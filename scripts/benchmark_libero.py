"""Compare OSMesa/EGL on the current evaluator, with fresh seeded GPU servers.

Run with the model Python; --libero-python points to the separate LIBERO runtime.
Profiling hooks live only in this script and do not change rollout observations.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def profile_worker(policy_url, connect_timeout, config):
    from post_train_vla import evaluator
    from post_train_vla.policy import WebsocketPolicy
    from robosuite.utils.binding_utils import MjSim

    seconds = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    previous = collections.defaultdict(float)
    previous_counts = collections.defaultdict(int)
    renderer = {}

    def timed(name, function):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                seconds[name] += time.perf_counter() - start
                counts[name] += 1
        return wrapped

    MjSim.render = timed("render", MjSim.render)
    create = evaluator._create_environment

    def create_environment(*args, **kwargs):
        env = timed("environment_create", create)(*args, **kwargs)
        from OpenGL import GL
        for name, enum in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION)):
            value = GL.glGetString(enum)
            renderer[name] = value.decode() if value else None
        for name in ("step", "reset", "set_init_state"):
            setattr(env, name, timed("environment_" + name, getattr(env, name)))
        return env

    evaluator._create_environment = create_environment
    evaluator.make_policy_observation = timed("preprocess", evaluator.make_policy_observation)
    append = evaluator._append_jsonl

    def append_profile(path, result):
        result["profile_seconds"] = {key: value - previous[key] for key, value in seconds.items()}
        previous.update(seconds)
        result["profile_calls"] = {key: value - previous_counts[key] for key, value in counts.items()}
        previous_counts.update(counts)
        result["gl"] = renderer.copy()
        append(path, result)

    evaluator._append_jsonl = append_profile
    with WebsocketPolicy(policy_url, connect_timeout=connect_timeout) as policy:
        policy.infer = timed("policy_request", policy.infer)
        return evaluator.evaluate(policy, config, policy.metadata)


def run_evaluation(args):
    from post_train_vla import eval_libero
    from post_train_vla.evaluator import EvalConfig

    eval_libero._evaluate_worker = profile_worker
    config = EvalConfig(task_ids=(args.task_id,), episodes_per_task=args.episodes, seed=args.seed,
                        output_dir=args.output_dir)
    start = time.perf_counter()
    if args.workers == 1:
        summary = profile_worker(args.policy_url, 120, config)
    else:
        summary = eval_libero.evaluate_parallel(args.policy_url, 120, config, args.workers)
    summary["evaluation_wall_seconds"] = time.perf_counter() - start
    records = [json.loads(line) for line in (args.output_dir / "episodes.jsonl").read_text().splitlines()]
    if os.environ.get("MUJOCO_GL") == "egl" and any(row["gl"]["vendor"] != "NVIDIA Corporation" for row in records):
        raise RuntimeError("EGL did not use NVIDIA hardware rendering; inspect the episode renderer records")
    steps = sum(row["policy_steps"] for row in records)
    requests = sum(row["inference_calls"] for row in records)
    totals = collections.defaultdict(float)
    calls = collections.defaultdict(int)
    for row in records:
        for key, value in row["profile_seconds"].items():
            totals[key] += value
        for key, value in row.get("profile_calls", {}).items():
            calls[key] += value
    summary.update(policy_steps=steps, policy_requests=requests,
                   errors=sum(row["error"] is not None for row in records),
                   episodes_per_minute=60 * len(records) / summary["evaluation_wall_seconds"],
                   steps_per_second=steps / summary["evaluation_wall_seconds"],
                   profile_worker_seconds=dict(totals), profile_worker_calls=dict(calls), gl=records[0]["gl"],
                   mean_batch_size=sum((row["mean_inference_batch_size"] or 1) * row["inference_calls"]
                                       for row in records) / requests if requests else None)
    (args.output_dir / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def run_server(args):
    import torch
    from post_train_vla.policy_server import PolicyServer
    from post_train_vla.torch_policy import TorchPolicy

    torch.manual_seed(args.seed)
    policy = TorchPolicy(args.checkpoint, args.tokenizer, device="cuda")
    PolicyServer(policy, host="127.0.0.1", port=args.port,
                 max_batch_size=args.max_batch_size, batch_wait_ms=10).serve_forever()


def stop(process):
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def benchmark(args):
    from post_train_vla.policy import WebsocketPolicy

    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    thread_variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    env.update({name: "1" for name in thread_variables})
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + ":" + str(args.libero_root)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for backend in args.backends:
        folder = args.output_dir / backend
        folder.mkdir()
        run_env = {**env, "MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend}
        server = monitor = None
        try:
            with (folder / "server.log").open("w") as log:
                start = time.perf_counter()
                server_command = [
                    sys.executable, script, "--role", "server", "--checkpoint", str(args.checkpoint),
                    "--tokenizer", str(args.tokenizer), "--seed", str(args.seed), "--port", str(args.port),
                    "--max-batch-size", str(args.max_batch_size), "--output-dir", str(folder),
                ]
                server = subprocess.Popen(server_command, env=env, stdout=log, stderr=subprocess.STDOUT)
                with WebsocketPolicy(args.policy_url, connect_timeout=120, retry_interval=0.2):
                    pass
                startup = time.perf_counter() - start
                print(f"{backend}: server ready in {startup:.2f}s; starting {args.episodes} episodes", flush=True)
                with (folder / "gpu.csv").open("w") as gpu, (folder / "eval.log").open("w") as evaluation_log:
                    monitor = subprocess.Popen(
                        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits", "-l", "1"], stdout=gpu
                    )
                    evaluation_command = [
                        str(args.libero_python), script, "--role", "eval", "--policy-url", args.policy_url,
                        "--episodes", str(args.episodes), "--workers", str(args.workers),
                        "--task-id", str(args.task_id), "--seed", str(args.seed), "--output-dir", str(folder),
                    ]
                    subprocess.run(evaluation_command, env=run_env,
                                   stdout=evaluation_log, stderr=subprocess.STDOUT, check=True)
                stop(monitor)
                monitor = None
                result = json.loads((folder / "benchmark.json").read_text())
                result["backend"] = backend
                result["server_startup_seconds"] = startup
                samples = [line.split(",") for line in (folder / "gpu.csv").read_text().splitlines() if line.strip()]
                result["sampled_gpu_utilization_percent"] = sum(float(row[1]) for row in samples) / len(samples)
                result["sampled_peak_vram_mib"] = max(float(row[2]) for row in samples)
                (folder / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
                results.append(result)
                (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
                print(
                    f"{backend}: {result['evaluation_wall_seconds']:.2f}s, "
                    f"{result['successes']}/{result['episodes']} successes, {result['errors']} errors",
                    flush=True,
                )
                if result["errors"]:
                    raise RuntimeError(f"{backend} produced rollout errors; inspect {folder / 'eval.log'}")
        finally:
            stop(monitor)
            stop(server)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("benchmark", "eval", "server"), default="benchmark")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/pi0-base-pytorch"))
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/paligemma_tokenizer.model"))
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/openpi_easy/third_party/libero"))
    parser.add_argument("--libero-python", type=Path,
                        default=Path("/home/ubuntu/openpi_easy/examples/libero/.venv/bin/python"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=("osmesa", "egl"), default=["osmesa", "egl"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=20)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--policy-url")
    args = parser.parse_args()
    if min(args.episodes, args.workers, args.max_batch_size) < 1:
        parser.error("episodes, workers and max-batch-size must be positive")
    if len(set(args.backends)) != len(args.backends):
        parser.error("backends must be unique")
    if args.role == "benchmark" and args.policy_url is not None:
        parser.error("benchmark starts its own local server; select its port with --port")
    args.policy_url = args.policy_url or f"ws://127.0.0.1:{args.port}"
    if args.role == "server":
        run_server(args)
    elif args.role == "eval":
        run_evaluation(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()

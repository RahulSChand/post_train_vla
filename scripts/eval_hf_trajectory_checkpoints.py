#!/usr/bin/env python3
"""Stream and evaluate hierarchical pi0.5 checkpoints from Hugging Face Hub."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download

from post_train_vla.eval_checkpoints import _summary_is_complete
from post_train_vla.finetune import _run_libero_evaluation, _wait_for_server


LOGGER = logging.getLogger(__name__)
CHECKPOINT_RE = re.compile(r"^(trajectories-(\d+))/(epoch-(\d+))/model\.safetensors$")
REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "metadata.json",
    "assets/physical-intelligence/libero/norm_stats.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--staging-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-batch-size", type=int, default=24)
    parser.add_argument("--batch-wait-ms", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=0.0,
        help="Exit after this much time with no newly completed evaluation; 0 performs one scan.",
    )
    parser.add_argument("--server-timeout", type=float, default=600.0)
    parser.add_argument(
        "--smoke-first",
        action="store_true",
        help="Run one task-0 episode once before the first full checkpoint evaluation.",
    )
    parser.add_argument(
        "--eval-python",
        type=pathlib.Path,
        default=pathlib.Path("/root/openpi_easy/examples/libero/.venv/bin/python"),
    )
    parser.add_argument("--openpi-root", type=pathlib.Path, default=pathlib.Path("/root/openpi_easy"))
    return parser.parse_args()


def _require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Policy server port {port} is already in use") from exc


def _discover_checkpoints(repo_id: str) -> tuple[str, list[tuple[int, int, str]]]:
    info = HfApi().model_info(repo_id, files_metadata=True)
    files = {item.rfilename: item.size for item in info.siblings}
    checkpoints = []
    for filename, size in files.items():
        match = CHECKPOINT_RE.match(filename)
        if not match or not size:
            continue
        trajectory_dir, trajectory_text, epoch_dir, epoch_text = match.groups()
        prefix = f"{trajectory_dir}/{epoch_dir}"
        if all(files.get(f"{prefix}/{relative}", 0) for relative in REQUIRED_FILES):
            checkpoints.append((int(trajectory_text), int(epoch_text), prefix))
    return info.sha, sorted(checkpoints)


def _belongs_to_shard(trajectory_count: int, epoch: int, shard_index: int, num_shards: int) -> bool:
    return (trajectory_count * 1000 + epoch) % num_shards == shard_index


def _read_complete_summary(path: pathlib.Path, episodes: int) -> dict | None:
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not _summary_is_complete(summary, episodes=episodes, suite="libero_spatial", task_id=None):
        return None
    if summary.get("config", {}).get("save_video") is not False:
        return None
    if "source" not in summary or len(summary.get("per_task", {})) != 10:
        return None
    return summary


def _download_checkpoint(repo_id: str, revision: str, prefix: str, local_dir: pathlib.Path) -> pathlib.Path:
    for relative in REQUIRED_FILES:
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{prefix}/{relative}",
            revision=revision,
            local_dir=local_dir,
        )
    checkpoint = local_dir / prefix
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("pi05") is not True:
        raise ValueError(f"Checkpoint does not declare pi05=true: {prefix}")
    return checkpoint


def _enrich_summary(
    summary_path: pathlib.Path,
    *,
    repo_id: str,
    revision: str,
    trajectory_count: int,
    epoch: int,
) -> dict:
    summary = json.loads(summary_path.read_text())
    episodes_path = summary_path.with_name("episodes.jsonl")
    task_results: dict[int, list[dict]] = collections.defaultdict(list)
    for line in episodes_path.read_text().splitlines():
        if line:
            result = json.loads(line)
            task_results[int(result["task_id"])].append(result)
    errors = [result for results in task_results.values() for result in results if result.get("error")]
    if errors:
        raise RuntimeError(f"Evaluation recorded {len(errors)} rollout errors; refusing to accept the summary")
    if len(task_results) != 10 or any(len(results) != 40 for results in task_results.values()):
        counts = {task_id: len(results) for task_id, results in sorted(task_results.items())}
        raise RuntimeError(f"Evaluation does not contain exactly 40 episodes for each of 10 tasks: {counts}")
    summary["per_task"] = {
        str(task_id): {
            "task": results[0]["task"],
            "episodes": len(results),
            "successes": sum(bool(result["success"]) for result in results),
            "success_rate": sum(bool(result["success"]) for result in results) / len(results),
        }
        for task_id, results in sorted(task_results.items())
    }
    summary["source"] = {
        "repo_id": repo_id,
        "revision": revision,
        "trajectory_count": trajectory_count,
        "epoch": epoch,
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    return summary


def _write_aggregate(output_dir: pathlib.Path) -> None:
    results = []
    for summary_path in output_dir.glob("trajectories-*/epoch-*/summary.json"):
        summary = _read_complete_summary(summary_path, episodes=40)
        if summary is not None and "source" in summary and "per_task" in summary:
            results.append(summary)
    results.sort(key=lambda result: (result["source"]["trajectory_count"], result["source"]["epoch"]))
    aggregate = {
        "checkpoints": len(results),
        "episodes_per_task": 40,
        "episodes_per_checkpoint": 400,
        "suite": "libero_spatial",
        "save_video": False,
        "results": results,
        "updated_at": time.time(),
    }
    temporary = output_dir / f".summary.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_dir / "summary.json")


def _evaluate_one(
    args: argparse.Namespace,
    *,
    revision: str,
    trajectory_count: int,
    epoch: int,
    prefix: str,
) -> None:
    evaluation_dir = args.output_dir / f"trajectories-{trajectory_count:03d}" / f"epoch-{epoch:03d}"
    summary_path = evaluation_dir / "summary.json"
    if _read_complete_summary(summary_path, args.episodes) is not None:
        return

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(pathlib.Path(__file__).resolve().parents[1] / "src"), str(args.openpi_root / "third_party/libero"))
            ),
        }
    )
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"gpu{args.gpu}-", dir=args.staging_dir) as temporary_text:
        temporary = pathlib.Path(temporary_text)
        LOGGER.info("Downloading %s at revision %s", prefix, revision)
        checkpoint = _download_checkpoint(args.repo_id, revision, prefix, temporary)
        server_command = [
            sys.executable,
            "-m",
            "post_train_vla.serve_torch",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(args.tokenizer),
            "--model",
            "pi05",
            "--device",
            "cuda",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--max-batch-size",
            str(args.max_batch_size),
            "--batch-wait-ms",
            str(args.batch_wait_ms),
        ]
        server = subprocess.Popen(server_command, cwd=pathlib.Path(__file__).resolve().parents[1], env=environment)
        try:
            _wait_for_server(server, "127.0.0.1", args.port, args.server_timeout)
            smoke_summary = args.output_dir / "smoke" / "summary.json"
            if args.smoke_first and not smoke_summary.is_file():
                LOGGER.info("Running one-episode task-0 smoke evaluation with %s", prefix)
                _run_libero_evaluation(
                    f"ws://127.0.0.1:{args.port}",
                    args.output_dir / "smoke",
                    eval_python=args.eval_python.absolute(),
                    openpi_root=args.openpi_root,
                    suite="libero_spatial",
                    task_id=0,
                    episodes=1,
                    save_video=False,
                    eval_workers=1,
                    environment=environment,
                )
            LOGGER.info("Evaluating %s: 40 episodes on each of 10 tasks, no video", prefix)
            _run_libero_evaluation(
                f"ws://127.0.0.1:{args.port}",
                evaluation_dir,
                eval_python=args.eval_python.absolute(),
                openpi_root=args.openpi_root,
                suite="libero_spatial",
                task_id=None,
                episodes=args.episodes,
                save_video=False,
                eval_workers=args.workers,
                environment=environment,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    summary = _enrich_summary(
        summary_path,
        repo_id=args.repo_id,
        revision=revision,
        trajectory_count=trajectory_count,
        epoch=epoch,
    )
    LOGGER.info(
        "Completed %s: %d/%d (%.1f%%)",
        prefix,
        summary["successes"],
        summary["episodes"],
        100 * summary["success_rate"],
    )
    _write_aggregate(args.output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= --shard-index < --num-shards")
    if args.episodes != 40:
        raise ValueError("This trajectory-efficiency run requires exactly --episodes 40")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.staging_dir = args.staging_dir.expanduser().resolve()
    args.tokenizer = args.tokenizer.expanduser().resolve()
    args.eval_python = args.eval_python.expanduser().absolute()
    args.openpi_root = args.openpi_root.expanduser().resolve()
    for required in (args.tokenizer, args.eval_python, args.openpi_root / "third_party/libero"):
        if not required.exists():
            raise FileNotFoundError(required)
    _require_free_port(args.port)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    last_completion = time.monotonic()
    while True:
        revision, discovered = _discover_checkpoints(args.repo_id)
        assigned = [
            item
            for item in discovered
            if _belongs_to_shard(item[0], item[1], args.shard_index, args.num_shards)
        ]
        pending = [
            item
            for item in assigned
            if _read_complete_summary(
                args.output_dir / f"trajectories-{item[0]:03d}" / f"epoch-{item[1]:03d}" / "summary.json",
                args.episodes,
            )
            is None
        ]
        LOGGER.info(
            "Revision %s: discovered=%d assigned=%d pending=%d",
            revision,
            len(discovered),
            len(assigned),
            len(pending),
        )
        for trajectory_count, epoch, prefix in pending:
            try:
                _evaluate_one(
                    args,
                    revision=revision,
                    trajectory_count=trajectory_count,
                    epoch=epoch,
                    prefix=prefix,
                )
            except Exception:
                LOGGER.exception("Evaluation failed for %s; it will be retried after the next scan", prefix)
            else:
                last_completion = time.monotonic()

        status = {
            "gpu": args.gpu,
            "shard_index": args.shard_index,
            "revision": revision,
            "discovered": len(discovered),
            "assigned": len(assigned),
            "pending_at_scan": len(pending),
            "updated_at": time.time(),
        }
        (args.output_dir / f"worker_{args.shard_index}_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n"
        )
        _write_aggregate(args.output_dir)
        if args.idle_exit_seconds == 0:
            return
        if not pending and time.monotonic() - last_completion >= args.idle_exit_seconds:
            LOGGER.info("No new checkpoints completed for %.0f seconds; exiting", args.idle_exit_seconds)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

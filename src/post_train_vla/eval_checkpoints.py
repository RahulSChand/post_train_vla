"""Fast, resumable LIBERO evaluation across periodic training checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import time

from post_train_vla.finetune import _evaluation_environment, _run_libero_evaluation, _wait_for_server
from post_train_vla.policy import WebsocketPolicy

LOGGER = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--every", type=int, default=500, help="Evaluate numeric checkpoint steps divisible by N")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=20)
    parser.add_argument("--batch-wait-ms", type=float, default=10.0)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Evaluate one task ID; omit to evaluate every task in the suite",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pi05", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--server-timeout", type=float, default=600.0)
    parser.add_argument(
        "--eval-python",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/openpi_easy/examples/libero/.venv/bin/python"),
    )
    parser.add_argument(
        "--openpi-root", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/openpi_easy")
    )
    return parser.parse_args()


def _checkpoint_steps(checkpoints_dir: pathlib.Path, every: int) -> list[int]:
    if every < 1:
        raise ValueError("--every must be positive")
    return sorted(
        int(path.name)
        for path in checkpoints_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and int(path.name) % every == 0
        and (path / "model.safetensors").is_file()
    )


def _summary_is_complete(summary: dict, *, episodes: int, suite: str, task_id: int | None) -> bool:
    config = summary.get("config", {})
    expected_total = episodes if task_id is not None else episodes * 10
    return (
        summary.get("episodes") == expected_total
        and summary.get("suite") == suite
        and config.get("task_ids") == ([task_id] if task_id is not None else None)
        and config.get("episodes_per_task") == episodes
        and config.get("episode_offset") == 0
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    checkpoints_dir = args.checkpoints_dir.expanduser().resolve()
    steps = _checkpoint_steps(checkpoints_dir, args.every)
    if not steps:
        raise FileNotFoundError(f"No checkpoints divisible by {args.every} below {checkpoints_dir}")

    LOGGER.info(
        "Evaluating %d checkpoints with %d workers and max batch size %d",
        len(steps),
        args.workers,
        args.max_batch_size,
    )
    started_at = time.monotonic()
    results = []
    completed = {}
    pending = []
    for step in steps:
        summary_path = checkpoints_dir / "eval" / f"step_{step:06d}" / "summary.json"
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text())
            if _summary_is_complete(existing, episodes=args.episodes, suite=args.suite, task_id=args.task_id):
                completed[step] = existing
                continue
        pending.append(step)

    server = None
    if pending:
        host = "127.0.0.1"
        policy_url = f"ws://{host}:{args.port}"
        project_root, environment = _evaluation_environment(args.openpi_root.expanduser().resolve())
        server_command = [
            sys.executable,
            "-m",
            "post_train_vla.serve_torch",
            "--checkpoint",
            str(checkpoints_dir / str(pending[0])),
            "--tokenizer",
            str(args.tokenizer.expanduser().resolve()),
            "--device",
            args.device,
            "--host",
            host,
            "--port",
            str(args.port),
            "--max-batch-size",
            str(args.max_batch_size),
            "--batch-wait-ms",
            str(args.batch_wait_ms),
        ]
        if args.pi05:
            server_command.extend(("--model", "pi05"))
        server = subprocess.Popen(server_command, cwd=project_root, env=environment)
        try:
            _wait_for_server(server, host, args.port, args.server_timeout)
            current_step = pending[0]
            for step in pending:
                index = steps.index(step) + 1
                if step != current_step:
                    reload_started = time.monotonic()
                    with WebsocketPolicy(policy_url, connect_timeout=args.server_timeout) as policy:
                        policy.load_checkpoint(str(checkpoints_dir / str(step)))
                    LOGGER.info("Loaded step %d in %.1fs", step, time.monotonic() - reload_started)
                    current_step = step
                LOGGER.info("[%d/%d] evaluating step %d", index, len(steps), step)
                evaluation_dir = (checkpoints_dir / "eval" / f"step_{step:06d}").resolve()
                summary = _run_libero_evaluation(
                    policy_url,
                    evaluation_dir,
                    # Keep the venv launcher path intact. Resolving its interpreter
                    # symlink bypasses the venv and loses its installed packages.
                    eval_python=args.eval_python.expanduser().absolute(),
                    openpi_root=args.openpi_root.expanduser().resolve(),
                    suite=args.suite,
                    task_id=args.task_id,
                    episodes=args.episodes,
                    save_video=args.save_video,
                    eval_workers=args.workers,
                    environment=environment,
                )
                completed[step] = summary
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    for index, step in enumerate(steps, start=1):
        if step not in pending:
            LOGGER.info("[%d/%d] step %d already complete; skipped", index, len(steps), step)
        results.append({"step": step, **completed[step]})

    aggregate = {
        "checkpoints": len(results),
        "elapsed_seconds": time.monotonic() - started_at,
        "results": results,
    }
    aggregate_path = checkpoints_dir / "eval" / "summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Finished %d checkpoints; aggregate=%s", len(results), aggregate_path)


if __name__ == "__main__":
    main()

"""Command-line entrypoint for LIBERO evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import multiprocessing
import pathlib
import shutil
import tempfile

from post_train_vla.evaluator import EvalConfig, evaluate
from post_train_vla.policy import WebsocketPolicy


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, action="append", dest="task_ids")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/libero"))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel LIBERO rollout processes")
    return parser.parse_args()


def _episode_chunks(episodes: int, workers: int) -> list[tuple[int, int]]:
    workers = min(workers, episodes)
    base, remainder = divmod(episodes, workers)
    chunks = []
    offset = 0
    for worker in range(workers):
        count = base + (worker < remainder)
        chunks.append((offset, count))
        offset += count
    return chunks


def _evaluate_worker(policy_url: str, connect_timeout: float, config: EvalConfig) -> dict:
    with WebsocketPolicy(policy_url, connect_timeout=connect_timeout) as policy:
        return evaluate(policy, config, policy.metadata)


def evaluate_parallel(policy_url: str, connect_timeout: float, config: EvalConfig, workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = _episode_chunks(config.episodes_per_task, workers)
    with tempfile.TemporaryDirectory(prefix="eval-workers-", dir=config.output_dir) as temporary:
        temporary_path = pathlib.Path(temporary)
        worker_configs = [
            dataclasses.replace(
                config,
                episode_offset=config.episode_offset + offset,
                episodes_per_task=count,
                output_dir=temporary_path / f"worker_{index:02d}",
            )
            for index, (offset, count) in enumerate(chunks)
        ]
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as pool:
            summaries = list(
                pool.map(
                    _evaluate_worker,
                    [policy_url] * len(chunks),
                    [connect_timeout] * len(chunks),
                    worker_configs,
                )
            )

        results = []
        for worker_config in worker_configs:
            episodes_path = worker_config.output_dir / "episodes.jsonl"
            results.extend(json.loads(line) for line in episodes_path.read_text().splitlines() if line)
            if config.save_video:
                videos_dir = config.output_dir / "videos"
                videos_dir.mkdir(exist_ok=True)
                for video in (worker_config.output_dir / "videos").glob("*.mp4"):
                    shutil.move(str(video), videos_dir / video.name)

    results.sort(key=lambda result: (result["task_id"], result["episode_index"]))
    episodes_path = config.output_dir / "episodes.jsonl"
    episodes_path.write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in results))
    successes = sum(result["success"] for result in results)
    summary = {
        "suite": config.suite,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "policy_metadata": summaries[0].get("policy_metadata", {}) if summaries else {},
        "workers": len(chunks),
        "config": {**dataclasses.asdict(config), "output_dir": str(config.output_dir)},
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    logging.getLogger(__name__).info(
        "Finished %d episodes with %d workers: %d successes (%.1f%%)",
        len(results), len(chunks), successes, 100 * summary["success_rate"],
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    config = EvalConfig(
        suite=args.suite,
        task_ids=tuple(args.task_ids) if args.task_ids else None,
        episodes_per_task=args.episodes_per_task,
        episode_offset=args.episode_offset,
        seed=args.seed,
        wait_steps=args.wait_steps,
        replan_steps=args.replan_steps,
        resize_size=args.resize_size,
        render_resolution=args.render_resolution,
        output_dir=args.output_dir,
        save_video=args.save_video,
    )
    if args.workers == 1:
        with WebsocketPolicy(args.policy_url, connect_timeout=args.connect_timeout) as policy:
            evaluate(policy, config, policy.metadata)
    else:
        evaluate_parallel(args.policy_url, args.connect_timeout, config, args.workers)


if __name__ == "__main__":
    main()

"""Command-line entrypoint for LIBERO evaluation."""

import argparse
import logging
import pathlib

from post_train_vla.evaluator import EvalConfig, evaluate
from post_train_vla.policy import WebsocketPolicy


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, action="append", dest="task_ids")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/libero"))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    config = EvalConfig(
        suite=args.suite,
        task_ids=tuple(args.task_ids) if args.task_ids else None,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        wait_steps=args.wait_steps,
        replan_steps=args.replan_steps,
        resize_size=args.resize_size,
        render_resolution=args.render_resolution,
        output_dir=args.output_dir,
        save_video=args.save_video,
    )
    with WebsocketPolicy(args.policy_url, connect_timeout=args.connect_timeout) as policy:
        evaluate(policy, config, policy.metadata)


if __name__ == "__main__":
    main()

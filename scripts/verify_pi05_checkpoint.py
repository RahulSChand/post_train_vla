#!/usr/bin/env python3
"""Strict-load a trained pi0.5 checkpoint and run one finite action sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from post_train_vla.torch_policy import TorchPolicy


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(quat[3]) / denominator).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.sample, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"]))
        state = np.concatenate(
            (
                np.asarray(source["observation__robot0_eef_pos"][0], dtype=np.float32),
                quat_to_axis_angle(source["observation__robot0_eef_quat"][0]),
                np.asarray(source["observation__robot0_gripper_qpos"][0], dtype=np.float32),
            )
        )
        raw = {
            "observation/image": np.ascontiguousarray(
                source["observation__agentview_image"][0, ::-1, ::-1]
            ),
            "observation/wrist_image": np.ascontiguousarray(
                source["observation__robot0_eye_in_hand_image"][0, ::-1, ::-1]
            ),
            "observation/state": state,
            "prompt": str(
                metadata.get("source_trace_metadata", {}).get("task_description")
                or metadata.get("instruction")
                or metadata.get("task")
            ).strip().lower().removesuffix("."),
        }

    policy = TorchPolicy(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
        pi05=True,
    )
    noise = np.zeros(
        (policy.config.action_horizon, policy.config.action_dim), dtype=np.float32
    )
    actions = policy.infer(raw, noise=noise)["actions"]
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "strict_load": True,
        "actions_shape": list(actions.shape),
        "actions_finite": bool(np.isfinite(actions).all()),
    }
    if not result["actions_finite"]:
        raise RuntimeError(f"Non-finite inference actions: {result}")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

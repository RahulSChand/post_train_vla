#!/usr/bin/env python3
"""Build a compact provenance and metrics summary for a pi0.5 training run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml
from safetensors import safe_open


STEP_PATTERN = re.compile(
    r"step=(?P<step>\d+) loss=(?P<loss>[0-9.eE+-]+) gpu_peak_memory_gib=(?P<memory>[0-9.]+)"
)
TRAINABLE_PATTERN = re.compile(r"trainable_parameters=(\d+)")


def checkpoint_parameter_count(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as tensors:
        return sum(int(np.prod(tensors.get_slice(key).get_shape())) for key in tensors.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--inference-result", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load((args.output_dir / "run_config.yaml").read_text())["arguments"]
    log_text = args.log.read_text()
    metrics = [
        (int(match["step"]), float(match["loss"]), float(match["memory"]))
        for match in STEP_PATTERN.finditer(log_text)
    ]
    if not metrics or metrics[-1][0] != args.step:
        raise ValueError(f"Training log does not end at requested step {args.step}")
    files = sorted((args.dataset_dir / "corrections").glob("*.npz"))
    frames = sum(len(np.load(path, allow_pickle=False)["expert_actions"]) for path in files)
    trainable_match = TRAINABLE_PATTERN.search(log_text)
    checkpoint = args.output_dir / str(args.step)
    trainable_parameters = (
        int(trainable_match.group(1))
        if trainable_match
        else checkpoint_parameter_count(checkpoint / "model.safetensors")
    )
    losses = [loss for _, loss, _ in metrics]
    summary = {
        "schema": "pi05_cross_policy_training_run_v1",
        "condition": args.condition,
        "checkpoint": str(checkpoint.resolve()),
        "starting_checkpoint": str(config["checkpoint"]),
        "correction_episodes": len(files),
        "correction_frames": frames,
        "optimizer_steps": args.step,
        "effective_batch_size": int(config["batch_size"])
        * int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "optimizer": config["optimizer"],
        "training_mode": "lora" if config["lora"] else "full",
        "trainable_parameters": trainable_parameters,
        "gpu_peak_memory_gib": max(memory for _, _, memory in metrics),
        "loss_first_50_mean": float(np.mean(losses[:50])),
        "loss_last_50_mean": float(np.mean(losses[-50:])),
        "loss_final": losses[-1],
        "inference_smoke": json.loads(args.inference_result.read_text()),
    }
    destination = args.output_dir / "training_summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.copy2(args.log, args.output_dir / "train.log")
    for source_name, destination_name in (
        ("dataset_manifest.json", "dataset_manifest.json"),
        ("summary.json", "dataset_collection_summary.json"),
    ):
        source = args.dataset_dir / source_name
        if source.is_file():
            shutil.copy2(source, args.output_dir / destination_name)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Minimal PyTorch/LeRobot fine-tuning entrypoint for standalone pi0/pi0.5."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

import numpy as np
import safetensors.torch
import torch
from torch.utils.data import DataLoader, Dataset

from post_train_vla.models import Pi0, Pi0Config
from post_train_vla.torch_policy import find_norm_stats
from post_train_vla.transforms import LiberoTransforms


class LeRobotLiberoDataset(Dataset):
    """Small adapter for the field layout produced by OpenPI's LIBERO converter."""

    def __init__(self, repo_id: str, action_horizon: int) -> None:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError("Install the training extra: uv sync --extra train") from exc
        self.meta = LeRobotDatasetMetadata(repo_id)
        self.dataset = LeRobotDataset(
            repo_id,
            delta_timestamps={"action": [step / self.meta.fps for step in range(action_horizon)]},
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        task_index = int(item["task_index"])
        return {
            "image": item["image"],
            "wrist_image": item["wrist_image"],
            "state": item["state"],
            "action": item["action"],
            "prompt": self.meta.tasks[task_index],
        }


def save_checkpoint(model: Pi0, optimizer: torch.optim.Optimizer, step: int, output_dir: pathlib.Path,
                    source_checkpoint: pathlib.Path) -> pathlib.Path:
    checkpoint = output_dir / str(step)
    temporary = output_dir / f".{step}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    safetensors.torch.save_model(model, str(temporary / "model.safetensors"))
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    (temporary / "metadata.json").write_text(json.dumps({"step": step, "created_at": time.time()}, indent=2) + "\n")
    shutil.copy2(source_checkpoint / "config.json", temporary / "config.json")
    stats = find_norm_stats(source_checkpoint)
    relative_stats = stats.relative_to(source_checkpoint)
    destination = temporary / relative_stats
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats, destination)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temporary.rename(checkpoint)
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Converted starting checkpoint")
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-repo", default="physical-intelligence/libero")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pi05", action="store_true")
    parser.add_argument(
        "--extra-delta-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Subtract current state from the first six action dimensions (default: enabled for pi0, disabled for pi0.5)",
    )
    parser.add_argument("--heads-only", action="store_true", help="Train only action/time projection layers")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    config = Pi0Config.from_checkpoint(checkpoint, pi05=args.pi05)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = Pi0(config)
    safetensors.torch.load_model(model, str(checkpoint / "model.safetensors"), strict=True)
    model.to(device).train()
    if args.heads_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("action_", "state_proj", "time_mlp_"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    transforms = LiberoTransforms(config, find_norm_stats(checkpoint), args.tokenizer)
    extra_delta_actions = not config.pi05 if args.extra_delta_actions is None else args.extra_delta_actions
    dataset = LeRobotLiberoDataset(args.dataset_repo, config.action_horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    iterator = iter(loader)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        observation, actions = transforms.encode_training_batch(
            batch, device, extra_delta_actions=extra_delta_actions
        )
        loss = model.loss(observation, actions).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"step={step} loss={loss.item():.6f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            saved = save_checkpoint(model, optimizer, step, args.output_dir, checkpoint)
            print(f"saved={saved}", flush=True)


if __name__ == "__main__":
    main()

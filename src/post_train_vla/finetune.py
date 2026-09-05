"""Minimal PyTorch/LeRobot fine-tuning entrypoint for standalone pi0/pi0.5."""

from __future__ import annotations

import argparse
import bisect
import io
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import safetensors.torch
import torch
from torch.utils.data import DataLoader, Dataset

from post_train_vla.models import Pi0, Pi0Config
from post_train_vla.torch_policy import find_norm_stats
from post_train_vla.transforms import LiberoTransforms


class LeRobotLiberoDataset(Dataset):
    """Adapter for local or Hub LeRobot datasets using the trainer's canonical fields."""

    def __init__(self, repo_id: str, action_horizon: int) -> None:
        local_root = pathlib.Path(repo_id).expanduser()
        if local_root.is_dir():
            self.dataset = _LocalLeRobotDataset(local_root, action_horizon)
            self.meta = self.dataset.meta
            return
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError("Install the training extra: uv sync --extra train") from exc
        self.meta = LeRobotDatasetMetadata(repo_id)
        action_key = "actions" if "actions" in self.meta.features else "action"
        self.dataset = LeRobotDataset(
            repo_id,
            delta_timestamps={action_key: [step / self.meta.fps for step in range(action_horizon)]},
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        task_index = int(item["task_index"])
        actions = item.get("actions", item.get("action"))
        if actions is None:
            raise KeyError("Dataset must contain an 'actions' or 'action' feature")
        return {
            "image": item["image"],
            "wrist_image": item["wrist_image"],
            "state": item["state"],
            "action": actions,
            "prompt": self.meta.tasks[task_index],
        }


class _LocalLeRobotDataset(Dataset):
    """Minimal v2.1 reader for image-backed local LeRobot datasets.

    The pinned LeRobot revision currently assumes an older Hugging Face Datasets
    API. Reading the parquet files directly keeps local training independent of
    that API mismatch while preserving the same sample semantics.
    """

    def __init__(self, root: pathlib.Path, action_horizon: int) -> None:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError("Install the training extra: uv sync --extra train") from exc

        self._parquet = parquet
        self.root = root.resolve()
        info = json.loads((self.root / "meta" / "info.json").read_text())
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"Expected a LeRobot v2.1 dataset, got {info.get('codebase_version')!r}")
        features = info.get("features", {})
        required = {"image", "wrist_image", "state", "task_index"}
        missing = required.difference(features)
        action_key = "actions" if "actions" in features else "action"
        missing_actions = {action_key}.difference(features)
        if missing or missing_actions:
            raise ValueError(f"Dataset is missing required features: {sorted(missing | missing_actions)}")

        self.action_horizon = action_horizon
        self.action_key = action_key
        self.data_path = info["data_path"]
        self.chunks_size = int(info["chunks_size"])
        self.meta = SimpleNamespace(
            fps=info["fps"],
            features=features,
            tasks=self._load_tasks(),
        )
        self._episodes = [
            json.loads(line) for line in (self.root / "meta" / "episodes.jsonl").read_text().splitlines()
        ]
        self._starts = []
        total = 0
        for episode in self._episodes:
            self._starts.append(total)
            total += int(episode["length"])
        if total != int(info["total_frames"]):
            raise ValueError(f"Episode lengths sum to {total}, expected {info['total_frames']}")
        self._length = total
        self._cached_episode = None
        self._cached_table = None

    def _load_tasks(self) -> dict[int, str]:
        tasks = {}
        for line in (self.root / "meta" / "tasks.jsonl").read_text().splitlines():
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
        return tasks

    def __len__(self) -> int:
        return self._length

    def _table_for_episode(self, episode_index: int):
        if self._cached_episode != episode_index:
            chunk = episode_index // self.chunks_size
            relative = self.data_path.format(episode_chunk=chunk, episode_index=episode_index)
            self._cached_table = self._parquet.read_table(self.root / relative)
            self._cached_episode = episode_index
        return self._cached_table

    @staticmethod
    def _decode_image(value) -> np.ndarray:
        payload = value.get("bytes") if isinstance(value, dict) else value
        if not payload:
            raise ValueError("Local LeRobot image has no embedded bytes")
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            return np.array(image.convert("RGB"), copy=True)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._starts, index) - 1
        row = index - self._starts[episode_index]
        episode_length = int(self._episodes[episode_index]["length"])
        table = self._table_for_episode(episode_index)

        def column_value(name: str, row_index: int):
            return table[name][row_index].as_py()

        action_rows = [
            column_value(self.action_key, min(row + offset, episode_length - 1))
            for offset in range(self.action_horizon)
        ]
        task_index = int(column_value("task_index", row))
        return {
            "image": self._decode_image(column_value("image", row)),
            "wrist_image": self._decode_image(column_value("wrist_image", row)),
            "state": np.asarray(column_value("state", row), dtype=np.float32),
            "action": np.asarray(action_rows, dtype=np.float32),
            "task_index": task_index,
            "prompt": self.meta.tasks[task_index],
        }


def save_checkpoint(
    model: Pi0,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: pathlib.Path,
    source_checkpoint: pathlib.Path,
    *,
    include_optimizer: bool = True,
) -> pathlib.Path:
    checkpoint = output_dir / str(step)
    temporary = output_dir / f".{step}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    safetensors.torch.save_model(model, str(temporary / "model.safetensors"))
    if include_optimizer:
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    (temporary / "metadata.json").write_text(
        json.dumps({"step": step, "created_at": time.time(), "resumable": include_optimizer}, indent=2) + "\n"
    )
    config_values = json.loads((source_checkpoint / "config.json").read_text())
    if isinstance(model, Pi0):
        # A LoRA checkpoint must declare its adapter layout so the standalone
        # policy loader constructs the same modules before strict loading.
        config_values.update(
            {
                "precision": model.config.dtype,
                "paligemma_lora_rank": model.config.paligemma_lora_rank,
                "action_expert_lora_rank": model.config.action_expert_lora_rank,
            }
        )
    (temporary / "config.json").write_text(json.dumps(config_values, indent=2, sort_keys=True) + "\n")
    stats = find_norm_stats(source_checkpoint)
    relative_stats = stats.relative_to(source_checkpoint)
    destination = temporary / relative_stats
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats, destination)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temporary.rename(checkpoint)
    return checkpoint


def _wait_for_server(process: subprocess.Popen, host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Evaluation policy server exited early with status {return_code}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"Evaluation policy server was not ready at {host}:{port} after {timeout:.0f}s")


def evaluate_checkpoint(
    checkpoint: pathlib.Path,
    tokenizer: pathlib.Path,
    step: int,
    output_dir: pathlib.Path,
    *,
    eval_python: pathlib.Path,
    openpi_root: pathlib.Path,
    suite: str,
    task_id: int,
    episodes: int,
    save_video: bool,
    port: int,
    server_timeout: float,
    device: str,
    pi05: bool,
) -> dict:
    """Evaluate a saved checkpoint in LIBERO using the separate Python 3.8 environment."""
    host = "127.0.0.1"
    # The evaluator runs with ``openpi_root`` as its working directory, so pass
    # an absolute output path and read the result from that same location.
    evaluation_dir = (output_dir / "eval" / f"step_{step:06d}").resolve()
    project_root = pathlib.Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_paths = [str(project_root / "src"), str(openpi_root / "third_party" / "libero")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.setdefault("MUJOCO_GL", "egl")

    server_command = [
        sys.executable,
        "-m",
        "post_train_vla.serve_torch",
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(tokenizer),
        "--device",
        device,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if pi05:
        server_command.extend(("--model", "pi05"))

    evaluator_command = [
        str(eval_python),
        "-m",
        "post_train_vla.eval_libero",
        "--policy-url",
        f"ws://{host}:{port}",
        "--suite",
        suite,
        "--task-id",
        str(task_id),
        "--episodes-per-task",
        str(episodes),
        "--output-dir",
        str(evaluation_dir),
    ]
    if save_video:
        evaluator_command.append("--save-video")

    server = subprocess.Popen(server_command, cwd=project_root, env=environment)
    try:
        _wait_for_server(server, host, port, server_timeout)
        subprocess.run(evaluator_command, cwd=openpi_root, env=environment, check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    summary_path = evaluation_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"LIBERO evaluation did not produce {summary_path}")
    return json.loads(summary_path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Converted starting checkpoint")
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset-repo",
        default="physical-intelligence/libero",
        help="Hub repo ID or local LeRobot dataset directory",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--save-every",
        type=int,
        default=3000,
        help="Checkpoint interval. With --train_only, these are evaluation-only checkpoints.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pi05", action="store_true")
    parser.add_argument("--wandb-entity", default="chandrahul0320")
    parser.add_argument("--wandb-project", default="post_vla")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument(
        "--train-only",
        "--train_only",
        dest="train_only",
        action="store_true",
        help="Disable in-training evaluation; save resumable checkpoints only at epoch boundaries and the final step.",
    )
    parser.add_argument("--eval-every", type=int, default=0, help="Run LIBERO evaluation every N steps (0 disables)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-suite", default="libero_spatial")
    parser.add_argument("--eval-task-id", type=int, default=0)
    parser.add_argument("--eval-save-video", action="store_true")
    parser.add_argument("--eval-port", type=int, default=8001)
    parser.add_argument("--eval-server-timeout", type=float, default=600.0)
    parser.add_argument(
        "--eval-python",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/openpi_easy/examples/libero/.venv/bin/python"),
    )
    parser.add_argument("--openpi-root", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/openpi_easy"))
    parser.add_argument(
        "--extra-delta-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Subtract current state from the first six action dimensions "
            "(default: enabled for pi0, disabled for pi0.5)"
        ),
    )
    trainable_group = parser.add_mutually_exclusive_group()
    trainable_group.add_argument(
        "--heads-only", action="store_true", help="Train only action/time projection layers"
    )
    trainable_group.add_argument(
        "--lora",
        action="store_true",
        help="Use OpenPI-compatible LoRA on Gemma attention and MLP projections",
    )
    parser.add_argument(
        "--lora-paligemma-rank",
        type=int,
        default=16,
        help="LoRA rank for the 2B PaliGemma stream (OpenPI default: 16)",
    )
    parser.add_argument(
        "--lora-action-expert-rank",
        type=int,
        default=32,
        help="LoRA rank for the 300M action expert (OpenPI default: 32)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recompute joint-backbone layers in backward to reduce training memory (enabled by default with --lora)",
    )
    return parser


def configure_lora_trainable_parameters(model: Pi0) -> list[torch.nn.Parameter]:
    """Train adapters and pi0 action/time heads while freezing converted base weights."""
    trainable_heads = ("action_", "state_proj", "time_mlp_")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_" in name or name.startswith(trainable_heads)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    if args.lora and (args.lora_paligemma_rank < 1 or args.lora_action_expert_rank < 1):
        raise ValueError("LoRA ranks must be positive")
    if args.eval_every < 0 or args.eval_episodes < 1:
        raise ValueError("--eval-every must be non-negative and --eval-episodes must be positive")
    if args.train_only and args.eval_every:
        raise ValueError("--train_only cannot be combined with --eval-every; evaluate saved checkpoints separately")
    if args.eval_every:
        if not args.eval_python.is_file():
            raise FileNotFoundError(f"LIBERO evaluation Python not found: {args.eval_python}")
        if not args.openpi_root.is_dir():
            raise FileNotFoundError(f"OpenPI root not found: {args.openpi_root}")
    checkpoint = args.checkpoint.expanduser().resolve()
    config = Pi0Config.from_checkpoint(checkpoint, pi05=args.pi05)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = Pi0(config)
    safetensors.torch.load_model(model, str(checkpoint / "model.safetensors"), strict=True)
    model.to(device).train()
    if args.lora:
        replaced = model.enable_lora(
            paligemma_rank=args.lora_paligemma_rank,
            action_expert_rank=args.lora_action_expert_rank,
        )
        parameters = configure_lora_trainable_parameters(model)
        print(
            "lora=" + ", ".join(f"{name}:{len(names)} projections" for name, names in replaced.items())
            + f" trainable_parameters={sum(parameter.numel() for parameter in parameters)}",
            flush=True,
        )
    elif args.heads_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("action_", "state_proj", "time_mlp_"))
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    else:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_checkpointing = args.lora if args.gradient_checkpointing is None else args.gradient_checkpointing
    model.set_gradient_checkpointing(gradient_checkpointing)
    if gradient_checkpointing:
        print("gradient_checkpointing=enabled", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    transforms = LiberoTransforms(config, find_norm_stats(checkpoint), args.tokenizer)
    extra_delta_actions = not config.pi05 if args.extra_delta_actions is None else args.extra_delta_actions
    dataset = LeRobotLiberoDataset(args.dataset_repo, config.action_horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    if not len(loader):
        raise ValueError("Dataset is smaller than --batch-size with drop_last=True")
    iterator = iter(loader)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_every < 1:
        raise ValueError("--save-every must be positive")

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("Install the training extra: uv sync --extra train") from exc
        wandb_config = {
            key: str(value) if isinstance(value, pathlib.Path) else value
            for key, value in vars(args).items()
        }
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )

    try:
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
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": args.learning_rate,
                "train/grad_norm": float(grad_norm),
            }
            if device.type == "cuda":
                metrics["train/gpu_peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            memory_message = (
                f" gpu_peak_memory_gib={metrics['train/gpu_peak_memory_gib']:.2f}"
                if "train/gpu_peak_memory_gib" in metrics
                else ""
            )
            print(f"step={step} loss={loss.item():.6f}{memory_message}", flush=True)
            should_evaluate = bool(args.eval_every and step % args.eval_every == 0)
            epoch_complete = step % len(loader) == 0
            final_step = step == args.steps
            periodic_save = step % args.save_every == 0
            should_save = periodic_save or final_step or should_evaluate or (args.train_only and epoch_complete)
            if should_save:
                resumable = not args.train_only or epoch_complete or final_step
                saved = save_checkpoint(
                    model,
                    optimizer,
                    step,
                    args.output_dir,
                    checkpoint,
                    include_optimizer=resumable,
                )
                print(f"saved={saved}", flush=True)
            if should_evaluate:
                # The evaluator loads a second copy of the checkpoint in its policy
                # server. Release references to the completed training graph first.
                del loss, observation, actions, batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                evaluation_started = time.monotonic()
                summary = evaluate_checkpoint(
                    saved,
                    args.tokenizer.expanduser().resolve(),
                    step,
                    args.output_dir,
                    # Do not resolve the venv's Python symlink: Python uses the
                    # invoked path to discover that venv's site-packages.
                    eval_python=args.eval_python.expanduser(),
                    openpi_root=args.openpi_root.expanduser().resolve(),
                    suite=args.eval_suite,
                    task_id=args.eval_task_id,
                    episodes=args.eval_episodes,
                    save_video=args.eval_save_video,
                    port=args.eval_port,
                    server_timeout=args.eval_server_timeout,
                    device=args.device,
                    pi05=args.pi05,
                )
                metrics.update(
                    {
                        "eval/success_rate": float(summary["success_rate"]),
                        "eval/successes": int(summary["successes"]),
                        "eval/episodes": int(summary["episodes"]),
                        "eval/elapsed_seconds": time.monotonic() - evaluation_started,
                    }
                )
                print(
                    f"eval_step={step} successes={summary['successes']}/{summary['episodes']} "
                    f"success_rate={summary['success_rate']:.6f}",
                    flush=True,
                )
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

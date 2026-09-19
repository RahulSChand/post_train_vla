"""Trajectory-budget sample-efficiency experiments for pi0 and pi0.5."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pathlib
import random
import shutil
import time
from dataclasses import dataclass

import numpy as np
import safetensors.torch
import torch
from torch.utils.data import DataLoader, Subset

from post_train_vla.finetune import LeRobotLiberoDataset, evaluate_checkpoint, save_checkpoint
from post_train_vla.models import Pi0, Pi0Config
from post_train_vla.torch_policy import find_norm_stats
from post_train_vla.transforms import LiberoTransforms

MANIFEST_VERSION = 1
ACTION_PARAMETER_PREFIXES = (
    "paligemma_with_expert.gemma_expert.",
    "action_",
    "state_proj",
    "time_mlp_",
)


@dataclass(frozen=True)
class EarlyStoppingState:
    best_successes: int
    best_epoch: int
    epochs_without_improvement: int


def _json_lines(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def create_or_validate_manifest(dataset_root: pathlib.Path, manifest_path: pathlib.Path, seed: int) -> dict:
    """Create one deterministic episode ordering, or validate an existing one."""
    dataset_root = dataset_root.resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    info_path = dataset_root / "meta" / "info.json"
    for required in (episodes_path, tasks_path, info_path):
        if not required.is_file():
            raise FileNotFoundError(f"Dataset metadata is missing: {required}")

    episodes = sorted(_json_lines(episodes_path), key=lambda item: int(item["episode_index"]))
    task_by_prompt = {item["task"]: int(item["task_index"]) for item in _json_lines(tasks_path)}
    episode_records = []
    for episode in episodes:
        prompts = episode.get("tasks", [])
        if len(prompts) != 1 or prompts[0] not in task_by_prompt:
            raise ValueError(f"Episode {episode.get('episode_index')} does not map to exactly one known task")
        episode_records.append(
            {
                "episode_index": int(episode["episode_index"]),
                "task_index": task_by_prompt[prompts[0]],
                "length": int(episode["length"]),
            }
        )

    metadata_digest = hashlib.sha256(episodes_path.read_bytes() + tasks_path.read_bytes()).hexdigest()
    ordered_indices = [item["episode_index"] for item in episode_records]
    random.Random(seed).shuffle(ordered_indices)
    expected = {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "selection": "global_without_replacement_python_random_v1",
        "dataset_metadata_sha256": metadata_digest,
        "total_episodes": len(episode_records),
        "ordered_episode_indices": ordered_indices,
        "episodes": episode_records,
    }

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != expected:
            raise ValueError(f"Existing manifest does not match seed={seed} and dataset metadata: {manifest_path}")
        return existing

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return expected


def manifest_sha256(manifest_path: pathlib.Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def frame_indices_for_episodes(dataset: LeRobotLiberoDataset, episode_indices: list[int]) -> list[int]:
    """Resolve complete local LeRobot episodes to their global frame indices."""
    local_dataset = dataset.dataset
    episodes = getattr(local_dataset, "_episodes", None)
    starts = getattr(local_dataset, "_starts", None)
    if episodes is None or starts is None:
        raise ValueError("Trajectory subsets currently require a local LeRobot v2.1 dataset directory")
    ranges = {
        int(episode["episode_index"]): (int(start), int(episode["length"]))
        for episode, start in zip(episodes, starts, strict=True)
    }
    missing = sorted(set(episode_indices).difference(ranges))
    if missing:
        raise ValueError(f"Manifest references episodes absent from the dataset: {missing}")
    frame_indices = []
    for episode_index in episode_indices:
        start, length = ranges[episode_index]
        frame_indices.extend(range(start, start + length))
    return frame_indices


def update_early_stopping(
    state: EarlyStoppingState | None,
    *,
    epoch: int,
    successes: int,
) -> EarlyStoppingState:
    if state is None or successes > state.best_successes:
        return EarlyStoppingState(successes, epoch, 0)
    return EarlyStoppingState(
        state.best_successes,
        state.best_epoch,
        state.epochs_without_improvement + 1,
    )


def should_stop_early(
    state: EarlyStoppingState,
    *,
    epoch: int,
    minimum_epochs: int,
    patience: int,
) -> bool:
    return epoch >= minimum_epochs and state.epochs_without_improvement >= patience


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_trainable_parameters(model: Pi0, *, freeze_vlm: bool) -> list[torch.nn.Parameter]:
    """Select either the full model or only the non-VLM action pathway for training."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not freeze_vlm or name.startswith(ACTION_PARAMETER_PREFIXES)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The requested freeze policy left no trainable parameters")
    return parameters


def _upload_and_verify(
    checkpoint: pathlib.Path,
    *,
    api,
    repo_id: str,
    path_in_repo: str,
    include_evaluation: bool,
) -> None:
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint),
        path_in_repo=path_in_repo,
        commit_message=f"Upload {path_in_repo}",
    )
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    required = {
        f"{path_in_repo}/model.safetensors",
        f"{path_in_repo}/config.json",
        f"{path_in_repo}/metadata.json",
        f"{path_in_repo}/trajectory_manifest.json",
        f"{path_in_repo}/assets/physical-intelligence/libero/norm_stats.json",
    }
    if include_evaluation:
        required.update(
            {
                f"{path_in_repo}/evaluation/summary.json",
                f"{path_in_repo}/evaluation/episodes.jsonl",
            }
        )
    missing = sorted(required.difference(remote_files))
    if missing:
        raise RuntimeError(f"Hugging Face upload verification failed; missing files: {missing}")


def _save_run_summary(run_dir: pathlib.Path, payload: dict) -> pathlib.Path:
    path = run_dir / "run_summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _build_model(args: argparse.Namespace, checkpoint: pathlib.Path) -> tuple[Pi0, list[torch.nn.Parameter]]:
    config = Pi0Config.from_checkpoint(checkpoint, pi05=args.model == "pi05")
    model = Pi0(config)
    safetensors.torch.load_model(model, str(checkpoint / "model.safetensors"), strict=True)
    parameters = configure_trainable_parameters(model, freeze_vlm=args.freeze_vlm)
    model.to(args.device).train()
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    return model, parameters


def _train_one_epoch(
    model: Pi0,
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    transforms: LiberoTransforms,
    *,
    device: str,
    extra_delta_actions: bool,
    gradient_accumulation_steps: int,
    starting_step: int,
) -> tuple[int, float, int]:
    model.train()
    global_step = starting_step
    epoch_loss = 0.0
    optimizer_steps = 0
    microbatch_count = len(loader)
    iterator = iter(loader)
    for group_start in range(0, microbatch_count, gradient_accumulation_steps):
        group_size = min(gradient_accumulation_steps, microbatch_count - group_start)
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for _ in range(group_size):
            batch = next(iterator)
            observation, actions = transforms.encode_training_batch(
                batch, device, extra_delta_actions=extra_delta_actions
            )
            loss = model.loss(observation, actions).mean()
            group_loss += float(loss.item()) / group_size
            (loss / group_size).backward()
            del loss, observation, actions, batch
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        global_step += 1
        optimizer_steps += 1
        epoch_loss += group_loss
        print(
            f"step={global_step} loss={group_loss:.6f} grad_norm={float(grad_norm):.6f}",
            flush=True,
        )
    # Free the final gradient tensors before the evaluation process loads a
    # second model copy on the same GPU. Adam state and model weights remain.
    optimizer.zero_grad(set_to_none=True)
    return global_step, epoch_loss / optimizer_steps, optimizer_steps


def run_budget(
    args: argparse.Namespace,
    *,
    budget: int,
    manifest: dict,
    manifest_path: pathlib.Path,
    api,
) -> dict:
    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = args.output_dir / f"trajectories-{budget:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_episodes = [int(value) for value in manifest["ordered_episode_indices"][:budget]]
    if len(selected_episodes) != budget:
        raise ValueError(f"Requested {budget} trajectories, but the manifest contains only {len(selected_episodes)}")

    _set_seed(args.seed)
    full_dataset = LeRobotLiberoDataset(
        str(args.dataset_root), Pi0Config.from_checkpoint(checkpoint, pi05=args.model == "pi05").action_horizon
    )
    frame_indices = frame_indices_for_episodes(full_dataset, selected_episodes)
    dataset = Subset(full_dataset, frame_indices)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        generator=generator,
    )
    if not len(loader):
        raise ValueError(f"The {budget}-trajectory dataset is empty")

    model, parameters = _build_model(args, checkpoint)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    transforms = LiberoTransforms(model.config, find_norm_stats(checkpoint), args.tokenizer)
    extra_delta_actions = args.model == "pi0"
    global_step = 0
    stopping_state = None
    epoch_records = []
    print(
        f"budget={budget} trajectories frames={len(dataset)} microbatch_size={args.batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch_size={args.batch_size * args.gradient_accumulation_steps} "
        f"optimizer_steps_per_epoch={math.ceil(len(loader) / args.gradient_accumulation_steps)} "
        f"freeze_vlm={args.freeze_vlm} "
        f"trainable_parameters={sum(parameter.numel() for parameter in parameters)} "
        f"total_parameters={sum(parameter.numel() for parameter in model.parameters())}",
        flush=True,
    )

    try:
        for epoch in range(1, args.max_epochs + 1):
            epoch_started = time.monotonic()
            global_step, train_loss, optimizer_steps = _train_one_epoch(
                model,
                parameters,
                optimizer,
                loader,
                transforms,
                device=args.device,
                extra_delta_actions=extra_delta_actions,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                starting_step=global_step,
            )
            checkpoint_dir = save_checkpoint(
                model,
                optimizer,
                global_step,
                run_dir / "checkpoints",
                checkpoint,
                include_optimizer=False,
            )
            # Full-model Adam training leaves a large CUDA allocator cache after
            # each epoch. Release only that unused cache so the separate policy
            # process has room to load the evaluation checkpoint; live model and
            # optimizer tensors remain on device and training continuity is kept.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_steps": int(optimizer_steps),
                "train_loss": train_loss,
                "elapsed_seconds": time.monotonic() - epoch_started,
            }
            if args.evaluate_after_epoch:
                summary = evaluate_checkpoint(
                    checkpoint_dir,
                    args.tokenizer.expanduser().resolve(),
                    global_step,
                    run_dir,
                    eval_python=args.eval_python.expanduser(),
                    openpi_root=args.openpi_root.expanduser().resolve(),
                    suite="libero_spatial",
                    task_id=0,
                    episodes=20,
                    save_video=False,
                    port=args.eval_port,
                    server_timeout=args.eval_server_timeout,
                    device=args.device,
                    pi05=args.model == "pi05",
                    eval_workers=args.eval_workers,
                    max_batch_size=args.eval_max_batch_size,
                    batch_wait_ms=args.eval_batch_wait_ms,
                )
                successes = int(summary["successes"])
                stopping_state = update_early_stopping(stopping_state, epoch=epoch, successes=successes)
                record.update(
                    {
                        "successes": successes,
                        "episodes": int(summary["episodes"]),
                        "success_rate": float(summary["success_rate"]),
                    }
                )
            epoch_records.append(record)
            metadata_path = checkpoint_dir / "metadata.json"
            checkpoint_metadata = json.loads(metadata_path.read_text())
            checkpoint_metadata.update(
                {
                    "epoch": epoch,
                    "trajectory_count": budget,
                    "selected_episode_indices": selected_episodes,
                    "trajectory_manifest_sha256": manifest_sha256(manifest_path),
                    "training": {
                        "model": args.model,
                        "seed": args.seed,
                        "batch_size": args.batch_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                        "learning_rate": args.learning_rate,
                        "full_model_finetune": not args.freeze_vlm,
                        "freeze_vlm": args.freeze_vlm,
                        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
                        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                        "optimizer_saved": False,
                    },
                    "evaluation": record if args.evaluate_after_epoch else None,
                }
            )
            metadata_path.write_text(json.dumps(checkpoint_metadata, indent=2, sort_keys=True) + "\n")
            if args.evaluate_after_epoch:
                evaluation_dir = checkpoint_dir / "evaluation"
                evaluation_dir.mkdir()
                rollout_dir = run_dir / "eval" / f"step_{global_step:06d}"
                shutil.copy2(rollout_dir / "summary.json", evaluation_dir / "summary.json")
                shutil.copy2(rollout_dir / "episodes.jsonl", evaluation_dir / "episodes.jsonl")
            shutil.copy2(manifest_path, checkpoint_dir / "trajectory_manifest.json")

            run_summary = {
                "model": args.model,
                "trajectory_count": budget,
                "selected_episode_indices": selected_episodes,
                "trajectory_manifest_sha256": manifest_sha256(manifest_path),
                "evaluate_after_epoch": args.evaluate_after_epoch,
                "best_epoch": stopping_state.best_epoch if stopping_state else None,
                "best_successes": stopping_state.best_successes if stopping_state else None,
                "epochs_without_improvement": stopping_state.epochs_without_improvement if stopping_state else None,
                "epochs": epoch_records,
            }
            summary_path = _save_run_summary(run_dir, run_summary)
            path_in_repo = f"trajectories-{budget:03d}/epoch-{epoch:03d}"
            _upload_and_verify(
                checkpoint_dir,
                api=api,
                repo_id=args.hf_repo_id,
                path_in_repo=path_in_repo,
                include_evaluation=args.evaluate_after_epoch,
            )
            api.upload_file(
                repo_id=args.hf_repo_id,
                repo_type="model",
                path_or_fileobj=str(summary_path),
                path_in_repo=f"trajectories-{budget:03d}/run_summary.json",
                commit_message=f"Update {budget}-trajectory run summary",
            )
            shutil.rmtree(checkpoint_dir)
            status = f"successes={successes}/20" if args.evaluate_after_epoch else "evaluation=skipped"
            print(f"uploaded={args.hf_repo_id}/{path_in_repo} deleted_local={checkpoint_dir} {status}", flush=True)
            if args.evaluate_after_epoch and should_stop_early(
                stopping_state,
                epoch=epoch,
                minimum_epochs=args.minimum_epochs,
                patience=args.patience,
            ):
                print(
                    f"early_stop budget={budget} epoch={epoch} best_epoch={stopping_state.best_epoch} "
                    f"best_successes={stopping_state.best_successes}/20",
                    flush=True,
                )
                break
        return run_summary
    finally:
        del optimizer, parameters, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi0", "pi05"), required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--trajectory-budgets", type=int, nargs="+", default=(5, 10, 15, 25, 50))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--minimum-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--freeze-vlm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze PaliGemma and train only the action expert plus action/time/state projections",
    )
    parser.add_argument(
        "--evaluate-after-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the 20-rollout task-0 evaluation and early-stopping check after each epoch",
    )
    parser.add_argument("--eval-workers", type=int, default=20)
    parser.add_argument("--eval-max-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-wait-ms", type=float, default=10.0)
    parser.add_argument("--eval-port", type=int, default=8765)
    parser.add_argument("--eval-server-timeout", type=float, default=600.0)
    parser.add_argument(
        "--eval-python",
        type=pathlib.Path,
        default=pathlib.Path("/root/openpi_easy/examples/libero/.venv/bin/python"),
    )
    parser.add_argument("--openpi-root", type=pathlib.Path, default=pathlib.Path("/root/openpi_easy"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.trajectory_budgets) < 1:
        raise ValueError("Trajectory budgets must be positive")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch size and gradient accumulation steps must be positive")
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.evaluate_after_epoch and (args.minimum_epochs < 1 or args.patience < 1 or args.max_epochs < args.minimum_epochs):
        raise ValueError("Require max_epochs >= minimum_epochs >= 1 and patience >= 1")
    if args.evaluate_after_epoch and not args.eval_python.is_file():
        raise FileNotFoundError(f"LIBERO evaluation Python not found: {args.eval_python}")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()

    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()
    print(f"huggingface_user={identity['name']}", flush=True)
    api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
    manifest = create_or_validate_manifest(args.dataset_root, args.manifest, args.seed)
    if max(args.trajectory_budgets) > manifest["total_episodes"]:
        raise ValueError("A trajectory budget exceeds the dataset's episode count")
    print(f"trajectory_manifest={args.manifest} sha256={manifest_sha256(args.manifest)}", flush=True)
    api.upload_file(
        repo_id=args.hf_repo_id,
        repo_type="model",
        path_or_fileobj=str(args.manifest),
        path_in_repo="trajectory_manifest.json",
        commit_message="Add shared trajectory-selection manifest",
    )

    results = []
    for budget in args.trajectory_budgets:
        results.append(run_budget(args, budget=budget, manifest=manifest, manifest_path=args.manifest, api=api))
    _save_run_summary(args.output_dir, {"model": args.model, "runs": results})


if __name__ == "__main__":
    main()

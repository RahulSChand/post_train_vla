"""Policy-agnostic LIBERO rollout loop."""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import pathlib
import re
import time
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from post_train_vla.libero_observation import make_policy_observation
from post_train_vla.policy import Policy

LOGGER = logging.getLogger(__name__)

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    suite: str = "libero_spatial"
    task_ids: tuple[int, ...] | None = None
    episodes_per_task: int = 50
    episode_offset: int = 0
    seed: int = 7
    wait_steps: int = 10
    replan_steps: int = 5
    resize_size: int = 224
    render_resolution: int = 256
    output_dir: pathlib.Path = pathlib.Path("outputs/libero")
    save_video: bool = False


class ActionChunker:
    def __init__(self, replan_steps: int) -> None:
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        self.replan_steps = replan_steps
        self._actions = collections.deque()

    def empty(self) -> bool:
        return not self._actions

    def add(self, actions: np.ndarray) -> None:
        chunk = np.asarray(actions)
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise ValueError(f"Expected [horizon, >=7] action chunk, got {chunk.shape}")
        if len(chunk) < self.replan_steps:
            raise ValueError(f"Policy horizon {len(chunk)} is shorter than replan_steps={self.replan_steps}")
        self._actions.extend(chunk[: self.replan_steps, :7])

    def pop(self) -> np.ndarray:
        return np.asarray(self._actions.popleft(), dtype=np.float32)


def _create_environment(task, resolution: int, seed: int):
    # Import lazily so preprocessing/tests do not require the legacy LIBERO environment.
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    environment = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=resolution, camera_widths=resolution)
    environment.seed(seed)
    return environment


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")[:120]


def _append_jsonl(path: pathlib.Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def evaluate(policy: Policy, config: EvalConfig, policy_metadata: dict | None = None) -> dict:
    from libero.libero import benchmark

    if config.suite not in MAX_STEPS:
        raise ValueError(f"Unknown suite {config.suite!r}; choose from {sorted(MAX_STEPS)}")
    if config.episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    if config.episode_offset < 0:
        raise ValueError("episode_offset must be non-negative")

    np.random.seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = config.output_dir / "videos"
    if config.save_video:
        videos_dir.mkdir(exist_ok=True)
    episodes_path = config.output_dir / "episodes.jsonl"

    suite = benchmark.get_benchmark_dict()[config.suite]()
    task_ids: Iterable[int] = config.task_ids if config.task_ids is not None else range(suite.n_tasks)
    results = []

    for task_id in tqdm(list(task_ids), desc="tasks"):
        if task_id < 0 or task_id >= suite.n_tasks:
            raise ValueError(f"task_id={task_id} is outside [0, {suite.n_tasks})")
        task = suite.get_task(task_id)
        prompt = str(task.language)
        initial_states = suite.get_task_init_states(task_id)
        episode_end = config.episode_offset + config.episodes_per_task
        if episode_end > len(initial_states):
            raise ValueError(
                f"Requested initial states [{config.episode_offset}, {episode_end}) but task {task_id} "
                f"only has {len(initial_states)} states"
            )

        environment = _create_environment(task, config.render_resolution, config.seed)
        try:
            for episode_index in tqdm(
                range(config.episode_offset, episode_end), desc=f"task {task_id}", leave=False
            ):
                started_at = time.monotonic()
                success = False
                error = None
                policy_steps = 0
                inference_calls = 0
                inference_ms = []
                frames = []
                policy.reset()
                try:
                    environment.reset()
                    observation = environment.set_init_state(initial_states[episode_index])
                    for _ in range(config.wait_steps):
                        observation, _, done, _ = environment.step(DUMMY_ACTION.tolist())
                        if done:
                            success = True
                            break

                    chunker = ActionChunker(config.replan_steps)
                    while not success and policy_steps < MAX_STEPS[config.suite]:
                        policy_observation = make_policy_observation(observation, prompt, config.resize_size)
                        if config.save_video:
                            frames.append(policy_observation["observation/image"])
                        if chunker.empty():
                            inference_started = time.monotonic()
                            response = policy.infer(policy_observation)
                            inference_ms.append((time.monotonic() - inference_started) * 1000.0)
                            inference_calls += 1
                            chunker.add(response["actions"])
                        observation, _, done, _ = environment.step(chunker.pop().tolist())
                        policy_steps += 1
                        success = bool(done)
                except Exception as exc:  # Record the failed episode, then continue the benchmark.
                    LOGGER.exception("Episode failed for task=%s episode=%s", task_id, episode_index)
                    error = f"{type(exc).__name__}: {exc}"

                result = {
                    "suite": config.suite,
                    "task_id": task_id,
                    "task": prompt,
                    "episode_index": episode_index,
                    "seed": config.seed,
                    "success": success,
                    "policy_steps": policy_steps,
                    "inference_calls": inference_calls,
                    "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "error": error,
                }
                results.append(result)
                _append_jsonl(episodes_path, result)

                if config.save_video and frames:
                    outcome = "success" if success else "failure"
                    filename = f"task_{task_id:02d}_episode_{episode_index:03d}_{_safe_name(prompt)}_{outcome}.mp4"
                    imageio.mimwrite(videos_dir / filename, frames, fps=10)
        finally:
            environment.close()

    successes = sum(result["success"] for result in results)
    summary = {
        "suite": config.suite,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "policy_metadata": policy_metadata or {},
        "config": {**dataclasses.asdict(config), "output_dir": str(config.output_dir)},
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    LOGGER.info("Finished %d episodes: %d successes (%.1f%%)", len(results), successes, 100 * summary["success_rate"])
    return summary

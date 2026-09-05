"""Standalone LIBERO normalization and model input/output transforms."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from post_train_vla.models.configuration import Pi0Config
from post_train_vla.models.observation import Observation
from post_train_vla.models.tokenizer import PaligemmaTokenizer


class LiberoTransforms:
    def __init__(self, config: Pi0Config, norm_stats_path: pathlib.Path, tokenizer_path: pathlib.Path) -> None:
        self.config = config
        self.tokenizer = PaligemmaTokenizer(tokenizer_path, max_len=config.max_token_len)
        payload = json.loads(norm_stats_path.read_text())
        self.stats = payload.get("norm_stats", payload)
        if "state" not in self.stats or "actions" not in self.stats:
            raise ValueError(f"Normalization stats must contain state and actions: {norm_stats_path}")

    def _normalize(self, value: np.ndarray, key: str) -> np.ndarray:
        stats = self.stats[key]
        if self.config.pi05:
            low = np.asarray(stats["q01"], dtype=np.float32)[: value.shape[-1]]
            high = np.asarray(stats["q99"], dtype=np.float32)[: value.shape[-1]]
            return (value - low) / (high - low + 1e-6) * 2.0 - 1.0
        mean = np.asarray(stats["mean"], dtype=np.float32)[: value.shape[-1]]
        std = np.asarray(stats["std"], dtype=np.float32)[: value.shape[-1]]
        return (value - mean) / (std + 1e-6)

    def _unnormalize(self, value: np.ndarray, key: str) -> np.ndarray:
        stats = self.stats[key]
        dimensions = len(stats["mean"])
        output = value.copy()
        if self.config.pi05:
            low = np.asarray(stats["q01"], dtype=np.float32)
            high = np.asarray(stats["q99"], dtype=np.float32)
            output[..., :dimensions] = (output[..., :dimensions] + 1.0) / 2.0 * (high - low + 1e-6) + low
        else:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            output[..., :dimensions] = output[..., :dimensions] * (std + 1e-6) + mean
        return output

    def encode(self, raw: dict, device: torch.device | str) -> tuple[Observation, np.ndarray]:
        state = np.asarray(raw["observation/state"], dtype=np.float32)
        normalized_state = self._normalize(state, "state")
        padded_state = np.pad(normalized_state, (0, self.config.action_dim - len(normalized_state)))
        tokens, token_mask = self.tokenizer.tokenize(str(raw["prompt"]), normalized_state if self.config.pi05 else None)
        base = np.asarray(raw["observation/image"], dtype=np.uint8)
        wrist = np.asarray(raw["observation/wrist_image"], dtype=np.uint8)

        def image_tensor(image: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(image.copy()).float().div(255.0).mul(2.0).sub(1.0).unsqueeze(0).to(device)

        observation = Observation(
            images={
                "base_0_rgb": image_tensor(base),
                "left_wrist_0_rgb": image_tensor(wrist),
                "right_wrist_0_rgb": image_tensor(np.zeros_like(base)),
            },
            image_masks={
                "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
                "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
            },
            state=torch.from_numpy(padded_state).float().unsqueeze(0).to(device),
            tokenized_prompt=torch.from_numpy(tokens).long().unsqueeze(0).to(device),
            tokenized_prompt_mask=torch.from_numpy(token_mask).bool().unsqueeze(0).to(device),
        )
        return observation, state

    def encode_batch(
        self, raw_observations: list[dict], device: torch.device | str
    ) -> tuple[Observation, list[np.ndarray]]:
        """Encode independent policy requests as one model batch."""
        if not raw_observations:
            raise ValueError("Cannot encode an empty observation batch")
        encoded = [self.encode(raw, "cpu") for raw in raw_observations]
        observations, states = zip(*encoded)
        batch = Observation(
            images={
                key: torch.cat([observation.images[key] for observation in observations], dim=0)
                for key in observations[0].images
            },
            image_masks={
                key: torch.cat([observation.image_masks[key] for observation in observations], dim=0)
                for key in observations[0].image_masks
            },
            state=torch.cat([observation.state for observation in observations], dim=0),
            tokenized_prompt=torch.cat([observation.tokenized_prompt for observation in observations], dim=0),
            tokenized_prompt_mask=torch.cat(
                [observation.tokenized_prompt_mask for observation in observations], dim=0
            ),
        ).to(device)
        return batch, list(states)

    def encode_training_batch(
        self, raw: dict, device: torch.device | str, *, extra_delta_actions: bool = False
    ) -> tuple[Observation, torch.Tensor]:
        """Encode a batched LIBERO/LeRobot sample for ``Pi0.loss``.

        The expected fields are the same canonical fields used by OpenPI's
        ``LeRobotLiberoDataConfig``: ``image``, ``wrist_image``, ``state``,
        ``action``, and ``prompt``. Images may be HWC numpy arrays or CHW torch
        tensors as returned by LeRobot.
        """
        def as_numpy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def images(value):
            array = as_numpy(value)
            if array.ndim != 4:
                raise ValueError(f"Expected batched images, got {array.shape}")
            if array.shape[1] == 3:  # LeRobot returns BCHW float images.
                array = np.moveaxis(array, 1, -1)
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return torch.from_numpy(array.copy()).float().div(255.0).mul(2.0).sub(1.0).to(device)

        state = as_numpy(raw["state"]).astype(np.float32)
        actions = as_numpy(raw["action"]).astype(np.float32)
        if state.ndim != 2:
            raise ValueError(f"Expected state [batch, dim], got {state.shape}")
        if actions.ndim != 3:
            raise ValueError(f"Expected action [batch, horizon, dim], got {actions.shape}")
        if actions.shape[1] != self.config.action_horizon:
            raise ValueError(
                f"Expected {self.config.action_horizon} actions per sample, got {actions.shape[1]}"
            )
        if actions.shape[-1] > self.config.action_dim:
            raise ValueError(f"Action dimension {actions.shape[-1]} exceeds model dimension {self.config.action_dim}")

        if extra_delta_actions:
            # Match OpenPI's legacy pi0_libero configuration: target Cartesian
            # actions are represented relative to the current end-effector pose.
            actions = actions.copy()
            dimensions = min(6, actions.shape[-1], state.shape[-1])
            actions[..., :dimensions] -= state[:, None, :dimensions]
        normalized_state = self._normalize(state, "state")
        normalized_actions = self._normalize(actions, "actions")
        padded_state = np.pad(normalized_state, ((0, 0), (0, self.config.action_dim - state.shape[-1])))
        padded_actions = np.pad(
            normalized_actions,
            ((0, 0), (0, 0), (0, self.config.action_dim - actions.shape[-1])),
        )
        prompts = raw["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens, masks = zip(
            *(self.tokenizer.tokenize(str(prompt), state_item if self.config.pi05 else None)
              for prompt, state_item in zip(prompts, normalized_state, strict=True))
        )
        base = images(raw["image"])
        wrist = images(raw["wrist_image"])
        batch = base.shape[0]
        observation = Observation(
            images={
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": torch.zeros_like(base),
            },
            image_masks={
                "base_0_rgb": torch.ones(batch, dtype=torch.bool, device=device),
                "left_wrist_0_rgb": torch.ones(batch, dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros(batch, dtype=torch.bool, device=device),
            },
            state=torch.from_numpy(padded_state).float().to(device),
            tokenized_prompt=torch.from_numpy(np.stack(tokens)).long().to(device),
            tokenized_prompt_mask=torch.from_numpy(np.stack(masks)).bool().to(device),
        )
        return observation, torch.from_numpy(padded_actions).float().to(device)

    def decode_actions(self, normalized_actions: np.ndarray, original_state: np.ndarray) -> np.ndarray:
        actions = self._unnormalize(normalized_actions, "actions")
        if not self.config.pi05:
            actions[..., :6] += original_state[:6]
        return np.asarray(actions[..., :7], dtype=np.float32)

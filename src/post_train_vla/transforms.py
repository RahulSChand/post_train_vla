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

    def decode_actions(self, normalized_actions: np.ndarray, original_state: np.ndarray) -> np.ndarray:
        actions = self._unnormalize(normalized_actions, "actions")
        if not self.config.pi05:
            actions[..., :6] += original_state[:6]
        return np.asarray(actions[..., :7], dtype=np.float32)

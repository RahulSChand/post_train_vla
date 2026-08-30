"""Standalone checkpoint loading and Torch policy inference."""

from __future__ import annotations

import pathlib
import time

import numpy as np
import safetensors.torch
import torch

from post_train_vla.models import Pi0, Pi0Config
from post_train_vla.transforms import LiberoTransforms


def find_norm_stats(checkpoint: pathlib.Path) -> pathlib.Path:
    matches = list((checkpoint / "assets").glob("**/norm_stats.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one norm_stats.json below {checkpoint / 'assets'}, found {matches}")
    return matches[0]


class TorchPolicy:
    def __init__(
        self,
        checkpoint: pathlib.Path,
        tokenizer_path: pathlib.Path,
        *,
        device: str = "cuda",
        pi05: bool = False,
        compile_model: bool = False,
    ) -> None:
        self.checkpoint = checkpoint.expanduser().resolve()
        self.device = torch.device(device)
        self.config = Pi0Config.from_checkpoint(self.checkpoint, pi05=pi05)
        self.model = Pi0(self.config)
        weights = self.checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Converted Torch checkpoint not found: {weights}")
        safetensors.torch.load_model(self.model, str(weights), strict=True)
        self.model.to(self.device).eval()
        if compile_model:
            self.model.sample_actions = torch.compile(self.model.sample_actions, mode="max-autotune")
        self.transforms = LiberoTransforms(self.config, find_norm_stats(self.checkpoint), tokenizer_path)
        self.metadata = {
            "backend": "pytorch-standalone",
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "model": "pi0.5" if pi05 else "pi0",
        }

    @torch.no_grad()
    def infer(self, raw_observation: dict, *, noise: np.ndarray | None = None) -> dict:
        observation, original_state = self.transforms.encode(raw_observation, self.device)
        noise_tensor = None if noise is None else torch.from_numpy(noise).float().unsqueeze(0).to(self.device)
        started = time.monotonic()
        actions = self.model.sample_actions(observation, noise=noise_tensor)
        inference_ms = (time.monotonic() - started) * 1000.0
        normalized = actions[0].detach().float().cpu().numpy()
        return {
            "actions": self.transforms.decode_actions(normalized, original_state),
            "policy_timing": {"infer_ms": inference_ms},
        }

    def reset(self) -> None:
        return None

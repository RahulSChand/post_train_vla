"""Tensor container used by the standalone policy and model."""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class Observation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor

    def to(self, device: torch.device | str) -> Observation:
        return Observation(
            images={key: value.to(device) for key, value in self.images.items()},
            image_masks={key: value.to(device) for key, value in self.image_masks.items()},
            state=self.state.to(device),
            tokenized_prompt=self.tokenized_prompt.to(device),
            tokenized_prompt_mask=self.tokenized_prompt_mask.to(device),
        )

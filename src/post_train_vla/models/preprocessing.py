"""Inference-only Torch preprocessing derived from OpenPI."""

from __future__ import annotations

import torch
from torch.nn import functional

from post_train_vla.models.observation import Observation

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)


def preprocess_observation(observation: Observation) -> Observation:
    missing = set(IMAGE_KEYS).difference(observation.images)
    if missing:
        raise ValueError(f"Missing model images: {sorted(missing)}")

    images = {}
    masks = {}
    for key in IMAGE_KEYS:
        image = observation.images[key]
        if image.ndim != 4:
            raise ValueError(f"Expected batched image for {key}, got {tuple(image.shape)}")
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[-2:] != IMAGE_RESOLUTION:
            image = functional.interpolate(image, size=IMAGE_RESOLUTION, mode="bilinear", align_corners=False)
        images[key] = image
        masks[key] = observation.image_masks.get(key, torch.ones(image.shape[0], dtype=torch.bool, device=image.device))
    return Observation(
        images=images,
        image_masks=masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )

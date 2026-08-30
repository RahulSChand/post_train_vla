"""LIBERO observation preprocessing matching OpenPI's training/evaluation path."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / denominator).astype(np.float32)


def resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {array.shape}")
    if array.shape[:2] == (height, width):
        return np.ascontiguousarray(array)

    pil_image = Image.fromarray(array)
    current_width, current_height = pil_image.size
    ratio = max(current_width / width, current_height / height)
    resized_width = int(current_width / ratio)
    resized_height = int(current_height / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    output = Image.new(resized.mode, (width, height), 0)
    output.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(output, dtype=np.uint8)


def make_policy_observation(observation: dict, prompt: str, resize_size: int = 224) -> dict:
    base_image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            quaternion_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected LIBERO state shape (8,), got {state.shape}")
    return {
        "observation/image": resize_with_pad(base_image, resize_size, resize_size),
        "observation/wrist_image": resize_with_pad(wrist_image, resize_size, resize_size),
        "observation/state": state,
        "prompt": str(prompt),
    }

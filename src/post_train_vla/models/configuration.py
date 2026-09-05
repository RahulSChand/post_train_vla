"""Small, JAX-free configuration objects for pi0/pi0.5."""

from __future__ import annotations

import dataclasses
import json
import pathlib


@dataclasses.dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


GEMMA_VARIANTS = {
    "gemma_300m": GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    "gemma_2b": GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
}


@dataclasses.dataclass(frozen=True)
class Pi0Config:
    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    pi05: bool = False
    max_token_len: int | None = None
    dtype: str = "bfloat16"
    num_inference_steps: int = 10
    compile_mode: str | None = None
    # Match OpenPI's low-memory pi0 configuration when both are set: rank 16
    # on the 2B PaliGemma stream and rank 32 on the 300M action expert.
    paligemma_lora_rank: int | None = None
    action_expert_lora_rank: int | None = None

    def __post_init__(self) -> None:
        if self.paligemma_variant not in GEMMA_VARIANTS:
            raise ValueError(f"Unsupported PaliGemma variant: {self.paligemma_variant}")
        if self.action_expert_variant not in GEMMA_VARIANTS:
            raise ValueError(f"Unsupported action expert variant: {self.action_expert_variant}")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        for name in ("paligemma_lora_rank", "action_expert_lora_rank"):
            rank = getattr(self, name)
            if rank is not None and rank < 1:
                raise ValueError(f"{name} must be positive when set, got {rank}")

    @classmethod
    def from_checkpoint(cls, checkpoint: pathlib.Path, *, pi05: bool | None = None) -> Pi0Config:
        config_path = checkpoint / "config.json"
        values = json.loads(config_path.read_text()) if config_path.is_file() else {}
        inferred_pi05 = pi05 if pi05 is not None else "pi05" in checkpoint.name.lower()
        return cls(
            action_dim=int(values.get("action_dim", 32)),
            action_horizon=int(values.get("action_horizon", 50)),
            paligemma_variant=values.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=values.get("action_expert_variant", "gemma_300m"),
            pi05=inferred_pi05,
            dtype=values.get("precision", "bfloat16"),
            paligemma_lora_rank=values.get("paligemma_lora_rank"),
            action_expert_lora_rank=values.get("action_expert_lora_rank"),
        )

    @property
    def paligemma(self) -> GemmaConfig:
        return GEMMA_VARIANTS[self.paligemma_variant]

    @property
    def action_expert(self) -> GemmaConfig:
        return GEMMA_VARIANTS[self.action_expert_variant]

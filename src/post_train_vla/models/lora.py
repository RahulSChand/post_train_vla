"""Small, dependency-free LoRA layers used by the standalone Torch model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class LoRALinear(nn.Linear):
    """A linear layer with a trainable low-rank update.

    The original ``weight`` and ``bias`` names are deliberately retained.  This
    makes a LoRA checkpoint a normal standalone model checkpoint plus two
    clearly named adapter tensors, instead of requiring PEFT at serving time.
    """

    def __init__(self, source: nn.Linear, *, rank: int, alpha: float | None = None) -> None:
        if rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        super().__init__(source.in_features, source.out_features, bias=source.bias is not None)
        self.weight = source.weight
        if source.bias is not None:
            self.bias = source.bias
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / rank
        self.lora_a = nn.Parameter(
            torch.empty(rank, source.in_features, device=source.weight.device, dtype=source.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.empty(source.out_features, rank, device=source.weight.device, dtype=source.weight.dtype)
        )
        # OpenPI's JAX reference initializes both low-rank factors from N(0,
        # 0.01), rather than using the common zero-B initialization.
        nn.init.normal_(self.lora_a, std=0.01)
        nn.init.normal_(self.lora_b, std=0.01)

    @classmethod
    def from_linear(cls, source: nn.Linear, *, rank: int, alpha: float | None = None) -> "LoRALinear":
        return cls(source, rank=rank, alpha=alpha)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = functional.linear(inputs, self.weight, self.bias)
        update = functional.linear(functional.linear(inputs, self.lora_a), self.lora_b)
        return base + update * self.scaling


def replace_gemma_linears(module: nn.Module, *, rank: int, alpha: float | None = None) -> list[str]:
    """Install LoRA in Gemma Q/K/V/O and gate/up/down projections below module."""
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in parent.named_children():
            path = f"{prefix}.{name}" if prefix else name
            if name in targets:
                if isinstance(child, LoRALinear):
                    continue
                if not isinstance(child, nn.Linear):
                    raise TypeError(f"Expected Gemma projection {path} to be nn.Linear, got {type(child).__name__}")
                setattr(parent, name, LoRALinear.from_linear(child, rank=rank, alpha=alpha))
                replaced.append(path)
            else:
                visit(child, path)

    visit(module, "")
    return replaced

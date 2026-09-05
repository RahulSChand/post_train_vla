import json

import torch
from torch import nn

from post_train_vla.finetune import configure_lora_trainable_parameters
from post_train_vla.models.configuration import Pi0Config
from post_train_vla.models.lora import LoRALinear, replace_gemma_linears


class _ToyGemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


def test_lora_replaces_openpi_projection_set_and_preserves_base_output():
    torch.manual_seed(0)
    module = _ToyGemma()
    inputs = torch.randn(2, 4)
    original = module.q_proj(inputs)

    replaced = replace_gemma_linears(module, rank=3, alpha=3)

    assert len(replaced) == 7
    assert isinstance(module.q_proj, LoRALinear)
    assert module.q_proj.rank == 3
    # Turning adapters off must exactly recover the converted base projection.
    with torch.no_grad():
        module.q_proj.lora_a.zero_()
        module.q_proj.lora_b.zero_()
    torch.testing.assert_close(module.q_proj(inputs), original)


def test_lora_config_round_trip(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"paligemma_lora_rank": 16, "action_expert_lora_rank": 32})
    )

    config = Pi0Config.from_checkpoint(tmp_path)

    assert config.paligemma_lora_rank == 16
    assert config.action_expert_lora_rank == 32


def test_lora_freeze_policy_keeps_only_adapters_in_gemma_backbones():
    model = nn.Module()
    model.paligemma_with_expert = nn.Module()
    model.paligemma_with_expert.paligemma = nn.Module()
    model.paligemma_with_expert.paligemma.model = nn.Module()
    model.paligemma_with_expert.paligemma.model.language_model = _ToyGemma()
    model.paligemma_with_expert.paligemma.lm_head = nn.Linear(4, 4)
    model.paligemma_with_expert.gemma_expert = nn.Module()
    model.paligemma_with_expert.gemma_expert.model = _ToyGemma()
    model.paligemma_with_expert.gemma_expert.lm_head = nn.Linear(4, 4)
    model.action_out_proj = nn.Linear(4, 4)
    replace_gemma_linears(model.paligemma_with_expert.paligemma.model.language_model, rank=2)
    replace_gemma_linears(model.paligemma_with_expert.gemma_expert.model, rank=2)

    trainable = configure_lora_trainable_parameters(model)

    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert "action_out_proj.weight" in names
    assert "paligemma_with_expert.paligemma.lm_head.weight" not in names
    assert "paligemma_with_expert.gemma_expert.lm_head.weight" not in names
    assert all(".lora_" in name or name.startswith("action_out_proj") for name in names)

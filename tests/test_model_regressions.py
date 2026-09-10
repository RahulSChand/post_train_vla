"""Numerical regressions for checkpoint loading and checkpointed backward."""

import copy
import json

import pytest
import safetensors.torch
import torch
from torch import nn
from transformers import GemmaConfig, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

from post_train_vla.models.gemma import PiGemmaModel
from post_train_vla.models.lora import replace_gemma_linears
from post_train_vla.models.paligemma_expert import PaliGemmaWithExpert
import post_train_vla.torch_policy as torch_policy


def _gemma_config(*, pi05=False):
    config = GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        use_adarms=pi05,
        adarms_cond_dim=16 if pi05 else None,
    )
    config._attn_implementation = "eager"
    return config


class _SmallPi0(nn.Module):
    """Real position/attention modules with small dimensions, using the actual loader."""

    def __init__(self, config):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        backbone = self.paligemma_with_expert
        backbone.paligemma = nn.Module()
        backbone.paligemma.model = nn.Module()
        paligemma = backbone.paligemma
        paligemma.model.language_model = PiGemmaModel(_gemma_config())
        paligemma.lm_head = nn.Linear(16, 32, bias=False)
        paligemma.lm_head.weight = paligemma.model.language_model.embed_tokens.weight
        paligemma.model.vision_tower = nn.Module()
        paligemma.model.vision_tower.vision_model = nn.Module()
        paligemma.model.vision_tower.vision_model.embeddings = SiglipVisionEmbeddings(
            SiglipVisionConfig(hidden_size=16, image_size=8, patch_size=4)
        )
        backbone.gemma_expert = nn.Module()
        backbone.gemma_expert.model = PiGemmaModel(_gemma_config(pi05=config.pi05))
        self.to(dtype=getattr(torch, config.dtype))

    def to_empty(self, **kwargs):
        result = super().to_empty(**kwargs)
        # Make missing initialization fail deterministically, irrespective of
        # whether the allocator happens to return previously correct values.
        for buffer in self.buffers():
            buffer.fill_(float("nan") if buffer.is_floating_point() else -1)
        return result


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("pi05", [False, True])
def test_policy_meta_load_matches_normal_initialization(tmp_path, monkeypatch, precision, pi05):
    monkeypatch.setattr(torch_policy, "Pi0", _SmallPi0)
    monkeypatch.setattr(torch_policy, "LiberoTransforms", lambda *args: None)
    monkeypatch.setattr(torch_policy, "find_norm_stats", lambda path: path / "norm_stats.json")
    (tmp_path / "config.json").write_text(json.dumps({"precision": precision}))
    config = torch_policy.Pi0Config.from_checkpoint(tmp_path, pi05=pi05)
    torch.manual_seed(23)
    reference = _SmallPi0(config).eval()
    safetensors.torch.save_model(reference, str(tmp_path / "model.safetensors"))

    policy = torch_policy.TorchPolicy(tmp_path, tmp_path / "tokenizer.model", device="cpu", pi05=pi05)
    expected_buffers = dict(reference.named_buffers())
    assert len(expected_buffers) == 3
    assert not set(expected_buffers).intersection(reference.state_dict())

    # Exercise initial loading and the hot-load path used by checkpoint sweeps.
    for reload in (False, True):
        if reload:
            policy.load_checkpoint(tmp_path)
        for name, actual in policy.model.named_buffers():
            torch.testing.assert_close(actual, expected_buffers[name], rtol=0, atol=0)
        for name, actual in policy.model.state_dict().items():
            torch.testing.assert_close(actual, reference.state_dict()[name], rtol=0, atol=0)
        backbone = policy.model.paligemma_with_expert
        assert backbone.paligemma.lm_head.weight is backbone.paligemma.model.language_model.embed_tokens.weight
        for model in (backbone.paligemma.model.language_model, backbone.gemma_expert.model):
            assert model.rotary_emb.original_inv_freq is model.rotary_emb.inv_freq

        # Check the positional effect on outputs, not only stored buffer values.
        images = torch.randn(1, 3, 8, 8)
        expected_vision = reference.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings
        actual_vision = backbone.paligemma.model.vision_tower.vision_model.embeddings
        torch.testing.assert_close(actual_vision(images), expected_vision(images), rtol=0, atol=0)
        tokens = torch.randn(1, 4, 16, dtype=getattr(torch, precision))
        positions = torch.arange(4).unsqueeze(0)
        mask = torch.zeros(1, 1, 4, 4)
        expected_lm = reference.paligemma_with_expert.paligemma.model.language_model
        expected = expected_lm(inputs_embeds=tokens, position_ids=positions, attention_mask=mask, use_cache=False)
        actual = backbone.paligemma.model.language_model(
            inputs_embeds=tokens, position_ids=positions, attention_mask=mask, use_cache=False
        )
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)


@pytest.mark.parametrize("lora", [False, True])
def test_checkpointed_joint_backward_matches_eager(lora):
    torch.manual_seed(42)
    model = PaliGemmaWithExpert.__new__(PaliGemmaWithExpert)
    nn.Module.__init__(model)
    model.paligemma = nn.Module()
    model.paligemma.model = nn.Module()
    model.paligemma.model.language_model = PiGemmaModel(_gemma_config())
    model.gemma_expert = nn.Module()
    model.gemma_expert.model = PiGemmaModel(_gemma_config())
    model.float().train()
    if lora:
        for stream in (model.paligemma.model.language_model, model.gemma_expert.model):
            replace_gemma_linears(stream, rank=2)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora_" in name
    checkpointed = copy.deepcopy(model)
    checkpointed.paligemma.model.language_model.gradient_checkpointing = True
    checkpointed.gemma_expert.model.gradient_checkpointing = True

    prefix = torch.randn(2, 4, 16)
    suffix = torch.randn(2, 3, 16)
    target = torch.randn(2, 3, 16)
    mask = torch.zeros(2, 1, 7, 7)
    mask[:, :, :4, 4:] = torch.finfo(torch.float32).min
    positions = torch.arange(7).unsqueeze(0).expand(2, -1)

    def backward(current):
        inputs = [prefix.clone().requires_grad_(), suffix.clone().requires_grad_()]
        outputs, _ = current(
            attention_mask=mask, position_ids=positions, inputs_embeds=inputs, use_cache=False
        )
        loss = (outputs[1] - target).square().mean()
        loss.backward()
        gradients = {name: p.grad.clone() for name, p in current.named_parameters() if p.grad is not None}
        return loss.detach(), gradients, [value.grad for value in inputs]

    expected_loss, expected_gradients, expected_inputs = backward(model)
    actual_loss, actual_gradients, actual_inputs = backward(checkpointed)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert actual_gradients.keys() == expected_gradients.keys()
    assert any("layers.0." in name and torch.count_nonzero(grad) for name, grad in expected_gradients.items())
    for name in expected_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], rtol=0, atol=0, msg=name)
    for actual, expected in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

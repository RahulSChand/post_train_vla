import importlib.util
import pathlib
import sys

import torch
from transformers import GemmaConfig
from transformers.models.gemma import modeling_gemma

from post_train_vla.models.configuration import Pi0Config
from post_train_vla.models.gemma import PiGemmaModel, PiGemmaRMSNorm


def _tiny_config(*, use_adarms=False):
    config = GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        use_adarms=use_adarms,
        adarms_cond_dim=16 if use_adarms else None,
    )
    config._attn_implementation = "eager"
    return config


def _load_openpi_reference_module():
    source = (
        pathlib.Path(__file__).parents[2]
        / "openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py"
    )
    name = "transformers.models.gemma._openpi_reference_modeling_gemma"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_import_does_not_patch_transformers():
    assert modeling_gemma.GemmaModel is not PiGemmaModel
    assert modeling_gemma.GemmaRMSNorm is not PiGemmaRMSNorm


def test_adaptive_rms_norm_parameter_contract():
    standard = PiGemmaRMSNorm(16)
    adaptive = PiGemmaRMSNorm(16, cond_dim=16)
    assert set(standard.state_dict()) == {"weight"}
    assert set(adaptive.state_dict()) == {"dense.weight", "dense.bias"}
    output, gate = adaptive(torch.randn(2, 3, 16), torch.randn(2, 16))
    assert output.shape == (2, 3, 16)
    assert gate.shape == (2, 1, 16)


def test_prefix_cache_is_not_mutated_by_suffix():
    model = PiGemmaModel(_tiny_config()).eval()
    prefix = torch.randn(1, 4, 16)
    prefix_mask = torch.zeros(1, 1, 4, 4)
    prefix_positions = torch.arange(4).unsqueeze(0)
    prefix_output = model(
        inputs_embeds=prefix,
        attention_mask=prefix_mask,
        position_ids=prefix_positions,
        use_cache=True,
    )
    cache = prefix_output.past_key_values
    assert cache.get_seq_length() == 4

    suffix = torch.randn(1, 2, 16)
    suffix_mask = torch.zeros(1, 1, 2, 6)
    suffix_positions = torch.arange(4, 6).unsqueeze(0)
    output = model(
        inputs_embeds=suffix,
        attention_mask=suffix_mask,
        position_ids=suffix_positions,
        past_key_values=cache,
        use_cache=False,
    )
    assert output.last_hidden_state.shape == (1, 2, 16)
    assert cache.get_seq_length() == 4


def test_pi05_configuration_defaults():
    assert Pi0Config(pi05=True).max_token_len == 200
    assert Pi0Config(pi05=False).max_token_len == 48


def test_small_gemma_matches_openpi_reference():
    reference_module = _load_openpi_reference_module()
    config = _tiny_config()
    torch.manual_seed(3)
    reference = reference_module.GemmaModel(config).eval()
    standalone = PiGemmaModel(config).eval()
    standalone.load_state_dict(reference.state_dict(), strict=True)

    embeddings = torch.randn(1, 4, 16)
    mask = torch.zeros(1, 1, 4, 4)
    positions = torch.arange(4).unsqueeze(0)
    expected = reference(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions, use_cache=True)
    actual = standalone(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions, use_cache=True)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
    for expected_layer, actual_layer in zip(expected.past_key_values, actual.past_key_values):
        torch.testing.assert_close(actual_layer[0], expected_layer[0])
        torch.testing.assert_close(actual_layer[1], expected_layer[1])


def test_small_adaptive_gemma_matches_openpi_reference():
    reference_module = _load_openpi_reference_module()
    config = _tiny_config(use_adarms=True)
    torch.manual_seed(4)
    reference = reference_module.GemmaModel(config).eval()
    standalone = PiGemmaModel(config).eval()
    standalone.load_state_dict(reference.state_dict(), strict=True)

    embeddings = torch.randn(2, 3, 16)
    condition = torch.randn(2, 16)
    mask = torch.zeros(2, 1, 3, 3)
    positions = torch.arange(3).unsqueeze(0).expand(2, -1)
    expected = reference(
        inputs_embeds=embeddings,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        adarms_cond=condition,
    )
    actual = standalone(
        inputs_embeds=embeddings,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        adarms_cond=condition,
    )
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)

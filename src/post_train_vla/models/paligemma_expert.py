"""PaliGemma backbone plus a separate Gemma action expert.

This is an inference-focused adaptation of OpenPI's Apache-2.0 Torch model.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING

from post_train_vla.models.configuration import GemmaConfig
from post_train_vla.models.gemma import PiGemmaModel
from post_train_vla.models.vision import embed_image_without_hf_scaling, install_pi_vision_forward

VOCAB_SIZE = 257_152


def _make_gemma_config(config: GemmaConfig, *, use_adarms: bool):
    result = CONFIG_MAPPING["gemma"](
        head_dim=config.head_dim,
        hidden_size=config.width,
        intermediate_size=config.mlp_dim,
        num_attention_heads=config.num_heads,
        num_hidden_layers=config.depth,
        num_key_value_heads=config.num_kv_heads,
        vocab_size=VOCAB_SIZE,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="float32",
        use_adarms=use_adarms,
        adarms_cond_dim=config.width if use_adarms else None,
    )
    result._attn_implementation = "eager"
    return result


class PaliGemmaWithExpert(nn.Module):
    def __init__(
        self,
        paligemma_config: GemmaConfig,
        action_expert_config: GemmaConfig,
        *,
        expert_use_adarms: bool,
        precision: str = "bfloat16",
    ) -> None:
        super().__init__()
        vlm_config = CONFIG_MAPPING["paligemma"]()
        vlm_config._vocab_size = VOCAB_SIZE
        vlm_config.image_token_index = VOCAB_SIZE
        vlm_config.text_config.hidden_size = paligemma_config.width
        vlm_config.text_config.intermediate_size = paligemma_config.mlp_dim
        vlm_config.text_config.num_attention_heads = paligemma_config.num_heads
        vlm_config.text_config.head_dim = paligemma_config.head_dim
        vlm_config.text_config.num_hidden_layers = paligemma_config.depth
        vlm_config.text_config.num_key_value_heads = paligemma_config.num_kv_heads
        vlm_config.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config.text_config.torch_dtype = "float32"
        vlm_config.text_config.vocab_size = VOCAB_SIZE
        vlm_config.text_config.use_adarms = False
        vlm_config.text_config.adarms_cond_dim = None
        vlm_config.text_config._attn_implementation = "eager"
        vlm_config.vision_config.intermediate_size = 4304
        vlm_config.vision_config.projection_dim = 2048
        vlm_config.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config)
        self.paligemma.model.language_model = PiGemmaModel(vlm_config.text_config)
        self.paligemma.lm_head.weight = self.paligemma.model.language_model.embed_tokens.weight
        install_pi_vision_forward(self.paligemma)

        expert_hf_config = _make_gemma_config(action_expert_config, use_adarms=expert_use_adarms)
        self.gemma_expert = GemmaForCausalLM(config=expert_hf_config)
        self.gemma_expert.model = PiGemmaModel(expert_hf_config)
        self.gemma_expert.model.embed_tokens = None
        self.set_precision(precision)

    def set_precision(self, precision: str) -> None:
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        keep_float32 = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, parameter in self.named_parameters():
            if any(selector in name for selector in keep_float32):
                parameter.data = parameter.data.float()

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        return embed_image_without_hf_scaling(self.paligemma, image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values=None,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        conditions = adarms_cond or [None, None]
        prefix, suffix = inputs_embeds
        if prefix is not None and suffix is None:
            output = self.paligemma.model.language_model(
                inputs_embeds=prefix,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=conditions[0],
            )
            return [output.last_hidden_state, None], output.past_key_values
        if prefix is None and suffix is not None:
            output = self.gemma_expert.model(
                inputs_embeds=suffix,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=conditions[1],
            )
            return [None, output.last_hidden_state], None
        raise NotImplementedError("Standalone PaliGemmaWithExpert currently supports inference only")


def scale_language_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings * math.sqrt(embeddings.shape[-1])

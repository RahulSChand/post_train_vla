"""PaliGemma backbone plus a separate Gemma action expert.

This is an inference-focused adaptation of OpenPI's Apache-2.0 Torch model.
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from post_train_vla.models.configuration import GemmaConfig
from post_train_vla.models.gemma import PiGemmaModel, _gated_residual
from post_train_vla.models.vision import embed_image_without_hf_scaling, install_pi_vision_forward

VOCAB_SIZE = 257_152


def _flatten_attention_heads(attention: torch.Tensor) -> torch.Tensor:
    """Flatten an eager-attention output without changing token order.

    Transformers' ``eager_attention_forward`` already returns attention in
    ``[batch, sequence, heads, head_dim]`` order.  The output projection
    expects the final two dimensions flattened to ``hidden_size``.
    """
    if attention.ndim != 4:
        raise ValueError(
            f"Expected attention [batch, sequence, heads, head_dim], got {tuple(attention.shape)}"
        )
    batch, sequence, heads, head_dim = attention.shape
    return attention.reshape(batch, sequence, heads * head_dim)


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
        if prefix is None or suffix is None:
            raise ValueError("Expected either prefix, suffix, or both embedding sequences")

        # Fine-tuning needs one attention pass over the VLM prefix and action-expert
        # suffix. They have separate weights but attend over their concatenated KV
        # sequence at every layer; this is the PyTorch equivalent of OpenPI's joint
        # prefix/suffix forward path.
        models = [self.paligemma.model.language_model, self.gemma_expert.model]
        hidden_states = [prefix, suffix]
        for layer_index in range(len(models[0].layers)):
            def run_layer(*layer_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                queries, keys, values, gates = [], [], [], []
                for model, states, condition in zip(models, layer_states, conditions, strict=True):
                    layer = model.layers[layer_index]
                    states, gate = layer.input_layernorm(states, condition)
                    shape = (*states.shape[:-1], -1, layer.self_attn.head_dim)
                    queries.append(layer.self_attn.q_proj(states).view(shape).transpose(1, 2))
                    keys.append(layer.self_attn.k_proj(states).view(shape).transpose(1, 2))
                    values.append(layer.self_attn.v_proj(states).view(shape).transpose(1, 2))
                    gates.append(gate)

                query = torch.cat(queries, dim=2)
                key = torch.cat(keys, dim=2)
                value = torch.cat(values, dim=2)
                rotary_input = torch.zeros(
                    query.shape[0], query.shape[2], query.shape[-1], dtype=query.dtype, device=query.device
                )
                cosine, sine = models[0].rotary_emb(rotary_input, position_ids)
                query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cosine, sine, unsqueeze_dim=1)
                attention, _ = modeling_gemma.eager_attention_forward(
                    models[0].layers[layer_index].self_attn,
                    query,
                    key,
                    value,
                    attention_mask,
                    models[0].layers[layer_index].self_attn.scaling,
                )
                attention = _flatten_attention_heads(attention)

                next_states, start = [], 0
                for model, states, gate, condition in zip(models, layer_states, gates, conditions, strict=True):
                    layer = model.layers[layer_index]
                    end = start + states.shape[1]
                    layer_attention = attention[:, start:end].to(layer.self_attn.o_proj.weight.dtype)
                    output = layer.self_attn.o_proj(layer_attention)
                    output = _gated_residual(states, output, gate)
                    residual = output
                    output, gate = layer.post_attention_layernorm(output, condition)
                    output = output.to(layer.mlp.up_proj.weight.dtype)
                    output = layer.mlp(output)
                    next_states.append(_gated_residual(residual, output, gate))
                    start = end
                return tuple(next_states)  # type: ignore[return-value]

            use_checkpoint = (
                self.training
                and torch.is_grad_enabled()
                and all(model.gradient_checkpointing for model in models)
                # pi0.5's adaRMS condition has its own gradient path. Keep the
                # eager path there until it is explicitly checkpointed as input.
                and all(condition is None for condition in conditions)
            )
            if use_checkpoint:
                hidden_states = list(checkpoint(run_layer, *hidden_states, use_reentrant=False))
            else:
                hidden_states = list(run_layer(*hidden_states))

        outputs = [
            model.norm(states, condition)[0]
            for model, states, condition in zip(models, hidden_states, conditions, strict=True)
        ]
        return outputs, None


def scale_language_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings * math.sqrt(embeddings.shape[-1])

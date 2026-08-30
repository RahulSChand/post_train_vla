"""Pi-specific Gemma layers without modifying the installed Transformers package.

Derived from the Apache-2.0 Transformers 4.53.2 Gemma implementation and
OpenPI's adaptive normalization/cache changes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma.modeling_gemma import (
    ALL_ATTENTION_FUNCTIONS,
    GemmaAttention,
    GemmaDecoderLayer,
    GemmaPreTrainedModel,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class PiGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is None:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None
        else:
            self.register_parameter("weight", None)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor | None = None):
        dtype = inputs.dtype
        variance = torch.mean(torch.square(inputs.float()), dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps)
        if condition is None or self.dense is None:
            return (normalized * (1.0 + self.weight.float())).to(dtype), None
        if condition.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected condition dimension {self.cond_dim}, got {condition.shape[-1]}")
        modulation = self.dense(condition)
        if inputs.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normalized = normalized * (1.0 + scale.float()) + shift.float()
        return normalized.to(dtype), gate.to(dtype)


def _gated_residual(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    return residual + update if gate is None else residual + update * gate


class PiGemmaAttention(GemmaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cosine, sine = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cosine, sine)

        if past_key_value is not None:
            if use_cache:
                cache_kwargs = {"sin": sine, "cos": cosine, "cache_position": cache_position}
                key, value = past_key_value.update(key, value, self.layer_idx, cache_kwargs)
            else:
                key = torch.cat([past_key_value[self.layer_idx][0], key], dim=2)
                value = torch.cat([past_key_value[self.layer_idx][1], value], dim=2)

        attention: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        output, weights = attention(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        output = output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), weights


class PiGemmaDecoderLayer(GemmaDecoderLayer):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        condition_dim = config.adarms_cond_dim if getattr(config, "use_adarms", False) else None
        self.self_attn = PiGemmaAttention(config, layer_idx)
        self.input_layernorm = PiGemmaRMSNorm(config.hidden_size, config.rms_norm_eps, condition_dim)
        self.post_attention_layernorm = PiGemmaRMSNorm(config.hidden_size, config.rms_norm_eps, condition_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, adarms_cond)
        hidden_states, attention_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = _gated_residual(residual, hidden_states, gate)
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, hidden_states, gate)
        return (hidden_states, attention_weights) if output_attentions else (hidden_states,)


class PiGemmaModel(GemmaPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        condition_dim = config.adarms_cond_dim if getattr(config, "use_adarms", False) else None
        self.layers = nn.ModuleList([PiGemmaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = PiGemmaRMSNorm(config.hidden_size, config.rms_norm_eps, condition_dim)
        self.rotary_emb = GemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_length, past_length + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        if self.layers and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        hidden_states, _ = self.norm(hidden_states, adarms_cond)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

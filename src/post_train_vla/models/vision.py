"""Small vision compatibility layer replacing OpenPI's global SigLIP patch."""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer


class PiSiglipVisionTransformer(SiglipVisionTransformer):
    def forward(
        self,
        pixel_values,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if self.encoder.layers and self.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooled = self.head(last_hidden_state) if self.use_head else None
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def install_pi_vision_forward(paligemma) -> None:
    """Replace only the nested vision transformer; parameter names stay unchanged."""
    current = paligemma.model.vision_tower.vision_model
    replacement = PiSiglipVisionTransformer(current.config)
    paligemma.model.vision_tower.vision_model = replacement


def embed_image_without_hf_scaling(paligemma, image: torch.Tensor) -> torch.Tensor:
    outputs = paligemma.model.vision_tower(image)
    return paligemma.model.multi_modal_projector(outputs.last_hidden_state)

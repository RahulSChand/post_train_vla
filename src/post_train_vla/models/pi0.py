"""Standalone Torch pi0/pi0.5 flow-matching inference model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from post_train_vla.models.configuration import Pi0Config
from post_train_vla.models.observation import Observation
from post_train_vla.models.paligemma_expert import PaliGemmaWithExpert, scale_language_embeddings
from post_train_vla.models.preprocessing import preprocess_observation


def make_attention_masks(padding: torch.Tensor, autoregressive: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cumsum(autoregressive, dim=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = padding[:, None, :] * padding[:, :, None]
    return causal & valid


def sinusoidal_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    if dimension % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=time.device)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    values = (1.0 / period * 2 * math.pi)[None, :] * time[:, None]
    return torch.cat([torch.sin(values), torch.cos(values)], dim=1)


class Pi0(nn.Module):
    def __init__(self, config: Pi0Config) -> None:
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.paligemma_with_expert = PaliGemmaWithExpert(
            config.paligemma,
            config.action_expert,
            expert_use_adarms=config.pi05,
            precision=config.dtype,
        )
        expert_width = config.action_expert.width
        self.action_in_proj = nn.Linear(config.action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.action_dim)
        if config.pi05:
            self.time_mlp_in = nn.Linear(expert_width, expert_width)
            self.time_mlp_out = nn.Linear(expert_width, expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, expert_width)
            self.action_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
            self.action_time_mlp_out = nn.Linear(expert_width, expert_width)
        torch.set_float32_matmul_precision("high")

    def _embed_prefix(self, observation: Observation):
        embeddings = []
        padding_masks = []
        attention_masks = []
        for image, image_mask in zip(observation.images.values(), observation.image_masks.values()):
            image_embedding = self.paligemma_with_expert.embed_image(image)
            batch, tokens = image_embedding.shape[:2]
            embeddings.append(image_embedding)
            padding_masks.append(image_mask[:, None].expand(batch, tokens))
            attention_masks.extend([0] * tokens)
        language = scale_language_embeddings(
            self.paligemma_with_expert.embed_language_tokens(observation.tokenized_prompt)
        )
        embeddings.append(language)
        padding_masks.append(observation.tokenized_prompt_mask)
        attention_masks.extend([0] * language.shape[1])
        embeddings = torch.cat(embeddings, dim=1)
        padding = torch.cat(padding_masks, dim=1)
        autoregressive = torch.tensor(attention_masks, dtype=torch.bool, device=padding.device)
        return embeddings, padding, autoregressive[None, :].expand(padding.shape[0], -1)

    def _embed_suffix(self, state: torch.Tensor, actions: torch.Tensor, time: torch.Tensor):
        embeddings = []
        masks = []
        autoregressive = []
        if not self.pi05:
            state_embedding = self.state_proj(state.float())
            embeddings.append(state_embedding[:, None, :])
            masks.append(torch.ones(state.shape[0], 1, dtype=torch.bool, device=state.device))
            autoregressive.append(1)
        time_embedding = sinusoidal_embedding(time, self.action_in_proj.out_features).to(time.dtype)
        action_embedding = self.action_in_proj(actions)
        if self.pi05:
            condition = functional.silu(self.time_mlp_out(functional.silu(self.time_mlp_in(time_embedding))))
            action_time_embedding = action_embedding
        else:
            expanded_time = time_embedding[:, None, :].expand_as(action_embedding)
            combined = torch.cat([action_embedding, expanded_time], dim=2)
            action_time_embedding = self.action_time_mlp_out(functional.silu(self.action_time_mlp_in(combined)))
            condition = None
        embeddings.append(action_time_embedding)
        masks.append(torch.ones(actions.shape[0], actions.shape[1], dtype=torch.bool, device=actions.device))
        autoregressive.extend([1] + [0] * (self.config.action_horizon - 1))
        embeddings = torch.cat(embeddings, dim=1)
        padding = torch.cat(masks, dim=1)
        autoregressive = torch.tensor(autoregressive, dtype=embeddings.dtype, device=embeddings.device)
        return embeddings, padding, autoregressive[None, :].expand(padding.shape[0], -1), condition

    @staticmethod
    def _mask_to_4d(mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[:, None, :, :], 0.0, -2.3819763e38)

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        observation = preprocess_observation(observation)
        batch = observation.state.shape[0]
        if noise is None:
            noise = torch.randn(
                batch,
                self.config.action_horizon,
                self.config.action_dim,
                dtype=torch.float32,
                device=observation.state.device,
            )
        prefix, prefix_padding, prefix_ar = self._embed_prefix(observation)
        prefix_mask = self._mask_to_4d(make_attention_masks(prefix_padding, prefix_ar))
        prefix_positions = torch.cumsum(prefix_padding, dim=1) - 1
        _, cache = self.paligemma_with_expert(
            attention_mask=prefix_mask,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )

        steps = num_steps or self.config.num_inference_steps
        delta = torch.tensor(-1.0 / steps, dtype=torch.float32, device=noise.device)
        actions = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        while time >= -delta / 2:
            suffix, suffix_padding, suffix_ar, condition = self._embed_suffix(
                observation.state, actions, time.expand(batch)
            )
            suffix_length = suffix_padding.shape[1]
            prefix_length = prefix_padding.shape[1]
            prefix_mask = prefix_padding[:, None, :].expand(batch, suffix_length, prefix_length)
            suffix_mask = make_attention_masks(suffix_padding, suffix_ar)
            full_mask = self._mask_to_4d(torch.cat([prefix_mask, suffix_mask], dim=2))
            positions = torch.sum(prefix_padding, dim=-1)[:, None] + torch.cumsum(suffix_padding, dim=1) - 1
            outputs, _ = self.paligemma_with_expert(
                attention_mask=full_mask,
                position_ids=positions,
                past_key_values=cache,
                inputs_embeds=[None, suffix],
                use_cache=False,
                adarms_cond=[None, condition],
            )
            velocity = self.action_out_proj(outputs[1][:, -self.config.action_horizon :].float())
            actions = actions + delta * velocity
            time = time + delta
        return actions

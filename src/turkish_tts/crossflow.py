from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CrossFlowModelConfig:
    """Architecture parameters for the independently implemented acoustic model."""

    vocab_size: int
    mel_channels: int = 100
    model_dim: int = 512
    depth: int = 12
    heads: int = 8
    ff_dim: int = 2048
    text_depth: int = 4
    text_kernel_size: int = 7
    dropout: float = 0.0
    checkpoint_activations: bool = True
    max_frames: int = 2400
    max_text_tokens: int = 512
    adapter_dim: int = 0
    speaker_count: int = 0
    speaker_embedding_dim: int = 256
    prosody_dim: int = 0
    context_conditioning: bool = False
    global_reference_conditioning: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CrossFlowModelConfig:
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sinusoidal_positions(length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / dim))
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


def _timestep_embedding(timestep: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0) * 1_000.0
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConvNeXtTextBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, dim * 4)
        self.project = nn.Linear(dim * 4, dim)

    def forward(self, hidden: Tensor, token_mask: Tensor) -> Tensor:
        residual = hidden
        hidden = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.norm(hidden)
        hidden = self.project(F.gelu(self.expand(hidden)))
        hidden = residual + hidden
        return hidden * token_mask.unsqueeze(-1)


class TextEncoder(nn.Module):
    def __init__(self, config: CrossFlowModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=0)
        self.blocks = nn.ModuleList(
            ConvNeXtTextBlock(config.model_dim, config.text_kernel_size) for _ in range(config.text_depth)
        )
        self.final_norm = nn.LayerNorm(config.model_dim)

    def forward(self, token_ids: Tensor, token_mask: Tensor) -> Tensor:
        if token_ids.shape[1] > self.config.max_text_tokens:
            raise ValueError(
                f"text length {token_ids.shape[1]} exceeds configured maximum {self.config.max_text_tokens}"
            )
        hidden = self.embedding(token_ids)
        hidden = hidden + _sinusoidal_positions(
            hidden.shape[1], hidden.shape[2], device=hidden.device, dtype=hidden.dtype
        ).unsqueeze(0)
        hidden = hidden * token_mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, token_mask)
        normalized: Tensor = self.final_norm(hidden)
        return normalized * token_mask.unsqueeze(-1)


class SpeakerStyleAdapter(nn.Module):
    """Zero-initialized residual adapter for one fixed speaker and style."""

    def __init__(self, dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, adapter_dim, bias=False)
        self.style = nn.Parameter(torch.zeros(adapter_dim))
        self.up = nn.Linear(adapter_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        adapted = self.down(self.norm(hidden)) + self.style
        output: Tensor = self.up(F.silu(adapted))
        return output


class CrossFlowBlock(nn.Module):
    def __init__(self, config: CrossFlowModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.self_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(dim, config.heads, dropout=config.dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, config.heads, dropout=config.dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, config.ff_dim * 2),
            SwiGLU(),
            nn.Linear(config.ff_dim, dim),
        )
        modulation_output = nn.Linear(dim, dim * 9)
        self.modulation = nn.Sequential(nn.SiLU(), modulation_output)
        nn.init.zeros_(modulation_output.weight)
        nn.init.zeros_(modulation_output.bias)
        self.speaker_style_adapter = SpeakerStyleAdapter(dim, config.adapter_dim) if config.adapter_dim > 0 else None
        self.adapter_scale = 1.0

    def set_adapter_scale(self, scale: float) -> None:
        self.adapter_scale = scale

    @staticmethod
    def _modulate(hidden: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return hidden * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden: Tensor,
        text_context: Tensor,
        timestep_context: Tensor,
        frame_mask: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        modulation = self.modulation(timestep_context).chunk(9, dim=-1)
        self_shift, self_scale, self_gate = modulation[0:3]
        cross_shift, cross_scale, cross_gate = modulation[3:6]
        ff_shift, ff_scale, ff_gate = modulation[6:9]

        self_input = self._modulate(self.self_norm(hidden), self_shift, self_scale)
        self_output, _ = self.self_attention(
            self_input,
            self_input,
            self_input,
            key_padding_mask=~frame_mask,
            need_weights=False,
        )
        hidden = hidden + self_gate.unsqueeze(1) * self_output

        cross_input = self._modulate(self.cross_norm(hidden), cross_shift, cross_scale)
        cross_output, _ = self.cross_attention(
            cross_input,
            text_context,
            text_context,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        hidden = hidden + cross_gate.unsqueeze(1) * cross_output

        ff_input = self._modulate(self.ff_norm(hidden), ff_shift, ff_scale)
        hidden = hidden + ff_gate.unsqueeze(1) * self.feed_forward(ff_input)
        if self.speaker_style_adapter is not None:
            hidden = hidden + self.adapter_scale * self.speaker_style_adapter(hidden)
        return hidden * frame_mask.unsqueeze(-1)


class SwiGLU(nn.Module):
    def forward(self, hidden: Tensor) -> Tensor:
        values, gates = hidden.chunk(2, dim=-1)
        return values * F.silu(gates)


class ConditioningProjection(nn.Module):
    """Zero-initialized projection for non-linguistic acoustic conditions."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, output_dim)
        self.output = nn.Linear(output_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: Tensor) -> Tensor:
        projected: Tensor = self.output(F.silu(self.input(values)))
        return projected


class CrossFlow(nn.Module):
    """Character-conditioned rectified-flow model over normalized log-mel frames.

    This implementation was designed from public flow-matching and Transformer concepts. It does not
    load or depend on FreyaTTS code, weights, audio, latent representations, or training data.
    """

    def __init__(self, config: CrossFlowModelConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.duration_head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, 1),
        )
        self.mel_input = nn.Linear(config.mel_channels, config.model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model_dim * 4, config.model_dim),
        )
        self.speaker_embedding: nn.Embedding | None
        self.speaker_projection: ConditioningProjection | None
        if config.speaker_count > 0:
            self.speaker_embedding = nn.Embedding(
                config.speaker_count,
                config.speaker_embedding_dim,
                padding_idx=0,
            )
            self.speaker_projection = ConditioningProjection(
                config.speaker_embedding_dim,
                config.model_dim,
            )
        else:
            self.speaker_embedding = None
            self.speaker_projection = None
        self.prosody_projection = (
            ConditioningProjection(config.prosody_dim, config.model_dim) if config.prosody_dim > 0 else None
        )
        self.context_encoder = (
            ConditioningProjection(config.mel_channels + 1, config.model_dim) if config.context_conditioning else None
        )
        self.blocks = nn.ModuleList(CrossFlowBlock(config) for _ in range(config.depth))
        if config.adapter_dim > 0:
            self.style_embedding = nn.Parameter(torch.zeros(config.model_dim))
        else:
            self.register_parameter("style_embedding", None)
        self.adapter_scale = 1.0
        self.final_norm = nn.LayerNorm(config.model_dim, elementwise_affine=False)
        final_modulation_output = nn.Linear(config.model_dim, config.model_dim * 2)
        self.final_modulation = nn.Sequential(nn.SiLU(), final_modulation_output)
        self.mel_output = nn.Linear(config.model_dim, config.mel_channels)
        nn.init.zeros_(final_modulation_output.weight)
        nn.init.zeros_(final_modulation_output.bias)
        nn.init.zeros_(self.mel_output.weight)
        nn.init.zeros_(self.mel_output.bias)

    def set_adapter_scale(self, scale: float) -> None:
        if not 0.0 <= scale <= 1.0:
            raise ValueError("adapter scale must be between zero and one")
        self.adapter_scale = scale
        for block in self.blocks:
            cast(CrossFlowBlock, block).set_adapter_scale(scale)

    def context_parameter_names(self) -> set[str]:
        return {name for name, _ in self.named_parameters() if name.startswith("context_encoder.")}

    def adapter_parameter_names(self) -> set[str]:
        if self.config.adapter_dim <= 0:
            return set()
        return {
            name
            for name, _ in self.named_parameters()
            if name == "style_embedding" or ".speaker_style_adapter." in name
        }

    def conditioning_parameter_names(self) -> set[str]:
        return {
            name
            for name, _ in self.named_parameters()
            if name.startswith(("speaker_embedding.", "speaker_projection.", "prosody_projection."))
        }

    def freeze_base_for_adapter_training(self) -> None:
        adapter_names = self.adapter_parameter_names()
        if not adapter_names:
            raise ValueError("adapter training requires a positive adapter_dim")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in adapter_names)

    def freeze_for_conditioning_training(self) -> None:
        conditioning_names = self.conditioning_parameter_names()
        if not conditioning_names:
            raise ValueError("conditioning-only training requires acoustic conditions")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in conditioning_names)

    def freeze_for_context_training(self) -> None:
        context_names = self.context_parameter_names()
        if not context_names:
            raise ValueError("context-only training requires context conditioning")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in context_names)

    def freeze_for_speaker_embedding_training(self) -> None:
        if self.speaker_embedding is None:
            raise ValueError("speaker embedding training requires speaker conditioning")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name == "speaker_embedding.weight")

    def freeze_linguistic_components(self) -> None:
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.duration_head.parameters():
            parameter.requires_grad_(False)

    def encode_text(self, token_ids: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor]:
        context = self.text_encoder(token_ids, token_mask)
        denominator = token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(context.dtype)
        pooled = context.sum(dim=1) / denominator
        predicted_log_frames = self.duration_head(pooled).squeeze(-1)
        return context, predicted_log_frames

    def forward(
        self,
        noisy_mel: Tensor,
        timestep: Tensor,
        token_ids: Tensor,
        frame_mask: Tensor,
        token_mask: Tensor,
        speaker_ids: Tensor | None = None,
        prosody_features: Tensor | None = None,
        context_mel: Tensor | None = None,
        context_frame_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if noisy_mel.shape[1] > self.config.max_frames:
            raise ValueError(f"mel length {noisy_mel.shape[1]} exceeds configured maximum {self.config.max_frames}")
        if (context_mel is None) != (context_frame_mask is None):
            raise ValueError("context_mel and context_frame_mask must be provided together")
        if context_mel is not None and self.context_encoder is None:
            raise ValueError("context conditioning is not enabled in this model")
        if context_mel is not None and context_frame_mask is not None:
            if context_mel.shape != noisy_mel.shape:
                raise ValueError("context_mel must match the noisy mel shape")
            if context_frame_mask.shape != noisy_mel.shape[:2]:
                raise ValueError("context_frame_mask must match the frame grid")
        text_context, predicted_log_frames = self.encode_text(token_ids, token_mask)
        timestep_context = self.time_mlp(_timestep_embedding(timestep, self.config.model_dim)).to(noisy_mel.dtype)
        if self.speaker_embedding is not None and self.speaker_projection is not None:
            if speaker_ids is None:
                speaker_ids = torch.zeros(
                    noisy_mel.shape[0],
                    device=noisy_mel.device,
                    dtype=torch.long,
                )
            if speaker_ids.shape != (noisy_mel.shape[0],):
                raise ValueError("speaker_ids must contain one ID per batch item")
            acoustic_condition = self.speaker_projection(self.speaker_embedding(speaker_ids))
            if self.prosody_projection is not None and prosody_features is not None:
                if prosody_features.shape != (noisy_mel.shape[0], self.config.prosody_dim):
                    raise ValueError(
                        f"prosody_features must have shape ({noisy_mel.shape[0]}, {self.config.prosody_dim})"
                    )
                acoustic_condition = acoustic_condition + self.prosody_projection(
                    prosody_features.to(timestep_context.dtype)
                )
            timestep_context = timestep_context + acoustic_condition * speaker_ids.ne(0).unsqueeze(1)
        elif self.prosody_projection is not None and prosody_features is not None:
            if prosody_features.shape != (noisy_mel.shape[0], self.config.prosody_dim):
                raise ValueError(f"prosody_features must have shape ({noisy_mel.shape[0]}, {self.config.prosody_dim})")
            timestep_context = timestep_context + self.prosody_projection(prosody_features.to(timestep_context.dtype))
        if self.style_embedding is not None:
            timestep_context = timestep_context + self.adapter_scale * self.style_embedding
        hidden = self.mel_input(noisy_mel)
        hidden = hidden + _sinusoidal_positions(
            hidden.shape[1], hidden.shape[2], device=hidden.device, dtype=hidden.dtype
        ).unsqueeze(0)
        hidden = hidden * frame_mask.unsqueeze(-1)
        if self.context_encoder is not None and context_mel is not None and context_frame_mask is not None:
            context_flags = context_frame_mask.unsqueeze(-1).to(hidden.dtype)
            context_features = torch.cat((context_mel.to(hidden.dtype) * context_flags, context_flags), dim=-1)
            encoded_context = self.context_encoder(context_features) * context_flags * frame_mask.unsqueeze(-1)
            hidden = hidden + encoded_context
            if self.config.global_reference_conditioning:
                denominator = context_frame_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
                pooled_context = encoded_context.sum(dim=1) / denominator
                has_context = context_frame_mask.any(dim=1, keepdim=True).to(hidden.dtype)
                timestep_context = timestep_context + pooled_context * has_context
        for block in self.blocks:
            if self.training and self.config.checkpoint_activations:
                hidden = checkpoint(
                    block,
                    hidden,
                    text_context,
                    timestep_context,
                    frame_mask,
                    token_mask,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, text_context, timestep_context, frame_mask, token_mask)
        shift, scale = self.final_modulation(timestep_context).chunk(2, dim=-1)
        hidden = self.final_norm(hidden) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        velocity = self.mel_output(hidden) * frame_mask.unsqueeze(-1)
        return velocity, predicted_log_frames

    def loss(
        self,
        clean_mel: Tensor,
        token_ids: Tensor,
        frame_mask: Tensor,
        token_mask: Tensor,
        *,
        duration_weight: float,
        speaker_ids: Tensor | None = None,
        prosody_features: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return flow_matching_loss(
            self,
            clean_mel,
            token_ids,
            frame_mask,
            token_mask,
            speaker_ids=speaker_ids,
            prosody_features=prosody_features,
            duration_weight=duration_weight,
        )

    @torch.no_grad()
    def sample(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        *,
        steps: int = 32,
        seed: int = 20260803,
        frame_count: int | None = None,
        duration_scale: float = 1.0,
        speaker_id: int = 0,
        prosody_features: Tensor | None = None,
        text_guidance_scale: float = 1.0,
        speaker_guidance_scale: float = 1.0,
        sway_coefficient: float = 0.0,
        solver: str = "euler",
        guidance_rescale: float = 0.0,
        mel_clamp: float | None = None,
        context_mel: Tensor | None = None,
        context_guidance_scale: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        if token_ids.shape[0] != 1:
            raise ValueError("sample currently requires a batch size of one")
        if text_guidance_scale < 0.0 or speaker_guidance_scale < 0.0 or context_guidance_scale < 0.0:
            raise ValueError("guidance scales cannot be negative")
        if not -1.0 <= sway_coefficient <= 1.0:
            raise ValueError("sway_coefficient must be between minus one and one")
        if solver not in ("euler", "midpoint"):
            raise ValueError("solver must be euler or midpoint")
        if speaker_guidance_scale != 1.0 and speaker_id == 0:
            raise ValueError("speaker guidance requires a conditioned speaker ID")
        if not 0.0 <= guidance_rescale <= 1.0:
            raise ValueError("guidance_rescale must be between zero and one")
        if mel_clamp is not None and mel_clamp <= 0.0:
            raise ValueError("mel_clamp must be positive")
        if context_guidance_scale != 1.0 and context_mel is None:
            raise ValueError("context guidance requires a reference context")
        if context_mel is not None:
            if context_mel.ndim != 3 or context_mel.shape[0] != 1:
                raise ValueError("context_mel must have shape (1, frames, mel_channels)")
            if context_mel.shape[2] != self.config.mel_channels:
                raise ValueError("context_mel channel count does not match the model")
        _, predicted_log_frames = self.encode_text(token_ids, token_mask)
        if frame_count is None:
            frame_count = round(math.exp(float(predicted_log_frames.item())) * duration_scale)
        frame_count = max(8, min(self.config.max_frames, frame_count))
        context_frames = 0
        if context_mel is not None:
            context_frames = context_mel.shape[1]
            frame_count = max(frame_count, context_frames + 8)
            if frame_count > self.config.max_frames:
                raise ValueError("reference context plus continuation exceeds the frame budget")
        speaker_ids = torch.tensor([speaker_id], device=token_ids.device, dtype=torch.long)
        unconditioned_speaker_ids = torch.zeros(1, device=token_ids.device, dtype=torch.long)
        if prosody_features is not None and prosody_features.shape != (1, self.config.prosody_dim):
            raise ValueError(f"prosody_features must have shape (1, {self.config.prosody_dim})")
        null_token_ids = torch.ones(1, 1, device=token_ids.device, dtype=torch.long)
        null_token_mask = torch.ones(1, 1, device=token_ids.device, dtype=torch.bool)
        generator = torch.Generator(device=token_ids.device).manual_seed(seed)
        mel = torch.randn(
            1,
            frame_count,
            self.config.mel_channels,
            device=token_ids.device,
            generator=generator,
            dtype=next(self.parameters()).dtype,
        )
        frame_mask = torch.ones(1, frame_count, device=token_ids.device, dtype=torch.bool)

        padded_context: Tensor | None = None
        padded_context_mask: Tensor | None = None
        if context_mel is not None:
            padded_context = torch.zeros(
                1,
                frame_count,
                self.config.mel_channels,
                device=token_ids.device,
                dtype=mel.dtype,
            )
            padded_context[:, :context_frames] = context_mel.to(mel.dtype)
            padded_context_mask = torch.zeros(1, frame_count, device=token_ids.device, dtype=torch.bool)
            padded_context_mask[:, :context_frames] = True
            initial_noise = mel[:, :context_frames].clone()

        def velocity(state: Tensor, time_value: float) -> Tensor:
            timestep = torch.full((1,), time_value, device=token_ids.device, dtype=torch.float32)
            conditioned, _ = self(
                state,
                timestep,
                token_ids,
                frame_mask,
                token_mask,
                speaker_ids=speaker_ids,
                prosody_features=prosody_features,
                context_mel=padded_context,
                context_frame_mask=padded_context_mask,
            )
            guided = conditioned
            if speaker_guidance_scale != 1.0:
                without_speaker, _ = self(
                    state,
                    timestep,
                    token_ids,
                    frame_mask,
                    token_mask,
                    speaker_ids=unconditioned_speaker_ids,
                    prosody_features=None,
                    context_mel=padded_context,
                    context_frame_mask=padded_context_mask,
                )
                guided = guided + (speaker_guidance_scale - 1.0) * (guided - without_speaker)
            if text_guidance_scale != 1.0:
                without_text, _ = self(
                    state,
                    timestep,
                    null_token_ids,
                    frame_mask,
                    null_token_mask,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_features,
                    context_mel=padded_context,
                    context_frame_mask=padded_context_mask,
                )
                guided = guided + (text_guidance_scale - 1.0) * (guided - without_text)
            if context_guidance_scale != 1.0:
                without_reference, _ = self(
                    state,
                    timestep,
                    token_ids,
                    frame_mask,
                    token_mask,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_features,
                )
                guided = guided + (context_guidance_scale - 1.0) * (guided - without_reference)
            if guidance_rescale > 0.0 and guided is not conditioned:
                conditioned_std = conditioned.float().std()
                guided_std = guided.float().std().clamp_min(1e-6)
                rescaled = guided * (conditioned_std / guided_std).to(guided.dtype)
                guided = guidance_rescale * rescaled + (1.0 - guidance_rescale) * guided
            return cast(Tensor, guided)

        uniform = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
        schedule = uniform + sway_coefficient * (torch.cos(math.pi / 2.0 * uniform) - 1.0 + uniform)
        for step in range(steps):
            current = float(schedule[step])
            width = float(schedule[step + 1]) - current
            if solver == "midpoint":
                midpoint_state = mel + velocity(mel, current) * (width / 2.0)
                mel = mel + velocity(midpoint_state, current + width / 2.0) * width
            else:
                mel = mel + velocity(mel, current) * width
            if mel_clamp is not None:
                mel = mel.clamp(-mel_clamp, mel_clamp)
            if padded_context is not None:
                interpolation = float(schedule[step + 1])
                mel[:, :context_frames] = (1.0 - interpolation) * initial_noise + interpolation * padded_context[
                    :, :context_frames
                ]
        return mel, predicted_log_frames

    @torch.no_grad()
    def sample_batched(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        *,
        frame_counts: Sequence[int],
        seeds: Sequence[int],
        steps: int = 32,
        speaker_ids: Tensor | None = None,
        prosody_features: Tensor | None = None,
        text_guidance_scale: float = 1.0,
        speaker_guidance_scale: float = 1.0,
        sway_coefficient: float = 0.0,
        solver: str = "euler",
        guidance_rescale: float = 0.0,
        mel_clamp: float | None = None,
        context_mel: Tensor | None = None,
        context_frame_mask: Tensor | None = None,
        context_guidance_scale: float = 1.0,
    ) -> Tensor:
        """Sample one mel per row; rows share text/condition layout but use independent seeds.

        Returns a padded (batch, max_frames, mel_channels) tensor; callers trim row i to
        frame_counts[i]. With ``context_mel``, every row shares the same reference context
        (production single-speaker anchoring).
        """
        mel, _ = self._sample_batched_rows(
            token_ids,
            token_mask,
            frame_counts=frame_counts,
            seeds=seeds,
            steps=steps,
            speaker_ids=speaker_ids,
            prosody_features=prosody_features,
            text_guidance_scale=text_guidance_scale,
            speaker_guidance_scale=speaker_guidance_scale,
            sway_coefficient=sway_coefficient,
            solver=solver,
            guidance_rescale=guidance_rescale,
            mel_clamp=mel_clamp,
            prune_scorer=None,
            prune_after_step=0,
            prune_keep=0,
            context_mel=context_mel,
            context_frame_mask=context_frame_mask,
            context_guidance_scale=context_guidance_scale,
        )
        return mel

    @torch.no_grad()
    def sample_batched_pruned(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        *,
        frame_counts: Sequence[int],
        seeds: Sequence[int],
        prune_scorer: Callable[[Tensor, Tensor], Tensor],
        prune_after_step: int,
        prune_keep: int,
        steps: int = 32,
        speaker_ids: Tensor | None = None,
        prosody_features: Tensor | None = None,
        text_guidance_scale: float = 1.0,
        speaker_guidance_scale: float = 1.0,
        sway_coefficient: float = 0.0,
        solver: str = "euler",
        guidance_rescale: float = 0.0,
        mel_clamp: float | None = None,
        context_mel: Tensor | None = None,
        context_frame_mask: Tensor | None = None,
        context_guidance_scale: float = 1.0,
    ) -> tuple[Tensor, list[int]]:
        """Best-of-N sampling with mid-flow candidate pruning.

        After ``prune_after_step`` flow steps the sampler forms each row's clean-mel estimate
        x1 = x_t + (1 - t) * v(x_t, t), scores the estimates with ``prune_scorer`` (higher is
        better), and continues integrating only the ``prune_keep`` best rows. Costs one extra
        velocity evaluation at the prune point; saves (steps - prune_after_step) evaluations
        for every pruned row plus its vocoder pass. Returns the surviving rows (original
        relative order preserved) and their original row indices.
        """
        return self._sample_batched_rows(
            token_ids,
            token_mask,
            frame_counts=frame_counts,
            seeds=seeds,
            steps=steps,
            speaker_ids=speaker_ids,
            prosody_features=prosody_features,
            text_guidance_scale=text_guidance_scale,
            speaker_guidance_scale=speaker_guidance_scale,
            sway_coefficient=sway_coefficient,
            solver=solver,
            guidance_rescale=guidance_rescale,
            mel_clamp=mel_clamp,
            prune_scorer=prune_scorer,
            prune_after_step=prune_after_step,
            prune_keep=prune_keep,
            context_mel=context_mel,
            context_frame_mask=context_frame_mask,
            context_guidance_scale=context_guidance_scale,
        )

    def _sample_batched_rows(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        *,
        frame_counts: Sequence[int],
        seeds: Sequence[int],
        steps: int,
        speaker_ids: Tensor | None,
        prosody_features: Tensor | None,
        text_guidance_scale: float,
        speaker_guidance_scale: float,
        sway_coefficient: float,
        solver: str,
        guidance_rescale: float,
        mel_clamp: float | None,
        prune_scorer: Callable[[Tensor, Tensor], Tensor] | None,
        prune_after_step: int,
        prune_keep: int,
        context_mel: Tensor | None = None,
        context_frame_mask: Tensor | None = None,
        context_guidance_scale: float = 1.0,
    ) -> tuple[Tensor, list[int]]:
        batch_size = token_ids.shape[0]
        if batch_size == 0:
            raise ValueError("sample_batched requires at least one row")
        if (context_mel is None) != (context_frame_mask is None):
            raise ValueError("context_mel and context_frame_mask must be provided together")
        if context_mel is not None and self.context_encoder is None:
            raise ValueError("context conditioning is not enabled in this model")
        if context_guidance_scale != 1.0 and context_mel is None:
            raise ValueError("context guidance requires a reference context")
        if prune_scorer is not None:
            if not 1 <= prune_keep <= batch_size:
                raise ValueError("prune_keep must be between one and the candidate count")
            if not 1 <= prune_after_step <= steps - 1:
                raise ValueError("prune_after_step must leave at least one step before and after pruning")
        if len(frame_counts) != batch_size or len(seeds) != batch_size:
            raise ValueError("frame_counts and seeds must contain one entry per row")
        if text_guidance_scale < 0.0 or speaker_guidance_scale < 0.0:
            raise ValueError("guidance scales cannot be negative")
        if not -1.0 <= sway_coefficient <= 1.0:
            raise ValueError("sway_coefficient must be between minus one and one")
        if solver not in ("euler", "midpoint"):
            raise ValueError("solver must be euler or midpoint")
        if not 0.0 <= guidance_rescale <= 1.0:
            raise ValueError("guidance_rescale must be between zero and one")
        if mel_clamp is not None and mel_clamp <= 0.0:
            raise ValueError("mel_clamp must be positive")
        if any(count < 8 or count > self.config.max_frames for count in frame_counts):
            raise ValueError("frame_counts must be between 8 and the configured frame budget")
        if speaker_ids is None:
            speaker_ids = torch.zeros(batch_size, device=token_ids.device, dtype=torch.long)
        if speaker_guidance_scale != 1.0 and bool(speaker_ids.eq(0).any()):
            raise ValueError("speaker guidance requires conditioned speaker IDs")
        unconditioned_speaker_ids = torch.zeros(batch_size, device=token_ids.device, dtype=torch.long)
        null_token_ids = torch.ones(batch_size, 1, device=token_ids.device, dtype=torch.long)
        null_token_mask = torch.ones(batch_size, 1, device=token_ids.device, dtype=torch.bool)
        max_frame_count = max(frame_counts)
        dtype = next(self.parameters()).dtype
        mel = torch.zeros(
            batch_size,
            max_frame_count,
            self.config.mel_channels,
            device=token_ids.device,
            dtype=dtype,
        )
        for row, (row_frames, row_seed) in enumerate(zip(frame_counts, seeds, strict=True)):
            generator = torch.Generator(device=token_ids.device).manual_seed(row_seed)
            mel[row, :row_frames] = torch.randn(
                row_frames,
                self.config.mel_channels,
                device=token_ids.device,
                generator=generator,
                dtype=dtype,
            )
        frame_positions = torch.arange(max_frame_count, device=token_ids.device).unsqueeze(0)
        frame_lengths = torch.tensor(list(frame_counts), device=token_ids.device, dtype=torch.long)
        frame_mask = frame_positions < frame_lengths.unsqueeze(1)
        padded_context = None
        padded_context_mask = None
        if context_mel is not None:
            assert context_frame_mask is not None
            padded_context = torch.zeros_like(mel)
            padded_context[:, : context_mel.shape[1]] = context_mel.to(dtype)
            padded_context_mask = torch.zeros_like(frame_mask)
            padded_context_mask[:, : context_frame_mask.shape[1]] = context_frame_mask

        def velocity(state: Tensor, time_value: float) -> Tensor:
            timestep = torch.full((batch_size,), time_value, device=token_ids.device, dtype=torch.float32)
            conditioned, _ = self(
                state,
                timestep,
                token_ids,
                frame_mask,
                token_mask,
                speaker_ids=speaker_ids,
                prosody_features=prosody_features,
                context_mel=padded_context,
                context_frame_mask=padded_context_mask,
            )
            guided = conditioned
            if speaker_guidance_scale != 1.0:
                without_speaker, _ = self(
                    state,
                    timestep,
                    token_ids,
                    frame_mask,
                    token_mask,
                    speaker_ids=unconditioned_speaker_ids,
                    prosody_features=None,
                    context_mel=padded_context,
                    context_frame_mask=padded_context_mask,
                )
                guided = guided + (speaker_guidance_scale - 1.0) * (guided - without_speaker)
            if text_guidance_scale != 1.0:
                without_text, _ = self(
                    state,
                    timestep,
                    null_token_ids,
                    frame_mask,
                    null_token_mask,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_features,
                    context_mel=padded_context,
                    context_frame_mask=padded_context_mask,
                )
                guided = guided + (text_guidance_scale - 1.0) * (guided - without_text)
            if context_guidance_scale != 1.0:
                without_reference, _ = self(
                    state,
                    timestep,
                    token_ids,
                    frame_mask,
                    token_mask,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_features,
                )
                guided = guided + (context_guidance_scale - 1.0) * (guided - without_reference)
            if guidance_rescale > 0.0 and guided is not conditioned:
                conditioned_std = conditioned.float().std(dim=(1, 2), keepdim=True)
                guided_std = guided.float().std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                rescaled = guided * (conditioned_std / guided_std).to(guided.dtype)
                guided = guidance_rescale * rescaled + (1.0 - guidance_rescale) * guided
            return cast(Tensor, guided)

        uniform = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
        schedule = uniform + sway_coefficient * (torch.cos(math.pi / 2.0 * uniform) - 1.0 + uniform)
        kept_rows = list(range(batch_size))
        for step in range(steps):
            current = float(schedule[step])
            width = float(schedule[step + 1]) - current
            if solver == "midpoint":
                midpoint_state = mel + velocity(mel, current) * (width / 2.0)
                mel = mel + velocity(midpoint_state, current + width / 2.0) * width
            else:
                mel = mel + velocity(mel, current) * width
            if mel_clamp is not None:
                mel = mel.clamp(-mel_clamp, mel_clamp)
            mel = mel * frame_mask.unsqueeze(-1)
            if padded_context is not None:
                assert padded_context_mask is not None
                interpolation = float(schedule[step + 1])
                # Re-impose the exact noise→reference interpolation on context frames
                # so the trajectory passes through the reference at t = 1.
                context_target = (1.0 - interpolation) * mel + interpolation * padded_context
                mel = torch.where(padded_context_mask.unsqueeze(-1), context_target, mel)
            if prune_scorer is not None and step + 1 == prune_after_step and prune_keep < batch_size:
                reached = float(schedule[step + 1])
                estimate = mel + velocity(mel, reached) * (1.0 - reached)
                estimate = estimate * frame_mask.unsqueeze(-1)
                scores = prune_scorer(estimate, frame_mask)
                if scores.shape != (batch_size,):
                    raise ValueError("prune scorer must return one score per candidate row")
                keep = torch.topk(scores.float(), prune_keep).indices.sort().values
                kept_rows = [kept_rows[int(index)] for index in keep]
                mel = mel[keep]
                token_ids = token_ids[keep]
                token_mask = token_mask[keep]
                frame_mask = frame_mask[keep]
                speaker_ids = speaker_ids[keep]
                unconditioned_speaker_ids = unconditioned_speaker_ids[keep]
                null_token_ids = null_token_ids[keep]
                null_token_mask = null_token_mask[keep]
                if padded_context is not None:
                    padded_context = padded_context[keep]
                if padded_context_mask is not None:
                    padded_context_mask = padded_context_mask[keep]
                if prosody_features is not None:
                    prosody_features = prosody_features[keep]
                batch_size = prune_keep
        return mel, kept_rows


def flow_matching_loss(
    model: nn.Module,
    clean_mel: Tensor,
    token_ids: Tensor,
    frame_mask: Tensor,
    token_mask: Tensor,
    speaker_ids: Tensor | None = None,
    prosody_features: Tensor | None = None,
    context_frame_mask: Tensor | None = None,
    *,
    duration_weight: float,
    duration_frame_lengths: Tensor | None = None,
    foundation_preservation_weight: float = 0.0,
    return_predictions: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    if context_frame_mask is not None and context_frame_mask.shape != clean_mel.shape[:2]:
        raise ValueError("context_frame_mask must match the frame grid")
    batch_size = clean_mel.shape[0]
    noise = torch.randn_like(clean_mel)
    timestep = torch.rand(batch_size, device=clean_mel.device, dtype=torch.float32)
    interpolation = timestep.to(clean_mel.dtype).view(batch_size, 1, 1)
    noisy_mel = (1.0 - interpolation) * noise + interpolation * clean_mel
    target_velocity = clean_mel - noise
    context_mel = clean_mel if context_frame_mask is not None else None
    foundation_velocity: Tensor | None = None
    if foundation_preservation_weight > 0.0:
        adapter_model = cast(Any, getattr(model, "module", model))
        adapter_model.set_adapter_scale(0.0)
        try:
            with torch.no_grad():
                foundation_velocity, _ = model(
                    noisy_mel,
                    timestep,
                    token_ids,
                    frame_mask,
                    token_mask,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_features,
                    context_mel=context_mel,
                    context_frame_mask=context_frame_mask,
                )
        finally:
            adapter_model.set_adapter_scale(1.0)
    predicted_velocity, predicted_log_frames = model(
        noisy_mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        speaker_ids=speaker_ids,
        prosody_features=prosody_features,
        context_mel=context_mel,
        context_frame_mask=context_frame_mask,
    )
    scored_frames = frame_mask
    if context_frame_mask is not None:
        scored_frames = frame_mask & ~context_frame_mask
    mask = scored_frames.unsqueeze(-1).to(clean_mel.dtype)
    flow_loss = ((predicted_velocity - target_velocity).square() * mask).sum() / (
        mask.sum() * clean_mel.shape[-1]
    ).clamp_min(1.0)
    if duration_frame_lengths is None:
        duration_frame_lengths = frame_mask.sum(dim=1)
    if duration_frame_lengths.shape != (batch_size,):
        raise ValueError("duration_frame_lengths must contain one target length per batch item")
    target_log_frames = duration_frame_lengths.float().clamp_min(1).log()
    duration_loss = F.mse_loss(predicted_log_frames.float(), target_log_frames)
    total = flow_loss + duration_weight * duration_loss
    components = {"flow": flow_loss.detach(), "duration": duration_loss.detach()}
    if return_predictions:
        # Rectified-flow interpolation x_t = (1 - t) * noise + t * x_1 with
        # v = x_1 - noise recovers the model's clean-mel estimate:
        # x_1_hat = x_t + (1 - t) * v_hat.
        predicted_mel = noisy_mel + (1.0 - interpolation) * predicted_velocity
        components["predicted_mel"] = predicted_mel
        components["scored_frame_mask"] = scored_frames
    if foundation_velocity is not None:
        preservation_loss = ((predicted_velocity - foundation_velocity).square() * mask).sum() / (
            mask.sum() * clean_mel.shape[-1]
        ).clamp_min(1.0)
        total = total + foundation_preservation_weight * preservation_loss
        components["foundation_preservation"] = preservation_loss.detach()
    return total, components


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

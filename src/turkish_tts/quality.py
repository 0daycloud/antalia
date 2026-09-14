"""Deterministic candidate scoring for mid-flow best-of-N pruning.

The scorer operates in the model's *normalized* mel space, where training data
is standardized to roughly zero mean and unit variance and clipped at
``mel_normalized_clip``. Flow-matching failure modes visible mid-generation --
collapsed/static output, saturated bands, distribution drift -- all show up as
departures from those statistics, so a cheap statistics distance is enough to
rank candidates before spending the remaining flow steps and the vocoder pass.

A learned multi-head quality model can replace ``MelStatisticsScorer`` behind
the same callable contract once trained; nothing in the sampler depends on the
heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


class MelCandidateScorer(Protocol):
    """Scores candidate rows; higher is better. Inputs are (batch, frames, mels) and (batch, frames)."""

    def __call__(self, estimated_mel: Tensor, frame_mask: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class MelStatisticsScorer:
    """Penalize normalized-mel candidates that drift from training statistics.

    Score = -(mean_weight * |mean| + std_weight * |std - 1| + clip_weight * clipped_fraction
              + static_weight * static_transition_fraction), computed over unmasked frames only.
    """

    clip_threshold: float = 5.0
    static_delta_threshold: float = 0.05
    mean_weight: float = 1.0
    std_weight: float = 1.0
    clip_weight: float = 4.0
    static_weight: float = 2.0

    def __call__(self, estimated_mel: Tensor, frame_mask: Tensor) -> Tensor:
        if estimated_mel.ndim != 3 or frame_mask.ndim != 2:
            raise ValueError("expected (batch, frames, mels) mel and (batch, frames) mask")
        if estimated_mel.shape[:2] != frame_mask.shape:
            raise ValueError("mel and frame mask disagree on the frame grid")
        mel = estimated_mel.float()
        mask = frame_mask.unsqueeze(-1).to(mel.dtype)
        element_counts = frame_mask.sum(dim=1).to(mel.dtype).clamp_min(1.0) * mel.shape[-1]
        masked = mel * mask
        mean = masked.sum(dim=(1, 2)) / element_counts
        centered = (mel - mean.view(-1, 1, 1)) * mask
        variance = centered.square().sum(dim=(1, 2)) / element_counts
        standard_deviation = variance.clamp_min(0.0).sqrt()
        clipped = ((mel.abs() >= self.clip_threshold).to(mel.dtype) * mask).sum(dim=(1, 2)) / element_counts

        transitions = (mel[:, 1:] - mel[:, :-1]).abs().mean(dim=-1)
        transition_mask = (frame_mask[:, 1:] & frame_mask[:, :-1]).to(mel.dtype)
        transition_counts = transition_mask.sum(dim=1).clamp_min(1.0)
        static = ((transitions < self.static_delta_threshold).to(mel.dtype) * transition_mask).sum(
            dim=1
        ) / transition_counts

        penalty = (
            self.mean_weight * mean.abs()
            + self.std_weight * (standard_deviation - 1.0).abs()
            + self.clip_weight * clipped
            + self.static_weight * static
        )
        return -penalty

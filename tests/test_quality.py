from pathlib import Path

import pytest
import torch

from turkish_tts.crossflow import CrossFlow, CrossFlowModelConfig
from turkish_tts.crossflow_train import (
    CharacterTokenizer,
    CrossFlowTrainConfig,
    MelNormalizer,
    _synthesize_loaded_crossflow_candidates,
)
from turkish_tts.quality import MelStatisticsScorer


def _tiny_model() -> CrossFlow:
    torch.manual_seed(3)
    return CrossFlow(
        CrossFlowModelConfig(
            vocab_size=32,
            mel_channels=8,
            model_dim=32,
            depth=1,
            heads=4,
            ff_dim=64,
            text_depth=1,
            text_kernel_size=3,
            checkpoint_activations=False,
            max_frames=32,
            max_text_tokens=32,
        )
    ).eval()


def test_mel_statistics_scorer_prefers_plausible_normalized_mel() -> None:
    scorer = MelStatisticsScorer()
    generator = torch.Generator().manual_seed(11)
    plausible = torch.randn(1, 24, 8, generator=generator)
    collapsed = torch.zeros(1, 24, 8)
    clipped = torch.full((1, 24, 8), 5.0)
    mask = torch.ones(1, 24, dtype=torch.bool)

    plausible_score = scorer(plausible, mask)
    assert plausible_score > scorer(collapsed, mask)
    assert plausible_score > scorer(clipped, mask)
    assert torch.equal(plausible_score, scorer(plausible, mask))


def test_mel_statistics_scorer_ignores_masked_frames() -> None:
    scorer = MelStatisticsScorer()
    generator = torch.Generator().manual_seed(12)
    mel = torch.randn(1, 24, 8, generator=generator)
    corrupted = mel.clone()
    corrupted[:, 16:] = 40.0
    mask = torch.arange(24).unsqueeze(0) < 16

    assert torch.equal(scorer(mel, mask), scorer(corrupted, mask))


def test_mel_statistics_scorer_rejects_mismatched_shapes() -> None:
    scorer = MelStatisticsScorer()
    with pytest.raises(ValueError, match="frame grid"):
        scorer(torch.zeros(2, 24, 8), torch.ones(2, 23, dtype=torch.bool))
    with pytest.raises(ValueError, match="expected"):
        scorer(torch.zeros(2, 24), torch.ones(2, 24, dtype=torch.bool))


def test_pruned_sampling_keeps_scorer_selected_rows_unchanged() -> None:
    model = _tiny_model()
    token_ids = torch.tensor([[2, 3, 4, 5, 6]]).expand(4, -1)
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    seeds = [101, 202, 303, 404]
    frame_counts = [16, 16, 16, 16]

    def rigged_scorer(estimated_mel: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        assert estimated_mel.shape == (4, 16, 8)
        assert frame_mask.shape == (4, 16)
        return torch.tensor([0.0, 10.0, 0.0, 10.0])

    pruned, kept = model.sample_batched_pruned(
        token_ids,
        token_mask,
        frame_counts=frame_counts,
        seeds=seeds,
        prune_scorer=rigged_scorer,
        prune_after_step=1,
        prune_keep=2,
        steps=4,
    )
    full = model.sample_batched(
        token_ids,
        token_mask,
        frame_counts=frame_counts,
        seeds=seeds,
        steps=4,
    )

    assert kept == [1, 3]
    assert pruned.shape == (2, 16, 8)
    assert torch.allclose(pruned[0], full[1], atol=1e-5)
    assert torch.allclose(pruned[1], full[3], atol=1e-5)


def test_pruned_sampling_validates_prune_window_and_scorer_output() -> None:
    model = _tiny_model()
    token_ids = torch.tensor([[2, 3, 4]]).expand(2, -1)
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    scorer = MelStatisticsScorer()

    with pytest.raises(ValueError, match="prune_after_step"):
        model.sample_batched_pruned(
            token_ids,
            token_mask,
            frame_counts=[16, 16],
            seeds=[1, 2],
            prune_scorer=scorer,
            prune_after_step=4,
            prune_keep=1,
            steps=4,
        )
    with pytest.raises(ValueError, match="prune_keep"):
        model.sample_batched_pruned(
            token_ids,
            token_mask,
            frame_counts=[16, 16],
            seeds=[1, 2],
            prune_scorer=scorer,
            prune_after_step=1,
            prune_keep=3,
            steps=4,
        )

    def broken_scorer(estimated_mel: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        return torch.zeros(estimated_mel.shape[0] + 1)

    with pytest.raises(ValueError, match="one score per candidate"):
        model.sample_batched_pruned(
            token_ids,
            token_mask,
            frame_counts=[16, 16],
            seeds=[1, 2],
            prune_scorer=broken_scorer,
            prune_after_step=1,
            prune_keep=1,
            steps=4,
        )


class _StubVocoder:
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.zeros(mel.shape[0], 1, mel.shape[2] * 256)


def test_candidate_synthesis_prunes_first_chunk_and_keeps_seed_identity(tmp_path: Path) -> None:
    model = _tiny_model()
    tokenizer = CharacterTokenizer.from_texts(["merhaba dünya. iyi günler."])
    train_config = CrossFlowTrainConfig(
        run_version="prune-check",
        train_arrow=str(tmp_path / "train.arrow"),
        validation_arrow=str(tmp_path / "validation.arrow"),
        output_dir=str(tmp_path / "run"),
        n_mels=8,
    )
    mel_normalizer = MelNormalizer(train_config, "cpu")

    results = _synthesize_loaded_crossflow_candidates(
        model=model,
        tokenizer=tokenizer,
        vocoder=_StubVocoder(),
        mel_normalizer=mel_normalizer,
        text="merhaba dünya. iyi günler.",
        sample_rate=24_000,
        hop_length=256,
        device="cpu",
        steps=4,
        seeds=[7, 8, 9],
        duration_scale=1.0,
        speaker_id=0,
        prosody=None,
        text_guidance_scale=1.0,
        speaker_guidance_scale=1.0,
        sway_coefficient=0.0,
        solver="euler",
        guidance_rescale=0.0,
        mel_clamp=None,
        min_seconds_per_char=0.0,
        chunk_character_limit=14,
        chunk_pause_seconds=0.1,
        prune_keep=2,
        prune_after_step=1,
    )

    assert len(results) == 2
    pause_samples = int(0.1 * 24_000)
    for waveform, joined_text, predicted_frames in results:
        assert joined_text == "merhaba dünya. iyi günler."
        assert predicted_frames > 0.0
        # Two 8-frame chunks at hop 256, joined by the configured pause.
        assert waveform.shape == (2 * 8 * 256 + pause_samples,)

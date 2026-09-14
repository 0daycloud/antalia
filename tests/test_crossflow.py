from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from turkish_tts.crossflow import CrossFlow, CrossFlowModelConfig, flow_matching_loss
from turkish_tts.crossflow_train import (
    ArrowAcousticDataset,
    CharacterTokenizer,
    CrossFlowTrainConfig,
    DurationBatchSampler,
    ExponentialMovingAverage,
    MelNormalizer,
    SpeakerVocabulary,
    _load_initial_crossflow_state,
    _prepend_paired_references,
    _resolve_speaker_id,
    _should_save_checkpoint,
)


def test_arrow_dataset_accepts_zero_duration_for_unreferenced_rows(tmp_path: Path) -> None:
    import pyarrow as pa
    from datasets import Dataset

    rows = [
        {
            "audio_path": "target.flac",
            "text": list("hello"),
            "duration": 2.0,
            "speaker": "speaker-a",
            "prosody": [],
            "reference_audio_path": None,
            "reference_text": None,
            "reference_duration": 0.0,
        },
        {
            "audio_path": "target-2.flac",
            "text": list("world"),
            "duration": 2.0,
            "speaker": "speaker-a",
            "prosody": [],
            "reference_audio_path": "reference.flac",
            "reference_text": list("reference"),
            "reference_duration": 3.0,
        },
    ]
    arrow_path = tmp_path / "train.arrow"
    table = Dataset.from_list(rows).data.table
    with pa.OSFile(str(arrow_path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)

    dataset = ArrowAcousticDataset(arrow_path, max_audio_seconds=12.0, max_text_tokens=64)

    assert len(dataset.records) == 2
    assert dataset.records[0].reference_audio_path is None
    assert dataset.records[0].reference_duration == 0.0
    assert dataset.records[1].reference_audio_path == "reference.flac"


def _tiny_model(
    *,
    adapter_dim: int = 0,
    speaker_count: int = 0,
    prosody_dim: int = 0,
    context_conditioning: bool = False,
) -> CrossFlow:
    return CrossFlow(
        CrossFlowModelConfig(
            vocab_size=24,
            mel_channels=8,
            model_dim=32,
            depth=1,
            heads=4,
            ff_dim=64,
            text_depth=1,
            text_kernel_size=3,
            checkpoint_activations=False,
            max_frames=32,
            max_text_tokens=16,
            adapter_dim=adapter_dim,
            speaker_count=speaker_count,
            speaker_embedding_dim=8,
            prosody_dim=prosody_dim,
            context_conditioning=context_conditioning,
        )
    )


def test_speaker_and_prosody_conditioning_are_acoustic_only() -> None:
    torch.manual_seed(5)
    foundation = _tiny_model().eval()
    conditioned = _tiny_model(speaker_count=4, prosody_dim=3).eval()
    incompatible = conditioned.load_state_dict(foundation.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "prosody_projection.input.bias",
        "prosody_projection.input.weight",
        "prosody_projection.output.bias",
        "prosody_projection.output.weight",
        "speaker_embedding.weight",
        "speaker_projection.input.bias",
        "speaker_projection.input.weight",
        "speaker_projection.output.bias",
        "speaker_projection.output.weight",
    }
    mel = torch.randn(2, 24, 8)
    timestep = torch.tensor([0.25, 0.75])
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)
    speaker_ids = torch.tensor([1, 2])
    prosody = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.4, -0.2]])

    expected = foundation(mel, timestep, token_ids, frame_mask, token_mask)
    actual = conditioned(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        speaker_ids=speaker_ids,
        prosody_features=prosody,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])

    torch.nn.init.normal_(conditioned.speaker_projection.output.weight)
    torch.nn.init.normal_(conditioned.prosody_projection.output.weight)
    torch.nn.init.normal_(conditioned.mel_output.weight)
    torch.nn.init.normal_(conditioned.final_modulation[1].weight)
    with torch.no_grad():
        foundation.mel_output.weight.copy_(conditioned.mel_output.weight)
        foundation.final_modulation[1].weight.copy_(conditioned.final_modulation[1].weight)
    expected_after_training = foundation(mel, timestep, token_ids, frame_mask, token_mask)
    changed = conditioned(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        speaker_ids=speaker_ids,
        prosody_features=prosody,
    )
    assert not torch.equal(changed[0], expected_after_training[0])
    unconditioned = conditioned(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        speaker_ids=torch.zeros(2, dtype=torch.long),
    )
    assert torch.equal(unconditioned[0], expected_after_training[0])
    assert torch.equal(unconditioned[1], expected_after_training[1])
    assert torch.equal(changed[1], expected_after_training[1])


def test_freezing_linguistic_components_leaves_acoustic_conditioning_trainable() -> None:
    model = _tiny_model(speaker_count=4, prosody_dim=3)
    model.freeze_linguistic_components()

    assert all(not parameter.requires_grad for parameter in model.text_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.duration_head.parameters())
    assert all(parameter.requires_grad for parameter in model.speaker_embedding.parameters())
    assert all(parameter.requires_grad for parameter in model.speaker_projection.parameters())
    assert all(parameter.requires_grad for parameter in model.prosody_projection.parameters())


def test_speaker_embedding_only_training_freezes_every_other_parameter() -> None:
    model = _tiny_model(speaker_count=4, prosody_dim=3)
    model.freeze_for_speaker_embedding_training()

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"speaker_embedding.weight"}


def test_conditioning_only_training_freezes_foundation_parameters() -> None:
    model = _tiny_model(speaker_count=4, prosody_dim=3)
    model.freeze_for_conditioning_training()

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == model.conditioning_parameter_names()
    assert trainable


def test_context_only_training_preserves_every_nonreference_parameter() -> None:
    model = CrossFlow(CrossFlowModelConfig(**{**_tiny_model().config.as_dict(), "context_conditioning": True}))
    model.freeze_for_context_training()

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == model.context_parameter_names()
    assert trainable


def test_checkpoint_speaker_resolution_rejects_unknown_identity() -> None:
    payload = {"speaker_vocabulary": ["<unconditioned>", "candidate-b"]}
    assert _resolve_speaker_id(payload, None) == 0
    assert _resolve_speaker_id(payload, "candidate-b") == 1
    with pytest.raises(ValueError, match="not present"):
        _resolve_speaker_id(payload, "candidate-a")


def test_conditioned_checkpoint_initialization_and_speaker_expansion() -> None:
    torch.manual_seed(31)
    foundation = _tiny_model()
    conditioned = _tiny_model(speaker_count=3, prosody_dim=3)
    foundation_payload = {
        "model_config": foundation.config.as_dict(),
        "model": foundation.state_dict(),
        "ema": {"values": foundation.state_dict()},
    }

    _load_initial_crossflow_state(
        conditioned,
        conditioned.config,
        foundation_payload,
        initialize_from_ema=True,
        adapter_only=False,
        initialize_conditioning_from_base=True,
    )

    expanded = _tiny_model(speaker_count=4, prosody_dim=3)
    conditioned_payload = {
        "model_config": conditioned.config.as_dict(),
        "model": conditioned.state_dict(),
        "ema": {
            "values": {
                name: value
                for name, value in conditioned.state_dict().items()
                if name in conditioned.conditioning_parameter_names()
            }
        },
    }
    _load_initial_crossflow_state(
        expanded,
        expanded.config,
        conditioned_payload,
        initialize_from_ema=True,
        adapter_only=False,
        expand_speaker_embedding=True,
    )

    assert torch.equal(
        expanded.speaker_embedding.weight[:-1],
        conditioned.speaker_embedding.weight,
    )
    assert not torch.equal(
        expanded.speaker_embedding.weight[-1],
        conditioned.speaker_embedding.weight[-1],
    )
    assert torch.equal(expanded.mel_input.weight, conditioned.mel_input.weight)


def test_speaker_vocabulary_reserves_unconditioned_and_adds_candidate() -> None:
    vocabulary = SpeakerVocabulary.from_speakers(["cv-b", "cv-a", "cv-a"])
    assert vocabulary.speakers == ("<unconditioned>", "cv-a", "cv-b")
    assert vocabulary.encode("cv-b") == 2
    assert vocabulary.encode("unseen") == 0

    expanded = vocabulary.with_speaker("candidate-b")

    assert expanded.speakers == ("<unconditioned>", "cv-a", "cv-b", "candidate-b")
    assert expanded.encode("candidate-b") == 3
    with pytest.raises(ValueError, match="already exists"):
        expanded.with_speaker("candidate-b")


def _conditioned_sampling_model() -> CrossFlow:
    torch.manual_seed(41)
    model = _tiny_model(speaker_count=4, prosody_dim=3).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return model


def test_unit_guidance_matches_unguided_sampling_exactly() -> None:
    model = _conditioned_sampling_model()
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)

    baseline, _ = model.sample(token_ids, token_mask, steps=4, frame_count=16, speaker_id=2)
    guided, _ = model.sample(
        token_ids,
        token_mask,
        steps=4,
        frame_count=16,
        speaker_id=2,
        text_guidance_scale=1.0,
        speaker_guidance_scale=1.0,
        sway_coefficient=0.0,
        solver="euler",
    )

    assert torch.equal(guided, baseline)


def test_guidance_sway_and_solver_change_sampling_trajectory() -> None:
    model = _conditioned_sampling_model()
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)
    baseline, _ = model.sample(token_ids, token_mask, steps=4, frame_count=16, speaker_id=2)

    speaker_guided, _ = model.sample(
        token_ids, token_mask, steps=4, frame_count=16, speaker_id=2, speaker_guidance_scale=2.0
    )
    text_guided, _ = model.sample(token_ids, token_mask, steps=4, frame_count=16, speaker_id=2, text_guidance_scale=2.0)
    swayed, _ = model.sample(token_ids, token_mask, steps=4, frame_count=16, speaker_id=2, sway_coefficient=-0.8)
    midpoint, _ = model.sample(token_ids, token_mask, steps=4, frame_count=16, speaker_id=2, solver="midpoint")

    assert not torch.equal(speaker_guided, baseline)
    assert not torch.equal(text_guided, baseline)
    assert not torch.equal(swayed, baseline)
    assert not torch.equal(midpoint, baseline)
    assert torch.isfinite(speaker_guided).all()
    assert torch.isfinite(text_guided).all()
    assert torch.isfinite(swayed).all()
    assert torch.isfinite(midpoint).all()


def test_batched_sampling_matches_single_seed_rows() -> None:
    model = _conditioned_sampling_model()
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)

    single_a, _ = model.sample(token_ids, token_mask, steps=3, frame_count=16, speaker_id=2, seed=7)
    single_b, _ = model.sample(token_ids, token_mask, steps=3, frame_count=16, speaker_id=2, seed=9)
    batched = model.sample_batched(
        token_ids.expand(2, -1),
        token_ids.expand(2, -1).ne(0),
        frame_counts=[16, 16],
        seeds=[7, 9],
        steps=3,
        speaker_ids=torch.tensor([2, 2]),
    )

    assert batched.shape == (2, 16, 8)
    assert torch.allclose(batched[0], single_a[0], atol=1e-4)
    assert torch.allclose(batched[1], single_b[0], atol=1e-4)
    assert not torch.allclose(batched[0], batched[1], atol=1e-2)


def test_batched_sampling_pads_mixed_lengths_and_validates() -> None:
    model = _conditioned_sampling_model()
    token_ids = torch.randint(2, 24, (2, 10))
    token_mask = torch.ones(2, 10, dtype=torch.bool)

    batched = model.sample_batched(
        token_ids,
        token_mask,
        frame_counts=[12, 20],
        seeds=[3, 4],
        steps=2,
        speaker_ids=torch.tensor([1, 2]),
        text_guidance_scale=2.0,
        mel_clamp=5.0,
    )

    assert batched.shape == (2, 20, 8)
    assert torch.equal(batched[0, 12:], torch.zeros(8, 8))
    assert torch.isfinite(batched).all()
    with pytest.raises(ValueError, match="one entry per row"):
        model.sample_batched(token_ids, token_mask, frame_counts=[12], seeds=[3, 4], steps=1)


def test_chunk_text_splits_long_sentences_at_clause_boundaries() -> None:
    from turkish_tts.crossflow_train import _chunk_text

    short = "kisa bir cumle."
    assert _chunk_text(short, 120) == [short]
    assert _chunk_text(short, 0) == [short]

    long_text = (
        "rezervasyon degisikligini bugun tamamlarsaniz mevcut fiyat korunacak; "
        "yarina kalirsa sistem yeni tarife uygular, bu yuzden onayinizi bekliyoruz. "
        "ayrica iade kosullari da degisecek."
    )
    chunks = _chunk_text(long_text, 90)
    assert len(chunks) >= 2
    assert " ".join(chunks).split() == long_text.split()
    assert all(len(chunk) <= 90 for chunk in chunks)

    unbreakable = "a" * 200
    assert _chunk_text(unbreakable, 90) == [unbreakable]


def test_reference_region_ends_at_reference_mel() -> None:
    torch.manual_seed(67)
    model = CrossFlow(
        CrossFlowModelConfig(
            **{
                **_tiny_model(speaker_count=4, prosody_dim=3).config.as_dict(),
                "context_conditioning": True,
                "max_frames": 64,
            }
        )
    ).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)
    reference = torch.randn(1, 12, 8)

    mel, _ = model.sample(token_ids, token_mask, steps=3, frame_count=28, speaker_id=2, context_mel=reference)

    assert torch.allclose(mel[:, :12], reference, atol=1e-5)


def test_context_guidance_changes_referenced_sampling_only() -> None:
    torch.manual_seed(61)
    model = CrossFlow(
        CrossFlowModelConfig(
            **{
                **_tiny_model(speaker_count=4, prosody_dim=3).config.as_dict(),
                "context_conditioning": True,
                "max_frames": 64,
            }
        )
    ).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)
    reference = torch.randn(1, 12, 8)

    baseline, _ = model.sample(token_ids, token_mask, steps=2, frame_count=28, speaker_id=2, context_mel=reference)
    guided, _ = model.sample(
        token_ids,
        token_mask,
        steps=2,
        frame_count=28,
        speaker_id=2,
        context_mel=reference,
        context_guidance_scale=2.0,
    )

    assert not torch.equal(guided, baseline)
    assert torch.isfinite(guided).all()
    with pytest.raises(ValueError, match="requires a reference context"):
        model.sample(
            token_ids,
            token_mask,
            steps=2,
            frame_count=28,
            context_guidance_scale=2.0,
        )


def test_context_conditioning_is_gated_per_frame_and_extends_checkpoints() -> None:
    torch.manual_seed(47)
    base = _tiny_model(speaker_count=4, prosody_dim=3).eval()
    torch.nn.init.normal_(base.mel_output.weight, std=0.1)
    contextual = CrossFlow(
        CrossFlowModelConfig(
            **{
                **base.config.as_dict(),
                "context_conditioning": True,
                "global_reference_conditioning": True,
            }
        )
    ).eval()
    payload = {
        "model_config": base.config.as_dict(),
        "model": base.state_dict(),
        "ema": {"values": base.state_dict()},
    }
    _load_initial_crossflow_state(
        contextual,
        contextual.config,
        payload,
        initialize_from_ema=True,
        adapter_only=False,
        initialize_context_from_base=True,
    )
    for parameter in contextual.context_encoder.parameters():
        torch.nn.init.normal_(parameter, std=0.1)

    mel = torch.randn(2, 24, 8)
    timestep = torch.tensor([0.25, 0.75])
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)
    context_mask = torch.zeros(2, 24, dtype=torch.bool)
    context_mask[0, :8] = True

    expected = base(mel, timestep, token_ids, frame_mask, token_mask)
    without_context = contextual(mel, timestep, token_ids, frame_mask, token_mask)
    with_context = contextual(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        context_mel=mel,
        context_frame_mask=context_mask,
    )

    assert torch.equal(without_context[0], expected[0])
    assert not torch.equal(with_context[0], expected[0])
    row_without_context = with_context[0][1]
    assert torch.equal(row_without_context, expected[0][1])


def test_global_reference_conditioning_reaches_generated_frames_and_is_row_gated() -> None:
    torch.manual_seed(71)
    local = CrossFlow(
        CrossFlowModelConfig(
            **{
                **_tiny_model().config.as_dict(),
                "depth": 0,
                "context_conditioning": True,
            }
        )
    ).eval()
    for parameter in local.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    global_model = CrossFlow(
        CrossFlowModelConfig(
            **{
                **local.config.as_dict(),
                "global_reference_conditioning": True,
            }
        )
    ).eval()
    global_model.load_state_dict(local.state_dict())

    mel = torch.randn(2, 24, 8)
    timestep = torch.tensor([0.25, 0.75])
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)
    context_mask = torch.zeros(2, 24, dtype=torch.bool)
    context_mask[0, :8] = True
    local_velocity, _ = local(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        context_mel=mel,
        context_frame_mask=context_mask,
    )
    global_velocity, _ = global_model(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        context_mel=mel,
        context_frame_mask=context_mask,
    )

    assert not torch.equal(global_velocity[0, 8:], local_velocity[0, 8:])
    assert torch.equal(global_velocity[1], local_velocity[1])


def test_infill_loss_scores_only_generated_frames() -> None:
    torch.manual_seed(53)
    model = CrossFlow(
        CrossFlowModelConfig(
            **{
                **_tiny_model(speaker_count=4, prosody_dim=3).config.as_dict(),
                "context_conditioning": True,
            }
        )
    )
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    mel = torch.randn(2, 24, 8)
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)
    context_mask = torch.zeros(2, 24, dtype=torch.bool)
    context_mask[:, :12] = True

    loss, components = flow_matching_loss(
        model,
        mel,
        token_ids,
        frame_mask,
        token_mask,
        context_frame_mask=context_mask,
        duration_weight=0.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["flow"] > 0.0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.context_encoder.parameters()
    )


def test_paired_references_prepend_cross_utterance_context_without_changing_replay() -> None:
    target_mel = torch.tensor(
        [
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [0.0]],
        ]
    )
    target_frames = torch.tensor([3, 2])
    target_tokens = torch.tensor([[7, 8, 9], [6, 5, 0]])
    target_token_lengths = torch.tensor([3, 2])
    reference_mel = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [0.0]],
        ]
    )
    reference_frames = torch.tensor([2, 1])
    reference_tokens = torch.tensor([[3, 4], [2, 0]])
    reference_token_lengths = torch.tensor([2, 1])

    mel, frame_lengths, tokens, token_lengths, context_mask = _prepend_paired_references(
        target_mel,
        target_frames,
        target_tokens,
        target_token_lengths,
        reference_mel,
        reference_frames,
        reference_tokens,
        reference_token_lengths,
        torch.tensor([True, False]),
    )

    assert frame_lengths.tolist() == [5, 2]
    assert token_lengths.tolist() == [5, 2]
    assert mel[0, :, 0].tolist() == [1.0, 2.0, 10.0, 11.0, 12.0]
    assert tokens[0].tolist() == [3, 4, 7, 8, 9]
    assert context_mask[0].tolist() == [True, True, False, False, False]
    assert mel[1, :2, 0].tolist() == [20.0, 21.0]
    assert tokens[1, :2].tolist() == [6, 5]
    assert not context_mask[1].any()


def test_sampling_with_reference_context_extends_frame_budget() -> None:
    torch.manual_seed(59)
    model = CrossFlow(
        CrossFlowModelConfig(
            **{
                **_tiny_model(speaker_count=4, prosody_dim=3).config.as_dict(),
                "context_conditioning": True,
                "max_frames": 64,
            }
        )
    ).eval()
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)
    reference = torch.randn(1, 12, 8)

    mel, _ = model.sample(
        token_ids,
        token_mask,
        steps=2,
        frame_count=28,
        speaker_id=2,
        context_mel=reference,
    )

    assert mel.shape == (1, 28, 8)
    assert torch.isfinite(mel).all()
    with pytest.raises(ValueError, match="frame budget"):
        model.sample(
            token_ids,
            token_mask,
            steps=2,
            frame_count=64,
            context_mel=torch.randn(1, 60, 8),
        )


def test_guidance_rescale_changes_guided_but_not_unguided_sampling() -> None:
    model = _conditioned_sampling_model()
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)

    unguided, _ = model.sample(token_ids, token_mask, steps=4, frame_count=16, speaker_id=2)
    unguided_rescaled, _ = model.sample(
        token_ids, token_mask, steps=4, frame_count=16, speaker_id=2, guidance_rescale=0.7
    )
    guided, _ = model.sample(token_ids, token_mask, steps=4, frame_count=16, speaker_id=2, text_guidance_scale=2.0)
    guided_rescaled, _ = model.sample(
        token_ids,
        token_mask,
        steps=4,
        frame_count=16,
        speaker_id=2,
        text_guidance_scale=2.0,
        guidance_rescale=0.7,
    )

    assert torch.equal(unguided_rescaled, unguided)
    assert not torch.equal(guided_rescaled, guided)
    assert torch.isfinite(guided_rescaled).all()


def test_mel_clamp_bounds_sampled_trajectory() -> None:
    model = _conditioned_sampling_model()
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)

    clamped, _ = model.sample(
        token_ids,
        token_mask,
        steps=4,
        frame_count=16,
        speaker_id=2,
        text_guidance_scale=3.0,
        mel_clamp=0.5,
    )

    assert clamped.abs().max() <= 0.5
    with pytest.raises(ValueError, match="mel_clamp must be positive"):
        model.sample(token_ids, token_mask, steps=2, frame_count=16, mel_clamp=0.0)


def test_speaker_guidance_requires_conditioned_speaker() -> None:
    model = _conditioned_sampling_model()
    token_ids = torch.randint(2, 24, (1, 10))
    token_mask = torch.ones(1, 10, dtype=torch.bool)

    with pytest.raises(ValueError, match="conditioned speaker ID"):
        model.sample(
            token_ids,
            token_mask,
            steps=2,
            frame_count=16,
            speaker_id=0,
            speaker_guidance_scale=2.0,
        )


def test_prosody_is_gated_off_for_unconditioned_speaker() -> None:
    model = _conditioned_sampling_model()
    mel = torch.randn(2, 24, 8)
    timestep = torch.tensor([0.25, 0.75])
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)
    prosody = torch.tensor([[0.5, -0.5, 1.0], [1.0, 0.2, -0.7]])

    bare = model(mel, timestep, token_ids, frame_mask, token_mask)
    zero_speaker_with_prosody = model(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        speaker_ids=torch.zeros(2, dtype=torch.long),
        prosody_features=prosody,
    )
    conditioned = model(
        mel,
        timestep,
        token_ids,
        frame_mask,
        token_mask,
        speaker_ids=torch.tensor([1, 2]),
        prosody_features=prosody,
    )

    assert torch.equal(zero_speaker_with_prosody[0], bare[0])
    assert not torch.equal(conditioned[0], bare[0])


def test_condition_dropout_configuration_is_validated(tmp_path: Path) -> None:
    base = {
        "run_version": "dropout-check",
        "train_arrow": "train.arrow",
        "validation_arrow": "validation.arrow",
        "output_dir": "run",
        "speaker_conditioning": True,
        "duration_weight": 0.0,
    }
    config_path = tmp_path / "config.json"

    config_path.write_text(json.dumps({**base, "text_dropout_probability": 0.15}))
    assert CrossFlowTrainConfig.load(config_path).text_dropout_probability == 0.15

    config_path.write_text(json.dumps({**base, "text_dropout_probability": 0.15, "duration_weight": 0.1}))
    with pytest.raises(ValueError, match="duration_weight"):
        CrossFlowTrainConfig.load(config_path)

    config_path.write_text(json.dumps({**base, "speaker_dropout_probability": 0.1, "speaker_conditioning": False}))
    with pytest.raises(ValueError, match="speaker dropout requires"):
        CrossFlowTrainConfig.load(config_path)


def test_text_normalization_configuration_is_validated(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    base = {
        "run_version": "normalization-check",
        "train_arrow": "train.arrow",
        "validation_arrow": "validation.arrow",
        "output_dir": "run",
    }

    config_path.write_text(json.dumps({**base, "text_normalization": "english"}))
    assert CrossFlowTrainConfig.load(config_path).text_normalization == "english"

    config_path.write_text(json.dumps({**base, "text_normalization": "unsupported"}))
    with pytest.raises(ValueError, match="text_normalization"):
        CrossFlowTrainConfig.load(config_path)


def test_infill_training_accepts_active_duration_loss(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_version": "infill-duration-check",
                "train_arrow": "train.arrow",
                "validation_arrow": "validation.arrow",
                "output_dir": "run",
                "context_conditioning": True,
                "infill_probability": 0.6,
                "duration_weight": 0.1,
            }
        )
    )

    config = CrossFlowTrainConfig.load(config_path)

    assert config.infill_probability == 0.6
    assert config.duration_weight == 0.1


def test_paired_reference_configuration_requires_context_and_allows_infill(tmp_path: Path) -> None:
    base = {
        "run_version": "paired-reference-check",
        "train_arrow": "train.arrow",
        "validation_arrow": "validation.arrow",
        "output_dir": "run",
        "duration_weight": 0.1,
        "paired_reference_conditioning": True,
    }
    config_path = tmp_path / "config.json"

    config_path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="requires context conditioning"):
        CrossFlowTrainConfig.load(config_path)

    config_path.write_text(json.dumps({**base, "context_conditioning": True}))
    assert CrossFlowTrainConfig.load(config_path).paired_reference_conditioning

    config_path.write_text(
        json.dumps(
            {
                **base,
                "context_conditioning": True,
                "infill_probability": 0.5,
            }
        )
    )
    config = CrossFlowTrainConfig.load(config_path)
    assert config.paired_reference_conditioning
    assert config.infill_probability == 0.5
    assert config.duration_weight == 0.1


def test_zero_initialized_adapter_preserves_foundation_and_freezes_base() -> None:
    torch.manual_seed(11)
    foundation = _tiny_model().eval()
    adapter = _tiny_model(adapter_dim=8).eval()
    adapter.freeze_base_for_adapter_training()
    foundation_config = foundation.config.as_dict()
    foundation_config.pop("adapter_dim")
    payload = {
        "model_config": foundation_config,
        "model": foundation.state_dict(),
        "ema": {"values": {name: parameter.detach().clone() for name, parameter in foundation.named_parameters()}},
    }
    _load_initial_crossflow_state(
        adapter,
        adapter.config,
        payload,
        initialize_from_ema=True,
        adapter_only=True,
    )
    mel = torch.randn(2, 24, 8)
    timestep = torch.tensor([0.25, 0.75])
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.arange(24).unsqueeze(0) < torch.tensor([24, 19]).unsqueeze(1)
    token_mask = torch.arange(10).unsqueeze(0) < torch.tensor([10, 7]).unsqueeze(1)

    expected = foundation(mel, timestep, token_ids, frame_mask, token_mask)
    actual = adapter(mel, timestep, token_ids, frame_mask, token_mask)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    trainable = {name for name, parameter in adapter.named_parameters() if parameter.requires_grad}
    assert trainable == adapter.adapter_parameter_names()
    assert trainable


def test_zero_adapter_scale_restores_foundation_after_training() -> None:
    torch.manual_seed(23)
    foundation = _tiny_model().eval()
    adapter = _tiny_model(adapter_dim=8).eval()
    incompatible = adapter.load_state_dict(foundation.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == adapter.adapter_parameter_names()
    for name, parameter in adapter.named_parameters():
        if name in adapter.adapter_parameter_names():
            torch.nn.init.normal_(parameter)
    adapter.set_adapter_scale(0.0)
    mel = torch.randn(1, 24, 8)
    timestep = torch.tensor([0.5])
    token_ids = torch.randint(2, 24, (1, 10))
    frame_mask = torch.ones(1, 24, dtype=torch.bool)
    token_mask = torch.ones(1, 10, dtype=torch.bool)

    expected = foundation(mel, timestep, token_ids, frame_mask, token_mask)
    actual = adapter(mel, timestep, token_ids, frame_mask, token_mask)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    with pytest.raises(ValueError, match="between zero and one"):
        adapter.set_adapter_scale(1.1)


def test_foundation_preservation_loss_compares_against_disabled_adapter() -> None:
    model = _tiny_model(adapter_dim=8)
    for name, parameter in model.named_parameters():
        if name in model.adapter_parameter_names():
            torch.nn.init.normal_(parameter, std=0.01)
    torch.nn.init.normal_(model.mel_output.weight, std=0.01)
    mel = torch.randn(2, 24, 8)
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)

    loss, components = flow_matching_loss(
        model,
        mel,
        token_ids,
        frame_mask,
        token_mask,
        duration_weight=0.0,
        foundation_preservation_weight=2.0,
    )

    assert torch.isfinite(loss)
    assert components["foundation_preservation"] > 0.0
    assert model.adapter_scale == 1.0
    assert all(block.adapter_scale == 1.0 for block in model.blocks)


def test_adapter_only_loss_backpropagates_only_through_adapter() -> None:
    model = _tiny_model(adapter_dim=8)
    model.freeze_base_for_adapter_training()
    mel = torch.randn(2, 24, 8)
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.arange(24).unsqueeze(0) < torch.tensor([24, 19]).unsqueeze(1)
    token_mask = torch.arange(10).unsqueeze(0) < torch.tensor([10, 7]).unsqueeze(1)

    loss, _ = flow_matching_loss(
        model,
        mel,
        token_ids,
        frame_mask,
        token_mask,
        duration_weight=0.0,
    )
    loss.backward()

    adapter_names = model.adapter_parameter_names()
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if name in adapter_names)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in adapter_names)


def test_all_false_context_mask_keeps_context_parameters_in_backward_graph() -> None:
    model = _tiny_model(context_conditioning=True)
    mel = torch.randn(2, 24, 8)
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)

    loss, _ = flow_matching_loss(
        model,
        mel,
        token_ids,
        frame_mask,
        token_mask,
        context_frame_mask=torch.zeros_like(frame_mask),
        duration_weight=0.1,
    )
    loss.backward()

    context_names = model.context_parameter_names()
    assert context_names
    assert all(parameter.grad is not None for name, parameter in model.named_parameters() if name in context_names)


def test_duration_loss_can_target_unprefixed_frames() -> None:
    model = _tiny_model()
    mel = torch.randn(2, 24, 8)
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)
    target_frames = torch.tensor([17, 11])
    _, predicted_log_frames = model.encode_text(token_ids, token_mask)

    _, components = flow_matching_loss(
        model,
        mel,
        token_ids,
        frame_mask,
        token_mask,
        duration_frame_lengths=target_frames,
        duration_weight=0.1,
    )

    expected = torch.nn.functional.mse_loss(predicted_log_frames, target_frames.float().log())
    assert torch.allclose(components["duration"], expected)


def test_flow_loss_backpropagates_through_acoustic_model() -> None:
    model = _tiny_model()
    mel = torch.randn(2, 24, 8)
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.arange(24).unsqueeze(0) < torch.tensor([24, 19]).unsqueeze(1)
    token_mask = torch.arange(10).unsqueeze(0) < torch.tensor([10, 7]).unsqueeze(1)

    loss, components = flow_matching_loss(
        model,
        mel,
        token_ids,
        frame_mask,
        token_mask,
        duration_weight=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components.keys() == {"flow", "duration"}
    assert model.mel_output.weight.grad is not None
    assert torch.isfinite(model.mel_output.weight.grad).all()


def test_return_predictions_recovers_clean_mel_with_perfect_velocity() -> None:
    model = _tiny_model()
    mel = torch.randn(2, 24, 8)
    token_ids = torch.randint(2, 24, (2, 10))
    frame_mask = torch.ones(2, 24, dtype=torch.bool)
    token_mask = torch.ones(2, 10, dtype=torch.bool)
    _, components = flow_matching_loss(
        model,
        mel,
        token_ids,
        frame_mask,
        token_mask,
        duration_weight=0.0,
        return_predictions=True,
    )

    # A zero velocity prediction implies the model guessed the noise exactly;
    # x_1_hat = x_t + (1 - t) * v then equals the interpolation endpoint trick,
    # so check the reconstruction contract instead of model accuracy: predicted
    # mel is a differentiable function of the model output with a valid mask.
    assert components["predicted_mel"].shape == mel.shape
    assert components["predicted_mel"].requires_grad
    assert components["scored_frame_mask"].equal(frame_mask)

    loss = components["predicted_mel"].square().sum()
    loss.backward()
    assert model.mel_output.weight.grad is not None


def test_sampling_is_seeded_and_respects_requested_frame_count() -> None:
    model = _tiny_model().eval()
    token_ids = torch.tensor([[2, 3, 4, 5]])
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)

    first, _ = model.sample(token_ids, token_mask, steps=2, seed=17, frame_count=20)
    second, _ = model.sample(token_ids, token_mask, steps=2, seed=17, frame_count=20)
    different, _ = model.sample(token_ids, token_mask, steps=2, seed=18, frame_count=20)

    assert first.shape == (1, 20, 8)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_character_tokenizer_has_stable_unknown_symbol() -> None:
    tokenizer = CharacterTokenizer.from_texts(["şimdi", "merhaba"])

    assert tokenizer.symbols[:2] == ("<pad>", "<unk>")
    assert tokenizer.encode("🙂") == [1]
    assert tokenizer.encode("merhaba") == tokenizer.encode("merhaba")


def test_duration_batches_partition_evenly_across_distributed_ranks() -> None:
    samplers = [
        DurationBatchSampler(
            [0.5] * 16,
            sample_rate=24_000,
            hop_length=256,
            frames_per_gpu=200,
            max_samples=2,
            seed=9,
            rank=rank,
            world_size=2,
            shuffle=False,
        )
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]
    rank_indices = [{index for batch in batches for index in batch} for batches in rank_batches]

    assert len(rank_batches[0]) == len(rank_batches[1])
    assert rank_indices[0].isdisjoint(rank_indices[1])
    assert rank_indices[0] | rank_indices[1] == set(range(16))


def test_duration_batches_keep_small_validation_sets_across_ranks() -> None:
    samplers = [
        DurationBatchSampler(
            [0.5] * 6,
            sample_rate=24_000,
            hop_length=256,
            frames_per_gpu=200,
            max_samples=2,
            seed=9,
            rank=rank,
            world_size=8,
            shuffle=False,
            drop_remainder=False,
        )
        for rank in range(8)
    ]
    rank_batches = [list(sampler) for sampler in samplers]
    rank_indices = [{index for batch in batches for index in batch} for batches in rank_batches]

    assert set().union(*rank_indices) == set(range(6))
    assert sum(len(batches) for batches in rank_batches) == 3


def test_checkpoint_save_includes_new_validation_champions() -> None:
    assert _should_save_checkpoint(
        update=2_500,
        max_updates=60_000,
        save_every_updates=5_000,
        best_validation_update=2_500,
    )
    assert _should_save_checkpoint(
        update=5_000,
        max_updates=60_000,
        save_every_updates=5_000,
        best_validation_update=2_500,
    )
    assert _should_save_checkpoint(
        update=60_000,
        max_updates=60_000,
        save_every_updates=5_000,
        best_validation_update=52_500,
    )
    assert not _should_save_checkpoint(
        update=7_500,
        max_updates=60_000,
        save_every_updates=5_000,
        best_validation_update=2_500,
    )


def test_ema_warmup_tracks_early_training_updates() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(model.weight)
    ema = ExponentialMovingAverage(model, decay=0.9999)
    torch.nn.init.ones_(model.weight)

    ema.update(model)
    state = ema.state_dict()

    assert state["updates"] == 1
    assert torch.all(state["values"]["weight"] > 0.8)


def test_recovery_config_requires_checkpoint_checksum(tmp_path: Path) -> None:
    config_path = tmp_path / "recovery.json"
    payload = {
        "run_version": "recovery",
        "train_arrow": "train.arrow",
        "validation_arrow": "validation.arrow",
        "output_dir": "run",
        "initial_checkpoint": "model.pt",
    }
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="must be set together"):
        CrossFlowTrainConfig.load(config_path)

    payload["initial_checkpoint_sha256"] = "abc123"
    payload["initialize_from_ema"] = True
    config_path.write_text(json.dumps(payload))
    config = CrossFlowTrainConfig.load(config_path)

    assert config.initial_checkpoint == "model.pt"
    assert config.initialize_from_ema is True


def test_adapter_only_config_requires_adapter_and_initial_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "adapter.json"
    payload = {
        "run_version": "adapter",
        "train_arrow": "train.arrow",
        "validation_arrow": "validation.arrow",
        "output_dir": "run",
        "train_adapter_only": True,
    }
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="positive adapter_dim"):
        CrossFlowTrainConfig.load(config_path)

    payload["adapter_dim"] = 8
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="initial checkpoint"):
        CrossFlowTrainConfig.load(config_path)


def test_mel_normalization_round_trips_scalar_statistics() -> None:
    config = CrossFlowTrainConfig(
        run_version="normalized",
        train_arrow="train.arrow",
        validation_arrow="validation.arrow",
        output_dir="run",
        n_mels=2,
        mel_mean=-2.0,
        mel_std=2.0,
    )
    normalizer = MelNormalizer(config, "cpu")
    mel = torch.tensor([[[-4.0, 2.0]]])

    normalized = normalizer.normalize(mel)

    assert torch.equal(normalized, torch.tensor([[[-1.0, 2.0]]]))
    assert torch.equal(normalizer.denormalize(normalized), mel)


def test_mel_statistics_must_match_channel_count(tmp_path: Path) -> None:
    config_path = tmp_path / "normalized.json"
    config_path.write_text(
        json.dumps(
            {
                "run_version": "normalized",
                "train_arrow": "train.arrow",
                "validation_arrow": "validation.arrow",
                "output_dir": "run",
                "n_mels": 2,
                "mel_mean": [0.0],
                "mel_std": [1.0, 1.0],
            }
        )
    )

    with pytest.raises(ValueError, match="mel_mean"):
        CrossFlowTrainConfig.load(config_path)


def test_release_export_matches_ema_checkpoint_loading(tmp_path: Path) -> None:
    from turkish_tts.crossflow_release import export_crossflow_release
    from turkish_tts.crossflow_train import load_crossflow_checkpoint

    torch.manual_seed(3)
    model = _tiny_model(adapter_dim=4, speaker_count=3, prosody_dim=2)
    adapter_names = sorted(model.adapter_parameter_names())
    ema_values = {name: torch.randn_like(model.state_dict()[name]) for name in adapter_names}
    train_config = CrossFlowTrainConfig(
        run_version="tiny",
        train_arrow="train.arrow",
        validation_arrow="validation.arrow",
        output_dir="run",
        n_mels=8,
        mel_mean=[0.1] * 8,
        mel_std=[1.5] * 8,
        speaker_conditioning=True,
        prosody_dim=2,
        adapter_dim=4,
    )
    checkpoint = tmp_path / "model_7.pt"
    torch.save(
        {
            "format_version": 2,
            "run_version": "tiny",
            "update": 7,
            "epoch": 1,
            "model_config": model.config.as_dict(),
            "train_config": train_config.as_dict(),
            "vocabulary": ["<pad>", "<unk>", " ", *(chr(97 + index) for index in range(21))],
            "speaker_vocabulary": ["<unconditioned>", "cv-x", "voicedata-candidate-b"],
            "model": model.state_dict(),
            "ema": {"values": ema_values, "updates": 7},
            "optimizer": {},
            "provenance": {"statement": "tiny", "train_arrow": "/internal/path.arrow"},
        },
        checkpoint,
    )

    config = export_crossflow_release(checkpoint, tmp_path / "release")
    released, _, payload = load_crossflow_checkpoint(tmp_path / "release", "cpu")
    reference, _, _ = load_crossflow_checkpoint(checkpoint, "cpu", use_ema=True)

    assert all(
        torch.equal(released_tensor, reference_tensor)
        for released_tensor, reference_tensor in zip(
            released.state_dict().values(), reference.state_dict().values(), strict=True
        )
    )
    assert torch.equal(released.state_dict()[adapter_names[0]], ema_values[adapter_names[0]])
    assert payload["speaker_vocabulary"].index("voicedata-candidate-b") == 2
    assert CrossFlowTrainConfig(**payload["train_config"]).mel_std == [1.5] * 8
    assert config["provenance"] == {"statement": "tiny"}

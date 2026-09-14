# ruff: noqa: RUF001 -- Turkish test fixtures intentionally use dotless i
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("faster_whisper")
pytest.importorskip("torchaudio")

from turkish_tts.asr_labeling import (
    CtcResult,
    LabelingThresholds,
    run_ctc_consensus_and_alignment,
    run_whisper_pseudolabels,
)
from turkish_tts.common_voice_prepare import AsrResult
from turkish_tts.dataset_finalize import CombinedDatasetConfig, build_combined_candidate_dataset
from turkish_tts.manifests import ClipRecord, CollectionFormat, RightsState
from turkish_tts.normalize import normalize_for_ctc_alignment, normalize_for_model
from turkish_tts.quality_gates import (
    PrivacyFinding,
    quality_gate_summary,
    run_acoustic_fingerprint_gate,
    run_privacy_gate,
    structured_pii_categories,
)
from turkish_tts.segments import SegmentationConfig, segment_voicedata_records


def test_model_normalization_expands_turkish_dates_money_times_and_percentages() -> None:
    text = "Dr. Ayşe 29.10.2026'da saat 14:05'te ₺12,50 ve %3 ödedi."

    assert normalize_for_model(text) == (
        "doktor Ayşe yirmi dokuz ekim iki bin yirmi altı'da saat on dört sıfır beş'te "
        "on iki lira elli kuruş ve yüzde üç ödedi."
    )
    assert normalize_for_model("₺1.275,50") == "bin iki yüz yetmiş beş lira elli kuruş"
    assert normalize_for_ctc_alignment("IĞDIR'da QWX") == "ığdır da qwx"
    assert normalize_for_model("1.000 dakika ve 25 GB") == "bin dakika ve yirmi beş gigabayt"
    assert normalize_for_model("2.099 TL; 08.00–22.00; 7/24") == (
        "iki bin doksan dokuz Türk lirası; sekiz–yirmi iki; yedi yirmi dört"
    )


def test_voicedata_segmentation_writes_deterministic_pcm_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import faster_whisper.audio
    import faster_whisper.vad

    sample_rate = 48_000
    seconds = 3
    timeline = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    envelope = np.where(np.sin(2 * np.pi * 2 * timeline) > 0, 0.20, 0.01)
    audio = (envelope * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(source, audio, sample_rate, subtype="PCM_16")
    monkeypatch.setattr(
        faster_whisper.audio,
        "decode_audio",
        lambda _path, sampling_rate: np.zeros(sampling_rate * seconds, dtype=np.float32),
    )
    monkeypatch.setattr(
        faster_whisper.vad,
        "get_speech_timestamps",
        lambda _audio, _options: [{"start": 1_600, "end": 33_600}],
    )

    result = segment_voicedata_records(
        [_record("parent", source, duration_seconds=seconds)],
        output_dir=tmp_path / "segments",
        config=SegmentationConfig(min_estimated_snr_db=-1),
    )

    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.clip_id.startswith("parent-seg-")
    assert segment.transcript is None
    assert segment.metadata["parent_clip_id"] == "parent"
    assert segment.metadata["quality_filter_reasons"] == []
    assert sf.info(segment.audio_path).samplerate == 24_000
    assert sf.info(segment.audio_path).channels == 1
    assert result.report["candidate_segments"] == 1


def test_dual_asr_gate_records_independent_evidence_and_rejections(tmp_path: Path) -> None:
    records = [
        _record("pass", tmp_path / "pass.wav"),
        _record("reject", tmp_path / "reject.wav"),
    ]

    class FixtureWhisper:
        batch_size = 2

        def transcribe_batch(self, batch: Sequence[ClipRecord]) -> Sequence[AsrResult]:
            return [AsrResult("Merhaba dünya.", "tr", 1.0, 1.5, True) for _record in batch]

    whispered = run_whisper_pseudolabels(
        records,
        token=None,
        checkpoint_path=tmp_path / "whisper.checkpoint.jsonl",
        output_path=tmp_path / "whisper.jsonl",
        batch_size=2,
        transcriber=FixtureWhisper(),
    )

    class FixtureCtc:
        def transcribe_and_align(
            self,
            batch: Sequence[ClipRecord],
            target_texts: Sequence[str],
        ) -> Sequence[CtcResult]:
            assert target_texts == ["Merhaba dünya.", "Merhaba dünya."]
            return [
                CtcResult("merhaba dünya", 0.95, _alignment(0.90, 0.02, 0.03)),
                CtcResult("tamamen farklı", 0.20, _alignment(0.30, 0.40, 0.35)),
            ]

    labeled = run_ctc_consensus_and_alignment(
        whispered,
        thresholds=_labeling_thresholds(),
        token=None,
        checkpoint_path=tmp_path / "ctc.checkpoint.jsonl",
        output_path=tmp_path / "ctc.jsonl",
        batch_size=2,
        recognizer=FixtureCtc(),
    )

    assert labeled[0].metadata["quality_filter_reasons"] == []
    assert labeled[0].metadata["forced_alignment"]["word_timings"]
    assert set(labeled[1].metadata["quality_filter_reasons"]) >= {
        "ctc_low_confidence",
        "dual_asr_disagreement",
        "low_alignment_probability",
        "speech_before_transcript",
        "speech_after_transcript",
    }
    assert labeled[1].metadata["dual_asr"]["calibration_id"] == "fixture-calibration"


def test_privacy_gate_redacts_training_text_and_alignment_tokens(tmp_path: Path) -> None:
    record = _record("private", tmp_path / "private.wav").model_copy(
        update={
            "transcript": "Ayşe'nin telefonu 0532 123 45 67.",
            "normalized_transcript": "Ayşe'nin telefonu sıfır beş üç iki.",
            "metadata": {
                "quality_filter_reasons": [],
                "quality_stage": "ctc_consensus_alignment",
                "whisper_asr": {"model": "fixture", "text": "Ayşe'nin telefonu."},
                "ctc_asr": {"model": "fixture", "text": "ayşe'nin telefonu"},
                "forced_alignment": {
                    "target": "ayşe'nin telefonu",
                    "mean_token_probability": 0.9,
                    "word_timings": [{"word": "ayşe"}],
                    "phoneme_timings": [{"grapheme": "a"}],
                },
            },
        }
    )

    class FixturePrivacyScanner:
        def scan(self, text: str) -> Sequence[PrivacyFinding]:
            assert "Ayşe" in text
            return [PrivacyFinding("person_name", "fixture")]

    gated = run_privacy_gate(
        [record],
        token=None,
        output_path=tmp_path / "privacy.jsonl",
        detector=FixturePrivacyScanner(),
    )

    assert gated[0].transcript is None
    assert gated[0].normalized_transcript is None
    assert gated[0].metadata["quality_filter_reasons"] == ["pii_detected"]
    assert "text" not in gated[0].metadata["whisper_asr"]
    assert "target" not in gated[0].metadata["forced_alignment"]
    assert "word_timings" not in gated[0].metadata["forced_alignment"]


def test_structured_pii_detector_covers_turkish_identifiers() -> None:
    findings = structured_pii_categories(
        "TC 10000000146, telefon 0532 123 45 67, e-posta kisi@example.com, IBAN TR33 0006 1005 1978 6457 8413 26"
    )

    assert findings == ("email", "iban", "phone_number", "turkish_identity_number")


def test_acoustic_fingerprint_rejects_duplicate_audio(tmp_path: Path) -> None:
    sample_rate = 16_000
    timeline = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 330 * timeline)).astype(np.float32)
    source = tmp_path / "same.wav"
    sf.write(source, audio, sample_rate, subtype="PCM_16")

    gated = run_acoustic_fingerprint_gate(
        [_record("first", source), _record("second", source)],
        output_path=tmp_path / "fingerprinted.jsonl",
    )

    assert gated[0].metadata["quality_filter_reasons"] == []
    assert gated[1].metadata["quality_filter_reasons"] == ["duplicate_acoustic_fingerprint"]
    assert gated[1].metadata["duplicate_of_clip_id"] == "first"
    summary = quality_gate_summary(gated)
    assert summary["accepted_records"] == 1
    assert summary["rejection_reasons"] == {"duplicate_acoustic_fingerprint": 1}


def test_combined_finalizer_preserves_public_splits_and_separates_target_parents(tmp_path: Path) -> None:
    audio_paths: list[Path] = []
    for index in range(9):
        path = tmp_path / f"{index}.wav"
        sf.write(path, np.zeros(1_600, dtype=np.float32), 16_000, subtype="PCM_16")
        audio_paths.append(path)
    public_splits = ("train", "validation", "test")
    common_voice = [
        _record(f"cv-{split}", audio_paths[index]).model_copy(
            update={
                "source_dataset": "mozilla-common-voice-scripted-speech",
                "source_version": "26.0",
                "source_split": split,
                "transcript": "2026 yılında merhaba.",
                "speaker_id": f"cv-speaker-{split}",
                "license_id": "CC0-1.0+Mozilla-Data-Collective-Terms",
                "metadata": {
                    "quality_filter_reasons": [],
                    "common_voice_asr": {"character_error_rate": 0.01},
                },
            }
        )
        for index, split in enumerate(public_splits)
    ]
    fleurs = [
        _record(f"fleurs-{split}", audio_paths[index + 3]).model_copy(
            update={
                "source_dataset": "google-fleurs",
                "source_version": "fixture-revision",
                "source_split": split,
                "transcript": "Merhaba dünya.",
                "speaker_id": None,
                "license_id": "CC-BY-4.0",
                "metadata": {"quality_filter_reasons": []},
            }
        )
        for index, split in enumerate(public_splits)
    ]
    voicedata = [
        _record(f"vd-{index}", audio_paths[index + 6]).model_copy(
            update={
                "transcript": "Bugün güzel bir gün.",
                "metadata": {
                    "candidate_a": True,
                    "parent_clip_id": f"parent-{index}",
                    "quality_filter_reasons": [],
                    "privacy_scan": {"passed": True},
                    "dual_asr": {"character_error_rate": 0.01},
                    "forced_alignment": {"mean_token_probability": 0.9},
                    "speaker_consistency": {"assessed": False},
                    "acoustic_fingerprint": f"fingerprint-{index}",
                },
            }
        )
        for index in range(3)
    ]
    voicedata.append(
        voicedata[0].model_copy(
            update={
                "clip_id": "vd-rejected",
                "metadata": {
                    **voicedata[0].metadata,
                    "quality_filter_reasons": ["dual_asr_disagreement"],
                },
            }
        )
    )

    result = build_combined_candidate_dataset(
        common_voice,
        fleurs,
        voicedata,
        output_dir=tmp_path / "combined",
        prefix="fixture",
        config=CombinedDatasetConfig(
            max_public_speaker_seconds=60,
            max_common_voice_seconds=60,
            max_fleurs_seconds=60,
        ),
    )

    assert len(result.accepted) == 9
    assert len(result.train) + len(result.validation) + len(result.test) == 9
    assert {record.source_split for record in result.validation} == {"validation"}
    assert {record.metadata["parent_clip_id"] for record in result.train if record.clip_id.startswith("vd-")} != {
        record.metadata["parent_clip_id"] for record in result.test if record.clip_id.startswith("vd-")
    }
    assert next(record for record in result.accepted if record.clip_id == "cv-train").normalized_transcript == (
        "iki bin yirmi altı yılında merhaba."
    )
    assert (tmp_path / "combined" / "fixture.accepted.jsonl").is_file()
    assert result.report["training"]["records"] == 9
    assert result.report["excluded_input_records"]["voicedata-turkish"] == 1


def _record(clip_id: str, audio_path: Path, *, duration_seconds: float = 2) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        source_dataset="voicedata-turkish",
        source_version="2026-07-30",
        source_split="candidate",
        source_row_id=clip_id,
        audio_path=str(audio_path),
        transcript=None,
        normalized_transcript=None,
        speaker_id="speaker-a",
        collection_format=CollectionFormat.MONOLOGUE,
        rights_state=RightsState.ALLOWED,
        license_id="voicedata-commercial-voice-consent-v1",
        duration_seconds=duration_seconds,
        sample_rate_hz=48_000,
        metadata={
            "candidate_a": True,
            "quality_filter_reasons": [],
            "quality_stage": "segmentation",
        },
    )


def _alignment(probability: float, leading: float, trailing: float) -> dict[str, object]:
    return {
        "model": "fixture",
        "revision": "fixture",
        "target": "merhaba dünya",
        "mean_token_probability": probability,
        "min_token_probability": probability,
        "leading_unaligned_ratio": leading,
        "trailing_unaligned_ratio": trailing,
        "aligned_span_ratio": 1 - leading - trailing,
        "word_timings": [{"word": "merhaba", "start_seconds": 0.1, "end_seconds": 0.8}],
        "phoneme_timings": [{"grapheme": "m", "phoneme": "m", "start_seconds": 0.1, "end_seconds": 0.2}],
    }


def _labeling_thresholds() -> LabelingThresholds:
    return LabelingThresholds(
        calibration_id="fixture-calibration",
        calibration_source="fixture",
        calibration_records=100,
        max_consensus_cer=0.20,
        min_ctc_confidence=0.50,
        min_alignment_probability=0.50,
        max_leading_unaligned_ratio=0.20,
        max_trailing_unaligned_ratio=0.20,
        min_characters_per_second=1.0,
        max_characters_per_second=20.0,
    )

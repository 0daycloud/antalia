from __future__ import annotations

# ruff: noqa: RUF001 -- Turkish fixture text intentionally uses dotless i
import importlib.util
import math
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from turkish_tts.manifests import ClipRecord, CollectionFormat, RightsState
from turkish_tts.normalize import normalize_orthography
from turkish_tts.scripted_segments import (
    ElevenLabsScribeTimedTranscriber,
    ScriptedSegmentationConfig,
    TimedTranscript,
    TimedWord,
    segment_scripted_records,
)

NEEDS_TORCHAUDIO = pytest.mark.skipif(
    importlib.util.find_spec("torchaudio") is None,
    reason="segmentation resampling requires torchaudio (install the local-asr extra)",
)


class _FakeTranscriber:
    model_name = "fixture-timed-asr"

    def __init__(self, words: list[str]) -> None:
        self._words = words
        self.calls = 0

    def transcribe(self, _path: Path) -> TimedTranscript:
        self.calls += 1
        timed = tuple(
            TimedWord(word, 0.5 + index * 0.5, 0.9 + index * 0.5, 0.99) for index, word in enumerate(self._words)
        )
        return TimedTranscript(
            text=" ".join(self._words),
            language="tr",
            language_probability=0.99,
            words=timed,
        )


@NEEDS_TORCHAUDIO
def test_scripted_segmentation_preserves_exact_script_and_reuses_asr_cache(tmp_path: Path) -> None:
    script = 'Birinci cümle bitti. "İkinci cümle başladı ve güzel ilerledi."'
    parent = _parent(tmp_path, script)
    transcriber = _FakeTranscriber(
        ["Birinci", "cümle", "bitti", "İkinci", "cümle", "başladı", "ve", "güzel", "ilerledi"]
    )
    config = ScriptedSegmentationConfig(
        target_min_seconds=1.0,
        target_max_seconds=4.0,
        absolute_max_seconds=6.0,
        min_estimated_snr_db=-20.0,
        min_active_frame_ratio=0.0,
    )
    cache_dir = tmp_path / "cache"

    first = segment_scripted_records(
        [parent],
        output_dir=tmp_path / "first",
        transcriber=transcriber,
        transcript_cache_dir=cache_dir,
        config=config,
    )
    second = segment_scripted_records(
        [parent],
        output_dir=tmp_path / "second",
        transcriber=transcriber,
        transcript_cache_dir=cache_dir,
        config=config,
    )

    assert transcriber.calls == 1
    assert len(first.segments) == len(second.segments) == 2
    assert first.report["accepted_segments"] == 2
    combined = " ".join(record.transcript or "" for record in first.segments)
    assert normalize_orthography(combined) == normalize_orthography(script)
    assert first.segments[0].transcript == "Birinci cümle bitti."
    assert first.segments[1].transcript == '"İkinci cümle başladı ve güzel ilerledi."'
    assert all(record.sample_rate_hz == 24_000 for record in first.segments)
    assert all((record.duration_seconds or 0) <= config.absolute_max_seconds for record in first.segments)
    assert all(not record.metadata["quality_filter_reasons"] for record in first.segments)


@NEEDS_TORCHAUDIO
def test_scripted_segmentation_marks_misread_audio_instead_of_trusting_script(tmp_path: Path) -> None:
    script = "Bugün sakin ve anlaşılır konuşuyorum."
    parent = _parent(tmp_path, script)
    transcriber = _FakeTranscriber(["tamamen", "farklı", "bir", "metin", "okundu"])
    result = segment_scripted_records(
        [parent],
        output_dir=tmp_path / "segments",
        transcriber=transcriber,
        config=ScriptedSegmentationConfig(min_active_frame_ratio=0.0, min_estimated_snr_db=-20.0),
    )

    assert len(result.segments) == 1
    reasons = result.segments[0].metadata["quality_filter_reasons"]
    assert "script_word_alignment_low" in reasons
    assert "script_asr_cer_high" in reasons
    assert result.report["accepted_segments"] == 0
    assert result.report["rejected_segments"] == 1


def test_scribe_timed_transcriber_requests_word_timestamps_and_maps_response(tmp_path: Path) -> None:
    audio_path = tmp_path / "recording.mp3"
    audio_path.write_bytes(b"fixture-audio")

    def respond(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert request.headers["xi-api-key"] == "fixture-key"
        assert request.url.params.get("enable_logging") is None
        assert b'name="model_id"' in body and b"scribe_v2" in body
        assert b'name="timestamps_granularity"' in body and b"word" in body
        return httpx.Response(
            200,
            json={
                "text": "Günaydın.",
                "language_code": "tur",
                "language_probability": 0.99,
                "words": [
                    {"text": "Günaydın.", "start": 0.2, "end": 0.9, "type": "word", "logprob": -0.1},
                    {"text": " ", "start": 0.9, "end": 0.95, "type": "spacing", "logprob": -0.01},
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        transcript = ElevenLabsScribeTimedTranscriber(api_key="fixture-key", client=client).transcribe(audio_path)

    assert transcript.text == "Günaydın."
    assert transcript.language == "tr"
    assert transcript.language_probability == 0.99
    assert len(transcript.words) == 1
    assert transcript.words[0].text == "Günaydın."
    assert math.isclose(transcript.words[0].probability, math.exp(-0.1))


def _parent(tmp_path: Path, script: str) -> ClipRecord:
    audio_path = tmp_path / f"parent-{len(list(tmp_path.glob('parent-*.wav')))}.wav"
    sample_rate = 48_000
    seconds = 8
    time = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 220 * time)
    sf.write(audio_path, audio, sample_rate, subtype="PCM_24")
    return ClipRecord(
        clip_id=f"parent-{audio_path.stem}",
        source_dataset="voicedata-scripted-campaign",
        source_version="fixture-v1",
        source_split="approved-scripted-parent",
        source_row_id=audio_path.stem,
        audio_path=str(audio_path),
        transcript=script,
        normalized_transcript=script,
        speaker_id="vd-speaker-fixture",
        collection_format=CollectionFormat.SCRIPTED,
        rights_state=RightsState.ALLOWED,
        license_id="VoiceData-Terms+Biometric-Consent",
        duration_seconds=seconds,
        sample_rate_hz=sample_rate,
        sha256="fixture-parent-sha256",
        metadata={"script_ground_truth": True, "quality_filter_reasons": []},
    )

import math
import struct
import wave
from pathlib import Path

from turkish_tts.audio import probe_audio
from turkish_tts.common_voice import iter_common_voice_validated
from turkish_tts.common_voice_prepare import (
    COMMON_VOICE_LICENSE_ID,
    AsrResult,
    annotate_common_voice_asr,
    audit_common_voice_records,
    deterministic_speaker_split,
    finalize_common_voice_splits,
    transcribe_common_voice_with_checkpoints,
)
from turkish_tts.fleurs import FLEURS_LICENSE_ID, materialize_fleurs_rows
from turkish_tts.manifests import ClipRecord, CollectionFormat, RightsState, iter_jsonl, write_jsonl
from turkish_tts.normalize import (
    normalize_for_asr_comparison,
    normalize_for_scoring,
    normalize_orthography,
    turkish_lower,
)


def test_turkish_normalization_preserves_dotted_and_dotless_i() -> None:
    assert turkish_lower("IĞDIR İZMİR") == "ığdır izmir"  # noqa: RUF001 -- Turkish casing contract
    assert normalize_orthography("  Merhaba,\n dünya!  ") == "Merhaba, dünya!"
    assert normalize_for_asr_comparison("İzmir'de saat 14.30.") == "izmir'de saat 14 30"


def test_scoring_normalization_spells_digits_like_spoken_turkish() -> None:
    assert normalize_for_scoring("3700") == "üç bin yedi yüz"
    assert normalize_for_scoring("Fiyat %12,5 arttı.") == "fiyat yüzde on iki virgül beş arttı"  # noqa: RUF001
    assert normalize_for_scoring("1.499 lira") == "bin dört yüz doksan dokuz lira"
    assert normalize_for_scoring("Sayaç 18,904 kWh gösteriyor.") == (
        "sayaç on sekiz virgül dokuz sıfır dört kwh gösteriyor"  # noqa: RUF001
    )
    spoken = "üç bin yedi yüz seksen iki lira doksan beş kuruş"
    assert normalize_for_scoring(spoken) == spoken


def test_common_voice_validated_rows_become_anonymous_allowed_records(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "sample.mp3").write_bytes(b"audio")
    tsv = tmp_path / "validated.tsv"
    tsv.write_text(
        "client_id\tpath\tsentence_id\tsentence\tup_votes\tdown_votes\tlocale\n"
        "private-client\tsample.mp3\tsentence-1\tIĞDIR ve İzmir.\t2\t0\ttr\n",
        encoding="utf-8",
    )

    records = list(iter_common_voice_validated(tsv_path=tsv, clips_dir=clips, version="26.0"))
    assert len(records) == 1
    record = records[0]
    assert record.transcript == "IĞDIR ve İzmir."
    assert record.language == "tr"
    assert record.collection_format == CollectionFormat.SCRIPTED
    assert record.rights_state == RightsState.ALLOWED
    assert record.license_id == COMMON_VOICE_LICENSE_ID
    assert record.speaker_id is not None and "private-client" not in record.speaker_id


def test_manifest_round_trip_is_lossless(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "sample.mp3").write_bytes(b"audio")
    tsv = tmp_path / "validated.tsv"
    tsv.write_text(
        "client_id\tpath\tsentence\nclient\tsample.mp3\tMerhaba dünya.\n",
        encoding="utf-8",
    )
    original = list(iter_common_voice_validated(tsv_path=tsv, clips_dir=clips, version="26.0"))
    manifest = tmp_path / "manifest.jsonl"

    assert write_jsonl(manifest, original) == 1
    assert list(iter_jsonl(manifest)) == original


def test_audio_probe_reports_signal_contract(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 1_600)

    probe = probe_audio(audio_path)

    assert probe.duration_seconds == 0.1
    assert probe.sample_rate_hz == 16_000
    assert probe.channels == 1
    assert probe.codec == "pcm_s16le"
    assert len(probe.sha256) == 64


def test_fleurs_rows_materialize_with_pinned_provenance_and_human_transcripts(tmp_path: Path) -> None:
    cached_audio = tmp_path / "cached.flac"
    cached_audio.write_bytes(b"flac-audio")
    rows = [
        {
            "id": 101,
            "num_samples": 32_000,
            "audio": {"path": "speaker-101.wav", "bytes": b"wav-audio"},
            "transcription": "  Merhaba,\n dünya! ",
            "raw_transcription": "Merhaba, dünya!",
            "gender": 1,
            "language": "Turkish",
            "lang_id": 80,
            "lang_group_id": 3,
        },
        {
            "id": 101,
            "num_samples": 16_000,
            "audio": {"path": str(cached_audio), "bytes": None},
            "transcription": "Bugün nasılsınız?",  # noqa: RUF001 -- Turkish text fixture
            "raw_transcription": "Bugün nasılsınız?",  # noqa: RUF001 -- Turkish text fixture
            "gender": 0,
            "language": "Turkish",
            "lang_id": 80,
            "lang_group_id": 3,
        },
    ]

    records = list(
        materialize_fleurs_rows(
            rows,
            split="train",
            output_dir=tmp_path / "fleurs",
            revision="commit-sha",
        )
    )

    assert len(records) == 2
    assert len({record.clip_id for record in records}) == 2
    assert len({record.audio_path for record in records}) == 2
    assert [record.source_row_id for record in records] == ["train:0", "train:1"]
    assert records[0].transcript == "Merhaba, dünya!"
    assert records[0].source_version == "commit-sha"
    assert records[0].source_split == "train"
    assert records[0].license_id == FLEURS_LICENSE_ID
    assert records[0].duration_seconds == 2
    assert records[0].sample_rate_hz == 16_000
    assert len(records[0].sha256 or "") == 64
    assert Path(records[0].audio_path).read_bytes() == b"wav-audio"
    assert Path(records[1].audio_path).read_bytes() == b"flac-audio"


def test_common_voice_audit_adds_rights_metrics_and_exact_duplicate_rejection(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    _write_speech_like_wav(audio_path)
    records = [
        ClipRecord(
            clip_id=f"clip-{index}",
            source_dataset="mozilla-common-voice-scripted-speech",
            source_version="26.0",
            source_split="validated",
            source_row_id=f"row-{index}",
            audio_path=str(audio_path),
            transcript="Merhaba dünya.",
            normalized_transcript="Merhaba dünya.",
            speaker_id=f"speaker-{index}",
            collection_format=CollectionFormat.SCRIPTED,
            rights_state=RightsState.ALLOWED,
            license_id="CC0-1.0",
        )
        for index in range(2)
    ]

    audited = audit_common_voice_records(records, max_concurrency=2)

    assert audited[0].license_id == COMMON_VOICE_LICENSE_ID
    assert audited[0].sha256 and len(audited[0].sha256) == 64
    assert audited[0].metadata["quality_filter_reasons"] == []
    assert audited[0].metadata["dataset_restrictions"] == [
        "do_not_attempt_speaker_identification",
        "do_not_rehost_source_dataset",
        "respect_mozilla_data_collective_terms",
    ]
    assert audited[1].metadata["quality_filter_reasons"] == ["duplicate_audio"]


def test_common_voice_asr_agreement_marks_language_and_transcript_failures(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    _write_speech_like_wav(audio_path)
    record = ClipRecord(
        clip_id="clip-asr",
        source_dataset="mozilla-common-voice-scripted-speech",
        source_version="26.0",
        source_split="validated",
        source_row_id="row-asr",
        audio_path=str(audio_path),
        transcript="Merhaba dünya.",
        normalized_transcript="Merhaba dünya.",
        speaker_id="speaker-asr",
        collection_format=CollectionFormat.SCRIPTED,
        rights_state=RightsState.ALLOWED,
        license_id=COMMON_VOICE_LICENSE_ID,
        duration_seconds=2,
        metadata={"quality_filter_reasons": []},
    )

    accepted = annotate_common_voice_asr(
        record,
        transcriber=lambda _path: AsrResult("Merhaba dünya.", "tr", 0.99, 1.5),
        model_name="fixture",
    )
    rejected = annotate_common_voice_asr(
        record,
        transcriber=lambda _path: AsrResult("completely unrelated", "en", 0.95, 1.5),
        model_name="fixture",
    )

    assert accepted.metadata["quality_filter_reasons"] == []
    assert rejected.metadata["quality_filter_reasons"] == ["transcript_mismatch", "wrong_language"]


def test_common_voice_asr_checkpoint_resumes_only_missing_records(tmp_path: Path) -> None:
    records = [
        ClipRecord(
            clip_id=f"clip-{index}",
            source_dataset="mozilla-common-voice-scripted-speech",
            source_version="26.0",
            source_split="validated",
            source_row_id=f"row-{index}",
            audio_path=str(tmp_path / f"{index}.wav"),
            transcript="Merhaba dünya.",
            normalized_transcript="Merhaba dünya.",
            speaker_id=f"speaker-{index}",
            collection_format=CollectionFormat.SCRIPTED,
            rights_state=RightsState.ALLOWED,
            license_id=COMMON_VOICE_LICENSE_ID,
            duration_seconds=2,
            metadata={"quality_filter_reasons": []},
        )
        for index in range(2)
    ]
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpointed = annotate_common_voice_asr(
        records[0],
        transcriber=lambda _path: AsrResult("Merhaba dünya.", "tr", 0.99, 1.5),
        model_name="fixture",
    )
    write_jsonl(checkpoint, [checkpointed])
    calls: list[Path] = []

    def transcriber(path: Path) -> AsrResult:
        calls.append(path)
        return AsrResult("Merhaba dünya.", "tr", 0.99, 1.5)

    completed = transcribe_common_voice_with_checkpoints(
        records,
        transcriber=transcriber,
        model_name="fixture",
        checkpoint_path=checkpoint,
        output_path=tmp_path / "output.jsonl",
        max_concurrency=2,
    )

    assert calls == [Path(records[1].audio_path)]
    assert [record.clip_id for record in completed] == ["clip-0", "clip-1"]
    assert [record.clip_id for record in iter_jsonl(tmp_path / "output.jsonl")] == ["clip-0", "clip-1"]


def test_common_voice_batch_asr_skips_acoustic_rejections(tmp_path: Path) -> None:
    base = ClipRecord(
        clip_id="clip-0",
        source_dataset="mozilla-common-voice-scripted-speech",
        source_version="26.0",
        source_split="validated",
        source_row_id="row-0",
        audio_path=str(tmp_path / "0.wav"),
        transcript="Merhaba dünya.",
        normalized_transcript="Merhaba dünya.",
        speaker_id="speaker-0",
        collection_format=CollectionFormat.SCRIPTED,
        rights_state=RightsState.ALLOWED,
        license_id=COMMON_VOICE_LICENSE_ID,
        duration_seconds=2,
        metadata={"quality_filter_reasons": []},
    )
    records = [
        base,
        base.model_copy(
            update={
                "clip_id": "clip-rejected",
                "metadata": {
                    "quality_filter_reasons": ["level_too_low"],
                    "quality_stage": "acoustic_audit",
                },
            }
        ),
        base.model_copy(update={"clip_id": "clip-2", "source_row_id": "row-2"}),
    ]

    class FixtureBatchTranscriber:
        batch_size = 3

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def transcribe_batch(self, batch: list[ClipRecord]) -> list[AsrResult]:
            self.calls.append([record.clip_id for record in batch])
            return [AsrResult("Merhaba dünya.", "tr", 0.99, 1.5) for _record in batch]

    transcriber = FixtureBatchTranscriber()
    completed = transcribe_common_voice_with_checkpoints(
        records,
        transcriber=transcriber,
        model_name="fixture-batch",
        checkpoint_path=tmp_path / "batch.checkpoint.jsonl",
        output_path=tmp_path / "batch.output.jsonl",
    )

    assert transcriber.calls == [["clip-0", "clip-2"]]
    assert completed[0].metadata["quality_filter_reasons"] == []
    assert completed[1].metadata["quality_filter_reasons"] == ["level_too_low"]
    assert completed[2].metadata["common_voice_asr"]["model"] == "fixture-batch"


def test_common_voice_finalization_creates_speaker_disjoint_splits(tmp_path: Path) -> None:
    speakers: dict[str, str] = {}
    index = 0
    while len(speakers) < 3:
        speaker_id = f"speaker-{index}"
        speakers.setdefault(deterministic_speaker_split(speaker_id), speaker_id)
        index += 1
    records = [
        ClipRecord(
            clip_id=f"clip-{split}",
            source_dataset="mozilla-common-voice-scripted-speech",
            source_version="26.0",
            source_split="validated",
            source_row_id=f"row-{split}",
            audio_path=str(tmp_path / f"{split}.wav"),
            transcript="Merhaba.",
            normalized_transcript="Merhaba.",
            speaker_id=speaker_id,
            collection_format=CollectionFormat.SCRIPTED,
            rights_state=RightsState.ALLOWED,
            license_id=COMMON_VOICE_LICENSE_ID,
            duration_seconds=2,
            metadata={"quality_filter_reasons": []},
        )
        for split, speaker_id in speakers.items()
    ]
    records.append(
        records[0].model_copy(
            update={
                "clip_id": "clip-rejected",
                "metadata": {"quality_filter_reasons": ["transcript_mismatch"]},
            }
        )
    )

    summary = finalize_common_voice_splits(records, output_dir=tmp_path / "splits", prefix="cv")

    assert summary["accepted_records"] == 3
    assert summary["rejected_records"] == 1
    split_speakers = {
        split: {record.speaker_id for record in iter_jsonl(tmp_path / "splits" / f"cv.{split}.jsonl")}
        for split in ("train", "validation", "test")
    }
    assert all(split_speakers.values())
    assert not split_speakers["train"] & split_speakers["validation"]
    assert not split_speakers["train"] & split_speakers["test"]
    assert not split_speakers["validation"] & split_speakers["test"]


def _write_speech_like_wav(path: Path) -> None:
    sample_rate = 16_000
    frames = []
    for index in range(sample_rate * 2):
        phase = 2 * math.pi * 220 * index / sample_rate
        amplitude = 0 if index < sample_rate // 2 else int(0.2 * 32767 * math.sin(phase))
        frames.append(struct.pack("<h", amplitude))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(frames))







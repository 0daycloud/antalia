from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest
import soundfile as sf

from turkish_tts.baseline import (
    BASELINE_DATASET_VERSION,
    F5_WEIGHT_LICENSE,
    REQUIRED_EVALUATION_CATEGORIES,
    VOICEDATA_DATASET_VERSION,
    architecture_decision,
    load_evaluation_suite,
    prepare_candidate_baseline_data,
    prepare_target_speaker_data,
    prepare_voicedata_adaptation_data,
)
from turkish_tts.manifests import ClipRecord, CollectionFormat, RightsState, write_jsonl


def test_architecture_decision_excludes_noncommercial_weights() -> None:
    decision = architecture_decision()
    selected = decision["selected"]
    assert isinstance(selected, dict)
    assert selected["base_weights"] is None
    assert selected["sample_rate_hz"] == 24000
    assert F5_WEIGHT_LICENSE == "CC-BY-NC-4.0"
    assert any(
        alternative.get("weight_license") == F5_WEIGHT_LICENSE and "excluded" in str(alternative.get("decision"))
        for alternative in decision["compared"]  # type: ignore[union-attr]
    )


def test_prepare_candidate_baseline_filters_and_separates_parent_recordings(tmp_path: Path) -> None:
    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "baseline"
    combined_dir.mkdir()
    audio_paths = []
    for index in range(4):
        path = tmp_path / f"{index}.wav"
        sf.write(path, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
        audio_paths.append(path)

    prefix = "combined"
    for index, split in enumerate(("train", "validation", "test")):
        candidate = _record(
            clip_id=f"candidate-{split}",
            audio_path=audio_paths[index],
            parent_id=f"parent-{split}",
            transcript=f"Aday sesi {index + 1} için temiz bir örnektir.",
        )
        public = candidate.model_copy(
            update={
                "clip_id": f"public-{split}",
                "source_dataset": "mozilla-common-voice-scripted-speech",
                "audio_path": str(audio_paths[3]),
                "metadata": {"quality_filter_reasons": []},
            }
        )
        write_jsonl(combined_dir / f"{prefix}.{split}.jsonl", [candidate, public])

    suite_path = tmp_path / "evaluation.jsonl"
    _write_evaluation_suite(suite_path)
    result = prepare_candidate_baseline_data(
        combined_dir=combined_dir,
        combined_prefix=prefix,
        output_dir=output_dir,
        evaluation_suite=suite_path,
    )

    assert result.records == {"train": 1, "validation": 1, "test": 1}
    assert result.parent_recordings == {"train": 1, "validation": 1, "test": 1}
    for split in ("train", "validation", "test"):
        manifest = output_dir / f"{BASELINE_DATASET_VERSION}.{split}.jsonl"
        row = orjson.loads(manifest.read_bytes())
        assert row["speaker"] == "candidate-a"
        assert row["normalized_text"].startswith("aday sesi")


def test_prepare_target_speaker_data_filters_and_separates_parents(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    target_speaker = "vd-speaker-target"
    records = [
        _record(
            clip_id=f"target-{index}",
            audio_path=audio,
            parent_id=f"parent-{index}",
            transcript=f"Hedef konuşmacı için temiz örnek {index}.",  # noqa: RUF001
        ).model_copy(update={"speaker_id": target_speaker})
        for index in range(8)
    ]
    records.append(
        _record(
            clip_id="other-speaker",
            audio_path=audio,
            parent_id="other-parent",
            transcript="Başka konuşmacının verisi seçilmemelidir.",  # noqa: RUF001
        )
    )
    input_manifest = tmp_path / "voicedata.jsonl"
    write_jsonl(input_manifest, records)
    suite_path = tmp_path / "evaluation.jsonl"
    _write_evaluation_suite(suite_path)

    result = prepare_target_speaker_data(
        input_manifest=input_manifest,
        target_speaker_id=target_speaker,
        target_alias="candidate-b",
        dataset_version="candidate-b-crossflow-v1",
        output_dir=tmp_path / "target",
        evaluation_suite=suite_path,
    )

    scripted_records = [
        record.model_copy(update={"source_dataset": "voicedata-scripted-campaign"}) for record in records
    ]
    scripted_manifest = tmp_path / "scripted.jsonl"
    write_jsonl(scripted_manifest, scripted_records)
    scripted_result = prepare_target_speaker_data(
        input_manifest=scripted_manifest,
        target_speaker_id=target_speaker,
        target_alias="candidate-b",
        dataset_version="candidate-b-scripted-v1",
        output_dir=tmp_path / "scripted-target",
        evaluation_suite=suite_path,
        source_dataset="voicedata-scripted-campaign",
    )
    assert scripted_result.records == result.records
    assert scripted_result.sources == {"voicedata-scripted-campaign": 8}

    assert result.records == {"train": 6, "validation": 1, "test": 1}
    assert result.parent_recordings == {"train": 6, "validation": 1, "test": 1}
    assert result.sources == {"voicedata-turkish": 8}
    parents_by_split = {}
    for split, manifest in result.manifests.items():
        rows = [orjson.loads(line) for line in Path(manifest).read_bytes().splitlines()]
        parents_by_split[split] = {str(row["parent_clip_id"]) for row in rows}
        assert {row["speaker"] for row in rows} == {"candidate-b"}
        assert {row["speaker_id"] for row in rows} == {target_speaker}
    assert not (parents_by_split["train"] & parents_by_split["validation"])
    assert not (parents_by_split["train"] & parents_by_split["test"])
    assert not (parents_by_split["validation"] & parents_by_split["test"])


def test_prepare_voicedata_adaptation_uses_candidate_parent_splits(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    records = []
    for split in ("train", "validation", "test"):
        parent_id = f"parent-{split}"
        (candidate_dir / f"{BASELINE_DATASET_VERSION}.{split}.jsonl").write_bytes(
            orjson.dumps({"parent_clip_id": parent_id}, option=orjson.OPT_APPEND_NEWLINE)
        )
        records.append(
            _record(
                clip_id=f"accepted-{split}",
                audio_path=audio,
                parent_id=parent_id,
                transcript=f"{split} için kabul edilen benzersiz örnek.",
            )
        )
    records.append(
        records[0].model_copy(
            update={
                "clip_id": "rejected-quality",
                "metadata": {
                    **records[0].metadata,
                    "quality_filter_reasons": ["dual_asr_disagreement"],
                },
            }
        )
    )
    input_manifest = tmp_path / "voicedata.jsonl"
    write_jsonl(input_manifest, records)
    suite_path = tmp_path / "evaluation.jsonl"
    _write_evaluation_suite(suite_path)

    result = prepare_voicedata_adaptation_data(
        input_manifest=input_manifest,
        candidate_dataset_dir=candidate_dir,
        output_dir=tmp_path / "output",
        evaluation_suite=suite_path,
    )

    assert result.dataset_version == VOICEDATA_DATASET_VERSION
    assert result.records == {"train": 1, "validation": 1, "test": 1}
    assert result.parent_recordings == {"train": 1, "validation": 1, "test": 1}
    assert result.sources == {"voicedata-turkish": 3}


def test_evaluation_suite_rejects_missing_category(tmp_path: Path) -> None:
    suite_path = tmp_path / "evaluation.jsonl"
    suite_path.write_bytes(orjson.dumps({"id": "one", "category": "general", "text": "Başka bir cümle."}) + b"\n")
    with pytest.raises(ValueError, match="missing categories"):
        load_evaluation_suite(suite_path)


def test_candidate_parent_cannot_cross_splits(tmp_path: Path) -> None:
    combined_dir = tmp_path / "combined"
    combined_dir.mkdir()
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    for split in ("train", "validation", "test"):
        write_jsonl(
            combined_dir / f"combined.{split}.jsonl",
            [_record(clip_id=split, audio_path=audio, parent_id="same-parent", transcript=f"{split} örneği")],
        )
    suite_path = tmp_path / "evaluation.jsonl"
    _write_evaluation_suite(suite_path)

    with pytest.raises(ValueError, match="parent recordings overlap"):
        prepare_candidate_baseline_data(
            combined_dir=combined_dir,
            combined_prefix="combined",
            output_dir=tmp_path / "output",
            evaluation_suite=suite_path,
        )


def _record(*, clip_id: str, audio_path: Path, parent_id: str, transcript: str) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        source_dataset="voicedata-turkish",
        source_version="2026-07-30",
        source_split="candidate",
        source_row_id=clip_id,
        audio_path=str(audio_path),
        transcript=transcript,
        normalized_transcript=transcript,
        speaker_id="candidate-a",
        collection_format=CollectionFormat.MONOLOGUE,
        rights_state=RightsState.ALLOWED,
        license_id="VoiceData-Terms+Biometric-Consent",
        duration_seconds=0.1,
        sample_rate_hz=24000,
        metadata={"candidate_a": True, "parent_clip_id": parent_id, "quality_filter_reasons": []},
    )


def _write_evaluation_suite(path: Path) -> None:
    with path.open("wb") as handle:
        for index, category in enumerate(sorted(REQUIRED_EVALUATION_CATEGORIES)):
            handle.write(
                orjson.dumps(
                    {"id": f"eval-{index}", "category": category, "text": f"Benzersiz değerlendirme cümlesi {index}."},
                    option=orjson.OPT_APPEND_NEWLINE,
                )
            )

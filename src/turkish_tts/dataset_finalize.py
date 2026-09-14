from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from turkish_tts.manifests import ClipRecord, RightsState, write_jsonl
from turkish_tts.normalize import normalize_for_model

COMMON_VOICE_DATASET = "mozilla-common-voice-scripted-speech"
FLEURS_DATASET = "google-fleurs"
VOICEDATA_DATASET = "voicedata-turkish"
COMBINED_DATASET_VERSION = "turkish-flagship-candidate-a-v1"


@dataclass(frozen=True, slots=True)
class CombinedDatasetConfig:
    max_public_speaker_seconds: float = 30 * 60
    max_common_voice_seconds: float = 120 * 3600
    max_fleurs_seconds: float = 15 * 3600
    validation_parent_fraction: float = 0.05
    test_parent_fraction: float = 0.05


@dataclass(frozen=True, slots=True)
class CombinedDatasetResult:
    accepted: list[ClipRecord]
    train: list[ClipRecord]
    validation: list[ClipRecord]
    test: list[ClipRecord]
    report: dict[str, object]


def build_combined_candidate_dataset(
    common_voice_records: Sequence[ClipRecord],
    fleurs_records: Sequence[ClipRecord],
    voicedata_records: Sequence[ClipRecord],
    *,
    output_dir: Path,
    prefix: str = COMBINED_DATASET_VERSION,
    config: CombinedDatasetConfig | None = None,
) -> CombinedDatasetResult:
    config = config or CombinedDatasetConfig()
    source_inputs = {
        COMMON_VOICE_DATASET: list(common_voice_records),
        FLEURS_DATASET: list(fleurs_records),
        VOICEDATA_DATASET: list(voicedata_records),
    }
    sources = {
        source: [record for record in records if not _quality_reasons(record)]
        for source, records in source_inputs.items()
    }
    for expected_source, records in sources.items():
        _validate_source_records(records, expected_source=expected_source)
    common_voice_accepted = sources[COMMON_VOICE_DATASET]
    fleurs_accepted = sources[FLEURS_DATASET]
    voicedata_accepted = sources[VOICEDATA_DATASET]
    accepted = _normalize_records([*common_voice_accepted, *fleurs_accepted, *voicedata_accepted])
    _validate_unique_identity(accepted)

    public = _cap_public_records(
        [*common_voice_accepted, *fleurs_accepted],
        config=config,
    )
    target = [record for record in voicedata_accepted if record.metadata.get("candidate_a") is True]
    if not target:
        raise ValueError("combined Candidate A dataset has no accepted target-voice records")
    split_by_clip_id = {
        **_public_split_assignments(public),
        **_voicedata_parent_split_assignments(
            target,
            validation_fraction=config.validation_parent_fraction,
            test_fraction=config.test_parent_fraction,
        ),
    }
    candidate_records = _normalize_records([*public, *target])
    split_records: dict[str, list[ClipRecord]] = {"train": [], "validation": [], "test": []}
    for record in candidate_records:
        split = split_by_clip_id[record.clip_id]
        split_records[split].append(_with_training_split(record, split=split))
    for records in split_records.values():
        records.sort(key=lambda item: item.clip_id)
    _validate_split_contract(split_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    accepted.sort(key=lambda item: (item.source_dataset, item.clip_id))
    write_jsonl(output_dir / f"{prefix}.accepted.jsonl", accepted)
    for split, records in split_records.items():
        write_jsonl(output_dir / f"{prefix}.{split}.jsonl", records)

    result = CombinedDatasetResult(
        accepted=accepted,
        train=split_records["train"],
        validation=split_records["validation"],
        test=split_records["test"],
        report={},
    )
    report = _combined_report(result, config=config, all_voicedata_records=voicedata_accepted)
    report["excluded_input_records"] = {
        source: len(source_inputs[source]) - len(sources[source]) for source in sorted(source_inputs)
    }
    return CombinedDatasetResult(
        accepted=result.accepted,
        train=result.train,
        validation=result.validation,
        test=result.test,
        report=report,
    )


def _validate_source_records(records: Sequence[ClipRecord], *, expected_source: str) -> None:
    if not records:
        raise ValueError(f"combined dataset source {expected_source} is empty")
    for record in records:
        if record.source_dataset != expected_source:
            raise ValueError(f"record {record.clip_id} belongs to unexpected source {record.source_dataset}")
        if record.rights_state != RightsState.ALLOWED or not record.license_id:
            raise ValueError(f"record {record.clip_id} does not have explicit training rights")
        if _quality_reasons(record):
            raise ValueError(f"record {record.clip_id} is not accepted")
        if not record.transcript or not record.transcript.strip():
            raise ValueError(f"record {record.clip_id} has no training transcript")
        if not Path(record.audio_path).is_file():
            raise FileNotFoundError(record.audio_path)
        if expected_source == COMMON_VOICE_DATASET and not isinstance(record.metadata.get("common_voice_asr"), dict):
            raise ValueError(f"Common Voice record {record.clip_id} has no ASR-agreement evidence")
        if expected_source == VOICEDATA_DATASET:
            _validate_voicedata_evidence(record)


def _validate_voicedata_evidence(record: ClipRecord) -> None:
    privacy = record.metadata.get("privacy_scan")
    if not isinstance(privacy, dict) or privacy.get("passed") is not True:
        raise ValueError(f"VoiceData record {record.clip_id} has not passed the privacy gate")
    required = ("dual_asr", "forced_alignment", "speaker_consistency", "acoustic_fingerprint")
    missing = [key for key in required if not record.metadata.get(key)]
    if missing:
        raise ValueError(f"VoiceData record {record.clip_id} is missing evidence: {', '.join(missing)}")


def _normalize_records(records: Sequence[ClipRecord]) -> list[ClipRecord]:
    return [
        record.model_copy(update={"normalized_transcript": normalize_for_model(record.transcript or "")})
        for record in records
    ]


def _validate_unique_identity(records: Sequence[ClipRecord]) -> None:
    if len({record.clip_id for record in records}) != len(records):
        raise ValueError("combined accepted manifest contains duplicate clip IDs")
    if len({record.audio_path for record in records}) != len(records):
        raise ValueError("combined accepted manifest contains duplicate audio paths")


def _cap_public_records(
    records: Sequence[ClipRecord],
    *,
    config: CombinedDatasetConfig,
) -> list[ClipRecord]:
    by_speaker: dict[tuple[str, str], list[ClipRecord]] = defaultdict(list)
    speakerless: list[ClipRecord] = []
    for record in records:
        if record.speaker_id:
            by_speaker[(record.source_dataset, record.speaker_id)].append(record)
        else:
            speakerless.append(record)
    capped = list(speakerless)
    for values in by_speaker.values():
        capped.extend(_duration_cap(values, config.max_public_speaker_seconds))
    source_caps = {
        COMMON_VOICE_DATASET: config.max_common_voice_seconds,
        FLEURS_DATASET: config.max_fleurs_seconds,
    }
    selected: list[ClipRecord] = []
    for source, cap_seconds in source_caps.items():
        source_records = [record for record in capped if record.source_dataset == source]
        selected.extend(_duration_cap(source_records, cap_seconds))
    return selected


def _duration_cap(records: Sequence[ClipRecord], cap_seconds: float) -> list[ClipRecord]:
    selected: list[ClipRecord] = []
    duration = 0.0
    for record in sorted(records, key=_quality_order):
        record_seconds = record.duration_seconds or 0
        if duration + record_seconds > cap_seconds:
            continue
        selected.append(record)
        duration += record_seconds
    return selected


def _quality_order(record: ClipRecord) -> tuple[float, float, str]:
    asr = record.metadata.get("common_voice_asr")
    acoustics = record.metadata.get("common_voice_acoustics")
    character_error_rate = _metadata_number(asr, "character_error_rate", default=0)
    estimated_snr = _metadata_number(acoustics, "estimated_snr_db", default=0)
    identity = hashlib.sha256(f"{COMBINED_DATASET_VERSION}:{record.clip_id}".encode()).hexdigest()
    return character_error_rate, -estimated_snr, identity


def _public_split_assignments(records: Sequence[ClipRecord]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for record in records:
        split = {"dev": "validation", "valid": "validation"}.get(
            record.source_split or "",
            record.source_split or "",
        )
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"public record {record.clip_id} has unsupported split {record.source_split}")
        assignments[record.clip_id] = split
    return assignments


def _voicedata_parent_split_assignments(
    records: Sequence[ClipRecord],
    *,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    by_parent: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in records:
        by_parent[str(record.metadata.get("parent_clip_id") or record.clip_id)].append(record)
    parents = sorted(
        by_parent, key=lambda value: hashlib.sha256(f"{COMBINED_DATASET_VERSION}:{value}".encode()).hexdigest()
    )
    if len(parents) < 3:
        raise ValueError("Candidate A needs at least three parent recordings for held-out evaluation")
    validation_count = max(1, round(len(parents) * validation_fraction))
    test_count = max(1, round(len(parents) * test_fraction))
    if validation_count + test_count >= len(parents):
        raise ValueError("VoiceData split fractions leave no training parents")
    validation_parents = set(parents[:validation_count])
    test_parents = set(parents[validation_count : validation_count + test_count])
    return {
        record.clip_id: (
            "validation" if parent in validation_parents else "test" if parent in test_parents else "train"
        )
        for parent, values in by_parent.items()
        for record in values
    }


def _with_training_split(record: ClipRecord, *, split: str) -> ClipRecord:
    return record.model_copy(
        update={
            "source_split": split,
            "metadata": {
                **record.metadata,
                "original_source_split": record.source_split,
                "training_split": split,
                "training_corpus_version": COMBINED_DATASET_VERSION,
                "target_voice": "candidate_a" if record.metadata.get("candidate_a") is True else None,
            },
        }
    )


def _validate_split_contract(split_records: dict[str, list[ClipRecord]]) -> None:
    if any(not records for records in split_records.values()):
        raise ValueError("combined dataset has an empty training, validation, or test split")
    seen: set[str] = set()
    for records in split_records.values():
        ids = {record.clip_id for record in records}
        if seen & ids:
            raise ValueError("combined dataset split clip IDs overlap")
        seen.update(ids)
    cv_speakers = {
        split: {
            record.speaker_id
            for record in records
            if record.source_dataset == COMMON_VOICE_DATASET and record.speaker_id
        }
        for split, records in split_records.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if cv_speakers[left] & cv_speakers[right]:
            raise ValueError(f"Common Voice speakers overlap between {left} and {right}")
    parent_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in split_records.items():
        for record in records:
            if record.source_dataset == VOICEDATA_DATASET:
                parent_splits[str(record.metadata.get("parent_clip_id") or record.clip_id)].add(split)
    if any(len(splits) != 1 for splits in parent_splits.values()):
        raise ValueError("VoiceData parent recordings cross training splits")


def _combined_report(
    result: CombinedDatasetResult,
    *,
    config: CombinedDatasetConfig,
    all_voicedata_records: Sequence[ClipRecord],
) -> dict[str, object]:
    candidate = [record for record in all_voicedata_records if record.metadata.get("candidate_a") is True]
    all_training = [*result.train, *result.validation, *result.test]
    return {
        "dataset_version": COMBINED_DATASET_VERSION,
        "config": asdict(config),
        "accepted": _aggregate(result.accepted),
        "candidate_a_available": _aggregate(candidate),
        "training": _aggregate(all_training),
        "splits": {
            "train": _aggregate(result.train),
            "validation": _aggregate(result.validation),
            "test": _aggregate(result.test),
        },
        "split_policy": {
            "common_voice": "source speaker-disjoint split preserved",
            "fleurs": "official source split preserved; source exposes no stable speaker IDs",
            "voicedata": "target-speaker parent-recording-disjoint split",
        },
        "rights": dict(sorted(Counter(record.license_id or "missing" for record in result.accepted).items())),
    }


def _aggregate(records: Sequence[ClipRecord]) -> dict[str, object]:
    by_source = Counter(record.source_dataset for record in records)
    source_seconds: defaultdict[str, float] = defaultdict(float)
    for record in records:
        source_seconds[record.source_dataset] += record.duration_seconds or 0
    return {
        "records": len(records),
        "hours": round(sum(record.duration_seconds or 0 for record in records) / 3600, 3),
        "speakers": len({record.speaker_id for record in records if record.speaker_id}),
        "records_by_source": dict(sorted(by_source.items())),
        "hours_by_source": {key: round(value / 3600, 3) for key, value in sorted(source_seconds.items())},
    }


def _quality_reasons(record: ClipRecord) -> tuple[str, ...]:
    reasons = record.metadata.get("quality_filter_reasons")
    if not isinstance(reasons, list):
        return ()
    return tuple(reason for reason in reasons if isinstance(reason, str))


def _metadata_number(metadata: object, key: str, *, default: float) -> float:
    if not isinstance(metadata, dict):
        return default
    value = metadata.get(key)
    return float(value) if isinstance(value, (int, float)) else default

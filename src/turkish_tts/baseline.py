from __future__ import annotations

import hashlib
import importlib
import platform
import re
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import orjson
from pydantic import BaseModel, ConfigDict, Field

from turkish_tts.manifests import ClipRecord, CollectionFormat, RightsState, iter_jsonl
from turkish_tts.normalize import (
    normalize_for_asr_comparison,
    normalize_for_model,
    normalize_for_scoring,
    turkish_lower,
)

BASELINE_DATASET_VERSION = "candidate-a-fastpitch-v1"
BASELINE_RUN_VERSION = "fastpitch-candidate-a-v1"
FOUNDATION_DATASET_VERSION = "commercial-turkish-foundation-fastpitch-v3"
FOUNDATION_RUN_VERSION = "fastpitch-commercial-turkish-foundation-v1"
VOICEDATA_DATASET_VERSION = "voicedata-turkish-fastpitch-v1"
VOICEDATA_RUN_VERSION = "fastpitch-voicedata-adaptation-v1"
NEMO_VERSION = "2.7.3"
NEMO_LICENSE = "Apache-2.0"
BIGVGAN_MODEL_ID = "nvidia/bigvgan_v2_24khz_100band_256x"
BIGVGAN_REVISION = "c329ede9e9bbc100ddf5c91e2330a61921262370"
BIGVGAN_CODE_REVISION = "7d2b454564a6c7d014227f635b7423881f14bdac"
BIGVGAN_LICENSE = "MIT"
F5_MODEL_ID = "SWivid/F5-TTS"
F5_REVISION = "84e5a410d9cead4de2f847e7c9369a6440bdfaca"
F5_WEIGHT_LICENSE = "CC-BY-NC-4.0"
REQUIRED_EVALUATION_CATEGORIES = frozenset(
    {
        "general",
        "voice_agent",
        "names_places",
        "numeric",
        "questions_confirmations",
        "acknowledgement",
        "long_form",
        "foreign_abbreviations",
        "emotional_style",
        "adversarial_normalization",
    }
)


class EvaluationPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)


class PreparedBaselineData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    manifests: dict[str, str]
    records: dict[str, int]
    hours: dict[str, float]
    sha256: dict[str, str]
    parent_recordings: dict[str, int]
    sources: dict[str, int] = Field(default_factory=dict)


def architecture_decision() -> dict[str, object]:
    """Return the machine-readable Phase 3 architecture and license decision."""
    return {
        "version": BASELINE_RUN_VERSION,
        "selected": {
            "acoustic_model": "FastPitch clean initialization",
            "code": {"project": "NVIDIA NeMo", "version": NEMO_VERSION, "license": NEMO_LICENSE},
            "base_weights": None,
            "vocoder": {
                "model_id": BIGVGAN_MODEL_ID,
                "revision": BIGVGAN_REVISION,
                "license": BIGVGAN_LICENSE,
            },
            "sample_rate_hz": 24000,
            "reason": (
                "Fast non-autoregressive inference, learned monotonic alignment, "
                "and no inherited acoustic-model weights."
            ),
        },
        "compared": [
            {
                "family": "flow-matching",
                "implementation": F5_MODEL_ID,
                "revision": F5_REVISION,
                "code_license": "MIT",
                "weight_license": F5_WEIGHT_LICENSE,
                "decision": (
                    "official weights excluded from every commercial run; clean-init experiment deferred to Phase 4"
                ),
            },
            {
                "family": "non-autoregressive transformer",
                "implementation": "NVIDIA NeMo FastPitch",
                "code_license": NEMO_LICENSE,
                "weight_license": "not applicable: clean initialization",
                "decision": "selected for the Candidate A diagnostic baseline",
            },
        ],
    }


def load_evaluation_suite(path: Path) -> list[EvaluationPrompt]:
    prompts: list[EvaluationPrompt] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                prompts.append(EvaluationPrompt.model_validate(orjson.loads(line)))
            except Exception as error:
                raise ValueError(f"invalid evaluation prompt at {path}:{line_number}") from error
    ids = [prompt.id for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation prompt IDs must be unique")
    categories = {prompt.category for prompt in prompts}
    missing = REQUIRED_EVALUATION_CATEGORIES - categories
    if missing:
        raise ValueError(f"evaluation suite is missing categories: {sorted(missing)}")
    return prompts


def assert_evaluation_excluded(prompts: Sequence[EvaluationPrompt], training_manifests: Sequence[Path]) -> None:
    training_text = set()
    for manifest in training_manifests:
        with manifest.open("rb") as handle:
            for line in handle:
                row = orjson.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"invalid manifest row at {manifest}")
                text = (
                    row.get("normalized_text")
                    or row.get("normalized_transcript")
                    or row.get("transcript")
                    or row.get("text")
                    or ""
                )
                training_text.add(normalize_for_asr_comparison(str(text)))
    collisions = [prompt.id for prompt in prompts if normalize_for_asr_comparison(prompt.text) in training_text]
    if collisions:
        raise ValueError(f"evaluation prompts collide with training text: {collisions}")


def prepare_candidate_baseline_data(
    *,
    combined_dir: Path,
    combined_prefix: str,
    output_dir: Path,
    evaluation_suite: Path,
) -> PreparedBaselineData:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_evaluation_suite(evaluation_suite)
    manifests: dict[str, str] = {}
    counts: dict[str, int] = {}
    hours: dict[str, float] = {}
    hashes: dict[str, str] = {}
    parents_by_split: dict[str, set[str]] = defaultdict(set)
    combined_paths: list[Path] = []

    for split in ("train", "validation", "test"):
        combined_path = combined_dir / f"{combined_prefix}.{split}.jsonl"
        combined_paths.append(combined_path)
        selected = []
        selected_seconds = 0.0
        for record in iter_jsonl(combined_path):
            if record.source_dataset != "voicedata-turkish" or record.metadata.get("candidate_a") is not True:
                continue
            if (
                record.rights_state is not RightsState.ALLOWED
                or not record.transcript
                or not record.audio_path
                or not record.duration_seconds
            ):
                raise ValueError(f"Candidate A record {record.clip_id} lacks text, audio, duration, or training rights")
            if not Path(record.audio_path).is_file():
                raise FileNotFoundError(f"Candidate A audio is missing: {record.audio_path}")
            parent_id = record.metadata.get("parent_clip_id")
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError(f"Candidate A record {record.clip_id} lacks a parent recording ID")
            parents_by_split[split].add(parent_id)
            selected_seconds += record.duration_seconds
            selected.append(
                {
                    "audio_filepath": record.audio_path,
                    "duration": record.duration_seconds,
                    "text": record.transcript,
                    "normalized_text": turkish_lower(normalize_for_model(record.transcript)),
                    "speaker": "candidate-a",
                    "clip_id": record.clip_id,
                    "parent_clip_id": parent_id,
                }
            )
        if not selected:
            raise ValueError(f"combined {split} split has no Candidate A records")
        manifest_path = output_dir / f"{BASELINE_DATASET_VERSION}.{split}.jsonl"
        _write_json_lines(manifest_path, selected)
        manifests[split] = str(manifest_path)
        counts[split] = len(selected)
        hours[split] = round(selected_seconds / 3600, 3)
        hashes[split] = _sha256_file(manifest_path)

    split_names = tuple(parents_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = parents_by_split[left] & parents_by_split[right]
            if overlap:
                raise ValueError(f"Candidate A parent recordings overlap between {left} and {right}: {sorted(overlap)}")
    assert_evaluation_excluded(prompts, combined_paths)

    result = PreparedBaselineData(
        dataset_version=BASELINE_DATASET_VERSION,
        manifests=manifests,
        records=counts,
        hours=hours,
        sha256=hashes,
        parent_recordings={split: len(parents) for split, parents in parents_by_split.items()},
    )
    _write_json(output_dir / f"{BASELINE_DATASET_VERSION}.report.json", result.model_dump(mode="json"))
    _write_json(output_dir / f"{BASELINE_RUN_VERSION}.architecture.json", architecture_decision())
    return result


def prepare_target_speaker_data(
    *,
    input_manifest: Path,
    target_speaker_id: str,
    target_alias: str,
    dataset_version: str,
    output_dir: Path,
    evaluation_suite: Path,
    source_dataset: str = "voicedata-turkish",
) -> PreparedBaselineData:
    if not target_speaker_id:
        raise ValueError("target_speaker_id must not be empty")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", target_alias):
        raise ValueError("target_alias must contain lowercase letters, numbers, and single hyphens")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", dataset_version):
        raise ValueError("dataset_version must contain lowercase letters, numbers, and single hyphens")

    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_evaluation_suite(evaluation_suite)
    records_by_parent: defaultdict[str, list[ClipRecord]] = defaultdict(list)
    for record in iter_jsonl(input_manifest):
        if record.source_dataset != source_dataset or record.speaker_id != target_speaker_id:
            continue
        if record.metadata.get("quality_filter_reasons") != []:
            continue
        if (
            record.rights_state is not RightsState.ALLOWED
            or not record.transcript
            or not record.audio_path
            or not record.duration_seconds
        ):
            raise ValueError(f"target record {record.clip_id} lacks text, audio, duration, or training rights")
        if not Path(record.audio_path).is_file():
            raise FileNotFoundError(f"target audio is missing: {record.audio_path}")
        parent_id = record.metadata.get("parent_clip_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(f"target record {record.clip_id} lacks a parent recording ID")
        records_by_parent[parent_id].append(record)

    if len(records_by_parent) < 3:
        raise ValueError("target speaker needs at least three accepted parent recordings")
    ordered_parents = sorted(
        records_by_parent,
        key=lambda parent_id: hashlib.sha256(f"{target_speaker_id}:{parent_id}".encode()).digest(),
    )
    holdout_count = max(1, round(len(ordered_parents) * 0.1))
    if holdout_count * 2 >= len(ordered_parents):
        holdout_count = 1
    validation_parents = set(ordered_parents[:holdout_count])
    test_parents = set(ordered_parents[holdout_count : holdout_count * 2])

    selected_by_split: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    seconds_by_split: defaultdict[str, float] = defaultdict(float)
    parents_by_split: defaultdict[str, set[str]] = defaultdict(set)
    for parent_id, records in records_by_parent.items():
        if parent_id in validation_parents:
            split = "validation"
        elif parent_id in test_parents:
            split = "test"
        else:
            split = "train"
        parents_by_split[split].add(parent_id)
        for record in records:
            assert record.transcript is not None
            assert record.audio_path is not None
            assert record.duration_seconds is not None
            seconds_by_split[split] += record.duration_seconds
            selected_by_split[split].append(
                {
                    "audio_filepath": record.audio_path,
                    "duration": record.duration_seconds,
                    "text": record.transcript,
                    "normalized_text": turkish_lower(normalize_for_model(record.transcript)),
                    "speaker": target_alias,
                    "speaker_id": target_speaker_id,
                    "clip_id": record.clip_id,
                    "parent_clip_id": parent_id,
                }
            )

    manifests: dict[str, str] = {}
    counts: dict[str, int] = {}
    hours: dict[str, float] = {}
    hashes: dict[str, str] = {}
    manifest_paths: list[Path] = []
    for split in ("train", "validation", "test"):
        selected = sorted(selected_by_split[split], key=lambda row: str(row["clip_id"]))
        if not selected:
            raise ValueError(f"target speaker {split} split is empty")
        manifest_path = output_dir / f"{dataset_version}.{split}.jsonl"
        _write_json_lines(manifest_path, selected)
        manifest_paths.append(manifest_path)
        manifests[split] = str(manifest_path)
        counts[split] = len(selected)
        hours[split] = round(seconds_by_split[split] / 3600, 3)
        hashes[split] = _sha256_file(manifest_path)

    assert_evaluation_excluded(prompts, manifest_paths)
    result = PreparedBaselineData(
        dataset_version=dataset_version,
        manifests=manifests,
        records=counts,
        hours=hours,
        sha256=hashes,
        parent_recordings={split: len(parents_by_split[split]) for split in ("train", "validation", "test")},
        sources={source_dataset: sum(counts.values())},
    )
    _write_json(output_dir / f"{dataset_version}.report.json", result.model_dump(mode="json"))
    return result


def prepare_fastpitch_stage_data(
    *,
    combined_dir: Path,
    combined_prefix: str,
    output_dir: Path,
    evaluation_suite: Path,
    stage: Literal["foundation"],
) -> PreparedBaselineData:
    dataset_version = FOUNDATION_DATASET_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_evaluation_suite(evaluation_suite)
    manifests: dict[str, str] = {}
    counts: dict[str, int] = {}
    hours: dict[str, float] = {}
    hashes: dict[str, str] = {}
    sources: defaultdict[str, int] = defaultdict(int)
    parents_by_split: dict[str, set[str]] = defaultdict(set)
    selected_combined_paths: list[Path] = []

    for split in ("train", "validation", "test"):
        combined_path = combined_dir / f"{combined_prefix}.{split}.jsonl"
        selected_combined_paths.append(combined_path)
        selected: list[dict[str, object]] = []
        selected_seconds = 0.0
        for record in iter_jsonl(combined_path):
            is_voicedata = record.source_dataset == "voicedata-turkish"
            if is_voicedata:
                continue
            if (
                record.rights_state is not RightsState.ALLOWED
                or not record.transcript
                or not record.audio_path
                or not record.duration_seconds
            ):
                raise ValueError(f"stage record {record.clip_id} lacks text, audio, duration, or training rights")
            if not Path(record.audio_path).is_file():
                raise FileNotFoundError(f"stage audio is missing: {record.audio_path}")
            parent_id = record.metadata.get("parent_clip_id")
            if not isinstance(parent_id, str) or not parent_id:
                parent_id = record.clip_id
            parents_by_split[split].add(parent_id)
            sources[record.source_dataset] += 1
            selected_seconds += record.duration_seconds
            selected.append(
                {
                    "audio_filepath": record.audio_path,
                    "duration": record.duration_seconds,
                    "text": record.transcript,
                    "normalized_text": turkish_lower(normalize_for_model(record.transcript)),
                    "speaker": record.speaker_id or f"{record.source_dataset}-unknown",
                    "clip_id": record.clip_id,
                    "parent_clip_id": parent_id,
                    "source_dataset": record.source_dataset,
                }
            )
        if not selected:
            raise ValueError(f"combined {split} split has no {stage} records")
        manifest_path = output_dir / f"{dataset_version}.{split}.jsonl"
        _write_json_lines(manifest_path, selected)
        manifests[split] = str(manifest_path)
        counts[split] = len(selected)
        hours[split] = round(selected_seconds / 3600, 3)
        hashes[split] = _sha256_file(manifest_path)

    split_names = tuple(parents_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = parents_by_split[left] & parents_by_split[right]
            if overlap:
                raise ValueError(f"{stage} parent recordings overlap between {left} and {right}: {sorted(overlap)}")
    assert_evaluation_excluded(prompts, selected_combined_paths)
    result = PreparedBaselineData(
        dataset_version=dataset_version,
        manifests=manifests,
        records=counts,
        hours=hours,
        sha256=hashes,
        parent_recordings={split: len(parents) for split, parents in parents_by_split.items()},
        sources=dict(sorted(sources.items())),
    )
    _write_json(output_dir / f"{dataset_version}.report.json", result.model_dump(mode="json"))
    return result


def prepare_voicedata_adaptation_data(
    *,
    input_manifest: Path,
    candidate_dataset_dir: Path,
    output_dir: Path,
    evaluation_suite: Path,
    target_dataset_version: str = BASELINE_DATASET_VERSION,
    dataset_version: str = VOICEDATA_DATASET_VERSION,
) -> PreparedBaselineData:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_evaluation_suite(evaluation_suite)
    fixed_parent_splits: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = candidate_dataset_dir / f"{target_dataset_version}.{split}.jsonl"
        with path.open("rb") as handle:
            for line in handle:
                row = orjson.loads(line)
                parent_id = row.get("parent_clip_id")
                if isinstance(parent_id, str) and parent_id:
                    existing = fixed_parent_splits.setdefault(parent_id, split)
                    if existing != split:
                        raise ValueError(f"target parent {parent_id} crosses prepared splits")

    selected_by_split: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    seconds_by_split: defaultdict[str, float] = defaultdict(float)
    parents_by_split: defaultdict[str, set[str]] = defaultdict(set)
    accepted = 0
    for record in iter_jsonl(input_manifest):
        if record.source_dataset != "voicedata-turkish":
            continue
        reasons = record.metadata.get("quality_filter_reasons")
        if reasons != []:
            continue
        if (
            record.rights_state is not RightsState.ALLOWED
            or not record.transcript
            or not record.audio_path
            or not record.duration_seconds
        ):
            raise ValueError(f"accepted VoiceData record {record.clip_id} lacks training requirements")
        parent_id = record.metadata.get("parent_clip_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(f"accepted VoiceData record {record.clip_id} lacks a parent recording")
        split = fixed_parent_splits.get(parent_id) or _stable_parent_split(parent_id)
        parents_by_split[split].add(parent_id)
        seconds_by_split[split] += record.duration_seconds
        selected_by_split[split].append(
            {
                "audio_filepath": record.audio_path,
                "duration": record.duration_seconds,
                "text": record.transcript,
                "normalized_text": turkish_lower(normalize_for_model(record.transcript)),
                "speaker": record.speaker_id or f"{record.source_dataset}-unknown",
                "clip_id": record.clip_id,
                "parent_clip_id": parent_id,
                "source_dataset": record.source_dataset,
            }
        )
        accepted += 1
    if accepted == 0:
        raise ValueError("VoiceData manifest has no accepted records")

    manifests: dict[str, str] = {}
    counts: dict[str, int] = {}
    hours: dict[str, float] = {}
    hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        rows = selected_by_split[split]
        if not rows:
            raise ValueError(f"VoiceData adaptation {split} split is empty")
        rows.sort(key=lambda row: str(row["clip_id"]))
        manifest_path = output_dir / f"{dataset_version}.{split}.jsonl"
        _write_json_lines(manifest_path, rows)
        manifests[split] = str(manifest_path)
        counts[split] = len(rows)
        hours[split] = round(seconds_by_split[split] / 3600, 3)
        hashes[split] = _sha256_file(manifest_path)
    assert_evaluation_excluded(prompts, [Path(path) for path in manifests.values()])
    result = PreparedBaselineData(
        dataset_version=dataset_version,
        manifests=manifests,
        records=counts,
        hours=hours,
        sha256=hashes,
        parent_recordings={split: len(parents_by_split[split]) for split in ("train", "validation", "test")},
        sources={"voicedata-turkish": accepted},
    )
    _write_json(output_dir / f"{dataset_version}.report.json", result.model_dump(mode="json"))
    return result


def _stable_parent_split(parent_id: str) -> Literal["train", "validation", "test"]:
    bucket = int(hashlib.sha256(parent_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def prepare_fastpitch_features(
    *,
    dataset_dir: Path,
    config_path: Path,
    feature_dir: Path,
    dataset_version: str = BASELINE_DATASET_VERSION,
    num_workers: int = 1,
) -> Path:
    """Precompute deterministic pitch and energy features. Requires the training extra."""
    from hydra.utils import instantiate
    from joblib import Parallel, delayed
    from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
    from omegaconf import OmegaConf

    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    cfg = OmegaConf.load(config_path)
    feature_config = OmegaConf.to_container(cfg.featurizers, resolve=True)
    feature_config_sha256 = hashlib.sha256(orjson.dumps(feature_config, option=orjson.OPT_SORT_KEYS)).hexdigest()
    manifests = [dataset_dir / f"{dataset_version}.{split}.jsonl" for split in ("train", "validation", "test")]
    fingerprint = {
        "feature_config_sha256": feature_config_sha256,
        "manifest_sha256": {path.name: _sha256_file(path) for path in manifests},
    }
    report_path = feature_dir / "feature-report.json"
    if report_path.is_file():
        existing = orjson.loads(report_path.read_bytes())
        if isinstance(existing, dict) and existing.get("inputs") == fingerprint:
            return report_path
        shutil.rmtree(feature_dir)
    elif feature_dir.is_dir():
        marker_path = feature_dir / "feature-inputs.json"
        marker = orjson.loads(marker_path.read_bytes()) if marker_path.is_file() else None
        if marker != fingerprint:
            shutil.rmtree(feature_dir)

    marker_path = feature_dir / "feature-inputs.json"
    feature_dir.mkdir(parents=True, exist_ok=True)
    _write_json(marker_path, fingerprint)

    featurizers = instantiate(cfg.featurizers)
    entries = [entry for manifest in manifests for entry in read_manifest(manifest)]
    for feature_name, featurizer in featurizers.items():
        Parallel(n_jobs=num_workers, prefer="processes")(
            delayed(featurizer.save)(
                manifest_entry=entry,
                audio_dir=Path("/"),
                feature_dir=feature_dir,
                overwrite=False,
            )
            for entry in entries
        )
        if not feature_name:
            raise ValueError("feature names must be non-empty")
    _write_json(
        report_path,
        {
            "version": dataset_version,
            "records": len(entries),
            "features": sorted(featurizers),
            "inputs": fingerprint,
        },
    )
    return report_path


def train_fastpitch_candidate(
    *,
    dataset_dir: Path,
    config_path: Path,
    output_dir: Path,
    dataset_version: str = BASELINE_DATASET_VERSION,
    run_version: str = BASELINE_RUN_VERSION,
    initial_model_path: Path | None = None,
    feature_workers: int = 1,
    max_epochs: int | None = None,
    max_steps: int | None = None,
    devices: int | None = None,
    strategy: str | None = None,
) -> Path:
    """Train or adapt a FastPitch model with a traceable dataset and optional parent checkpoint."""
    import lightning.pytorch as pl
    import nemo
    import torch
    from nemo.collections.common.callbacks import LogEpochTimeCallback
    from nemo.collections.tts.models import FastPitchModel
    from nemo.utils.exp_manager import exp_manager
    from omegaconf import OmegaConf

    if str(nemo.__version__) != NEMO_VERSION:
        raise RuntimeError(f"expected nemo-toolkit {NEMO_VERSION}, found {nemo.__version__}")
    torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(config_path)
    train_manifest = dataset_dir / f"{dataset_version}.train.jsonl"
    validation_manifest = dataset_dir / f"{dataset_version}.validation.jsonl"
    feature_dir = dataset_dir / "features-v1"
    prepare_fastpitch_features(
        dataset_dir=dataset_dir,
        config_path=config_path,
        feature_dir=feature_dir,
        dataset_version=dataset_version,
        num_workers=feature_workers,
    )
    cfg.train_ds_meta = _nemo_dataset_meta(train_manifest, feature_dir)
    cfg.val_ds_meta = _nemo_dataset_meta(validation_manifest, feature_dir)
    cfg.log_ds_meta = _nemo_dataset_meta(validation_manifest, feature_dir)
    speaker_names = sorted(
        {
            str(row.get("speaker") or "unknown")
            for manifest in (train_manifest, validation_manifest)
            for row in (orjson.loads(line) for line in manifest.open("rb"))
        }
    )
    speaker_map_path = dataset_dir / f"{dataset_version}.speaker-map.json"
    if speaker_map_path.is_file():
        existing_map = orjson.loads(speaker_map_path.read_bytes())
        if not isinstance(existing_map, dict) or not existing_map:
            raise ValueError(f"invalid speaker map: {speaker_map_path}")
        unknown_speakers = [name for name in speaker_names if name not in existing_map]
        if unknown_speakers:
            raise ValueError(f"manifest speakers missing from speaker map: {unknown_speakers}")
        cfg.n_speakers = len(existing_map)
        cfg.model.train_ds.dataset.speaker_path = str(speaker_map_path)
        cfg.model.validation_ds.dataset.speaker_path = str(speaker_map_path)
    elif len(speaker_names) > 1:
        _write_json(speaker_map_path, {name: index for index, name in enumerate(speaker_names)})
        cfg.n_speakers = len(speaker_names)
        cfg.model.train_ds.dataset.speaker_path = str(speaker_map_path)
        cfg.model.validation_ds.dataset.speaker_path = str(speaker_map_path)
    cfg.log_dir = str(output_dir / "samples")
    cfg.exp_manager.exp_dir = str(output_dir)
    cfg.exp_manager.name = run_version
    if max_epochs is not None:
        cfg.trainer.max_epochs = max_epochs
    if max_steps is not None:
        cfg.trainer.max_steps = max_steps
    if devices is not None:
        cfg.trainer.devices = devices
    if strategy is not None:
        cfg.trainer.strategy = strategy
    pl.seed_everything(int(cfg.seed), workers=True)
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not isinstance(trainer_kwargs, dict):
        raise TypeError("trainer config must resolve to a mapping")
    trainer: Any = pl.Trainer(**cast(dict[str, Any], trainer_kwargs))
    exp_manager(trainer, cfg.exp_manager)
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)
    parent_model_sha256 = None
    parent_skipped_keys: list[str] = []
    if initial_model_path is not None:
        if not initial_model_path.is_file():
            raise FileNotFoundError(f"initial FastPitch model is missing: {initial_model_path}")
        parent_model = FastPitchModel.restore_from(str(initial_model_path), map_location="cpu")
        parent_state = parent_model.state_dict()
        model_state = model.state_dict()
        original_parent_keys = set(parent_state)
        shape_mismatched = [
            key for key, value in parent_state.items() if key in model_state and model_state[key].shape != value.shape
        ]
        for key in shape_mismatched:
            parent_state.pop(key)
        load_result = model.load_state_dict(parent_state, strict=False)
        # A missing key is acceptable only when the parent never had it (a parameter
        # introduced by the new config, e.g. a larger speaker table) or when it was
        # deliberately dropped for a shape mismatch; it then keeps its fresh init.
        unexpected_missing = [
            key for key in load_result.missing_keys if key in original_parent_keys and key not in shape_mismatched
        ]
        parent_skipped_keys = sorted(set(shape_mismatched) | set(load_result.missing_keys))
        if unexpected_missing or load_result.unexpected_keys:
            raise RuntimeError(
                f"parent model keys do not match: missing={unexpected_missing} unexpected={load_result.unexpected_keys}"
            )
        del parent_model
        parent_model_sha256 = _sha256_file(initial_model_path)
    trainer.callbacks.extend([pl.callbacks.LearningRateMonitor(), LogEpochTimeCallback()])
    started = time.monotonic()
    trainer.fit(model)
    elapsed = time.monotonic() - started

    checkpoint_callback = trainer.checkpoint_callback
    best_checkpoint_path = checkpoint_callback.best_model_path if checkpoint_callback is not None else ""
    export_model = (
        FastPitchModel.load_from_checkpoint(best_checkpoint_path, map_location="cpu") if best_checkpoint_path else model
    )
    model_path = output_dir / f"{run_version}.nemo"
    export_model.save_to(str(model_path))
    report = {
        "run_version": run_version,
        "dataset_version": dataset_version,
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "parent_model_path": str(initial_model_path) if initial_model_path is not None else None,
        "parent_model_sha256": parent_model_sha256,
        "parent_reinitialized_keys": parent_skipped_keys,
        "elapsed_seconds": round(elapsed, 3),
        "best_checkpoint_path": best_checkpoint_path or None,
        "best_validation_loss": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None
            else None
        ),
        "environment": {
            "python": platform.python_version(),
            "nemo": str(nemo.__version__),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
        "architecture": architecture_decision(),
    }
    _write_json(output_dir / f"{run_version}.report.json", report)
    return model_path


def synthesize_fastpitch_prompts(
    *,
    model_path: Path,
    prompts: Sequence[EvaluationPrompt],
    output_dir: Path,
    bigvgan_source: Path,
    vocoder_dir: Path,
    device: str = "cuda",
    speaker_index: int | None = None,
    warmup_text: str = "Merhaba, bugün size destek olabilirim.",
) -> Path:
    """Synthesize Turkish prompts and write a latency report. Requires the training extra."""
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from nemo.collections.tts.models import FastPitchModel

    if not prompts:
        raise ValueError("at least one synthesis prompt is required")
    if not model_path.is_file():
        raise FileNotFoundError(f"FastPitch model is missing: {model_path}")
    if not bigvgan_source.joinpath("bigvgan.py").is_file():
        raise FileNotFoundError(f"pinned BigVGAN source is missing: {bigvgan_source}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA synthesis requested but CUDA is unavailable")

    vocoder_dir.mkdir(parents=True, exist_ok=True)
    required_vocoder_files = (vocoder_dir / "config.json", vocoder_dir / "bigvgan_generator.pt")
    if not all(path.is_file() for path in required_vocoder_files):
        snapshot_download(
            repo_id=BIGVGAN_MODEL_ID,
            revision=BIGVGAN_REVISION,
            local_dir=vocoder_dir,
            allow_patterns=["config.json", "bigvgan_generator.pt", "LICENSE"],
        )

    source_path = str(bigvgan_source.resolve())
    sys.path.insert(0, source_path)
    try:
        bigvgan_module = importlib.import_module("bigvgan")
    finally:
        sys.path.remove(source_path)
    bigvgan_class: Any = bigvgan_module.BigVGAN

    torch.set_float32_matmul_precision("high")
    load_started = time.monotonic()
    acoustic_model = FastPitchModel.restore_from(str(model_path), map_location=device)
    acoustic_model = acoustic_model.eval().to(device)
    n_speakers = int(acoustic_model.cfg.get("n_speakers", 1))
    if n_speakers > 1 and speaker_index is None:
        raise ValueError(f"model has {n_speakers} speakers; pass --speaker-index")
    if speaker_index is not None and not 0 <= speaker_index < n_speakers:
        raise ValueError(f"speaker index {speaker_index} out of range for {n_speakers} speakers")
    speaker_tensor = torch.tensor([speaker_index], device=device) if speaker_index is not None else None
    vocoder = bigvgan_class.from_pretrained(str(vocoder_dir), use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    _assert_bigvgan_mel_contract(acoustic_model.cfg.preprocessor, vocoder.h)
    _synchronize_device(torch, device)
    model_load_seconds = time.monotonic() - load_started

    output_dir.mkdir(parents=True, exist_ok=True)

    def generate(text: str) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float, float]:
        normalized_text = turkish_lower(normalize_for_model(text))
        with torch.inference_mode():
            _synchronize_device(torch, device)
            acoustic_started = time.monotonic()
            tokens = acoustic_model.parse(normalized_text, normalize=False)
            mel = acoustic_model.generate_spectrogram(tokens=tokens, speaker=speaker_tensor)
            _synchronize_device(torch, device)
            acoustic_seconds = time.monotonic() - acoustic_started
            vocoder_started = time.monotonic()
            waveform = vocoder(mel.float()).squeeze().clamp(-1.0, 1.0)
            _synchronize_device(torch, device)
            vocoder_seconds = time.monotonic() - vocoder_started
        return waveform.detach().cpu().numpy().astype(np.float32, copy=False), acoustic_seconds, vocoder_seconds

    warmup_started = time.monotonic()
    generate(warmup_text)
    warmup_seconds = time.monotonic() - warmup_started

    sample_rate = int(vocoder.h.sampling_rate)
    samples: list[dict[str, object]] = []
    total_audio_seconds = 0.0
    total_generation_seconds = 0.0
    for index, prompt in enumerate(prompts):
        waveform, acoustic_seconds, vocoder_seconds = generate(prompt.text)
        audio_seconds = len(waveform) / sample_rate
        generation_seconds = acoustic_seconds + vocoder_seconds
        output_path = output_dir / f"{index + 1:03d}-{_safe_artifact_name(prompt.id)}.wav"
        sf.write(output_path, waveform, sample_rate, subtype="PCM_16")
        total_audio_seconds += audio_seconds
        total_generation_seconds += generation_seconds
        samples.append(
            {
                "id": prompt.id,
                "category": prompt.category,
                "text": prompt.text,
                "normalized_text": turkish_lower(normalize_for_model(prompt.text)),
                "audio_path": str(output_path),
                "audio_seconds": round(audio_seconds, 3),
                "acoustic_seconds": round(acoustic_seconds, 4),
                "vocoder_seconds": round(vocoder_seconds, 4),
                "latency_seconds": round(generation_seconds, 4),
                "real_time_factor": round(generation_seconds / audio_seconds, 4),
            }
        )

    report_path = output_dir / f"{BASELINE_RUN_VERSION}.synthesis.json"
    _write_json(
        report_path,
        {
            "run_version": BASELINE_RUN_VERSION,
            "model": {"path": str(model_path), "sha256": _sha256_file(model_path)},
            "vocoder": {
                "model_id": BIGVGAN_MODEL_ID,
                "model_revision": BIGVGAN_REVISION,
                "code_revision": BIGVGAN_CODE_REVISION,
                "weight_sha256": _sha256_file(vocoder_dir / "bigvgan_generator.pt"),
            },
            "device": device,
            "sample_rate_hz": sample_rate,
            "model_load_seconds": round(model_load_seconds, 3),
            "warmup_seconds": round(warmup_seconds, 3),
            "samples": samples,
            "totals": {
                "prompts": len(samples),
                "audio_seconds": round(total_audio_seconds, 3),
                "generation_seconds": round(total_generation_seconds, 3),
                "real_time_factor": round(total_generation_seconds / total_audio_seconds, 4),
            },
        },
    )
    return report_path


def _assert_bigvgan_mel_contract(preprocessor: Any, vocoder_config: Any) -> None:
    vocoder_fmax = vocoder_config.fmax
    if vocoder_fmax is None:
        vocoder_fmax = int(vocoder_config.sampling_rate) // 2
    comparisons = {
        "sample_rate": (int(preprocessor.sample_rate), int(vocoder_config.sampling_rate)),
        "mel_channels": (int(preprocessor.features), int(vocoder_config.num_mels)),
        "n_fft": (int(preprocessor.n_fft), int(vocoder_config.n_fft)),
        "window_size": (int(preprocessor.n_window_size), int(vocoder_config.win_size)),
        "hop_size": (int(preprocessor.n_window_stride), int(vocoder_config.hop_size)),
        "fmin": (int(preprocessor.lowfreq), int(vocoder_config.fmin)),
        "fmax": (int(preprocessor.highfreq), int(vocoder_fmax)),
        "mag_power": (float(preprocessor.mag_power), 1.0),
        "log_zero_guard_type": (str(preprocessor.log_zero_guard_type), "clamp"),
        "log_zero_guard_value": (float(preprocessor.log_zero_guard_value), 1e-5),
        "mel_norm": (str(preprocessor.mel_norm), "slaney"),
    }
    mismatches = {
        name: {"acoustic_model": acoustic, "vocoder": vocoder}
        for name, (acoustic, vocoder) in comparisons.items()
        if acoustic != vocoder
    }
    if mismatches:
        raise RuntimeError(f"FastPitch and BigVGAN mel contracts differ: {mismatches}")


def _synchronize_device(torch_module: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch_module.cuda.synchronize()


def _safe_artifact_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return normalized or "sample"


def evaluate_fastpitch_synthesis(
    *,
    synthesis_report: Path,
    reference_manifest: Path,
    output_path: Path,
    token: str | None,
    device: str = "cuda",
) -> Path:
    """Measure ASR intelligibility, speaker identity, and signal health for synthesized audio."""
    import gc

    import jiwer
    import numpy as np
    import soundfile as sf

    from turkish_tts.asr_labeling import WHISPER_MODEL_ID, WHISPER_MODEL_REVISION
    from turkish_tts.audio import analyze_signal
    from turkish_tts.common_voice_prepare import TransformersWhisperBatchTranscriber
    from turkish_tts.quality_gates import SpeakerEmbedder

    payload = orjson.loads(synthesis_report.read_bytes())
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"invalid synthesis report: {synthesis_report}")
    run_version = payload.get("run_version")
    if not isinstance(run_version, str) or not run_version.strip():
        raise ValueError(f"synthesis report has no run_version: {synthesis_report}")
    generated_records: list[ClipRecord] = []
    samples_by_id: dict[str, dict[str, object]] = {}
    for sample in payload["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("synthesis report samples must be objects")
        clip_id = str(sample.get("id", ""))
        audio_path = Path(str(sample.get("audio_path", "")))
        text = str(sample.get("text", ""))
        if not clip_id or not text or not audio_path.is_file():
            raise ValueError(f"invalid synthesized sample: {sample}")
        generated_records.append(
            ClipRecord(
                clip_id=clip_id,
                source_dataset="turkish-tts-generated",
                source_version=run_version,
                source_split="evaluation",
                audio_path=str(audio_path),
                transcript=text,
                normalized_transcript=turkish_lower(normalize_for_model(text)),
                speaker_id="candidate-a",
                collection_format=CollectionFormat.SCRIPTED,
                rights_state=RightsState.ALLOWED,
                license_id="generated-model-output",
            )
        )
        samples_by_id[clip_id] = sample

    transcriber = TransformersWhisperBatchTranscriber(
        model_name=WHISPER_MODEL_ID,
        revision=WHISPER_MODEL_REVISION,
        token=token,
        device=device,
        batch_size=min(32, len(generated_records)),
    )
    asr_results = transcriber.transcribe_batch(generated_records)
    del transcriber
    gc.collect()

    reference_records = _load_speaker_reference_manifest(reference_manifest)
    if not reference_records:
        raise ValueError("speaker reference manifest is empty")
    embedder = SpeakerEmbedder(token=token, device=device, batch_size=32)
    embeddings = embedder.embed([*reference_records, *generated_records])
    reference_vectors = [embeddings[record.clip_id] for record in reference_records]
    centroid = np.asarray(np.mean(np.stack(reference_vectors), axis=0), dtype=np.float32)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)

    sample_metrics: list[dict[str, object]] = []
    wers: list[float] = []
    cers: list[float] = []
    similarities: list[float] = []
    snrs: list[float] = []
    clipping: list[float] = []
    for record, asr_result in zip(generated_records, asr_results, strict=True):
        target = normalize_for_scoring(record.transcript or "")
        hypothesis = normalize_for_scoring(asr_result.text)
        audio, sample_rate = sf.read(record.audio_path, dtype="float32", always_2d=True)
        signal = analyze_signal(audio, sample_rate)
        speaker_similarity = float(np.dot(centroid, embeddings[record.clip_id]))
        synthesis_sample = samples_by_id[record.clip_id]
        wer = jiwer.wer(target, hypothesis)
        cer = cast(float, jiwer.cer(target, hypothesis))
        wers.append(wer)
        cers.append(cer)
        similarities.append(speaker_similarity)
        snrs.append(signal.estimated_snr_db)
        clipping.append(signal.clipping_ratio)
        sample_metrics.append(
            {
                "id": record.clip_id,
                "category": synthesis_sample.get("category"),
                "target": record.transcript,
                "asr_text": asr_result.text,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "speaker_similarity": round(speaker_similarity, 4),
                "signal": signal.as_metadata(),
            }
        )

    _write_json(
        output_path,
        {
            "run_version": run_version,
            "synthesis_report": str(synthesis_report),
            "reference_manifest": str(reference_manifest),
            "reference_records": len(reference_records),
            "models": {
                "asr": {"id": WHISPER_MODEL_ID, "revision": WHISPER_MODEL_REVISION},
                "speaker": {"purpose": "held-out target-speaker centroid similarity"},
            },
            "summary": {
                "samples": len(sample_metrics),
                "wer_mean": round(float(np.mean(wers)), 4),
                "wer_p90": round(_percentile(wers, 0.9), 4),
                "cer_mean": round(float(np.mean(cers)), 4),
                "cer_p90": round(_percentile(cers, 0.9), 4),
                "speaker_similarity_mean": round(float(np.mean(similarities)), 4),
                "speaker_similarity_p10": round(_percentile(similarities, 0.1), 4),
                "estimated_snr_db_mean": round(float(np.mean(snrs)), 3),
                "clipping_ratio_max": round(max(clipping), 8),
            },
            "samples": sample_metrics,
        },
    )
    return output_path


def _load_speaker_reference_manifest(path: Path) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = orjson.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid speaker reference row at {path}:{line_number}")
            if "audio_filepath" not in row:
                records.append(ClipRecord.model_validate(row))
                continue
            audio_path = str(row.get("audio_filepath", ""))
            transcript = str(row.get("text", ""))
            clip_id = str(row.get("clip_id") or f"candidate-reference-{line_number}")
            if not audio_path or not transcript:
                raise ValueError(f"incomplete speaker reference row at {path}:{line_number}")
            records.append(
                ClipRecord(
                    clip_id=clip_id,
                    source_dataset="voicedata-turkish",
                    source_version=BASELINE_DATASET_VERSION,
                    source_split="test",
                    audio_path=audio_path,
                    transcript=transcript,
                    normalized_transcript=str(row.get("normalized_text") or normalize_for_model(transcript)),
                    speaker_id=str(row.get("speaker") or "candidate-a"),
                    collection_format=CollectionFormat.MONOLOGUE,
                    rights_state=RightsState.ALLOWED,
                    license_id="voicedata-commercial-voice-consent-v1",
                    duration_seconds=float(row["duration"]) if row.get("duration") is not None else None,
                )
            )
    return records


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _nemo_dataset_meta(manifest_path: Path, feature_dir: Path) -> dict[str, dict[str, object]]:
    return {
        "candidate_a": {
            "manifest_path": str(manifest_path),
            "audio_dir": "/",
            "feature_dir": str(feature_dir),
            "sample_weight": 1.0,
        }
    }


def _write_json_lines(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

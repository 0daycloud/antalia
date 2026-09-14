from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pyarrow as pa  # type: ignore[import-untyped]
from datasets import Dataset  # type: ignore[import-untyped]
from numpy.typing import NDArray

PROSODY_FEATURES = (
    "log_seconds_per_character",
    "log_energy_mean",
    "log_energy_std",
    "log_f0_mean",
    "log_f0_std",
    "voiced_ratio",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a speaker-conditioned Common Voice foundation Arrow dataset.")
    parser.add_argument("--source-arrow", type=Path, required=True)
    parser.add_argument("--common-voice-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _speaker_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = orjson.loads(line)
            audio_path = str(row.get("audio_path", ""))
            speaker = str(row.get("speaker_id", ""))
            if not audio_path or not speaker:
                raise ValueError(f"missing audio_path or speaker_id at {path}:{line_number}")
            previous = mapping.setdefault(audio_path, speaker)
            if previous != speaker:
                raise ValueError(f"conflicting speaker IDs for {audio_path}")
    return mapping


def _feature_path(feature_root: Path, kind: str, audio_path: str) -> Path:
    relative = Path(audio_path.removeprefix("/")).with_suffix(".npy")
    return feature_root / kind / relative


def _prosody(feature_root: Path, audio_path: str, duration: float, text: str) -> NDArray[np.float64] | None:
    paths = {kind: _feature_path(feature_root, kind, audio_path) for kind in ("pitch", "energy", "voiced_mask")}
    if not all(path.is_file() for path in paths.values()):
        return None
    pitch = np.load(paths["pitch"])
    energy = np.load(paths["energy"])
    voiced = np.load(paths["voiced_mask"]).astype(bool)
    if pitch.ndim != 1 or energy.ndim != 1 or voiced.ndim != 1:
        raise ValueError(f"prosody arrays must be one-dimensional: {audio_path}")
    if not (len(pitch) == len(energy) == len(voiced)) or not len(pitch):
        raise ValueError(f"prosody arrays have incompatible lengths: {audio_path}")
    voiced_pitch = pitch[voiced & (pitch > 0.0)]
    if not len(voiced_pitch):
        return None
    log_energy = np.log1p(np.maximum(energy.astype(np.float64), 0.0))
    log_pitch = np.log(voiced_pitch.astype(np.float64))
    character_count = max(1, sum(not character.isspace() for character in text))
    values = np.array(
        [
            math.log(max(duration / character_count, 1e-6)),
            float(log_energy.mean()),
            float(log_energy.std()),
            float(log_pitch.mean()),
            float(log_pitch.std()),
            float(voiced.mean()),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite prosody features: {audio_path}")
    return values


def _split_validation(
    rows: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for speaker, speaker_rows in sorted(by_speaker.items()):
        ordered = sorted(
            speaker_rows,
            key=lambda row: hashlib.sha256(f"{seed}\0{speaker}\0{row['audio_path']}".encode()).digest(),
        )
        validation_count = 0
        if len(ordered) > 1:
            validation_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
    train.sort(key=lambda row: row["audio_path"])
    validation.sort(key=lambda row: row["audio_path"])
    return train, validation


def _write_arrow(path: Path, rows: list[dict[str, Any]]) -> None:
    table = Dataset.from_list(rows).data.table
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    speakers = _speaker_map(args.common_voice_manifest)
    raw = Dataset.from_file(str(args.source_arrow))
    prepared: list[dict[str, Any]] = []
    excluded_non_common_voice = 0
    excluded_missing_prosody = 0
    for row in raw:
        audio_path = str(row["audio_path"])
        speaker = speakers.get(audio_path)
        if speaker is None:
            excluded_non_common_voice += 1
            continue
        raw_text = row["text"]
        text = "".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
        duration = float(row["duration"])
        prosody = _prosody(args.feature_root, audio_path, duration, text)
        if prosody is None:
            excluded_missing_prosody += 1
            continue
        prepared.append(
            {
                "audio_path": audio_path,
                "text": list(text),
                "duration": duration,
                "speaker": speaker,
                "raw_prosody": prosody.tolist(),
            }
        )
    if not prepared:
        raise ValueError("source Arrow contains no Common Voice records with speaker IDs")
    train, validation = _split_validation(prepared, args.validation_fraction, args.seed)
    train_features = np.asarray([row.pop("raw_prosody") for row in train], dtype=np.float64)
    validation_features = np.asarray([row.pop("raw_prosody") for row in validation], dtype=np.float64)
    means = train_features.mean(axis=0)
    standard_deviations = train_features.std(axis=0)
    if np.any(standard_deviations <= 0.0):
        raise ValueError("prosody feature standard deviations must be positive")
    train_normalized = np.clip((train_features - means) / standard_deviations, -5.0, 5.0)
    validation_normalized = np.clip((validation_features - means) / standard_deviations, -5.0, 5.0)
    for row, values in zip(train, train_normalized, strict=True):
        row["prosody"] = values.astype(np.float32).tolist()
    for row, values in zip(validation, validation_normalized, strict=True):
        row["prosody"] = values.astype(np.float32).tolist()

    train_path = args.output_dir / "raw.arrow"
    validation_path = args.output_dir / "validation.arrow"
    _write_arrow(train_path, train)
    _write_arrow(validation_path, validation)
    speakers_in_train = {row["speaker"] for row in train}
    report = {
        "dataset_version": args.output_dir.name,
        "seed": args.seed,
        "source_arrow": str(args.source_arrow),
        "source_arrow_sha256": _sha256(args.source_arrow),
        "common_voice_manifest": str(args.common_voice_manifest),
        "common_voice_manifest_sha256": _sha256(args.common_voice_manifest),
        "feature_root": str(args.feature_root),
        "excluded_non_common_voice_records": excluded_non_common_voice,
        "train_records": len(train),
        "excluded_missing_prosody_records": excluded_missing_prosody,
        "validation_records": len(validation),
        "train_hours": sum(float(row["duration"]) for row in train) / 3600.0,
        "validation_hours": sum(float(row["duration"]) for row in validation) / 3600.0,
        "speakers": len(speakers_in_train),
        "validation_speakers_are_in_train": all(row["speaker"] in speakers_in_train for row in validation),
        "prosody": {
            "features": list(PROSODY_FEATURES),
            "mean": means.tolist(),
            "standard_deviation": standard_deviations.tolist(),
            "normalized_clip": 5.0,
        },
        "outputs": {
            "train_arrow": str(train_path),
            "train_arrow_sha256": _sha256(train_path),
            "validation_arrow": str(validation_path),
            "validation_arrow_sha256": _sha256(validation_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

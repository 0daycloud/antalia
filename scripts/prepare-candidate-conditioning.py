from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
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
    parser = argparse.ArgumentParser(
        description="Add one dedicated speaker ID and normalized prosody to Voice B Arrow data."
    )
    parser.add_argument("--train-arrow", type=Path, required=True)
    parser.add_argument("--validation-arrow", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--foundation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--audio-path-contains", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    source = Dataset.from_file(str(path))
    rows: list[dict[str, Any]] = []
    for row in source:
        raw_text = row["text"]
        text = "".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
        rows.append(
            {
                "audio_path": str(row["audio_path"]),
                "text": text,
                "duration": float(row["duration"]),
            }
        )
    if not rows:
        raise ValueError(f"Arrow dataset is empty: {path}")
    return rows


def _precompute_features(rows: list[dict[str, Any]], feature_config: Path, feature_dir: Path, workers: int) -> None:
    from hydra.utils import instantiate
    from joblib import Parallel, delayed
    from omegaconf import OmegaConf

    if workers < 1:
        raise ValueError("workers must be positive")
    cfg = OmegaConf.load(feature_config)
    featurizers = instantiate(cfg.featurizers)
    entries = [
        {
            "audio_filepath": row["audio_path"],
            "duration": row["duration"],
            "text": row["text"],
        }
        for row in rows
    ]
    feature_dir.mkdir(parents=True, exist_ok=True)
    for feature_name, featurizer in featurizers.items():
        if not feature_name:
            raise ValueError("feature names must be non-empty")
        Parallel(n_jobs=workers, prefer="processes")(
            delayed(featurizer.save)(
                manifest_entry=entry,
                audio_dir=Path("/"),
                feature_dir=feature_dir,
                overwrite=False,
            )
            for entry in entries
        )


def _feature_path(feature_dir: Path, kind: str, audio_path: str) -> Path:
    relative = Path(audio_path.removeprefix("/")).with_suffix(".npy")
    return feature_dir / kind / relative


def _prosody(feature_dir: Path, audio_path: str, duration: float, text: str) -> NDArray[np.float64]:
    pitch = np.load(_feature_path(feature_dir, "pitch", audio_path))
    energy = np.load(_feature_path(feature_dir, "energy", audio_path))
    voiced = np.load(_feature_path(feature_dir, "voiced_mask", audio_path)).astype(bool)
    if pitch.ndim != 1 or energy.ndim != 1 or voiced.ndim != 1:
        raise ValueError(f"prosody arrays must be one-dimensional: {audio_path}")
    if not (len(pitch) == len(energy) == len(voiced)) or not len(pitch):
        raise ValueError(f"prosody arrays have incompatible lengths: {audio_path}")
    voiced_pitch = pitch[voiced & (pitch > 0.0)]
    if not len(voiced_pitch):
        raise ValueError(f"Voice B clip has no voiced pitch frames: {audio_path}")
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


def _condition_rows(
    rows: list[dict[str, Any]],
    *,
    feature_dir: Path,
    speaker_id: str,
    mean: NDArray[np.float64],
    standard_deviation: NDArray[np.float64],
    normalized_clip: float,
) -> list[dict[str, Any]]:
    conditioned: list[dict[str, Any]] = []
    for row in rows:
        raw = _prosody(
            feature_dir,
            str(row["audio_path"]),
            float(row["duration"]),
            str(row["text"]),
        )
        normalized = np.clip((raw - mean) / standard_deviation, -normalized_clip, normalized_clip)
        conditioned.append(
            {
                "audio_path": row["audio_path"],
                "text": list(str(row["text"])),
                "duration": row["duration"],
                "speaker": speaker_id,
                "prosody": normalized.astype(np.float32).tolist(),
            }
        )
    return conditioned


def _write_arrow(path: Path, rows: list[dict[str, Any]]) -> None:
    table = Dataset.from_list(rows).data.table
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if not args.speaker_id or args.speaker_id == "<unconditioned>":
        raise ValueError("speaker-id must identify a conditioned speaker")
    raw_train = _read_rows(args.train_arrow)
    raw_validation = _read_rows(args.validation_arrow)
    train = [row for row in raw_train if args.audio_path_contains in str(row["audio_path"])]
    validation = [row for row in raw_validation if args.audio_path_contains in str(row["audio_path"])]
    if not train or not validation:
        raise ValueError("audio path filter removed every train or validation record")
    _precompute_features(train + validation, args.feature_config, args.feature_dir, args.workers)

    foundation_report = json.loads(args.foundation_report.read_text())
    prosody = foundation_report["prosody"]
    if tuple(prosody["features"]) != PROSODY_FEATURES:
        raise ValueError("foundation prosody feature contract does not match Voice B builder")
    mean = np.asarray(prosody["mean"], dtype=np.float64)
    standard_deviation = np.asarray(prosody["standard_deviation"], dtype=np.float64)
    normalized_clip = float(prosody["normalized_clip"])
    if mean.shape != (len(PROSODY_FEATURES),) or standard_deviation.shape != mean.shape:
        raise ValueError("foundation prosody statistics have invalid dimensions")
    if np.any(standard_deviation <= 0.0) or normalized_clip <= 0.0:
        raise ValueError("foundation prosody normalization is invalid")

    conditioned_train = _condition_rows(
        train,
        feature_dir=args.feature_dir,
        speaker_id=args.speaker_id,
        mean=mean,
        standard_deviation=standard_deviation,
        normalized_clip=normalized_clip,
    )
    conditioned_validation = _condition_rows(
        validation,
        feature_dir=args.feature_dir,
        speaker_id=args.speaker_id,
        mean=mean,
        standard_deviation=standard_deviation,
        normalized_clip=normalized_clip,
    )
    train_output = args.output_dir / "raw.arrow"
    validation_output = args.output_dir / "validation.arrow"
    _write_arrow(train_output, conditioned_train)
    _write_arrow(validation_output, conditioned_validation)
    report = {
        "dataset_version": args.output_dir.name,
        "speaker_id": args.speaker_id,
        "train_records": len(conditioned_train),
        "validation_records": len(conditioned_validation),
        "excluded_train_records": len(raw_train) - len(train),
        "excluded_validation_records": len(raw_validation) - len(validation),
        "train_hours": sum(float(row["duration"]) for row in conditioned_train) / 3600.0,
        "validation_hours": sum(float(row["duration"]) for row in conditioned_validation) / 3600.0,
        "prosody": {
            "features": list(PROSODY_FEATURES),
            "foundation_report": str(args.foundation_report),
            "foundation_report_sha256": _sha256(args.foundation_report),
        },
        "inputs": {
            "audio_path_contains": args.audio_path_contains,
            "train_arrow": str(args.train_arrow),
            "train_arrow_sha256": _sha256(args.train_arrow),
            "validation_arrow": str(args.validation_arrow),
            "validation_arrow_sha256": _sha256(args.validation_arrow),
            "feature_config": str(args.feature_config),
            "feature_config_sha256": _sha256(args.feature_config),
        },
        "outputs": {
            "train_arrow": str(train_output),
            "train_arrow_sha256": _sha256(train_output),
            "validation_arrow": str(validation_output),
            "validation_arrow_sha256": _sha256(validation_output),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

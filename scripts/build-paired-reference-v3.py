"""Build Voice B paired-reference arrows from consented VoiceData segments.

Pairs each accepted scripted segment with a same-parent reference (another
segment from the same recording, or the parent recording itself), so the
context-conditioned model learns to anchor timbre from real Voice B audio.
Splits come from the production v2 prepared manifests; foundation replay rows
come from the CFG foundation arrows (no reference).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
from datasets import Dataset  # type: ignore[import-untyped]

PROSODY_DIM = 6
SPEAKER_ID = "voicedata-candidate-b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prepare_prosody_map(arrow_path: Path) -> dict[str, list[float]]:
    dataset = Dataset.from_file(str(arrow_path))
    return {str(row["audio_path"]): [float(value) for value in row["prosody"]] for row in dataset}


def _segment_view(segment: dict[str, Any]) -> tuple[str, str, float]:
    """(audio_path, normalized_text, duration) for a segments-manifest row."""
    audio = str(segment.get("audio_path") or segment.get("audio_filepath") or "")
    text = str(segment.get("normalized_transcript") or segment.get("normalized_text") or segment.get("text") or "")
    duration = float(segment.get("duration_seconds") or segment.get("duration") or 0.0)
    return audio, text, duration


def _pick_reference(
    row: dict[str, Any],
    parent_audio: str,
    siblings: list[dict[str, Any]],
) -> tuple[str, str, float]:
    row_audio = str(row.get("audio_filepath", ""))
    candidates = []
    for sibling in siblings:
        audio, text, duration = _segment_view(sibling)
        if audio and audio != row_audio and 2.0 <= duration <= 12.0 and text.strip():
            candidates.append((audio, text, duration))
    if candidates:
        # Prefer mid-length references: enough speech to anchor timbre without
        # consuming the frame budget for the target.
        candidates.sort(key=lambda item: abs(item[2] - 8.0))
        chosen = candidates[0]
        return chosen
    if parent_audio and Path(parent_audio).is_file():
        normalized = str(row.get("normalized_text") or row.get("text", ""))
        return parent_audio, normalized, float(row.get("duration", 0.0))
    # No in-band same-parent reference: keep the row unconditioned rather than
    # dropping training data; the context encoder still sees plenty of pairs.
    return "", "", 0.0


def _build_split(
    rows: list[dict[str, Any]],
    prosody_map: dict[str, list[float]],
    segments_by_parent: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    built: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get("clip_id"))):
        audio_path = str(row.get("audio_filepath", ""))
        if not audio_path or not Path(audio_path).is_file():
            raise FileNotFoundError(f"training audio missing: {audio_path}")
        parent_id = str(row.get("parent_clip_id") or "")
        parent_audio = str(row.get("parent_audio_filepath", ""))
        reference_audio, reference_text, reference_duration = _pick_reference(
            row,
            parent_audio,
            segments_by_parent.get(parent_id, []),
        )
        prosody = prosody_map.get(audio_path)
        if prosody is None or len(prosody) != PROSODY_DIM:
            raise ValueError(f"missing prosody features for {audio_path}")
        built.append(
            {
                "audio_path": audio_path,
                "text": str(row.get("normalized_text") or row["text"]),
                "duration": float(row["duration"]),
                "speaker": SPEAKER_ID,
                "prosody": prosody,
                "reference_audio_path": reference_audio or None,
                "reference_text": reference_text or None,
                "reference_duration": reference_duration or None,
            }
        )
    return built


def _write_arrow(rows: list[dict[str, Any]], path: Path) -> None:
    dataset = Dataset.from_list(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_stream(sink, dataset.data.table.schema) as writer:
        writer.write_table(dataset.data.table)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True, help="candidate-b v2 prepared manifest dir")
    parser.add_argument("--segments-manifest", type=Path, required=True, help="candidate-b v2 segments manifest")
    parser.add_argument("--prosody-arrow", type=Path, required=True, help="arrow supplying speaker prosody features")
    parser.add_argument("--prosody-validation-arrow", type=Path, help="validation arrow supplying prosody features")
    parser.add_argument("--foundation-train-arrow", type=Path, required=True)
    parser.add_argument("--foundation-validation-arrow", type=Path, required=True)
    parser.add_argument("--foundation-replay-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    segments = _load_jsonl(args.segments_manifest)
    segments_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        parent_id = str(metadata.get("parent_clip_id") or segment.get("parent_clip_id") or segment.get("clip_id"))
        segments_by_parent[parent_id].append(segment)

    prosody_map = _prepare_prosody_map(args.prosody_arrow)
    if args.prosody_validation_arrow is not None:
        prosody_map.update(_prepare_prosody_map(args.prosody_validation_arrow))
    prepared_name = args.prepared_dir.name
    if prepared_name == "prepared":
        prepared_name = args.prepared_dir.parent.name
    train_rows = _build_split(
        _load_jsonl(args.prepared_dir / f"{prepared_name}.train.jsonl"),
        prosody_map,
        segments_by_parent,
    )
    validation_rows = _build_split(
        _load_jsonl(args.prepared_dir / f"{prepared_name}.validation.jsonl"),
        prosody_map,
        segments_by_parent,
    )

    random_generator = random.Random(args.seed)
    for arrow_path, target_rows, name in (
        (args.foundation_train_arrow, train_rows, "train"),
        (args.foundation_validation_arrow, validation_rows, "validation"),
    ):
        foundation = Dataset.from_file(str(arrow_path))
        indices = list(range(len(foundation)))
        random_generator.shuffle(indices)
        candidate_seconds = sum(row["duration"] for row in target_rows)
        target_seconds = candidate_seconds * (args.foundation_replay_fraction / (1.0 - args.foundation_replay_fraction))
        replay_rows: list[dict[str, Any]] = []
        replay_seconds = 0.0
        for index in indices:
            row = foundation[index]
            speaker = str(row["speaker"])
            if speaker == SPEAKER_ID:
                continue
            replay_rows.append(
                {
                    "audio_path": str(row["audio_path"]),
                    "text": str(row["text"]),
                    "duration": float(row["duration"]),
                    "speaker": speaker,
                    "prosody": [float(value) for value in row["prosody"]],
                    "reference_audio_path": None,
                    "reference_text": None,
                    "reference_duration": None,
                }
            )
            replay_seconds += float(row["duration"])
            if replay_seconds >= target_seconds:
                break
        merged = target_rows + replay_rows
        random_generator.shuffle(merged)
        output_path = args.output_dir / f"{name}.arrow"
        _write_arrow(merged, output_path)
        print(
            json.dumps(
                {
                    "split": name,
                    "output": str(output_path),
                    "output_sha256": _sha256(output_path),
                    "candidate_records": len(target_rows),
                    "foundation_records": len(replay_rows),
                    "total_records": len(merged),
                    "candidate_hours": round(candidate_seconds / 3600.0, 3),
                    "foundation_hours": round(replay_seconds / 3600.0, 3),
                    "reference_pairs": sum(row["reference_audio_path"] is not None for row in merged),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
from datasets import Dataset  # type: ignore[import-untyped]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build explicit different-utterance reference/target pairs with unconditioned replay."
    )
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--foundation-arrow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references-per-target", type=int, default=2)
    parser.add_argument("--foundation-replay-ratio", type=float, default=1.0)
    parser.add_argument("--reference-min-seconds", type=float, default=4.0)
    parser.add_argument("--reference-max-seconds", type=float, default=12.0)
    parser.add_argument("--max-total-seconds", type=float, default=24.0)
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _text_characters(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(character) for character in value]
    return list(str(value))


def _write_arrow(path: Path, rows: list[dict[str, Any]]) -> None:
    table = Dataset.from_list(rows).data.table
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.references_per_target < 1:
        raise ValueError("references-per-target must be positive")
    if args.foundation_replay_ratio < 0.0:
        raise ValueError("foundation-replay-ratio cannot be negative")
    if not 0.0 < args.reference_min_seconds <= args.reference_max_seconds:
        raise ValueError("reference duration bounds must satisfy 0 < min <= max")

    targets = _read_jsonl(args.target_manifest)
    if not targets:
        raise ValueError("target manifest is empty")
    target_paths = {str(row["audio_filepath"]) for row in targets}
    reference_bank = sorted(
        (row for row in targets if args.reference_min_seconds <= float(row["duration"]) <= args.reference_max_seconds),
        key=lambda row: str(row["clip_id"]),
    )
    if not reference_bank:
        raise ValueError("no records satisfy the reference duration bounds")

    paired_rows: list[dict[str, Any]] = []
    skipped_targets = 0
    for target in sorted(targets, key=lambda row: str(row["clip_id"])):
        target_text = str(target.get("normalized_text") or target["text"])
        eligible = [
            reference
            for reference in reference_bank
            if reference["parent_clip_id"] != target["parent_clip_id"]
            and reference["speaker_id"] == target["speaker_id"]
            and float(reference["duration"]) + float(target["duration"]) <= args.max_total_seconds
            and len(str(reference.get("normalized_text") or reference["text"])) + len(target_text) + 1
            <= args.max_text_tokens
        ]
        if not eligible:
            skipped_targets += 1
            continue
        target_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}:{target['clip_id']}".encode()).digest()[:8],
            "big",
        )
        selected = random.Random(target_seed).sample(eligible, k=min(args.references_per_target, len(eligible)))
        for reference in selected:
            reference_text = str(reference.get("normalized_text") or reference["text"])
            paired_rows.append(
                {
                    "audio_path": str(target["audio_filepath"]),
                    "text": list(target_text),
                    "duration": float(target["duration"]),
                    "reference_audio_path": str(reference["audio_filepath"]),
                    "reference_text": list(reference_text),
                    "reference_duration": float(reference["duration"]),
                }
            )

    if not paired_rows:
        raise ValueError("pairing produced no records")
    foundation = Dataset.from_file(str(args.foundation_arrow))
    foundation_rows = [
        {
            "audio_path": str(row["audio_path"]),
            "text": _text_characters(row["text"]),
            "duration": float(row["duration"]),
            "reference_audio_path": None,
            "reference_text": None,
            "reference_duration": None,
        }
        for row in foundation
        if str(row["audio_path"]) not in target_paths
        and 0.0 < float(row["duration"]) <= args.max_total_seconds
        and len(_text_characters(row["text"])) <= args.max_text_tokens
    ]
    replay_count = min(len(foundation_rows), round(len(paired_rows) * args.foundation_replay_ratio))
    replay_rows = random.Random(args.seed).sample(foundation_rows, k=replay_count)
    output_rows = paired_rows + replay_rows
    random.Random(args.seed).shuffle(output_rows)
    _write_arrow(args.output, output_rows)

    report = {
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "target_manifest": str(args.target_manifest),
        "target_manifest_sha256": _sha256(args.target_manifest),
        "foundation_arrow": str(args.foundation_arrow),
        "foundation_arrow_sha256": _sha256(args.foundation_arrow),
        "targets": len(targets),
        "skipped_targets": skipped_targets,
        "paired_records": len(paired_rows),
        "distinct_reference_clips": len({row["reference_audio_path"] for row in paired_rows}),
        "cross_parent_pairs": sum(
            target_path != reference_path
            for target_path, reference_path in ((row["audio_path"], row["reference_audio_path"]) for row in paired_rows)
        ),
        "foundation_replay_records": len(replay_rows),
        "records": len(output_rows),
        "paired_fraction": len(paired_rows) / len(output_rows),
        "target_hours": sum(float(row["duration"]) for row in paired_rows) / 3600.0,
        "reference_hours": sum(float(row["reference_duration"]) for row in paired_rows) / 3600.0,
        "foundation_replay_hours": sum(float(row["duration"]) for row in replay_rows) / 3600.0,
        "seed": args.seed,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

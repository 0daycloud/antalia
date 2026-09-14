from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
from datasets import Dataset  # type: ignore[import-untyped]

EXPECTED_COLUMNS = {"audio_path", "text", "duration", "speaker", "prosody"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge speaker-conditioned Arrow datasets with identical schemas.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for input_path in args.inputs:
        source = Dataset.from_file(str(input_path))
        if set(source.column_names) != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected columns in {input_path}: {source.column_names}")
        source_rows = [dict(row) for row in source]
        counts[str(input_path)] = len(source_rows)
        rows.extend(source_rows)
    if not rows:
        raise ValueError("merged dataset is empty")
    random.Random(args.seed).shuffle(rows)
    table = Dataset.from_list(rows).data.table
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    temporary.replace(args.output)
    report = {
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "records": len(rows),
        "hours": sum(float(row["duration"]) for row in rows) / 3600.0,
        "inputs": {path: {"records": count, "sha256": _sha256(Path(path))} for path, count in counts.items()},
        "seed": args.seed,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

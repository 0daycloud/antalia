"""Assemble the public evaluation/manifest dataset from internal manifests.

Public manifests keep what is needed to reproduce the filtering (source clip name, transcripts,
duration, checksum, acoustic QA, split) and drop local paths, speaker ids, demographics, and vote
counts so that the release does not add speaker-identification capability beyond the upstream
datasets. Audio is never copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PUBLIC_FIELDS = (
    "clip_id",
    "source_dataset",
    "source_version",
    "source_split",
    "transcript",
    "normalized_transcript",
    "language",
    "collection_format",
    "license_id",
    "duration_seconds",
    "sample_rate_hz",
    "sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True, help="directory with internal manifests")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    public = {field: record.get(field) for field in PUBLIC_FIELDS}
    public["source_file"] = Path(record["audio_path"]).name
    metadata = record.get("metadata") or {}
    for key in (
        "common_voice_acoustics",
        "acoustics",
        "asr_check",
        "quality_filter_reasons",
        "quality_stage",
        "attribution",
        "license_url",
        "dataset_terms_url",
    ):
        if key in metadata:
            public[key] = metadata[key]
    return public


def _strip_manifest(source: Path, destination: Path) -> int:
    count = 0
    with source.open() as reader, destination.open("w") as writer:
        for line in reader:
            if not line.strip():
                continue
            writer.write(json.dumps(_public_record(json.loads(line)), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = _parse_args()
    out = args.output_dir
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "evaluation").mkdir(exist_ok=True)
    (out / "listening").mkdir(exist_ok=True)
    (out / "selection").mkdir(exist_ok=True)
    counts: dict[str, int] = {}
    for source in sorted(args.manifests.glob("*.jsonl")):
        if source.name.endswith(".accepted.jsonl"):
            continue  # accepted = train + validation + test; do not ship it twice
        name = source.name.replace("fleurs-tr-70bb2e84b976b7e960aa89f1c648e09c59f894dd", "fleurs-tr")
        counts[name] = _strip_manifest(source, out / "manifests" / name)
    for source in sorted(args.manifests.glob("*.report.json")):
        name = source.name.replace("fleurs-tr-70bb2e84b976b7e960aa89f1c648e09c59f894dd", "fleurs-tr")
        shutil.copyfile(source, out / "manifests" / name)
    for suite in ("turkish-v2.jsonl", "turkish-v1.jsonl", "turkish-pronunciation-v1.jsonl"):
        shutil.copyfile(Path("configs/evaluation") / suite, out / "evaluation" / suite)
    cmos = Path("reports/cmos-session-v1")
    shutil.copyfile(cmos / "instructions.md", out / "listening" / "cmos-v1-instructions.md")
    shutil.copyfile(cmos / "trials.csv", out / "listening" / "cmos-v1-trials.csv")
    shutil.copyfile(cmos / "key.json", out / "listening" / "cmos-v1-key.json")
    shutil.copyfile(cmos / "results-sezgin.csv", out / "listening" / "cmos-v1-results-listener-1.csv")
    shutil.copyfile(cmos / "RESULTS.md", out / "listening" / "cmos-v1-results.md")
    shutil.copyfile(Path("reports/native-listening-cmos-protocol.md"), out / "listening" / "cmos-protocol.md")
    for name in ("prosody-presets.json", "timbre-profile.json", "envelope-stats.json", "inference-recipe.json"):
        shutil.copyfile(Path("release/hf/antalia-1") / name, out / "selection" / name)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

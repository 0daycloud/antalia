"""Assemble the public sample gallery from a finished best-of-8 evaluation run.

For every evaluation category the gallery pairs the best-of-8 selected take with the raw
single-seed take (candidate-00, seed 20260803) of the same prompt, and adds explicit failure
cases: the prompts where the raw take fails hardest and the prompts where even selection fails.
Writes ``gallery.json`` plus copied WAV files into ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="best-of-N run with candidate-*/ and selected/")
    parser.add_argument("--suite", type=Path, required=True, help="evaluation suite JSONL with id, category, text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--raw-failures", type=int, default=6)
    parser.add_argument("--selected-failures", type=int, default=3)
    return parser.parse_args()


def _load_quality(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {sample["id"]: sample for sample in payload["samples"]}


def _copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination.name


def main() -> None:
    args = _parse_args()
    suite_rows = (json.loads(line) for line in args.suite.read_text().splitlines() if line.strip())
    prompts = {row["id"]: row for row in suite_rows}
    candidate_dirs = sorted(path for path in args.run_dir.glob("candidate-*") if path.is_dir())
    candidate_quality = [_load_quality(path / "quality.json") for path in candidate_dirs]
    raw_quality = candidate_quality[0]
    selection = json.loads((args.run_dir / "selected" / "selection.json").read_text())
    selected_rows = {row["id"]: row for row in selection["selections"]}
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def entry(prompt_id: str, role: str) -> dict[str, Any]:
        prompt = prompts[prompt_id]
        raw = raw_quality[prompt_id]
        chosen = selected_rows[prompt_id]
        spread = [quality[prompt_id]["cer"] for quality in candidate_quality]
        raw_audio = Path(raw["audio_path"]) if "audio_path" in raw else None
        if raw_audio is None or not raw_audio.exists():
            matches = sorted((candidate_dirs[0] / "synthesis").glob(f"*-{prompt_id}.wav"))
            raw_audio = matches[0]
        return {
            "id": prompt_id,
            "role": role,
            "category": prompt["category"],
            "text": prompt["text"],
            "raw": {
                "audio": _copy(raw_audio, audio_dir / f"{prompt_id}-raw.wav"),
                "seed": 20260803,
                "cer": raw["cer"],
                "wer": raw.get("wer"),
                "speaker_similarity": raw.get("speaker_similarity"),
                "asr_text": raw.get("asr_text"),
            },
            "selected": {
                "audio": _copy(Path(chosen["audio_path"]), audio_dir / f"{prompt_id}-selected.wav"),
                "candidate": Path(chosen["chosen_run"]).name,
                "cer": chosen["cer"],
                "wer": chosen.get("wer"),
                "speaker_similarity": chosen.get("speaker_similarity"),
                "timbre_penalty": chosen.get("timbre_penalty"),
                "envelope_penalty": chosen.get("envelope_penalty"),
            },
            "candidate_cer_min": min(spread),
            "candidate_cer_max": max(spread),
        }

    showcase: list[dict[str, Any]] = []
    by_category: dict[str, list[str]] = {}
    for prompt_id, prompt in prompts.items():
        by_category.setdefault(prompt["category"], []).append(prompt_id)
    for category in sorted(by_category):
        ranked = sorted(
            by_category[category],
            key=lambda pid: (selected_rows[pid]["cer"], -selected_rows[pid]["speaker_similarity"]),
        )
        showcase.extend(entry(pid, "showcase") for pid in ranked[: args.per_category])
    used = {row["id"] for row in showcase}
    raw_failures = sorted((pid for pid in prompts if pid not in used), key=lambda pid: -raw_quality[pid]["cer"])
    failures = [entry(pid, "raw_failure") for pid in raw_failures[: args.raw_failures]]
    used.update(row["id"] for row in failures)
    selected_failures = sorted((pid for pid in prompts if pid not in used), key=lambda pid: -selected_rows[pid]["cer"])
    failures.extend(entry(pid, "selected_failure") for pid in selected_failures[: args.selected_failures])

    manifest = {
        "run": args.run_dir.name,
        "candidates": len(candidate_dirs),
        "summary_selected": selection["summary"],
        "summary_raw": json.loads((candidate_dirs[0] / "quality.json").read_text())["summary"],
        "showcase": showcase,
        "failures": failures,
    }
    (args.output_dir / "gallery.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"showcase": len(showcase), "failures": len(failures), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()

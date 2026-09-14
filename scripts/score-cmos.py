#!/usr/bin/env python3
"""Score CMOS listening-session results against the blinded key.

Applies the decision rule from reports/native-listening-cmos-protocol.md:
  1. Pooled CMOS >= 0.0 toward the candidate package (not worse),
  2. Same-person rate >= 0.95 on reference-anchored pairs,
  3. No category with pooled CMOS <= -1.0,
  plus catch-trial listener validity (discard a listener with >1 non-zero catch rating).

Usage:
  python scripts/score-cmos.py reports/cmos-session-v1/results-<name>.csv [more results...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SESSION_DIR = Path("reports/cmos-session-v1")


def load_results(path: Path) -> dict[int, tuple[int, str]]:
    rows: dict[int, tuple[int, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trial = int(row["trial"])
            cmos = int(row["cmos"])
            same = row["same_person"].strip().upper()
            if not -3 <= cmos <= 3:
                raise ValueError(f"{path.name}: trial {trial} cmos {cmos} out of range")
            if same not in {"E", "H"}:
                raise ValueError(f"{path.name}: trial {trial} same_person must be E or H")
            rows[trial] = (cmos, same)
    return rows


def mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="results-<name>.csv files")
    parser.add_argument("--key", type=Path, default=SESSION_DIR / "key.json")
    args = parser.parse_args()

    key = {entry["trial"]: entry for entry in json.loads(args.key.read_text(encoding="utf-8"))}
    with (args.key.parent / "trials.csv").open(encoding="utf-8") as handle:
        category_by_trial = {int(row["trial"]): row["category"] for row in csv.DictReader(handle)}

    valid_listeners: dict[str, dict[int, tuple[int, str]]] = {}
    for path in args.results:
        listener = path.stem.removeprefix("results-")
        rows = load_results(path)
        missing = sorted(set(key) - set(rows))
        if missing:
            print(f"[{listener}] WARNING: missing trials {missing}")
        catch_violations = sum(
            1 for trial, entry in key.items()
            if entry["type"] == "catch" and trial in rows and rows[trial][0] != 0
        )
        if catch_violations > 1:
            print(f"[{listener}] DISCARDED: {catch_violations} non-zero catch ratings (protocol allows at most 1)")
            continue
        print(f"[{listener}] catch trials: {catch_violations} non-zero rating(s) — listener valid")
        valid_listeners[listener] = rows

    if not valid_listeners:
        print("No valid listeners; nothing to score.")
        return 1

    # Orient every rating toward the CANDIDATE side of the pair:
    #   ab pairs: candidate = the gated best-of-N package (vs plain selection)
    #   anchored pairs: candidate = synthesized clip (vs real recording)
    # Rating convention: -3 = A clearly better ... +3 = B clearly better,
    # so score-toward-candidate = cmos if B is the candidate else -cmos.
    pooled: list[float] = []
    by_pool: dict[str, list[float]] = {"ab (gated vs plain)": [], "anchored (synth vs real)": []}
    by_category: dict[str, list[float]] = {}
    same_person_anchored: list[bool] = []

    for rows in valid_listeners.values():
        for trial, (cmos, same) in rows.items():
            entry = key[trial]
            kind = entry["type"]
            if kind == "catch":
                continue
            if kind == "ab":
                candidate = "gated"
                pool = "ab (gated vs plain)"
            else:
                candidate = "synth"
                pool = "anchored (synth vs real)"
                same_person_anchored.append(same == "E")
            oriented = cmos if entry["B"] == candidate else -cmos
            pooled.append(oriented)
            by_pool[pool].append(oriented)
            category = category_by_trial.get(trial, "general")
            by_category.setdefault(category, []).append(oriented)

    print()
    print(f"Listeners pooled: {', '.join(sorted(valid_listeners))}")
    mean, ci = mean_ci(pooled)
    print(f"Pooled CMOS toward candidate: {mean:+.3f} (95% CI ±{ci:.3f}, n={len(pooled)})")
    for pool, values in by_pool.items():
        pool_mean, pool_ci = mean_ci(values)
        print(f"  {pool}: {pool_mean:+.3f} (±{pool_ci:.3f}, n={len(values)})")

    anchored_n = len(same_person_anchored)
    same_rate = sum(same_person_anchored) / anchored_n if anchored_n else float("nan")
    print(f"Same-person rate on anchored pairs: {same_rate:.1%} ({sum(same_person_anchored)}/{anchored_n})")

    print("Per-category CMOS toward candidate:")
    worst_category, worst_mean = None, math.inf
    for category in sorted(by_category):
        cat_mean, _ = mean_ci(by_category[category])
        flag = "  <-- at/below -1.0" if cat_mean <= -1.0 else ""
        print(f"  {category:>16}: {cat_mean:+.3f} (n={len(by_category[category])}){flag}")
        if cat_mean < worst_mean:
            worst_category, worst_mean = category, cat_mean

    criterion_1 = mean >= 0.0
    criterion_2 = same_rate >= 0.95
    criterion_3 = worst_mean > -1.0
    print()
    print(f"[{'PASS' if criterion_1 else 'FAIL'}] 1. pooled CMOS >= 0.0 ({mean:+.3f})")
    print(f"[{'PASS' if criterion_2 else 'FAIL'}] 2. anchored same-person >= 95% ({same_rate:.1%})")
    print(f"[{'PASS' if criterion_3 else 'FAIL'}] 3. no category <= -1.0 (worst: {worst_category} {worst_mean:+.3f})")
    verdict = criterion_1 and criterion_2 and criterion_3
    print()
    print("VERDICT: PROMOTE" if verdict else "VERDICT: DO NOT PROMOTE — see failing criteria above")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())

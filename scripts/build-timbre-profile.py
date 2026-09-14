#!/usr/bin/env python3
"""Build a Candidate timbre profile (band-spectrum shape statistics) from real recordings.

The profile feeds `select-best-of-n.py --timbre-stats` so candidate takes are ranked by how
closely their long-term spectral shape matches the real voice — the dimension the speaker
verifier is saturated on (CMOS session v1: same-person 0/12 while verifier similarity read 0.91+).

Usage:
  python scripts/build-timbre-profile.py --audio real1.wav real2.wav ... --output profile.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_SELECTOR = Path(__file__).with_name("select-best-of-n.py")
_spec = importlib.util.spec_from_file_location("select_best_of_n", _SELECTOR)
assert _spec is not None and _spec.loader is not None
_selector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_selector)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, nargs="+", required=True, help="real recordings of the target voice")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profiles = []
    for path in args.audio:
        features = _selector._timbre_features(str(path))
        if features is None:
            print(f"skipped (no active frames): {path}", file=sys.stderr)
            continue
        profiles.append(features)
    if len(profiles) < 3:
        raise SystemExit("need at least 3 usable real recordings for a stable profile")

    count = len(profiles)
    bands = len(profiles[0])
    means = [sum(profile[band] for profile in profiles) / count for band in range(bands)]
    stds = [
        (sum((profile[band] - means[band]) ** 2 for profile in profiles) / (count - 1)) ** 0.5
        for band in range(bands)
    ]
    payload = {
        "band_edges_hz": list(_selector.TIMBRE_BAND_EDGES),
        "clips": count,
        "candidate_mean": [round(value, 4) for value in means],
        "candidate_std": [round(value, 4) for value in stds],
        "sources": [str(path) for path in args.audio],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "clips": count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

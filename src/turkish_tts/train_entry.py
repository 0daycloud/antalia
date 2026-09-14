"""Minimal torchrun entry point: ``python -m turkish_tts.train_entry <config.json> [resume.pt]``.

Replaces the box-local ``scripts/train_entry.py`` shims so automation depends only on the
deployed package. Rank/world-size setup is handled inside ``train_crossflow`` via the
standard torchrun environment variables.
"""

from __future__ import annotations

import sys
from pathlib import Path

from turkish_tts.crossflow_train import train_crossflow


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python -m turkish_tts.train_entry <config.json> [resume-checkpoint.pt]")
    config_path = Path(sys.argv[1])
    resume_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    if not config_path.is_file():
        raise SystemExit(f"config does not exist: {config_path}")
    if resume_path is not None and not resume_path.is_file():
        raise SystemExit(f"resume checkpoint does not exist: {resume_path}")
    final = train_crossflow(config_path, resume_path=resume_path)
    print(f"training complete: {final}")


if __name__ == "__main__":
    main()

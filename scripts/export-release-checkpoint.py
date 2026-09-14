from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CrossFlow training checkpoint to the public release format.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="training checkpoint (.pt)")
    parser.add_argument("--output-dir", type=Path, required=True, help="release directory to write")
    parser.add_argument("--raw-model", action="store_true", help="export raw weights instead of EMA weights")
    parser.add_argument("--vocoder-repo", default="nvidia/bigvgan_v2_24khz_100band_256x")
    parser.add_argument("--vocoder-revision")
    parser.add_argument("--extra", type=Path, help="JSON object merged into config.json (e.g. license, name)")
    return parser.parse_args()


def main() -> None:
    from turkish_tts.crossflow_release import export_crossflow_release

    args = _parse_args()
    extra = json.loads(args.extra.read_text()) if args.extra is not None else None
    config = export_crossflow_release(
        args.checkpoint,
        args.output_dir,
        use_ema=not args.raw_model,
        vocoder_repo=args.vocoder_repo,
        vocoder_revision=args.vocoder_revision,
        extra=extra,
    )
    summary = {key: config[key] for key in ("format", "source", "parameters", "text_normalization")}
    summary["speakers"] = len(config["speaker_vocabulary"] or [])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

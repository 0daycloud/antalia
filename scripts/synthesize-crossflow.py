from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize a CrossFlow checkpoint in the pinned F5/BigVGAN environment."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--evaluation-suite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--speaker")
    parser.add_argument("--prosody", nargs="*", type=float)
    parser.add_argument("--prosody-presets", type=Path)
    parser.add_argument("--preset")
    parser.add_argument("--preset-strength", type=float, default=1.0)
    parser.add_argument("--auto-style", action="store_true")
    parser.add_argument("--text-guidance", type=float, default=1.0)
    parser.add_argument("--speaker-guidance", type=float, default=1.0)
    parser.add_argument("--sway", type=float, default=0.0)
    parser.add_argument("--solver", choices=("euler", "midpoint"), default="euler")
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--mel-clamp", type=float)
    parser.add_argument(
        "--mel-correction", type=Path, help="JSON with a per-band mel_correction vector applied before the vocoder"
    )
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--context-guidance", type=float, default=1.0)
    parser.add_argument("--min-seconds-per-char", type=float, default=0.0)
    parser.add_argument("--chunk-chars", type=int, default=0)
    parser.add_argument("--chunk-pause-ms", type=float, default=160.0)
    parser.add_argument("--seed-candidates", type=int, default=1)
    parser.add_argument("--raw-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    from turkish_tts.crossflow_train import (
        blend_prosody_preset,
        load_prosody_presets,
        synthesize_crossflow,
        synthesize_crossflow_suite,
    )

    args = _parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least one")
    if not 0.5 <= args.duration_scale <= 2.0:
        raise ValueError("--duration-scale must be between 0.5 and 2.0")
    if not 0.0 <= args.adapter_scale <= 1.0:
        raise ValueError("--adapter-scale must be between zero and one")
    if args.text_guidance < 0.0 or args.speaker_guidance < 0.0:
        raise ValueError("guidance scales cannot be negative")
    if not -1.0 <= args.sway <= 1.0:
        raise ValueError("--sway must be between minus one and one")
    if not 0.0 <= args.guidance_rescale <= 1.0:
        raise ValueError("--guidance-rescale must be between zero and one")
    if args.mel_clamp is not None and args.mel_clamp <= 0.0:
        raise ValueError("--mel-clamp must be positive")
    if (args.reference_audio is None) != (args.reference_text is None):
        raise ValueError("--reference-audio and --reference-text must be provided together")
    if args.context_guidance < 0.0:
        raise ValueError("--context-guidance cannot be negative")
    if args.context_guidance != 1.0 and args.reference_audio is None:
        raise ValueError("--context-guidance requires --reference-audio")
    if args.min_seconds_per_char < 0.0:
        raise ValueError("--min-seconds-per-char cannot be negative")
    if args.chunk_chars < 0:
        raise ValueError("--chunk-chars cannot be negative")
    if not 0.0 <= args.chunk_pause_ms <= 2000.0:
        raise ValueError("--chunk-pause-ms must be between 0 and 2000")
    if args.seed_candidates < 1:
        raise ValueError("--seed-candidates must be at least one")
    if args.seed_candidates > 1 and args.text is not None:
        raise ValueError("--seed-candidates requires --evaluation-suite")
    if (args.preset is not None or args.auto_style) and args.prosody_presets is None:
        raise ValueError("--preset and --auto-style require --prosody-presets")
    if not 0.0 <= args.preset_strength <= 1.5:
        raise ValueError("--preset-strength must be between zero and one and a half")
    prosody = args.prosody
    mel_correction = None
    if args.mel_correction is not None:
        correction_payload = json.loads(args.mel_correction.read_text())
        mel_correction = correction_payload["mel_correction"]
        if not isinstance(mel_correction, list) or not mel_correction:
            raise ValueError("--mel-correction JSON must contain a non-empty mel_correction array")
    auto_style = None
    if args.prosody_presets is not None:
        presets_payload = load_prosody_presets(args.prosody_presets)
        if prosody is None:
            if args.preset is not None:
                prosody = blend_prosody_preset(presets_payload, args.preset, args.preset_strength)
            else:
                prosody = list(presets_payload["global_mean"])
        if args.auto_style:
            auto_style = {"payload": presets_payload, "strength": args.preset_strength}
    if args.text is not None:
        if args.output is None or args.output_dir is not None:
            raise ValueError("--text requires --output and does not accept --output-dir")
        report = synthesize_crossflow(
            checkpoint_path=args.checkpoint,
            text=args.text,
            output_path=args.output,
            vocoder_dir=args.vocoder,
            device=args.device,
            steps=args.steps,
            seed=args.seed,
            duration_scale=args.duration_scale,
            adapter_scale=args.adapter_scale,
            speaker=args.speaker,
            prosody=prosody,
            text_guidance_scale=args.text_guidance,
            speaker_guidance_scale=args.speaker_guidance,
            sway_coefficient=args.sway,
            solver=args.solver,
            guidance_rescale=args.guidance_rescale,
            mel_clamp=args.mel_clamp,
            reference_audio=args.reference_audio,
            reference_text=args.reference_text,
            context_guidance_scale=args.context_guidance,
            min_seconds_per_char=args.min_seconds_per_char,
            chunk_character_limit=args.chunk_chars,
            chunk_pause_seconds=args.chunk_pause_ms / 1000.0,
            auto_style=auto_style,
            mel_correction=mel_correction,
            use_ema=not args.raw_model,
        )
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.output_dir is None or args.output is not None:
        raise ValueError("--evaluation-suite requires --output-dir and does not accept --output")
    prompts = [json.loads(line) for line in args.evaluation_suite.read_text().splitlines() if line.strip()]
    report_path = synthesize_crossflow_suite(
        checkpoint_path=args.checkpoint,
        prompts=prompts,
        output_dir=args.output_dir,
        vocoder_dir=args.vocoder,
        device=args.device,
        steps=args.steps,
        seed=args.seed,
        duration_scale=args.duration_scale,
        adapter_scale=args.adapter_scale,
        speaker=args.speaker,
        prosody=prosody,
        text_guidance_scale=args.text_guidance,
        speaker_guidance_scale=args.speaker_guidance,
        sway_coefficient=args.sway,
        solver=args.solver,
        guidance_rescale=args.guidance_rescale,
        mel_clamp=args.mel_clamp,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        context_guidance_scale=args.context_guidance,
        min_seconds_per_char=args.min_seconds_per_char,
        chunk_character_limit=args.chunk_chars,
        chunk_pause_seconds=args.chunk_pause_ms / 1000.0,
        auto_style=auto_style,
        mel_correction=mel_correction,
        seed_candidates=args.seed_candidates,
        use_ema=not args.raw_model,
    )
    print(report_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a per-band mel correction vector: mean real log-mel minus mean synthetic log-mel.

The vector is applied additively to denormalized mels before the vocoder
(`synthesize-crossflow.py --mel-correction`), shifting the average spectral shape of
generated speech toward the target voice. Motivation: CMOS v1 failed on voice tone;
band analysis measured a +4 dB presence-band excess in synthesis that the champion
model reproduces consistently, so a static correction removes the average bias.

Uses the checkpoint's own LogMelFrontend so featurization matches the vocoder contract exactly.

Usage (on the training VM):
  .venv/bin/python scripts/build-mel-correction.py \
    --checkpoint /mnt/disks/tts-data/runs/crossflow-candidate-consistency-v1/model_6000.pt \
    --real-manifest .../candidate-b-scripted-complete-v1.train.jsonl \
    --synth-dir /mnt/disks/tts-data/runs/crossflow-longform-fixes/u6000-fc-seed*/synthesis \
    --output /mnt/disks/tts-data/manifests/candidate-b-mel-correction-v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ENERGY_QUANTILE = 0.3  # drop the quietest frames so silence does not dilute the profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True, help="jsonl manifest of real recordings")
    parser.add_argument("--synth-dir", type=Path, nargs="+", required=True, help="directories of synthesized wavs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips", type=int, default=1000, help="cap per group")
    return parser.parse_args()


def _manifest_audio_paths(manifest: Path) -> list[Path]:
    paths = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("audio_filepath") or row.get("audio_path") or row.get("audio")
        if value:
            paths.append(Path(value))
    if not paths:
        raise ValueError(f"no audio paths found in {manifest}")
    return paths


def main() -> None:
    args = _parse_args()
    import torch

    from turkish_tts.crossflow_train import CrossFlowTrainConfig, LogMelFrontend, load_crossflow_checkpoint

    _, _, payload = load_crossflow_checkpoint(args.checkpoint, args.device, use_ema=True)
    train_config = CrossFlowTrainConfig(**payload["train_config"])
    frontend = LogMelFrontend(train_config, args.device)

    import soundfile as sf

    def band_mean(paths: list[Path]) -> tuple[list[float], int, int]:
        total = None
        frames = 0
        used = 0
        for path in paths[: args.max_clips]:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            mono = torch.from_numpy(audio.mean(axis=1)).to(args.device)
            if rate != train_config.sample_rate:
                import torchaudio

                mono = torchaudio.functional.resample(mono, rate, train_config.sample_rate)
            lengths = torch.tensor([mono.shape[0]], device=args.device)
            mel, frame_lengths = frontend(mono.unsqueeze(0), lengths)
            mel = mel[0, : int(frame_lengths[0])]  # [frames, bands]
            frame_energy = mel.mean(dim=-1)
            threshold = torch.quantile(frame_energy, ENERGY_QUANTILE)
            active = mel[frame_energy >= threshold]
            if active.shape[0] == 0:
                continue
            band_sum = active.sum(dim=0)
            total = band_sum if total is None else total + band_sum
            frames += int(active.shape[0])
            used += 1
        if total is None or frames == 0:
            raise ValueError("no usable frames")
        return (total / frames).tolist(), used, frames

    real_paths = _manifest_audio_paths(args.real_manifest)
    synth_paths: list[Path] = []
    for directory in args.synth_dir:
        synth_paths.extend(sorted(directory.glob("*.wav")))
    if not synth_paths:
        raise ValueError("no synthesized wavs found")

    real_mean, real_clips, real_frames = band_mean(real_paths)
    synth_mean, synth_clips, synth_frames = band_mean(synth_paths)
    correction = [round(r - s, 6) for r, s in zip(real_mean, synth_mean, strict=True)]

    payload_out = {
        "correction_version": "candidate-b-mel-correction-v1",
        "checkpoint": str(args.checkpoint),
        "mel_correction": correction,
        "bands": len(correction),
        "real": {"manifest": str(args.real_manifest), "clips": real_clips, "frames": real_frames},
        "synth": {"dirs": [str(d) for d in args.synth_dir], "clips": synth_clips, "frames": synth_frames},
        "energy_quantile": ENERGY_QUANTILE,
        "note": "additive on denormalized log-mel before BigVGAN; positive = synth was too quiet in band",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n")
    largest = sorted(enumerate(correction), key=lambda item: abs(item[1]), reverse=True)[:5]
    print(json.dumps({
        "output": str(args.output),
        "real_clips": real_clips,
        "synth_clips": synth_clips,
        "correction_l2": round(sum(value * value for value in correction) ** 0.5, 4),
        "largest_bands": [{"band": index, "delta": round(value, 3)} for index, value in largest],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

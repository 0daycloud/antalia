from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a best-of-N listening package by picking, per prompt, the seed variant "
            "with the best verifier score across evaluated synthesis runs."
        )
    )
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cer-weight", type=float, default=1.0)
    parser.add_argument("--envelope-stats", type=Path)
    parser.add_argument("--envelope-weight", type=float, default=0.0)
    parser.add_argument("--timbre-stats", type=Path)
    parser.add_argument("--timbre-weight", type=float, default=0.0)
    return parser.parse_args()


@functools.lru_cache(maxsize=4096)
def _envelope_features(audio_path: str) -> tuple[float, float, float] | None:
    """Measure log-F0 mean, log-F0 spread, and voiced ratio of one waveform."""
    import numpy as np
    import pyworld
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    mono = np.mean(audio, axis=1)
    pitch, positions = pyworld.dio(mono, sample_rate, f0_floor=50.0, f0_ceil=500.0)
    pitch = pyworld.stonemask(mono, pitch, positions, sample_rate)
    voiced = pitch > 0.0
    if not voiced.any():
        return None
    log_pitch = np.log(pitch[voiced])
    return float(log_pitch.mean()), float(log_pitch.std()), float(voiced.mean())


def _envelope_penalty(audio_path: str, stats: dict[str, list[float]]) -> float:
    """Root-mean-square z-score of measured F0 features against the Candidate envelope."""
    features = _envelope_features(audio_path)
    if features is None:
        return 10.0
    squared = 0.0
    for index, value in zip((3, 4, 5), features, strict=True):
        normalized = (value - stats["foundation_mean"][index]) / stats["foundation_std"][index]
        z = (normalized - stats["candidate_mean"][index]) / stats["candidate_std"][index]
        squared += z * z
    return math.sqrt(squared / 3.0)


@functools.lru_cache(maxsize=4096)
def _timbre_features(audio_path: str) -> tuple[float, ...] | None:
    """Level-normalized band spectrum (dB per band) — the long-term timbre shape of one waveform."""
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    window = np.hanning(2048)
    frames = []
    for start in range(0, len(mono) - 2048, 1024):
        segment = mono[start : start + 2048]
        if float(np.sqrt((segment**2).mean())) < 0.01:
            continue
        frames.append(np.abs(np.fft.rfft(segment * window)) ** 2)
    if not frames:
        return None
    psd = np.mean(frames, axis=0)
    freqs = np.fft.rfftfreq(2048, 1.0 / sample_rate)
    bands = [
        10.0 * math.log10(float(psd[(freqs >= lo) & (freqs < hi)].mean()) + 1e-12)
        for lo, hi in itertools.pairwise(TIMBRE_BAND_EDGES)
    ]
    mean_level = sum(bands) / len(bands)
    return tuple(band - mean_level for band in bands)


TIMBRE_BAND_EDGES = (0, 200, 400, 700, 1000, 1500, 2200, 3200, 4700, 6800, 10000, 12000)


def _timbre_penalty(audio_path: str, stats: dict[str, list[float]]) -> float:
    """RMS z-score of the band-spectrum shape against the Candidate's real-voice timbre profile."""
    features = _timbre_features(audio_path)
    if features is None:
        return 10.0
    squared = 0.0
    for value, mean, std in zip(features, stats["candidate_mean"], stats["candidate_std"], strict=True):
        z = (value - mean) / max(std, 0.5)
        squared += z * z
    return math.sqrt(squared / len(features))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of no values")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_candidate(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    quality = json.loads((root / "quality.json").read_text())
    synthesis = json.loads((root / "synthesis" / "crossflow.synthesis.json").read_text())
    audio_paths = {sample["id"]: sample["audio_path"] for sample in synthesis["samples"]}
    metrics = {sample["id"]: sample for sample in quality["samples"]}
    if set(audio_paths) != set(metrics):
        raise ValueError(f"synthesis and quality reports disagree in {root}")
    return metrics, audio_paths


def main() -> None:
    args = _parse_args()
    if len(args.candidates) < 2:
        raise ValueError("best-of-N selection needs at least two candidate runs")
    if (args.envelope_stats is None) != (args.envelope_weight == 0.0):
        raise ValueError("--envelope-stats and a nonzero --envelope-weight must be used together")
    if (args.timbre_stats is None) != (args.timbre_weight == 0.0):
        raise ValueError("--timbre-stats and a nonzero --timbre-weight must be used together")
    envelope_stats = json.loads(args.envelope_stats.read_text()) if args.envelope_stats is not None else None
    timbre_stats = json.loads(args.timbre_stats.read_text()) if args.timbre_stats is not None else None
    candidates = [_load_candidate(root) for root in args.candidates]
    prompt_ids = list(candidates[0][0])
    for metrics, _ in candidates[1:]:
        if list(metrics) != prompt_ids:
            raise ValueError("candidate runs cover different prompt sets")

    args.output.mkdir(parents=True, exist_ok=True)
    selections: list[dict[str, Any]] = []
    chosen_metrics: list[dict[str, Any]] = []
    envelope_penalties: list[float] = []
    timbre_penalties: list[float] = []
    for index, prompt_id in enumerate(prompt_ids, start=1):
        best_candidate = None
        best_score = float("-inf")
        best_penalty = 0.0
        best_timbre = 0.0
        for candidate_index, (metrics, audio_paths) in enumerate(candidates):
            sample = metrics[prompt_id]
            score = float(sample["speaker_similarity"]) - args.cer_weight * float(sample["cer"])
            penalty = 0.0
            timbre = 0.0
            if envelope_stats is not None:
                penalty = _envelope_penalty(audio_paths[prompt_id], envelope_stats)
                score -= args.envelope_weight * penalty
            if timbre_stats is not None:
                timbre = _timbre_penalty(audio_paths[prompt_id], timbre_stats)
                score -= args.timbre_weight * timbre
            if score > best_score:
                best_score = score
                best_penalty = penalty
                best_timbre = timbre
                best_candidate = (candidate_index, sample, audio_paths[prompt_id])
        assert best_candidate is not None
        candidate_index, sample, audio_path = best_candidate
        destination = args.output / f"{index:03d}-{prompt_id}.wav"
        shutil.copy2(audio_path, destination)
        chosen_metrics.append(sample)
        envelope_penalties.append(best_penalty)
        timbre_penalties.append(best_timbre)
        selections.append(
            {
                "id": prompt_id,
                "chosen_run": str(args.candidates[candidate_index]),
                "score": round(best_score, 4),
                "cer": sample["cer"],
                "wer": sample["wer"],
                "speaker_similarity": sample["speaker_similarity"],
                "envelope_penalty": round(best_penalty, 4),
                "timbre_penalty": round(best_timbre, 4),
                "audio_path": str(destination),
            }
        )

    cers = [float(sample["cer"]) for sample in chosen_metrics]
    wers = [float(sample["wer"]) for sample in chosen_metrics]
    similarities = [float(sample["speaker_similarity"]) for sample in chosen_metrics]
    snrs = [float(sample["signal"]["estimated_snr_db"]) for sample in chosen_metrics]
    report = {
        "candidates": [str(root) for root in args.candidates],
        "cer_weight": args.cer_weight,
        "envelope_weight": args.envelope_weight,
        "timbre_weight": args.timbre_weight,
        "summary": {
            "samples": len(chosen_metrics),
            "cer_mean": round(sum(cers) / len(cers), 4),
            "cer_p90": round(_percentile(cers, 0.9), 4),
            "wer_mean": round(sum(wers) / len(wers), 4),
            "wer_p90": round(_percentile(wers, 0.9), 4),
            "speaker_similarity_mean": round(sum(similarities) / len(similarities), 4),
            "speaker_similarity_p10": round(_percentile(similarities, 0.1), 4),
            "estimated_snr_db_mean": round(sum(snrs) / len(snrs), 3),
            "envelope_penalty_mean": (
                round(sum(envelope_penalties) / len(envelope_penalties), 4) if envelope_stats is not None else None
            ),
            "timbre_penalty_mean": (
                round(sum(timbre_penalties) / len(timbre_penalties), 4) if timbre_stats is not None else None
            ),
        },
        "selections": selections,
    }
    (args.output / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

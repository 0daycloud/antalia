from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from turkish_tts.manifests import ClipRecord


@dataclass(frozen=True, slots=True)
class AudioProbe:
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    codec: str
    format_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    peak_amplitude: float
    rms_dbfs: float
    clipping_ratio: float
    active_frame_ratio: float
    estimated_snr_db: float

    def as_metadata(self) -> dict[str, object]:
        return {
            "duration_seconds": round(self.duration_seconds, 6),
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "peak_amplitude": round(self.peak_amplitude, 8),
            "rms_dbfs": round(self.rms_dbfs, 3),
            "clipping_ratio": round(self.clipping_ratio, 8),
            "active_frame_ratio": round(self.active_frame_ratio, 6),
            "estimated_snr_db": round(self.estimated_snr_db, 3),
        }


def analyze_signal(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    *,
    active_floor_dbfs: float = -45.0,
) -> SignalMetrics:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.ndim != 2 or audio.size == 0 or sample_rate <= 0:
        raise ValueError("audio must be a non-empty frame-by-channel array")
    mono = np.asarray(np.mean(audio, axis=1, dtype=np.float32), dtype=np.float32)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    frame_rms = _frame_rms(mono, sample_rate)
    noise_rms = float(np.percentile(frame_rms, 10)) if frame_rms.size else 0.0
    signal_rms = float(np.percentile(frame_rms, 90)) if frame_rms.size else 0.0
    active_threshold = max(10 ** (active_floor_dbfs / 20), noise_rms * 2.0)
    return SignalMetrics(
        duration_seconds=len(mono) / sample_rate,
        sample_rate_hz=sample_rate,
        channels=int(audio.shape[1]),
        peak_amplitude=peak,
        rms_dbfs=20 * math.log10(max(rms, 1e-12)),
        clipping_ratio=float(np.mean(np.abs(mono) >= 0.995)),
        active_frame_ratio=float(np.mean(frame_rms >= active_threshold)) if frame_rms.size else 0.0,
        estimated_snr_db=20 * math.log10((signal_rms + 1e-8) / (noise_rms + 1e-8)),
    )


def signal_quality_reasons(
    metrics: SignalMetrics,
    *,
    min_duration_seconds: float,
    max_duration_seconds: float,
    min_rms_dbfs: float,
    max_clipping_ratio: float,
    min_active_frame_ratio: float,
    min_estimated_snr_db: float,
    require_mono: bool = True,
) -> list[str]:
    reasons: list[str] = []
    if metrics.duration_seconds < min_duration_seconds:
        reasons.append("duration_too_short")
    if metrics.duration_seconds > max_duration_seconds:
        reasons.append("duration_too_long")
    if require_mono and metrics.channels != 1:
        reasons.append("not_mono")
    if metrics.rms_dbfs < min_rms_dbfs:
        reasons.append("level_too_low")
    if metrics.clipping_ratio > max_clipping_ratio:
        reasons.append("excessive_clipping")
    if metrics.active_frame_ratio < min_active_frame_ratio:
        reasons.append("insufficient_active_audio")
    if metrics.estimated_snr_db < min_estimated_snr_db:
        reasons.append("low_estimated_snr")
    return reasons


def _frame_rms(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    frame_size = max(1, sample_rate // 50)
    frame_count = len(audio) // frame_size
    if frame_count == 0:
        return np.array([], dtype=np.float64)
    framed = audio[: frame_count * frame_size].reshape(frame_count, frame_size)
    return np.sqrt(np.mean(np.square(framed), axis=1, dtype=np.float64))


def probe_audio(path: Path) -> AudioProbe:
    if not path.is_file():
        raise FileNotFoundError(path)
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:stream=codec_name,sample_rate,channels",
            "-select_streams",
            "a:0",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected one primary audio stream in {path}, got {len(streams)}")
    stream = streams[0]
    format_info = payload.get("format") or {}
    duration = float(format_info["duration"])
    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    if duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise ValueError(f"invalid audio metadata for {path}")
    return AudioProbe(
        duration_seconds=duration,
        sample_rate_hz=sample_rate,
        channels=channels,
        codec=str(stream.get("codec_name") or "unknown"),
        format_name=str(format_info.get("format_name") or "unknown"),
        sha256=sha256_file(path),
    )


def enrich_record(record: ClipRecord) -> ClipRecord:
    probe = probe_audio(Path(record.audio_path))
    metadata = {
        **record.metadata,
        "channels": probe.channels,
        "codec": probe.codec,
        "format_name": probe.format_name,
    }
    return record.model_copy(
        update={
            "duration_seconds": probe.duration_seconds,
            "sample_rate_hz": probe.sample_rate_hz,
            "sha256": probe.sha256,
            "metadata": metadata,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

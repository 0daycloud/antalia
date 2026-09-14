from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import pyworld
import soundfile as sf
from nemo.collections.tts.parts.preprocessing.features import PitchFeaturizer
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_audio_filepaths, normalize_volume
from numpy.typing import NDArray
from scipy.signal import resample_poly


class WorldPitchFeaturizer(PitchFeaturizer):  # type: ignore[misc]
    """NeMo-compatible F0 extraction using WORLD DIO with StoneMask refinement."""

    def compute_pitch(
        self,
        manifest_entry: dict[str, Any],
        audio_dir: Path,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
        audio_path, _ = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
        mono = np.mean(audio, axis=1)
        if sample_rate != self.sample_rate:
            divisor = gcd(sample_rate, self.sample_rate)
            mono = resample_poly(mono, self.sample_rate // divisor, sample_rate // divisor)
        if self.volume_norm:
            mono = normalize_volume(mono)
        frame_period_ms = self.hop_length / self.sample_rate * 1000.0
        pitch, temporal_positions = pyworld.dio(
            mono,
            self.sample_rate,
            f0_floor=self.pitch_fmin,
            f0_ceil=self.pitch_fmax,
            frame_period=frame_period_ms,
        )
        pitch = pyworld.stonemask(mono, pitch, temporal_positions, self.sample_rate)
        pitch = np.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        voiced_mask = pitch > 0.0
        voiced_prob = voiced_mask.astype(np.float32)
        return pitch, voiced_mask, voiced_prob

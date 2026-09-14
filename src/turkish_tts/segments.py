from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from turkish_tts.audio import analyze_signal, sha256_file, signal_quality_reasons
from turkish_tts.manifests import ClipRecord, CollectionFormat

SEGMENTATION_VERSION = "voicedata-silero-vad-v1"
VAD_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 500
    max_speech_duration_seconds: float = 15.0
    min_silence_duration_ms: int = 350
    speech_pad_ms: int = 120
    min_segment_duration_seconds: float = 0.75
    max_segment_duration_seconds: float = 15.25
    min_rms_dbfs: float = -50.0
    max_clipping_ratio: float = 0.02
    min_active_frame_ratio: float = 0.20
    min_estimated_snr_db: float = 3.0
    output_sample_rate: int = OUTPUT_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    segments: list[ClipRecord]
    report: dict[str, object]


def segment_voicedata_records(
    records: Sequence[ClipRecord],
    *,
    output_dir: Path,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    import torch
    import torchaudio
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if config is None:
        config = SegmentationConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    vad_options = VadOptions(
        threshold=config.vad_threshold,
        min_speech_duration_ms=config.min_speech_duration_ms,
        max_speech_duration_s=config.max_speech_duration_seconds,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
    )
    resamplers: dict[int, Any] = {}
    segments: list[ClipRecord] = []
    parent_stats: dict[str, dict[str, object]] = {}
    for parent in sorted(records, key=lambda item: item.clip_id):
        source_audio, source_sample_rate = sf.read(parent.audio_path, dtype="float32", always_2d=True)
        if source_audio.size == 0 or source_sample_rate <= 0:
            raise ValueError(f"VoiceData source {parent.clip_id} could not be decoded")
        source_mono = np.asarray(np.mean(source_audio, axis=1, dtype=np.float32), dtype=np.float32)
        vad_audio = decode_audio(parent.audio_path, sampling_rate=VAD_SAMPLE_RATE)
        timestamps = get_speech_timestamps(vad_audio, vad_options)
        parent_speech_seconds = 0.0
        for segment_index, timestamp in enumerate(timestamps):
            vad_start = int(timestamp["start"])
            vad_end = int(timestamp["end"])
            source_start = max(0, round(vad_start * source_sample_rate / VAD_SAMPLE_RATE))
            source_end = min(len(source_mono), round(vad_end * source_sample_rate / VAD_SAMPLE_RATE))
            if source_end <= source_start:
                continue
            segment_audio = torch.from_numpy(source_mono[source_start:source_end].copy())
            if source_sample_rate != config.output_sample_rate:
                resampler = resamplers.get(source_sample_rate)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(source_sample_rate, config.output_sample_rate)
                    resamplers[source_sample_rate] = resampler
                segment_audio = resampler(segment_audio)
            segment_array = segment_audio.numpy().astype(np.float32, copy=False)
            metrics = analyze_signal(segment_array[:, np.newaxis], config.output_sample_rate)
            reasons = signal_quality_reasons(
                metrics,
                min_duration_seconds=config.min_segment_duration_seconds,
                max_duration_seconds=config.max_segment_duration_seconds,
                min_rms_dbfs=config.min_rms_dbfs,
                max_clipping_ratio=config.max_clipping_ratio,
                min_active_frame_ratio=config.min_active_frame_ratio,
                min_estimated_snr_db=config.min_estimated_snr_db,
            )
            identity = f"{SEGMENTATION_VERSION}:{parent.clip_id}:{vad_start}:{vad_end}:{config.output_sample_rate}"
            segment_digest = hashlib.sha256(identity.encode()).hexdigest()
            clip_id = f"{parent.clip_id}-seg-{segment_digest[:16]}"
            audio_path = output_dir / f"{clip_id}.wav"
            temporary_path = audio_path.with_suffix(".wav.part")
            sf.write(temporary_path, segment_array, config.output_sample_rate, format="WAV", subtype="PCM_16")
            temporary_path.replace(audio_path)
            segment_sha256 = sha256_file(audio_path)
            parent_speech_seconds += metrics.duration_seconds
            metadata = {
                **parent.metadata,
                "parent_clip_id": parent.clip_id,
                "parent_sha256": parent.sha256,
                "parent_duration_seconds": parent.duration_seconds,
                "segment_index": segment_index,
                "segment_start_seconds": round(source_start / source_sample_rate, 6),
                "segment_end_seconds": round(source_end / source_sample_rate, 6),
                "segmentation_version": SEGMENTATION_VERSION,
                "segmentation_config": asdict(config),
                "audio_transform": {
                    "source_sample_rate_hz": source_sample_rate,
                    "output_sample_rate_hz": config.output_sample_rate,
                    "channel_mix": "mean_to_mono",
                    "encoding": "pcm_s16le",
                },
                "signal_metrics": metrics.as_metadata(),
                "quality_filter_reasons": sorted(set(reasons)),
                "quality_stage": "segmentation",
            }
            segments.append(
                parent.model_copy(
                    update={
                        "clip_id": clip_id,
                        "source_row_id": f"{parent.source_row_id or parent.clip_id}:{vad_start}-{vad_end}",
                        "audio_path": str(audio_path.resolve()),
                        "duration_seconds": metrics.duration_seconds,
                        "sample_rate_hz": config.output_sample_rate,
                        "sha256": segment_sha256,
                        "transcript": None,
                        "normalized_transcript": None,
                        "transcription_model": None,
                        "transcription_language": None,
                        "metadata": metadata,
                    }
                )
            )
        parent_stats[parent.clip_id] = {
            "segments": len(timestamps),
            "speech_seconds": round(parent_speech_seconds, 6),
            "source_seconds": parent.duration_seconds,
        }

    segments = _mark_exact_duplicates(segments)
    reason_counts = Counter(reason for record in segments for reason in _quality_reasons(record))
    accepted = [record for record in segments if not _quality_reasons(record)]
    report: dict[str, object] = {
        "segmentation_version": SEGMENTATION_VERSION,
        "config": asdict(config),
        "source_records": len(records),
        "source_hours": round(sum(record.duration_seconds or 0 for record in records) / 3600, 3),
        "segments": len(segments),
        "candidate_segments": len(accepted),
        "rejected_segments": len(segments) - len(accepted),
        "segment_hours": round(sum(record.duration_seconds or 0 for record in segments) / 3600, 3),
        "candidate_hours": round(sum(record.duration_seconds or 0 for record in accepted) / 3600, 3),
        "by_format": _aggregate_by_format(segments),
        "candidate_a": _aggregate_candidate(segments),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "parents": parent_stats,
    }
    return SegmentationResult(segments=segments, report=report)


def _mark_exact_duplicates(records: Sequence[ClipRecord]) -> list[ClipRecord]:
    first_by_sha: dict[str, str] = {}
    output: list[ClipRecord] = []
    for record in sorted(records, key=lambda item: item.clip_id):
        reasons = list(_quality_reasons(record))
        if record.sha256:
            first_clip_id = first_by_sha.setdefault(record.sha256, record.clip_id)
            if first_clip_id != record.clip_id:
                reasons.append("duplicate_audio")
        output.append(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "quality_filter_reasons": sorted(set(reasons)),
                    }
                }
            )
        )
    return output


def _aggregate_by_format(records: Sequence[ClipRecord]) -> dict[str, object]:
    return {
        collection_format.value: {
            "segments": len(selected),
            "hours": round(sum(record.duration_seconds or 0 for record in selected) / 3600, 3),
        }
        for collection_format in (CollectionFormat.MONOLOGUE, CollectionFormat.CONVERSATION)
        if (selected := [record for record in records if record.collection_format == collection_format])
    }


def _aggregate_candidate(records: Sequence[ClipRecord]) -> dict[str, object]:
    selected = [record for record in records if record.metadata.get("candidate_a") is True]
    return {
        "segments": len(selected),
        "hours": round(sum(record.duration_seconds or 0 for record in selected) / 3600, 3),
        "conversation_hours": round(
            sum(
                record.duration_seconds or 0
                for record in selected
                if record.collection_format == CollectionFormat.CONVERSATION
            )
            / 3600,
            3,
        ),
        "monologue_hours": round(
            sum(
                record.duration_seconds or 0
                for record in selected
                if record.collection_format == CollectionFormat.MONOLOGUE
            )
            / 3600,
            3,
        ),
    }


def _quality_reasons(record: ClipRecord) -> tuple[str, ...]:
    reasons = record.metadata.get("quality_filter_reasons")
    if not isinstance(reasons, list):
        return ()
    return tuple(reason for reason in reasons if isinstance(reason, str))

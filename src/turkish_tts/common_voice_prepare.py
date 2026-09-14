from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import jiwer
import numpy as np
import orjson
import soundfile as sf

from turkish_tts.audio import analyze_signal, sha256_file, signal_quality_reasons
from turkish_tts.manifests import ClipRecord, iter_jsonl, write_jsonl
from turkish_tts.normalize import normalize_for_asr_comparison

COMMON_VOICE_DATASET_TERMS_URL = "https://datacollective.mozillafoundation.org/datasets/cmqinosfq00x4nr07gnk0rdf9"
COMMON_VOICE_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
COMMON_VOICE_LICENSE_ID = "CC0-1.0+Mozilla-Data-Collective-Terms"
COMMON_VOICE_RESTRICTIONS = (
    "do_not_attempt_speaker_identification",
    "do_not_rehost_source_dataset",
    "respect_mozilla_data_collective_terms",
)
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 20.0
MIN_RMS_DBFS = -45.0
MAX_CLIPPING_RATIO = 0.01
MIN_ACTIVE_FRAME_RATIO = 0.25
MIN_ESTIMATED_SNR_DB = 6.0
LONG_TRANSCRIPT_MAX_CER = 0.40
SHORT_TRANSCRIPT_MAX_CER = 0.60
SHORT_TRANSCRIPT_CHARACTERS = 20
HIGH_CONFIDENCE_WRONG_LANGUAGE = 0.80


@dataclass(frozen=True, slots=True)
class AsrResult:
    text: str
    language: str
    language_probability: float
    speech_seconds: float
    language_was_forced: bool = False


Transcriber = Callable[[Path], AsrResult]


@runtime_checkable
class BatchTranscriber(Protocol):
    batch_size: int

    def transcribe_batch(self, records: Sequence[ClipRecord]) -> Sequence[AsrResult]: ...


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        num_workers: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )

    def __call__(self, path: Path) -> AsrResult:
        segments, info = self._model.transcribe(
            str(path),
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        materialized = list(segments)
        text = " ".join(segment.text.strip() for segment in materialized if segment.text.strip()).strip()
        speech_seconds = sum(max(0.0, float(segment.end) - float(segment.start)) for segment in materialized)
        return AsrResult(
            text=text,
            language=str(info.language or "unknown"),
            language_probability=float(info.language_probability or 0),
            speech_seconds=speech_seconds,
        )


class TransformersWhisperBatchTranscriber:
    def __init__(
        self,
        *,
        model_name: str = "openai/whisper-large-v3",
        device: str = "cuda",
        batch_size: int = 128,
        revision: str | None = None,
        token: str | None = None,
    ) -> None:
        import torch
        import torchaudio
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.batch_size = batch_size
        self._device = device
        self._dtype = torch.float16
        self._torch = torch
        self._torchaudio = torchaudio
        auto_processor: Any = AutoProcessor
        self._processor = auto_processor.from_pretrained(model_name, revision=revision, token=token)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            revision=revision,
            token=token,
            dtype=self._dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        self._model.eval()
        self._resamplers: dict[int, Any] = {}

    def transcribe_batch(self, records: Sequence[ClipRecord]) -> Sequence[AsrResult]:
        audio_arrays = [self._load_audio(Path(record.audio_path)) for record in records]
        features = self._processor.feature_extractor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            return_attention_mask=True,
        )
        with self._torch.inference_mode():
            tokens = self._model.generate(
                input_features=features.input_features.to(device=self._device, dtype=self._dtype),
                attention_mask=features.attention_mask.to(device=self._device),
                language="tr",
                task="transcribe",
                num_beams=1,
                max_new_tokens=128,
            )
        texts = self._processor.batch_decode(tokens, skip_special_tokens=True)
        if len(texts) != len(records):
            raise RuntimeError("Whisper batch output count did not match its input count")
        return [
            AsrResult(
                text=text.strip(),
                language="tr",
                language_probability=0,
                speech_seconds=_estimated_active_seconds(record),
                language_was_forced=True,
            )
            for record, text in zip(records, texts, strict=True)
        ]

    def _load_audio(self, path: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = self._torch.from_numpy(np.mean(audio, axis=1, dtype=np.float32))
        if sample_rate != 16_000:
            resampler = self._resamplers.get(sample_rate)
            if resampler is None:
                resampler = self._torchaudio.transforms.Resample(sample_rate, 16_000)
                self._resamplers[sample_rate] = resampler
            mono = resampler(mono)
        return mono.numpy()


def audit_common_voice_records(records: Sequence[ClipRecord], *, max_concurrency: int) -> list[ClipRecord]:
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        audited = list(executor.map(audit_common_voice_record, records))

    first_by_sha: dict[str, str] = {}
    deduplicated: list[ClipRecord] = []
    for record in sorted(audited, key=lambda item: item.clip_id):
        reasons = list(_quality_reasons(record))
        if record.sha256:
            first_clip_id = first_by_sha.setdefault(record.sha256, record.clip_id)
            if first_clip_id != record.clip_id:
                reasons.append("duplicate_audio")
        deduplicated.append(_with_quality_reasons(record, reasons, stage="acoustic_audit"))
    return sorted(deduplicated, key=lambda item: item.clip_id)


def audit_common_voice_record(record: ClipRecord) -> ClipRecord:
    metadata = _rights_metadata(record.metadata)
    reasons: list[str] = []
    try:
        audio, sample_rate = sf.read(record.audio_path, dtype="float32", always_2d=True)
        if audio.size == 0 or sample_rate <= 0:
            raise ValueError("empty decoded audio")
        metrics = analyze_signal(audio, sample_rate, active_floor_dbfs=MIN_RMS_DBFS)
        digest = sha256_file(Path(record.audio_path))
        metadata["common_voice_acoustics"] = metrics.as_metadata()
        reasons.extend(
            signal_quality_reasons(
                metrics,
                min_duration_seconds=MIN_DURATION_SECONDS,
                max_duration_seconds=MAX_DURATION_SECONDS,
                min_rms_dbfs=MIN_RMS_DBFS,
                max_clipping_ratio=MAX_CLIPPING_RATIO,
                min_active_frame_ratio=MIN_ACTIVE_FRAME_RATIO,
                min_estimated_snr_db=MIN_ESTIMATED_SNR_DB,
            )
        )
        return record.model_copy(
            update={
                "license_id": COMMON_VOICE_LICENSE_ID,
                "duration_seconds": metrics.duration_seconds,
                "sample_rate_hz": metrics.sample_rate_hz,
                "sha256": digest,
                "metadata": {
                    **metadata,
                    "quality_filter_reasons": sorted(set(reasons)),
                    "quality_stage": "acoustic_audit",
                },
            }
        )
    except Exception as error:
        metadata["common_voice_acoustic_error"] = type(error).__name__
        reasons.append("audio_decode_failed")
        return record.model_copy(
            update={
                "license_id": COMMON_VOICE_LICENSE_ID,
                "metadata": {
                    **metadata,
                    "quality_filter_reasons": reasons,
                    "quality_stage": "acoustic_audit",
                },
            }
        )


def annotate_common_voice_asr(record: ClipRecord, *, transcriber: Transcriber, model_name: str) -> ClipRecord:
    if _quality_reasons(record):
        return record
    result = transcriber(Path(record.audio_path))
    reference = normalize_for_asr_comparison(record.transcript or "")
    hypothesis = normalize_for_asr_comparison(result.text)
    error_rate = cast(float, jiwer.cer(reference, hypothesis)) if reference else 1.0
    speech_ratio = result.speech_seconds / (record.duration_seconds or 1)
    max_cer = SHORT_TRANSCRIPT_MAX_CER if len(reference) < SHORT_TRANSCRIPT_CHARACTERS else LONG_TRANSCRIPT_MAX_CER
    reasons: list[str] = []
    if not hypothesis:
        reasons.append("asr_empty")
    if result.language != "tr" and result.language_probability >= HIGH_CONFIDENCE_WRONG_LANGUAGE:
        reasons.append("wrong_language")
    if error_rate > max_cer:
        reasons.append("transcript_mismatch")
    if speech_ratio < MIN_ACTIVE_FRAME_RATIO:
        reasons.append("insufficient_asr_speech")
    metadata = {
        **record.metadata,
        "common_voice_asr": {
            "model": model_name,
            "text": result.text,
            "normalized_text": hypothesis,
            "language": result.language,
            "language_probability": round(result.language_probability, 6),
            "language_was_forced": result.language_was_forced,
            "character_error_rate": round(error_rate, 6),
            "speech_seconds": round(result.speech_seconds, 6),
            "speech_ratio": round(speech_ratio, 6),
            "max_allowed_character_error_rate": max_cer,
        },
        "quality_filter_reasons": sorted(set(reasons)),
        "quality_stage": "asr_agreement",
    }
    return record.model_copy(update={"metadata": metadata})


def transcribe_common_voice_with_checkpoints(
    records: Sequence[ClipRecord],
    *,
    transcriber: Transcriber | BatchTranscriber,
    model_name: str,
    checkpoint_path: Path,
    output_path: Path,
    max_concurrency: int = 1,
) -> list[ClipRecord]:
    completed = {record.clip_id: record for record in iter_jsonl(checkpoint_path)} if checkpoint_path.is_file() else {}
    _validate_asr_checkpoint(completed.values(), model_name=model_name)
    pending = [record for record in records if record.clip_id not in completed]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("ab") as checkpoint:
        if isinstance(transcriber, BatchTranscriber):
            annotated_records = _iter_batch_asr_annotations(
                pending,
                transcriber=transcriber,
                model_name=model_name,
            )
            _checkpoint_annotations(annotated_records, checkpoint=checkpoint, completed=completed)
        else:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                annotated_records = executor.map(
                    lambda record: _annotate_common_voice_asr_or_reject(
                        record,
                        transcriber=transcriber,
                        model_name=model_name,
                    ),
                    pending,
                )
                _checkpoint_annotations(annotated_records, checkpoint=checkpoint, completed=completed)
    ordered = [completed[record.clip_id] for record in records]
    write_jsonl(output_path, ordered)
    return ordered


def _annotate_common_voice_asr_or_reject(
    record: ClipRecord,
    *,
    transcriber: Transcriber,
    model_name: str,
) -> ClipRecord:
    try:
        return annotate_common_voice_asr(record, transcriber=transcriber, model_name=model_name)
    except Exception as error:
        return _asr_failure_record(record, model_name=model_name, error=error)


def _constant_transcriber(result: AsrResult) -> Transcriber:
    def transcribe(_path: Path) -> AsrResult:
        return result

    return transcribe


def _iter_batch_asr_annotations(
    records: Sequence[ClipRecord],
    *,
    transcriber: BatchTranscriber,
    model_name: str,
) -> Iterator[ClipRecord]:
    for offset in range(0, len(records), transcriber.batch_size):
        batch = records[offset : offset + transcriber.batch_size]
        eligible = [record for record in batch if not _quality_reasons(record)]
        annotated_by_id: dict[str, ClipRecord] = {}
        if eligible:
            try:
                results = transcriber.transcribe_batch(eligible)
                if len(results) != len(eligible):
                    raise RuntimeError("ASR batch output count did not match its input count")
                for record, result in zip(eligible, results, strict=True):
                    annotated_by_id[record.clip_id] = _annotate_common_voice_asr_or_reject(
                        record,
                        transcriber=_constant_transcriber(result),
                        model_name=model_name,
                    )
            except Exception as error:
                annotated_by_id.update(
                    (record.clip_id, _asr_failure_record(record, model_name=model_name, error=error))
                    for record in eligible
                )
        for record in batch:
            yield annotated_by_id.get(record.clip_id, record)


def _checkpoint_annotations(
    records: Iterable[ClipRecord],
    *,
    checkpoint: Any,
    completed: dict[str, ClipRecord],
) -> None:
    for completed_count, annotated in enumerate(records, start=1):
        checkpoint.write(orjson.dumps(annotated.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE))
        if completed_count % 100 == 0:
            checkpoint.flush()
        completed[annotated.clip_id] = annotated


def _validate_asr_checkpoint(records: Iterable[ClipRecord], *, model_name: str) -> None:
    for record in records:
        if _quality_reasons(record) and record.metadata.get("quality_stage") == "acoustic_audit":
            continue
        asr_metadata = record.metadata.get("common_voice_asr")
        if not isinstance(asr_metadata, dict) or asr_metadata.get("model") != model_name:
            raise ValueError(f"ASR checkpoint record {record.clip_id} does not match model {model_name}")


def _asr_failure_record(record: ClipRecord, *, model_name: str, error: Exception) -> ClipRecord:
    return _with_quality_reasons(
        record.model_copy(
            update={
                "metadata": {
                    **record.metadata,
                    "common_voice_asr": {
                        "model": model_name,
                        "error": type(error).__name__,
                    },
                }
            }
        ),
        [*_quality_reasons(record), "asr_failed"],
        stage="asr_agreement",
    )


def finalize_common_voice_splits(
    records: Sequence[ClipRecord],
    *,
    output_dir: Path,
    prefix: str,
) -> dict[str, object]:
    accepted: list[ClipRecord] = []
    rejected: list[ClipRecord] = []
    split_records: dict[str, list[ClipRecord]] = {"train": [], "validation": [], "test": []}
    for record in records:
        reasons = list(_quality_reasons(record))
        speaker_id = record.speaker_id
        if not speaker_id:
            reasons.append("missing_speaker_id")
        if reasons:
            rejected_record = _with_quality_reasons(record, reasons, stage="finalized")
            rejected.append(rejected_record.model_copy(update={"source_split": "rejected"}))
            continue
        assert speaker_id is not None
        split = deterministic_speaker_split(speaker_id)
        finalized = record.model_copy(
            update={
                "source_split": split,
                "metadata": {
                    **record.metadata,
                    "source_release_split": record.source_split,
                    "quality_filter_reasons": [],
                    "quality_stage": "finalized",
                },
            }
        )
        accepted.append(finalized)
        split_records[split].append(finalized)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"{prefix}.accepted.jsonl", accepted)
    write_jsonl(output_dir / f"{prefix}.rejected.jsonl", rejected)
    for split, values in split_records.items():
        write_jsonl(output_dir / f"{prefix}.{split}.jsonl", values)

    speaker_sets = {
        split: {record.speaker_id for record in values if record.speaker_id} for split, values in split_records.items()
    }
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(speaker_sets[left] & speaker_sets[right] for left, right in split_pairs):
        raise ValueError("Common Voice speaker-disjoint split invariant failed")
    rejection_reasons = Counter(reason for record in rejected for reason in _quality_reasons(record))
    return {
        "records": len(records),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "accepted_hours": round(sum(record.duration_seconds or 0 for record in accepted) / 3600, 3),
        "rejected_hours": round(sum(record.duration_seconds or 0 for record in rejected) / 3600, 3),
        "splits": {
            split: {
                "records": len(values),
                "hours": round(sum(record.duration_seconds or 0 for record in values) / 3600, 3),
                "speakers": len(speaker_sets[split]),
            }
            for split, values in split_records.items()
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def deterministic_speaker_split(speaker_id: str) -> str:
    bucket = int(hashlib.sha256(f"common-voice-split-v1:{speaker_id}".encode()).hexdigest()[:8], 16) % 10_000
    if bucket < 9_000:
        return "train"
    if bucket < 9_500:
        return "validation"
    return "test"


def acoustic_audit_summary(records: Sequence[ClipRecord]) -> dict[str, object]:
    reasons = Counter(reason for record in records for reason in _quality_reasons(record))
    candidates = [record for record in records if not _quality_reasons(record)]
    return {
        "records": len(records),
        "candidate_records": len(candidates),
        "rejected_before_asr": len(records) - len(candidates),
        "candidate_hours": round(sum(record.duration_seconds or 0 for record in candidates) / 3600, 3),
        "rejection_reasons": dict(sorted(reasons.items())),
        "license_id": COMMON_VOICE_LICENSE_ID,
        "dataset_terms_url": COMMON_VOICE_DATASET_TERMS_URL,
    }


def _rights_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        **metadata,
        "license_url": COMMON_VOICE_LICENSE_URL,
        "dataset_terms_url": COMMON_VOICE_DATASET_TERMS_URL,
        "dataset_restrictions": list(COMMON_VOICE_RESTRICTIONS),
        "attribution": "Mozilla Common Voice contributors",
    }


def _estimated_active_seconds(record: ClipRecord) -> float:
    acoustics = record.metadata.get("common_voice_acoustics")
    active_ratio = acoustics.get("active_frame_ratio") if isinstance(acoustics, dict) else None
    ratio = float(active_ratio) if isinstance(active_ratio, int | float) else 1.0
    return (record.duration_seconds or 0) * min(1.0, max(0.0, ratio))


def _quality_reasons(record: ClipRecord) -> tuple[str, ...]:
    value = record.metadata.get("quality_filter_reasons")
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _with_quality_reasons(record: ClipRecord, reasons: Iterable[str], *, stage: str) -> ClipRecord:
    return record.model_copy(
        update={
            "metadata": {
                **record.metadata,
                "quality_filter_reasons": sorted(set(reasons)),
                "quality_stage": stage,
            }
        }
    )

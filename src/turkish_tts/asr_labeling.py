from __future__ import annotations

# ruff: noqa: RUF001 -- Turkish grapheme and IPA mappings are intentional
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import jiwer
import numpy as np
import orjson
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from turkish_tts.common_voice_prepare import BatchTranscriber, TransformersWhisperBatchTranscriber
from turkish_tts.manifests import ClipRecord, iter_jsonl, write_jsonl
from turkish_tts.normalize import (
    normalize_for_asr_comparison,
    normalize_for_ctc_alignment,
    normalize_for_model,
)

WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
CTC_MODEL_ID = "m3hrdadfi/wav2vec2-large-xlsr-turkish"
CTC_MODEL_REVISION = "8699cf317b8f9a834ddf192608324ae5bbd191f0"
LABELING_VERSION = "voicedata-dual-asr-ctc-align-v1"


class LabelingThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_id: str = Field(min_length=1)
    calibration_source: str = Field(min_length=1)
    calibration_records: int = Field(gt=0)
    max_consensus_cer: float = Field(ge=0, le=1)
    min_ctc_confidence: float = Field(ge=0, le=1)
    min_alignment_probability: float = Field(ge=0, le=1)
    max_leading_unaligned_ratio: float = Field(ge=0, le=1)
    max_trailing_unaligned_ratio: float = Field(ge=0, le=1)
    min_characters_per_second: float = Field(gt=0)
    max_characters_per_second: float = Field(gt=0)


@dataclass(frozen=True, slots=True)
class CtcResult:
    text: str
    confidence: float
    alignment: dict[str, object] | None
    alignment_error: str | None = None


class CtcRecognizer(Protocol):
    def transcribe_and_align(
        self,
        records: Sequence[ClipRecord],
        target_texts: Sequence[str],
    ) -> Sequence[CtcResult]: ...


class TurkishCtcRecognizer:
    def __init__(
        self,
        *,
        token: str | None,
        device: str = "cuda",
        model_id: str = CTC_MODEL_ID,
        revision: str = CTC_MODEL_REVISION,
    ) -> None:
        import torch
        import torchaudio
        from transformers import AutoModelForCTC, AutoProcessor

        auto_processor: Any = AutoProcessor
        auto_model: Any = AutoModelForCTC
        self._processor = auto_processor.from_pretrained(model_id, revision=revision, token=token)
        self._model = auto_model.from_pretrained(model_id, revision=revision, token=token).to(device)
        self._model.eval()
        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device
        self._blank_id = int(self._model.config.pad_token_id)
        self._unknown_id = int(self._processor.tokenizer.unk_token_id)
        self._resamplers: dict[int, Any] = {}

    def transcribe_and_align(
        self,
        records: Sequence[ClipRecord],
        target_texts: Sequence[str],
    ) -> list[CtcResult]:
        if len(records) != len(target_texts):
            raise ValueError("CTC records and target texts must have identical lengths")
        audios = [self._load_audio(Path(record.audio_path)) for record in records]
        encoded = self._processor(
            audios,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_values = encoded.input_values.to(self._device)
        attention_mask = encoded.attention_mask.to(self._device)
        with self._torch.inference_mode():
            logits = self._model(input_values, attention_mask=attention_mask).logits
        log_probs = self._torch.log_softmax(logits.float(), dim=-1).cpu()
        predicted_ids = self._torch.argmax(logits, dim=-1).cpu()
        decoded = self._processor.batch_decode(predicted_ids)
        input_lengths = attention_mask.sum(dim=-1)
        model: Any = self._model
        output_lengths = model._get_feat_extract_output_lengths(input_lengths).cpu()
        results: list[CtcResult] = []
        for index, (record, target_text, hypothesis) in enumerate(zip(records, target_texts, decoded, strict=True)):
            valid_frames = int(output_lengths[index])
            record_log_probs = log_probs[index, :valid_frames]
            record_predictions = predicted_ids[index, :valid_frames]
            confidence = _ctc_confidence(record_log_probs, record_predictions, blank_id=self._blank_id)
            try:
                alignment = self._align(
                    record_log_probs,
                    target_text=target_text,
                    duration_seconds=record.duration_seconds or 0,
                )
                alignment_error = None
            except Exception as error:
                alignment = None
                alignment_error = type(error).__name__
            results.append(
                CtcResult(
                    text=str(hypothesis).strip(),
                    confidence=confidence,
                    alignment=alignment,
                    alignment_error=alignment_error,
                )
            )
        return results

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

    def _align(
        self,
        log_probs: Any,
        *,
        target_text: str,
        duration_seconds: float,
    ) -> dict[str, object]:
        normalized_target = normalize_for_ctc_alignment(target_text)
        if not normalized_target:
            raise ValueError("alignment target is empty")
        target_ids = self._processor.tokenizer(normalized_target, add_special_tokens=False).input_ids
        if not target_ids or self._unknown_id in target_ids:
            raise ValueError("alignment target contains unsupported graphemes")
        targets = self._torch.tensor([target_ids], dtype=self._torch.int32)
        if len(target_ids) >= log_probs.shape[0]:
            raise ValueError("alignment target is longer than the CTC emission sequence")
        aligned_tokens, scores = self._torchaudio.functional.forced_align(
            log_probs.unsqueeze(0),
            targets,
            blank=self._blank_id,
        )
        spans = self._torchaudio.functional.merge_tokens(
            aligned_tokens[0],
            scores[0],
            blank=self._blank_id,
        )
        if len(spans) != len(target_ids):
            raise ValueError("forced alignment did not cover every target token")
        frame_count = int(log_probs.shape[0])
        seconds_per_frame = duration_seconds / frame_count
        leading_ratio = spans[0].start / frame_count
        trailing_ratio = (frame_count - spans[-1].end) / frame_count
        token_probabilities = [math.exp(float(span.score)) for span in spans]
        token_timings = [
            {
                "grapheme": str(self._processor.tokenizer.convert_ids_to_tokens(int(span.token))),
                "start_seconds": round(span.start * seconds_per_frame, 6),
                "end_seconds": round(span.end * seconds_per_frame, 6),
                "probability": round(probability, 6),
            }
            for span, probability in zip(spans, token_probabilities, strict=True)
        ]
        return {
            "model": CTC_MODEL_ID,
            "revision": CTC_MODEL_REVISION,
            "target": normalized_target,
            "mean_token_probability": round(float(np.mean(token_probabilities)), 6),
            "min_token_probability": round(float(np.min(token_probabilities)), 6),
            "leading_unaligned_ratio": round(leading_ratio, 6),
            "trailing_unaligned_ratio": round(trailing_ratio, 6),
            "aligned_span_ratio": round((spans[-1].end - spans[0].start) / frame_count, 6),
            "word_timings": _word_timings(token_timings),
            "phoneme_timings": _phoneme_timings(token_timings),
        }


def calibrate_labeling_thresholds(
    records: Sequence[ClipRecord],
    *,
    recognizer: CtcRecognizer,
    sample_size: int,
    source_name: str,
) -> tuple[LabelingThresholds, dict[str, object]]:
    eligible = [
        record
        for record in records
        if record.transcript and _common_voice_asr_text(record) and not _quality_reasons(record)
    ]
    sample = sorted(eligible, key=_calibration_order)[:sample_size]
    rows: list[dict[str, float]] = []
    for offset in range(0, len(sample), 24):
        batch = sample[offset : offset + 24]
        human_texts = [record.transcript or "" for record in batch]
        results = recognizer.transcribe_and_align(batch, human_texts)
        for record, result in zip(batch, results, strict=True):
            if result.alignment is None:
                continue
            whisper_text = _common_voice_asr_text(record)
            if whisper_text is None:
                continue
            human = normalize_for_asr_comparison(record.transcript or "")
            whisper = normalize_for_asr_comparison(whisper_text)
            ctc = normalize_for_asr_comparison(result.text)
            alignment = result.alignment
            rows.append(
                {
                    "whisper_human_cer": _cer(human, whisper),
                    "ctc_human_cer": _cer(human, ctc),
                    "consensus_cer": _cer(whisper, ctc),
                    "ctc_confidence": result.confidence,
                    "alignment_probability": _as_float(alignment["mean_token_probability"]),
                    "leading_unaligned_ratio": _as_float(alignment["leading_unaligned_ratio"]),
                    "trailing_unaligned_ratio": _as_float(alignment["trailing_unaligned_ratio"]),
                    "characters_per_second": len(human.replace(" ", "")) / (record.duration_seconds or 1),
                }
            )
    clean = [row for row in rows if row["whisper_human_cer"] <= 0.25 and row["ctc_human_cer"] <= 0.35]
    if len(clean) < max(50, sample_size // 4):
        raise ValueError("insufficient clean Turkish calibration rows")
    calibration_id = hashlib.sha256(
        (LABELING_VERSION + ":" + ":".join(record.clip_id for record in sample)).encode()
    ).hexdigest()[:20]
    thresholds = LabelingThresholds(
        calibration_id=calibration_id,
        calibration_source=source_name,
        calibration_records=len(clean),
        max_consensus_cer=min(0.40, max(0.15, _percentile(clean, "consensus_cer", 95) + 0.02)),
        min_ctc_confidence=max(0.10, _percentile(clean, "ctc_confidence", 5) - 0.02),
        min_alignment_probability=max(0.10, _percentile(clean, "alignment_probability", 5) - 0.02),
        max_leading_unaligned_ratio=min(
            0.30,
            _percentile(clean, "leading_unaligned_ratio", 95) + 0.02,
        ),
        max_trailing_unaligned_ratio=min(
            0.30,
            _percentile(clean, "trailing_unaligned_ratio", 95) + 0.02,
        ),
        min_characters_per_second=max(0.5, _percentile(clean, "characters_per_second", 1) * 0.8),
        max_characters_per_second=min(30.0, _percentile(clean, "characters_per_second", 99) * 1.2),
    )
    report: dict[str, object] = {
        "labeling_version": LABELING_VERSION,
        "source": source_name,
        "sample_requested": sample_size,
        "sample_aligned": len(rows),
        "clean_calibration_records": len(clean),
        "sample_identity_sha256": hashlib.sha256("\n".join(record.clip_id for record in sample).encode()).hexdigest(),
        "models": {
            "whisper": {"id": WHISPER_MODEL_ID, "revision": WHISPER_MODEL_REVISION},
            "ctc": {"id": CTC_MODEL_ID, "revision": CTC_MODEL_REVISION},
        },
        "distributions": {
            key: _distribution(rows, key)
            for key in (
                "whisper_human_cer",
                "ctc_human_cer",
                "consensus_cer",
                "ctc_confidence",
                "alignment_probability",
                "leading_unaligned_ratio",
                "trailing_unaligned_ratio",
                "characters_per_second",
            )
        },
        "thresholds": thresholds.model_dump(mode="json"),
    }
    return thresholds, report


def run_whisper_pseudolabels(
    records: Sequence[ClipRecord],
    *,
    token: str | None,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 128,
    transcriber: BatchTranscriber | None = None,
) -> list[ClipRecord]:
    completed = _load_checkpoint(checkpoint_path, stage="whisper_pseudolabel")
    pending = [record for record in records if record.clip_id not in completed]
    if transcriber is None:
        transcriber = TransformersWhisperBatchTranscriber(
            model_name=WHISPER_MODEL_ID,
            revision=WHISPER_MODEL_REVISION,
            token=token,
            batch_size=batch_size,
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("ab") as checkpoint:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            eligible = [record for record in batch if not _quality_reasons(record)]
            labeled_by_id: dict[str, ClipRecord] = {}
            if eligible:
                try:
                    results = transcriber.transcribe_batch(eligible)
                except Exception as error:
                    results = []
                    for record in eligible:
                        labeled_by_id[record.clip_id] = _add_reasons(
                            record,
                            ["whisper_failed"],
                            stage="whisper_pseudolabel",
                            extra={"whisper_error": type(error).__name__},
                        )
                else:
                    for record, result in zip(eligible, results, strict=True):
                        reasons = ["whisper_empty"] if not normalize_for_asr_comparison(result.text) else []
                        labeled_by_id[record.clip_id] = _add_reasons(
                            record.model_copy(
                                update={
                                    "transcript": result.text,
                                    "normalized_transcript": normalize_for_model(result.text),
                                    "transcription_model": WHISPER_MODEL_ID,
                                    "transcription_language": "tr",
                                }
                            ),
                            reasons,
                            stage="whisper_pseudolabel",
                            extra={
                                "whisper_asr": {
                                    "model": WHISPER_MODEL_ID,
                                    "revision": WHISPER_MODEL_REVISION,
                                    "text": result.text,
                                    "language": "tr",
                                    "language_was_forced": True,
                                }
                            },
                        )
            for record in batch:
                labeled = labeled_by_id.get(record.clip_id, record)
                _append_checkpoint(checkpoint, labeled)
                completed[labeled.clip_id] = labeled
    ordered = [completed[record.clip_id] for record in records]
    write_jsonl(output_path, ordered)
    return ordered


def run_ctc_consensus_and_alignment(
    records: Sequence[ClipRecord],
    *,
    thresholds: LabelingThresholds,
    token: str | None,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 24,
    recognizer: CtcRecognizer | None = None,
) -> list[ClipRecord]:
    completed = _load_checkpoint(checkpoint_path, stage="ctc_consensus_alignment")
    pending = [record for record in records if record.clip_id not in completed]
    if recognizer is None:
        recognizer = TurkishCtcRecognizer(token=token)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("ab") as checkpoint:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            eligible = [record for record in batch if not _quality_reasons(record) and record.transcript]
            labeled_by_id: dict[str, ClipRecord] = {}
            if eligible:
                target_texts = [record.transcript or "" for record in eligible]
                try:
                    results = recognizer.transcribe_and_align(eligible, target_texts)
                except Exception as error:
                    results = []
                    for record in eligible:
                        labeled_by_id[record.clip_id] = _add_reasons(
                            record,
                            ["ctc_failed"],
                            stage="ctc_consensus_alignment",
                            extra={"ctc_error": type(error).__name__},
                        )
                else:
                    for record, result in zip(eligible, results, strict=True):
                        labeled_by_id[record.clip_id] = _apply_ctc_result(
                            record,
                            result,
                            thresholds=thresholds,
                        )
            for record in batch:
                labeled = labeled_by_id.get(record.clip_id, record)
                _append_checkpoint(checkpoint, labeled)
                completed[labeled.clip_id] = labeled
    ordered = [completed[record.clip_id] for record in records]
    write_jsonl(output_path, ordered)
    return ordered


def labeling_summary(records: Sequence[ClipRecord]) -> dict[str, object]:
    accepted = [record for record in records if not _quality_reasons(record)]
    reasons = Counter(reason for record in records for reason in _quality_reasons(record))
    return {
        "labeling_version": LABELING_VERSION,
        "records": len(records),
        "accepted_records": len(accepted),
        "rejected_records": len(records) - len(accepted),
        "accepted_hours": round(sum(record.duration_seconds or 0 for record in accepted) / 3600, 3),
        "rejected_hours": round(
            sum(record.duration_seconds or 0 for record in records if _quality_reasons(record)) / 3600,
            3,
        ),
        "rejection_reasons": dict(sorted(reasons.items())),
        "models": {
            "whisper": {"id": WHISPER_MODEL_ID, "revision": WHISPER_MODEL_REVISION},
            "ctc": {"id": CTC_MODEL_ID, "revision": CTC_MODEL_REVISION},
        },
    }


def write_labeling_config(path: Path, *, thresholds: LabelingThresholds, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"thresholds": thresholds.model_dump(mode="json"), "calibration": report}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_labeling_thresholds(path: Path) -> LabelingThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds") if isinstance(payload, dict) else None
    return LabelingThresholds.model_validate(thresholds)


def _apply_ctc_result(
    record: ClipRecord,
    result: CtcResult,
    *,
    thresholds: LabelingThresholds,
) -> ClipRecord:
    whisper = normalize_for_asr_comparison(record.transcript or "")
    ctc = normalize_for_asr_comparison(result.text)
    consensus_cer = _cer(whisper, ctc)
    characters_per_second = len(whisper.replace(" ", "")) / (record.duration_seconds or 1)
    reasons: list[str] = []
    if not ctc:
        reasons.append("ctc_empty")
    if result.confidence < thresholds.min_ctc_confidence:
        reasons.append("ctc_low_confidence")
    if consensus_cer > thresholds.max_consensus_cer:
        reasons.append("dual_asr_disagreement")
    if not thresholds.min_characters_per_second <= characters_per_second <= thresholds.max_characters_per_second:
        reasons.append("implausible_transcript_rate")
    alignment = result.alignment
    if alignment is None:
        reasons.append("forced_alignment_failed")
    else:
        if _as_float(alignment["mean_token_probability"]) < thresholds.min_alignment_probability:
            reasons.append("low_alignment_probability")
        if _as_float(alignment["leading_unaligned_ratio"]) > thresholds.max_leading_unaligned_ratio:
            reasons.append("speech_before_transcript")
        if _as_float(alignment["trailing_unaligned_ratio"]) > thresholds.max_trailing_unaligned_ratio:
            reasons.append("speech_after_transcript")
    return _add_reasons(
        record,
        reasons,
        stage="ctc_consensus_alignment",
        extra={
            "ctc_asr": {
                "model": CTC_MODEL_ID,
                "revision": CTC_MODEL_REVISION,
                "text": result.text,
                "confidence": round(result.confidence, 6),
            },
            "dual_asr": {
                "character_error_rate": round(consensus_cer, 6),
                "max_allowed_character_error_rate": thresholds.max_consensus_cer,
                "characters_per_second": round(characters_per_second, 6),
                "calibration_id": thresholds.calibration_id,
            },
            "forced_alignment": alignment
            or {
                "model": CTC_MODEL_ID,
                "revision": CTC_MODEL_REVISION,
                "error": result.alignment_error,
            },
        },
    )


def _ctc_confidence(log_probs: Any, predictions: Any, *, blank_id: int) -> float:
    frame_probabilities = log_probs.exp().max(dim=-1).values
    speech_probabilities = frame_probabilities[predictions != blank_id]
    if speech_probabilities.numel() == 0:
        return 0.0
    return float(speech_probabilities.mean())


def _word_timings(tokens: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    graphemes: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    probabilities: list[float] = []
    token_stream = list(tokens)
    token_stream.append({"grapheme": "|"})
    for token in token_stream:
        grapheme = str(token["grapheme"])
        if grapheme == "|":
            if graphemes:
                words.append(
                    {
                        "word": "".join(graphemes),
                        "start_seconds": round(starts[0], 6),
                        "end_seconds": round(ends[-1], 6),
                        "probability": round(float(np.mean(probabilities)), 6),
                    }
                )
                graphemes, starts, ends, probabilities = [], [], [], []
            continue
        graphemes.append(grapheme)
        starts.append(_as_float(token["start_seconds"]))
        ends.append(_as_float(token["end_seconds"]))
        probabilities.append(_as_float(token["probability"]))
    return words


def _phoneme_timings(tokens: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    phonemes = {
        "c": "dʒ",
        "ç": "tʃ",
        "ğ": "ɰ",
        "ı": "ɯ",
        "j": "ʒ",
        "ö": "œ",
        "r": "ɾ",
        "ş": "ʃ",
        "ü": "y",
        "y": "j",
        "q": "k",
        "w": "v",
        "x": "ks",
    }
    return [
        {
            "grapheme": token["grapheme"],
            "phoneme": phonemes.get(str(token["grapheme"]), token["grapheme"]),
            "start_seconds": token["start_seconds"],
            "end_seconds": token["end_seconds"],
            "probability": token["probability"],
        }
        for token in tokens
        if token["grapheme"] != "|"
    ]


def _load_checkpoint(path: Path, *, stage: str) -> dict[str, ClipRecord]:
    if not path.is_file():
        return {}
    records = {record.clip_id: record for record in iter_jsonl(path)}
    allowed_stages = {
        "whisper_pseudolabel": {"segmentation", "whisper_pseudolabel"},
        "ctc_consensus_alignment": {
            "segmentation",
            "whisper_pseudolabel",
            "ctc_consensus_alignment",
        },
    }[stage]
    for record in records.values():
        if record.metadata.get("quality_stage") not in allowed_stages:
            raise ValueError(f"checkpoint {path} contains an incompatible stage for {record.clip_id}")
    return records


def _append_checkpoint(checkpoint: Any, record: ClipRecord) -> None:
    checkpoint.write(orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE))
    checkpoint.flush()


def _add_reasons(
    record: ClipRecord,
    reasons: Iterable[str],
    *,
    stage: str,
    extra: dict[str, object] | None = None,
) -> ClipRecord:
    existing = _quality_reasons(record)
    return record.model_copy(
        update={
            "metadata": {
                **record.metadata,
                **(extra or {}),
                "quality_filter_reasons": sorted(set([*existing, *reasons])),
                "quality_stage": stage,
                "labeling_version": LABELING_VERSION,
            }
        }
    )


def _quality_reasons(record: ClipRecord) -> tuple[str, ...]:
    reasons = record.metadata.get("quality_filter_reasons")
    if not isinstance(reasons, list):
        return ()
    return tuple(reason for reason in reasons if isinstance(reason, str))


def _common_voice_asr_text(record: ClipRecord) -> str | None:
    metadata = record.metadata.get("common_voice_asr")
    if not isinstance(metadata, dict):
        return None
    text = metadata.get("text")
    return text if isinstance(text, str) and text.strip() else None


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric metadata, got {type(value).__name__}")
    return float(value)


def _cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 1.0
    return cast(float, jiwer.cer(reference, hypothesis))


def _calibration_order(record: ClipRecord) -> str:
    return hashlib.sha256(f"{LABELING_VERSION}:{record.clip_id}".encode()).hexdigest()


def _percentile(rows: Sequence[dict[str, float]], key: str, percentile: float) -> float:
    return float(np.percentile([row[key] for row in rows], percentile))


def _distribution(rows: Sequence[dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }

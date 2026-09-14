from __future__ import annotations

# ruff: noqa: RUF001 -- Turkish model output and fixture text can contain dotless i
import hashlib
import math
import mimetypes
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import jiwer
import numpy as np
import orjson
import soundfile as sf

from turkish_tts.audio import analyze_signal, sha256_file, signal_quality_reasons
from turkish_tts.manifests import ClipRecord
from turkish_tts.normalize import normalize_for_asr_comparison, normalize_for_model

SCRIPTED_SEGMENTATION_VERSION = "scripted-whisper-alignment-v1"
OUTPUT_SAMPLE_RATE = 24_000
_WORD_PATTERN = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüQWXqwx]+(?:['’][0-9A-Za-zÇĞİÖŞÜçğıöşüQWXqwx]+)?")
_SENTENCE_END = re.compile(r"[.!?]")
_CLAUSE_END = re.compile(r"[,;:]")


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start_seconds: float
    end_seconds: float
    probability: float


@dataclass(frozen=True, slots=True)
class TimedTranscript:
    text: str
    language: str
    language_probability: float
    words: tuple[TimedWord, ...]


class TimedTranscriber(Protocol):
    model_name: str

    def transcribe(self, path: Path) -> TimedTranscript: ...


class FasterWhisperTimedTranscriber:
    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        num_workers: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )

    def transcribe(self, path: Path) -> TimedTranscript:
        segments, info = self._model.transcribe(
            str(path),
            language="tr",
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=True,
        )
        materialized = list(segments)
        words = tuple(
            TimedWord(
                text=str(word.word).strip(),
                start_seconds=float(word.start),
                end_seconds=float(word.end),
                probability=float(word.probability),
            )
            for segment in materialized
            for word in (segment.words or ())
            if str(word.word).strip() and float(word.end) > float(word.start)
        )
        text = " ".join(segment.text.strip() for segment in materialized if segment.text.strip()).strip()
        if not words or not text:
            raise ValueError(f"timed ASR produced no speech for {path}")
        return TimedTranscript(
            text=text,
            language=str(info.language or "unknown"),
            language_probability=float(info.language_probability or 0),
            words=words,
        )


class ElevenLabsScribeTimedTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "scribe_v2",
        language_code: str = "tur",
        client: httpx.Client | None = None,
        zero_retention: bool = False,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ElevenLabs API key is required")
        self.model_name = f"elevenlabs/{model_name}"
        self._api_key = api_key
        self._api_model_name = model_name
        self._language_code = language_code
        self._zero_retention = zero_retention
        self._client = client

    def transcribe(self, path: Path) -> TimedTranscript:
        if not path.is_file():
            raise FileNotFoundError(path)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as audio:
            headers = {"xi-api-key": self._api_key}
            params = {"enable_logging": "false"} if self._zero_retention else {}
            data = {
                "model_id": self._api_model_name,
                "language_code": self._language_code,
                "timestamps_granularity": "word",
                "diarize": "false",
                "tag_audio_events": "false",
                "no_verbatim": "false",
            }
            files = {"file": (path.name, audio, mime_type)}
            if self._client is not None:
                response = self._client.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers=headers,
                    params=params,
                    data=data,
                    files=files,
                )
            else:
                with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                    response = client.post(
                        "https://api.elevenlabs.io/v1/speech-to-text",
                        headers=headers,
                        params=params,
                        data=data,
                        files=files,
                    )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Scribe returned an invalid response for {path}")
        words_payload = payload.get("words")
        if not isinstance(words_payload, list):
            raise ValueError(f"Scribe returned no word timestamps for {path}")
        words = tuple(
            TimedWord(
                text=str(word["text"]).strip(),
                start_seconds=float(word["start"]),
                end_seconds=float(word["end"]),
                probability=_probability_from_logprob(word.get("logprob")),
            )
            for word in words_payload
            if isinstance(word, dict)
            and word.get("type") == "word"
            and str(word.get("text") or "").strip()
            and isinstance(word.get("start"), int | float)
            and isinstance(word.get("end"), int | float)
            and float(word["end"]) > float(word["start"])
        )
        text = str(payload.get("text") or "").strip()
        if not words or not text:
            raise ValueError(f"Scribe produced no timed speech for {path}")
        return TimedTranscript(
            text=text,
            language=_canonical_language_code(str(payload.get("language_code") or self._language_code)),
            language_probability=float(payload.get("language_probability") or 0),
            words=words,
        )


def _probability_from_logprob(value: object) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return 0.0
    return min(1.0, max(0.0, math.exp(float(value))))


def _canonical_language_code(value: str) -> str:
    return "tr" if value.casefold() in {"tr", "tur"} else value


@dataclass(frozen=True, slots=True)
class ScriptedSegmentationConfig:
    target_min_seconds: float = 3.0
    target_max_seconds: float = 20.0
    absolute_max_seconds: float = 24.0
    audio_pad_seconds: float = 0.12
    min_exact_word_ratio: float = 0.70
    max_cer: float = 0.30
    min_mean_word_probability: float = 0.45
    min_language_probability: float = 0.70
    min_rms_dbfs: float = -50.0
    max_clipping_ratio: float = 0.02
    min_active_frame_ratio: float = 0.20
    min_estimated_snr_db: float = 3.0
    output_sample_rate: int = OUTPUT_SAMPLE_RATE
    peak_limit_dbfs: float = -1.0


@dataclass(frozen=True, slots=True)
class ScriptedSegmentationResult:
    segments: list[ClipRecord]
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _TextToken:
    normalized: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class _AsrToken:
    normalized: str
    word_index: int


def segment_scripted_records(
    records: Sequence[ClipRecord],
    *,
    output_dir: Path,
    transcriber: TimedTranscriber,
    transcript_cache_dir: Path | None = None,
    config: ScriptedSegmentationConfig | None = None,
) -> ScriptedSegmentationResult:
    resolved_config = config or ScriptedSegmentationConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    if transcript_cache_dir is not None:
        transcript_cache_dir.mkdir(parents=True, exist_ok=True)
    segments: list[ClipRecord] = []
    parent_reports: dict[str, object] = {}
    resamplers: dict[int, Any] = {}

    for parent in sorted(records, key=lambda item: item.clip_id):
        if not parent.transcript:
            raise ValueError(f"scripted parent {parent.clip_id} has no ground-truth transcript")
        if parent.metadata.get("script_ground_truth") is not True:
            raise ValueError(f"scripted parent {parent.clip_id} is not marked as script ground truth")
        source_audio, source_sample_rate = sf.read(parent.audio_path, dtype="float32", always_2d=True)
        if source_audio.size == 0 or source_sample_rate <= 0:
            raise ValueError(f"scripted parent {parent.clip_id} could not be decoded")
        source_mono = np.asarray(np.mean(source_audio, axis=1, dtype=np.float32), dtype=np.float32)
        transcript = _transcribe_parent(parent, transcriber=transcriber, cache_dir=transcript_cache_dir)
        script_tokens = _text_tokens(parent.transcript)
        asr_tokens = _asr_tokens(transcript.words)
        if not script_tokens or not asr_tokens:
            raise ValueError(f"scripted parent {parent.clip_id} did not produce alignable words")
        token_mapping = _align_tokens(script_tokens, asr_tokens)
        ranges = _chunk_ranges(
            parent.transcript,
            script_tokens,
            asr_tokens,
            transcript.words,
            token_mapping,
            resolved_config,
        )
        parent_segments: list[ClipRecord] = []
        for segment_index, (script_start, script_end) in enumerate(ranges):
            record = _materialize_segment(
                parent,
                parent.transcript,
                transcript,
                script_tokens,
                asr_tokens,
                token_mapping,
                script_start=script_start,
                script_end=script_end,
                segment_index=segment_index,
                source_audio=source_mono,
                source_sample_rate=source_sample_rate,
                output_dir=output_dir,
                transcriber_model=transcriber.model_name,
                config=resolved_config,
                resamplers=resamplers,
            )
            parent_segments.append(record)
            segments.append(record)
        aligned_exact = sum(
            mapping is not None and script_tokens[index].normalized == asr_tokens[mapping].normalized
            for index, mapping in enumerate(token_mapping)
        )
        parent_reports[parent.clip_id] = {
            "source_seconds": parent.duration_seconds,
            "asr_language": transcript.language,
            "asr_language_probability": round(transcript.language_probability, 6),
            "script_words": len(script_tokens),
            "asr_words": len(asr_tokens),
            "exact_aligned_words": aligned_exact,
            "exact_word_ratio": round(aligned_exact / len(script_tokens), 6),
            "segments": len(parent_segments),
            "accepted_segments": sum(not _quality_reasons(record) for record in parent_segments),
            "accepted_seconds": round(
                sum((record.duration_seconds or 0) for record in parent_segments if not _quality_reasons(record)), 6
            ),
        }

    reason_counts = Counter(reason for record in segments for reason in _quality_reasons(record))
    accepted = [record for record in segments if not _quality_reasons(record)]
    report: dict[str, object] = {
        "segmentation_version": SCRIPTED_SEGMENTATION_VERSION,
        "transcriber_model": transcriber.model_name,
        "config": asdict(resolved_config),
        "source_records": len(records),
        "source_hours": round(sum(record.duration_seconds or 0 for record in records) / 3600, 6),
        "segments": len(segments),
        "accepted_segments": len(accepted),
        "rejected_segments": len(segments) - len(accepted),
        "segment_hours": round(sum(record.duration_seconds or 0 for record in segments) / 3600, 6),
        "accepted_hours": round(sum(record.duration_seconds or 0 for record in accepted) / 3600, 6),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "parents": parent_reports,
    }
    return ScriptedSegmentationResult(segments=segments, report=report)


def _materialize_segment(
    parent: ClipRecord,
    script: str,
    transcript: TimedTranscript,
    script_tokens: Sequence[_TextToken],
    asr_tokens: Sequence[_AsrToken],
    token_mapping: Sequence[int | None],
    *,
    script_start: int,
    script_end: int,
    segment_index: int,
    source_audio: np.ndarray[Any, np.dtype[np.float32]],
    source_sample_rate: int,
    output_dir: Path,
    transcriber_model: str,
    config: ScriptedSegmentationConfig,
    resamplers: dict[int, Any],
) -> ClipRecord:
    mapped = _mapped_token_indices(token_mapping, script_start, script_end)
    if not mapped:
        raise ValueError(f"scripted range {script_start}:{script_end} has no timed words")
    first_word = transcript.words[asr_tokens[mapped[0]].word_index]
    last_word = transcript.words[asr_tokens[mapped[-1]].word_index]
    audio_start = max(0.0, first_word.start_seconds - config.audio_pad_seconds)
    audio_end = min(len(source_audio) / source_sample_rate, last_word.end_seconds + config.audio_pad_seconds)
    source_start = max(0, round(audio_start * source_sample_rate))
    source_end = min(len(source_audio), round(audio_end * source_sample_rate))
    segment_audio = source_audio[source_start:source_end].copy()
    if source_sample_rate != config.output_sample_rate:
        import torch
        import torchaudio

        resampler = resamplers.get(source_sample_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(source_sample_rate, config.output_sample_rate)
            resamplers[source_sample_rate] = resampler
        segment_audio = resampler(torch.from_numpy(segment_audio)).numpy().astype(np.float32, copy=False)
    peak = float(np.max(np.abs(segment_audio))) if segment_audio.size else 0.0
    peak_limit = 10 ** (config.peak_limit_dbfs / 20)
    gain = min(1.0, peak_limit / peak) if peak > 0 else 1.0
    if gain < 1.0:
        segment_audio *= gain
    metrics = analyze_signal(segment_audio[:, np.newaxis], config.output_sample_rate)

    text_start = 0 if script_start == 0 else _text_boundary(script, script_tokens, script_start - 1)
    text_end = (
        len(script) if script_end + 1 == len(script_tokens) else _text_boundary(script, script_tokens, script_end)
    )
    segment_text = script[text_start:text_end].strip()
    asr_start = min(mapped)
    asr_end = max(mapped)
    asr_text = " ".join(transcript.words[asr_tokens[index].word_index].text for index in range(asr_start, asr_end + 1))
    normalized_target = normalize_for_asr_comparison(segment_text)
    normalized_asr = normalize_for_asr_comparison(asr_text)
    cer_result = jiwer.cer(normalized_target, normalized_asr) if normalized_target else 1.0
    if not isinstance(cer_result, (int, float)):
        raise TypeError("character error rate must be numeric for one transcript pair")
    cer = float(cer_result)
    exact_words = 0
    for index in range(script_start, script_end + 1):
        mapped_index = token_mapping[index]
        if mapped_index is not None and script_tokens[index].normalized == asr_tokens[mapped_index].normalized:
            exact_words += 1
    exact_word_ratio = exact_words / (script_end - script_start + 1)
    word_indices = sorted({asr_tokens[index].word_index for index in mapped})
    mean_probability = float(np.mean([transcript.words[index].probability for index in word_indices]))
    reasons = signal_quality_reasons(
        metrics,
        min_duration_seconds=1.0,
        max_duration_seconds=config.absolute_max_seconds,
        min_rms_dbfs=config.min_rms_dbfs,
        max_clipping_ratio=config.max_clipping_ratio,
        min_active_frame_ratio=config.min_active_frame_ratio,
        min_estimated_snr_db=config.min_estimated_snr_db,
    )
    if exact_word_ratio < config.min_exact_word_ratio:
        reasons.append("script_word_alignment_low")
    if cer > config.max_cer:
        reasons.append("script_asr_cer_high")
    if mean_probability < config.min_mean_word_probability:
        reasons.append("asr_word_probability_low")
    if transcript.language != "tr" or transcript.language_probability < config.min_language_probability:
        reasons.append("turkish_language_confidence_low")

    identity = (
        f"{SCRIPTED_SEGMENTATION_VERSION}:{parent.clip_id}:{source_start}:{source_end}:"
        f"{script_start}:{script_end}:{config.output_sample_rate}"
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    clip_id = f"{parent.clip_id}-script-seg-{digest[:16]}"
    audio_path = output_dir / f"{clip_id}.wav"
    temporary_path = audio_path.with_suffix(".wav.part")
    sf.write(temporary_path, segment_audio, config.output_sample_rate, format="WAV", subtype="PCM_16")
    temporary_path.replace(audio_path)
    segment_sha256 = sha256_file(audio_path)
    metadata = {
        **parent.metadata,
        "parent_clip_id": parent.clip_id,
        "parent_sha256": parent.sha256,
        "parent_duration_seconds": parent.duration_seconds,
        "segment_index": segment_index,
        "segment_start_seconds": round(source_start / source_sample_rate, 6),
        "segment_end_seconds": round(source_end / source_sample_rate, 6),
        "script_start_word": script_start,
        "script_end_word": script_end,
        "scripted_segmentation_version": SCRIPTED_SEGMENTATION_VERSION,
        "scripted_segmentation_config": asdict(config),
        "audio_transform": {
            "source_sample_rate_hz": source_sample_rate,
            "output_sample_rate_hz": config.output_sample_rate,
            "channel_mix": "mean_to_mono",
            "encoding": "pcm_s16le",
            "peak_limit_dbfs": config.peak_limit_dbfs,
            "applied_gain_db": round(20 * math.log10(gain), 6) if gain > 0 else None,
        },
        "script_alignment": {
            "asr_model": transcriber_model,
            "asr_text": asr_text,
            "cer": round(cer, 6),
            "exact_word_ratio": round(exact_word_ratio, 6),
            "mean_word_probability": round(mean_probability, 6),
            "language": transcript.language,
            "language_probability": round(transcript.language_probability, 6),
        },
        "signal_metrics": metrics.as_metadata(),
        "quality_filter_reasons": sorted(set(reasons)),
        "quality_stage": "scripted_alignment_segmentation",
    }
    return parent.model_copy(
        update={
            "clip_id": clip_id,
            "source_row_id": f"{parent.source_row_id or parent.clip_id}:{script_start}-{script_end}",
            "audio_path": str(audio_path.resolve()),
            "transcript": segment_text,
            "normalized_transcript": normalize_for_model(segment_text),
            "duration_seconds": metrics.duration_seconds,
            "sample_rate_hz": config.output_sample_rate,
            "sha256": segment_sha256,
            "transcription_model": "script-ground-truth",
            "transcription_language": "tr",
            "metadata": metadata,
        }
    )


def _text_tokens(text: str) -> list[_TextToken]:
    tokens: list[_TextToken] = []
    for match in _WORD_PATTERN.finditer(text):
        normalized_parts = normalize_for_asr_comparison(match.group()).split()
        tokens.extend(_TextToken(part, match.start(), match.end()) for part in normalized_parts if part)
    return tokens


def _asr_tokens(words: Sequence[TimedWord]) -> list[_AsrToken]:
    tokens: list[_AsrToken] = []
    for word_index, word in enumerate(words):
        normalized_parts = normalize_for_asr_comparison(word.text).split()
        tokens.extend(_AsrToken(part, word_index) for part in normalized_parts if part)
    return tokens


def _align_tokens(script: Sequence[_TextToken], asr: Sequence[_AsrToken]) -> list[int | None]:
    script_count = len(script)
    asr_count = len(asr)
    costs = np.empty((script_count + 1, asr_count + 1), dtype=np.int32)
    trace = np.zeros((script_count + 1, asr_count + 1), dtype=np.uint8)
    costs[:, 0] = np.arange(script_count + 1)
    costs[0, :] = np.arange(asr_count + 1)
    trace[1:, 0] = 2
    trace[0, 1:] = 3
    for script_index in range(1, script_count + 1):
        for asr_index in range(1, asr_count + 1):
            substitution = 0 if script[script_index - 1].normalized == asr[asr_index - 1].normalized else 1
            diagonal = int(costs[script_index - 1, asr_index - 1]) + substitution
            deletion = int(costs[script_index - 1, asr_index]) + 1
            insertion = int(costs[script_index, asr_index - 1]) + 1
            if diagonal <= deletion and diagonal <= insertion:
                costs[script_index, asr_index] = diagonal
                trace[script_index, asr_index] = 1
            elif deletion <= insertion:
                costs[script_index, asr_index] = deletion
                trace[script_index, asr_index] = 2
            else:
                costs[script_index, asr_index] = insertion
                trace[script_index, asr_index] = 3
    mapping: list[int | None] = [None] * script_count
    script_index = script_count
    asr_index = asr_count
    while script_index or asr_index:
        operation = int(trace[script_index, asr_index])
        if operation == 1:
            mapping[script_index - 1] = asr_index - 1
            script_index -= 1
            asr_index -= 1
        elif operation == 2:
            script_index -= 1
        elif operation == 3:
            asr_index -= 1
        else:
            raise RuntimeError("word alignment backtrace reached an invalid state")
    return mapping


def _chunk_ranges(
    script: str,
    script_tokens: Sequence[_TextToken],
    asr_tokens: Sequence[_AsrToken],
    words: Sequence[TimedWord],
    mapping: Sequence[int | None],
    config: ScriptedSegmentationConfig,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(script_tokens):
        mapped_after_cursor = [index for index in range(cursor, len(mapping)) if mapping[index] is not None]
        if not mapped_after_cursor:
            break
        first_mapped = mapped_after_cursor[0]
        first_asr = mapping[first_mapped]
        if first_asr is None:
            raise RuntimeError("mapped token unexpectedly missing")
        start_time = words[asr_tokens[first_asr].word_index].start_seconds
        hard_end = first_mapped
        for index in mapped_after_cursor:
            asr_index = mapping[index]
            if asr_index is None:
                continue
            end_time = words[asr_tokens[asr_index].word_index].end_seconds
            if end_time - start_time > config.target_max_seconds:
                break
            hard_end = index
        if hard_end == first_mapped and first_mapped + 1 < len(script_tokens):
            hard_end = first_mapped + 1
        candidates = [
            index
            for index in range(first_mapped, hard_end + 1)
            if _boundary_priority(script, script_tokens, index) > 0
            and _mapped_duration(index, first_asr, mapping, asr_tokens, words) >= config.target_min_seconds
        ]
        if candidates:
            end = max(candidates, key=lambda index: (_boundary_priority(script, script_tokens, index), index))
        else:
            end = hard_end
        while end > cursor:
            mapped = _mapped_token_indices(mapping, cursor, end)
            if mapped:
                duration = (
                    words[asr_tokens[mapped[-1]].word_index].end_seconds
                    - words[asr_tokens[mapped[0]].word_index].start_seconds
                    + 2 * config.audio_pad_seconds
                )
                if duration <= config.absolute_max_seconds:
                    break
            end -= 1
        if end < cursor:
            end = cursor
        ranges.append((cursor, end))
        cursor = end + 1
    return ranges


def _boundary_priority(script: str, tokens: Sequence[_TextToken], index: int) -> int:
    separator_end = tokens[index + 1].start_char if index + 1 < len(tokens) else len(script)
    separator = script[tokens[index].end_char : separator_end]
    if _SENTENCE_END.search(separator) or index + 1 == len(tokens):
        return 2
    if _CLAUSE_END.search(separator):
        return 1
    return 0


def _mapped_duration(
    script_index: int,
    first_asr: int,
    mapping: Sequence[int | None],
    asr_tokens: Sequence[_AsrToken],
    words: Sequence[TimedWord],
) -> float:
    asr_index = mapping[script_index]
    if asr_index is None:
        return 0.0
    return words[asr_tokens[asr_index].word_index].end_seconds - words[asr_tokens[first_asr].word_index].start_seconds


def _mapped_token_indices(mapping: Sequence[int | None], start: int, end: int) -> list[int]:
    indices: list[int] = []
    for index in range(start, end + 1):
        mapped = mapping[index]
        if mapped is not None:
            indices.append(mapped)
    return indices


def _transcribe_parent(
    parent: ClipRecord,
    *,
    transcriber: TimedTranscriber,
    cache_dir: Path | None,
) -> TimedTranscript:
    if cache_dir is None:
        return transcriber.transcribe(Path(parent.audio_path))
    cache_path = cache_dir / f"{parent.clip_id}.json"
    if cache_path.is_file():
        payload = orjson.loads(cache_path.read_bytes())
        if (
            isinstance(payload, dict)
            and payload.get("model_name") == transcriber.model_name
            and payload.get("parent_sha256") == parent.sha256
        ):
            words = payload.get("words")
            if isinstance(words, list):
                return TimedTranscript(
                    text=str(payload.get("text") or ""),
                    language=_canonical_language_code(str(payload.get("language") or "unknown")),
                    language_probability=float(payload.get("language_probability") or 0),
                    words=tuple(TimedWord(**word) for word in words if isinstance(word, dict)),
                )
    transcript = transcriber.transcribe(Path(parent.audio_path))
    payload = {
        "segmentation_version": SCRIPTED_SEGMENTATION_VERSION,
        "model_name": transcriber.model_name,
        "parent_sha256": parent.sha256,
        "text": transcript.text,
        "language": transcript.language,
        "language_probability": transcript.language_probability,
        "words": [asdict(word) for word in transcript.words],
    }
    temporary_path = cache_path.with_suffix(".json.part")
    temporary_path.write_bytes(orjson.dumps(payload))
    temporary_path.replace(cache_path)
    return transcript


def _text_boundary(script: str, tokens: Sequence[_TextToken], index: int) -> int:
    separator_end = tokens[index + 1].start_char if index + 1 < len(tokens) else len(script)
    separator = script[tokens[index].end_char : separator_end]
    opening_mark = re.search(r"\s+([\"“‘«(]+)\s*$", separator)
    if opening_mark:
        return tokens[index].end_char + opening_mark.start(1)
    return separator_end


def _quality_reasons(record: ClipRecord) -> tuple[str, ...]:
    reasons = record.metadata.get("quality_filter_reasons")
    if not isinstance(reasons, list):
        return ()
    return tuple(reason for reason in reasons if isinstance(reason, str))

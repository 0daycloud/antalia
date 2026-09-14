from __future__ import annotations

# ruff: noqa: RUF001 -- Turkish PII patterns intentionally use dotless i
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from turkish_tts.manifests import ClipRecord, write_jsonl

SPEAKER_MODEL_ID = "microsoft/wavlm-base-plus-sv"
SPEAKER_MODEL_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
SPEAKER_MODEL_LICENSE = "MIT"
NER_MODEL_ID = "stanfordnlp/stanza-tr"
NER_MODEL_REVISION = "14240124e40c5c9f597a1fd15d3432cbdfb131d3"
NER_MODEL_LICENSE = "Apache-2.0"
NER_RESOURCES_COMMIT = "ca19085f5b979371bbf41109fd896ac987b5ee69"
NER_RESOURCES_SHA256 = "f7b0c91ec3648a892c1a75a93b59aa2b8be0d3221997bd98665d2840fa90885c"
NER_RESOURCES_URL = (
    f"https://raw.githubusercontent.com/stanfordnlp/stanza-resources/{NER_RESOURCES_COMMIT}/resources_1.14.0.json"
)
PRIVACY_GATE_VERSION = "turkish-pii-regex-stanza-v1"
SPEAKER_GATE_VERSION = "wavlm-parent-consistency-v1"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-zÇĞİÖŞÜçğıöşü]{2,}\b")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_IBAN = re.compile(r"\bTR\s*\d{2}(?:\s*\d{4}){5}\s*\d{2}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?90\s*)?(?:\(?0?5\d{2}\)?[\s.-]*)\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
_TC_IDENTITY = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
_SOCIAL_HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,30}\b")
_ADDRESS = re.compile(
    r"\b(?:cadde(?:si)?|sokak|mahallesi?|bulvar(?:ı)?|apartman(?:ı)?|daire)\b.{0,40}\b\d{1,4}\b",
    re.IGNORECASE,
)
_ADDRESS_WORD = re.compile(r"\b(?:cadde|sokak|mahalle|bulvar|apartman|daire|adres)\b", re.IGNORECASE)


class SpeakerThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_id: str = Field(min_length=1)
    calibration_source: str = Field(min_length=1)
    calibration_speakers: int = Field(gt=1)
    calibration_pairs: int = Field(gt=1)
    min_parent_cosine_similarity: float = Field(ge=-1, le=1)


@dataclass(frozen=True, slots=True)
class PrivacyFinding:
    category: str
    detector: str


class PrivacyScanner(Protocol):
    def scan(self, text: str) -> Sequence[PrivacyFinding]: ...


class TurkishPiiDetector:
    def __init__(self, *, token: str | None, use_gpu: bool = True) -> None:
        import stanza
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=NER_MODEL_ID,
                revision=NER_MODEL_REVISION,
                allow_patterns=["models/default.zip"],
                token=token,
            )
        )
        model_root = _prepare_stanza_runtime(snapshot)
        self._pipeline = stanza.Pipeline(
            lang="tr",
            dir=str(model_root),
            processors="tokenize,ner",
            download_method=stanza.DownloadMethod.NONE,
            use_gpu=use_gpu,
            verbose=False,
        )

    def scan(self, text: str) -> list[PrivacyFinding]:
        findings = _regex_privacy_findings(text)
        if findings:
            return findings
        document = self._pipeline(text)
        for entity in document.ents:
            entity_type = str(entity.type).upper()
            if entity_type in {"PER", "PERSON"}:
                findings.append(PrivacyFinding(category="person_name", detector="stanza_ner"))
            elif entity_type in {"LOC", "LOCATION"} and _ADDRESS_WORD.search(text):
                findings.append(PrivacyFinding(category="street_address", detector="stanza_ner"))
        return findings


class SpeakerEmbedder:
    def __init__(
        self,
        *,
        token: str | None,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        import torch
        import torchaudio
        from transformers import AutoFeatureExtractor, WavLMForXVector

        feature_extractor: Any = AutoFeatureExtractor
        model_class: Any = WavLMForXVector
        self._feature_extractor = feature_extractor.from_pretrained(
            SPEAKER_MODEL_ID,
            revision=SPEAKER_MODEL_REVISION,
            token=token,
        )
        self._model = model_class.from_pretrained(
            SPEAKER_MODEL_ID,
            revision=SPEAKER_MODEL_REVISION,
            token=token,
        ).to(device)
        self._model.eval()
        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device
        self._batch_size = batch_size
        self._resamplers: dict[int, Any] = {}

    def embed(self, records: Sequence[ClipRecord]) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
        embeddings: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
        for offset in range(0, len(records), self._batch_size):
            batch = records[offset : offset + self._batch_size]
            audio = [self._load_audio(Path(record.audio_path)) for record in batch]
            inputs = self._feature_extractor(
                audio,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                vectors = self._model(**model_inputs).embeddings
                vectors = self._torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
            embeddings.update(
                (record.clip_id, vector.astype(np.float32, copy=False))
                for record, vector in zip(batch, vectors, strict=True)
            )
        return embeddings

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


def run_privacy_gate(
    records: Sequence[ClipRecord],
    *,
    token: str | None,
    output_path: Path,
    use_gpu: bool = True,
    detector: PrivacyScanner | None = None,
) -> list[ClipRecord]:
    if detector is None:
        detector = TurkishPiiDetector(token=token, use_gpu=use_gpu)
    gated: list[ClipRecord] = []
    for record in records:
        if _quality_reasons(record) or not record.transcript:
            gated.append(record)
            continue
        findings = detector.scan(record.transcript)
        metadata: dict[str, object] = {
            **record.metadata,
            "privacy_scan": {
                "version": PRIVACY_GATE_VERSION,
                "ner_model": NER_MODEL_ID,
                "ner_revision": NER_MODEL_REVISION,
                "ner_license": NER_MODEL_LICENSE,
                "resources_commit": NER_RESOURCES_COMMIT,
                "resources_sha256": NER_RESOURCES_SHA256,
                "findings": sorted({finding.category for finding in findings}),
                "passed": not findings,
            },
            "quality_stage": "privacy_gate",
        }
        if findings:
            reasons = sorted(set([*_quality_reasons(record), "pii_detected"]))
            metadata["quality_filter_reasons"] = reasons
            metadata = _redact_text_metadata(metadata)
            gated.append(
                record.model_copy(
                    update={
                        "transcript": None,
                        "normalized_transcript": None,
                        "metadata": metadata,
                    }
                )
            )
        else:
            gated.append(record.model_copy(update={"metadata": metadata}))
    write_jsonl(output_path, gated)
    return gated


def calibrate_speaker_thresholds(
    records: Sequence[ClipRecord],
    *,
    embedder: SpeakerEmbedder,
    source_name: str,
    max_speakers: int = 300,
) -> tuple[SpeakerThresholds, dict[str, object]]:
    by_speaker: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in records:
        if record.speaker_id and not _quality_reasons(record):
            by_speaker[record.speaker_id].append(record)
    eligible = [
        (speaker_id, sorted(values, key=lambda item: item.clip_id)[:3])
        for speaker_id, values in sorted(by_speaker.items())
        if len(values) >= 2
    ]
    selected = sorted(eligible, key=lambda item: _stable_hash(item[0]))[:max_speakers]
    sample = [record for _, values in selected for record in values]
    embeddings = embedder.embed(sample)
    within: list[float] = []
    centroids: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for _, values in selected:
        vectors = [embeddings[record.clip_id] for record in values]
        centroid = _normalized_mean(vectors)
        centroids.append(centroid)
        within.extend(float(np.dot(vector, centroid)) for vector in vectors)
    between = [
        float(np.dot(centroids[index], centroids[(index + 1) % len(centroids)])) for index in range(len(centroids))
    ]
    if len(within) < 20 or len(centroids) < 10:
        raise ValueError("insufficient speaker calibration data")
    threshold_value = max(0.35, min(0.90, float(np.percentile(within, 1)) - 0.03))
    calibration_id = hashlib.sha256(
        (SPEAKER_GATE_VERSION + ":" + ":".join(speaker_id for speaker_id, _ in selected)).encode()
    ).hexdigest()[:20]
    thresholds = SpeakerThresholds(
        calibration_id=calibration_id,
        calibration_source=source_name,
        calibration_speakers=len(selected),
        calibration_pairs=len(within),
        min_parent_cosine_similarity=threshold_value,
    )
    report: dict[str, object] = {
        "speaker_gate_version": SPEAKER_GATE_VERSION,
        "model": {
            "id": SPEAKER_MODEL_ID,
            "revision": SPEAKER_MODEL_REVISION,
            "license": SPEAKER_MODEL_LICENSE,
        },
        "source": source_name,
        "speaker_count": len(selected),
        "within_speaker_similarity": _distribution(within),
        "between_speaker_similarity": _distribution(between),
        "thresholds": thresholds.model_dump(mode="json"),
    }
    return thresholds, report


def run_speaker_consistency_gate(
    records: Sequence[ClipRecord],
    *,
    embedder: SpeakerEmbedder,
    thresholds: SpeakerThresholds,
    output_path: Path,
) -> list[ClipRecord]:
    candidates = [record for record in records if not _quality_reasons(record)]
    embeddings = embedder.embed(candidates)
    by_parent: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in candidates:
        parent_id = str(record.metadata.get("parent_clip_id") or record.clip_id)
        by_parent[parent_id].append(record)
    similarity_by_id: dict[str, float | None] = {}
    for values in by_parent.values():
        if len(values) == 1:
            similarity_by_id[values[0].clip_id] = None
            continue
        centroid = _normalized_mean([embeddings[record.clip_id] for record in values])
        for record in values:
            similarity_by_id[record.clip_id] = float(np.dot(embeddings[record.clip_id], centroid))
    gated: list[ClipRecord] = []
    for record in records:
        if record.clip_id not in similarity_by_id:
            gated.append(record)
            continue
        similarity = similarity_by_id[record.clip_id]
        reasons = list(_quality_reasons(record))
        if similarity is not None and similarity < thresholds.min_parent_cosine_similarity:
            reasons.append("speaker_inconsistent_with_parent")
        gated.append(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "speaker_consistency": {
                            "model": SPEAKER_MODEL_ID,
                            "revision": SPEAKER_MODEL_REVISION,
                            "license": SPEAKER_MODEL_LICENSE,
                            "parent_cosine_similarity": None if similarity is None else round(similarity, 6),
                            "minimum_similarity": thresholds.min_parent_cosine_similarity,
                            "calibration_id": thresholds.calibration_id,
                            "assessed": similarity is not None,
                        },
                        "quality_filter_reasons": sorted(set(reasons)),
                        "quality_stage": "speaker_consistency",
                    }
                }
            )
        )
    write_jsonl(output_path, gated)
    return gated


def run_acoustic_fingerprint_gate(
    records: Sequence[ClipRecord],
    *,
    output_path: Path,
) -> list[ClipRecord]:
    first_by_fingerprint: dict[str, str] = {}
    gated: list[ClipRecord] = []
    for record in records:
        if _quality_reasons(record):
            gated.append(record)
            continue
        fingerprint = _acoustic_fingerprint(Path(record.audio_path))
        first_clip_id = first_by_fingerprint.setdefault(fingerprint, record.clip_id)
        reasons = list(_quality_reasons(record))
        if first_clip_id != record.clip_id:
            reasons.append("duplicate_acoustic_fingerprint")
        gated.append(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "acoustic_fingerprint": fingerprint,
                        "duplicate_of_clip_id": first_clip_id if first_clip_id != record.clip_id else None,
                        "quality_filter_reasons": sorted(set(reasons)),
                        "quality_stage": "duplicate_gate",
                    }
                }
            )
        )
    write_jsonl(output_path, gated)
    return gated


def quality_gate_summary(records: Sequence[ClipRecord]) -> dict[str, object]:
    accepted = [record for record in records if not _quality_reasons(record)]
    candidate = [record for record in accepted if record.metadata.get("candidate_a") is True]
    rejection_reasons = Counter(reason for record in records for reason in _quality_reasons(record))
    privacy_findings: Counter[str] = Counter()
    speaker_similarities: list[float] = []
    for record in records:
        privacy = record.metadata.get("privacy_scan")
        if isinstance(privacy, dict):
            findings = privacy.get("findings")
            if isinstance(findings, list):
                privacy_findings.update(finding for finding in findings if isinstance(finding, str))
        consistency = record.metadata.get("speaker_consistency")
        if isinstance(consistency, dict):
            similarity = consistency.get("parent_cosine_similarity")
            if isinstance(similarity, (int, float)):
                speaker_similarities.append(float(similarity))
    return {
        "records": len(records),
        "accepted_records": len(accepted),
        "accepted_hours": round(sum(record.duration_seconds or 0 for record in accepted) / 3600, 3),
        "rejected_records": len(records) - len(accepted),
        "rejected_hours": round(
            sum(record.duration_seconds or 0 for record in records if _quality_reasons(record)) / 3600,
            3,
        ),
        "candidate_a": {
            "records": len(candidate),
            "hours": round(sum(record.duration_seconds or 0 for record in candidate) / 3600, 3),
        },
        "accepted_by_format": {
            collection_format: {
                "records": len(selected),
                "hours": round(sum(record.duration_seconds or 0 for record in selected) / 3600, 3),
            }
            for collection_format in sorted({record.collection_format.value for record in accepted})
            if (selected := [record for record in accepted if record.collection_format.value == collection_format])
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "privacy_findings": dict(sorted(privacy_findings.items())),
        "speaker_similarity": _distribution(speaker_similarities) if speaker_similarities else None,
        "accepted_missing_evidence": sum(
            not all(
                record.metadata.get(key)
                for key in (
                    "privacy_scan",
                    "dual_asr",
                    "forced_alignment",
                    "speaker_consistency",
                    "acoustic_fingerprint",
                )
            )
            for record in accepted
        ),
        "versions": {
            "privacy_gate": PRIVACY_GATE_VERSION,
            "speaker_gate": SPEAKER_GATE_VERSION,
            "speaker_model": {"id": SPEAKER_MODEL_ID, "revision": SPEAKER_MODEL_REVISION},
            "ner_model": {"id": NER_MODEL_ID, "revision": NER_MODEL_REVISION},
            "ner_resources_commit": NER_RESOURCES_COMMIT,
        },
    }


def write_speaker_config(
    path: Path,
    *,
    thresholds: SpeakerThresholds,
    report: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"thresholds": thresholds.model_dump(mode="json"), "calibration": report},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_speaker_thresholds(path: Path) -> SpeakerThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds") if isinstance(payload, dict) else None
    return SpeakerThresholds.model_validate(thresholds)


def _prepare_stanza_runtime(snapshot: Path) -> Path:
    import httpx

    model_root = Path.home() / ".cache" / "turkish_tts" / f"stanza-tr-{NER_MODEL_REVISION}"
    language_root = model_root / "tr"
    tokenizer_path = language_root / "tokenize" / "imst.pt"
    ner_path = language_root / "ner" / "starlang.pt"
    if not tokenizer_path.is_file() or not ner_path.is_file():
        language_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(snapshot / "models" / "default.zip") as archive:
            archive.extractall(language_root)
    resources_path = model_root / "resources.json"
    if not resources_path.is_file():
        response = httpx.get(NER_RESOURCES_URL, timeout=60)
        response.raise_for_status()
        content = response.content
        if hashlib.sha256(content).hexdigest() != NER_RESOURCES_SHA256:
            raise ValueError("Stanza resources metadata checksum mismatch")
        temporary_path = resources_path.with_suffix(".json.part")
        temporary_path.write_bytes(content)
        temporary_path.replace(resources_path)
    return model_root


def _regex_privacy_findings(text: str) -> list[PrivacyFinding]:
    checks = (
        ("email", _EMAIL),
        ("url", _URL),
        ("iban", _IBAN),
        ("phone_number", _PHONE),
        ("social_handle", _SOCIAL_HANDLE),
        ("street_address", _ADDRESS),
    )
    findings = [
        PrivacyFinding(category=category, detector="regex") for category, pattern in checks if pattern.search(text)
    ]
    if any(_valid_tc_identity(match[0]) for match in _TC_IDENTITY.finditer(text)):
        findings.append(PrivacyFinding(category="turkish_identity_number", detector="checksum"))
    return findings


def structured_pii_categories(text: str) -> tuple[str, ...]:
    return tuple(sorted({finding.category for finding in _regex_privacy_findings(text)}))


def _valid_tc_identity(value: str) -> bool:
    digits = [int(character) for character in value]
    if len(digits) != 11 or digits[0] == 0:
        return False
    tenth = ((sum(digits[0:9:2]) * 7) - sum(digits[1:8:2])) % 10
    eleventh = sum(digits[:10]) % 10
    return digits[9] == tenth and digits[10] == eleventh


def _redact_text_metadata(metadata: dict[str, object]) -> dict[str, object]:
    redacted = dict(metadata)
    for key in ("whisper_asr", "ctc_asr"):
        value = redacted.get(key)
        if isinstance(value, dict):
            redacted[key] = {field: field_value for field, field_value in value.items() if field != "text"}
    alignment = redacted.get("forced_alignment")
    if isinstance(alignment, dict):
        redacted["forced_alignment"] = {
            field: value
            for field, value in alignment.items()
            if field not in {"target", "word_timings", "phoneme_timings"}
        }
    return redacted


def _normalized_mean(vectors: Sequence[np.ndarray[Any, np.dtype[np.float32]]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    centroid = np.asarray(np.mean(np.stack(vectors), axis=0), dtype=np.float32)
    return np.asarray(centroid / max(float(np.linalg.norm(centroid)), 1e-8), dtype=np.float32)


def _acoustic_fingerprint(path: Path) -> str:
    import torch
    import torchaudio

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(np.mean(audio, axis=1, dtype=np.float32))
    if sample_rate != 8_000:
        mono = torchaudio.functional.resample(mono, sample_rate, 8_000)
    mono = mono / max(float(torch.max(torch.abs(mono))), 1e-6)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=8_000,
        n_fft=512,
        hop_length=160,
        n_mels=32,
    )(mono)
    log_mel = torch.log1p(mel).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(log_mel, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    centered = resized - torch.mean(resized)
    bits = np.packbits((centered.numpy() >= 0).astype(np.uint8))
    return hashlib.sha256(bits.tobytes()).hexdigest()


def _quality_reasons(record: ClipRecord) -> tuple[str, ...]:
    reasons = record.metadata.get("quality_filter_reasons")
    if not isinstance(reasons, list):
        return ()
    return tuple(reason for reason in reasons if isinstance(reason, str))


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }

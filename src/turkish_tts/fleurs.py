from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset  # type: ignore[import-untyped]
from huggingface_hub import HfApi

from turkish_tts.manifests import ClipRecord, CollectionFormat, RightsState
from turkish_tts.normalize import normalize_orthography

FLEURS_DATASET_ID = "google/fleurs"
FLEURS_CONFIG = "tr_tr"
FLEURS_LICENSE_ID = "CC-BY-4.0"
FLEURS_SAMPLE_RATE_HZ = 16_000
FLEURS_SPLITS = ("train", "validation", "test")
JsonRow = Mapping[str, Any]


def resolve_fleurs_revision(*, token: str | None) -> str:
    info = HfApi(token=token).dataset_info(FLEURS_DATASET_ID)
    if not info.sha:
        raise ValueError("Hugging Face did not return an immutable FLEURS revision")
    return info.sha


def iter_fleurs_turkish(
    *,
    output_dir: Path,
    revision: str,
    token: str | None,
    splits: Iterable[str] = FLEURS_SPLITS,
) -> Iterator[ClipRecord]:
    for split in splits:
        if split not in FLEURS_SPLITS:
            raise ValueError(f"unsupported FLEURS split: {split}")
        dataset = load_dataset(
            FLEURS_DATASET_ID,
            FLEURS_CONFIG,
            split=split,
            revision=revision,
            token=token,
        ).cast_column("audio", Audio(decode=False))
        yield from materialize_fleurs_rows(
            dataset,
            split=split,
            output_dir=output_dir,
            revision=revision,
        )


def materialize_fleurs_rows(
    rows: Iterable[JsonRow],
    *,
    split: str,
    output_dir: Path,
    revision: str,
) -> Iterator[ClipRecord]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        fleurs_id = row.get("id")
        source_row_id = f"{split}:{index}"
        clip_digest = hashlib.sha256(
            f"fleurs:{revision}:{FLEURS_CONFIG}:{source_row_id}:{fleurs_id}".encode()
        ).hexdigest()
        audio_payload = row.get("audio")
        if not isinstance(audio_payload, Mapping):
            raise ValueError(f"FLEURS {split}/{source_row_id} has no audio payload")
        source_path = audio_payload.get("path")
        extension = _audio_extension(source_path)
        audio_path = split_dir / f"fleurs-tr-{clip_digest[:24]}{extension}"
        _materialize_audio(audio_payload, audio_path)

        transcript = normalize_orthography(str(row.get("transcription") or ""))
        if not transcript:
            raise ValueError(f"FLEURS {split}/{source_row_id} has an empty transcript")
        num_samples = _positive_int(row.get("num_samples"))
        metadata: dict[str, object] = {
            "dataset_id": FLEURS_DATASET_ID,
            "dataset_config": FLEURS_CONFIG,
            "attribution": "Google FLEURS dataset authors and source speakers",
        }
        if fleurs_id is not None:
            metadata["fleurs_id"] = fleurs_id
        for key in ("raw_transcription", "gender", "language", "lang_id", "lang_group_id"):
            value = row.get(key)
            if value is not None and value != "":
                metadata[key] = value
        if isinstance(source_path, str) and source_path:
            metadata["source_file_name"] = Path(source_path).name

        yield ClipRecord(
            clip_id=f"fleurs-tr-{clip_digest[:24]}",
            source_dataset="google-fleurs",
            source_version=revision,
            source_split=split,
            source_row_id=source_row_id,
            audio_path=str(audio_path.resolve()),
            transcript=transcript,
            normalized_transcript=transcript,
            language="tr",
            collection_format=CollectionFormat.SCRIPTED,
            rights_state=RightsState.ALLOWED,
            license_id=FLEURS_LICENSE_ID,
            duration_seconds=num_samples / FLEURS_SAMPLE_RATE_HZ if num_samples else None,
            sample_rate_hz=FLEURS_SAMPLE_RATE_HZ,
            sha256=_sha256_file(audio_path),
            metadata=metadata,
        )


def _materialize_audio(payload: Mapping[str, Any], destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    temporary_path.unlink(missing_ok=True)
    audio_bytes = payload.get("bytes")
    if isinstance(audio_bytes, (bytes, bytearray, memoryview)):
        temporary_path.write_bytes(bytes(audio_bytes))
    else:
        source_path = payload.get("path")
        if not isinstance(source_path, str) or not Path(source_path).is_file():
            raise ValueError(f"FLEURS audio payload is missing bytes and a local path for {destination.name}")
        shutil.copyfile(source_path, temporary_path)
    if temporary_path.stat().st_size == 0:
        raise ValueError(f"FLEURS audio payload is empty for {destination.name}")
    temporary_path.replace(destination)


def _audio_extension(source_path: object) -> str:
    if isinstance(source_path, str):
        suffix = Path(source_path).suffix.lower()
        if suffix in {".wav", ".flac", ".mp3", ".ogg", ".opus"}:
            return suffix
    return ".wav"


def _positive_int(value: object) -> int | None:
    if not isinstance(value, (int, float, str)):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from turkish_tts.common_voice_prepare import (
    COMMON_VOICE_DATASET_TERMS_URL,
    COMMON_VOICE_LICENSE_ID,
    COMMON_VOICE_LICENSE_URL,
    COMMON_VOICE_RESTRICTIONS,
)
from turkish_tts.manifests import ClipRecord, CollectionFormat, RightsState
from turkish_tts.normalize import normalize_orthography


def _anonymous_speaker_id(version: str, client_id: str) -> str:
    digest = hashlib.sha256(f"common-voice:{version}:{client_id}".encode()).hexdigest()
    return f"cv-{digest[:20]}"


def iter_common_voice_validated(
    *,
    tsv_path: Path,
    clips_dir: Path,
    version: str,
    require_audio: bool = True,
) -> Iterator[ClipRecord]:
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"client_id", "path", "sentence"}
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"missing Common Voice columns: {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            relative_path = row["path"].strip()
            audio_path = clips_dir / relative_path
            if require_audio and not audio_path.is_file():
                raise FileNotFoundError(f"missing Common Voice clip at row {row_number}: {audio_path}")

            client_id = row["client_id"].strip()
            sentence = normalize_orthography(row["sentence"])
            row_key = row.get("sentence_id", "").strip() or relative_path
            clip_digest = hashlib.sha256(f"cv26:tr:{row_key}:{relative_path}".encode()).hexdigest()

            metadata: dict[str, object] = {
                "up_votes": _optional_int(row.get("up_votes")),
                "down_votes": _optional_int(row.get("down_votes")),
                "license_url": COMMON_VOICE_LICENSE_URL,
                "dataset_terms_url": COMMON_VOICE_DATASET_TERMS_URL,
                "dataset_restrictions": list(COMMON_VOICE_RESTRICTIONS),
                "attribution": "Mozilla Common Voice contributors",
            }
            for key in ("age", "gender", "accent", "accents", "variant", "segment", "sentence_domain"):
                value = (row.get(key) or "").strip()
                if value:
                    metadata[key] = value

            yield ClipRecord(
                clip_id=f"cv26-tr-{clip_digest[:24]}",
                source_dataset="mozilla-common-voice-scripted-speech",
                source_version=version,
                source_split="validated",
                source_row_id=row_key,
                audio_path=str(audio_path.resolve()),
                transcript=sentence,
                normalized_transcript=sentence,
                language="tr",
                speaker_id=_anonymous_speaker_id(version, client_id),
                collection_format=CollectionFormat.SCRIPTED,
                rights_state=RightsState.ALLOWED,
                license_id=COMMON_VOICE_LICENSE_ID,
                metadata=metadata,
            )


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator


def write_export_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


class RightsState(StrEnum):
    ALLOWED = "allowed"
    RESEARCH_ONLY = "research_only"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class CollectionFormat(StrEnum):
    SCRIPTED = "scripted"
    MONOLOGUE = "monologue"
    CONVERSATION = "conversation"


class ClipRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_split: str | None = None
    source_row_id: str | None = None
    audio_path: str = Field(min_length=1)
    transcript: str | None = None
    normalized_transcript: str | None = None
    language: str = "tr"
    speaker_id: str | None = None
    session_id: str | None = None
    collection_format: CollectionFormat
    rights_state: RightsState
    license_id: str = Field(min_length=1)
    duration_seconds: float | None = Field(default=None, ge=0)
    sample_rate_hz: int | None = Field(default=None, gt=0)
    sha256: str | None = None
    transcription_model: str | None = None
    transcription_language: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("language")
    @classmethod
    def require_turkish(cls, value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in {"tr", "tr-tr", "tur"}:
            raise ValueError(f"expected Turkish language code, got {value!r}")
        return "tr"


def iter_jsonl(path: Path) -> Iterator[ClipRecord]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield ClipRecord.model_validate(orjson.loads(line))
            except Exception as error:
                raise ValueError(f"invalid manifest row at {path}:{line_number}") from error


def write_jsonl(path: Path, records: Iterable[ClipRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE))
            count += 1
    return count

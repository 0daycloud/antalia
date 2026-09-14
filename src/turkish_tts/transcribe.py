from __future__ import annotations

import asyncio
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from turkish_tts.manifests import ClipRecord
from turkish_tts.normalize import normalize_orthography


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    index: int
    record: ClipRecord


async def transcribe_records(
    records: list[ClipRecord],
    *,
    api_key: str,
    model: str = "gpt-transcribe",
    language: str = "tr",
    max_concurrency: int = 8,
    overwrite: bool = False,
) -> list[ClipRecord]:
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(index: int, record: ClipRecord) -> TranscriptionResult:
        if record.transcript and not overwrite:
            return TranscriptionResult(index=index, record=record)
        async with semaphore:
            text, detected_language = await _transcribe_one(client, Path(record.audio_path), model, language)
        updated = record.model_copy(
            update={
                "transcript": text,
                "normalized_transcript": normalize_orthography(text),
                "transcription_model": model,
                "transcription_language": detected_language or language,
            }
        )
        return TranscriptionResult(index=index, record=updated)

    tasks = [asyncio.create_task(run(index, record)) for index, record in enumerate(records)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda item: item.index)
    return [item.record for item in results]


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=30),
    reraise=True,
)
async def _transcribe_one(client: AsyncOpenAI, path: Path, model: str, language: str) -> tuple[str, str | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as handle:
        response = await client.audio.transcriptions.create(
            model=model,
            file=(path.name, handle, mime_type),
            extra_body={"languages": [language]},
        )
    text = normalize_orthography(response.text)
    languages = getattr(response, "languages", None) or []
    detected = getattr(languages[0], "code", None) if languages else None
    if not text:
        raise ValueError(f"empty transcription for {path}")
    return text, detected

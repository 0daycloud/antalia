from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    artifact_uri: str = Field(default="", alias="TURKISH_TTS_ARTIFACT_URI")
    transcription_model: str = Field(default="gpt-transcribe", alias="TURKISH_TTS_TRANSCRIPTION_MODEL")
    language: str = Field(default="tr", alias="TURKISH_TTS_LANGUAGE")
    max_concurrency: int = Field(default=8, ge=1, le=64, alias="TURKISH_TTS_MAX_CONCURRENCY")
    speaker_hmac_key: str | None = Field(default=None, alias="TURKISH_TTS_SPEAKER_HMAC_KEY")

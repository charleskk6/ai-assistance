"""Application settings.

Everything is overridable with environment variables prefixed ``COACH_`` (or a
``.env`` file in the project root). Model profiles and voices live in
``config/models.toml`` so they can be edited without touching code.

API keys for the optional cloud providers are read from the standard,
unprefixed ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` variables and are never
written anywhere by this application.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMBackendName = Literal["mlx", "ollama", "openai_compat", "mock"]
STTBackendName = Literal["faster_whisper", "whispercpp", "mock"]
TTSBackendName = Literal["piper", "macos_say", "mock"]
VADBackendName = Literal["silero", "energy"]
BargeInMode = Literal["headphones", "speakers", "off"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COACH_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "info"

    # --- LLM ----------------------------------------------------------------
    llm_backend: LLMBackendName = "mlx"
    llm_profile: Literal["small", "medium", "large"] = "medium"
    llm_model: str = ""  # overrides the profile's model id for the chosen backend
    llm_max_tokens: int = 110  # ~ three short spoken sentences
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    mlx_prompt_cache: bool = True  # reuse the KV cache for the shared prompt prefix
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_keep_alive: str = "30m"
    openai_compat_url: str = "http://127.0.0.1:8080/v1"  # llama.cpp `llama-server`
    context_max_messages: int = 14
    context_char_budget: int = 6000  # ~1.5k tokens of history

    # --- STT ------------------------------------------------------------------
    stt_backend: STTBackendName = "faster_whisper"
    stt_model: str = "base.en"
    stt_threads: int = 4
    stt_partials: bool = True
    stt_partial_interval_ms: int = 700
    stt_speculative_ms: int = 240  # start the final transcription after this much silence

    # --- VAD / turn taking -------------------------------------------------------
    vad_backend: VADBackendName = "silero"
    vad_model_path: str = "models/vad/silero_vad.onnx"
    vad_threshold: float = 0.5
    min_speech_ms: int = 160
    end_of_turn_ms: int = 560
    max_turn_s: float = 30.0
    pre_roll_ms: int = 320

    # --- barge-in -----------------------------------------------------------------
    barge_in_mode: BargeInMode = "headphones"
    barge_in_threshold_headphones: float = 0.55
    barge_in_min_ms_headphones: int = 200
    barge_in_threshold_speakers: float = 0.8
    barge_in_min_ms_speakers: int = 380
    echo_similarity_threshold: float = 0.55

    # --- TTS --------------------------------------------------------------------------
    tts_backend: TTSBackendName = "piper"
    tts_voice: str = "en_GB-cori-medium"
    tts_length_scale: float = 1.0
    tts_first_chunk_min_chars: int = 40  # flush the first clause early to cut time-to-audio

    # --- conversation -------------------------------------------------------------------
    default_mode: str = "free"
    idle_prompt_s: float = 14.0  # proactively re-engage after this much user silence

    # --- cloud (disabled by default) ------------------------------------------------------
    cloud_enabled: bool = False
    cloud_provider: Literal["anthropic", "openai"] = "anthropic"
    cloud_allow_review: bool = True
    cloud_allow_turn_corrections: bool = False
    cloud_escalation: Literal["explicit_only", "local_failure"] = "explicit_only"
    anthropic_model: str = "claude-opus-5-5"
    openai_model: str = "gpt-5-mini"
    cloud_timeout_s: float = 60.0

    # --- storage --------------------------------------------------------------------------
    save_history: bool = True
    db_path: str = "data/coach.sqlite3"
    models_dir: str = "models"

    # --- paths ----------------------------------------------------------------------------
    frontend_dist: str = "frontend/dist"
    models_toml: str = "config/models.toml"

    # Not settable via COACH_ prefix: read from the conventional variable names.
    anthropic_api_key: str | None = Field(default=None, exclude=True)
    openai_api_key: str | None = Field(default=None, exclude=True)

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or None
        self.openai_api_key = os.environ.get("OPENAI_API_KEY") or None

    def path(self, value: str) -> Path:
        """Resolve a project-relative path setting."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    def public_dict(self) -> dict:
        """Settings safe to show in the UI (never includes credentials)."""
        d = self.model_dump(exclude={"anthropic_api_key", "openai_api_key"})
        d["anthropic_key_present"] = bool(self.anthropic_api_key)
        d["openai_key_present"] = bool(self.openai_api_key)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_catalog(path: str | None = None) -> dict:
    """Model profiles, voices and their licences from ``config/models.toml``."""
    p = Path(path) if path else get_settings().path(get_settings().models_toml)
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def resolve_llm_model(settings: Settings, backend: str | None = None, profile: str | None = None) -> str:
    backend = backend or settings.llm_backend
    if settings.llm_model and backend == settings.llm_backend and profile is None:
        return settings.llm_model
    profile = profile or settings.llm_profile
    if backend == "mock":
        return "mock-coach"
    cat = load_catalog()
    return cat["llm"]["profiles"][profile][backend]

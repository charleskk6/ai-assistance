"""Central configuration, loaded from environment / .env.

Everything the assistant needs to be re-pointed at a different runtime, model or
search provider lives here. Nothing else in the codebase reads os.environ.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the project, not the process working directory: a relative ".env"
# is silently ignored whenever the server is started from another directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # --- server -------------------------------------------------------------
    host: str = "0.0.0.0"          # LAN-reachable so the iPhone can call in
    port: int = 8000
    log_level: str = "INFO"

    # --- auth ---------------------------------------------------------------
    # Required. The server refuses to start without it (see main.py).
    local_assistant_token: str = ""

    # --- local LLM ----------------------------------------------------------
    llm_provider: Literal["ollama", "mlx", "echo"] = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.4
    # Qwen3 is a hybrid reasoning model; thinking blocks cost many seconds of
    # latency for a spoken answer, so it is off by default.
    llm_enable_thinking: bool = False
    # How long Ollama keeps the weights resident after a request. Long enough to
    # avoid a cold reload between back-to-back Siri questions, short enough that
    # an idle laptop gets its RAM back.
    llm_keep_alive: str = "10m"
    # Ollama allocates a KV cache for the whole window; 4096 comfortably holds
    # the trimmed evidence budget below plus the answer.
    llm_num_ctx: int = 4096
    llm_timeout_s: float = 120.0
    # Spoken answers must stay short; API/CLI callers may want more.
    llm_max_tokens_siri: int = 400
    llm_max_tokens_default: int = 900

    # --- router -------------------------------------------------------------
    router_default_mode: Literal["auto", "local", "web"] = "auto"

    # --- search -------------------------------------------------------------
    search_provider: Literal["duckduckgo", "brave", "tavily", "searxng"] = "duckduckgo"
    search_results: int = 8          # how many results to ask the provider for
    search_timeout_s: float = 10.0
    brave_api_key: str = ""
    tavily_api_key: str = ""
    searxng_url: str = "http://127.0.0.1:8080"

    # --- retrieval ----------------------------------------------------------
    fetch_max_pages: int = 4         # pages actually downloaded per query
    fetch_timeout_s: float = 8.0
    fetch_max_bytes: int = 2_000_000  # hard cap; oversized responses are dropped
    fetch_concurrency: int = 4        # bounded, this is a fanless laptop
    fetch_user_agent: str = "LocalAssistant/0.1 (personal use)"
    # Off by default: search results are attacker-influenceable, so fetching a
    # private address would let a poisoned result probe your LAN. Turn it on only
    # to point the assistant at an intranet you trust.
    fetch_allow_private_urls: bool = False

    # --- RAG ----------------------------------------------------------------
    # Measured on an M3 Air: prefill, not generation, dominates the web route -
    # 6200 characters of evidence cost ~17s of the 19s the model took, for an
    # answer of about 45 tokens. Smaller chunks also retrieve more precisely,
    # since a chunk is either relevant or it is not.
    chunk_chars: int = 800
    chunk_overlap_chars: int = 120
    rag_top_chunks: int = 5
    rag_context_budget_chars: int = 3000
    rag_min_evidence_chars: int = 300  # below this we admit we found nothing

    # --- end-to-end ---------------------------------------------------------
    request_timeout_s: float = 150.0
    # A fanless Air should not be running four inferences at once; queue instead.
    max_concurrent_requests: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()

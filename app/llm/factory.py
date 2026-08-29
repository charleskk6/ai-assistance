from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.local import EchoProvider, OllamaProvider


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model=settings.llm_model,
            host=settings.ollama_host,
            keep_alive=settings.llm_keep_alive,
            num_ctx=settings.llm_num_ctx,
            timeout_s=settings.llm_timeout_s,
            enable_thinking=settings.llm_enable_thinking,
        )
    if settings.llm_provider == "echo":
        return EchoProvider(settings.llm_model)
    # "mlx" is reserved: drop in an MlxProvider subclass and wire it here.
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

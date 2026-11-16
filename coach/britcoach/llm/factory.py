"""Create LLM backends by name. Imports are lazy so non-Mac machines never import MLX."""

from __future__ import annotations

import platform

from ..config import Settings, resolve_llm_model
from .base import LLMBackend


def mlx_supported() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def create_llm(settings: Settings, backend: str | None = None, model: str | None = None,
               profile: str | None = None) -> LLMBackend:
    backend = backend or settings.llm_backend
    model = model or resolve_llm_model(settings, backend, profile)
    if backend == "mlx":
        if not mlx_supported():
            raise RuntimeError("The MLX backend needs macOS on Apple Silicon. Use ollama, openai_compat or mock.")
        from .mlx_backend import MLXBackend

        return MLXBackend(model, prompt_cache=settings.mlx_prompt_cache)
    if backend == "ollama":
        from .http_backends import OllamaBackend

        return OllamaBackend(model, settings.ollama_url, settings.ollama_keep_alive)
    if backend == "openai_compat":
        from .http_backends import OpenAICompatBackend

        return OpenAICompatBackend(model, settings.openai_compat_url)
    if backend == "mock":
        from .mock import MockLLM

        return MockLLM()
    raise ValueError(f"Unknown LLM backend: {backend}")

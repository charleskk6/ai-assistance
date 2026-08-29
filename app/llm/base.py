"""Provider-agnostic interface for the local LLM.

Deliberately tiny: two calls. Swapping Ollama for MLX-LM or llama.cpp means
writing one new subclass, not touching the pipeline.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """Raised when the local model could not produce an answer."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Completion:
    text: str
    # True when the runtime stopped because it hit the token ceiling, rather
    # than because the model finished its sentence. Guessing this from the text
    # is unreliable - plenty of complete answers end without punctuation.
    truncated: bool = False


@dataclass
class LLMHealth:
    runtime_ok: bool
    model_ok: bool
    provider: str
    model: str
    detail: str = ""
    available_models: list[str] = field(default_factory=list)


class LLMProvider(abc.ABC):
    name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    async def complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> Completion:
        """Return the assistant's plain-text reply."""

    @abc.abstractmethod
    async def health(self) -> LLMHealth:
        """Check the runtime is up and the configured model is present."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None

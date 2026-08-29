"""Local LLM providers.

`OllamaProvider` is the Phase 1 default. `EchoProvider` exists so the API and RAG
pipeline can be tested without a model on the machine.
"""

from __future__ import annotations

import re

import httpx

from app.llm.base import LLMError, LLMHealth, LLMProvider

# Qwen3 and friends can emit a reasoning block. We ask them not to, but strip it
# defensively so it can never reach Apple's text-to-speech.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK_RE = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    # An unterminated block means the model ran out of tokens while thinking.
    if _ORPHAN_THINK_RE.match(text):
        return ""
    return text.strip()


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str,
        *,
        keep_alive: str = "10m",
        num_ctx: int = 8192,
        timeout_s: float = 120.0,
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(model)
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.enable_thinking = enable_thinking
        self._client = httpx.AsyncClient(
            base_url=self.host, timeout=httpx.Timeout(timeout_s, connect=5.0)
        )

    def _system_prompt(self, system: str) -> str:
        # Qwen3's documented soft switch for its hybrid reasoning mode. Harmless
        # for models that do not implement it.
        if not self.enable_thinking and self.model.lower().startswith("qwen3"):
            return f"{system}\n\n/no_think"
        return system

    async def complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt(system)},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.enable_thinking,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError("llm_timeout", "The local model took too long.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "llm_unavailable", f"Could not reach the local model runtime: {exc}"
            ) from exc

        if resp.status_code == 404:
            raise LLMError(
                "model_not_found",
                f"Model '{self.model}' is not installed. Run: ollama pull {self.model}",
            )
        if resp.status_code >= 400:
            raise LLMError(
                "llm_error", f"Local model runtime returned {resp.status_code}."
            )

        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        answer = strip_thinking(content)
        if not answer:
            raise LLMError("llm_empty", "The local model returned an empty answer.")
        return answer

    async def health(self) -> LLMHealth:
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return LLMHealth(
                runtime_ok=False,
                model_ok=False,
                provider=self.name,
                model=self.model,
                detail=f"Ollama unreachable at {self.host}: {exc}",
            )

        names = [m.get("name", "") for m in resp.json().get("models", [])]
        # Ollama reports "qwen3:8b"; a configured bare "qwen3" should match too.
        model_ok = any(
            n == self.model or n.split(":")[0] == self.model.split(":")[0]
            for n in names
        )
        return LLMHealth(
            runtime_ok=True,
            model_ok=model_ok,
            provider=self.name,
            model=self.model,
            detail="" if model_ok else f"Run: ollama pull {self.model}",
            available_models=names,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class EchoProvider(LLMProvider):
    """Deterministic stand-in used by tests and by `LLM_PROVIDER=echo`."""

    name = "echo"

    def __init__(self, model: str = "echo") -> None:
        super().__init__(model)

    async def complete(
        self, system: str, user: str, *, max_tokens: int, temperature: float
    ) -> str:
        return f"[echo:{self.model}] {user.strip()[:max_tokens]}"

    async def health(self) -> LLMHealth:
        return LLMHealth(
            runtime_ok=True, model_ok=True, provider=self.name, model=self.model
        )

"""HTTP LLM backends: Ollama and any OpenAI-compatible server (llama.cpp ``llama-server``,
LM Studio, MLX's own ``mlx_lm.server``). Closing the HTTP stream aborts generation server-side.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from .base import CancelToken, GenParams, LLMBackend, LLMChunk, Message


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model_id: str, base_url: str, keep_alive: str = "30m",
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(120, connect=5),
                                         transport=transport)

    async def load(self) -> None:
        # An empty chat request loads the model and keeps it resident for keep_alive.
        r = await self._client.post("/api/chat", json={"model": self.model_id, "messages": [], "keep_alive": self.keep_alive})
        r.raise_for_status()

    async def unload(self) -> None:
        try:
            await self._client.post("/api/chat", json={"model": self.model_id, "messages": [], "keep_alive": 0})
        except httpx.HTTPError:
            pass

    async def stream(self, messages: list[Message], params: GenParams, cancel: CancelToken) -> AsyncIterator[LLMChunk]:
        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": params.max_tokens, "temperature": params.temperature, "top_p": params.top_p},
        }
        t0 = time.perf_counter()
        first = None
        n = 0
        final: dict = {}
        async with self._client.stream("POST", "/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if cancel.cancelled:
                    break  # leaving the context manager closes the connection
                if not line.strip():
                    continue
                data = json.loads(line)
                piece = data.get("message", {}).get("content", "")
                if piece:
                    first = first or time.perf_counter()
                    n += 1
                    yield LLMChunk(text=piece)
                if data.get("done"):
                    final = data
                    break
        eval_ns = final.get("eval_duration") or 0
        stats = {
            "prompt_tokens": final.get("prompt_eval_count"),
            "generated_tokens": final.get("eval_count", n),
            "ttft_s": (first - t0) if first else None,
            "tokens_per_s": (final.get("eval_count", 0) / (eval_ns / 1e9)) if eval_ns else None,
            "cancelled": cancel.cancelled,
        }
        yield LLMChunk(text="", done=True, stats=stats)


class OpenAICompatBackend(LLMBackend):
    name = "openai_compat"

    def __init__(self, model_id: str, base_url: str, api_key: str = "local",
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(120, connect=5),
                                         headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    async def stream(self, messages: list[Message], params: GenParams, cancel: CancelToken) -> AsyncIterator[LLMChunk]:
        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        t0 = time.perf_counter()
        first = None
        n = 0
        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if cancel.cancelled:
                    break
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                choices = data.get("choices") or []
                piece = (choices[0].get("delta", {}).get("content") or "") if choices else ""
                if piece:
                    first = first or time.perf_counter()
                    n += 1
                    yield LLMChunk(text=piece)
        dt = time.perf_counter() - (first or t0)
        stats = {
            "generated_tokens": n,  # chunks ~ tokens for llama-server
            "ttft_s": (first - t0) if first else None,
            "tokens_per_s": (n / dt) if first and dt > 0 else None,
            "cancelled": cancel.cancelled,
        }
        yield LLMChunk(text="", done=True, stats=stats)

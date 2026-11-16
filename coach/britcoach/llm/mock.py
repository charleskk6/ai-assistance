"""Deterministic offline LLM for tests, CI and pipeline benchmarks without weights."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator

from .base import CancelToken, GenParams, LLMBackend, LLMChunk, Message


class MockLLM(LLMBackend):
    """Streams a scripted (or rule-based) reply word by word with a fixed per-token delay."""

    name = "mock"

    def __init__(self, replies: list[str] | None = None, token_delay_s: float = 0.01, ttft_s: float = 0.02) -> None:
        self.model_id = "mock-coach"
        self.replies = list(replies or [])
        self.token_delay_s = token_delay_s
        self.ttft_s = ttft_s
        self.requests: list[list[Message]] = []
        self.cancelled_count = 0

    def _reply(self, messages: list[Message]) -> str:
        if self.replies:
            return self.replies.pop(0)
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "JSON" in (messages[0].get("content", "") if messages else ""):
            return '{"mistakes": [], "alternatives": []}'
        m = re.search(r"\b(?:I|we)\s+(bring|go)\b.*\byesterday\b|\byesterday\b.*\b(?:I|we)\s+(bring|go)\b", last, re.I)
        if m:
            return "Oh, you took your son to school yesterday. How did he get on?"
        return "Lovely, that sounds brilliant. What are you up to this weekend, then?"

    async def stream(self, messages: list[Message], params: GenParams, cancel: CancelToken) -> AsyncIterator[LLMChunk]:
        self.requests.append(messages)
        text = self._reply(messages)
        t0 = time.perf_counter()
        await asyncio.sleep(self.ttft_s)
        words = re.findall(r"\S+\s*", text)[: params.max_tokens]
        n = 0
        completed = False
        try:
            for w in words:
                if cancel.cancelled:
                    break
                n += 1
                yield LLMChunk(text=w)
                await asyncio.sleep(self.token_delay_s)
            completed = n == len(words)
        finally:
            if not completed:
                self.cancelled_count += 1  # stopped early (token cancelled or task cancelled)
        dt = time.perf_counter() - t0
        yield LLMChunk(text="", done=True, stats={
            "generated_tokens": n, "ttft_s": self.ttft_s, "tokens_per_s": n / dt if dt else None,
            "cancelled": cancel.cancelled,
        })

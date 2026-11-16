"""LLM backend interface and a thread -> asyncio streaming bridge."""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Executor
from dataclasses import dataclass, field


class CancelToken:
    """Cooperative cancellation shared between the event loop and worker threads."""

    def __init__(self) -> None:
        self._ev = threading.Event()
        self.reason = ""

    def cancel(self, reason: str = "") -> None:
        self.reason = self.reason or reason
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()


@dataclass
class GenParams:
    max_tokens: int = 110
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] = field(default_factory=list)


@dataclass
class LLMChunk:
    text: str
    done: bool = False
    stats: dict | None = None  # set on the final chunk: tokens, tps, ttft, memory...


Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLMBackend(ABC):
    name = "base"
    model_id = ""

    async def load(self) -> None:
        """Load weights. Must be idempotent."""

    async def unload(self) -> None:
        """Release weights and accelerator memory."""

    async def warmup(self) -> None:
        tok = CancelToken()
        async for _ in self.stream([{"role": "user", "content": "Hi"}], GenParams(max_tokens=4), tok):
            pass

    @abstractmethod
    def stream(self, messages: list[Message], params: GenParams, cancel: CancelToken) -> AsyncIterator[LLMChunk]:
        """Yield text deltas; the last chunk has ``done=True`` and ``stats``."""

    async def complete(self, messages: list[Message], params: GenParams, cancel: CancelToken | None = None) -> str:
        parts = []
        async for chunk in self.stream(messages, params, cancel or CancelToken()):
            parts.append(chunk.text)
        return "".join(parts)

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_id}


async def iterate_in_thread(
    make_iter: Callable[[], Iterator],
    executor: Executor,
    cancel: CancelToken,
    max_queue: int = 256,
) -> AsyncIterator:
    """Run a blocking iterator on ``executor`` and yield its items on the event loop.

    The producer checks ``cancel`` between items, so a cancelled generation stops
    within one token. Closing the async generator before it is exhausted also sets
    ``cancel``; normal exhaustion does not (the token may be shared by a whole turn).
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
    sentinel = object()

    def put(item) -> None:
        # Called from the worker thread.
        fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
        try:
            fut.result(timeout=30)
        except Exception:
            cancel.cancel("event loop unavailable")
            raise

    def produce() -> None:
        try:
            for item in make_iter():
                if cancel.cancelled:
                    break
                put(item)
        except BaseException as exc:  # forwarded to the consumer
            if not cancel.cancelled:
                put(exc)
        finally:
            try:
                put(sentinel)
            except Exception:
                pass

    fut = loop.run_in_executor(executor, produce)
    finished = False
    try:
        while True:
            item = await q.get()
            if item is sentinel:
                finished = True
                break
            if isinstance(item, BaseException):
                finished = True
                raise item
            yield item
    finally:
        if not finished:
            # The consumer stopped early (cancelled or closed): stop the producer too.
            cancel.cancel("consumer closed")
        # Drain so a producer blocked on a full queue can finish.
        while not fut.done():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
        await asyncio.shield(fut)


def now() -> float:
    return time.perf_counter()

"""Apple MLX backend (mlx-lm). Apple Silicon only.

* The model stays loaded for the life of the process.
* All MLX work runs on one dedicated thread (MLX streams are per-thread).
* Prompt-prefix KV-cache reuse: the system prompt and earlier history are not
  re-prefilled each turn; only the new suffix is. Per-turn coaching hints are
  appended to the *last* user message so the cached prefix stays stable.
  If anything about cache reuse fails, it falls back to a full prefill.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from .base import CancelToken, GenParams, LLMBackend, LLMChunk, Message, iterate_in_thread

log = logging.getLogger(__name__)


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class MLXBackend(LLMBackend):
    name = "mlx"

    def __init__(self, model_id: str, prompt_cache: bool = True) -> None:
        self.model_id = model_id
        self.use_prompt_cache = prompt_cache
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._model = None
        self._tok = None
        self._cache = None
        self._cache_tokens: list[int] = []
        self.last_stats: dict = {}

    # ----------------------------------------------------------------- lifecycle
    async def load(self) -> None:
        import asyncio

        if self._model is None:
            await asyncio.get_running_loop().run_in_executor(self._ex, self._load_sync)

    def _load_sync(self) -> None:
        from mlx_lm import load

        t0 = time.perf_counter()
        self._model, self._tok = load(self.model_id)
        log.info("Loaded %s in %.1fs", self.model_id, time.perf_counter() - t0)

    async def unload(self) -> None:
        import asyncio

        def _free() -> None:
            import mlx.core as mx

            self._model = self._tok = self._cache = None
            self._cache_tokens = []
            gc.collect()
            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()

        await asyncio.get_running_loop().run_in_executor(self._ex, _free)

    # ----------------------------------------------------------------- generation
    def _encode(self, messages: list[Message]) -> list[int]:
        prompt = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        return list(self._tok.encode(prompt, add_special_tokens=False))

    def _prepare_cache(self, tokens: list[int]) -> tuple[object | None, list[int], int]:
        """Return (cache, tokens_to_feed, reused_count)."""
        if not self.use_prompt_cache:
            return None, tokens, 0
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

        try:
            if self._cache is not None:
                common = _common_prefix(self._cache_tokens, tokens)
                if common == len(tokens):
                    common -= 1  # must feed at least one token
                to_trim = len(self._cache_tokens) - common
                if common > 0 and (to_trim == 0 or can_trim_prompt_cache(self._cache)):
                    if to_trim:
                        trim_prompt_cache(self._cache, to_trim)
                    self._cache_tokens = self._cache_tokens[:common]
                    return self._cache, tokens[common:], common
            self._cache = make_prompt_cache(self._model)
            self._cache_tokens = []
            return self._cache, tokens, 0
        except Exception:  # pragma: no cover - defensive; depends on mlx-lm version
            log.exception("Prompt cache reuse failed; using a full prefill")
            self._cache, self._cache_tokens = None, []
            return None, tokens, 0

    def _generate_iter(self, messages: list[Message], params: GenParams, cancel: CancelToken):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        t0 = time.perf_counter()
        tokens = self._encode(messages)
        cache, feed, reused = self._prepare_cache(tokens)
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)
        kwargs = {"max_tokens": params.max_tokens, "sampler": sampler}
        if cache is not None:
            kwargs["prompt_cache"] = cache

        generated: list[int] = []
        first_t = None
        last = None
        try:
            for resp in stream_generate(self._model, self._tok, feed, **kwargs):
                if first_t is None:
                    first_t = time.perf_counter()
                generated.append(resp.token)
                last = resp
                if cancel.cancelled:
                    break
                yield LLMChunk(text=resp.text)
        finally:
            if cache is not None:
                try:
                    offset = cache[0].offset
                    self._cache_tokens = (self._cache_tokens + feed + generated)[:offset]
                except Exception:
                    self._cache, self._cache_tokens = None, []
        stats = {
            "prompt_tokens": len(tokens),
            "prefill_tokens": len(feed),
            "reused_prefix_tokens": reused,
            "generated_tokens": len(generated),
            "ttft_s": (first_t - t0) if first_t else None,
            "tokens_per_s": getattr(last, "generation_tps", None),
            "prompt_tps": getattr(last, "prompt_tps", None),
            "peak_memory_gb": getattr(last, "peak_memory", None),
            "cancelled": cancel.cancelled,
            "truncated": getattr(last, "finish_reason", None) == "length",
        }
        self.last_stats = stats
        yield LLMChunk(text="", done=True, stats=stats)

    async def stream(self, messages: list[Message], params: GenParams, cancel: CancelToken) -> AsyncIterator[LLMChunk]:
        await self.load()
        async for chunk in iterate_in_thread(lambda: self._generate_iter(messages, params, cancel), self._ex, cancel):
            yield chunk

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_id, "prompt_cache": self.use_prompt_cache, **self.last_stats}

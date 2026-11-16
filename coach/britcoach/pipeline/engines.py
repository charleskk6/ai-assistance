"""Process-wide model engines: loaded once at startup, warmed up, shared by sessions.

Loading failures are recorded, not fatal, so the UI can show exactly what is
missing (e.g. "run scripts/setup_models.py") while everything else works.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from ..audio.vad import VAD, create_vad
from ..config import Settings, load_catalog, resolve_llm_model
from ..llm.base import LLMBackend
from ..llm.factory import create_llm
from ..stt.base import STTBackend
from ..stt.engines import create_stt
from ..tts.engines import TTSBackend, create_tts

log = logging.getLogger(__name__)


class Engines:
    def __init__(self, settings: Settings, llm: LLMBackend | None = None, stt: STTBackend | None = None,
                 tts: TTSBackend | None = None, vad: VAD | None = None) -> None:
        self.settings = settings
        self.catalog = load_catalog(str(settings.path(settings.models_toml)))
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.status: dict[str, dict] = {}
        self._switch_lock = asyncio.Lock()

    # ------------------------------------------------------------------ startup
    async def start(self, warmup: bool = True) -> None:
        s = self.settings
        models_dir = s.path(s.models_dir)
        loop = asyncio.get_running_loop()

        if self.vad is None:
            self.vad = create_vad(s.vad_backend, s.path(s.vad_model_path))
        self.status["vad"] = {"ok": True, "backend": self.vad.name}

        async def step(name: str, coro_factory) -> None:
            t0 = time.perf_counter()
            try:
                await coro_factory()
                self.status[name] = {"ok": True, "load_s": round(time.perf_counter() - t0, 2)}
            except Exception as e:  # noqa: BLE001 - report every failure to the UI
                log.exception("%s failed to load", name)
                self.status[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        async def load_stt() -> None:
            if self.stt is None:
                self.stt = create_stt(s.stt_backend, s.stt_model, models_dir, s.stt_threads)
            await loop.run_in_executor(self.stt_executor, self.stt.load)
            if warmup:
                await loop.run_in_executor(self.stt_executor, self.stt.warmup)

        async def load_tts() -> None:
            if self.tts is None:
                self.tts = create_tts(s.tts_backend, models_dir, self.catalog, s.tts_length_scale)
            if warmup:
                await loop.run_in_executor(self.tts_executor, self.tts.warmup, self.default_voice())

        async def load_llm() -> None:
            if self.llm is None:
                self.llm = create_llm(s)
            await self.llm.load()
            if warmup:
                await self.llm.warmup()

        # STT/TTS load on their own threads concurrently with the LLM.
        await asyncio.gather(step("stt", load_stt), step("tts", load_tts), step("llm", load_llm))
        for name, eng in (("stt", self.stt), ("tts", self.tts), ("llm", self.llm)):
            if eng is not None and self.status.get(name, {}).get("ok"):
                self.status[name].update(eng.info() if hasattr(eng, "info") and name != "tts" else {"backend": eng.name})

    def default_voice(self) -> str:
        s = self.settings
        if self.tts and self.tts.name == "macos_say" and s.tts_voice.startswith("en_GB-"):
            voices = self.tts.voices()
            return voices[0]["id"] if voices else "Daniel"
        if self.tts and self.tts.name == "mock":
            return "mock"
        return s.tts_voice

    def ready(self) -> tuple[bool, str]:
        missing = [f"{k}: {v.get('error')}" for k, v in self.status.items() if not v.get("ok")]
        return (not missing, "; ".join(missing))

    # ------------------------------------------------------------------ model switching
    async def switch_llm(self, backend: str, model: str | None = None, profile: str | None = None) -> dict:
        """Unload the current model and load another. In-flight turns finish on the old one."""
        async with self._switch_lock:
            new = create_llm(self.settings, backend, model or resolve_llm_model(self.settings, backend, profile))
            t0 = time.perf_counter()
            await new.load()
            await new.warmup()
            old, self.llm = self.llm, new
            if old is not None and old is not new:
                await old.unload()
            self.status["llm"] = {"ok": True, "load_s": round(time.perf_counter() - t0, 2), **new.info()}
            return self.status["llm"]

    def shutdown(self) -> None:
        self.stt_executor.shutdown(wait=False, cancel_futures=True)
        self.tts_executor.shutdown(wait=False, cancel_futures=True)

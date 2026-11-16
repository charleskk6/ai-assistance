"""Concrete STT engines: faster-whisper (CTranslate2, CPU int8), whisper.cpp, mock."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

from .base import STTBackend

log = logging.getLogger(__name__)


class FasterWhisperSTT(STTBackend):
    """CTranslate2 Whisper on the CPU.

    Running STT on the CPU leaves the Apple GPU to the MLX LLM, so the two
    don't contend for the same Metal queue mid-turn.
    """

    name = "faster_whisper"

    def __init__(self, model: str, models_dir: Path, threads: int = 4) -> None:
        self.model_name = model
        self.models_dir = models_dir
        self.threads = threads
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        local = self.models_dir / "stt" / f"faster-whisper-{self.model_name}"
        source = str(local) if local.exists() else self.model_name
        self._model = WhisperModel(source, device="cpu", compute_type="int8", cpu_threads=self.threads,
                                   download_root=str(self.models_dir / "stt"))

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=1, best_of=1, temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments)

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name, "device": "cpu-int8", "threads": self.threads}


class WhisperCppSTT(STTBackend):
    """whisper.cpp via pywhispercpp (Metal-accelerated when built for Apple Silicon)."""

    name = "whispercpp"

    def __init__(self, model: str, models_dir: Path, threads: int = 4) -> None:
        self.model_name = model
        self.models_dir = models_dir
        self.threads = threads
        self._model = None

    def load(self) -> None:
        from pywhispercpp.model import Model

        path = self.models_dir / "stt" / f"ggml-{self.model_name}.bin"
        self._model = Model(str(path) if path.exists() else self.model_name,
                            models_dir=str(self.models_dir / "stt"), n_threads=self.threads,
                            print_progress=False, print_realtime=False, redirect_whispercpp_logs_to=None)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments = self._model.transcribe(audio, language="en", no_context=True, single_segment=True)
        return " ".join(s.text.strip() for s in segments)

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name, "threads": self.threads}


class MockSTT(STTBackend):
    """Returns scripted transcripts in order; for tests and pipeline benchmarks without weights."""

    name = "mock"

    def __init__(self, script: list[str] | None = None, latency_s: float = 0.0) -> None:
        self.script = deque(script or [])
        self.latency_s = latency_s
        self.calls = 0

    def load(self) -> None:
        pass

    def warmup(self) -> None:
        pass

    def push(self, *texts: str) -> None:
        self.script.extend(texts)

    def peek(self) -> str:
        return self.script[0] if self.script else "Hello there"

    def _transcribe(self, audio: np.ndarray) -> str:
        import time

        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        return self.peek()

    def on_final(self, text: str) -> None:
        """Advance the script once a *final* transcript has been produced."""
        if self.script:
            self.script.popleft()


def create_stt(backend: str, model: str, models_dir: Path, threads: int) -> STTBackend:
    if backend == "faster_whisper":
        return FasterWhisperSTT(model, models_dir, threads)
    if backend == "whispercpp":
        return WhisperCppSTT(model, models_dir, threads)
    if backend == "mock":
        return MockSTT()
    raise ValueError(f"Unknown STT backend: {backend}")

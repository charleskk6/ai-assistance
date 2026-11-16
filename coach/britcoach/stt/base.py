"""Speech-to-text interface.

Whisper-family engines are *not* natively streaming: they transcribe a whole
buffer. "Partial" transcripts in this app are produced by re-transcribing the
growing utterance buffer every ``stt_partial_interval_ms`` (chunked
pseudo-streaming), and the final transcript is started speculatively as soon as
the speaker pauses (see ``pipeline/session.py``).

All engines are synchronous and CPU-bound; the session runs them on a dedicated
single-thread executor so they never block the asyncio loop.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Whisper hallucinations on silence / noise that should be treated as "nothing said".
_HALLUCINATIONS = {
    "", "you", "thank you", "thanks for watching", "thank you for watching", "bye",
    "[blank_audio]", "(silence)", "[silence]", "[music]", "(music)",
}


@dataclass
class STTResult:
    text: str
    audio_s: float
    latency_s: float


def clean_transcript(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)  # [BLANK_AUDIO], (laughs) ...
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower().strip(" .!?,") in _HALLUCINATIONS:
        return ""
    return text


class STTBackend(ABC):
    name = "base"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def _transcribe(self, audio: np.ndarray, prompt: str = "") -> str: ...

    def transcribe(self, audio: np.ndarray, final: bool = True, prompt: str = "") -> STTResult:
        """``prompt``: recent conversation text (e.g. the coach's last question). Whisper uses it
        as context, which helps with names and topic words."""
        t0 = time.perf_counter()
        text = clean_transcript(self._transcribe(audio.astype(np.float32), prompt)) if len(audio) else ""
        if final:
            self.on_final(text)
        return STTResult(text=text, audio_s=len(audio) / 16000, latency_s=time.perf_counter() - t0)

    def on_final(self, text: str) -> None:
        """Hook called after a final (not partial) transcription."""

    def warmup(self) -> None:
        # One second of low noise exercises the whole graph and allocates buffers.
        self._transcribe((np.random.default_rng(0).standard_normal(16000) * 0.001).astype(np.float32))

    def info(self) -> dict:
        return {"backend": self.name}

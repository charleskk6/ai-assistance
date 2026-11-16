"""TTS engines. All are synchronous and run on a dedicated executor thread.

* ``PiperTTS``   – offline neural TTS (onnxruntime, CPU). Note: piper-tts >= 1.3 is GPL-3.0.
* ``MacSayTTS``  – the built-in macOS voices (e.g. "Daniel", en_GB). No download needed.
* ``MockTTS``    – a quiet tone whose length tracks the text; for tests and benchmarks.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
import time
import wave
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TTSAudio:
    pcm16: bytes
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate


class TTSBackend(ABC):
    name = "base"

    def load(self, voice: str) -> None:
        """Preload a voice (idempotent)."""

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> Iterator[TTSAudio]:
        """Yield audio for ``text``; may yield several chunks."""

    def voices(self) -> list[dict]:
        return []

    def warmup(self, voice: str) -> None:
        for _ in self.synthesize("Right then.", voice):
            pass


class PiperTTS(TTSBackend):
    name = "piper"

    def __init__(self, voices_dir: Path, catalog: list[dict], length_scale: float = 1.0) -> None:
        self.voices_dir = voices_dir
        self.catalog = catalog
        self.length_scale = length_scale
        self._loaded: OrderedDict[str, object] = OrderedDict()  # tiny LRU: switching voices is instant

    def _path(self, voice: str) -> Path:
        return self.voices_dir / f"{voice}.onnx"

    def voices(self) -> list[dict]:
        out = []
        for v in self.catalog:
            out.append({"id": v["id"], "label": v["label"], "installed": self._path(v["id"]).exists(), "backend": self.name})
        return out

    def load(self, voice: str) -> object:
        if voice in self._loaded:
            self._loaded.move_to_end(voice)
            return self._loaded[voice]
        from piper import PiperVoice

        path = self._path(voice)
        if not path.exists():
            raise FileNotFoundError(f"Piper voice {voice} not installed at {path}. Run scripts/setup_models.py --voices {voice}")
        pv = PiperVoice.load(str(path))
        self._loaded[voice] = pv
        while len(self._loaded) > 2:
            self._loaded.popitem(last=False)
        return pv

    def synthesize(self, text: str, voice: str) -> Iterator[TTSAudio]:
        from piper import SynthesisConfig

        pv = self.load(voice)
        cfg = SynthesisConfig(length_scale=self.length_scale)
        for chunk in pv.synthesize(text, syn_config=cfg):
            yield TTSAudio(chunk.audio_int16_bytes, chunk.sample_rate)


class MacSayTTS(TTSBackend):
    name = "macos_say"
    RATE = 22050

    def __init__(self, preferred: list[str]) -> None:
        self.preferred = preferred

    def voices(self) -> list[dict]:
        if platform.system() != "Darwin":
            return []
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        found = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and "en_GB" in line:
                name = line.split("en_GB")[0].strip()
                found.append({"id": name, "label": f"{name} (macOS, en_GB)", "installed": True, "backend": self.name})
        found.sort(key=lambda v: self.preferred.index(v["id"]) if v["id"] in self.preferred else 99)
        return found

    def synthesize(self, text: str, voice: str) -> Iterator[TTSAudio]:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(["say", "-v", voice or "Daniel", "-o", path, "--file-format=WAVE",
                            f"--data-format=LEI16@{self.RATE}", text], check=True, timeout=30)
            with wave.open(path, "rb") as w:
                yield TTSAudio(w.readframes(w.getnframes()), w.getframerate())
        finally:
            os.unlink(path)


class MockTTS(TTSBackend):
    """~60 ms of audio per character at 16 kHz, with a configurable synthesis delay."""

    name = "mock"
    RATE = 16000

    def __init__(self, delay_s: float = 0.0, ms_per_char: float = 60.0, amplitude: float = 0.05) -> None:
        self.delay_s = delay_s
        self.ms_per_char = ms_per_char
        self.amplitude = amplitude
        self.calls: list[str] = []

    def voices(self) -> list[dict]:
        return [{"id": "mock", "label": "Mock tone", "installed": True, "backend": self.name}]

    def synthesize(self, text: str, voice: str) -> Iterator[TTSAudio]:
        self.calls.append(text)
        if self.delay_s:
            time.sleep(self.delay_s)
        n = int(self.RATE * len(text) * self.ms_per_char / 1000)
        t = np.arange(n) / self.RATE
        audio = (self.amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        yield TTSAudio((audio * 32767).astype("<i2").tobytes(), self.RATE)


def create_tts(backend: str, models_dir: Path, catalog: dict, length_scale: float = 1.0) -> TTSBackend:
    if backend == "piper":
        return PiperTTS(models_dir / "tts", catalog.get("tts", {}).get("piper_voices", []), length_scale)
    if backend == "macos_say":
        return MacSayTTS(catalog.get("tts", {}).get("macos_say", {}).get("preferred", []))
    if backend == "mock":
        return MockTTS()
    raise ValueError(f"Unknown TTS backend: {backend}")

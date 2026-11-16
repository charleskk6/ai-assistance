from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from britcoach.audio.pcm import FRAME_SAMPLES
from britcoach.audio.vad import EnergyVAD
from britcoach.config import Settings
from britcoach.llm.mock import MockLLM
from britcoach.pipeline.engines import Engines
from britcoach.stt.engines import MockSTT
from britcoach.tts.engines import MockTTS

ROOT = Path(__file__).resolve().parent.parent


def make_settings(**kw) -> Settings:
    base = dict(_env_file=None, llm_backend="mock", stt_backend="mock", tts_backend="mock", vad_backend="energy",
                save_history=False, stt_partials=False, idle_prompt_s=0, min_speech_ms=96, end_of_turn_ms=320,
                stt_speculative_ms=160, barge_in_min_ms_headphones=96, barge_in_min_ms_speakers=160,
                barge_in_threshold_headphones=0.5, barge_in_threshold_speakers=0.5, cloud_enabled=False)
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


def speech_frame(rng=np.random.default_rng(1)) -> np.ndarray:
    return (rng.standard_normal(FRAME_SAMPLES) * 0.2).astype(np.float32)


def silence_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.float32)


def to_pcm(frames: list[np.ndarray]) -> bytes:
    return (np.concatenate(frames) * 32767).astype("<i2").tobytes()


class Recorder:
    """Collects everything a Session sends to the client."""

    def __init__(self) -> None:
        self.json: list[dict] = []
        self.audio: list[bytes] = []
        self.order: list[tuple[str, object]] = []

    async def send_json(self, msg: dict) -> None:
        self.json.append(msg)
        self.order.append(("json", msg))

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)
        self.order.append(("bytes", data))

    def of(self, kind: str) -> list[dict]:
        return [m for m in self.json if m.get("type") == kind]


async def make_engines(settings: Settings, llm=None, stt=None, tts=None) -> Engines:
    eng = Engines(settings, llm=llm or MockLLM(), stt=stt or MockSTT(), tts=tts or MockTTS(), vad=EnergyVAD())
    await eng.start(warmup=False)
    return eng


async def wait_for(pred, timeout: float = 3.0, step: float = 0.005) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(step)


@pytest.fixture(scope="session")
def silero_path() -> Path | None:
    """The Silero model from the installed silero-vad wheel (or models/vad)."""
    p = ROOT / "models" / "vad" / "silero_vad.onnx"
    if p.exists():
        return p
    try:
        import importlib.util

        spec = importlib.util.find_spec("silero_vad")
        if spec and spec.submodule_search_locations:
            cand = Path(list(spec.submodule_search_locations)[0]) / "data" / "silero_vad.onnx"
            if cand.exists():
                return cand
    except Exception:
        pass
    return None


@pytest.fixture(scope="session")
def spoken_wav(tmp_path_factory) -> Path | None:
    """A real spoken English sentence at 16 kHz (espeak-ng or macOS `say`), or None."""
    out = tmp_path_factory.mktemp("speech") / "speech.wav"
    if shutil.which("espeak-ng"):
        cmd = ["espeak-ng", "-v", "en-gb", "-s", "150", "-w", str(out),
               "Yesterday I took my son to school and then I went to work."]
    elif shutil.which("say"):
        cmd = ["say", "-v", "Daniel", "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@16000",
               "Yesterday I took my son to school and then I went to work."]
    else:
        return None
    subprocess.run(cmd, check=True, capture_output=True)
    return out

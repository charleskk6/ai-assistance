"""Voice activity detection.

``SileroVAD`` runs the Silero VAD ONNX model directly on onnxruntime (no torch).
The model file ships inside the MIT-licensed ``silero-vad`` wheel on PyPI;
``scripts/setup_models.py`` copies it to ``models/vad/silero_vad.onnx``.

``EnergyVAD`` is a dependency-free fallback used when the model is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .pcm import FRAME_SAMPLES, SAMPLE_RATE, rms_dbfs

log = logging.getLogger(__name__)


class VAD:
    name = "base"

    def prob(self, frame: np.ndarray) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def clone(self) -> "VAD":
        """A fresh instance with independent recurrent state (one per session)."""
        raise NotImplementedError


class SileroVAD(VAD):
    name = "silero"
    CONTEXT = 64  # samples of left context the v5+ model expects at 16 kHz

    def __init__(self, model_path: str | Path, _session=None) -> None:
        import onnxruntime as ort

        self.model_path = str(model_path)
        if _session is None:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            _session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._session = _session
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def clone(self) -> "SileroVAD":
        return SileroVAD(self.model_path, _session=self._session)  # sessions are thread-safe to share

    def prob(self, frame: np.ndarray) -> float:
        if frame.shape[-1] != FRAME_SAMPLES:
            raise ValueError(f"Silero expects {FRAME_SAMPLES} samples, got {frame.shape[-1]}")
        x = np.concatenate([self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._state = state
        self._context = x[:, -self.CONTEXT:]
        return float(out[0][0])


class EnergyVAD(VAD):
    """Maps frame loudness to a pseudo-probability. Crude, but has no dependencies."""

    name = "energy"

    def __init__(self, speech_dbfs: float = -35.0, silence_dbfs: float = -50.0) -> None:
        self.speech_dbfs = speech_dbfs
        self.silence_dbfs = silence_dbfs

    def prob(self, frame: np.ndarray) -> float:
        db = rms_dbfs(frame)
        span = self.speech_dbfs - self.silence_dbfs
        return float(np.clip((db - self.silence_dbfs) / span, 0.0, 1.0))

    def clone(self) -> "EnergyVAD":
        return EnergyVAD(self.speech_dbfs, self.silence_dbfs)


def create_vad(backend: str, model_path: Path) -> VAD:
    if backend == "silero":
        if model_path.exists():
            return SileroVAD(model_path)
        log.warning("Silero model not found at %s; falling back to energy VAD. Run scripts/setup_models.py", model_path)
    return EnergyVAD()

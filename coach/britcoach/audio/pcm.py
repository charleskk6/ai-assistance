"""PCM helpers and the binary audio frame format used on the WebSocket."""

from __future__ import annotations

import struct

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32 ms at 16 kHz: Silero VAD's native frame size
FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE

# Server -> client audio: 16-byte header then int16 little-endian mono PCM.
#   uint32 turn_id | uint32 seq | uint32 sample_rate | uint32 sentence_idx
AUDIO_HEADER = struct.Struct("<IIII")


def int16_bytes_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pack_audio(turn_id: int, seq: int, sample_rate: int, sentence_idx: int, pcm16: bytes) -> bytes:
    return AUDIO_HEADER.pack(turn_id, seq, sample_rate, sentence_idx) + pcm16


def unpack_audio(msg: bytes) -> tuple[int, int, int, int, bytes]:
    turn_id, seq, sr, idx = AUDIO_HEADER.unpack_from(msg)
    return turn_id, seq, sr, idx, msg[AUDIO_HEADER.size:]


class FrameAssembler:
    """Re-chunks arbitrary-length int16 PCM into fixed 512-sample float frames."""

    def __init__(self, frame_samples: int = FRAME_SAMPLES) -> None:
        self.frame_samples = frame_samples
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, data: bytes) -> list[np.ndarray]:
        if len(data) % 2:
            data = data[:-1]
        self._buf = np.concatenate([self._buf, int16_bytes_to_float(data)])
        n = len(self._buf) // self.frame_samples
        frames = [self._buf[i * self.frame_samples:(i + 1) * self.frame_samples] for i in range(n)]
        self._buf = self._buf[n * self.frame_samples:]
        return frames


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / src_rate
    n_out = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def rms_dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)

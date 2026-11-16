"""End-of-turn detection: a pure state machine over per-frame VAD probabilities.

Kept free of I/O and clocks so it can be unit-tested deterministically. Time is
"audio time" (frame count x 32 ms); the session maps events to wall-clock time
for latency metrics.

States: SILENCE -> CANDIDATE -> SPEECH -> (end of turn) -> SILENCE.
A candidate becomes speech only after ``min_speech_ms`` of voiced frames, which
rejects clicks, coughs and short echo bursts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .pcm import FRAME_MS


class TState(str, Enum):
    SILENCE = "silence"
    CANDIDATE = "candidate"
    SPEECH = "speech"


@dataclass
class TurnConfig:
    threshold: float = 0.5
    min_speech_ms: float = 160
    end_of_turn_ms: float = 560
    speculative_ms: float = 240
    pre_roll_ms: float = 320
    candidate_gap_ms: float = 160  # an unvoiced gap this long aborts a candidate
    max_turn_s: float = 30.0
    trailing_keep_ms: float = 120

    @property
    def neg_threshold(self) -> float:
        return max(0.05, self.threshold - 0.15)


@dataclass
class SpeechStarted:
    start_ms: float  # audio time of the first voiced frame (excluding pre-roll)
    confirm_ms: float  # audio time at which the start was confirmed


@dataclass
class SilenceReached:
    """Speech has paused for ``speculative_ms``: a good moment to start transcribing early."""

    audio: np.ndarray
    voiced_frames: int  # lets the caller tell whether more speech arrived since


@dataclass
class SpeechEnded:
    audio: np.ndarray
    start_ms: float
    last_voice_ms: float
    end_ms: float
    voiced_frames: int
    forced: bool = False


Event = SpeechStarted | SilenceReached | SpeechEnded


@dataclass
class TurnDetector:
    cfg: TurnConfig = field(default_factory=TurnConfig)

    def __post_init__(self) -> None:
        self._pre = deque(maxlen=max(1, int(self.cfg.pre_roll_ms / FRAME_MS)))
        self.reset()

    # ------------------------------------------------------------------ control
    def reset(self) -> None:
        self.state = TState.SILENCE
        self.t_ms = 0.0
        self._frames: list[np.ndarray] = []
        self._voiced = 0
        self._unvoiced_run = 0
        self._silence_run = 0
        self._voiced_flag = False
        self._start_ms = 0.0
        self._last_voice_ms = 0.0
        self._spec_sent = False
        self._pre.clear()

    def set_sensitivity(self, threshold: float, min_speech_ms: float) -> None:
        """Used to tighten detection while the assistant is talking (barge-in)."""
        self.cfg.threshold = threshold
        self.cfg.min_speech_ms = min_speech_ms

    def current_audio(self) -> np.ndarray:
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)

    @property
    def silence_ms(self) -> float:
        return self._silence_run * FRAME_MS

    @property
    def last_voice_ms(self) -> float:
        return self._last_voice_ms

    @property
    def voiced_frames(self) -> int:
        return self._voiced

    # ------------------------------------------------------------------ core
    def _is_voiced(self, prob: float) -> bool:
        if prob >= self.cfg.threshold:
            self._voiced_flag = True
        elif prob < self.cfg.neg_threshold:
            self._voiced_flag = False
        return self._voiced_flag

    def process(self, prob: float, frame: np.ndarray) -> list[Event]:
        events: list[Event] = []
        self.t_ms += FRAME_MS
        voiced = self._is_voiced(prob)
        c = self.cfg

        if self.state is TState.SILENCE:
            if voiced:
                self.state = TState.CANDIDATE
                self._frames = list(self._pre) + [frame]
                self._pre.clear()
                self._voiced = 1
                self._unvoiced_run = 0
                self._start_ms = self.t_ms - FRAME_MS
                self._last_voice_ms = self.t_ms
                if self._voiced * FRAME_MS >= c.min_speech_ms:
                    self.state = TState.SPEECH
                    events.append(SpeechStarted(self._start_ms, self.t_ms))
            else:
                self._pre.append(frame)
            return events

        self._frames.append(frame)

        if self.state is TState.CANDIDATE:
            if voiced:
                self._voiced += 1
                self._unvoiced_run = 0
                self._last_voice_ms = self.t_ms
                if self._voiced * FRAME_MS >= c.min_speech_ms:
                    self.state = TState.SPEECH
                    self._silence_run = 0
                    self._spec_sent = False
                    events.append(SpeechStarted(self._start_ms, self.t_ms))
            else:
                self._unvoiced_run += 1
                if self._unvoiced_run * FRAME_MS >= c.candidate_gap_ms:
                    # Too short to be speech: back to silence, keep the tail as pre-roll.
                    tail = self._frames[-self._pre.maxlen:]
                    self.state = TState.SILENCE
                    self._frames = []
                    self._voiced = 0
                    self._pre.extend(tail)
            return events

        # SPEECH
        if voiced:
            self._voiced += 1
            self._silence_run = 0
            self._spec_sent = False
            self._last_voice_ms = self.t_ms
        else:
            self._silence_run += 1

        too_long = (self.t_ms - self._start_ms) >= c.max_turn_s * 1000
        if self.silence_ms >= c.end_of_turn_ms or too_long:
            events.append(self._finish(forced=too_long and self.silence_ms < c.end_of_turn_ms))
        elif not self._spec_sent and c.speculative_ms > 0 and self.silence_ms >= c.speculative_ms:
            self._spec_sent = True
            events.append(SilenceReached(self._trimmed_audio(), self._voiced))
        return events

    def _trimmed_audio(self) -> np.ndarray:
        keep = int(self.cfg.trailing_keep_ms / FRAME_MS)
        drop = max(0, self._silence_run - keep)
        frames = self._frames[: len(self._frames) - drop] if drop else self._frames
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)

    def _finish(self, forced: bool) -> SpeechEnded:
        ev = SpeechEnded(
            audio=self._trimmed_audio(),
            start_ms=self._start_ms,
            last_voice_ms=self._last_voice_ms,
            end_ms=self.t_ms,
            voiced_frames=self._voiced,
            forced=forced,
        )
        t = self.t_ms
        self.reset()
        self.t_ms = t
        return ev

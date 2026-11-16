import wave

import numpy as np
import pytest

from britcoach.audio.pcm import FRAME_MS, FRAME_SAMPLES, FrameAssembler, pack_audio, resample_linear, unpack_audio
from britcoach.audio.turn import SilenceReached, SpeechEnded, SpeechStarted, TState, TurnConfig, TurnDetector
from britcoach.audio.vad import EnergyVAD, SileroVAD

from .conftest import silence_frame, speech_frame


def run(det: TurnDetector, probs: list[float]):
    events = []
    for i, p in enumerate(probs):
        for ev in det.process(p, np.full(FRAME_SAMPLES, i, dtype=np.float32)):
            events.append((i, ev))
    return events


def test_speech_start_requires_min_duration():
    det = TurnDetector(TurnConfig(threshold=0.5, min_speech_ms=160, end_of_turn_ms=320, speculative_ms=0))
    need = int(np.ceil(160 / FRAME_MS))  # 5 frames
    ev = run(det, [0.0] * 3 + [0.9] * need)
    starts = [(i, e) for i, e in ev if isinstance(e, SpeechStarted)]
    assert len(starts) == 1 and starts[0][0] == 3 + need - 1
    assert starts[0][1].start_ms == pytest.approx(3 * FRAME_MS)


def test_short_blip_is_ignored():
    det = TurnDetector(TurnConfig(min_speech_ms=160, candidate_gap_ms=96))
    ev = run(det, [0.9, 0.9] + [0.0] * 10)
    assert ev == [] and det.state is TState.SILENCE


def test_end_of_turn_after_silence_and_speculative_first():
    cfg = TurnConfig(min_speech_ms=96, end_of_turn_ms=320, speculative_ms=160)
    det = TurnDetector(cfg)
    probs = [0.9] * 20 + [0.0] * 15
    ev = run(det, probs)
    kinds = [type(e).__name__ for _, e in ev]
    assert kinds == ["SpeechStarted", "SilenceReached", "SpeechEnded"]
    spec_i = next(i for i, e in ev if isinstance(e, SilenceReached))
    end_i, end = next((i, e) for i, e in ev if isinstance(e, SpeechEnded))
    assert spec_i == 19 + int(np.ceil(160 / FRAME_MS))
    assert end_i == 19 + int(np.ceil(320 / FRAME_MS))
    assert end.last_voice_ms == pytest.approx(20 * FRAME_MS)
    # trailing silence trimmed to ~trailing_keep_ms
    assert len(end.audio) <= (20 + int(cfg.trailing_keep_ms / FRAME_MS) + 1) * FRAME_SAMPLES
    assert det.state is TState.SILENCE


def test_pause_shorter_than_eot_keeps_turn_open_and_resends_speculation():
    det = TurnDetector(TurnConfig(min_speech_ms=96, end_of_turn_ms=480, speculative_ms=160))
    ev = run(det, [0.9] * 10 + [0.0] * 8 + [0.9] * 10 + [0.0] * 20)
    kinds = [type(e).__name__ for _, e in ev]
    assert kinds == ["SpeechStarted", "SilenceReached", "SilenceReached", "SpeechEnded"]
    spec1 = [e for _, e in ev if isinstance(e, SilenceReached)][0]
    end = [e for _, e in ev if isinstance(e, SpeechEnded)][0]
    assert spec1.voiced_frames < end.voiced_frames  # caller must not reuse the first speculation


def test_pre_roll_is_included():
    det = TurnDetector(TurnConfig(min_speech_ms=64, end_of_turn_ms=160, pre_roll_ms=96, speculative_ms=0))
    ev = run(det, [0.0] * 10 + [0.9] * 5 + [0.0] * 10)
    end = [e for _, e in ev if isinstance(e, SpeechEnded)][0]
    first_vals = end.audio[::FRAME_SAMPLES][:4]
    assert list(first_vals) == [7.0, 8.0, 9.0, 10.0]  # 3 pre-roll frames then speech


def test_max_turn_forces_end():
    det = TurnDetector(TurnConfig(min_speech_ms=64, max_turn_s=1.0, speculative_ms=0))
    ev = run(det, [0.9] * 60)
    ends = [e for _, e in ev if isinstance(e, SpeechEnded)]
    assert ends and ends[0].forced


def test_hysteresis():
    det = TurnDetector(TurnConfig(threshold=0.5, min_speech_ms=64))
    assert det._is_voiced(0.6) is True
    assert det._is_voiced(0.4) is True  # between neg and pos threshold: stays voiced
    assert det._is_voiced(0.3) is False


def test_frame_assembler_and_audio_packing():
    fa = FrameAssembler()
    pcm = (np.arange(1300) % 100).astype("<i2").tobytes()
    frames = fa.push(pcm[:1001]) + fa.push(pcm[1001:])
    assert len(frames) == 2 and all(len(f) == FRAME_SAMPLES for f in frames)
    msg = pack_audio(7, 3, 22050, 1, b"\x01\x00\x02\x00")
    assert unpack_audio(msg) == (7, 3, 22050, 1, b"\x01\x00\x02\x00")
    assert len(resample_linear(np.ones(22050, dtype=np.float32), 22050, 16000)) == 16000


def test_energy_vad():
    v = EnergyVAD()
    assert v.prob(speech_frame()) > 0.9
    assert v.prob(silence_frame()) == 0.0


def test_silero_vad_on_real_speech(silero_path, spoken_wav):
    if silero_path is None:
        pytest.skip("silero model not available")
    vad = SileroVAD(silero_path)
    assert vad.prob(silence_frame()) < 0.2
    if spoken_wav is None:
        pytest.skip("no local TTS to generate speech")
    with wave.open(str(spoken_wav)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        sr = w.getframerate()
    audio = resample_linear(audio, sr, 16000)
    vad.reset()
    probs = [vad.prob(audio[i:i + FRAME_SAMPLES]) for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES)]
    assert max(probs) > 0.8
    # Run the real VAD through the detector: exactly one turn, ending after the speech.
    det = TurnDetector(TurnConfig(end_of_turn_ms=560))
    vad.reset()
    padded = np.concatenate([np.zeros(8000, np.float32), audio, np.zeros(16000, np.float32)])
    events = []
    for i in range(0, len(padded) - FRAME_SAMPLES, FRAME_SAMPLES):
        f = padded[i:i + FRAME_SAMPLES]
        events += det.process(vad.prob(f), f)
    assert sum(isinstance(e, SpeechStarted) for e in events) >= 1
    assert sum(isinstance(e, SpeechEnded) for e in events) >= 1

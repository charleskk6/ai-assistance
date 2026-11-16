"""One voice conversation over one WebSocket: VAD -> STT -> LLM -> TTS with barge-in.

Concurrency model (all on the asyncio loop unless noted):

* Microphone frames arrive as binary WS messages and go through VAD + the turn
  detector inline (~0.1–0.5 ms per 32 ms frame).
* STT runs on the engines' single STT thread; TTS on the single TTS thread; MLX
  on its own thread. Nothing CPU-heavy runs on the loop.
* A reply is a ``ResponseTurn``: an LLM producer task feeding a bounded
  sentence queue, and a TTS consumer that sends audio as soon as each sentence
  is synthesised. Every audio frame carries the turn id, so the client drops
  anything stale.
* Barge-in: ``stop_audio`` is sent *first*, then the turn's cancel token is set
  (the LLM stops within one token) and its tasks are cancelled without waiting.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from ..audio.pcm import FRAME_MS, FrameAssembler, pack_audio
from ..audio.turn import SilenceReached, SpeechEnded, SpeechStarted, TState, TurnConfig, TurnDetector
from ..cloud.router import CloudPrefs, CloudRouter
from ..coaching import detector as rules
from ..coaching.policy import CorrectionPolicy, Strategy
from ..coaching.review import build_review, enrich_review, parse_llm_json, review_messages
from ..config import Settings
from ..conversation import prompts
from ..conversation.modes import MODES
from ..conversation.state import ConversationState
from ..llm.base import CancelToken, GenParams, iterate_in_thread
from ..storage.db import Store
from ..tts.text import SentenceSplitter, ThinkFilter, clean_for_speech
from .engines import Engines
from .metrics import TurnMetrics, memory_snapshot, ms

log = logging.getLogger(__name__)

SendJson = Callable[[dict], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


def word_overlap(a: str, b: str) -> float:
    """Share of the words in ``a`` that also occur in ``b`` (echo detection)."""
    wa = re.findall(r"[a-z']+", a.lower())
    wb = set(re.findall(r"[a-z']+", b.lower()))
    return sum(1 for w in wa if w in wb) / len(wa) if wa else 0.0


@dataclass
class ResponseTurn:
    id: int
    kind: str
    metrics: TurnMetrics
    cancel: CancelToken = field(default_factory=CancelToken)
    task: asyncio.Task | None = None
    t_request: float = 0.0
    speech_end_wall: float | None = None
    sentences: list[str] = field(default_factory=list)  # sent to TTS, in order
    durations: list[float] = field(default_factory=list)  # seconds of audio per sentence index
    first_audio_sent: float | None = None
    play_start: float | None = None  # client-reported (falls back to first_audio_sent)
    playback_done: bool = False
    text_done: bool = False
    full_text: str = ""

    def audio_total(self) -> float:
        return sum(self.durations)

    def playing(self, now: float) -> bool:
        if self.playback_done:
            return False
        start = self.play_start or self.first_audio_sent
        if start is None:
            return False
        return now < start + self.audio_total() + 0.25 or not self.text_done

    def heard_text(self, now: float) -> str:
        """Sentences whose playback had started by ``now``."""
        start = self.play_start or self.first_audio_sent
        if start is None:
            return ""
        elapsed, t, heard = now - start, 0.0, []
        for s, d in zip(self.sentences, self.durations):
            if t > elapsed:
                break
            heard.append(s)
            t += d
        return " ".join(heard)


class Session:
    def __init__(self, engines: Engines, settings: Settings, send_json: SendJson, send_bytes: SendBytes,
                 store: Store | None = None, cloud: CloudRouter | None = None) -> None:
        self.engines = engines
        self.s = settings
        self.send_json = send_json
        self.send_bytes = send_bytes
        self.store = store
        self.cloud = cloud
        self.id = uuid.uuid4().hex[:12]
        self.active = False
        self.state = "idle"
        mode = settings.default_mode if settings.default_mode in MODES else "free"
        self.conv = ConversationState(mode=mode, max_messages=settings.context_max_messages,
                                      char_budget=settings.context_char_budget)
        self.policy = CorrectionPolicy(sensitivity=MODES[mode].default_sensitivity, tutor=(mode == "tutor"))
        self.vad = engines.vad.clone()
        self.turn_cfg = TurnConfig(threshold=settings.vad_threshold, min_speech_ms=settings.min_speech_ms,
                                   end_of_turn_ms=settings.end_of_turn_ms, speculative_ms=settings.stt_speculative_ms,
                                   pre_roll_ms=settings.pre_roll_ms, max_turn_s=settings.max_turn_s)
        self.detector = TurnDetector(self.turn_cfg)
        self.assembler = FrameAssembler()
        self.voice = engines.default_voice()
        self.barge_in_mode = settings.barge_in_mode
        self.cloud_prefs = CloudPrefs(enabled=settings.cloud_enabled, provider=settings.cloud_provider,
                                      allow_turn_corrections=settings.cloud_allow_turn_corrections)
        self.response: ResponseTurn | None = None
        self._turn_counter = 0
        self.metrics_log: list[dict] = []
        self._pending_prefix = ""
        self._spec: tuple[asyncio.Future, int] | None = None
        self._partial_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._echo_task: asyncio.Task | None = None
        self._bargein: dict | None = None
        self._speech_first_voice_wall: float | None = None
        self._last_voice_wall: float | None = None
        self._vad_times: list[float] = []
        self._idle_nudges = 0
        self._bg: set[asyncio.Task] = set()
        self._started_store = False
        self._pending_bargein_metrics: TurnMetrics | None = None

    # ================================================================== helpers
    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        return t

    async def _emit(self, msg: dict) -> None:
        try:
            await self.send_json(msg)
        except Exception:  # client went away
            log.debug("send failed", exc_info=True)

    async def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            await self._emit({"type": "state", "state": state})

    def _ai_audible(self, now: float | None = None) -> bool:
        rt = self.response
        return bool(rt and rt.playing(now or time.perf_counter()))

    def _apply_detector_sensitivity(self) -> None:
        if self._ai_audible() and self.barge_in_mode in ("headphones", "speakers"):
            if self.barge_in_mode == "headphones":
                self.detector.set_sensitivity(self.s.barge_in_threshold_headphones, self.s.barge_in_min_ms_headphones)
            else:
                self.detector.set_sensitivity(self.s.barge_in_threshold_speakers, self.s.barge_in_min_ms_speakers)
        else:
            self.detector.set_sensitivity(self.s.vad_threshold, self.s.min_speech_ms)

    # ================================================================== control messages
    async def handle_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "start":
            await self.configure(msg.get("config") or {})
            await self.start()
        elif kind == "config":
            await self.configure(msg.get("config") or {})
        elif kind == "stop":
            await self.stop()
        elif kind == "text_input":  # typed message: same path as a spoken final transcript
            text = (msg.get("text") or "").strip()
            if text:
                await self._cancel_response("typed input", unheard_merge=False)
                await self._on_user_text(text, TurnMetrics(kind="typed"))
        elif kind == "playback":
            await self._on_playback(msg)
        elif kind == "end_session":
            await self.end_session(enrich=bool(msg.get("enrich")), use_cloud=bool(msg.get("use_cloud")),
                                   consent=bool(msg.get("consent")))
        elif kind == "ping":
            await self._emit({"type": "pong", "t": msg.get("t")})

    async def configure(self, cfg: dict) -> None:
        if "mode" in cfg or "scenario" in cfg or "difficulty" in cfg or "level" in cfg:
            self.conv.configure(cfg.get("mode"), cfg.get("scenario"), cfg.get("difficulty"), cfg.get("level"))
            if "mode" in cfg and "sensitivity" not in cfg:
                self.policy.configure(MODES[self.conv.mode].default_sensitivity)
            self.policy.configure(tutor=(self.conv.mode == "tutor"))
        if cfg.get("sensitivity"):
            self.policy.configure(cfg["sensitivity"])
        if cfg.get("voice"):
            self.voice = cfg["voice"]
        if cfg.get("barge_in_mode") in ("headphones", "speakers", "off"):
            self.barge_in_mode = cfg["barge_in_mode"]
        if "end_of_turn_ms" in cfg:
            self.turn_cfg.end_of_turn_ms = float(max(200, min(2000, cfg["end_of_turn_ms"])))
        if "cloud_enabled" in cfg:
            self.cloud_prefs.enabled = bool(cfg["cloud_enabled"])
        if cfg.get("cloud_provider") in ("anthropic", "openai"):
            self.cloud_prefs.provider = cfg["cloud_provider"]
        if "cloud_turn_corrections" in cfg:
            self.cloud_prefs.allow_turn_corrections = bool(cfg["cloud_turn_corrections"])
        await self._emit({"type": "config", "config": self.public_config()})

    def public_config(self) -> dict:
        return {"mode": self.conv.mode, "scenario": self.conv.scenario, "difficulty": self.conv.difficulty,
                "level": self.conv.level, "sensitivity": self.policy.sensitivity, "voice": self.voice,
                "barge_in_mode": self.barge_in_mode, "end_of_turn_ms": self.turn_cfg.end_of_turn_ms,
                "cloud_enabled": self.cloud_prefs.enabled, "cloud_provider": self.cloud_prefs.provider,
                "cloud_turn_corrections": self.cloud_prefs.allow_turn_corrections, "session_id": self.id}

    async def start(self) -> None:
        ok, why = self.engines.ready()
        if not ok:
            await self._emit({"type": "error", "fatal": False, "message": f"Some engines are not ready: {why}"})
            if not (self.engines.llm and self.engines.status.get("llm", {}).get("ok")):
                return
        self.active = True
        self.detector.reset()
        self.vad.reset()
        if self.store and not self._started_store:
            self._started_store = True
            await self.store.start_session(self.id, self.conv.mode, self.conv.scenario,
                                           getattr(self.engines.llm, "model_id", ""))
        await self._set_state("listening")
        if not self.conv.turns:
            await self._start_response(prompts.opening_note(self.conv.mode, self.conv.scenario), "opening", TurnMetrics(kind="opening"))

    async def stop(self) -> None:
        self.active = False
        await self._cancel_response("stopped", unheard_merge=False)
        for t in (self._partial_task, self._idle_task, self._echo_task):
            if t:
                t.cancel()
        self.detector.reset()
        await self._set_state("idle")

    async def close(self) -> None:
        await self.stop()
        if self.store and self.conv.turns:
            await self.store.save_transcript(self.id, self.conv.transcript(), build_review(self.conv).stats)
        for t in list(self._bg):
            t.cancel()

    # ================================================================== audio input
    async def handle_audio(self, data: bytes) -> None:
        if not self.active:
            return
        for frame in self.assembler.push(data):
            await self._process_frame(frame)

    async def _process_frame(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        audible = self._ai_audible(now)
        if audible and self.barge_in_mode == "off":
            # Half-duplex: ignore the microphone while the coach is talking.
            if self.detector.state is not TState.SILENCE:
                self.detector.reset()
            return
        self._apply_detector_sensitivity()
        t0 = time.perf_counter()
        prob = self.vad.prob(frame)
        self._vad_times.append(time.perf_counter() - t0)
        if len(self._vad_times) > 200:
            self._vad_times = self._vad_times[-200:]
        prev_state = self.detector.state
        events = self.detector.process(prob, frame)
        if self.detector.state is not TState.SILENCE and self.detector.last_voice_ms == self.detector.t_ms:
            self._last_voice_wall = now
            if prev_state is TState.SILENCE:
                self._speech_first_voice_wall = now
        for ev in events:
            if isinstance(ev, SpeechStarted):
                await self._on_speech_started(now, audible)
            elif isinstance(ev, SilenceReached):
                self._on_silence(ev)
            elif isinstance(ev, SpeechEnded):
                await self._on_speech_ended(ev, now)

    async def _on_speech_started(self, now: float, audible: bool) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        self._idle_nudges = 0
        rt = self.response
        if rt is not None and audible:
            detect_s = now - (self._speech_first_voice_wall or now)
            if self.barge_in_mode == "speakers":
                await self._emit({"type": "duck", "on": True, "turn_id": rt.id})
                self._bargein = {"t_confirm": now, "detect_s": detect_s, "turn": rt, "resolved": asyncio.Event(), "echo": False}
                self._echo_task = self._spawn(self._verify_bargein(self._bargein))
            else:
                await self._barge_in(now, detect_s)
        elif rt is not None and not rt.playback_done:
            # The coach was still thinking (nothing audible yet): the user carried on talking.
            await self._cancel_response("user continued", unheard_merge=True)
        await self._set_state("user_speaking")
        if self.s.stt_partials and (self._partial_task is None or self._partial_task.done()):
            self._partial_task = self._spawn(self._partials_loop())

    async def _barge_in(self, t_confirm: float, detect_s: float) -> None:
        rt = self.response
        if rt is None:
            return
        await self._cancel_response("barge-in", unheard_merge=True)
        m = TurnMetrics(turn_id=rt.id, kind="barge_in", bargein_detect_ms=ms(detect_s),
                        bargein_cancel_ms=ms(time.perf_counter() - t_confirm))
        self._pending_bargein_metrics = m
        await self._emit({"type": "barge_in", "turn_id": rt.id, "metrics": m.to_dict()})

    async def _verify_bargein(self, b: dict) -> None:
        """Speakers mode: the coach is ducked; decide whether the 'speech' is just our own echo."""
        try:
            await asyncio.sleep(0.45)  # gather a little more audio
            audio = self.detector.current_audio()
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(self.engines.stt_executor, self.engines.stt.transcribe, audio, False)
            rt: ResponseTurn = b["turn"]
            ai_recent = " ".join(rt.sentences[-3:])
            overlap = word_overlap(res.text, ai_recent)
            if not res.text or overlap >= self.s.echo_similarity_threshold:
                b["echo"] = True
                self.detector.reset()
                await self._emit({"type": "duck", "on": False, "turn_id": rt.id, "echo": True,
                                  "heard": res.text, "overlap": round(overlap, 2)})
                if rt.metrics:
                    rt.metrics.echo_rejected += 1
                await self._set_state("speaking" if self._ai_audible() else "listening")
            else:
                await self._barge_in(b["t_confirm"], b["detect_s"])
        finally:
            b["resolved"].set()

    def _on_silence(self, ev: SilenceReached) -> None:
        # Speculatively start the final transcription during the end-of-turn wait.
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self.engines.stt_executor, self.engines.stt.transcribe, ev.audio, False)
        self._spec = (fut, ev.voiced_frames)

    async def _partials_loop(self) -> None:
        loop = asyncio.get_running_loop()
        interval = self.s.stt_partial_interval_ms / 1000
        try:
            while self.detector.state is TState.SPEECH:
                await asyncio.sleep(interval)
                if self.detector.state is not TState.SPEECH or self.detector.silence_ms > 100:
                    continue  # a pause: leave the STT thread free for the final transcript
                audio = self.detector.current_audio()
                if len(audio) < 8000:
                    continue
                res = await loop.run_in_executor(self.engines.stt_executor, self.engines.stt.transcribe, audio[-16000 * 12:], False)
                if res.text and self.detector.state is TState.SPEECH:
                    await self._emit({"type": "partial_transcript", "text": res.text})
        except asyncio.CancelledError:
            pass

    async def _on_speech_ended(self, ev: SpeechEnded, now: float) -> None:
        m = TurnMetrics(kind="user")
        pending = self._pending_bargein_metrics
        if pending:
            m.bargein_detect_ms, m.bargein_cancel_ms = pending.bargein_detect_ms, pending.bargein_cancel_ms
            m.bargein_client_stop_ms = pending.bargein_client_stop_ms
            self._pending_bargein_metrics = None
        speech_end_wall = self._last_voice_wall or now
        m.eot_ms = ms(now - speech_end_wall)
        if self._vad_times:
            m.vad_frame_ms = round(1000 * sum(self._vad_times) / len(self._vad_times), 3)

        if self._bargein and not self._bargein["resolved"].is_set():
            await self._bargein["resolved"].wait()
        if self._bargein and self._bargein.get("echo"):
            self._bargein = None
            self._spec = None
            return
        self._bargein = None
        if self._partial_task:
            self._partial_task.cancel()

        await self._set_state("thinking")
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        spec, self._spec = self._spec, None
        res = None
        if spec and spec[1] == ev.voiced_frames:
            res = await spec[0]
            self.engines.stt.on_final(res.text)
            m.stt_speculative_hit = True
        else:
            if spec:
                spec[0].cancel()
            res = await loop.run_in_executor(self.engines.stt_executor, self.engines.stt.transcribe, ev.audio, True)
            m.stt_speculative_hit = False
        m.stt_wait_ms = ms(time.perf_counter() - t0)
        m.stt_compute_ms = ms(res.latency_s)
        m.stt_audio_s = round(res.audio_s, 2)
        if not res.text:
            await self._emit({"type": "partial_transcript", "text": ""})
            await self._set_state("speaking" if self._ai_audible() else "listening")
            return
        await self._on_user_text(res.text, m, speech_end_wall)

    # ================================================================== turns
    async def _on_user_text(self, text: str, m: TurnMetrics, speech_end_wall: float | None = None) -> None:
        if self._pending_prefix:
            text = f"{self._pending_prefix} {text}"
            self._pending_prefix = ""
        await self._start_response(text, "user", m, speech_end_wall)

    async def _start_response(self, text: str, kind: str, m: TurnMetrics, speech_end_wall: float | None = None) -> None:
        await self._cancel_response("superseded", unheard_merge=False)
        if self._idle_task:
            self._idle_task.cancel()
        self._turn_counter += 1
        m.turn_id = self._turn_counter
        rt = ResponseTurn(id=self._turn_counter, kind=kind, metrics=m, speech_end_wall=speech_end_wall)
        self.response = rt
        rt.task = asyncio.create_task(self._run_response(rt, text))

    def _coach_note(self, text: str) -> tuple[list, str, dict]:
        issues = rules.detect(text)
        decision = self.policy.decide(issues)
        notes = []
        if decision.strategy is Strategy.RECAST and decision.issue:
            notes.append(prompts.recast_note(decision.issue.original, decision.issue.suggestion))
        elif decision.strategy is Strategy.BRIEF and decision.issue:
            notes.append(prompts.brief_note(decision.issue.original, decision.issue.suggestion, decision.issue.explanation))
        if self.conv.short_answer_streak() >= 1 and len(text.split()) <= 3:
            notes.append(prompts.SHORT_ANSWERS_NOTE)
        if self.conv.recent_questions:
            notes.append("don't repeat these questions you already asked: " + " / ".join(self.conv.recent_questions[-2:]))
        info = {"strategy": decision.strategy.value, "reason": decision.reason,
                "issue": decision.issue.to_dict() if decision.issue else None}
        return issues, prompts.combine_notes(notes), info

    async def _run_response(self, rt: ResponseTurn, text: str) -> None:
        m = rt.metrics
        try:
            if rt.kind == "user":
                issues, note, info = self._coach_note(text)
                self.conv.add_user(text, issues, note=note)
                await self._emit({"type": "user_final", "turn_id": rt.id, "text": text,
                                  "issues": [i.to_dict() for i in issues], "correction": info})
                self._maybe_cloud_correction(text)
            else:  # opening / idle: an app-generated prompt, hidden from the transcript
                placeholder = "(The learner is listening.)" if rt.kind == "opening" else "(The learner is quiet.)"
                self.conv.add_user(placeholder, note=text, synthetic=True)

            await self._set_state("thinking")
            messages = self.conv.build_messages()
            llm = self.engines.llm
            params = GenParams(max_tokens=self.s.llm_max_tokens, temperature=self.s.llm_temperature, top_p=self.s.llm_top_p)
            queue: asyncio.Queue = asyncio.Queue(maxsize=3)
            rt.t_request = time.perf_counter()

            producer = asyncio.create_task(self._produce_text(rt, llm, messages, params, queue))
            consumer = asyncio.create_task(self._speak(rt, queue))
            try:
                await asyncio.gather(producer, consumer)
            finally:
                for t in (producer, consumer):
                    t.cancel()
            if rt.cancel.cancelled:
                return
            rt.text_done = True
            spoken = " ".join(rt.sentences)
            self.conv.add_assistant(spoken)
            snap = memory_snapshot()
            m.rss_mb, m.sys_available_gb = snap.get("rss_mb"), snap.get("sys_available_gb")
            await self._emit({"type": "assistant_done", "turn_id": rt.id, "text": spoken,
                              "topic": self.conv.topic, "level": self.conv.level})
            if not rt.sentences:
                rt.playback_done = True
                await self._after_playback(rt)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            log.exception("response failed")
            await self._emit({"type": "error", "message": f"Reply failed: {type(e).__name__}: {e}"})
            if self.response is rt:
                self.response = None
            await self._set_state("listening")

    async def _produce_text(self, rt: ResponseTurn, llm, messages, params: GenParams, queue: asyncio.Queue) -> None:
        m = rt.metrics
        splitter = SentenceSplitter(first_chunk_min_chars=self.s.tts_first_chunk_min_chars)
        think = ThinkFilter()
        first = True
        try:
            async for chunk in llm.stream(messages, params, rt.cancel):
                if rt.cancel.cancelled:
                    break
                if chunk.done:
                    st = chunk.stats or {}
                    m.llm_ttft_ms = ms(st.get("ttft_s"))
                    m.llm_tokens = st.get("generated_tokens")
                    tps = st.get("tokens_per_s")
                    m.llm_tokens_per_s = round(tps, 1) if tps else None
                    m.llm_prompt_tokens = st.get("prompt_tokens")
                    m.llm_reused_prefix_tokens = st.get("reused_prefix_tokens")
                    m.llm_peak_memory_gb = st.get("peak_memory_gb")
                    break
                piece = think.push(chunk.text)
                if not piece:
                    continue
                rt.full_text += piece
                await self._emit({"type": "assistant_delta", "turn_id": rt.id, "text": piece})
                for sentence in splitter.push(piece):
                    sentence = clean_for_speech(sentence)
                    if sentence:
                        if first:
                            m.first_sentence_ms = ms(time.perf_counter() - rt.t_request)
                            first = False
                        await queue.put(sentence)
            tail = think.flush()
            rt.full_text += tail
            for sentence in splitter.push(tail) + splitter.flush():
                sentence = clean_for_speech(sentence)
                if sentence and not rt.cancel.cancelled:
                    if first:
                        m.first_sentence_ms = ms(time.perf_counter() - rt.t_request)
                        first = False
                    await queue.put(sentence)
        finally:
            await queue.put(None)

    async def _speak(self, rt: ResponseTurn, queue: asyncio.Queue) -> None:
        m = rt.metrics
        tts = self.engines.tts
        seq = 0
        while True:
            sentence = await queue.get()
            if sentence is None or rt.cancel.cancelled:
                return
            idx = len(rt.sentences)
            rt.sentences.append(sentence)
            rt.durations.append(0.0)
            await self._emit({"type": "assistant_sentence", "turn_id": rt.id, "index": idx, "text": sentence})
            t_synth = time.perf_counter()
            first_chunk = True
            async for audio in iterate_in_thread(lambda s=sentence: tts.synthesize(s, self.voice), self.engines.tts_executor, rt.cancel):
                if rt.cancel.cancelled:
                    return
                now = time.perf_counter()
                if first_chunk and idx == 0:
                    m.tts_first_audio_ms = ms(now - t_synth)
                    m.tts_rtf = round((now - t_synth) / max(audio.duration_s, 1e-3), 3)
                first_chunk = False
                rt.durations[idx] += audio.duration_s
                await self.send_bytes(pack_audio(rt.id, seq, audio.sample_rate, idx, audio.pcm16))
                seq += 1
                if rt.first_audio_sent is None:
                    rt.first_audio_sent = now
                    if rt.speech_end_wall:
                        m.e2e_server_ms = ms(now - rt.speech_end_wall)
                    await self._set_state("speaking")

    async def _cancel_response(self, reason: str, unheard_merge: bool) -> None:
        rt = self.response
        if rt is None:
            return
        now = time.perf_counter()
        if rt.text_done and (rt.playback_done or not rt.playing(now)):
            # Already finished (the client's "ended" report may simply not have arrived yet).
            self.response = None
            if not rt.playback_done:
                rt.playback_done = True
                self.metrics_log.append(rt.metrics.to_dict())
                await self._emit({"type": "metrics", "turn": rt.metrics.to_dict()})
            return
        self.response = None
        rt.cancel.cancel(reason)
        heard = rt.heard_text(now) if (rt.first_audio_sent or rt.play_start) else ""
        await self._emit({"type": "stop_audio", "turn_id": rt.id, "reason": reason, "heard": heard})
        if rt.task and not rt.task.done():
            rt.task.cancel()
        rt.metrics.cancelled = True
        if rt.text_done:
            # Reply was fully generated but playback was cut short: keep only what was heard.
            self._truncate_last_assistant(heard)
        elif heard:
            self.conv.add_assistant(heard, interrupted=True)
        elif unheard_merge and rt.kind == "user":
            # Nothing was heard: merge the unanswered utterance with what comes next.
            prev = self.conv.pop_unanswered_user()
            if prev:
                self._pending_prefix = prev.text
        await self._emit({"type": "assistant_interrupted", "turn_id": rt.id, "heard": heard, "reason": reason})

    def _truncate_last_assistant(self, heard: str) -> None:
        for t in reversed(self.conv.turns):
            if t.role == "assistant":
                if heard and heard != t.text:
                    t.text, t.interrupted = heard.rstrip() + " —", True
                elif not heard:
                    self.conv.turns.remove(t)
                return

    async def _on_playback(self, msg: dict) -> None:
        rt = self.response
        now = time.perf_counter()
        event = msg.get("event")
        if event == "stopped":
            # Client acknowledges a stop_audio: how long silencing took on its side.
            if msg.get("stop_ms") is not None:
                if self._pending_bargein_metrics:
                    self._pending_bargein_metrics.bargein_client_stop_ms = round(float(msg["stop_ms"]), 2)
                await self._emit({"type": "metric_update", "bargein_client_stop_ms": msg["stop_ms"]})
            return
        if rt is None or msg.get("turn_id") != rt.id:
            return
        if event == "started" and rt.play_start is None:
            rt.play_start = now
            if rt.speech_end_wall:
                rt.metrics.e2e_audible_ms = ms(now - rt.speech_end_wall)
        elif event == "ended":
            if rt.text_done:
                rt.playback_done = True
                await self._after_playback(rt)

    async def _after_playback(self, rt: ResponseTurn) -> None:
        row = rt.metrics.to_dict()
        self.metrics_log.append(row)
        await self._emit({"type": "metrics", "turn": row})
        if self.store:
            self._spawn(self.store.save_metrics(self.id, row))
        if self.response is rt:
            self.response = None
        if self.active:
            await self._set_state("listening")
            if self.s.idle_prompt_s > 0 and self._idle_nudges < 1:
                self._idle_task = self._spawn(self._idle_timer())

    async def _idle_timer(self) -> None:
        try:
            await asyncio.sleep(self.s.idle_prompt_s)
            if self.active and self.response is None and self.detector.state is TState.SILENCE:
                self._idle_nudges += 1
                await self._start_response(prompts.IDLE_NOTE, "idle", TurnMetrics(kind="idle"))
        except asyncio.CancelledError:
            pass

    # ================================================================== cloud + review
    def _maybe_cloud_correction(self, text: str) -> None:
        if not self.cloud:
            return
        decision = self.cloud.decide("turn_correction", self.cloud_prefs)
        if not decision.allowed:
            return
        turn = self.conv.turns[-1]

        async def run() -> None:
            from ..conversation.prompts import REVIEW_SYSTEM

            try:
                out = await self.cloud.complete(decision, "turn_correction", REVIEW_SYSTEM,
                                                f"Learner's lines:\n- {text}\n\nReturn the JSON now.", 600)
            except Exception as e:  # noqa: BLE001
                await self._emit({"type": "cloud_error", "message": str(e)})
                return
            data = parse_llm_json(out) or {}
            added = []
            for item in data.get("mistakes") or []:
                orig, corr = (item or {}).get("original", ""), (item or {}).get("correction", "")
                if orig and corr and orig.lower() in text.lower() and orig.lower() not in {i.original.lower() for i in turn.issues}:
                    iss = rules.Issue("other", orig, corr, item.get("explanation", ""), "minor", source="cloud")
                    turn.issues.append(iss)
                    added.append(iss.to_dict())
            if added:
                await self._emit({"type": "cloud_correction", "text": text, "issues": added})

        self._spawn(run())

    async def end_session(self, enrich: bool = False, use_cloud: bool = False, consent: bool = False) -> dict:
        await self.stop()
        review = build_review(self.conv)
        local_failed = False
        if enrich and self.conv.user_turns() and self.engines.llm:
            try:
                out = await asyncio.wait_for(self.engines.llm.complete(review_messages(self.conv), GenParams(max_tokens=500, temperature=0.2)), 90)
                data = parse_llm_json(out)
                local_failed = data is None
                enrich_review(review, self.conv, data, "local-llm")
            except Exception:  # noqa: BLE001
                log.exception("local review enrichment failed")
                local_failed = True
        if self.cloud and self.conv.user_turns() and (use_cloud or local_failed):
            decision = self.cloud.decide("review", self.cloud_prefs, explicit_request=use_cloud, consent=consent,
                                         local_failed=local_failed)
            if decision.allowed:
                msgs = review_messages(self.conv)
                try:
                    out = await self.cloud.complete(decision, "review", msgs[0]["content"], msgs[1]["content"])
                    enrich_review(review, self.conv, parse_llm_json(out), f"cloud:{decision.provider}")
                except Exception as e:  # noqa: BLE001
                    review.notes.append(f"Cloud analysis failed: {e}")
            elif use_cloud:
                review.notes.append(f"Cloud analysis not run: {decision.reason}.")
        result = {**review.to_dict(), "session_id": self.id, "transcript": self.conv.transcript(),
                  "metrics": self.metrics_log}
        if self.store:
            await self.store.save_transcript(self.id, self.conv.transcript(), review.stats)
            await self.store.save_review(self.id, result)
        await self._emit({"type": "review", "review": result})
        return result

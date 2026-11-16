# Architecture

## Goals, in priority order

1. Feel conversational: short time from the end of your speech to the coach's voice, and instant interruptions.
2. Run entirely on a fanless MacBook Air M3 (24 GB) without thermal or memory trouble.
3. Coach well: natural British English, gentle and grounded corrections.
4. Keep every component replaceable.

## Process layout

One Python process (FastAPI + uvicorn, a single asyncio loop) and one browser tab.

```
┌──────────────────────── browser ───────────────────────────┐
│ getUserMedia (AEC/NS/AGC) → AudioWorklet: 48k→16k, 512-sample│
│ Int16 frames ─────────────────────────────┐                  │
│ Web Audio player ◄─ [turn|seq|rate|idx|PCM] │  JSON events    │
└─────────────────────────────────────────────┼──────────────────┘
                                              │ WebSocket /ws
┌──────────────────────── server ─────────────┼──────────────────┐
│ Session (per connection, asyncio)           ▼                  │
│   FrameAssembler → SileroVAD (onnxruntime, ~0.1–0.5 ms/frame)  │
│   → TurnDetector (pure state machine)                          │
│      SpeechStarted ─► barge-in / partial STT loop               │
│      SilenceReached ─► speculative final STT  ─┐  STT thread    │
│      SpeechEnded ──► reuse speculation or STT ◄┘  (1 worker)    │
│   → correction detector + policy → coach note                  │
│   → ConversationState.build_messages()                         │
│   → ResponseTurn                                               │
│        producer: LLM stream ─► ThinkFilter ─► SentenceSplitter │
│                   (MLX thread)        bounded queue (3)        │
│        consumer: TTS per sentence (TTS thread) ─► WS bytes     │
│ Engines (process-wide): llm, stt, tts, vad + executors         │
│ CloudRouter (per session) · Store (SQLite via to_thread)       │
└────────────────────────────────────────────────────────────────┘
```

### Threads

| Work | Where | Why |
|---|---|---|
| VAD | event loop, inline | ~0.1–0.5 ms per 32 ms frame, cheaper than a thread hop |
| STT | `Engines.stt_executor` (1 thread) | CPU-bound; serialises partials, speculation and finals |
| LLM (MLX) | `MLXBackend._ex` (1 thread) | MLX streams are per-thread; one generation at a time |
| TTS | `Engines.tts_executor` (1 thread) | CPU-bound ONNX (Piper) or a `say` subprocess |
| SQLite | `asyncio.to_thread` | tiny writes, never on the loop |

`llm/base.py:iterate_in_thread` bridges a blocking iterator to an async generator. The worker
checks a shared `CancelToken` between items, so a cancelled MLX generation stops within one token.

### CPU/GPU split on the M3

The LLM runs on the GPU (MLX, Metal). STT (faster-whisper int8) and TTS (Piper on onnxruntime) run on the
CPU, so the two halves of a turn don't queue on the same Metal command stream. `whispercpp` can use Metal
instead; benchmark both on your machine (`COACH_STT_BACKEND`). `OMP_NUM_THREADS=4` and `COACH_STT_THREADS=4`
keep CPU engines to the performance cores.

## A turn, step by step

1. **Capture.** The worklet box-filters the context rate down to 16 kHz and posts 512-sample Int16 frames (32 ms).
2. **VAD + turn detection.** Silero gives a speech probability per frame. The detector uses hysteresis
   (enter at `threshold`, leave below `threshold-0.15`). It needs `min_speech_ms` of voiced frames to
   confirm speech, which rejects clicks and coughs, and keeps `pre_roll_ms` of audio from before the start.
3. **Partial transcripts.** While you talk, the growing buffer is re-transcribed every ~700 ms (Whisper is not
   a streaming model, so this is chunked pseudo-streaming). Partials are skipped once you pause, keeping the
   STT thread free.
4. **Speculative final transcription.** After `stt_speculative_ms` (240 ms) of silence, the final
   transcription starts *while* the detector is still waiting out `end_of_turn_ms` (560 ms). If you don't
   resume, the result is used and STT disappears from the critical path (`stt_speculative_hit`). If you do
   resume, it is discarded.
5. **Coaching.** The rule detector finds issues, the policy picks a strategy, and a `[Coach note: …]` is
   appended to *your latest message only*. The system prompt stays byte-identical across turns, so the MLX
   KV cache for it (and earlier history) is reused (`reused_prefix_tokens`).
6. **Generation.** Tokens stream into `SentenceSplitter`. The first clause is flushed early at a comma once
   it is ≥ 40 characters; after that, whole sentences go out. `max_tokens=110` keeps replies to one to
   three spoken sentences.
7. **Speech.** Each sentence is synthesised on the TTS thread while the LLM keeps generating. Audio goes out
   with a 16-byte header `(turn_id, seq, sample_rate, sentence_idx)`; the browser schedules chunks back to back.
8. **Playback reports.** The browser sends `playback started/ended`. The server uses these for
   `e2e_audible_ms`, to know when the coach is audible (barge-in sensitivity), and to estimate which
   sentences you actually heard.

## Barge-in

```
user speech confirmed while coach audible
  ├─ headphones: _cancel_response()
  │     1. send {"type":"stop_audio","turn_id"}         ← first, before any cleanup
  │     2. CancelToken.cancel() → MLX thread stops at the next token
  │     3. producer/consumer tasks cancelled (not awaited)
  │     4. history keeps only the sentences already heard, marked "—" (interrupted)
  ├─ speakers: duck to 15 % → transcribe ~0.5 s → word overlap with the coach's recent
  │            sentences ≥ 0.55 (or empty) ⇒ echo: unduck, ignore; otherwise ⇒ cancel as above
  └─ off: microphone ignored while the coach is audible
```

The browser drops any audio whose `turn_id` has been stopped, so packets already in flight can never
play. If you start talking while the coach is still *thinking* (nothing audible yet), the reply is
cancelled and your two utterances are merged into one turn.

Self-interruption is handled with defence in depth: browser AEC, stricter thresholds while the coach is
audible, the echo check in speakers mode, and headphones as the recommended setup.

## Conversation management

* `ConversationState` holds the turns, current topic (keyword classifier), topics covered, recent questions
  (passed in the note so the coach doesn't repeat itself) and an estimated CEFR level (B1/B2/C1, re-estimated
  every four turns from utterance length, lexical variety and error rate).
* Context is the system prompt plus the most recent messages within `context_max_messages` /
  `context_char_budget`. When older turns are dropped, a one-line "earlier you talked about …" is added.
* The coach opens the conversation, and re-engages once after `idle_prompt_s` of silence. These
  app-generated prompts are hidden from your transcript and review.

## Replaceable components

| Interface | Implementations | Selected by |
|---|---|---|
| `llm.base.LLMBackend` (`stream`, `load`, `unload`) | `MLXBackend`, `OllamaBackend`, `OpenAICompatBackend`, `MockLLM` | `COACH_LLM_BACKEND`, `/api/llm/select` at runtime |
| `stt.base.STTBackend` (`transcribe`) | `FasterWhisperSTT`, `WhisperCppSTT`, `MockSTT` | `COACH_STT_BACKEND` |
| `tts.engines.TTSBackend` (`synthesize` → chunks) | `PiperTTS`, `MacSayTTS`, `MockTTS` | `COACH_TTS_BACKEND`, voice per session |
| `audio.vad.VAD` (`prob`) | `SileroVAD`, `EnergyVAD` | `COACH_VAD_BACKEND` |
| `cloud.providers.CloudProvider` | `AnthropicProvider`, `OpenAIProvider` | `COACH_CLOUD_PROVIDER` / UI |

Model switching (`Engines.switch_llm`) loads and warms up the new model before swapping and unloading the
old one, so a failed load leaves the working model in place. Peak memory briefly holds both (e.g. 4B + 7B
≈ 7 GB of weights, fine within 24 GB).

## Memory budget (approximate weight sizes, not measurements)

| Component | Approx. resident |
|---|---|
| LLM 1.5B / 4B / 7B at 4-bit | ~1 / ~2.5 / ~4.5 GB + KV cache |
| Whisper base.en (int8) | ~0.2 GB |
| Piper medium voice | ~0.1 GB per loaded voice (two cached) |
| Silero VAD | ~2 MB |

Measured RSS, free system memory and MLX peak memory are reported per turn in the latency panel.

## WebSocket protocol

Client → server: binary Int16 16 kHz PCM; JSON `start{config}`, `config{…}`, `stop`, `text_input{text}`,
`playback{event: started|ended|stopped, turn_id, stop_ms}`, `end_session{enrich, use_cloud, consent}`, `ping`.

Server → client: binary audio (see above); JSON `hello`, `config`, `state`, `partial_transcript`,
`user_final{issues, correction}`, `assistant_delta`, `assistant_sentence`, `assistant_done`, `stop_audio`,
`assistant_interrupted{heard}`, `barge_in{metrics}`, `duck`, `metrics{turn}`, `metric_update`,
`cloud_status{active, provider, purpose, chars_sent}`, `cloud_correction`, `review`, `error`.

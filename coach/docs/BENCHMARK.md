# Benchmarking

Every number the app reports is measured on the machine it runs on. This repository contains **no M3
results**: it was developed in a Linux container without Apple hardware or access to the model weights.
Please run the steps below on the MacBook Air and commit the JSON files from `benchmarks/results/`.

## Before you start

* Plug in to power and close heavy apps. The Air is fanless, so a long run throttles; note the run length.
* `./scripts/setup_models.py` has completed, and `curl -s localhost:8765/api/health` says `"status": "ok"`.
* Test speech is synthesised with macOS `say -v Daniel`, so runs are repeatable without a microphone.

## 1. Components

```bash
.venv/bin/python scripts/benchmark.py components --runs 5
.venv/bin/python scripts/benchmark.py components --runs 3 --profiles small medium large
```

Measures, with median / p90 / min / max:

| Key | Meaning |
|---|---|
| `vad_frame_ms` | Silero inference per 32 ms frame |
| `stt_ms`, `stt_rtf_x1000`, `stt_word_accuracy` | final-transcription latency, real-time factor ×1000, crude word recall vs the script |
| `tts_first_audio_ms`, `tts_rtf_x1000` | time to the first audio chunk for a short sentence |
| `llm.<profile>.llm_ttft_ms` | time to first token with the real coach system prompt and growing history |
| `llm.<profile>.llm_tokens_per_s` | decode speed |
| `llm.<profile>.reused_prefix_tokens` | MLX KV-cache reuse across turns (should be most of the prompt) |
| `memory`, `peak_memory_gb` | process RSS, system free memory, MLX peak |

To compare STT engines, run it twice with `COACH_STT_BACKEND=faster_whisper` and `=whispercpp`
(and `COACH_STT_MODEL=tiny.en` vs `base.en`).

## 2. End-to-end over the WebSocket

```bash
./scripts/start.sh                       # terminal 1
.venv/bin/python scripts/benchmark.py e2e --turns 6     # terminal 2
```

The client streams each utterance in real time (32 ms frames), exactly like the browser, then records:

* `client_e2e_first_audio_ms`: from the last speech frame sent to the first reply audio received.
  **This is the headline number for the ~1 s target.** It includes the end-of-turn silence (`COACH_END_OF_TURN_MS`).
* `server_summary`: every stage from `/api/metrics` (`eot_ms`, `stt_wait_ms`, `llm_ttft_ms`, `first_sentence_ms`,
  `tts_first_audio_ms`, `e2e_server_ms`, `e2e_audible_ms`, `bargein_*`).
* `bargein`: interrupts every second reply mid-playback; time from speech onset to `stop_audio` received.

The browser also shows these per turn in **Latency diagnostics**. Browser-side audible latency adds the Web Audio
output buffer, roughly 10–40 ms depending on the device.

## 3. Real browser

`./scripts/e2e_browser.sh` drives headless Chromium with a fake microphone (looped WAV) against the real UI.
By default it uses mock STT/LLM/TTS and the real Silero VAD; export `COACH_LLM_BACKEND=mlx` and so on to run it with
the real engines.

## Reading the result: where the time goes

```
end of speech ─┬─ end_of_turn wait (COACH_END_OF_TURN_MS, default 560 ms)   ← deliberate
               │    └─ speculative STT runs inside this window
               ├─ STT remainder (0 when speculation hits)
               ├─ LLM prefill + first ~8–15 tokens (first clause)
               ├─ TTS for the first clause
               └─ network + browser audio buffer
```

Levers, roughly in order of impact:

1. `COACH_END_OF_TURN_MS`: 450 feels snappier but cuts slow speakers off; 700 is more patient.
2. LLM size: `small` decodes much faster than `large`. Coaching quality drops with size.
3. `COACH_TTS_FIRST_CHUNK_MIN_CHARS`: smaller flushes the first clause sooner.
4. `COACH_STT_MODEL=tiny.en` if `stt_speculative_hit` is often false and `stt_wait_ms` is large.
5. KV-cache reuse (`COACH_MLX_PROMPT_CACHE=true`): check `reused_prefix_tokens` is large.

## Harness validation (Linux container, mock models: not a performance result)

To check the harness itself, it was run in the build container (x86-64, 4 vCPU, no GPU) with **mock**
STT/LLM/TTS and the **real** Silero VAD:

* Silero VAD: median 0.1 ms per 32 ms frame (component mode), 0.5 ms inside the live server.
* `client_e2e_first_audio_ms` 600–696 ms (n=4), of which `eot_ms` ≈ 576 ms was the deliberate end-of-turn
  wait. So the orchestration adds roughly 50–100 ms on top of the models' own time.
* Barge-in, speech onset to `stop_audio` received: 206 and 238 ms (dominated by the 200 ms minimum-speech
  guard); server-side cancellation 0.7–1.1 ms.

These numbers say nothing about Whisper, Qwen or Piper speed on the M3; only a run on the Mac does.

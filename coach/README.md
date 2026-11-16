# British Conversation Coach

A local-first voice app for practising natural British English. You talk; a
friendly British "coach" talks back, keeps the conversation going, quietly
corrects your English, and gives you a review at the end. Speech recognition,
the language model and speech synthesis all run on your Mac. There are no API costs.

```
Browser mic ─► AudioWorklet (16 kHz) ─► WebSocket ─► Silero VAD ─► turn detector
                                                          │ speculative STT on pause
                                                          ▼
Browser speaker ◄─ Web Audio player ◄─ WebSocket ◄─ TTS (per sentence) ◄─ LLM token stream
                         ▲                                                  (MLX, 4-bit)
                         └──── barge-in: stop_audio + cancel, within one token ─────┘
```

Target hardware: MacBook Air M3, 24 GB. Linux works too (with Ollama or llama.cpp instead of MLX).

## Quick start (macOS, Apple Silicon)

```bash
cd coach
./scripts/install.sh        # Python venv, MLX, Piper, builds the UI, downloads models (~3-4 GB)
./scripts/start.sh          # loads + warms up the models, then serves http://127.0.0.1:8765
```

Open **http://127.0.0.1:8765** in Safari or Chrome, allow the microphone, and press **Start conversation**.
Use **headphones** for the best experience. See [Interruptions and echo](#interruptions-and-echo).

Prerequisites: Python 3.11+ (`brew install python@3.12`) and Node 18+ (`brew install node`) for building the UI.

<details><summary>Manual install / other options</summary>

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt -r requirements-mac.txt
.venv/bin/pip install -r requirements-tts.txt          # Piper (GPL-3.0, see Licences)
(cd frontend && npm ci && npm run build)
cp .env.example .env
.venv/bin/python scripts/setup_models.py --verify-only  # check repos + licences first
.venv/bin/python scripts/setup_models.py                # VAD, Whisper base.en, 2 voices, Qwen3-4B (MLX)
```

* No Piper? Set `COACH_TTS_BACKEND=macos_say` to use the built-in British voices ("Daniel", etc.). Nothing to download.
* Ollama instead of MLX: `COACH_LLM_BACKEND=ollama`, then `setup_models.py --llm-backend ollama`.
* llama.cpp: run `llama-server -m model.gguf --port 8080` and set `COACH_LLM_BACKEND=openai_compat`.
* UI development: `cd frontend && npm run dev` (proxies to the Python server on :8765).
</details>

## What you get

| Feature | Where |
|---|---|
| Real-time two-way voice, sentence-streamed replies | `britcoach/pipeline/session.py` |
| Barge-in: interrupt at any moment, stale audio never plays | session + `frontend/src/lib/audio.ts` |
| Coach persona that drives the conversation (follow-ups, new topics, re-engages after silence) | `britcoach/conversation/` |
| Corrections: **A** natural recast · **B** brief tip for repeated/important errors · **C** end-of-session review | `britcoach/coaching/` |
| Modes: Free Conversation, British Social Practice, Role-play (easy/normal/challenging), English Tutor, Session Review | UI, `conversation/modes.py` |
| Live latency diagnostics for every stage, plus memory | UI side panel, `/api/metrics` |
| Model, voice, correction sensitivity, barge-in mode and end-of-turn pause in the UI | UI settings |
| Optional cloud analysis (Anthropic or OpenAI): off by default, consent-gated, always shown | `britcoach/cloud/` |
| Local session history and progress (SQLite) | `data/coach.sqlite3`, `/api/sessions`, `/api/progress` |

### How corrections work

Every utterance passes through a fast rule-based detector (`coaching/detector.py`, < 1 ms).
It recognises common learner errors such as past tense after "yesterday", *I am agree*, *since 3 years*,
*explain me*, *he don't* and *informations*. Every issue quotes the exact words you said. The policy
(`coaching/policy.py`) picks **at most one** issue per turn:

* **Recast** (default): the coach is privately told the correct form and reuses it naturally.
  *"Yesterday I bring my son to school"* → *"Oh, you took your son to school yesterday? How did he get on?"*
* **Quick tip**: only for important mistakes repeated 3+ times (medium sensitivity), with a cooldown so it never nags.
  Tutor mode tips more readily.
* **Review**: everything is logged. The end-of-session review lists your top three mistakes, five useful British
  expressions (marked if the coach used them today), more natural versions of your sentences,
  and practice suggestions.

The review is grounded: rule-based findings quote your words verbatim. Anything an LLM adds (local or cloud) is
**dropped unless its quoted text actually appears in what you said**. Pronunciation is never assessed, because
the app only has a speech-recognition transcript.

> Whisper's language model sometimes "corrects" grammar while transcribing, so some real mistakes never reach the
> transcript. The detector favours precision: a false correction is worse than a missed one.

## If Whisper mishears you

* Raise **Pause before the coach replies** (UI slider) to 800–900 ms if you pause mid-sentence; otherwise your turn ends early and only half of it is transcribed.
* Try `COACH_STT_MODEL=small.en` (more accurate, slower) and/or `COACH_STT_BEAM_SIZE=5`, then re-run the benchmark to see the latency cost on your Mac.
* Check the mic level meter moves well when you speak; a quiet or distant mic hurts accuracy.

## Interruptions and echo

When the coach speaks through the MacBook speakers, the microphone hears it. The app offers three modes (in settings):

| Mode | Behaviour |
|---|---|
| **Headphones** (default) | Any speech over ~200 ms stops the coach immediately. |
| **Laptop speakers** | Stricter detection; the coach is *ducked*, a quick transcription checks whether the "speech" is just the coach's own words, and only real interruptions stop it. |
| **Off** | Half-duplex: the mic is ignored while the coach talks. |

The browser's echo cancellation is always requested (`echoCancellation: true`). Safari applies system-level
echo cancellation; Chrome's may not cover Web Audio playback. An **experimental** option routes playback through
a WebRTC loopback so Chrome's echo canceller can see it. It hasn't been verified on the target Mac.

## Configuration

Everything is in `.env` (see `.env.example`); each `COACH_*` variable maps to a field in `britcoach/config.py`.
Model profiles and voices are in `config/models.toml`.

| Setting | Default | Notes |
|---|---|---|
| `COACH_LLM_BACKEND` | `mlx` | `mlx`, `ollama`, `openai_compat`, `mock` |
| `COACH_LLM_PROFILE` | `medium` | `small` 1.5B · `medium` 3B/4B · `large` 7B/8B |
| `COACH_STT_BACKEND` / `_MODEL` | `faster_whisper` / `base.en` | `whispercpp`; `tiny.en` for speed, `small.en` for accuracy |
| `COACH_STT_BEAM_SIZE` | `1` | `5` is more accurate (faster-whisper) at extra decode time |
| `COACH_STT_CONTEXT_PROMPT` | `true` | Gives Whisper the coach's last sentence as context |
| `COACH_LLM_MAX_TOKENS` | `160` | Safety cap; if a reply hits it mid-sentence, the half-sentence isn't spoken |
| `COACH_TTS_BACKEND` / `_VOICE` | `piper` / `en_GB-cori-medium` | `macos_say` / `Daniel` |
| `COACH_END_OF_TURN_MS` | `560` | Silence that ends your turn: the biggest single latency lever |
| `COACH_BARGE_IN_MODE` | `headphones` | `speakers`, `off` |
| `COACH_CLOUD_ENABLED` | `false` | See below |

## Cloud fallback (optional, off by default)

Ordinary conversation **never** leaves your machine. When cloud is enabled (in `.env` or with the UI toggle), it can be used for:

* **Advanced review**: only when you press *Advanced analysis (cloud)* and tick the consent box naming the provider and word count.
* **Cloud-assisted corrections**: only if you tick that option. Each sentence is sent in the background; it never blocks the voice loop.
* **Escalation**: with `COACH_CLOUD_ESCALATION=local_failure`, a failed local review analysis may fall back to the cloud. Consent is still required.

Keys come only from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. The header shows **☁ Cloud** whenever a call is in flight.
Every call (provider, purpose, characters sent) is logged locally in the `cloud_calls` table. The Anthropic provider
defaults to `claude-opus-5-5` (`COACH_ANTHROPIC_MODEL`) and enables server-side refusal fallbacks
(`fallbacks="default"`), so if that model declines a request, the API retries it on a fallback model.
Set `COACH_OPENAI_MODEL` to an OpenAI model you have access to.

## Tests

```bash
.venv/bin/python -m pytest              # 94 tests, ~5 s, no model weights needed
node --test frontend/tests/*.test.mjs    # microphone resampler (anti-aliasing) tests
./scripts/e2e_browser.sh                # headless Chromium + fake microphone + real Silero VAD
```

The suite covers conversation state, correction policy, streaming (thread→asyncio bridge, Ollama and
OpenAI-compatible streams), audio-queue cancellation, barge-in (including stale audio and echo rejection),
cloud routing, model backend switching, and the REST/WebSocket API.

## Benchmarks

See [docs/BENCHMARK.md](docs/BENCHMARK.md). In short, on the Mac:

```bash
.venv/bin/python scripts/benchmark.py components --profiles small medium large
./scripts/start.sh &  .venv/bin/python scripts/benchmark.py e2e
```

**No performance numbers for the M3 are included in this repository.** It was built in a Linux container
with no Apple hardware and no access to the model weights. Run the benchmark and commit the JSON from `benchmarks/results/`.

## Licences of the moving parts

| Component | Licence |
|---|---|
| This code | the repository's licence |
| Silero VAD | MIT |
| faster-whisper, pywhispercpp / whisper.cpp, Whisper weights | MIT |
| Piper (`piper-tts` ≥ 1.3) | **GPL-3.0**. Fine for personal local use; check before redistributing. Voices have their own dataset licences (printed by `setup_models.py`). |
| Qwen2.5-1.5B, Qwen3-4B(-Instruct-2507), Qwen2.5-7B | Apache-2.0 (verify with `setup_models.py --verify-only`) |
| mlx-lm | MIT |

Qwen2.5-3B (non-commercial licence) and Llama 3.2 (community licence) are deliberately not defaults.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

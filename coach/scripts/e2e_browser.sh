#!/usr/bin/env bash
# Browser end-to-end smoke test with a fake microphone.
# Uses the real Silero VAD and the real web UI; STT/LLM/TTS are mocks unless you
# export COACH_*_BACKEND yourself (e.g. run it on the Mac with the real engines).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-benchmarks/e2e}"
PORT="${PORT:-8799}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"  # absolute: Chromium needs a real path to the WAV

# 2 s silence + one spoken sentence + 6 s silence, 16 kHz mono (Chromium loops it).
.venv/bin/python - "$OUT/mic.wav" <<'EOF'
import shutil, subprocess, sys, tempfile, wave
import numpy as np
out = sys.argv[1]
tmp = tempfile.mktemp(suffix=".wav")
text = "Yesterday I bring my son to school and then I go to work."
if shutil.which("say"):
    subprocess.run(["say", "-v", "Daniel", "-o", tmp, "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)
else:
    subprocess.run(["espeak-ng", "-v", "en-gb", "-s", "150", "-w", tmp, text], check=True)
with wave.open(tmp) as w:
    sr, a = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
if sr != 16000:
    x = np.linspace(0, len(a) / sr, len(a), endpoint=False)
    a = np.interp(np.linspace(0, len(a) / sr, int(len(a) * 16000 / sr), endpoint=False), x, a)
a = np.concatenate([np.zeros(32000), a, np.zeros(96000)]).astype("<i2")
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(a.tobytes())
EOF

export COACH_LLM_BACKEND="${COACH_LLM_BACKEND:-mock}" COACH_STT_BACKEND="${COACH_STT_BACKEND:-mock}"
export COACH_TTS_BACKEND="${COACH_TTS_BACKEND:-mock}" COACH_VAD_BACKEND=silero COACH_SAVE_HISTORY=true
export COACH_DB_PATH="$OUT/e2e.sqlite3" COACH_IDLE_PROMPT_S=0
.venv/bin/python -m uvicorn britcoach.main:app --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break; sleep 0.5; done
NODE_PATH="${NODE_PATH:-$(npm root -g)}" node scripts/e2e_browser.mjs "http://127.0.0.1:$PORT" "$OUT/mic.wav" "$OUT"

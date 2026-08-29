#!/usr/bin/env bash
# Run the two acceptance questions against several local models and print the
# latency, the stage breakdown and the actual answer for each.
#
#   ./scripts/compare-models.sh qwen3:8b qwen3:14b
#   ./scripts/compare-models.sh qwen3:8b qwen3:30b-a3b
#
# Each model is served on a scratch port by a backend this script starts and
# stops itself, so your running assistant is left alone. Pull the models first:
#   ollama pull qwen3:14b
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || { echo "Usage: $0 <model> [model...]   e.g. $0 qwen3:8b qwen3:14b"; exit 1; }

TOKEN="${LOCAL_ASSISTANT_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -x .venv/bin/python ] && [ -f .env ]; then
  TOKEN="$(env -u LOCAL_ASSISTANT_TOKEN .venv/bin/python -c \
    'from app.config import get_settings; print(get_settings().local_assistant_token)' 2>/dev/null)"
fi
[ -n "$TOKEN" ] || { echo "No token. Run ./scripts/set-token.sh, or export LOCAL_ASSISTANT_TOKEN."; exit 1; }

PORT="${COMPARE_PORT:-8765}"
LOG=$(mktemp)
trap 'kill %1 2>/dev/null; rm -f "$LOG"' EXIT

ask() {  # ask <json>
  curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/ask" \
    -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d "$1"
}

report() {  # report <label> <json>
  printf '%s' "$2" | python3 -c "
import json, sys
label = sys.argv[1]
d = json.load(sys.stdin)
if 'error' in d:
    print(f'  {label:12} FAILED: {d[\"error\"]} - {d[\"message\"][:80]}')
else:
    t = d.get('timings') or {}
    stages = '  '.join(f'{k}={v}' for k, v in t.items()) or '(local route)'
    print(f'  {label:12} {d[\"latency_ms\"]:>6} ms   route={d[\"route\"]}')
    print(f'               {stages}')
    print(f'               {d[\"answer\"][:180]}')
" "$1"
}

for MODEL in "${MODELS[@]}"; do
  echo "==================================================================="
  echo "$MODEL"
  echo "==================================================================="

  LOCAL_ASSISTANT_TOKEN="$TOKEN" LLM_MODEL="$MODEL" \
    .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
    --log-level warning > "$LOG" 2>&1 &

  ready=""
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then ready=1; break; fi
    sleep 0.5
  done
  if [ -z "$ready" ]; then
    echo "  backend did not start:"; tail -5 "$LOG" | sed 's/^/    /'
    kill %1 2>/dev/null; continue
  fi

  H=$(curl -s "http://127.0.0.1:$PORT/health")
  if ! printf '%s' "$H" | grep -q '"model_ok":true'; then
    echo "  model not installed - run: ollama pull $MODEL"
    kill %1 2>/dev/null; wait 2>/dev/null; continue
  fi

  # A warm-up request: the first call pays for loading the weights from disk,
  # which is not the number anyone cares about.
  ask '{"query":"hi","mode":"local","source":"cli"}' > /dev/null

  report "local" "$(ask '{"query":"解釋下 dependency injection","mode":"auto","source":"siri"}')"
  report "web" "$(ask '{"query":"而家最新 stable Python version 係邊個？","mode":"auto","source":"siri"}')"

  kill %1 2>/dev/null; wait 2>/dev/null
  echo
done

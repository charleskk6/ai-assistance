#!/usr/bin/env bash
# Start the coach on http://127.0.0.1:8765 (localhost only: the browser allows the
# microphone on localhost without HTTPS).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || { echo "No .venv. Run ./scripts/install.sh first."; exit 1; }
[ -f .env ] || cp .env.example .env

HOST="$(grep -E '^COACH_HOST=' .env | tail -n 1 | cut -d= -f2- || true)"
PORT="$(grep -E '^COACH_PORT=' .env | tail -n 1 | cut -d= -f2- || true)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

# Keep the CPU-side engines from oversubscribing the M3's 4 performance cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Only exported here so the Python SDKs can find them; never written anywhere.
set -a
# shellcheck disable=SC1091
. ./.env
set +a

echo "British Conversation Coach -> http://${HOST}:${PORT}"
echo "(models load and warm up before the server accepts connections)"
exec .venv/bin/python -m uvicorn britcoach.main:app --host "$HOST" --port "$PORT" --log-level warning

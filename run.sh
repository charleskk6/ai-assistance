#!/usr/bin/env bash
# Start the assistant on the LAN so the iPhone Shortcut can reach it.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "No .env found. Copy .env.example to .env and set LOCAL_ASSISTANT_TOKEN."; exit 1; }
[ -d .venv ] || { echo "No .venv found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }

# shellcheck disable=SC1091
set -a; source .env; set +a

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Assistant listening on http://${HOST}:${PORT}"
echo "From the iPhone, try: http://$(scutil --get LocalHostName 2>/dev/null || hostname -s).local:${PORT}/health"
exec .venv/bin/python -m uvicorn app.main:app --host "$HOST" --port "$PORT"

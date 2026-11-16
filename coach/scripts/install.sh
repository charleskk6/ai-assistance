#!/usr/bin/env bash
# One-shot installer for macOS on Apple Silicon (also works on Linux without MLX).
#   ./scripts/install.sh            # deps + UI + default models (medium profile, Piper voices)
#   ./scripts/install.sh --no-models
# Portable bash: no GNU-only flags, so it behaves the same with macOS's BSD tools.
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_MODELS=1
for a in "$@"; do
  case "$a" in
    --no-models) WITH_MODELS=0 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

OS="$(uname -s)"; ARCH="$(uname -m)"
echo "==> Platform: $OS $ARCH"
if [ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ]; then
  echo "    Warning: MLX needs Apple Silicon. Use COACH_LLM_BACKEND=ollama on Intel Macs."
fi

# ---- Python 3.11+ ---------------------------------------------------------------
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then PY="$cand"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11+ not found. On macOS: brew install python@3.12"; exit 1
fi
echo "==> Using $($PY --version) ($PY)"

if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
.venv/bin/python -m pip install -q --upgrade pip
echo "==> Installing Python dependencies"
.venv/bin/pip install -q -r requirements.txt -r requirements-dev.txt
.venv/bin/pip install -q -r requirements-tts.txt || echo "    Piper install failed; the app can use macOS voices (COACH_TTS_BACKEND=macos_say)"
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  .venv/bin/pip install -q -r requirements-mac.txt
fi

# ---- Frontend --------------------------------------------------------------------
if command -v npm >/dev/null 2>&1; then
  echo "==> Building the web UI"
  (cd frontend && npm ci --no-audit --no-fund && npm run build)
else
  echo "npm not found: install Node.js 18+ (brew install node), then: cd frontend && npm ci && npm run build"
fi

# ---- Config ----------------------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then
    printf '\n# Not Apple Silicon: MLX unavailable\nCOACH_LLM_BACKEND=ollama\n' >> .env
  fi
  echo "==> Created .env from .env.example"
fi

# ---- Models ----------------------------------------------------------------------
if [ "$WITH_MODELS" = "1" ]; then
  echo "==> Downloading models (several GB on first run)"
  BACKEND=mlx
  if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then BACKEND=ollama; fi
  .venv/bin/python scripts/setup_models.py --llm-backend "$BACKEND"
fi

echo
echo "Installed. Start with:  ./scripts/start.sh   then open http://127.0.0.1:8765"

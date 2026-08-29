#!/usr/bin/env bash
# Put exactly one working LOCAL_ASSISTANT_TOKEN in .env.
#
#   ./scripts/set-token.sh              generate a new token
#   ./scripts/set-token.sh <token>      set a specific one (e.g. to match a
#                                       backend already running)
#
# Safe to re-run. Existing assignments are removed first, so a leftover blank
# placeholder cannot shadow the real value - the last assignment is the one every
# reader takes.
set -euo pipefail
cd "$(dirname "$0")/.."

TOKEN="${1:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }
cp .env .env.bak

# Drop every existing assignment (commented ones are left alone), then append one.
grep -v -E '^[[:space:]]*(export[[:space:]]+)?LOCAL_ASSISTANT_TOKEN[[:space:]]*=' .env > .env.tmp
printf '\nLOCAL_ASSISTANT_TOKEN=%s\n' "$TOKEN" >> .env.tmp
mv .env.tmp .env

echo "Set LOCAL_ASSISTANT_TOKEN in $(pwd)/.env  (previous file saved as .env.bak)"
echo
grep -n "LOCAL_ASSISTANT_TOKEN" .env | sed 's/^/  line /'
echo
echo "Now restart the backend so it picks this up:  ./run.sh"

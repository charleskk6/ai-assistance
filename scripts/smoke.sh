#!/usr/bin/env bash
# Acceptance checks against a running assistant. Run on the Mac:
#   ./scripts/smoke.sh                       (defaults to localhost:8000)
#   ./scripts/smoke.sh http://my-mac.local:8000
#   ./scripts/smoke.sh http://my-mac.local:8000 <token>   (bypass .env entirely)
set -uo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-http://127.0.0.1:8000}"
ENV_FILE="${ENV_FILE:-.env}"

# Token resolution, in order: 2nd argument, exported variable, then .env.
# .env is parsed rather than sourced: a stray line in it should not be able to
# execute, and `source` silently gives up on CRLF line endings.
# A blank "LOCAL_ASSISTANT_TOKEN=" is the placeholder .env.example ships with,
# never a real value - so skip blanks and take the last assignment that has one.
read_env_token() {
  [ -f "$ENV_FILE" ] || return 1
  sed -n 's/\r$//; s/^[[:space:]]*\(export[[:space:]]\+\)\?LOCAL_ASSISTANT_TOKEN[[:space:]]*=[[:space:]]*//p' \
    "$ENV_FILE" | sed 's/^["'"'"']//; s/["'"'"']$//' | grep -v '^[[:space:]]*$' | tail -n1
}

TOKEN="${2:-${LOCAL_ASSISTANT_TOKEN:-}}"
TOKEN_FROM="argument"
[ -n "$TOKEN" ] && [ -z "${2:-}" ] && TOKEN_FROM="environment"
if [ -z "$TOKEN" ]; then
  TOKEN="$(read_env_token || true)"
  TOKEN_FROM="$ENV_FILE"
fi

if [ -z "$TOKEN" ]; then
  echo "No LOCAL_ASSISTANT_TOKEN found."
  echo "  looked in: the 2nd argument, the environment, and $(pwd)/$ENV_FILE"
  if [ ! -f "$ENV_FILE" ]; then
    echo "  -> $ENV_FILE does not exist. Create it and generate a token:"
    echo
    echo "     ./scripts/set-token.sh"
  elif ! grep -q "LOCAL_ASSISTANT_TOKEN" "$ENV_FILE"; then
    echo "  -> $ENV_FILE has no LOCAL_ASSISTANT_TOKEN line at all. Add one:"
    echo
  else
    echo "  -> $ENV_FILE assigns LOCAL_ASSISTANT_TOKEN, but every assignment is blank:"
    grep -n "LOCAL_ASSISTANT_TOKEN" "$ENV_FILE" | sed 's/^/       line /'
    echo "     Set it with:"
    echo
  fi
  echo "     ./scripts/set-token.sh"
  echo
  echo "  Then RESTART the backend - it reads .env once at startup - and re-run this."
  echo "  Or pass the token directly, without touching .env:"
  echo
  echo "     ./scripts/smoke.sh $BASE <token>"
  exit 1
fi

pass=0; fail=0
ask() {  # ask <json> -> prints route (or error code)
  curl -s --max-time 180 -X POST "$BASE/ask" \
    -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d "$1"
}
check() { # check <label> <expected> <actual>
  if [ "$2" = "$3" ]; then echo "  PASS  $1"; pass=$((pass+1));
  else echo "  FAIL  $1 (expected '$2', got '$3')"; fail=$((fail+1)); fi
}
route_of() { printf '%s' "$1" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("route") or d.get("error",""))'; }
field()    { printf '%s' "$1" | python3 -c "import json,sys; print(json.load(sys.stdin).get('$2',''))"; }

echo "Target: $BASE   (token from $TOKEN_FROM)"
echo
echo "1. Health"
H=$(curl -s --max-time 10 "$BASE/health") || { echo "  FAIL  backend unreachable"; exit 1; }
check "backend reachable"   "True" "$(printf '%s' "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin)["backend"])')"
check "llm runtime up"      "True" "$(printf '%s' "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin)["llm"]["runtime_ok"])')"
check "model available"     "True" "$(printf '%s' "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin)["llm"]["model_ok"])')"
echo "  model: $(printf '%s' "$H" | python3 -c 'import json,sys;print(json.load(sys.stdin)["llm"]["model"])')"

echo
echo "2. Auth"
check "no token rejected" "401" "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/ask" -H 'Content-Type: application/json' -d '{"query":"hi"}')"

# A running backend read its config at startup. If .env was edited since, the
# token here and the token it is enforcing differ, and every check below would
# fail as "unauthorized" for no obvious reason.
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 180 -X POST "$BASE/ask" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"ping","mode":"local","source":"cli"}')
if [ "$CODE" = "401" ]; then
  echo "  FAIL  the backend rejected the token from .env"
  echo
  echo "  The backend at $BASE is running with a different token than .env holds."
  echo "  It reads .env once, at startup - so if you edited .env after starting it,"
  echo "  stop it (Ctrl-C in the run.sh terminal) and run ./run.sh again."
  exit 1
fi
check "valid token accepted" "yes" "$([ "$CODE" = "200" ] && echo yes || echo "HTTP $CODE")"

echo
echo "3. Routing"
check "local: 解釋下 DI"      "local" "$(route_of "$(ask '{"query":"解釋下 dependency injection","source":"siri"}')")"
check "override /local"      "local" "$(route_of "$(ask '{"query":"/local latest Python release","source":"cli"}')")"

echo
echo "4. Acceptance Test A - local, spoken Cantonese"
A=$(ask '{"query":"解釋下 dependency injection","mode":"auto","source":"siri"}')
check "route" "local" "$(route_of "$A")"
ANS=$(field "$A" answer)
case "$ANS" in *'#'*|*'|'*|*'http'*|*'**'*) echo "  FAIL  answer contains markup a TTS voice would read"; fail=$((fail+1));;
  *) echo "  PASS  answer is speech-clean"; pass=$((pass+1));; esac
echo "  latency: $(field "$A" latency_ms) ms"
echo "  answer (${#ANS} bytes): $ANS"

echo
echo "5. Acceptance Test B - web RAG, grounded, sources separate"
B=$(ask '{"query":"而家最新 stable Python version 係邊個？","mode":"auto","source":"siri"}')
check "route" "web" "$(route_of "$B")"
NSRC=$(printf '%s' "$B" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("sources",[])))')
[ "$NSRC" -gt 0 ] && { echo "  PASS  $NSRC sources returned"; pass=$((pass+1)); } || { echo "  FAIL  no sources"; fail=$((fail+1)); }
ANS=$(field "$B" answer)
case "$ANS" in *http*) echo "  FAIL  a URL leaked into the spoken answer"; fail=$((fail+1));;
  *) echo "  PASS  no URL in the spoken answer"; pass=$((pass+1));; esac
echo "  latency: $(field "$B" latency_ms) ms"
echo "  answer (${#ANS} bytes): $ANS"
printf '%s' "$B" | python3 -c 'import json,sys
for s in json.load(sys.stdin).get("sources",[]): print("  source:  %s - %s" % (s["domain"], s["title"][:60]))'

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]

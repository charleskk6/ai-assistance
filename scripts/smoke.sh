#!/usr/bin/env bash
# Acceptance checks against a running assistant. Run on the Mac:
#   ./scripts/smoke.sh                       (defaults to localhost:8000)
#   ./scripts/smoke.sh http://my-mac.local:8000
set -uo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-http://127.0.0.1:8000}"
[ -f .env ] && { set -a; . ./.env; set +a; }
TOKEN="${LOCAL_ASSISTANT_TOKEN:?set LOCAL_ASSISTANT_TOKEN or source .env}"

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

echo "Target: $BASE"
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
echo "  answer:  ${ANS:0:160}"

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
echo "  answer:  ${ANS:0:200}"
printf '%s' "$B" | python3 -c 'import json,sys
for s in json.load(sys.stdin).get("sources",[]): print("  source:  %s - %s" % (s["domain"], s["title"][:60]))'

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]

#!/usr/bin/env bash
# Check a running service: health, model list, one greedy chat request, one streamed request with its speed.
#   scripts/smoke_test.sh [--wait]      # --wait: poll /health for up to 90 minutes first (e.g. during warm-up)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-$ROOT/config/serve.env}
HOST=127.0.0.1; PORT=8000; SERVED_MODEL_NAME=qwen3.8-27b-fp8-mtp; API_KEY=
# shellcheck disable=SC1090
[ -f "$CONFIG" ] && . "$CONFIG"
[ "$HOST" = "0.0.0.0" ] && HOST=127.0.0.1
export BASE_URL=http://$HOST:$PORT MODEL=$SERVED_MODEL_NAME API_KEY
AUTH=(); [ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")

if [ "${1:-}" = "--wait" ]; then
  printf 'waiting for %s/health ' "$BASE_URL"
  for _ in $(seq 1 1080); do
    [ "$(curl -s -m3 -o /dev/null -w '%{http_code}' "$BASE_URL/health")" = 200 ] && break
    printf .; sleep 5
  done; echo
fi
code=$(curl -s -m5 -o /dev/null -w '%{http_code}' "$BASE_URL/health" || true)
[ "$code" = 200 ] || { echo "health: $code (not ready; the first start warms up for 25-40 minutes)"; exit 1; }
echo "health: ok"
curl -s -m10 "${AUTH[@]}" "$BASE_URL/v1/models" | python3 -c 'import json,sys; print("models:", [m["id"] for m in json.load(sys.stdin)["data"]])'

echo "greedy request:"
curl -s -m300 "${AUTH[@]}" "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\", \"temperature\": 0, \"max_tokens\": 200,
  \"messages\": [{\"role\": \"user\", \"content\": \"用三句话解释为什么天空是蓝色的。\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false}}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print("  " + r["choices"][0]["message"]["content"].replace("\n", "\n  ")); print("  usage:", r["usage"])'

echo "streamed request (512 tokens, code prompt):"
python3 "$ROOT/tools/bench/bench.py" smoke 512 0

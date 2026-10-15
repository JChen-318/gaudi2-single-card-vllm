#!/usr/bin/env bash
# Decode-speed benchmark against a running service (single stream, two streams, ~30K-token context).
#   scripts/bench.sh            # about 5 minutes
#   scripts/bench.sh --full     # also the 15-prompt mix of tools/bench/bench_multi.py (about 15 more minutes)
# Speed depends on how many draft tokens each round accepts, so it varies with the prompt and the sampling.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-$ROOT/config/serve.env}
HOST=127.0.0.1; PORT=8000; SERVED_MODEL_NAME=qwen3.8-27b-fp8-mtp; API_KEY=
# shellcheck disable=SC1090
[ -f "$CONFIG" ] && . "$CONFIG"
[ "$HOST" = "0.0.0.0" ] && HOST=127.0.0.1
export BASE_URL=http://$HOST:$PORT MODEL=$SERVED_MODEL_NAME API_KEY
B=$ROOT/tools/bench
[ "$(curl -s -m3 -o /dev/null -w '%{http_code}' "$BASE_URL/health")" = 200 ] || { echo "service not ready at $BASE_URL"; exit 1; }

echo "warm-up requests"
python3 "$B/bench.py" warm 128 0 >/dev/null; python3 "$B/bench.py" warm 128 2 >/dev/null
echo "single stream (3 prompts x 512 tokens)"
for p in 0 1 2; do python3 "$B/bench.py" single-$p 512 $p; done
echo "two streams (both at once)"
python3 "$B/bench.py" pair-a 512 0 & python3 "$B/bench.py" pair-b 512 2; wait
echo "about 30K tokens of context (prefill, then 400 tokens)"
python3 "$B/bench_ctx.py" ctx30k 15000 400
if [ "${1:-}" = "--full" ]; then
  echo "prompt mix"
  python3 "$B/bench_multi.py" 512
fi

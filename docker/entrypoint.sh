#!/usr/bin/env bash
# Container entry point: start the service in the foreground with config/serve.env (mounted from the host),
# keep a copy of its output in $DATA_DIR/logs like scripts/serve.sh --daemon does, and prime the service after
# /health answers (tools/prime.py, PRIME_AFTER_START=0 turns it off). Extra arguments go to scripts/serve.sh.
set -euo pipefail
ROOT=/opt/gaudi2-single-card-vllm
CONFIG=${CONFIG:-$ROOT/config/serve.env}
[ -f "$CONFIG" ] || { echo "missing $CONFIG: mount your serve.env there (docker/compose.yaml does)"; exit 1; }
# shellcheck disable=SC1090
. "$CONFIG"
DATA_DIR=${DATA_DIR:-/data}
PORT=${PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.8-27b-fp8-mtp}
mkdir -p "$DATA_DIR/logs"
# After a host reboot the model share may come up later than Docker: wait for it instead of failing.
for _ in $(seq 1 "${MODEL_WAIT_SECONDS:-1800}"); do
  [ -f "${TARGET_MODEL:-}/config.json" ] && [ -f "${DRAFT_MODEL:-}/config.json" ] && break
  [ "${_waited:-0}" = 0 ] && echo "waiting for $TARGET_MODEL and $DRAFT_MODEL ..." && _waited=1
  sleep 1
done
LOG=$DATA_DIR/logs/serve-$(date +%Y%m%d-%H%M%S).log
echo "gaudi2-single-card-vllm container: log $LOG"
if [ "${PRIME_AFTER_START:-1}" = 1 ]; then
  API_KEY=${API_KEY:-} python3.12 "$ROOT/tools/prime.py" --base "http://127.0.0.1:$PORT" --model "$SERVED_MODEL_NAME" \
    >> "${LOG%.log}-prime.log" 2>&1 < /dev/null &
fi
exec > >(tee -a "$LOG") 2>&1
exec "$ROOT/scripts/serve.sh" --config "$CONFIG" "$@"

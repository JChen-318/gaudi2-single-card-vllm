#!/usr/bin/env bash
# Stop a service started with scripts/serve.sh --daemon (TERM, then KILL after 2 minutes).
#   scripts/stop.sh [--config config/serve.env]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=$ROOT/config/serve.env
[ "${1:-}" = "--config" ] && CONFIG=$2
DATA_DIR=$HOME/.cache/gaudi2-single-card-vllm
# shellcheck disable=SC1090
[ -f "$CONFIG" ] && . "$CONFIG"
PIDFILE=$DATA_DIR/serve.pid
[ -f "$PIDFILE" ] || { echo "no $PIDFILE: nothing started with --daemon"; exit 0; }
PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then echo "process $PID is not running"; rm -f "$PIDFILE"; exit 0; fi
# serve.sh starts the server with setsid, so the API server and its engine share process group $PID
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID"
for _ in $(seq 1 120); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
if kill -0 "$PID" 2>/dev/null || pgrep -g "$PID" >/dev/null 2>&1; then
  echo "still running after 120 s: sending KILL"
  kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID"
fi
rm -f "$PIDFILE"
echo "stopped"

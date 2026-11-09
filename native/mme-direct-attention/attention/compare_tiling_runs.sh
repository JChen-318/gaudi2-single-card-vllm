#!/usr/bin/env bash
# Existing full-head input required. Each run checks every output on every replay.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
input=${1:-inputs/head-layer25-h0-g20.bin}
group=${2:-20}
for entry in old-a new-a new-b old-b; do
 rows=128;kernel=exp_f16_logq_ms8;extra=()
 if [[ $entry == new-* ]]; then rows=256;kernel=exp_f16_tile_logq_ms8;extra+=(--coalesce-qk); fi
 LOG_NAME="tiling-abba-$entry" PROBE_TIMEOUT=180 ALLOW_UNVALIDATED_ARC=1 \
 bash run_context_probe.sh scal-batch --input "$input" --group-blocks "$group" \
 --repeats 8 --tile-rows "$rows" --scores-fp16 --exp-kernel "$kernel" --reload-head-kv "${extra[@]}"
done

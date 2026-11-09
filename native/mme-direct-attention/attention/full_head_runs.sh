#!/usr/bin/env bash
# Complete distinct-query head; all outputs returned. Input packing and CPU
# reference are preparation, outside the reported native/DMA time.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
head=${1:-0}
group=${2:-20}
tile=${3:-128}
kernel=exp_f16_logq_ms8
extra=()
case "$tile" in
 128) ;;
 256) kernel=exp_f16_tile_logq_ms8;extra+=(--coalesce-qk);;
 512) kernel=exp_f16_tile_logq_ms8
      (( group % 4 == 0 && group <= 16 )) || { echo '512 rows needs group multiple of 4 <=16' >&2; exit 2; };;
 *) echo 'tile rows: 128|256|512' >&2;exit 2;;
esac
input="inputs/head-layer25-h${head}-g${group}.bin"
PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    /home/a/h3-omni-venv/bin/python export_query_batch.py \
    /mnt/sas/tmp/minimax-h3/qkv/qkv-call1-layer25.pt "$input" \
    --head "$head" --blocks "$group" --full-head
PROBE_TIMEOUT=180 LOG_NAME="full-head-h${head}-g${group}-t${tile}" ALLOW_UNVALIDATED_ARC=1 \
    bash run_context_probe.sh scal-batch --input "$input" --group-blocks "$group" \
    --repeats 5 --tile-rows "$tile" --scores-fp16 --exp-kernel "$kernel" --reload-head-kv "${extra[@]}"

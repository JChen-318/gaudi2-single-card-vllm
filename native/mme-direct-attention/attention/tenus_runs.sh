#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
config=${1:-wide}
mode=${2:-real}
input=${3:-inputs/head-layer25-h0.bin}
group=${GROUP_BLOCKS:-16}
repeats=${REPEATS:-10}
args=(--input "$input" --group-blocks "$group" --repeats "$repeats" --reload-head-kv
      --tile-rows 512 --scores-fp16 --coalesce-qk --interleave-arm --overlap-dma
      --split-fence --cross-group --multi-monitor --tpc-cmd 0xa)
case "$config" in
 baseline) args+=(--exp-kernel exp_f16_tile_affine_skewidx_ms8);;
 fixed512) args+=(--exp-kernel exp_f16_tile_fixed0_g15pad --mme-no-barrier --mme-async-commit --local-qk-barrier);;
 wide) args+=(--exp-kernel exp_f16_tile_fixed0_g15pad --mme-no-barrier --mme-async-commit --local-qk-barrier --wide-query);;
 wide-relaxed) args+=(--exp-kernel exp_f16_tile_fixed0_g15pad --mme-no-barrier --mme-async-commit --local-qk-barrier --wide-query --relaxed-qk-reuse);;
 *) echo 'config: baseline|fixed512|wide|wide-relaxed' >&2;exit 2;;
esac
if [[ $config != baseline ]]; then args+=(--fixed0-exponent-shift "${FIXED0_EXPONENT_SHIFT:-8}");fi
case "$mode" in
 real) ;;
 mme|tpc) args+=(--ablation "$mode-only" --profile-only);;
 *) echo 'mode: real|mme|tpc' >&2;exit 2;;
esac
LOG_NAME=${LOG_NAME:-tenus-final-$config-$mode} PROBE_TIMEOUT=${PROBE_TIMEOUT:-240} ALLOW_UNVALIDATED_ARC=1  bash run_context_probe.sh scal-batch "${args[@]}"

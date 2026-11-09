#!/usr/bin/env bash
# Stage ablation / A-B runs of the full-head batch paths (2026-11-09).
# Entries are NAME[.tag]-gGROUP. "real" runs check every output on every replay;
# profile-only runs (noop exp, tpc-only) are NOT valid attention.
# 128/256-row runs use the 20-block-group input; 512-row runs need groups that
# are multiples of four <=16, so they use the 16-block-group input and no
# QK coalescing (not validated for 512 rows).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
input20=${INPUT:-inputs/head-layer25-h0-g20.bin}
input16=${INPUT16:-inputs/head-layer25-h0.bin}
run() { local name=$1; shift
  LOG_NAME="ablation-$name" PROBE_TIMEOUT=240 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-batch "$@" | grep -E 'RESULT|Error|error' | cut -c1-400 || true
}
for entry in ${RUNS:-real-g20 inter-g20 noop-g20 noop-inter-g20 tpc-g20 tpc-inter-g20 real-g10 inter-g10 real-g4 inter-g4}; do
  name=${entry%-g*}; name=${name%%.*}; g=${entry##*-g}
  if [[ $name == *512* || $name == xg* || $name == xk* ]]; then
    common=(--input "$input16" --repeats 6 --scores-fp16 --reload-head-kv --group-blocks "$g")
  else
    common=(--input "$input20" --repeats 6 --tile-rows 128 --scores-fp16 --coalesce-qk --reload-head-kv --group-blocks "$g")
  fi
  extra=()
  case $name in
    real) extra+=(--exp-kernel exp_f16_logq_ms8) ;;
    inter) extra+=(--exp-kernel exp_f16_logq_ms8 --interleave-arm) ;;
    ovl) extra+=(--exp-kernel exp_f16_logq_ms8 --interleave-arm --overlap-dma) ;;
    aff) extra+=(--exp-kernel exp_f16_affine_ms8 --interleave-arm --overlap-dma) ;;
    affnp) extra+=(--exp-kernel exp_f16_affine_np_ms8 --interleave-arm --overlap-dma) ;;
    afftpc) extra+=(--exp-kernel exp_f16_affine_ms8 --profile-only --ablation tpc-only --interleave-arm) ;;
    t256) extra+=(--tile-rows 256 --exp-kernel exp_f16_tile_logq_ms8 --interleave-arm --overlap-dma) ;;
    t256aff) extra+=(--tile-rows 256 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma) ;;
    t256noop) extra+=(--tile-rows 256 --exp-kernel noop_exp3 --profile-only --interleave-arm) ;;
    q512) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_logq_ms8 --interleave-arm --overlap-dma) ;;
    q512aff) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma) ;;
    affsf) extra+=(--exp-kernel exp_f16_affine_ms8 --interleave-arm --overlap-dma --split-fence) ;;
    q512affsf) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence) ;;
    old256) extra+=(--tile-rows 256 --exp-kernel exp_f16_tile_logq_ms8) ;;
    old512) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_logq_ms8) ;;
    mme128) extra+=(--exp-kernel exp_f16_affine_ms8 --profile-only --ablation mme-only --interleave-arm --split-fence) ;;
    noop128sf) extra+=(--exp-kernel noop_exp3 --profile-only --interleave-arm --split-fence) ;;
    q512mme) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation mme-only --interleave-arm --split-fence) ;;
    q512noopsf) extra+=(--tile-rows 512 --exp-kernel noop_exp3 --profile-only --interleave-arm --split-fence) ;;
    pf128) extra+=(--exp-kernel exp_f16_affine_ms8 --interleave-arm --overlap-dma --split-fence --prefetch-desc) ;;
    pf512) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence --prefetch-desc) ;;
    pf128mme) extra+=(--exp-kernel exp_f16_affine_ms8 --profile-only --ablation mme-only --interleave-arm --split-fence --prefetch-desc) ;;
    pf512mme) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation mme-only --interleave-arm --split-fence --prefetch-desc) ;;
    nb128) extra+=(--exp-kernel exp_f16_affine_ms8 --interleave-arm --overlap-dma --split-fence --prefetch-desc --mme-no-barrier) ;;
    nb512) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence --prefetch-desc --mme-no-barrier) ;;
    nb128mme) extra+=(--exp-kernel exp_f16_affine_ms8 --profile-only --ablation mme-only --interleave-arm --split-fence --prefetch-desc --mme-no-barrier) ;;
    nb512mme) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation mme-only --interleave-arm --split-fence --prefetch-desc --mme-no-barrier) ;;
    xg512) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence --cross-group) ;;
    xg512mme) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation mme-only --interleave-arm --overlap-dma --split-fence --cross-group) ;;
    xg512tpc) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation tpc-only --interleave-arm --overlap-dma --split-fence --cross-group) ;;
    xg512tpcnm) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_nomain_ms8 --profile-only --ablation tpc-only --interleave-arm --overlap-dma --split-fence --cross-group) ;;
    xg512tpcnoop) extra+=(--tile-rows 512 --exp-kernel noop_exp3 --profile-only --ablation tpc-only --interleave-arm --overlap-dma --split-fence --cross-group) ;;
    xgmm) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor) ;;
    xgmmtpc) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation tpc-only --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor) ;;
    xgmmtpcnoop) extra+=(--tile-rows 512 --exp-kernel noop_exp3 --profile-only --ablation tpc-only --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor) ;;
    xgc10) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xgc2) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xgc2tpc) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation tpc-only --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xgc2tpcnoop) extra+=(--tile-rows 512 --exp-kernel noop_exp3 --profile-only --ablation tpc-only --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xgcq) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xgcqmme) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --coalesce-qk --profile-only --ablation mme-only --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xgmme10) extra+=(--tile-rows 512 --exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation mme-only --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xkskew) extra+=(--exp-kernel exp_f16_tile_affine_skew_ms8 --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xkskewtpc) extra+=(--exp-kernel exp_f16_tile_affine_skew_ms8 --profile-only --ablation tpc-only --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xkcur) extra+=(--exp-kernel exp_f16_tile_affine_ms8 --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xkcurtpc) extra+=(--exp-kernel exp_f16_tile_affine_ms8 --profile-only --ablation tpc-only --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xkidx) extra+=(--exp-kernel exp_f16_tile_affine_skewidx_ms8 --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xkidxtpc) extra+=(--exp-kernel exp_f16_tile_affine_skewidx_ms8 --profile-only --ablation tpc-only --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xknomain) extra+=(--exp-kernel exp_f16_tile_affine_nomain_ms8 --profile-only --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xknoop) extra+=(--exp-kernel noop_exp3 --profile-only --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    xkmme) extra+=(--exp-kernel exp_f16_tile_affine_skewidx_ms8 --profile-only --ablation mme-only --tile-rows 512 --coalesce-qk --interleave-arm --overlap-dma --split-fence --cross-group --multi-monitor --tpc-cmd 0xa) ;;
    q512noop) extra+=(--tile-rows 512 --exp-kernel noop_exp3 --profile-only --interleave-arm) ;;
    noop) extra+=(--exp-kernel noop_exp3 --profile-only) ;;
    noop-inter) extra+=(--exp-kernel noop_exp3 --profile-only --interleave-arm) ;;
    tpc) extra+=(--exp-kernel exp_f16_logq_ms8 --profile-only --ablation tpc-only) ;;
    tpc-inter) extra+=(--exp-kernel exp_f16_logq_ms8 --profile-only --ablation tpc-only --interleave-arm) ;;
    *) echo "unknown entry $entry" >&2; exit 2 ;;
  esac
  echo "== $entry"; run "$entry" "${common[@]}" "${extra[@]}"
done
echo ABLATION_DONE

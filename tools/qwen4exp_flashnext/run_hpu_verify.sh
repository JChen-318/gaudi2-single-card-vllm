#!/usr/bin/env bash
# Tier-4 integration verification for the Qwen4Exp Strata+NVFP4 Gaudi2 port.
#
#   tools/qwen4exp_flashnext/run_hpu_verify.sh              tiers 1-3 (CPU) then on-card micro tests
#   tools/qwen4exp_flashnext/run_hpu_verify.sh --offline    tiers 1-3 only (no device probing)
#   tools/qwen4exp_flashnext/run_hpu_verify.sh --skip-tier123
#   tools/qwen4exp_flashnext/run_hpu_verify.sh --serve-bench
#                                                           delegate to validation/run_hpu_verify.sh
#
# Pipeline (mirrors the 4-tier micro test loop; never loads the full 48-layer
# model, never starts vllm serve on its own):
#   Tier 1  csf.py golden decode tests (CPU, real checkpoint bytes)
#   Tier 2  tpc-clang compile gate + kernel-math op tests   (test_nvfp4_tpc_op.py)
#   Tier 3  2-layer micro-config forward + block tests      (test_qwen4_exp_forward.py)
#   Tier 4  on-card micro tests with VLLM_SKIP_WARMUP=true: per-kernel bit-exact
#           checks + single-layer HPU tests. Recipe cache is cleared first.
#
# Device detection: hl-smi on this host spuriously prints "driver not loaded"
# even when the card works, so the PRIMARY probe is a child-process
# torch.hpu.is_available() (this script's own process never touches the device:
# Gaudi2 allows a single acquiring process).
#
# Exit codes: 0 ok, 2 usage, 3 no device (tier 4 requested), 4 verification failure.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")"/../.. && pwd)
PY=${PYTHON:-$ROOT/.venv/bin/python}
TOOLS=$ROOT/tools/qwen4exp_flashnext
NATIVE=$ROOT/build/native

OFFLINE=0
SKIP_TIER123=0
SERVE_BENCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --offline) OFFLINE=1; shift ;;
    --skip-tier123) SKIP_TIER123=1; shift ;;
    --serve-bench) SERVE_BENCH=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s\n' "$*"; }
fail() { printf '[run_hpu_verify] %s\n' "$*" >&2; exit 4; }
# These files are scripts (main()), not pytest modules. pytest treats their
# arguments as fixtures and, for files with no test_* functions, collects
# nothing and exits 0 without running the check.
run_script() {
  local label=$1
  shift
  for t in "$@"; do
    [ -f "$t" ] || fail "$label: missing $t"
    say "$label: $(basename "$t")"
    "$PY" "$t" || fail "$label ($(basename "$t"))"
  done
}

[ -x "$PY" ] || { echo "no interpreter at $PY (set PYTHON=...)" >&2; exit 2; }
export VLLM_HW=cpu   # tiers 1-3 are CPU-only; tier 4 subprocesses override

# ---------------------------------------------------------------------------
# Tiers 1-3: CPU gates (must pass before any device run)
# ---------------------------------------------------------------------------
if [ "$SKIP_TIER123" = 0 ]; then
  run_script "Tier 1" "$TOOLS/test_csf.py" "$TOOLS/test_pad_spans.py"

  TIER2=("$TOOLS/test_nvfp4_tpc_math.py")
  [ -f "$TOOLS/test_nvfp4_tpc_op.py" ] && TIER2+=("$TOOLS/test_nvfp4_tpc_op.py")
  run_script "Tier 2" "${TIER2[@]}"

  BLOCKS=(test_hyperconnection.py test_qsa_torch.py test_qsa_tile_replay.py
          test_ple_runtime.py test_ple_embedding.py test_moe_grouped.py
          test_mtp.py test_weights_loader.py test_model_skeleton.py)
  TIER3=("${BLOCKS[@]/#/$TOOLS/}")
  [ -f "$TOOLS/test_qwen4_exp_forward.py" ] && TIER3+=("$TOOLS/test_qwen4_exp_forward.py")
  run_script "Tier 3" "${TIER3[@]}"
fi

if [ "$OFFLINE" = 1 ]; then
  say "done (offline): tiers 1-3 passed"
  exit 0
fi

# ---------------------------------------------------------------------------
# Device probe: child process only (this parent never acquires the card)
# ---------------------------------------------------------------------------
say "device probe (child-process torch.hpu; hl-smi is advisory here)"
hl-smi -L 2>/dev/null | sed 's/^/  hl-smi: /' || echo "  hl-smi: unavailable (not fatal on this host)"
DEVICE_OK=$("$PY" - <<'EOF'
import sys
try:
    import torch
    ok = hasattr(torch, "hpu") and torch.hpu.is_available()
except Exception:
    ok = False
print("yes" if ok else "no")
EOF
)
if [ "$DEVICE_OK" != yes ]; then
  cat >&2 <<'EOF'
[run_hpu_verify] no usable Gaudi2 device right now. Tiers 1-3 above are still valid.
                 Re-run without --offline when the card is attached. Not marking
                 the goal blocked: this is exactly the offline/on-card split.
EOF
  exit 3
fi

# ---------------------------------------------------------------------------
# Tier 4: on-card micro tests. Fresh recipe cache, warmup skipped.
# ---------------------------------------------------------------------------
say "clearing TPC recipe caches"
CACHE_ROOT=${PT_HPU_RECIPE_CACHE_CONFIG:-$HOME/.cache/gaudi2-single-card-vllm/recipe-cache}
rm -rf "$HOME"/.cache/gaudi2-single-card-vllm/recipe-cache-* 2>/dev/null || true
[ "${PT_HPU_RECIPE_CACHE_CONFIG:-}" = "" ] || rm -rf "${PT_HPU_RECIPE_CACHE_CONFIG:?}" 2>/dev/null || true
echo "  cleared under $CACHE_ROOT*"

Q4X_KERNELS=$ROOT/vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so
[ -f "$Q4X_KERNELS" ] || fail "missing $Q4X_KERNELS (build the csrc TPC library first)"
export GC_KERNEL_PATH="$NATIVE/libkiro_tpc_shim.so"
# csrc lib carries indexed-dequant BF16 + FP8 GUIDs. First match wins, so it
# is listed ahead of the decode-GEMV library.
export KIRO_TPC_LIBS="$Q4X_KERNELS:$NATIVE/libmtp_sampling_kernels_v2.so:$NATIVE/libflashinfer_gaudi_kernels.so:$NATIVE/libqwen4exp_nvfp4_moe_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so"
MOE_OP=$(ls "$NATIVE"/hpu_qwen4exp_nvfp4_moe_pt2*.so 2>/dev/null | head -1 || true)
[ -n "$MOE_OP" ] && export VLLM_QWEN4EXP_MOE_LIB="$MOE_OP"
export VLLM_SKIP_WARMUP=true   # no full-model warmup: micro tests only
export PT_HPU_LAZY_MODE=0      # micro tests compare bit-exact eager results
unset VLLM_HW

run_script "Tier 4" \
  "$TOOLS/test_nvfp4_tpc_hpu.py" \
  "$TOOLS/test_nvfp4_gemv_hpu.py" \
  "$TOOLS/test_gdn_layer_hpu.py" \
  "$TOOLS/test_qsa_layer_hpu.py" \
  "$TOOLS/test_ple_layer_hpu.py" \
  "$TOOLS/test_moe_layer_hpu.py" \
  "$TOOLS/test_moe_block_hpu.py"

if [ "$SERVE_BENCH" = 1 ]; then
  say "serve bench: delegating to validation/run_hpu_verify.sh --serve-bench"
  exec bash "$ROOT/validation/run_hpu_verify.sh" --serve-bench
fi

say "done: tiers 1-4 passed (no serve, no full-model load)"

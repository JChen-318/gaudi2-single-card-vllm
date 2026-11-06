#!/usr/bin/env bash
# On-card verification for the Qwen4Exp (Qwen3.8-Flash-Next) HPU port - Track D.
#
#   validation/run_hpu_verify.sh                 full verification on a Gaudi2 card
#   validation/run_hpu_verify.sh --offline       host-only: build + offline pytest suites
#   validation/run_hpu_verify.sh --skip-build    do not rebuild the native libraries
#   validation/run_hpu_verify.sh --full-serve    ...and serve the real checkpoint (long)
#   validation/run_hpu_verify.sh --serve-bench   ...serve, time a fixed prompt, stop the server
#
# What it does, in order:
#   (a) checks that the Gaudi2 device is really present (hl-smi + torch.hpu). A host without the
#       card exits 3 with a message instead of failing inside a kernel. --offline skips (a), (d)
#       and (e), which is the only path that may run without the device / in CI.
#   (b) builds the native pieces: native/qwen4exp-nvfp4-moe (Track C decode GEMV + bucketing),
#       flashinfer_gaudi, mtp-sampling, and the single-file GC_KERNEL_PATH shim. scripts/build_native.sh
#       also runs the host geometry/reference/shim checks, so a green build already proves every
#       kernel GUID is reachable and the decode math matches its C reference.
#   (c) runs the offline suites: validation/qwen4exp (Track A/B/C/D checkers), the Track D model
#       test, and the ModelOpt loader tests.
#   (d) runs validation/qwen4exp/hpu_device_smoke.py: 4-layer dummy model on the device, one prefill
#       forward, then a fixed-prompt decode loop that prints measured tok/s.
#   (e) optionally serves the real checkpoint with GC_KERNEL_PATH (the shim) and KIRO_TPC_LIBS (all
#       kernel libraries, including the Track C one). --serve-bench waits for /health, runs one
#       fixed-prompt completion, prints tok/s from the reported usage, and stops the server.
#
# Environment:
#   PYTHON              interpreter (default <repo>/.venv/bin/python)
#   VLLM_QWEN4EXP_MODEL checkpoint directory (default $HOME/models/qwen4exp-flashnext-plain)
#   SMOKE_STEPS         decode steps timed by the device smoke (default 16)
#   SERVE_PORT          port for --full-serve/--serve-bench (default 8000)
#
# Exit codes: 0 ok, 2 usage, 3 no device, 4 smoke/verification failure, 5 serve failure.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$ROOT/.venv/bin/python}
MODEL=${VLLM_QWEN4EXP_MODEL:-$HOME/models/qwen4exp-flashnext-plain}
NATIVE=$ROOT/build/native
STEPS=${SMOKE_STEPS:-16}
PORT=${SERVE_PORT:-8000}

OFFLINE=0
BUILD=1
FULL_SERVE=0
SERVE_BENCH=0

usage() {
  sed -n '2,40p' "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --offline) OFFLINE=1; shift ;;
    --skip-build) BUILD=0; shift ;;
    --full-serve) FULL_SERVE=1; shift ;;
    --serve-bench) FULL_SERVE=1; SERVE_BENCH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s\n' "$*"; }
fail() { printf '[run_hpu_verify] %s\n' "$*" >&2; exit 4; }

say "Qwen4Exp Track D verification ($([ "$OFFLINE" = 1 ] && echo 'offline' || echo 'on-card'))"
echo "  repo:        $ROOT"
echo "  python:      $PY"
echo "  checkpoint:  $MODEL"
[ -x "$PY" ] || { echo "no interpreter at $PY (set PYTHON=...)" >&2; exit 2; }

# ---------------------------------------------------------------------------
# (a) device presence
# ---------------------------------------------------------------------------
if [ "$OFFLINE" = 1 ]; then
  say "device check: skipped (--offline)"
else
  say "device check"
  if ! command -v hl-smi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[run_hpu_verify] hl-smi not found: this looks like a host without the Intel Gaudi stack.
                 Nothing here was run. Use --offline for the host-only suites, or run this
                 script inside the Gaudi2 container with the device attached.
EOF
    exit 3
  fi
  if ! hl-smi -L >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[run_hpu_verify] hl-smi cannot see a device (driver not loaded or the card is disconnected).
                 Use --offline for the host-only suites; (d)/(e) need the card.
EOF
    exit 3
  fi
  "$PY" - <<'EOF' || { echo "[run_hpu_verify] torch.hpu reports no usable device" >&2; exit 3; }
import torch
assert hasattr(torch, "hpu") and torch.hpu.is_available(), "torch.hpu.is_available() == False"
print(f"  torch.hpu: {torch.hpu.get_device_name(0)}")
EOF
  hl-smi -L | sed 's/^/  /'
fi

# ---------------------------------------------------------------------------
# (b) native libraries
# ---------------------------------------------------------------------------
if [ "$BUILD" = 1 ]; then
  say "(b) build native libraries (scripts/build_native.sh)"
  bash "$ROOT/scripts/build_native.sh"
else
  say "(b) build skipped (--skip-build)"
fi

for lib in libkiro_tpc_shim.so libqwen4exp_nvfp4_moe_kernels.so libflashinfer_gaudi_kernels.so \
           libmtp_sampling_kernels_v2.so; do
  [ -f "$NATIVE/$lib" ] || fail "missing native library $NATIVE/$lib (run without --skip-build)"
done
MOE_OP=$(ls "$NATIVE"/hpu_qwen4exp_nvfp4_moe_pt2*.so 2>/dev/null | head -1 || true)
[ -n "$MOE_OP" ] || fail "missing Track C torch op in $NATIVE (run without --skip-build)"
echo "  Track C op: $MOE_OP"

export GC_KERNEL_PATH="$NATIVE/libkiro_tpc_shim.so"
Q4X_KERNELS=$ROOT/vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so
[ -f "$Q4X_KERNELS" ] || fail "missing $Q4X_KERNELS (indexed-dequant BF16/FP8 TPC library)"
export KIRO_TPC_LIBS="$Q4X_KERNELS:$NATIVE/libmtp_sampling_kernels_v2.so:$NATIVE/libflashinfer_gaudi_kernels.so:$NATIVE/libqwen4exp_nvfp4_moe_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so"
export VLLM_QWEN4EXP_MOE_LIB="$MOE_OP"

# ---------------------------------------------------------------------------
# (c) offline suites (host only; safe everywhere)
# ---------------------------------------------------------------------------
say "(c) offline pytest suites"
# The full-checkpoint dry-run (meta model + real 301,759-tensor index) is gated so
# plain `pytest validation` stays fast; here it is part of the verification.
export VLLM_QWEN4EXP_FULL_LOAD=1
"$PY" -m pytest -q \
  "$ROOT/validation/qwen4exp" \
  "$ROOT/vllm-gaudi/tests/unit_tests/ops/test_qwen4_exp_model.py" \
  "$ROOT/vllm-gaudi/tests/unit_tests/ops/test_modelopt_csf.py" \
  "$ROOT/vllm-gaudi/tests/unit_tests/ops/test_hpu_modelopt.py"

if [ "$OFFLINE" = 1 ]; then
  say "done (offline): native build + host-only suites passed"
  exit 0
fi

# ---------------------------------------------------------------------------
# (d) device smoke: dummy 4-layer model, one forward, measured tok/s
# ---------------------------------------------------------------------------
say "(d) device smoke (4-layer dummy on HPU, fixed prompt, measured tok/s)"
"$PY" "$ROOT/validation/qwen4exp/hpu_device_smoke.py" --steps "$STEPS" || exit 4

# ---------------------------------------------------------------------------
# (e) optional: serve the real checkpoint with the Track D kernel libraries
# ---------------------------------------------------------------------------
if [ "$FULL_SERVE" = 0 ]; then
  say "done: host checks + device smoke passed (add --full-serve for the real checkpoint)"
  exit 0
fi

say "(e) serve the real checkpoint"
[ -f "$MODEL/config.json" ] || fail "no checkpoint at $MODEL (set VLLM_QWEN4EXP_MODEL)"
LOG=${VLLM_QWEN4EXP_SERVE_LOG:-$ROOT/build/qwen4exp-serve.log}
mkdir -p "$(dirname "$LOG")"
SERVE_CMD=(
  "$PY" -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --served-model-name qwen4exp
  --language-model-only
  --dtype bfloat16
  --max-model-len 4096
  --gpu-memory-utilization 0.90
  --port "$PORT"
)
echo "  GC_KERNEL_PATH=$GC_KERNEL_PATH"
echo "  KIRO_TPC_LIBS=$KIRO_TPC_LIBS"
echo "  ${SERVE_CMD[*]}"
echo "  log: $LOG"
if [ "$SERVE_BENCH" = 0 ]; then
  exec "${SERVE_CMD[@]}"
fi

"${SERVE_CMD[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

"$PY" - "$PORT" <<'EOF' || { echo "[run_hpu_verify] server did not become healthy; see the log" >&2; exit 5; }
import json, sys, time, urllib.request
port = sys.argv[1]
deadline = time.time() + 3600  # first start compiles shape buckets
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            if response.status == 200:
                print("  server healthy")
                raise SystemExit(0)
    except Exception:
        time.sleep(5)
raise SystemExit(1)
EOF

say "serve bench: fixed prompt, 32 new tokens"
"$PY" - "$PORT" <<'EOF'
import json, sys, time, urllib.request
port = sys.argv[1]
prompt = "The capital of France is"
body = json.dumps({
    "model": "qwen4exp",
    "prompt": prompt,
    "max_tokens": 32,
    "temperature": 0.0,
}).encode()
request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=1800) as response:
    payload = json.loads(response.read())
elapsed = time.perf_counter() - started
tokens = payload["usage"]["completion_tokens"]
print(f"  prompt: {prompt!r}")
print(f"  completion tokens: {tokens} in {elapsed:.2f}s -> {tokens / elapsed:.2f} tok/s "
      f"(single request, includes prefill)")
print(f"  text: {payload['choices'][0]['text'][:160]!r}")
EOF
say "done: device smoke + serve bench passed"

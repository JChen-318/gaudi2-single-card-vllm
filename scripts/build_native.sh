#!/usr/bin/env bash
# Build the native pieces the service loads. Host-only: compiles, never opens the device.
#   build/native/libmtp_sampling_kernels_v2.so          TPC kernels for random rejection sampling
#   build/native/mtp_sampling_v2.cpython-312-*.so        its PyTorch op
#   build/native/libflashinfer_gaudi_kernels.so, flashinfer_gaudi_ops.*.so   FlashInfer-Gaudi kernels (incl. the fused
#                                                        GDN core) + op
#   build/native/libqwen4exp_nvfp4_moe_kernels.so, hpu_qwen4exp_nvfp4_moe_pt2.*.so   NVFP4 MoE TPC kernels (decode
#                                                        GEMV + prefill bucketing) + op, host geometry test, scalar C
#                                                        reference for the decode math
#   build/native/libkiro_tpc_shim.so                     single-file TPC library front-end (recipe cache needs one file)
# After rebuilding any of them, start the service with a fresh recipe cache (scripts/serve.sh does this by itself:
# the cache directory name contains the libraries' hashes).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-$( [ -f "$ROOT/.python" ] && cat "$ROOT/.python" || echo python3.12 )}
OUT=$ROOT/build/native
JOBS=${MAX_JOBS:-4}
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=$JOBS
mkdir -p "$OUT"

echo "== mtp-sampling kernels"
cmake -S "$ROOT/native/mtp-sampling" -B "$OUT/mtp-sampling" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$OUT/mtp-sampling" --target mtp_sampling_kernels_v2 --parallel "$JOBS"
cp "$OUT/mtp-sampling/libmtp_sampling_kernels_v2.so" "$OUT/"

echo "== mtp-sampling op"
(cd "$ROOT/native/mtp-sampling" && "$PY" setup.py -q build_ext --build-lib "$OUT" --build-temp "$OUT/mtp-sampling/ext")

echo "== flashinfer_gaudi kernels and op"
# Kept out of vllm-gaudi/flashinfer_gaudi/lib on purpose: a kernel library there is prepended to GC_KERNEL_PATH on
# import, which breaks the single-file GC_KERNEL_PATH that the recipe cache needs.
"$PY" "$ROOT/vllm-gaudi/tools/build_flashinfer_gaudi.py" --build-root "$OUT/flashinfer_gaudi" --output-dir "$OUT"

# QWEN4EXP=0 skips the Qwen4Exp (Flash-Next) NVFP4 MoE library: the 27B service does not load it.
if [ "${QWEN4EXP:-1}" = 1 ]; then
  echo "== qwen4exp NVFP4 MoE kernels (decode GEMV + prefill bucketing)"
  # TPC-C sources under native/qwen4exp-nvfp4-moe/kernels, host glue under host/, ELFs embedded. Built as a separate
  # library so the existing vllm-gaudi/csrc/qwen4exp kernels stay untouched; the shim merges its GUIDs with the rest.
  cmake -S "$ROOT/native/qwen4exp-nvfp4-moe" -B "$OUT/qwen4exp-nvfp4-moe" -DCMAKE_BUILD_TYPE=Release >/dev/null
  # Builds the library, the host geometry test, and (where the Habana simulator libraries are installed) the
  # TPC-simulator harness.
  cmake --build "$OUT/qwen4exp-nvfp4-moe" --parallel "$JOBS"
  cp "$OUT/qwen4exp-nvfp4-moe/libqwen4exp_nvfp4_moe_kernels.so" "$OUT/"
  cp "$OUT/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_geometry_test" "$OUT/"
  # Scalar C reference for the decode GEMV math; validation/qwen4exp/test_nvfp4_moe_reference.py runs it.
  cp "$OUT/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_reference" "$OUT/"
  # The TPC-simulator harness is only configured where the Habana simulator libraries exist.
  if [ -x "$OUT/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_sim" ]; then cp "$OUT/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_sim" "$OUT/"; fi
  # Per-call latency estimate at the checkpoint's layer shapes (same simulator; ~40 s, run by ctest too).
  if [ -x "$OUT/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_latency" ]; then cp "$OUT/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_latency" "$OUT/"; fi
  echo "== qwen4exp NVFP4 MoE op"
  (cd "$ROOT/native/qwen4exp-nvfp4-moe/pytorch" && "$PY" setup.py -q build_ext --build-lib "$OUT" --build-temp "$OUT/qwen4exp-nvfp4-moe/ext")
fi

echo "== TPC library front-end"
g++ -O2 -fPIC -shared -std=c++17 -I/usr/include/habanalabs "$ROOT/native/tpc-kernel-shim/tpc_kernel_shim.cpp" \
  -o "$OUT/libkiro_tpc_shim.so" -Wl,--no-as-needed -L/usr/lib/habanalabs -ltpc_kernels -ldl -lpthread \
  -Wl,-rpath,/usr/lib/habanalabs
g++ -O1 -std=c++17 -I/usr/include/habanalabs "$ROOT/native/tpc-kernel-shim/shim_check.cpp" -o "$OUT/shim_check" -ldl

MOE=; [ "${QWEN4EXP:-1}" = 1 ] && MOE="$OUT/libqwen4exp_nvfp4_moe_kernels.so:"
LIBS="$OUT/libmtp_sampling_kernels_v2.so:$OUT/libflashinfer_gaudi_kernels.so:${MOE}/usr/lib/habanalabs/libtpc_kernels.so"
echo "== check: every kernel of the libraries is reachable through the front-end"
KIRO_TPC_LIBS=$LIBS "$OUT/shim_check" "$OUT/libkiro_tpc_shim.so" ${LIBS//:/ }
if [ "${QWEN4EXP:-1}" = 1 ]; then
  echo "== check: qwen4exp NVFP4 MoE glue geometry (no device)"
  "$OUT/qwen4exp_nvfp4_moe_geometry_test" "$OUT/libqwen4exp_nvfp4_moe_kernels.so"
  if [ -x "$OUT/qwen4exp_nvfp4_moe_sim" ]; then
    echo "== check: qwen4exp NVFP4 MoE kernels in the TPC simulator (no device)"
    "$OUT/qwen4exp_nvfp4_moe_sim"
  fi
fi
ls -l "$OUT"/*.so
echo "Built. Next: cp config/serve.env.example config/serve.env, edit the model paths, scripts/serve.sh"

#!/usr/bin/env bash
# Build the three native pieces the service loads. Host-only: compiles, never opens the device.
#   build/native/libmtp_sampling_kernels_v2.so          TPC kernels for random rejection sampling
#   build/native/mtp_sampling_v2.cpython-312-*.so        its PyTorch op
#   build/native/libflashinfer_gaudi_kernels.so, flashinfer_gaudi_ops.*.so   FlashInfer-Gaudi kernels (incl. the fused
#                                                        GDN core) + op
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

echo "== TPC library front-end"
g++ -O2 -fPIC -shared -std=c++17 -I/usr/include/habanalabs "$ROOT/native/tpc-kernel-shim/tpc_kernel_shim.cpp" \
  -o "$OUT/libkiro_tpc_shim.so" -Wl,--no-as-needed -L/usr/lib/habanalabs -ltpc_kernels -ldl -lpthread \
  -Wl,-rpath,/usr/lib/habanalabs
g++ -O1 -std=c++17 -I/usr/include/habanalabs "$ROOT/native/tpc-kernel-shim/shim_check.cpp" -o "$OUT/shim_check" -ldl

LIBS="$OUT/libmtp_sampling_kernels_v2.so:$OUT/libflashinfer_gaudi_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so"
echo "== check: every kernel of the three libraries is reachable through the front-end"
KIRO_TPC_LIBS=$LIBS "$OUT/shim_check" "$OUT/libkiro_tpc_shim.so" ${LIBS//:/ }
ls -l "$OUT"/*.so
echo "Built. Next: cp config/serve.env.example config/serve.env, edit the model paths, scripts/serve.sh"

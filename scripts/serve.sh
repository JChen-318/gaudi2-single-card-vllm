#!/usr/bin/env bash
# Start the OpenAI-compatible service with the tuned single-Gaudi2 configuration (docs/OPTIMIZATION.md, v14).
#
#   scripts/serve.sh                      # foreground, settings from config/serve.env
#   scripts/serve.sh --daemon             # background; log in $DATA_DIR/logs, stop with scripts/stop.sh
#   scripts/serve.sh --config my.env --skip-warmup
#
# First start: model load ~4 min + warm-up ~25-40 min (compiles every shape bucket; device recipes are cached
# on disk, so later starts take ~25 min). --skip-warmup starts in ~6 min and compiles shapes on first use:
# fine for a functional check, never for benchmarks.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=$ROOT/config/serve.env
DAEMON=0; SKIP_WARMUP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --daemon) DAEMON=1; shift ;;
    --skip-warmup) SKIP_WARMUP=1; shift ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
[ -f "$CONFIG" ] || { echo "missing $CONFIG: cp config/serve.env.example config/serve.env and edit it"; exit 1; }
# shellcheck disable=SC1090
. "$CONFIG"
: "${TARGET_MODEL:?set TARGET_MODEL in $CONFIG}" "${DRAFT_MODEL:?set DRAFT_MODEL in $CONFIG}"
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.8-27b-fp8-mtp}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}; MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}; KV_CACHE_GIB=${KV_CACHE_GIB:-28}
DATA_DIR=${DATA_DIR:-$HOME/.cache/gaudi2-single-card-vllm}
GENERATION_CONFIG=${GENERATION_CONFIG:-'{"temperature":1.0,"top_p":0.95,"top_k":20}'}
PY=${PYTHON:-$( [ -f "$ROOT/.python" ] && cat "$ROOT/.python" || echo python3.12 )}
for d in "$TARGET_MODEL" "$DRAFT_MODEL"; do [ -f "$d/config.json" ] || { echo "no config.json in $d"; exit 1; }; done
[ "$MAX_NUM_SEQS" -ge 1 ] && [ "$MAX_NUM_SEQS" -le 16 ] || { echo "MAX_NUM_SEQS must be 1-16 (DFlash2 limit)"; exit 1; }
# Decode buckets: batch 1..MAX_NUM_SEQS; pages (128 tokens) up to every sequence at full length, capped by the
# KV cache (about 18,800 FP8 tokens per GiB), rounded up to the 128-page step. Measured: 2 x 262144 -> 4096.
PAGES=$(( (MAX_MODEL_LEN * MAX_NUM_SEQS + 127) / 128 ))
KV_PAGES=$(( KV_CACHE_GIB * 18832 / 128 ))
[ "$PAGES" -gt "$KV_PAGES" ] && PAGES=$KV_PAGES
BLOCK_MAX=$(( (PAGES + 127) / 128 * 128 ))

NATIVE=$ROOT/build/native
FI=$NATIVE
SAMPLING_KERNELS=$NATIVE/libmtp_sampling_kernels_v2.so
SAMPLING_OP=$(ls "$NATIVE"/mtp_sampling_v2.cpython-312*.so 2>/dev/null | head -1)
FI_KERNELS=$FI/libflashinfer_gaudi_kernels.so
FI_OP=$(ls "$FI"/flashinfer_gaudi_ops.cpython-312*.so 2>/dev/null | head -1)
SHIM=$NATIVE/libkiro_tpc_shim.so
if ls "$ROOT"/vllm-gaudi/flashinfer_gaudi/lib/*.so >/dev/null 2>&1; then
  echo "vllm-gaudi/flashinfer_gaudi/lib/ contains .so files: flashinfer_gaudi would add them to GC_KERNEL_PATH and"
  echo "break the recipe cache. Remove them (scripts/build_native.sh puts the libraries in build/native/)."; exit 1
fi
for f in "$SAMPLING_KERNELS" "$SAMPLING_OP" "$FI_KERNELS" "$FI_OP" "$SHIM"; do
  [ -n "$f" ] && [ -f "$f" ] || { echo "native library missing ($f): run scripts/build_native.sh"; exit 1; }
done

# Device recipes are cached per set of kernel libraries: a rebuild gets a fresh cache automatically.
LIBHASH=$(cat "$SAMPLING_KERNELS" "$FI_KERNELS" "$SHIM" /usr/lib/habanalabs/libtpc_kernels.so | sha256sum | cut -c1-12)
RUN=$DATA_DIR/run; RECIPES=$DATA_DIR/recipe-cache-$LIBHASH
mkdir -p "$RUN" "$RECIPES" "$DATA_DIR/logs" "$DATA_DIR/habana-logs" "$DATA_DIR/torch-cache"
# Runtime switches read by the engine (all on = the measured configuration)
echo '{"lookahead": true, "fast_topk": true, "presample": true, "delta": true, "predraft": true, "guard_skip": true, "rng_pool": true, "early_readback": true}' > "$RUN/control.json"
echo '{"cpu_inputs": true, "context_region": true}' > "$RUN/decode-control.json"
echo '{"handoff": true, "sampling_cache": true, "draft_inputs": true, "packed_copies": false}' > "$RUN/host-control.json"

SITE=$(PYTHONNOUSERSITE=1 "$PY" -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("torch").origin))')
HF_TORCH_LIB=$(dirname "$SITE")/habana_frameworks/torch/lib
EXTRA=()
[ "$SKIP_WARMUP" = 1 ] && EXTRA+=(VLLM_SKIP_WARMUP=true)
[ -n "${PIN_MAIN_CPU:-}" ] && EXTRA+=(VLLM_HPU_PIN_MAIN_CPU="$PIN_MAIN_CPU")
[ -n "${STEP_LOG:-}" ] && EXTRA+=(VLLM_HPU_STEP_TIMER="$STEP_LOG")
ARGS=()
[ -n "${API_KEY:-}" ] && ARGS+=(--api-key "$API_KEY")

CMD=(env -u QUANT_CONFIG -u HABANA_VISIBLE_MODULES -u CUDA_VISIBLE_DEVICES -u HABANA_PROF_CONFIG
  PATH="/opt/habanalabs/bin:$PATH"
  PYTHONPATH="$ROOT/vllm-gaudi"
  PYTHONNOUSERSITE=1
  LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
  LD_LIBRARY_PATH="$SITE/lib:$HF_TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  HABANA_LOGS="$DATA_DIR/habana-logs/"
  HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw
  HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins
  PT_HPU_LAZY_MODE=0
  ENABLE_EXPERIMENTAL_FLAGS=true
  ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true
  # kernel libraries: one file for the graph compiler, the real list behind it; on-disk recipe cache
  GC_KERNEL_PATH="$SHIM"
  KIRO_TPC_LIBS="$SAMPLING_KERNELS:$FI_KERNELS:/usr/lib/habanalabs/libtpc_kernels.so"
  PT_HPU_RECIPE_CACHE_CONFIG="$RECIPES,false,20480"
  FLASHINFER_GAUDI_NATIVE_LIBRARY="$FI_OP"
  VLLM_HPU_NATIVE_SAMPLING_LIBRARY="$SAMPLING_OP"
  # Dynamo remembers which dims turned out dynamic and compiles them dynamic from the start next time
  TORCH_COMPILE_JOB_ID=gaudi2-single-card-vllm
  TORCHINDUCTOR_CACHE_DIR="$DATA_DIR/torch-cache"
  # host-side decode preparation (sections 3, 9)
  VLLM_HPU_FAST_DECODE_PREP=all VLLM_HPU_FAST_DECODE_PREP_VERIFY=256 VLLM_HPU_ENVS_CACHE=1
  VLLM_HPU_SHARED_PREFIX_DECODE=1 VLLM_HPU_SKIP_UNBACKED_BLOCK_LIST=1 VLLM_HPU_SHARE_GROUP_MASKS=1
  VLLM_HPU_UNIQUE_COMPILE_CODE=1 VLLM_HPU_FUSED_MODEL_ENTRY=1 VLLM_HPU_WARMUP_PHASE_GC=1
  # full attention over the whole context per request: KV gathered once, FP8 matmuls, one softmax (section 17)
  VLLM_HPU_SHARED_PREFIX_ATTN=dense VLLM_HPU_DENSE_ATTN_QK=fp8 VLLM_HPU_DENSE_ATTN_AV=fp8
  # DFlash2 speculative decoding pipeline (sections 9-12)
  VLLM_HPU_DFLASH2_PAGE_BUCKETS=1 VLLM_HPU_DFLASH2_DRAFT_STACK=1 VLLM_HPU_DFLASH2_DIRECT_CHECKPOINTS=1
  VLLM_HPU_DFLASH2_LOOKAHEAD=1 VLLM_HPU_DFLASH2_LOOKAHEAD_SAMPLE=1 VLLM_HPU_DFLASH2_LOOKAHEAD_DELTA=1
  VLLM_HPU_DFLASH2_PREDRAFT=1 VLLM_HPU_DFLASH2_PREDRAFT_VERIFY=32 VLLM_HPU_DFLASH2_EARLY_READBACK=1
  VLLM_HPU_DFLASH2_CPU_INPUTS=1 VLLM_HPU_DFLASH2_CONTEXT_REGION=1
  VLLM_HPU_DFLASH2_DECODE_CONTROL="$RUN/decode-control.json"
  VLLM_HPU_DFLASH2_HOST_OPT=1 VLLM_HPU_DFLASH2_HOST_OPT_CONTROL="$RUN/host-control.json"
  VLLM_HPU_V6_CONTROL="$RUN/control.json"
  VLLM_HPU_GUARD_SKIP=1 VLLM_HPU_RNG_POOL=32
  # sampling (sections 9, 11)
  VLLM_HPU_FAST_TOPK_TOPP=1 VLLM_HPU_DFLASH2_CHUNKED_TOPK=1 VLLM_HPU_LM_HEAD_FP8=1 VLLM_HPU_NATIVE_CORRECTION=1
  VLLM_HPU_BOUNDED_TOPK=0 VLLM_HPU_STABLE_SAMPLING_RNG=0
  # GDN layers through FlashInfer-Gaudi, fused verify core on gathered states (sections 13, 15)
  VLLM_HPU_FLASHINFER_GDN=1 VLLM_HPU_FLASHINFER_DFLASH2=1 VLLM_COMPACT_GDN=1
  FLASHINFER_GAUDI_BACKEND=auto FLASHINFER_GAUDI_ENABLE_MTP_PREPARED=true FLASHINFER_GAUDI_MTP_PREPARED_MAX_BATCH=2
  FLASHINFER_GAUDI_MTP_FUSED=1 FLASHINFER_GAUDI_MTP_FUSED_GATHER=1
  FLASHINFER_GAUDI_MTP_TRACE=0 FLASHINFER_GAUDI_ENABLE_MTP5_T6=0
  # MTP-head alternatives that this DFlash2 configuration does not use
  VLLM_HPU_MTP_CHAIN_COMPILE=0 VLLM_HPU_MTP_METADATA_POOL=0 VLLM_HPU_MTP_HANDOFF_OPT=0 VLLM_HPU_MTP_SAMPLING_CACHE=0
  VLLM_HPU_MTP_SINGLE_HEAD_COMPILE=0 VLLM_HPU_MTP_RANDOM_WARMUP=0
  VLLM_HPU_SPEC_OPTIMIZATION=1 VLLM_HPU_SPEC_LAYER_GROUP_SIZE=4
  VLLM_GRAPH_RESERVED_MEM=0.3
  # decode shape buckets (computed above; the measured configuration gives batch 1-2 and 4096 pages)
  VLLM_DECODE_BUCKETING_STRATEGY=lin VLLM_DECODE_BS_BUCKET_MIN=1 VLLM_DECODE_BS_BUCKET_STEP=3 VLLM_DECODE_BS_BUCKET_MAX="$MAX_NUM_SEQS"
  VLLM_DECODE_BLOCK_BUCKET_MIN=1 VLLM_DECODE_BLOCK_BUCKET_STEP=128 VLLM_DECODE_BLOCK_BUCKET_MAX="$BLOCK_MAX"
  VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "${EXTRA[@]}"
  "$PY" -m vllm.entrypoints.cli.main serve "$TARGET_MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size 1 --pipeline-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --kv-cache-memory-bytes $((KV_CACHE_GIB * 1024 * 1024 * 1024)) --kv-cache-dtype fp8_inc
  --no-enable-prefix-caching
  --override-generation-config "$GENERATION_CONFIG"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --middleware dflash2_request_guard.DFlash2RequestGuard
  --host "$HOST" --port "$PORT"
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":7}"
  "${ARGS[@]}")

cd "$ROOT/vllm-gaudi"
if [ "$DAEMON" = 1 ]; then
  LOG=$DATA_DIR/logs/serve-$(date +%Y%m%d-%H%M%S).log
  setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
  echo $! > "$DATA_DIR/serve.pid"
  echo "started (pid $!), log: $LOG"
  echo "ready when: curl -s http://$HOST:$PORT/health returns 200 (watch: tail -f $LOG)"
else
  exec "${CMD[@]}"
fi

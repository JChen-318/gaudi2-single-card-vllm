#!/bin/sh
# Example launcher for the v10 production configuration in docs/OPTIMIZATION.md (single Gaudi2,
# Qwen3.5-architecture 27B FP8 target + DFlash2 drafter, up to 2 concurrent requests, 262K context, no profiler).
#
# This is a record of one working setup, not a portable installer: every path below is site-specific, the native
# libraries have to be built for your SDK, and warm-up takes about 30 minutes. Read the notes at the bottom first.

VLLM_GAUDI_ROOT=/path/to/vllm-gaudi                 # this repository's vllm-gaudi directory (on PYTHONPATH)
SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages
TARGET_MODEL=/path/to/models/qwen3.8-27b-fp8-mtp
DRAFT_MODEL=/path/to/models/Qwen3.8-27B-DFlash2
SAMPLING_NATIVE=/path/to/mtp-sampling-build         # built from native/mtp-sampling
HOST_OPT_CONTROL=/path/to/dflash-host-control.json  # {"handoff": true, "sampling_cache": true, "draft_inputs": true, "packed_copies": false}
LOG_DIR=/path/to/service-logs/
TPC_SHIM=/path/to/libkiro_tpc_shim.so                # built from native/tpc-kernel-shim
RECIPE_CACHE=/path/to/recipe-cache                   # clear it whenever any kernel library changes
RUN_DIR=/path/to/run                                 # v6 control file and per-round step log

cd "$VLLM_GAUDI_ROOT" || exit 1
exec env -u HABANA_VISIBLE_MODULES -u CUDA_VISIBLE_DEVICES -u HABANA_PROF_CONFIG \
  PYTHONPATH="$VLLM_GAUDI_ROOT" \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4 \
  GC_KERNEL_PATH="$TPC_SHIM" \
  KIRO_TPC_LIBS="$SAMPLING_NATIVE/build/libmtp_sampling_kernels_v2.so:$SITE_PACKAGES/flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so" \
  PT_HPU_RECIPE_CACHE_CONFIG="$RECIPE_CACHE,false,20480" \
  FLASHINFER_GAUDI_NATIVE_LIBRARY="$SITE_PACKAGES/flashinfer_gaudi/lib/flashinfer_gaudi_ops.cpython-312-x86_64-linux-gnu.so" \
  LD_LIBRARY_PATH="$SITE_PACKAGES/torch/lib:$SITE_PACKAGES/habana_frameworks/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  HABANA_LOGS="$LOG_DIR" \
  HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
  HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
  PT_HPU_LAZY_MODE=0 \
  ENABLE_EXPERIMENTAL_FLAGS=true \
  ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
  \
  VLLM_HPU_FAST_DECODE_PREP=all \
  VLLM_HPU_FAST_DECODE_PREP_VERIFY=256 \
  VLLM_HPU_ENVS_CACHE=1 \
  VLLM_HPU_SHARED_PREFIX_DECODE=1 \
  VLLM_HPU_SKIP_UNBACKED_BLOCK_LIST=1 \
  VLLM_HPU_SHARE_GROUP_MASKS=1 \
  \
  VLLM_HPU_UNIQUE_COMPILE_CODE=1 \
  VLLM_HPU_DFLASH2_DRAFT_STACK=1 \
  VLLM_HPU_FUSED_MODEL_ENTRY=1 \
  VLLM_HPU_DFLASH2_LOOKAHEAD=1 \
  VLLM_HPU_DFLASH2_LOOKAHEAD_SAMPLE=1 \
  VLLM_HPU_DFLASH2_LOOKAHEAD_DELTA=1 \
  VLLM_HPU_FAST_TOPK_TOPP=1 \
  VLLM_HPU_DFLASH2_CHUNKED_TOPK=1 \
  VLLM_HPU_LM_HEAD_FP8=1 \
  VLLM_HPU_DFLASH2_DIRECT_CHECKPOINTS=1 \
  VLLM_HPU_DFLASH2_PAGE_BUCKETS=1 \
  VLLM_HPU_DFLASH2_CPU_INPUTS=1 \
  VLLM_HPU_DFLASH2_CONTEXT_REGION=1 \
  VLLM_HPU_DFLASH2_DECODE_CONTROL="$RUN_DIR/decode-control.json" \
  VLLM_HPU_V6_CONTROL="$RUN_DIR/control.json" \
  VLLM_HPU_STEP_TIMER="$RUN_DIR/steps.log" \
  \
  VLLM_HPU_FLASHINFER_GDN=1 \
  VLLM_HPU_FLASHINFER_DFLASH2=1 \
  VLLM_HPU_DFLASH2_HOST_OPT=1 \
  VLLM_HPU_DFLASH2_HOST_OPT_CONTROL="$HOST_OPT_CONTROL" \
  VLLM_COMPACT_GDN=1 \
  FLASHINFER_GAUDI_BACKEND=auto \
  FLASHINFER_GAUDI_ENABLE_MTP_PREPARED=true \
  FLASHINFER_GAUDI_MTP_PREPARED_MAX_BATCH=2 \
  FLASHINFER_GAUDI_MTP_TRACE=0 \
  FLASHINFER_GAUDI_ENABLE_MTP5_T6=0 \
  VLLM_HPU_MTP_CHAIN_COMPILE=0 \
  VLLM_HPU_BOUNDED_TOPK=0 \
  VLLM_HPU_STABLE_SAMPLING_RNG=0 \
  VLLM_HPU_MTP_METADATA_POOL=0 \
  VLLM_HPU_MTP_HANDOFF_OPT=0 \
  VLLM_HPU_MTP_SAMPLING_CACHE=0 \
  VLLM_HPU_MTP_SINGLE_HEAD_COMPILE=0 \
  VLLM_HPU_MTP_RANDOM_WARMUP=0 \
  VLLM_HPU_NATIVE_CORRECTION=1 \
  VLLM_HPU_NATIVE_SAMPLING_LIBRARY="$SAMPLING_NATIVE/mtp_sampling_v2.cpython-312-x86_64-linux-gnu.so" \
  VLLM_HPU_SPEC_OPTIMIZATION=1 \
  VLLM_HPU_SPEC_LAYER_GROUP_SIZE=4 \
  VLLM_GRAPH_RESERVED_MEM=0.3 \
  \
  VLLM_DECODE_BUCKETING_STRATEGY=lin \
  VLLM_DECODE_BS_BUCKET_MIN=1 \
  VLLM_DECODE_BS_BUCKET_STEP=3 \
  VLLM_DECODE_BS_BUCKET_MAX=2 \
  VLLM_DECODE_BLOCK_BUCKET_MIN=1 \
  VLLM_DECODE_BLOCK_BUCKET_STEP=128 \
  VLLM_DECODE_BLOCK_BUCKET_MAX=2048 \
  \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONNOUSERSITE=1 \
  vllm serve "$TARGET_MODEL" \
  --served-model-name qwen3.8-27b-fp8-mtp \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --kv-cache-memory-bytes 30064771072 \
  --no-enable-prefix-caching \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --middleware dflash2_request_guard.DFlash2RequestGuard \
  --host 127.0.0.1 \
  --port 8000 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":7}"

# Notes
# - VLLM_DECODE_BLOCK_BUCKET_MAX=2048 is the page count of one decode batch (2048 pages x 128 tokens = 262K). It
#   only means this with VLLM_HPU_SHARED_PREFIX_DECODE=1; without it, decode shapes are pages x 8 and the old
#   value of 512 stops at 65.5K tokens.
# - --max-num-seqs 2 is deliberate: fewer batch shapes to warm up, and the measurements only cover 1-2 streams.
# - For a quick functional check after changing code, add VLLM_SKIP_WARMUP=true: the service is up in about
#   6 minutes and compiles each shape on first use. Never benchmark in that mode.
# - For diagnostics add VLLM_PROFILER_ENABLED=full, TORCH_LOGS=graph_breaks and --profiler-config (see
#   tools/README.md). The hardware trace is on by default; capture one or two rounds only.
# - The measured service listened on 0.0.0.0; this example binds to 127.0.0.1. Expose it deliberately.
# - The request guard rejects logprobs, penalties, logit bias and similar options for DFlash2 requests.
# - v6-v10 switches are described in docs/OPTIMIZATION.md. Each is off by default and independent.
#   $RUN_DIR/control.json toggles the runtime-only ones in the same process (read when the batch changes), e.g.
#   {"lookahead": true, "fast_topk": true, "presample": true, "delta": true}.
# - VLLM_HPU_DFLASH2_CPU_INPUTS / _CONTEXT_REGION also need VLLM_HPU_DFLASH2_DECODE_CONTROL pointing to a file with
#   {"cpu_inputs": true, "context_region": true}.
# - VLLM_HPU_LM_HEAD_FP8 changes logits slightly (measured top-1 agreement 98.4%, mean KL 0.0016); set
#   VLLM_HPU_LM_HEAD_FP8_CHECK=1 to log the comparison (diagnostic: it adds a BF16 projection).
# - VLLM_HPU_STEP_TIMER is cheap; tools/trace/steps_stats.py and phase_stats.py read it.
# - PT_HPU_RECIPE_CACHE_CONFIG requires GC_KERNEL_PATH to be a single file, hence the shim.

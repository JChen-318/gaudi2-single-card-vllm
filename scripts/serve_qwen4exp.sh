#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (NVFP4) on one Gaudi2 with the measured host and graph setup.
#
#   scripts/serve_qwen4exp.sh [extra vllm serve arguments]
#
# Environment: MODEL, PORT, MAX_MODEL_LEN, GPU_BLOCKS. DECODE_GRAPH=0, GDN_PREFILL_GRAPH=0 and
# MASKED_DENSE_MAX_KEYS=0 together give the byte-identical eager path. WARMUP=0 skips the startup
# requests that compile the graphs before the first user request. PIN=1 pins the engine threads to one CCD:
# it paid off while decode launched ~2,600 eager ops per token, but with the decode graph the free
# layout is as fast for decode and about 5% faster for prefill.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$HOME/models/qwen4exp-flashnext-plain}"
PORT="${PORT:-8000}"

export PATH="/usr/bin/habanatools:$PATH"
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export HABANA_LOGS="${HABANA_LOGS:-/var/log/habana_logs/}"
export HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw
export HABANA_PLUGINS_LIB_PATH=/usr/lib/habanatools/habana_plugins
export PT_HPU_LAZY_MODE=0
export VLLM_HPU_PREFILL_TILE_REPLAY=0
# glibc malloc is 43% of the eager lowering thread; tcmalloc takes decode from 51 to 40 ms per token.
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
export VLLM_QWEN4EXP_DECODE_GRAPH="${DECODE_GRAPH:-1}"
export VLLM_QWEN4EXP_GDN_PREFILL_GRAPH="${GDN_PREFILL_GRAPH:-1}"
export VLLM_QWEN4EXP_QSA_MASKED_DENSE_MAX_KEYS="${MASKED_DENSE_MAX_KEYS:-24576}"
# One log line per Dynamo recompilation with the guard that failed: a compile during a user request
# shows up with its cause (scripts/serve.sh does the same for the 27B).
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
# Allocator state every N engine steps (0 = off).
export VLLM_HPU_MEM_LOG="${MEM_LOG:-0}"
# H31: host-side decode preparation, as scripts/serve.sh runs the 27B. The fast state-index and
# group-routing builders are cross-checked against the reference builders for the first 256 live
# rounds and switch themselves off on any mismatch. A decode step spent 16% of its Python time in the
# reference builders, and the host issue path (12.7 ms of a 16.6 ms step) sets the decode pace.
export VLLM_HPU_FAST_DECODE_PREP="${FAST_DECODE_PREP:-all}"
export VLLM_HPU_FAST_DECODE_PREP_VERIFY="${FAST_DECODE_PREP_VERIFY:-256}"
export VLLM_HPU_ENVS_CACHE=1
export VLLM_HPU_SHARE_GROUP_MASKS=1

# H31: the model's full 262,144-token context. QSA keeps its own cache (about 7.2 GiB at 262K) and the
# paged KV is virtual, so the 4,096 scheduler blocks below cost 60 MiB; a 200K prefill peaks at 88 GiB.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# H31: prefill chunk. Real text routes a chunk to nearly all 512 experts, so a chunk dequantizes
# about the same expert weights whatever its length; 4,096 tokens amortize that over twice the rows
# of 2,048 (12K prompt 7.2 -> 5.9 s, 255K 259 -> 227 s, peak 89.4 GiB). 8,192 is another 5-7 %
# faster but leaves about 3 GiB of headroom at 255K.
# The decode attention graphs are compiled for every buffer capacity up to this length at the first
# decode step above the 2,048-token budget (the 8K warmup request below).
export VLLM_QWEN4EXP_WARM_MAX_TOKENS="$MAX_MODEL_LEN"

if [ "${WARMUP:-1}" = 1 ]; then
  # The graphs compile on first use (about 1 s each). One prompt per prefill query bucket of the
  # 4,096-token chunk (128, 256, 384, 512, 640, 896, 1280, 1664, 2304, 3072, 4096 rows; one unit is
  # 12 tokens on top of ~57 template tokens) and an 8K prompt whose decode compiles every attention
  # capacity take that cost before a user does.
  (
    until curl -fs -m 2 -o /dev/null "http://127.0.0.1:$PORT/health"; do sleep 2; done
    for units in 1 12 22 33 45 63 95 125 170 230 320 640; do
      text=$(printf 'The quick survey of trade routes shows how logistics shaped politics. %.0s' $(seq 1 "$units"))
      curl -fs -m 600 -o /dev/null "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$text Summarize.\"}],\"max_tokens\":8}" || true
    done
  ) &
fi

if [ "${PIN:-0}" = 1 ]; then
  # Pipeline thread names only appear once the engine has run a request, so warm it up first.
  (
    until curl -fs -m 2 -o /dev/null "http://127.0.0.1:$PORT/health"; do sleep 2; done
    curl -fs -m 300 -o /dev/null "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" || true
    "$ROOT/.venv/bin/python" "$ROOT/tools/qwen4exp_flashnext/pin_engine.py" pin
  ) &
fi

cd "$ROOT"
exec .venv/bin/python .venv/bin/vllm serve "$MODEL" \
  --load-format qwen4exp_plain --language-model-only --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "${BATCHED_TOKENS:-4096}" --max-num-seqs 1 \
  --num-gpu-blocks "${GPU_BLOCKS:-4096}" --no-enable-prefix-caching --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --port "$PORT" "$@"

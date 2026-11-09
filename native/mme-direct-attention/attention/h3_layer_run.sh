#!/usr/bin/env bash
# One full H3 attention layer: direct path vs production max_free_attention, isolated container.
# Usage: h3_layer_run.sh [layer=25] [call=1]   (card must be idle)
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
layer=${1:-25}; call=${2:-1}
[[ -z $(fuser /dev/accel/accel0 2>/dev/null) ]] || { echo 'Accelerator is in use.' >&2; exit 90; }
log=${LOG_NAME:-h3layer-call$call-layer$layer}
[[ ! -e "$HERE/evidence/$log.log" ]] || { echo "Log already exists: $log; choose a fresh LOG_NAME" >&2;exit 2; }
mkdir -p "$HERE/evidence/$log-hl" "$HERE/evidence/$log-ecb"
docker run --rm --name mme-h3-layer --network host --ipc host --cap-add SYS_NICE --ulimit memlock=-1 \
    --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
    -e HLS_MODULE_ID=7 -e HABANA_VISIBLE_MODULES=7 -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
    -e PT_HPU_LAZY_MODE=0 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e "SCAL_CONFIG_FILE_PATH=$HERE/scal-attention-h3-1.24.1.json" -e "H3DIRECT_ROOT=$HERE" \
    -e "H3_LAYER_FULL_REF_HEADS=${H3_LAYER_FULL_REF_HEADS:-}" \
    -e "H3_LAYER_REF_HEADS=${H3_LAYER_REF_HEADS:-0,27,55}" \
    -e "H3_ECB_AUDIT=${H3_ECB_AUDIT:-0}" \
    -e "H3_LAYER_HEAD_LIMIT=${H3_LAYER_HEAD_LIMIT:-0}" \
    -e "H3DIRECT_POW2Q=${H3DIRECT_POW2Q:-0}" \
    -e "H3DIRECT_RELAXED=${H3DIRECT_RELAXED:-0}" -e "H3DIRECT_STREAM=${H3DIRECT_STREAM:-0}" -e "H3DIRECT_FAST_GATES=${H3DIRECT_FAST_GATES:-0}" -e "H3DIRECT_COMPILED_GATES=${H3DIRECT_COMPILED_GATES:-0}" \
    -e "MME_ECB_DEBUG_DIR=$HERE/evidence/$log-ecb" \
    -e VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE=fast -e PT_HPU_SDPA_HEAD_FACTOR=2 -e VLLM_GAUDI_OMNI_HPU_COMPILE=1 \
    -e VLLM_GAUDI_OMNI_SDPA_MAXFREE=1 \
    -e "HABANA_LOGS=$HERE/evidence/$log-hl" \
    -v /home/a/h3-deploy/vllm-omni:/home/a/h3-deploy/vllm-omni:ro \
    -v /home/a/h3-omni-venv:/home/a/h3-omni-venv:ro -v /home/a/gsv-docker:/home/a/gsv-docker:ro \
    -v /mnt/sas/tmp/minimax-h3:/mnt/sas/tmp/minimax-h3:ro -v "$HERE:$HERE" -w "$HERE" \
    --entrypoint bash gaudi2-single-card-vllm:v17 -c '
        source /etc/profile.d/habanalabs.sh
        export HABANA_LOGS='"$HERE/evidence/$log-hl"'

        [[ $H3_ECB_AUDIT == 1 ]] || unset MME_ECB_DEBUG_DIR
        export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
        export PYTHONPATH=/home/a/h3-deploy/vllm-omni:/home/a/gsv-docker/vllm-core:/home/a/gsv-docker/vllm-gaudi:'"$HERE"'
        exec timeout 1500 /home/a/h3-omni-venv/bin/python "$@"
    ' bash "$HERE/${H3_LAYER_SCRIPT:-h3_direct_layer_test.py}" "/mnt/sas/tmp/minimax-h3/qkv/qkv-call$call-layer$layer.pt" \
      --repeats "${REPEATS:-3}" >"$HERE/evidence/$log.log" 2>&1
grep -E 'H3_LAYER|Error|error' "$HERE/evidence/$log.log" | cut -c1-2500 | tail -5

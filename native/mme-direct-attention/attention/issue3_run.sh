#!/usr/bin/env bash
# Issue #3 checks (issue3_test.py) in an isolated container with the video configuration
# (POW2Q, stream, fast + compiled gates) and the fixed attention.py as an overlay. Card must be idle.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
layer=${1:-25}
[[ -z $(fuser /dev/accel/accel0 2>/dev/null) ]] || { echo 'Accelerator is in use.' >&2; exit 90; }
log=${LOG_NAME:-issue3-layer$layer}
[[ ! -e "$HERE/evidence/$log.log" ]] || { echo "Log exists: $log" >&2; exit 2; }
SAS=${SAS_OUT:-/mnt/sas/tmp/mme-direct-run}/$log   # HL logs and graph dumps off the system disk
mkdir -p "$SAS"
docker run --rm --name mme-h3-issue3 --network host --ipc host --cap-add SYS_NICE --ulimit memlock=-1 \
    --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
    -e HLS_MODULE_ID=7 -e HABANA_VISIBLE_MODULES=7 -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
    -e PT_HPU_LAZY_MODE=0 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e "SCAL_CONFIG_FILE_PATH=$HERE/scal-attention-h3-1.24.1.json" -e "H3DIRECT_ROOT=$HERE" \
    -e "H3_ATTENTION_OVERLAY=$HERE/overlay/attention_issue3.py" \
    -e H3DIRECT_POW2Q=1 -e "H3DIRECT_KEYS=${H3DIRECT_KEYS:-37760}" -e "ISSUE3_LENGTHS=${ISSUE3_LENGTHS:-37760,37700,28800}" -e H3DIRECT_STREAM=1 -e H3DIRECT_FAST_GATES=1 -e H3DIRECT_COMPILED_GATES=1 -e H3DIRECT_LOG=0 \
    -e VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE=fast -e PT_HPU_SDPA_HEAD_FACTOR=2 -e VLLM_GAUDI_OMNI_HPU_COMPILE=1 \
    -e VLLM_GAUDI_OMNI_SDPA_MAXFREE=1 -e "HABANA_LOGS=$SAS" \
    -v /home/a/h3-deploy/vllm-omni:/home/a/h3-deploy/vllm-omni:ro \
    -v /home/a/h3-omni-venv:/home/a/h3-omni-venv:ro -v /home/a/gsv-docker:/home/a/gsv-docker:ro \
    -v /mnt/sas/tmp/minimax-h3:/mnt/sas/tmp/minimax-h3:ro -v "$HERE:$HERE" -v /mnt/sas/tmp:/mnt/sas/tmp -w "$SAS" \
    --entrypoint bash gaudi2-single-card-vllm:v17 -c '
        source /etc/profile.d/habanalabs.sh
        export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
        export PYTHONPATH=/home/a/h3-deploy/vllm-omni:/home/a/gsv-docker/vllm-core:/home/a/gsv-docker/vllm-gaudi:'"$HERE"'
        exec timeout 2400 /home/a/h3-omni-venv/bin/python "$@"
    ' bash "$HERE/issue3_test.py" "/mnt/sas/tmp/minimax-h3/qkv/qkv-call1-layer$layer.pt" \
      --expect-sha "${EXPECT_SHA-0b66cf40ed2bbcadd4c19ab94754fdadbd3da34b3ebab445e6cf291061d48138}" \
      >"$HERE/evidence/$log.log" 2>&1
grep -E 'ISSUE3 |ISSUE4 |QUAD_SET_VALID|QUAD_RELEASE|Error|error|Traceback' "$HERE/evidence/$log.log" | cut -c1-600 | tail -20

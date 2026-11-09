#!/usr/bin/env bash
# Run compiler_matmul_bw.py in an isolated container (card must be idle). Extra args pass through.
set -eu
O=/home/a/gsv-data/media/mme-direct-20261108/repro/attention
test -z "$(fuser /dev/accel/accel0 2>/dev/null)" || { echo 'Card busy'; exit 90; }
docker run --rm --name mme-compiler-matmul-bw --network host --ipc host --cap-add SYS_NICE --ulimit memlock=-1 \
  --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
  -e HLS_MODULE_ID=7 -e HABANA_VISIBLE_MODULES=7 \
  -v /home/a/h3-omni-venv:/home/a/h3-omni-venv:ro -v "$O:$O" -w "$O" \
  --entrypoint bash gaudi2-single-card-vllm:v17 -c "
set -eu
source /etc/profile.d/habanalabs.sh
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PT_HPU_LAZY_MODE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
timeout 900 /home/a/h3-omni-venv/bin/python $O/compiler_matmul_bw.py $*
"

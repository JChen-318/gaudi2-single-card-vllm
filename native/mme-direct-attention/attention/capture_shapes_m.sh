#!/usr/bin/env bash
# Capture compiler MME descriptors (PRINT_MME_DESCRIPTORS) for C[m,n] = A[m,k] @ B[k,n].
# Usage: capture_shapes_m.sh TAG "m n k" ["m n k" ...]   (runs isolated; card must be idle)
set -eu
O=/home/a/gsv-data/media/mme-direct-20261108/repro/attention
TAG=$1; shift
mkdir -p "$O/fixtures"
test -z "$(fuser /dev/accel/accel0 2>/dev/null)" || exit 90
for shape in "$@"; do
  read -r m n k <<< "$shape"
  out="$O/fixtures/$TAG-m${m}n${n}k${k}"
  rm -rf "$out"; mkdir -p "$out"
  docker run --rm --name mme-attention-shapes-m --network host --ipc host --cap-add SYS_NICE --ulimit memlock=-1 \
    --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
    -e HLS_MODULE_ID=7 -e HABANA_VISIBLE_MODULES=7 -e ENABLE_PIPELINE_MANAGEMENT="${ENABLE_PIPELINE_MANAGEMENT:-1}" -e CAPTURE_DTYPE="${CAPTURE_DTYPE:-bfloat16}" -e CAPTURE_OUTPUT_DTYPE="${CAPTURE_OUTPUT_DTYPE:-${CAPTURE_DTYPE:-bfloat16}}" -v /home/a/h3-omni-venv:/home/a/h3-omni-venv:ro -v "$O:$O" -w "$out" \
    -e HABANA_LOGS="$out" --entrypoint bash gaudi2-single-card-vllm:v17 -c "
set -eu
source /etc/profile.d/habanalabs.sh
export HABANA_LOGS=$out
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PT_HPU_LAZY_MODE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
export ENABLE_PIPELINE_MANAGEMENT=${ENABLE_PIPELINE_MANAGEMENT:-1}
export LOG_LEVEL_ALL=0 PRINT_MME_DESCRIPTORS=1 LOG_LEVEL_MME_DESC_DUMP=0 LOG_LEVEL_MME=0 ENABLE_EXPERIMENTAL_FLAGS=1
timeout 170 /home/a/h3-omni-venv/bin/python $O/capture_shapes.py $m $n $k
" > "$out/run.log" 2>&1 || echo "capture failed: $shape"
  echo "$shape: $(grep CAPTURE "$out/run.log") descriptors=$(find "$out" -name '*descriptor_dump*' | wc -l)"
done

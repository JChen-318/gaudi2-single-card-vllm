#!/bin/bash
# One-call VAE decoder traces with production VAE settings (tools/minimax_h3/trace_vae_decoder_batch.py).
# Loads only the video VAE. Batch 1 first (records the reference), then the given batch sizes.
# Usage: vae_decoder_trace.sh TAG [batch ...]   (card must be idle; never kills other workloads)
set -euo pipefail
tag=${1:?tag}; shift
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
O=/home/a/gsv-data/media/mme-direct-20261108/repro/attention
P="$O/evidence/vae-dec-$tag"
if fuser /dev/accel/accel0 >/dev/null 2>&1; then echo 'Card busy'; exit 1; fi
[ ! -e "$P" ] || { echo 'Output exists'; exit 1; }
mkdir -p "$P"
for b in 1 "$@"; do
  docker run --rm --name "gaudi2-h3-vaetrace-$tag" --network host --ipc host --cap-add SYS_NICE --ulimit memlock=-1 \
   --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
   --mount type=bind,src=/home/a/h3-deploy,dst=/home/a/h3-deploy,readonly \
   --mount type=bind,src=/home/a/h3-omni-venv,dst=/home/a/h3-omni-venv,readonly \
   --mount type=bind,src=/home/a/gsv-docker,dst=/home/a/gsv-docker,readonly \
   --mount type=bind,src=/mnt/sas/models/minimax-h3,dst=/mnt/sas/models/minimax-h3,readonly \
   --mount type=bind,src="$O",dst="$O" \
   -e HLS_MODULE_ID=7 -e HABANA_VISIBLE_MODULES=7 -e PT_HPU_LAZY_MODE=0 -e TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
   -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e DIFFUSERS_OFFLINE=1 \
   -e VLLM_GAUDI_H3_VAE_SOFTMAX_MODE=fast -e VLLM_GAUDI_H3_VAE_KV_ALIGN=128 \
   -e PYTHONPATH=/home/a/gsv-docker/vllm-gaudi/tools/minimax_h3:/home/a/gsv-docker/vllm-core:/home/a/gsv-docker/vllm-gaudi \
   -w /home/a/gsv-docker/vllm-gaudi/tools/minimax_h3 \
   --entrypoint /bin/bash gaudi2-single-card-vllm:v17 -c "
     source /etc/profile.d/habanalabs.sh 2>/dev/null || true
     export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4 HABANA_LOGS=$P/hl-b$b
     mkdir -p \$HABANA_LOGS
     timeout 1200 /home/a/h3-omni-venv/bin/python trace_vae_decoder_batch.py /mnt/sas/models/minimax-h3/FL2VA/video_vae \
       --batch-size $b --warmups 2 --output-dir $P/b$b --reference-npy $P/reference-b1.npy" > "$P/run-b$b.log" 2>&1 \
   || { echo "batch $b failed"; tail -5 "$P/run-b$b.log"; exit 1; }
  echo "batch $b: $(grep -oE '"device_ms": [0-9.]+|"mismatch_count": [0-9]+|"trace_bytes": [0-9]+' "$P/b$b/result.json" | tr '\n' ' ')"
done
echo VAE_TRACE_DONE

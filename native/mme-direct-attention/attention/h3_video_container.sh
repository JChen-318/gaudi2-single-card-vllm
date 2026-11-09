#!/bin/bash
set -euo pipefail
if [ -f /etc/profile.d/habanalabs.sh ]; then source /etc/profile.d/habanalabs.sh; fi
O=/home/a/gsv-data/media/mme-direct-20261108/repro/attention
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$O/h3inject:$O:/home/a/gsv-docker/vllm-core:/home/a/gsv-docker/vllm-gaudi"
export H3_BASE_SITECUSTOMIZE=/home/a/gsv-data/media/h3prof/sitecustomize.py
export H3DIRECT_ROOT="$O" H3DIRECT_RELAXED=0 H3DIRECT_LOG=1
export H3DIRECT_STREAM="${H3DIRECT_STREAM:-0}" H3DIRECT_FAST_GATES="${H3DIRECT_FAST_GATES:-0}" H3DIRECT_COMPILED_GATES="${H3DIRECT_COMPILED_GATES:-0}"
export SCAL_CONFIG_FILE_PATH="$O/scal-attention-h3-1.24.1.json"
export PT_HPU_LAZY_MODE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
export IMAGEIO_FFMPEG_EXE=/home/a/h3-deploy/media-bin/ffmpeg
export HABANA_LOGS="$H3PROF_DIR/hl" VLLM_GAUDI_LOCK_DIR=/tmp/h3locks
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Reference variants (defaults = production r24 settings).
export VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE=${H3_SDPA_SOFTMAX_MODE:-fast} PT_HPU_SDPA_HEAD_FACTOR=2
export VLLM_GAUDI_OMNI_HPU_COMPILE=1 VLLM_GAUDI_OMNI_SDPA_MAXFREE=${H3_SDPA_MAXFREE:-1}
export VLLM_GAUDI_OMNI_SDPA_MAXFREE_SKIP_LAYERS=45,48,49
export VLLM_GAUDI_H3_VAE_SOFTMAX_MODE=fast VLLM_GAUDI_H3_VAE_KV_ALIGN=128
unset VLLM_PLUGINS MME_ECB_DEBUG_DIR
mkdir -p "$HABANA_LOGS" "$VLLM_GAUDI_LOCK_DIR" /mnt/h3-runtime/engine
cd /home/a/gsv-docker/vllm-gaudi
# Opt-in PCIe experiments (defaults reproduce the production launch exactly).
#  H3_TE_PIN=1        : text-encoder layerwise offload from pinned host memory (later CLI value wins)
#  H3_PHASE_OFFLOAD=0 : keep the DiT resident (t2va/FL2VA only; launcher then picks VAE tile batch 4)
#  H3_VAE_TILE_BATCH  : explicit --vae-tile-batch-size
launch_extra=(); serve_extra=()
[ "${H3_PHASE_OFFLOAD:-1}" = 1 ] || launch_extra+=(--no-phase-offload)
[ -z "${H3_VAE_TILE_BATCH:-}" ] || launch_extra+=(--vae-tile-batch-size "$H3_VAE_TILE_BATCH")
[ "${H3_TE_PIN:-0}" = 0 ] || serve_extra+=(--diffusion-offload-config '{"mode":"layer","components":["text_encoder"],"pin_memory":true}')
exec /home/a/h3-omni-venv/bin/python /home/a/h3-deploy/serve_single_hpu_fp8_experiment.py \
 /mnt/sas/models/minimax-h3 --partition FL2VA "${launch_extra[@]}" \
 --lightx2v-4step-lora /mnt/sas/models/minimax-h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
 --online-fp8 --experimental-lightx2v-fp8 --module 7 --host 127.0.0.1 --port "$H3_VIDEO_PORT" \
 --temp-dir /mnt/h3-runtime/engine --media-bin /home/a/h3-deploy/media-bin \
 -- --served-model-name MiniMaxAI/MiniMax-H3 --init-timeout 3600 --stage-init-timeout 3600 "${serve_extra[@]}"

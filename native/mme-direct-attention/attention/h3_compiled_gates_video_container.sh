#!/bin/bash
set -euo pipefail
if [ -f /etc/profile.d/habanalabs.sh ]; then source /etc/profile.d/habanalabs.sh; fi
O=/home/a/gsv-data/media/mme-direct-20261108/repro/attention
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$O/h3inject_compiled_gates:$O:/home/a/gsv-docker/vllm-core:/home/a/gsv-docker/vllm-gaudi"
export H3_BASE_SITECUSTOMIZE=/home/a/gsv-data/media/h3prof/sitecustomize.py
export H3DIRECT_ROOT="$O" H3DIRECT_RELAXED=0 H3DIRECT_LOG=1
export SCAL_CONFIG_FILE_PATH="$O/scal-attention-h3-1.24.1.json"
export PT_HPU_LAZY_MODE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
export IMAGEIO_FFMPEG_EXE=/home/a/h3-deploy/media-bin/ffmpeg
export HABANA_LOGS="$H3PROF_DIR/hl" VLLM_GAUDI_LOCK_DIR=/tmp/h3locks
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE=fast PT_HPU_SDPA_HEAD_FACTOR=2
export VLLM_GAUDI_OMNI_HPU_COMPILE=1 VLLM_GAUDI_OMNI_SDPA_MAXFREE=1
export VLLM_GAUDI_OMNI_SDPA_MAXFREE_SKIP_LAYERS=45,48,49
export VLLM_GAUDI_H3_VAE_SOFTMAX_MODE=fast VLLM_GAUDI_H3_VAE_KV_ALIGN=128
unset VLLM_PLUGINS MME_ECB_DEBUG_DIR
mkdir -p "$HABANA_LOGS" "$VLLM_GAUDI_LOCK_DIR" /mnt/h3-runtime/engine
cd /home/a/gsv-docker/vllm-gaudi
exec /home/a/h3-omni-venv/bin/python /home/a/h3-deploy/serve_single_hpu_fp8_experiment.py \
 /mnt/sas/models/minimax-h3 --partition FL2VA \
 --lightx2v-4step-lora /mnt/sas/models/minimax-h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
 --online-fp8 --experimental-lightx2v-fp8 --module 7 --host 127.0.0.1 --port "$H3_VIDEO_PORT" \
 --temp-dir /mnt/h3-runtime/engine --media-bin /home/a/h3-deploy/media-bin \
 -- --served-model-name MiniMaxAI/MiniMax-H3 --init-timeout 3600 --stage-init-timeout 3600

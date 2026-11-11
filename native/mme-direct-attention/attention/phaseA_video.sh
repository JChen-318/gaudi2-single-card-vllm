#!/usr/bin/env bash
# Phase A end-to-end: one service, H3DIRECT_KEYS=auto, four requests whose prompts select different buckets
# (forest warmup -> long ~600+ tokens -> fox 149 tokens -> forest again). Request 3 (forest, seed 2102) should
# be bit-identical to video-direct-pinres-20261109b/out1.mp4 (same config, fixed 37760 plan). Detached; SAS.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TAG=${1:-phaseA-auto-20261111a}
if [ -z "${PV_CHILD:-}" ]; then PV_CHILD=1 setsid nohup bash "$0" "$TAG" > "/mnt/sas/tmp/mme-direct-run/$TAG.out" 2>&1 < /dev/null & echo launched; exit 0; fi
H3DIRECT_STREAM=1 H3DIRECT_FAST_GATES=1 H3DIRECT_COMPILED_GATES=1 H3_TE_PIN=1 H3_PHASE_OFFLOAD=0 \
H3DIRECT_KEYS=auto H3_ATTENTION_OVERLAY="$HERE/overlay/attention_issue3.py" \
H3_VIDEO_PROMPTS_FILE="$HERE/phaseA_prompts.txt" H3_VIDEO_NREQ=4 \
  bash "$HERE/h3_direct_video_run.sh" direct "$TAG"
echo "VIDEO_DONE rc=$?"

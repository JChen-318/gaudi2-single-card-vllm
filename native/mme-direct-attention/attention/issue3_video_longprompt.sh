#!/usr/bin/env bash
# Issue #3 end-to-end check: a long prompt (well over 50 text tokens, so the attention length exceeds 37760)
# through the direct path with the 38272-key bucket, the fixed attention.py overlay and the 79.7 s
# configuration (stream, fast + compiled gates, pinned text encoder, resident DiT). Detached; outputs on SAS.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TAG=${1:-issue3-longprompt-20261111a}
if [ -z "${LP_CHILD:-}" ]; then LP_CHILD=1 setsid nohup bash "$0" "$TAG" > "/mnt/sas/tmp/mme-direct-run/$TAG.out" 2>&1 < /dev/null & echo launched; exit 0; fi
export H3_VIDEO_PROMPT="At golden hour, a red fox trots along a frosty ridge above a misty pine valley. The camera follows from a low angle as the fox pauses, ears turning toward the distant call of a raven, then leaps over a fallen log dusted with snow. Its breath forms small clouds in the cold air, and dry grass crunches under its paws. Far below, a narrow river glints between the trees while a light wind moves through the branches. The fox reaches a rocky outcrop, sits, and looks out over the valley as the sun touches the mountains. Natural ambient sound only: wind, footsteps on frozen ground, a raven calling, the faint rush of the river. No music, no voiceover."
H3DIRECT_STREAM=1 H3DIRECT_FAST_GATES=1 H3DIRECT_COMPILED_GATES=1 H3_TE_PIN=1 H3_PHASE_OFFLOAD=0 \
H3DIRECT_KEYS=38272 H3_ATTENTION_OVERLAY="$HERE/overlay/attention_issue3.py" \
  bash "$HERE/h3_direct_video_run.sh" direct "$TAG"
echo "VIDEO_DONE rc=$?"

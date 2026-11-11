#!/usr/bin/env bash
# Phase A, 8 buckets (every 512 keys): layer suite with H3DIRECT_KEYS=auto, then the long-prompt video
# (forest warmup, 607-token prompt twice -> 38784 bucket). Detached; logs in evidence/ and on SAS.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${PA2_CHILD:-}" ]; then PA2_CHILD=1 setsid nohup bash "$0" > "$HERE/evidence/phaseA2.out" 2>&1 < /dev/null & echo launched; exit 0; fi
cd "$HERE"
[[ -z $(docker ps -q --filter name=mme-h3) ]] || { echo "mme-h3 container running"; exit 1; }
for w in 9728 9856 9984 10112 10240 10368; do
  for d in g-m512n${w}k128 g-m512n128k${w}; do [ -d "fixtures/$d/mme_debug" ] || { echo "missing fixture $d"; exit 1; }; done
done
LOG_NAME=issue3-layer25-auto8 H3DIRECT_KEYS=auto bash issue3_run.sh 25
echo LAYER_DONE
PV2_CHILD=1 bash phaseA2_video.sh phaseA2-auto-20261111a
echo PHASEA2_DONE

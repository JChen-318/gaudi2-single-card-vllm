#!/usr/bin/env bash
# Phase A (automatic key-capacity buckets): rebuild the bridge (plan release + kernel-binary cache), then the
# issue #3 suite with H3DIRECT_KEYS=auto, which adds t5 (bucket switches in both directions). Detached.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${PA_CHILD:-}" ]; then PA_CHILD=1 setsid nohup bash "$0" > "$HERE/evidence/phaseA.out" 2>&1 < /dev/null & echo launched; exit 0; fi
cd "$HERE"
[[ -z $(docker ps -q --filter name=mme-h3) ]] || { echo "mme-h3 container running"; exit 1; }
for d in g-m512n10368k128 g-m512n128k10368; do [ -d "fixtures/$d/mme_debug" ] || { echo "missing fixture $d"; exit 1; }; done
BUILD_CONTEXT_PROBE=1 bash build.sh > evidence/phaseA-build.log 2>&1 || { echo BUILD_FAILED; grep -m5 error evidence/phaseA-build.log; exit 1; }
sha256sum -c backup-pre-issue3-20261111/bins.sha256 2>&1 | grep -v ': OK' && { echo KERNEL_BINS_CHANGED; exit 1; }
echo BUILD_OK
LOG_NAME=${LOG_NAME:-issue3-layer25-auto} H3DIRECT_KEYS=auto bash issue3_run.sh 25
echo PHASEA_DONE

#!/usr/bin/env bash
# Detached: rebuild the bridge, then the issue #3 suite on both key-capacity buckets. Output in evidence/.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${IB_CHILD:-}" ]; then IB_CHILD=1 setsid nohup bash "$0" > "$HERE/evidence/issue3-buckets.out" 2>&1 < /dev/null & echo launched; exit 0; fi
cd "$HERE"
[[ -z $(docker ps -q --filter name=mme-h3) ]] || { echo "mme-h3 container running"; exit 1; }
BUILD_CONTEXT_PROBE=1 bash build.sh > evidence/issue3-build2.log 2>&1 || { echo BUILD_FAILED; grep -m5 error evidence/issue3-build2.log; exit 1; }
sha256sum -c backup-pre-issue3-20261111/bins.sha256 2>&1 | grep -v ': OK' && { echo KERNEL_BINS_CHANGED; exit 1; }
echo BUILD_OK
LOG_NAME=issue3-layer25-k37760 H3DIRECT_KEYS=37760 bash issue3_run.sh 25
echo K37760_DONE
LOG_NAME=issue3-layer25-k38272 H3DIRECT_KEYS=38272 EXPECT_SHA= ISSUE3_LENGTHS=38272,38000,37760,29000 bash issue3_run.sh 25
echo K38272_DONE

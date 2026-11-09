#!/usr/bin/env bash
# Isolated experiment: MME descriptor writes + program end without engine
# barrier (commit keeps it). Small-integer exact checks. Stop on first failure.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawnb-$name" PROBE_TIMEOUT=120 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --scores-fp16 --repeats 10 "$@" \
    | grep -oE '"no_barrier": [a-z]+|"bulk": [a-z]+|"descriptors": [0-9]+|"median_us": [0-9.]+|"exact": [a-z]+|Error.*|assert.*' | tr '\n' ' '; echo
}
for e in ${RUNS:-bulk6 nb6 bulk2 nb2}; do
  echo -n "$e: "
  case $e in
    bulk6) run $e --kind qk --rows 128 --size 37760 --bulk ;;
    nb6) run $e --kind qk --rows 128 --size 37760 --no-barrier ;;
    bulk2) run $e --kind qk --rows 128 --size 37760 --bulk --coalesce-qk ;;
    nb2) run $e --kind qk --rows 128 --size 37760 --no-barrier --coalesce-qk ;;
  esac
done
echo RAWNB_DONE

#!/usr/bin/env bash
# QK efficiency vs M and output placement (2026-11-09). Small-integer inputs,
# exact output check; time = submit -> all completions (includes ~6 us floor).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawbig-$name" PROBE_TIMEOUT=300 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --kind qk --scores-fp16 --repeats 10 "$@" \
    | grep -oE '"rows": [0-9]+|"size": [0-9]+|"out_hbm": [a-z]+|"in_hbm": [a-z]+|"descriptors": [0-9]+|"median_us": [0-9.]+|"exact": [a-z]+|Error.*' | tr '\n' ' '; echo
}
for e in ${RUNS:-floor m128 m128h m128hh m1024 m1024h m2560s m2560sh m2560full m2560fullhh}; do
  echo -n "$e: "
  case $e in
    floor) run $e --rows 128 --size 128 ;;
    m128) run $e --rows 128 --size 37760 ;;
    m128h) run $e --rows 128 --size 37760 --out-hbm ;;
    m128hh) run $e --rows 128 --size 37760 --out-hbm --in-hbm ;;
    m1024) run $e --rows 1024 --size 4736 ;;
    m1024h) run $e --rows 1024 --size 4736 --out-hbm ;;
    m2560s) run $e --rows 2560 --size 1920 ;;
    m2560sh) run $e --rows 2560 --size 1920 --out-hbm ;;
    m2560full) run $e --rows 2560 --size 37760 --out-hbm ;;
    m2560fullhh) run $e --rows 2560 --size 37760 --out-hbm --in-hbm ;;
  esac
done
echo RAWBIG_DONE

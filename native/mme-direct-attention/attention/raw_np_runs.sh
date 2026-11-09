#!/usr/bin/env bash
# QK descriptors captured with ENABLE_PIPELINE_MANAGEMENT=0 (no SRAM slicing),
# replayed in the probe with exact small-integer checks (2026-11-09).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawnp-$name" PROBE_TIMEOUT=200 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --kind qk --scores-fp16 --repeats 10 --bulk "$@" \
    | grep -oE '"rows": [0-9]+|"size": [0-9]+|"fixture_tag": "[a-z]+"|"descriptors": [0-9]+|"median_us": [0-9.]+|"exact": [a-z]+|"mismatch_fraction": [0-9.e-]+|Error.*' | tr '\n' ' '; echo
}
for e in ${RUNS:-np512 g512 np1024 g1024 np2560}; do
  echo -n "$e: "
  case $e in
    np512) run $e --fixture-tag np --rows 512 --size 9472 ;;
    g512) run $e --fixture-tag g --rows 512 --size 9472 ;;
    np1024) run $e --fixture-tag np --rows 1024 --size 4736 ;;
    g1024) run $e --fixture-tag g --rows 1024 --size 4736 ;;
    np2560) run $e --fixture-tag np --rows 2560 --size 1920 ;;
  esac
done
echo RAWNP_DONE

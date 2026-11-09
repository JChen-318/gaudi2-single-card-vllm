#!/usr/bin/env bash
# storeEn1 experiment (2026-11-09): QK 128xN FP16 coalesced, Cout1 in its own
# poisoned SRAM buffer. Reports timing and what lands in Cout0/Cout1.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawst1-$name" PROBE_TIMEOUT=200 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --kind qk --rows 128 --scores-fp16 --coalesce-qk --bulk --repeats 20 "$@" > /dev/null 2>&1
  grep -oE '"median_us": [0-9.]+|"exact": [a-z]+|"written_fraction": [0-9.]+|"cout[01]_written_fraction": [0-9.]+|"cout1_matches_reference_where_written": [a-zA-Z]+|"union_written_fraction": [0-9.]+|"cout1_equals_cout0_full": [a-z]+|SCAL_TIMED_OUT|run [0-9]+: -?[0-9]+|Error[^,]*' "evidence/rawst1-$name.log" | sort -u | tr '\n' ' '; echo
}
for n in 1024 37760; do
  echo -n "n$n-st0: "; run n$n-st0 --size $n
  echo -n "n$n-st1: "; run n$n-st1 --size $n --store-en1
done
echo RAWST1_DONE

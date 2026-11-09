#!/usr/bin/env bash
# Where is the QK write wall? (2026-11-09) Single-engine QK and QK under
# background DMA write traffic. Small-integer inputs, exact checks.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawwall-$name" PROBE_TIMEOUT=300 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --kind qk --scores-fp16 --coalesce-qk --bulk --repeats 20 "$@" \
    | grep -oE '"descriptors": [0-9]+|"median_us": [0-9.]+|"exact": [a-z]+|"written_fraction": [0-9.]+|"bg_alone_us": \[[^]]*\]|Error.*' | tr '\n' ' '; echo
}
for n in 1024 37760; do
  echo -n "n$n-both: "; run n$n-both --size $n
  echo -n "n$n-e0: ";   run n$n-e0 --size $n --qk-engine 0
  echo -n "n$n-e1: ";   run n$n-e1 --size $n --qk-engine 1
done
echo -n "n37760-both-again: "; run n37760-both-again --size 37760
echo -n "n37760-bg-sram: "; run n37760-bg-sram --size 37760 --bg-dma sram --bg-mib 256
echo -n "n37760-bg-hbm: ";  run n37760-bg-hbm --size 37760 --bg-dma hbm --bg-mib 256
echo -n "n37760-e0-bg-sram: "; run n37760-e0-bg-sram --size 37760 --qk-engine 0 --bg-dma sram --bg-mib 256
echo RAWWALL_DONE

#!/usr/bin/env bash
# Cross-checks of the QK write wall (2026-11-09):
#  A) multi-point fit M128 FP16 (coalesced): N = 1024, 9472, 18944, 37760
#  B) read-light shape: M1024 FP16, N = 4736 / 9472 (same write bytes per N as above, far fewer K reads)
#  C) FP32 vs FP16 output with the difference method, uncoalesced: N = 1024 / 18944 / 37760
# Small-integer inputs, exact checks, 20 repeats, medians.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawbw2-$name" PROBE_TIMEOUT=300 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --kind qk --bulk --repeats 20 "$@" \
    | grep -oE '"rows": [0-9]+|"size": [0-9]+|"descriptors": [0-9]+|"median_us": [0-9.]+|"exact": [a-z]+|"mismatch_fraction": [0-9.e-]+|Error.*' | tr '\n' ' '; echo
}
for n in 1024 9472 18944 37760; do echo -n "A-m128-fp16c-n$n: "; run A-n$n --rows 128 --size $n --scores-fp16 --coalesce-qk; done
for n in 4736 9472; do echo -n "B-m1024-fp16-n$n: "; run B-n$n --rows 1024 --size $n --scores-fp16; done
for n in 1024 18944 37760; do echo -n "C-m128-fp16-n$n: "; run C16-n$n --rows 128 --size $n --scores-fp16; done
for n in 1024 18944 37760; do echo -n "C-m128-fp32-n$n: "; run C32-n$n --rows 128 --size $n --out-fp32 --fixture-tag g32; done
echo RAWBW2_DONE

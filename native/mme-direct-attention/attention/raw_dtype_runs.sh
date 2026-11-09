#!/usr/bin/env bash
# Is QK limited by output elements or output bytes? Same shapes, FP16 vs FP32
# output (compiler captures g / g32), exact small-integer checks.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawdt-$name" PROBE_TIMEOUT=200 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --kind qk --repeats 10 --bulk "$@" \
    | grep -oE '"rows": [0-9]+|"out_fp32": [a-z]+|"descriptors": [0-9]+|"median_us": [0-9.]+|"exact": [a-z]+|Error.*' | tr '\n' ' '; echo
}
echo -n "m128fp16: "; run m128fp16 --rows 128 --size 37760 --scores-fp16
echo -n "m128fp32: "; run m128fp32 --rows 128 --size 37760 --out-fp32 --fixture-tag g32
echo -n "m512fp16: "; run m512fp16 --rows 512 --size 9472 --scores-fp16
echo -n "m512fp32: "; run m512fp32 --rows 512 --size 9472 --out-fp32 --fixture-tag g32
echo RAWDT_DONE

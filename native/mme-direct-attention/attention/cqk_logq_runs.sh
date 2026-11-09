#!/usr/bin/env bash
# Isolated resident replay benchmark; does not measure full attention/video.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
for input in layer25-h0-q0 layer25-h55-q10000 layer25-h20-q37622; do
    PROBE_TIMEOUT=120 LOG_NAME="cqk-logq-$input" ALLOW_UNVALIDATED_ARC=1 \
    bash run_context_probe.sh scal-attention --input "inputs/$input.bin" \
        --pipeline --tpcs 24 --rearm-only --dual-qk --lut --split-pv --half-max \
        --defer-norm --pipelined 16 --repeats 20 --compiler-qk --wreg-bulk \
        --exp-kernel exp_lut_half_logq_b80_ms8
done
# Different queries on alternating blocks, every block normalized; check that
# reversing the odd result restores the even result exactly. Also wraps ring.
PROBE_TIMEOUT=120 LOG_NAME=cqk-logq-alternate ALLOW_UNVALIDATED_ARC=1 \
bash run_context_probe.sh scal-attention --input inputs/layer25-h0-q0.bin \
    --pipeline --tpcs 24 --rearm-only --dual-qk --lut --split-pv --half-max \
    --pipelined 16 --repeats 100 --compiler-qk --wreg-bulk --alternate-query \
    --exp-kernel exp_lut_half_logq_b80_ms8

#!/usr/bin/env bash
# Write wall or compute? (2026-11-09) QK at constant output, reduction depth K varied.
# Compiler captures g-m<M>n<N>k<K>; FP16 output, small-integer inputs, exact checks.
# Output-bound: time flat in K. Compute/geometry-bound: time grows with K.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawk-$name" PROBE_TIMEOUT=300 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw --kind qk --scores-fp16 --bulk --repeats 20 "$@" \
    | grep -oE '"descriptors": [0-9]+|"median_us": [0-9.]+|"exact": [a-z]+|"written_fraction": [0-9.]+|Error.*' | tr '\n' ' '; echo
}
for k in ${KS:-32 64 128 256 512}; do
  echo -n "k$k-m128n1024: "; run k$k-m128n1024 --rows 128 --size 1024 --k $k
  echo -n "k$k-m512n9472: "; run k$k-m512n9472 --rows 512 --size 9472 --k $k
  echo -n "k$k-m512n9472-e0: "; run k$k-m512n9472-e0 --rows 512 --size 9472 --k $k --qk-engine 0
done
echo RAWK_DONE

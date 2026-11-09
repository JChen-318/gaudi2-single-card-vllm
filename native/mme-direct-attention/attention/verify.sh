#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n $(fuser /dev/accel/accel0 2>/dev/null) ]]; then
    echo 'Accelerator is in use; verification did not start.' >&2; exit 90
fi
mkdir -p "$HERE/evidence"
exec 9>"/tmp/gaudi2-mme-attention-$(id -u).lock"
flock -n 9 || exit 90
{
    timeout 20 "$HERE/build/tpc_runner" "$HERE/build/relu.bin"
    timeout 20 "$HERE/build/tpc_runner" "$HERE/build/exp_poly.bin" exp
    timeout 30 "$HERE/build/attention_runner" "$HERE/build/exp_sum.bin" "$HERE/build/normalize.bin" 17 uniform 128 1 2
    for keys in 128 256 1024 8192 32768 37760; do
        timeout 60 "$HERE/build/attention_runner" "$HERE/build/exp_sum.bin" "$HERE/build/normalize.bin" 17 random "$keys" 24 5
    done
    # Optional exported real QKV files supplied as arguments.
    for input in "$@"; do
        timeout 60 "$HERE/build/attention_runner" "$HERE/build/exp_sum.bin" "$HERE/build/normalize.bin" 17 random 37760 24 20 "$input"
    done
} 2>&1 | tee "$HERE/evidence/verify-attention.log"

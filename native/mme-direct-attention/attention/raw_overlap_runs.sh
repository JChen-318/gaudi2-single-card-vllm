#!/usr/bin/env bash
# QK -> PV overlap on the same MMEs (2026-11-09). Production quarter geometry (M512, 9472 keys).
# Each case twice (A/B order) to expose drift. Small-integer inputs, QK exact, PV rel < 1%.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
run() { local name=$1; shift
  LOG_NAME="rawov-$name" PROBE_TIMEOUT=300 ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-overlap --repeats 30 "$@" \
    | grep -oE '"descriptors": [0-9]+|"median_us": [0-9.]+|"qk_exact": [a-z]+|"pv_rel_error": [0-9.e-]+|"pv_live_blocks": \[[^]]*\]|Error.*|assert.*' | tr '\n' ' '; echo
}
for pass in 1 2; do
  for nb in barrier nobarrier; do
    f=(); [ $nb = nobarrier ] && f=(--no-barrier)
    for m in qk pv qkpv pvqk; do
      echo -n "p$pass-$nb-$m: "; run p$pass-$nb-$m --mode $m "${f[@]}"
    done
  done
done
echo RAWOV_DONE

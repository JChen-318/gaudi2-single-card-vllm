#!/usr/bin/env bash
# Runs "log-name|probe args" lines from stdin, one isolated process each.
cd /home/a/gsv-data/media/mme-direct-20261108/repro/attention
while IFS='|' read -r name args; do
    [[ -n $name ]] || continue
    [[ -z $(fuser /dev/accel/accel0 2>/dev/null) ]] || { echo "busy before $name"; exit 90; }
    # shellcheck disable=SC2086
    LOG_NAME=$name ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-attention $args >/dev/null 2>&1
    echo "$name rc=$? replays=$(grep -c PIPELINE_REPLAY evidence/$name.log)"
done
echo SWEEP_DONE

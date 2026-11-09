#!/usr/bin/env bash
# Stage-timing ablation sweep; one isolated process per mode.
cd /home/a/gsv-data/media/mme-direct-20261108/repro/attention
input=${INPUT:-inputs/layer25-h0-q0.bin}
tag=${TAG:-real-h0}
for mode in ${MODES:-floor lbw qk qk-exp qk-exp-pv exp pv norm}; do
    [[ -z $(fuser /dev/accel/accel0 2>/dev/null) ]] || { echo "busy before $mode"; exit 90; }
    LOG_NAME=stage-$mode-$tag ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-attention \
        --input "$input" --pipeline --tpcs ${TPCS:-24} --repeats 10 --mode $mode >/dev/null 2>&1
    echo "$mode rc=$? $(grep -c PIPELINE_REPLAY evidence/stage-$mode-$tag.log) replays"
done
echo SWEEP_DONE

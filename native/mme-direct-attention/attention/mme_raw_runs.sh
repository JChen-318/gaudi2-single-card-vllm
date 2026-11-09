#!/usr/bin/env bash
# Usage: mme_raw_runs.sh "tag|args" ...  ; one isolated process each.
cd /home/a/gsv-data/media/mme-direct-20261108/repro/attention
for spec in "$@"; do
  name=${spec%%|*}; args=${spec#*|}
  [[ -z $(fuser /dev/accel/accel0 2>/dev/null) ]] || { echo busy; exit 90; }
  # shellcheck disable=SC2086
  LOG_NAME=$name ALLOW_UNVALIDATED_ARC=1 bash run_context_probe.sh scal-mme-raw $args >/dev/null 2>&1
  echo "$name rc=$? $(grep -h 'RESULT_MME_RAW\|Error' evidence/$name.log | cut -c1-400)"
done
echo "habanalabs_msgs=$(sudo -n dmesg 2>/dev/null | grep -c habanalabs)"

#!/usr/bin/env bash
# Detached runner for issue3_headfactor_probe.py. Output: evidence/issue3-headfactor.log; HL logs on SAS.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=/mnt/sas/tmp/mme-direct-run/headfactor
if [ -z "${HF_CHILD:-}" ]; then HF_CHILD=1 setsid nohup bash "$0" > "$HERE/evidence/issue3-headfactor.log" 2>&1 < /dev/null & echo launched; exit 0; fi
[[ -z $(docker ps -q --filter name=mme-h3) ]] || { echo "mme-h3 container running"; exit 1; }
mkdir -p "$OUT"
docker run --rm --name mme-h3-headfactor --network host --ipc host --cap-add SYS_NICE \
  --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
  -e HLS_MODULE_ID=7 -e HABANA_VISIBLE_MODULES=7 -e PT_HPU_LAZY_MODE=0 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e HABANA_LOGS=$OUT -v /home/a/h3-omni-venv:/home/a/h3-omni-venv:ro -v "$HERE:$HERE" -v /mnt/sas/tmp:/mnt/sas/tmp -w $OUT \
  --entrypoint bash gaudi2-single-card-vllm:v17 -c "source /etc/profile.d/habanalabs.sh; timeout 1500 /home/a/h3-omni-venv/bin/python $HERE/issue3_headfactor_probe.py"
echo "PROBE_DONE rc=$?"

#!/usr/bin/env bash
# Run the DSpine overlay on the existing v19 image; keep the original service intact.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:-$ROOT/config/dspine.env}
[ -f "$CONFIG" ] || { echo "copy config/dspine.env.example to $CONFIG and edit paths"; exit 1; }
CONFIG=$(realpath "$CONFIG")
[ $# -eq 0 ] || shift
# shellcheck disable=SC1090
. "$CONFIG"
: "${DATA_DIR:?set DATA_DIR in $CONFIG}"
mkdir -p "$DATA_DIR"
exec docker run -d --name "${CONTAINER_NAME:-gaudi2-dspine}" --restart unless-stopped \
  --network host --ipc host --cap-add SYS_NICE \
  --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
  --mount type=bind,src=/mnt/sas,dst=/mnt/sas,readonly \
  --mount "type=bind,src=$DATA_DIR,dst=$DATA_DIR" \
  --mount "type=bind,src=$CONFIG,dst=/run/dspine.env,readonly" \
  --mount "type=bind,src=$ROOT/vllm-gaudi,dst=/opt/gaudi2-single-card-vllm/vllm-gaudi,readonly" \
  --mount "type=bind,src=$ROOT/scripts/serve.sh,dst=/opt/gaudi2-single-card-vllm/scripts/serve.sh,readonly" \
  --entrypoint /bin/bash "${IMAGE:-gaudi2-single-card-vllm:v19}" \
  /opt/gaudi2-single-card-vllm/scripts/serve.sh --config /run/dspine.env "$@"

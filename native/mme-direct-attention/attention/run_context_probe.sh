#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-readonly}
flags=()
extra=()
script=context_probe.py
case "$mode" in
    readonly) ;;
    runtime-copy) flags+=(--runtime-copy) ;;
    scal-nop) flags+=(--scal-nop --runtime-copy) ;;
    scal-mme-ecb|scal-tpc)
        [[ ${ALLOW_UNVALIDATED_ARC:-0} == 1 ]] || { echo 'Unvalidated ARC experiment; read HANDOFF.md first.' >&2; exit 91; }
        flags+=(--scal-mme-ecb --runtime-copy)
        [[ $mode != scal-tpc ]] || flags+=(--scal-tpc) ;;
    scal-attention)
        [[ ${ALLOW_UNVALIDATED_ARC:-0} == 1 ]] || { echo 'Read HANDOFF.md before the isolated attention probe.' >&2; exit 91; }
        script=runtime_attention_probe.py
        shift
        flags+=("$@") ;;
    scal-mme-raw)
        [[ ${ALLOW_UNVALIDATED_ARC:-0} == 1 ]] || exit 91
        script=mme_raw_probe.py
        shift
        flags+=("$@") ;;
    scal-mme-overlap)
        [[ ${ALLOW_UNVALIDATED_ARC:-0} == 1 ]] || exit 91
        script=mme_overlap_probe.py
        shift
        flags+=("$@") ;;
    scal-batch)
        [[ ${ALLOW_UNVALIDATED_ARC:-0} == 1 ]] || exit 91
        script=batch_attention_probe.py
        shift
        flags+=("$@") ;;
    special-ops)
        # Default runtime config (no private SCAL profile): HPU special functions vs CPU.
        script=special_ops_check.py
        shift
        flags+=("$@") ;;
    scal-fence)
        [[ ${ALLOW_UNVALIDATED_ARC:-0} == 1 ]] || exit 91
        flags+=(--scal-fence --runtime-copy) ;;
    scal-tpc-fence)
        [[ ${ALLOW_UNVALIDATED_ARC:-0} == 1 ]] || exit 91
        flags+=(--scal-tpc-fence --runtime-copy) ;;
    *) echo 'Modes: readonly | runtime-copy | scal-nop | scal-mme-ecb' >&2; exit 2 ;;
esac
if [[ $mode == scal-* ]]; then
    extra+=(-e "SCAL_CONFIG_FILE_PATH=$HERE/scal-attention-1.24.1.json")
fi
[[ -z $(fuser /dev/accel/accel0 2>/dev/null) ]] || { echo 'Accelerator is in use.' >&2; exit 90; }
log=${LOG_NAME:-context-$mode}
mkdir -p "$HERE/evidence/$log-hl"
docker run --rm --name mme-attention-context-probe --network host --ipc host --cap-add SYS_NICE --ulimit memlock=-1 \
    --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
    -e HLS_MODULE_ID=7 -e HABANA_VISIBLE_MODULES=7 -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
    -e PT_HPU_LAZY_MODE=0 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e "PROBE_TIMEOUT=${PROBE_TIMEOUT:-60}" \
    -e "HABANA_LOGS=$HERE/evidence/$log-hl" "${extra[@]}" \
    -v /home/a/h3-omni-venv:/home/a/h3-omni-venv:ro -v "$HERE:$HERE" -w "$HERE" \
    --entrypoint bash gaudi2-single-card-vllm:v17 -c '
        source /etc/profile.d/habanalabs.sh
        export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
        exec timeout "$PROBE_TIMEOUT" /home/a/h3-omni-venv/bin/python "$@"
    ' bash "$HERE/$script" "${flags[@]}" >"$HERE/evidence/$log.log" 2>&1
tail -12 "$HERE/evidence/$log.log"

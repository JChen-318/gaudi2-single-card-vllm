#!/bin/bash
# Isolated, matched baseline/direct H3 benchmark. Never kills unrelated workloads.
set -euo pipefail
mode=${1:?baseline or direct}; tag=${2:?unique tag}
[[ "$mode" == baseline || "$mode" == direct ]] || exit 2
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
O=/home/a/gsv-data/media/mme-direct-20261108/repro/attention
OUT=${H3_VIDEO_OUT:-/mnt/sas/tmp/mme-direct-run}   # outputs (MP4, logs, HL logs) off the system disk
mkdir -p "$OUT"
P="$OUT/video-$tag"
name="gaudi2-h3-mme-$tag"; port=${H3_VIDEO_PORT:-8098}; nreq=${H3_VIDEO_NREQ:-3}
[[ "$port" =~ ^[0-9]+$ && "$nreq" =~ ^[1-9][0-9]*$ ]] || exit 2
if fuser /dev/accel/accel0 >/dev/null 2>&1; then echo 'Card busy'; exit 1; fi
if ss -ltn | awk '{print $4}' | grep -qE ":$port$"; then echo 'Port busy'; exit 1; fi
if docker container inspect "$name" >/dev/null 2>&1; then echo 'Container name exists'; exit 1; fi
if [ -e "$P" ]; then echo 'Output exists; choose a unique tag'; exit 1; fi
mkdir "$P"
printf 'starting\n' > "$P/status"
enable=0; [ "$mode" != direct ] || enable=1
pow2=${H3DIRECT_POW2Q:-$enable}
[[ "$pow2" == 0 || "$pow2" == 1 ]] || exit 2
printf 'mode=%s\ntag=%s\nport=%s\nnreq=%s\npow2_q=%s\nfirst_request=warmup\n' "$mode" "$tag" "$port" "$nreq" "$pow2" > "$P/config.txt"
printf 'sdpa_softmax_mode=%s\nsdpa_maxfree=%s\nstream=%s\nfast_gates=%s\ncompiled_gates=%s\n' "${H3_SDPA_SOFTMAX_MODE:-fast}" "${H3_SDPA_MAXFREE:-1}" "${H3DIRECT_STREAM:-0}" "${H3DIRECT_FAST_GATES:-0}" "${H3DIRECT_COMPILED_GATES:-0}" >> "$P/config.txt"
printf 'keys=%s\noverlay=%s\nprompt=%s\n' "${H3DIRECT_KEYS:-37760}" "${H3_ATTENTION_OVERLAY:-}" "${H3_VIDEO_PROMPT:-forest}" >> "$P/config.txt"
printf 'te_pin=%s\nphase_offload=%s\nvae_tile_batch=%s\ntrace_call=%s\ntrace_decode=%s\n' "${H3_TE_PIN:-0}" "${H3_PHASE_OFFLOAD:-1}" "${H3_VAE_TILE_BATCH:-default}" "${H3_TRACE_CALL:--1}" "${H3_TRACE_DECODE:-0}" >> "$P/config.txt"
sha256sum "$O"/h3_direct_attention.py "$O"/h3inject/sitecustomize.py "$O"/h3_video_container.sh "$O"/h3_direct_video_run.sh "$O"/build/libcontext_bridge.so "$O"/scal-attention-h3-1.24.1.json > "$P/source.sha256"
hl-smi -Q timestamp,memory.used,utilization.aip,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total -f csv > "$P/health-before.csv"
cid=''; monitor=''
cleanup() {
 rc=$?
 trap - EXIT
 if [ -n "$monitor" ]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
 if [ -n "$cid" ]; then
  docker logs "$cid" > "$P/server.log" 2>&1 || true
  docker inspect "$cid" > "$P/container.json" 2>/dev/null || true
  docker stop -t 60 "$cid" > "$P/stop.log" 2>&1 || true
  docker rm "$cid" >> "$P/stop.log" 2>&1 || true
 fi
 hl-smi -Q timestamp,memory.used,utilization.aip,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total -f csv > "$P/health-after.csv" || true
 printf 'finished rc=%s\n' "$rc" > "$P/status"
 exit "$rc"
}
trap cleanup EXIT
cid=$(docker run -d --name "$name" \
 --network host --ipc host --pid host --cap-add SYS_NICE --ulimit memlock=-1 \
 --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
 --mount type=bind,src=/home/a/h3-deploy,dst=/home/a/h3-deploy,readonly \
 --mount type=bind,src=/home/a/h3-omni-venv,dst=/home/a/h3-omni-venv,readonly \
 --mount type=bind,src=/home/a/gsv-docker,dst=/home/a/gsv-docker,readonly \
 --mount type=bind,src=/mnt/sas/models/minimax-h3,dst=/mnt/sas/models/minimax-h3,readonly \
 --mount type=bind,src=/mnt/sas/tmp/minimax-h3,dst=/mnt/h3-runtime \
 --mount type=bind,src=/opt/habanalabs/media,dst=/opt/habanalabs/media,readonly \
 --mount type=bind,src=/usr/bin/fuser,dst=/usr/local/bin/fuser,readonly \
 --mount type=bind,src=/home/a/gsv-data/media/h3prof,dst=/home/a/gsv-data/media/h3prof,readonly \
 --mount type=bind,src="$O",dst="$O" \
 --mount type=bind,src="$OUT",dst="$OUT" \
 -e H3DIRECT_STREAM="${H3DIRECT_STREAM:-0}" -e H3DIRECT_FAST_GATES="${H3DIRECT_FAST_GATES:-0}" -e H3DIRECT_COMPILED_GATES="${H3DIRECT_COMPILED_GATES:-0}" -e H3_SDPA_SOFTMAX_MODE="${H3_SDPA_SOFTMAX_MODE:-fast}" -e H3_SDPA_MAXFREE="${H3_SDPA_MAXFREE:-1}" -e H3DIRECT_KEYS="${H3DIRECT_KEYS:-37760}" -e H3_ATTENTION_OVERLAY="${H3_ATTENTION_OVERLAY:-}" -e H3_TE_PIN="${H3_TE_PIN:-0}" -e H3_PHASE_OFFLOAD="${H3_PHASE_OFFLOAD:-1}" -e H3_VAE_TILE_BATCH="${H3_VAE_TILE_BATCH:-}" -e H3PROF_DIR="$P" -e H3PROF_TRACE_CALL="${H3_TRACE_CALL:--1}" -e H3PROF_TRACE_DECODE="${H3_TRACE_DECODE:-0}" -e H3DIRECT_ENABLE="$enable" -e H3DIRECT_POW2Q="$pow2" -e H3_VIDEO_PORT="$port" \
 --entrypoint /bin/bash gaudi2-single-card-vllm:v17 "$O/h3_video_container.sh")
printf '%s\n' "$cid" > "$P/container-id"
(
 while docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null | grep -qx true; do
  hl-smi -Q timestamp,memory.used,utilization.aip,temperature.aip,power.draw -f csv,noheader || true
  sleep 5
 done
) > "$P/health-samples.csv" & monitor=$!
start=$(date +%s)
until curl -fs -m 3 -o /dev/null "http://127.0.0.1:$port/health"; do
 [ "$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null)" = true ] || { echo 'Container exited'; exit 1; }
 [ $(( $(date +%s) - start )) -lt 3000 ] || { echo 'Startup timeout'; exit 1; }
 sleep 10
done
printf 'healthy after %ss\n' "$(( $(date +%s) - start ))" | tee "$P/status"
for (( i=0; i<nreq; i++ )); do
 printf 'request %s running\n' "$i" > "$P/status"
 docker exec "$cid" /home/a/h3-omni-venv/bin/python /home/a/gsv-docker/vllm-gaudi/tools/minimax_h3/request_video.py \
  --api-url "http://127.0.0.1:$port/v1/videos/sync" --task t2va \
  --prompt "${H3_VIDEO_PROMPT:-A small stream flows through a sunlit forest. The camera slowly moves toward the water. Birds chirp in the trees and water gurgles over smooth stones. Natural continuous sound, no music.}" \
  --output "$P/out$i.mp4" --metadata "$P/out$i.json" --width 1344 --height 768 --duration 5.167 --fps 24 \
  --lightx2v-4step-lora /mnt/sas/models/minimax-h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --seed "$((2101 + i % 2))" > "$P/request$i.log" 2>&1
 echo "request $i completed at $(date -u +%FT%TZ)"
 docker logs "$cid" > "$P/server.log" 2>&1
done
echo 'All requests completed'

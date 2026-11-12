#!/bin/bash
# Runs inside the container after stack840.sh. STAGE=smoke: startup checks of one worker on GPU 0.
# STAGE=pilot: every slot runs one real episode. STAGE=loop: every slot runs episodes until none is left or the deadline.
# WORKERS_PER_GPU slots per visible GPU; a slot whose worker fails is retried once, a second failure stops the job.
set -euo pipefail
: "${STAGE:?}" "${L840_MODEL:?}" "${ARTIFACTS:?}" "${L840_PROTOCOL:?}" "${WORKERS_PER_GPU:?}"
WORKER=$L840_CODE/$(python -c "import json; print(json.load(open('$L840_PROTOCOL'))['models']['$L840_MODEL']['worker'])")
mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:blank:]')
(( ${#UUIDS[@]} > 0 )) || { echo "no GPU visible" >&2; exit 65; }
echo "SLOTS model=$L840_MODEL stage=$STAGE gpus=${#UUIDS[@]} workers_per_gpu=$WORKERS_PER_GPU threads=${L840_THREADS:-2} $(date -u +%FT%TZ)"

fail() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$ARTIFACTS/FAILED"; echo "FAILED $*"; }

slot_loop() {  # slot uuid
  local slot=$1 gpu=$2 rc failures=0 extra=()
  [[ "$STAGE" == pilot ]] && extra=(--max-episodes 1)
  [[ "$STAGE" == smoke ]] && extra=(--smoke)
  while :; do
    [[ -e "$ARTIFACTS/FAILED" ]] && return 1
    echo "SLOT_START slot=$slot gpu=$gpu $(date -u +%H:%M:%S)"
    set +e
    CUDA_VISIBLE_DEVICES=$gpu python "$WORKER" --protocol "$L840_PROTOCOL" --worker "$slot" "${extra[@]}" \
      >> "$ARTIFACTS/slot_$slot.log" 2>&1
    rc=$?
    set -e
    if (( rc == 0 )); then echo "SLOT_DONE slot=$slot $(date -u +%H:%M:%S)"; return 0; fi
    tail -c 3000 "$ARTIFACTS/slot_$slot.log"
    failures=$(( failures + 1 ))
    if (( failures >= 2 )); then fail "slot $slot exited $rc (second failure)"; return 1; fi
    echo "SLOT_RETRY slot=$slot rc=$rc"
    sleep 30
  done
}

if [[ "$STAGE" == t5prep ]]; then
  CUDA_VISIBLE_DEVICES=${UUIDS[0]} python "$L840_CODE/t5prep840.py" --protocol "$L840_PROTOCOL" 2>&1 | tee "$ARTIFACTS/t5prep.log"
  exit "${PIPESTATUS[0]}"
fi
left=$(python -c "import sys; sys.path.insert(0, '$L840_CODE'); import l840; P = l840.read('$L840_PROTOCOL'); print(P['benchmark']['episodes'] - l840.done(l840.results(P, '$L840_MODEL'), P['benchmark']['episodes']))")
if [[ "$STAGE" == loop && "$left" == 0 ]]; then
  echo "NOTHING_LEFT all episodes are saved"
  python "$L840_CODE/analysis840.py" --protocol "$L840_PROTOCOL" --model "$L840_MODEL"
  exit 0
fi
PIDS=()
if [[ "$STAGE" == smoke ]]; then
  slot_loop 0 "${UUIDS[0]}" & PIDS+=("$!")
else
  slot=0
  for uuid in "${UUIDS[@]}"; do
    for (( w = 0; w < WORKERS_PER_GPU; w++ )); do
      slot_loop "$slot" "$uuid" & PIDS+=("$!")
      slot=$(( slot + 1 ))
      sleep 5
    done
  done
fi
rc=0
while :; do
  if [[ -e "$ARTIFACTS/FAILED" ]]; then
    for pid in "${PIDS[@]}"; do pkill -TERM -P "$pid" 2>/dev/null || true; kill -TERM "$pid" 2>/dev/null || true; done
    rc=1
    break
  fi
  alive=0
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=$(( alive + 1 )); done
  (( alive == 0 )) && break
  sleep 20
done
for pid in "${PIDS[@]}"; do wait "$pid" || rc=1; done
grep -h '^STARTUP' "$ARTIFACTS"/slot_*.log || true
python "$L840_CODE/analysis840.py" --protocol "$L840_PROTOCOL" --model "$L840_MODEL" || rc=1
exit "$rc"

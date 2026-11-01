# Shared by run.pbs (node 0) and node.sh (node 1); sourced after environment.sh. Needs ARTIFACTS, RESULTS, NODE_TAG.
# A failure anywhere writes $ARTIFACTS/FAILED; every slot loop and both node supervisors stop when it appears.

fail() {
  printf '%s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NODE_TAG" "$*" >> "$ARTIFACTS/FAILED"
  echo "FAILED $NODE_TAG $*"
}

monitor() {  # ncpus uuid...: GPU_SAMPLE every 60 s, CPU_SAMPLE (busy cores of this user over 2 s) every 120 s
  local ncpus=$1; shift
  local me n=0
  me=$(id -un)
  while :; do
    for uuid in "$@"; do
      printf 'GPU_SAMPLE timestamp=%s node=%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NODE_TAG"
      nvidia-smi -i "$uuid" --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader \
        || echo GPU_SAMPLE_ERROR
    done
    if (( n % 2 == 0 )); then
      top -b -n 2 -d 2 -u "$me" 2>/dev/null | awk -v ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)" -v req="$ncpus" -v node="$NODE_TAG" \
        '/^top -/ {frame++} frame == 2 && $1 ~ /^[0-9]+$/ {s += $9} END {printf "CPU_SAMPLE timestamp=%s node=%s busy_cores=%.1f requested=%s\n", ts, node, s / 100, req}' \
        || echo CPU_SAMPLE_ERROR
    fi
    n=$(( n + 1 ))
    sleep 60
  done
}

slot_loop() {  # slot gpu_uuid threads: load the arm the scheduler names, run its queue, repeat until it names none
  local slot=$1 gpu=$2 threads=$3 arm rc
  while :; do
    [[ -e "$ARTIFACTS/FAILED" ]] && return 1
    if ! arm=$(python3.11 "$ARTIFACTS/sched.py" choose --protocol "$ARTIFACTS/protocol.json" --slot "$slot"); then
      fail "scheduler failed for slot $slot"; return 1
    fi
    if [[ -z "$arm" ]]; then echo "SLOT_EXIT slot=$slot $(date -u +%H:%M:%S)"; return 0; fi
    echo "SLOT_START slot=$slot arm=$arm gpu=$gpu node=$NODE_TAG $(date -u +%H:%M:%S)"
    set +e
    CUDA_VISIBLE_DEVICES=$gpu FW_CPU_THREADS=$threads run_python "$ARTIFACTS/panel_worker.py" \
      --protocol "$ARTIFACTS/protocol.json" --arm "$arm" --worker "$slot" >> "$ARTIFACTS/slot_$slot.log" 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
      tail -c 4000 "$ARTIFACTS/slot_$slot.log"
      fail "slot $slot arm $arm exited $rc"; return 1
    fi
    echo "SLOT_DONE slot=$slot arm=$arm $(date -u +%H:%M:%S)"
  done
}

supervise() {  # pid...: wait for all; on FAILED kill the rest. Returns 0 only if every pid exited 0 and no FAILED.
  local pids=("$@") pid alive rc=0
  while :; do
    if [[ -e "$ARTIFACTS/FAILED" ]]; then
      for pid in "${pids[@]}"; do pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; done
      break
    fi
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=$(( alive + 1 )); done
    (( alive == 0 )) && break
    sleep 20
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  [[ -e "$ARTIFACTS/FAILED" ]] && rc=1
  return "$rc"
}

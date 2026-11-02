# Sourced by run_loop.pbs after environment.sh. Needs ARTIFACTS, DEADLINE_EPOCH.
# A failure anywhere writes $ARTIFACTS/FAILED; every slot loop and the supervisor stop when it appears.

fail() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$ARTIFACTS/FAILED"
  echo "FAILED $*"
}

slot_loop() {  # slot gpu_uuid threads: load the arm the scheduler names, run its queue, repeat until it names none
  local slot=$1 gpu=$2 threads=$3 arm rc failures=0
  while :; do
    [[ -e "$ARTIFACTS/FAILED" ]] && return 1
    if (( $(date +%s) > DEADLINE_EPOCH - 1800 )); then echo "SLOT_DEADLINE slot=$slot"; return 0; fi
    if ! arm=$(python3.11 "$ARTIFACTS/sched.py" choose --protocol "$ARTIFACTS/protocol.json" --slot "$slot"); then
      fail "scheduler failed for slot $slot"; return 1
    fi
    if [[ -z "$arm" ]]; then echo "SLOT_EXIT slot=$slot $(date -u +%H:%M:%S)"; return 0; fi
    echo "SLOT_START slot=$slot arm=$arm gpu=$gpu $(date -u +%H:%M:%S)"
    set +e
    CUDA_VISIBLE_DEVICES=$gpu COSMOS_CPU_THREADS=$threads run_python "$ARTIFACTS/loop_worker.py" \
      --protocol "$ARTIFACTS/protocol.json" --arm "$arm" --worker "$slot" >> "$ARTIFACTS/slot_$slot.log" 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
      tail -c 3000 "$ARTIFACTS/slot_$slot.log"
      failures=$(( failures + 1 ))
      if (( failures >= 2 )); then fail "slot $slot arm $arm exited $rc (second failure)"; return 1; fi
      echo "SLOT_RETRY slot=$slot arm=$arm rc=$rc"
      sleep 30
      continue
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

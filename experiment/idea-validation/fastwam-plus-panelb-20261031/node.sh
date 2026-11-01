#!/bin/bash
# pbsdsh task: started by run.pbs on every node of the allocation. The task on node 0 exits at once (run.pbs drives
# node 0 itself); the first task on node 1 runs that node's worker slots; any further task on node 1 exits.
set -euo pipefail

ENV_FILE=${1:?node_env.sh path required}
source "$ENV_FILE"
die() { echo "ERROR node.sh $(hostname -s): $*" >&2; printf '%s node1 %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$ARTIFACTS/FAILED"; exit 64; }
[[ "${PBS_JOBID:-}" == "$PARENT_JOBID" ]] || die "PBS_JOBID ${PBS_JOBID:-missing} differs from the parent $PARENT_JOBID"
HOST=$(hostname -s)
[[ "$HOST" == "$NODE0_HOST" ]] && exit 0
[[ "$HOST" == "$NODE1_HOST" ]] || die "Host $HOST is not node 1 ($NODE1_HOST)"
mkdir "$ARTIFACTS/node1_claim" 2>/dev/null || exit 0
case "$HOST" in *login*) die 'Refusing login-node execution';; esac
export PBS_NODEFILE=${PBS_NODEFILE:-$PARENT_NODEFILE}
[[ -r "$PBS_NODEFILE" ]] || die "PBS_NODEFILE $PBS_NODEFILE unreadable on node 1"
awk -F. -v host="$HOST" '$1 == host {found=1} END {exit !found}' "$PBS_NODEFILE" || die 'Node 1 is outside the PBS allocation'

NODE_TAG=node1
exec > >(tee -a "$ARTIFACTS/node1.log") 2>&1
echo "NODE1 host=$HOST PBS_TASKNUM=${PBS_TASKNUM:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} slots=$NODE1_FIRST_SLOT+$NODE1_SLOTS"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=, read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
  [[ ${#DEVICES[@]} -eq $NODE1_GPUS ]] || die "CUDA_VISIBLE_DEVICES names ${#DEVICES[@]} GPUs, expected $NODE1_GPUS"
  UUIDS=()
  for gpu in "${DEVICES[@]}"; do UUIDS+=("$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"); done
else
  mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:blank:]')
  [[ ${#UUIDS[@]} -eq $NODE1_GPUS ]] || die "No CUDA_VISIBLE_DEVICES and nvidia-smi shows ${#UUIDS[@]} GPUs, expected $NODE1_GPUS"
fi
for uuid in "${UUIDS[@]}"; do [[ "$uuid" == GPU-* ]] || die "Bad GPU UUID '$uuid'"; done
nvidia-smi -L

source "$ARTIFACTS/environment.sh"
source "$ARTIFACTS/slots.sh"
command -v python3.11 > /dev/null || die "python3.11 missing after module load"
MONITOR_PID=
cleanup() { [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null; true; }
trap cleanup EXIT
monitor "$NODE1_NCPUS" "${UUIDS[@]}" & MONITOR_PID=$!

THREADS=$(( NODE1_NCPUS / NODE1_SLOTS > 0 ? NODE1_NCPUS / NODE1_SLOTS : 1 ))
PIDS=()
for (( i = 0; i < NODE1_SLOTS; i++ )); do
  slot=$(( NODE1_FIRST_SLOT + i ))
  slot_loop "$slot" "${UUIDS[$(( i / SLOTS_PER_GPU ))]}" "$THREADS" & PIDS+=("$!")
  sleep 10
done
if supervise "${PIDS[@]}"; then echo "NODE1_DONE"; else echo "NODE1_FAILED"; exit 1; fi

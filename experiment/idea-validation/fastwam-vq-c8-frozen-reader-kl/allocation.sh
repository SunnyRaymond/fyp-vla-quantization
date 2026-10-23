#!/bin/bash
set -euo pipefail

test -n "${PBS_JOBID:-}" -a -s "${PBS_NODEFILE:-}"
case "$(hostname -s)" in *login*) echo 'Refusing login-node execution'; exit 64;; esac
awk -F. -v host="$(hostname -s)" '$1 == host {found=1} END {exit !found}' "$PBS_NODEFILE"
qstat -f "$PBS_JOBID" | grep -Eq '^[[:space:]]*job_state[[:space:]]*=[[:space:]]*R([[:space:]]|$)'

NEW_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-c8-frozen-reader-kl-20261022
CLIP_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-activation-clipping-20261017
PARENT=/scratch/users/ntu/yguo017/fastwam-vq-c8-action-latent-plus-20261015
ROOT=/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005
DIAG_ROOT=$NEW_ROOT
SMOOTH_ROOT=/scratch/users/ntu/yguo017/fastwam-smooth-vq-phases12-20261006
SOURCE_ROOT=/scratch/users/ntu/yguo017/fastwam-a4-phases12-20261006
BASE=/scratch/users/ntu/yguo017/fastwam-smoke
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
RUNTIME_ARTIFACTS=$PARENT/artifacts/25814264.pbs101
ARTIFACTS=$NEW_ROOT/artifacts/$PBS_JOBID
SOURCE_RUNTIME_ARTIFACTS=$(python3 - "$NEW_ROOT/protocol.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as stream:
    protocol = json.load(stream)
print(protocol['source_runtime_artifacts'])
PY
)
FW_JOB_START_EPOCH=$(date +%s)
export NEW_ROOT ROOT PARENT DIAG_ROOT SMOOTH_ROOT SOURCE_ROOT BASE SIF ARTIFACTS
export RUNTIME_ARTIFACTS SOURCE_RUNTIME_ARTIFACTS FW_JOB_START_EPOCH

mkdir -p "$ARTIFACTS"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
GPU_MONITOR_PID=
finish() {
  rc=$?
  trap - EXIT
  if [ -n "$GPU_MONITOR_PID" ]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  printf '%s\n' "$rc" > "$ARTIFACTS/exit_code.txt"
  exit "$rc"
}
trap finish EXIT

# The source snapshot supplies the exact C8 preparation/bank scripts and runtime.
cp "$SOURCE_RUNTIME_ARTIFACTS/"*.py "$ARTIFACTS/"
test -f "$ARTIFACTS/train_c8.py"
mv "$ARTIFACTS/train_c8.py" "$ARTIFACTS/train_c8_reference.py"
cp "$SOURCE_RUNTIME_ARTIFACTS/environment.sh" "$ARTIFACTS/"
cp "$SOURCE_RUNTIME_ARTIFACTS/protocol.json" "$ARTIFACTS/baseline_protocol.json"

# Overlay only the new experiment's local Python and frozen protocol.
cp "$NEW_ROOT/"*.py "$ARTIFACTS/"
cp "$NEW_ROOT/protocol.json" "$ARTIFACTS/"
source "$ARTIFACTS/environment.sh"

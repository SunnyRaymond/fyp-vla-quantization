#!/bin/bash
set -euo pipefail
test -n "${PBS_JOBID:-}" -a -s "${PBS_NODEFILE:-}"
case "$(hostname -s)" in *login*) echo 'Refusing login-node execution'; exit 64;; esac
awk -F. -v host="$(hostname -s)" '$1 == host {found=1} END {exit !found}' "$PBS_NODEFILE"
qstat -f "$PBS_JOBID" | grep -Eq '^[[:space:]]*job_state[[:space:]]*=[[:space:]]*R([[:space:]]|$)'
NEW_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-teacher-anchor-kl-20261021
CLIP_ROOT=/scratch/users/ntu/yguo017/fastwam-vq-activation-clipping-20261017
PARENT=/scratch/users/ntu/yguo017/fastwam-vq-c8-action-latent-plus-20261015
ROOT=/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005
DIAG_ROOT=$NEW_ROOT
SMOOTH_ROOT=/scratch/users/ntu/yguo017/fastwam-smooth-vq-phases12-20261006
SOURCE_ROOT=/scratch/users/ntu/yguo017/fastwam-a4-phases12-20261006
BASE=/scratch/users/ntu/yguo017/fastwam-smoke
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
RUNTIME_ARTIFACTS=$PARENT/artifacts/25814264.pbs101
SOURCE_ARTIFACTS=/scratch/users/ntu/yguo017/fastwam-vq-dual-kl-20261019/artifacts/25878493.pbs101
ARTIFACTS=$NEW_ROOT/artifacts/$PBS_JOBID
FW_JOB_START_EPOCH=$(date +%s)
export NEW_ROOT ROOT PARENT DIAG_ROOT SMOOTH_ROOT SOURCE_ROOT BASE SIF ARTIFACTS FW_JOB_START_EPOCH
mkdir -p "$ARTIFACTS"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
GPU_MONITOR_PID=
finish() {
 rc=$?
 trap - EXIT
 if [ -n "$GPU_MONITOR_PID" ]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi
 printf '%s\n' "$rc" > "$ARTIFACTS/exit_code.txt"
 exit "$rc"
}
trap finish EXIT
cp "$SOURCE_ARTIFACTS/"*.py "$ARTIFACTS/"
cp "$SOURCE_ARTIFACTS/source_protocol.json" "$ARTIFACTS/"
cp "$SOURCE_ARTIFACTS/environment.sh" "$ARTIFACTS/"
cp /scratch/users/ntu/yguo017/fastwam-vq-c80-clip-joint-plus-20261017/artifacts/25858606.pbs101/vq_adapter.py "$ARTIFACTS/previous_vq_adapter.py"
cp "$NEW_ROOT/"*.py "$ARTIFACTS/"
cp "$NEW_ROOT/protocol.json" "$ARTIFACTS/"
source "$ARTIFACTS/environment.sh"

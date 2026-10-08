#!/usr/bin/env bash
set -euo pipefail

# Run only inside an approved PBS GPU allocation. This starts the model server;
# it does not start RoboLab, render a scene, or run an episode.
TASK_ROOT="/scratch/users/ntu/yguo017/cosmos3-edge-robolab120"
SIF="/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif"
[[ -n "${PBS_JOBID:-}" ]] || { echo "FAIL_CLOSED: PBS_JOBID is missing"; exit 90; }
[[ -n "${PBS_NODEFILE:-}" && -r "$PBS_NODEFILE" ]] || { echo "FAIL_CLOSED: PBS_NODEFILE is missing or unreadable"; exit 91; }
HOST_SHORT="$(hostname -s)"
case "$HOST_SHORT" in asp2a-login*|*login*) echo "FAIL_CLOSED: login node $HOST_SHORT"; exit 92;; esac
awk -v h="$HOST_SHORT" '$1 == h || index($1, h ".") == 1 { found=1 } END { exit !found }' "$PBS_NODEFILE" || {
  echo "FAIL_CLOSED: current host is not in the PBS allocation"; exit 93;
}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "FAIL_CLOSED: PBS did not assign a visible GPU"; exit 94; }
[[ -r "$SIF" ]] || { echo "FAIL_CLOSED: reusable Ubuntu SIF is unavailable"; exit 95; }

RECEIPT="$TASK_ROOT/logs/cpu_stage_receipt.env"
[[ -r "$RECEIPT" ]] && grep -qx 'RESOURCES_STATUS=PASS' "$RECEIPT" && \
  grep -qx 'DEPENDENCIES_STATUS=PASS' "$RECEIPT" && \
  grep -qx 'COSMOS_DEPENDENCIES_STATUS=PASS' "$RECEIPT" || {
    echo "FAIL_CLOSED: Cosmos resources/server dependencies are not ready"; exit 96;
  }
[[ -d "$TASK_ROOT/envs/cosmos3-edge-policy-cu128" ]] || { echo "FAIL_CLOSED: Cosmos policy-server environment is missing"; exit 97; }
[[ -s "$TASK_ROOT/vendor/cosmos-framework/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth" ]] || {
  echo "FAIL_CLOSED: pinned Wan VAE is missing"; exit 98;
}
POLICY_SNAPSHOT="$(sed -n 's/^POLICY_SNAPSHOT=//p' "$RECEIPT")"
[[ -n "$POLICY_SNAPSHOT" && -d "$POLICY_SNAPSHOT" ]] || { echo "FAIL_CLOSED: pinned Edge policy snapshot is missing"; exit 99; }

if ! command -v apptainer >/dev/null 2>&1; then
  type module >/dev/null 2>&1 && module load apptainer/1.5.0
fi
command -v apptainer >/dev/null 2>&1 || { echo "FAIL_CLOSED: apptainer/1.5.0 is unavailable"; exit 100; }

mkdir -p "$TASK_ROOT/logs"
LOG="$TASK_ROOT/logs/policy-server-${PBS_JOBID}.log"
exec > >(tee -a "$LOG") 2>&1
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "$gpu_id" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "FAIL_CLOSED: invalid allocated GPU identifier"; exit 101; }
done

echo "PBS_JOBID=$PBS_JOBID"
echo "HOST=$(hostname -f)"
echo "ALLOCATED_GPU_IDS=$CUDA_VISIBLE_DEVICES"
echo "POLICY_SNAPSHOT=$POLICY_SNAPSHOT"
echo "SERVER_LOG=$LOG"
echo "SIMULATOR_STATUS=NOT_STARTED_POLICY_SERVER_ONLY"

(
  while kill -0 "$$" 2>/dev/null; do
    for gpu_id in "${GPU_IDS[@]}"; do
      echo "GPU_SAMPLE_BEGIN=$(date -Is) GPU_ID=$gpu_id"
      nvidia-smi --id="$gpu_id" --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader || {
        echo "GPU_SAMPLE_FAILED GPU_ID=$gpu_id"
        kill -TERM "$$"
        exit 1
      }
    done
    sleep 10
  done
) &
GPU_MONITOR_PID=$!
SERVER_PID=""
cleanup() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export HF_HOME="$TASK_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COSMOS_ROOT="$TASK_ROOT/vendor/cosmos-framework"
apptainer exec --nv --bind "$TASK_ROOT:$TASK_ROOT" \
  --env "TASK_ROOT=$TASK_ROOT" \
  --env "POLICY_SNAPSHOT=$POLICY_SNAPSHOT" \
  --env "HF_HOME=$HF_HOME" \
  --env "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --pwd "$COSMOS_ROOT" "$SIF" /bin/bash -c '
    set -euo pipefail
    export PATH="$TASK_ROOT/tools/uv:$PATH"
    exec "$TASK_ROOT/envs/cosmos3-edge-policy-cu128/bin/python" \
      -m cosmos_framework.scripts.action_policy_server_robolab \
      --checkpoint-path "$POLICY_SNAPSHOT" --port 8000 \
      --format-prompt-as-json True --guidance-interval 960 1001
  ' &
SERVER_PID=$!
set +e
wait "$SERVER_PID"
SERVER_STATUS=$?
set -e
SERVER_PID=""
exit "$SERVER_STATUS"

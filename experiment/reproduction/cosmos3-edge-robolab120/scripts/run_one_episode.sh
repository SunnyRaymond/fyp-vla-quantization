#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="/scratch/users/ntu/yguo017/cosmos3-edge-robolab120"
SIF="/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif"
[[ -n "${PBS_JOBID:-}" && -n "${PBS_NODEFILE:-}" ]] || { echo "FAIL_CLOSED: launch only inside an allocated PBS job"; exit 90; }
[[ -r "$PBS_NODEFILE" ]] || { echo "FAIL_CLOSED: PBS_NODEFILE is unreadable"; exit 91; }
HOST_SHORT="$(hostname -s)"
case "$HOST_SHORT" in asp2a-login*|*login*) echo "FAIL_CLOSED: refusing launch on login node $HOST_SHORT"; exit 92;; esac
awk -v h="$HOST_SHORT" '$1 == h || index($1, h ".") == 1 { found=1 } END { exit !found }' "$PBS_NODEFILE" || {
  echo "FAIL_CLOSED: current host is not in PBS_NODEFILE"; exit 93;
}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "FAIL_CLOSED: CUDA_VISIBLE_DEVICES must identify allocated GPU(s)"; exit 94; }
[[ -d "$TASK_ROOT/envs/cosmos3-edge-policy-cu128" ]] || {
  echo "FAIL_CLOSED: Cosmos Edge policy-server environment is missing"; exit 95;
}
[[ -d "$TASK_ROOT/envs/robolab-isaac51" ]] || {
  echo "SIM_ENV_NOT_INSTALLED: RoboLab/Isaac runtime was deliberately skipped; prepare it only on a compatible RTX-enabled machine"; exit 95;
}
[[ -s "$TASK_ROOT/vendor/cosmos-framework/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth" ]] || {
  echo "FAIL_CLOSED: pinned Wan VAE is not staged at the framework lookup path"; exit 96;
}
[[ -r "$SIF" ]] || { echo "FAIL_CLOSED: reusable Ubuntu SIF is unavailable"; exit 97; }
RECEIPT="$TASK_ROOT/logs/cpu_stage_receipt.env"
[[ -r "$RECEIPT" ]] && grep -qx 'RESOURCES_STATUS=PASS' "$RECEIPT" && grep -qx 'DEPENDENCIES_STATUS=PASS' "$RECEIPT" || {
  echo "FAIL_CLOSED: CPU-stage completion receipt is missing"; exit 99;
}
POLICY_SNAPSHOT="$(sed -n 's/^POLICY_SNAPSHOT=//p' "$RECEIPT")"
[[ -d "$POLICY_SNAPSHOT" ]] || { echo "FAIL_CLOSED: pinned policy snapshot is missing"; exit 100; }
if ! command -v apptainer >/dev/null 2>&1; then
  module load apptainer/1.5.0
fi

export HF_HOME="$TASK_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN_ID="cosmos3_banana_single_$(date -u +%Y%m%dT%H%M%SZ)"
COSMOS_ROOT="$TASK_ROOT/vendor/cosmos-framework"
ROBOLAB_ROOT="$TASK_ROOT/vendor/RoboLab"
mkdir -p "$TASK_ROOT/logs"
exec > >(tee -a "$TASK_ROOT/logs/gpu-run-$PBS_JOBID.log") 2>&1
GPU_LOG="$TASK_ROOT/logs/gpu-util-$PBS_JOBID.csv"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "$gpu_id" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "FAIL_CLOSED: invalid allocated GPU identifier"; exit 98; }
done
(
  while kill -0 "$$" 2>/dev/null; do
    for gpu_id in "${GPU_IDS[@]}"; do
      nvidia-smi --id="$gpu_id" --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader | tee -a "$GPU_LOG"
    done
    sleep 5
  done
) &
GPU_MONITOR_PID=$!
SERVER_PID=""
cleanup() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

container_args=(exec --nv --bind "$TASK_ROOT:$TASK_ROOT" --env "TASK_ROOT=$TASK_ROOT"
  --env "HF_HOME=$HF_HOME" --env "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
  --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1"
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" --env "RUN_ID=$RUN_ID"
  --env "POLICY_SNAPSHOT=$POLICY_SNAPSHOT")

echo "PBS_JOBID=$PBS_JOBID"
echo "HOST=$(hostname -f)"
echo "ALLOCATED_GPU_IDS=$CUDA_VISIBLE_DEVICES"
echo "RUN_ID=$RUN_ID"
echo "GPU_METRICS_LOG=$GPU_LOG"

apptainer "${container_args[@]}" --pwd "$COSMOS_ROOT" "$SIF" /bin/bash -c '
    set -euo pipefail
    export PATH="$TASK_ROOT/tools/uv:$PATH"
    export ISAACSIM_CACHE_DIR="$TASK_ROOT/cache/isaac-sim"
    exec "$TASK_ROOT/envs/cosmos3-edge-policy-cu128/bin/python" -m cosmos_framework.scripts.action_policy_server_robolab \
      --checkpoint-path "$POLICY_SNAPSHOT" --port 8000 \
      --format-prompt-as-json True --guidance-interval 960 1001
  ' > "$TASK_ROOT/logs/cosmos-server-$PBS_JOBID.log" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 120); do
  if apptainer "${container_args[@]}" --pwd "$COSMOS_ROOT" "$SIF" "$TASK_ROOT/envs/cosmos3-edge-policy-cu128/bin/python" -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=2).read()' >/dev/null 2>&1; then
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "FAIL: Cosmos server exited before becoming healthy"; exit 101; }
  sleep 5
done
[[ "$READY" -eq 1 ]] || { echo "FAIL: Cosmos server health timeout"; exit 102; }

apptainer "${container_args[@]}" --pwd "$ROBOLAB_ROOT" "$SIF" /bin/bash -c '
    set -euo pipefail
    export ISAACSIM_CACHE_DIR="$TASK_ROOT/cache/isaac-sim"
    export PATH="$TASK_ROOT/tools/runtime-bin:$PATH"
    exec "$TASK_ROOT/envs/robolab-isaac51/bin/python" policies/cosmos3/run.py \
      --task BananaInBowlTask --num-envs 1 --num-runs 1 --headless \
      --remote-host 127.0.0.1 --remote-port 8000 --instruction-type default \
      --video-mode all --output-folder-name "$RUN_ID"
  '

RESULTS="$TASK_ROOT/vendor/RoboLab/output/$RUN_ID/episode_results.jsonl"
apptainer "${container_args[@]}" "$SIF" "$TASK_ROOT/envs/robolab-isaac51/bin/python" - "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 1, f"expected exactly one episode row, found {len(rows)}"
r = rows[0]
assert r.get("task_name", r.get("env_name")) == "BananaInBowlTask", r
assert r.get("policy") == "cosmos3", r
assert r.get("run") == 0 and r.get("episode") == 0 and r.get("env_id") == 0, r
assert int(r.get("episode_step", 0)) > 0, r
assert isinstance(r.get("success"), bool), r
print("ONE_EPISODE_RESULT=PASS")
print(json.dumps(r, sort_keys=True))
PY
echo "EPISODE_RESULT=$RESULTS"
echo "ONE_EPISODE_STATUS=PASS"

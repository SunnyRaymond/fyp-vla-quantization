#!/bin/bash
set -euo pipefail
STAGE=${1:?stage is required}
case "$STAGE" in smoke|spinquant|vq|evaluation) ;; *) exit 64;; esac
test -n "${PBS_JOBID:-}" && test -s "${PBS_NODEFILE:-}" || exit 64
case "$(hostname -s)" in *login*) exit 64;; esac
grep -Fqx "$(hostname -s)" <(awk -F. '{print $1}' "$PBS_NODEFILE") || exit 64
COSMOS_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015
ARTIFACTS=$COSMOS_ROOT/artifacts/$PBS_JOBID
COSMOS_JOB_START_EPOCH=$(date +%s)
export COSMOS_ROOT ARTIFACTS COSMOS_JOB_START_EPOCH
mkdir -p "$ARTIFACTS" "$COSMOS_ROOT/results/$STAGE"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
GPU_MONITOR_PID=
finish() {
  rc=$?
  trap - EXIT TERM INT
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  printf '%s\n' "$rc" > "$ARTIFACTS/exit_code.txt"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT
printf 'PBS_JOBID=%s STAGE=%s HOST=%s ALLOCATION_START=%s\n' "$PBS_JOBID" "$STAGE" "$(hostname -f)" "$COSMOS_JOB_START_EPOCH"
qstat -f "$PBS_JOBID" > "$ARTIFACTS/allocation.txt"
grep -Eq '^[[:space:]]*job_state[[:space:]]*=[[:space:]]*R([[:space:]]|$)' "$ARTIFACTS/allocation.txt" || exit 64
test -s "$COSMOS_ROOT/environment_ready.json"
test -n "${CUDA_VISIBLE_DEVICES:-}" || { echo 'Allocated GPU identifiers are unavailable'; exit 64; }
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
EXPECTED_GPUS=1
if [[ "$STAGE" == evaluation ]]; then
  EXPECTED_GPUS=4
  grep -Eq '^[[:space:]]*queue[[:space:]]*=[[:space:]]*g3([[:space:]]|$)' "$ARTIFACTS/allocation.txt" || {
    echo 'Final evaluation requires the g3 execution queue'; exit 64;
  }
fi
[[ ${#GPU_IDS[@]} -eq $EXPECTED_GPUS ]] || exit 64
GPU_UUIDS=()
for gpu in "${GPU_IDS[@]}"; do
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
  [[ -n "$uuid" ]] || exit 64
  for previous in "${GPU_UUIDS[@]:-}"; do [[ "$previous" != "$uuid" ]] || exit 64; done
  GPU_UUIDS+=("$uuid")
done
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_UUIDS[*]}")
export CUDA_VISIBLE_DEVICES
(
  while :; do
    for uuid in "${GPU_UUIDS[@]}"; do
      printf 'GPU_SAMPLE timestamp=%s ' "$(date -Is)"
      nvidia-smi -i "$uuid" --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || echo GPU_SAMPLE_ERROR
    done
    sleep 15
  done
) &
GPU_MONITOR_PID=$!
for file in "$COSMOS_ROOT"/*.py "$COSMOS_ROOT"/*.sh "$COSMOS_ROOT"/*.pbs "$COSMOS_ROOT/protocol.json"; do
  cp "$file" "$ARTIFACTS/"
done
mkdir -p "$ARTIFACTS/helpers"
cp /scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008/rotation.py "$ARTIFACTS/helpers/"
cp /scratch/users/ntu/yguo017/fastwam-vq-micro-latent-search-20261014/micro_codebook.py "$ARTIFACTS/helpers/"
cp /scratch/users/ntu/yguo017/fastwam-vq-micro-latent-search-20261014/train_banks.py "$ARTIFACTS/helpers/"
cp /scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009/artifacts/25743693.pbs101/codebook.py "$ARTIFACTS/helpers/"
cp /scratch/users/ntu/yguo017/fastwam-smooth-vq-phases12-20261006/smooth_vq.py "$ARTIFACTS/helpers/"
cp /scratch/users/ntu/yguo017/fastwam-vq-c8-action-latent-plus-20261015/batched_transform.py "$ARTIFACTS/helpers/"
source "$ARTIFACTS/gpu_environment.sh"
run_python - "$ARTIFACTS" <<'PY'
import pathlib, py_compile, sys
for path in pathlib.Path(sys.argv[1]).glob('*.py'):
    py_compile.compile(str(path), doraise=True)
print('PYTHON_SYNTAX_CHECK_PASSED', flush=True)
PY
run_python -m unittest discover -s "$ARTIFACTS" -p test_evaluation.py
if [[ "$STAGE" == smoke ]]; then
  run_python "$ARTIFACTS/prepare_text_embeddings.py"
  run_python "$ARTIFACTS/smoke.py" --protocol "$ARTIFACTS/protocol.json" --root "$COSMOS_ROOT" --out "$COSMOS_ROOT/results/smoke"
elif [[ "$STAGE" == spinquant || "$STAGE" == vq ]]; then
  run_python "$ARTIFACTS/calibrate.py" --arm "$STAGE" --protocol "$ARTIFACTS/protocol.json" --root "$COSMOS_ROOT" --out "$COSMOS_ROOT/results/$STAGE" --soft-deadline-seconds 6600
else
  export COSMOS_SOFT_DEADLINE_SECONDS=85500
  MANIFEST_SOURCE=/scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008/results/manifest.json
  if [[ ! -e "$COSMOS_ROOT/results/manifest.json" ]]; then
    cp "$MANIFEST_SOURCE" "$COSMOS_ROOT/results/manifest.json"
  fi
  cmp -s "$MANIFEST_SOURCE" "$COSMOS_ROOT/results/manifest.json" || {
    echo 'Saved evaluation cohort differs from the frozen 280-variant source'; exit 65;
  }
  run_python "$ARTIFACTS/evaluate.py" --protocol "$ARTIFACTS/protocol.json" --manifest "$COSMOS_ROOT/results/manifest.json" --root "$COSMOS_ROOT" --out "$COSMOS_ROOT/results/evaluation" --worker-count 4 --soft-deadline-seconds 85500
fi
run_python - "$COSMOS_ROOT/results/$STAGE/summary.json" "$ARTIFACTS" <<'PY'
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
status = summary.get('status')
if status == 'complete':
    (out/'PIPELINE_COMPLETE').touch()
elif status == 'checkpointed':
    (out/'PIPELINE_CHECKPOINTED').touch()
else:
    raise SystemExit('Missing complete/checkpointed summary: '+str(status))
print('STAGE_STATUS='+status, flush=True)
PY

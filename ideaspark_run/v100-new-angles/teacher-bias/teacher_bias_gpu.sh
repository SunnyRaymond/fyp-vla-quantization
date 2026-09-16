#!/usr/bin/env bash
# Bounded Wall teacher-bias feature screen.  Heavy work is allocation-only.
# The input manifest is prepared separately and is never rewritten here.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=teacher_bias_v100
#SBATCH --output=artifacts/teacher_bias_%j.log

set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
OLD="${HOME}/cem_update_ccds"
ROOT="${OLD}/modelroot"
CONTROL="${TOP}/control"
PY="${OLD}/venv/bin/python"
MANIFEST="${TEACHER_BIAS_MANIFEST:-${TOP}/teacher_bias_ready2/manifest.json}"

# allocation_guard is intentionally the first operational check.  It rejects
# login-node execution, a missing PBS/SLURM allocation, or a non-V100 GPU.
test -f "${CONTROL}/allocation_guard.sh"
source "${CONTROL}/allocation_guard.sh"
test -x "${PY}"
test -f "${CONTROL}/teacher_bias_input_freeze.json"
MANIFEST_SHA="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "${CONTROL}/teacher_bias_input_freeze.json")"
test -d "${ROOT}"
test -f "${CONTROL}/teacher_bias_screen.py"
test -f "${CONTROL}/allocation_guard.py"
test -f "${CONTROL}/teacher_bias_protocol.zh.md"
test -f "${OLD}/control/smoke_runner.py"
test -f "${OLD}/control/screen_runner.py"
test -f "${MANIFEST}"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${CONTROL}/teacher_bias_screen.py" "${OUT}/teacher_bias_screen.py"
cp "${OLD}/control/smoke_runner.py" "${OLD}/control/screen_runner.py" "${OUT}/"
cp "${CONTROL}/allocation_guard.py" "${OUT}/"
cp "${CONTROL}/teacher_bias_protocol.zh.md" "${OUT}/teacher_bias_protocol.zh.md"
cp "${CONTROL}/teacher_bias_gpu.sh" "${OUT}/"
cp "${CONTROL}/teacher_bias_input_freeze.json" "${OUT}/"

export PYTHONPATH="${OUT}:${OLD}/control:${ROOT}/source"
export TORCH_HOME="${OLD}/cache/torch"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

"${PY}" "${OUT}/allocation_guard.py" > "${OUT}/allocation.json"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader > "${OUT}/gpu_identity.txt"
timeout --signal=TERM --kill-after=30s 4m "${PY}" "${OUT}/teacher_bias_screen.py" \
  --root "${ROOT}" \
  --output "${OUT}" \
  --input-manifest "${MANIFEST}" \
  --input-sha256 "${MANIFEST_SHA}" \
  --helper-dir "${OUT}" \
  --max-seconds 240 2>&1 | tee "${OUT}/run.log"

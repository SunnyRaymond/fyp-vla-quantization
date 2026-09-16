#!/usr/bin/env bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=action_gradient_geometry_v100
#SBATCH --output=artifacts/action_gradient_geometry_%j.log

set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
OLD="${HOME}/cem_update_ccds"
ROOT="${OLD}/modelroot"
CONTROL="${TOP}/control"
PY="${OLD}/venv/bin/python"

# The controller uploads the guard and runner by basename.  This is a cheap
# shell precheck; gradient_screen.py repeats the guard as its first workload
# action before importing the model runtime or touching data.
test -f "${CONTROL}/allocation_guard.sh"
source "${CONTROL}/allocation_guard.sh"
test -x "${PY}"
test -d "${ROOT}"
test -f "${CONTROL}/gradient_screen.py"
test -f "${CONTROL}/allocation_guard.py"
test -f "${CONTROL}/PROTOCOL.zh.md"
test -f "${CONTROL}/IMPLEMENTATION.zh.md"
test -f "${OLD}/control/smoke_runner.py"
test -f "${OLD}/control/screen_runner.py"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${CONTROL}/gradient_screen.py" "${OUT}/gradient_screen.py"
cp "${OLD}/control/smoke_runner.py" "${OLD}/control/screen_runner.py" "${OUT}/"
cp "${CONTROL}/allocation_guard.py" "${OUT}/"
cp "${CONTROL}/PROTOCOL.zh.md" "${OUT}/PROTOCOL.zh.md"
cp "${CONTROL}/IMPLEMENTATION.zh.md" "${OUT}/IMPLEMENTATION.zh.md"

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
timeout --signal=TERM --kill-after=30s 18m "${PY}" "${OUT}/gradient_screen.py" \
  --root "${ROOT}" \
  --output "${OUT}" \
  --helper-dir "${OUT}" \
  --max-seconds 1080 2>&1 | tee "${OUT}/run.log"

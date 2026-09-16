#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=antithetic_rounding_v100
#SBATCH --output=artifacts/antithetic_rounding_%j.log
set -euo pipefail

# This campaign is intentionally separate from the old CEM-Update artifacts.
TOP="${HOME}/v100_newangles_ccds"
OLD="${HOME}/cem_update_ccds"
ROOT="${OLD}/modelroot"
CONTROL="${TOP}/control"
PY="${OLD}/venv/bin/python"

# The shell guard is a cheap fail-closed precheck.  The Python runner repeats
# allocation_guard.require_allocation() as its first workload action.
source "${CONTROL}/allocation_guard.sh"
test -x "${PY}"
test -d "${ROOT}"
test -f "${CONTROL}/ensemble_screen.py"
test -f "${CONTROL}/allocation_guard.py"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${CONTROL}/ensemble_screen.py" "${OUT}/ensemble_screen.py"
cp "${OLD}/control/smoke_runner.py" "${OLD}/control/screen_runner.py" "${OUT}/"
cp "${CONTROL}/allocation_guard.py" "${OUT}/"
cp "${CONTROL}/IDEA_AND_PROTOCOL.zh.md" "${OUT}/IDEA_AND_PROTOCOL.zh.md"

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
timeout --signal=TERM --kill-after=30s 40m "${PY}" "${OUT}/ensemble_screen.py" \
  --root "${ROOT}" \
  --output "${OUT}" \
  --max-seconds 2400 2>&1 | tee "${OUT}/run.log"

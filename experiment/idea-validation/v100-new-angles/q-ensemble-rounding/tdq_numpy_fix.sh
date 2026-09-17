#!/bin/bash
# Isolated NumPy compatibility overlay, then fresh reset-only preparation.
# Fixes dm-control 1.0.16 versus NumPy 2.x API incompatibility without modifying
# the original venv or completed extra2 overlay; no model load or GPU request.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=tdq_numpy_fix
#SBATCH --output=artifacts/tdq_numpy_fix_%j.log
set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
VENDOR="${TOP}/tdmpc2_q_coupling_runtime"
EXTRA2="${TOP}/tdmpc2_q_coupling_runtime_extra2"
EXTRA3="${TOP}/tdmpc2_q_coupling_runtime_extra3"
READY="${TOP}/tdmpc2_q_coupling_ready5"

# Must be the first scheduler-dependent action.
source "${TOP}/control/allocation_guard.sh"

if [[ -e "${OUT}" || -e "${EXTRA3}" || -e "${READY}" ]]; then
  echo "refusing to overwrite output, extra3, or ready5: ${OUT} ${EXTRA3} ${READY}" >&2
  exit 3
fi
if [[ ! -d "${VENDOR}" || ! -d "${EXTRA2}" ]]; then
  echo "required prior vendor/extra2 is missing: ${VENDOR} ${EXTRA2}" >&2
  exit 3
fi
mkdir -p "${OUT}"
cp "${CONTROL}/tdq_numpy_fix.py" "${CONTROL}/tdq_numpy_fix.sh" "${CONTROL}/tdq_prepare.py" "${CONTROL}/tdq_prepare.sh" "${CONTROL}/PREPARATION.zh.md" "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing CPU runtime is unavailable: ${PY}" >&2
  exit 3
fi
export PYTHONPATH="${EXTRA2}:${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"${PY}" "${OUT}/tdq_numpy_fix.py" --python "${PY}" --extra "${EXTRA3}" --output "${OUT}"

export PYTHONPATH="${EXTRA3}:${EXTRA2}:${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
"${PY}" "${OUT}/tdq_prepare.py" --root "${TOP}" --assetdir "${READY}" --output "${OUT}"

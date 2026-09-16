#!/bin/bash
# One bounded CPU dependency-overlay repair followed by the fresh preparation.
# Never modifies the SmolVLA/DINO venv or the failed tdmpc2_q_coupling directory.
# pip dry-run/report must exclude Torch/NVIDIA/Triton and stay below 500 MiB.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=tdq_runtime_extra
#SBATCH --output=artifacts/tdq_runtime_extra_%j.log
set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
VENDOR="${TOP}/tdmpc2_q_coupling_runtime"
EXTRA="${TOP}/tdmpc2_q_coupling_runtime_extra"
READY="${TOP}/tdmpc2_q_coupling_ready"

# Must be the first scheduler-dependent action.
source "${TOP}/control/allocation_guard.sh"

if [[ -e "${OUT}" || -e "${EXTRA}" || -e "${READY}" ]]; then
  echo "refusing to overwrite existing repair output, extra overlay, or ready asset: ${OUT} ${EXTRA} ${READY}" >&2
  exit 3
fi
if [[ ! -d "${VENDOR}" ]]; then
  echo "previous isolated vendor is missing: ${VENDOR}" >&2
  exit 3
fi
mkdir -p "${OUT}"
cp "${CONTROL}/tdq_runtime_extra.py" "${CONTROL}/tdq_runtime_extra.sh" "${CONTROL}/tdq_prepare.py" "${CONTROL}/tdq_prepare.sh" "${CONTROL}/PREPARATION.zh.md" "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing CPU runtime is unavailable: ${PY}" >&2
  exit 3
fi
export PYTHONPATH="${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"${PY}" "${OUT}/tdq_runtime_extra.py" \
  --python "${PY}" --vendor "${VENDOR}" --extra "${EXTRA}" \
  --output "${OUT}" --control "${CONTROL}" --root "${TOP}"

export PYTHONPATH="${EXTRA}:${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
"${PY}" "${OUT}/tdq_prepare.py" \
  --root "${TOP}" --assetdir "${READY}" --output "${OUT}"

#!/bin/bash
# CPU-only conditional-action-distribution preparation. Submit through ccds_campaign_control.py.
# No download, install, GPU request, model load, or inference is performed.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=conditional_marginal_prepare
#SBATCH --output=artifacts/conditional_marginal_prepare_%j.log
set -euo pipefail

TOP="/tc1home/UG/yguo017/v100_newangles_ccds"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
ASSETDIR="${TOP}/conditional_marginal"

# This must be the first scheduler-dependent action.
source "${TOP}/control/allocation_guard.sh"

mkdir -p "${OUT}"
cp "${CONTROL}/conditional_prepare.py" "${OUT}/conditional_prepare.py"
cp "${CONTROL}/conditional_prepare.sh" "${OUT}/conditional_prepare.sh"
cp "${CONTROL}/PREPARATION.zh.md" "${OUT}/PREPARATION.zh.md"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing SmolVLA venv is unavailable: ${PY}" >&2
  exit 3
fi
export PYTHONPATH="${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

exec "${PY}" "${OUT}/conditional_prepare.py" \
  --root "${TOP}" --assetdir "${ASSETDIR}" \
  --padded-manifest "${TOP}/padded_feedback/manifest.json" --output "${OUT}"

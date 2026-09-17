#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=conditional_prepare_ready
#SBATCH --output=artifacts/conditional_prepare_ready_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${TOP}/control/conditional_prepare.py" "${TOP}/control/conditional_download.py" "${TOP}/control/conditional_prepare_ready.sh" "${TOP}/control/allocation_guard.py" "${OUT}/"
export PYTHONPATH="${OUT}"
PY="${TOP}/smolvla/venv/bin/python"
exec > >(tee -a "${OUT}/run.log") 2>&1
"${PY}" "${OUT}/conditional_download.py"
"${PY}" "${OUT}/conditional_prepare.py" --root "${TOP}" \
  --assetdir "${TOP}/conditional_marginal_ready" --identity-file "${TOP}/conditional_marginal_ready/identity.json" \
  --padded-manifest "${TOP}/padded_feedback/manifest.json" --output "${OUT}"

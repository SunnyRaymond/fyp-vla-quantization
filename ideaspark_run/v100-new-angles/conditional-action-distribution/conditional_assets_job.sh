#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --job-name=conditional_asset_probe
#SBATCH --output=artifacts/conditional_asset_probe_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${TOP}/control/conditional_assets.py" "${TOP}/control/conditional_assets_job.sh" "${TOP}/control/allocation_guard.py" "${OUT}/"
export PYTHONPATH="${OUT}"
"${TOP}/smolvla/venv/bin/python" "${OUT}/conditional_assets.py" 2>&1 | tee "${OUT}/run.log"

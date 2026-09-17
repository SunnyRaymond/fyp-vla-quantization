#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:04:00
#SBATCH --job-name=smolvla_metadata_probe
#SBATCH --output=artifacts/smolvla_metadata_probe_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${TOP}/control/smolvla_metadata_probe.py" "${TOP}/control/allocation_guard.py" "${OUT}/"
export PYTHONPATH="${OUT}"
"${TOP}/smolvla/venv/bin/python" "${OUT}/smolvla_metadata_probe.py" 2>&1 | tee "${OUT}/run.log"

#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=verify_gradient_64767
#SBATCH --output=artifacts/verify_gradient_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${TOP}/control/verify_gradient.py" "${TOP}/control/allocation_guard.py" "${OUT}/"
export PYTHONPATH="${OUT}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"${HOME}/cem_update_ccds/venv/bin/python" "${OUT}/verify_gradient.py" \
  --input "${TOP}/artifacts/64767" --output "${OUT}" 2>&1 | tee "${OUT}/run.log"

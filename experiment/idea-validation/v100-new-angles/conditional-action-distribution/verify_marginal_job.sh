#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=verify_marginal_64778
#SBATCH --output=artifacts/verify_marginal_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${TOP}/control/verify_marginal.py" "${TOP}/control/verify_marginal_job.sh" "${TOP}/control/allocation_guard.py" "${OUT}/"
export PYTHONPATH="${OUT}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"${TOP}/smolvla/venv/bin/python" "${OUT}/verify_marginal.py" \
  --input "${TOP}/artifacts/64778" --output "${OUT}" 2>&1 | tee "${OUT}/run.log"

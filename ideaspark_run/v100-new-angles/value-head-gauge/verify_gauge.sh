#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=verify_gauge
#SBATCH --output=artifacts/verify_gauge_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
INPUT="${TOP}/artifacts/64807"
test ! -e "${OUT}"
mkdir -p "${OUT}"
cp "${TOP}/control/verify_gauge.py" "${TOP}/control/verify_gauge.sh" "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1
export PYTHONPATH="${TOP}/tdmpc2_q_coupling_runtime_extra3:${TOP}/tdmpc2_q_coupling_runtime_extra2:${TOP}/tdmpc2_q_coupling_runtime:${TOP}/control:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
"${TOP}/smolvla/venv/bin/python" "${OUT}/verify_gauge.py" --input "${INPUT}" --output "${OUT}"


#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --job-name=persistence_manifest
#SBATCH --output=artifacts/persistence_manifest_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
test ! -e "${OUT}"
mkdir -p "${OUT}"
cp "${TOP}/control/compact_persistence_manifest.py" "${TOP}/control/compact_persistence_manifest.sh" "${OUT}/"
export PYTHONPATH="${TOP}/control:${PYTHONPATH:-}"
"${TOP}/smolvla/venv/bin/python" "${OUT}/compact_persistence_manifest.py" > "${OUT}/run.log" 2>&1

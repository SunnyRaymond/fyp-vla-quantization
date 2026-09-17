#!/usr/bin/env bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --job-name=teacher_source_probe
#SBATCH --output=artifacts/teacher_source_probe_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
source "${TOP}/control/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
exec > "${OUT}/probe.log" 2>&1
SOURCE="${HOME}/cem_update_ccds/modelroot/source"
command -v git || true
git --version || true
git -C "${SOURCE}" rev-parse HEAD || true
git -C "${SOURCE}" diff --name-only HEAD || true
git -C "${SOURCE}" show HEAD:models/dino.py || true
stat -c '%n %F %U %G' "${SOURCE}" "${SOURCE}/.git" || true

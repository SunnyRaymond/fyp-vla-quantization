#!/bin/bash
# One isolated CPU dependency repair; submit through ccds_campaign_control.py.
# Installs only dm-control and mujoco with --no-deps into a new vendor directory.
# It never changes the SmolVLA/DINO venv, installs Torch, loads a model, or creates an env.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=tdq_runtime_repair
#SBATCH --output=artifacts/tdq_runtime_repair_%j.log
set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
VENDOR="${TOP}/tdmpc2_q_coupling_runtime"

# Must be the first scheduler-dependent action.
source "${TOP}/control/allocation_guard.sh"

if [[ -e "${OUT}" || -e "${VENDOR}" ]]; then
  echo "refusing to overwrite existing repair output or vendor: ${OUT} ${VENDOR}" >&2
  exit 3
fi
mkdir -p "${OUT}"
cp "${CONTROL}/tdq_runtime_repair.py" "${CONTROL}/tdq_runtime_repair.sh" "${CONTROL}/PREPARATION.zh.md" "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing CPU runtime is unavailable: ${PY}" >&2
  exit 3
fi
mkdir "${VENDOR}"
"${PY}" -m pip install --disable-pip-version-check --no-cache-dir --no-deps --target "${VENDOR}" \
  "dm-control==1.0.16" "mujoco==3.1.2"

export PYTHONPATH="${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec "${PY}" "${OUT}/tdq_runtime_repair.py" --vendor "${VENDOR}" --output "${OUT}"

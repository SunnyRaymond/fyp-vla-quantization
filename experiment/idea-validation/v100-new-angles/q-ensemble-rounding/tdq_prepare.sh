#!/bin/bash
# Bounded CPU-only TD-MPC2 source/checkpoint/reset preparation.
# Submit through ../ccds_campaign_control.py; never run this on a login node.
# No model load, env.step, env.render, full rollout, or GPU request.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=tdq_prepare
#SBATCH --output=artifacts/tdq_prepare_%j.log
set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"

# Must be the first scheduler-dependent action.
source "${TOP}/control/allocation_guard.sh"

if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing job artifact directory: ${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}"
cp "${CONTROL}/tdq_prepare.py" "${CONTROL}/tdq_prepare.sh" "${CONTROL}/PREPARATION.zh.md" "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing CPU runtime is unavailable: ${PY}" >&2
  exit 3
fi
export PYTHONPATH="${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

exec "${PY}" "${OUT}/tdq_prepare.py" \
  --root "${TOP}" \
  --assetdir "${TOP}/tdmpc2_q_coupling" \
  --output "${OUT}"

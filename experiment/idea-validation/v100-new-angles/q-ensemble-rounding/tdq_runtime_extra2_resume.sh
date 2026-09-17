#!/bin/bash
# Resume only the completed extra2 overlay: verify imports, then run fresh prep.
# No resolver, reinstall, model load, env.step, env.render, or GPU request.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=tdq_extra2_resume
#SBATCH --output=artifacts/tdq_extra2_resume_%j.log
set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
VENDOR="${TOP}/tdmpc2_q_coupling_runtime"
EXTRA="${TOP}/tdmpc2_q_coupling_runtime_extra2"
READY="${TOP}/tdmpc2_q_coupling_ready4"

# Must be the first scheduler-dependent action.
source "${TOP}/control/allocation_guard.sh"

if [[ -e "${OUT}" || ! -d "${VENDOR}" || ! -d "${EXTRA}" || -e "${READY}" ]]; then
  echo "refusing overwrite or missing prerequisite: ${OUT} ${VENDOR} ${EXTRA} ${READY}" >&2
  exit 3
fi
mkdir -p "${OUT}"
cp "${CONTROL}/tdq_runtime_extra2_resume.py" "${CONTROL}/tdq_runtime_extra2_resume.sh" "${CONTROL}/tdq_prepare.py" "${CONTROL}/tdq_prepare.sh" "${CONTROL}/PREPARATION.zh.md" "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing CPU runtime is unavailable: ${PY}" >&2
  exit 3
fi
export PYTHONPATH="${EXTRA}:${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"${PY}" "${OUT}/tdq_runtime_extra2_resume.py" \
  --extra "${EXTRA}" --vendor "${VENDOR}" --control "${CONTROL}" --output "${OUT}"
"${PY}" "${OUT}/tdq_prepare.py" \
  --root "${TOP}" --assetdir "${READY}" --output "${OUT}"

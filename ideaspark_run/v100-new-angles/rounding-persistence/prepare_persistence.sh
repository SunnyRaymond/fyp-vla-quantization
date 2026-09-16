#!/bin/bash
# CPU-only preparation for the denoising-call rounding-persistence screen.
# It never downloads, loads a model, runs inference, or performs an env rollout.
# Submit only to a real CCDS SLURM allocation.

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=rounding_persistence_prepare
#SBATCH --output=artifacts/rounding_persistence_prepare_%j.log

set -euo pipefail

TOP="${ROUNDING_PERSISTENCE_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"

# The real-allocation guard is the first scheduler-dependent action.
source "${CONTROL}/allocation_guard.sh"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
ASSET="${ROUNDING_PERSISTENCE_ASSET:-${TOP}/rounding_persistence_ready}"
BASE_IDENTITY="${ROUNDING_PERSISTENCE_BASE_IDENTITY:-${TOP}/smolvla/identity.json}"
EXTENSION_IDENTITY="${ROUNDING_PERSISTENCE_EXTENSION_IDENTITY:-${TOP}/conditional_marginal_ready/identity.json}"
HELPER="${ROUNDING_PERSISTENCE_HELPER:-${TOP}/artifacts/64758/smolvla_cpu_prepare.py}"
PY="${ROUNDING_PERSISTENCE_PYTHON:-${TOP}/smolvla/venv/bin/python}"

if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing job output: ${OUT}" >&2
  exit 3
fi
if [[ ! -x "${PY}" ]]; then
  echo "pinned CPU Python executable is unavailable: ${PY}" >&2
  exit 3
fi
for required in \
  "${CONTROL}/prepare_persistence.py" \
  "${CONTROL}/prepare_persistence.sh" \
  "${CONTROL}/allocation_guard.py" \
  "${CONTROL}/allocation_guard.sh" \
  "${BASE_IDENTITY}" \
  "${EXTENSION_IDENTITY}" \
  "${HELPER}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required pinned input is missing: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${OUT}"
cp "${CONTROL}/prepare_persistence.py" \
   "${CONTROL}/prepare_persistence.sh" \
   "${CONTROL}/allocation_guard.py" \
   "${CONTROL}/allocation_guard.sh" \
   "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

export PYTHONUNBUFFERED=1
export PYTHONPATH="${OUT}:${CONTROL}:${PYTHONPATH:-}"

set +e
timeout --signal=TERM --kill-after=10s 285s \
  "${PY}" "${OUT}/prepare_persistence.py" \
  --root "${TOP}" \
  --base-identity "${BASE_IDENTITY}" \
  --extension-identity "${EXTENSION_IDENTITY}" \
  --helper "${HELPER}" \
  --assetdir "${ASSET}" \
  --output "${OUT}"
status=$?
set -e
exit "${status}"

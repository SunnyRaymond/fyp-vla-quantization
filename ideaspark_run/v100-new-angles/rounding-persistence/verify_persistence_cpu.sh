#!/usr/bin/env bash
# Independent CPU raw replay for the matched temporal persistence screen.
# Set ROUNDING_PERSISTENCE_INPUT to the GPU artifact directory. The default is
# the frozen first GPU screen job; the verifier also handles a timed-out,
# summary-less artifact without interpreting partial arrays as science.

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=verify_rounding_persistence
#SBATCH --output=artifacts/verify_rounding_persistence_%j.log

set -euo pipefail

TOP="${ROUNDING_PERSISTENCE_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"

# The copied campaign guard is the first scheduler-dependent operation.
source "${CONTROL}/allocation_guard.sh"

INPUT="${ROUNDING_PERSISTENCE_INPUT:-${TOP}/artifacts/64815}"
PYTHON="${ROUNDING_PERSISTENCE_PYTHON:-${TOP}/smolvla/venv/bin/python}"
OUT="${ROUNDING_PERSISTENCE_OUTPUT:-${TOP}/artifacts/${SLURM_JOB_ID}}"

for required in \
  "${CONTROL}/verify_persistence.py" \
  "${CONTROL}/verify_persistence_cpu.sh" \
  "${CONTROL}/allocation_guard.py" \
  "${INPUT}/engineering.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "required verifier input is missing: ${required}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "approved CPU verifier Python is unavailable: ${PYTHON}" >&2
  exit 3
fi
if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing verifier output: ${OUT}" >&2
  exit 3
fi

mkdir -p "${OUT}"
cp "${CONTROL}/verify_persistence.py" "${CONTROL}/verify_persistence_cpu.sh" "${CONTROL}/allocation_guard.py" "${OUT}/"

exec > >(tee -a "${OUT}/run.log") 2>&1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="${OUT}:${CONTROL}${PYTHONPATH:+:${PYTHONPATH}}"

timeout --signal=TERM --kill-after=15s 5m \
  "${PYTHON}" "${OUT}/verify_persistence.py" \
  --input "${INPUT}" \
  --output "${OUT}" \
  --max-seconds 300

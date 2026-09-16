#!/usr/bin/env bash
# Independent CPU receipt/raw replay for the B2 persistence repair.
# Set ROUNDING_PERSISTENCE_INPUT to one completed or partial B2 GPU artifact.
# A partial artifact is classified without loading its science arrays.

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=verify_rounding_persistence_b2
#SBATCH --output=artifacts/verify_rounding_persistence_b2_%j.log

set -euo pipefail

TOP="${ROUNDING_PERSISTENCE_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"

# The copied campaign guard is the first scheduler-dependent operation.
source "${CONTROL}/allocation_guard.sh"

INPUT="${ROUNDING_PERSISTENCE_INPUT:-${TOP}/artifacts/64825}"
PYTHON="${ROUNDING_PERSISTENCE_PYTHON:-${TOP}/smolvla/venv/bin/python}"
OUT="${ROUNDING_PERSISTENCE_OUTPUT:-${TOP}/artifacts/${SLURM_JOB_ID}}"
if [[ -z "${INPUT}" ]]; then
  echo "ROUNDING_PERSISTENCE_INPUT must name a B2 GPU artifact" >&2
  exit 2
fi

for required in \
  "${CONTROL}/verify_persistence_batch2.py" \
  "${CONTROL}/verify_persistence_batch2_cpu.sh" \
  "${CONTROL}/allocation_guard.py" \
  "${INPUT}/engineering.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "required B2 verifier input is missing: ${required}" >&2
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
cp "${CONTROL}/verify_persistence_batch2.py" "${OUT}/verify_persistence_batch2.py"
cp "${CONTROL}/verify_persistence_batch2_cpu.sh" "${OUT}/verify_persistence_batch2_cpu.sh"
cp "${CONTROL}/allocation_guard.py" "${OUT}/allocation_guard.py"

exec > >(tee -a "${OUT}/run.log") 2>&1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="${OUT}:${CONTROL}${PYTHONPATH:+:${PYTHONPATH}}"

timeout --signal=TERM --kill-after=15s 5m \
  "${PYTHON}" "${OUT}/verify_persistence_batch2.py" \
  --input "${INPUT}" \
  --output "${OUT}" \
  --max-seconds 300

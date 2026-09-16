#!/usr/bin/env bash
# Independent CPU-only replay of teacher-bias raw features.
# Submit only in a real CCDS SLURM allocation; it never loads the model.
# TEACHER_BIAS_INPUT must name one completed or failed GPU artifact directory.

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=teacher_bias_verify
#SBATCH --output=artifacts/teacher_bias_verify_%j.log

set -euo pipefail

TOP="${TEACHER_BIAS_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TEACHER_BIAS_CONTROL:-${TOP}/control}"
PY="${TEACHER_BIAS_PYTHON:-${HOME}/cem_update_ccds/venv/bin/python}"
INPUT="${TEACHER_BIAS_INPUT:-${TOP}/artifacts/64828}"
OUT="${TEACHER_BIAS_VERIFY_OUTPUT:-${TOP}/artifacts/${SLURM_JOB_ID:-unallocated}}"

# This is the first scheduler-dependent action.  It rejects login nodes,
# missing allocations, and NodeList/hostname mismatches.
test -f "${CONTROL}/allocation_guard.sh"
source "${CONTROL}/allocation_guard.sh"

if [[ -z "${INPUT}" ]]; then
  echo "TEACHER_BIAS_INPUT must point to an actual GPU job artifact" >&2
  exit 3
fi
if [[ ! -x "${PY}" ]]; then
  echo "pinned CPU Python executable is unavailable: ${PY}" >&2
  exit 3
fi
if [[ ! -f "${CONTROL}/verify_teacher_bias.py" || ! -f "${CONTROL}/allocation_guard.py" ]]; then
  echo "verifier or allocation helper is missing" >&2
  exit 3
fi
if [[ ! -d "${INPUT}" ]]; then
  echo "GPU artifact directory is missing: ${INPUT}" >&2
  exit 3
fi
if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite verifier output: ${OUT}" >&2
  exit 3
fi

mkdir -p "${OUT}"
cp "${CONTROL}/verify_teacher_bias.py" \
   "${CONTROL}/teacher_bias_cpu_verify.sh" \
   "${CONTROL}/allocation_guard.py" \
   "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="${OUT}:${CONTROL}:${PYTHONPATH:-}"

SHA_ARGS=()
if [[ -n "${TEACHER_BIAS_INPUT_SHA256:-}" ]]; then
  SHA_ARGS=(--input-sha256 "${TEACHER_BIAS_INPUT_SHA256}")
fi

timeout --signal=TERM --kill-after=15s 270s \
  "${PY}" "${OUT}/verify_teacher_bias.py" \
  --input "${INPUT}" \
  --output "${OUT}" \
  "${SHA_ARGS[@]}"

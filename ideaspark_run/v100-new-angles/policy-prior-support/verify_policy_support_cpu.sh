#!/bin/bash
# CPU-only verifier for one explicit producer job directory.
# Submit through the CCDS controller; never run on a login/head node.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=policy_support_verify
#SBATCH --output=artifacts/policy_support_verify_%j.log
set -euo pipefail

TOP="${POLICY_SUPPORT_TOP:-/tc1home/UG/yguo017/v100_newangles_ccds}"
CONTROL="${POLICY_SUPPORT_CONTROL:-${TOP}/control}"
PY="${POLICY_SUPPORT_PYTHON:-${TOP}/smolvla/venv/bin/python}"
EXTRA3="${POLICY_SUPPORT_EXTRA3:-${TOP}/tdmpc2_q_coupling_runtime_extra3}"
EXTRA2="${POLICY_SUPPORT_EXTRA2:-${TOP}/tdmpc2_q_coupling_runtime_extra2}"
VENDOR="${POLICY_SUPPORT_VENDOR:-${TOP}/tdmpc2_q_coupling_runtime}"

# The guard is the first scheduler-dependent action.
source "${CONTROL}/allocation_guard.sh"

INPUT="${POLICY_SUPPORT_INPUT:-${TOP}/artifacts/64833}"
OUTPUT="${POLICY_SUPPORT_OUTPUT:-${TOP}/artifacts/${SLURM_JOB_ID}}"
REPLAY="${POLICY_SUPPORT_REPLAY:-${CONTROL}/policy_support_replay.py}"
if [[ -z "$INPUT" || -z "$OUTPUT" || -z "$REPLAY" ]]; then
  echo "policy-support input, output, and replay paths must be non-empty" >&2
  exit 3
fi
if [[ -e "${OUTPUT}/verification.json" ]]; then
  echo "refusing to overwrite verifier report: ${OUTPUT}/verification.json" >&2
  exit 3
fi
if [[ "${OUTPUT}" == "${INPUT}" ]]; then
  echo "verifier output must be separate from producer input" >&2
  exit 3
fi
if [[ ! -x "$PY" ]]; then
  echo "existing CPU Python runtime is unavailable: $PY" >&2
  exit 3
fi
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT}"
cp "${CONTROL}/verify_policy_support.py" "${CONTROL}/verify_policy_support_cpu.sh" "${REPLAY}" "${CONTROL}/allocation_guard.py" "${CONTROL}/allocation_guard.sh" "${OUTPUT}/"
REPLAY="${OUTPUT}/policy_support_replay.py"
exec > >(tee -a "${OUTPUT}/run.log") 2>&1

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4
export PYTHONPATH="${EXTRA3}:${EXTRA2}:${VENDOR}:${CONTROL}:${PYTHONPATH:-}"

exec "$PY" "${OUTPUT}/verify_policy_support.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --replay "${REPLAY}"

#!/usr/bin/env bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=tdq_coupling_v100
#SBATCH --output=artifacts/tdq_coupling_%j.log

set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"

# This is intentionally the first scheduler-dependent action.  The guard
# checks the real hostname, job owner, partition, and allocated NodeList.
source "${TOP}/control/allocation_guard.sh"

ASSET_ROOT="${TDQ_ASSET_ROOT:-${TOP}/tdq_compatible_checkpoint}"
RUNTIME_NUMPY="${TDQ_RUNTIME_NUMPY:-${TOP}/tdmpc2_q_coupling_runtime_extra3}"
RUNTIME_EXTRA="${TDQ_RUNTIME_EXTRA:-${TOP}/tdmpc2_q_coupling_runtime_extra2}"
RUNTIME_VENDOR="${TDQ_RUNTIME_VENDOR:-${TOP}/tdmpc2_q_coupling_runtime}"
PY="${TDQ_PYTHON:-${TOP}/smolvla/venv/bin/python}"
MANIFEST="${TDQ_MANIFEST:-${ASSET_ROOT}/manifest.json}"
SOURCE="${TDQ_SOURCE_ROOT:-${TOP}/tdmpc2_q_coupling_ready5/source}"
CHECKPOINT="${TDQ_CHECKPOINT:-${ASSET_ROOT}/cartpole-balance-3.pt}"
CONFIG="${TDQ_CONFIG:-${SOURCE}/tdmpc2/config.yaml}"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"

if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing output: ${OUT}" >&2
  exit 3
fi
if [[ ! -x "${PY}" ]]; then
  echo "pinned Python executable is unavailable: ${PY}" >&2
  exit 3
fi
for required in \
  "${CONTROL}/tdq_screen.py" \
  "${CONTROL}/IMPLEMENTATION.zh.md" \
  "${CONTROL}/PROTOCOL.zh.md" \
  "${CONTROL}/tdq_prepare.py" \
  "${CONTROL}/verify_tdq.py" \
  "${CONTROL}/allocation_guard.sh" \
  "${CONTROL}/allocation_guard.py" \
  "${MANIFEST}" \
  "${CONFIG}" \
  "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required pinned input is missing: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1

cp "${CONTROL}/tdq_screen.py" \
  "${CONTROL}/tdq_screen.sh" \
  "${CONTROL}/IMPLEMENTATION.zh.md" \
  "${CONTROL}/PROTOCOL.zh.md" \
  "${CONTROL}/tdq_prepare.py" \
  "${CONTROL}/verify_tdq.py" \
  "${CONTROL}/allocation_guard.sh" \
  "${CONTROL}/allocation_guard.py" \
  "${OUT}/"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTHONPATH="${RUNTIME_NUMPY}:${RUNTIME_EXTRA}:${RUNTIME_VENDOR}:${SOURCE}/tdmpc2:${SOURCE}:${CONTROL}:${PYTHONPATH:-}"

set +e
timeout --signal=TERM --kill-after=30s 14m \
  "${PY}" "${OUT}/tdq_screen.py" \
  --manifest "${MANIFEST}" \
  --source-root "${SOURCE}" \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --output "${OUT}" \
  --max-seconds 720
status=$?
set -e
exit "${status}"

#!/usr/bin/env bash
# Policy-prior support: one V100, first CEM update observable only.
# Submit through the CCDS controller; this file is not a login-node launcher.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=policy_prior_support
#SBATCH --output=artifacts/policy_prior_support_%j.log

set -euo pipefail

TOP="${POLICY_SUPPORT_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"

# The guard is deliberately the first scheduler-dependent action.
source "${CONTROL}/allocation_guard.sh"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
SOURCE="${POLICY_SUPPORT_SOURCE_ROOT:-${TOP}/tdmpc2_q_coupling_ready5/source}"
ASSET="${POLICY_SUPPORT_ASSET_ROOT:-${TOP}/policy_prior_support_ready}"
CHECKPOINT="${POLICY_SUPPORT_CHECKPOINT:-${TOP}/tdq_compatible_checkpoint/cartpole-balance-3.pt}"
CONFIG="${POLICY_SUPPORT_CONFIG:-${SOURCE}/tdmpc2/config.yaml}"
MANIFEST="${POLICY_SUPPORT_MANIFEST:-${ASSET}/manifest.json}"
HELPER="${POLICY_SUPPORT_HELPER:-${CONTROL}/tdq_screen.py}"
RUNNER="${POLICY_SUPPORT_RUNNER:-${CONTROL}/policy_support_screen.py}"
PROTOCOL="${POLICY_SUPPORT_PROTOCOL:-${CONTROL}/policy_support_protocol.zh.md}"
INPUT_FREEZE="${POLICY_SUPPORT_INPUT_FREEZE:-${CONTROL}/policy_support_input_freeze.json}"
PY="${POLICY_SUPPORT_PYTHON:-${TOP}/smolvla/venv/bin/python}"
RUNTIME_NUMPY="${POLICY_SUPPORT_RUNTIME_NUMPY:-${TOP}/tdmpc2_q_coupling_runtime_extra3}"
RUNTIME_EXTRA="${POLICY_SUPPORT_RUNTIME_EXTRA:-${TOP}/tdmpc2_q_coupling_runtime_extra2}"
RUNTIME_VENDOR="${POLICY_SUPPORT_RUNTIME_VENDOR:-${TOP}/tdmpc2_q_coupling_runtime}"

if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing output: ${OUT}" >&2
  exit 3
fi
for required in "${PY}" "${RUNNER}" "${HELPER}" "${CONTROL}/allocation_guard.sh" \
  "${CONTROL}/allocation_guard.py" "${PROTOCOL}" "${INPUT_FREEZE}" \
  "${MANIFEST}" "${SOURCE}" "${CONFIG}" "${CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required pinned input is missing: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
cp "${CONTROL}/policy_support_gpu.sh" "${RUNNER}" "${HELPER}" "${CONTROL}/allocation_guard.sh" \
  "${CONTROL}/allocation_guard.py" "${PROTOCOL}" "${INPUT_FREEZE}" "${OUT}/"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTHONPATH="${OUT}:${RUNTIME_NUMPY}:${RUNTIME_EXTRA}:${RUNTIME_VENDOR}:${SOURCE}/tdmpc2:${SOURCE}:${CONTROL}:${PYTHONPATH:-}"

set +e
timeout --signal=TERM --kill-after=15s 540s "${PY}" "${OUT}/policy_support_screen.py" \
  --manifest "${MANIFEST}" --input-freeze "${OUT}/policy_support_input_freeze.json" \
  --source-root "${SOURCE}" --checkpoint "${CHECKPOINT}" --config "${CONFIG}" \
  --protocol "${OUT}/policy_support_protocol.zh.md" \
  --output "${OUT}" --max-seconds 480
status=$?
set -e
printf '{"exit_code":%s}\n' "${status}" > "${OUT}/exit_status.json"
exit "${status}"

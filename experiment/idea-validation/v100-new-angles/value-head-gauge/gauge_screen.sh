#!/usr/bin/env bash
# Bounded TD-MPC2 value-head gauge screen; submit only to a real CCDS V100
# allocation.  The screen is fake W4 quantization and does not claim kernels.
# The allocation guard must run before copying files or starting Python.

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=value_head_gauge_v100
#SBATCH --output=artifacts/value_head_gauge_%j.log

set -euo pipefail

TOP="${VALUE_GAUGE_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"

source "${TOP}/control/allocation_guard.sh"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
ASSET="${VALUE_GAUGE_ASSET:-${TOP}/value_head_gauge_ready}"
MANIFEST="${VALUE_GAUGE_MANIFEST:-${ASSET}/manifest.json}"
SOURCE="${VALUE_GAUGE_SOURCE:-${TOP}/tdmpc2_q_coupling_ready5/source}"
CHECKPOINT="${VALUE_GAUGE_CHECKPOINT:-${TOP}/tdq_compatible_checkpoint/cartpole-balance-3.pt}"
CONFIG="${VALUE_GAUGE_CONFIG:-${SOURCE}/tdmpc2/config.yaml}"
FREEZE="${VALUE_GAUGE_FREEZE:-${CONTROL}/FREEZE_AMENDMENT.zh.md}"
DRAFT="${VALUE_GAUGE_DRAFT:-${CONTROL}/PROTOCOL_DRAFT.zh.md}"
PY="${VALUE_GAUGE_PYTHON:-${TOP}/smolvla/venv/bin/python}"

if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing output: ${OUT}" >&2
  exit 3
fi
if [[ ! -x "${PY}" ]]; then
  echo "pinned Python executable is unavailable: ${PY}" >&2
  exit 3
fi
for required in \
  "${CONTROL}/gauge_screen.py" \
  "${CONTROL}/gauge_screen.sh" \
  "${CONTROL}/tdq_screen.py" \
  "${CONTROL}/allocation_guard.py" \
  "${CONTROL}/allocation_guard.sh" \
  "${FREEZE}" \
  "${DRAFT}" \
  "${MANIFEST}" \
  "${SOURCE}/tdmpc2/config.yaml" \
  "${CONFIG}" \
  "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required pinned input is missing: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1

# Import the helper copied into this job output, so the running screen is
# coupled to the recorded helper hash rather than a later control directory.
cp "${CONTROL}/gauge_screen.py" \
   "${CONTROL}/verify_gauge.py" \
   "${CONTROL}/gauge_screen.sh" \
   "${CONTROL}/tdq_screen.py" \
   "${CONTROL}/allocation_guard.py" \
   "${CONTROL}/allocation_guard.sh" \
   "${FREEZE}" \
   "${DRAFT}" \
   "${OUT}/"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export PYTHONPATH="${TOP}/tdmpc2_q_coupling_runtime_extra3:${TOP}/tdmpc2_q_coupling_runtime_extra2:${TOP}/tdmpc2_q_coupling_runtime:${OUT}:${SOURCE}/tdmpc2:${SOURCE}:${CONTROL}:${PYTHONPATH:-}"

set +e
timeout --signal=TERM --kill-after=20s 9m \
  "${PY}" "${OUT}/gauge_screen.py" \
  --manifest "${MANIFEST}" \
  --source-root "${SOURCE}" \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --freeze "${OUT}/FREEZE_AMENDMENT.zh.md" \
  --draft "${OUT}/PROTOCOL_DRAFT.zh.md" \
  --output "${OUT}" \
  --max-seconds 480
status=$?
set -e
exit "${status}"

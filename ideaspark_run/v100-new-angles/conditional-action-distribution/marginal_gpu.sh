#!/usr/bin/env bash
# Two-arm conditional-action marginal raw screen.  Submit through the campaign
# controller; this file does not install dependencies or download assets.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=conditional_marginal_v100
#SBATCH --output=artifacts/conditional_marginal_%j.log

set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
ASSET="${MARGINAL_MODEL_ASSET:-${TOP}/smolvla}"
ASSET_ROOT="${MARGINAL_ASSET_ROOT:-${TOP}}"
MANIFEST="${MARGINAL_MANIFEST:-${TOP}/conditional_marginal_ready/manifest.json}"
PY="${MARGINAL_PYTHON:-${ASSET}/venv/bin/python}"
MODEL="${MARGINAL_MODEL_PATH:-${ASSET}/model}"
CHECKPOINT="${MARGINAL_CHECKPOINT_PATH:-${ASSET}/model/model.safetensors}"
VLM="${MARGINAL_VLM_PATH:-${ASSET}/base_vlm_metadata}"
HELPER="${MARGINAL_FLOW_HELPER:-${CONTROL}/flow_screen.py}"

# This is the first scheduler-dependent operation.  Do not move file I/O,
# hashing, imports, or model work before the real allocation guard.
source "${CONTROL}/allocation_guard.sh"

test -x "${PY}"
test -f "${CONTROL}/marginal_screen.py"
test -f "${CONTROL}/marginal_gpu.sh"
test -f "${CONTROL}/allocation_guard.py"
test -f "${CONTROL}/IMPLEMENTATION.zh.md"
test -f "${CONTROL}/PROTOCOL.zh.md"
test -f "${HELPER}"
test -f "${MANIFEST}"
test -d "${MODEL}"
test -f "${CHECKPOINT}"
test -d "${VLM}"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${CONTROL}/marginal_screen.py" "${OUT}/marginal_screen.py"
cp "${CONTROL}/marginal_gpu.sh" "${OUT}/marginal_gpu.sh"
cp "${CONTROL}/allocation_guard.py" "${OUT}/allocation_guard.py"
cp "${CONTROL}/IMPLEMENTATION.zh.md" "${OUT}/IMPLEMENTATION.zh.md"
cp "${CONTROL}/PROTOCOL.zh.md" "${OUT}/PROTOCOL.zh.md"
cp "${HELPER}" "${OUT}/flow_screen.py"

export PYTHONPATH="${OUT}:${ASSET}/source:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

"${PY}" "${OUT}/allocation_guard.py" > "${OUT}/allocation.json"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader > "${OUT}/gpu_identity.txt"

EXTRA_ARGS=()
if [[ -n "${MARGINAL_LEROBOT_SOURCE:-}" ]]; then
    test -d "${MARGINAL_LEROBOT_SOURCE}"
    EXTRA_ARGS+=(--lerobot-source "${MARGINAL_LEROBOT_SOURCE}")
fi

set +e
timeout --signal=TERM --kill-after=30s 14m "${PY}" "${OUT}/marginal_screen.py" \
  --manifest "${MANIFEST}" --asset-root "${ASSET_ROOT}" \
  --model-path "${MODEL}" --checkpoint "${CHECKPOINT}" --vlm-path "${VLM}" \
  --helper-path "${OUT}/flow_screen.py" --output "${OUT}" --max-seconds 840 \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${OUT}/run.log"
status=${PIPESTATUS[0]}
set -e
exit "${status}"

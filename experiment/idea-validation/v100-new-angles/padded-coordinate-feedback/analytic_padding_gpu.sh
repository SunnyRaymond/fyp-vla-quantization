#!/usr/bin/env bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=analytic_padding_feedback_v100
#SBATCH --output=artifacts/analytic_padding_feedback_%j.log

set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
ASSET="${PADDED_ASSET_ROOT:-${TOP}/smolvla}"
MANIFEST="${PADDED_MANIFEST:-${TOP}/padded_feedback/manifest.json}"
CONTROL="${TOP}/control"
PY="${PADDED_PYTHON:-${ASSET}/venv/bin/python}"
MODEL="${PADDED_MODEL_PATH:-${ASSET}/model}"
CHECKPOINT="${PADDED_CHECKPOINT_PATH:-${ASSET}/model/model.safetensors}"
VLM="${PADDED_VLM_PATH:-${ASSET}/base_vlm_metadata}"

test -f "${CONTROL}/allocation_guard.sh"
source "${CONTROL}/allocation_guard.sh"
test -x "${PY}"
test -f "${CONTROL}/analytic_padding_screen.py"
test -f "${CONTROL}/analytic_padding_gpu.sh"
test -f "${CONTROL}/allocation_guard.py"
test -f "${CONTROL}/IMPLEMENTATION.zh.md"
test -f "${CONTROL}/PROTOCOL.zh.md"
test -f "${CONTROL}/INDEPENDENT_METHOD_AUDIT.zh.md"
test -f "${MANIFEST}"
test -d "${MODEL}"
test -f "${CHECKPOINT}"
test -d "${VLM}"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
cp "${CONTROL}/analytic_padding_screen.py" "${OUT}/analytic_padding_screen.py"
cp "${CONTROL}/analytic_padding_gpu.sh" "${OUT}/analytic_padding_gpu.sh"
cp "${CONTROL}/allocation_guard.py" "${OUT}/allocation_guard.py"
cp "${CONTROL}/IMPLEMENTATION.zh.md" "${OUT}/IMPLEMENTATION.zh.md"
cp "${CONTROL}/PROTOCOL.zh.md" "${OUT}/PROTOCOL.zh.md"
cp "${CONTROL}/INDEPENDENT_METHOD_AUDIT.zh.md" "${OUT}/INDEPENDENT_METHOD_AUDIT.zh.md"
FLOW_HELPER="${PADDED_FLOW_HELPER:-${CONTROL}/flow_screen.py}"
test -f "${FLOW_HELPER}"
cp "${FLOW_HELPER}" "${OUT}/flow_screen.py"

export PYTHONPATH="${OUT}:${ASSET}/source"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

"${PY}" "${OUT}/allocation_guard.py" > "${OUT}/allocation.json"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader > "${OUT}/gpu_identity.txt"
EXTRA_ARGS=()
if [[ -n "${PADDED_LEROBOT_SOURCE:-}" ]]; then
    test -d "${PADDED_LEROBOT_SOURCE}"
    EXTRA_ARGS+=(--lerobot-source "${PADDED_LEROBOT_SOURCE}")
fi
timeout --signal=TERM --kill-after=30s 12m "${PY}" "${OUT}/analytic_padding_screen.py" \
  --manifest "${MANIFEST}" --asset-root "${TOP}" --model-path "${MODEL}" \
  --checkpoint "${CHECKPOINT}" --vlm-path "${VLM}" --helper-dir "${OUT}" \
  --output "${OUT}" --max-seconds 720 "${EXTRA_ARGS[@]}" 2>&1 | tee "${OUT}/run.log"

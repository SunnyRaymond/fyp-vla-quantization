#!/usr/bin/env bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=flow_geometry_drift_v100
#SBATCH --output=artifacts/flow_geometry_drift_%j.log

set -euo pipefail

# This script is intentionally explicit about every large input.  The
# controller must provide paths prepared inside approved compute allocations;
# no login-node fallback, download, or guessed model layout is permitted.
TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
test -f "${CONTROL}/allocation_guard.sh"
source "${CONTROL}/allocation_guard.sh"

test -f "${CONTROL}/flow_screen.py"
test -f "${CONTROL}/IDEA_AND_PROTOCOL.zh.md"
test -f "${CONTROL}/INDEPENDENT_AUDIT.zh.md"

ASSET="${FLOW_ASSET_ROOT:-${TOP}/smolvla}"
FLOW_PYTHON="${FLOW_PYTHON:-${ASSET}/venv/bin/python}"
FLOW_INPUT_MANIFEST="${FLOW_INPUT_MANIFEST:-${ASSET}/manifest.json}"
FLOW_MODEL_PATH="${FLOW_MODEL_PATH:-${ASSET}/model}"
FLOW_CHECKPOINT_PATH="${FLOW_CHECKPOINT_PATH:-${ASSET}/model/model.safetensors}"
FLOW_VLM_PATH="${FLOW_VLM_PATH:-${ASSET}/base_vlm_metadata}"
test -x "${FLOW_PYTHON}"
test -f "${FLOW_INPUT_MANIFEST}"
test -d "${FLOW_MODEL_PATH}"
test -f "${FLOW_CHECKPOINT_PATH}"
test -d "${FLOW_VLM_PATH}"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
test -f "${CONTROL}/allocation_guard.py"
cp "${CONTROL}/allocation_guard.py" "${OUT}/allocation_guard.py"
cp "${CONTROL}/flow_screen.py" "${OUT}/flow_screen.py"
cp "${CONTROL}/IDEA_AND_PROTOCOL.zh.md" "${OUT}/IDEA_AND_PROTOCOL.zh.md"
cp "${CONTROL}/INDEPENDENT_AUDIT.zh.md" "${OUT}/INDEPENDENT_AUDIT.zh.md"

"${FLOW_PYTHON}" "${OUT}/allocation_guard.py" > "${OUT}/allocation.json"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader > "${OUT}/gpu_identity.csv"

EXTRA_ARGS=()
if [[ -n "${FLOW_LEROBOT_SOURCE:-}" ]]; then
    test -d "${FLOW_LEROBOT_SOURCE}"
    EXTRA_ARGS+=(--lerobot-source "${FLOW_LEROBOT_SOURCE}")
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

timeout --signal=TERM --kill-after=30s 40m \
    "${FLOW_PYTHON}" "${OUT}/flow_screen.py" \
    --input-manifest "${FLOW_INPUT_MANIFEST}" \
    --model-path "${FLOW_MODEL_PATH}" \
    --checkpoint "${FLOW_CHECKPOINT_PATH}" \
    --vlm-path "${FLOW_VLM_PATH}" \
    --output "${OUT}" \
    --max-seconds 2400 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT}/run.log"

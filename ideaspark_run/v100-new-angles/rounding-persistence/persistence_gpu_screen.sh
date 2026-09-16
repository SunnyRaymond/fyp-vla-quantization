#!/usr/bin/env bash
# Matched temporal rounding-persistence screen for one V100.
# All model loading, hashing, input processing and inference require the real
# CCDS SLURM allocation checked by allocation_guard.sh below.

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=rounding_persistence_v100
#SBATCH --output=artifacts/rounding_persistence_%j.log

set -euo pipefail

TOP="${ROUNDING_PERSISTENCE_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"

# The allocation guard is deliberately the first scheduler-dependent action.
source "${CONTROL}/allocation_guard.sh"

PYTHON="${ROUNDING_PERSISTENCE_PYTHON:-${TOP}/smolvla/venv/bin/python}"
ASSET="${ROUNDING_PERSISTENCE_ASSET:-${TOP}/smolvla}"
MANIFEST="${ROUNDING_PERSISTENCE_MANIFEST:-${TOP}/rounding_persistence_ready/manifest.json}"
MODEL="${ROUNDING_PERSISTENCE_MODEL:-${ASSET}/model}"
CHECKPOINT="${ROUNDING_PERSISTENCE_CHECKPOINT:-${ASSET}/model/model.safetensors}"
VLM="${ROUNDING_PERSISTENCE_VLM:-${ASSET}/base_vlm_metadata}"
FLOW_HELPER="${ROUNDING_PERSISTENCE_FLOW_HELPER:-${CONTROL}/flow_screen.py}"
PROTOCOL_COPY="${ROUNDING_PERSISTENCE_PROTOCOL_COPY:-${CONTROL}/persistence_protocol.zh.md}"

for required in \
  "${CONTROL}/persistence_screen.py" \
  "${CONTROL}/allocation_guard.py" \
  "${CONTROL}/allocation_guard.sh" \
  "${FLOW_HELPER}" \
  "${PROTOCOL_COPY}" \
  "${MANIFEST}" \
  "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required pinned input is missing: ${required}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON}" || ! -d "${MODEL}" || ! -d "${VLM}" ]]; then
  echo "pinned SmolVLA runtime paths are unavailable" >&2
  exit 3
fi

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing output: ${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}"
cp "${CONTROL}/persistence_screen.py" "${OUT}/persistence_screen.py"
cp "${CONTROL}/allocation_guard.py" "${OUT}/allocation_guard.py"
cp "${CONTROL}/allocation_guard.sh" "${OUT}/allocation_guard.sh"
cp "${FLOW_HELPER}" "${OUT}/flow_screen.py"
cp "${PROTOCOL_COPY}" "${OUT}/persistence_protocol.zh.md"

exec > >(tee -a "${OUT}/run.log") 2>&1
"${PYTHON}" "${OUT}/allocation_guard.py" > "${OUT}/allocation.json"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader > "${OUT}/gpu_identity.csv"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${OUT}:${CONTROL}${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_ARGS=()
if [[ -n "${ROUNDING_PERSISTENCE_LEROBOT_SOURCE:-}" ]]; then
  test -d "${ROUNDING_PERSISTENCE_LEROBOT_SOURCE}"
  EXTRA_ARGS+=(--lerobot-source "${ROUNDING_PERSISTENCE_LEROBOT_SOURCE}")
fi

timeout --signal=TERM --kill-after=15s 9m \
  "${PYTHON}" "${OUT}/persistence_screen.py" \
  --input-manifest "${MANIFEST}" \
  --model-path "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --vlm-path "${VLM}" \
  --flow-helper "${OUT}/flow_screen.py" \
  --protocol-copy "${PROTOCOL_COPY}" \
  --output "${OUT}" \
  --max-seconds 540 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUT}/python.log"

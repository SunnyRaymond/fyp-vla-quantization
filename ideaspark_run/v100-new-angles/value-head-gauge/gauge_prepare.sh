#!/bin/bash
# CPU-only fresh Cartpole reset preparation for value-head-gauge.
# Submit through the CCDS campaign controller; never run on a login node.
# No model/checkpoint load, env.step, env.render, GPU, or full rollout.
# TDQ_PARENT_MANIFEST must be set explicitly to the selected compatible parent.
# There is intentionally no default checkpoint path.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=value_head_gauge_prep
#SBATCH --output=artifacts/value_head_gauge_prep_%j.log
set -euo pipefail

TOP="${VALUE_GAUGE_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
ASSET="${VALUE_GAUGE_ASSETDIR:-${TOP}/value_head_gauge_ready}"
EXTRA3="${TOP}/tdmpc2_q_coupling_runtime_extra3"
EXTRA2="${TOP}/tdmpc2_q_coupling_runtime_extra2"
VENDOR="${TOP}/tdmpc2_q_coupling_runtime"

# This must be the first scheduler-dependent action.
source "${TOP}/control/allocation_guard.sh"
export TDQ_PARENT_MANIFEST="${TOP}/tdq_compatible_checkpoint/manifest.json"
export TDQ_PARENT_MANIFEST_SHA256="9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369"

if [[ -z "${TDQ_PARENT_MANIFEST:-}" ]]; then
  echo "TDQ_PARENT_MANIFEST is required; refusing an implicit checkpoint choice" >&2
  exit 3
fi
if [[ ! -f "${TDQ_PARENT_MANIFEST}" ]]; then
  echo "explicit TDQ parent manifest is missing: ${TDQ_PARENT_MANIFEST}" >&2
  exit 3
fi
if [[ -e "${OUT}" || -e "${ASSET}" ]]; then
  echo "refusing to overwrite output or gauge asset: ${OUT} ${ASSET}" >&2
  exit 3
fi
for required in "${EXTRA3}" "${EXTRA2}" "${VENDOR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "required CPU runtime overlay is missing: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${OUT}"
cp "${CONTROL}/gauge_prepare.py" "${CONTROL}/gauge_prepare.sh" "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing CPU runtime is unavailable: ${PY}" >&2
  exit 3
fi
export PYTHONPATH="${EXTRA3}:${EXTRA2}:${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

exec "${PY}" "${OUT}/gauge_prepare.py" \
  --root "${TOP}" \
  --assetdir "${ASSET}" \
  --output "${OUT}" \
  --parent-manifest "${TDQ_PARENT_MANIFEST}"

#!/bin/bash
# CPU-only fresh Cartpole reset preparation for policy-prior-support.
# Submit through the CCDS campaign controller; never run on a login node.
# No model/checkpoint load, env.step, env.render, GPU, or full rollout.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=policy_prior_support_prep
#SBATCH --output=artifacts/policy_prior_support_prep_%j.log
set -euo pipefail

TOP="${POLICY_SUPPORT_TOP:-${HOME}/v100_newangles_ccds}"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID:-unknown}"
ASSET="${POLICY_SUPPORT_ASSETDIR:-${TOP}/policy_prior_support_ready}"
PARENT="${TDQ_PARENT_MANIFEST:-${TOP}/tdq_compatible_checkpoint/manifest.json}"
PROTOCOL_ALIAS="${POLICY_SUPPORT_PROTOCOL:-${CONTROL}/policy_support_protocol.zh.md}"
EXTRA3="${TOP}/tdmpc2_q_coupling_runtime_extra3"
EXTRA2="${TOP}/tdmpc2_q_coupling_runtime_extra2"
VENDOR="${TOP}/tdmpc2_q_coupling_runtime"
EXPECTED_PARENT_SHA="9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369"

# This must be the first scheduler-dependent action.
source "${CONTROL}/allocation_guard.sh"
export TDQ_PARENT_MANIFEST="${PARENT}"
export TDQ_PARENT_MANIFEST_SHA256="${EXPECTED_PARENT_SHA}"
export POLICY_SUPPORT_PROTOCOL="${PROTOCOL_ALIAS}"

if [[ -z "${TDQ_PARENT_MANIFEST:-}" || ! -f "${TDQ_PARENT_MANIFEST}" ]]; then
  echo "explicit compatible TDQ parent manifest is missing: ${TDQ_PARENT_MANIFEST}" >&2
  exit 3
fi
if [[ ! -f "${POLICY_SUPPORT_PROTOCOL}" ]]; then
  echo "root-generated policy_support_protocol.zh.md is missing: ${POLICY_SUPPORT_PROTOCOL}" >&2
  exit 3
fi
if [[ -e "${OUT}" || -e "${ASSET}" ]]; then
  echo "refusing to overwrite output or policy-support asset: ${OUT} ${ASSET}" >&2
  exit 3
fi
for required in "${EXTRA3}" "${EXTRA2}" "${VENDOR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "required CPU runtime overlay is missing: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${OUT}"
cp "${CONTROL}/policy_support_prepare.py" "${CONTROL}/policy_support_prepare.sh" "${OUT}/"
cp "${POLICY_SUPPORT_PROTOCOL}" "${OUT}/policy_support_protocol.zh.md"
exec > >(tee -a "${OUT}/run.log") 2>&1

PY="${TOP}/smolvla/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "existing CPU runtime is unavailable: ${PY}" >&2
  exit 3
fi
export PYTHONPATH="${EXTRA3}:${EXTRA2}:${VENDOR}:${CONTROL}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="-1"

exec "${PY}" "${OUT}/policy_support_prepare.py" \
  --root "${TOP}" \
  --assetdir "${ASSET}" \
  --output "${OUT}" \
  --parent-manifest "${TDQ_PARENT_MANIFEST}" \
  --protocol "${OUT}/policy_support_protocol.zh.md"

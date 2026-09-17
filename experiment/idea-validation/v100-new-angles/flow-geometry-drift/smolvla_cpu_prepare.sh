#!/bin/bash
# CPU-only bounded preparation.  Submit through ccds_campaign_control.py.
# No --gres is requested; the guard must succeed before any heavy operation.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --job-name=smolvla_prepare
#SBATCH --output=artifacts/smolvla_prepare_%j.log
set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
ASSETDIR="${TOP}/smolvla"

# Keep the guard as the first scheduler-dependent action, while leaving a
# bounded diagnostic if the guard itself rejects the allocation.
guard_failure_record() {
  rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    failure_out="${TOP}/artifacts/${SLURM_JOB_ID}"
    mkdir -p "${failure_out}"
    cat > "${failure_out}/guard_failure.json" <<EOF
{"schema":"smolvla-cpu-preparation-status-v1","state":"failed","phase":"allocation_guard","job_id":"${SLURM_JOB_ID}","hostname":"$(hostname -s)","exit_code":${rc}}
EOF
  fi
  exit "${rc}"
}
trap guard_failure_record EXIT

# This is intentionally the first operation that depends on the scheduler.
source "${TOP}/control/allocation_guard.sh"
trap - EXIT

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
  echo "CPU-only preparation refuses a visible CUDA device" >&2
  exit 2
fi
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
mkdir -p "${OUT}" "${ASSETDIR}"
cp "${CONTROL}/smolvla_cpu_prepare.py" "${OUT}/smolvla_cpu_prepare.py"
cp "${CONTROL}/smolvla_cpu_prepare.sh" "${OUT}/smolvla_cpu_prepare.sh"
cp "${CONTROL}/PREPARATION.zh.md" "${OUT}/PREPARATION.zh.md"
cat > "${OUT}/status.json" <<EOF
{"schema":"smolvla-cpu-preparation-status-v1","state":"running","phase":"dependency_installing","job_id":"${SLURM_JOB_ID}","hostname":"$(hostname -s)","allocation_guard":"shell_passed"}
EOF
cp "${OUT}/status.json" "${ASSETDIR}/status.json"
exec > >(tee -a "${OUT}/run.log") 2>&1

BOOTSTRAP="${HOME}/cem_update_ccds/venv/bin/python"
if [[ ! -x "${BOOTSTRAP}" ]]; then
  echo "known Python 3.10 bootstrap is unavailable: ${BOOTSTRAP}" >&2
  exit 3
fi
VENV="${ASSETDIR}/venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BOOTSTRAP}" -m venv "${VENV}"
fi
PY="${VENV}/bin/python"
export PYTHONPATH="${CONTROL}:${PYTHONPATH:-}"
export PIP_CACHE_DIR="${ASSETDIR}/pip-cache"
export PYTHONUNBUFFERED=1
export HF_HOME="${ASSETDIR}/hf-home"

"${PY}" -m pip install --disable-pip-version-check \
  --index-url https://pypi.org/simple "pip==25.3"
"${PY}" -m pip install --disable-pip-version-check \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124"
"${PY}" -m pip install --disable-pip-version-check \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "lerobot[smolvla]==0.4.4" "torchcodec==0.2.1"

"${PY}" "${OUT}/smolvla_cpu_prepare.py" --assetdir "${ASSETDIR}" --output "${OUT}" --resume-from-job 64757

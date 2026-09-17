#!/bin/bash
# Reference-branch coupling: guarded single-V100 producer.
# Heavy loading, hashing, data access, and inference run only after the guard.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=reference_branch
#SBATCH --output=artifacts/reference_%j.log
set -euo pipefail

TOP="${HOME}/v100_newangles_ccds"
OLD="${HOME}/cem_update_ccds"
ROOT="${OLD}/modelroot"
CONTROL="${TOP}/control"
source "${CONTROL}/allocation_guard.sh"

OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
test ! -e "${OUT}"
mkdir -p "${OUT}"
for f in reference_screen.py reference_gpu.sh RAW_CONTRACT.json reference_protocol.zh.md reference_input_freeze.json teacher_bias_screen.py; do
    cp "${CONTROL}/${f}" "${OUT}/${f}"
done
cp "${OLD}/control/smoke_runner.py" "${OUT}/smoke_runner.py"
cp "${OLD}/control/screen_runner.py" "${OUT}/screen_runner.py"
cp "${CONTROL}/allocation_guard.py" "${OUT}/allocation_guard.py"
cp "${CONTROL}/allocation_guard.sh" "${OUT}/allocation_guard.sh"

exec > >(tee -a "${OUT}/run.log") 2>&1
export PYTHONPATH="${OUT}:${OLD}/control:${ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${OLD}/cache/torch"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
PY="${OLD}/venv/bin/python"
"${PY}" "${OUT}/allocation_guard.py" > "${OUT}/allocation.json"

set +e
timeout --signal=TERM --kill-after=15s 270s "${PY}" "${OUT}/reference_screen.py" \
    --root "${ROOT}" \
    --input-manifest "${TOP}/teacher_bias_ready2/manifest.json" \
    --teacher-helper "${OUT}/teacher_bias_screen.py" \
    --helper-dir "${OUT}" \
    --protocol "${OUT}/reference_protocol.zh.md" \
    --freeze "${OUT}/reference_input_freeze.json" \
    --output "${OUT}" \
    --max-seconds 180
code=$?
printf '{"exit_code":%s}\n' "${code}" > "${OUT}/exit_status.json"
exit "${code}"

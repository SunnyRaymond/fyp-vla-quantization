#!/usr/bin/env bash
# CPU-only preparation for the frozen DINO-WM Wall teacher-bias diagnostic.
# It never loads a model, runs inference, calls an environment, downloads, or installs.
# Submit only inside a real CCDS SLURM allocation.

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=teacher_bias_prepare
#SBATCH --output=artifacts/teacher_bias_prepare_%j.log

set -euo pipefail

TOP="${TEACHER_BIAS_TOP:-${HOME}/v100_newangles_ccds}"
OLD="${TEACHER_BIAS_OLD:-${HOME}/cem_update_ccds}"
ROOT="${TEACHER_BIAS_MODELROOT:-${OLD}/modelroot}"
CONTROL="${TOP}/control"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
ASSET="${TEACHER_BIAS_ASSET:-${TOP}/teacher_bias_ready2}"
PY="${TEACHER_BIAS_PYTHON:-${OLD}/venv/bin/python}"
SOURCE="${ROOT}/source"
CONFIG="${ROOT}/checkpoints/outputs/wall_single/hydra.yaml"
CHECKPOINT="${ROOT}/checkpoints/outputs/wall_single/checkpoints/model_latest.pth"
DATASET_ROOT="${ROOT}/data"

# The real-allocation guard is the first scheduler-dependent action.
source "${CONTROL}/allocation_guard.sh"

if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite existing job output: ${OUT}" >&2
  exit 3
fi
if [[ ! -x "${PY}" ]]; then
  echo "pinned CPU Python executable is unavailable: ${PY}" >&2
  exit 3
fi
for required in \
  "${CONTROL}/prepare_teacher_bias.py" \
  "${CONTROL}/prepare_teacher_bias.sh" \
  "${CONTROL}/allocation_guard.py" \
  "${CONTROL}/allocation_guard.sh" \
  "${CONTROL}/teacher_bias_protocol.zh.md" \
  "${SOURCE}/datasets/wall_dset.py" \
  "${SOURCE}/datasets/traj_dset.py" \
  "${SOURCE}/datasets/img_transforms.py" \
  "${SOURCE}/models/visual_world_model.py" \
  "${CONFIG}" \
  "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required pinned input is missing: ${required}" >&2
    exit 3
  fi
done
test -d "${DATASET_ROOT}/wall_single"

mkdir -p "${OUT}"
cp "${CONTROL}/prepare_teacher_bias.py" \
   "${CONTROL}/prepare_teacher_bias.sh" \
   "${CONTROL}/allocation_guard.py" \
   "${CONTROL}/allocation_guard.sh" \
   "${CONTROL}/teacher_bias_protocol.zh.md" \
   "${OUT}/"
exec > >(tee -a "${OUT}/run.log") 2>&1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export DATASET_DIR="${DATASET_ROOT}"
export PYTHONPATH="${OUT}:${SOURCE}:${CONTROL}:${PYTHONPATH:-}"

set +e
timeout --signal=TERM --kill-after=10s 270s \
  "${PY}" "${OUT}/prepare_teacher_bias.py" \
  --root "${ROOT}" \
  --source "${SOURCE}" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --assetdir "${ASSET}" \
  --output "${OUT}"
status=$?
set -e
exit "${status}"

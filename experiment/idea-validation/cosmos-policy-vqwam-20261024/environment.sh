#!/bin/bash
set -euo pipefail

COSMOS_ROOT=${COSMOS_ROOT:-/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024}
COSMOS_CODE_ROOT=${COSMOS_CODE_ROOT:-$COSMOS_ROOT}
ASSET_ROOT=${ASSET_ROOT:-/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015}
LEGACY_ROOT=${LEGACY_ROOT:-$ASSET_ROOT}
ARTIFACTS_TMP=${ARTIFACTS_TMP:?Per-job temporary directory is required}
ARTIFACTS=${ARTIFACTS:?Per-job artifact directory is required}
REPO=$LEGACY_ROOT/vendor/cosmos-policy
PLUS_ROOT=/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005
PLUS_NATIVE=$PLUS_ROOT/native/imagemagick-jammy
PLUS_NATIVE_LIB=$PLUS_NATIVE/usr/lib/x86_64-linux-gnu
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
PYTHON=$LEGACY_ROOT/env/bin/python

test -d "$REPO" -a -x "$PYTHON" -a -s "$SIF"
test -d "$PLUS_ROOT/libero-plus-compat" -a -d "$PLUS_ROOT/LIBERO-plus" -a -d "$PLUS_ROOT/deps"

run_python() {
  local nv_args=(--nv) cuda_devices
  if [[ "${COSMOS_CPU_ONLY:-0}" == 1 ]]; then
    nv_args=()
    cuda_devices=
  else
    cuda_devices=${CUDA_VISIBLE_DEVICES:?PBS did not assign CUDA_VISIBLE_DEVICES}
  fi
  local bind_args=(--bind /scratch,/scratch/GPFS/app/apps:/app/apps
                   --bind "$COSMOS_ROOT:$COSMOS_ROOT:rw"
                   --bind "$LEGACY_ROOT:$LEGACY_ROOT:ro"
                   --bind "$PLUS_ROOT:$PLUS_ROOT:ro"
                   --bind "$ARTIFACTS_TMP:/tmp:rw"
                   --bind "$PBS_NODEFILE:$PBS_NODEFILE:ro")
  local source_path
  while IFS= read -r source_path; do
    [[ -n "$source_path" && -e "$source_path" ]] && bind_args+=(--bind "$source_path:$source_path:ro")
  done < <(python3 - "$COSMOS_CODE_ROOT/protocol.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
for value in (p.get("source_paths", {}).get("teacher_original"),
              p.get("calibration", {}).get("source_root"),
              p.get("cohort", {}).get("manifest")):
    if value:
        print(value)
PY
)
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$cuda_devices \
  APPTAINERENV_TMPDIR=/tmp APPTAINERENV_TEMP=/tmp APPTAINERENV_TMP=/tmp \
  APPTAINERENV_HOME=/tmp/cosmos-home APPTAINERENV_PYTHONUNBUFFERED=1 \
  APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1 \
  APPTAINERENV_HF_HOME="$LEGACY_ROOT/cache/huggingface" \
  APPTAINERENV_COSMOS_JOB_START_EPOCH="$COSMOS_JOB_START_EPOCH" \
  APPTAINERENV_COSMOS_SOFT_DEADLINE_SECONDS="$COSMOS_SOFT_DEADLINE_SECONDS" \
  APPTAINERENV_COSMOS_SAVE_RESERVE_SECONDS=120 \
  apptainer exec --cleanenv "${nv_args[@]}" "${bind_args[@]}" --pwd "$REPO" \
    --env "ROOT=$COSMOS_ROOT,ASSET_ROOT=$ASSET_ROOT,LEGACY_ROOT=$LEGACY_ROOT,COSMOS_ROOT=$COSMOS_ROOT" \
    --env "PBS_JOBID=$PBS_JOBID,PBS_NODEFILE=$PBS_NODEFILE" \
    --env "COSMOS_CODE_ROOT=$COSMOS_CODE_ROOT,ARTIFACTS=$ARTIFACTS" \
    --env "PYTHONPATH=$ARTIFACTS/engine_sources:$COSMOS_CODE_ROOT:$ARTIFACTS/legacy_sources:$ARTIFACTS/source_helpers:$LEGACY_ROOT:$REPO:$COSMOS_HELPERS_PYTHONPATH:$PLUS_ROOT/libero-plus-compat:$PLUS_ROOT/LIBERO-plus:$PLUS_ROOT/deps" \
    --env "LIBERO_CONFIG_PATH=$PLUS_ROOT/libero-config" \
    --env "LD_LIBRARY_PATH=$PLUS_NATIVE_LIB:$PLUS_NATIVE/usr/lib:/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
    --env "MAGICK_HOME=$PLUS_NATIVE,MAGICK_CONFIGURE_PATH=$PLUS_NATIVE/etc/ImageMagick-6:$PLUS_NATIVE_LIB/ImageMagick-6.9.11/config-Q16" \
    --env "MAGICK_CODER_MODULE_PATH=$PLUS_NATIVE_LIB/ImageMagick-6.9.11/modules-Q16/coders,MAGICK_FILTER_MODULE_PATH=$PLUS_NATIVE_LIB/ImageMagick-6.9.11/modules-Q16/filters" \
    --env "OMP_NUM_THREADS=${COSMOS_CPU_THREADS:-8},OPENBLAS_NUM_THREADS=${COSMOS_CPU_THREADS:-8},LP_NUM_THREADS=${COSMOS_CPU_THREADS:-8}" \
    --env "HF_HUB_DISABLE_TELEMETRY=1,TOKENIZERS_PARALLELISM=false,MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LIBGL_ALWAYS_SOFTWARE=1,DETERMINISTIC=true,CUBLAS_WORKSPACE_CONFIG=:4096:8" \
    "$SIF" "$PYTHON" "$@"
}

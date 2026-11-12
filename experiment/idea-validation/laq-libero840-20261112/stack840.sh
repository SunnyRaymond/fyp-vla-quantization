#!/bin/bash
# Runs inside the LIBERO container: stack840.sh fastwam|cosmos command...
# Sets the model's software stack (venv / env, PYTHONPATH with this job's code first, then the panel-A LAQ code exactly as
# it ran on ASPIRE2A) on the ORIGINAL LIBERO suites, then execs the command.
set -euo pipefail
MODEL=$1; shift
U=/scratch/users/ntu/yguo017
F=$U/fastwam-smoke
CODE=${L840_CODE:?}
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 GIT_PYTHON_REFRESH=quiet
export LIBERO_CONFIG_PATH=$F/libero-config
T=${L840_THREADS:-2}
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T LP_NUM_THREADS=$T
case "$MODEL" in
  fastwam)
    FWA=$U/laq-fastwam-20261111/artifacts/26198651.pbs101
    PILOT=$U/fastwam-libero-plus-pilot-20261005
    export PATH=$F/venv/bin:$PATH
    export PYTHONPATH=$CODE:$FWA:$U/fastwam-smooth-vq-phases12-20261006:$U/fastwam-a4-phases12-20261006:$PILOT:$F/libero-compat:$F/libero:$F/FastWAM:$F/FastWAM/src:$F/FastWAM/experiments/libero
    export DIFFSYNTH_MODEL_BASE_PATH=$F/model-cache DIFFSYNTH_DOWNLOAD_SOURCE=huggingface DIFFSYNTH_SKIP_DOWNLOAD=true
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$F/cache/huggingface
    export PYTHONHASHSEED=20261022 FW_REQUIRE_REAL_QUANT=1 TRITON_CACHE_DIR=/projects/yguo017/.tmp/triton
    export BASE=$F ROOT=$PILOT NEW_ROOT=$U/laq-fastwam-20261111 DIAG_ROOT=$U/laq-fastwam-20261111
    export SMOOTH_ROOT=$U/fastwam-smooth-vq-phases12-20261006 SOURCE_ROOT=$U/fastwam-a4-phases12-20261006
    ;;
  cosmos)
    C=$U/cosmos-policy-c8-w4a4-20261015
    COA=$U/laq-cosmos-20261111/artifacts/26204098.pbs101
    export PATH=$C/env/bin:$PATH
    export PYTHONPATH=$CODE:$COA:$C:$C/vendor/cosmos-policy:$F/libero-compat:$F/libero
    export HF_HOME=$C/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
    export DETERMINISTIC=true
    unset ROOT
    cd "$C/vendor/cosmos-policy"
    ;;
  *) echo "unknown model $MODEL" >&2; exit 64;;
esac
export L840_MODEL=$MODEL
exec "$@"

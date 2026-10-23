# Source only after the calling PBS wrapper verifies the allocation.
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0 cuda/12.2.2
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
PLUS=/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005
REPO=$COSMOS_ROOT/vendor/cosmos-policy
NATIVE=$PLUS/native/imagemagick-jammy
NATIVE_LIB=$NATIVE/usr/lib/x86_64-linux-gnu
test -s "$SIF" -a -x "$COSMOS_ROOT/env/bin/python" -a -d "$REPO"
mkdir -p "$ARTIFACTS/tmp"
run_python() {
  APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" apptainer exec --cleanenv --nv \
    --bind /scratch,/scratch/GPFS/app/apps:/app/apps \
    --bind "$ARTIFACTS/tmp:/tmp" \
    --bind "$PBS_NODEFILE:$PBS_NODEFILE:ro" \
    --pwd "$REPO" \
    --env "PBS_JOBID=$PBS_JOBID,PBS_NODEFILE=$PBS_NODEFILE" \
    --env "COSMOS_ROOT=$COSMOS_ROOT,COSMOS_JOB_START_EPOCH=$COSMOS_JOB_START_EPOCH,COSMOS_SOFT_DEADLINE_SECONDS=${COSMOS_SOFT_DEADLINE_SECONDS:-6600},COSMOS_SAVE_RESERVE_SECONDS=120" \
    --env "PYTHONPATH=$ARTIFACTS:$ARTIFACTS/helpers:$COSMOS_ROOT:$REPO:$PLUS/libero-plus-compat:$PLUS/LIBERO-plus:$PLUS/deps" \
    --env "LIBERO_CONFIG_PATH=$PLUS/libero-config,HF_HOME=$COSMOS_ROOT/cache/huggingface" \
    --env "LD_LIBRARY_PATH=$NATIVE_LIB:$NATIVE/usr/lib:/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
    --env "MAGICK_HOME=$NATIVE/usr,MAGICK_CONFIGURE_PATH=$NATIVE/etc/ImageMagick-6:$NATIVE/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16" \
    --env "MAGICK_CODER_MODULE_PATH=$NATIVE/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders,MAGICK_FILTER_MODULE_PATH=$NATIVE/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/filters" \
    --env "OMP_NUM_THREADS=${COSMOS_CPU_THREADS:-8},OPENBLAS_NUM_THREADS=${COSMOS_CPU_THREADS:-8},LP_NUM_THREADS=${COSMOS_CPU_THREADS:-8}" \
    --env "HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HUB_DISABLE_TELEMETRY=1,TOKENIZERS_PARALLELISM=false" \
    --env "MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LIBGL_ALWAYS_SOFTWARE=1,DETERMINISTIC=true,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONUNBUFFERED=1" \
    "$SIF" "$COSMOS_ROOT/env/bin/python" "$@"
}

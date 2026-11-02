# Sourced by the PBS scripts after the allocation check. Needs ASSET_ROOT, ARTIFACTS (source snapshot of this job).
# run_python runs the Cosmos env python inside the LIBERO container with the same binds and variables as the
# BF16 panel-A run (cosmos-policy-c8-w4a4 gpu_environment.sh), plus this job's sources first on PYTHONPATH.
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0 cuda/12.2.2
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
PLUS=/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005
REPO=$ASSET_ROOT/vendor/cosmos-policy
NATIVE=$PLUS/native/imagemagick-jammy
NATIVE_LIB=$NATIVE/usr/lib/x86_64-linux-gnu
test -s "$SIF" -a -x "$ASSET_ROOT/env/bin/python" -a -d "$REPO"
mkdir -p "$ARTIFACTS/tmp"
run_python() {
  local nv=(--nv)
  [[ "${COSMOS_CPU_ONLY:-0}" == 1 ]] && nv=()
  APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" apptainer exec --cleanenv "${nv[@]}" \
    --bind /scratch,/scratch/GPFS/app/apps:/app/apps \
    --bind "$ARTIFACTS/tmp:/tmp" \
    --bind "$PBS_NODEFILE:$PBS_NODEFILE:ro" \
    --pwd "$REPO" \
    --env "PBS_JOBID=$PBS_JOBID,PBS_NODEFILE=$PBS_NODEFILE,COSMOS_JOB_START_EPOCH=$COSMOS_JOB_START_EPOCH" \
    --env "PYTHONPATH=$ARTIFACTS:$ASSET_ROOT:$REPO:$PLUS/libero-plus-compat:$PLUS/LIBERO-plus:$PLUS/deps" \
    --env "LIBERO_CONFIG_PATH=$PLUS/libero-config,HF_HOME=$ASSET_ROOT/cache/huggingface" \
    --env "LD_LIBRARY_PATH=$NATIVE_LIB:$NATIVE/usr/lib:/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
    --env "MAGICK_HOME=$NATIVE/usr,MAGICK_CONFIGURE_PATH=$NATIVE/etc/ImageMagick-6:$NATIVE/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16" \
    --env "MAGICK_CODER_MODULE_PATH=$NATIVE/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders,MAGICK_FILTER_MODULE_PATH=$NATIVE/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/filters" \
    --env "OMP_NUM_THREADS=${COSMOS_CPU_THREADS:-8},OPENBLAS_NUM_THREADS=${COSMOS_CPU_THREADS:-8},LP_NUM_THREADS=${COSMOS_CPU_THREADS:-8},MKL_NUM_THREADS=${COSMOS_CPU_THREADS:-8}" \
    --env "HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HUB_DISABLE_TELEMETRY=1,TOKENIZERS_PARALLELISM=false" \
    --env "MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LIBGL_ALWAYS_SOFTWARE=1,DETERMINISTIC=true,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONUNBUFFERED=1" \
    "$SIF" "$ASSET_ROOT/env/bin/python" "$@"
}

monitor() {  # ncpus uuid...: GPU_SAMPLE every 60 s, CPU_SAMPLE (busy cores of this user over 2 s) every 120 s
  local ncpus=$1; shift
  local me n=0
  me=$(id -un)
  while :; do
    for uuid in "$@"; do
      printf 'GPU_SAMPLE timestamp=%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi -i "$uuid" --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader \
        || echo GPU_SAMPLE_ERROR
    done
    if (( n % 2 == 0 )); then
      top -b -n 2 -d 2 -u "$me" 2>/dev/null | awk -v ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)" -v req="$ncpus" \
        '/^top -/ {frame++} frame == 2 && $1 ~ /^[0-9]+$/ {s += $9} END {printf "CPU_SAMPLE timestamp=%s busy_cores=%.1f requested=%s\n", ts, s / 100, req}' \
        || echo CPU_SAMPLE_ERROR
    fi
    n=$(( n + 1 ))
    sleep 60
  done
}

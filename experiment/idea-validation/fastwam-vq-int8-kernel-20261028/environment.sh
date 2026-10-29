# Source only after the PBS allocation guards in run.pbs.
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0 cuda/12.2.2
BASE=/scratch/users/ntu/yguo017/fastwam-smoke
PILOT=/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif

run_python() {
  apptainer exec --cleanenv --nv \
    --bind /scratch,/scratch/GPFS/app/apps:/app/apps \
    --bind "$PBS_NODEFILE:$PBS_NODEFILE:ro" \
    --bind /usr/lib64/libffi.so.6:/usr/lib/x86_64-linux-gnu/libffi.so.6 \
    --bind /usr/lib64/libssl.so.1.1:/usr/lib/x86_64-linux-gnu/libssl.so.1.1 \
    --bind /usr/lib64/libcrypto.so.1.1:/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 \
    --env "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:${LD_LIBRARY_PATH:-}" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env "CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=42" \
    --env "PBS_JOBID=$PBS_JOBID,PBS_NODEFILE=$PBS_NODEFILE,ARTIFACTS=$ARTIFACTS,ROOT=$BASE,BASE=$BASE" \
    --env "PYTHONPATH=$ARTIFACTS:$BASE/libero-compat:$BASE/libero:$PILOT/deps:$BASE/FastWAM:$BASE/FastWAM/src:$BASE/FastWAM/experiments/libero" \
    --env "LIBERO_CONFIG_PATH=$BASE/libero-config" \
    --env "DIFFSYNTH_MODEL_BASE_PATH=$BASE/model-cache" \
    --env DIFFSYNTH_DOWNLOAD_SOURCE=huggingface,DIFFSYNTH_SKIP_DOWNLOAD=true,HF_HUB_OFFLINE=1 \
    --env "HF_HOME=$BASE/cache/huggingface" \
    --env MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LIBGL_ALWAYS_SOFTWARE=1 \
    --env OMP_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,LP_NUM_THREADS=8 \
    --env GIT_PYTHON_REFRESH=quiet,PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false \
    "$SIF" "$BASE/venv/bin/python" "$@"
}

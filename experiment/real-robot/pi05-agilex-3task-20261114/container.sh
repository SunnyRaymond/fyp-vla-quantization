# Sourced by prep.pbs and train.pbs after the PBS allocation is verified. Needs RUN ENV_ROOT SIF ARTIFACTS NCPUS
# PBS_JOBID; GPU jobs set NV=1. openpi runs from its uv venv inside the Ubuntu 22.04 SIF, called directly (never
# `uv run`, which would reinstall torchcodec; the SIF has no FFmpeg libraries). All caches stay on /scratch.
module purge
module load apptainer/1.5.0
NV_ARGS=()
if [[ "${NV:-0}" == 1 ]]; then NV_ARGS=(--nv); fi
mkdir -p "$RUN/cache/hf_datasets" "$RUN/tmp/$PBS_JOBID" "$RUN/data/lerobot"

openpi_python() {
  local extra=()
  if [[ -n "${JAX_PLATFORMS:-}" ]]; then extra+=(--env "JAX_PLATFORMS=$JAX_PLATFORMS"); fi
  apptainer exec --cleanenv "${NV_ARGS[@]}" --home "$ENV_ROOT/home" --bind /scratch \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env "RUN=$RUN,ENV_ROOT=$ENV_ROOT,OPENPI_REPO=$ENV_ROOT/openpi,PBS_JOBID=$PBS_JOBID,NCPUS=$NCPUS" \
    --env "OPENPI_DATA_HOME=$ENV_ROOT/cache/openpi,HF_HOME=$ENV_ROOT/cache/huggingface,XDG_CACHE_HOME=$ENV_ROOT/cache/xdg" \
    --env "HF_LEROBOT_HOME=$RUN/data/lerobot,HF_DATASETS_CACHE=$RUN/cache/hf_datasets" \
    --env "HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,WANDB_MODE=disabled,TMPDIR=$RUN/tmp/$PBS_JOBID" \
    --env "OMP_NUM_THREADS=${OMP_THREADS:-4},PYTHONUNBUFFERED=1,PYTHONIOENCODING=utf-8,TOKENIZERS_PARALLELISM=false" \
    --env "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9,TQDM_MININTERVAL=60" \
    "${extra[@]}" \
    "$SIF" "$ENV_ROOT/openpi/.venv/bin/python" "$@"
}

# Snapshot the uploaded sources into the job's artifact dir and run from there.
stage_sources() {
  local name
  for name in protocol.json container.sh agilex_openpi.py convert_openpi.py; do
    test -s "$RUN/$name" || { echo "Uploaded source missing: $name" >&2; exit 65; }
    cp "$RUN/$name" "$ARTIFACTS/"
  done
}

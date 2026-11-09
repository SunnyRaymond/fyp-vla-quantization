# Sourced by prep.pbs and train.pbs after the PBS allocation is verified.
# Needs ROOT BASE SIF ARTIFACTS NCPUS FW_THREADS; GPU jobs set NV=1. Same container, venv, binds and model cache as
# the LIBERO quantization runs (fastwam-smoke); no LIBERO/MuJoCo paths are needed for training.
module purge
if [[ "${NV:-0}" == 1 ]]; then
  module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0 cuda/12.2.2
  NV_ARGS=(--nv)
  # `import deepspeed` on a GPU node probes `$CUDA_HOME/bin/nvcc -V`; the container's /usr/local/cuda has no nvcc.
  # Point it at the host toolkit matching torch's cu128 (only the version probe uses it; ZeRO-2 builds no ops).
  CONTAINER_CUDA_HOME=/app/apps/cuda/12.8.1
else
  module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0
  NV_ARGS=()
  CONTAINER_CUDA_HOME=
fi
export LD_LIBRARY_PATH="/.singularity.d/libs:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/cache/hf_datasets" "$ROOT/cache/torch_extensions" "$ROOT/cache/triton"

run_python() {
  apptainer exec --cleanenv "${NV_ARGS[@]}" \
    --bind /scratch,/scratch/GPFS/app/apps:/app/apps \
    --bind /usr/lib64/libffi.so.6:/usr/lib/x86_64-linux-gnu/libffi.so.6 \
    --bind /usr/lib64/libssl.so.1.1:/usr/lib/x86_64-linux-gnu/libssl.so.1.1 \
    --bind /usr/lib64/libcrypto.so.1.1:/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 \
    --env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env "CUDA_HOME=$CONTAINER_CUDA_HOME" \
    --env "PYTHONPATH=$ARTIFACTS:$BASE/FastWAM:$BASE/FastWAM/src" \
    --env "DIFFSYNTH_MODEL_BASE_PATH=${FW_MODEL_CACHE:-$BASE/model-cache}" \
    --env DIFFSYNTH_DOWNLOAD_SOURCE=huggingface,DIFFSYNTH_SKIP_DOWNLOAD=true,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1 \
    --env "HF_HOME=$BASE/cache/huggingface,HF_DATASETS_CACHE=$ROOT/cache/hf_datasets" \
    --env "TORCH_EXTENSIONS_DIR=$ROOT/cache/torch_extensions,TRITON_CACHE_DIR=$ROOT/cache/triton" \
    --env "OMP_NUM_THREADS=$FW_THREADS,MKL_NUM_THREADS=$FW_THREADS,NCPUS=$NCPUS" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "FW_OPT_GROUPS=${FW_OPT_GROUPS:-16},FW_INIT_REPORT=${FW_INIT_REPORT:-},FW_DUMP_CONFIG=${FW_DUMP_CONFIG:-}" \
    --env "FW_OFFLOAD=${FW_OFFLOAD:-0},FW_MEMPROBE=${FW_MEMPROBE:-},FW_LAYER_CHECKPOINT=${FW_LAYER_CHECKPOINT:-1}" \
    --env "PATH=$BASE/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/app/apps/gcc/11.4.0-nscc/bin" \
    --env "PBS_JOBID=$PBS_JOBID" \
    --env PYTHONUNBUFFERED=1,PYTHONIOENCODING=utf-8,TOKENIZERS_PARALLELISM=false,GIT_PYTHON_REFRESH=quiet \
    "$SIF" "$BASE/venv/bin/python" "$@"
}

# Copy the uploaded sources into the job's artifact dir: FastWAM's configs with the AgileX data/task configs added,
# and the training / text-embedding entry points under scripts/ so their hydra config path ../configs resolves there.
stage_sources() {
  local name
  for name in protocol.json environment.sh fw_agilex.py train_agilex.py convert_agilex.py check_data.py check_model.py \
      ds_zero2.json accelerate_zero2.yaml configs/data/agilex_2cam.yaml configs/task/agilex_optional_idm_2cam224.yaml; do
    test -s "$ROOT/$name" || { echo "Uploaded source missing: $name" >&2; exit 65; }
  done
  rm -rf "$ARTIFACTS/configs" "$ARTIFACTS/scripts"
  cp -r "$BASE/FastWAM/configs" "$ARTIFACTS/configs"
  cp "$ROOT/configs/data/agilex_2cam.yaml" "$ARTIFACTS/configs/data/"
  cp "$ROOT/configs/task/agilex_optional_idm_2cam224.yaml" "$ARTIFACTS/configs/task/"
  mkdir -p "$ARTIFACTS/scripts"
  cp "$ROOT/train_agilex.py" "$ARTIFACTS/scripts/"
  cp "$BASE/FastWAM/scripts/precompute_text_embeds.py" "$ARTIFACTS/scripts/"
  for name in protocol.json fw_agilex.py convert_agilex.py check_data.py check_model.py ds_zero2.json; do
    cp "$ROOT/$name" "$ARTIFACTS/"
  done
  sed "s#DS_CONFIG_PATH#$ARTIFACTS/ds_zero2.json#" "$ROOT/accelerate_zero2.yaml" > "$ARTIFACTS/accelerate_zero2.yaml"
}

#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=cem_cpu_prepare
#SBATCH --output=artifacts/prepare_%j.log
set -euo pipefail
BASE="$HOME/cem_update_ccds"
source "$BASE/control/allocation_guard.sh"
mkdir "$BASE/.prepare_lock"
trap 'rmdir "$BASE/.prepare_lock"' EXIT
ROOT="$BASE/modelroot"
mkdir -p "$ROOT/downloads" "$ROOT/checkpoints" "$ROOT/data" "$BASE/run"
module load python/3.10 uv
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export UV_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/cem-uv-${SLURM_JOB_ID}" UV_HTTP_TIMEOUT=120
export TORCH_HOME="$BASE/cache/torch" MPLBACKEND=Agg PYTHONUNBUFFERED=1
if [ ! -x "$BASE/venv/bin/python" ]; then
  uv venv --python 3.10 "$BASE/venv"
fi
uv pip install --quiet --python "$BASE/venv/bin/python" 'torch==2.2.0' 'torchvision==0.17.0' 'numpy==1.26.4' 'gym==0.23.1' 'hydra-core==1.3.2' 'omegaconf==2.3.0' 'wandb==0.13.1' 'protobuf==3.20.3' 'submitit==1.5.1' 'einops==0.4.1' 'decord==0.6.0' 'imageio==2.34.1' 'imageio-ffmpeg==0.4.9' 'matplotlib==3.7.5' 'scipy==1.13.1' 'psutil==5.9.8' 'tqdm==4.66.4' 'h5py==3.11.0' 'opencv-python-headless==4.6.0.66' 'accelerate==0.26.1' 'setuptools==70.0.0'
git init "$ROOT/source"
git -C "$ROOT/source" remote add origin https://github.com/gaoyuezhou/dino_wm.git
git -C "$ROOT/source" fetch --depth 1 origin 0a9492fa12044b852ae9e001cc74604b79c8bb0c
git -C "$ROOT/source" checkout --detach FETCH_HEAD
git -C "$ROOT/source" rev-parse HEAD
curl -fLsS --retry 2 --max-time 1500 https://osf.io/download/xvzs4/ -o "$ROOT/downloads/outputs.zip.partial"
mv "$ROOT/downloads/outputs.zip.partial" "$ROOT/downloads/outputs.zip"
curl -fLsS --retry 2 --max-time 1500 https://osf.io/download/49rnx/ -o "$ROOT/downloads/wall_single.zip.partial"
mv "$ROOT/downloads/wall_single.zip.partial" "$ROOT/downloads/wall_single.zip"
"$BASE/venv/bin/python" "$BASE/control/prepare_assets_ccds.py" --base "$BASE"
uv pip freeze --python "$BASE/venv/bin/python" > "$BASE/requirements-resolved.txt"

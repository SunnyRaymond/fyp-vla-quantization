#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=frt_core_probe
#SBATCH --output=artifacts/core_probe_%j.log
set -euo pipefail
TOP="$HOME/frt_ccds"
OLD="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
JOB_TRES=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_TRES" == *"gres/gpu=1"* ]]
test -n "${CUDA_VISIBLE_DEVICES:-}"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
for name in core_probe.py frt_core.py allocation_guard.py; do cp "$TOP/control/$name" "$OUT/$name"; done
export PYTHONPATH="$OUT:$OLD/control:$OLD/modelroot/source"
export TORCH_HOME="$OLD/cache/torch" MPLBACKEND=Agg WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
timeout --signal=TERM --kill-after=15s 8m "$OLD/venv/bin/python" "$OUT/core_probe.py" --root "$OLD/modelroot" --output "$OUT"

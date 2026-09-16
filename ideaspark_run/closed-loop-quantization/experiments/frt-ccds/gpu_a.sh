#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=frt_stage_a
#SBATCH --output=artifacts/stage_a_%j.log
set -euo pipefail
TOP="$HOME/frt_ccds"
OLD="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
JOB_TRES=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_TRES" == *"gres/gpu=1"* ]]
test -n "${CUDA_VISIBLE_DEVICES:-}"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
for name in frt_core.py stage_a_entry.py frt_records.py allocation_guard.py; do
  cp "$TOP/control/$name" "$OUT/$name"
done
cp "$TOP/control/manifest_a.json" "$OUT/manifest.json"
export PYTHONPATH="$OUT:$OLD/control:$OLD/modelroot/source"
export TORCH_HOME="$OLD/cache/torch" MPLBACKEND=Agg WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$OLD/venv/bin/python" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
timeout --signal=TERM --kill-after=15s 10m "$OLD/venv/bin/python" "$OUT/frt_records.py" --stage a --root "$OLD/modelroot" --manifest "$OUT/manifest.json" --output "$OUT/records" --helper-dir "$OLD/control"
timeout --signal=TERM --kill-after=15s 17m "$OLD/venv/bin/python" "$OUT/stage_a_entry.py" --stage a --root "$OLD/modelroot" --output "$OUT" --manifest "$OUT/records/manifest.json" --run-id "frt-a-$SLURM_JOB_ID"

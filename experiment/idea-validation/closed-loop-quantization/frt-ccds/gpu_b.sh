#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=frt_stage_b
#SBATCH --output=artifacts/stage_b_%j.log
set -euo pipefail
TOP="$HOME/frt_ccds"
OLD="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
JOB_TRES=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_TRES" == *"gres/gpu=1"* ]]
test -n "${CUDA_VISIBLE_DEVICES:-}"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
for name in frt_core.py frt_stage_b.py allocation_guard.py check_b_preflight.py; do
  cp "$TOP/control/$name" "$OUT/$name"
done
cp "$TOP/control/stage_b_runner.py" "$OUT/frt_runner.py"
cp "$TOP/control/manifest_b.json" "$OUT/manifest.json"
cp "$TOP/control/manifest_b_preflight.json" "$OUT/manifest_preflight.json"
export PYTHONPATH="$OUT:$OLD/control:$OLD/modelroot/source"
export TORCH_HOME="$OLD/cache/torch" MPLBACKEND=Agg WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$OLD/venv/bin/python" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
timeout --signal=TERM --kill-after=15s 3m "$OLD/venv/bin/python" "$OUT/frt_runner.py" --stage b --root "$OLD/modelroot" --output "$OUT/preflight" --manifest "$OUT/manifest_preflight.json" --a-summary "$TOP/artifacts/64688/summary.json" --run-id "frt-b-preflight-$SLURM_JOB_ID"
"$OLD/venv/bin/python" "$OUT/check_b_preflight.py" "$OUT/preflight"
timeout --signal=TERM --kill-after=15s 54m "$OLD/venv/bin/python" "$OUT/frt_runner.py" --stage b --root "$OLD/modelroot" --output "$OUT" --manifest "$OUT/manifest.json" --a-summary "$TOP/artifacts/64688/summary.json" --run-id "frt-b-$SLURM_JOB_ID"

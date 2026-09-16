#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=prr_engineering
#SBATCH --output=artifacts/engineering_%j.log
set -euo pipefail
TOP="$HOME/prr_ccds"
OLD="$HOME/cem_update_ccds"
FRT="$HOME/frt_ccds/artifacts/64694"
source "$TOP/control/allocation_guard.sh"
JOB_TRES=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_TRES" == *"gres/gpu=1"* ]]
test -n "${CUDA_VISIBLE_DEVICES:-}"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
for name in prr_quant.py prr_runner.py prr_records.py allocation_guard.py; do
  cp "$TOP/control/$name" "$OUT/$name"
done
cp "$TOP/control/manifest_engineering.json" "$OUT/manifest.json"
export PYTHONPATH="$OUT:$FRT:$HOME/frt_ccds/artifacts/64688:$OLD/control:$OLD/modelroot/source"
export TORCH_HOME="$OLD/cache/torch" MPLBACKEND=Agg WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$OLD/venv/bin/python" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
timeout --signal=TERM --kill-after=15s 10m "$OLD/venv/bin/python" "$OUT/prr_records.py" --stage smoke --root "$OLD/modelroot" --manifest "$OUT/manifest.json" --output "$OUT/records" --helper-dir "$OLD/control"
timeout --signal=TERM --kill-after=15s 17m "$OLD/venv/bin/python" "$OUT/prr_runner.py" --stage engineering --root "$OLD/modelroot" --manifest "$OUT/records/manifest.json" --output "$OUT"

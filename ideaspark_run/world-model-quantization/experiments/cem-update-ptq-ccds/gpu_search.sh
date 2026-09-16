#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:16:30
#SBATCH --job-name=cem_v100_search
#SBATCH --output=artifacts/search_%j.log
set -euo pipefail
TOP="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
ROOT="$TOP/modelroot"
BASE="$TOP/run"
test -f "$TOP/prepare_summary.json"
test -f "$BASE/pools/newcal.pkl"
test -f "$BASE/pools/newdev.pkl"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
cp "$TOP/control"/*.py "$OUT/"
cp "$TOP/control/initial_scoreerror.json" "$TOP/control/old_target_fingerprints.json" "$TOP/control/manifest.json" "$TOP/control/BUDGET_FREEZE.json" "$OUT/"
export PYTHONPATH="$ROOT/source" TORCH_HOME="$TOP/cache/torch" MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
PY="$TOP/venv/bin/python"
"$PY" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
timeout --signal=TERM --kill-after=15s 4530s "$PY" "$OUT/joint_search.py" --root "$ROOT" --cal "$BASE/pools/newcal.pkl" --dev "$BASE/pools/newdev.pkl" --initial "$OUT/initial_scoreerror.json" --output "$OUT" --rounds 2 --max-seconds 4500

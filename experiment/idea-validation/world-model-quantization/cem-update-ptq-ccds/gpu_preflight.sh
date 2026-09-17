#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=cem_v100_preflight
#SBATCH --output=artifacts/preflight_%j.log
set -euo pipefail
TOP="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
test -f "$TOP/prepare_summary.json"
ROOT="$TOP/modelroot"
BASE="$TOP/run"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
cp "$TOP/control"/*.py "$OUT/"
cp "$TOP/control/initial_scoreerror.json" "$TOP/control/old_target_fingerprints.json" "$OUT/"
cp "$TOP/control/manifest.json" "$OUT/manifest.json"
export PYTHONPATH="$ROOT/source" TORCH_HOME="$TOP/cache/torch" MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
PY="$TOP/venv/bin/python"
"$PY" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" "$OUT/joint_search.py" --self-test
"$PY" "$OUT/verify_stage_b.py" --self-test
timeout --signal=TERM --kill-after=15s 18m "$PY" "$OUT/prepare_pools.py" --root "$ROOT" --base "$BASE" --output "$OUT/collection" --old-fingerprints "$OUT/old_target_fingerprints.json"
timeout --signal=TERM --kill-after=15s 10m "$PY" "$OUT/joint_search.py" --root "$ROOT" --cal "$BASE/pools/newcal.pkl" --dev "$BASE/pools/newdev.pkl" --initial "$OUT/initial_scoreerror.json" --output "$OUT/benchmark" --benchmark-only --max-seconds 540
"$PY" "$OUT/estimate_budget.py" --benchmark "$OUT/benchmark/benchmark_summary.json"

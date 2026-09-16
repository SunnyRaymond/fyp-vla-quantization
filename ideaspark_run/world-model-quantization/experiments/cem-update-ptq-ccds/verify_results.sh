#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=cem_cpu_verify
#SBATCH --output=artifacts/verify_%j.log
set -euo pipefail
TOP="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
PY="$TOP/venv/bin/python"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
cp "$TOP/control/verify_stage_b.py" "$TOP/control/allocation_guard.py" "$TOP/control/old_target_fingerprints.json" "$TOP/control/search_job.json" "$OUT/"
SEARCH_JOB=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$OUT/search_job.json")
[[ "$SEARCH_JOB" =~ ^[0-9]+$ ]]
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$PY" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
"$PY" "$OUT/verify_stage_b.py" --self-test
timeout --signal=TERM --kill-after=15s 8m "$PY" "$OUT/verify_stage_b.py" --base "$TOP/run" --search "$TOP/artifacts/$SEARCH_JOB" --output "$OUT/verification.json" --old-fingerprints "$OUT/old_target_fingerprints.json"

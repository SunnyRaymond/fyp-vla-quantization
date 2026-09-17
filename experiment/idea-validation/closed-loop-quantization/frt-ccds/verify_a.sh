#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=frt_verify_a
#SBATCH --output=artifacts/verify_a_%j.log
set -euo pipefail
TOP="$HOME/frt_ccds"
OLD="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
cp "$TOP/control/verify_a_entry.py" "$OUT/verify_frt.py"
cp "$TOP/control/allocation_guard.py" "$OUT/allocation_guard.py"
export PYTHONPATH="$OUT:$OLD/control:$OLD/modelroot/source"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$OLD/venv/bin/python" "$OUT/verify_frt.py" --stage a --artifact-dir "$TOP/artifacts/64688" --manifest "$TOP/artifacts/64688/records/manifest.json" --output "$OUT/verification_a.json"

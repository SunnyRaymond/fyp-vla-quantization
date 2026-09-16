#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=frt_verify_b
#SBATCH --dependency=afterok:64694
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=artifacts/verify_b_%j.log
set -euo pipefail
TOP="$HOME/frt_ccds"
OLD="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
for name in verify_b_driver.py verify_b_helpers.py allocation_guard.py; do
  cp "$TOP/control/$name" "$OUT/$name"
done
export PYTHONPATH="$OUT:$OLD/control:$OLD/modelroot/source"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$OLD/venv/bin/python" "$OUT/verify_b_driver.py" "$TOP/artifacts/64694" "$TOP/artifacts/64689/verification_a.json" "$OUT/verification_b.json"

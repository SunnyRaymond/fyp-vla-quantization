#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --dependency=afterany:64706
#SBATCH --job-name=prr_verify_r1
#SBATCH --output=artifacts/verify_%j.log
set -euo pipefail
TOP="$HOME/prr_ccds"
OLD="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
for name in verify_prr.py allocation_guard.py manifest_template.json; do
  cp "$TOP/control/$name" "$OUT/$name"
done
export PYTHONPATH="$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
"$OLD/venv/bin/python" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
sacct -nP -j 64705,64706 --format=JobID,State,ExitCode,ElapsedRaw,AllocTRES > "$OUT/gpu_accounting.txt"
timeout --signal=TERM --kill-after=15s 8m "$OLD/venv/bin/python" "$OUT/verify_prr.py" --artifact-dir "$TOP/artifacts/64706" --manifest "$TOP/artifacts/64706/records/manifest.json" --output "$OUT/verification.json"

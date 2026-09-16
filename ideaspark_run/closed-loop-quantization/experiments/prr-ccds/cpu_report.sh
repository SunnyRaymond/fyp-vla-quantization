#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=prr_report
#SBATCH --output=artifacts/report_%j.log
set -euo pipefail
TOP="$HOME/prr_ccds"
source "$TOP/control/allocation_guard.sh"
OUT="$TOP/artifacts/$SLURM_JOB_ID"
mkdir "$OUT"
cp "$TOP/control/report_metrics.py" "$TOP/control/allocation_guard.py" "$OUT/"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
"$HOME/cem_update_ccds/venv/bin/python" "$OUT/report_metrics.py"

#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=frt_cpu_reuse
#SBATCH --output=artifacts/resources_%j.log
set -euo pipefail
TOP="$HOME/frt_ccds"
source "$TOP/control/allocation_guard.sh"
OLD="$HOME/cem_update_ccds"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$TOP/control:$OLD/modelroot/source"
"$OLD/venv/bin/python" "$TOP/control/inspect_resources.py" --base "$OLD" --output "$TOP/artifacts/resources_$SLURM_JOB_ID.json"

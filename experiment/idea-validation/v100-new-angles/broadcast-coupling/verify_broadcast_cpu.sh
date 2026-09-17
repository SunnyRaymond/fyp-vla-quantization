#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=broadcast_verify
#SBATCH --output=artifacts/broadcast_verify_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
OLD="${HOME}/cem_update_ccds"
CONTROL="$TOP/control"
source "$CONTROL/allocation_guard.sh"
OUT="$TOP/artifacts/${SLURM_JOB_ID}"
INPUT="$TOP/artifacts/64838"
test ! -e "$OUT"
mkdir -p "$OUT"
cp "$CONTROL/verify_broadcast.py" "$CONTROL/broadcast_replay.py" "$CONTROL/allocation_guard.py" "$CONTROL/verify_broadcast_cpu.sh" "$OUT/"
exec > >(tee -a "$OUT/run.log") 2>&1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$OUT:$CONTROL"
timeout --signal=TERM --kill-after=10s 285s "$OLD/venv/bin/python" "$OUT/verify_broadcast.py" \
 --input "$INPUT" --output "$OUT" --replay "$OUT/broadcast_replay.py"

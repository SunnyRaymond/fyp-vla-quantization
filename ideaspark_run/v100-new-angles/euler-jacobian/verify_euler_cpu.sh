#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=euler_verify
#SBATCH --output=artifacts/euler_verify_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
CONTROL="$TOP/control"
source "$CONTROL/allocation_guard.sh"
OUT="$TOP/artifacts/${SLURM_JOB_ID}"
INPUT="$TOP/artifacts/64835"
test ! -e "$OUT"
mkdir -p "$OUT"
cp "$CONTROL/verify_euler.py" "$CONTROL/euler_replay.py" "$CONTROL/allocation_guard.py" "$CONTROL/verify_euler_cpu.sh" "$OUT/"
exec > >(tee -a "$OUT/run.log") 2>&1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$OUT:$CONTROL"
timeout --signal=TERM --kill-after=10s 285s "$TOP/smolvla/venv/bin/python" "$OUT/verify_euler.py" \
 --input "$INPUT" --output "$OUT" --replay "$OUT/euler_replay.py"

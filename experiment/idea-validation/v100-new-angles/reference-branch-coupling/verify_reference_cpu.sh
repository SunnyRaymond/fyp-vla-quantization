#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=verify_reference
#SBATCH --output=artifacts/verify_reference_%j.log
set -euo pipefail

INPUT="${1:-${INPUT:-}}"
OUTPUT="${2:-${OUTPUT:-}}"
TOP="${TOP:-$HOME/v100_newangles_ccds}"
CONTROL="${CONTROL:-$TOP/control}"
PYTHON="${PYTHON:-$HOME/cem_update_ccds/venv/bin/python}"

# Keep the login-node controller separate: this guard must pass on the real
# SLURM compute node before Python reads producer artifacts.
source "$CONTROL/allocation_guard.sh"
INPUT="${INPUT:-$TOP/artifacts/64840}"
OUTPUT="${OUTPUT:-$TOP/artifacts/$SLURM_JOB_ID}"
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/run.log") 2>&1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Snapshot only the small verifier inputs into this CPU job's own artifact
# directory.  The producer raw arrays and checkpoint remain in the producer
# artifact directory and are read only after its complete status is checked.
for name in verify_reference.py verify_reference_cpu.sh reference_replay.py RAW_CONTRACT.json allocation_guard.py allocation_guard.sh; do
  cp -- "$CONTROL/$name" "$OUTPUT/$name"
done
export PYTHONPATH="$OUTPUT:$CONTROL:$TOP:${PYTHONPATH:-}"

timeout --signal=TERM --kill-after=10s 285s "$PYTHON" \
  "$OUTPUT/verify_reference.py" \
  --input "$INPUT" --output "$OUTPUT" \
  --replay "$OUTPUT/reference_replay.py" --contract "$OUTPUT/RAW_CONTRACT.json"

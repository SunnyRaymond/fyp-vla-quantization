#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=broadcast_coupling
#SBATCH --output=artifacts/broadcast_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
OLD="${HOME}/cem_update_ccds"
ROOT="$OLD/modelroot"
CONTROL="$TOP/control"
source "$CONTROL/allocation_guard.sh"
OUT="$TOP/artifacts/${SLURM_JOB_ID}"
test ! -e "$OUT"
mkdir -p "$OUT"
for f in broadcast_screen.py broadcast_gpu.sh broadcast_protocol.zh.md broadcast_input_freeze.json teacher_bias_screen.py allocation_guard.py allocation_guard.sh; do
 cp "$CONTROL/$f" "$OUT/"
done
cp "$OLD/control/smoke_runner.py" "$OLD/control/screen_runner.py" "$OUT/"
exec > >(tee -a "$OUT/run.log") 2>&1
export PYTHONPATH="$OUT:$OLD/control:$ROOT/source"
export TORCH_HOME="$OLD/cache/torch" MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$OLD/venv/bin/python"
"$PY" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
set +e
timeout --signal=TERM --kill-after=15s 270s "$PY" "$OUT/broadcast_screen.py" \
 --root "$ROOT" --input-manifest "$TOP/teacher_bias_ready2/manifest.json" \
 --teacher-helper "$OUT/teacher_bias_screen.py" --helper-dir "$OUT" \
 --protocol "$OUT/broadcast_protocol.zh.md" --freeze "$OUT/broadcast_input_freeze.json" --output "$OUT"
code=$?
printf '{"exit_code":%s}\n' "$code" > "$OUT/exit_status.json"
exit "$code"

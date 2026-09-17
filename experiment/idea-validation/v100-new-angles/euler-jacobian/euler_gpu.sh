#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=euler_jacobian
#SBATCH --output=artifacts/euler_jacobian_%j.log
set -euo pipefail
TOP="${HOME}/v100_newangles_ccds"
CONTROL="${TOP}/control"
source "${CONTROL}/allocation_guard.sh"
OUT="${TOP}/artifacts/${SLURM_JOB_ID}"
test ! -e "$OUT"
mkdir -p "$OUT"
for f in euler_screen.py euler_protocol.zh.md euler_input_freeze.json persistence_batch2_screen.py flow_screen.py allocation_guard.py allocation_guard.sh euler_gpu.sh; do
  cp "${CONTROL}/${f}" "$OUT/"
done
exec > >(tee -a "$OUT/run.log") 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${OUT}:${CONTROL}"
PY="${TOP}/smolvla/venv/bin/python"
"$PY" "$OUT/allocation_guard.py" > "$OUT/allocation.json"
set +e
timeout --signal=TERM --kill-after=15s 540s "$PY" "$OUT/euler_screen.py" \
 --input-manifest "$TOP/rounding_persistence_ready/manifest.json" \
 --model-path "$TOP/smolvla/model" --checkpoint "$TOP/smolvla/model/model.safetensors" \
 --vlm-path "$TOP/smolvla/base_vlm_metadata" --flow-helper "$OUT/flow_screen.py" \
 --persistence-helper "$OUT/persistence_batch2_screen.py" --protocol "$OUT/euler_protocol.zh.md" \
 --freeze "$OUT/euler_input_freeze.json" --output "$OUT"
code=$?
printf '{"exit_code":%s}\n' "$code" > "$OUT/exit_status.json"
exit "$code"

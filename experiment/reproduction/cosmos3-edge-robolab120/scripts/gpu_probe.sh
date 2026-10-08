#!/usr/bin/env bash
#SBATCH --job-name=cosmos3_gpu_probe
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --account=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --output=/tc1home/UG/yguo017/cosmos3-edge-robolab120/logs/%x-%j.log
#SBATCH --error=/tc1home/UG/yguo017/cosmos3-edge-robolab120/logs/%x-%j.log
set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: not in a SLURM allocation' >&2; exit 10; }
[[ "${SLURM_JOB_PARTITION:-}" == 'UGGPU-TC1' ]] || { echo 'ERROR: unexpected partition' >&2; exit 11; }
node="${SLURMD_NODENAME:-$(hostname -s)}"
[[ "$node" == TC1N* ]] || { echo 'ERROR: not on an approved compute node' >&2; exit 12; }
visible="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "$visible" && "$visible" != *,* ]] || { echo 'ERROR: expected exactly one scheduler-visible GPU' >&2; exit 13; }

printf 'job=%s node=%s partition=%s\n' "$SLURM_JOB_ID" "$node" "$SLURM_JOB_PARTITION"
printf 'gpu model / memory / driver: '
nvidia-smi -i "$visible" --query-gpu=name,memory.total,driver_version --format=csv,noheader
for sample in {1..10}; do
  printf 'sample=%s utc=%s ' "$sample" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi -i "$visible" --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  sleep 5
done

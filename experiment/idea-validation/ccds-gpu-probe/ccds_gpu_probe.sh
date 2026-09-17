#!/bin/bash
# Minimal CCDS TC1 GPU identity probe; no Python, model, package install, or benchmark.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=5
#SBATCH --job-name=ccds_gpu_probe
#SBATCH --output=ccds_gpu_probe_%j.out
#SBATCH --error=ccds_gpu_probe_%j.err

set -euo pipefail
printf '=== HOST ===\n'
hostname -f
date -Is
printf '\n=== GPU ===\n'
nvidia-smi -L
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,memory.used --format=csv,noheader
printf '\n=== ALLOCATION ===\n'
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nCUDA_VISIBLE_DEVICES=%s\n' "${SLURM_JOB_ID:-}" "${SLURM_JOB_NODELIST:-}" "${CUDA_VISIBLE_DEVICES:-}"

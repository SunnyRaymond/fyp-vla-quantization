#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=cem_cpu_probe
#SBATCH --output=artifacts/probe_%j.log
set -euo pipefail
BASE="$HOME/cem_update_ccds"
source "$BASE/control/allocation_guard.sh"
hostname
module avail 2>&1
ls -d /tc1apps/2_conda_env/*
ls -d "$HOME"/.conda/envs/* "$HOME"/miniconda* "$HOME"/anaconda* 2>/dev/null || true
df -h "$HOME"
curl -ILsS -o /dev/null -w 'GitHub HTTP %{http_code}\n' --max-time 20 https://github.com/gaoyuezhou/dino_wm
curl -ILsS -o /dev/null -w 'OSF HTTP %{http_code}\n' --max-time 20 https://osf.io/download/xvzs4/

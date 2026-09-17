#!/usr/bin/env bash
set -euo pipefail
case "${PBS_JOBID:-}" in
  '') echo 'Refusing probe: PBS_JOBID is missing.' >&2; exit 64 ;;
esac
case "$(hostname -s)" in
  *login*|asp2a-login*) echo 'Refusing probe: login node is not a compute allocation.' >&2; exit 64 ;;
esac
source /etc/profile
module load apptainer
container=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
apptainer exec "$container" sh -lc 'sed -n "1,200p" /app/libero/requirements.txt; printf "--- setup ---\n"; sed -n "1,160p" /app/libero/setup.py'

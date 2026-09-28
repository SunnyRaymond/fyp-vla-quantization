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
apptainer exec "$container" sh -lc 'find /app/libero -maxdepth 3 -type f | sort | head -100'

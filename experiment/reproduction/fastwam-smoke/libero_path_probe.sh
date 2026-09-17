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
apptainer exec "$container" sh -lc 'printf "%s\n" "--- config locations ---"; find /root /app/libero -maxdepth 3 -path "*/.libero/config.yaml" -o -name config.yaml 2>/dev/null | head -20; printf "%s\n" "--- config ---"; for f in /root/.libero/config.yaml /home/*/.libero/config.yaml; do [ -f "$f" ] && { echo "$f"; sed -n "1,120p" "$f"; }; done'

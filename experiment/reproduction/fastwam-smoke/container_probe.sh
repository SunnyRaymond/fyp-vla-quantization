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
apptainer exec "$container" sh -lc 'printf "root paths:\n"; ls -la /app 2>&1 | head -50; printf "python paths:\n"; python - <<"PY"
import sys
print("\n".join(sys.path))
PY
printf "installed packages:\n"; python -m pip list 2>&1 | head -80'

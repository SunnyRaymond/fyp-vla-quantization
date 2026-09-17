#!/usr/bin/env bash
set -euo pipefail
case "${PBS_JOBID:-}" in
  '') echo 'Refusing quarantine move: PBS_JOBID is missing.' >&2; exit 64 ;;
esac
case "$(hostname -s)" in
  *login*|asp2a-login*) echo 'Refusing quarantine move: login node is not a compute allocation.' >&2; exit 64 ;;
esac
base=/scratch/users/ntu/yguo017/fastwam-smoke/checkpoints
if [ -f "$base/libero_optional_idm_2cam224.pt" ]; then
  mv "$base/libero_optional_idm_2cam224.pt" "$base/libero_optional_idm_2cam224.concurrent-download.quarantine.pt"
fi
ls -lh "$base"

#!/usr/bin/env bash
set -euo pipefail
case "${PBS_JOBID:-}" in
  '') echo 'Refusing search: PBS_JOBID is missing.' >&2; exit 64 ;;
esac
case "$(hostname -s)" in
  *login*|asp2a-login*) echo 'Refusing search: login node is not a compute allocation.' >&2; exit 64 ;;
esac
find /scratch/users/ntu/yguo017 -type f -size +100M -printf '%s %p\n' 2>/dev/null | grep -E 'Wan2\.2_VAE|models_t5|\.safetensors' | head -100

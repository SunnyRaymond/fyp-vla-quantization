#!/usr/bin/env bash
set -euo pipefail
case "${PBS_JOBID:-}" in
  '') echo 'Refusing probe: PBS_JOBID is missing.' >&2; exit 64 ;;
esac
case "$(hostname -s)" in
  *login*|asp2a-login*) echo 'Refusing probe: login node is not a compute allocation.' >&2; exit 64 ;;
esac
source /etc/profile
module load python/3.10.4
printf 'python='; python --version
printf 'pip='; python -m pip --version
python - <<'PY'
import importlib.util
for name in ['torch','torchvision','mujoco','transformers','hydra','av','imageio','einops']:
    print(f'{name}={bool(importlib.util.find_spec(name))}')
PY

#!/usr/bin/env bash
set -euo pipefail
test -n "${PBS_JOBID:-}" || { echo 'Refusing container probe: PBS_JOBID is missing.' >&2; exit 64; }
case "$(hostname -s)" in
  *login*|asp2a-login*) echo 'Refusing container probe: login node is not a compute allocation.' >&2; exit 64 ;;
esac
source /etc/profile
module load apptainer
container=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
printf 'apptainer='; apptainer --version
printf 'container_python='; apptainer exec "$container" python --version
apptainer exec "$container" python - <<'PY'
import importlib.util
modules = ['torch', 'torchvision', 'libero', 'mujoco', 'transformers', 'diffsynth', 'hydra', 'av']
for name in modules:
    print(f'{name}={bool(importlib.util.find_spec(name))}')
PY

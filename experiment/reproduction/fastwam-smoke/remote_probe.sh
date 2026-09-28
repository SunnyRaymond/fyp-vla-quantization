#!/usr/bin/env bash
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

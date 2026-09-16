#!/usr/bin/env bash
set -euo pipefail
case "${PBS_JOBID:-}" in
  '') echo 'Refusing environment preparation: PBS_JOBID is missing.' >&2; exit 64 ;;
esac
case "$(hostname -s)" in
  *login*|asp2a-login*) echo 'Refusing environment preparation: login node is not a compute allocation.' >&2; exit 64 ;;
esac
source /etc/profile
module load python/3.10.4
venv=/scratch/users/ntu/yguo017/fastwam-smoke/venv
if [ ! -x "$venv/bin/python" ]; then
  python -m venv "$venv"
fi
"$venv/bin/python" --version
"$venv/bin/python" -m pip install --upgrade pip setuptools wheel
echo "venv=$venv"

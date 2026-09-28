#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/rolling-ball-lewm
[[ "$(hostname)" == autodl-container-e7e742ba1d-c91a1edb && -z "${PBS_JOBID:-}" ]] || exit 1
PYTHON="$ROOT/envs/sim/bin/python"
LAB="$ROOT/src/IsaacLab-v2.3.1"
OUT="$ROOT/results/cpu-lab-core"
[[ -x "$PYTHON" && -f "$LAB/source/isaaclab/setup.py" && ! -e "$OUT" ]] || exit 1
mkdir "$OUT"
exec > >(tee "$OUT/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"; exit "$rc"' EXIT
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" PIP_NO_CACHE_DIR=0 PIP_DISABLE_PIP_VERSION_CHECK=1
printf 'torch==2.7.0+cu128\nnumpy==1.26.0\n' > "$OUT/constraints.txt"
# flatdict 4.0.1 imports pkg_resources, removed by newer setuptools.
"$PYTHON" -m pip install 'setuptools==80.9.0'
"$PYTHON" -m pip install --no-build-isolation 'flatdict==4.0.1'
"$PYTHON" -m pip install --constraint "$OUT/constraints.txt" -e "$LAB/source/isaaclab"
"$PYTHON" - "$OUT/ready.json" <<'PY'
import importlib.metadata as m, json, sys
from pathlib import Path
names=('isaaclab','isaaclab-assets','isaaclab-tasks','isaaclab-rl','isaaclab-mimic','flatdict','torch','numpy')
versions={n:m.version(n) for n in names}
assert versions['isaaclab']=='0.48.0'
assert versions['torch']=='2.7.0+cu128' and versions['numpy']=='1.26.0'
report={'status':'PASS_LAB_CORE_PACKAGES','versions':versions,'runtime_verified':False,'checkpoint_loaded':False}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY

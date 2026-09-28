#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/rolling-ball-lewm
BUNDLE="$ROOT/bundle/rolling-ball-lewm-epoch100"
[[ "$(hostname)" == autodl-container-e7e742ba1d-c91a1edb && -z "${PBS_JOBID:-}" ]] || exit 1
[[ -s "$ROOT/install_record.txt" && -s "$BUNDLE/BUNDLE.json" ]] || exit 1
export TMPDIR="$ROOT/tmp" PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
OUT="$ROOT/results/cpu-packages"
[[ ! -e "$OUT" ]] || { printf 'Refusing to replace CPU package evidence\n'; exit 1; }
mkdir "$OUT"
exec > >(tee "$OUT/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"; exit "$rc"' EXIT
"$ROOT/envs/sim/bin/python" -c 'import importlib.metadata as m; assert m.version("isaaclab") == "0.48.0"'
"$ROOT/envs/policy/bin/python" - "$BUNDLE" "$OUT/constraints.txt" <<'PY'
import json, sys
from pathlib import Path
versions=json.loads((Path(sys.argv[1])/'package_versions.json').read_text())['policy_environment_observed_during_export']
Path(sys.argv[2]).write_text(''.join(f'{name}=={version}\n' for name,version in versions.items()))
PY
"$ROOT/envs/policy/bin/python" -m pip install --no-cache-dir \
  --constraint "$OUT/constraints.txt" --find-links "$ROOT/cache/wheels" \
  -e "$BUNDLE/upstream_stablewm" 'datasets==5.0.1' 'opencv-python-headless==4.11.0.86' imageio
"$ROOT/envs/sim/bin/python" -m pip install --no-cache-dir -e "$BUNDLE/reflexbench/source/reflexbench"
"$ROOT/envs/sim/bin/python" - "$OUT/sim_versions.json" <<'PY'
import importlib.metadata as m, json, sys
from pathlib import Path
names=('isaacsim','isaaclab','reflexbench','torch','numpy')
Path(sys.argv[1]).write_text(json.dumps({n:m.version(n) for n in names},indent=2)+'\n')
PY
"$ROOT/envs/policy/bin/python" - "$BUNDLE" "$OUT/ready.json" <<'PY'
import importlib, importlib.metadata as m, json, sys
from pathlib import Path
bundle=Path(sys.argv[1])
sys.path[:0]=[str(bundle/'upstream_lewm'),str(bundle/'upstream_stablewm')]
for name in ('jepa','utils','stable_worldmodel.solver.cem'):
    importlib.import_module(name)
expected=json.loads((bundle/'package_versions.json').read_text())['policy_environment_observed_during_export']
actual={name:m.version(name) for name in expected}
assert actual==expected, {'expected':expected,'actual':actual}
report={'status':'PASS_CPU_POLICY_IMPORTS','policy_versions':actual,
        'sim_versions':json.loads((Path(sys.argv[2]).parent/'sim_versions.json').read_text()),
        'imported_modules':['jepa','utils','stable_worldmodel.solver.cem'],
        'checkpoint_loaded':False,'explicit_cuda_calls':False,'closed_loop_evaluated':False}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY

#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/autodl-tmp/rolling-ball-lewm
BUNDLE="$ROOT/bundle/rolling-ball-lewm-epoch100"
[[ "$(hostname)" == autodl-container-e7e742ba1d-c91a1edb && -z "${PBS_JOBID:-}" ]] || exit 1
PRIOR="$ROOT/results/cpu-packages"
[[ "$(cat "$PRIOR/exit_code")" == 1 && -s "$PRIOR/constraints.txt" && -s "$PRIOR/sim_versions.json" ]] || exit 1
OUT="$ROOT/results/cpu-policy-import-repair"
[[ ! -e "$OUT" ]] || exit 1
mkdir "$OUT"
exec > >(tee "$OUT/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"; exit "$rc"' EXIT
export TMPDIR="$ROOT/tmp" PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$ROOT/envs/policy/bin/python" -m pip install --constraint "$PRIOR/constraints.txt" 'datasets==5.0.1'
"$ROOT/envs/policy/bin/python" - "$BUNDLE" "$PRIOR" "$OUT/ready.json" <<'PY'
import importlib, importlib.metadata as m, json, sys
from pathlib import Path
bundle, prior, output = map(Path, sys.argv[1:])
sys.path[:0] = [str(bundle/'upstream_lewm'), str(bundle/'upstream_stablewm')]
modules = ('jepa', 'utils', 'stable_worldmodel.solver.cem')
for name in modules:
    importlib.import_module(name)
expected = json.loads((bundle/'package_versions.json').read_text())['policy_environment_observed_during_export']
actual = {name: m.version(name) for name in expected}
assert actual == expected, {'expected': expected, 'actual': actual}
report = {'status': 'PASS_CPU_POLICY_IMPORTS', 'policy_versions': actual,
          'sim_versions': json.loads((prior/'sim_versions.json').read_text()),
          'repaired_transitive_versions': {name: m.version(name) for name in ('datasets', 'pyarrow')},
          'prior_failure': str(prior/'job.log'), 'imported_modules': list(modules),
          'checkpoint_loaded': False, 'explicit_cuda_calls': False, 'closed_loop_evaluated': False}
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
PY

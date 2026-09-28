#!/usr/bin/env bash
# Frozen baseline campaign; shutdown follows local retrieval and review, not this driver.
set -euo pipefail
R=/root/autodl-tmp/rolling-ball-lewm
H=autodl-container-59db4f87a9-11fcb48a
test "$(hostname)" = "$H"
OUT="$R/results/campaign-physical-001"
test ! -e "$OUT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/cache/xdg"
export XDG_DATA_HOME="$R/omniverse/data" XDG_CONFIG_HOME="$R/omniverse/config"
export XDG_RUNTIME_DIR="$R/tmp/xdg-runtime" OMNI_CONFIG_PATH="$R/omniverse/config"
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export CUDA_CACHE_PATH="$R/cache/cuda" TRITON_CACHE_DIR="$R/cache/triton"
export VK_ICD_FILENAMES="$R/controls/nvidia_headless_icd.json"
COMMON=(--rented-host --expected-host "$H"
  --reflexbench-root "$R/bundle/rolling-ball-lewm-epoch100/reflexbench"
  --asset-mirror "$R/assets/local" --policy-url http://127.0.0.1:8000
  --fixture-bank "$R/results/pair-physical-50-newhost-002/fixtures.pt"
  --pairing-protocol physical-v2 --num-episodes 50 --seed-base 2026092700)
validate_stage() {
  "$R/envs/policy/bin/python" - "$1" "$2" "$OUT/status.json" <<'PY'
import json,pathlib,sys
result_path=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; status_path=pathlib.Path(sys.argv[3])
d=json.loads(result_path.read_text())
counts=6 if mode=='fixed-delay' else 1
ok=(d.get('status')=='COMPLETED' and not d.get('error') and d.get('closed_loop_evaluated')
    and d.get('paired') and d.get('pairing_protocol')=='physical-v2'
    and not d.get('smoke_only') and len(d.get('conditions',[]))==counts
    and all(c['summary']['episodes']==50 for c in d.get('conditions',[])))
s=json.loads(status_path.read_text()) if status_path.exists() else {'stages':[]}
s['stages'].append({'mode':mode,'passed_completion_gate':bool(ok),'result':str(result_path),
                    'summaries':[c['summary'] for c in d.get('conditions',[])]})
s['status']='COMPLETED' if ok and mode=='true-async' else ('RUNNING' if ok else 'FAIL')
status_path.write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s['stages'][-1]),flush=True)
if not ok: raise SystemExit(65)
PY
}
for mode in sync fixed-delay true-async; do
  stage="$R/results/${mode}-50-physical-001"
  bash "$R/controls/run_rented.sh" "$H" "$R/results/${mode}-50-physical-wrapper-001" \
    "$R/envs/sim/bin/python" "$R/controls/eval_rolling.py" "${COMMON[@]}" \
    --mode "$mode" --output "$stage/result.json"
  validate_stage "$stage/result.json" "$mode"
done

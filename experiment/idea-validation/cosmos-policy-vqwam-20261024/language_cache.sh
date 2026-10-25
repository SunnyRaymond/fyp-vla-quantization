#!/bin/bash -l
set -euo pipefail

ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024
ASSET_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015
HOST=$(hostname -s)
[[ -n "${PBS_JOBID:-}" && -s "${PBS_NODEFILE:-}" && "${HOST,,}" != *login* ]] || {
  echo 'Missing PBS allocation or refusing login node' >&2
  exit 1
}
grep -Fqx "$HOST" <(awk -F. '{print $1}' "$PBS_NODEFILE") || {
  echo "Current host is outside PBS_NODEFILE: $HOST" >&2
  exit 1
}

ARTIFACTS=$ROOT/artifacts/$PBS_JOBID
mkdir -p "$ARTIFACTS/control_sources" "$ARTIFACTS/tmp"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
finish() {
  local rc=$? temporary
  trap - EXIT
  if [[ "$rc" == 0 && -s "$ROOT/results/language_cache/summary.json" ]]; then
    if python3 - "$ROOT/results/language_cache/summary.json" "$PBS_JOBID" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        value = json.load(stream)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if value.get("pbs_job_id") == sys.argv[2]
                 and value.get("status") == "complete"
                 and value.get("complete") is True else 1)
PY
    then
      touch "$ARTIFACTS/PIPELINE_COMPLETE"
    else
      rc=1
    fi
  fi
  temporary=$ARTIFACTS/exit_code.txt.tmp.$$
  printf '%s\n' "$rc" > "$temporary"
  mv -f "$temporary" "$ARTIFACTS/exit_code.txt"
  exit "$rc"
}
trap finish EXIT

QSTAT_OUTPUT=$(qstat -f "$PBS_JOBID") || { echo "Cannot query PBS job $PBS_JOBID" >&2; exit 1; }
JOB_STATE=$(awk -F= '/^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
JOB_NAME=$(awk -F= '/^[[:space:]]*Job_Name[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
EXECUTION_QUEUE=$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
if [[ "$JOB_STATE" != R || "$JOB_NAME" != cpvq24lang ]] || [[ "$EXECUTION_QUEUE" != qdev && "$EXECUTION_QUEUE" != q1 && "$EXECUTION_QUEUE" != q2 ]]; then
  echo "PBS identity mismatch: state=${JOB_STATE:-unknown} Job_Name=${JOB_NAME:-unknown} queue=${EXECUTION_QUEUE:-unknown}" >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'CPU-only language cache job unexpectedly has CUDA_VISIBLE_DEVICES' >&2
  exit 1
fi
echo "PBS_JOBID=$PBS_JOBID JOB_NAME=$JOB_NAME STATE=$JOB_STATE QUEUE=$EXECUTION_QUEUE HOST=$(hostname -f) CPU_ONLY=1"

module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0
for name in prepare_language_cache.py common.py protocol.json environment.sh; do
  source=$ROOT/$name
  [[ -s "$source" ]] || { echo "Missing language-cache source snapshot: $source" >&2; exit 1; }
  cp "$source" "$ARTIFACTS/control_sources/$name.tmp"
  mv -f "$ARTIFACTS/control_sources/$name.tmp" "$ARTIFACTS/control_sources/$name"
done

export COSMOS_ROOT=$ROOT COSMOS_CODE_ROOT=$ARTIFACTS/control_sources
export ASSET_ROOT LEGACY_ROOT=$ASSET_ROOT ARTIFACTS
export ARTIFACTS_TMP=$ARTIFACTS/tmp COSMOS_CPU_ONLY=1 COSMOS_CPU_THREADS=4
export COSMOS_EXECUTION_QUEUE=$EXECUTION_QUEUE APPTAINERENV_COSMOS_CPU_ONLY=1
export APPTAINERENV_COSMOS_EXECUTION_QUEUE=$EXECUTION_QUEUE
export COSMOS_JOB_START_EPOCH=$(date +%s) COSMOS_SOFT_DEADLINE_SECONDS=900
export COSMOS_HELPERS_PYTHONPATH=$ASSET_ROOT
source "$COSMOS_CODE_ROOT/environment.sh"
run_python "$COSMOS_CODE_ROOT/prepare_language_cache.py" \
  --protocol "$COSMOS_CODE_ROOT/protocol.json" --root "$ROOT" --asset-root "$ASSET_ROOT"

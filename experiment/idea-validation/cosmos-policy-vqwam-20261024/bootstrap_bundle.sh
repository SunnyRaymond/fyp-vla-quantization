#!/bin/bash -l
set -euo pipefail

ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024

if [[ -z "${PBS_JOBID:-}" || -z "${PBS_NODEFILE:-}" || ! -s "$PBS_NODEFILE" ]]; then
  echo 'Missing PBS_JOBID or PBS_NODEFILE; refusing bootstrap' >&2
  exit 1
fi
HOST=$(hostname -s)
if [[ "${HOST,,}" == *login* ]]; then
  echo "Refusing login node: $HOST" >&2
  exit 1
fi
if ! grep -Fqx "$HOST" <(awk -F. '{print $1}' "$PBS_NODEFILE"); then
  echo "Current host is outside PBS_NODEFILE: $HOST" >&2
  exit 1
fi

ARTIFACTS=$ROOT/artifacts/$PBS_JOBID
mkdir -p "$ARTIFACTS"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
finish() {
  local rc=$? temporary
  trap - EXIT
  temporary=$ARTIFACTS/exit_code.txt.tmp.$$
  printf '%s\n' "$rc" > "$temporary"
  mv -f "$temporary" "$ARTIFACTS/exit_code.txt"
  if [[ ! -e "$ARTIFACTS/PIPELINE_COMPLETE" && -s "$ROOT/results/bootstrap_bundle/summary.json" ]]; then
    if python3 - "$ROOT/results/bootstrap_bundle/summary.json" "$PBS_JOBID" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if value.get("pbs_job_id") == sys.argv[2]
                 and value.get("status") == "checkpointed" else 1)
PY
    then
      touch "$ARTIFACTS/PIPELINE_CHECKPOINTED"
    fi
  fi
  exit "$rc"
}
trap finish EXIT

QSTAT_OUTPUT=$(qstat -f "$PBS_JOBID") || { echo "Cannot query PBS job $PBS_JOBID" >&2; exit 1; }
JOB_STATE=$(awk -F= '/^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
JOB_NAME=$(awk -F= '/^[[:space:]]*Job_Name[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
EXECUTION_QUEUE=$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
if [[ "$JOB_STATE" != R || "$JOB_NAME" != cpvq24b2 || "$EXECUTION_QUEUE" != g2 ]]; then
  echo "PBS identity mismatch: state=${JOB_STATE:-unknown} Job_Name=${JOB_NAME:-unknown} queue=${EXECUTION_QUEUE:-unknown}" >&2
  exit 1
fi

echo "PBS_JOBID=$PBS_JOBID JOB_NAME=$JOB_NAME STATE=$JOB_STATE QUEUE=$EXECUTION_QUEUE HOST=$(hostname -f)"
module purge
module load gcc/11.4.0-nscc
module load python/3.11.7-gcc11
module load apptainer/1.5.0
module load cuda/12.2.2
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is unavailable' >&2; exit 1; }
python3 -c 'import sys; print("PYTHON=" + sys.executable + " VERSION=" + sys.version.split()[0]); sys.exit(0 if sys.version_info[:3] == (3, 11, 7) else 1)'
nvidia-smi -L

CONTROL_SOURCES=$ARTIFACTS/control_sources
mkdir -p "$CONTROL_SOURCES"
for name in bootstrap_bundle.py bootstrap_race.py common.py protocol.json; do
  source=$ROOT/$name
  [[ -s "$source" ]] || { echo "Missing bootstrap control source: $source" >&2; exit 1; }
  cp "$source" "$CONTROL_SOURCES/$name.tmp"
  mv -f "$CONTROL_SOURCES/$name.tmp" "$CONTROL_SOURCES/$name"
done

export COSMOS_ROOT=$ROOT
set +e
python3 "$CONTROL_SOURCES/bootstrap_bundle.py"
rc=$?
set -e
exit "$rc"

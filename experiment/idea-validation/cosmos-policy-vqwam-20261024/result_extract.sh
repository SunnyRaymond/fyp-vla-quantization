#!/bin/bash -l
set -euo pipefail

ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024
HOST=$(hostname -s)
[[ -n "${PBS_JOBID:-}" && -s "${PBS_NODEFILE:-}" && "${HOST,,}" != *login* ]] || {
  echo 'Missing native PBS allocation or refusing login host' >&2
  exit 64
}
grep -Fqx "$HOST" <(awk -F. '{print $1}' "$PBS_NODEFILE") || {
  echo "Current host is absent from PBS_NODEFILE: $HOST" >&2
  exit 64
}
QSTAT_OUTPUT=$(qstat -f "$PBS_JOBID") || { echo 'Cannot query current extraction job' >&2; exit 64; }
JOB_STATE=$(awk -F= '/^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
JOB_NAME=$(awk -F= '/^[[:space:]]*Job_Name[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
EXECUTION_QUEUE=$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$QSTAT_OUTPUT")
[[ "$JOB_STATE" == R && ( "$JOB_NAME" == cpvq24extract || "$JOB_NAME" == cpvq24grip ) ]] || { echo 'Current extraction job is not an authorized running CPU result job' >&2; exit 64; }
[[ "$EXECUTION_QUEUE" == qdev || "$EXECUTION_QUEUE" == q1 || "$EXECUTION_QUEUE" == q2 ]] || {
  echo "Extraction is outside approved CPU queues: $EXECUTION_QUEUE" >&2; exit 64;
}
if grep -Eq 'Resource_List\.ngpus[[:space:]]*=[[:space:]]*[1-9]' <<<"$QSTAT_OUTPUT" \
    || grep -Eq 'ngpus[[:space:]]*=[[:space:]]*[1-9][0-9]*' <<<"$QSTAT_OUTPUT" \
    || [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Refusing non-CPU extraction allocation' >&2
  exit 64
fi

ARTIFACTS=$ROOT/artifacts/$PBS_JOBID
mkdir -p "$ARTIFACTS/control_sources" "$ARTIFACTS/tmp"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
finish() {
  local rc=$? temporary
  trap - EXIT
  if [[ "$rc" == 0 ]]; then
    touch "$ARTIFACTS/PIPELINE_COMPLETE" || rc=1
  fi
  temporary=$ARTIFACTS/exit_code.txt.tmp.$$
  printf '%s\n' "$rc" > "$temporary"
  mv -f "$temporary" "$ARTIFACTS/exit_code.txt"
  exit "$rc"
}
trap finish EXIT

echo "PBS_JOBID=$PBS_JOBID JOB_NAME=$JOB_NAME STATE=$JOB_STATE QUEUE=$EXECUTION_QUEUE HOST=$HOST CPU_ONLY=1"
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11
MODULE_PYTHON=$(type -P python3.11) || { echo 'Python 3.11 module executable is unavailable' >&2; exit 64; }
MODULE_PYTHON=$(readlink -f -- "$MODULE_PYTHON")
[[ "$MODULE_PYTHON" == /* && -x "$MODULE_PYTHON" ]] || { echo 'Invalid Python module executable' >&2; exit 64; }
"$MODULE_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo 'Loaded module Python is older than 3.11' >&2; exit 64;
}

for name in result_extract.py protocol.json; do
  source_file=$ROOT/$name
  [[ -s "$source_file" ]] || { echo "Missing extraction source: $source_file" >&2; exit 1; }
  cp "$source_file" "$ARTIFACTS/control_sources/$name.tmp"
  mv -f "$ARTIFACTS/control_sources/$name.tmp" "$ARTIFACTS/control_sources/$name"
done
HANDLE_SOURCE=$ROOT/control/evaluation_handle.json
[[ -s "$HANDLE_SOURCE" ]] || { echo "Missing current evaluation handle: $HANDLE_SOURCE" >&2; exit 1; }
cp "$HANDLE_SOURCE" "$ARTIFACTS/control_sources/evaluation_handle.json.tmp"
mv -f "$ARTIFACTS/control_sources/evaluation_handle.json.tmp" "$ARTIFACTS/control_sources/evaluation_handle.json"

EXTRACT_ARGS=(--protocol "$ARTIFACTS/control_sources/protocol.json"
  --evaluation-handle "$ARTIFACTS/control_sources/evaluation_handle.json"
  --root "$ROOT" --out "$ROOT/results/reporting/summary.json")
RECOVERY_SOURCE=$ROOT/control/result_extract_queue_recovery.json
if [[ -e "$RECOVERY_SOURCE" ]]; then
  [[ -s "$RECOVERY_SOURCE" ]] || { echo 'Queue recovery receipt exists but is empty' >&2; exit 1; }
  for name in evaluate.py aggregate.py; do
    source_file=$ROOT/$name
    [[ -s "$source_file" ]] || { echo "Missing recovery validator source: $source_file" >&2; exit 1; }
    cp "$source_file" "$ARTIFACTS/control_sources/$name.tmp"
    mv -f "$ARTIFACTS/control_sources/$name.tmp" "$ARTIFACTS/control_sources/$name"
  done
  cp "$RECOVERY_SOURCE" "$ARTIFACTS/control_sources/result_extract_queue_recovery.json.tmp"
  mv -f "$ARTIFACTS/control_sources/result_extract_queue_recovery.json.tmp" \
    "$ARTIFACTS/control_sources/result_extract_queue_recovery.json"
  EXTRACT_ARGS+=(--queue-recovery-receipt "$ARTIFACTS/control_sources/result_extract_queue_recovery.json")
fi
"$MODULE_PYTHON" "$ARTIFACTS/control_sources/result_extract.py" "${EXTRACT_ARGS[@]}"

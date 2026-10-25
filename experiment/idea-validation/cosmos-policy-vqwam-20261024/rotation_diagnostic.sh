#!/bin/bash -l
set -euo pipefail

ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024
ASSET_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015
host=$(hostname -s)
[[ -n "${PBS_JOBID:-}" && -s "${PBS_NODEFILE:-}" && "${host,,}" != *login* ]] || exit 64
grep -Fqx "$host" <(awk -F. '{print $1}' "$PBS_NODEFILE") || exit 64
qstat_output=$(qstat -f "$PBS_JOBID") || exit 64
job_state=$(awk -F= '/^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' <<<"$qstat_output")
job_name=$(awk -F= '/^[[:space:]]*Job_Name[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$qstat_output")
execution_queue=$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/,"",$2); print $2; exit}' <<<"$qstat_output")
[[ "$job_state" == R && "$job_name" == cpvq24orth ]] || exit 64
[[ "$execution_queue" == qdev || "$execution_queue" == q1 || "$execution_queue" == q2 ]] || exit 64
if grep -Eq 'Resource_List\.ngpus[[:space:]]*=[[:space:]]*[1-9]' <<<"$qstat_output" \
    || [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Refusing non-CPU allocation for rotation diagnostic' >&2
  exit 64
fi

ARTIFACTS=$ROOT/artifacts/$PBS_JOBID
mkdir -p "$ARTIFACTS/control_sources" "$ARTIFACTS/tmp"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
finish() {
  local rc=$? temporary
  trap - EXIT
  temporary=$ARTIFACTS/exit_code.txt.tmp.$$
  printf '%s\n' "$rc" > "$temporary"
  mv -f "$temporary" "$ARTIFACTS/exit_code.txt"
  exit "$rc"
}
trap finish EXIT

echo "PBS_JOBID=$PBS_JOBID JOB_NAME=$job_name STATE=$job_state QUEUE=$execution_queue HOST=$host CPU_ONLY=1 NO_MODEL_FORWARDS=1"
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0
for name in rotation_diagnostic.py common.py environment.sh protocol.json training_state.py runtime.py; do
  source_file=$ROOT/$name
  [[ -s "$source_file" ]] || { echo "Missing diagnostic source: $source_file" >&2; exit 1; }
  cp "$source_file" "$ARTIFACTS/control_sources/$name.tmp"
  mv -f "$ARTIFACTS/control_sources/$name.tmp" "$ARTIFACTS/control_sources/$name"
done

export COSMOS_ROOT=$ROOT COSMOS_CODE_ROOT=$ARTIFACTS/control_sources
export ASSET_ROOT LEGACY_ROOT=$ASSET_ROOT ARTIFACTS
export ARTIFACTS_TMP=$ARTIFACTS/tmp COSMOS_CPU_ONLY=1 COSMOS_CPU_THREADS=4
export COSMOS_EXECUTION_QUEUE=$execution_queue
export APPTAINERENV_COSMOS_CPU_ONLY=1 APPTAINERENV_COSMOS_EXECUTION_QUEUE=$execution_queue
export COSMOS_JOB_START_EPOCH=$(date +%s) COSMOS_SOFT_DEADLINE_SECONDS=900
export COSMOS_HELPERS_PYTHONPATH=$ASSET_ROOT
source "$COSMOS_CODE_ROOT/environment.sh"
run_python "$COSMOS_CODE_ROOT/rotation_diagnostic.py" \
  --root "$ROOT" --out "$ROOT/results/rotation_diagnostic" \
  --source-ours-pbs-job-id 25965916.pbs101
touch "$ARTIFACTS/PIPELINE_COMPLETE"

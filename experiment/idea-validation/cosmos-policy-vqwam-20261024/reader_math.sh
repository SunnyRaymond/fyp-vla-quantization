#!/bin/bash -l
set -euo pipefail
ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024
LEGACY_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015
host=$(hostname -s)
[[ -n "${PBS_JOBID:-}" && -s "${PBS_NODEFILE:-}" && "${host,,}" != *login* ]] || exit 64
grep -Fqx "$host" <(awk -F. '{print $1}' "$PBS_NODEFILE") || exit 64
state=$(qstat -f "$PBS_JOBID") || exit 64
grep -Eq 'job_state[[:space:]]*=[[:space:]]*R' <<<"$state" || exit 64
grep -Eq 'Job_Name[[:space:]]*=[[:space:]]*cpvq24math[[:space:]]*$' <<<"$state" || exit 64
ARTIFACTS=$ROOT/artifacts/$PBS_JOBID
mkdir -p "$ARTIFACTS/control_sources" "$ARTIFACTS/tmp"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
finish() { local rc=$?; trap - EXIT; printf '%s\n' "$rc" > "$ARTIFACTS/exit_code.txt"; exit "$rc"; }
trap finish EXIT
echo "PBS_JOBID=$PBS_JOBID CPU_READER_MATH_HOST=$host NO_MODEL_FORWARDS=1"
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0
for name in test_reader_math.py reader.py softbin_loss.py common.py protocol.json environment.sh; do
  cp "$ROOT/$name" "$ARTIFACTS/control_sources/$name"
done
export COSMOS_ROOT=$ROOT COSMOS_CODE_ROOT=$ARTIFACTS/control_sources ASSET_ROOT=$LEGACY_ROOT
export ARTIFACTS ARTIFACTS_TMP=$ARTIFACTS/tmp COSMOS_CPU_ONLY=1 COSMOS_CPU_THREADS=4
export COSMOS_JOB_START_EPOCH=$(date +%s) COSMOS_SOFT_DEADLINE_SECONDS=900
export COSMOS_HELPERS_PYTHONPATH=$LEGACY_ROOT
source "$COSMOS_CODE_ROOT/environment.sh"
run_python "$COSMOS_CODE_ROOT/test_reader_math.py" --protocol "$COSMOS_CODE_ROOT/protocol.json" \
  --root "$ROOT" --out "$ROOT/results/reader_math"
touch "$ARTIFACTS/PIPELINE_COMPLETE"

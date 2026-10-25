#!/bin/bash -l
set -euo pipefail

ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024
MODE=${1:-parent}

load_python() {
  module purge
  module load gcc/11.4.0-nscc python/3.11.7-gcc11
  PYTHON_BIN=$(command -v python3)
  [[ "$PYTHON_BIN" = /* ]] || { echo 'python3 module did not resolve to an absolute path' >&2; return 1; }
  "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
    echo 'selected module Python must be version 3.11 or newer' >&2
    return 1
  }
}

if [[ "$MODE" == worker ]]; then
  [[ $# == 2 ]] || { echo 'worker mode requires one shared launch JSON path' >&2; exit 64; }
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  load_python
  exec "$PYTHON_BIN" "$SCRIPT_DIR/multinode_probe.py" worker "$2"
fi

[[ "$MODE" == parent && $# == 0 ]] || { echo 'usage: multinode_probe.sh [worker <launch-json>]' >&2; exit 64; }
HOST=$(hostname -s)
[[ -n "${PBS_JOBID:-}" && -s "${PBS_NODEFILE:-}" && "${HOST,,}" != *login* ]] || {
  echo 'Missing native PBS allocation or refusing login host' >&2
  exit 64
}
grep -Fqx "$HOST" <(awk -F. '{print $1}' "$PBS_NODEFILE") || {
  echo "Parent host is absent from native PBS_NODEFILE: $HOST" >&2
  exit 64
}
[[ "$PBS_JOBID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Unexpected PBS job ID format' >&2; exit 64; }

ARTIFACTS=$ROOT/artifacts/$PBS_JOBID
SUMMARY=$ROOT/results/multinode_probe/summary.json
mkdir -p "$ARTIFACTS/control_sources" "$ARTIFACTS/multinode_probe" "$(dirname "$SUMMARY")"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
finish() {
  local rc=$? tmp
  trap - EXIT
  if [[ ! -s "$SUMMARY" ]] || ! grep -Fqx "  \"pbs_job_id\": \"$PBS_JOBID\"," "$SUMMARY"; then
    tmp=$SUMMARY.tmp.$$
    printf '{"schema":"cosmos-multinode-probe-v1","pbs_job_id":"%s","status":"failed","complete":false,"failure_reason":"parent exited before writing summary"}\n' "$PBS_JOBID" > "$tmp"
    mv -f "$tmp" "$SUMMARY"
  fi
  tmp=$ARTIFACTS/exit_code.txt.tmp.$$
  printf '%s\n' "$rc" > "$tmp"
  mv -f "$tmp" "$ARTIFACTS/exit_code.txt"
  exit "$rc"
}
trap finish EXIT

for name in multinode_probe.py multinode_probe.sh multinode_probe.pbs; do
  source_file=$ROOT/$name
  [[ -s "$source_file" ]] || { echo "Missing probe source snapshot: $source_file" >&2; exit 1; }
  cp "$source_file" "$ARTIFACTS/control_sources/$name.tmp"
  mv -f "$ARTIFACTS/control_sources/$name.tmp" "$ARTIFACTS/control_sources/$name"
done

load_python
echo "PBS_JOBID=$PBS_JOBID JOB_NAME=cpvq24mprobe PARENT_HOST=$HOST CPU_ONLY=1"
"$PYTHON_BIN" "$ARTIFACTS/control_sources/multinode_probe.py" parent --root "$ROOT" --artifacts "$ARTIFACTS"

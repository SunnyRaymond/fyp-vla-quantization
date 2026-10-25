#!/bin/bash
set -euo pipefail

LAUNCH=${1:?launch receipt path required}
[[ -s "$LAUNCH" ]] || { echo 'Missing shared PBS evaluation launch receipt' >&2; exit 64; }
CONTROL_SOURCES=$(dirname "$LAUNCH")/control_sources
COORDINATOR=$CONTROL_SOURCES/evaluation_multinode.py
[[ -s "$COORDINATOR" ]] || { echo 'Snapshotted multinode coordinator is missing' >&2; exit 64; }

module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0 cuda/12.2.2
MODULE_PYTHON=$(type -P python3.11) || { echo 'Python module executable is unavailable' >&2; exit 64; }
MODULE_PYTHON=$(readlink -f -- "$MODULE_PYTHON")
[[ "$MODULE_PYTHON" == /* && -x "$MODULE_PYTHON" ]] || { echo 'Invalid module Python executable' >&2; exit 64; }
NODE_INFO=$(python3 "$COORDINATOR" claim-node --launch "$LAUNCH") || exit 64
IFS=$'\t' read -r NODE_RANK NODE_DIR CLAIM_DIR ROOT OUT ARTIFACTS_ROOT CONTROL_SOURCES ENGINE_SOURCES \
  LEGACY_SOURCES SOURCE_HELPERS PROTOCOL_PATH DEADLINE_EPOCH SOFT_DEADLINE_SECONDS PBS_JOBID_RECEIPT \
  PBS_TASKNUM_RECEIPT EFFECTIVE_NODEFILE NODEFILE_SOURCE NATIVE_NODEFILE \
  <<<"$(python3 - "$NODE_INFO" <<'PY'
import json, sys
v = json.loads(sys.argv[1])
print("\t".join(str(v[key]) for key in (
    "node_rank", "node_dir", "claim_dir", "root", "out", "artifacts_root",
    "control_sources", "engine_sources", "legacy_sources", "source_helpers",
    "protocol_path", "deadline_epoch", "soft_deadline_seconds", "pbs_job_id",
    "pbs_tasknum", "effective_nodefile", "nodefile_source", "native_nodefile")))
PY
)"
[[ "$PBS_JOBID" == "$PBS_JOBID_RECEIPT" ]] || { echo 'Native PBS_JOBID changed after node claim' >&2; exit 64; }
[[ "${PBS_TASKNUM:-}" == "$PBS_TASKNUM_RECEIPT" ]] || { echo 'Native PBS_TASKNUM changed after node claim' >&2; exit 64; }
NATIVE_NODEFILE_NOW=${PBS_NODEFILE:-'(missing)'}
[[ "$NATIVE_NODEFILE_NOW" == "$NATIVE_NODEFILE" ]] || { echo 'Native PBS_NODEFILE changed after node claim' >&2; exit 64; }
PBS_NODEFILE=$EFFECTIVE_NODEFILE
export PBS_NODEFILE COSMOS_NODEFILE_SOURCE=$NODEFILE_SOURCE COSMOS_NATIVE_PBS_NODEFILE=$NATIVE_NODEFILE

mkdir -p "$NODE_DIR/tmp"
exec > >(tee -a "$NODE_DIR/job.log") 2>&1
GPU_MONITOR_PID=
write_node_receipt() {
  local rc=$1
  "$MODULE_PYTHON" "$COORDINATOR" write-node-receipt --launch "$LAUNCH" \
    --node-rank "$NODE_RANK" --exit-code "$rc" || {
      echo 'Could not persist this node receipt; coordinator must treat node as incomplete' >&2
      [[ "$rc" != 0 ]] || rc=1
    }
  return "$rc"
}
finish() {
  local rc=$?
  trap - EXIT
  if [[ -n "$GPU_MONITOR_PID" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi
  printf '%s\n' "$rc" > "$NODE_DIR/exit_code.txt"
  write_node_receipt "$rc" || rc=$?
  exit "$rc"
}
trap finish EXIT

printf 'PBS_JOBID=%s PBS_TASKNUM=%s NODE_RANK=%s HOST=%s PBS_NODEFILE_NATIVE=%s PBS_NODEFILE_SOURCE=%s PBS_NODEFILE_EFFECTIVE=%s\n' \
  "$PBS_JOBID" "$PBS_TASKNUM" "$NODE_RANK" "$(hostname -f)" "$NATIVE_NODEFILE" "$NODEFILE_SOURCE" "$PBS_NODEFILE"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=, read -r -a VISIBLE_DEVICES <<<"$CUDA_VISIBLE_DEVICES"
  if [[ ${#VISIBLE_DEVICES[@]} -ne 4 ]] || [[ $(printf '%s\n' "${VISIBLE_DEVICES[@]}" | sort -u | wc -l) -ne 4 ]]; then
    echo 'Native CUDA_VISIBLE_DEVICES must name four unique allocated GPUs on this node' >&2; exit 64
  fi
else
  mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
  if [[ ${#GPU_UUIDS[@]} -ne 4 ]] || [[ $(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l) -ne 4 ]]; then
    echo 'Cannot derive devices: PBS allocation is not verified as all four GPUs on this node' >&2; exit 64
  fi
  CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_UUIDS[*]}")
  export CUDA_VISIBLE_DEVICES
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi -L

GPU_MONITOR_INTERVAL=$(python3 - "$PROTOCOL_PATH" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("gpu_monitor_interval_seconds", 15)))
PY
)
(
  while true; do
    printf 'GPU_SAMPLE_UTC=%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      || echo 'GPU_SAMPLE_ERROR=nvidia-smi_failed'
    sleep "$GPU_MONITOR_INTERVAL"
  done
) >> "$NODE_DIR/job.log" 2>&1 &
GPU_MONITOR_PID=$!

ASSET_ROOT=$(python3 - "$PROTOCOL_PATH" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p["asset_root"])
PY
)
LEGACY_ROOT=$(python3 - "$PROTOCOL_PATH" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p.get("legacy_root", p["asset_root"]))
PY
)
export ROOT LEGACY_ROOT ARTIFACTS=$ARTIFACTS_ROOT ARTIFACTS_TMP="$NODE_DIR/tmp"
export COSMOS_ROOT=$ROOT COSMOS_CODE_ROOT=$CONTROL_SOURCES ASSET_ROOT
export COSMOS_JOB_START_EPOCH=$(python3 - "$LAUNCH" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["started_epoch"]))
PY
)
export COSMOS_SOFT_DEADLINE_SECONDS=$SOFT_DEADLINE_SECONDS
COSMOS_HELPERS_PYTHONPATH=$(python3 - "$PROTOCOL_PATH" "$SOURCE_HELPERS" <<'PY'
import json, pathlib, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(":".join(dict.fromkeys((str(pathlib.Path(sys.argv[2])),
                               str(pathlib.Path(p["source_paths"]["teacher_original"]))))))
PY
)
export COSMOS_HELPERS_PYTHONPATH
source "$CONTROL_SOURCES/environment.sh"

run_python "$CONTROL_SOURCES/evaluate.py" --protocol "$PROTOCOL_PATH" --root "$ROOT" --out "$OUT" \
  --node-only --node-rank "$NODE_RANK" --launch-receipt "$LAUNCH"

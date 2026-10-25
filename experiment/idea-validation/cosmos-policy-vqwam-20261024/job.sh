#!/bin/bash -l
set -euo pipefail

ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024
ASSET_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015
LEGACY_ROOT=$ASSET_ROOT
STAGE=${1:?stage is required}
EVALUATION_WORKERS=0
case "$STAGE" in
  prepare|teacher|quarot|svdquant|viditq|vidcal|ours_short|ours_long|perf_bf16|perf_spinquant|perf_quarot|perf_svdquant|perf_viditq|perf_ours|evaluation) ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 64 ;;
esac

require_allocation() {
  local host state job_name
  [[ -n "${PBS_JOBID:-}" && -s "${PBS_NODEFILE:-}" ]] || { echo 'Missing PBS allocation identity' >&2; exit 64; }
  host=$(hostname -s)
  [[ "${host,,}" != *login* ]] || { echo "Refusing login node: $host" >&2; exit 64; }
  grep -Fqx "$host" <(awk -F. '{print $1}' "$PBS_NODEFILE") || { echo "Host not listed in PBS_NODEFILE: $host" >&2; exit 64; }
  local qstat_out
  qstat_out=$(qstat -f "$PBS_JOBID") || { echo "Cannot query current PBS job $PBS_JOBID" >&2; exit 64; }
  state=$(awk -F= '/^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' <<<"$qstat_out")
  [[ "$state" == R ]] || { echo "Current PBS job is not running (state=${state:-unknown})" >&2; exit 64; }
  if [[ "$STAGE" == evaluation ]]; then
    local profile expected_queue expected_gpus expected_workers expected_nodes gpus_per_node actual_queue
    profile=$(python3 - "$ROOT/protocol.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
r = p.get("resources", {}).get("evaluation", {})
w = p.get("load_balancing", {}).get("persistent_workers")
s = p.get("stages", {}).get("evaluation", {})
key = (r.get("gpus"), r.get("required_execution_queue"), w,
       s.get("script", "evaluation.pbs"))
single = (key == (4, "g3", 4, "evaluation.pbs") and r.get("ncpus") == 64
          and r.get("memory_gb") == 440 and r.get("nodes", 1) == 1
          and r.get("gpus_per_node", 4) == 4 and r.get("ncpus_per_node", 64) == 64
          and r.get("memory_gb_per_node", 440) == 440)
multi = (key == (8, "g4", 8, "evaluation_g4.pbs") and r.get("ncpus") == 128
         and r.get("memory_gb") == 880 and r.get("nodes") == 2
         and r.get("gpus_per_node") == 4 and r.get("ncpus_per_node") == 64
         and r.get("memory_gb_per_node") == 440)
if (not (single or multi) or r.get("walltime") != "24:00:00"
        or r.get("queue") != "normal" or s.get("job_name", "cpvq24eval") != "cpvq24eval"):
    raise SystemExit("Evaluation protocol is not an approved g3/4-GPU or g4/8-GPU two-node profile")
nodes = 1 if single else 2
gpus_per_node = 4
print(key[1], key[0], w, nodes, gpus_per_node)
PY
) || { echo 'Cannot verify evaluation profile from root/protocol.json' >&2; exit 64; }
    read -r expected_queue expected_gpus expected_workers expected_nodes gpus_per_node <<<"$profile"
    actual_queue=$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' <<<"$qstat_out")
    [[ "$actual_queue" == "$expected_queue" ]] || {
      echo "Evaluation allocation queue $actual_queue differs from protocol $expected_queue" >&2; exit 64;
    }
    local visible_devices=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=, read -r -a visible_devices <<<"$CUDA_VISIBLE_DEVICES"
    fi
    [[ "$expected_workers" -eq "$expected_gpus" ]] || {
      echo 'Evaluation worker count differs from total allocated GPUs' >&2; exit 64;
    }
    if [[ "$expected_nodes" == 1 || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      [[ ${#visible_devices[@]} -eq $gpus_per_node ]] || {
        echo "Evaluation profile requires $gpus_per_node local visible GPUs" >&2; exit 64;
      }
    fi
    EVALUATION_WORKERS=$expected_workers
    echo "EVALUATION_PROFILE queue=$expected_queue nodes=$expected_nodes gpus=$expected_gpus workers=$expected_workers local_gpus=$gpus_per_node"
  fi
  if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24c2 ]]; then
    [[ "$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' <<<"$qstat_out")" == g2 ]] || {
      echo 'Baseline follow-up bundle was not routed to g2' >&2; exit 64;
    }
  fi
  case "$STAGE" in
    prepare) job_name=cpvq24prep ;;
    teacher) job_name=cpvq24teach ;;
    quarot) job_name=cpvq24qrot ;;
    svdquant) job_name=cpvq24svd ;;
    viditq) job_name=cpvq24vid ;;
    vidcal) job_name=cpvq24c2 ;;
    ours_short) job_name=cpvq24osht ;;
    ours_long) job_name=cpvq24olon ;;
    perf_bf16) job_name=cpvq24pbf16 ;;
    perf_spinquant) job_name=cpvq24pspin ;;
    perf_quarot) job_name=cpvq24pqrot ;;
    perf_svdquant) job_name=cpvq24psvd ;;
    perf_viditq) job_name=cpvq24pvid ;;
    perf_ours) job_name=cpvq24pours ;;
    evaluation) job_name=cpvq24eval ;;
  esac
  if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24b2 ]]; then
    case "$STAGE" in teacher|perf_bf16|perf_spinquant) job_name=cpvq24b2 ;;
      *) echo 'Stage is outside the authorized bootstrap bundle' >&2; exit 64 ;;
    esac
  fi
  if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24c2 ]]; then
    case "$STAGE" in perf_svdquant|vidcal|perf_viditq) job_name=cpvq24c2 ;;
      *) echo 'Stage is outside the authorized baseline follow-up bundle' >&2; exit 64 ;;
    esac
  elif [[ "$STAGE" == vidcal ]]; then
    echo 'vidcal is an internal baseline-follow-up bundle stage' >&2; exit 64
  fi
  [[ "$(awk -F= '/^[[:space:]]*Job_Name[[:space:]]*=/{sub(/^[[:space:]]*/,"",$2); sub(/[[:space:]]*$/, "", $2); print $2; exit}' <<<"$qstat_out")" == "$job_name" ]] || {
    echo "PBS Job_Name does not match stage $STAGE" >&2; exit 64;
  }
}

main() {
  require_allocation
  ARTIFACTS=$ROOT/artifacts/$PBS_JOBID
  if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24b2 || "${COSMOS_PARENT_BUNDLE:-}" == cpvq24c2 ]]; then
    ARTIFACTS=$ARTIFACTS/stages/$STAGE
  fi
  mkdir -p "$ARTIFACTS"
  exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
  GPU_MONITOR_PID=
  RACE_RESUME_LOCK=0
  export COSMOS_JOB_START_EPOCH=$(date +%s)
  finish() {
    local rc=$?
    trap - EXIT
    if [[ -n "$GPU_MONITOR_PID" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi
    if [[ "$RACE_RESUME_LOCK" == 1 ]]; then release_resume_lock; fi
    printf '%s\n' "$rc" > "$ARTIFACTS/exit_code.txt"
    exit "$rc"
  }
  trap finish EXIT
  printf 'PBS_JOBID=%s\nSTAGE=%s\nHOST=%s\n' "$PBS_JOBID" "$STAGE" "$(hostname -f)"

  case "$STAGE" in
    ours_short|ours_long)
      module purge
      module load gcc/11.4.0-nscc
      module load python/3.11.7-gcc11
      claim_ours_race ;;
  esac

  if [[ "$STAGE" == prepare ]]; then
    module purge
    module load gcc/11.4.0-nscc
    module load python/3.11.7-gcc11
    MODULE_PYTHON=$(type -P python3.11) || { echo 'Python 3.11 module executable is unavailable' >&2; exit 64; }
    MODULE_PYTHON=$(readlink -f -- "$MODULE_PYTHON")
    [[ "$MODULE_PYTHON" == /* && -x "$MODULE_PYTHON" ]] || { echo 'Invalid absolute module Python executable' >&2; exit 64; }
    "$MODULE_PYTHON" -c 'import sys; print("MODULE_PYTHON=" + sys.executable + " VERSION=" + sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
      echo 'Loaded Python module is older than 3.11' >&2; exit 64;
    }
    snapshot_control_sources
    "$MODULE_PYTHON" "$ARTIFACTS/control_sources/prepare.py" --protocol "$ARTIFACTS/control_sources/protocol.json" \
      --root "$ROOT" --out "$ROOT/results/preparation" --soft-deadline-seconds 13500
    mark_stage_result "$ROOT/results/preparation/summary.json"
    exit 0
  fi

  module purge
  module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0 cuda/12.2.2
  case "$STAGE" in
    teacher|perf_bf16|perf_spinquant)
      if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24b2 ]]; then
        python3 "$ROOT/bootstrap_race.py" claim --candidate bundle
      else
        python3 "$ROOT/bootstrap_race.py" claim --candidate single --stage "$STAGE"
      fi ;;
    perf_svdquant|viditq)
      if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24c2 ]]; then
        python3 "$ROOT/bootstrap_race.py" claim --candidate bundle --race-directory control/baseline_followup_race
      elif [[ -s "$ROOT/control/baseline_followup_race/submissions_ready.json" ]]; then
        python3 "$ROOT/bootstrap_race.py" claim --candidate single --stage "$STAGE" \
          --race-directory control/baseline_followup_race
      fi ;;
    vidcal)
      python3 "$ROOT/bootstrap_race.py" claim --candidate bundle --race-directory control/baseline_followup_race ;;
    perf_viditq)
      if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24c2 ]]; then
        python3 "$ROOT/bootstrap_race.py" claim --candidate bundle --race-directory control/baseline_followup_race
      fi ;;
  esac
  command -v nvidia-smi >/dev/null
  nvidia-smi -L
  snapshot_sources
  if [[ "$STAGE" == evaluation && "$EVALUATION_WORKERS" == 8 ]]; then
    echo 'EVALUATION_GPU_MONITORING=per-node'
  else
    gpu_monitor
  fi
  export COSMOS_ROOT=$ROOT COSMOS_CODE_ROOT="$ARTIFACTS/control_sources" ASSET_ROOT LEGACY_ROOT ARTIFACTS_TMP="$ARTIFACTS/tmp"
  mkdir -p "$ARTIFACTS_TMP"
  COSMOS_HELPERS_PYTHONPATH=$(helper_pythonpath)
  export COSMOS_HELPERS_PYTHONPATH
  case "$STAGE" in
    ours_short) export COSMOS_SOFT_DEADLINE_SECONDS=6600 ;;
    ours_long) export COSMOS_SOFT_DEADLINE_SECONDS=39600 ;;
    evaluation) export COSMOS_SOFT_DEADLINE_SECONDS=85500 ;;
    *) export COSMOS_SOFT_DEADLINE_SECONDS=6600 ;;
  esac
  source "$COSMOS_CODE_ROOT/environment.sh"

  local stage_rc=0 summary
  case "$STAGE" in
    teacher)
      summary=$ROOT/results/teacher/summary.json
      run_python "$COSMOS_CODE_ROOT/teacher.py" --protocol "$COSMOS_CODE_ROOT/protocol.json" --root "$ROOT" --out "$ROOT/results/teacher" \
        --soft-deadline-seconds "$COSMOS_SOFT_DEADLINE_SECONDS" || stage_rc=$? ;;
    quarot|svdquant|viditq)
      ARM=$(stage_arm "$STAGE")
      summary=$ROOT/results/$STAGE/summary.json
      run_python "$COSMOS_CODE_ROOT/baseline_worker.py" --arm "$ARM" --protocol "$COSMOS_CODE_ROOT/protocol.json" --root "$ROOT" \
        --out "$ROOT/results/$STAGE" --soft-deadline-seconds "$COSMOS_SOFT_DEADLINE_SECONDS" || stage_rc=$? ;;
    ours_short|ours_long)
      summary=$ROOT/results/ours/summary.json
      run_python "$COSMOS_CODE_ROOT/calibrate_ours.py" --stage "$STAGE" --protocol "$COSMOS_CODE_ROOT/protocol.json" --root "$ROOT" \
        --out "$ROOT/results/ours" --soft-deadline-seconds "$COSMOS_SOFT_DEADLINE_SECONDS" || stage_rc=$? ;;
    vidcal)
      summary=$ROOT/results/viditq/summary.json
      run_python "$COSMOS_CODE_ROOT/baseline_worker.py" --arm viditq_w4a4 --protocol "$COSMOS_CODE_ROOT/protocol.json" --root "$ROOT" \
        --out "$ROOT/results/viditq" --soft-deadline-seconds "$COSMOS_SOFT_DEADLINE_SECONDS" || stage_rc=$? ;;
    perf_bf16|perf_spinquant|perf_quarot|perf_svdquant|perf_viditq|perf_ours)
      ARM=$(stage_arm "$STAGE")
      summary=$ROOT/results/performance/$ARM/summary.json
      run_python "$COSMOS_CODE_ROOT/benchmark.py" --arm "$ARM" --protocol "$COSMOS_CODE_ROOT/protocol.json" --root "$ROOT" \
        --out "$ROOT/results/performance/$ARM" --soft-deadline-seconds 6600 || stage_rc=$? ;;
    evaluation)
      summary=$ROOT/results/evaluation/summary.json
      if [[ "$EVALUATION_WORKERS" == 8 ]]; then
        python3 "$COSMOS_CODE_ROOT/evaluation_multinode.py" run --protocol "$COSMOS_CODE_ROOT/protocol.json" \
          --root "$ROOT" --out "$ROOT/results/evaluation" --artifacts "$ARTIFACTS" \
          --soft-deadline-seconds "$COSMOS_SOFT_DEADLINE_SECONDS" || stage_rc=$?
      else
        run_python "$COSMOS_CODE_ROOT/evaluate.py" --protocol "$COSMOS_CODE_ROOT/protocol.json" --root "$ROOT" --out "$ROOT/results/evaluation" \
          --soft-deadline-seconds "$COSMOS_SOFT_DEADLINE_SECONDS" || stage_rc=$?
      fi ;;
  esac
  mark_stage_result "$summary"
  if [[ "$stage_rc" == 0 || "$stage_rc" == 2 ]] && [[ -e "$ARTIFACTS/PIPELINE_CHECKPOINTED" ]]; then
    stage_rc=0
    # afterok children must only be released by a complete C80 teacher.
    [[ "$STAGE" != teacher ]] || stage_rc=2
  fi
  return "$stage_rc"
}

function stage_arm() {
  python3 - "$ARTIFACTS/control_sources/protocol.json" "$1" <<'PY'
import json, sys
stage = sys.argv[2]
print("viditq_w4a4" if stage == "vidcal" else
      json.load(open(sys.argv[1], encoding="utf-8"))["stage_arms"][stage])
PY
}

function snapshot_sources() {
  local engine_root name source bytes
  snapshot_control_sources
  mkdir -p "$ARTIFACTS/engine_sources" "$ARTIFACTS/legacy_sources" "$ARTIFACTS/source_helpers"
  engine_root=$(python3 - "$ARTIFACTS/control_sources/protocol.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_paths"]["engine"])
PY
)
  for name in decoder_runtime.py fused_vq.py fused_input.py layer_runtime.py; do
    source=$engine_root/$name
    [[ -s "$source" ]] || { echo "Missing frozen engine helper: $source" >&2; return 1; }
    bytes=$(stat -c '%s' "$source")
    (( bytes <= 2097152 )) || { echo "Refusing oversized engine source snapshot: $source ($bytes bytes)" >&2; return 1; }
    cp "$source" "$ARTIFACTS/engine_sources/$name.tmp"
    mv -f "$ARTIFACTS/engine_sources/$name.tmp" "$ARTIFACTS/engine_sources/$name"
  done
  for name in engine.py export_bundle.py; do
    source=$ARTIFACTS/control_sources/$name
    [[ -s "$source" ]] || { echo "Missing new engine source: $source" >&2; return 1; }
    cp "$source" "$ARTIFACTS/engine_sources/$name.tmp"
    mv -f "$ARTIFACTS/engine_sources/$name.tmp" "$ARTIFACTS/engine_sources/$name"
  done
  for name in cosmos_runtime.py endpoint.py calibrate.py quantization.py; do
    source=$LEGACY_ROOT/$name
    [[ -s "$source" ]] || { echo "Missing frozen legacy helper: $source" >&2; return 1; }
    bytes=$(stat -c '%s' "$source")
    (( bytes <= 2097152 )) || { echo "Refusing oversized legacy source snapshot: $source ($bytes bytes)" >&2; return 1; }
    cp "$source" "$ARTIFACTS/legacy_sources/$name.tmp"
    mv -f "$ARTIFACTS/legacy_sources/$name.tmp" "$ARTIFACTS/legacy_sources/$name"
  done
  while IFS=$'\t' read -r name source; do
    [[ "$name" =~ ^[a-zA-Z0-9_]+$ && -s "$source" ]] || { echo "Invalid or missing protocol helper: $name $source" >&2; return 1; }
    bytes=$(stat -c '%s' "$source")
    (( bytes <= 2097152 )) || { echo "Refusing oversized protocol helper snapshot: $source ($bytes bytes)" >&2; return 1; }
    cp "$source" "$ARTIFACTS/source_helpers/$name.py.tmp"
    mv -f "$ARTIFACTS/source_helpers/$name.py.tmp" "$ARTIFACTS/source_helpers/$name.py"
  done < <(python3 - "$ARTIFACTS/control_sources/protocol.json" <<'PY'
import json, pathlib, sys
protocol = json.load(open(sys.argv[1], encoding="utf-8"))
for name, value in protocol["source_paths"]["helpers"].items():
    print(name + "\t" + str(pathlib.Path(value)))
PY
)
}

function snapshot_control_sources() {
  local source target bytes
  mkdir -p "$ARTIFACTS/control_sources"
  while IFS= read -r -d '' source; do
    bytes=$(stat -c '%s' "$source")
    (( bytes <= 5242880 )) || { echo "Refusing oversized control source snapshot: $source ($bytes bytes)" >&2; return 1; }
    target=$ARTIFACTS/control_sources/${source##*/}
    cp "$source" "$target.tmp"
    mv -f "$target.tmp" "$target"
  done < <(find "$ROOT" -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.pbs' -o -name 'protocol.json' \) -print0)
  [[ -s "$ARTIFACTS/control_sources/protocol.json" && -s "$ARTIFACTS/control_sources/job.sh" ]] || {
    echo 'Control snapshot is missing protocol.json or job.sh' >&2; return 1;
  }
}

function helper_pythonpath() {
  python3 - "$ARTIFACTS/control_sources/protocol.json" "$ARTIFACTS/source_helpers" <<'PY'
import json, pathlib, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
paths = [sys.argv[2], p["source_paths"]["teacher_original"]]
print(":".join(dict.fromkeys(str(pathlib.Path(path)) for path in paths)))
PY
}

function gpu_monitor() {
  local interval
  interval=$(python3 - "$ARTIFACTS/control_sources/protocol.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("gpu_monitor_interval_seconds", 15)))
PY
)
  (
    while true; do
      printf 'GPU_SAMPLE_UTC=%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || echo 'GPU_SAMPLE_ERROR=nvidia-smi_failed'
      sleep "$interval"
    done
  ) >> "$ARTIFACTS/job.log" 2>&1 &
  GPU_MONITOR_PID=$!
}

function read_field() {
  python3 - "$1" "$2" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

function race_job_name() {
  case "$1" in ours_short) echo cpvq24osht ;; ours_long) echo cpvq24olon ;; *) return 64 ;; esac
}

function race_ready_pair() {
  local marker=$ROOT/control/ours_race/submissions_ready.json waited=0
  while [[ ! -s "$marker" && $waited -lt 300 ]]; do sleep 2; waited=$((waited+2)); done
  [[ -s "$marker" ]] || { echo 'Ours race pair receipt did not arrive; refusing to load model' >&2; return 1; }
  local expected_self expected_sibling self_id sibling_id
  expected_self=$(race_job_name "$STAGE")
  if [[ "$STAGE" == ours_short ]]; then
    expected_sibling=cpvq24olon
    self_id=$(read_field "$marker" short.job_id)
    sibling_id=$(read_field "$marker" long.job_id)
    [[ "$(read_field "$marker" short.job_name)" == "$expected_self" && "$(read_field "$marker" long.job_name)" == "$expected_sibling" ]]
  else
    expected_sibling=cpvq24osht
    self_id=$(read_field "$marker" long.job_id)
    sibling_id=$(read_field "$marker" short.job_id)
    [[ "$(read_field "$marker" long.job_name)" == "$expected_self" && "$(read_field "$marker" short.job_name)" == "$expected_sibling" ]]
  fi
  [[ "$self_id" == "$PBS_JOBID" ]] || { echo 'Race pair receipt does not bind this PBS job' >&2; return 1; }
  [[ "$sibling_id" =~ ^[0-9]+\.[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid sibling PBS handle in race receipt' >&2; return 1; }
  printf '%s %s\n' "$sibling_id" "$expected_sibling"
}

function verify_checkpointed_ours() {
  python3 - "$ROOT/results/ours/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file() or json.loads(p.read_text(encoding="utf-8")).get("status") != "checkpointed":
    raise SystemExit(1)
PY
}

function mark_stage_result() {
  local summary=$1 status
  status=$(python3 - "$summary" "$STAGE" "$PBS_JOBID" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("Stage summary is missing: " + str(path))
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("status") not in ("complete", "checkpointed"):
    raise SystemExit("Stage summary has no terminal complete/checkpointed status")
job_id = value.get("pbs_job_id")
if job_id is not None and job_id != sys.argv[3]:
    raise SystemExit("Stage summary belongs to a different PBS job")
if sys.argv[2] in ("ours_short", "ours_long") and job_id != sys.argv[3]:
    raise SystemExit("Ours training summary lacks the current PBS job identity")
print(value["status"])
PY
)
  case "$status" in
    complete) touch "$ARTIFACTS/PIPELINE_COMPLETE" ;;
    checkpointed) touch "$ARTIFACTS/PIPELINE_CHECKPOINTED" ;;
  esac
  echo "STAGE_STATUS=$status SUMMARY=$summary"
}

function qstat_fields() {
  local output
  output=$(qstat -xf "$1") || return 1
  awk -F= '
    /^[[:space:]]*Job_Name[[:space:]]*=/{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); name=$2}
    /^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); state=$2}
    /^[[:space:]]*Exit_status[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); exitcode=$2}
    END {print name " " state " " exitcode}
  ' <<<"$output"
}

function cancel_queued_sibling() {
  local sibling_id=$1 sibling_name=$2 fields name state exitcode matches attempt
  fields=$(qstat_fields "$sibling_id") || {
    matches=$(qselect -u yguo017 -N "$sibling_name" 2>&1) || return 1
    [[ -z "$matches" ]] || { echo "Cannot verify sibling handle $sibling_id; matching jobs remain: $matches" >&2; return 1; }
    echo "Sibling $sibling_id no longer appears in PBS; verified no live $sibling_name job"
    return 0
  }
  read -r name state exitcode <<<"$fields"
  [[ "$name" == "$sibling_name" ]] || { echo 'Sibling PBS Job_Name identity mismatch' >&2; return 1; }
  if [[ "$state" == Q || "$state" == H ]]; then
    qdel "$sibling_id"
    for attempt in {1..15}; do
      if ! fields=$(qstat_fields "$sibling_id"); then
        matches=$(qselect -u yguo017 -N "$sibling_name" 2>&1) || return 1
        [[ -z "$matches" ]] || { echo "Sibling disappeared but matching stage job remains: $matches" >&2; return 1; }
        state=gone
        break
      fi
      read -r name state exitcode <<<"$fields"
      [[ "$name" == "$sibling_name" ]] || return 1
      [[ "$state" == F || "$state" == X ]] && break
      sleep 1
    done
    if [[ "$state" != F && "$state" != X && "$state" != gone ]]; then
      echo "Sibling did not reach a verified terminal state after qdel: $sibling_id state=$state" >&2; return 1
    fi
  fi
  echo "Sibling identity checked: job=$sibling_id name=$name state=$state exit=$exitcode"
  return 0
}

function write_race_owner() {
  local target=$ROOT/control/ours_race/winner.json temporary=$ROOT/control/ours_race/winner.json.tmp.$PBS_JOBID
  python3 - "$temporary" "$STAGE" "$PBS_JOBID" "$(race_job_name "$STAGE")" "${1:-}" <<'PY'
import json, os, socket, sys
path, stage, job_id, job_name, resumed_from = sys.argv[1:]
value = {"stage": stage, "job_id": job_id, "job_name": job_name,
         "host": socket.gethostname(), "pid": os.getpid(), "resumed_from": resumed_from or None}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
  mv -f "$temporary" "$target"
}

function write_startup_repair_owner() {
  local old_id=$1 target=$ROOT/control/ours_race/winner.json temporary=$ROOT/control/ours_race/winner.json.startup.$PBS_JOBID
  python3 - "$temporary" "$STAGE" "$PBS_JOBID" "$(race_job_name "$STAGE")" "$old_id" <<'PY'
import json, os, socket, sys
path, stage, job_id, job_name, old_id = sys.argv[1:]
value = {"stage": stage, "job_id": job_id, "job_name": job_name,
         "host": socket.gethostname(), "pid": os.getpid(),
         "startup_repair": True, "startup_repair_from": old_id}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
  mv -f "$temporary" "$target"
}

function claim_startup_repair() {
  local race=$ROOT/control/ours_race marker=$ROOT/control/ours_race/startup_repair.json
  local lock=$ROOT/control/ours_race/startup_repair_claim owner=$ROOT/control/ours_race/winner.json
  local old_id=$1 sibling_id=$2 sibling_name=$3
  mkdir "$lock" 2>/dev/null || { echo 'Startup-repair claim is already held; refusing duplicate retry' >&2; return 1; }
  python3 - "$ROOT" "$marker" "$owner" "$STAGE" "$PBS_JOBID" "$old_id" "$sibling_id" "$sibling_name" \
    "$lock/owner.json" <<'PY'
import json, os, re, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

root, marker_path, owner_path, stage, current_id, old_id, sibling_id, sibling_name, lock_owner = sys.argv[1:]
marker_path, owner_path, root = Path(marker_path), Path(owner_path), Path(root)
marker = json.loads(marker_path.read_text(encoding="utf-8"))
owner = json.loads(owner_path.read_text(encoding="utf-8"))
protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
if marker.get("protocol") != protocol.get("protocol") or marker.get("status") != "ready_for_release":
    raise SystemExit("Startup-repair marker is not ready or protocol-bound")
if marker.get("zero_updates") is not True or marker.get("no_checkpoint") is not True:
    raise SystemExit("Startup-repair marker lacks its zero-update/no-checkpoint proof")
old = marker.get("old_winner", {})
new = marker.get("new_winner", {})
sibling = marker.get("sibling", {})
if (old.get("job_id") != old_id or owner.get("job_id") != old_id
        or old.get("stage") != stage or owner.get("stage") != stage
        or old.get("job_name") != owner.get("job_name")
        or new.get("job_id") != current_id or new.get("stage") != stage
        or sibling.get("job_id") != sibling_id or sibling.get("job_name") != sibling_name):
    raise SystemExit("Startup-repair marker, old race winner, or current PBS ID differs")
proof = marker.get("proof", {})
archive_rel = proof.get("old_winner_handle_archive")
if (not isinstance(archive_rel, str) or not archive_rel.startswith("control/")
        or ".." in Path(archive_rel).parts):
    raise SystemExit("Startup-repair marker lacks a safe archived old handle path")
archived_handle_path = root / archive_rel
if not archived_handle_path.is_file():
    raise SystemExit("Startup-repair archived old winner handle is missing")
archived_handle = json.loads(archived_handle_path.read_text(encoding="utf-8"))
if (archived_handle.get("job_id") != old_id or archived_handle.get("job_name") != old.get("job_name")
        or archived_handle.get("stage") not in (None, stage)):
    raise SystemExit("Startup-repair archived handle does not match the old winner")
race_root = root / "control/ours_race"
if not (race_root / "claim").is_dir() or (race_root / "resume_claim").exists():
    raise SystemExit("Startup-repair race claim state is missing or already resuming")
if not (race_root / "startup_repair_claim").is_dir():
    raise SystemExit("Atomic startup-repair claim is missing")
expected_name = {"ours_short": "cpvq24osht", "ours_long": "cpvq24olon"}.get(stage)
if not expected_name or old.get("job_name") != expected_name or new.get("job_name") != expected_name:
    raise SystemExit("Startup-repair stage name is not an Ours stage")

def attrs(job_id):
    result = subprocess.run(["qstat", "-xf", job_id], check=False, capture_output=True,
                            text=True, timeout=25)
    if result.returncode:
        raise SystemExit("Cannot verify PBS job " + job_id + ": " + result.stderr.strip())
    values = {}
    current = None
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)$", line)
        if match:
            current = match.group(1)
            values[current] = match.group(2).strip()
        elif current and line.strip():
            values[current] += " " + line.strip()
    return values

old_status = attrs(old_id)
if (old_status.get("Job_Name") != expected_name or old_status.get("job_state") != "F"
        or not old_status.get("Exit_status", "").lstrip("-").isdigit()
        or int(old_status["Exit_status"]) == 0):
    raise SystemExit("Old race winner is not exact F with explicit nonzero Exit_status")
if owner.get("job_id") != old_id or owner.get("job_name") != expected_name:
    raise SystemExit("Current winner receipt no longer binds the failed startup")

sibling_status = attrs(sibling_id)
if (sibling_status.get("Job_Name") != sibling_name
        or sibling_status.get("job_state") not in ("F", "X")
        or sibling_status.get("Exit_status", "").strip()
        or sibling_status.get("stime", "").strip()
        or sibling_status.get("exec_host", "").strip()
        or sibling_status.get("resources_used.walltime", "").strip()):
    raise SystemExit("Old sibling is not exact terminal and never-started")

current_status = attrs(current_id)
if current_status.get("Job_Name") != expected_name or current_status.get("job_state") != "R":
    raise SystemExit("Current startup-repair PBS job is not the identity-verified running winner")

pair = json.loads((root / "control/ours_race/submissions_ready.json").read_text(encoding="utf-8"))
winner_label = "short" if stage == "ours_short" else "long"
sibling_label = "long" if winner_label == "short" else "short"
if (pair.get("protocol") != protocol.get("protocol")
        or pair.get(winner_label, {}).get("job_id") != current_id
        or pair.get(winner_label, {}).get("job_name") != expected_name
        or pair.get(sibling_label, {}).get("job_id") != sibling_id
        or pair.get(sibling_label, {}).get("job_name") != sibling_name):
    raise SystemExit("Current Ours pair receipt does not bind the retry and old sibling")

checkpoint = root / "results/ours/checkpoint.pt"
if checkpoint.exists():
    raise SystemExit("Ours checkpoint.pt exists; refusing startup retry")
summary_path = root / "results/ours/summary.json"
summary = None
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("protocol") not in (None, protocol.get("protocol"))
            or summary.get("pbs_job_id") not in (None, old_id, sibling_id)
            or summary.get("status") in ("complete", "checkpointed")):
        raise SystemExit("Ours summary is unrelated or describes a resumable/completed run")

def check_counts(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("completed_updates", "rotation_completed"):
                try:
                    if float(item) != 0:
                        raise SystemExit("Ours summary records completed updates")
                except (TypeError, ValueError):
                    raise SystemExit("Ours summary update counter is unverifiable")
            else:
                check_counts(item)
    elif isinstance(value, list):
        for item in value:
            check_counts(item)
if summary is not None:
    check_counts(summary)

if marker_path.exists():
    marker["status"] = "claimed"
    marker["claimed_job_id"] = current_id
    marker["claimed_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = marker_path.with_name(marker_path.name + ".tmp." + current_id.replace(".", "_"))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, ensure_ascii=False, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, marker_path)
with open(lock_owner, "x", encoding="utf-8") as stream:
    json.dump({"stage": stage, "job_id": current_id, "startup_repair_from": old_id}, stream)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
  write_startup_repair_owner "$old_id"
  echo "RACE_STARTUP_REPAIR_WINNER stage=$STAGE job=$PBS_JOBID repaired_from=$old_id"
  cancel_queued_sibling "$sibling_id" "$sibling_name" || {
    echo 'Could not verify old sibling terminal state; refusing model load' >&2; return 1;
  }
}

function release_resume_lock() {
  local path=$ROOT/control/ours_race/resume_claim/owner.json
  if [[ -s "$path" ]] && [[ "$(read_field "$path" job_id 2>/dev/null || true)" == "$PBS_JOBID" ]]; then
    rm -f "$path"
    rmdir "$ROOT/control/ours_race/resume_claim" 2>/dev/null || true
  fi
}

function claim_ours_race() {
  local race=$ROOT/control/ours_race lock=$ROOT/control/ours_race/claim owner=$ROOT/control/ours_race/winner.json
  local sibling_id sibling_name old_id old_stage old_name old_state old_exit active=$ROOT/control/ours_race/resume_claim
  local pair
  mkdir -p "$race"
  pair=$(race_ready_pair) || exit 64
  read -r sibling_id sibling_name <<<"$pair"
  if mkdir "$lock" 2>/dev/null; then
    write_race_owner ""
    echo "RACE_WINNER stage=$STAGE job=$PBS_JOBID"
  else
    [[ -s "$owner" ]] || { echo 'Race claim exists without a durable winner receipt; refusing to train' >&2; exit 64; }
    old_id=$(read_field "$owner" job_id)
    old_stage=$(read_field "$owner" stage)
    old_name=$(read_field "$owner" job_name)
    old_state=$(qstat_fields "$old_id") || { echo "Cannot verify prior race owner $old_id" >&2; exit 64; }
    read -r _ old_state old_exit <<<"$old_state"
    local startup_marker=$race/startup_repair.json
    local startup_new_winner
    startup_new_winner=$(read_field "$startup_marker" new_winner.job_id 2>/dev/null || true)
    if [[ -s "$startup_marker" && "$startup_new_winner" == "$PBS_JOBID" ]]; then
      [[ "$old_stage" == "$STAGE" && "$old_state" == F && "$old_exit" =~ ^-?[0-9]+$ && "$old_exit" -ne 0 ]] || {
        echo 'Startup-repair marker does not match the exact failed old winner' >&2; exit 64;
      }
      [[ "$(qstat_fields "$old_id" | awk '{print $1}')" == "$old_name" ]] || {
        echo 'Startup-repair old winner PBS identity mismatch' >&2; exit 64;
      }
      claim_startup_repair "$old_id" "$sibling_id" "$sibling_name" || exit 64
      return 0
    fi
    if [[ "$old_stage" != "$STAGE" || ( "$old_state" != F && "$old_state" != X ) ]] || ! verify_checkpointed_ours; then
      echo "RACE_LOSER stage=$STAGE job=$PBS_JOBID winner=$old_stage/$old_id state=$old_state; no model loaded"
      exit 0
    fi
    [[ "$(qstat_fields "$old_id" | awk '{print $1}')" == "$old_name" ]] || { echo 'Prior winner identity mismatch' >&2; exit 64; }
    if [[ -d "$active" ]]; then
      local active_id active_stage active_fields active_name active_state active_exit
      [[ -s "$active/owner.json" ]] || { echo 'Active resume lock has no receipt; refusing to clear it' >&2; exit 64; }
      active_id=$(read_field "$active/owner.json" job_id)
      active_stage=$(read_field "$active/owner.json" stage)
      [[ "$active_stage" == "$STAGE" ]] || { echo 'Active resume lock names another stage' >&2; exit 64; }
      active_fields=$(qstat_fields "$active_id") || { echo "Cannot verify active resume job $active_id" >&2; exit 64; }
      read -r active_name active_state active_exit <<<"$active_fields"
      [[ "$active_name" == "$(race_job_name "$active_stage")" ]] || { echo 'Active resume PBS identity mismatch' >&2; exit 64; }
      if [[ "$active_state" != F && "$active_state" != X ]]; then
        echo "RACE_LOSER duplicate resume job active=$active_id state=$active_state; no model loaded"
        exit 0
      fi
      rm -f "$active/owner.json"
      rmdir "$active" 2>/dev/null || { echo 'Cannot clear stale resume claim safely' >&2; exit 64; }
    fi
    mkdir "$active" || { echo 'Another resume claimant won the atomic lock' >&2; exit 0; }
    RACE_RESUME_LOCK=1
    write_race_owner "$old_id"
    python3 - "$active/owner.json" "$STAGE" "$PBS_JOBID" <<'PY'
import json, os, sys
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump({"stage": sys.argv[2], "job_id": sys.argv[3]}, stream)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
PY
    echo "RACE_RESUME_WINNER stage=$STAGE job=$PBS_JOBID resumed_from=$old_id"
  fi
  cancel_queued_sibling "$sibling_id" "$sibling_name" || { echo 'Could not verify sibling cancellation/identity; refusing model load' >&2; exit 64; }
}

main "$@"

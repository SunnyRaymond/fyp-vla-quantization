#!/bin/bash -l
set -euo pipefail
TASK_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-native-perf-20261025
OLD_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024
stage_arg=${1:?stage required}
STAGE=${NATIVE_STAGE:-$stage_arg}
case "$STAGE" in prepare|startup|atom_general_build|profile|build|build_fused|hadamard|svd_fused_build|svd_fused_gate|diffusion_repack|svd_fused_repack|toolkit|optimized|analyze|diffusion|capture|probe|probe_all|gates|ours_gate|panel|aggregate|gate_spin|gate_qua|gate_svd|gate_vidit|gate_wide|gate_wide_g1|gate_wide_g2|gate_wide_g3|perf_bf16|perf_spin|perf_qua|perf_svd|perf_vidit|perf_ours|perf_bundle|rotation_parity|rotation_parity_long) ;; *) exit 64 ;; esac
[[ -n "${PBS_JOBID:-}" && -s "${PBS_NODEFILE:-}" ]] || { echo 'Missing PBS identity'; exit 64; }
actual_host=$(hostname -s)
[[ "${actual_host,,}" != *login* ]] || exit 64
grep -Fqx "$actual_host" <(awk -F. '{print $1}' "$PBS_NODEFILE") || exit 64
allocation_status=$(qstat -f "$PBS_JOBID")
grep -Eq 'job_state[[:space:]]*=[[:space:]]*R' <<<"$allocation_status" || exit 64
expected_name=cpnp25prof
[[ "$STAGE" != prepare ]] || expected_name=cpnp25prep
[[ "$STAGE" != startup ]] || expected_name=cpnp25start
[[ "$STAGE" != atom_general_build ]] || expected_name=cpnp25atom
[[ "$STAGE" != build ]] || expected_name=cpnp25build
[[ "$STAGE" != toolkit ]] || expected_name=cpnp25cuda
[[ "$STAGE" != optimized ]] || expected_name=cpnp25opt
[[ "$STAGE" != analyze ]] || expected_name=cpnp25trace
[[ "$STAGE" != diffusion ]] || expected_name=cpnp25diff
[[ "$STAGE" != capture ]] || expected_name=cpnp25cap
[[ "$STAGE" != probe ]] || expected_name=cpnp25probe
[[ "$STAGE" != probe_all ]] || expected_name=cpnp25probeall
[[ "$STAGE" != build_fused ]] || expected_name=cpnp25qepi
[[ "$STAGE" != hadamard ]] || expected_name=cpnp25fht
[[ "$STAGE" != svd_fused_build ]] || expected_name=cpnp25svdf
[[ "$STAGE" != diffusion_repack ]] || expected_name=cpnp25drepack
[[ "$STAGE" != svd_fused_repack ]] || expected_name=cpnp25sfrepack
[[ "$STAGE" != svd_fused_gate ]] || expected_name=cpnp25svdg
[[ "$STAGE" != gates ]] || expected_name=cpnp25gate
[[ "$STAGE" != ours_gate ]] || expected_name=cpnp25ogate
[[ "$STAGE" != panel ]] || expected_name=cpnp25panel
[[ "$STAGE" != aggregate ]] || expected_name=cpnp25agg
[[ "$STAGE" != rotation_parity ]] || expected_name=cpnp25rdiag
[[ "$STAGE" != rotation_parity_long ]] || expected_name=cpnp25rdlong
race_stage=
case "$STAGE" in
  gate_spin) expected_name=cpnp25gspin; single_arm=spinquant_w4a4 ;;
  gate_qua) expected_name=cpnp25gqua; single_arm=quarot_w4a4 ;;
  gate_svd) expected_name=cpnp25gsvd; single_arm=svdquant_w4a4 ;;
  gate_vidit) expected_name=cpnp25gvidit; single_arm=viditq_w4a4 ;;
  gate_wide) expected_name=cpnp25gwide; wide_expected_gpus=6; wide_soft_seconds=6600 ;;
  gate_wide_g1) expected_name=cpnp25gw1; wide_expected_gpus=1; wide_soft_seconds=13800 ;;
  gate_wide_g2) expected_name=cpnp25gw2; wide_expected_gpus=3; wide_soft_seconds=8460 ;;
  gate_wide_g3) expected_name=cpnp25gw3; wide_expected_gpus=4; wide_soft_seconds=8460 ;;
  perf_bf16) expected_name=cpnp25pbf16; single_arm=bf16 ;;
  perf_spin) expected_name=cpnp25pspin; single_arm=spinquant_w4a4; race_stage=gate_spin ;;
  perf_qua) expected_name=cpnp25pqua; single_arm=quarot_w4a4; race_stage=gate_qua ;;
  perf_svd) expected_name=cpnp25psvd; single_arm=svdquant_w4a4; race_stage=gate_svd ;;
  perf_vidit) expected_name=cpnp25pvidit; single_arm=viditq_w4a4; race_stage=gate_vidit ;;
  perf_ours) expected_name=cpnp25pours; single_arm=ours_vqwam; race_stage=ours_gate ;;
  perf_bundle) expected_name=cpnp25pbundle; race_stage=gate_wide ;;
esac
grep -Eq "Job_Name[[:space:]]*=[[:space:]]*$expected_name$" <<<"$allocation_status" || exit 64
ARTIFACTS=$TASK_ROOT/artifacts/$PBS_JOBID
mkdir -p "$ARTIFACTS/source" "$ARTIFACTS/tmp"
exec > >(tee -a "$ARTIFACTS/job.log") 2>&1
monitor_pid=
finish() {
  rc=$?
  trap - EXIT
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
  printf '%s\n' "$rc" > "$ARTIFACTS/exit_code.txt"
  [[ "$rc" != 0 ]] || touch "$ARTIFACTS/PIPELINE_COMPLETE"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
echo "PBS_JOBID=$PBS_JOBID HOST=$actual_host STAGE=$STAGE"
for source in "$TASK_ROOT"/*.py "$TASK_ROOT"/*.sh "$TASK_ROOT"/*.json "$TASK_ROOT"/*.cpp "$TASK_ROOT"/*.cu "$TASK_ROOT"/*.h; do
  [[ -f "$source" ]] && cp "$source" "$ARTIFACTS/source/"
done
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0 cuda/12.2.2
if [[ "$STAGE" == prepare ]]; then
  python3 "$ARTIFACTS/source/prepare_sources.py" --root "$TASK_ROOT"
  command -v nvcc || true
  nvcc --version || true
  exit 0
fi
if [[ "$STAGE" == toolkit ]]; then
  python3 "$ARTIFACTS/source/prepare_toolkit.py" --root "$TASK_ROOT"
  exit 0
fi
if [[ "$STAGE" == analyze ]]; then
  python3 "$ARTIFACTS/source/analyze_profile.py" --root "$TASK_ROOT"
  exit 0
fi
if [[ "$STAGE" == aggregate ]]; then
  python3 "$ARTIFACTS/source/aggregate_native.py" --root "$TASK_ROOT"
  exit 0
fi
if [[ "$STAGE" == build || "$STAGE" == diffusion || "$STAGE" == diffusion_repack || "$STAGE" == build_fused || "$STAGE" == hadamard || "$STAGE" == svd_fused_build || "$STAGE" == svd_fused_repack || "$STAGE" == atom_general_build ]]; then
  CUDA_TOOLKIT=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
  builder=build_native.py
  builder_args=(--root "$TASK_ROOT")
  [[ "$STAGE" != build_fused ]] || builder_args+=(--fused)
  if [[ "$STAGE" == diffusion || "$STAGE" == diffusion_repack || "$STAGE" == hadamard || "$STAGE" == svd_fused_build || "$STAGE" == svd_fused_repack || "$STAGE" == atom_general_build ]]; then
    CUDA_TOOLKIT=$TASK_ROOT/toolkits/cuda-12.8.1
    [[ -s "$TASK_ROOT/results/preparation/toolkit.json" && -x "$CUDA_TOOLKIT/bin/nvcc" ]] || exit 64
    builder=build_diffusion.py
    [[ "$STAGE" != hadamard ]] || builder=build_hadamard.py
    [[ "$STAGE" != svd_fused_build && "$STAGE" != svd_fused_repack ]] || builder=build_svd_fused.py
    [[ "$STAGE" != atom_general_build ]] || builder=build_atom_general.py
  fi
  if [[ "$STAGE" == diffusion_repack || "$STAGE" == svd_fused_repack ]]; then
    receipt_name=native_diffusion_build.json
    [[ "$STAGE" != svd_fused_repack ]] || receipt_name=native_svd_fused_build.json
    python3 - "$TASK_ROOT/results/preparation/$receipt_name" "$ARTIFACTS/previous_receipts" <<'PY'
import json, pathlib, shutil, sys
source = pathlib.Path(sys.argv[1])
if source.is_file() and json.loads(source.read_text()).get("status") == "complete":
    target = pathlib.Path(sys.argv[2]) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"Preserved previous successful build receipt: {target}")
PY
  fi
  SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
  PYTHON=/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015/env/bin/python
  echo "CUDA_TOOLKIT=$CUDA_TOOLKIT"
  apptainer exec --cleanenv --bind /scratch,/scratch/GPFS/app/apps:/app/apps \
    --bind "$CUDA_TOOLKIT:$CUDA_TOOLKIT:ro" --bind "$PBS_NODEFILE:$PBS_NODEFILE:ro" \
    --bind "$ARTIFACTS/tmp:/tmp:rw" \
    --env "PBS_JOBID=$PBS_JOBID,PBS_NODEFILE=$PBS_NODEFILE,CUDA_HOME=$CUDA_TOOLKIT" \
    --env "PATH=$TASK_ROOT/native_dependencies/bin:$CUDA_TOOLKIT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "LD_LIBRARY_PATH=$CUDA_TOOLKIT/lib64:/usr/lib/x86_64-linux-gnu" \
    --env "HOME=/tmp,PYTHONUNBUFFERED=1,MAX_JOBS=2,TORCH_CUDA_ARCH_LIST=8.0" \
    "$SIF" "$PYTHON" "$ARTIFACTS/source/$builder" "${builder_args[@]}"
  exit 0
fi
if [[ "$STAGE" == startup ]]; then
  export COSMOS_CPU_ONLY=1
else
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 64
  nvidia-smi -L
  if [[ -n "$race_stage" ]]; then
    race_root=$TASK_ROOT/control/formal_n40_queue_race
    if [[ -f "$race_root/queue_race.json" ]]; then
      gpu_names=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader)
      while IFS= read -r gpu_name; do
        [[ "$gpu_name" == 'NVIDIA A100-SXM4-40GB' ]] || { echo 'Formal N40 race requires the frozen A100 GPU SKU'; exit 64; }
      done <<<"$gpu_names"
      python3 "$ARTIFACTS/source/queue_race.py" --root "$race_root" --stage "$race_stage"
    fi
  fi
  if [[ "$STAGE" == gate_wide* ]]; then
    IFS=, read -r -a wide_gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
    [[ ${#wide_gpu_ids[@]} -eq $wide_expected_gpus ]] || { echo 'Wide allocation GPU count mismatch'; exit 64; }
  fi
  if [[ -z "$race_stage" && -f "$TASK_ROOT/queue_race.json" ]]; then
    gpu_names=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader)
    while IFS= read -r gpu_name; do
      [[ "$gpu_name" == 'NVIDIA A100-SXM4-40GB' ]] || { echo 'Race requires the frozen A100 GPU SKU'; exit 64; }
    done <<<"$gpu_names"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    python3 "$ARTIFACTS/source/queue_race.py" --root "$TASK_ROOT" --stage "$STAGE"
  fi
  if [[ "$STAGE" == rotation_parity* ]]; then
    mkdir -p "$TASK_ROOT/control"
    parity_claim=$TASK_ROOT/control/rotation_parity_v1_claim
    mkdir "$parity_claim" || exit 75
    printf '%s\n' "$PBS_JOBID" > "$parity_claim/owner.txt"
  fi
  (
    while true; do
      printf 'GPU_TELEMETRY utc=%s\n' "$(date -u +%FT%TZ)"
      nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
      sleep 15
    done
  ) &
  monitor_pid=$!
fi
export COSMOS_ROOT=$OLD_ROOT COSMOS_CODE_ROOT=$OLD_ROOT
export ASSET_ROOT=/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015
export LEGACY_ROOT=$ASSET_ROOT
export ARTIFACTS_TMP=$ARTIFACTS/tmp ARTIFACTS
export COSMOS_JOB_START_EPOCH=$(date +%s) COSMOS_SOFT_DEADLINE_SECONDS=6600 COSMOS_CPU_THREADS=8
[[ "$STAGE" != gate_wide* ]] || export COSMOS_SOFT_DEADLINE_SECONDS=$wide_soft_seconds
[[ "$STAGE" != startup ]] || export COSMOS_CPU_THREADS=4
[[ "$STAGE" != capture ]] || export COSMOS_SOFT_DEADLINE_SECONDS=3300
mkdir -p "$ARTIFACTS/source_helpers" "$ARTIFACTS/legacy_sources" "$ARTIFACTS/engine_sources"
python3 - "$OLD_ROOT/protocol.json" "$ARTIFACTS" <<'PY'
import json, pathlib, shutil, sys
p=json.load(open(sys.argv[1])); out=pathlib.Path(sys.argv[2])
for name, value in p['source_paths']['helpers'].items():
    shutil.copyfile(value, out/'source_helpers'/f'{name}.py')
for name in ['endpoint.py','calibrate.py','quantization.py','cosmos_runtime.py']:
    shutil.copyfile(pathlib.Path(p['asset_root'])/name, out/'legacy_sources'/name)
for name in ['fused_input.py','layer_runtime.py','fused_vq.py','decoder_runtime.py']:
    shutil.copyfile(pathlib.Path(p['source_paths']['engine'])/name, out/'engine_sources'/name)
PY
export COSMOS_HELPERS_PYTHONPATH=$ARTIFACTS/source:$ARTIFACTS/source_helpers:/scratch/users/ntu/yguo017/fastwam-vq-calibration-suites-20261013
source "$OLD_ROOT/environment.sh"
if [[ "$STAGE" == rotation_parity* ]]; then
  run_python "$ARTIFACTS/source/diagnose_rotation_parity.py" \
    --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/rotation_parity"
  exit 0
fi
if [[ "$STAGE" == startup ]]; then
  run_python "$ARTIFACTS/source/startup_imports.py" --out "$ARTIFACTS/work/startup"
  exit 0
fi
if [[ "$STAGE" == gate_wide* ]]; then
  wide_arms=(spinquant_w4a4 quarot_w4a4 svdquant_w4a4 viditq_w4a4 ours_vqwam bf16)
  wide_gpu_count=${#wide_gpu_ids[@]}
  wide_pids=()
  for ((worker=0; worker<wide_gpu_count; worker++)); do
    (
      trap - EXIT
      export CUDA_VISIBLE_DEVICES=${wide_gpu_ids[$worker]}
      export COSMOS_CPU_THREADS=8
      worker_rc=0
      for ((index=worker; index<6; index+=wide_gpu_count)); do
        if [[ ${wide_arms[$index]} == bf16 ]]; then
          run_python "$ARTIFACTS/source/panel_arm.py" --arm bf16 --variant current \
            --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/perf_bf16" \
            > "$ARTIFACTS/wide_worker_$index.log" 2>&1 || worker_rc=1
        else
          run_python "$ARTIFACTS/source/gates_worker.py" --arm "${wide_arms[$index]}" \
            --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/gate_${wide_arms[$index]}" \
            > "$ARTIFACTS/wide_worker_$index.log" 2>&1 || worker_rc=1
        fi
      done
      exit "$worker_rc"
    ) &
    wide_pids+=("$!")
  done
  wide_rc=0
  for pid in "${wide_pids[@]}"; do wait "$pid" || wide_rc=1; done
  exit "$wide_rc"
fi
if [[ "$STAGE" == gate_* ]]; then
  run_python "$ARTIFACTS/source/gates_worker.py" --arm "$single_arm" \
    --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/gate_$single_arm"
  exit 0
fi
if [[ "$STAGE" == perf_bundle ]]; then
  bundle_arms=(spinquant_w4a4 quarot_w4a4 svdquant_w4a4 viditq_w4a4 ours_vqwam)
  bundle_variants=(current current current current optimized)
  bundle_rc=0
  for index in "${!bundle_arms[@]}"; do
    arm=${bundle_arms[$index]}
    variant=${bundle_variants[$index]}
    log="$ARTIFACTS/perf_bundle_$arm.log"
    printf 'PERF_BUNDLE_STAGE PBS_JOBID=%s arm=%s variant=%s\n' \
      "$PBS_JOBID" "$arm" "$variant" > "$log"
    if run_python "$ARTIFACTS/source/panel_arm.py" --arm "$arm" --variant "$variant" \
      --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/perf_bundle_$arm" \
      >> "$log" 2>&1; then
      printf 'PERF_BUNDLE_RESULT arm=%s status=complete\n' "$arm" >> "$log"
    else
      arm_rc=$?
      printf 'PERF_BUNDLE_RESULT arm=%s status=failed rc=%s\n' "$arm" "$arm_rc" >> "$log"
      bundle_rc=1
    fi
  done
  exit "$bundle_rc"
fi
if [[ "$STAGE" == perf_* ]]; then
  variant=$(python3 - "$TASK_ROOT/native_protocol.json" "$single_arm" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['selected_variants'][sys.argv[2]])
PY
  )
  run_python "$ARTIFACTS/source/panel_arm.py" --arm "$single_arm" --variant "$variant" \
    --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/perf_$single_arm"
  exit 0
fi
run_arm_gate() {
  local arm=$1 variant=$2 scope
  for scope in sampler dit_replay; do
    run_python "$ARTIFACTS/source/perf_native.py" --mode gate --old-root "$OLD_ROOT" --root "$TASK_ROOT" \
      --out "$ARTIFACTS/work/gate_${arm}_${variant}_${scope}" --arm "$arm" --variant "$variant" --scope "$scope" || return $?
  done
}
run_svd_fused_work() {
  local core_gate_rc=${1:-0}
  run_python "$ARTIFACTS/source/probe_svd_fused.py" --root "$TASK_ROOT" --out "$TASK_ROOT/results/preparation/svd_fused_kernel_probe.json" || return $?
  run_arm_gate svdquant_w4a4 native_fused || return $?
  if [[ "$core_gate_rc" == 0 ]]; then
    run_python "$ARTIFACTS/source/compare_svd_runtime.py" --old-root "$OLD_ROOT" --root "$TASK_ROOT" \
      --out "$ARTIFACTS/work/svd_fusion_diagnostic" || return $?
  else
    echo 'SVD_DIAGNOSTIC_SKIPPED: core fidelity gate failed; fused fidelity evidence retained separately'
  fi
}
if [[ "$STAGE" == gates ]]; then
  IFS=, read -r -a allocated_gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
  [[ ${#allocated_gpu_ids[@]} -eq 4 ]] || { echo 'Four actually allocated GPUs required'; exit 64; }
  gate_arms=(spinquant_w4a4 quarot_w4a4 svdquant_w4a4 viditq_w4a4)
  gate_pids=()
  for index in 0 1 2 3; do
    (
      trap - EXIT
      export CUDA_VISIBLE_DEVICES=${allocated_gpu_ids[$index]}
      export COSMOS_CPU_THREADS=8
      run_python "$ARTIFACTS/source/gates_worker.py" --arm "${gate_arms[$index]}" \
        --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/gate_${gate_arms[$index]}"
    ) > "$ARTIFACTS/gate_worker_$index.log" 2>&1 &
    gate_pids+=("$!")
  done
  gate_rc=0
  for pid in "${gate_pids[@]}"; do wait "$pid" || gate_rc=1; done
  exit "$gate_rc"
fi
if [[ "$STAGE" == svd_fused_gate ]]; then
  run_svd_fused_work
  exit 0
fi
if [[ "$STAGE" == ours_gate ]]; then
  run_python "$ARTIFACTS/source/gates_worker.py" --arm ours_vqwam \
    --old-root "$OLD_ROOT" --root "$TASK_ROOT" --out "$ARTIFACTS/work/gate_ours_vqwam"
  exit 0
fi
if [[ "$STAGE" == panel ]]; then
  python3 - "$TASK_ROOT/native_protocol.json" "$ARTIFACTS/source" "$TASK_ROOT" <<'PY'
import json, pathlib, sys
sys.path.insert(0, sys.argv[2])
from perf_native import ARMS, _require_gate, validate_variant
p=json.load(open(sys.argv[1]))
if p.get('selection_status') != 'ready_for_formal_panel':
    raise RuntimeError('Native backend selection is not ready for the formal panel')
for arm in ARMS:
    variant=p['selected_variants'][arm]
    validate_variant(arm, variant)
    _require_gate(pathlib.Path(sys.argv[3]), arm, variant, 'sampler')
PY
  run_python "$ARTIFACTS/source/panel_native.py" --old-root "$OLD_ROOT" --root "$TASK_ROOT" \
    --out "$ARTIFACTS/work/panel"
  python3 "$ARTIFACTS/source/aggregate_native.py" --root "$TASK_ROOT"
  exit 0
fi
if [[ "$STAGE" == capture ]]; then
  run_python "$ARTIFACTS/source/perf_native.py" --mode capture --old-root "$OLD_ROOT" --root "$TASK_ROOT" \
    --out "$ARTIFACTS/work" --arm bf16 --variant current --scope dit_replay --soft-deadline-seconds "$COSMOS_SOFT_DEADLINE_SECONDS"
  exit 0
fi
if [[ "$STAGE" == probe ]]; then
  run_python "$ARTIFACTS/source/kernel_probe.py" --root "$TASK_ROOT" --out "$TASK_ROOT/results/preparation/kernel_probe.json"
  exit 0
fi
if [[ "$STAGE" == probe_all ]]; then
  run_python "$ARTIFACTS/source/probe_all_worker.py" --root "$TASK_ROOT"
  exit 0
fi
if [[ "$STAGE" == optimized ]]; then
  run_python "$ARTIFACTS/source/optimize_worker.py" --old-root "$OLD_ROOT" --root "$TASK_ROOT"
  exit 0
fi
for arm in bf16 ours_vqwam; do
  run_python "$ARTIFACTS/source/engine_profile.py" --old-root "$OLD_ROOT" --root "$TASK_ROOT" \
    --out "$TASK_ROOT/results/profile_initial/$arm" --arm "$arm" --profile-only
done

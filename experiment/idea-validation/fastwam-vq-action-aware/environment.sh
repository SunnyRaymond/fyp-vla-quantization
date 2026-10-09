# Source only after the PBS allocation guard in the calling script.
module purge
module load gcc/11.4.0-nscc python/3.11.7-gcc11 apptainer/1.5.0
NV_ARGS=()
if [[ "${FW_REQUIRE_GPU:-0}" == 1 ]]; then
  module load cuda/12.2.2
  NV_ARGS=(--nv)
else
  export CUDA_VISIBLE_DEVICES=''
fi
mapfile -t ENV_PATHS < <(python3 - "$ROOT/protocol.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
r = d["roots"]
base, pilot, source, banks = (Path(r[k]) for k in ("base", "pilot", "old_inputs", "old_banks"))
env_path = pilot / "env_status.json"
status = json.loads(env_path.read_text(encoding="utf-8"))
if status.get("checks", {}).get("render_preflight_ready") is not True:
    raise SystemExit("Actual LIBERO-Plus rendering preflight is not ready")
p = status.get("paths", status)
native = json.loads(Path(p["native_setup_receipt"]).read_text(encoding="utf-8"))
if native.get("ready") is not True:
    raise SystemExit("Native ImageMagick setup is not ready")
aliases = (
    ("plus_repo", "libero_plus", "libero_plus_root", "plus_root"),
    ("dependency_target", "deps", "dependencies", "dependency_dir"),
    ("libero_config_path", "libero_config", "config_dir"),
    ("libero_compat",), ("native_root",), ("native_library_dir",), ("magick_home",),
    ("magick_configure_path",), ("magick_coder_module_path",), ("magick_filter_module_path",),
    ("native_setup_receipt",),
)
print(base); print(pilot); print(source); print(banks)
for names in aliases:
    value = next((p.get(k) for k in names if p.get(k)), None)
    if not value:
        raise SystemExit(f"env_status.json lacks one of {names}")
    if names[0] in ("magick_configure_path", "magick_filter_module_path"):
        print(value)
        continue
    path = Path(value)
    if not path.is_absolute(): path = pilot / path
    if not path.exists(): raise SystemExit(f"Configured environment path is unavailable: {path}")
    print(path)
PY
)
test "${#ENV_PATHS[@]}" -eq 15 || { echo 'Could not resolve LIBERO-Plus environment paths' >&2; exit 65; }
BASE=${ENV_PATHS[0]}
PILOT_ROOT=${ENV_PATHS[1]}
SOURCE_ROOT=${ENV_PATHS[2]}
OLD_BANK_ROOT=${ENV_PATHS[3]}
PLUS_ROOT=${ENV_PATHS[4]}
DEPS=${ENV_PATHS[5]}
LIBERO_CONFIG=${ENV_PATHS[6]}
COMPAT=${ENV_PATHS[7]}
NATIVE_ROOT=${ENV_PATHS[8]}
NATIVE_LIBDIR=${ENV_PATHS[9]}
MAGICK_HOME=${ENV_PATHS[10]}
MAGICK_CONFIGURE_PATH=${ENV_PATHS[11]}
MAGICK_CODER_MODULE_PATH=${ENV_PATHS[12]}
MAGICK_FILTER_MODULE_PATH=${ENV_PATHS[13]}
NATIVE_RECEIPT=${ENV_PATHS[14]}
DIAG_ROOT=$ROOT
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
export BASE PILOT_ROOT SOURCE_ROOT OLD_BANK_ROOT PLUS_ROOT DEPS LIBERO_CONFIG COMPAT
export NATIVE_ROOT NATIVE_LIBDIR MAGICK_HOME MAGICK_CONFIGURE_PATH MAGICK_CODER_MODULE_PATH
export MAGICK_FILTER_MODULE_PATH NATIVE_RECEIPT DIAG_ROOT SIF ARTIFACTS ROOT
export LD_LIBRARY_PATH="$NATIVE_LIBDIR:$NATIVE_ROOT/usr/lib:/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:${LD_LIBRARY_PATH:-}"
test -d "$PLUS_ROOT" -a -d "$DEPS" -a -d "$LIBERO_CONFIG" -a -d "$COMPAT"
test -d "$BASE/FastWAM" -a -d "$BASE/checkpoints"

export FW_REQUIRE_REAL_QUANT=1
run_python() {
  apptainer exec --cleanenv "${NV_ARGS[@]}" \
    --bind /scratch,/scratch/GPFS/app/apps:/app/apps \
    --bind "$PBS_NODEFILE:$PBS_NODEFILE:ro" \
    --bind /usr/lib64/libffi.so.6:/usr/lib/x86_64-linux-gnu/libffi.so.6 \
    --bind /usr/lib64/libssl.so.1.1:/usr/lib/x86_64-linux-gnu/libssl.so.1.1 \
    --bind /usr/lib64/libcrypto.so.1.1:/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 \
    --env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    --env "MAGICK_HOME=$MAGICK_HOME,MAGICK_CONFIGURE_PATH=$MAGICK_CONFIGURE_PATH,MAGICK_CODER_MODULE_PATH=$MAGICK_CODER_MODULE_PATH,MAGICK_FILTER_MODULE_PATH=$MAGICK_FILTER_MODULE_PATH" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env "PBS_JOBID=$PBS_JOBID,PBS_NODEFILE=$PBS_NODEFILE,ARTIFACTS=$ARTIFACTS,ROOT=$ROOT,DIAG_ROOT=$DIAG_ROOT,SOURCE_ROOT=$SOURCE_ROOT,OLD_BANK_ROOT=$OLD_BANK_ROOT,BASE=$BASE,PILOT_ROOT=$PILOT_ROOT" \
    --env "PYTHONPATH=$ARTIFACTS:$ROOT:$PILOT_ROOT:$SOURCE_ROOT:$OLD_BANK_ROOT:$COMPAT:$PLUS_ROOT:$DEPS:$BASE/FastWAM:$BASE/FastWAM/src:$BASE/FastWAM/experiments/libero" \
    --env "LIBERO_CONFIG_PATH=$LIBERO_CONFIG" \
    --env "DIFFSYNTH_MODEL_BASE_PATH=$BASE/model-cache" \
    --env DIFFSYNTH_DOWNLOAD_SOURCE=huggingface,DIFFSYNTH_SKIP_DOWNLOAD=true,HF_HUB_OFFLINE=1 \
    --env "HF_HOME=$BASE/cache/huggingface" \
    --env MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LIBGL_ALWAYS_SOFTWARE=1 \
    --env "OMP_NUM_THREADS=${FW_CPU_THREADS:-8},OPENBLAS_NUM_THREADS=${FW_CPU_THREADS:-8},LP_NUM_THREADS=${FW_CPU_THREADS:-8}" \
    --env GIT_PYTHON_REFRESH=quiet,PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false \
    --env FW_REQUIRE_REAL_QUANT=1 \
    "$SIF" "$BASE/venv/bin/python" "$@"
}

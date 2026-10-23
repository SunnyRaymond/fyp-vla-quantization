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

mapfile -t ENV_PATHS < <(python3 - "$ARTIFACTS/protocol.json" <<'PY'
import json, sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
roots = protocol["roots"]
base, pilot = Path(roots["base"]), Path(roots["pilot"])
source, old_inputs, old_banks = (Path(protocol["source_root"]),
                                 Path(roots["old_inputs"]), Path(roots["old_banks"]))
if str(base) != "/scratch/users/ntu/yguo017/fastwam-smoke":
    raise SystemExit(f"Unexpected model base: {base}")
status = json.loads((pilot / "env_status.json").read_text(encoding="utf-8"))
if status.get("checks", {}).get("render_preflight_ready") is not True:
    raise SystemExit("Original-LIBERO render environment is not ready")
paths = status.get("paths", status)
native = json.loads(Path(paths["native_setup_receipt"]).read_text(encoding="utf-8"))
if native.get("ready") is not True:
    raise SystemExit("Native ImageMagick setup is not ready")
for value in (base, pilot, source, old_inputs, old_banks):
    if not value.is_dir():
        raise SystemExit(f"Configured campaign root is unavailable: {value}")
native_names = ("native_root", "native_library_dir", "magick_home", "magick_configure_path",
                "magick_coder_module_path", "magick_filter_module_path", "native_setup_receipt")
for name in native_names:
    value = paths.get(name)
    if not value:
        raise SystemExit(f"env_status.json lacks {name}")
    path = Path(value)
    if name not in ("magick_configure_path", "magick_filter_module_path") and not path.exists():
        raise SystemExit(f"Configured rendering path is unavailable: {path}")
print(base); print(pilot); print(source); print(old_inputs); print(old_banks)
for name in native_names:
    print(paths[name])
PY
)
test "${#ENV_PATHS[@]}" -eq 12 || { echo 'Could not resolve original-LIBERO runtime paths' >&2; exit 65; }
BASE=${ENV_PATHS[0]}
PILOT_ROOT=${ENV_PATHS[1]}
SOURCE_ROOT=${ENV_PATHS[2]}
OLD_INPUTS=${ENV_PATHS[3]}
OLD_BANK_ROOT=${ENV_PATHS[4]}
NATIVE_ROOT=${ENV_PATHS[5]}
NATIVE_LIBDIR=${ENV_PATHS[6]}
MAGICK_HOME=${ENV_PATHS[7]}
MAGICK_CONFIGURE_PATH=${ENV_PATHS[8]}
MAGICK_CODER_MODULE_PATH=${ENV_PATHS[9]}
MAGICK_FILTER_MODULE_PATH=${ENV_PATHS[10]}
NATIVE_RECEIPT=${ENV_PATHS[11]}
DIAG_ROOT=$ROOT
SIF=/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif
ORIGINAL_LIBERO=$BASE/libero/libero
ORIGINAL_COMPAT=$BASE/libero-compat
DEPS=$PILOT_ROOT/deps
export BASE PILOT_ROOT SOURCE_ROOT OLD_INPUTS OLD_BANK_ROOT DIAG_ROOT SIF ORIGINAL_LIBERO ORIGINAL_COMPAT DEPS
export NATIVE_ROOT NATIVE_LIBDIR MAGICK_HOME MAGICK_CONFIGURE_PATH MAGICK_CODER_MODULE_PATH
export MAGICK_FILTER_MODULE_PATH NATIVE_RECEIPT
export LD_LIBRARY_PATH="$NATIVE_LIBDIR:$NATIVE_ROOT/usr/lib:/usr/lib/x86_64-linux-gnu:/.singularity.d/libs:${LD_LIBRARY_PATH:-}"
HF_HUB_OFFLINE=1
if [[ "${FW_ALLOW_DATA_HTTP:-0}" == 1 ]]; then HF_HUB_OFFLINE=0; fi
HTTP_CA_ARGS=()
if [[ "${FW_ALLOW_DATA_HTTP:-0}" == 1 ]]; then
  HOST_CA_FILE=$(python3 -c 'import ssl; print(ssl.get_default_verify_paths().cafile or "")')
  test -n "$HOST_CA_FILE" -a -s "$HOST_CA_FILE" || { echo 'Trusted host CA bundle is unavailable' >&2; exit 65; }
  # The host verifies public HTTPS; the container lacks its current CA bundle.
  HTTP_CA_ARGS=(--bind "$HOST_CA_FILE:/etc/calibration-host-ca.pem:ro"
                --env SSL_CERT_FILE=/etc/calibration-host-ca.pem,REQUESTS_CA_BUNDLE=/etc/calibration-host-ca.pem)
fi
test -d "$BASE/FastWAM" -a -d "$BASE/venv" -a -d "$ORIGINAL_LIBERO" \
  -a -d "$ORIGINAL_COMPAT" -a -d "$BASE/libero-config" -a -d "$DEPS"

export FW_REQUIRE_REAL_QUANT=1
run_python() {
  apptainer exec --cleanenv "${NV_ARGS[@]}" "${HTTP_CA_ARGS[@]}" \
    --bind /scratch,/scratch/GPFS/app/apps:/app/apps \
    --bind "$PBS_NODEFILE:$PBS_NODEFILE:ro" \
    --bind /usr/lib64/libffi.so.6:/usr/lib/x86_64-linux-gnu/libffi.so.6 \
    --bind /usr/lib64/libssl.so.1.1:/usr/lib/x86_64-linux-gnu/libssl.so.1.1 \
    --bind /usr/lib64/libcrypto.so.1.1:/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1 \
    --env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    --env "MAGICK_HOME=$MAGICK_HOME,MAGICK_CONFIGURE_PATH=$MAGICK_CONFIGURE_PATH,MAGICK_CODER_MODULE_PATH=$MAGICK_CODER_MODULE_PATH,MAGICK_FILTER_MODULE_PATH=$MAGICK_FILTER_MODULE_PATH" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --env "PBS_JOBID=$PBS_JOBID,PBS_NODEFILE=$PBS_NODEFILE,ARTIFACTS=$ARTIFACTS,ROOT=$ROOT,DIAG_ROOT=$DIAG_ROOT,SOURCE_ROOT=$SOURCE_ROOT,OLD_INPUTS=$OLD_INPUTS,OLD_BANK_ROOT=$OLD_BANK_ROOT,BASE=$BASE,PILOT_ROOT=$PILOT_ROOT,FW_JOB_START_EPOCH=$FW_JOB_START_EPOCH,FW_SOFT_DEADLINE_SECONDS=${FW_SOFT_DEADLINE_SECONDS:-6300}" \
    --env "PYTHONPATH=$ARTIFACTS:$ROOT:$ORIGINAL_COMPAT:$ORIGINAL_LIBERO:$DEPS:$PILOT_ROOT:$SOURCE_ROOT:$OLD_INPUTS:$OLD_BANK_ROOT:$BASE/FastWAM:$BASE/FastWAM/src:$BASE/FastWAM/experiments/libero" \
    --env "LIBERO_CONFIG_PATH=$BASE/libero-config" \
    --env "DIFFSYNTH_MODEL_BASE_PATH=$BASE/model-cache" \
    --env "DIFFSYNTH_DOWNLOAD_SOURCE=huggingface,DIFFSYNTH_SKIP_DOWNLOAD=true,HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
    --env "HF_HOME=$BASE/cache/huggingface" \
    --env MUJOCO_GL=osmesa,PYOPENGL_PLATFORM=osmesa,LIBGL_ALWAYS_SOFTWARE=1 \
    --env "OMP_NUM_THREADS=${FW_CPU_THREADS:-8},OPENBLAS_NUM_THREADS=${FW_CPU_THREADS:-8},LP_NUM_THREADS=${FW_CPU_THREADS:-8}" \
    --env GIT_PYTHON_REFRESH=quiet,PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false \
    --env FW_REQUIRE_REAL_QUANT=1 \
    "$SIF" "$BASE/venv/bin/python" "$@"
}

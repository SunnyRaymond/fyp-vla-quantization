#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot installer for the explicitly rented AutoDL RTX 4090 host.
# It refuses a different host, any PBS context, or an existing target tree.
readonly EXPECTED_HOST="autodl-container-e7e742ba1d-c91a1edb"
readonly BASE="/root/autodl-tmp/rolling-ball-lewm"
readonly BASE_PARENT="/root/autodl-tmp"
readonly CONDA_BIN="/root/miniconda3/bin/conda"
readonly LAB_TAG="v2.3.1"
readonly STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly LOG="${BASE_PARENT}/rolling-ball-lewm-install-${STAMP}.log"
readonly EXIT_FILE="${BASE_PARENT}/rolling-ball-lewm-install-${STAMP}.exit_code"
stage="preflight"
mode="gpu"
resume_cpu=0

mkdir -p "$BASE_PARENT"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; printf "exit_code=%s stage=%s finished_utc=%s\n" "$rc" "$stage" "$(date -u +%FT%TZ)" | tee -a "$LOG"; printf "%s\n" "$rc" > "$EXIT_FILE"; exit "$rc"' EXIT

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
set_stage() { stage="$1"; printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$stage"; }
cached_torch_dependencies() {
  "$SIM_ENV/bin/python" - "$1" "$BASE/cache/wheels" <<'PY'
import email, sys, zipfile
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import parse_wheel_filename
wheel=Path(sys.argv[1])
if not wheel.is_file(): raise SystemExit(0)
with zipfile.ZipFile(wheel) as z:
    metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
cache=[]
for p in Path(sys.argv[2]).glob('*.whl'):
    name, version, _, _ = parse_wheel_filename(p.name)
    cache.append((name, version, p))
for value in metadata.get_all('Requires-Dist', []):
    req=Requirement(value)
    if req.marker and not req.marker.evaluate(): continue
    matches=[(v,p) for n,v,p in cache if n==req.name.replace('_','-').lower() and v in req.specifier]
    if matches: print(max(matches)[1])
PY
}
if [[ "${1:-}" == "--prepare-cpu" ]]; then
  mode="cpu-preparation"
  shift
elif [[ "${1:-}" == "--resume-cpu" ]]; then
  # Only resume the interrupted SDK stage on this exact task host.
  mode="cpu-preparation"
  resume_cpu=1
  shift
fi
[[ "$#" -eq 0 ]] || fail "usage: bash install_rented.sh [--prepare-cpu|--resume-cpu]"

set_stage host-and-allocation-guard
[[ "$(uname -s)" == Linux ]] || fail "Linux is required"
[[ "$(hostname)" == "$EXPECTED_HOST" ]] || fail "expected hostname $EXPECTED_HOST; got $(hostname)"
[[ -z "${PBS_JOBID:-}" && -z "${PBS_NODEFILE:-}" ]] || fail "PBS variables are set; this installer is for the rented host only"
[[ "$(id -u)" -eq 0 ]] || fail "run as root on the rented container"
[[ -r /etc/os-release ]] || fail "cannot identify the OS"
grep -q '^ID=ubuntu$' /etc/os-release || fail "Ubuntu is required"
grep -q '^VERSION_ID="22.04"$' /etc/os-release || fail "Ubuntu 22.04 is required"
[[ "$(uname -m)" == x86_64 ]] || fail "x86_64 is required"

set_stage gpu-and-storage-guard
if [[ "$mode" == "cpu-preparation" ]]; then
  gpu_label="CPU_PREPARATION_NOT_EVALUATED"
  printf 'gpu=%s\n' "$gpu_label"
else
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
  mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
  [[ "${#gpu_rows[@]}" -eq 1 ]] || fail "expected exactly one visible GPU; found ${#gpu_rows[@]}"
  [[ "${gpu_rows[0]}" == *"RTX 4090"* ]] || fail "expected RTX 4090; got ${gpu_rows[0]}"
  gpu_label="${gpu_rows[0]}"
  printf 'gpu=%s\n' "$gpu_label"
fi
if (( resume_cpu )); then
  [[ -x "$BASE/envs/sim/bin/python" ]] || fail "resume requires the existing simulator Python"
  "$BASE/envs/sim/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11)'
  [[ -s "$BASE/omniverse/config/omniverse.toml" ]] || fail "resume requires the existing task cache config"
else
  [[ ! -e "$BASE/envs/sim" && ! -e "$BASE/omniverse/config/omniverse.toml" ]] || fail "refusing to overwrite the simulator environment or cache config"
fi
for owned_path in "$BASE/envs/policy" "$BASE/src/IsaacLab-v2.3.1" "$BASE/install_record.txt"; do
  [[ ! -e "$owned_path" ]] || fail "refusing to overwrite existing installation path: $owned_path"
done
avail_target_kb=$(df -Pk "$BASE_PARENT" | awk 'END {print $4}')
avail_root_kb=$(df -Pk / | awk 'END {print $4}')
required_target_gib=45
(( ! resume_cpu )) || required_target_gib=30
(( avail_target_kb >= required_target_gib * 1024 * 1024 )) || fail "need at least ${required_target_gib} GiB free under $BASE_PARENT"
(( avail_root_kb >= 5 * 1024 * 1024 )) || fail "need at least 5 GiB free on / for small OS graphics packages"
[[ -x "$CONDA_BIN" ]] || fail "expected Miniconda not found at $CONDA_BIN"
libc_version=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
awk -v v="$libc_version" 'BEGIN {split(v,a,"."); exit !(a[1] > 2 || (a[1] == 2 && a[2] >= 35))}' || fail "Isaac Sim pip needs GLIBC 2.35+; found $libc_version"
printf 'glibc=%s\n' "$libc_version"

set_stage vulkan-prerequisites
if ! ldconfig -p | grep 'libvulkan\.so\.1' >/dev/null || ! command -v vulkaninfo >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libvulkan1 vulkan-tools
  apt-get clean
fi
ldconfig -p | grep 'libvulkan\.so\.1' >/dev/null || fail "libvulkan.so.1 remains unavailable"
if [[ "$mode" == "gpu" ]]; then
  icd=""
  for candidate in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    if [[ -r "$candidate" ]]; then icd="$candidate"; break; fi
  done
  [[ -n "$icd" ]] || fail "NVIDIA Vulkan ICD JSON is missing"
  ldconfig -p | grep 'libGLX_nvidia\.so\.0' >/dev/null || fail "NVIDIA GLX driver library is missing"
  command -v vulkaninfo >/dev/null || fail "vulkan-tools did not provide vulkaninfo"
  printf 'vulkan_icd=%s\n' "$icd"
  VK_ICD_FILENAMES="$icd" vulkaninfo --summary
else
  printf 'vulkan_runtime=NOT_TESTED_CPU_PREPARATION\n'
fi

set_stage target-and-cache-layout
mkdir -p "$BASE" "$BASE/tmp" "$BASE/cache/conda/pkgs" "$BASE/cache/pip" \
  "$BASE/cache/xdg" "$BASE/cache/cuda" "$BASE/cache/triton" \
  "$BASE/omniverse/config" "$BASE/omniverse/data" "$BASE/omniverse/cache" "$BASE/omniverse/logs"
export TMPDIR="$BASE/tmp"
export PIP_CACHE_DIR="$BASE/cache/pip"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CONDA_PKGS_DIRS="$BASE/cache/conda/pkgs"
export XDG_CACHE_HOME="$BASE/cache/xdg"
export XDG_DATA_HOME="$BASE/omniverse/data"
export XDG_CONFIG_HOME="$BASE/omniverse/config"
export OMNI_CONFIG_PATH="$BASE/omniverse/config"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export CUDA_CACHE_PATH="$BASE/cache/cuda"
export TRITON_CACHE_DIR="$BASE/cache/triton"
if (( ! resume_cpu )); then
cat > "$OMNI_CONFIG_PATH/omniverse.toml" <<EOF
[paths]
data_root = "$BASE/omniverse/data"
cache_root = "$BASE/omniverse/cache"
logs_root = "$BASE/omniverse/logs"
EOF
fi

readonly SIM_ENV="$BASE/envs/sim"
readonly POLICY_ENV="$BASE/envs/policy"
readonly LAB_ROOT="$BASE/src/IsaacLab-v2.3.1"
mkdir -p "$BASE/envs" "$BASE/src"

set_stage simulator-python311
if (( ! resume_cpu )); then
"$CONDA_BIN" create -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main -p "$SIM_ENV" python=3.11 pip
fi
"$CONDA_BIN" run --no-capture-output -p "$SIM_ENV" python -m pip install --no-cache-dir --upgrade pip

# Install the final CUDA build first so SDK dependencies do not fetch a second
# Torch/CUDA stack. This is package installation only, with no CUDA calls.
set_stage simulator-torch-cu128
sim_torch='https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
sim_torch_wheel="$BASE/cache/wheels/torch-2.7.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
[[ ! -f "$sim_torch_wheel" ]] || sim_torch="$sim_torch_wheel"
sim_vision='https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
sim_vision_wheel="$BASE/cache/wheels/torchvision-0.22.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
[[ ! -f "$sim_vision_wheel" ]] || sim_vision="$sim_vision_wheel"
mapfile -t sim_cached_deps < <(cached_torch_dependencies "$sim_torch_wheel")
"$CONDA_BIN" run --no-capture-output -p "$SIM_ENV" python -m pip install --no-cache-dir \
  "$sim_torch" "$sim_vision" "${sim_cached_deps[@]}" \
  --index-url https://mirrors.aliyun.com/pypi/simple --find-links "$BASE/cache/wheels"

set_stage isaac-sim-5.1
# Install the documented core packages needed for physics, robot assets, and
# RTX camera sensors. Isaac Lab learning frameworks and optional ROS/collector
# packages are deliberately excluded.
sdk_requirements=( \
  'isaacsim==5.1.0' \
  'isaacsim-kernel==5.1.0' 'isaacsim-app==5.1.0' 'isaacsim-core==5.1.0' \
  'isaacsim-asset==5.1.0' 'isaacsim-robot==5.1.0' 'isaacsim-sensor==5.1.0' \
  'isaacsim-extscache-kit==5.1.0' 'isaacsim-extscache-kit-sdk==5.1.0' \
  'isaacsim-extscache-physics==5.1.0' )
sdk_cached_roots=()
for sdk_wheel in "$BASE/cache/wheels"/isaacsim*.whl; do
  [[ ! -f "$sdk_wheel" ]] || sdk_cached_roots+=("$sdk_wheel")
done
if ! "$CONDA_BIN" run --no-capture-output -p "$SIM_ENV" python -m pip install --no-cache-dir \
  --no-index --find-links "$BASE/cache/wheels" "${sdk_requirements[@]}"; then
  "$CONDA_BIN" run --no-capture-output -p "$SIM_ENV" python -m pip install --no-cache-dir \
    "${sdk_cached_roots[@]}" "${sdk_requirements[@]}" --extra-index-url https://pypi.nvidia.com \
    --find-links "$BASE/cache/wheels"
fi

set_stage isaac-lab-v2.3.1
# flatdict 4.0.1 needs pkg_resources; preinstall it with compatible build tools.
"$SIM_ENV/bin/python" -m pip install 'setuptools==80.9.0'
"$SIM_ENV/bin/python" -m pip install --no-build-isolation 'flatdict==4.0.1'
git clone --depth 1 --branch "$LAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$LAB_ROOT"
(
  cd "$LAB_ROOT"
  "$CONDA_BIN" run --no-capture-output -p "$SIM_ENV" ./isaaclab.sh --install none
)
# The upstream installer can continue after an extension's pip failure.
"$SIM_ENV/bin/python" -c 'import importlib.metadata as m; assert m.version("isaaclab") == "0.48.0"'

set_stage policy-python311
"$CONDA_BIN" create -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main -p "$POLICY_ENV" python=3.11 pip
"$CONDA_BIN" run --no-capture-output -p "$POLICY_ENV" python -m pip install --no-cache-dir --upgrade pip
policy_torch='https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
policy_torch_wheel="$BASE/cache/wheels/torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
[[ ! -f "$policy_torch_wheel" ]] || policy_torch="$policy_torch_wheel"
policy_vision='https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.23.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
policy_vision_wheel="$BASE/cache/wheels/torchvision-0.23.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
[[ ! -f "$policy_vision_wheel" ]] || policy_vision="$policy_vision_wheel"
mapfile -t policy_cached_deps < <(cached_torch_dependencies "$policy_torch_wheel")
"$CONDA_BIN" run --no-capture-output -p "$POLICY_ENV" python -m pip install --no-cache-dir \
  "$policy_torch" "$policy_vision" "${policy_cached_deps[@]}" \
  --index-url https://mirrors.aliyun.com/pypi/simple --find-links "$BASE/cache/wheels"
"$CONDA_BIN" run --no-capture-output -p "$POLICY_ENV" python -m pip install --no-cache-dir \
  'numpy==2.4.6' 'hydra-core==1.3.7' 'omegaconf==2.3.1' \
  'stable-pretraining==0.1.7' 'datasets==5.0.1' 'gymnasium==1.3.0' 'Pillow==12.3.0' 'lightning==2.6.6'

set_stage install-record
{
  printf 'host=%s\n' "$(hostname)"
  printf 'os=%s\n' "$(. /etc/os-release; printf '%s %s' "$PRETTY_NAME" "")"
  printf 'gpu=%s\n' "$gpu_label"
  printf 'isaaclab_tag=%s\n' "$LAB_TAG"
  printf 'sim_env=%s\npolicy_env=%s\n' "$SIM_ENV" "$POLICY_ENV"
  printf 'eula_env=OMNI_KIT_ACCEPT_EULA=YES\n'
  printf 'mode=%s\n' "$mode"
  printf 'source_bundle=not-installed-by-this-script\n'
} > "$BASE/install_record.txt"
printf 'Installation stages finished. Next: run the separate SDK and native Rolling task RGB smoke; no simulator or model smoke was run by this script.\n'

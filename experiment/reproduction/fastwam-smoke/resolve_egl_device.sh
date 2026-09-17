#!/bin/bash

# Resolve the scheduler-provided GPU for MuJoCo/robosuite without changing
# PBS's CUDA isolation. nvidia-smi does not honor CUDA_VISIBLE_DEVICES, so
# UUID identity is checked through PyTorch and the EGL device is matched to
# that UUID through CUDA-OpenGL interop.
resolve_egl_device() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  local resolved=""
  local torch_uuid=""
  local torch_uuid_raw=""
  local egl_count=""
  local egl_index=""
  local probe_output=""
  local probe_status=0

  if [ -z "$visible" ]; then
    echo 'CUDA_VISIBLE_DEVICES is unset; refusing to guess an EGL device.' >&2
    return 65
  fi
  case "$visible" in
    *,*)
      echo "Expected one PBS-visible GPU, got CUDA_VISIBLE_DEVICES=$visible" >&2
      return 65
      ;;
  esac

  export CUDA_VISIBLE_DEVICES_ORIGINAL="$visible"
  if [[ "$visible" == GPU-* ]]; then
    if [ -z "${VENV:-}" ] || [ ! -x "$VENV/bin/python" ]; then
      echo 'VENV/bin/python is required for UUID identity verification.' >&2
      return 65
    fi
    if probe_output="$("$VENV/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one CUDA device under the PBS mask")
props = torch.cuda.get_device_properties(0)
uuid = getattr(props, "uuid", None)
if uuid is None:
    raise SystemExit("PyTorch did not expose the CUDA device UUID")
print(str(uuid).strip())
PY
  2>&1)"; then
      probe_status=0
    else
      probe_status=$?
    fi
    if [ "$probe_status" -ne 0 ]; then
      echo "PyTorch UUID probe failed (status=$probe_status): $probe_output" >&2
      return 65
    fi
    torch_uuid_raw="$(printf '%s\n' "$probe_output" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r')"
    case "$torch_uuid_raw" in
      GPU-*) torch_uuid="$torch_uuid_raw" ;;
      *) torch_uuid="GPU-$torch_uuid_raw" ;;
    esac
    if [ "$torch_uuid" != "$visible" ]; then
      echo "PBS UUID does not match PyTorch device UUID: expected=$visible actual=$torch_uuid" >&2
      return 65
    fi

    if probe_output="$("$VENV/bin/python" - "$visible" <<'PY'
import ctypes
import ctypes.util
import sys

from mujoco.egl import egl_ext as EGL

target_uuid = sys.argv[1].removeprefix("GPU-").replace("-", "").lower()

libcuda_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
libcuda = ctypes.CDLL(libcuda_name)
cuInit = libcuda.cuInit
cuInit.argtypes = [ctypes.c_uint]
cuInit.restype = ctypes.c_int
cuGLGetDevices = libcuda.cuGLGetDevices
cuGLGetDevices.argtypes = [
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_uint,
    ctypes.c_uint,
]
cuGLGetDevices.restype = ctypes.c_int
cuDeviceGetUuid = libcuda.cuDeviceGetUuid
cuDeviceGetUuid.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
cuDeviceGetUuid.restype = ctypes.c_int

if cuInit(0) != 0:
    raise SystemExit("CUDA driver initialization failed")

EGL_ATTRIBUTES = (
    EGL.EGL_RED_SIZE, 8,
    EGL.EGL_GREEN_SIZE, 8,
    EGL.EGL_BLUE_SIZE, 8,
    EGL.EGL_ALPHA_SIZE, 8,
    EGL.EGL_DEPTH_SIZE, 24,
    EGL.EGL_STENCIL_SIZE, 8,
    EGL.EGL_COLOR_BUFFER_TYPE, EGL.EGL_RGB_BUFFER,
    EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
    EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
    EGL.EGL_NONE,
)

def cuda_uuids_for_current_gl():
    count = ctypes.c_uint()
    devices = (ctypes.c_int * 16)()
    result = cuGLGetDevices(ctypes.byref(count), devices, 16, 1)
    if result != 0:
        raise RuntimeError(f"cuGLGetDevices failed with CUDA result {result}")
    uuids = []
    for ordinal in devices[:count.value]:
        raw = (ctypes.c_ubyte * 16)()
        result = cuDeviceGetUuid(raw, ordinal)
        if result != 0:
            raise RuntimeError(f"cuDeviceGetUuid failed with CUDA result {result}")
        uuids.append(bytes(raw).hex())
    return uuids

def probe_device(device):
    display = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
    if display == EGL.EGL_NO_DISPLAY:
        raise RuntimeError("eglGetPlatformDisplayEXT returned EGL_NO_DISPLAY")
    context = None
    try:
        if EGL.eglInitialize(display, None, None) != EGL.EGL_TRUE:
            raise RuntimeError("eglInitialize failed")
        configs = (EGL.EGLConfig * 1)()
        config_count = EGL.EGLint()
        if EGL.eglChooseConfig(display, EGL_ATTRIBUTES, configs, 1, config_count) != EGL.EGL_TRUE:
            raise RuntimeError("eglChooseConfig failed")
        if config_count.value < 1:
            raise RuntimeError("eglChooseConfig returned no config")
        if EGL.eglBindAPI(EGL.EGL_OPENGL_API) != EGL.EGL_TRUE:
            raise RuntimeError("eglBindAPI failed")
        context = EGL.eglCreateContext(display, configs[0], EGL.EGL_NO_CONTEXT, None)
        if not context:
            raise RuntimeError("eglCreateContext failed")
        if EGL.eglMakeCurrent(display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, context) != EGL.EGL_TRUE:
            raise RuntimeError("eglMakeCurrent failed")
        return cuda_uuids_for_current_gl()
    finally:
        if context:
            EGL.eglDestroyContext(display, context)
        EGL.eglTerminate(display)
        EGL.eglReleaseThread()

devices = EGL.eglQueryDevicesEXT()
print(f"egl_device_count={len(devices)}")
candidate_indices = []
for index, device in enumerate(devices):
    try:
        cuda_uuids = probe_device(device)
        print(f"egl_index={index} cuda_uuids={','.join(cuda_uuids) or 'none'}")
        if target_uuid in cuda_uuids:
            candidate_indices.append(index)
    except Exception as exc:
        print(f"egl_index={index} cuda_gl_query_error={type(exc).__name__}:{exc}")

if len(candidate_indices) != 1:
    raise SystemExit(
        f"expected one EGL device matching CUDA UUID {target_uuid}, got {candidate_indices}"
    )
print(f"selected_egl_index={candidate_indices[0]}")
PY
  2>&1)"; then
      probe_status=0
    else
      probe_status=$?
    fi
    if [ "$probe_status" -ne 0 ]; then
      echo "EGL CUDA interop probe failed (status=$probe_status): $probe_output" >&2
      return 65
    fi
    egl_count="$(printf '%s\n' "$probe_output" | sed -n 's/^egl_device_count=//p' | tail -n 1 | tr -d '\r')"
    egl_index="$(printf '%s\n' "$probe_output" | sed -n 's/^selected_egl_index=//p' | tail -n 1 | tr -d '\r')"
    if ! [[ "$egl_count" =~ ^[0-9]+$ ]] || ! [[ "$egl_index" =~ ^[0-9]+$ ]]; then
      echo "EGL CUDA interop mapping failed: $probe_output" >&2
      return 65
    fi
    resolved="$egl_index"
    export MUJOCO_EGL_DEVICE_ID="$resolved"
    export CUDA_VISIBLE_DEVICES="$visible"
  elif [[ "$visible" =~ ^[0-9]+$ ]]; then
    resolved="$visible"
    export MUJOCO_EGL_DEVICE_ID="$resolved"
  else
    echo "Could not resolve scheduler GPU UUID/index: CUDA_VISIBLE_DEVICES=$visible" >&2
    return 65
  fi

  printf 'CUDA_VISIBLE_DEVICES_ORIGINAL=%s\n' "$CUDA_VISIBLE_DEVICES_ORIGINAL"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'torch_cuda_device_uuid_raw=%s\n' "${torch_uuid_raw:-not-queried-numeric-mask}"
  printf 'torch_cuda_device_uuid_normalized=%s\n' "${torch_uuid:-not-queried-numeric-mask}"
  printf 'egl_device_count=%s\n' "${egl_count:-not-queried-numeric-mask}"
  printf 'selected_egl_index=%s\n' "${egl_index:-$resolved}"
  printf 'MUJOCO_EGL_DEVICE_ID=%s\n' "$MUJOCO_EGL_DEVICE_ID"
  if [[ "$visible" == GPU-* ]]; then
    printf 'resolution_source=torch-uuid-plus-CUDA-GL-EGL-map\n'
  else
    printf 'resolution_source=PBS-numeric-index\n'
  fi
}

# Copy robosuite into the job artifact directory and patch only that copy.
# Its original assertion assumes CUDA_VISIBLE_DEVICES contains integers,
# while PBS may deliberately provide a UUID. MuJoCo receives the EGL index
# only after resolve_egl_device has matched its GL context to the UUID.
prepare_robosuite_uuid_patch() {
  if [ -z "${VENV:-}" ] || [ -z "${ARTIFACTS:-}" ]; then
    echo 'VENV and ARTIFACTS are required for the robosuite compatibility copy.' >&2
    return 65
  fi
  local source_pkg="$VENV/lib/python3.11/site-packages/robosuite"
  local target_root="$ARTIFACTS/robosuite-compat"
  local target_pkg="$target_root/robosuite"
  local binding="$target_pkg/utils/binding_utils.py"

  if [ ! -f "$source_pkg/utils/binding_utils.py" ]; then
    echo "robosuite binding_utils.py not found: $source_pkg/utils/binding_utils.py" >&2
    return 65
  fi
  rm -rf "$target_root"
  mkdir -p "$target_root"
  cp -a "$source_pkg" "$target_pkg"
  "$VENV/bin/python" - "$binding" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = '''CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"
'''
new = '''CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        if CUDA_VISIBLE_DEVICES.startswith("GPU-"):
            # PBS UUID identity was validated through torch.cuda before import.
            assert MUJOCO_EGL_DEVICE_ID.isdigit(), "MUJOCO_EGL_DEVICE_ID must be numeric"
        else:
            assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
                MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
            ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"
'''
if text.count(old) != 1:
    raise SystemExit(f"expected one robosuite assertion block, found {text.count(old)}")
path.write_text(text.replace(old, new))
PY
  export PYTHONPATH="$target_root${PYTHONPATH:+:$PYTHONPATH}"
  printf 'robosuite_source=%s\n' "$source_pkg"
  printf 'robosuite_compat=%s\n' "$target_pkg"
  printf 'robosuite_patch=binding_utils_UUID_mask\n'
}
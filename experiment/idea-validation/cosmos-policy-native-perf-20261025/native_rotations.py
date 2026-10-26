"""Native kernel adapters for the frozen Cosmos SpinQuant and QuaRot banks.

The official QuaRot CUTLASS INT4 GEMM and Fast Hadamard kernels are used for
their respective operations. G128 scale handling, activation quantization,
sign multiplication, and method artifact integration remain task-local; this
is not the official full QuaRot/SpinQuant inference engine.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path


SPIN_ARM = "spinquant_w4a4"
QUAROT_ARM = "quarot_w4a4"
EXPECTED_MODULES = 280
GROUP_SIZE = 128
BACKEND_MODULE = "quarot_official_gemm"
FUSED_BACKEND_MODULE = "quarot_official_gemm_fused"
HADAMARD_BACKEND_MODULE = "quarot_native_hadamard_cuda"
QUAROT_SOURCE_REVISION = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
CUTLASS_SOURCE_REVISION = "ffa34e70756b0bc744e1dfcc115b5a991a68f132"
HADAMARD_SOURCE_REPOSITORY = "https://github.com/Dao-AILab/fast-hadamard-transform.git"
HADAMARD_SUBMODULE_PATH = "third-party/fast-hadamard-transform"
ADAPTER_VERSION = "cosmos_rotation_int4_g128_v3_native_quarot_h128"


def _group_ranges(padded_width: int, group_size: int = GROUP_SIZE):
    if group_size != GROUP_SIZE or padded_width <= 0 or padded_width % group_size:
        raise ValueError("native adapter requires positive width padded to G128")
    return tuple((start // group_size, start, start + group_size)
                 for start in range(0, padded_width, group_size))


def _offset8_to_twos_complement_byte(value: int) -> int:
    """Rebase two offset-8 nibbles without unpacking or changing their values."""
    if not 0 <= value <= 255:
        raise ValueError("packed byte must be in [0, 255]")
    return value ^ 0x88


def _load_backend(backend=None):
    fused_search_path = any(
        Path(entry).name == FUSED_BACKEND_MODULE
        for entry in sys.path if isinstance(entry, str) and entry)
    fused_present = FUSED_BACKEND_MODULE in sys.modules or fused_search_path
    if backend is None and not fused_present:
        fused_spec = importlib.util.find_spec(FUSED_BACKEND_MODULE)
        fused_present = fused_spec is not None
    if backend is None and fused_present:
        try:
            backend = importlib.import_module(FUSED_BACKEND_MODULE)
        except Exception as exc:
            raise RuntimeError(
                f"{FUSED_BACKEND_MODULE} is present but failed to load; refusing legacy fallback") from exc
        if not callable(getattr(backend, "scaled_add", None)):
            raise RuntimeError(f"{FUSED_BACKEND_MODULE} must expose scaled_add(output, accum, a_scales, w_scales, group)")
    elif backend is None:
        backend = importlib.import_module(BACKEND_MODULE)
    if not callable(getattr(backend, "matmul", None)):
        module_name = getattr(backend, "__name__", BACKEND_MODULE)
        raise RuntimeError(f"{module_name} must expose matmul(A_uint8, B_uint8) -> int32")
    return backend


def _safe_bank_path(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).is_absolute() or ".." in Path(filename).parts:
        raise RuntimeError("packed bank contains an unsafe relative path")
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("packed bank path escapes its artifact directory") from exc
    return path


def _pack_weight_groups(packed, padded_width, torch, *, offset8=False):
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise RuntimeError("W4 bank codes must be a packed uint8 matrix")
    if packed.shape[1] * 2 != padded_width:
        raise RuntimeError("W4 bank packed width differs from its padded K")
    if offset8:
        packed = packed.bitwise_xor(0x88)
    return tuple(packed[:, start // 2:(start + GROUP_SIZE) // 2].contiguous()
                 for _, start, _ in _group_ranges(padded_width))


def _spin_banks(runtime, artifact_dir, torch):
    root = Path(artifact_dir)
    bank_root = root / "packed_bank"
    manifest = json.loads((bank_root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "fastwam-rotation-native-bank-v1"
            or manifest.get("rotation_mode") != "learned"
            or manifest.get("group_size") != GROUP_SIZE
            or manifest.get("weight_bits") != 4
            or manifest.get("activation_bits") != 4
            or manifest.get("kv_cache_dtype") != "bf16"):
        raise RuntimeError("SpinQuant artifact is not the frozen learned W4A4/G128/BF16-KV bank")
    modules = dict(sorted(runtime.target_linears.items()))
    entries = {row.get("name"): row for row in manifest.get("modules", [])}
    expected = {"video_expert." + name for name in modules}
    if len(entries) != len(manifest.get("modules", [])) or set(entries) != expected:
        raise RuntimeError("SpinQuant native bank target names differ")
    rotation_file = manifest.get("rotation_file")
    rotation_payload = torch.load(_safe_bank_path(bank_root, rotation_file),
                                  map_location="cpu", weights_only=True)
    if rotation_payload.get("format") != manifest.get("format"):
        raise RuntimeError("SpinQuant rotation payload format differs")
    streams = rotation_payload.get("streams", {})
    if set(streams) != {"video"}:
        raise RuntimeError("Cosmos SpinQuant requires the single shared video stream")
    rotation = streams["video"]
    matrix, signs = rotation.get("matrix"), rotation.get("signs")
    if (not torch.is_tensor(matrix) or tuple(matrix.shape) != (128, 128)
            or matrix.dtype != torch.float32 or not torch.is_tensor(signs)
            or signs.dtype != torch.float32 or signs.ndim != 1
            or signs.numel() < max(module.in_features for module in modules.values())):
        raise RuntimeError("SpinQuant R128 matrix/sign tensors have invalid shape or dtype")

    banks = {}
    for name, module in modules.items():
        alias = "video_expert." + name
        row = entries[alias]
        shape = [int(module.out_features), int(module.in_features)]
        if row.get("stream") != "video" or row.get("shape") != shape:
            raise RuntimeError(f"SpinQuant native bank shape/stream mismatch: {name}")
        payload = torch.load(_safe_bank_path(bank_root, row.get("file")),
                             map_location="cpu", weights_only=True)
        padded = math.ceil(module.in_features / GROUP_SIZE) * GROUP_SIZE
        packed, scales = payload.get("packed_weight"), payload.get("weight_scales_bf16")
        if (payload.get("module") != alias or payload.get("shape") != shape
                or payload.get("group_size") != GROUP_SIZE
                or payload.get("padded_in_features") != padded
                or payload.get("code_range") != [-7, 7]
                or payload.get("nibble_order") != "low nibble first; signed two's complement"
                or not torch.is_tensor(packed) or packed.dtype != torch.uint8
                or tuple(packed.shape) != (module.out_features, padded // 2)
                or not torch.is_tensor(scales) or scales.dtype != torch.bfloat16
                or tuple(scales.shape) != (module.out_features, padded // GROUP_SIZE)):
            raise RuntimeError(f"SpinQuant packed W4/G128 payload mismatch: {name}")
        device = module.weight.device
        groups = _pack_weight_groups(packed.to(device=device), padded, torch)
        banks[name] = {
            "weight_groups": groups,
            "scales": scales.to(device=device).contiguous(),
            "padded_width": padded,
            "scale_dtype": "torch.bfloat16",
            "rotation_mode": "learned_shared_cayley_r128",
        }
    return banks


def _quarot_banks(runtime, artifact_dir, torch):
    root = Path(artifact_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("arm") != QUAROT_ARM
            or manifest.get("recipe", {}).get("group_size") != GROUP_SIZE):
        raise RuntimeError("QuaRot artifact is not the completed frozen W4A4/G128 bank")
    modules = dict(sorted(runtime.target_linears.items()))
    rows = manifest.get("target_manifest")
    expected_rows = [{"name": name,
                      "shape": [int(module.out_features), int(module.in_features)],
                      "weight_numel": int(module.out_features * module.in_features)}
                     for name, module in modules.items()]
    if rows != expected_rows:
        raise RuntimeError("QuaRot artifact target manifest differs from runtime modules")
    entries = manifest.get("entries", [])
    if len(entries) != len(modules) or [row.get("name") for row in entries] != list(modules):
        raise RuntimeError("QuaRot bank entries differ from sorted target modules")

    banks = {}
    for index, (name, module) in enumerate(modules.items()):
        shape = [int(module.out_features), int(module.in_features)]
        if module.in_features % GROUP_SIZE:
            raise RuntimeError(f"QuaRot local H128 requires K divisible by 128: {name}")
        row = entries[index]
        if row.get("shape") != shape:
            raise RuntimeError(f"QuaRot bank module identity differs: {name}")
        payload = torch.load(_safe_bank_path(root, row.get("file")),
                             map_location="cpu", weights_only=True)
        encoded = payload.get("encoded", {})
        packed, scales = encoded.get("codes"), encoded.get("scales")
        padded = math.ceil(module.in_features / GROUP_SIZE) * GROUP_SIZE
        signs = payload.get("signs")
        if (payload.get("module") != name or payload.get("module_index") != index
                or payload.get("shape") != tuple(shape)
                or encoded.get("shape") != tuple(shape)
                or encoded.get("group_size") != GROUP_SIZE
                or encoded.get("bits") != 4 or encoded.get("affine") is not False
                or not torch.is_tensor(packed) or packed.dtype != torch.uint8
                or tuple(packed.shape) != (module.out_features, padded // 2)
                or not torch.is_tensor(scales) or scales.dtype != torch.float16
                or tuple(scales.shape) != (module.out_features, padded // GROUP_SIZE)
                or not torch.is_tensor(signs) or signs.ndim != 1
                or signs.numel() != module.in_features):
            raise RuntimeError(f"QuaRot packed W4/G128 payload mismatch: {name}")
        device = module.weight.device
        # The old bank stores q+8; xor 8 in each nibble converts it exactly to
        # signed two's-complement without changing scales or quantized values.
        groups = _pack_weight_groups(packed.to(device=device), padded, torch, offset8=True)
        banks[name] = {
            "weight_groups": groups,
            "scales": scales.to(device=device).contiguous(),
            "signs": signs.to(device=device, dtype=torch.int8).contiguous(),
            "padded_width": padded,
            "scale_dtype": "torch.float16",
            "rotation_mode": "random_local_signed_hadamard_h128",
        }
    return banks


def _resolve_artifact_dir(p, old_root, arm, artifact_dir=None):
    """Resolve the frozen method bank even when the decoded arm is already active."""
    if artifact_dir is not None:
        return Path(artifact_dir)
    if arm == SPIN_ARM:
        try:
            return Path(p["reuse"]["spinquant_artifact"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("SpinQuant frozen artifact path is missing from protocol reuse metadata") from exc
    if arm == QUAROT_ARM:
        return Path(old_root) / "results" / "quarot"
    raise ValueError(f"native rotation adapter supports only {SPIN_ARM} and {QUAROT_ARM}")


def _validate_hadamard_build(module, receipt, extension_dir=None):
    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        raise RuntimeError("QuaRot native H128 requires a complete FHT build receipt")
    if (receipt.get("module_name") != HADAMARD_BACKEND_MODULE
            or receipt.get("source_repository") != HADAMARD_SOURCE_REPOSITORY
            or receipt.get("submodule_path") != HADAMARD_SUBMODULE_PATH
            or receipt.get("quarot_source_revision") != QUAROT_SOURCE_REVISION):
        raise RuntimeError("FHT build receipt is not bound to the pinned QuaRot submodule")
    revision = receipt.get("source_commit")
    if not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision.lower()):
        raise RuntimeError("FHT build receipt has an invalid upstream source revision")
    if not receipt.get("pbs_job_id") or not receipt.get("host"):
        raise RuntimeError("FHT build receipt is missing its PBS build producer")
    module_file = getattr(module, "__file__", None)
    receipt_module = receipt.get("module")
    if not module_file or not receipt_module:
        raise RuntimeError("FHT extension and receipt must identify their compiled module")
    actual_path = Path(module_file).resolve()
    if actual_path != Path(receipt_module).resolve():
        raise RuntimeError("Loaded FHT module path differs from the compiled build receipt")
    if extension_dir is not None and actual_path.parent != Path(extension_dir).resolve():
        raise RuntimeError("Loaded FHT module is outside the selected QuaRot GEMM extension directory")
    if not callable(getattr(module, "fast_hadamard_transform", None)):
        raise RuntimeError("FHT extension does not expose fast_hadamard_transform")
    return receipt


def _load_hadamard_backend(gemm_backend):
    module_file = getattr(gemm_backend, "__file__", None)
    if not module_file:
        raise RuntimeError("Cannot locate the FHT extension directory from the official GEMM module")
    extension_dir = Path(module_file).resolve().parent
    directory = str(extension_dir)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    receipt_path = (extension_dir.parents[1] / "results" / "preparation"
                    / "quarot_hadamard_build.json")
    if not receipt_path.is_file():
        raise RuntimeError(f"QuaRot native H128 build receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        module = importlib.import_module(HADAMARD_BACKEND_MODULE)
    except Exception as exc:
        raise RuntimeError("QuaRot FHT extension is present in the protocol but failed to load") from exc
    _validate_hadamard_build(module, receipt, extension_dir)
    return module, receipt, receipt_path


def _fwht_h128(value, hadamard_backend):
    if value.shape[-1] != GROUP_SIZE:
        raise ValueError("native FHT adapter accepts H128 blocks only")
    if hadamard_backend is None:
        raise RuntimeError("QuaRot native H128 extension is unavailable; Torch fallback is disabled")
    transform = getattr(hadamard_backend, "fast_hadamard_transform", None)
    if not callable(transform):
        raise RuntimeError("QuaRot native H128 extension has no fast_hadamard_transform")
    return transform(value.float().contiguous(), 1.0 / math.sqrt(GROUP_SIZE))


def _rotate_quarot(value, signs, torch, hadamard_backend):
    if value.shape[-1] % GROUP_SIZE:
        raise ValueError("QuaRot local H128 input width must be divisible by 128")
    blocks = value.float().reshape(*value.shape[:-1], -1, GROUP_SIZE)
    signs = signs.to(device=value.device, dtype=torch.float32).reshape(-1, GROUP_SIZE)
    return _fwht_h128(blocks * signs, hadamard_backend).reshape_as(value)


def _quantize_activation(value, padded_width, torch):
    rows = value.float().reshape(-1, value.shape[-1]).contiguous()
    if rows.shape[0] == 0 or rows.shape[1] > padded_width:
        raise ValueError("native A4 GEMM received an empty or over-wide Linear input")
    maximum = rows.abs().amax(dim=1, keepdim=True)
    scales = maximum / 7.0
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(rows / safe).clamp(-7, 7).to(torch.int8)
    codes = torch.where(scales > 0, codes, torch.zeros_like(codes))
    if padded_width != rows.shape[1]:
        codes = torch.cat((codes, torch.zeros((rows.shape[0], padded_width - rows.shape[1]),
                                              device=rows.device, dtype=torch.int8)), dim=1)
    groups = codes.reshape(rows.shape[0], -1, GROUP_SIZE).permute(1, 0, 2).contiguous()
    nibbles = groups.to(torch.int16).bitwise_and(15).to(torch.uint8)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).contiguous()
    return packed, scales.reshape(-1).contiguous()


def _accumulate_group(output, accum, activation_scales, weight_scales, group, backend):
    """Apply one G128 scale and accumulate, preserving the stored weight-scale dtype."""
    scaled_add = getattr(backend, "scaled_add", None)
    if callable(scaled_add):
        scaled_add(output, accum, activation_scales, weight_scales, group)
        return "fused_native_scaled_add"
    factor = activation_scales[:, None] * weight_scales[:, group].float()[None, :]
    output.add_(accum.float() * factor)
    return "torch_fp32_group_scale_multiply_add"


def _native_linear(module, value, torch, backend, evidence, hadamard_backend=None):
    context = module.__dict__["_native_rotation_context"]
    if context["arm"] == SPIN_ARM:
        state = module.__dict__.get("_cosmos_spin_state")
        if not isinstance(state, dict) or not callable(state.get("rotate")):
            raise RuntimeError("SpinQuant learned R128 transform state is unavailable")
        transformed = state["rotate"](
            value.to(dtype=module._native_activation_dtype), state["matrix"], state["signs"])
    else:
        transformed = _rotate_quarot(value, module._native_rotation_signs, torch,
                                     hadamard_backend)
        evidence["native_hadamard_kernel_used"] = True
        evidence["native_hadamard_calls"] += 1
        evidence["native_hadamard_input_rows"] += int(value.numel() // GROUP_SIZE)

    packed_a, activation_scales = _quantize_activation(
        transformed, context["padded_width"], torch)
    output = torch.zeros((packed_a.shape[1], module.out_features),
                         device=value.device, dtype=torch.float32)
    epilogue_modes = set()
    for group, activation_group in enumerate(packed_a):
        accum = backend.matmul(activation_group, context["weight_groups"][group])
        if (not torch.is_tensor(accum) or accum.dtype != torch.int32
                or tuple(accum.shape) != tuple(output.shape) or accum.device != value.device):
            raise RuntimeError("official CUTLASS matmul must return int32[M,N] on the input GPU")
        epilogue_modes.add(_accumulate_group(
            output, accum, activation_scales, context["scales"], group, backend))
        evidence["native_scaled_add_calls"] += int(
            callable(getattr(backend, "scaled_add", None)))
        evidence["native_integer_gemm_calls"] += 1
    if len(epilogue_modes) != 1:
        raise RuntimeError(f"native linear mixed group epilogues: {sorted(epilogue_modes)}")
    evidence["adapter_epilogue"] = next(iter(epilogue_modes))
    if module.bias is not None:
        output.add_(module.bias.float())
    evidence["native_target_linear_calls"] += 1
    return output.reshape(*value.shape[:-1], module.out_features).to(torch.bfloat16)


def _restore_native(runtime, state):
    torch = state["torch"]
    controller = runtime.controller
    with torch.no_grad():
        for name, module in runtime.target_linears.items():
            for buffer_name in ("_native_rotation_scales", "_native_rotation_signs"):
                module._buffers.pop(buffer_name, None)
            module.__dict__.pop("_native_rotation_context", None)
            module.__dict__.pop("_native_rotation_signs", None)
            module.__dict__.pop("_native_activation_dtype", None)
            module.__dict__.pop("_cosmos_adapted_baseline_arm", None)
            if module.weight is None:
                base = controller._base_weights_cpu[name]
                module.register_parameter(
                    "weight", torch.nn.Parameter(base.to(device=state["devices"][name]),
                                                  requires_grad=controller._weight_requires_grad[name]))
    controller._restore_current()
    controller.arm = "bf16"
    controller.metadata = controller._bf16_metadata()
    runtime.current_arm = "bf16"
    runtime.quantization_evidence = controller.metadata
    runtime.native_kernel_stats = None
    runtime.__dict__.pop("restore_native_rotation", None)
    state["installed"] = False


def install_native(runtime, p, old_root, arm, artifact_dir=None, backend=None):
    """Install a native rotation baseline and return (metadata, restore_hook).

    `old_root` is the legacy experiment root. The method artifact is installed
    through its existing decoded reference first, then its exact packed bank is
    attached to the native CUTLASS split-G128 adapter.
    """
    if arm not in (SPIN_ARM, QUAROT_ARM):
        raise ValueError(f"native rotation adapter supports only {SPIN_ARM} and {QUAROT_ARM}")
    artifact_dir = _resolve_artifact_dir(p, old_root, arm, artifact_dir)
    import torch

    previous = getattr(runtime, "restore_native_rotation", None)
    if callable(previous):
        previous()
    controller = getattr(runtime, "controller", None)
    if controller is None:
        raise RuntimeError("CosmosRuntime controller is required for safe arm restore")
    if not runtime.target_linears:
        raise RuntimeError("runtime.target_linears is empty")
    backend = _load_backend(backend)
    hadamard_backend = hadamard_receipt = hadamard_receipt_path = None
    if arm == QUAROT_ARM:
        hadamard_backend, hadamard_receipt, hadamard_receipt_path = _load_hadamard_backend(backend)
    if not torch.cuda.is_available():
        raise RuntimeError("native W4A4 requires a CUDA allocation")
    device = next(iter(runtime.target_linears.values())).weight.device
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"QuaRot CUTLASS INT4 Tensor Core requires SM80+, got SM{major}{minor}")
    modules = dict(sorted(runtime.target_linears.items()))
    if len(modules) != EXPECTED_MODULES:
        raise RuntimeError(f"expected the frozen {EXPECTED_MODULES} target Linears, got {len(modules)}")
    if any(module.weight is None or module.weight.dtype != torch.bfloat16
           for module in modules.values()):
        raise RuntimeError("native installation requires the decoded BF16 reference arm")
    if any(module.bias is not None and module.bias.dtype != torch.bfloat16
           for module in modules.values()):
        raise RuntimeError("native installation requires BF16 target biases")
    if any(module.weight.device != device for module in modules.values()):
        raise RuntimeError("all target Linears must be on one CUDA device")

    controller._backup_weights()
    try:
        if controller.arm != arm:
            controller._restore_current()
            if arm == SPIN_ARM:
                controller.set_arm(arm, artifact_dir=artifact_dir)
            else:
                baseline_path = Path(old_root) / "baseline_quant.py"
                spec = importlib.util.spec_from_file_location(
                    "_native_legacy_baseline_quant", baseline_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot load legacy QuaRot installer at {baseline_path}")
                baseline_quant = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(baseline_quant)
                baseline_quant.install_baseline(runtime, p, arm, artifact_dir)
    except Exception:
        controller._restore_current()
        controller.arm = "bf16"
        controller.metadata = controller._bf16_metadata()
        runtime.current_arm = "bf16"
        runtime.quantization_evidence = controller.metadata
        runtime.native_kernel_stats = None
        raise
    runtime.current_arm = arm

    try:
        if arm == SPIN_ARM:
            banks = _spin_banks(runtime, artifact_dir, torch)
            scale_dtype = "torch.bfloat16"
            transform = "learned_shared_cayley_r128"
        else:
            banks = _quarot_banks(runtime, artifact_dir, torch)
            scale_dtype = "torch.float16"
            transform = "random_local_signed_hadamard_h128"
        source_artifact = str(Path(artifact_dir).resolve())
    except Exception:
        controller._restore_current()
        controller.arm = "bf16"
        controller.metadata = controller._bf16_metadata()
        runtime.current_arm = "bf16"
        runtime.quantization_evidence = controller.metadata
        runtime.native_kernel_stats = None
        raise

    base_device = {name: module.weight.device for name, module in modules.items()}
    evidence = {
        "arm": arm,
        "adapter_version": ADAPTER_VERSION,
        "execution": "official_quarot_cutlass_int4_gemm_split_g128",
        "native_backend": "official QuaRot CUTLASS int4b_t TensorOp GEMM with a Cosmos group-scale adapter",
        "native_backend_module": getattr(
            backend, "__name__",
            FUSED_BACKEND_MODULE if callable(getattr(backend, "scaled_add", None)) else BACKEND_MODULE),
        "native_backend_source_revision": QUAROT_SOURCE_REVISION,
        "native_backend_cutlass_revision": CUTLASS_SOURCE_REVISION,
        "native_hadamard_backend": ("Dao-AILab fast-hadamard-transform CUDA kernel from the pinned QuaRot submodule"
                                    if arm == QUAROT_ARM else
                                    "not used: frozen learned SpinQuant R128 remains a Torch FP32 dense matmul"),
        "native_hadamard_module": HADAMARD_BACKEND_MODULE if arm == QUAROT_ARM else None,
        "native_hadamard_source_repository": HADAMARD_SOURCE_REPOSITORY if arm == QUAROT_ARM else None,
        "native_hadamard_source_commit": (hadamard_receipt["source_commit"]
                                          if hadamard_receipt else None),
        "native_hadamard_quarot_source_revision": (hadamard_receipt["quarot_source_revision"]
                                                   if hadamard_receipt else None),
        "native_hadamard_submodule_path": (hadamard_receipt["submodule_path"]
                                           if hadamard_receipt else None),
        "native_hadamard_build_receipt": (str(hadamard_receipt_path.resolve())
                                          if hadamard_receipt_path else None),
        "native_hadamard_build_producer": ({
            "pbs_job_id": hadamard_receipt["pbs_job_id"],
            "host": hadamard_receipt["host"],
        } if hadamard_receipt else None),
        "native_hadamard_build_pbs_job_id": (hadamard_receipt.get("pbs_job_id")
                                             if hadamard_receipt else None),
        "native_hadamard_build_host": (hadamard_receipt.get("host")
                                       if hadamard_receipt else None),
        "native_hadamard_build_torch": (hadamard_receipt.get("torch")
                                        if hadamard_receipt else None),
        "native_hadamard_build_cuda_runtime": (hadamard_receipt.get("torch_cuda")
                                               if hadamard_receipt else None),
        "native_hadamard_kernel_used": False,
        "native_hadamard_calls": 0,
        "native_hadamard_input_rows": 0,
        "rotation_sign_multiply_backend": "torch_fp32_elementwise",
        "adapter_epilogue": ("fused_cuda_fp32_group_scaled_add" if callable(
            getattr(backend, "scaled_add", None)) else "python_torch_fp32_group_scaled_add"),
        "adapter_epilogue_source": "task-local G128 scale/add adapter around official QuaRot GEMM; not upstream full inference engine",
        "native_int4_kernel_used": True,
        "native_integer_gemm_calls": 0,
        "native_scaled_add_calls": 0,
        "native_target_linear_calls": 0,
        "native_int4_tensorcore": True,
        "native_integer_gemm_output_dtype": "torch.int32",
        "packed_gpu_resident": True,
        "packed_weight_gpu_resident": True,
        "packed_activation_gpu_resident_during_gemm": True,
        "packed_gemm_input_dtype": "torch.uint8",
        "packed_weight_storage": "signed_int4_two_complement_low_nibble_first",
        "packed_activation_storage": "per_token_signed_int4_two_complement_low_nibble_first",
        "weight_group_size": GROUP_SIZE,
        "weight_scale_dtype": scale_dtype,
        "activation_quantizer": "dynamic_per_token_symmetric_signed_int4_qmax7",
        "activation_quantizer_backend": "torch_fp32_dynamic_quantization_and_nibble_packing",
        "activation_scale_dtype": "torch.float32",
        "bias_dtype": "torch.bfloat16",
        "output_dtype": "torch.bfloat16",
        "reference_a4_elementwise_bf16_rounding": "not applied before integer GEMM; requires measured GPU parity",
        "rotation": transform,
        "artifact_dir": source_artifact,
        "target_module_count": len(modules),
        "target_weight_numel": sum(int(module.out_features * module.in_features)
                                    for module in modules.values()),
        "bf16_target_weight_numel_resident": 0,
        "bf16_target_weight_numel_cuda_resident": 0,
        "bf16_restore_shadow_cpu_numel": sum(int(module.out_features * module.in_features)
                                               for module in modules.values()),
        "groupwise_scale_rounding_changed": False,
    }
    packed_bytes = scale_bytes = rotation_bytes = 0
    rotation_storage_seen = set()
    installed = []
    state = {"torch": torch, "devices": base_device, "installed": False}
    try:
        for name, module in modules.items():
            bank = banks[name]
            context = {
                "arm": arm,
                "padded_width": bank["padded_width"],
                "weight_groups": bank["weight_groups"],
                "scales": bank["scales"],
            }
            module._native_rotation_context = context
            module._native_activation_dtype = module.weight.dtype
            if arm == QUAROT_ARM:
                module.register_buffer("_native_rotation_signs", bank["signs"], persistent=False)
                rotation_bytes += bank["signs"].numel() * bank["signs"].element_size()
            else:
                spin_state = module.__dict__["_cosmos_spin_state"]
                for tensor in (spin_state["matrix"], spin_state["signs"]):
                    key = (tensor.device, tensor.untyped_storage().data_ptr(),
                           tensor.untyped_storage().nbytes())
                    if key not in rotation_storage_seen:
                        rotation_storage_seen.add(key)
                        rotation_bytes += tensor.untyped_storage().nbytes()
            module.register_buffer("_native_rotation_scales", bank["scales"], persistent=False)
            packed_bytes += sum(t.numel() * t.element_size() for t in bank["weight_groups"])
            scale_bytes += bank["scales"].numel() * bank["scales"].element_size()
            installed.append((name, module))

            def forward(current, value, _torch=torch, _backend=backend, _evidence=evidence,
                        _hadamard=hadamard_backend):
                return _native_linear(current, value, _torch, _backend, _evidence, _hadamard)

            module.forward = types.MethodType(forward, module)
        # The original W4 weights now live only as packed CUDA buffers.
        for _, module in installed:
            module.register_parameter("weight", None)
        state["installed"] = True
    except Exception:
        for name, module in modules.items():
            for buffer_name in ("_native_rotation_scales", "_native_rotation_signs"):
                module._buffers.pop(buffer_name, None)
            module.__dict__.pop("_native_rotation_context", None)
            module.__dict__.pop("_native_rotation_signs", None)
            module.__dict__.pop("_native_activation_dtype", None)
            module.forward = controller._forwards[name]
            if module.weight is None:
                module.register_parameter("weight", torch.nn.Parameter(
                    controller._base_weights_cpu[name].to(device=base_device[name],
                                                           dtype=torch.bfloat16),
                    requires_grad=controller._weight_requires_grad[name]))
        controller._restore_current()
        controller.arm = "bf16"
        controller.metadata = controller._bf16_metadata()
        runtime.current_arm = "bf16"
        runtime.quantization_evidence = controller.metadata
        runtime.native_kernel_stats = None
        raise

    evidence.update({
        "packed_weight_tensor_bytes": packed_bytes,
        "weight_scale_tensor_bytes": scale_bytes,
        "rotation_tensor_bytes": rotation_bytes,
        "packed_target_payload_bytes": packed_bytes + scale_bytes + rotation_bytes,
        "weight_scale_values_preserved": True,
        "weight_scale_dtype": scale_dtype,
        "rotation_artifact_source": "existing method-specific packed checkpoint",
        "method_artifact_is_full_official_engine": False,
    })
    runtime.current_arm = arm
    controller.arm = arm
    controller.metadata = evidence
    runtime.quantization_evidence = evidence
    runtime.native_kernel_stats = evidence

    def restore():
        if state["installed"]:
            _restore_native(runtime, state)

    runtime.restore_native_rotation = restore
    return evidence, restore

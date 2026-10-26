"""Install existing Cosmos SVDQuant/ViDiT-Q banks through native INT4 GEMMs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

VIDITQ_SOURCE_COMMIT = "43657d11b32192ec0f50f0dac8d3a16c3d841196"
VIDITQ_GENERAL_MODULE = "viditq_atom_per_channel_native"


def _load_baseline(old_root: Path):
    here = Path(__file__).resolve().parent
    candidates = [
        old_root / "baseline_quant.py",
        old_root / "experiment" / "idea-validation" / "cosmos-policy-vqwam-20261024" / "baseline_quant.py",
        here.parent / "cosmos-policy-vqwam-20261024" / "baseline_quant.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("Legacy baseline_quant.py is required to validate and read the frozen banks")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    name = "_cosmos_vqwam_20261024_baseline_quant"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load baseline module at {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _legacy_protocol(old_root: Path, fallback: dict) -> dict:
    path = old_root / "protocol.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if fallback.get("protocol") == "cosmos-policy-vqwam-v1":
        return fallback
    raise FileNotFoundError(f"Legacy frozen protocol is required to validate the bank: {path}")


def _bank_directory(old_root: Path, arm: str, artifact_dir) -> Path:
    key = "svdquant" if arm == "svdquant_w4a4" else "viditq"
    candidates = []
    if artifact_dir is not None:
        root = Path(artifact_dir)
        candidates.extend((root, root / "results" / key, root / key))
    candidates.extend((old_root / "results" / key,
                       old_root / "results" / "baselines" / key))
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    looked = ", ".join(str(path / "manifest.json") for path in candidates)
    raise FileNotFoundError(f"Existing complete {arm} bank was not found; checked {looked}")


def _extension(arm: str, artifact_dir):
    import importlib.machinery
    if arm == "svdquant_w4a4":
        name, module_name = "nunchaku_svdq_native", "nunchaku_svdq_native"
    elif arm == "viditq_w4a4":
        return _viditq_general_extension(artifact_dir)[0]
    else:
        raise ValueError(f"Unsupported native diffusion extension arm: {arm}")
    here = Path(__file__).resolve().parent
    bases = [here]
    if artifact_dir is not None:
        bases.insert(0, Path(artifact_dir).resolve())
    dirs = []
    for base in bases:
        dirs.extend((base / "native_extensions" / arm, base / arm, base))
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    binary = next((file for directory in dirs if directory.is_dir()
                   for file in directory.glob(name + "*")
                   if file.is_file() and file.name.endswith(suffixes)), None)
    if binary is None:
        raise RuntimeError(f"Native extension {name} is missing; build it with build_diffusion.py in CPU PBS")
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, binary)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native kernel extension: {binary}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _viditq_general_build(artifact_dir=None) -> dict:
    root = (Path(artifact_dir).resolve() if artifact_dir is not None
            else Path(__file__).resolve().parent)
    receipt_path = root / "results" / "preparation" / "native_viditq_general_build.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"General per-column ViDiT-Q build receipt is required: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise RuntimeError("ViDiT-Q general per-column build receipt is incomplete")
    if receipt.get("source_commit") != VIDITQ_SOURCE_COMMIT:
        raise RuntimeError("ViDiT-Q general build receipt does not match the pinned Atom source")
    producer = receipt.get("producer") if isinstance(receipt.get("producer"), dict) else {}
    pbs_job_id = receipt.get("pbs_job_id") or producer.get("pbs_job_id")
    host = receipt.get("host") or producer.get("host")
    if not isinstance(pbs_job_id, str) or not pbs_job_id.strip() \
            or not isinstance(host, str) or not host.strip():
        raise RuntimeError("ViDiT-Q general build receipt lacks its PBS producer job and host")
    module_name = receipt.get("module_name", VIDITQ_GENERAL_MODULE)
    if module_name != VIDITQ_GENERAL_MODULE:
        raise RuntimeError(f"Unexpected ViDiT-Q general module name: {module_name!r}")
    raw_module = receipt.get("module")
    if not isinstance(raw_module, str) or not Path(raw_module).is_absolute():
        raise RuntimeError("ViDiT-Q general build receipt must name an absolute module path")
    module_path = Path(raw_module).resolve()
    extension_dir = (root / "native_extensions" / "viditq_w4a4_general").resolve()
    import importlib.machinery
    if (module_path.parent != extension_dir or not module_path.is_file()
            or not module_path.name.startswith(VIDITQ_GENERAL_MODULE)
            or not module_path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))):
        raise RuntimeError("ViDiT-Q general receipt module is not the built module in its required directory")
    return {"receipt_path": str(receipt_path.resolve()),
            "source_commit": VIDITQ_SOURCE_COMMIT,
            "module_name": VIDITQ_GENERAL_MODULE,
            "module": str(module_path),
            "pbs_job_id": pbs_job_id,
            "host": host,
            "producer": {"pbs_job_id": pbs_job_id, "host": host}}


def _viditq_general_extension(artifact_dir=None) -> tuple[object, dict]:
    build = _viditq_general_build(artifact_dir)
    name = VIDITQ_GENERAL_MODULE
    module_path = Path(build["module"])
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != module_path:
            raise RuntimeError(f"A different {name} binary is already loaded")
        module = cached
    else:
        spec = importlib.util.spec_from_file_location(name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load ViDiT-Q general extension: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    if Path(getattr(module, "__file__", "")).resolve() != module_path:
        raise RuntimeError("Loaded ViDiT-Q module path differs from its build receipt")
    if not callable(getattr(module, "matmul", None)):
        raise RuntimeError("ViDiT-Q general extension must expose its native Atom matmul entry point")
    return module, build


def _unpack_int4(codes, torch):
    return torch.stack((codes & 15, codes >> 4), dim=-1).reshape(codes.shape[0], -1)


def _signed_nibble(value: int) -> int:
    value &= 15
    return value - 16 if value >= 8 else value


def _split_w8_code(value: int) -> tuple[int, int]:
    centered = int(value) - 128
    return centered // 16, centered % 16 - 8


def _affine_group_correction(a_values, w_values, a_offset: int, w_offset: int) -> float:
    if len(a_values) != len(w_values):
        raise ValueError("Affine correction groups must have matching widths")
    return (a_offset * sum(w_values) + w_offset * sum(a_values)
            + len(a_values) * a_offset * w_offset)


def _module_bank_path(directory: Path, entry: dict, index: int) -> Path:
    path = (directory / entry["file"]).resolve()
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise RuntimeError("Baseline manifest bank path escapes its artifact directory") from exc
    return path


def _restore_bf16_parameters(modules, controller, devices, torch):
    """Recreate released GPU parameters from the controller's CPU shadow."""
    if controller is None or controller._base_weights_cpu is None:
        raise RuntimeError("Cannot restore released native weights without the controller CPU shadow")
    for name, module in modules.items():
        if module.weight is None:
            base = controller._base_weights_cpu[name]
            module.register_parameter(
                "weight",
                torch.nn.Parameter(
                    base.to(device=devices[name], dtype=torch.bfloat16),
                    requires_grad=controller._weight_requires_grad[name],
                ),
            )


def _unique_cuda_storage_bytes(values, torch) -> int:
    """Count the actual CUDA storages retained by an adapter without double-counting views."""
    seen = set()

    def visit(value):
        if torch.is_tensor(value):
            if value.is_cuda:
                storage = value.untyped_storage()
                key = (str(value.device), int(storage.data_ptr()), int(storage.nbytes()))
                seen.add(key)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(values)
    return sum(nbytes for _, _, nbytes in seen)


def _nunchaku_output_shape(batch: int, padded_m: int, padded_n: int,
                           out_features: int) -> tuple[int, int]:
    """Keep true M for the official epilogue mask while A and W stay tile-padded."""
    if (batch <= 0 or padded_m < batch or padded_m % 256 != 0
            or padded_m - batch >= 256):
        raise ValueError("Nunchaku activation M must be padded to the next 256-row tile")
    if (out_features <= 0 or padded_n < out_features or padded_n % 128 != 0
            or padded_n - out_features >= 128):
        raise ValueError("Nunchaku output N must be padded to the next 128-column tile")
    return batch, padded_n


NATIVE_ADAPTER_VERSIONS = {
    "svdquant_w4a4": "cosmos_nunchaku_w4a4_g128_rank16_v1",
    "viditq_w4a4": "cosmos_viditq_atom_w4a4_g128_native_h128_precomputed_affine_half2_general_column_v4",
}


def _viditq_activation_scales(scale_rows, torch):
    """Pack [G,M] FP16 scales in Atom's 16-row lane-copy order."""
    if not torch.is_tensor(scale_rows) or scale_rows.dtype != torch.float16:
        raise TypeError("ViDiT-Q Atom activation scales must be an FP16 tensor")
    groups, padded_m = map(int, scale_rows.shape)
    if groups <= 0 or padded_m < 128 or padded_m % 128:
        raise ValueError("ViDiT-Q Atom activation M must be padded to a positive 128-row multiple")
    # ldmatrix.x4 consumes the row pair as four adjacent half2 copies:
    # (scale[r], scale[r+8]) repeated four times per lane.
    return (scale_rows.reshape(groups, padded_m // 16, 2, 8)
            .transpose(-2, -1)
            .unsqueeze(-2)
            .expand(-1, -1, -1, 4, -1)
            .reshape(groups, 4 * padded_m)
            .contiguous())


def _nunchaku_bank(bank, module, extension, torch, functional, rank):
    encoded = bank["encoded"]
    rows, columns = map(int, encoded["shape"])
    padded_k = int(encoded["codes"].shape[1]) * 2
    padded_n = ((rows + 127) // 128) * 128
    device = module.weight.device
    weight_codes = (_unpack_int4(encoded["codes"], torch).to(torch.int16) - 8).to(torch.int8)
    if padded_n != rows:
        weight_codes = functional.pad(weight_codes, (0, 0, 0, padded_n - rows), value=0)
    weight_scales = encoded["scales"].transpose(0, 1).contiguous()
    weight_scales = weight_scales.repeat_interleave(2, dim=0)
    if padded_n != rows:
        weight_scales = functional.pad(weight_scales, (0, padded_n - rows), value=1)
    packed_weight = extension.pack_weight(weight_codes.contiguous())
    packed_wscales = extension.pack_weight_scales(weight_scales.contiguous())
    return {
        "packed_weight": packed_weight,
        "packed_wscales": packed_wscales,
        "padded_k": padded_k,
        "padded_n": padded_n,
        "scales": bank["smoothing"],
        "up": bank["up"],
        "down": bank["down"],
        "rank": rank,
        "shape": (rows, columns),
    }


def _viditq_weight_correction(weight_signed, weight_scales_gn, weight_offset_gn, torch):
    """Precompute all W-only affine correction terms in [N,G] orientation."""
    rows, columns = map(int, weight_signed.shape)
    groups = int(weight_scales_gn.shape[0])
    if (columns <= 0 or columns % 128 or tuple(weight_scales_gn.shape) != (groups, rows)
            or tuple(weight_offset_gn.shape) != (groups, rows)):
        raise ValueError("ViDiT-Q correction expects signed W[N,K] and scales/offsets[G,N] with K%128=0")
    group_sum_ng = weight_signed.to(torch.float32).reshape(rows, groups, 128).sum(-1)
    scales_ng = weight_scales_gn.to(torch.float32).transpose(0, 1)
    sum_weight_scale_n = (group_sum_ng * scales_ng).sum(1).contiguous()
    if bool(torch.any(weight_offset_gn != 0).item()):
        scaled_offset_ng = (weight_offset_gn.to(torch.float32)
                            * weight_scales_gn.to(torch.float32)).transpose(0, 1).contiguous()
        sum_scaled_offset_n = scaled_offset_ng.sum(1).contiguous()
    else:
        # The W4A4 common zero-point-8 path needs no stored [N,G] zero matrix.
        scaled_offset_ng = None
        sum_scaled_offset_n = None
    return {"sum_weight_scale_n": sum_weight_scale_n,
            "scaled_offset_ng": scaled_offset_ng,
            "sum_scaled_offset_n": sum_scaled_offset_n}


def _viditq_bank(bank, module, extension, torch, functional, retain_probe_offsets=True):
    encoded = bank["encoded"]
    rows, columns = map(int, encoded["shape"])
    device = module.weight.device
    padded_k = int(encoded["codes"].shape[1]) * (2 if int(encoded["bits"]) == 4 else 1)
    padded_n = ((rows + 127) // 128) * 128
    if int(encoded["bits"]) == 4:
        weight_codes = _unpack_int4(encoded["codes"], torch).to(torch.int16)
    elif int(encoded["bits"]) == 8:
        weight_codes = encoded["codes"].to(torch.int16)
    else:
        raise RuntimeError(f"ViDiT-Q bank has unsupported W{encoded['bits']} codes")
    zeros = encoded["zero_points"].to(torch.int16)
    scales = encoded["scales"]
    if padded_n != rows:
        weight_codes = functional.pad(weight_codes, (0, 0, 0, padded_n - rows), value=8 if int(encoded["bits"]) == 4 else 128)
        zeros = functional.pad(zeros, (0, 0, 0, padded_n - rows),
                               value=8 if int(encoded["bits"]) == 4 else 128)
        scales = functional.pad(scales, (0, 0, 0, padded_n - rows), value=1)
    if int(encoded["bits"]) == 4:
        signed = weight_codes - 8
        packed_codes = ((signed[:, 0::2] & 15) | ((signed[:, 1::2] & 15) << 4)).to(torch.uint8)
        packed_weight = packed_codes.contiguous()
        scales_by_group = scales.transpose(0, 1).to(torch.float16).contiguous()
        weight_offset_gn = (8 - zeros).transpose(0, 1).contiguous()
        corrections = [_viditq_weight_correction(signed, scales_by_group,
                                                  weight_offset_gn, torch)]
        passes = [(packed_weight, scales_by_group,
                   weight_offset_gn if retain_probe_offsets else None, 1)]
    else:
        centered = weight_codes - 128
        high = torch.div(centered, 16, rounding_mode="floor")
        low = torch.remainder(centered, 16) - 8
        high_pack = ((high[:, 0::2] & 15) | ((high[:, 1::2] & 15) << 4)).to(torch.uint8)
        low_pack = ((low[:, 0::2] & 15) | ((low[:, 1::2] & 15) << 4)).to(torch.uint8)
        scales_by_group = scales.transpose(0, 1).to(torch.float16).contiguous()
        high_offset_gn = torch.zeros((), device=device, dtype=torch.int16).expand_as(scales_by_group)
        low_offset_gn = (136 - zeros).transpose(0, 1).contiguous()
        passes = [
            (high_pack.contiguous(), scales_by_group,
             high_offset_gn if retain_probe_offsets else None, 16),
            (low_pack.contiguous(), scales_by_group,
             low_offset_gn if retain_probe_offsets else None, 1),
        ]
        corrections = [
            _viditq_weight_correction(high, scales_by_group, high_offset_gn, torch),
            _viditq_weight_correction(low, scales_by_group, low_offset_gn, torch),
        ]
    return {"passes": passes, "padded_k": padded_k, "padded_n": padded_n,
            "corrections": corrections,
            "weight_bits": int(encoded["bits"]), "scales": bank["smoothing"],
            "signs": bank["signs"], "shape": (rows, columns)}


def install_native(runtime, p, old_root, arm, artifact_dir=None):
    """Install a frozen bank using native INT4 GEMM; return metadata and restore hook."""
    import torch
    import torch.nn.functional as functional

    if arm not in ("svdquant_w4a4", "viditq_w4a4"):
        raise ValueError(f"Unsupported native diffusion arm: {arm}")
    if not torch.cuda.is_available():
        raise RuntimeError("Native diffusion GEMM requires the approved CUDA runtime")
    root = Path(old_root).resolve()
    baseline = _load_baseline(root)
    legacy_protocol = _legacy_protocol(root, p)
    directory = _bank_directory(root, arm, artifact_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modules = dict(sorted(runtime.target_linears.items()))
    rows = baseline._target_rows(modules)
    if (manifest.get("status") != "complete" or manifest.get("arm") != arm
            or manifest.get("protocol") != legacy_protocol.get("protocol")
            or manifest.get("target_manifest") != rows
            or len(rows) != baseline.EXPECTED_MODULES):
        raise RuntimeError("Only the complete frozen Cosmos bank matching these 280 Linear modules can be installed")
    _, _, group, _ = baseline._target_config(legacy_protocol)
    if group != 128:
        raise RuntimeError(f"Frozen diffusion bank group size changed: expected 128, got {group}")
    native_build = None
    if arm == "viditq_w4a4":
        extension, native_build = _viditq_general_extension(artifact_dir)
    else:
        extension = _extension(arm, artifact_dir)
    if not callable(getattr(extension, "matmul", None)):
        raise RuntimeError(f"Native extension for {arm} has no callable matmul; fake quant fallback is disabled")
    if arm == "svdquant_w4a4" and not all(callable(getattr(extension, name, None)) for name in
                                           ("pack_activation", "pack_weight", "pack_activation_scales",
                                            "pack_weight_scales")):
        raise RuntimeError("Nunchaku extension is missing direct-code packing entry points")
    if arm == "viditq_w4a4" and not callable(getattr(extension, "matmul", None)):
        raise RuntimeError("ViDiT-Q Atom extension is missing its official GEMM entry point")

    native_rotations = hadamard_backend = hadamard_receipt = hadamard_receipt_path = None
    if arm == "viditq_w4a4":
        import native_rotations
        rotation_backend = native_rotations._load_backend()
        hadamard_backend, hadamard_receipt, hadamard_receipt_path = (
            native_rotations._load_hadamard_backend(rotation_backend))

    controller = getattr(runtime, "controller", None)
    previous_native_restore = getattr(runtime, "_cosmos_native_restore", None)
    if callable(previous_native_restore):
        previous_native_restore()
    if controller is None:
        raise RuntimeError("Native diffusion requires the runtime controller's CPU BF16 restore shadow")
    controller._restore_current()
    if controller._base_weights_cpu is None:
        controller._backup_weights()
    if controller._base_weights_cpu is None or set(controller._base_weights_cpu) != set(modules):
        raise RuntimeError("Controller CPU BF16 shadow does not cover the complete native target set")
    if any(module.weight.dtype != torch.bfloat16
           or (module.bias is not None and module.bias.dtype != torch.bfloat16)
           for module in modules.values()):
        raise RuntimeError("Native diffusion installation requires restored BF16 target Linear modules")

    prepared = []
    protected = []
    base_device = {name: module.weight.device for name, module in modules.items()}
    old_forwards = {name: module.forward for name, module in modules.items()}
    old_runtime = {key: (hasattr(runtime, key), getattr(runtime, key, None))
                   for key in ("quantization_evidence", "current_arm")}
    old_controller = {key: (hasattr(controller, key), getattr(controller, key, None))
                      for key in ("arm", "metadata")}

    def restore():
        # Restore storage before asking the controller to restore its forwards/state.
        _restore_bf16_parameters(modules, controller, base_device, torch)
        for name, module in modules.items():
            module.forward = old_forwards[name]
            module.__dict__.pop("_cosmos_native_arm", None)
        controller._restore_current()
        controller.arm = "bf16"
        controller.metadata = controller._bf16_metadata()
        runtime.current_arm = "bf16"
        runtime.quantization_evidence = controller.metadata
        for key, (present, value) in old_runtime.items():
            if key not in ("current_arm", "quantization_evidence"):
                continue
            if not present:
                runtime.__dict__.pop(key, None)
        for key, (present, value) in old_controller.items():
            if key not in ("arm", "metadata"):
                continue
            if not present:
                controller.__dict__.pop(key, None)
        runtime.__dict__.pop("_cosmos_native_restore", None)

    for index, (name, module) in enumerate(modules.items()):
        try:
            shape = (int(module.weight.shape[0]), int(module.weight.shape[1]))
            entry = manifest["entries"][index]
            if entry.get("name") != name or tuple(entry.get("shape", ())) != shape:
                raise RuntimeError(f"Baseline manifest target identity differs at module {index}: {name}")
            bank_path = _module_bank_path(directory, entry, index)
            bank = baseline._read_bank(bank_path, arm, name, index, shape, group, torch)
            bank = baseline._device_tree(bank, module.weight.device, torch)
            if arm == "svdquant_w4a4":
                rank = int(bank["rank"])
                if rank != 16:
                    raise RuntimeError(f"Frozen SVDQuant branch rank changed at {name}: expected 16, got {rank}")
                item = _nunchaku_bank(bank, module, extension, torch, functional, rank)
            else:
                item = _viditq_bank(bank, module, extension, torch, functional,
                                    retain_probe_offsets=False)
                if item["weight_bits"] == 8:
                    protected.append(name)
            prepared.append((name, module, item))
            # Release each BF16 CUDA matrix as soon as its checkpoint representation is ready.
            module.register_parameter("weight", None)
        except Exception:
            restore()
            raise

    try:
        metadata = {}
        for name, module, item in prepared:
            if arm == "svdquant_w4a4":
                packed_weight = item["packed_weight"]
                packed_wscales = item["packed_wscales"]
                padded_k, padded_n = item["padded_k"], item["padded_n"]
                smoothing, up, down = item["scales"], item["up"], item["down"]

                out_features = int(item["shape"][0])
                def forward(value, _module=module, _out_features=out_features, _packed_weight=packed_weight,
                            _packed_wscales=packed_wscales, _padded_k=padded_k, _padded_n=padded_n,
                            _smoothing=smoothing, _up=up, _down=down):
                    original_shape = value.shape
                    columns = original_shape[-1]
                    flat = value.reshape(-1, columns)
                    normalized = flat.float() / _smoothing
                    scale = normalized.abs().amax(dim=-1, keepdim=True) / 7
                    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                    activation_codes = normalized.div(safe).round().clamp(-7, 7).to(torch.int8)
                    if _padded_k > columns:
                        activation_codes = functional.pad(activation_codes, (0, _padded_k - columns), value=0)
                    batch = int(flat.shape[0])
                    padded_m = ((batch + 255) // 256) * 256
                    if padded_m != batch:
                        activation_codes = functional.pad(activation_codes, (0, 0, 0, padded_m - batch), value=0)
                    groups = _padded_k // 64
                    a_scales = safe.to(torch.float16).reshape(1, batch).expand(groups, batch)
                    if padded_m != batch:
                        a_scales = functional.pad(a_scales, (0, padded_m - batch), value=1)
                    a_scales = extension.pack_activation_scales(a_scales.contiguous())
                    packed_activation = extension.pack_activation(activation_codes.contiguous())
                    output_shape = _nunchaku_output_shape(batch, padded_m, _padded_n, _out_features)
                    output = torch.empty(output_shape,
                                         device=value.device, dtype=torch.float16)
                    extension.matmul(packed_activation, _packed_weight, output,
                                     a_scales, _packed_wscales)
                    main = output[:, :_out_features].to(torch.bfloat16)
                    if _module.bias is not None:
                        main = main + _module.bias
                    low = functional.linear(normalized.to(torch.bfloat16), _down)
                    result = main + functional.linear(low, _up)
                    return result.reshape(*original_shape[:-1], _out_features)

            else:
                passes = item["passes"]
                corrections = item["corrections"]
                padded_k, padded_n = item["padded_k"], item["padded_n"]
                smoothing, signs = item["scales"], item["signs"]
                weight_bits = item["weight_bits"]
                out_features = int(item["shape"][0])
                def forward(value, _module=module, _out_features=out_features, _passes=passes, _padded_k=padded_k,
                            _padded_n=padded_n, _smoothing=smoothing, _signs=signs,
                            _weight_bits=weight_bits, _corrections=corrections,
                            _hadamard_backend=hadamard_backend, _metadata=metadata):
                    original_shape = value.shape
                    columns = original_shape[-1]
                    flat = value.reshape(-1, columns)
                    transformed = native_rotations._rotate_quarot(
                        flat.float() * _smoothing, _signs, torch, _hadamard_backend)
                    _metadata["native_hadamard_kernel_used"] = True
                    _metadata["native_hadamard_calls"] += 1
                    _metadata["native_hadamard_input_rows"] += int(flat.numel() // 128)
                    x = transformed.float()
                    lower = x.amin(dim=-1, keepdim=True).clamp_max(0)
                    upper = x.amax(dim=-1, keepdim=True).clamp_min(0)
                    scale = (upper - lower) / 15
                    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                    zero = (-lower / safe).round().clamp(0, 15).to(torch.int16)
                    q = x.div(safe).round().add(zero).clamp(0, 15).to(torch.int16)
                    batch = int(flat.shape[0])
                    padded_m = max(128, ((batch + 127) // 128) * 128)
                    if _padded_k > columns:
                        q = functional.pad(q, (0, _padded_k - columns), value=0)
                        q[:, columns:] = zero.expand(batch, _padded_k - columns)
                    a_offset = 8 - zero.reshape(batch)
                    if padded_m != batch:
                        q = torch.cat((q, torch.full((padded_m - batch, _padded_k), 8,
                                                     device=value.device, dtype=torch.int16)), dim=0)
                        a_offset = torch.cat((a_offset, torch.zeros(padded_m - batch,
                                                                    device=value.device, dtype=torch.int16)))
                    a_signed = (q - 8).to(torch.int8).contiguous()
                    packed_activation = ((a_signed[:, 0::2] & 15)
                                         | ((a_signed[:, 1::2] & 15) << 4)).to(torch.uint8).contiguous()
                    a_scales_row = safe.reshape(batch).to(torch.float16)
                    if padded_m != batch:
                        a_scales_row = torch.cat((a_scales_row,
                                                  torch.ones(padded_m - batch, device=value.device,
                                                             dtype=torch.float16)))
                    a_scales = a_scales_row.reshape(1, padded_m).expand(_padded_k // 128, padded_m)
                    a_scales = _viditq_activation_scales(a_scales, torch)
                    a_offset = a_offset.to(torch.float32)
                    a_group_sum = a_signed.to(torch.float32).reshape(padded_m, _padded_k // 128, 128).sum(-1)
                    activation_keeper = torch.zeros((padded_m, 128), device=value.device, dtype=torch.int8)
                    output = torch.zeros((padded_m, _padded_n), device=value.device, dtype=torch.float16)
                    correction_total = torch.zeros((padded_m, _padded_n), device=value.device, dtype=torch.float32)
                    for pass_index, (weight_codes, weight_scales, _weight_offset, multiplier) in enumerate(_passes):
                        if weight_codes.dtype == torch.uint8:
                            packed = weight_codes
                        else:
                            raise RuntimeError("ViDiT-Q weight pass was not packed INT4")
                        correction = _corrections[pass_index]
                        weight_keeper = torch.zeros((128, _padded_n), device=value.device, dtype=torch.int8)
                        weight_keeper_scale = torch.ones(_padded_n, device=value.device, dtype=torch.float16)
                        activation_keeper_scale = torch.ones(4 * padded_m, device=value.device,
                                                            dtype=torch.float16)
                        partial = torch.empty_like(output)
                        extension.matmul(packed_activation, packed, partial, a_scales, weight_scales,
                                         activation_keeper, weight_keeper,
                                         activation_keeper_scale, weight_keeper_scale)
                        a_scale_float = a_scales_row.to(torch.float32)
                        corr = a_offset[:, None] * correction["sum_weight_scale_n"][None, :]
                        if correction["scaled_offset_ng"] is not None:
                            corr = (corr + a_group_sum @ correction["scaled_offset_ng"].transpose(0, 1)
                                    + 128 * a_offset[:, None]
                                    * correction["sum_scaled_offset_n"][None, :])
                        corr = corr * a_scale_float[:, None]
                        if padded_m > batch:
                            corr[batch:] = 0
                        correction_total += (partial.to(torch.float32) + corr) * multiplier
                    result = correction_total[:batch, :_out_features].to(torch.bfloat16)
                    if _module.bias is not None:
                        result = result + _module.bias
                    return result.reshape(*original_shape[:-1], _out_features)

            module.forward = forward
            module._cosmos_native_arm = arm

        metadata.update({
            "protocol": legacy_protocol.get("protocol"),
            "arm": arm,
            "adapter_version": NATIVE_ADAPTER_VERSIONS[arm],
            "implementation": ("nunchaku_official_w4a4_int4_g128_checkpoint_adapter"
                               if arm == "svdquant_w4a4" else
                               "viditq_official_atom_mma_task_local_general_per_column_scale_epilogue_affine_adapter"),
            "target_linears": len(modules),
            "target_weight_numel": manifest["target_weight_numel"],
            "weight_group_size": 128,
            "weight_scale_dtype": "torch.float16",
            "activation_bits": 4,
            "native_int4_kernel_used": True,
            "native_kernel": ("Nunchaku W4A4 INT4" if arm == "svdquant_w4a4" else
                              "ViDiT-Q Atom INT4 MMA + task-local general per-column scale epilogue"),
            "native_weight_format": ("Nunchaku GEMM-specific signed INT4 layout" if arm == "svdquant_w4a4"
                                     else "ViDiT Atom MMA signed INT4 layout with task-local general per-column scale epilogue; W8 layers are exact two-pass signed-W4 decompositions"),
            "original_weight_bank": str(manifest_path),
            "weight_requantized": False,
            "weight_recalibrated": False,
            "new_training_updates": 0,
            "new_closed_loop_episodes": 0,
            "global_svd_rank": manifest.get("global_svd_rank"),
            "protected_modules": protected,
            "execution": "packed_gpu_resident_native_int4_gemm",
            "native_activation_scale_dtype": "torch.float16",
            "packed_gpu_resident": True,
            "packed_weight_gpu_resident": True,
            "bf16_target_weight_numel_resident": 0,
            "bf16_target_weight_numel_cuda_resident": 0,
            "bf16_restore_shadow_cpu_numel": sum(int(m.out_features * m.in_features)
                                                    for m in modules.values()),
            "packed_target_payload_bytes": _unique_cuda_storage_bytes(
                [item for _, _, item in prepared], torch),
            "packed_target_payload_scope": (
                "unique resident CUDA storage of native packed weights, native group scales, "
                "affine offsets, Smooth transforms, ViDiT signs, and the SVD rank-16 branch; "
                "measured from tensor untyped_storage, excluding transient activations/output"),
            "bf16_target_bias_storage_bytes": _unique_cuda_storage_bytes(
                [module.bias for module in modules.values() if module.bias is not None], torch),
            "reference_module_weights_retained": False,
            "adaptation_boundary": manifest.get("adaptation_boundary"),
            "artifact_manifest": str(manifest_path),
            "native_extension": extension.__file__ if hasattr(extension, "__file__") else "already_loaded",
        })
        if arm == "viditq_w4a4":
            metadata.update({
                "native_extension_name": native_build["module_name"],
                "native_extension_build_receipt": native_build["receipt_path"],
                "native_extension_source_commit": native_build["source_commit"],
                "native_extension_build_pbs_job_id": native_build["pbs_job_id"],
                "native_extension_build_host": native_build["host"],
                "native_extension_build_producer": native_build["producer"],
                "native_scale_epilogue": "task-local general per-column FP16 half2 scale products over official Atom INT4 MMA accumulator",
                "rotation": "random_local_signed_hadamard_h128",
                "native_hadamard_backend": "Dao-AILab fast-hadamard-transform CUDA kernel from the pinned QuaRot submodule",
                "native_hadamard_module": native_rotations.HADAMARD_BACKEND_MODULE,
                "native_hadamard_source_repository": native_rotations.HADAMARD_SOURCE_REPOSITORY,
                "native_hadamard_source_commit": hadamard_receipt["source_commit"],
                "native_hadamard_quarot_source_revision": hadamard_receipt["quarot_source_revision"],
                "native_hadamard_submodule_path": hadamard_receipt["submodule_path"],
                "native_hadamard_build_receipt": str(hadamard_receipt_path.resolve()),
                "native_hadamard_build_producer": {
                    "pbs_job_id": hadamard_receipt["pbs_job_id"],
                    "host": hadamard_receipt["host"],
                },
                "native_hadamard_build_pbs_job_id": hadamard_receipt["pbs_job_id"],
                "native_hadamard_build_host": hadamard_receipt["host"],
                "native_hadamard_build_torch": hadamard_receipt.get("torch"),
                "native_hadamard_build_cuda_runtime": hadamard_receipt.get("torch_cuda"),
                "native_hadamard_kernel_used": False,
                "native_hadamard_calls": 0,
                "native_hadamard_input_rows": 0,
                "rotation_sign_multiply_backend": "torch_fp32_elementwise",
            })
        metadata["native_resident_tensor_bytes"] = (
            metadata["packed_target_payload_bytes"] + metadata["bf16_target_bias_storage_bytes"])
        metadata["native_resident_tensor_scope"] = (
            "native packed target payload plus the still-resident BF16 target biases; "
            "excludes transient activations/output and all unquantized non-target model tensors")
        controller.arm = arm
        controller.metadata = metadata
        runtime.quantization_evidence = metadata
        runtime.current_arm = arm
        runtime._cosmos_native_restore = restore
        return metadata, restore
    except Exception:
        restore()
        raise

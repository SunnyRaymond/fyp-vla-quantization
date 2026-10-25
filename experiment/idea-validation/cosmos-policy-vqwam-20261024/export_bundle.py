"""Export a target-weight-free Cosmos DiT plus the packed VQ deployment banks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def _tensor_inventory(values, torch):
    seen, references, logical, unique = set(), 0, 0, 0

    def visit(value):
        nonlocal references, logical, unique
        if torch.is_tensor(value):
            references += 1
            size = value.numel() * value.element_size()
            logical += size
            key = _storage_key(value)
            if key not in seen:
                seen.add(key)
                unique += value.untyped_storage().nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(values)
    return {"tensor_references": references, "logical_tensor_bytes": logical,
            "unique_storage_count": len(seen), "unique_storage_bytes": unique}


def _cpu_shared_state(state, torch):
    memo, result = {}, {}
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            result[name] = tensor
            continue
        value = tensor.detach()
        key = _storage_key(value)
        if key not in memo:
            storage = value.untyped_storage()
            raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                storage, 0, (storage.nbytes(),), (1,))
            memo[key] = raw.cpu().clone() if value.device.type != "cpu" else raw
        typed = memo[key].view(value.dtype)
        result[name] = torch.as_strided(typed, value.shape, value.stride(), value.storage_offset())
    return result


def _target_weight_names(model, target_linears):
    target_ids = {id(module) for module in target_linears.values()}
    return {f"{name}.weight" if name else "weight"
            for name, module in model.named_modules(remove_duplicate=False)
            if id(module) in target_ids}


def _file_in(root, relative):
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes source directory: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def export(runtime, p, artifact_dir, out) -> dict:
    """Package the unquantized DiT and exact packed VQ/transform/U deployment payload."""
    import engine

    source, destination = Path(artifact_dir).resolve(), Path(out).resolve()
    engine._compute_guard(runtime.task_root)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Export output and training artifact directories must be disjoint")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = engine.validate_manifest(manifest, runtime.target_linears,
                                    expected_protocol=engine._protocol_value(p))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    torch = runtime.torch
    model = runtime.model.net
    target_names = {name.removeprefix("net.") for name in _target_weight_names(runtime.model,
                                                                                runtime.target_linears)}
    state = {name: value for name, value in model.state_dict().items() if name not in target_names}
    parameter_rows = [(name, value) for name, value in model.named_parameters()
                      if name not in target_names]
    parameter_numel = sum(int(value.numel()) for _, value in parameter_rows)
    unquantized_parameters = _tensor_inventory([value for _, value in parameter_rows], torch)
    state_inventory = _tensor_inventory(state, torch)

    state_path = destination / "dit_state.pt"
    temporary_state = state_path.with_suffix(".pt.tmp")
    torch.save(_cpu_shared_state(state, torch), temporary_state)
    os.replace(temporary_state, state_path)

    (destination / "bank").mkdir()
    copied_payload_bytes = 0
    bank_unique_bytes = 0
    for index, row in enumerate(rows):
        relative = manifest["modules"][index]["file"]
        source_file = _file_in(source, engine._safe_bank_file(relative))
        payload = torch.load(source_file, map_location="cpu", weights_only=True)
        if not isinstance(payload.get("encoding"), dict) or not isinstance(payload.get("transform"), dict):
            raise ValueError(f"Bank payload is incomplete at {row['name']}")
        bank_unique_bytes += _tensor_inventory(
            (payload["encoding"], payload["transform"]), torch)["unique_storage_bytes"]
        target_file = destination / engine._safe_bank_file(relative)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target_file)
        copied_payload_bytes += source_file.stat().st_size

    rotation_source = _file_in(source, Path(manifest["rotation_file"]))
    rotation_payload = torch.load(rotation_source, map_location="cpu", weights_only=True)
    matrix = rotation_payload.get("matrix")
    if (rotation_payload.get("stream") != "cosmos"
            or rotation_payload.get("parameterization") != "H128_then_U"
            or not torch.is_tensor(matrix) or tuple(matrix.shape) != (128, 128)
            or matrix.dtype != torch.float32):
        raise ValueError("rotation.pt does not satisfy the shared FP32 Cosmos U contract")
    rotation_bytes = _tensor_inventory(rotation_payload, torch)["unique_storage_bytes"]
    shutil.copyfile(rotation_source, destination / "rotation.pt")
    shutil.copyfile(source / "manifest.json", destination / "manifest.json")

    target_weight_numel = sum(row["weight_numel"] for row in rows)
    target_payload_bytes = bank_unique_bytes + rotation_bytes
    target_effective_bpw = 8.0 * target_payload_bytes / target_weight_numel
    whole_parameter_numel = parameter_numel + target_weight_numel
    whole_payload_bytes = unquantized_parameters["unique_storage_bytes"] + target_payload_bytes
    whole_dit_bpw = 8.0 * whole_payload_bytes / whole_parameter_numel
    report = {
        "format": "cosmos_vqwam_export_v1",
        "arm": engine.ARM,
        "source_job": manifest["source_job"],
        "completed_updates": manifest["completed_updates"],
        "target_module_count": len(rows),
        "target_weight_numel": target_weight_numel,
        "target_packed_and_transform_tensor_bytes": target_payload_bytes,
        "target_effective_bpw": target_effective_bpw,
        "unquantized_dit_parameter_numel": parameter_numel,
        "whole_dit_parameter_numel": whole_parameter_numel,
        "whole_dit_effective_bpw": whole_dit_bpw,
        "unquantized_dit_state_unique_tensor_bytes": state_inventory["unique_storage_bytes"],
        "unquantized_dit_parameter_unique_tensor_bytes": unquantized_parameters["unique_storage_bytes"],
        "unique_tensor_payload_bytes": state_inventory["unique_storage_bytes"] + target_payload_bytes,
        "deployment_files_gib": None,
        "deployment_files_bytes": None,
        "copied_bank_serialized_bytes": copied_payload_bytes,
        "dense_target_weights_in_export": False,
        "teacher_optimizer_training_masters_in_export": False,
        "files": ["dit_state.pt", "manifest.json", "rotation.pt", "bank/*.pt"],
        "accounting": "unique backing storage within tensors; serialized file bytes include container overhead",
    }
    report_path = destination / "storage.json"
    base_bytes = sum(path.stat().st_size for path in destination.rglob("*")
                     if path.is_file() and path != report_path)
    report["deployment_files_bytes"] = base_bytes
    for _ in range(4):
        report["deployment_files_gib"] = report["deployment_files_bytes"] / (1024 ** 3)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                          allow_nan=False) + "\n", encoding="utf-8")
        total_bytes = base_bytes + report_path.stat().st_size
        if total_bytes == report["deployment_files_bytes"]:
            break
        report["deployment_files_bytes"] = total_bytes
    return report


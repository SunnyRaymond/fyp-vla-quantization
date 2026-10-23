"""Offline full-matrix weight and fixed-input mechanism analysis for Fast-WAM."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import sys
import time
from typing import Any


SCHEMA = "weight-analysis-v1"
METHODS = ("spinquant_native_w4a4", "row_scale_vq_a4", "micro_scale_vq_a4")
METHOD_ALIASES = ("spinquant", "row_scale_vq", "micro_scale_vq")
METHOD_LANES = ("bf16", *METHODS)
BASIS_LANES = ("raw", "learned_spinquant", "smooth05_hadamard_row",
               "smooth05_hadamard_micro")
REF_WINDOW_ATOL = 5e-6
OUTPUT_EQUIVALENCE_RMSE_TOL = 1e-5


def guard() -> None:
    """Require an actual PBS compute allocation before loading tensors or data."""
    host = socket.gethostname().split(".", 1)[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("Approved PBS compute allocation required")
    nodes = {line.split(".", 1)[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_npz(np: Any, path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
    os.replace(temporary, path)


def _load_sources(capture_protocol: dict[str, Any], torch_threads: int) -> tuple[Any, ...]:
    roots = capture_protocol["roots"]
    paths = [
        roots["pilot"], capture_protocol["rotation_root"], roots["old_banks"],
        capture_protocol["source_root"], capture_protocol["micro_root"],
    ]
    sys.path[:0] = [str(Path(path)) for path in paths]
    import numpy as np
    import torch
    import codebook
    import micro_codebook
    import quantization
    import rotation
    import smooth_vq

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return np, torch, codebook, micro_codebook, quantization, rotation, smooth_vq


def _read_npz(np: Any, root: Path, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / record["file"]
    if not path.is_file():
        raise FileNotFoundError(f"capture record is missing: {record['file']}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("record_id") != record.get("record_id"):
        raise RuntimeError(f"capture record identity mismatch: {record.get('record_id')}")
    return arrays, metadata


def _record_index(records: list[dict[str, Any]], *, kind: str,
                  arms: tuple[str, ...] | None = None) -> dict[tuple, list[dict[str, Any]]]:
    result: dict[tuple, list[dict[str, Any]]] = {}
    for row in records:
        if row.get("kind") != kind or (arms is not None and row.get("arm") not in arms):
            continue
        if kind == "weight":
            key = (int(row["module_index"]), row["arm"])
        elif kind == "activation":
            key = (int(row["module_index"]), int(row["case_id"]), row["stage"], int(row["step"]))
        else:
            key = (int(row["module_index"]), int(row["case_id"]), row["stage"], int(row["step"]))
        result.setdefault(key, []).append(row)
    return result


def _primary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"missing primary {label}")
    indexed = []
    for row in rows:
        if row.get("call_index") is None:
            raise RuntimeError(f"primary {label} has null call_index")
        indexed.append((int(row["call_index"]), str(row.get("record_id", "")), row))
    indexed.sort(key=lambda item: (item[0], item[1]))
    if len(indexed) > 1 and indexed[0][0] == indexed[1][0]:
        raise RuntimeError(f"primary {label} has tied lowest call_index")
    return indexed[0][2]


def _module_rows(capture_protocol: dict[str, Any], analysis_protocol: dict[str, Any],
                 manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("protocol") != capture_protocol.get("protocol"):
        raise RuntimeError("capture manifest does not match the frozen capture protocol")
    if manifest.get("status") != "complete":
        raise RuntimeError("weight analysis requires the complete existing capture manifest")
    if not str(manifest.get("primary_activation_record_rule", "")).startswith(
            "lowest call_index for"):
        raise RuntimeError("capture manifest does not declare the lowest-call primary rule")
    expected = {
        (stream, int(block), suffix)
        for stream in analysis_protocol["streams"]
        for block in analysis_protocol["blocks"]
        for suffix in analysis_protocol["module_suffixes"]
    }
    by_name: dict[str, dict[str, Any]] = {}
    for row in manifest["records"]:
        if row.get("kind") != "weight" or row.get("arm") != "bf16":
            continue
        name = str(row["module"])
        match = re.fullmatch(r"(video_expert|action_expert)\.blocks\.(\d+)\.(.+)", name)
        if not match or (match.group(1), int(match.group(2)), match.group(3)) not in expected:
            continue
        item = {"module": name, "module_index": int(row["module_index"]),
                "stream": match.group(1), "block_index": int(match.group(2)),
                "suffix": match.group(3), "shape": [int(v) for v in row["weight_shape"]]}
        previous = by_name.setdefault(name, item)
        if previous != item:
            raise RuntimeError(f"conflicting capture identities for module {name}")
    modules = sorted(by_name.values(), key=lambda row: row["module_index"])
    if len(modules) != int(analysis_protocol["expected_modules"]):
        raise RuntimeError(f"expected {analysis_protocol['expected_modules']} modules, found {len(modules)}")
    if len({row["module_index"] for row in modules}) != len(modules):
        raise RuntimeError("selected module indices are not unique")
    return modules


def _tensor_dict(torch: Any, values: dict[str, Any]) -> dict[str, Any]:
    return {key: torch.as_tensor(value).detach().cpu() for key, value in values.items()}


def _captured_transform(np: Any, root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = root / record["transform_file"]
    if not path.is_file():
        raise FileNotFoundError(f"captured transform is missing: {record['transform_file']}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files if key != "metadata_json"}


def _assert_transform_matches(torch: Any, transform: dict[str, Any],
                              captured: dict[str, Any], label: str) -> None:
    tensor_fields = [key for key, value in transform.items() if torch.is_tensor(value)]
    for key in tensor_fields:
        if key not in captured:
            raise RuntimeError(f"{label} transform capture lacks {key}")
        actual = transform[key].detach().cpu().float().numpy()
        expected = captured[key].astype("float32", copy=False)
        if actual.shape != expected.shape or not bool((abs(actual - expected) <= 0).all()):
            raise RuntimeError(f"{label} transform does not match its captured source artifact: {key}")


def _transforms_equal(torch: Any, first: dict[str, Any], second: dict[str, Any]) -> bool:
    if set(first) != set(second):
        return False
    for key in first:
        left, right = first[key], second[key]
        if torch.is_tensor(left) or torch.is_tensor(right):
            if not (torch.is_tensor(left) and torch.is_tensor(right)
                    and left.dtype == right.dtype and left.shape == right.shape
                    and torch.equal(left.detach().cpu(), right.detach().cpu())):
                return False
        elif left != right:
            return False
    return True


def _checkpoint_weight(torch: Any, state: dict[str, Any], module: dict[str, Any],
                       analysis_protocol: dict[str, Any]) -> Any:
    stream = module["stream"]
    prefix = analysis_protocol["checkpoint_prefixes"][stream]
    suffix = module["module"].split(stream + ".", 1)[1]
    key = f"{prefix}.{suffix}.weight"
    if key not in state:
        raise KeyError(f"checkpoint state is missing target Linear weight {key}")
    value = state[key]
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(module["shape"]):
        raise RuntimeError(f"checkpoint tensor shape differs for {module['module']}: {key}")
    # Runtime constructs the model in BF16 before loading this checkpoint.
    return value.detach().to(dtype=torch.bfloat16).float().contiguous()


def _decode_spin(torch: Any, quantization: Any, bank: dict[str, Any],
                 shape: tuple[int, int]) -> tuple[Any, Any]:
    out_features, in_features = shape
    padded = int(bank["padded_in_features"])
    packed = bank["packed_weight"]
    scales_bf16 = bank["weight_scales_bf16"]
    if tuple(bank["shape"]) != shape or packed.dtype != torch.uint8:
        raise RuntimeError("SpinQuant packed bank shape or dtype is invalid")
    if padded < in_features or padded % 128 or tuple(packed.shape) != (out_features, padded // 2):
        raise RuntimeError("SpinQuant packed bank has invalid padded input width")
    if tuple(scales_bf16.shape) != (out_features, padded // 128) or scales_bf16.dtype != torch.bfloat16:
        raise RuntimeError("SpinQuant BF16 G128 scale shape/dtype is invalid")
    codes = quantization._unpack_nibbles(packed, padded, signed=True).float()
    scale_profile = scales_bf16.float()
    dense = (codes * scale_profile.repeat_interleave(128, dim=1))[:, :in_features]
    return dense.contiguous(), scale_profile.contiguous()


def _window_bounds(length: int, count: int) -> int:
    return min(length, count)


def _stats(np: Any, value: Any) -> dict[str, Any]:
    flat = value.detach().cpu().numpy().astype("float64", copy=False).reshape(-1)
    absolute = np.abs(flat)
    abs_mean = float(absolute.mean())
    abs_std = float(absolute.std())
    return {
        "count": int(flat.size), "min": float(flat.min()), "max": float(flat.max()),
        "mean": float(flat.mean()), "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "mean_abs": abs_mean, "std_abs": abs_std,
        "cv_abs": abs_std / abs_mean if abs_mean > 0 else None,
        "zero_fraction": float(np.count_nonzero(flat == 0) / max(flat.size, 1)),
    }


def _weight_distribution_stats(np: Any, value: Any) -> dict[str, Any]:
    matrix = value.detach().cpu().numpy().astype("float64", copy=False)
    result = _stats(np, value)
    input_rms = np.sqrt(np.square(matrix).mean(axis=0))
    output_rms = np.sqrt(np.square(matrix).mean(axis=1))
    input_mean = float(input_rms.mean())
    output_mean = float(output_rms.mean())
    global_rms = float(np.sqrt(np.square(matrix).mean()))
    result.update({
        "input_channel_rms_cv": float(input_rms.std() / input_mean) if input_mean else None,
        "output_channel_rms_cv": float(output_rms.std() / output_mean) if output_mean else None,
        "max_abs_to_rms": float(np.abs(matrix).max() / global_rms) if global_rms else None,
    })
    return result


def _ratio(numerator: float, denominator: float) -> tuple[float | None, str]:
    if denominator > 0:
        return numerator / denominator, "measured"
    return (0.0, "zero_reference_zero_error") if numerator == 0 else (
        None, "zero_reference_nonzero_error")


def _relative_improvement(baseline: float | None, candidate: float | None) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {"value_pct": None, "status": "undefined_nmse"}
    if baseline > 0:
        return {"value_pct": 100.0 * (baseline - candidate) / baseline, "status": "measured"}
    if candidate == 0:
        return {"value_pct": 0.0, "status": "both_zero"}
    return {"value_pct": None, "status": "zero_baseline_nonzero_candidate"}


def _weight_error(np: Any, reference: Any, actual: Any) -> tuple[dict[str, Any], Any]:
    difference = actual.float() - reference.float()
    ref64 = reference.detach().cpu().numpy().astype("float64", copy=False)
    diff64 = difference.detach().cpu().numpy().astype("float64", copy=False)
    numerator = float(np.square(diff64).sum())
    denominator = float(np.square(ref64).sum())
    nmse, status = _ratio(numerator, denominator)
    return {
        "mse": numerator / max(diff64.size, 1),
        "rmse": float(np.sqrt(numerator / max(diff64.size, 1))),
        "nmse": nmse, "nmse_status": status,
        "relative_frobenius": float(np.sqrt(numerator / denominator)) if denominator > 0 else None,
        "max_abs_error": float(np.abs(diff64).max()),
        "reference_rms": float(np.sqrt(denominator / max(ref64.size, 1))),
        "actual_rms": float(np.sqrt(np.square(actual.detach().cpu().numpy().astype("float64")).mean())),
    }, difference


def _block_error(np: Any, reference: Any, difference: Any, block_size: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    ref = reference.detach().cpu().numpy().astype("float64", copy=False)
    err = difference.detach().cpu().numpy().astype("float64", copy=False)
    rows, cols = ref.shape
    col_blocks = math.ceil(cols / block_size)
    padded_cols = col_blocks * block_size
    if padded_cols != cols:
        ref = np.pad(ref, ((0, 0), (0, padded_cols - cols)))
        err = np.pad(err, ((0, 0), (0, padded_cols - cols)))
    ref_tiles = ref.reshape(rows, col_blocks, block_size)
    err_tiles = err.reshape(rows, col_blocks, block_size)
    ref_energy = np.square(ref_tiles).sum(axis=2)
    error_energy = np.square(err_tiles).sum(axis=2)
    nmse = np.full((rows, col_blocks), np.nan, dtype=np.float64)
    status = np.zeros((rows, col_blocks), dtype=np.uint8)
    nonzero_reference = ref_energy > 0
    np.divide(error_energy, ref_energy, out=nmse, where=nonzero_reference)
    zero_zero = (~nonzero_reference) & (error_energy == 0)
    zero_nonzero = (~nonzero_reference) & (error_energy != 0)
    nmse[zero_zero] = 0.0
    status[zero_zero] = 1
    status[zero_nonzero] = 2
    nmse = nmse.astype(np.float32)
    ref_energy = ref_energy.astype(np.float32)
    error_energy = error_energy.astype(np.float32)
    valid = nmse[np.isfinite(nmse)]
    total_nmse, _ = _ratio(float(error_energy.sum()), float(ref_energy.sum()))
    summary = {
        "block_size": block_size, "output_rows": rows, "input_blocks": col_blocks,
        "tile_definition": "one output channel row by one contiguous B32 input-channel block; no pooling across rows",
        "defined_block_count": int(valid.size),
        "zero_reference_block_count": int((status != 0).sum()),
        "zero_reference_nonzero_error_count": int((status == 2).sum()),
        "mean_block_nmse": float(valid.mean()) if valid.size else None,
        "median_block_nmse": float(np.median(valid)) if valid.size else None,
        "p95_block_nmse": float(np.percentile(valid, 95)) if valid.size else None,
        "max_block_nmse": float(valid.max()) if valid.size else None,
        "element_weighted_nmse": total_nmse,
        "zero_energy_rule": "0 for zero/zero tiles; NaN in NPZ and explicit status 2 when reference energy is zero but error is nonzero",
    }
    return nmse, ref_energy, error_energy, {
        **summary, "status": status
    }


def _hist(np: Any, arrays: dict[str, Any], basis: str,
          values: dict[str, Any]) -> tuple[str, dict[str, str]]:
    maximum = max((float(value.detach().abs().max()) for value in values.values()), default=0.0)
    maximum = maximum or 1.0
    edges = np.linspace(-maximum, maximum, 257, dtype=np.float32)
    edge_key = f"hist_edges_{basis}"
    arrays[edge_key] = edges
    count_keys = {}
    for label, value in values.items():
        key = f"hist_{label}"
        arrays[key] = np.histogram(value.detach().cpu().numpy().reshape(-1), bins=edges)[0].astype(np.int64)
        count_keys[label] = key
    return edge_key, count_keys


def _top_channels(np: Any, values: Any, count: int = 5) -> list[dict[str, Any]]:
    profile = values.detach().cpu().numpy().astype("float64", copy=False)
    order = np.argsort(profile)[::-1][:min(count, profile.size)]
    return [{"channel": int(index), "rms": float(profile[index])} for index in order]


def _basis_transform(
    torch: Any, method: str, stream: str,
    transforms: dict[str, Any], rotation: Any, smooth_vq: Any, raw: Any,
) -> Any:
    if method == METHODS[0]:
        stream = rotation._stream_for_name(stream)
        return rotation._rotate_with_matrix(
            raw, transforms["spin"][stream]["matrix"], transforms["spin"][stream]["signs"])
    if method == METHODS[1]:
        return smooth_vq.transform_input(raw, transforms["row"])
    return smooth_vq.transform_input(raw, transforms["micro"])


def _basis_weight(
    torch: Any, method: str, raw_weight: Any, transforms: dict[str, Any],
    rotation: Any, smooth_vq: Any,
) -> Any:
    if method == METHODS[0]:
        stream = rotation._stream_for_name(transforms["stream"])
        return rotation._rotate_with_matrix(
            raw_weight, transforms["spin"][stream]["matrix"], transforms["spin"][stream]["signs"])
    return smooth_vq.transform_weight(
        raw_weight, transforms["row" if method == METHODS[1] else "micro"])


def _module_compute(
    np: Any, torch: Any, codebook: Any, micro_codebook: Any, quantization: Any,
    rotation: Any, smooth_vq: Any, capture_protocol: dict[str, Any],
    analysis_protocol: dict[str, Any], capture_root: Path, manifest: dict[str, Any],
    module: dict[str, Any], checkpoint_state: dict[str, Any],
    record_maps: dict[str, Any], rotation_payload: dict[str, Any],
    spin_manifest_doc: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    name, module_index = module["module"], int(module["module_index"])
    shape = tuple(module["shape"])
    out_features, in_features = shape
    raw_weight = _checkpoint_weight(torch, checkpoint_state, module, analysis_protocol)
    if tuple(raw_weight.shape) != shape:
        raise RuntimeError(f"checkpoint BF16 weight shape mismatch for {name}")

    weight_records = {}
    for arm in ("bf16", *METHODS):
        rows = record_maps["weights"].get((module_index, arm), [])
        if len(rows) != 1:
            raise RuntimeError(f"missing captured weight window: module={name}, arm={arm}")
        record = rows[0]
        arrays, _meta = _read_npz(np, capture_root, record)
        weight_records[arm] = (record, arrays)
        expected_window = (_window_bounds(out_features, int(analysis_protocol["weight_plot_window"]["output_count"])),
                           _window_bounds(in_features, int(analysis_protocol["weight_plot_window"]["input_count"])))
        if tuple(arrays["weight"].shape) != expected_window or tuple(arrays["reference"].shape) != expected_window:
            raise RuntimeError(f"captured weight window shape differs for {name}/{arm}")

    spin_root = Path(capture_protocol["spinquant_bank"])
    source_rotation = rotation_payload
    stream_rotation = source_rotation["streams"][rotation._stream_for_name(module["stream"])]
    capture_spin_transform = _captured_transform(np, capture_root, weight_records[METHODS[0]][0])
    _assert_transform_matches(torch, stream_rotation, capture_spin_transform, f"{name} learned SpinQuant")
    transforms: dict[str, Any] = {"spin": source_rotation["streams"], "stream": module["stream"]}

    row_root = Path(capture_protocol["source_root"]) / "results/full/banks/row_scale_vq_a4"
    micro_root = Path(capture_protocol["micro_root"]) / "results/full/banks/micro_scale_vq_a4"
    row_bank = torch.load(row_root / f"{module_index:04d}.pt", map_location="cpu", weights_only=True)
    micro_bank = torch.load(micro_root / f"{module_index:04d}.pt", map_location="cpu", weights_only=True)
    for bank, label in ((row_bank, "row-scale VQ"), (micro_bank, "micro-scale VQ")):
        if bank.get("module") != name or tuple(bank.get("encoded", {}).get("shape", ())) != shape:
            raise RuntimeError(f"{label} bank module identity/shape mismatch for {name}")
        if not isinstance(bank.get("transform"), dict):
            raise RuntimeError(f"{label} bank is missing its saved transform for {name}")
    transforms["row"], transforms["micro"] = row_bank["transform"], micro_bank["transform"]
    capture_row_transform = _captured_transform(np, capture_root, weight_records[METHODS[1]][0])
    capture_micro_transform = _captured_transform(np, capture_root, weight_records[METHODS[2]][0])
    _assert_transform_matches(torch, transforms["row"], capture_row_transform, f"{name} row VQ")
    _assert_transform_matches(torch, transforms["micro"], capture_micro_transform, f"{name} micro VQ")
    row_micro_same = _transforms_equal(torch, transforms["row"], transforms["micro"])
    if not row_micro_same:
        raise RuntimeError(f"Row-scale and Micro-scale VQ banks do not share the frozen Smooth basis for {name}")

    spin_entry = next((row for row in spin_manifest_doc["modules"] if row.get("name") == name), None)
    if spin_entry is None:
        raise RuntimeError(f"learned SpinQuant manifest lacks {name}")
    spin_bank = torch.load(spin_root / spin_entry["file"], map_location="cpu", weights_only=True)
    if spin_bank.get("module") != name:
        raise RuntimeError(f"learned SpinQuant packed file identity mismatch for {name}")
    decoded_spin, spin_scales = _decode_spin(torch, quantization, spin_bank, shape)
    encoded_row, encoded_micro = row_bank["encoded"], micro_bank["encoded"]
    decoded_row = codebook.decode_vq(encoded_row).float()
    decoded_micro = micro_codebook.decode_vq(encoded_micro).float()
    scale_profiles = {
        METHODS[0]: spin_scales,
        METHODS[1]: encoded_row["row_scales"].float().reshape(out_features, 1),
        METHODS[2]: micro_codebook.block_scales(encoded_micro).float(),
    }
    decoded = {METHODS[0]: decoded_spin, METHODS[1]: decoded_row, METHODS[2]: decoded_micro}
    if any(tuple(weight.shape) != shape for weight in decoded.values()):
        raise RuntimeError(f"a decoded bank weight shape differs from {name}: expected {shape}")

    transform_file_ids = {}
    for arm in METHODS:
        record = weight_records[arm][0]
        transform_file_ids[arm] = record["transform_id"]

    arrays: dict[str, Any] = {}
    weight_dist: dict[str, Any] = {"raw_reference": _weight_distribution_stats(np, raw_weight)}
    distribution_values = {"raw": {"raw_bf16": raw_weight}, "spin": {}, "smooth": {}}
    same_basis_refs = {}
    errors = {}
    block_summaries = {}
    for arm in METHODS:
        ref = _basis_weight(torch, arm, raw_weight, transforms, rotation, smooth_vq)
        actual = decoded[arm]
        if arm == METHODS[0]:
            distribution_values["spin"].update({"spin_reference": ref, "spin_actual": actual})
        elif arm == METHODS[1]:
            distribution_values["smooth"]["row_reference"] = ref
            distribution_values["smooth"]["row_actual"] = actual
        else:
            distribution_values["smooth"]["micro_reference"] = ref
            distribution_values["smooth"]["micro_actual"] = actual
        metrics, diff = _weight_error(np, ref, actual)
        same_basis_refs[arm], errors[arm] = ref, diff
        metrics["reference_distribution"] = _weight_distribution_stats(np, ref)
        metrics["actual_distribution"] = _weight_distribution_stats(np, actual)
        nmse, ref_energy, err_energy, block_summary = _block_error(
            np, ref, diff, int(analysis_protocol["block_size"]))
        block_status = block_summary.pop("status")
        arrays[f"block_nmse__{arm}"] = nmse
        arrays[f"block_status__{arm}"] = block_status
        valid = nmse[np.isfinite(nmse)]
        hotspot_order = np.argsort(np.nan_to_num(nmse, nan=-1.0).reshape(-1))[::-1]
        block_cols = nmse.shape[1]
        block_summary["hotspots"] = [
            {"output_channel": int(index // block_cols), "input_block": int(index % block_cols),
             "nmse": float(nmse.reshape(-1)[index]),
             "reference_energy": float(ref_energy.reshape(-1)[index]),
             "error_energy": float(err_energy.reshape(-1)[index])}
            for index in hotspot_order[:8] if np.isfinite(nmse.reshape(-1)[index])
        ]
        block_summaries[arm] = block_summary
        weight_dist[arm] = metrics

    raw_basis_edges, raw_histograms = _hist(np, arrays, "raw", distribution_values["raw"])
    spin_basis_edges, spin_histograms = _hist(np, arrays, "spin", distribution_values["spin"])
    smooth_basis_edges, smooth_histograms = _hist(np, arrays, "smooth", distribution_values["smooth"])

    weight_window = analysis_protocol["weight_plot_window"]
    weight_r0, weight_c0 = int(weight_window["output_start"]), int(weight_window["input_start"])
    window_rows = min(out_features - weight_r0, int(weight_window["output_count"]))
    window_cols = min(in_features - weight_c0, int(weight_window["input_count"]))
    if weight_r0 < 0 or weight_c0 < 0 or window_rows <= 0 or window_cols <= 0:
        raise RuntimeError(f"invalid weight distribution window for {name}: {weight_window}")
    if (window_rows != int(weight_window["output_count"])
            or window_cols != int(weight_window["input_count"])):
        raise RuntimeError(f"full requested weight surface window is unavailable for {name}: {shape}")
    crop = lambda value: value[weight_r0:weight_r0 + window_rows,
                               weight_c0:weight_c0 + window_cols].cpu().float()
    arrays["weight_reference_window"] = torch.stack(
        [crop(raw_weight), *[crop(same_basis_refs[arm]) for arm in METHODS]]
    ).numpy()
    arrays["weight_actual_window"] = torch.stack(
        [crop(raw_weight), *[crop(decoded[arm]) for arm in METHODS]]
    ).numpy()
    arrays["weight_error_window"] = torch.stack([crop(errors[arm]) for arm in METHODS]).numpy()
    weight_references = [raw_weight, *[same_basis_refs[arm] for arm in METHODS]]
    weight_actuals = [raw_weight, *[decoded[arm] for arm in METHODS]]
    arrays["weight_input_rms_reference"] = torch.stack([
        torch.sqrt(torch.mean(weight.float() * weight.float(), dim=0)) for weight in weight_references
    ]).cpu().numpy()
    arrays["weight_input_rms_actual"] = torch.stack([
        torch.sqrt(torch.mean(weight.float() * weight.float(), dim=0)) for weight in weight_actuals
    ]).cpu().numpy()
    arrays["weight_output_rms_reference"] = torch.stack([
        torch.sqrt(torch.mean(weight.float() * weight.float(), dim=1)) for weight in weight_references
    ]).cpu().numpy()
    arrays["weight_output_rms_actual"] = torch.stack([
        torch.sqrt(torch.mean(weight.float() * weight.float(), dim=1)) for weight in weight_actuals
    ]).cpu().numpy()

    parity = {}
    for arm in ("bf16", *METHODS):
        _record, captured = weight_records[arm]
        if arm == "bf16":
            actual_window = raw_weight[weight_r0:weight_r0 + window_rows,
                                       weight_c0:weight_c0 + window_cols]
            ref_window = raw_weight[weight_r0:weight_r0 + window_rows,
                                    weight_c0:weight_c0 + window_cols]
        else:
            actual_window = decoded[arm][weight_r0:weight_r0 + window_rows,
                                         weight_c0:weight_c0 + window_cols]
            ref_window = same_basis_refs[arm][weight_r0:weight_r0 + window_rows,
                                              weight_c0:weight_c0 + window_cols]
        actual_diff = np.abs(actual_window.cpu().numpy().astype("float64")
                             - captured["weight"].astype("float64"))
        ref_diff = np.abs(ref_window.cpu().numpy().astype("float64")
                          - captured["reference"].astype("float64"))
        actual_max = float(actual_diff.max()) if actual_diff.size else 0.0
        ref_max = float(ref_diff.max()) if ref_diff.size else 0.0
        if actual_max != 0:
            raise RuntimeError(f"decoded {arm} actual weight window differs from capture for {name}: {actual_max}")
        if ref_max > REF_WINDOW_ATOL:
            raise RuntimeError(f"decoded {arm} same-basis reference window differs for {name}: {ref_max}")
        parity[arm] = {
            "actual_max_abs": actual_max, "actual_exact": True,
            "reference_max_abs": ref_max, "reference_atol": REF_WINDOW_ATOL,
            "window_shape": [window_rows, window_cols],
        }

    activation_index = record_maps["activations"]
    paired_index = record_maps["paired"]
    cases = [int(value) for value in analysis_protocol["cases"]]
    steps = [int(value) for value in analysis_protocol["snapshot_steps"]]
    context_count = len(cases) * len(steps)
    context_ids = []
    contexts = []
    window = analysis_protocol["output_plot_window"]
    token_start = int(window["token_start"])
    channel_start = int(window["output_start"])
    first_stage = "video" if module["stream"] == "video_expert" else "action"
    first_key = (module_index, cases[0], first_stage, steps[0])
    first_paired = _primary(paired_index.get(first_key, []), f"{name}/{first_key}/paired raw input")
    first_pair_rows = first_paired.get("metadata", {}).get("token_rows")
    if (not isinstance(first_pair_rows, list) or len(first_pair_rows) != 2
            or int(first_pair_rows[1]) <= int(first_pair_rows[0])):
        raise RuntimeError(f"paired input lacks valid token_rows metadata for {name}/{first_key}")
    module_token_count = int(first_pair_rows[1]) - int(first_pair_rows[0])
    token_count = min(int(window["token_count"]), module_token_count - token_start)
    channel_count = min(int(window["output_count"]), out_features - channel_start)
    if token_start < 0 or channel_start < 0 or token_count <= 0 or channel_count <= 0:
        raise RuntimeError(f"invalid fixed-input output window for {name}: {window}")
    if channel_count != int(window["output_count"]):
        raise RuntimeError(f"full requested fixed-input output-channel window is unavailable for {name}: {shape}")
    input_rms = torch.empty((context_count, 4, in_features), dtype=torch.float32)
    output_ref_rms = torch.empty((context_count, 4, out_features), dtype=torch.float32)
    output_actual_rms = torch.empty_like(output_ref_rms)
    output_error_rms = torch.empty((context_count, 3, out_features), dtype=torch.float32)
    token_actual_rms = torch.empty((context_count, 4, module_token_count), dtype=torch.float32)
    token_error_rms = torch.empty((context_count, 3, module_token_count), dtype=torch.float32)
    output_nmse = torch.empty((context_count, 3), dtype=torch.float64)
    output_nmse_status = torch.zeros((context_count, 3), dtype=torch.uint8)
    basis_equivalence_nmse = torch.empty_like(output_nmse)
    basis_equivalence_max_abs = torch.empty_like(output_nmse)
    saved_output_check = torch.empty((context_count,), dtype=torch.float64)
    output_ref_window = torch.empty((context_count, 4, token_count, channel_count), dtype=torch.float32)
    output_actual_window = torch.empty_like(output_ref_window)
    output_error_window = torch.empty((context_count, 3, token_count, channel_count), dtype=torch.float32)

    for case_index, case in enumerate(cases):
        for step_index, step in enumerate(steps):
            stage = "video" if module["stream"] == "video_expert" else "action"
            context_id = f"{case}|{stage}|{step}"
            context_index = case_index * len(steps) + step_index
            context_ids.append(context_id)
            key = (module_index, int(case), stage, int(step))
            activation = _primary(activation_index.get(key, []), f"{name}/{context_id}/BF16 input")
            paired = _primary(paired_index.get(key, []), f"{name}/{context_id}/paired raw input")
            if int(activation["call_index"]) != int(paired["call_index"]):
                raise RuntimeError(f"paired raw input does not match the selected BF16 call for {name}/{context_id}")
            input_arrays, input_meta = _read_npz(np, capture_root, paired)
            saved_arrays, saved_meta = _read_npz(np, capture_root, activation)
            raw_x = torch.from_numpy(input_arrays["paired_raw"]).float()
            if raw_x.ndim != 2 or raw_x.shape[1] != in_features:
                raise RuntimeError(f"paired raw input shape differs for {name}/{context_id}: {tuple(raw_x.shape)}")
            pair_rows = input_meta.get("token_rows")
            activation_rows = saved_meta.get("token_rows")
            if (not isinstance(pair_rows, list) or len(pair_rows) != 2
                    or not isinstance(activation_rows, list) or len(activation_rows) != 2
                    or [int(value) for value in pair_rows] != [int(value) for value in activation_rows]):
                raise RuntimeError(f"paired input and activation token_rows metadata differ for {name}/{context_id}")
            pair_start, pair_end = (int(value) for value in pair_rows)
            expected_ids = np.arange(pair_start, pair_end, dtype=np.int64)
            pair_ids = np.asarray(input_arrays.get("token_row_ids", []), dtype=np.int64)
            activation_ids = np.asarray(saved_arrays.get("token_row_ids", []), dtype=np.int64)
            if (pair_end - pair_start != raw_x.shape[0]
                    or pair_ids.shape != (raw_x.shape[0],)
                    or activation_ids.shape != (raw_x.shape[0],)
                    or not np.array_equal(pair_ids, expected_ids)
                    or not np.array_equal(activation_ids, expected_ids)):
                raise RuntimeError(f"saved token_row_ids do not match token_rows metadata for {name}/{context_id}")
            if raw_x.shape[0] != module_token_count:
                raise RuntimeError(
                    f"selected contexts have inconsistent saved token-row counts for {name}/{context_id}: "
                    f"expected {module_token_count}, found {raw_x.shape[0]}")
            if token_start + token_count > raw_x.shape[0]:
                raise RuntimeError(f"requested output window exceeds saved token rows for {name}/{context_id}")
            raw_y = raw_x @ raw_weight.T
            input_basis = {"raw": raw_x}
            refs_y = {"bf16": raw_y}
            actual_y = {"bf16": raw_y}
            errors_y = {}
            basis_eq = {}
            input_basis["spin"] = _basis_transform(
                torch, METHODS[0], module["stream"], transforms, rotation, smooth_vq, raw_x)
            input_basis["row"] = _basis_transform(
                torch, METHODS[1], module["stream"], transforms, rotation, smooth_vq, raw_x)
            input_basis["micro"] = _basis_transform(
                torch, METHODS[2], module["stream"], transforms, rotation, smooth_vq, raw_x)
            for arm, lane in zip(METHODS, ("spin", "row", "micro")):
                x_basis = input_basis[lane]
                ref_y = x_basis @ same_basis_refs[arm].T
                error_y = x_basis @ errors[arm].T
                equivalent_diff = ref_y - raw_y
                denominator = float(torch.sum(raw_y.double() * raw_y.double()).item())
                numerator = float(torch.sum(equivalent_diff.double() * equivalent_diff.double()).item())
                eq_nmse, eq_status = _ratio(numerator, denominator)
                eq_rmse_relative = math.sqrt(eq_nmse) if eq_nmse is not None else None
                if eq_rmse_relative is None or eq_rmse_relative > OUTPUT_EQUIVALENCE_RMSE_TOL:
                    raise RuntimeError(
                        f"{name}/{context_id}/{arm} same-basis reference output fails equivalence gate: "
                        f"relative_rmse={eq_rmse_relative}, status={eq_status}")
                actual = ref_y + error_y
                refs_y[arm], errors_y[arm], actual_y[arm] = ref_y, error_y, actual
                basis_eq[arm] = {
                    "nmse_vs_raw_reference": eq_nmse, "relative_rmse_vs_raw_reference": eq_rmse_relative,
                    "max_abs_difference": float(torch.max(torch.abs(equivalent_diff)).item()),
                    "status": eq_status, "tolerance_relative_rmse": OUTPUT_EQUIVALENCE_RMSE_TOL,
                }

            context_input = torch.stack([
                input_basis["raw"], input_basis["spin"],
                input_basis["row"], input_basis["micro"],
            ])
            context_ref = torch.stack([refs_y["bf16"], refs_y[METHODS[0]],
                                       refs_y[METHODS[1]], refs_y[METHODS[2]]])
            context_actual = torch.stack([actual_y["bf16"], actual_y[METHODS[0]],
                                          actual_y[METHODS[1]], actual_y[METHODS[2]]])
            context_errors = torch.stack([errors_y[arm] for arm in METHODS])
            input_rms[context_index] = torch.sqrt(torch.mean(context_input * context_input, dim=1))
            output_ref_rms[context_index] = torch.sqrt(torch.mean(context_ref * context_ref, dim=1))
            output_actual_rms[context_index] = torch.sqrt(torch.mean(context_actual * context_actual, dim=1))
            output_error_rms[context_index] = torch.sqrt(torch.mean(context_errors * context_errors, dim=1))
            token_actual_rms[context_index] = torch.sqrt(torch.mean(context_actual * context_actual, dim=2))
            token_error_rms[context_index] = torch.sqrt(torch.mean(context_errors * context_errors, dim=2))
            output_nmse_context = {}
            for method_index, arm in enumerate(METHODS):
                error_energy = float(torch.sum(errors_y[arm].double() ** 2).item())
                ref_energy = float(torch.sum(refs_y[arm].double() ** 2).item())
                metric, metric_status = _ratio(error_energy, ref_energy)
                output_nmse[context_index, method_index] = math.nan if metric is None else metric
                output_nmse_status[context_index, method_index] = {
                    "measured": 0, "zero_reference_zero_error": 1,
                    "zero_reference_nonzero_error": 2,
                }[metric_status]
                output_nmse_context[arm] = {"value": metric, "status": metric_status}
                basis_equivalence_nmse[context_index, method_index] = (
                    math.nan if basis_eq[arm]["nmse_vs_raw_reference"] is None
                    else basis_eq[arm]["nmse_vs_raw_reference"])
                basis_equivalence_max_abs[context_index, method_index] = float(
                    torch.max(torch.abs(refs_y[arm] - raw_y)).item())
            output_ref_window[context_index] = context_ref[:, token_start:token_start + token_count,
                                                          channel_start:channel_start + channel_count]
            output_actual_window[context_index] = context_actual[:, token_start:token_start + token_count,
                                                                 channel_start:channel_start + channel_count]
            output_error_window[context_index] = context_errors[:, token_start:token_start + token_count,
                                                                channel_start:channel_start + channel_count]
            captured_y = saved_arrays["output"].astype("float64", copy=False)
            if (captured_y.ndim != 2 or captured_y.shape[0] != raw_y.shape[0]
                    or captured_y.shape[1] > raw_y.shape[1]):
                raise RuntimeError(f"captured BF16 output window shape differs for {name}/{context_id}")
            delta = (raw_y[:, :captured_y.shape[1]].detach().cpu().numpy().astype("float64")
                     - captured_y)
            captured_rms = float(np.sqrt(np.square(captured_y).mean()))
            saved_output_check[context_index] = float(
                np.sqrt(np.square(delta).mean()) / max(captured_rms, 1e-12))
            improvement = _relative_improvement(
                output_nmse_context[METHODS[1]]["value"],
                output_nmse_context[METHODS[2]]["value"])
            contexts.append({
                "id": context_id, "case_id": int(case), "stage": stage, "step": int(step),
                "index": context_index,
                "token_rows": [pair_start, pair_end], "token_count": int(raw_x.shape[0]),
                "output_window_token_count": int(token_count),
                "input_basis_rms_key": "input_basis_rms",
                "output_reference_rms_key": "output_reference_rms",
                "output_actual_rms_key": "output_actual_rms",
                "output_error_rms_key": "output_error_rms",
                "token_actual_rms_key": "token_actual_rms",
                "token_error_rms_key": "token_error_rms",
                "output_nmse_key": "output_nmse",
                "output_nmse_status_key": "output_nmse_status",
                "basis_equivalence_nmse_key": "basis_equivalence_nmse",
                "basis_equivalence_max_abs_key": "basis_equivalence_max_abs",
                "output_reference_window_key": "output_reference_window",
                "output_actual_window_key": "output_actual_window",
                "output_error_window_key": "output_error_window",
                "output_nmse": {
                    alias: output_nmse_context[arm]
                    for arm, alias in zip(METHODS, METHOD_ALIASES)
                },
                "basis_equivalence": basis_eq,
                "bf16_captured_output_relative_rms_difference": float(saved_output_check[context_index]),
                "output_error_hotspots": {
                    alias: _top_channels(np, output_error_rms[context_index, i])
                    for i, (arm, alias) in enumerate(zip(METHODS, METHOD_ALIASES))
                },
                "relative_improvement_vs_row": {"micro_vq": improvement["value_pct"]},
                "relative_improvement_vs_row_status": {"micro_vq": improvement["status"]},
            })

    arrays.update({
        "context_ids": np.asarray([row["id"] for row in contexts]),
        "input_basis_rms": input_rms.numpy(),
        "output_reference_rms": output_ref_rms.numpy(),
        "output_actual_rms": output_actual_rms.numpy(),
        "output_error_rms": output_error_rms.numpy(),
        "token_actual_rms": token_actual_rms.numpy(),
        "token_error_rms": token_error_rms.numpy(),
        "output_nmse": output_nmse.numpy(),
        "output_nmse_status": output_nmse_status.numpy(),
        "basis_equivalence_nmse": basis_equivalence_nmse.numpy(),
        "basis_equivalence_max_abs": basis_equivalence_max_abs.numpy(),
        "bf16_captured_output_relative_rms_difference": saved_output_check.numpy(),
        "output_reference_window": output_ref_window.numpy(),
        "output_actual_window": output_actual_window.numpy(),
        "output_error_window": output_error_window.numpy(),
        "scale_profile__spinquant_native_w4a4": scale_profiles[METHODS[0]].cpu().numpy(),
        "scale_profile__row_scale_vq_a4": scale_profiles[METHODS[1]].cpu().numpy(),
        "scale_profile__micro_scale_vq_a4": scale_profiles[METHODS[2]].cpu().numpy(),
    })
    context_docs = {row["id"]: row for row in contexts}
    weight_improvement = _relative_improvement(
        weight_dist[METHODS[1]]["nmse"], weight_dist[METHODS[2]]["nmse"])
    weight_metrics = {
        alias: {"nmse": weight_dist[arm]["nmse"]}
        for arm, alias in zip(METHODS, METHOD_ALIASES)
    }
    weight_metrics["micro_vs_row_improvement_pct"] = weight_improvement["value_pct"]
    weight_metrics["micro_vs_row_improvement_status"] = weight_improvement["status"]
    basis_labels = {
        "raw": "BF16 checkpoint weight, raw basis",
        "learned_spinquant": "FP32 R128 learned SpinQuant basis",
        "smooth05_hadamard_row": "saved Row-scale VQ Smooth+H128 basis",
        "smooth05_hadamard_micro": "saved Micro-scale VQ Smooth+H128 basis",
    }
    module_doc = {
        "status": "complete", "id": name, "module_index": module_index, "stream": module["stream"],
        "block_index": module["block_index"], "suffix": module["suffix"], "shape": list(shape),
        "npz": f"data/modules/m{module_index:04d}.npz",
        "checkpoint_weight_effective_dtype": "bfloat16 loaded from original checkpoint into BF16 Runtime",
        "transform_ids": transform_file_ids,
        "raw_bf16_is_reference_and_actual": True,
        "focus": (module["block_index"] == int(analysis_protocol["focus_block"])
                  and module["suffix"] == analysis_protocol["focus_suffix"]),
        "weight_parity": parity, "row_micro_transform_exact_equal": row_micro_same,
        "distribution": {
            "basis_edges": {"raw": raw_basis_edges, "spin": spin_basis_edges, "smooth": smooth_basis_edges},
            "counts": {**raw_histograms, **spin_histograms, **smooth_histograms},
            "basis_labels": basis_labels,
            "full_matrix_statistics": weight_dist,
        },
        "block_nmse": {arm: f"block_nmse__{arm}" for arm in METHODS},
        "block_status": {arm: f"block_status__{arm}" for arm in METHODS},
        "block_statistics": block_summaries,
        "scale_profile": {arm: f"scale_profile__{arm}" for arm in METHODS},
        "scale_profile_tile_semantics": {
            METHODS[0]: "one learned BF16 scale per output row and G128 input group",
            METHODS[1]: "one scale per output row",
            METHODS[2]: "one scale per output row and B32 input group",
        },
        "weight_metrics": weight_metrics,
        "contexts": context_docs,
        "weight_windows": {
            "reference": "weight_reference_window", "actual": "weight_actual_window",
            "error": "weight_error_window",
        },
        "weight_surface_lane_order": list(BASIS_LANES),
        "weight_window_start": [weight_r0, weight_c0],
        "weight_window_shape": [window_rows, window_cols],
        "output_window_token_start": token_start,
        "output_window_token_count": int(token_count),
        "saved_token_count": int(module_token_count),
        "weight_profile_keys": {
            "input_rms_reference": "weight_input_rms_reference",
            "input_rms_actual": "weight_input_rms_actual",
            "output_rms_reference": "weight_output_rms_reference",
            "output_rms_actual": "weight_output_rms_actual",
        },
        "basis_lane_order": ["raw", "learned_spinquant", "smooth05_hadamard_row",
                             "smooth05_hadamard_micro"],
        "basis_labels": basis_labels,
        "method_lane_order": list(METHOD_LANES),
        "error_lane_order": list(METHODS),
    }
    return module_doc, arrays


def _selfcheck(np: Any, torch: Any, codebook: Any, micro_codebook: Any,
               quantization: Any, rotation: Any, smooth_vq: Any) -> None:
    generator = torch.Generator(device="cpu").manual_seed(20261012)
    width = 128
    raw_weight = torch.randn((5, width), generator=generator).to(torch.bfloat16).float()
    raw_input = torch.randn((7, width), generator=generator).to(torch.bfloat16).float()
    matrix = torch.eye(width, dtype=torch.float32)
    signs = torch.where(torch.arange(width) % 2 == 0, 1.0, -1.0)
    spin_weight = rotation._rotate_with_matrix(raw_weight, matrix, signs)
    spin_input = rotation._rotate_with_matrix(raw_input, matrix, signs)
    spin_nmse = torch.sum((spin_input @ spin_weight.T - raw_input @ raw_weight.T) ** 2) / torch.sum(
        (raw_input @ raw_weight.T) ** 2)
    assert float(torch.sqrt(spin_nmse)) <= OUTPUT_EQUIVALENCE_RMSE_TOL

    transform = {"scale": torch.linspace(0.5, 1.5, width), "signs": signs,
                 "hadamard": True, "block": 128, "input_dim": width}
    smooth_weight = smooth_vq.transform_weight(raw_weight, transform)
    smooth_input = smooth_vq.transform_input(raw_input, transform)
    smooth_nmse = torch.sum((smooth_input @ smooth_weight.T - raw_input @ raw_weight.T) ** 2) / torch.sum(
        (raw_input @ raw_weight.T) ** 2)
    assert float(torch.sqrt(smooth_nmse)) <= OUTPUT_EQUIVALENCE_RMSE_TOL
    wrong_basis = raw_input @ smooth_weight.T
    assert float(torch.linalg.vector_norm(wrong_basis - raw_input @ raw_weight.T)) > 1e-3

    packed = torch.tensor([[0xF1, 0x87]], dtype=torch.uint8)
    decoded = quantization._unpack_nibbles(packed, 4, signed=True)
    assert decoded.tolist() == [[1, -1, 7, -8]]
    denom_nmse, denom_status = _ratio(0.0, 0.0)
    assert denom_nmse == 0.0 and denom_status == "zero_reference_zero_error"
    denom_nmse, denom_status = _ratio(1.0, 0.0)
    assert denom_nmse is None and denom_status == "zero_reference_nonzero_error"
    block_reference = torch.zeros((2, 35), dtype=torch.float32)
    block_error = torch.zeros_like(block_reference)
    block_reference[0, :32] = 2.0
    block_error[0, :32] = 1.0
    block_reference[0, 32:] = 4.0
    block_error[0, 32:] = 2.0
    block_error[1, 32:] = 1.0
    block_nmse, _, _, block_summary = _block_error(np, block_reference, block_error, 32)
    assert block_nmse.shape == (2, 2)
    assert np.allclose(block_nmse[0], [0.25, 0.25])
    assert block_nmse[1, 0] == 0.0 and np.isnan(block_nmse[1, 1])
    assert block_summary["output_rows"] == 2 and block_summary["input_blocks"] == 2
    print("WEIGHT_ANALYSIS_SELFCHECK_PASSED basis_equivalence wrong_basis signed_nibbles zero_energy",
          flush=True)


def _compact_aggregate_module(module: dict[str, Any], capture_protocol: dict[str, Any],
                              analysis_protocol: dict[str, Any]) -> dict[str, Any]:
    """Keep plot inputs and all cohort scalars without repeating per-context proof detail."""
    contexts = module.get("contexts", {})
    context_rows = list(contexts.values()) if isinstance(contexts, dict) else list(contexts)
    context_rows.sort(key=lambda row: int(row.get("index", 0)))
    requested_window = analysis_protocol["output_plot_window"]
    token_start = int(requested_window["token_start"])
    saved_token_count = module.get("saved_token_count")
    if saved_token_count is None:
        saved_token_count = next((row.get("token_count") for row in context_rows
                                  if row.get("token_count") is not None),
                                 int(capture_protocol["capture_tokens"]))
    saved_token_count = int(saved_token_count)
    output_window_token_count = int(module.get(
        "output_window_token_count",
        min(int(requested_window["token_count"]), saved_token_count - token_start)))
    def finite_max(values: list[Any]) -> float | None:
        clean = [float(value) for value in values
                 if value is not None and math.isfinite(float(value))]
        return max(clean) if clean else None

    equivalence = {}
    for alias, arm in zip(METHOD_ALIASES, METHODS):
        proof = [row.get("basis_equivalence", {}).get(arm, {}) for row in context_rows]
        equivalence[alias] = finite_max(
            [item.get("relative_rmse_vs_raw_reference") for item in proof])
    captured_differences = [
        row.get("bf16_captured_output_relative_rms_difference") for row in context_rows]
    compact_contexts = {}
    for row in context_rows:
        compact = {
            key: row[key] for key in (
                "id", "index", "case_id", "stage", "step", "token_rows",
                "token_count", "output_window_token_count") if key in row
        }
        compact["output_nmse"] = row.get("output_nmse", {})
        compact["relative_improvement_vs_row"] = row.get("relative_improvement_vs_row", {})
        compact["relative_improvement_vs_row_status"] = row.get(
            "relative_improvement_vs_row_status", {})
        compact.setdefault("token_rows", [int(capture_protocol.get("token_start", 0)),
                                           int(capture_protocol.get("token_start", 0)) + saved_token_count])
        compact.setdefault("token_count", saved_token_count)
        compact.setdefault("output_window_token_count", output_window_token_count)
        compact_contexts[str(row.get("id", len(compact_contexts)))] = compact

    compact_block_stats = {}
    for arm, stats in module.get("block_statistics", {}).items():
        compact_block_stats[arm] = {
            key: value for key, value in stats.items()
            if key not in ("tile_definition", "zero_energy_rule", "hotspots")
        }
        compact_block_stats[arm]["hotspots"] = [
            {key: hotspot[key] for key in ("output_channel", "input_block", "nmse")
             if key in hotspot}
            for hotspot in stats.get("hotspots", [])[:3]
        ]

    parity_rows = list(module.get("weight_parity", {}).values())
    parity_summary = {
        "all_actual_windows_exact": all(row.get("actual_exact") is True for row in parity_rows),
        "max_actual_abs_error": finite_max([row.get("actual_max_abs") for row in parity_rows]),
        "max_reference_abs_error": finite_max([row.get("reference_max_abs") for row in parity_rows]),
        "reference_atol": REF_WINDOW_ATOL,
    }
    distribution = module.get("distribution", {})
    compact_distribution = {key: distribution[key] for key in
                            ("basis_edges", "counts", "full_matrix_statistics") if key in distribution}
    result = {
        key: module[key] for key in (
            "status", "id", "module_index", "stream", "block_index", "suffix", "shape",
            "npz", "focus", "row_micro_transform_exact_equal",
            "block_nmse", "block_status", "scale_profile",
            "weight_metrics", "weight_windows", "weight_profile_keys",
            "output_window_token_start", "output_window_token_count", "saved_token_count",
        ) if key in module
    }
    result.update({
        "distribution": compact_distribution,
        "block_statistics": compact_block_stats,
        "contexts": compact_contexts,
        "basis_equivalence_summary": equivalence,
        "bf16_captured_output_relative_rms_difference_max": finite_max(captured_differences),
        "weight_parity": parity_summary,
        "output_window_token_start": token_start,
        "output_window_token_count": output_window_token_count,
        "saved_token_count": saved_token_count,
    })
    return result


def _aggregate(np: Any, root: Path, capture_protocol: dict[str, Any],
               analysis_protocol: dict[str, Any], modules: list[dict[str, Any]],
               completed: list[dict[str, Any]], start_epoch: float,
               status: str) -> dict[str, Any]:
    ordered = sorted(completed, key=lambda row: row["module_index"])
    contexts = [
        {"id": f"{case}|{stage}|{step}", "case_id": int(case), "stage": stage, "step": int(step)}
        for stage in ("video", "action")
        for case in analysis_protocol["cases"]
        for step in analysis_protocol["snapshot_steps"]
    ]
    grouped: dict[str, dict[str, Any]] = {}
    for module in ordered:
        group_key = f"{module['stream']}|{module['suffix']}"
        group = grouped.setdefault(group_key, {"weight": [], "output": [],
                                               "weight_output_pairs": []})
        weight_pct = module["weight_metrics"].get("micro_vs_row_improvement_pct")
        if weight_pct is not None and math.isfinite(float(weight_pct)):
            group["weight"].append(float(weight_pct))
        for context in module["contexts"].values():
            output_pct = context["relative_improvement_vs_row"]["micro_vq"]
            if output_pct is not None and math.isfinite(float(output_pct)):
                group["output"].append(float(output_pct))
            if (weight_pct is not None and output_pct is not None
                    and math.isfinite(float(weight_pct)) and math.isfinite(float(output_pct))):
                group["weight_output_pairs"].append((float(weight_pct), float(output_pct)))

    def paired_stats(values: list[float]) -> dict[str, Any]:
        return {
            "paired_win_count": sum(value > 0 for value in values),
            "total": len(values),
            "median_relative_improvement_pct": float(np.median(values)) if values else None,
            "min_relative_improvement_pct": float(min(values)) if values else None,
            "max_relative_improvement_pct": float(max(values)) if values else None,
            "positive_means_micro_lower_nmse": True,
        }

    paired_summary = {}
    for group_key, values in grouped.items():
        pairs = values["weight_output_pairs"]
        paired_summary[group_key] = {
            "weight": paired_stats(values["weight"]),
            "output": paired_stats(values["output"]),
            "tradeoffs": {
                "micro_weight_helps_output_worsens_count": sum(w > 0 and o < 0 for w, o in pairs),
                "micro_output_helps_weight_worsens_count": sum(o > 0 and w < 0 for w, o in pairs),
                "paired_weight_output_count": len(pairs),
            },
        }
    summary = {
        "schema": SCHEMA, "status": status,
        "metadata": {
            "capture_protocol": capture_protocol["protocol"],
            "analysis_protocol": analysis_protocol["protocol"],
            "source_full_job": capture_protocol["source_full_job"],
            "bf16_checkpoint": analysis_protocol["bf16_checkpoint"],
            "checkpoint_state_key": analysis_protocol["checkpoint_state_key"],
            "checkpoint_prefixes": analysis_protocol["checkpoint_prefixes"],
            "method_lane_order": list(METHOD_LANES),
            "method_alias_order": ["bf16", *METHOD_ALIASES],
            "error_lane_order": list(METHODS),
            "basis_lane_order": list(BASIS_LANES),
            "basis_labels": {
                "raw": "BF16 checkpoint weight, raw basis",
                "learned_spinquant": "FP32 R128 learned SpinQuant basis",
                "smooth05_hadamard_row": "saved Row-scale VQ Smooth+H128 basis",
                "smooth05_hadamard_micro": "saved Micro-scale VQ Smooth+H128 basis",
            },
            "weight_window": {
                "token_axis": "output_channel", "channel_axis": "input_channel",
                "start": [analysis_protocol["weight_plot_window"]["output_start"],
                          analysis_protocol["weight_plot_window"]["input_start"]], "requested_shape": [
                    analysis_protocol["weight_plot_window"]["output_count"],
                    analysis_protocol["weight_plot_window"]["input_count"]],
                "lane_order": list(BASIS_LANES), "value": "signed weights; renderer may display absolute magnitude",
            },
            "windows": {
                "output": {
                    "token_start": analysis_protocol["output_plot_window"]["token_start"],
                    "token_count": analysis_protocol["output_plot_window"]["token_count"],
                    "channel_start": analysis_protocol["output_plot_window"]["output_start"],
                    "channel_count": analysis_protocol["output_plot_window"]["output_count"],
                },
            },
            "output_error_definition": analysis_protocol["output_error_definition"],
            "offline_scope": "FP32 basis transforms and bias-free Linear products from the existing BF16 paired input; no A4, model construction, whole-model forward, training, or refit",
            "new_whole_model_forwards": 0, "model_constructed": False,
            "training": False, "refit": False, "activation_quantization": False,
            "native_kernel_executed": False, "fixed_input_context_count": int(analysis_protocol["expected_primary_inputs"]),
            "output_comparison_count": int(analysis_protocol["expected_output_comparisons"]),
            "weight_comparison_count": int(analysis_protocol["expected_weight_comparisons"]),
            "equivalence_tolerance_relative_rmse": OUTPUT_EQUIVALENCE_RMSE_TOL,
            "reference_window_tolerance_max_abs": REF_WINDOW_ATOL,
            "zero_reference_energy_rule": "NMSE is 0 for zero/zero and null with an explicit status for nonzero error over zero reference energy.",
            "relative_improvement_positive_means": "Micro VQ has lower NMSE than Row VQ; percent reduction relative to Row VQ.",
            "expected_module_count": int(analysis_protocol["expected_modules"]),
            "completed_module_count": len(ordered),
            "soft_deadline_seconds": int(analysis_protocol["soft_deadline_seconds"]),
            "elapsed_seconds": round(time.time() - start_epoch, 3),
        },
        "methods": list(METHODS),
        "method_aliases": list(METHOD_ALIASES),
        "contexts": contexts,
        "modules": [_compact_aggregate_module(row, capture_protocol, analysis_protocol)
                    for row in ordered],
        "paired_micro_vs_row_by_stream_suffix": paired_summary,
        "gates": {
            "module_count": len(ordered) == int(analysis_protocol["expected_modules"]),
            "all_module_weight_window_parity": all(
                all(item["actual_exact"] and item["reference_max_abs"] <= REF_WINDOW_ATOL
                    for item in row["weight_parity"].values()) for row in ordered),
            "all_actual_weight_windows_exact": all(
                all(item["actual_exact"] and item["actual_max_abs"] == 0
                    for item in row["weight_parity"].values()) for row in ordered),
            "all_reference_weight_windows_within_tolerance": all(
                all(item["reference_max_abs"] <= REF_WINDOW_ATOL
                    for item in row["weight_parity"].values()) for row in ordered),
            "all_row_micro_transform_exact_equal": all(
                row["row_micro_transform_exact_equal"] for row in ordered),
            "all_basis_equivalence_gates": all(
                float(context["basis_equivalence"][arm]["relative_rmse_vs_raw_reference"])
                <= OUTPUT_EQUIVALENCE_RMSE_TOL
                for row in ordered for context in row["contexts"].values() for arm in METHODS),
            "primary_input_count": sum(len(row["contexts"]) for row in ordered),
            "output_comparison_count": sum(len(row["contexts"]) * len(METHODS) for row in ordered),
            "weight_comparison_count": sum(len(METHODS) for _ in ordered),
            "focus_module_count": sum(bool(row["focus"]) for row in ordered),
            "main_output_map_count": sum(any(
                context["case_id"] == int(analysis_protocol["focus_case"])
                and context["step"] == int(analysis_protocol["focus_step"])
                for context in row["contexts"].values()) for row in ordered),
            "expected_main_output_map_count": int(analysis_protocol["expected_modules"]),
            "weight_channel_profile_module_count": sum(
                "weight_profile_keys" in row for row in ordered),
        },
    }
    path = root / "weight_analysis_data.json"
    payload = json.dumps(summary, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")) + "\n"
    size = len(payload.encode("utf-8"))
    print(f"WEIGHT_AGGREGATE_SIZE modules={len(ordered)} status={status} json_bytes={size}",
          flush=True)
    if size > 262144:
        raise RuntimeError(f"aggregate weight analysis JSON exceeds 256 KiB: {size} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return summary


def _reuse_complete_analysis(analysis_root: Path, capture_protocol: dict[str, Any],
                            analysis_protocol: dict[str, Any],
                            modules: list[dict[str, Any]]) -> bool:
    aggregate_path = analysis_root / "weight_analysis_data.json"
    progress_path = analysis_root / "progress.json"
    if not aggregate_path.is_file() or not progress_path.is_file():
        return False
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected_count = int(analysis_protocol["expected_modules"])
    expected_gates = {
        "module_count": True, "all_module_weight_window_parity": True,
        "all_actual_weight_windows_exact": True,
        "all_reference_weight_windows_within_tolerance": True,
        "all_row_micro_transform_exact_equal": True,
        "all_basis_equivalence_gates": True,
        "primary_input_count": int(analysis_protocol["expected_primary_inputs"]),
        "output_comparison_count": int(analysis_protocol["expected_output_comparisons"]),
        "weight_comparison_count": int(analysis_protocol["expected_weight_comparisons"]),
        "focus_module_count": 2, "main_output_map_count": expected_count,
        "weight_channel_profile_module_count": expected_count,
    }
    metadata = aggregate.get("metadata", {})
    if (aggregate.get("schema") != SCHEMA or aggregate.get("status") != "complete"
            or metadata.get("capture_protocol") != capture_protocol.get("protocol")
            or metadata.get("analysis_protocol") != analysis_protocol.get("protocol")
            or metadata.get("bf16_checkpoint") != analysis_protocol.get("bf16_checkpoint")
            or metadata.get("completed_module_count") != expected_count
            or progress.get("status") != "complete"
            or progress.get("completed_modules") != expected_count
            or progress.get("primary_inputs") != int(analysis_protocol["expected_primary_inputs"])
            or progress.get("output_comparisons") != int(analysis_protocol["expected_output_comparisons"])):
        return False
    gates = aggregate.get("gates", {})
    if any(gates.get(key) != value for key, value in expected_gates.items()):
        return False
    rows = aggregate.get("modules", [])
    if len(rows) != expected_count or [row.get("id") for row in rows] != [row["module"] for row in modules]:
        return False
    data_dir, checkpoint_dir = analysis_root / "data/modules", analysis_root / "checkpoints"
    for module in modules:
        index = int(module["module_index"])
        marker_path = checkpoint_dir / f"module_m{index:04d}.json"
        module_json_path = data_dir / f"m{index:04d}.json"
        module_npz_path = data_dir / f"m{index:04d}.npz"
        if not marker_path.is_file() or not module_json_path.is_file() or not module_npz_path.is_file():
            return False
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("complete") is not True or marker.get("module") != module["module"]
                or marker.get("analysis_protocol") != analysis_protocol
                or marker.get("capture_protocol") != capture_protocol):
            return False
    return aggregate_path.stat().st_size <= 262144


def _validate_complete_summary(summary: dict[str, Any], analysis_protocol: dict[str, Any]) -> None:
    gates = summary["gates"]
    required_true = (
        "module_count", "all_module_weight_window_parity", "all_actual_weight_windows_exact",
        "all_reference_weight_windows_within_tolerance", "all_row_micro_transform_exact_equal",
        "all_basis_equivalence_gates",
    )
    if not all(gates.get(key) is True for key in required_true):
        raise RuntimeError(f"weight analysis gates failed: {gates}")
    expected_modules = int(analysis_protocol["expected_modules"])
    expected_primary = int(analysis_protocol["expected_primary_inputs"])
    expected_outputs = int(analysis_protocol["expected_output_comparisons"])
    expected_weights = int(analysis_protocol["expected_weight_comparisons"])
    if (gates.get("primary_input_count") != expected_primary
            or gates.get("output_comparison_count") != expected_outputs
            or gates.get("weight_comparison_count") != expected_weights
            or gates.get("focus_module_count") != 2
            or gates.get("main_output_map_count") != expected_modules
            or gates.get("weight_channel_profile_module_count") != expected_modules):
        raise RuntimeError(f"weight analysis coverage gates failed: {gates}")


def _aggregate_existing(args: argparse.Namespace, capture_protocol: dict[str, Any],
                        analysis_protocol: dict[str, Any]) -> None:
    """Rebuild only the compact summary from completed per-module JSON artifacts."""
    capture_root = args.out.resolve()
    analysis_root = capture_root / analysis_protocol.get("result_subdirectory", "weight_analysis")
    manifest_path = capture_root / "raw/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"existing capture manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modules = _module_rows(capture_protocol, analysis_protocol, manifest)
    if _reuse_complete_analysis(analysis_root, capture_protocol, analysis_protocol, modules):
        print("WEIGHT_ANALYSIS_COMPLETE_REUSED aggregate-only; preserve plot-cache identities", flush=True)
        return
    data_dir, checkpoint_dir = analysis_root / "data/modules", analysis_root / "checkpoints"
    completed = []
    for module in modules:
        index = int(module["module_index"])
        marker_path = checkpoint_dir / f"module_m{index:04d}.json"
        module_json_path = data_dir / f"m{index:04d}.json"
        module_npz_path = data_dir / f"m{index:04d}.npz"
        if not all(path.is_file() for path in (marker_path, module_json_path, module_npz_path)):
            raise RuntimeError(f"aggregate-only requires all 16 existing module artifacts; missing index {index}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        module_doc = json.loads(module_json_path.read_text(encoding="utf-8"))
        if (marker.get("complete") is not True or marker.get("module") != module["module"]
                or marker.get("analysis_protocol") != analysis_protocol
                or marker.get("capture_protocol") != capture_protocol
                or module_doc.get("status") != "complete"
                or module_doc.get("id") != module["module"]
                or int(module_doc.get("module_index", -1)) != index):
            raise RuntimeError(f"completed module identity/protocol mismatch at index {index}")
        completed.append(module_doc)

    start_epoch = time.time()
    aggregate_path = analysis_root / "weight_analysis_data.json"
    if aggregate_path.is_file():
        try:
            prior = json.loads(aggregate_path.read_text(encoding="utf-8"))
            prior_meta = prior.get("metadata", {})
            elapsed = float(prior_meta.get("elapsed_seconds"))
            if (prior_meta.get("capture_protocol") == capture_protocol["protocol"]
                    and prior_meta.get("analysis_protocol") == analysis_protocol["protocol"]
                    and math.isfinite(elapsed) and elapsed >= 0):
                start_epoch -= elapsed
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    summary = _aggregate(statistics, analysis_root, capture_protocol, analysis_protocol,
                         modules, completed, start_epoch, "complete")
    _validate_complete_summary(summary, analysis_protocol)
    _save_json(analysis_root / "progress.json", {
        "status": "complete", "completed_modules": len(completed),
        "expected_modules": len(modules),
        "primary_inputs": int(analysis_protocol["expected_primary_inputs"]),
        "output_comparisons": int(analysis_protocol["expected_output_comparisons"]),
    })
    print(f"WEIGHT_ANALYSIS_AGGREGATE_ONLY_COMPLETE modules={len(completed)} "
          f"json_bytes={(analysis_root / 'weight_analysis_data.json').stat().st_size}", flush=True)


def _run(np: Any, torch: Any, codebook: Any, micro_codebook: Any,
         quantization: Any, rotation: Any, smooth_vq: Any,
         args: argparse.Namespace, capture_protocol: dict[str, Any],
         analysis_protocol: dict[str, Any]) -> None:
    capture_root = args.out.resolve()
    analysis_root = capture_root / analysis_protocol.get("result_subdirectory", "weight_analysis")
    manifest_path = capture_root / "raw/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"existing capture manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modules = _module_rows(capture_protocol, analysis_protocol, manifest)
    records = manifest["records"]
    selected_indices = {row["module_index"] for row in modules}
    weight_rows = [row for row in records if row.get("kind") == "weight"
                   and int(row.get("module_index", -1)) in selected_indices]
    activation_rows = [row for row in records if row.get("kind") == "activation"
                       and row.get("arm") == "bf16" and int(row.get("module_index", -1)) in selected_indices]
    paired_rows = [row for row in records if row.get("kind") == "paired_local"
                   and int(row.get("module_index", -1)) in selected_indices]
    record_maps = {
        "weights": _record_index(weight_rows, kind="weight", arms=("bf16", *METHODS)),
        "activations": _record_index(activation_rows, kind="activation"),
        "paired": _record_index(paired_rows, kind="paired_local"),
    }
    for module in modules:
        for arm in ("bf16", *METHODS):
            if (module["module_index"], arm) not in record_maps["weights"]:
                raise RuntimeError(f"weight record coverage missing for {module['module']}/{arm}")
        stage = "video" if module["stream"] == "video_expert" else "action"
        for case in analysis_protocol["cases"]:
            for step in analysis_protocol["snapshot_steps"]:
                key = (module["module_index"], int(case), stage, int(step))
                _primary(record_maps["activations"].get(key, []), f"{module['module']}/{key}/activation")
                _primary(record_maps["paired"].get(key, []), f"{module['module']}/{key}/paired input")
    expected_primary = len(modules) * len(analysis_protocol["cases"]) * len(analysis_protocol["snapshot_steps"])
    if expected_primary != int(analysis_protocol["expected_primary_inputs"]):
        raise RuntimeError(f"expected primary input count mismatch: {expected_primary}")
    if expected_primary * len(METHODS) != int(analysis_protocol["expected_output_comparisons"]):
        raise RuntimeError("expected output comparison count does not match the frozen protocol")
    if len(modules) * len(METHODS) != int(analysis_protocol["expected_weight_comparisons"]):
        raise RuntimeError("expected full-matrix weight comparison count does not match the frozen protocol")

    analysis_root.mkdir(parents=True, exist_ok=True)
    data_dir = analysis_root / "data/modules"
    checkpoint_dir = analysis_root / "checkpoints"
    plots_dir = analysis_root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if _reuse_complete_analysis(analysis_root, capture_protocol, analysis_protocol, modules):
        print(f"WEIGHT_ANALYSIS_REUSE_COMPLETE modules={len(modules)}; aggregate left unchanged",
              flush=True)
        return
    start_epoch = float(os.environ.get("FW_JOB_START_EPOCH", time.time()))
    deadline = start_epoch + int(analysis_protocol["soft_deadline_seconds"])
    checkpoint_path = Path(analysis_protocol["bf16_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state_key = analysis_protocol["checkpoint_state_key"]
    if state_key not in checkpoint or not isinstance(checkpoint[state_key], dict):
        raise RuntimeError(f"BF16 checkpoint lacks required state dictionary {state_key}")
    checkpoint_state = checkpoint[state_key]
    spin_manifest = json.loads((Path(capture_protocol["spinquant_bank"]) / "manifest.json")
                               .read_text(encoding="utf-8"))
    rotation_payload = torch.load(
        Path(capture_protocol["spinquant_bank"]) / spin_manifest["rotation_file"],
        map_location="cpu", weights_only=True)
    completed = []
    for module in modules:
        marker_path = checkpoint_dir / f"module_m{module['module_index']:04d}.json"
        module_json_path = data_dir / f"m{module['module_index']:04d}.json"
        module_npz_path = data_dir / f"m{module['module_index']:04d}.npz"
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("complete") is not True or marker.get("module") != module["module"]:
                raise RuntimeError(f"invalid per-module checkpoint: {marker_path}")
            if marker.get("analysis_protocol") != analysis_protocol or marker.get("capture_protocol") != capture_protocol:
                raise RuntimeError(f"frozen protocol changed since module checkpoint: {module['module']}")
            if not module_json_path.is_file() or not module_npz_path.is_file():
                raise RuntimeError(f"completed module checkpoint lacks data files: {module['module']}")
            completed.append(json.loads(module_json_path.read_text(encoding="utf-8")))
            continue
        if deadline - time.time() < 240:
            _aggregate(np, analysis_root, capture_protocol, analysis_protocol, modules,
                       completed, start_epoch, "checkpointed")
            _save_json(analysis_root / "progress.json", {
                "status": "checkpointed", "completed_modules": len(completed),
                "expected_modules": len(modules), "next_module": module["module"],
            })
            return
        module_doc, arrays = _module_compute(
            np, torch, codebook, micro_codebook, quantization, rotation, smooth_vq,
            capture_protocol, analysis_protocol, capture_root, manifest, module,
            checkpoint_state, record_maps, rotation_payload, spin_manifest)
        _save_npz(np, module_npz_path, arrays)
        _save_json(module_json_path, module_doc)
        _save_json(marker_path, {
            "schema": SCHEMA, "complete": True, "module": module["module"],
            "module_index": module["module_index"],
            "analysis_protocol": analysis_protocol, "capture_protocol": capture_protocol,
        })
        completed.append(module_doc)
        _aggregate(np, analysis_root, capture_protocol, analysis_protocol, modules,
                   completed, start_epoch, "checkpointed")
        _save_json(analysis_root / "progress.json", {
            "status": "checkpointed", "completed_modules": len(completed),
            "expected_modules": len(modules), "next_module": None,
        })
        print(f"WEIGHT_MODULE_COMPLETE {len(completed)}/{len(modules)} {module['module']}",
              flush=True)
    summary = _aggregate(np, analysis_root, capture_protocol, analysis_protocol, modules,
                         completed, start_epoch, "complete")
    if not all(summary["gates"][key] for key in (
            "module_count", "all_module_weight_window_parity", "all_actual_weight_windows_exact",
            "all_reference_weight_windows_within_tolerance", "all_row_micro_transform_exact_equal",
            "all_basis_equivalence_gates")):
        raise RuntimeError(f"weight analysis gates failed: {summary['gates']}")
    if (summary["gates"]["primary_input_count"] != int(analysis_protocol["expected_primary_inputs"])
            or summary["gates"]["output_comparison_count"] != int(analysis_protocol["expected_output_comparisons"])
            or summary["gates"]["weight_comparison_count"] != int(analysis_protocol["expected_weight_comparisons"])
            or summary["gates"]["focus_module_count"] != 2
            or summary["gates"]["main_output_map_count"] != int(analysis_protocol["expected_modules"])
            or summary["gates"]["weight_channel_profile_module_count"] != int(analysis_protocol["expected_modules"])):
        raise RuntimeError(f"weight analysis coverage gates failed: {summary['gates']}")
    _save_json(analysis_root / "progress.json", {
        "status": "complete", "completed_modules": len(completed),
        "expected_modules": len(modules), "primary_inputs": expected_primary,
        "output_comparisons": expected_primary * len(METHODS),
    })
    print(f"WEIGHT_ANALYSIS_COMPLETE modules={len(completed)} inputs={expected_primary} "
          f"outputs={expected_primary * len(METHODS)} json_bytes="
          f"{(analysis_root / 'weight_analysis_data.json').stat().st_size}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True,
                        help="the original capture protocol.json")
    parser.add_argument("--analysis-protocol", type=Path, required=True,
                        help="the frozen weight_protocol.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="the existing captured results/full directory")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run a small numerical mechanism check and exit")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="rebuild the summary from all completed module JSONs without loading tensors")
    args = parser.parse_args()
    if args.selfcheck and args.aggregate_only:
        parser.error("--selfcheck and --aggregate-only are mutually exclusive")
    guard()
    capture_protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    analysis_protocol = json.loads(args.analysis_protocol.read_text(encoding="utf-8"))
    if analysis_protocol.get("protocol") != "fastwam-weight-analysis-v1":
        raise ValueError("unexpected weight analysis protocol")
    if analysis_protocol.get("capture_protocol") != capture_protocol.get("protocol"):
        raise ValueError("weight analysis protocol does not match the capture protocol")
    if analysis_protocol.get("new_whole_model_forwards") != 0 or analysis_protocol.get("training") is not False:
        raise ValueError("weight analysis protocol must freeze zero whole-model forwards and no training")
    if args.aggregate_only:
        _aggregate_existing(args, capture_protocol, analysis_protocol)
        return
    np, torch, codebook, micro_codebook, quantization, rotation, smooth_vq = _load_sources(
        capture_protocol, int(analysis_protocol.get("cpu_threads", 16)))
    if args.selfcheck:
        _selfcheck(np, torch, codebook, micro_codebook, quantization, rotation, smooth_vq)
        return
    _run(np, torch, codebook, micro_codebook, quantization, rotation, smooth_vq,
         args, capture_protocol, analysis_protocol)


if __name__ == "__main__":
    main()

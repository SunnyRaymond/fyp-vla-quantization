"""Collect bounded Fast-WAM weight/activation diagnostics inside PBS only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Any


PROTOCOL_NAME = "fastwam-smooth-vq-visualization-v1"
SOURCE_PROTOCOL_NAME = "fastwam-smooth-vq-phases12-v1"
EXPECTED_CASES = [1, 13, 21]
EXPECTED_SEEDS = [0]
WEIGHT_CONFIGS = ["smooth05_hadamard", "smooth1_hadamard"]
QUANTIZERS = ["scalar", "vq"]
EXPECTED_MODULE_COUNT = 614
SNAPSHOT_STEPS = [0, 4, 9]
ACTIVATION_BITS = 4
ACTION_DRIFT_LIMIT = 0.01
MAX_POOL_SIDE = 64


class CollectorStageTagger:
    """Small stage interface for run_diagnostic's real scheduler wrappers."""

    def __init__(self) -> None:
        self.stage = "conditioning"
        self.step = -1

    def set_stage(self, stage: str, step: int) -> None:
        if stage in ("conditioning", "video_conditioning_prefill"):
            if step != -1:
                raise ValueError(f"conditioning stage must use step -1, got {step}")
        elif stage in ("video", "action"):
            if type(step) is not int or step not in range(10):
                raise ValueError(f"{stage} denoising step must be in 0..9, got {step}")
        else:
            raise ValueError(f"unknown scheduler stage: {stage!r}")
        self.stage, self.step = stage, step


def _json_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonl_append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _resolve_path(value: str | Path, bases: list[Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = [(base / path).resolve() for base in bases]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _validate_visualization_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"unexpected visualization protocol: {protocol.get('protocol')!r}")
    if protocol.get("source_protocol") != SOURCE_PROTOCOL_NAME:
        raise ValueError(f"unexpected source protocol: {protocol.get('source_protocol')!r}")
    fixed = {
        "cases": EXPECTED_CASES,
        "seed_indices": EXPECTED_SEEDS,
        "activation_configs": ["identity", "hadamard", "smooth05_hadamard", "smooth1_hadamard"],
        "weight_configs": WEIGHT_CONFIGS,
        "weight_quantizers": QUANTIZERS,
        "local_output_rows": 16,
        "snapshot_steps": SNAPSHOT_STEPS,
        "selected_module_count": 6,
        "restoration_experiments": False,
        "fit_codebooks": False,
        "evaluation_episodes": 0,
        "native_latency_claim": False,
    }
    for key, expected in fixed.items():
        if protocol.get(key) != expected:
            raise ValueError(f"visualization protocol {key} must be {expected!r}, got {protocol.get(key)!r}")
    if not isinstance(protocol.get("source_full_job"), str) or not protocol["source_full_job"].strip():
        raise ValueError("visualization protocol must pin source_full_job")


def _verify_full_run(full_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    expected_job = protocol["source_full_job"]
    evidence: dict[str, Any] = {}
    for filename in ("summary.json", "runtime.json"):
        path = full_root / filename
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            actual_job = value.get("pbs_jobid")
            if actual_job != expected_job:
                raise RuntimeError(f"{filename} PBS job {actual_job!r} != protocol source_full_job {expected_job!r}")
            evidence[filename] = {"pbs_jobid": actual_job, "status": value.get("status")}
    if not evidence:
        raise FileNotFoundError(f"full result metadata missing under {full_root}")
    summary_path = full_root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise RuntimeError(f"source full run is not complete: {summary.get('status')!r}")
        if summary.get("protocol") != SOURCE_PROTOCOL_NAME:
            raise RuntimeError(f"source full run protocol mismatch: {summary.get('protocol')!r}")
    for config in WEIGHT_CONFIGS:
        for kind in QUANTIZERS:
            directory = full_root / "banks" / f"{config}_{kind}"
            if not directory.is_dir():
                raise FileNotFoundError(f"saved bank directory missing: {directory}")
    if not (full_root / "actions.jsonl").is_file():
        raise FileNotFoundError(f"source full actions are missing: {full_root / 'actions.jsonl'}")
    return evidence


def _same_transform(torch: Any, left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("scale", "signs"):
        if key not in left or key not in right or not torch.equal(left[key].cpu(), right[key].cpu()):
            return False
    return all(left.get(key) == right.get(key) for key in ("hadamard", "block", "input_dim"))


def _to_device_transform(transform: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device=device, dtype=torch.float32) if torch.is_tensor(value) else value
        for key, value in transform.items()
    }


def _load_saved_banks(torch: Any, full_root: Path, modules: dict[str, Any], progress: Any) -> tuple[dict, dict]:
    """Keep encoded tensors on CPU; never fit or retain decoded full-model banks."""
    encoded_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    transforms_cpu: dict[str, dict[str, Any]] = {config: {} for config in WEIGHT_CONFIGS}
    module_items = list(modules.items())
    for module_index, (name, _module) in enumerate(module_items):
        for config in WEIGHT_CONFIGS:
            transform = None
            for kind in QUANTIZERS:
                path = full_root / "banks" / f"{config}_{kind}" / f"{module_index:04d}.pt"
                bank = torch.load(path, map_location="cpu", weights_only=True)
                if bank.get("module") != name:
                    raise RuntimeError(f"stored module identity differs at {path}: {bank.get('module')!r} != {name!r}")
                current_transform = bank.get("transform")
                encoded = bank.get("encoded")
                if not isinstance(current_transform, dict) or not isinstance(encoded, dict):
                    raise RuntimeError(f"malformed saved bank: {path}")
                if encoded.get("kind") != kind:
                    raise RuntimeError(f"saved bank kind mismatch at {path}: {encoded.get('kind')!r}")
                if transform is None:
                    transform = current_transform
                elif not _same_transform(torch, transform, current_transform):
                    raise RuntimeError(f"scalar/VQ transforms differ for {config}/{name}")
                encoded_cache[(config, name, kind)] = encoded
                del bank
            transforms_cpu[config][name] = transform
        if (module_index + 1) % 25 == 0 or module_index + 1 == len(module_items):
            progress("bank_cache", modules_completed=module_index + 1, module_count=len(module_items), current_module=name)
    return encoded_cache, transforms_cpu


def _pool_absmax(torch: Any, matrix: Any) -> tuple[Any, dict[str, Any]]:
    import torch.nn.functional as functional

    values = matrix.detach().float().abs()
    rows, columns = map(int, values.shape)
    block_rows = max(1, math.ceil(rows / MAX_POOL_SIDE))
    block_columns = max(1, math.ceil(columns / MAX_POOL_SIDE))
    padded_rows = math.ceil(rows / block_rows) * block_rows
    padded_columns = math.ceil(columns / block_columns) * block_columns
    padded = functional.pad(values[None, None], (0, padded_columns - columns, 0, padded_rows - rows), value=0.0)
    pooled = functional.max_pool2d(padded, kernel_size=(block_rows, block_columns),
                                   stride=(block_rows, block_columns))[0, 0]
    out_rows, out_columns = map(int, pooled.shape)
    row_starts = [i * block_rows for i in range(out_rows)]
    row_ends = [min((i + 1) * block_rows, rows) for i in range(out_rows)]
    column_starts = [i * block_columns for i in range(out_columns)]
    column_ends = [min((i + 1) * block_columns, columns) for i in range(out_columns)]
    boundaries = {
        "row_starts": row_starts,
        "row_ends": row_ends,
        "column_starts": column_starts,
        "column_ends": column_ends,
    }
    return pooled.detach().cpu().numpy(), boundaries


def _numpy(tensor: Any) -> Any:
    return tensor.detach().cpu().numpy()


def _top8(torch: Any, error: Any) -> tuple[list[dict[str, Any]], list[int], float]:
    flat = error.detach().float().abs().reshape(-1)
    count = min(8, flat.numel())
    values, indices = torch.topk(flat, count, largest=True, sorted=True)
    in_features = int(error.shape[1])
    coords = [
        {"row": int(index.item() // in_features), "column": int(index.item() % in_features), "abs_error": float(value.item())}
        for value, index in zip(values, indices)
    ]
    peak_index = int(indices[0].item())
    return coords, [peak_index // in_features, peak_index % in_features], float(values[0].item())


def _weight_error_stats(torch: Any, base: Any, decoded: Any) -> dict[str, Any]:
    difference = decoded.detach().float() - base
    rmse = difference.square().mean().sqrt()
    weight_rms = base.square().mean().sqrt()
    row_rms = base.square().mean(dim=1).sqrt()
    row_error_rms = difference.square().mean(dim=1).sqrt()
    top, position, peak = _top8(torch, difference)
    return {
        "rmse": float(rmse.item()),
        "rrmse": float((rmse / weight_rms.clamp_min(torch.finfo(torch.float32).eps)).item()),
        "peak": peak,
        "position": position,
        "top8": top,
        "row_rrmse": row_error_rms / row_rms.clamp_min(torch.finfo(torch.float32).eps),
        "output_channel_energy": difference.square().sum(dim=1),
        "input_channel_energy": difference.square().sum(dim=0),
        "pooled_absmax": _pool_absmax(torch, difference)[0],
    }


def _decode(torch: Any, core: Any, encoded: dict[str, Any], device: Any) -> Any:
    kind = encoded["kind"]
    if kind == "scalar":
        return core.decode_scalar_quantized(
            encoded["packed"].to(device=device), encoded["scales"].to(device=device),
            tuple(encoded["shape"]), int(encoded["group"]),
        )
    if kind == "vq":
        return core.decode_vq_quantized(
            encoded["indices"].to(device=device), encoded["codebooks"].to(device=device), tuple(encoded["shape"]),
        )
    raise ValueError(f"unsupported saved bank kind: {kind!r}")


def _scan_weights(
    torch: Any, np: Any, core: Any, out: Path, modules: dict[str, Any], streams: dict[str, str],
    transforms: dict[str, dict[str, dict[str, Any]]], encoded_cache: dict, progress: Any,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    module_items = list(modules.items())
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    weight_records: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for module_index, (name, module) in enumerate(module_items):
        original = module.weight.detach().float()
        for config in WEIGHT_CONFIGS:
            transform = transforms[config][name]
            base = core.transform_weight(original, transform)
            weight_rms = float(base.square().mean().sqrt().item())
            scalar_decoded = _decode(torch, core, encoded_cache[(config, name, "scalar")], base.device)
            scalar_stats = _weight_error_stats(torch, base, scalar_decoded)
            del scalar_decoded
            vq_decoded = _decode(torch, core, encoded_cache[(config, name, "vq")], base.device)
            vq_stats = _weight_error_stats(torch, base, vq_decoded)
            del vq_decoded
            ratio = vq_stats["rrmse"] / scalar_stats["rrmse"] if scalar_stats["rrmse"] > 0 else None
            record = {
                "config": config, "module": name, "module_index": module_index,
                "stream": streams[name], "shape": list(base.shape), "weight_rms": weight_rms,
                "scalar_rmse": scalar_stats["rmse"], "vq_rmse": vq_stats["rmse"],
                "scalar_rrmse": scalar_stats["rrmse"], "vq_rrmse": vq_stats["rrmse"],
                "ratio": ratio, "scalar_peak_error": scalar_stats["peak"],
                "vq_peak_error": vq_stats["peak"], "scalar_peak_position": scalar_stats["position"],
                "vq_peak_position": vq_stats["position"],
                "top8_absolute_error_coordinates": {"scalar": scalar_stats["top8"], "vq": vq_stats["top8"]},
                "weight_coordinate_space": "saved transformed coordinates",
            }
            snapshot_path = out / "snapshots" / f"weight_{module_index:04d}_{config}.npz"
            record["snapshot_file"] = str(snapshot_path.relative_to(out))
            weight_records.append(record)
            lookup[(config, name)] = record
            weight_absmax, boundaries = _pool_absmax(torch, base)
            np.savez_compressed(
                snapshot_path,
                weight_absmax=weight_absmax,
                scalar_error_absmax=scalar_stats["pooled_absmax"],
                vq_error_absmax=vq_stats["pooled_absmax"],
                row_rms=_numpy(base.square().mean(dim=1).sqrt()),
                row_scalar_rrmse=_numpy(scalar_stats["row_rrmse"]),
                row_vq_rrmse=_numpy(vq_stats["row_rrmse"]),
                scalar_column_energy=_numpy(scalar_stats["input_channel_energy"]),
                vq_column_energy=_numpy(vq_stats["input_channel_energy"]),
                scalar_output_channel_error_energy=_numpy(scalar_stats["output_channel_energy"]),
                vq_output_channel_error_energy=_numpy(vq_stats["output_channel_energy"]),
                row_starts=np.asarray(boundaries["row_starts"], dtype=np.int32),
                row_ends=np.asarray(boundaries["row_ends"], dtype=np.int32),
                column_starts=np.asarray(boundaries["column_starts"], dtype=np.int32),
                column_ends=np.asarray(boundaries["column_ends"], dtype=np.int32),
                original_shape=np.asarray(base.shape, dtype=np.int64),
                coordinate_space=np.asarray("saved transformed coordinates"),
                pooling=np.asarray("deterministic adaptive max pooling over the complete matrix; no random sampling"),
            )
            del base
        del original
        if (module_index + 1) % 25 == 0 or module_index + 1 == len(module_items):
            progress("weight_scan", modules_completed=module_index + 1, module_count=len(module_items), current_module=name)
    with (out / "weights.jsonl").open("w", encoding="utf-8") as stream:
        for record in weight_records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return weight_records, lookup


def _select_modules(weight_records: list[dict[str, Any]], module_count: int) -> list[dict[str, Any]]:
    alpha05 = [row for row in weight_records if row["config"] == "smooth05_hadamard"]
    if len(alpha05) != module_count:
        raise RuntimeError(f"expected {module_count} alpha0.5 weight rows, got {len(alpha05)}")
    selected: dict[int, dict[str, Any]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        index = int(row["module_index"])
        if index in selected:
            selected[index]["reason"] += f"; also {reason}"
        else:
            selected[index] = {"module": row["module"], "module_index": index, "stream": row["stream"], "reason": reason}

    actions = [row for row in alpha05 if row["stream"] == "action"]
    videos = [row for row in alpha05 if row["stream"] == "video"]
    proprios = [row for row in alpha05 if row["stream"] == "proprio"]
    if not actions or not videos or not proprios:
        raise RuntimeError("expected Action, Video, and proprio modules for selected-layer rules")
    add(max(actions, key=lambda row: (row["ratio"] if row["ratio"] is not None else -1, -row["module_index"])), "Action largest VQ/Scalar RMSE ratio")
    add(max(actions, key=lambda row: (row["vq_rrmse"], -row["module_index"])), "Action largest VQ normalized error")
    action_ratios = sorted(actions, key=lambda row: (row["ratio"] if row["ratio"] is not None else float("inf"), row["module_index"]))
    ratios = [float(row["ratio"]) for row in action_ratios if row["ratio"] is not None]
    median_ratio = statistics.median(ratios)
    add(min(actions, key=lambda row: (abs((row["ratio"] or 0.0) - (median_ratio or 0.0)), row["module_index"])), "Action median-ratio representative")
    add(max(videos, key=lambda row: (row["ratio"] if row["ratio"] is not None else -1, -row["module_index"])), "Video largest VQ/Scalar RMSE ratio")
    add(min(videos, key=lambda row: (row["ratio"] if row["ratio"] is not None else float("inf"), row["module_index"])), "Video smallest VQ/Scalar RMSE ratio control")
    add(min(proprios, key=lambda row: (row["module"].count("."), row["module_index"])), "shallowest proprio encoder module")
    for row in sorted(alpha05, key=lambda item: (-(item["ratio"] if item["ratio"] is not None else -1), -item["vq_rrmse"], item["module_index"])):
        if len(selected) >= 6:
            break
        add(row, "deterministic fill by VQ/Scalar ratio")
    if len(selected) != 6:
        raise RuntimeError(f"selected module count is {len(selected)}, expected 6")
    return [selected[index] for index in sorted(selected)]


def _uniform_rows(torch: Any, rows: int, requested: int) -> Any:
    count = min(rows, requested)
    if count == rows:
        return torch.arange(rows, device="cuda", dtype=torch.long)
    return torch.linspace(0, rows - 1, count, device="cuda", dtype=torch.float64).round().long()


def _local_output_decomposition(
    torch: Any, core: Any, x: Any, z: Any, qz: Any, weight: Any,
    transform: dict[str, Any], decoded: Any, row_indices: Any,
    equivalence_check: dict[str, Any],
) -> dict[str, Any]:
    x_sample = x.index_select(0, row_indices).float()
    z_sample = z.index_select(0, row_indices).float()
    qz_sample = qz.index_select(0, row_indices).float()
    delta = qz_sample - z_sample
    original_weight = weight.detach().float()
    reference = x_sample @ original_weight.T
    bhat = decoded.detach().float()
    activation_error = core.inverse_transform_input(delta, transform) @ original_weight.T
    weight_error = z_sample @ bhat.T - reference
    cross_error = delta @ bhat.T - activation_error
    total_error = qz_sample @ bhat.T - reference
    decomposed = activation_error + weight_error + cross_error
    norm = reference.square().sum().sqrt().clamp_min(torch.finfo(torch.float32).eps)

    def error_metrics(value: Any) -> tuple[float, float, float]:
        energy = value.square().sum()
        rmse = value.square().mean().sqrt()
        return float(rmse.item()), float((energy.sqrt() / norm).item()), float(energy.item())

    weight_rmse, weight_rrmse, weight_energy = error_metrics(weight_error)
    activation_rmse, activation_rrmse, activation_energy = error_metrics(activation_error)
    cross_rmse, cross_rrmse, cross_energy = error_metrics(cross_error)
    total_rmse, total_rrmse, total_energy = error_metrics(total_error)
    return {
        "weight_rmse": weight_rmse, "activation_rmse": activation_rmse,
        "cross_rmse": cross_rmse, "total_rmse": total_rmse,
        "weight_rrmse": weight_rrmse, "activation_rrmse": activation_rrmse,
        "cross_rrmse": cross_rrmse, "total_rrmse": total_rrmse,
        "reference_output_energy": float(reference.square().sum().item()),
        "weight_error_energy": weight_energy, "activation_error_energy": activation_energy,
        "cross_error_energy": cross_energy, "total_error_energy": total_energy,
        "decomposition_absmax": float((total_error - decomposed).abs().max().item()),
        "normalization": "Frobenius error norm divided by ||Z B^T||_F; reference is evaluated from the original BF16 input/weight in FP32",
        "arithmetic": "FP32 local Linear proxy; not native BF16 or INT4 kernel error",
        "reference_provenance": "original BF16 trajectory input and original BF16 checkpoint weight, both evaluated in FP32; transformed quantized weight is decoded from the read-only saved bank",
        "weight_error_definition": "z @ Bhat.T - x @ W.T memory-bounded proxy; it includes transform-equivalence float drift instead of materializing B on every callback",
        "transform_equivalence_drift_toy_max_abs": equivalence_check["transform_equivalence_drift_max_abs"],
        "activation_transform_equivalence_drift_toy_max_abs": equivalence_check["activation_transform_equivalence_drift_max_abs"],
        "transform_equivalence_drift_scope": "small deterministic CUDA toy self-check with explicit B; not a per-layer runtime estimate",
    }


def _selfcheck_decomposition(torch: Any, core: Any) -> dict[str, float]:
    class ForwardProbe:
        def forward(self):
            return None

    probe = ForwardProbe()
    original_forward = probe.forward
    assert probe.forward == original_forward
    probe.forward = lambda: None
    assert probe.forward != original_forward
    device = "cuda"
    generator = torch.Generator(device=device)
    generator.manual_seed(20261006)
    weight = torch.randn((11, 16), device=device, dtype=torch.bfloat16, generator=generator)
    x = torch.randn((19, 16), device=device, dtype=torch.bfloat16, generator=generator)
    scale = torch.rand(16, device=device, generator=generator) + 0.1
    transform = core.make_transform(scale, weight, 0.5, True, 8, 20261006)
    z = core.transform_input(x, transform)
    base = core.transform_weight(weight.float(), transform)
    qz = core.quant_activation(z, ACTIVATION_BITS)[0]
    noise = torch.randn(base.shape, device=device, dtype=base.dtype, generator=generator)
    synthetic_decoded = (base + 0.002 * noise).to(torch.bfloat16)
    indices = _uniform_rows(torch, x.shape[0], 16)
    x_rows = x.float().reshape(-1, 16)
    z_rows = z.reshape(-1, 16)
    qz_rows = qz.float().reshape(-1, 16)
    result = _local_output_decomposition(
        torch, core, x_rows, z_rows, qz_rows, weight, transform, synthetic_decoded,
        indices, {"transform_equivalence_drift_max_abs": 0.0,
                  "activation_transform_equivalence_drift_max_abs": 0.0},
    )
    if result["decomposition_absmax"] > 2e-4:
        raise RuntimeError(f"local-output decomposition self-check failed: {result['decomposition_absmax']}")
    # Explicitly materialize B only in this tiny self-check to quantify the
    # transform-equivalence drift hidden inside the production proxy terms.
    x_sample = x_rows.index_select(0, indices).float()
    z_sample = z_rows.index_select(0, indices).float()
    qz_sample = qz_rows.index_select(0, indices).float()
    base_sample = base.float()
    reference_b = x_sample @ weight.float().T
    transformed_reference = z_sample @ base_sample.T
    delta = qz_sample - z_sample
    explicit_activation = delta @ base_sample.T
    inverse_activation = core.inverse_transform_input(delta, transform) @ weight.float().T
    explicit_weight = z_sample @ (synthetic_decoded.float() - base_sample).T
    explicit_cross = delta @ (synthetic_decoded.float() - base_sample).T
    direct_total = qz_sample @ synthetic_decoded.float().T - reference_b
    explicit_total = explicit_activation + explicit_weight + explicit_cross
    transform_drift = (transformed_reference - reference_b).abs().max()
    activation_drift = (explicit_activation - inverse_activation).abs().max()
    explicit_decomposition_drift = (direct_total - explicit_total).abs().max()
    if float(explicit_decomposition_drift.item()) > 2e-4:
        raise RuntimeError(f"explicit-B toy decomposition self-check failed: {float(explicit_decomposition_drift.item())}")
    torch.testing.assert_close(transformed_reference, reference_b, atol=2e-4, rtol=2e-4)
    return {
        "transform_equivalence_drift_max_abs": float(transform_drift.item()),
        "activation_transform_equivalence_drift_max_abs": float(activation_drift.item()),
        "explicit_b_decomposition_drift_max_abs": float(explicit_decomposition_drift.item()),
    }


def _block_snapshot(torch: Any, np: Any, original: Any, transformed: Any, quantized: Any, error: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    maps = {}
    boundaries = None
    for key, value in (("original_absmax", original), ("transformed_absmax", transformed),
                       ("quantized_absmax", quantized), ("error_absmax", error)):
        pooled, current_boundaries = _pool_absmax(torch, value)
        maps[key] = pooled
        if boundaries is None:
            boundaries = current_boundaries
        elif boundaries != current_boundaries:
            raise RuntimeError("activation heatmap pooling boundaries diverged")
    return maps, boundaries


def _load_full_actions(path: Path, wanted: dict[tuple[int, int, int], int]) -> dict[tuple[int, int, int], dict[str, Any]]:
    found: dict[tuple[int, int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("case_id"), row.get("seed_index"), row.get("sampler_seed"))
            if key not in wanted or row.get("arm") != "bf16":
                continue
            if key in found:
                raise RuntimeError(f"duplicate full BF16 action row for {key}")
            if "action" not in row:
                raise ValueError(f"full action row lacks action tensor for {key}")
            found[key] = row
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"full/actions.jsonl lacks matching BF16 rows: {sorted(missing)}")
    return found


def _import_sources(script_dir: Path, source_root: Path) -> tuple[Any, Any, Any]:
    for path in (source_root, script_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_validation as smooth_runner
    import run_diagnostic as source
    import smooth_vq as core
    return smooth_runner, source, core


def run(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) in sys.path:
        sys.path.remove(str(script_dir))
    sys.path.insert(0, str(script_dir))
    import run_validation as smooth_runner

    # This must precede torch imports, bank reads, model loading, or numerical self-checks.
    smooth_runner.guard()
    stage = "guarded"
    torch = np = runtime = None
    hook_handles: list[Any] = []
    scheduler_handles: list[Any] = []
    conditioning_handles: list[Any] = []
    previous_tf32 = None
    old_root = None
    root_was_modified = False
    output_dir: Path | None = None
    try:
        import torch as torch_module
        import numpy as numpy_module

        torch, np = torch_module, numpy_module
        diag_root = Path(os.environ.get("DIAG_ROOT", script_dir)).resolve()
        source_root = _resolve_path(args.source_root or os.environ.get("SOURCE_ROOT", "fastwam-a4-phases12"), [Path.cwd(), diag_root, script_dir.parent])
        full_root = _resolve_path(args.full or "full-results", [diag_root, Path.cwd(), script_dir])
        protocol_path = _resolve_path(args.protocol or "visualization_protocol.json", [script_dir, Path.cwd(), diag_root])
        output_dir = _resolve_path(args.out or "results/visualize", [diag_root, Path.cwd(), script_dir])
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        _validate_visualization_protocol(protocol)
        _json_save(output_dir / "protocol.json", protocol)
        source_job_evidence = _verify_full_run(full_root, protocol)

        smooth_protocol_path = script_dir / "protocol.json"
        if not smooth_protocol_path.is_file():
            smooth_protocol_path = diag_root / "protocol.json"
        smooth_protocol = json.loads(smooth_protocol_path.read_text(encoding="utf-8"))
        if smooth_protocol.get("protocol") != SOURCE_PROTOCOL_NAME:
            raise RuntimeError("saved full-run protocol does not match the visualization source protocol")
        if int(smooth_protocol.get("transform_seed", -1)) != 20261006:
            raise RuntimeError("saved source protocol transform_seed must remain the fixed seed 20261006")
        plan_path = source_root / "plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"fixed-input plan missing: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        old_root = os.environ.get("ROOT")
        os.environ["ROOT"] = os.environ.get("PILOT_ROOT", str(Path("/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005")))
        root_was_modified = True
        os.environ["DIAG_ROOT"] = str(source_root)
        smooth_runner, source, core = _import_sources(script_dir, source_root)
        if plan.get("protocol") != source.PROTOCOL:
            raise RuntimeError(f"fixed-input plan protocol mismatch: {plan.get('protocol')!r}")

        if torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one PBS-allocated CUDA GPU is required")
        props = torch.cuda.get_device_properties(0)
        cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_mask or str(props.uuid).removeprefix("GPU-").lower() != cuda_mask.removeprefix("GPU-").lower():
            raise RuntimeError("CUDA GPU UUID does not match the visible PBS allocation")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        equivalence_check = _selfcheck_decomposition(torch, core)

        pilot = source.load_pilot_runner()
        stage = "model_loading"
        runtime = pilot.Runtime(output_dir)
        runtime.arm("bf16")
        if runtime.q.weights_converted or any(module.weight is None for module in runtime.q.modules.values()):
            raise RuntimeError("collector requires original BF16 target weights throughout")
        module_items = list(runtime.q.modules.items())
        if len(module_items) != EXPECTED_MODULE_COUNT:
            raise RuntimeError(f"expected {EXPECTED_MODULE_COUNT} target Linear modules, got {len(module_items)}")
        module_names = [name for name, _module in module_items]
        if len(set(module_names)) != EXPECTED_MODULE_COUNT:
            raise RuntimeError("target module names are not unique")
        module_index = {name: index for index, (name, _module) in enumerate(module_items)}
        module_id_to_name = {id(module): name for name, module in module_items}
        streams = {name: source_scope(name) for name in module_names}
        module_counts = dict(Counter(streams.values()))

        def progress(current_stage: str, **fields: Any) -> None:
            value = {"status": "running", "stage": current_stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **fields}
            _json_save(output_dir / "progress.json", value)
            print("VISUALIZATION_PROGRESS " + json.dumps(value, ensure_ascii=False), flush=True)

        stage = "bank_cache"
        encoded_cache, transforms_cpu = _load_saved_banks(torch, full_root, runtime.q.modules, progress)
        device = next(iter(runtime.q.modules.values())).weight.device
        transforms: dict[str, dict[str, dict[str, Any]]] = {config: {} for config in protocol["activation_configs"]}
        for name, module in module_items:
            transforms["identity"][name] = core.make_transform(None, module.weight, None, False, int(smooth_protocol["hadamard_block"]), 20261006)
            transforms["hadamard"][name] = core.make_transform(None, module.weight, None, True, int(smooth_protocol["hadamard_block"]), 20261006)
            for config in WEIGHT_CONFIGS:
                transforms[config][name] = _to_device_transform(transforms_cpu[config][name], device)

        stage = "weight_scan"
        weight_records, weight_lookup = _scan_weights(torch, np, core, output_dir, runtime.q.modules, streams, transforms, encoded_cache, progress)
        selected = _select_modules(weight_records, len(module_items))
        _json_save(output_dir / "selected.json", selected)
        selected_indices = {item["module_index"] for item in selected}
        selected_snapshots: set[tuple[int, str, str, str, int]] = set()
        channel_stats_count = 0
        (output_dir / "channels").mkdir(parents=True, exist_ok=True)

        # Keep identity checks to prove the hooks observe without replacing weights or forwards.
        weight_identity = {name: id(module.weight) for name, module in module_items}
        weight_version = {name: module.weight._version for name, module in module_items}
        forward_identity = {name: module.forward for name, module in module_items}
        tagger = CollectorStageTagger()
        scheduler_handles = source.install_stage_schedulers(runtime.model, tagger)
        conditioning_handles = source.install_video_conditioning_wrappers(runtime.model, tagger, scheduler_handles[0][3])
        active: dict[str, Any] = {}
        total_callbacks = 0
        last_progress_bucket = 0
        activation_count = 0

        def collect_activation(module: Any, inputs: tuple[Any, ...], _name: str, _index: int) -> None:
            nonlocal total_callbacks, last_progress_bucket, activation_count, channel_stats_count
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            x_tensor = inputs[0].detach()
            if x_tensor.shape[-1] != int(module.in_features):
                raise RuntimeError(f"activation width mismatch for {_name}: {tuple(x_tensor.shape)}")
            x = x_tensor.reshape(-1, x_tensor.shape[-1]).float()
            if not x.numel():
                return
            stage_name, step_number = tagger.stage, tagger.step
            call_index = int(active["call_index"])
            active["call_index"] += 1
            active["observed_modules"].add(_name)
            active["calls_by_stage"][stage_name] += 1
            active["steps_by_stage"].setdefault(stage_name, set()).add(step_number)
            active["calls_by_module_stage_step"][( _name, stage_name, step_number)] += 1
            total_callbacks += 1
            activation_count += 1
            row_indices = _uniform_rows(torch, int(x.shape[0]), int(protocol["local_output_rows"]))
            for config in protocol["activation_configs"]:
                transform = transforms[config][_name]
                z = core.transform_input(x, transform)
                qz, _quant_stats = core.quant_activation(z, ACTIVATION_BITS)
                delta = qz.float() - z
                error_flat = delta.abs().reshape(-1)
                error_absmax_tensor, error_flat_index = error_flat.max(dim=0)
                error_absmax = float(error_absmax_tensor.item())
                error_peak_position = [int(error_flat_index.item() // delta.shape[1]), int(error_flat_index.item() % delta.shape[1])]
                channel_error_energy_all = delta.square().sum(dim=0)
                top_channel_count = min(4, channel_error_energy_all.numel())
                top_channel_values, top_channel_indices = torch.topk(channel_error_energy_all, top_channel_count, largest=True, sorted=True)
                signal_squared = z.square().sum()
                error_squared = delta.square().sum()
                zero_count = (qz == 0).sum()
                elements = int(z.numel())
                rms = signal_squared.sqrt() / max(elements, 1) ** 0.5
                absmax = z.abs().amax()
                local_output = None
                if config in WEIGHT_CONFIGS:
                    local_output = {}
                    for kind in QUANTIZERS:
                        decoded = _decode(torch, core, encoded_cache[(config, _name, kind)], device)
                        result = _local_output_decomposition(
                            torch, core, x, z, qz, module.weight, transform, decoded,
                            row_indices, equivalence_check,
                        )
                        result["weight_reconstruction_rmse"] = weight_lookup[(config, _name)][f"{kind}_rmse"]
                        result["weight_reconstruction_rrmse"] = weight_lookup[(config, _name)][f"{kind}_rrmse"]
                        result["row_indices"] = row_indices.detach().cpu().tolist()
                        local_output[kind] = result
                        del decoded
                    # The local output is an FP32 proxy and leaves the BF16 model output untouched.

                record: dict[str, Any] = {
                    "case_id": active["case_id"], "seed_index": active["seed_index"],
                    "sampler_seed": active["sampler_seed"], "stage": stage_name, "step": step_number,
                    "call_index": call_index, "module": _name, "module_index": _index,
                    "stream": streams[_name], "rows": int(x.shape[0]), "input_shape": list(x_tensor.shape),
                    "config": config, "activation_rrmse": float((error_squared / signal_squared.clamp_min(torch.finfo(torch.float32).tiny)).sqrt().item()),
                    "zero_fraction": float((zero_count.float() / elements).item()),
                    "absmax": float(absmax.item()), "rms": float(rms.item()),
                    "error_absmax": error_absmax, "error_peak_position": error_peak_position,
                    "error_peak_position_space": "flattened token row, transformed activation channel",
                    "top4_channel_error_energy": [float(value.item()) for value in top_channel_values],
                    "top4_channel_error_channel": [int(index.item()) for index in top_channel_indices],
                    "outlier_ratio": float((absmax / rms.clamp_min(torch.finfo(torch.float32).eps)).item()),
                    "signal_squared": float(signal_squared.item()), "error_squared": float(error_squared.item()),
                    "zero_count": int(zero_count.item()), "elements": elements,
                    "activation_bits": ACTIVATION_BITS,
                    "transform_source": "saved full bank" if config in WEIGHT_CONFIGS else "fixed seed 20261006",
                    "aggregation": "all flattened Linear input rows and channels; local output uses deterministic uniform 16-row sample",
                }
                if local_output is not None:
                    record["local_output"] = local_output

                if _index in selected_indices:
                    channel_path = output_dir / "channels" / (
                        f"act_c{active['case_id']}_m{_index:04d}_{config}_{stage_name}_s{step_number}_call{call_index}.npz"
                    )
                    np.savez_compressed(
                        channel_path,
                        channel_rms=_numpy(z.square().mean(dim=0).sqrt()),
                        channel_absmax=_numpy(z.abs().amax(dim=0)),
                        channel_error_energy=_numpy(channel_error_energy_all),
                        channel_zero_fraction=_numpy((qz == 0).float().mean(dim=0)),
                        stage=np.asarray(stage_name), step=np.asarray(step_number, dtype=np.int32),
                        shape=np.asarray(x_tensor.shape, dtype=np.int64),
                        channel_space=np.asarray("transformed activation z; exact across all rows"),
                        scope=np.asarray("full per-channel statistics for this selected-module call/config"),
                    )
                    record["channel_stats_file"] = str(channel_path.relative_to(output_dir))
                    record["channel_stats_scope"] = "all rows and full channel dimension; every call/config for selected six modules"
                    channel_stats_count += 1

                snapshot_key = (active["case_id"], _name, config, stage_name, step_number)
                if (_index in selected_indices and stage_name in ("video", "action")
                        and step_number in protocol["snapshot_steps"] and snapshot_key not in selected_snapshots):
                    maps, boundaries = _block_snapshot(torch, np, x, z, qz.float(), delta)
                    snapshot_path = output_dir / "snapshots" / f"act_c{active['case_id']}_m{_index:04d}_{config}_{stage_name}_s{step_number}_call{call_index}.npz"
                    np.savez_compressed(
                        snapshot_path,
                        **maps,
                        channel_rms=_numpy(z.square().mean(dim=0).sqrt()), channel_absmax=_numpy(z.abs().amax(dim=0)),
                        channel_error_energy=_numpy(channel_error_energy_all), channel_zero_fraction=_numpy((qz == 0).float().mean(dim=0)),
                        row_starts=np.asarray(boundaries["row_starts"], dtype=np.int32),
                        row_ends=np.asarray(boundaries["row_ends"], dtype=np.int32),
                        column_starts=np.asarray(boundaries["column_starts"], dtype=np.int32),
                        column_ends=np.asarray(boundaries["column_ends"], dtype=np.int32),
                        step=np.asarray(step_number, dtype=np.int32), stage=np.asarray(stage_name),
                        shape=np.asarray(x_tensor.shape, dtype=np.int64),
                        flattened_shape=np.asarray(z.shape, dtype=np.int64),
                        channel_space=np.asarray("transformed activation z; exact across all rows"),
                        pooling=np.asarray("nonoverlapping deterministic block absmax over all flattened rows; at most 64x64"),
                    )
                    record["snapshot_file"] = str(snapshot_path.relative_to(output_dir))
                    record["snapshot_call"] = "first selected-module call for this case/config/stage/step"
                    selected_snapshots.add(snapshot_key)
                _jsonl_append(output_dir / "activations.jsonl", record)
                del z, qz, delta
            progress_bucket = total_callbacks // 100
            if progress_bucket > last_progress_bucket:
                last_progress_bucket = progress_bucket
                progress("activation_callbacks", callback_count=total_callbacks, case_id=active["case_id"],
                         call_index=call_index, current_module=_name)

        for name, module in module_items:
            index = module_index[name]

            def _hook(_module: Any, inputs: tuple[Any, ...], _name: str = name, _index: int = index) -> None:
                collect_activation(_module, inputs, _name, _index)

            hook_handles.append(module.register_forward_pre_hook(_hook))

        expected_rows: dict[tuple[int, int, int], int] = {}
        case_inputs: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]] = {}
        for case_id in protocol["cases"]:
            row = source.validate_plan(plan, case_id)
            observation, metadata = source.load_case(source_root, row)
            seed_index = int(protocol["seed_indices"][0])
            sampler_seed = int(row["sampler_seeds"][seed_index])
            expected_rows[(case_id, seed_index, sampler_seed)] = case_id
            case_inputs[case_id] = (row, observation, metadata, seed_index, sampler_seed)
        full_actions = _load_full_actions(full_root / "actions.jsonl", expected_rows)

        stage = "activation_collection"
        query_coverage = []
        for case_id in protocol["cases"]:
            row, observation, metadata, seed_index, sampler_seed = case_inputs[case_id]
            datum = runtime.datum(observation)
            runtime.description = metadata["description"]
            runtime.current_arm = "bf16"
            source.reset_stage_calls(scheduler_handles)
            tagger.set_stage("conditioning", -1)
            active.update(case_id=case_id, seed_index=seed_index, sampler_seed=sampler_seed,
                          call_index=0, observed_modules=set(), calls_by_stage=Counter(),
                          steps_by_stage={}, calls_by_module_stage_step=Counter())
            progress("query_running", case_id=case_id, seed_index=seed_index, sampler_seed=sampler_seed,
                     completed_queries=len(query_coverage), expected_queries=len(protocol["cases"]))
            started = time.perf_counter()
            with torch.inference_mode():
                action, _metrics = runtime.infer(datum, sampler_seed, measure=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if runtime.q.weights_converted or runtime.q.summary().get("native_int4_gemm_calls", 0):
                raise RuntimeError("collector unexpectedly left the BF16 path")
            if tuple(action.shape) != (32, 7) or not bool(torch.isfinite(action).all()):
                raise RuntimeError(f"invalid normalized action for case {case_id}")
            scheduler_coverage = source.validate_stage_calls(scheduler_handles)
            expected_names = set(module_names)
            observed = active["observed_modules"]
            if observed != expected_names:
                raise RuntimeError(f"case {case_id} observed {len(observed)}/{len(expected_names)} target modules")
            denoising_steps = {
                stage_name: sorted(step for step in active["steps_by_stage"].get(stage_name, set()) if type(step) is int)
                for stage_name in ("video", "action")
            }
            if denoising_steps != {"video": list(range(10)), "action": list(range(10))}:
                raise RuntimeError(f"case {case_id} activation hooks missed denoising step coverage: {denoising_steps}")
            if active["steps_by_stage"].get("conditioning", set()) and active["steps_by_stage"]["conditioning"] != {-1}:
                raise RuntimeError("conditioning callbacks were mixed into denoising steps")
            if active["steps_by_stage"].get("video_conditioning_prefill", set()) not in (set(), {-1}):
                raise RuntimeError("video conditioning prefill was mixed into denoising steps")
            for name, module in module_items:
                if id(module.weight) != weight_identity[name] or module.weight._version != weight_version[name]:
                    raise RuntimeError(f"collector changed/replaced BF16 weight for {name}")
                # Bound methods are recreated on attribute access; equality checks their instance/function.
                if module.forward != forward_identity[name]:
                    raise RuntimeError(f"collector replaced Linear forward for {name}")

            action_cpu = action.detach().float().cpu()
            key = (case_id, seed_index, sampler_seed)
            full_action = torch.as_tensor(full_actions[key]["action"], dtype=torch.float32)
            if tuple(full_action.shape) != (32, 7):
                raise RuntimeError(f"full action shape mismatch for case {case_id}: {tuple(full_action.shape)}")
            difference = action_cpu - full_action
            max_abs_drift = float(difference.abs().max().item())
            drift_record = {
                "case_id": case_id, "seed_index": seed_index, "sampler_seed": sampler_seed,
                "matched_full_arm": full_actions[key].get("arm"),
                "max_abs_drift": max_abs_drift,
                "rmse_drift": float(difference.square().mean().sqrt().item()),
                "peak_position": [int(value) for value in torch.nonzero(difference.abs() == difference.abs().max(), as_tuple=False)[0].tolist()],
                "anomaly_limit": ACTION_DRIFT_LIMIT,
                "within_anomaly_limit": max_abs_drift <= ACTION_DRIFT_LIMIT,
                "limit_interpretation": "diagnostic anomaly cutoff only; not a scientific equivalence threshold",
            }
            _jsonl_append(output_dir / "action_alignment.jsonl", drift_record)
            action_snapshot = output_dir / "snapshots" / f"action_drift_c{case_id}.npz"
            np.savez_compressed(action_snapshot, reference_action=full_action.numpy(), diagnostic_action=action_cpu.numpy(),
                                signed_error=difference.numpy(), abs_error=difference.abs().numpy(),
                                case_id=np.asarray(case_id, dtype=np.int32), sampler_seed=np.asarray(sampler_seed, dtype=np.int64))
            if max_abs_drift > ACTION_DRIFT_LIMIT:
                raise RuntimeError(f"case {case_id} BF16 action drift {max_abs_drift} exceeds anomaly cutoff {ACTION_DRIFT_LIMIT}")

            observed_by_stage = {
                stage_name: {"calls": int(active["calls_by_stage"].get(stage_name, 0)),
                             "steps": sorted(active["steps_by_stage"].get(stage_name, set()))}
                for stage_name in ("conditioning", "video_conditioning_prefill", "video", "action")
            }
            selected_module_names = [item["module"] for item in selected]
            selected_names = set(selected_module_names)
            selected_actual_coverage: dict[str, list[dict[str, Any]]] = {name: [] for name in selected_module_names}
            stage_order = {name: index for index, name in enumerate(("conditioning", "video_conditioning_prefill", "video", "action"))}
            for (module_name, stage_name, step_number), calls in active["calls_by_module_stage_step"].items():
                if module_name in selected_names:
                    selected_actual_coverage[module_name].append(
                        {"stage": stage_name, "step": int(step_number), "calls": int(calls)}
                    )
            for module_name in selected_module_names:
                selected_actual_coverage[module_name].sort(
                    key=lambda item: (stage_order.get(item["stage"], 99), item["step"])
                )
            query_record = {
                "case_id": case_id, "seed_index": seed_index, "sampler_seed": sampler_seed,
                "label": "exploratory known input; not held-out evaluation",
                "query_index": len(query_coverage), "bf16_query_completed": True,
                "runtime_calls": runtime.calls, "observed_module_count": len(observed),
                "observed_module_names": sorted(observed), "module_counts_by_stream": module_counts,
                "stage_scheduler_coverage": scheduler_coverage,
                "activation_hook_coverage": observed_by_stage,
                "selected_module_actual_stage_step_coverage": selected_actual_coverage,
                "denoising_steps": denoising_steps,
                "call_index_scope": "global Linear-hook order within this BF16 query; not a scheduler step",
                "action_alignment": drift_record,
                "query_wall_seconds_diagnostic_only": elapsed,
                "episodes": 0, "post_query_env_steps": 0,
            }
            _jsonl_append(output_dir / "queries.jsonl", query_record)
            query_coverage.append(query_record)
            progress("query_complete", case_id=case_id, seed_index=seed_index, sampler_seed=sampler_seed,
                     completed_queries=len(query_coverage), expected_queries=len(protocol["cases"]),
                     observed_modules=len(observed), denoising_steps=denoising_steps)

        if runtime.calls != len(protocol["cases"]):
            raise RuntimeError(f"expected {len(protocol['cases'])} BF16 Runtime calls, got {runtime.calls}")
        if activation_count == 0:
            raise RuntimeError("collector produced no activation records")
        if any(item["observed_module_count"] != EXPECTED_MODULE_COUNT for item in query_coverage):
            raise RuntimeError("not all 614 target modules were observed for each fixed input")
        expected_snapshots = {
            (query["case_id"], module_name, config, event["stage"], event["step"])
            for query in query_coverage
            for module_name, events in query["selected_module_actual_stage_step_coverage"].items()
            for event in events
            if event["stage"] in ("video", "action") and event["step"] in protocol["snapshot_steps"] and event["calls"] > 0
            for config in protocol["activation_configs"]
        }
        if selected_snapshots != expected_snapshots:
            missing = sorted(expected_snapshots - selected_snapshots)
            extra = sorted(selected_snapshots - expected_snapshots)
            raise RuntimeError(f"selected activation heatmap coverage mismatch; missing={missing[:8]}, extra={extra[:8]}")
        if channel_stats_count == 0:
            raise RuntimeError("selected-module per-call channel statistics were not collected")

        activation_stage_counts = Counter()
        activation_step_counts = Counter()
        activation_config_counts = Counter()
        local_output_counts = Counter()
        with (output_dir / "activations.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                activation_stage_counts[record["stage"]] += 1
                activation_step_counts[f"{record['stage']}:{record['step']}"] += 1
                activation_config_counts[record["config"]] += 1
                if "local_output" in record:
                    for kind in QUANTIZERS:
                        local_output_counts[kind] += 1
        summary = {
            "protocol": protocol["protocol"], "source_protocol": protocol["source_protocol"],
            "pbs_jobid": os.environ["PBS_JOBID"], "hostname": socket.gethostname(),
            "source_full_job": protocol["source_full_job"], "source_full_run_evidence": source_job_evidence,
            "status": "complete", "context_scope": protocol["context_scope"],
            "cases": protocol["cases"], "seed_indices": protocol["seed_indices"],
            "query_count": len(query_coverage), "bf16_queries_completed": len(query_coverage),
            "runtime_calls": runtime.calls, "target_module_count": EXPECTED_MODULE_COUNT,
            "observed_module_coverage_per_query": [record["observed_module_count"] for record in query_coverage],
            "module_counts_by_stream": module_counts,
            "weight_records": len(weight_records), "weight_records_by_config": dict(Counter(row["config"] for row in weight_records)),
            "weight_snapshots": len(weight_records),
            "weight_snapshot_scope": "all 614 modules x both weight configs; full row/channel summaries plus deterministic max-pooled matrices",
            "selected_modules": selected,
            "activation_records": activation_count * len(protocol["activation_configs"]),
            "activation_records_by_config": dict(activation_config_counts),
            "activation_records_by_stage": dict(activation_stage_counts),
            "activation_records_by_stage_step": dict(activation_step_counts),
            "activation_snapshots": len(selected_snapshots),
            "activation_snapshot_coverage": {
                "observed": len(selected_snapshots), "expected": len(expected_snapshots),
                "cases": protocol["cases"], "configs": protocol["activation_configs"],
                "denoising_stages": ["video", "action"], "steps": protocol["snapshot_steps"],
                "selected_modules_only": True, "stage_step_basis": "actual per-module forward-hook calls, not stream name",
                "coverage_exact": True,
            },
            "selected_module_actual_stage_step_coverage_per_query": [
                {"case_id": query["case_id"], "modules": query["selected_module_actual_stage_step_coverage"]}
                for query in query_coverage
            ],
            "local_output_records_by_quantizer": dict(local_output_counts),
            "activation_scalar_aggregation": "all elements of each target Linear input; exact per callback, energies additive across calls",
            "activation_channel_aggregation": "exact full-row channel summaries for every call/config of selected six modules; other modules retain scalar summaries",
            "activation_channel_stats_files": channel_stats_count,
            "snapshot_pooling": "deterministic adaptive/block max abs over full matrix, at most 64x64; bins recorded",
            "local_output_proxy": "FP32 local Linear proxy from same BF16 trajectory inputs and stored transformed weights; not actual BF16/INT4 kernel error or closed-loop behavior",
            "local_output_reference_provenance": "original BF16 model/input values converted to FP32; decoded candidate weight comes from the read-only saved full-run bank",
            "local_output_transform_equivalence": {
                **equivalence_check,
                "scope": "small deterministic CUDA toy self-check with explicit transformed B; production per-callback decomposition keeps the memory-bounded proxy and can include small transform-equivalence float drift",
            },
            "outlier_ratio_definition": "transformed activation absmax divided by transformed activation RMS for the full flattened Linear input",
            "local_output_row_sampling": protocol["row_sampling"],
            "action_alignment": {"records": [row["action_alignment"] for row in query_coverage],
                                 "anomaly_limit": ACTION_DRIFT_LIMIT,
                                 "max_abs_drift": max(row["action_alignment"]["max_abs_drift"] for row in query_coverage)},
            "coverage": query_coverage,
            "flags": {
                "exploratory_known_inputs": True, "new_held_out_conclusion": False,
                "original_bf16_weights_retained": True, "weight_conversion": False,
                "codebook_fit": False, "restoration_experiment": False,
                "evaluation_episodes": 0, "action_environment_steps": 0,
                "observational_hooks": True, "no_model_weight_replacement": True,
                "no_output_replacement": True, "native_latency_claim": False,
            },
            "outputs": {
                "weights.jsonl": ["config", "module", "module_index", "stream", "shape", "weight_rms", "scalar_rmse", "vq_rmse", "scalar_rrmse", "vq_rrmse", "ratio", "scalar_peak_error", "vq_peak_error", "scalar_peak_position", "vq_peak_position", "top8_absolute_error_coordinates", "snapshot_file"],
                "selected.json": "six-element array of {module,module_index,stream,reason}",
                "activations.jsonl": ["case_id", "seed_index", "sampler_seed", "stage", "step", "call_index", "module", "module_index", "stream", "rows", "input_shape", "config", "activation_rrmse", "zero_fraction", "absmax", "rms", "error_absmax", "error_peak_position", "top4_channel_error_energy", "top4_channel_error_channel", "outlier_ratio", "signal_squared", "error_squared", "zero_count", "local_output", "channel_stats_file", "snapshot_file"],
                "channels/*.npz": ["channel_rms", "channel_absmax", "channel_error_energy", "channel_zero_fraction", "stage", "step", "shape"],
                "weight_snapshot_npz": ["weight_absmax", "scalar_error_absmax", "vq_error_absmax", "row_rms", "row_scalar_rrmse", "row_vq_rrmse", "scalar_column_energy", "vq_column_energy", "scalar_output_channel_error_energy", "vq_output_channel_error_energy", "pooling boundaries", "original_shape"],
                "activation_snapshot_npz": ["original_absmax", "transformed_absmax", "quantized_absmax", "error_absmax", "channel_rms", "channel_absmax", "channel_error_energy", "channel_zero_fraction", "stage", "step", "shape", "flattened_shape", "pooling boundaries"],
                "action_alignment.jsonl": "per-query BF16 action max-absolute drift against matching source full/actions.jsonl row",
            },
        }
        _json_save(output_dir / "summary.json", summary)
        _json_save(output_dir / "progress.json", {"status": "complete", "queries_completed": len(query_coverage),
                                                     "activation_callbacks": total_callbacks, "activation_records": activation_count * len(protocol["activation_configs"])})
        print("VISUALIZATION_COMPLETE " + json.dumps({"queries": len(query_coverage), "modules": EXPECTED_MODULE_COUNT,
                                                        "selected": selected, "output": str(output_dir)}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        if output_dir is not None:
            _json_save(output_dir / "failure.json", {"stage": stage, "error": f"{type(exc).__name__}: {exc}"})
            _json_save(output_dir / "progress.json", {"status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        try:
            for handle in hook_handles:
                handle.remove()
            if runtime is not None:
                if conditioning_handles:
                    source.restore_methods(conditioning_handles)
                if scheduler_handles:
                    source.restore_stage_schedulers(scheduler_handles)
                runtime.q.disable()
                runtime.q.close()
                if runtime.env is not None:
                    runtime.env.close()
        finally:
            if previous_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            if root_was_modified:
                if old_root is not None:
                    os.environ["ROOT"] = old_root
                else:
                    os.environ.pop("ROOT", None)


def source_scope(name: str) -> str:
    parts = name.split(".")
    if "video_expert" in parts or any(parts[i:i + 3] == ["mot", "mixtures", "video"] for i in range(len(parts) - 2)):
        return "video"
    if "action_expert" in parts or any(parts[i:i + 3] == ["mot", "mixtures", "action"] for i in range(len(parts) - 2)):
        return "action"
    if "proprio_encoder" in parts:
        return "proprio"
    raise ValueError(f"module outside video/action/proprio scopes: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--full", type=Path, help="full result directory containing banks/, actions.jsonl, summary.json")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = Path("results") / f"visualize_{stamp}"
    run(args)


if __name__ == "__main__":
    main()

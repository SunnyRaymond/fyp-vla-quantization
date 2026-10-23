#!/usr/bin/env python3
"""Render weight and fixed-input Linear error figures from captured arrays."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path, PureWindowsPath


SCHEMA = "weight-analysis-v1"
RENDERER_SCHEMA = "fastwam-weight-analysis-renderer-v2"
METHODS = ("spinquant", "row_scale_vq", "micro_vq")
METHOD_LABELS = {"spinquant": "Learned SpinQuant", "row_scale_vq": "Row-scale VQ",
                 "micro_vq": "MicroVQ", "micro_scale_vq": "MicroVQ"}
METHOD_COLORS = {"spinquant": "#d66b4d", "row_scale_vq": "#377eb8",
                 "micro_vq": "#8c68aa", "micro_scale_vq": "#8c68aa"}
METHOD_ALIASES = {
    "spin": "spinquant", "learned_spinquant": "spinquant",
    "spinquant_native_w4a4": "spinquant", "spinquant_native_w4a4": "spinquant",
    "row": "row_scale_vq", "row_vq": "row_scale_vq", "row_scale_vq": "row_scale_vq",
    "row_scale_vq_a4": "row_scale_vq", "micro": "micro_vq", "micro_vq": "micro_vq",
    "micro_scale_vq": "micro_vq", "micro_scale_vq_a4": "micro_vq",
}
MAX_PREVIEW_BYTES = 220 * 1024


def canonical_method(value: object) -> str:
    key = str(value).lower().replace("-", "_")
    return METHOD_ALIASES.get(key, key)


def slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "item"


def require_compute_allocation() -> None:
    # Reuse the collector's exact PBS allocation and host check before reading arrays.
    import collect
    collect.guard()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_input_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("module record lacks its NPZ path")
    normalized = value.replace("\\", "/")
    rel = Path(normalized)
    if rel.is_absolute() or PureWindowsPath(normalized).is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"NPZ path must stay inside the analysis result: {value!r}")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"NPZ path escapes the analysis result: {value!r}") from exc
    if not path.is_file():
        raise RuntimeError(f"module NPZ does not exist: {path}")
    return path


def source_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def protocol_source_identity(path: Path) -> dict:
    # The parsed protocol is embedded in renderer identity, so file metadata
    # must not invalidate otherwise identical plot receipts.
    return {"path": path.name}


def normalize_identity(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    protocol = normalized.get("protocol")
    if isinstance(protocol, dict):
        protocol = dict(protocol)
        protocol.pop("_path", None)
        normalized["protocol"] = protocol
    return normalized


def normalize_source_identities(value: object) -> object:
    if not isinstance(value, list):
        return value
    normalized = [dict(row) if isinstance(row, dict) else row for row in value]
    # The first row is weight_protocol.json. Older receipts included its size
    # and mtime; parsed contents already participate in renderer identity.
    if normalized and isinstance(normalized[0], dict):
        normalized[0].pop("mtime_ns", None)
        normalized[0].pop("size", None)
    return normalized


def canonical_map(mapping: dict) -> dict:
    return {canonical_method(key): value for key, value in mapping.items()}


def array_for(npz, key: object, label: str, ndim: int | None = None):
    if not isinstance(key, str) or not key:
        raise RuntimeError(f"{label} lacks an NPZ key")
    if key not in npz.files:
        raise RuntimeError(f"{label} references missing NPZ array {key!r}")
    value = npz[key]
    if not value.size or (ndim is not None and value.ndim != ndim):
        raise RuntimeError(f"{label} has invalid shape {value.shape}")
    return value


def stat_value(value: object, label: str) -> float:
    if isinstance(value, dict):
        for field in ("nmse", "weight_nmse", "output_nmse", "value", "value_pct"):
            if field in value:
                value = value[field]
                break
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{label} must be a finite scalar; got {value!r}")
    return float(value)


def contexts_by_module(data: dict, module: dict) -> list[dict]:
    all_contexts = {str(row["id"]): row for row in data.get("contexts", []) if isinstance(row, dict)}
    module_contexts = module.get("contexts")
    if not isinstance(module_contexts, dict) or len(module_contexts) != 6:
        raise RuntimeError(f"{module.get('id')} must contain its six stream contexts")
    result = []
    for context_id, row in module_contexts.items():
        if not isinstance(row, dict):
            raise RuntimeError(f"{module['id']} context {context_id} must be an object")
        meta = all_contexts.get(str(context_id), {})
        merged = {**meta, **row, "id": str(context_id)}
        if not isinstance(merged.get("case_id"), int) or not isinstance(merged.get("step"), int):
            match = re.fullmatch(r"(\d+)\|[^|]+\|(\d+)", str(context_id))
            if match:
                merged["case_id"], merged["step"] = int(match.group(1)), int(match.group(2))
        if not isinstance(merged.get("index"), int):
            raise RuntimeError(f"{module['id']} context {context_id} lacks its NPZ context index")
        result.append(merged)
    result.sort(key=lambda row: (row["case_id"], row["step"]))
    if [(row["case_id"], row["step"]) for row in result] != [
        (case, step) for case in (1004, 1014) for step in (0, 4, 9)
    ]:
        raise RuntimeError(f"{module['id']} contexts differ from the frozen 2-case × 3-step set")
    return result


def context_array_key(module: dict, context: dict, array_name: str) -> str:
    """Resolve compact module-level NPZ references, with legacy row fallback."""
    row_key = context.get(f"{array_name}_key")
    if isinstance(row_key, str) and row_key:
        return row_key
    module_keys = module.get("context_profile_keys", {})
    key = module_keys.get(array_name) if isinstance(module_keys, dict) else None
    return key if isinstance(key, str) and key else array_name


def validate_input(data: dict, protocol: dict, data_path: Path) -> tuple[list, list, dict]:
    if data.get("schema") != SCHEMA or data.get("status") != "complete":
        raise RuntimeError("weight-analysis JSON must use weight-analysis-v1 and status=complete")
    if protocol.get("protocol") != "fastwam-weight-analysis-v1":
        raise RuntimeError("unexpected frozen weight-analysis protocol")
    if protocol.get("expected_modules") != 16 or protocol.get("new_whole_model_forwards") != 0:
        raise RuntimeError("protocol must declare the frozen 16 modules and zero whole-model forwards")
    modules = data.get("modules")
    contexts = data.get("contexts")
    if not isinstance(modules, list) or len(modules) != 16:
        raise RuntimeError("expected all 16 selected Linear modules")
    if not isinstance(contexts, list) or len(contexts) not in (6, 12):
        raise RuntimeError("expected six contexts per stream in the two-stream scope")
    if len({str(module.get("id")) for module in modules}) != 16:
        raise RuntimeError("module IDs must be present and unique")
    for module in modules:
        if not isinstance(module, dict):
            raise RuntimeError("module entries must be objects")
        for field in ("npz", "distribution", "block_nmse", "scale_profile", "weight_metrics", "contexts"):
            if field not in module:
                raise RuntimeError(f"module {module.get('id')} lacks {field}")
        if len(contexts_by_module(data, module)) != 6:
            raise RuntimeError(f"module {module['id']} has incomplete context coverage")
        if not all(method in canonical_map(module["block_nmse"]) for method in METHODS):
            raise RuntimeError(f"module {module['id']} lacks B32 NMSE for all three arms")
    # Ensure the manifest describes all method and transform lanes used to interpret arrays.
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("error_lane_order"), list):
        raise RuntimeError("metadata.error_lane_order is required")
    error_methods = {canonical_method(key) for key in metadata["error_lane_order"]}
    if not all(method in error_methods for method in METHODS):
        raise RuntimeError("metadata.error_lane_order must include all three comparison arms")
    basis_lanes = metadata.get("basis_lane_order")
    if not isinstance(basis_lanes, list) or len(basis_lanes) != 4:
        raise RuntimeError("metadata.basis_lane_order must define raw, SpinQuant, Row and Micro bases")
    npz_paths = [safe_input_path(data_path.parent, row["npz"]) for row in modules]
    source_rows = [source_identity(data_path), *[source_identity(path) for path in npz_paths]]
    return modules, contexts, {"npz_paths": dict(zip((str(row["id"]) for row in modules), npz_paths)),
                              "source_identities": source_rows, "metadata": metadata}


def distribution_specs(module: dict) -> list[dict]:
    dist = module["distribution"]
    edges, counts, labels = dist.get("basis_edges", {}), dist.get("counts", {}), dist.get("basis_labels", {})
    required = (
        ("raw", "raw_bf16", None, "Raw BF16 weights", None),
        ("spinquant", "spin_reference", "spin_actual", "Learned SpinQuant basis", "spinquant"),
        ("smooth", "row_reference", "row_actual", "Smooth + H128 basis · Row-scale VQ", "row_scale_vq"),
        ("smooth", "row_reference", "micro_actual", "Smooth + H128 basis · MicroVQ", "micro_vq"),
    )
    result = []
    for basis, reference, actual, title, method in required:
        edge_key = edges.get(basis)
        if not edge_key and basis == "smooth":
            edge_key = edges.get("smooth05_hadamard")
        if not edge_key and basis == "spinquant":
            edge_key = edges.get("spin") or edges.get("learned_spinquant")
        result.append({"basis": basis, "edges_key": edge_key,
                       "reference_key": counts.get(reference),
                       "actual_key": counts.get(actual) if actual else None,
                       "title": labels.get(basis, title) if isinstance(labels, dict) else title,
                       "actual_label": METHOD_LABELS.get(method) if method else None,
                       "method": method})
    return result


def plot_distributions(module: dict, npz, plt):
    specs = distribution_specs(module)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    for ax, spec in zip(axes.flat, specs):
        edges = array_for(npz, spec["edges_key"], "histogram edges", 1)
        reference = array_for(npz, spec["reference_key"], "same-basis FP reference histogram", 1)
        if edges.size != reference.size + 1:
            raise RuntimeError("histogram edges/counts disagree")
        ax.stairs(reference, edges, color="#29384d", linewidth=1.2,
                  label="FP32 reference in this basis")
        if spec["actual_key"]:
            actual = array_for(npz, spec["actual_key"], "quantized-weight histogram", 1)
            if actual.size != reference.size:
                raise RuntimeError("same-basis reference and quantized histogram bins disagree")
            method = spec["method"]
            ax.stairs(actual, edges, color=METHOD_COLORS[method], linewidth=1.05,
                      label=spec["actual_label"])
        else:
            ax.text(0.98, 0.94, "Raw basis", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="#59677a")
        ax.set_title(str(spec["title"]), fontsize=9)
        ax.set_xlabel("Weight value", fontsize=8)
        ax.set_ylabel("Precomputed bin count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#d9e0e8", linewidth=0.4)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{module['id']} · weight distributions · each arm's compared basis", fontsize=13)
    return fig


def plot_weight_surfaces(module: dict, npz, metadata: dict, plt, np):
    windows = module.get("weight_windows", {})
    reference_key = windows.get("reference", module.get("weight_reference_window_key", "weight_reference_window"))
    actual_key = windows.get("actual", module.get("weight_actual_window_key", "weight_actual_window"))
    if not reference_key or not actual_key:
        raise RuntimeError(f"{module['id']} lacks the saved FP32/quantized weight windows for 3D plots")
    references = array_for(npz, reference_key, "basis-matched FP32 weight windows", 3)
    actuals = array_for(npz, actual_key, "quantized weight windows", 3)
    basis_order = metadata.get("basis_lane_order", ["raw", "learned_spinquant", "smooth05_hadamard_row", "smooth05_hadamard_micro"])
    basis_names = [str(name).lower().replace("-", "_") for name in basis_order]
    if references.shape[0] != len(basis_order) or actuals.shape[0] != len(basis_order):
        raise RuntimeError("saved weight-window lanes disagree with metadata.basis_lane_order")

    def basis_index(name: str) -> int:
        aliases = {
            "raw": ("raw",),
            "spinquant": ("spinquant", "learned_spinquant", "spin"),
            "row_scale_vq": ("row_scale_vq", "smooth05_hadamard_row", "smooth_row"),
            "micro_vq": ("micro_vq", "smooth05_hadamard_micro", "smooth_micro"),
        }[name]
        for index, label in enumerate(basis_names):
            if any(alias in label for alias in aliases):
                return index
        raise RuntimeError(f"cannot find {name} in basis_lane_order={basis_order}")
    pairs = [
        ("raw", "raw", None, "Raw BF16 weights"),
        ("spinquant", "spinquant", "spinquant", "Learned SpinQuant basis"),
        ("smooth", "row_scale_vq", "row_scale_vq", "Smooth + H128 · Row-scale VQ"),
        ("smooth", "micro_vq", "micro_vq", "Smooth + H128 · MicroVQ"),
    ]
    arrays = []
    for basis, method, actual_method, title in pairs:
        ref_lane = basis_index(method)
        value = actuals[basis_index(actual_method)] if actual_method else references[ref_lane]
        arrays.append(np.abs(value))
    zmax = max(float(np.max(array)) for array in arrays) or 1.0
    rows, cols = references.shape[1:]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    for panel, (basis, method, actual_method, title) in enumerate(pairs, 1):
        ax = fig.add_subplot(2, 2, panel, projection="3d")
        if actual_method:
            actual = np.abs(actuals[basis_index(actual_method)])
            ax.plot_surface(x, y, actual, rstride=1, cstride=1,
                            color=METHOD_COLORS[canonical_method(actual_method)], alpha=1.0,
                            linewidth=0, antialiased=False, shade=False)
            ax.set_title(f"{title}\neffective quantized weight; FP reference on other pages", fontsize=8, pad=2)
        else:
            ref = np.abs(references[basis_index(method)])
            ax.plot_surface(x, y, ref, rstride=1, cstride=1, color="#d7a0ae", alpha=1.0,
                            linewidth=0, antialiased=False, shade=False)
            ax.set_title(f"{title}\nraw reference", fontsize=8, pad=2)
        ax.set_zlim(0, zmax)
        ax.set_xlabel("Input channel", fontsize=7, labelpad=1)
        ax.set_ylabel("Output channel", fontsize=7, labelpad=1)
        ax.set_zlabel("|W|", fontsize=7, labelpad=1)
        ax.set_xticks(np.linspace(0, cols - 1, 4, dtype=int))
        ax.set_yticks(np.linspace(0, rows - 1, 4, dtype=int))
        ax.tick_params(labelsize=6, pad=0)
        ax.view_init(elev=27, azim=-58)
    shape = f"{rows}×{cols}"
    fig.suptitle(f"{module['id']} · Prism-style |W| surface · full {shape} protocol window · shared height 0–{zmax:.3g}",
                 fontsize=13)
    return fig


def _cv_label(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{float(value):.3g}"


def plot_weight_rms_profiles(module: dict, metadata: dict, npz, plt, np):
    """Show full-matrix weight channel RMS profiles and precomputed CV values."""
    profile_keys = module.get("weight_profile_keys", {})
    arrays = {
        "input_reference": array_for(npz, profile_keys.get("input_rms_reference"),
                                     "full-matrix input-channel reference RMS", 2),
        "input_actual": array_for(npz, profile_keys.get("input_rms_actual"),
                                  "full-matrix input-channel actual RMS", 2),
        "output_reference": array_for(npz, profile_keys.get("output_rms_reference"),
                                       "full-matrix output-channel reference RMS", 2),
        "output_actual": array_for(npz, profile_keys.get("output_rms_actual"),
                                   "full-matrix output-channel actual RMS", 2),
    }
    basis_order = metadata.get("basis_lane_order", module.get("basis_lane_order", []))
    if len(basis_order) != 4 or any(array.shape[0] != 4 for array in arrays.values()):
        raise RuntimeError("weight RMS arrays and basis_lane_order must have four matched lanes")
    if (arrays["input_reference"].shape != arrays["input_actual"].shape
            or arrays["output_reference"].shape != arrays["output_actual"].shape):
        raise RuntimeError("weight reference/actual channel RMS profiles must share axes")

    full_stats = module.get("distribution", {}).get("full_matrix_statistics", {})
    stats_lanes = {
        "raw": ("raw_reference", "raw_reference"),
        "spinquant": ("spinquant_native_w4a4", "spinquant_native_w4a4"),
        "row_scale_vq": ("row_scale_vq_a4", "row_scale_vq_a4"),
        "micro_vq": ("micro_scale_vq_a4", "micro_scale_vq_a4"),
    }
    lane_specs = []
    basis_labels = module.get("basis_labels", {})
    for lane, basis in enumerate(basis_order):
        basis_name = str(basis).lower().replace("-", "_")
        if basis_name in ("bf16", "raw") or "raw" in basis_name:
            method = "raw"
        elif "spin" in basis_name:
            method = "spinquant"
        elif "row" in basis_name:
            method = "row_scale_vq"
        elif "micro" in basis_name:
            method = "micro_vq"
        else:
            raise RuntimeError(f"cannot map weight RMS basis lane {basis!r} to a method")
        if method == "raw":
            stat_names = stats_lanes["raw"]
            label = "Raw BF16"
        else:
            label = (basis_labels.get(basis, METHOD_LABELS.get(method, str(basis)))
                     if isinstance(basis_labels, dict) else METHOD_LABELS.get(method, str(basis)))
            stat_names = stats_lanes[method]
        method_stats = []
        for stat_name, arm in zip(stat_names, ("reference", "actual")):
            root = full_stats.get(stat_name)
            if method != "raw":
                root = root.get(f"{arm}_distribution", {}) if isinstance(root, dict) else {}
            else:
                root = root if isinstance(root, dict) else {}
            method_stats.append(root)
        lane_specs.append({"lane": lane, "method": method, "label": label,
                           "input_ref_cv": method_stats[0].get("input_channel_rms_cv"),
                           "input_actual_cv": method_stats[1].get("input_channel_rms_cv"),
                           "output_ref_cv": method_stats[0].get("output_channel_rms_cv"),
                           "output_actual_cv": method_stats[1].get("output_channel_rms_cv")})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    panels = (
        (axes[0], arrays["input_reference"], arrays["input_actual"], "input", "Input channel"),
        (axes[1], arrays["output_reference"], arrays["output_actual"], "output", "Output channel"),
    )
    for ax, references, actuals, kind, channel_label in panels:
        for spec in lane_specs:
            lane, method = spec["lane"], spec["method"]
            color = "#29384d" if method == "raw" else METHOD_COLORS[method]
            ref_cv, actual_cv = spec[f"{kind}_ref_cv"], spec[f"{kind}_actual_cv"]
            ax.plot(np.arange(references.shape[1]), references[lane], color=color,
                    linewidth=0.8, linestyle="-", alpha=0.9)
            ax.plot(np.arange(actuals.shape[1]), actuals[lane], color=color,
                    linewidth=0.85, linestyle="--", alpha=0.9,
                    label=f"{spec['label']} · CV ref/actual {_cv_label(ref_cv)}/{_cv_label(actual_cv)}")
        ax.set_title(f"Full-matrix {kind} RMS · all {references.shape[1]} channels", fontsize=10)
        ax.set_xlabel(channel_label, fontsize=8)
        ax.set_ylabel("Weight RMS", fontsize=8)
        ax.grid(True, color="#d9e0e8", linewidth=0.4)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.4, frameon=False, loc="best",
                  title="solid: FP reference · dashed: actual", title_fontsize=6.4)
    fig.suptitle(f"{module['id']} · full-matrix weight RMS channel profiles · precomputed CV", fontsize=13)
    return fig


def plot_block_error(module: dict, npz, plt, np):
    block_keys, scale_keys = canonical_map(module["block_nmse"]), canonical_map(module["scale_profile"])
    matrices, scales = {}, {}
    for method in METHODS:
        matrices[method] = array_for(npz, block_keys[method], f"{method} B32 block NMSE", 2)
        scales[method] = array_for(npz, scale_keys[method], f"{method} B32 scale profile")
    if len({matrix.shape for matrix in matrices.values()}) != 1:
        raise RuntimeError(f"B32 NMSE matrices must share block axes within {module['id']}")
    positive = np.concatenate([matrix[matrix > 0] for matrix in matrices.values()])
    vmax = max(float(np.max(matrix)) for matrix in matrices.values()) or 1.0
    vmin = float(np.min(positive)) if positive.size else vmax
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e7ebf0")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.0, 0.86]})
    image = None
    for ax, method in zip(axes[0], METHODS):
        matrix = matrices[method]
        image = ax.imshow(np.ma.masked_less_equal(matrix, 0), origin="lower", aspect="auto",
                          interpolation="nearest", cmap=cmap,
                          norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.001)),
                          rasterized=True)
        row, col = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        ax.scatter([col], [row], marker="x", s=26, linewidths=1, color="cyan")
        ax.set_title(f"{METHOD_LABELS[method]} · hotspot output {row}, input-block {col}", fontsize=8)
        ax.set_xlabel("Input B32 group index", fontsize=7)
        ax.set_ylabel("Output channel", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.colorbar(image, ax=list(axes[0]), shrink=0.84,
                 label="B32 block NMSE · shared log range; gray = zero")
    scale_min = min(float(np.min(a)) for a in scales.values())
    scale_max = max(float(np.max(a)) for a in scales.values())
    for ax, method in zip(axes[1], METHODS):
        values = scales[method]
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim == 1:
            ax.plot(np.arange(values.size), values, color=METHOD_COLORS[method], linewidth=0.8)
            ax.set_xlabel("Output channel", fontsize=7)
            ax.set_ylabel("Scale", fontsize=7)
            if scale_min != scale_max:
                ax.set_ylim(scale_min, scale_max)
        else:
            ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
                      cmap="viridis", vmin=scale_min, vmax=scale_max if scale_max != scale_min else scale_min + 1e-12,
                      rasterized=True)
            ax.set_xlabel("Scale column", fontsize=7)
            ax.set_ylabel("Scale row", fontsize=7)
        ax.set_title(f"{METHOD_LABELS[method]} · full scale profile", fontsize=8)
        if values.ndim == 1:
            ax.grid(True, color="#d9e0e8", linewidth=0.4)
        else:
            ax.grid(False)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{module['id']} · B32 block quantization error and scale profiles · full block grid", fontsize=13)
    return fig


def lane_index(order: list, method: str) -> int:
    for index, lane in enumerate(order):
        if canonical_method(lane) == method:
            return index
    raise RuntimeError(f"method lane {method} not found in {order}")


def plot_rms_profiles(module: dict, context: dict, metadata: dict, npz, plt, np):
    idx = context["index"]
    ctx = module["contexts"][context["id"]]
    bases = metadata.get("basis_lane_order", ["raw", "learned_spinquant", "smooth05_hadamard_row", "smooth05_hadamard_micro"])
    input_all = array_for(npz, context_array_key(module, ctx, "input_basis_rms"),
                          "full-channel input RMS", 3)[idx]
    reference_all = array_for(npz, context_array_key(module, ctx, "output_reference_rms"),
                              "full-channel reference output RMS", 3)[idx]
    actual_all = array_for(npz, context_array_key(module, ctx, "output_actual_rms"),
                           "full-channel actual output RMS", 3)[idx]
    error_all = array_for(npz, context_array_key(module, ctx, "output_error_rms"),
                          "full-channel output-error RMS", 3)[idx]
    method_order = metadata.get("method_lane_order", ["bf16", "spinquant_native_w4a4", "row_scale_vq_a4", "micro_scale_vq_a4"])
    error_order = metadata.get("error_lane_order", list(METHODS))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    basis_labels = metadata.get("basis_labels", {})
    for lane, basis in enumerate(bases):
        label = basis_labels.get(basis, str(basis)) if isinstance(basis_labels, dict) else str(basis)
        axes[0].plot(np.arange(input_all.shape[-1]), input_all[lane], linewidth=0.75, label=label)
    axes[0].set_title(f"Input RMS by basis · all {input_all.shape[-1]} channels", fontsize=9)
    axes[0].set_xlabel("Input channel", fontsize=8)
    axes[0].set_ylabel("RMS", fontsize=8)
    axes[0].legend(ncol=4, fontsize=6, frameon=False)
    for lane, raw_method in enumerate(method_order):
        method = canonical_method(raw_method)
        label = "BF16" if method in ("bf16", "bf16_reference") else METHOD_LABELS.get(method, str(raw_method))
        color = "#27364a" if method in ("bf16", "bf16_reference") else METHOD_COLORS.get(method, "#7d8795")
        axes[1].plot(np.arange(reference_all.shape[-1]), reference_all[lane], color=color,
                     linewidth=0.65, alpha=0.48, linestyle="--", label=f"{label} reference")
        axes[1].plot(np.arange(actual_all.shape[-1]), actual_all[lane], color=color,
                     linewidth=0.9, label=f"{label} actual")
    axes[1].set_title(f"Output RMS by method · all {actual_all.shape[-1]} channels", fontsize=9)
    axes[1].set_xlabel("Output channel", fontsize=8)
    axes[1].set_ylabel("RMS", fontsize=8)
    axes[1].legend(ncol=4, fontsize=6, frameon=False)
    for lane, raw_method in enumerate(error_order):
        method = canonical_method(raw_method)
        axes[2].plot(np.arange(error_all.shape[-1]), error_all[lane], linewidth=0.8,
                     color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axes[2].set_title("Fixed-input output-error RMS · full output width", fontsize=9)
    axes[2].set_xlabel("Output channel", fontsize=8)
    axes[2].set_ylabel("Error RMS", fontsize=8)
    axes[2].legend(ncol=3, fontsize=7, frameon=False)
    for ax in axes:
        ax.grid(True, color="#d9e0e8", linewidth=0.4)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{module['id']} · full-channel input/output RMS · {context['id']}", fontsize=13)
    return fig


def plot_output_error(module: dict, context: dict, metadata: dict, npz, plt, np):
    idx = context["index"]
    ctx = module["contexts"][context["id"]]
    errors = array_for(npz, context_array_key(module, ctx, "output_error_window"),
                       "fixed-input output-error window", 4)[idx]
    lane_order = metadata.get("error_lane_order", list(METHODS))
    selected = {canonical_method(lane): errors[index] for index, lane in enumerate(lane_order)}
    if not all(method in selected for method in METHODS):
        raise RuntimeError("output-error window misses one of the three comparison arms")
    shapes = {selected[method].shape for method in METHODS}
    if len(shapes) != 1:
        raise RuntimeError("output-error panels must share true token and output-channel axes")
    vmax = max(float(np.max(np.abs(selected[m]))) for m in METHODS) or 1.0
    window = metadata.get("windows", {}).get("output", {})
    token_start, channel_start = int(window.get("token_start", 0)), int(window.get("channel_start", 0))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        image = ax.imshow(selected[method], origin="lower", aspect="auto", interpolation="nearest",
                          cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set_title(f"{METHOD_LABELS[method]} · signed Δ output", fontsize=8)
        ax.set_xlabel(f"Output channel · window starts at {channel_start}", fontsize=7)
        ax.set_ylabel(f"Token row · window starts at {token_start}", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.colorbar(image, ax=list(axes), shrink=0.84,
                 label="Signed fixed-input output error · shared symmetric range")
    fig.suptitle(f"{module['id']} · {context['id']} · {selected['spinquant'].shape[0]}×{selected['spinquant'].shape[1]} window",
                 fontsize=12)
    return fig


def module_weight_nmse(module: dict, method: str) -> float:
    return stat_value(canonical_map(module["weight_metrics"])[method],
                      f"{module['id']} {method} weight NMSE")


def output_nmse(module: dict, context: dict, method: str) -> float:
    metrics = canonical_map(context.get("output_nmse", {}))
    if method not in metrics:
        raise RuntimeError(f"{module['id']} {context['id']} lacks {method} output NMSE")
    return stat_value(metrics[method], f"{module['id']} {context['id']} {method} output NMSE")


def improvement_vs_row(module: dict, context: dict, method: str, is_weight: bool) -> float:
    if is_weight:
        stats = canonical_map(module["weight_metrics"])
        value = stats.get("micro_vs_row_improvement_pct")
        if value is None:
            raise RuntimeError(f"{module['id']} lacks precomputed MicroVQ/Row-scale weight improvement")
        return stat_value(value, "weight micro_vs_row_improvement_pct")
    else:
        direct = context.get("relative_improvement_vs_row", {})
        value = canonical_map(direct).get(method)
        if value is None:
            raise RuntimeError(f"{module['id']} {context['id']} lacks precomputed MicroVQ/Row-scale improvement")
        return stat_value(value, "relative_improvement_vs_row")


def module_context_table(data: dict, module: dict) -> list[dict]:
    return contexts_by_module(data, module)


def plot_output_nmse(modules: list, data: dict, plt, np):
    contexts = [module_context_table(data, module) for module in modules]
    values = {method: np.asarray([[output_nmse(module, row, method) for row in rows]
                                  for module, rows in zip(modules, contexts)], dtype=float)
              for method in METHODS}
    positive = np.concatenate([value[value > 0] for value in values.values()])
    vmin = float(np.min(positive)) if positive.size else 1e-12
    vmax = max(float(np.max(value)) for value in values.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.2), constrained_layout=True)
    labels = [str(module["id"]) for module in modules]
    context_labels = [f"case {row['case_id']}\nstep {row['step']}"
                      for row in contexts[0]]
    image = None
    for ax, method in zip(axes, METHODS):
        image = ax.imshow(values[method], aspect="auto", interpolation="nearest", cmap="viridis",
                          norm=plt.matplotlib.colors.LogNorm(vmin=max(vmin, 1e-12),
                                                             vmax=max(vmax, vmin * 1.001)))
        ax.set_title(METHOD_LABELS[method], fontsize=9)
        ax.set_xticks(np.arange(6), context_labels, fontsize=6)
        ax.set_yticks(np.arange(16), labels, fontsize=5)
        ax.set_xlabel("Case / stage / step", fontsize=7)
        ax.tick_params(axis="y", length=0)
    fig.colorbar(image, ax=list(axes), shrink=0.8, label="Fixed-input output NMSE · shared log range")
    fig.suptitle("Output NMSE · all 16 modules × 6 fixed inputs", fontsize=13)
    return fig


def plot_weight_nmse(modules: list, plt, np):
    values = np.asarray([[module_weight_nmse(module, method) for method in METHODS]
                         for module in modules], dtype=float)
    positive = values[values > 0]
    low = float(np.min(positive)) if positive.size else 1e-12
    high = float(np.max(values))
    fig, ax = plt.subplots(figsize=(10, 7.2), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis",
                      norm=plt.matplotlib.colors.LogNorm(vmin=max(low, 1e-12), vmax=max(high, low * 1.001)))
    ax.set_xticks(np.arange(3), [METHOD_LABELS[method] for method in METHODS], fontsize=8)
    ax.set_yticks(np.arange(len(modules)), [str(module["id"]) for module in modules], fontsize=6)
    ax.set_xlabel("Compared arm", fontsize=8)
    ax.set_title("Weight NMSE · complete Linear matrices · each arm's own basis", fontsize=11)
    fig.colorbar(image, ax=ax, label="Weight NMSE · shared log range")
    return fig


def plot_nmse_scatter(modules: list, data: dict, plt, np):
    markers = ("o", "s", "^", "D", "v", "P")
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for method in METHODS:
        for ci, context in enumerate(module_context_table(data, modules[0])):
            xs, ys = [], []
            for module in modules:
                pair_context = next(row for row in module_context_table(data, module)
                                    if row["case_id"] == context["case_id"] and row["step"] == context["step"])
                xs.append(module_weight_nmse(module, method))
                ys.append(output_nmse(module, pair_context, method))
            ax.scatter(xs, ys, marker=markers[ci], s=24, alpha=0.65,
                       color=METHOD_COLORS[method],
                       label=f"{METHOD_LABELS[method]} · case {context['case_id']} / step {context['step']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Weight NMSE · method's compared basis", fontsize=9)
    ax.set_ylabel("Fixed-input output NMSE", fontsize=9)
    ax.set_title("Weight error versus output error · all modules and contexts", fontsize=11)
    ax.grid(True, which="both", color="#d9e0e8", linewidth=0.4)
    ax.legend(fontsize=6, ncol=3, frameon=False)
    return fig


def plot_micro_vs_row(modules: list, data: dict, plt, np):
    output_rows = [module_context_table(data, module) for module in modules]
    output = np.asarray([[improvement_vs_row(module, row, "micro_vq", False)
                          for row in contexts] for module, contexts in zip(modules, output_rows)], dtype=float)
    weight = np.asarray([improvement_vs_row(module, {}, "micro_vq", True) for module in modules], dtype=float)
    limit = max(float(np.max(np.abs(output))), float(np.max(np.abs(weight)))) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.65, 1.35]})
    module_labels = [str(module["id"]) for module in modules]
    image = axes[0].imshow(weight[:, None], aspect="auto", interpolation="nearest", cmap="coolwarm",
                           vmin=-limit, vmax=limit)
    axes[0].set_yticks(np.arange(16), module_labels, fontsize=6)
    axes[0].set_xticks([0], ["Complete weight\nmatrix"], fontsize=7)
    axes[0].set_title("Weight NMSE improvement", fontsize=9)
    contexts = output_rows[0]
    context_labels = [f"case {row['case_id']}\nstep {row['step']}" for row in contexts]
    axes[1].imshow(output, aspect="auto", interpolation="nearest", cmap="coolwarm",
                   vmin=-limit, vmax=limit)
    axes[1].set_yticks(np.arange(16), module_labels, fontsize=6)
    axes[1].set_xticks(np.arange(6), context_labels, fontsize=6)
    axes[1].set_title("Fixed-input output NMSE improvement", fontsize=9)
    axes[1].set_xlabel("Case / stage / step", fontsize=8)
    fig.colorbar(image, ax=list(axes), shrink=0.82,
                 label="MicroVQ reduction versus Row-scale VQ (%) · positive favors MicroVQ")
    fig.suptitle("Where MicroVQ lowers error relative to Row-scale VQ", fontsize=13)
    return fig


def save_preview(fig, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    for dpi in (80, 64, 48, 36, 28, 20, 14, 10):
        fig.savefig(path, format="jpeg", dpi=dpi, bbox_inches="tight",
                    pil_kwargs={"quality": 78, "optimize": True})
        size = path.stat().st_size
        if size <= MAX_PREVIEW_BYTES:
            return size
        path.unlink(missing_ok=True)
    raise RuntimeError(f"could not fit JPEG preview under {MAX_PREVIEW_BYTES} bytes")


def save_figure(fig, plot_dir: Path, record: dict) -> dict:
    from matplotlib.backends.backend_pdf import PdfPages

    figures = fig if isinstance(fig, list) else [fig]
    stem = record["stem"]
    pdf = plot_dir / f"{stem}.pdf"
    page_descriptions = (
        "weight histogram by compared basis",
        "Prism-style |W| 3D surface window",
        "full-matrix weight RMS profiles with precomputed channel CV",
    )
    pages = []
    with PdfPages(pdf) as bundle:
        for index, page_fig in enumerate(figures, 1):
            png = plot_dir / f"{stem}__page{index}.png"
            preview = plot_dir / "previews" / f"{stem}__page{index}.jpg"
            page_fig.savefig(png, dpi=170, bbox_inches="tight")
            bundle.savefig(page_fig, dpi=150, bbox_inches="tight")
            preview_size = save_preview(page_fig, preview)
            pages.append({"page": index,
                          "description": page_descriptions[index - 1] if record["family"] == "distribution" else record["title"],
                          "png": png.relative_to(plot_dir).as_posix(),
                          "preview": preview.relative_to(plot_dir).as_posix(),
                          "preview_bytes": preview_size})
    return {"pdf": pdf.relative_to(plot_dir).as_posix(), "pages": pages,
            "png": pages[0]["png"], "preview": pages[0]["preview"],
            "preview_bytes": pages[0]["preview_bytes"]}


def focus_modules(modules: list) -> list:
    selected = [row for row in modules if row.get("focus") is True]
    if not selected:
        selected = [row for row in modules if int(row.get("block_index", -1)) == 10
                    and str(row.get("suffix", "")).endswith("ffn.2")]
    if len(selected) != 2:
        raise RuntimeError("expected two FFN.2 / block 10 focus modules")
    return selected


def figure_records(modules: list, data: dict, contexts: list) -> list[dict]:
    records = []
    for module in modules:
        contexts_local = contexts_by_module(data, module)
        main = next((row for row in contexts_local if row["case_id"] == 1004 and row["step"] == 4), None)
        if main is None:
            raise RuntimeError(f"{module['id']} lacks main case 1004 / step 4")
        for family in ("distribution", "block_error", "input_output_rms", "output_error"):
            records.append({"id": f"{family}:{slug(module['id'])}", "family": family,
                            "module_id": str(module["id"]), "stream": module.get("stream"),
                            "block_index": module.get("block_index"), "suffix": module.get("suffix"),
                            "context_id": main["id"], "title": f"{module['id']} · {family}",
                            "context_index": main["index"]})
    for module in focus_modules(modules):
        for context in contexts_by_module(data, module):
            records.append({"id": f"focus_output_error:{slug(module['id'])}:{slug(context['id'])}",
                            "family": "focus_output_error", "module_id": str(module["id"]),
                            "stream": module.get("stream"), "block_index": module.get("block_index"),
                            "suffix": module.get("suffix"), "context_id": context["id"],
                            "context_index": context["index"],
                            "title": f"{module['id']} · focus · {context['id']}"})
    for key, title in (
        ("aggregate_output_nmse", "Output NMSE · all modules and contexts"),
        ("aggregate_weight_nmse", "Weight NMSE · all modules"),
        ("aggregate_weight_vs_output", "Weight NMSE versus output NMSE"),
        ("aggregate_micro_vs_row", "MicroVQ relative to Row-scale VQ"),
    ):
        records.append({"id": key, "family": "aggregate", "title": title})
    return records


def render_record(record: dict, module: dict | None, data: dict, metadata: dict, npz, plt, np):
    family = record["family"]
    if family == "aggregate":
        modules = data["modules"]
        return {
            "aggregate_output_nmse": lambda: plot_output_nmse(modules, data, plt, np),
            "aggregate_weight_nmse": lambda: plot_weight_nmse(modules, plt, np),
            "aggregate_weight_vs_output": lambda: plot_nmse_scatter(modules, data, plt, np),
            "aggregate_micro_vs_row": lambda: plot_micro_vs_row(modules, data, plt, np),
        }[record["id"]]()
    context = module["contexts"][record["context_id"]]
    if family == "distribution":
        return [plot_distributions(module, npz, plt),
                plot_weight_surfaces(module, npz, metadata, plt, np),
                plot_weight_rms_profiles(module, metadata, npz, plt, np)]
    if family == "block_error":
        return plot_block_error(module, npz, plt, np)
    if family == "input_output_rms":
        merged = {**data["contexts_by_id"].get(record["context_id"], {}), **context,
                  "id": record["context_id"], "index": record["context_index"]}
        return plot_rms_profiles(module, merged, metadata, npz, plt, np)
    if family in ("output_error", "focus_output_error"):
        merged = {**data["contexts_by_id"].get(record["context_id"], {}), **context,
                  "id": record["context_id"], "index": record["context_index"]}
        return plot_output_error(module, merged, metadata, npz, plt, np)
    raise RuntimeError(f"unknown plot family: {family}")


def output_manifest(plot_dir: Path, identity: dict, records: list, status: str) -> dict:
    figures = records
    counts = {family: sum(row.get("family") == family for row in figures)
              for family in ("distribution", "block_error", "input_output_rms", "output_error",
                             "focus_output_error", "aggregate")}
    expected = {"distribution": 16, "block_error": 16, "input_output_rms": 16,
                "output_error": 16, "focus_output_error": 12, "aggregate": 4}
    families = {key: {"complete": counts[key], "expected": expected[key]} for key in expected}
    ids = {row.get("id") for row in figures}
    expected_ids = {row["id"] for row in identity["figure_specs"]}
    page_expectations = {"distribution": 3}
    expected_pages = sum(page_expectations.get(row["family"], 1) for row in identity["figure_specs"])
    actual_pages = sum(len(row.get("pages", [])) for row in figures)
    family_pages = {}
    for family, expected_count in expected.items():
        actual_count = sum(len(row.get("pages", [])) for row in figures if row.get("family") == family)
        family_pages[family] = {
            "complete": actual_count,
            "expected": expected_count * page_expectations.get(family, 1),
        }
    page_complete = all(
        len(row.get("pages", [])) == page_expectations.get(row["family"], 1)
        for row in figures
    )
    complete = (status == "complete" and ids == expected_ids and counts == expected
                and page_complete and actual_pages == expected_pages)
    return {
        "schema": RENDERER_SCHEMA, "status": "complete" if complete else "checkpointed",
        "identity": identity["identity"], "source_identities": identity["source_identities"],
        "module_count": 16, "expected_module_count": 16,
        "focus_count": 2, "aggregate_count": 4,
        "figure_count": len(figures), "expected_figure_count": len(identity["figure_specs"]),
        "families": families, "coverage": counts,
        "page_count": actual_pages, "expected_page_count": expected_pages,
        "family_pages": family_pages, "figures": figures,
        "preview_max_bytes": MAX_PREVIEW_BYTES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="frozen weight_analysis protocol")
    parser.add_argument("--out", type=Path, required=True, help="results/full/weight_analysis root")
    parser.add_argument("--data", type=Path, help="computed JSON; defaults inside --out")
    parser.add_argument("--selfcheck", action="store_true", help="run pure helper assertions")
    args = parser.parse_args()
    if args.selfcheck:
        assert canonical_method("micro_scale_vq_a4") == "micro_vq"
        assert slug("video_expert.blocks.10.ffn.2") == "video_expert.blocks.10.ffn.2"
        print("weight analysis renderer helper selfcheck passed")
        return 0

    require_compute_allocation()
    out_root = args.out.resolve()
    data_path = (args.data or (out_root / "weight_analysis_data.json")).resolve()
    protocol_path = args.protocol.resolve()
    data, protocol = read_json(data_path), read_json(protocol_path)
    protocol["_path"] = str(protocol_path)
    modules, contexts, normalized = validate_input(data, protocol, data_path)
    metadata = normalized["metadata"]
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    context_by_id = {str(row["id"]): row for row in contexts if isinstance(row, dict)}
    data["contexts_by_id"] = context_by_id
    records = figure_records(modules, data, contexts)
    protocol_identity = dict(protocol)
    protocol_identity.pop("_path", None)
    identity_only = {"renderer_schema": RENDERER_SCHEMA, "protocol": protocol_identity,
                     "data_schema": data["schema"], "main_case": protocol["focus_case"],
                     "main_step": protocol["focus_step"]}
    identity = {"identity": identity_only, "source_identities": [protocol_source_identity(protocol_path),
                *normalized["source_identities"]]}
    manifest_path = plot_dir / "manifest.json"
    old = read_json(manifest_path) if manifest_path.is_file() else {}
    old_rows = {row.get("id"): row for row in old.get("figures", []) if isinstance(row, dict)}
    if (normalize_identity(old.get("identity")) != normalize_identity(identity_only)
            or normalize_source_identities(old.get("source_identities"))
            != normalize_source_identities(identity["source_identities"])):
        old_rows = {}
    specs = [{key: value for key, value in row.items() if key != "context_index"} for row in records]
    identity["figure_specs"] = specs
    figure_module = {str(row["id"]): row for row in modules}
    start_text = os.environ.get("FW_JOB_START_EPOCH")
    try:
        start_epoch = float(start_text) if start_text else time.time()
        soft_seconds = int(protocol.get("soft_deadline_seconds", 6600))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PBS start epoch and soft deadline must be numeric") from exc
    if not math.isfinite(start_epoch) or not 0 < soft_seconds <= 6600:
        raise RuntimeError("renderer deadline must be finite, positive, and no longer than 6600 seconds")
    deadline = start_epoch + soft_seconds

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    done = []
    for number, record in enumerate(records, 1):
        record["render_version"] = 2 if record["family"] in ("distribution", "block_error") else 1
        record["stem"] = slug(record["id"])
        previous = old_rows.get(record["id"], {})
        previous_pages = previous.get("pages", []) if isinstance(previous, dict) else []
        previous_files = [previous.get("pdf")] + [page.get(field) for page in previous_pages
                                                   for field in ("png", "preview")]
        required_pages = 3 if record["family"] == "distribution" else 1
        if previous and previous.get("render_version", 1) == record["render_version"] and len(previous_pages) == required_pages and all(value and (plot_dir / value).is_file()
                                                for value in previous_files):
            done.append(previous)
            continue
        if time.time() >= deadline:
            atomic_json(manifest_path, output_manifest(plot_dir, identity, done, "checkpointed"))
            print(f"checkpointed after {len(done)}/{len(records)} figures")
            return 0
        module = figure_module.get(record.get("module_id"))
        if module is None:
            fig = render_record(record, None, data, metadata, None, plt, np)
        else:
            npz_path = normalized["npz_paths"][str(module["id"])]
            with np.load(npz_path, allow_pickle=False) as npz:
                fig = render_record(record, module, data, metadata, npz, plt, np)
        try:
            files = save_figure(fig, plot_dir, record)
        finally:
            for page_fig in (fig if isinstance(fig, list) else [fig]):
                plt.close(page_fig)
        done.append({**record, **files})
        atomic_json(manifest_path, output_manifest(plot_dir, identity, done, "checkpointed"))
        print(f"rendered {number}/{len(records)}: {record['id']}")

    final = output_manifest(plot_dir, identity, done, "complete")
    if final["status"] != "complete":
        raise RuntimeError(f"plot bundle coverage is incomplete: {final['families']}")
    atomic_json(manifest_path, final)
    print(f"complete: {final['figure_count']} figures; all previews <= {MAX_PREVIEW_BYTES} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

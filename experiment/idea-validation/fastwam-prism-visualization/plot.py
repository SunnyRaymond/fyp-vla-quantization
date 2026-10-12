#!/usr/bin/env python3
"""Render the frozen Fast-WAM Prism-style activation and weight captures."""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from urllib.parse import quote


MAX_PREVIEW_BYTES = 220 * 1024
PRIMARY_RULE = "lowest call_index for (case_id,seed_index,arm,module,stage,step)"
STAGE_BY_STREAM = {"video_expert": "video", "action_expert": "action"}
RENDERER_SCHEMA = "fastwam-prism-renderer-v2"


def window_shape(shape: tuple[int, ...], row_start: int, row_limit: int,
                 col_start: int, col_limit: int) -> tuple[int, int]:
    if len(shape) != 2 or min(row_start, col_start, row_limit, col_limit) < 0:
        raise ValueError(f"expected a non-negative 2D window, got shape={shape}")
    return (
        max(0, min(shape[0], row_start + row_limit) - row_start),
        max(0, min(shape[1], col_start + col_limit) - col_start),
    )


def selfcheck() -> None:
    assert window_shape((128, 4096), 0, 64, 0, 512) == (64, 512)
    assert window_shape((32, 300), 0, 64, 0, 512) == (32, 300)
    assert window_shape((128, 4096), 10, 64, 20, 512) == (64, 512)
    assert window_shape((4, 8), 4, 64, 8, 512) == (0, 0)
    try:
        window_shape((4, 8, 2), 0, 64, 0, 512)
    except ValueError:
        pass
    else:
        raise AssertionError("non-matrix input must be rejected")
    print("plot.py window helper selfcheck passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, help="frozen protocol JSON")
    parser.add_argument("--out", type=Path, help="capture result root containing raw/manifest.json")
    parser.add_argument("--selfcheck", action="store_true", help="run pure window-boundary checks")
    return parser.parse_args()


def require_compute_allocation() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("rendering requires an active PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("rendering requires a valid PBS_NODEFILE")
    host = socket.gethostname().lower()
    if "login" in host:
        raise RuntimeError(f"refusing to render on login node {host!r}")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def int_value(value, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be an integer, got {value!r}") from exc


def classify(row: dict) -> str:
    kind = str(row.get("kind", "")).lower()
    if "weight" in kind:
        return "weight"
    if "paired" in kind or row.get("transform") is not None:
        return "paired"
    if "activation" in kind:
        return "actual"
    raise RuntimeError(f"unrecognized record kind/fields: {row.get('record_id')!r}, {kind!r}")


def module_suffix(module: str, suffixes: list[str]) -> str | None:
    value = str(module)
    for suffix in suffixes:
        if value == suffix or value.endswith("." + suffix) or value.endswith(suffix):
            return suffix
    return None


def safe_record_path(out_root: Path, raw_root: Path, value: str, label: str) -> Path:
    given = Path(str(value).replace("\\", "/"))
    if given.is_absolute():
        raise RuntimeError(f"{label} must be a path relative to the result root: {value!r}")
    roots = (out_root, raw_root)
    for root in roots:
        candidate = (root / given).resolve()
        try:
            candidate.relative_to(out_root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"{label} does not exist under the result root: {value!r}")


def record_path(row: dict, out_root: Path, raw_root: Path) -> Path:
    if not row.get("file"):
        raise RuntimeError(f"record {row.get('record_id')!r} has no file path")
    return safe_record_path(out_root, raw_root, row["file"], "record file")


def resolve_sources(row: dict, out_root: Path, raw_root: Path) -> list[Path]:
    result = [record_path(row, out_root, raw_root)]
    if row.get("transform_file"):
        result.append(safe_record_path(out_root, raw_root, row["transform_file"], "transform_file"))
    return result


def parse_npz_metadata(data, record_id: str) -> dict:
    if "metadata_json" not in data.files:
        raise RuntimeError(f"{record_id}: NPZ is missing metadata_json")
    scalar = data["metadata_json"]
    value = scalar.item() if getattr(scalar, "shape", None) == () else scalar
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise RuntimeError(f"{record_id}: metadata_json must be a NumPy string scalar")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{record_id}: invalid metadata_json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{record_id}: metadata_json must contain an object")
    return parsed


def check_row_metadata(row: dict, meta: dict, category: str) -> None:
    row_fields = ["record_id", "module", "module_index", "stream", "block_index"]
    meta_fields = ["module", "module_index", "stream", "block_index"]
    if category in ("actual", "paired"):
        row_fields.extend(("case_id", "seed_index", "stage", "step", "call_index", "shape"))
        meta_fields.extend(
            ("record_id", "case_id", "seed_index", "stage", "step", "call_index",
             "input_shape", "output_shape", "token_rows", "batch_index", "token_order")
        )
        if category == "actual":
            row_fields.extend(("input_shape", "output_shape"))
            meta_fields.extend((
                "arm", "input_width", "output_width", "output_window_channels",
                "token_count", "a4_applied",
            ))
        else:
            meta_fields.append("shape")
    else:
        row_fields.extend(("arm", "stage", "step", "call_index", "shape", "weight_shape", "axes"))
        meta_fields.extend(("record_id", "arm", "stage", "step", "call_index", "weight_shape", "window_shape", "axes", "basis"))
    missing = [field for field in meta_fields if field not in meta]
    if missing:
        raise RuntimeError(
            f"{row.get('record_id')!r}: NPZ metadata_json lacks required fields {missing}"
        )
    for field in row_fields:
        if field not in row:
            raise RuntimeError(f"{row.get('record_id')!r}: manifest lacks {field}")
    if meta.get("record_id") != row.get("record_id"):
        raise RuntimeError(f"{row.get('record_id')!r}: NPZ record_id differs from manifest")
    if category in ("actual", "paired"):
        nested = row.get("metadata")
        if not isinstance(nested, dict):
            raise RuntimeError(f"{row.get('record_id')!r}: manifest lacks nested capture metadata")
        for field in ("batch_index", "token_order"):
            if nested.get(field) != meta[field]:
                raise RuntimeError(
                    f"{row.get('record_id')!r}: manifest/NPZ metadata disagree for {field}"
                )
    for field in (
        "module", "module_index", "block_index", "stream", "arm", "stage", "step",
        "case_id", "seed_index", "call_index", "input_shape", "output_shape",
        "shape", "weight_shape", "axes", "basis",
    ):
        if field in row and field in meta and row[field] != meta[field]:
            raise RuntimeError(
                f"{row.get('record_id')!r}: manifest/NPZ metadata disagree for {field}"
            )
    if category == "weight":
        if row.get("transform") is not None and row.get("transform") != meta["basis"]:
            raise RuntimeError(
                f"{row.get('record_id')!r}: manifest weight basis and NPZ basis disagree"
            )
        axes = meta["axes"]
        axis_text = " ".join(str(value) for value in axes).lower()
        if "out" not in axis_text or "in" not in axis_text:
            raise RuntimeError(
                f"{row.get('record_id')!r}: weight axes must identify output and input channels"
            )


def select_primary(groups: dict, expected_keys: list[tuple], label: str) -> tuple[dict, list[dict]]:
    selected = {}
    extras = []
    missing = []
    ambiguous = []
    for key in expected_keys:
        rows = groups.get(key, [])
        if not rows:
            missing.append(key)
            continue
        indexed = []
        for row in rows:
            if row.get("call_index") is None:
                raise RuntimeError(f"{label} {row.get('record_id')!r} has null call_index")
            indexed.append((int_value(row["call_index"], "call_index"), row))
        indexed.sort(key=lambda pair: (pair[0], str(pair[1].get("record_id", ""))))
        if len(indexed) > 1 and indexed[0][0] == indexed[1][0]:
            ambiguous.append(key)
            continue
        selected[key] = indexed[0][1]
        extras.extend(row for _, row in indexed[1:])
    if missing or ambiguous:
        reason = []
        if missing:
            reason.append(f"{len(missing)} missing")
        if ambiguous:
            reason.append(f"{len(ambiguous)} tied lowest-call groups")
        sample = missing[:5] + ambiguous[:5]
        raise RuntimeError(
            f"{label} coverage failed ({', '.join(reason)}; example keys: {sample})"
        )
    return selected, extras


def preflight(protocol: dict, manifest: dict, records: list[dict],
              out_root: Path, raw_root: Path) -> tuple[dict, dict, dict, list[dict]]:
    arms = list(protocol["arms"])
    transforms = list(protocol["paired_transforms"])
    streams = list(protocol["streams"])
    suffixes = list(protocol["module_suffixes"])
    cases = [int_value(v, "protocol case") for v in protocol["cases"]]
    steps = [int_value(v, "protocol snapshot step") for v in protocol["snapshot_steps"]]
    blocks = [int_value(v, "protocol block") for v in protocol["blocks"]]
    seed = int_value(protocol["seed_index"], "protocol seed_index")

    declared = manifest.get("protocol")
    declared_name = declared.get("protocol") if isinstance(declared, dict) else declared
    if declared_name != protocol.get("protocol"):
        raise RuntimeError(f"manifest protocol {declared_name!r} does not match frozen protocol")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"raw manifest status must be 'complete', got {manifest.get('status')!r}")
    if manifest.get("batch_index") != protocol.get("batch_index") or manifest.get("token_order") != protocol.get("token_order"):
        raise RuntimeError("manifest batch/token-axis selection differs from the frozen protocol")
    for field, expected in (
        ("cases", set(cases)), ("arms", set(arms)), ("paired_transforms", set(transforms)),
        ("blocks", set(blocks)), ("streams", set(streams)),
        ("module_suffixes", set(suffixes)), ("snapshot_steps", set(steps)),
    ):
        if set(str(v) for v in manifest.get(field, [])) != set(str(v) for v in expected):
            raise RuntimeError(f"manifest {field} do not match the frozen protocol")
    if set(str(v) for v in manifest.get("seed_indices", [])) != {str(seed)}:
        raise RuntimeError("manifest seed_indices do not match the frozen seed")
    if not str(manifest.get("primary_activation_record_rule", "")).startswith(PRIMARY_RULE):
        raise RuntimeError("manifest does not declare the required lowest-call primary-record rule")

    classified = {"actual": [], "paired": [], "weight": []}
    existing_paths = {}
    seen_ids = set()
    for row in records:
        if not isinstance(row, dict):
            raise RuntimeError("manifest records must be JSON objects")
        record_id = row.get("record_id")
        if not record_id or str(record_id) in seen_ids:
            raise RuntimeError(f"manifest record_id is empty or duplicated: {record_id!r}")
        seen_ids.add(str(record_id))
        category = classify(row)
        classified[category].append(row)
        paths = resolve_sources(row, out_root, raw_root)
        existing_paths[str(row.get("record_id"))] = paths

    actual_groups: dict[tuple, list[dict]] = {}
    paired_groups: dict[tuple, list[dict]] = {}
    weight_groups: dict[tuple, list[dict]] = {}
    for row in classified["actual"]:
        suffix = module_suffix(row.get("module", ""), suffixes)
        if suffix is None or row.get("stream") not in streams:
            continue
        stage = str(row.get("stage"))
        if stage != STAGE_BY_STREAM.get(row.get("stream")):
            continue
        if row.get("arm") not in arms:
            raise RuntimeError(f"unexpected actual arm {row.get('arm')!r}")
        key = (
            int_value(row.get("case_id"), "case_id"), int_value(row.get("seed_index"), "seed_index"),
            str(row["arm"]), str(row["stream"]), suffix, int_value(row.get("block_index"), "block_index"),
            stage, int_value(row.get("step"), "step"),
        )
        actual_groups.setdefault(key, []).append(row)
    for row in classified["paired"]:
        suffix = module_suffix(row.get("module", ""), suffixes)
        if suffix is None or row.get("stream") not in streams:
            continue
        stage = str(row.get("stage"))
        if stage != STAGE_BY_STREAM.get(row.get("stream")):
            continue
        transform_value = row.get("transform")
        if transform_value in transforms:
            row_transforms = [str(transform_value)]
        elif transform_value is None or transform_value in ("paired_local", "all"):
            row_transforms = transforms
        elif isinstance(transform_value, list) and set(transform_value) == set(transforms):
            row_transforms = transforms
        else:
            raise RuntimeError(f"unexpected paired transform declaration {transform_value!r}")
        for transform in row_transforms:
            key = (
                int_value(row.get("case_id"), "case_id"), int_value(row.get("seed_index"), "seed_index"),
                transform, str(row["stream"]), suffix,
                int_value(row.get("block_index"), "block_index"), stage, int_value(row.get("step"), "step"),
            )
            paired_groups.setdefault(key, []).append(row)
    for row in classified["weight"]:
        suffix = module_suffix(row.get("module", ""), suffixes)
        if suffix is None or row.get("stream") not in streams:
            continue
        if row.get("arm") not in arms:
            raise RuntimeError(f"unexpected weight arm {row.get('arm')!r}")
        key = (
            str(row["arm"]), str(row["stream"]), suffix,
            int_value(row.get("block_index"), "block_index"),
        )
        weight_groups.setdefault(key, []).append(row)

    expected_actual = [
        (case, seed, arm, stream, suffix, block, STAGE_BY_STREAM[stream], step)
        for case in cases for arm in arms for stream in streams for suffix in suffixes
        for block in blocks for step in steps
    ]
    expected_paired = [
        (case, seed, transform, stream, suffix, block, STAGE_BY_STREAM[stream], step)
        for case in cases for transform in transforms for stream in streams for suffix in suffixes
        for block in blocks for step in steps
    ]
    expected_weights = [
        (arm, stream, suffix, block)
        for arm in arms for stream in streams for suffix in suffixes for block in blocks
    ]
    actual_primary, actual_extras = select_primary(actual_groups, expected_actual, "actual activation")
    paired_primary, paired_extras = select_primary(paired_groups, expected_paired, "paired-local activation")
    for case in cases:
        for stream in streams:
            for suffix in suffixes:
                for block in blocks:
                    for step in steps:
                        key = (case, seed, "bf16", stream, suffix, block,
                               STAGE_BY_STREAM[stream], step)
                        bf16_call = int_value(actual_primary[key]["call_index"], "BF16 call_index")
                        for transform in transforms:
                            paired_key = (
                                case, seed, transform, stream, suffix, block,
                                STAGE_BY_STREAM[stream], step,
                            )
                            paired_call = int_value(
                                paired_primary[paired_key]["call_index"], "paired call_index"
                            )
                            if paired_call != bf16_call:
                                raise RuntimeError(
                                    f"paired-local {paired_key} call_index={paired_call} "
                                    f"does not match BF16 call_index={bf16_call}"
                                )

    weights = {}
    missing_weights = []
    duplicate_weights = []
    for key in expected_weights:
        rows = weight_groups.get(key, [])
        if not rows:
            missing_weights.append(key)
        elif len(rows) != 1:
            duplicate_weights.append(key)
        else:
            weights[key] = rows[0]
    if missing_weights or duplicate_weights:
        raise RuntimeError(
            f"weight coverage failed: {len(missing_weights)} missing, "
            f"{len(duplicate_weights)} duplicated; examples: "
            f"{(missing_weights + duplicate_weights)[:8]}"
        )

    main_case = int_value(protocol.get("main_case", 1004), "main_case")
    main_step = int_value(protocol.get("main_step", 4), "main_step")
    if main_case not in cases or main_step not in steps:
        raise RuntimeError("primary case/step are outside the frozen protocol")
    main_actual = {}
    main_paired = {}
    for stream in streams:
        for suffix in suffixes:
            block_actual = {}
            block_paired = {}
            for block in blocks:
                for arm in arms:
                    key = (main_case, seed, arm, stream, suffix, block,
                           STAGE_BY_STREAM[stream], main_step)
                    block_actual[(arm, block)] = actual_primary[key]
                for transform in transforms:
                    key = (main_case, seed, transform, stream, suffix, block,
                           STAGE_BY_STREAM[stream], main_step)
                    block_paired[(transform, block)] = paired_primary[key]
            main_actual[(stream, suffix)] = block_actual
            main_paired[(stream, suffix)] = block_paired

    extras_by_id = {str(row["record_id"]): row for row in actual_extras + paired_extras}
    extras = list(extras_by_id.values())
    return (
        {"actual": main_actual, "paired": main_paired},
        weights,
        existing_paths,
        extras,
    )


def import_plot_libraries():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.cm import ScalarMappable
    except ImportError as exc:
        raise RuntimeError(
            "renderer needs the environment's NumPy and Matplotlib; do not install them on a login node"
        ) from exc
    return np, plt, cm, Normalize, PdfPages, ScalarMappable


def numeric_matrix(np, data, key: str, record_id: str):
    if key not in data.files:
        raise RuntimeError(f"{record_id}: NPZ is missing required array {key!r}")
    array = data[key]
    if array.ndim != 2 or array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise RuntimeError(
            f"{record_id}: {key} must be a signed float32 2D array; "
            f"got shape={array.shape}, dtype={array.dtype}"
        )
    if not np.isfinite(array).all():
        raise RuntimeError(f"{record_id}: {key} contains NaN or infinity")
    return array


def validate_record_shapes(np, data, row: dict, metadata: dict, category: str,
                           arrays: dict, protocol: dict) -> None:
    record_id = str(row.get("record_id"))
    if category == "actual":
        required = ("pre_quant", "post_quant", "output", "token_row_ids", "original_flat_row_ids")
        if any(key not in data.files for key in required):
            raise RuntimeError(f"{record_id}: activation NPZ is missing one of {required}")
        input_shape = tuple(int(v) for v in metadata["input_shape"])
        batch_index = int_value(metadata["batch_index"], "batch_index")
        branch_rows = int(np.prod(input_shape[1:-1])) if len(input_shape) >= 3 else (
            int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
        )
        start = int_value(protocol.get("token_start", 0), "token_start")
        end = min(branch_rows, start + int_value(protocol["capture_tokens"], "capture_tokens"))
        rows = end - start
        width = int_value(metadata["input_width"], "input_width")
        if rows <= 0 or metadata["token_rows"] != [start, end] or metadata["token_count"] != rows:
            raise RuntimeError(f"{record_id}: captured token window does not match first-branch protocol")
        if tuple(row["shape"]) != (rows, width):
            raise RuntimeError(f"{record_id}: manifest activation shape is not token_count × full input width")
        for key in ("pre_quant", "post_quant"):
            if arrays[key].shape != (rows, width):
                raise RuntimeError(f"{record_id}: {key} is not the full-channel captured input matrix")
        output_channels = metadata.get("output_window_channels")
        if not isinstance(output_channels, list) or len(output_channels) != 2:
            raise RuntimeError(f"{record_id}: output_window_channels metadata is missing")
        output_width = int_value(output_channels[1], "output_window_channels end") - int_value(
            output_channels[0], "output_window_channels start"
        )
        if arrays["output"].shape != (rows, output_width):
            raise RuntimeError(f"{record_id}: output array shape disagrees with its channel window")
        for key in ("token_row_ids", "original_flat_row_ids"):
            ids = data[key]
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.shape[0] != rows:
                raise RuntimeError(f"{record_id}: {key} must be an integer vector matching the token window")
        expected_local = np.arange(start, end, dtype=data["token_row_ids"].dtype)
        expected_flat = np.arange(
            batch_index * branch_rows + start, batch_index * branch_rows + end,
            dtype=data["original_flat_row_ids"].dtype,
        )
        if not np.array_equal(data["token_row_ids"], expected_local):
            raise RuntimeError(f"{record_id}: token_row_ids are not branch-local row indices")
        if not np.array_equal(data["original_flat_row_ids"], expected_flat):
            raise RuntimeError(f"{record_id}: original_flat_row_ids disagree with batch and shape metadata")
        if bool(metadata["a4_applied"]) != (row.get("arm") != "bf16"):
            raise RuntimeError(f"{record_id}: a4_applied disagrees with the actual arm")
    elif category == "paired":
        required = (
            "paired_raw", "paired_learned_spinquant", "paired_smooth05_hadamard",
            "paired_raw_qdq", "paired_learned_spinquant_qdq", "paired_smooth05_hadamard_qdq",
            "token_row_ids", "original_flat_row_ids",
        )
        if any(key not in data.files for key in required):
            raise RuntimeError(f"{record_id}: paired-local NPZ is missing one of {required}")
        rows, width = (int(v) for v in metadata["shape"])
        if tuple(row["shape"]) != (rows, width):
            raise RuntimeError(f"{record_id}: paired-local manifest and metadata shapes disagree")
        for key in required[:6]:
            if arrays[key].shape != (rows, width):
                raise RuntimeError(f"{record_id}: {key} is not the full-channel paired matrix")
        for key in required[6:]:
            ids = data[key]
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.shape[0] != rows:
                raise RuntimeError(f"{record_id}: {key} must be an integer vector matching the paired window")
        if metadata["batch_index"] != protocol["batch_index"]:
            raise RuntimeError(f"{record_id}: paired-local batch selection differs from the protocol")
        input_shape = tuple(int(v) for v in metadata["input_shape"])
        branch_rows = int(np.prod(input_shape[1:-1])) if len(input_shape) >= 3 else (
            int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
        )
        start = int_value(protocol.get("token_start", 0), "token_start")
        end = min(branch_rows, start + int_value(protocol["capture_tokens"], "capture_tokens"))
        if metadata["token_rows"] != [start, end] or rows != end - start:
            raise RuntimeError(f"{record_id}: paired token window does not match first-branch protocol")
        expected_local = np.arange(start, end, dtype=data["token_row_ids"].dtype)
        expected_flat = np.arange(
            int(metadata["batch_index"]) * branch_rows + start,
            int(metadata["batch_index"]) * branch_rows + end,
            dtype=data["original_flat_row_ids"].dtype,
        )
        if not np.array_equal(data["token_row_ids"], expected_local):
            raise RuntimeError(f"{record_id}: paired token_row_ids are not branch-local indices")
        if not np.array_equal(data["original_flat_row_ids"], expected_flat):
            raise RuntimeError(f"{record_id}: paired original_flat_row_ids disagree with metadata")
    else:
        if any(key not in data.files for key in ("weight", "reference")):
            raise RuntimeError(f"{record_id}: weight NPZ must contain weight and reference")
        expected = tuple(int(v) for v in metadata["window_shape"])
        if arrays["weight"].shape != expected or arrays["reference"].shape != expected:
            raise RuntimeError(f"{record_id}: weight/reference shapes disagree with window metadata")
        if arrays["weight"].shape != tuple(int(v) for v in row["shape"]):
            raise RuntimeError(f"{record_id}: weight window and manifest shape disagree")
        if metadata["axes"] != ["output_channel", "input_channel"]:
            raise RuntimeError(f"{record_id}: weight axes must be [output_channel, input_channel]")


def window_array(np, array, row_start: int, row_limit: int,
                 col_start: int, col_limit: int):
    height, width = window_shape(array.shape, row_start, row_limit, col_start, col_limit)
    return array[row_start:row_start + height, col_start:col_start + width]


def matrix_summary(np, array, row_start: int, row_limit: int,
                   col_start: int, col_limit: int, category: str) -> dict:
    window = window_array(np, array, row_start, row_limit, col_start, col_limit)
    values = window.astype(np.float64, copy=False)
    result = {
        "shape": [int(v) for v in array.shape],
        "window_shape": [int(v) for v in window.shape],
        "window_start": [row_start, col_start],
        "max_abs": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
    }
    if category == "weight":
        result["per_output_channel_rms"] = np.sqrt(np.mean(values * values, axis=1)).tolist()
        result["per_input_channel_rms"] = np.sqrt(np.mean(values * values, axis=0)).tolist()
    else:
        result["per_token_rms"] = np.sqrt(np.mean(values * values, axis=1)).tolist()
        result["per_channel_rms"] = np.sqrt(np.mean(values * values, axis=0)).tolist()
    return result


def nmse(np, source, candidate, row_start: int, row_limit: int,
         col_start: int, col_limit: int):
    source_window = window_array(np, source, row_start, row_limit, col_start, col_limit)
    candidate_window = window_array(np, candidate, row_start, row_limit, col_start, col_limit)
    if source_window.shape != candidate_window.shape:
        raise RuntimeError("QDQ/reference arrays do not share the same plot window shape")
    source64 = source_window.astype(np.float64, copy=False)
    candidate64 = candidate_window.astype(np.float64, copy=False)
    denominator = float(np.mean(source64 * source64))
    if denominator == 0:
        return None
    delta = source64 - candidate64
    return float(np.mean(delta * delta) / denominator)


def scale_summaries(np, data) -> dict:
    summaries = {}
    for key in data.files:
        lower = key.lower()
        if "scale" not in lower:
            continue
        value = data[key]
        if value.dtype.kind not in "fiu" or value.size == 0:
            continue
        values = value.astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            continue
        summaries[key] = {
            "shape": [int(v) for v in value.shape],
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return summaries


def calculate_all_metrics(np, records: list[dict], path_map: dict,
                          out_root: Path, raw_root: Path, protocol: dict) -> dict:
    metrics = []
    token_start = int_value(protocol.get("token_start", 0), "token_start")
    channel_start = int_value(protocol.get("channel_start", 0), "channel_start")
    plot_tokens = int_value(protocol.get("plot_tokens", 64), "plot_tokens")
    plot_channels = int_value(protocol.get("plot_channels", 512), "plot_channels")
    output_start = int_value(protocol.get("weight_output_start", 0), "weight_output_start")
    output_count = int_value(protocol.get("weight_output_count", 128), "weight_output_count")
    scale_cache = {}

    for row in records:
        category = classify(row)
        row_start, row_limit = (
            (output_start, output_count) if category == "weight" else (token_start, plot_tokens)
        )
        path = path_map[str(row.get("record_id"))][0]
        arrays = {}
        with np.load(path, allow_pickle=False) as data:
            meta = parse_npz_metadata(data, str(row.get("record_id")))
            check_row_metadata(row, meta, category)
            if category == "actual":
                candidates = ("pre_quant", "post_quant", "output")
            elif category == "paired":
                candidates = (
                    "paired_raw", "paired_learned_spinquant", "paired_smooth05_hadamard",
                    "paired_raw_qdq", "paired_learned_spinquant_qdq", "paired_smooth05_hadamard_qdq",
                )
            else:
                candidates = ("weight", "reference")
            for key in candidates:
                if key in data.files:
                    arrays[key] = numeric_matrix(np, data, key, str(row.get("record_id")))
            if not arrays:
                raise RuntimeError(f"{row.get('record_id')!r}: no recognized numeric arrays")
            validate_record_shapes(np, data, row, meta, category, arrays, protocol)
            array_metrics = {
                key: matrix_summary(
                    np, value, row_start, row_limit, channel_start, plot_channels, category
                )
                for key, value in arrays.items()
            }
            qdq_nmse = {}
            if category == "actual" and "pre_quant" in arrays and "post_quant" in arrays:
                qdq_nmse["pre_quant_to_post_quant"] = nmse(
                    np, arrays["pre_quant"], arrays["post_quant"],
                    row_start, row_limit, channel_start, plot_channels,
                )
            elif category == "paired":
                for candidate_key in tuple(arrays):
                    if candidate_key.endswith("_qdq"):
                        source_key = candidate_key[:-4]
                        if source_key in arrays:
                            qdq_nmse[source_key + "_to_qdq"] = nmse(
                                np, arrays[source_key], arrays[candidate_key],
                                row_start, row_limit, channel_start, plot_channels,
                            )
            elif category == "weight" and "weight" in arrays and "reference" in arrays:
                qdq_nmse["weight_to_reference"] = nmse(
                    np, arrays["reference"], arrays["weight"],
                    row_start, row_limit, channel_start, plot_channels,
                )
            scales = scale_summaries(np, data)

        transform_scale_summary = {}
        transform_path = path_map[str(row.get("record_id"))][1:]
        if transform_path:
            transform_file = transform_path[0]
            cache_key = str(transform_file)
            if cache_key not in scale_cache:
                with np.load(transform_file, allow_pickle=False) as transform_data:
                    scale_cache[cache_key] = scale_summaries(np, transform_data)
            transform_scale_summary = scale_cache[cache_key]
        basis = meta.get("basis", row.get("basis"))
        if basis is None and isinstance(meta.get("metadata"), dict):
            basis = meta["metadata"].get("basis")
        metrics.append({
            "record_id": row.get("record_id"),
            "kind": row.get("kind"),
            "file": row.get("file"),
            "category": category,
            "case_id": row.get("case_id"),
            "seed_index": row.get("seed_index"),
            "arm": row.get("arm"),
            "transform": row.get("transform"),
            "stream": row.get("stream"),
            "module": row.get("module"),
            "block_index": row.get("block_index"),
            "stage": row.get("stage"),
            "step": row.get("step"),
            "call_index": row.get("call_index"),
            "basis": basis,
            "arrays": array_metrics,
            "nmse": qdq_nmse,
            "scales": {**scales, **transform_scale_summary},
        })
    return {
        "metric_window": {
            "activation": {
                "token_start": token_start, "token_limit": plot_tokens,
                "channel_start": channel_start, "channel_limit": plot_channels,
                "batch_index": int_value(protocol.get("batch_index", 0), "batch_index"),
                "token_order": protocol.get("token_order", "not declared"),
            },
            "weight": {
                "output_start": output_start, "output_limit": output_count,
                "input_start": channel_start, "input_limit": plot_channels,
            },
        },
        "records": metrics,
    }


def open_record(np, row: dict, path_map: dict, category: str):
    path = path_map[str(row.get("record_id"))][0]
    data = np.load(path, allow_pickle=False)
    metadata = parse_npz_metadata(data, str(row.get("record_id")))
    check_row_metadata(row, metadata, category)
    return data, metadata


def get_window(np, data, key: str, row: dict, protocol: dict, category: str):
    array = numeric_matrix(np, data, key, str(row.get("record_id")))
    if category == "weight":
        row_start = int_value(protocol.get("weight_output_start", 0), "weight_output_start")
        row_limit = int_value(protocol.get("weight_output_count", 128), "weight_output_count")
    else:
        row_start = int_value(protocol.get("token_start", 0), "token_start")
        row_limit = int_value(protocol.get("plot_tokens", 64), "plot_tokens")
    col_start = int_value(protocol.get("channel_start", 0), "channel_start")
    col_limit = int_value(protocol.get("plot_channels", 512), "plot_channels")
    return window_array(np, array, row_start, row_limit, col_start, col_limit)


def set_pale_axes(ax) -> None:
    ax.set_facecolor("#fbfcfe")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((0.96, 0.975, 0.99, 1.0))
            axis.pane.set_edgecolor("#d5deea")
        except AttributeError:
            pass
    ax.grid(True, color="#d5deea", linewidth=0.35, alpha=0.65)


def draw_surface(ax, values, plt, cmap, norm, row_label: str, col_label: str,
                 z_label: str, rcount: int | None = None, ccount: int | None = None):
    height, width = values.shape
    if height < 2 or width < 2:
        raise RuntimeError(
            f"actual surface window is only {height}x{width}; need at least 2 rows and channels"
        )
    numpy = __import__("numpy")
    x, y = numpy.meshgrid(range(width), range(height), indexing="xy")
    magnitude = numpy.abs(values)
    ax.plot_surface(
        x, y, magnitude,
        rcount=height if rcount is None else rcount,
        ccount=width if ccount is None else ccount,
        cmap=cmap, norm=norm, linewidth=0, antialiased=False, shade=False,
        rasterized=True,
    )
    ax.set_xlabel(col_label, fontsize=6, labelpad=1)
    ax.set_ylabel(row_label, fontsize=6, labelpad=1)
    ax.set_zlabel(z_label, fontsize=6, labelpad=1)
    ax.tick_params(axis="both", labelsize=5, pad=0)
    ax.tick_params(axis="z", labelsize=5, pad=0)
    ax.view_init(elev=25, azim=-58)
    ax.set_box_aspect((1.3, 1.0, 0.48))
    set_pale_axes(ax)


def ensure_comparable(windows: list, label: str) -> tuple[int, int]:
    shapes = {tuple(value.shape) for value in windows}
    if len(shapes) != 1:
        raise RuntimeError(f"{label} panels have different clamped window shapes: {sorted(shapes)}")
    return next(iter(shapes))


def make_activation_figure(np, plt, cm, Normalize, ScalarMappable,
                           rows: list[tuple[str, list[tuple[dict, str]]]],
                           blocks: list[int], protocol: dict, title: str,
                           path_map: dict, category: str):
    height = int_value(protocol.get("plot_tokens", 64), "plot_tokens")
    width = int_value(protocol.get("plot_channels", 512), "plot_channels")
    data_windows = {}
    for label, cells in rows:
        for block, (row, key) in zip(blocks, cells):
            data, _ = open_record(np, row, path_map, category)
            try:
                data_windows[(label, block)] = get_window(np, data, key, row, protocol, category)
            finally:
                data.close()
    for block in blocks:
        ensure_comparable(
            [data_windows[(label, block)] for label, _ in rows],
            f"{title}, block {block}",
        )

    col_max = {
        block: max(
            float(np.max(np.abs(data_windows[(label, block)])))
            for label, _ in rows
        ) for block in blocks
    }
    fig = plt.figure(figsize=(22, 3.4 * len(rows) + 1.0), constrained_layout=True)
    axes = {}
    cmap = cm.get_cmap("viridis")
    for row_index, (label, _) in enumerate(rows):
        for col_index, block in enumerate(blocks):
            ax = fig.add_subplot(len(rows), len(blocks), row_index * len(blocks) + col_index + 1,
                                 projection="3d")
            values = data_windows[(label, block)]
            vmax = col_max[block] or 1.0
            norm = Normalize(0.0, vmax)
            draw_surface(
                ax, values, plt, cmap, norm, "Token row", "Channel", "|activation|",
                rcount=values.shape[0], ccount=values.shape[1],
            )
            ax.set_zlim(0.0, vmax)
            ax.set_title(f"block {block}", fontsize=8, pad=1)
            axes[(row_index, col_index)] = ax
        axes[(row_index, 0)].text2D(
            -0.12, 0.5, label, transform=axes[(row_index, 0)].transAxes,
            ha="right", va="center", fontsize=8,
        )
    for col_index, block in enumerate(blocks):
        vmax = col_max[block] or 1.0
        mapper = ScalarMappable(norm=Normalize(0.0, vmax), cmap=cmap)
        fig.colorbar(mapper, ax=[axes[(r, col_index)] for r in range(len(rows))],
                     shrink=0.63, pad=0.01, aspect=24, label=f"|activation|, 0–{vmax:.3g}")
    fig.suptitle(
        f"{title} | first {height} token rows × {width} channels; "
        f"shared abs heights/colors within each depth",
        fontsize=12,
    )
    return fig, data_windows


def plot_error_heatmap(ax, error, np, cmap, vmin, vmax, row_label: str,
                       col_label: str, title: str):
    image = ax.imshow(
        error, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
        vmin=vmin, vmax=vmax, rasterized=True,
        extent=(-0.5, error.shape[1] - 0.5, -0.5, error.shape[0] - 0.5),
    )
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(col_label, fontsize=6)
    ax.set_ylabel(row_label, fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_facecolor("#fbfcfe")
    return image


def weight_scale_data(np, row, path_map, record_id: str):
    values = {}
    data, _ = open_record(np, row, path_map, "weight")
    try:
        for key in data.files:
            if "scale" in key.lower():
                value = data[key]
                if value.dtype.kind in "fiu" and value.size and np.isfinite(value).all():
                    values[key] = value.astype(np.float64, copy=False)
    finally:
        data.close()
    for transform_path in path_map[str(row.get("record_id"))][1:]:
        with np.load(transform_path, allow_pickle=False) as transform_data:
            for key in transform_data.files:
                if "scale" in key.lower():
                    value = transform_data[key]
                    if value.dtype.kind in "fiu" and value.size and np.isfinite(value).all():
                        values[key] = value.astype(np.float64, copy=False)
    return values


def make_weight_pages(np, plt, cm, Normalize, ScalarMappable, PdfPages,
                      rows_by_arm: dict, arms: list[str], blocks: list[int],
                      protocol: dict, title: str, path_map: dict):
    weights = {}
    references = {}
    scale_values = {}
    basis_labels = {}
    for arm in arms:
        for block in blocks:
            row = rows_by_arm[(arm, block)]
            data, metadata = open_record(np, row, path_map, "weight")
            try:
                w = get_window(np, data, "weight", row, protocol, "weight")
                ref = get_window(np, data, "reference", row, protocol, "weight")
            finally:
                data.close()
            if w.shape != ref.shape:
                raise RuntimeError(f"{row.get('record_id')}: weight/reference windows differ")
            weights[(arm, block)] = w
            references[(arm, block)] = ref
            scale_values[(arm, block)] = weight_scale_data(np, row, path_map, str(row.get("record_id")))
            basis = metadata.get("basis", row.get("basis"))
            if basis is None and isinstance(metadata.get("metadata"), dict):
                basis = metadata["metadata"].get("basis")
            basis_labels[(arm, block)] = str(basis) if basis is not None else ""
    for block in blocks:
        ensure_comparable([weights[(arm, block)] for arm in arms], f"{title}, block {block} weights")
        ensure_comparable([references[(arm, block)] for arm in arms], f"{title}, block {block} references")

    weight_max = {
        block: max(float(np.max(np.abs(weights[(arm, block)]))) for arm in arms) or 1.0
        for block in blocks
    }
    error_max = {
        block: max(
            float(np.max(np.abs(weights[(arm, block)] - references[(arm, block)])))
            for arm in arms
        ) or 1.0
        for block in blocks
    }
    surfaces = []
    fig = plt.figure(figsize=(22, 3.25 * len(arms) + 1), constrained_layout=True)
    axes = {}
    cmap = cm.get_cmap("cividis")
    for r, arm in enumerate(arms):
        for c, block in enumerate(blocks):
            ax = fig.add_subplot(len(arms), len(blocks), r * len(blocks) + c + 1, projection="3d")
            value = weights[(arm, block)]
            vmax = weight_max[block]
            draw_surface(
                ax, value, plt, cmap, Normalize(0.0, vmax), "Output channel",
                "Input channel", "|weight|", rcount=value.shape[0], ccount=value.shape[1],
            )
            ax.set_zlim(0.0, vmax)
            ax.set_title(f"block {block}", fontsize=7, pad=1)
            axes[(r, c)] = ax
        row_basis = basis_labels[(arm, blocks[0])]
        row_label = arm.replace("_", " ") + (f" | {row_basis}" if row_basis else "")
        axes[(r, 0)].text2D(
            -0.12, 0.5, row_label, transform=axes[(r, 0)].transAxes,
            ha="right", va="center", fontsize=7,
        )
    for c, block in enumerate(blocks):
        fig.colorbar(
            ScalarMappable(norm=Normalize(0.0, weight_max[block]), cmap=cmap),
            ax=[axes[(r, c)] for r in range(len(arms))], shrink=0.65, pad=0.01,
            aspect=24, label=f"|weight|, 0–{weight_max[block]:.3g}",
        )
    fig.suptitle(
        f"{title} | output channel × input channel; shared abs scale within each depth",
        fontsize=12,
    )
    surfaces.append(fig)

    errors = []
    fig = plt.figure(figsize=(22, 3.0 * len(arms) + 1), constrained_layout=True)
    axes = {}
    error_cmap = cm.get_cmap("magma")
    for r, arm in enumerate(arms):
        for c, block in enumerate(blocks):
            ax = fig.add_subplot(len(arms), len(blocks), r * len(blocks) + c + 1)
            error = np.abs(weights[(arm, block)] - references[(arm, block)])
            vmax = error_max[block]
            image = plot_error_heatmap(
                ax, error, np, error_cmap, 0.0, vmax, "Output channel",
                "Input channel", f"block {block}",
            )
            axes[(r, c)] = ax
        row_basis = basis_labels[(arm, blocks[0])]
        row_label = arm.replace("_", " ") + (f" | {row_basis}" if row_basis else "")
        axes[(r, 0)].text(
            -0.12, 0.5, row_label, transform=axes[(r, 0)].transAxes,
            ha="right", va="center", fontsize=7,
        )
    for c, block in enumerate(blocks):
        fig.colorbar(
            ScalarMappable(norm=Normalize(0.0, error_max[block]), cmap=error_cmap),
            ax=[axes[(r, c)] for r in range(len(arms))], shrink=0.72, pad=0.01,
            aspect=24, label=f"|weight-reference|, 0–{error_max[block]:.3g}",
        )
    fig.suptitle(
        f"{title} | absolute error against each arm's equivalent unquantized reference",
        fontsize=12,
    )
    surfaces.append(fig)

    if any(scale_values.values()):
        fig = plt.figure(figsize=(22, 2.8 * len(arms) + 1), constrained_layout=True)
        axes = {}
        scale_cmap = cm.get_cmap("viridis")
        scale_bounds = {}
        for block in blocks:
            all_values = [
                value for arm in arms for value in scale_values[(arm, block)].values()
            ]
            if all_values:
                scale_bounds[block] = (
                    min(float(np.min(value)) for value in all_values),
                    max(float(np.max(value)) for value in all_values),
                )
            else:
                scale_bounds[block] = (0.0, 1.0)
        for r, arm in enumerate(arms):
            for c, block in enumerate(blocks):
                ax = fig.add_subplot(len(arms), len(blocks), r * len(blocks) + c + 1)
                values = scale_values[(arm, block)]
                low, high = scale_bounds[block]
                if not values:
                    ax.text(0.5, 0.5, "scale not saved", ha="center", va="center", fontsize=7)
                    ax.set_axis_off()
                else:
                    key = sorted(values)[0]
                    array = values[key]
                    ax.set_title(f"block {block}: {key} {tuple(array.shape)}", fontsize=6)
                    if array.ndim == 0:
                        ax.text(0.5, 0.5, f"{float(array):.4g}", ha="center", va="center")
                        ax.set_axis_off()
                    elif array.ndim == 1:
                        ax.plot(np.arange(array.size), array, color="#315e9e", linewidth=0.8)
                        if low != high:
                            ax.set_ylim(low, high)
                        ax.set_xlabel("scale index", fontsize=6)
                        ax.tick_params(labelsize=5)
                        ax.grid(True, color="#d5deea", linewidth=0.35)
                    else:
                        image = ax.imshow(
                            array.reshape(array.shape[0], -1), origin="lower", aspect="auto",
                            interpolation="nearest", cmap=scale_cmap, vmin=low,
                            vmax=high if high != low else low + 1e-12, rasterized=True,
                        )
                        ax.tick_params(labelsize=5)
                axes[(r, c)] = ax
            row_basis = basis_labels[(arm, blocks[0])]
            row_label = arm.replace("_", " ") + (f" | {row_basis}" if row_basis else "")
            axes[(r, 0)].text(
                -0.12, 0.5, row_label, transform=axes[(r, 0)].transAxes,
                ha="right", va="center", fontsize=7,
            )
        fig.suptitle(f"{title} | saved scale arrays (first scale array per record)", fontsize=12)
        surfaces.append(fig)
    return surfaces, {
        "weight_abs_max_by_block": {str(k): v for k, v in weight_max.items()},
        "error_abs_max_by_block": {str(k): v for k, v in error_max.items()},
        "has_saved_scales": any(scale_values.values()),
    }


def save_preview(fig, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for dpi in (80, 64, 48, 36, 28, 20, 14, 10):
        buffer = io.BytesIO()
        try:
            fig.savefig(
                buffer, format="jpeg", dpi=dpi, bbox_inches="tight",
                pil_kwargs={"quality": 78, "optimize": True},
            )
        except TypeError:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="jpeg", dpi=dpi, bbox_inches="tight")
        content = buffer.getvalue()
        if len(content) <= MAX_PREVIEW_BYTES:
            destination.write_bytes(content)
            return len(content)
        last = len(content)
    raise RuntimeError(
        f"could not fit preview under {MAX_PREVIEW_BYTES} bytes; last size was {last}"
    )


def save_bundle(PdfPages, pages: list, plot_dir: Path, stem: str,
                source_rows: list[dict], out_root: Path, raw_root: Path,
                extra: dict | None = None,
                preview_page_numbers: set[int] | None = None) -> dict:
    pdf_rel = f"{stem}.pdf"
    pdf_path = plot_dir / pdf_rel
    page_entries = []
    with PdfPages(pdf_path) as pdf:
        for index, (fig, description) in enumerate(pages, start=1):
            png_rel = f"{stem}__page{index}.png"
            fig.savefig(plot_dir / png_rel, dpi=140, bbox_inches="tight")
            pdf.savefig(fig, dpi=140, bbox_inches="tight")
            entry = {
                "page": index,
                "description": description,
                "png": png_rel,
            }
            if preview_page_numbers is None or index in preview_page_numbers:
                preview_rel = f"previews/{stem}__page{index}.jpg"
                entry["preview"] = preview_rel
                entry["preview_bytes"] = save_preview(fig, plot_dir / preview_rel)
            page_entries.append(entry)
    sources = {}
    for row in source_rows:
        for path in resolve_sources(row, out_root, raw_root):
            sources[path.relative_to(out_root.resolve()).as_posix()] = True
    result = {
        "id": stem,
        "title": stem.replace("_", " "),
        "pdf": pdf_rel,
        "pages": page_entries,
        "preview_paths": [page["preview"] for page in page_entries if "preview" in page],
        "source_npz": sorted(sources),
        "source_record_ids": sorted({str(row.get("record_id")) for row in source_rows}),
    }
    if extra:
        result.update(extra)
    for fig, _ in pages:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return result


def link_from_plot(path: str) -> str:
    return quote(path.replace("\\", "/"), safe="/._-")


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def make_bundle_specs(protocol: dict, main_rows: dict, weight_rows: dict,
                      arms: list[str], transforms: list[str], streams: list[str],
                      suffixes: list[str], blocks: list[int], main_case: int,
                      main_step: int) -> list[dict]:
    specs = []
    transform_keys = {
        "raw": "paired_raw",
        "learned_spinquant": "paired_learned_spinquant",
        "smooth05_hadamard": "paired_smooth05_hadamard",
    }
    paired_labels = {
        "raw": "Raw BF16",
        "learned_spinquant": "SpinQuant-adapted learned R128",
        "smooth05_hadamard": "Smooth0.5 Hadamard",
    }
    actual_labels = {
        "bf16": "BF16 raw; A4 skipped",
        "spinquant_native_w4a4": "learned R128, SpinQuant-adapted; before A4",
        "smooth_scalar_native_w4a4": "Smooth+H128 basis; before A4",
        "row_scale_vq_a4": "Smooth+H128 basis; before decoded-BF16 A4",
        "micro_scale_vq_a4": "Smooth+H128 basis; before decoded-BF16 A4",
    }
    post_labels = {
        "bf16": "BF16 identity copy",
        "spinquant_native_w4a4": "SpinQuant-adapted native INT4 QDQ",
        "smooth_scalar_native_w4a4": "SmoothScalar native INT4 QDQ",
        "row_scale_vq_a4": "Row-scale VQ decoded BF16 after A4",
        "micro_scale_vq_a4": "Micro-scale VQ decoded BF16 after A4",
    }
    for stream in streams:
        for suffix in suffixes:
            paired_rows = []
            paired_plot_rows = []
            for transform in transforms:
                cells = []
                for block in blocks:
                    row = main_rows["paired"][(stream, suffix)][(transform, block)]
                    cells.append((row, transform_keys[transform]))
                    paired_rows.append(row)
                paired_plot_rows.append((paired_labels[transform], cells))
            specs.append({
                "id": f"activation_paired_{stream}_{suffix.replace('.', '_')}",
                "kind": "paired", "stream": stream, "suffix": suffix,
                "plot_rows": paired_plot_rows, "source_rows": paired_rows,
                "title": f"Paired-local activation | {stream} | {suffix} | case {main_case}, step {main_step}",
                "page_descriptions": ["paired-local 3 transforms × 4 depths"],
                "preview_page_numbers": None,
            })

            actual_rows = []
            pre_rows = []
            post_rows = []
            for arm in arms:
                cells = []
                for block in blocks:
                    row = main_rows["actual"][(stream, suffix)][(arm, block)]
                    cells.append((row, "pre_quant"))
                    actual_rows.append(row)
                pre_rows.append((actual_labels.get(arm, arm.replace("_", " ")), cells))
                post_rows.append((post_labels.get(arm, arm.replace("_", " ")),
                                  [(row, "post_quant") for row, _ in cells]))
            specs.append({
                "id": f"activation_actual_{stream}_{suffix.replace('.', '_')}",
                "kind": "actual", "stream": stream, "suffix": suffix,
                "pre_rows": pre_rows, "post_rows": post_rows, "source_rows": actual_rows,
                "title": f"Actual activation | {stream} | {suffix} | case {main_case}, step {main_step}",
                "page_descriptions": [
                    "A4 前：各实际 arm 的 pre_quant；已应用该 arm 输入变换，BF16 为 raw identity",
                    "A4 后：native INT4 的反量化值、VQ 的 decoded BF16；BF16 为 identity copy",
                ],
                "preview_page_numbers": None,
            })

            rows_by_arm = {
                (arm, block): weight_rows[(arm, stream, suffix, block)]
                for arm in arms for block in blocks
            }
            weight_source_rows = [rows_by_arm[(arm, block)] for arm in arms for block in blocks]
            specs.append({
                "id": f"weight_{stream}_{suffix.replace('.', '_')}",
                "kind": "weight", "stream": stream, "suffix": suffix,
                "rows_by_arm": rows_by_arm, "source_rows": weight_source_rows,
                "title": f"Weight | {stream} | {suffix}",
                "preview_page_numbers": {1},
            })
    return specs


def render_bundle(spec: dict, np, plt, cm, Normalize, PdfPages, ScalarMappable,
                  protocol: dict, blocks: list[int], arms: list[str],
                  path_map: dict, plot_dir: Path, out_root: Path,
                  raw_root: Path) -> dict:
    if spec["kind"] == "paired":
        fig, _ = make_activation_figure(
            np, plt, cm, Normalize, ScalarMappable, spec["plot_rows"], blocks,
            protocol, spec["title"], path_map, "paired",
        )
        pages = [(fig, spec["page_descriptions"][0])]
        return save_bundle(
            PdfPages, pages, plot_dir, spec["id"], spec["source_rows"],
            out_root, raw_root, preview_page_numbers=spec["preview_page_numbers"],
        )
    if spec["kind"] == "actual":
        pre_fig, _ = make_activation_figure(
            np, plt, cm, Normalize, ScalarMappable, spec["pre_rows"], blocks,
            protocol, spec["title"] + " | before A4", path_map, "actual",
        )
        post_fig, _ = make_activation_figure(
            np, plt, cm, Normalize, ScalarMappable, spec["post_rows"], blocks,
            protocol, spec["title"] + " | after A4", path_map, "actual",
        )
        pages = [(pre_fig, spec["page_descriptions"][0]),
                 (post_fig, spec["page_descriptions"][1])]
        return save_bundle(
            PdfPages, pages, plot_dir, spec["id"], spec["source_rows"],
            out_root, raw_root, preview_page_numbers=spec["preview_page_numbers"],
        )

    pages, scales_meta = make_weight_pages(
        np, plt, cm, Normalize, ScalarMappable, PdfPages, spec["rows_by_arm"],
        arms, blocks, protocol, spec["title"], path_map,
    )
    descriptions = [
        "weight magnitude surfaces: 5 arms × 4 depths",
        "absolute weight error against each arm's equivalent reference",
    ]
    if len(pages) == 3:
        descriptions.append("saved quantization scales (first named scale per record)")
    return save_bundle(
        PdfPages, list(zip(pages, descriptions)), plot_dir, spec["id"],
        spec["source_rows"], out_root, raw_root, scales_meta,
        preview_page_numbers=spec["preview_page_numbers"],
    )


def output_file_exists(plot_dir: Path, relative_path: str, max_bytes: int | None = None) -> bool:
    path = Path(str(relative_path).replace("\\", "/"))
    if path.is_absolute():
        return False
    candidate = (plot_dir / path).resolve()
    try:
        candidate.relative_to(plot_dir.resolve())
    except ValueError:
        return False
    if not candidate.is_file() or candidate.stat().st_size == 0:
        return False
    return max_bytes is None or candidate.stat().st_size <= max_bytes


def reusable_bundle(existing: dict, spec: dict, out_root: Path, raw_root: Path,
                    plot_dir: Path) -> bool:
    if existing.get("id") != spec["id"]:
        return False
    expected_ids = sorted({str(row.get("record_id")) for row in spec["source_rows"]})
    expected_sources = sorted({
        path.relative_to(out_root.resolve()).as_posix()
        for row in spec["source_rows"] for path in resolve_sources(row, out_root, raw_root)
    })
    if existing.get("source_record_ids") != expected_ids or existing.get("source_npz") != expected_sources:
        return False
    if not output_file_exists(plot_dir, existing.get("pdf", "")):
        return False
    pages = existing.get("pages")
    if not isinstance(pages, list) or not pages:
        return False
    if spec["kind"] in ("paired", "actual") and len(pages) != len(spec["page_descriptions"]):
        return False
    if spec["kind"] == "weight" and len(pages) not in (2, 3):
        return False
    for index, page in enumerate(pages, start=1):
        if page.get("page") != index or not output_file_exists(plot_dir, page.get("png", "")):
            return False
        expects_preview = spec["preview_page_numbers"] is None or index in spec["preview_page_numbers"]
        if expects_preview != ("preview" in page):
            return False
        if expects_preview:
            preview = page.get("preview", "")
            if not output_file_exists(plot_dir, preview, MAX_PREVIEW_BYTES):
                return False
            if page.get("preview_bytes") != (plot_dir / Path(preview)).stat().st_size:
                return False
    return True


def cached_metrics(plot_dir: Path, protocol: dict, source_identities: dict,
                   record_ids: set[str]) -> dict | None:
    path = plot_dir / "metrics.json"
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except RuntimeError:
        return None
    if value.get("renderer_schema") != RENDERER_SCHEMA:
        return None
    if value.get("protocol_snapshot") != protocol or value.get("source_identities") != source_identities:
        return None
    rows = value.get("records")
    if not isinstance(rows, list) or {str(row.get("record_id")) for row in rows} != record_ids:
        return None
    return value


def write_gallery(plot_dir: Path, protocol_path: Path, out_root: Path,
                  figures: list[dict], metrics_path: str, raw_manifest_path: Path,
                  additional_rows: list[dict], protocol: dict,
                  source_identities: dict, status: str,
                  expected_bundle_count: int, metrics_ready: bool) -> None:
    protocol_copy = plot_dir / "protocol.json"
    shutil.copyfile(protocol_path, protocol_copy)
    manifest_doc = {
        "renderer_schema": RENDERER_SCHEMA,
        "status": status,
        "bundle_count": len(figures),
        "expected_bundle_count": expected_bundle_count,
        "protocol": read_json(protocol_path).get("protocol"),
        "protocol_snapshot": protocol,
        "source_identities": source_identities,
        "source_manifest": Path(os.path.relpath(raw_manifest_path, plot_dir)).as_posix(),
        "protocol_copy": "protocol.json",
        "metrics": metrics_path,
        "metrics_ready": metrics_ready,
        "figures": figures,
        "additional_records": [
            {
                "record_id": row.get("record_id"),
                "call_index": row.get("call_index"),
                "file": row.get("file"),
            } for row in additional_rows
        ],
    }

    lines = [
        "<!doctype html>", '<html lang="zh-Hans"><meta charset="utf-8">',
        "<title>Fast-WAM Prism visualization</title>",
        "<style>body{font:15px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#182431;background:#fafcff}"
        "a{color:#1759a6}article{border:1px solid #d6e0ec;border-radius:10px;padding:1rem;margin:1rem 0;background:white}"
        "img{max-width:100%;height:auto;background:#f5f8fc}.meta{color:#526579;font-size:.9rem}code{overflow-wrap:anywhere}</style>",
        "<h1>Fast-WAM Prism-style activation / weight gallery</h1>",
        "<p>该 gallery 展示冻结输入上的局部数值形状。signed NPZ 是原始数据；surface 显示幅值。图本身不提供 accuracy、latency 或 closed-loop 因果证据。</p>",
        "<p>",
        f'<a href="{link_from_plot(manifest_doc["source_manifest"])}">raw manifest</a> · ',
        '<a href="protocol.json">protocol copy</a> · ',
        '<a href="manifest.json">visualization manifest</a>',
        "</p>",
    ]
    if metrics_ready:
        lines[lines.index('<a href="manifest.json">visualization manifest</a>')] = (
            '<a href="metrics.json">per-record metrics</a> · '
            + '<a href="manifest.json">visualization manifest</a>'
        )
    if status == "checkpointed":
        lines.append(
            f'<p class="meta">Render checkpoint: {len(figures)} of {expected_bundle_count} bundles complete; '
            "rerun with the same protocol and source identities to resume.</p>"
        )
    for figure in figures:
        lines.append("<article>")
        lines.append(f"<h2>{html.escape(figure['title'])}</h2>")
        lines.append(
            f'<p class="meta"><a href="{link_from_plot(figure["pdf"])}">PDF bundle</a> · '
            + "Source NPZ: "
            + ", ".join(
                f'<a href="{link_from_plot(os.path.relpath(out_root / Path(src), plot_dir).replace(os.sep, "/"))}">{html.escape(Path(src).name)}</a>'
                for src in figure["source_npz"]
            )
            + "</p>"
        )
        for page in figure["pages"]:
            lines.append(f"<h3>{html.escape(page['description'])}</h3>")
            preview_summary = (
                f"Preview {page['preview_bytes']} bytes · "
                if "preview" in page else "Full-resolution image · "
            )
            lines.append(
                f'<p class="meta">{preview_summary}'
                f'<a href="{link_from_plot(page["png"])}">full PNG</a> · '
                f'<a href="{link_from_plot(figure["pdf"])}#page={page["page"]}">PDF page {page["page"]}</a></p>'
            )
            if "preview" in page:
                lines.append(
                    f'<a href="{link_from_plot(page["png"])}"><img src="{link_from_plot(page["preview"])}" '
                    f'alt="{html.escape(figure["title"])} preview"></a>'
                )
        lines.append("</article>")
    if additional_rows:
        lines.append(
            f"<p class=\"meta\">Primary panels follow the manifest rule, lowest call_index. "
            f"{len(additional_rows)} repeated call records remain listed in the visualization manifest "
            "and per-record metrics.</p>"
        )
    lines.extend(("</html>", ""))
    atomic_write_text(plot_dir / "index.html", "\n".join(lines))
    # Write the receipt last, after every listed bundle and its gallery links exist.
    atomic_write_json(plot_dir / "manifest.json", manifest_doc)


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if args.protocol is None or args.out is None:
        raise RuntimeError("--protocol and --out are required")
    require_compute_allocation()
    protocol_path = args.protocol.resolve()
    out_root = args.out.resolve()
    raw_root = (out_root / "raw").resolve()
    raw_manifest_path = raw_root / "manifest.json"
    protocol = read_json(protocol_path)
    manifest = read_json(raw_manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError("raw manifest records must be a list")
    required_protocol = (
        "protocol", "cases", "arms", "paired_transforms", "blocks", "streams",
        "module_suffixes", "snapshot_steps", "batch_index", "token_order",
    )
    missing = [key for key in required_protocol if key not in protocol]
    if missing:
        raise RuntimeError(f"protocol lacks required fields: {missing}")
    if int_value(protocol["batch_index"], "batch_index") != 0:
        raise RuntimeError("renderer requires the frozen first batch/CFG branch (batch_index=0)")

    main_rows, weight_rows, path_map, extras = preflight(
        protocol, manifest, records, out_root, raw_root
    )
    np, plt, cm, Normalize, PdfPages, ScalarMappable = import_plot_libraries()

    plot_dir = out_root / "plots"
    preview_dir = plot_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    metrics_rel = "metrics.json"

    arms = list(protocol["arms"])
    transforms = list(protocol["paired_transforms"])
    streams = list(protocol["streams"])
    suffixes = list(protocol["module_suffixes"])
    blocks = [int_value(v, "protocol block") for v in protocol["blocks"]]
    main_case = int_value(protocol.get("main_case", 1004), "main_case")
    main_step = int_value(protocol.get("main_step", 4), "main_step")
    bundle_specs = make_bundle_specs(
        protocol, main_rows, weight_rows, arms, transforms, streams, suffixes,
        blocks, main_case, main_step,
    )
    source_identities = manifest.get("source_identities", {})
    if not isinstance(source_identities, dict):
        raise RuntimeError("raw manifest source_identities must be an object")
    start_text = os.environ.get("FW_JOB_START_EPOCH")
    if not start_text:
        raise RuntimeError("PBS renderer must receive FW_JOB_START_EPOCH from its job wrapper")
    try:
        start_epoch = float(start_text)
    except ValueError as exc:
        raise RuntimeError("FW_JOB_START_EPOCH must be numeric when provided") from exc
    if not math.isfinite(start_epoch):
        raise RuntimeError("FW_JOB_START_EPOCH must be finite")
    soft_seconds = int_value(protocol.get("soft_deadline_seconds", 6600), "soft_deadline_seconds")
    if not 0 < soft_seconds <= 6600:
        raise RuntimeError("renderer soft deadline must be positive and no more than 6600 seconds")
    soft_deadline = start_epoch + soft_seconds

    existing = None
    existing_path = plot_dir / "manifest.json"
    if existing_path.is_file():
        try:
            existing = read_json(existing_path)
        except RuntimeError:
            existing = None
    existing_figures = {}
    if (
        isinstance(existing, dict)
        and existing.get("renderer_schema") == RENDERER_SCHEMA
        and existing.get("protocol_snapshot") == protocol
        and existing.get("source_identities") == source_identities
        and isinstance(existing.get("figures"), list)
    ):
        existing_figures = {
            str(row.get("id")): row for row in existing["figures"] if isinstance(row, dict)
        }
    reusable = {
        spec["id"]: existing_figures[spec["id"]]
        for spec in bundle_specs
        if spec["id"] in existing_figures
        and reusable_bundle(existing_figures[spec["id"]], spec, out_root, raw_root, plot_dir)
    }
    figure_map = dict(reusable)
    record_ids = {str(row.get("record_id")) for row in records}
    metric_doc = cached_metrics(plot_dir, protocol, source_identities, record_ids)

    write_gallery(
        plot_dir, protocol_path, out_root,
        [figure_map[spec["id"]] for spec in bundle_specs if spec["id"] in figure_map],
        metrics_rel, raw_manifest_path, extras, protocol, source_identities,
        "checkpointed", len(bundle_specs), metric_doc is not None,
    )

    if metric_doc is None:
        if time.time() >= soft_deadline:
            print("Render checkpointed before metrics; rerun this result directory to resume.")
            return 0
        metric_doc = calculate_all_metrics(np, records, path_map, out_root, raw_root, protocol)
        metric_doc.update({
            "renderer_schema": RENDERER_SCHEMA,
            "protocol_snapshot": protocol,
            "source_identities": source_identities,
        })
        atomic_write_json(plot_dir / metrics_rel, metric_doc)
        write_gallery(
            plot_dir, protocol_path, out_root,
            [figure_map[spec["id"]] for spec in bundle_specs if spec["id"] in figure_map],
            metrics_rel, raw_manifest_path, extras, protocol, source_identities,
            "checkpointed", len(bundle_specs), True,
        )

    for spec in bundle_specs:
        if spec["id"] in figure_map:
            continue
        if time.time() >= soft_deadline:
            ordered = [figure_map[item["id"]] for item in bundle_specs if item["id"] in figure_map]
            write_gallery(
                plot_dir, protocol_path, out_root, ordered, metrics_rel,
                raw_manifest_path, extras, protocol, source_identities,
                "checkpointed", len(bundle_specs), True,
            )
            print(f"Render checkpointed after {len(ordered)}/{len(bundle_specs)} complete bundles.")
            return 0
        figure_map[spec["id"]] = render_bundle(
            spec, np, plt, cm, Normalize, PdfPages, ScalarMappable, protocol,
            blocks, arms, path_map, plot_dir, out_root, raw_root,
        )
        ordered = [figure_map[item["id"]] for item in bundle_specs if item["id"] in figure_map]
        write_gallery(
            plot_dir, protocol_path, out_root, ordered, metrics_rel,
            raw_manifest_path, extras, protocol, source_identities,
            "checkpointed", len(bundle_specs), True,
        )

    figures = [figure_map[spec["id"]] for spec in bundle_specs]
    write_gallery(
        plot_dir, protocol_path, out_root, figures, metrics_rel,
        raw_manifest_path, extras, protocol, source_identities,
        "complete", len(bundle_specs), True,
    )
    print(
        f"Rendered/resumed {len(figures)} bundles; "
        f"{sum(len(figure.get('preview_paths', [])) for figure in figures)} preview images are <= "
        f"{MAX_PREVIEW_BYTES} bytes; metrics cover {len(metric_doc['records'])} records."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

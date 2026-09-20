#!/usr/bin/env python3
"""Diagnose paired decision-trace structure and non-finite values."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELDS = ("step", "topk_values", "topk_indices", "mean", "variance", "costs")
MISSING = object()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def category(value: Any) -> str:
    if not is_number(value):
        return f"type:{type(value).__name__}"
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return "finite"


def shape_signature(value: Any) -> Any:
    if value is MISSING:
        return "<missing>"
    if isinstance(value, list):
        child_shapes = [shape_signature(item) for item in value]
        if not child_shapes:
            return [0]
        first = child_shapes[0]
        return [len(value), first] if all(item == first for item in child_shapes) else [len(value), "ragged", child_shapes[:3]]
    if isinstance(value, dict):
        return {str(key): shape_signature(value[key]) for key in sorted(value)}
    return category(value)


def flatten(value: Any, path: str = "") -> Iterable[tuple[str, str, Any]]:
    if value is MISSING:
        yield path, "<missing>", None
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            yield from flatten(item, child)
    elif isinstance(value, dict):
        for key in sorted(value):
            child = f"{path}.{key}" if path else str(key)
            yield from flatten(value[key], child)
    else:
        yield path, category(value), value


def maps(value: Any) -> dict[str, tuple[str, Any]]:
    return {path: (kind, raw) for path, kind, raw in flatten(value)}


def exact_equal(left: Any, right: Any) -> bool:
    left_map = maps(left)
    right_map = maps(right)
    if set(left_map) != set(right_map):
        return False
    for path in left_map:
        left_kind, left_raw = left_map[path]
        right_kind, right_raw = right_map[path]
        if left_kind != right_kind or left_kind != "finite" or left_raw != right_raw:
            return False
    return True


def compare_field(left: Any, right: Any) -> dict[str, Any]:
    left_map = maps(left)
    right_map = maps(right)
    paths = sorted(set(left_map) | set(right_map))
    finite_diffs: list[float] = []
    left_counts = defaultdict(int)
    right_counts = defaultdict(int)
    mask_match = set(left_map) == set(right_map)
    signs_match = mask_match
    first_difference: dict[str, str] | None = None
    for path in paths:
        left_kind, left_raw = left_map.get(path, ("<missing>", None))
        right_kind, right_raw = right_map.get(path, ("<missing>", None))
        if left_kind in {"nan", "+inf", "-inf"}:
            left_counts[left_kind] += 1
        if right_kind in {"nan", "+inf", "-inf"}:
            right_counts[right_kind] += 1
        if left_kind != right_kind and {left_kind, right_kind} <= {"finite", "nan", "+inf", "-inf"}:
            mask_match = False
        if left_kind in {"nan", "+inf", "-inf"} or right_kind in {"nan", "+inf", "-inf"}:
            if left_kind != right_kind:
                signs_match = False
        if left_kind == "finite" and right_kind == "finite":
            finite_diffs.append(abs(float(left_raw) - float(right_raw)))
            different = left_raw != right_raw
        else:
            different = left_kind != right_kind or left_kind == "<missing>" or right_kind == "<missing>"
        if first_difference is None and different:
            first_difference = {"path": path, "left": left_kind, "right": right_kind}
    return {
        "shape_equal": shape_signature(left) == shape_signature(right),
        "left_shape": shape_signature(left),
        "right_shape": shape_signature(right),
        "exact_equal": exact_equal(left, right),
        "finite_max_abs_both": max(finite_diffs) if finite_diffs else None,
        "left_nonfinite_counts": dict(sorted(left_counts.items())),
        "right_nonfinite_counts": dict(sorted(right_counts.items())),
        "nonfinite_masks_match": mask_match,
        "nonfinite_signs_match": signs_match,
        "first_difference": first_difference,
    }


def trace_field(row: dict[str, Any], field: str) -> Any:
    if field == "costs":
        return row.get("costs", MISSING)
    trace = row.get("decision_trace", MISSING)
    if isinstance(trace, list):
        return [entry.get(field, MISSING) if isinstance(entry, dict) else MISSING for entry in trace]
    if isinstance(trace, dict):
        return trace.get(field, MISSING)
    return MISSING


def pair_diagnosis(key: tuple[Any, Any, Any], unit: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = unit["baseline"]
    cached = unit["iteration_cache"]
    fields = {field: compare_field(trace_field(baseline, field), trace_field(cached, field)) for field in FIELDS}
    first_field = next((field for field in FIELDS if fields[field]["first_difference"] is not None), None)
    first = fields[first_field]["first_difference"] if first_field else None
    if first is not None:
        first = {"field": first_field, **first}
    return {
        "observation_id": key[0],
        "warmup": bool(key[1]),
        "technical_repeat": key[2],
        "fields": fields,
        "first_differing_field_path": first,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"pair_count": len(rows), "fields": {}}
    for field in FIELDS:
        field_rows = [row["fields"][field] for row in rows]
        finite_values = [item["finite_max_abs_both"] for item in field_rows if item["finite_max_abs_both"] is not None]
        left_counts = defaultdict(int)
        right_counts = defaultdict(int)
        for item in field_rows:
            for key, value in item["left_nonfinite_counts"].items():
                left_counts[key] += value
            for key, value in item["right_nonfinite_counts"].items():
                right_counts[key] += value
        result["fields"][field] = {
            "pair_count": len(field_rows),
            "shape_equal_count": sum(item["shape_equal"] for item in field_rows),
            "exact_equal_count": sum(item["exact_equal"] for item in field_rows),
            "finite_max_abs_both_max": max(finite_values) if finite_values else None,
            "left_nonfinite_counts": dict(sorted(left_counts.items())),
            "right_nonfinite_counts": dict(sorted(right_counts.items())),
            "nonfinite_masks_match_count": sum(item["nonfinite_masks_match"] for item in field_rows),
            "nonfinite_signs_match_count": sum(item["nonfinite_signs_match"] for item in field_rows),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    grouped: dict[tuple[Any, Any, Any], dict[str, dict[str, Any]]] = {}
    for row in summary.get("records", []):
        key = (row.get("observation_id"), bool(row.get("warmup")), row.get("technical_repeat"))
        grouped.setdefault(key, {})[row.get("path")] = row
    pairs = [pair_diagnosis(key, grouped[key]) for key in sorted(grouped) if set(grouped[key]) == {"baseline", "iteration_cache"}]
    output = {
        "source_schema": summary.get("schema"),
        "source_job_id": summary.get("job_id"),
        "pair_count": len(pairs),
        "pairs": pairs,
        "aggregates": {
            "all": aggregate(pairs),
            "warmup": aggregate([row for row in pairs if row["warmup"]]),
            "technical": aggregate([row for row in pairs if not row["warmup"]]),
            "by_observation": {
                observation_id: aggregate([row for row in pairs if row["observation_id"] == observation_id])
                for observation_id in sorted({row["observation_id"] for row in pairs})
            },
        },
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "pair_count": len(pairs), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

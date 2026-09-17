"""CPU-only audit of frozen pilot-test configuration and archived jobs.

The audit is intentionally evidence based: it compares parsed JSON fields and
normalized source text, never SHA-256 digests.  It can be run while only a
subset of the seven jobs is archived; the report then remains explicitly
``partial`` and can be rerun after more job directories arrive.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

import numpy as np


EXPECTED_JOBS = {
    "16177203.pbs101": "all_W4",
    "16177204.pbs101": "RankCal",
    "16177205.pbs101": "LocalMSE",
    "16177206.pbs101": "ScoreError",
    "16177210.pbs101": "FP32",
    "16177211.pbs101": "Random1",
    "16177212.pbs101": "Random2",
}
ALLOCATION_METHODS = {"RankCal", "LocalMSE", "ScoreError", "Random1", "Random2"}
EXPECTED_MODE_SPECS = {"all_W4": "test_all_W4.json", "FP32": "test_FP32.json"}
SIGNALS = ("rank_disagreement", "local_block_output_nmse", "planner_score_nmse")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _normal_text(path: Path) -> str:
    """Normalize only transport whitespace; preserve source content."""
    text = path.read_text(encoding="utf-8")
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).rstrip() + "\n"


def _first_difference(left: str, right: str) -> int | None:
    left_lines, right_lines = left.splitlines(), right.splitlines()
    for index, (left_line, right_line) in enumerate(zip(left_lines, right_lines), start=1):
        if left_line != right_line:
            return index
    if len(left_lines) != len(right_lines):
        return min(len(left_lines), len(right_lines)) + 1
    return None


def _check_source_files(job_dir: Path, control_dir: Path) -> Dict[str, Any]:
    required = ("screen_runner.py", "smoke_runner.py")
    optional = ("run_stage.py", "joint_fidelity.py", "probe_runner.py")
    comparisons: Dict[str, Any] = {}
    errors: List[str] = []
    for name in required + optional:
        archived = job_dir / name
        control = control_dir / name
        if not archived.exists():
            if name in required:
                errors.append(f"missing archived required source {name}")
            comparisons[name] = {"status": "missing", "required": name in required}
            continue
        if not control.is_file():
            errors.append(f"missing local control source {name}")
            comparisons[name] = {"status": "missing_control", "required": name in required}
            continue
        archived_text, control_text = _normal_text(archived), _normal_text(control)
        equal = archived_text == control_text
        comparisons[name] = {
            "status": "match" if equal else "mismatch",
            "normalized_comparison": "LF line endings; trailing whitespace removed; final newline normalized",
            "first_difference_line": None if equal else _first_difference(archived_text, control_text),
            "archived_line_count": len(archived_text.splitlines()),
            "control_line_count": len(control_text.splitlines()),
        }
        if not equal:
            errors.append(f"archived {name} differs from local control source")
    return {"passed": not errors, "files": comparisons, "errors": errors}


def _expected_target_map(freeze: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = freeze.get("test_targets")
    if not isinstance(raw, list) or len(raw) != 24:
        raise ValueError("TEST_FREEZE.test_targets must contain 24 entries")
    result: Dict[str, Dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("TEST_FREEZE target entry is not a mapping")
        episode_id = str(entry.get("episode_id"))
        if episode_id in result:
            raise ValueError(f"duplicate frozen target {episode_id}")
        result[episode_id] = {
            key: entry.get(key)
            for key in ("episode_id", "dataset_index", "env_seed", "cem_seed", "target_fingerprint")
        }
    expected_ids = {f"pilot_test:{index:03d}" for index in range(24)}
    if set(result) != expected_ids:
        raise ValueError("TEST_FREEZE target episode IDs are not pilot_test:000..023")
    return result


def _target_manifest_check(root: Path, freeze: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = root / "artifacts" / "screen" / "targets" / "episode_manifest_pilot_test.json"
    result: Dict[str, Any] = {"path": str(manifest_path.resolve()), "passed": False}
    if not manifest_path.is_file():
        result["error"] = "pilot_test target manifest is missing"
        return result
    try:
        manifest = _load_json(manifest_path)
        expected = _expected_target_map(freeze)
        entries = manifest.get("episodes")
        actual = {
            str(entry.get("episode_id")): {
                key: entry.get(key)
                for key in ("episode_id", "dataset_index", "env_seed", "cem_seed", "target_fingerprint")
            }
            for entry in entries
        }
        mismatches = [episode_id for episode_id in expected if actual.get(episode_id) != expected[episode_id]]
        result.update({
            "manifest_schema": manifest.get("schema"),
            "episode_count": len(entries) if isinstance(entries, list) else None,
            "mismatches": mismatches,
            "passed": manifest.get("schema") == "rankcal-wall-targets-v1" and not mismatches and set(actual) == set(expected),
        })
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def _summary_path(job_dir: Path) -> Path | None:
    summaries = sorted((job_dir / "evaluation" / "pilot_test").glob("*/summary.json"))
    return summaries[0] if len(summaries) == 1 else None


def _mapping_from_summary(summary: Mapping[str, Any], method: str, freeze: Mapping[str, Any]) -> Tuple[bool, Dict[str, int] | None, List[str]]:
    errors: List[str] = []
    metadata = summary.get("mode_metadata")
    if not isinstance(metadata, Mapping):
        return False, None, ["summary.mode_metadata is missing"]
    if method == "FP32":
        passed = metadata.get("mode") == "FP32" and "allocation" not in metadata
        if not passed:
            errors.append("FP32 mode_metadata is not an unquantized baseline")
        return passed, None, errors
    if method == "all_W4":
        allocation = metadata.get("allocation")
        if allocation is not None:
            errors.append("all_W4 unexpectedly contains an allocation mapping")
        passed = metadata.get("mode") == "all_W4" and metadata.get("bits") == 4 and allocation is None
        if not passed and not errors:
            errors.append("all_W4 mode_metadata does not declare bits=4")
        return passed, None, errors
    expected = freeze.get("allocations", {}).get(method)
    allocation = metadata.get("allocation")
    if not isinstance(expected, Mapping) or not isinstance(allocation, Mapping):
        return False, None, [f"missing frozen or recorded allocation for {method}"]
    actual = {str(key): int(value) for key, value in allocation.items()}
    expected_map = {str(key): int(value) for key, value in expected.items()}
    if actual != expected_map:
        errors.append(f"{method} allocation mapping differs from TEST_FREEZE")
    selected = sorted(key for key, value in actual.items() if value == 8)
    expected_selected = sorted(key for key, value in expected_map.items() if value == 8)
    if metadata.get("selected_sites") != expected_selected:
        errors.append(f"{method} selected_sites differs from frozen mapping")
    w8_counts = metadata.get("w8_counts")
    expected_counts = {
        "encoder": sum(value == 8 and key.startswith("encoder.") for key, value in expected_map.items()),
        "predictor": sum(value == 8 and key.startswith("predictor.") for key, value in expected_map.items()),
    }
    if w8_counts != expected_counts:
        errors.append(f"{method} w8_counts differs from frozen quota")
    passed = not errors
    return passed, actual, errors


def _fit_source_check(stage_spec: Mapping[str, Any], summary: Mapping[str, Any], method: str, freeze: Mapping[str, Any]) -> Dict[str, Any]:
    arguments = [str(value) for value in stage_spec.get("arguments", [])]
    source = summary.get("mode_metadata", {}).get("source") if isinstance(summary.get("mode_metadata"), Mapping) else None
    errors: List[str] = []
    if method in ALLOCATION_METHODS:
        expected_fragment = f"{freeze.get('calibration_job')}/probes/{method}.json"
        allocation_args = [value for value in arguments if "allocation-json" in value or value.endswith(f"/{method}.json")]
        if not any(expected_fragment in value or value.endswith(f"/16176985.pbs101/probes/{method}.json") for value in allocation_args):
            errors.append("stage spec allocation source is not the frozen calibration artifact")
        source_path = str(source.get("source_probe_artifact")) if isinstance(source, Mapping) else ""
        if "16176985.pbs101" not in source_path or "16176943.pbs101" in source_path or "pilot_test" in source_path:
            errors.append("summary allocation source is not calibration-only")
        if not isinstance(source, Mapping) or source.get("calibration_only") is not True:
            errors.append("summary allocation source is not marked calibration_only")
    else:
        if any("probe" in value.lower() or "allocation-json" in value for value in arguments):
            errors.append(f"{method} baseline stage unexpectedly references a probe/allocation input")
    return {"passed": not errors, "errors": errors}


def _episode_target_check(summary: Mapping[str, Any], frozen_targets: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = summary.get("episodes")
    errors: List[str] = []
    if not isinstance(rows, list) or len(rows) != 24:
        return {"passed": False, "episode_count": len(rows) if isinstance(rows, list) else None, "errors": ["summary must contain 24 episodes"]}
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("summary episode is not a mapping")
            continue
        episode_id = str(row.get("episode_id"))
        if episode_id in seen:
            errors.append(f"duplicate episode {episode_id}")
            continue
        seen.add(episode_id)
        expected = frozen_targets.get(episode_id)
        if expected is None:
            errors.append(f"unexpected episode {episode_id}")
            continue
        for key in ("dataset_index", "env_seed", "cem_seed", "target_fingerprint"):
            if row.get(key) != expected[key]:
                errors.append(f"{episode_id} {key} differs from frozen target")
    if seen != set(frozen_targets):
        errors.append("summary episode IDs do not cover all frozen targets")
    return {"passed": not errors, "episode_count": len(rows), "errors": errors}


def _job_audit(job_id: str, method: str, artifacts_root: Path, control_root: Path, freeze: Mapping[str, Any], frozen_targets: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any] | None) -> Dict[str, Any]:
    job_dir = artifacts_root / job_id
    result: Dict[str, Any] = {"job_id": job_id, "method": method, "present": job_dir.is_dir(), "passed": False}
    if not job_dir.is_dir():
        result["status"] = "missing"
        result["errors"] = ["archive directory is missing"]
        return result
    errors: List[str] = []
    stage_spec_path = job_dir / "stage_spec.json"
    stage_status_path = job_dir / "stage_status.json"
    stage_spec = _load_json(stage_spec_path) if stage_spec_path.is_file() else {}
    stage_status = _load_json(stage_status_path) if stage_status_path.is_file() else {}
    expected_spec_path = control_root / ({"all_W4": "test_all_W4.json", "FP32": "test_FP32.json"}.get(method, f"test_{method}.json"))
    expected_spec = _load_json(expected_spec_path) if expected_spec_path.is_file() else None
    spec_match = expected_spec is not None and stage_spec == expected_spec
    if not spec_match:
        errors.append("stage_spec.json differs from the frozen local test spec")
    status_passed = stage_status.get("status") == "complete" and stage_status.get("exit_code") == 0
    if not status_passed:
        errors.append("stage_status is not complete with exit_code 0")
    summary_path = _summary_path(job_dir)
    summary = _load_json(summary_path) if summary_path is not None else {}
    summary_checks = {
        "path": None if summary_path is None else str(summary_path.resolve()),
        "present": summary_path is not None,
        "status_complete": summary.get("status") == "complete" and summary.get("failed") == 0,
        "split": summary.get("split") == "pilot_test",
        "mode": summary.get("mode") == method,
        "protocol": protocol is None or summary.get("protocol") == protocol,
    }
    if not all(value for key, value in summary_checks.items() if key not in {"path", "present"}):
        errors.append("summary identity/status/protocol check failed")
    mapping_passed, mapping, mapping_errors = _mapping_from_summary(summary, method, freeze)
    errors.extend(mapping_errors)
    target_check = _episode_target_check(summary, frozen_targets)
    errors.extend(target_check["errors"])
    fit_check = _fit_source_check(stage_spec, summary, method, freeze)
    errors.extend(fit_check["errors"])
    source_check = _check_source_files(job_dir, control_root)
    errors.extend(source_check["errors"])
    result.update({
        "status": "pass" if not errors else "fail",
        "passed": not errors,
        "stage_spec": {"path": str(stage_spec_path.resolve()), "matches_control_spec": spec_match},
        "stage_status": {key: stage_status.get(key) for key in ("status", "exit_code", "stage")},
        "summary": summary_checks,
        "summary_mode_metadata": {
            "mapping_matches_freeze_or_baseline": mapping_passed,
            "allocation_mapping": mapping,
            "errors": mapping_errors,
        },
        "targets": target_check,
        "fit_source": fit_check,
        "runner_sources": source_check,
        "errors": errors,
    })
    return result


def _pool_aggregation_equivalence(root: Path) -> Dict[str, Any]:
    path = root / "artifacts" / "screen" / "artifacts" / "16176985.pbs101" / "probes" / "probes.pkl"
    result: Dict[str, Any] = {"path": str(path.resolve()), "passed": False}
    if not path.is_file():
        result["error"] = "calibration probes.pkl is missing"
        return result
    try:
        probe = _load_pickle(path)
        pools = probe["pools"]
        records = probe["records"]
        pool_to_episode = {str(pool["pool_id"]): str(pool["episode_id"]) for pool in pools}
        pool_to_point = {str(pool["pool_id"]): int(pool["mpc_point"]) for pool in pools}
        # Build explicit pool means so the equality proof does not depend on
        # the order or count of the record rows.
        rows_by_pool: MutableMapping[Tuple[str, int, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
        for row in records:
            pool_id = str(row["pool_id"])
            rows_by_pool[(str(row["group_id"]), int(row["bits"]), pool_to_episode[pool_id], pool_to_point[pool_id], pool_id)].append(row)
        max_abs = {signal: 0.0 for signal in SIGNALS}
        checked = 0
        point_pool_counts: Dict[Tuple[str, int], set[str]] = defaultdict(set)
        for pool_id, episode_id in pool_to_episode.items():
            point_pool_counts[(episode_id, pool_to_point[pool_id])].add(pool_id)
        invalid_point_counts = {
            f"{episode_id}/point{point}": len(pool_ids)
            for (episode_id, point), pool_ids in point_pool_counts.items()
            if len(pool_ids) != 2
        }
        grouped: MutableMapping[Tuple[str, int, str, int], List[Tuple[str, Dict[str, float]]]] = defaultdict(list)
        for (group_id, bits, episode_id, point, pool_id), rows in rows_by_pool.items():
            pool_mean = {signal: float(np.mean([float(row[signal]) for row in rows])) for signal in SIGNALS}
            grouped[(group_id, bits, episode_id, point)].append((pool_id, pool_mean))
        for (group_id, bits, episode_id), _ in sorted({key[:3]: None for key in grouped}.items()):
            points = sorted(point for (g, b, e, point) in grouped if (g, b, e) == (group_id, bits, episode_id))
            pool_values = []
            point_values = []
            for point in points:
                entries = grouped[(group_id, bits, episode_id, point)]
                point_values.append({signal: float(np.mean([value[signal] for _, value in entries])) for signal in SIGNALS})
                pool_values.extend(value for _, value in entries)
            for signal in SIGNALS:
                pool_mean = float(np.mean([value[signal] for value in pool_values]))
                point_mean = float(np.mean([value[signal] for value in point_values]))
                max_abs[signal] = max(max_abs[signal], abs(pool_mean - point_mean))
            checked += 1
        result.update({
            "passed": not invalid_point_counts and all(value <= 1e-12 for value in max_abs.values()),
            "checked_site_bit_episode_cells": checked,
            "max_abs_difference": max_abs,
            "point_pool_counts": invalid_point_counts,
            "condition": "each visited MPC point has exactly two pools (CEM iterations 1 and 5)",
        })
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    parser.add_argument("--freeze", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    artifacts_root = (args.artifacts_root or root / "artifacts" / "screen" / "artifacts").resolve()
    freeze_path = (args.freeze or root / "TEST_FREEZE.json").resolve()
    output_path = (args.output or root / "artifacts" / "screen" / "freeze_audit.json").resolve()
    errors: List[str] = []
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    freeze = _load_json(freeze_path)
    frozen_targets = _expected_target_map(freeze)
    target_manifest = _target_manifest_check(root, freeze)
    if not target_manifest.get("passed"):
        errors.append("frozen pilot_test target manifest check failed")
    protocol = None
    control_protocol_path = root / "screen_runner.py"
    try:
        import importlib.util
        import sys

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("rankcal_wall_audit_screen_runner", control_protocol_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            protocol = module._protocol_config()
    except Exception as exc:
        errors.append(f"could not load control protocol: {exc!r}")
    jobs = [_job_audit(job_id, method, artifacts_root, root, freeze, frozen_targets, protocol) for job_id, method in EXPECTED_JOBS.items()]
    for job in jobs:
        if not job["passed"]:
            errors.extend(f"{job['job_id']}: {error}" for error in job.get("errors", []))
    stability = _pool_aggregation_equivalence(root)
    if not stability.get("passed"):
        errors.append("calibration pool/point aggregation equivalence check failed")
    missing = [job["job_id"] for job in jobs if not job["present"]]
    complete = not missing
    report = {
        "schema": "rankcal-wall-freeze-audit-v1",
        "status": "complete" if complete else "partial",
        "audit_pass": not errors and complete,
        "freeze": str(freeze_path),
        "artifacts_root": str(artifacts_root),
        "expected_job_count": len(EXPECTED_JOBS),
        "present_job_count": len(EXPECTED_JOBS) - len(missing),
        "missing_jobs": missing,
        "target_manifest": target_manifest,
        "jobs": jobs,
        "stability_aggregation_equivalence": stability,
        "errors": errors,
        "scope": {
            "source_comparison": "normalized source text equality; no SHA-256",
            "allocation_source": "calibration freeze only; no dev/test refit evidence checked",
            "test_target_count": 24,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "audit_pass": report["audit_pass"], "present_jobs": report["present_job_count"], "missing_jobs": missing, "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()

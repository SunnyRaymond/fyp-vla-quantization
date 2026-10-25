"""Merge existing Cosmos T5 caches on a verified CPU PBS allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import pickle
import socket
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
EXPECTED_PLAN_PROTOCOL = "fastwam-vq-calibration-suites-v1"
EXPECTED_CALIBRATION_CASES = 80
EXPECTED_PERFORMANCE_CASES = 40
EXPECTED_LEGACY_INSTRUCTIONS = 289


class CachePreparationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CachePreparationError(f"Expected a JSON object at {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_cpu_allocation() -> dict[str, str]:
    if os.environ.get("COSMOS_CPU_ONLY") != "1":
        raise CachePreparationError("COSMOS_CPU_ONLY=1 is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise CachePreparationError("CPU cache preparation must not receive visible GPUs")
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    host = socket.gethostname().split(".", 1)[0]
    queue = os.environ.get("COSMOS_EXECUTION_QUEUE", "")
    if not job_id or not nodefile:
        raise CachePreparationError("PBS_JOBID and PBS_NODEFILE are required")
    if "login" in host.lower():
        raise CachePreparationError(f"Refusing login node {host}")
    node_path = Path(nodefile)
    if not node_path.is_file():
        raise CachePreparationError(f"PBS_NODEFILE is missing: {node_path}")
    allocated = {line.strip().split(".", 1)[0] for line in node_path.read_text().splitlines()
                 if line.strip()}
    if host not in allocated:
        raise CachePreparationError(f"Current host {host} is not listed in PBS_NODEFILE")
    if queue not in ('qdev', 'q1', 'q2'):
        raise CachePreparationError(f"Expected a verified CPU execution queue, got {queue!r}")

    # common.torch_module performs its own PBS/nodefile guard before importing Torch.
    import common
    torch = common.torch_module()
    return {"pbs_job_id": job_id, "host": socket.gethostname(), "queue": queue,
            "torch_version": str(torch.__version__)}


def _contained_metadata_path(data_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise CachePreparationError(f"Plan metadata path must be relative: {relative!r}")
    full = (data_root / candidate).resolve()
    try:
        full.relative_to(data_root.resolve())
    except ValueError as exc:
        raise CachePreparationError(f"Plan metadata path escapes source root: {relative!r}") from exc
    if not full.is_file():
        raise CachePreparationError(f"Plan metadata file is missing: {full}")
    return full


def collect_required_instructions(protocol: dict, asset_requirements: dict) -> dict:
    calibration = protocol.get("calibration", {})
    performance = protocol.get("performance", {})
    calibration_ids = calibration.get("case_ids", [])
    performance_ids = performance.get("case_ids", [])
    if len(calibration_ids) != EXPECTED_CALIBRATION_CASES or len(set(calibration_ids)) != EXPECTED_CALIBRATION_CASES:
        raise CachePreparationError("Frozen calibration panel must contain 80 unique case IDs")
    if len(performance_ids) != EXPECTED_PERFORMANCE_CASES or len(set(performance_ids)) != EXPECTED_PERFORMANCE_CASES:
        raise CachePreparationError("Frozen performance panel must contain 40 unique case IDs")
    if set(calibration_ids) & set(performance_ids):
        raise CachePreparationError("Calibration and performance case panels must be disjoint")

    plan_path = Path(calibration.get("plan_file", ""))
    source_root = Path(calibration.get("source_root", ""))
    if not plan_path.is_file() or not source_root.is_dir():
        raise CachePreparationError("Frozen calibration plan or source root is unavailable")
    plan = _read_json(plan_path)
    if plan.get("status") != "complete" or plan.get("protocol") != EXPECTED_PLAN_PROTOCOL:
        raise CachePreparationError("Frozen source plan is incomplete or has a different protocol")
    rows = plan.get("inputs")
    if not isinstance(rows, list):
        raise CachePreparationError("Frozen source plan has no inputs list")
    wanted = set(calibration_ids) | set(performance_ids)
    by_id: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "case_id" not in row:
            continue
        try:
            case_id = int(row["case_id"])
        except (TypeError, ValueError):
            continue
        if case_id in wanted:
            if case_id in by_id:
                raise CachePreparationError(f"Source plan repeats case ID {case_id}")
            by_id[case_id] = row
    absent = sorted(wanted - by_id.keys())
    if absent:
        raise CachePreparationError(f"Source plan is missing requested case IDs: {absent}")

    data_root = source_root / "data"
    descriptions: list[str] = []
    for case_id in sorted(wanted):
        row = by_id[case_id]
        metadata_path = _contained_metadata_path(data_root, row.get("metadata_file", ""))
        metadata = _read_json(metadata_path)
        if (metadata.get("domain") != "original"
                or int(metadata.get("case_id", -1)) != case_id
                or metadata.get("suite") != row.get("suite")
                or int(metadata.get("task_id", -1)) != int(row.get("task_id", -2))):
            raise CachePreparationError(f"Frozen source metadata binding differs for case {case_id}")
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            raise CachePreparationError(f"Case {case_id} lacks a non-empty exact description")
        descriptions.append(description)

    legacy = asset_requirements.get("instructions")
    if not isinstance(legacy, list) or len(legacy) != EXPECTED_LEGACY_INSTRUCTIONS:
        raise CachePreparationError(
            f"language_requirements.instructions must contain {EXPECTED_LEGACY_INSTRUCTIONS} entries")
    if any(not isinstance(value, str) or not value.strip() for value in legacy):
        raise CachePreparationError("language_requirements.instructions contains an invalid string")

    required = set(legacy) | set(descriptions)
    return {
        "instructions": required,
        "calibration_case_count": len(calibration_ids),
        "performance_case_count": len(performance_ids),
        "metadata_description_count": len(descriptions),
        "metadata_description_unique_count": len(set(descriptions)),
        "legacy_instruction_count": len(legacy),
        "legacy_instruction_unique_count": len(set(legacy)),
        "required_exact_keys": sorted(required),
    }


def combine_caches(merged_existing: dict, official: dict) -> dict:
    """Keep existing merged values on collisions; use official values for gaps."""
    result = dict(merged_existing)
    for key, value in official.items():
        result.setdefault(key, value)
    return result


def _torch_load(path_or_stream, torch):
    try:
        return torch.load(path_or_stream, map_location="cpu", weights_only=False)
    except TypeError:
        if hasattr(path_or_stream, "seek"):
            path_or_stream.seek(0)
        return torch.load(path_or_stream, map_location="cpu")


class _CPUUnpickler(pickle.Unpickler):
    """Load ordinary pickle caches while remapping embedded Torch storages to CPU."""
    def __init__(self, stream, torch):
        super().__init__(stream)
        self._torch = torch

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            torch = self._torch

            def load_storage(data):
                return _torch_load(io.BytesIO(data), torch)

            return load_storage
        return super().find_class(module, name)


def load_cache_cpu(path: Path, torch) -> dict:
    if not path.is_file():
        raise CachePreparationError(f"Cache source is missing: {path}")
    if zipfile.is_zipfile(path):
        value = _torch_load(path, torch)
    else:
        with path.open("rb") as stream:
            try:
                value = _CPUUnpickler(stream, torch).load()
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError):
                # Also accept a legacy torch.save serialization if the source uses one.
                value = _torch_load(path, torch)
    if not isinstance(value, dict):
        raise CachePreparationError(f"Cache must contain a dictionary: {path}")
    if any(not isinstance(key, str) for key in value):
        raise CachePreparationError(f"Cache contains a non-string language key: {path}")
    for item in value.values():
        _require_cpu_tensors(item, torch, path)
    return value


def _require_cpu_tensors(value, torch, path: Path) -> None:
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise CachePreparationError(f"Cache tensor was not mapped to CPU: {path}")
    elif isinstance(value, dict):
        for child in value.values():
            _require_cpu_tensors(child, torch, path)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_cpu_tensors(child, torch, path)


def _write_pickle_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _base_summary(protocol: dict, allocation: dict[str, str], protocol_path: Path) -> dict:
    return {
        "protocol": protocol.get("protocol"),
        "status": "running",
        "complete": False,
        "pbs_job_id": allocation["pbs_job_id"],
        "job_name": "cpvq24lang",
        "host": allocation["host"],
        "execution_queue": allocation["queue"],
        "allocation": {"pbs_job_id": allocation["pbs_job_id"], "host": allocation["host"],
                       "execution_queue": allocation["queue"], "cpu_only": True,
                       "torch_version": allocation["torch_version"]},
        "protocol_file": str(protocol_path),
        "source_paths": {},
    }


def run(protocol_path: Path, root: Path, asset_root: Path, torch, allocation: dict[str, str]) -> int:
    verified = require_cpu_allocation()
    if verified["pbs_job_id"] != allocation["pbs_job_id"]:
        raise CachePreparationError("PBS allocation identity changed before cache loading")
    protocol = _read_json(protocol_path)
    if root.resolve() != Path(protocol["remote_root"]).resolve():
        raise CachePreparationError("Output root differs from the frozen protocol remote_root")
    if asset_root.resolve() != Path(protocol["asset_root"]).resolve():
        raise CachePreparationError("ASSET_ROOT differs from the frozen protocol asset_root")

    output_dir = root / "results" / "language_cache"
    output_path = output_dir / "embeddings.pkl"
    summary_path = output_dir / "summary.json"
    requirements_path = asset_root / "language_requirements.json"
    try:
        requirements = _read_json(requirements_path)
        official_path = Path(requirements["source_cache"])
        if not official_path.is_absolute():
            official_path = asset_root / official_path
        merged_path = asset_root / "results" / "text_embeddings.pkl"
        sources = {"official_source_cache": official_path, "existing_merged_cache": merged_path}
        if output_path.resolve() in {path.resolve() for path in sources.values()}:
            raise CachePreparationError("Output cache must be disjoint from both read-only sources")
        plan_info = collect_required_instructions(protocol, requirements)
        summary = _base_summary(protocol, allocation, protocol_path)
        summary["source_paths"] = {key: str(value) for key, value in sources.items()}
        summary["panels"] = {
            "calibration_cases": plan_info["calibration_case_count"],
            "performance_cases": plan_info["performance_case_count"],
            "metadata_descriptions": plan_info["metadata_description_count"],
            "metadata_description_unique_count": plan_info["metadata_description_unique_count"],
            "legacy_instruction_entries": plan_info["legacy_instruction_count"],
            "legacy_instruction_unique_count": plan_info["legacy_instruction_unique_count"],
        }
        summary["required_exact_keys"] = plan_info["required_exact_keys"]
        summary["required_exact_key_count"] = len(plan_info["instructions"])

        official = load_cache_cpu(official_path, torch)
        merged_existing = load_cache_cpu(merged_path, torch)
        required = plan_info["instructions"]
        combined = combine_caches(merged_existing, official)
        missing = sorted(required - combined.keys())
        summary["sources_inventory"] = {
            "official_source_cache": {
                "key_count": len(official),
                "required_key_coverage": len(required & official.keys()),
                "required_key_missing_count": len(required - official.keys()),
            },
            "existing_merged_cache": {
                "key_count": len(merged_existing),
                "required_key_coverage": len(required & merged_existing.keys()),
                "required_key_missing_count": len(required - merged_existing.keys()),
            },
        }
        summary["merge"] = {
            "precedence": "existing_merged_cache_then_official_source_cache_for_missing_keys",
            "combined_key_count": len(combined),
            "output_path": str(output_path),
            "missing_required_keys": missing,
            "missing_required_key_count": len(missing),
        }

        # Match Cosmos' official lock filename: <pickle path>.lock.
        from filelock import FileLock
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(output_path) + ".lock")
        with FileLock(str(lock_path), timeout=30):
            _write_pickle_atomic(output_path, combined)
        summary["merge"]["output_written"] = True
        summary["status"] = "complete" if not missing else "failed"
        summary["complete"] = not missing
        if missing:
            summary["error"] = "Required exact language keys are absent from both existing caches"
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(summary_path, summary)
        print(f"LANGUAGE_CACHE_STATUS={summary['status']} required={len(required)} "
              f"combined={len(combined)} missing={len(missing)} output={output_path}", flush=True)
        return 0 if not missing else 1
    except Exception as exc:
        failure = locals().get("summary")
        if not isinstance(failure, dict):
            failure = _base_summary(protocol, allocation, protocol_path)
            failure["source_paths"] = {
                "requirements": str(requirements_path),
                "official_source_cache": str(locals().get("official_path", "")),
                "existing_merged_cache": str(locals().get("merged_path", "")),
                "output": str(output_path),
            }
        failure.update(status="failed", complete=False,
                       error=f"{type(exc).__name__}: {exc}",
                       finished_utc=datetime.now(timezone.utc).isoformat())
        try:
            _atomic_json(summary_path, failure)
        except Exception as summary_exc:
            print(f"LANGUAGE_CACHE_SUMMARY_ERROR={type(summary_exc).__name__}: {summary_exc}",
                  file=sys.stderr, flush=True)
        print(f"LANGUAGE_CACHE_ERROR={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()

    # Do not import Torch or unpickle any cache until the allocation gate passes.
    allocation = require_cpu_allocation()
    protocol = _read_json(args.protocol)
    root = args.root or Path(os.environ.get("COSMOS_ROOT", protocol["remote_root"]))
    asset_root = args.asset_root or Path(os.environ.get("ASSET_ROOT", protocol["asset_root"]))
    import common
    torch = common.torch_module()
    return run(args.protocol, root, asset_root, torch, allocation)


if __name__ == "__main__":
    raise SystemExit(main())

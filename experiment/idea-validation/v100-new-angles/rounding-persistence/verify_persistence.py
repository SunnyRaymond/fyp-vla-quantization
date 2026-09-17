"""Independent CPU replay and receipt verification for persistence screen.

The module loads no model. It checks the saved raw arithmetic, binds every
common-path hash to the saved FP trajectory, and verifies explicit producer
receipts. It must run inside a real CCDS SLURM allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple


RAW_SCHEMA = "rounding-persistence-raw-v1"
SCREEN_SCHEMA = "rounding-persistence-screen-v1"
VERIFICATION_SCHEMA = "rounding-persistence-verification-v1"
INPUT_SCHEMA = "rounding-persistence-raw-input-manifest-v1"
IDENTITY_SCHEMA = "smolvla-cpu-preparation-identity-v1"
ARM_NAMES = ("FP", "RTN", "F0", "F1", "F2", "C0", "C1", "C2")
SAMPLE_COUNT, NOISE_COUNT, DRAW_COUNT = 6, 2, 3
HORIZON, PADDED_ACTION_DIM, PHYSICAL_ACTION_DIM, VELOCITY_STEPS = 50, 32, 7, 10
EXPERT_LINEAR_COUNT = 112
NOISE_SEEDS, DRAW_SEEDS = (2201, 2202), (2101, 2102, 2103)
MAX_SECONDS = 300.0
EXPECTED_MANIFEST_SHA256 = "71243c83702ada092481abb2772787ebfc01774d8b92d63c4a5812e1771a04ef"
EXPECTED_BASE_IDENTITY_SHA256 = "be4a49ebe588a49a29bd26ed98b8a01a648a247e66d45e12a935ac7d8d0c4e64"
EXPECTED_FLOW_HELPER_SHA256 = "ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9"
EXPECTED_PROTOCOL_SHA256 = "0b9c9129ac8c81ff01fc26a304bc9ebd57ea25eeb0999505557198db713ac931"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.casefold())


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _scalar_string(value: Any, label: str) -> str:
    if getattr(value, "ndim", None) != 0 or getattr(getattr(value, "dtype", None), "kind", None) not in {"U", "S"}:
        raise ValueError(f"{label} must be a scalar string array")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).expanduser().resolve() == expected.resolve()


def _file_receipt(
    record: Mapping[str, Any], label: str, checks: MutableMapping[str, bool], receipts: MutableMapping[str, Any], cache: MutableMapping[str, str],
) -> bool:
    path_value, expected = record.get("path"), record.get("sha256")
    if not isinstance(path_value, str) or not _is_sha(expected):
        checks[label] = False
        receipts[label] = {"path": path_value, "expected_sha256": expected, "error": "malformed receipt"}
        return False
    path = Path(path_value).expanduser().resolve()
    try:
        actual = cache.get(str(path))
        if actual is None:
            if not path.is_file():
                raise FileNotFoundError(str(path))
            actual = sha(path)
            cache[str(path)] = actual
        checks[label] = actual.casefold() == expected.casefold()
        receipts[label] = {"path": str(path), "expected_sha256": expected.casefold(), "actual_sha256": actual}
        return checks[label]
    except OSError as exc:
        checks[label] = False
        receipts[label] = {"path": str(path), "expected_sha256": expected.casefold(), "error": str(exc)}
        return False


def _expected_schedule() -> Any:
    import numpy as np

    frozen = np.asarray([[draw] * VELOCITY_STEPS for draw in range(DRAW_COUNT)], dtype=np.int64)
    cyclic = np.asarray([[(draw + step) % DRAW_COUNT for step in range(VELOCITY_STEPS)] for draw in range(DRAW_COUNT)], dtype=np.int64)
    return np.stack([frozen, cyclic], axis=0)


def scientific_replay(data: Mapping[str, Any], np: Any) -> Dict[str, Any]:
    """Recompute the frozen E/S/D metrics from raw arrays only."""
    x, v, common, noise, dt, schedule = (data[k] for k in ("raw_x", "raw_v", "common_q_v", "noise", "dt", "schedule"))
    shapes = {"raw_x": (6, 2, 8, 11, 50, 32), "raw_v": (6, 2, 8, 10, 50, 32), "common_q_v": (6, 2, 3, 10, 50, 32), "noise": (2, 50, 32), "dt": (10,), "times": (10,), "schedule": (2, 3, 10)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid shape/nonfinite {key}")
    expected_schedule = _expected_schedule()
    if schedule.dtype.kind not in "iu" or not np.array_equal(schedule, expected_schedule):
        raise ValueError("Schedule differs from frozen 3-draw coupling")
    checks = {
        "same_initial_noise_every_trajectory": bool(np.array_equal(x[:, :, :, 0], np.broadcast_to(noise[None, :, None], (6, 2, 8, 50, 32)))),
        "ten_step_time_grid": bool(np.allclose(data["times"], 1 - np.arange(10) / 10, atol=1e-7, rtol=0)),
        "fixed_Euler_dt": bool(np.allclose(dt, -0.1, atol=1e-8, rtol=0)),
        "schedule_multiset_exact": bool(all(np.array_equal(np.sort(schedule[f, :, t]), np.arange(3)) for f in range(2) for t in range(10))),
    }
    checks["all_trajectories_Euler_recurrence"] = bool(np.allclose(x[:, :, :, 1:], x[:, :, :, :-1] + v * dt[None, None, None, :, None, None], atol=1e-6, rtol=0))
    endpoint = x[:, :, :, -1, :8, :7].astype(np.float64)
    endpoint_mse = np.square(endpoint - endpoint[:, :, :1]).mean(axis=(-1, -2))
    e_frozen, e_cyclic, e_rtn = endpoint_mse[:, :, 2:5].mean((1, 2)), endpoint_mse[:, :, 5:8].mean((1, 2)), endpoint_mse[:, :, 1].mean(1)
    error = common[:, :, :, :, :8, :7].astype(np.float64) - v[:, :, 0, None, :, :8, :7].astype(np.float64)
    weighted = error * dt.astype(np.float64)[None, None, None, :, None, None]
    forcing, diagonal, marginal = np.empty((6, 2)), np.empty((6, 2)), []
    for family in range(2):
        assigned = np.stack([np.stack([weighted[:, :, int(schedule[family, draw, step]), step] for step in range(10)], axis=2) for draw in range(3)], axis=2)
        forcing[:, family] = np.square(assigned.sum(axis=3)).mean(axis=(1, 2, 3, 4))
        diagonal[:, family] = np.square(assigned).sum(axis=3).mean(axis=(1, 2, 3, 4))
        marginal.append(np.square(assigned).mean(axis=(2, 4, 5)))
    checks["common_path_stepwise_marginal_exact"] = bool(np.allclose(marginal[0], marginal[1], atol=1e-12, rtol=1e-10))
    checks["common_path_diagonal_energy_matched"] = bool(np.allclose(diagonal[:, 0], diagonal[:, 1], atol=1e-12, rtol=1e-10))
    binding = (e_frozen > 1e-12) & (forcing[:, 0] > 1e-12)
    joint = binding & (forcing[:, 1] <= 0.75 * forcing[:, 0]) & (e_cyclic <= 0.90 * e_frozen)
    numeric = "implementation_inconclusive" if not all(checks.values()) else "inconclusive_degenerate" if not bool(binding.all()) else "scope_limited_preliminary_go" if int(joint.sum()) >= 5 else "mechanism_no_go"
    states = []
    for state in range(6):
        states.append({"state_index": state, "E_frozen": float(e_frozen[state]), "E_cyclic": float(e_cyclic[state]), "E_RTN": float(e_rtn[state]), "S_frozen": float(forcing[state, 0]), "S_cyclic": float(forcing[state, 1]), "D_frozen": float(diagonal[state, 0]), "D_cyclic": float(diagonal[state, 1]), "cross_term_frozen": float(forcing[state, 0] - diagonal[state, 0]), "cross_term_cyclic": float(forcing[state, 1] - diagonal[state, 1]), "endpoint_gain": float(1 - e_cyclic[state] / e_frozen[state]) if e_frozen[state] > 0 else None, "forcing_gain": float(1 - forcing[state, 1] / forcing[state, 0]) if forcing[state, 0] > 0 else None, "nondegenerate": bool(binding[state]), "joint_pass": bool(joint[state])})
    return {"numeric_decision": numeric, "raw_science_checks": checks, "nondegenerate_count": int(binding.sum()), "joint_pass_count": int(joint.sum()), "mean_E_frozen": float(e_frozen.mean()), "mean_E_cyclic": float(e_cyclic.mean()), "mean_E_RTN": float(e_rtn.mean()), "states": states}


def _load_raw(input_dir: Path, np: Any) -> Tuple[Dict[str, Any], Path]:
    path = input_dir / "raw_persistence.npz"
    required = {"schema", "arm_names", "raw_x", "raw_v", "actions", "common_q_v", "common_x_sha256", "actual_times", "noise", "times", "dt", "schedule", "actual_schedule", "completed", "episode_ids", "metadata_json"}
    with np.load(path, allow_pickle=False) as packed:
        missing = required - set(packed.files)
        if missing:
            raise ValueError(f"raw archive is missing keys: {sorted(missing)}")
        data = {key: packed[key].copy() for key in packed.files}
    if _scalar_string(data["schema"], "raw schema") != RAW_SCHEMA or data["arm_names"].tolist() != list(ARM_NAMES):
        raise ValueError("raw schema or arm order mismatch")
    shapes = {"raw_x": (6, 2, 8, 11, 50, 32), "raw_v": (6, 2, 8, 10, 50, 32), "actions": (6, 2, 8, 50, 7), "common_q_v": (6, 2, 3, 10, 50, 32), "common_x_sha256": (6, 2, 3, 10), "actual_times": (6, 2, 8, 10), "noise": (2, 50, 32), "times": (10,), "dt": (10,), "schedule": (2, 3, 10), "actual_schedule": (6, 2, 8, 10), "completed": (6, 2), "episode_ids": (6,)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"Wrong {key} shape: {data[key].shape}")
    dtypes = {k: "float32" for k in ("raw_x", "raw_v", "actions", "common_q_v", "actual_times", "noise", "times", "dt")}
    dtypes.update({"schedule": "int64", "actual_schedule": "int8", "completed": "bool"})
    for key, dtype in dtypes.items():
        if data[key].dtype != np.dtype(dtype):
            raise ValueError(f"Unexpected {key} dtype: {data[key].dtype}")
    for key in dtypes:
        if dtypes[key] == "float32" and not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite raw field: {key}")
    if not bool(data["completed"].all()) or data["common_x_sha256"].dtype.kind not in {"U", "S"} or data["episode_ids"].dtype.kind not in {"U", "S"}:
        raise ValueError("raw archive is incomplete or has invalid string fields")
    metadata = json.loads(_scalar_string(data["metadata_json"], "raw metadata_json"))
    if not isinstance(metadata, dict) or metadata.get("schema") != RAW_SCHEMA:
        raise ValueError("raw metadata schema mismatch")
    data["metadata"] = metadata
    return data, path


def _common_hash_check(data: Mapping[str, Any], np: Any) -> Dict[str, Any]:
    mismatch, mismatch_count = [], 0
    for state in range(6):
        for noise in range(2):
            for draw in range(3):
                for step in range(10):
                    expected = hashlib.sha256(np.ascontiguousarray(data["raw_x"][state, noise, 0, step], dtype=np.float32).tobytes()).hexdigest()
                    actual = data["common_x_sha256"][state, noise, draw, step]
                    actual = actual.decode("utf-8") if isinstance(actual, bytes) else str(actual)
                    if actual.casefold() != expected:
                        mismatch_count += 1
                        if len(mismatch) < 4:
                            mismatch.append({"state": state, "noise": noise, "draw": draw, "step": step, "expected": expected, "actual": actual})
    return {"common_x_sha256_matches_fp_trajectory": mismatch_count == 0, "common_x_hash_count": 360, "common_x_hash_mismatch_count": mismatch_count, "common_x_hash_mismatch_examples": mismatch, "hash_formula": "sha256(contiguous float32 bytes of raw_x[state,noise,FP,step])"}


def _schedule_time_check(data: Mapping[str, Any], np: Any) -> Dict[str, Any]:
    expected = _expected_schedule()
    actual = np.full((6, 2, 8, 10), -1, dtype=np.int8)
    actual[:, :, 2:5] = expected[0].astype(np.int8)
    actual[:, :, 5:8] = expected[1].astype(np.int8)
    times = np.broadcast_to(data["times"][None, None, None, :], (6, 2, 8, 10))
    return {"actual_schedule_matches_frozen_cyclic_definition": bool(np.array_equal(data["actual_schedule"], actual)), "FP_RTN_actual_schedule_sentinel_minus_one": bool(np.all(data["actual_schedule"][:, :, :2] == -1)), "actual_times_match_frozen_grid": bool(np.allclose(data["actual_times"], times, atol=1e-6, rtol=0)), "actual_times_max_abs_error": float(np.max(np.abs(data["actual_times"].astype(np.float64) - times.astype(np.float64))))}


def _input_receipts(engineering: Mapping[str, Any], data: Mapping[str, Any], cache: MutableMapping[str, str]) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    checks, receipts = {}, {}
    info = engineering.get("input_manifest")
    if not isinstance(info, Mapping) or not isinstance(info.get("path"), str) or not _is_sha(info.get("sha256")):
        raise ValueError("engineering.input_manifest is malformed")
    manifest_path, manifest_expected = Path(info["path"]).expanduser().resolve(), info["sha256"]
    manifest_actual = sha(manifest_path)
    checks["input_manifest_hash"] = manifest_actual.casefold() == manifest_expected.casefold()
    checks["input_manifest_frozen_pin"] = manifest_actual.casefold() == EXPECTED_MANIFEST_SHA256
    receipts["manifest"] = {"path": str(manifest_path), "recorded_sha256": manifest_expected, "actual_sha256": manifest_actual}
    manifest = _read_json(manifest_path, "input manifest")
    samples = manifest.get("samples")
    checks["input_manifest_schema"] = manifest.get("schema") == INPUT_SCHEMA and isinstance(samples, list) and len(samples) == 6
    identity_path, identity_expected = Path(info["identity_path"]).expanduser().resolve(), info.get("identity_sha256")
    if not _is_sha(identity_expected):
        raise ValueError("base identity SHA receipt is malformed")
    identity_actual = sha(identity_path)
    checks["base_identity_hash"] = identity_actual.casefold() == identity_expected.casefold()
    checks["base_identity_frozen_pin"] = identity_actual.casefold() == EXPECTED_BASE_IDENTITY_SHA256
    receipts["base_identity"] = {"path": str(identity_path), "recorded_sha256": identity_expected, "actual_sha256": identity_actual}
    identity = _read_json(identity_path, "base identity")
    checkpoint, dataset = identity.get("checkpoint"), identity.get("dataset")
    checks["base_identity_schema_and_pins"] = identity.get("schema") == IDENTITY_SCHEMA and isinstance(checkpoint, Mapping) and checkpoint.get("repo") == "lerobot/smolvla_libero" and checkpoint.get("revision") == "31d453f7edd78c839a8bbc39744a292686daf0de" and isinstance(dataset, Mapping) and dataset.get("repo") == "lerobot/libero" and dataset.get("revision") == "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
    checks["manifest_identity_binding"] = manifest.get("base_identity_path") == str(identity_path) and str(manifest.get("base_identity_sha256", "")).casefold() == identity_actual.casefold()
    sample_ids, sample_hashes = [], []
    for index, sample in enumerate(samples if isinstance(samples, list) else []):
        if not isinstance(sample, Mapping) or not isinstance(sample.get("sample_path"), str) or not _is_sha(sample.get("sha256")):
            raise ValueError(f"input sample receipt {index} is malformed")
        path = (manifest_path.parent / sample["sample_path"]).expanduser().resolve()
        actual = sha(path)
        sample_hashes.append(actual)
        episode = sample.get("dataset_episode_index", sample.get("episode_index"))
        if episode is None:
            raise ValueError(f"input sample {index} lacks episode index")
        sample_ids.append(f"task{int(sample['task_index'])}:episode{int(episode)}:frame{int(sample['frame_index'])}")
        checks[f"sample_{index}_hash"] = actual.casefold() == str(sample["sha256"]).casefold()
    checks["input_sample_hash_receipts"] = sample_hashes == [str(v).casefold() for v in info.get("sample_sha256", [])]
    checks["input_episode_ids_receipt"] = sample_ids == [str(v) for v in info.get("episode_ids", [])] == data["episode_ids"].tolist()
    receipts["samples"] = {"count": len(sample_ids), "sha256": sample_hashes, "episode_ids": sample_ids}
    return checks, receipts


def _snapshot_receipts(input_dir: Path, engineering: Mapping[str, Any], checks: MutableMapping[str, bool], receipts: MutableMapping[str, Any], cache: MutableMapping[str, str]) -> None:
    info = engineering.get("snapshots")
    if not isinstance(info, Mapping) or not isinstance(info.get("path"), str) or not _is_sha(info.get("sha256")):
        checks["snapshot_receipt"] = False
        return
    path = Path(info["path"]).expanduser().resolve()
    actual = sha(path)
    checks["snapshot_receipt"] = actual.casefold() == info["sha256"].casefold()
    checks["snapshot_path_in_gpu_artifact"] = path.parent == input_dir.resolve()
    receipts["snapshot"] = {"path": str(path), "recorded_sha256": info["sha256"], "actual_sha256": actual}
    if not checks["snapshot_receipt"]:
        return
    try:
        import torch

        record = torch.load(path, map_location="cpu", weights_only=True)
        modules = record.get("modules") if isinstance(record, Mapping) else None
        owners = [item for item in modules.values() if isinstance(item, Mapping) and "alias_of" not in item] if isinstance(modules, Mapping) else []
        aliases = [item for item in modules.values() if isinstance(item, Mapping) and "alias_of" in item] if isinstance(modules, Mapping) else []
        module_names = set(modules) if isinstance(modules, Mapping) else set()
        alias_targets_ok = all(isinstance(item.get("alias_of"), str) and item["alias_of"] in module_names for item in aliases)
        owner_receipts = all(isinstance(item.get("fp_weight"), torch.Tensor) and str(item["fp_weight"].dtype) == "torch.float32" and torch.isfinite(item["fp_weight"]).all() and all(key in item for key in ("rtn_codes", "scales", "draw_codes")) for item in owners)
        checks["snapshot_record_schema"] = isinstance(record, Mapping) and record.get("schema") == "rounding-persistence-snapshot-v1" and record.get("bits") == 4 and record.get("qmax") == 7 and record.get("seeds") == list(DRAW_SEEDS) and record.get("module_count") == 112 and len(owners) + len(aliases) == 112 and record.get("unique_weight_count") == len(owners)
        checks["snapshot_weight_receipts"] = bool(owner_receipts and alias_targets_ok)
        checks["snapshot_map_digest_receipt"] = isinstance(record.get("map_digests"), Mapping) and dict(record["map_digests"]) == dict(info.get("map_digests", {})) and set(record["map_digests"]) == {"RTN", "F0", "F1", "F2"}
        receipts["snapshot_record"] = {"module_count": len(modules) if isinstance(modules, Mapping) else None, "unique_weight_count": len(owners), "alias_count": len(aliases), "map_digests": dict(record.get("map_digests", {})) if isinstance(record, Mapping) and isinstance(record.get("map_digests"), Mapping) else None}
    except Exception as exc:
        checks["snapshot_record_schema"] = checks["snapshot_weight_receipts"] = checks["snapshot_map_digest_receipt"] = False
        receipts["snapshot_record"] = {"error": f"{type(exc).__name__}: {exc}"}


def _verify_artifact(args: argparse.Namespace, allocation: Mapping[str, Any], np: Any) -> Dict[str, Any]:
    started, input_dir, output_dir = time.monotonic(), args.input.resolve(), args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    def deadline(label: str) -> None:
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError(f"verifier deadline reached at {label}")
    engineering_path, summary_path = input_dir / "engineering.json", input_dir / "summary.json"
    engineering = _read_json(engineering_path, "engineering receipt")
    expected_completed = [[True, True] for _ in range(6)]
    if engineering.get("status") != "complete" or engineering.get("completed") != expected_completed:
        status = "inconclusive_budget" if engineering.get("status") == "running" else "implementation_inconclusive"
        return {"schema": VERIFICATION_SCHEMA, "allocation": dict(allocation), "input_job_dir": str(input_dir), "output_job_dir": str(output_dir), "engineering_sha256": sha(engineering_path), "checks": {"engineering_complete": False}, "receipts": {"engineering_status": engineering.get("status"), "completed": engineering.get("completed")}, "numeric_decision": status, "decision": status, "independence_boundary": "Partial GPU arrays are never interpreted as science; this status records only the allocation/output boundary.", "scope": "GPU artifact is incomplete; no scientific metrics were computed."}
    data, raw_path = _load_raw(input_dir, np)
    deadline("raw load")
    summary = _read_json(summary_path, "screen summary")
    checks: Dict[str, bool] = {}
    receipts: Dict[str, Any] = {}
    checks["engineering_complete"] = engineering.get("schema") == SCREEN_SCHEMA and engineering.get("status") == "complete" and engineering.get("completed") == expected_completed
    checks["summary_complete"] = summary.get("schema") == SCREEN_SCHEMA and summary.get("status") == "complete" and summary.get("raw_schema") == RAW_SCHEMA
    checks["summary_raw_hash"] = _same_path(summary.get("raw_npz"), raw_path) and summary.get("raw_sha256") == sha(raw_path)
    checks["summary_engineering_hash"] = _same_path(summary.get("engineering_json"), engineering_path) and summary.get("engineering_json") == str(engineering_path)
    info, metadata = engineering.get("input_manifest", {}), data["metadata"]
    checks["raw_metadata_receipt_binding"] = metadata.get("manifest_path") == info.get("path") and metadata.get("manifest_sha256") == info.get("sha256") and metadata.get("identity_sha256") == info.get("identity_sha256") and metadata.get("schedule_definition") == "frozen[d,t]=d; cyclic[d,t]=(d+t)%3" and metadata.get("physical_readout") == "first 7 action coordinates; common path retains padded 32D velocity" and metadata.get("common_x_sha256_formula") == "sha256(contiguous float32 bytes of exact x_t captured immediately before each denoise_step)"
    params = engineering.get("parameters", {})
    expected_params = {"arms": list(ARM_NAMES), "draw_seeds": list(DRAW_SEEDS), "noise_seeds": list(NOISE_SEEDS), "noise_shape": [1, 50, 32], "raw_x_shape": [6, 2, 8, 11, 50, 32], "raw_v_shape": [6, 2, 8, 10, 50, 32], "action_shape": [6, 2, 8, 50, 7], "common_q_v_shape": [6, 2, 3, 10, 50, 32], "common_x_sha256_shape": [6, 2, 3, 10], "actual_times_shape": [6, 2, 8, 10], "expert_linear_count": 112, "dt": -0.1, "bits": 4, "qmax": 7}
    checks["engineering_frozen_parameters"] = all(params.get(k) == v for k, v in expected_params.items())
    checks["engineering_manual_fp_gate"] = engineering.get("manual_fp_gate", {}).get("allclose_atol_1e-6_rtol_0") is True
    checks["engineering_rtn_negative_control"] = engineering.get("rtn_schedule_negative_control", {}).get("allclose_atol_1e-6_rtol_0") is True
    cache_digests, expected_cache_keys = engineering.get("prefix_cache_digests", {}), {f"state{s}:noise{n}" for s in range(6) for n in range(2)}
    checks["prefix_cache_receipts"] = isinstance(cache_digests, Mapping) and set(cache_digests) == expected_cache_keys and all(_is_sha(v) for v in cache_digests.values())
    binding = engineering.get("expert_binding", {})
    checks["expert_binding_receipt"] = binding.get("count") == 112 and binding.get("weight_binding") == "actual module.weight; no state_dict key rewrite" and _is_sha(binding.get("nonexpert_state_digest_before"))
    checks["weight_restore_receipt"] = engineering.get("weight_restore") == "exact after every arm and final FP restore"
    runtime, gpu = engineering.get("runtime_identity", {}), engineering.get("gpu", {})
    checks["runtime_recorded_FP32"] = runtime.get("dtype") == "float32" and runtime.get("device") == "cuda:0" and runtime.get("load_vlm_weights") is False and runtime.get("compile_model") is False and runtime.get("rtc_config") is None and runtime.get("rtc_processor") is None and runtime.get("model_rtc_processor") is None
    checks["recorded_V100_runtime"] = isinstance(gpu, Mapping) and "v100" in str(gpu.get("name", "")).casefold() and gpu.get("device_index") is not None and gpu.get("total_memory_bytes", 0) >= 30 * 1024**3
    cache: Dict[str, str] = {}
    flow_helper = engineering.get("flow_helper")
    if isinstance(flow_helper, Mapping):
        _file_receipt(flow_helper, "flow_helper_actual_hash", checks, receipts, cache)
        checks["flow_helper_frozen_pin"] = receipts.get("flow_helper_actual_hash", {}).get("actual_sha256") == EXPECTED_FLOW_HELPER_SHA256
    else:
        checks["flow_helper_actual_hash"] = checks["flow_helper_frozen_pin"] = False
    input_checks, input_receipts = _input_receipts(engineering, data, cache)
    checks.update(input_checks)
    receipts["input"] = input_receipts
    deadline("input receipts")
    source = engineering.get("source_identity")
    source_checks, source_receipts = {}, {}
    if isinstance(source, Mapping):
        for name, record in source.items():
            if isinstance(record, Mapping) and "path" in record:
                _file_receipt(record, f"source_{name}", source_checks, source_receipts, cache)
        checks.update(source_checks)
    checks["runtime_source_receipts_present"] = isinstance(source, Mapping) and runtime.get("source_identity") == source and len(source_checks) >= 3
    receipts["source"] = source_receipts
    for key in ("checkpoint", "model_config", "vlm_config"):
        record = runtime.get(key) if isinstance(runtime, Mapping) else None
        if isinstance(record, Mapping):
            _file_receipt(record, f"runtime_{key}", checks, receipts, cache)
        else:
            checks[f"runtime_{key}"] = False
    processor_files = runtime.get("processor_files", {}) if isinstance(runtime, Mapping) else {}
    processor_count = 0
    if isinstance(processor_files, Mapping):
        for group in ("policy_files", "vlm_files"):
            for index, item in enumerate(processor_files.get(group, [])):
                if isinstance(item, Mapping):
                    actual_record = item.get("actual", item)
                    if isinstance(actual_record, Mapping):
                        _file_receipt(actual_record, f"processor_{group}_{index}", checks, receipts, cache)
                        processor_count += 1
                        if _is_sha(item.get("cpu_downloaded_sha256")):
                            checks[f"processor_{group}_{index}_cpu_hash"] = item["cpu_downloaded_sha256"].casefold() == receipts[f"processor_{group}_{index}"].get("actual_sha256", "").casefold()
    checks["processor_receipts_present"] = processor_count >= 3
    protocol = engineering.get("protocol_identity")
    if isinstance(protocol, Mapping):
        _file_receipt(protocol, "protocol_actual_hash", checks, receipts, cache)
        checks["protocol_frozen_pin"] = receipts.get("protocol_actual_hash", {}).get("actual_sha256") == EXPECTED_PROTOCOL_SHA256
    else:
        checks["protocol_actual_hash"] = checks["protocol_frozen_pin"] = False
    dataset_parent = engineering.get("dataset_parent_identity", {})
    checks["dataset_parent_identity_receipt"] = isinstance(dataset_parent, Mapping) and _same_path(dataset_parent.get("base_identity_path"), Path(info.get("identity_path", "")).expanduser()) and dataset_parent.get("base_identity_sha256") == info.get("identity_sha256")
    if isinstance(dataset_parent, Mapping):
        extension_path, extension_sha = dataset_parent.get("extension_identity_path"), dataset_parent.get("extension_identity_sha256")
    else:
        extension_path, extension_sha = None, None
    if isinstance(extension_path, str) and _is_sha(extension_sha):
        extension_record = {"path": extension_path, "sha256": extension_sha}
        _file_receipt(extension_record, "extension_identity", checks, receipts, cache)
    else:
        checks["extension_identity"] = False
    _snapshot_receipts(input_dir, engineering, checks, receipts, cache)
    snapshot_info = engineering.get("snapshots", {})
    checks["summary_snapshot_receipt"] = isinstance(snapshot_info, Mapping) and isinstance(snapshot_info.get("path"), str) and _same_path(summary.get("snapshot_path"), Path(snapshot_info["path"]).expanduser())
    deadline("producer receipts")
    common, schedule_time = _common_hash_check(data, np), _schedule_time_check(data, np)
    checks.update({"common_x_sha256_matches_fp_trajectory": common["common_x_sha256_matches_fp_trajectory"], **schedule_time})
    science = scientific_replay(data, np)
    checks.update({f"science_{key}": bool(value) for key, value in science["raw_science_checks"].items()})
    deadline("scientific replay")
    report = {"schema": VERIFICATION_SCHEMA, "allocation": dict(allocation), "input_job_dir": str(input_dir), "output_job_dir": str(output_dir), "raw_sha256": sha(raw_path), "engineering_sha256": sha(engineering_path), "summary_sha256": sha(summary_path), "checks": {k: bool(v) for k, v in sorted(checks.items())}, "receipts": receipts, "common_path_receipt": common, "schedule_time_receipt": schedule_time, "scientific_replay": science, "numeric_decision": science["numeric_decision"], "decision": science["numeric_decision"] if all(checks.values()) else "implementation_inconclusive", "elapsed_seconds": time.monotonic() - started, "independence_boundary": "CPU independently checks raw arithmetic, FP-path hashes, schedules, and file receipts; it does not reconstruct GPU model calls or prove velocity provenance beyond the saved common-x binding.", "scope": "Six pinned states, two shared noises, three SR draws, 10-step FP32 fake-quantized Euler replay; no model reload, environment step, task success, native low-bit or full-validation claim."}
    if len(json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")) >= 64 * 1024:
        raise ValueError("verification.json exceeds 64 KiB")
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise ValueError("verification.json exceeds 64 KiB")
    temporary = path / "verification.json.writing"
    temporary.write_bytes(encoded)
    temporary.replace(path / "verification.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_SECONDS:g}]")
    # Do not import NumPy, torch, or touch artifacts before the allocation guard.
    from allocation_guard import require_allocation

    allocation = require_allocation()
    import numpy as np

    started = time.monotonic()
    try:
        report = _verify_artifact(args, allocation, np)
        _write_report(args.output.resolve(), report)
        print(json.dumps({"decision": report["decision"], "numeric_decision": report["numeric_decision"]}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {"schema": VERIFICATION_SCHEMA, "allocation": dict(allocation), "input_job_dir": str(args.input.resolve()), "decision": "implementation_inconclusive", "numeric_decision": "implementation_inconclusive", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started, "independence_boundary": "Failure report only; producer booleans are not treated as independent science.", "scope": "CPU raw replay failed before a bound scientific decision."}
        _write_report(args.output.resolve(), failure)
        raise


if __name__ == "__main__":
    main()

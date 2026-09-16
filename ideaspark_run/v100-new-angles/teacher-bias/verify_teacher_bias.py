"""Independent CPU verifier for the recorded-future teacher-bias screen.

The verifier consumes only the producer's raw NPZ plus small engineering
receipts.  It never loads the model.  It is intended to run inside a real
CCDS CPU allocation; the wrapper imports allocation_guard before importing
numpy or torch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


RAW_SCHEMA = "teacher-bias-raw-v1"
MANIFEST_SCHEMA = "teacher-bias-recorded-future-manifest-v1"
SAMPLE_SCHEMA = "dino-wm-wall-recorded-future-raw-v1"
VALID_INDICES = tuple(range(124, 130))
ARM_NAMES = ("FP32", "W4", "W8")
TARGET_FRAMES = (5, 25)
CHECKPOINT_SHA256 = "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b"
SOURCE_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CHECKPOINT_EPOCH = 65
RUNNER_SHA256 = "6e475ff4c75dda269b6d916ac28f8d63879abe6aae6427da6955bb35d36ed559"
PROTOCOL_SHA256 = "b553d6a011bfc990a55c3f41dc1319fc3990b7e6449b27c3cd4b0b60a35e0140"
MAX_REPORT_BYTES = 64 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _scalar_text(value: Any, label: str) -> str:
    array = value
    if getattr(array, "ndim", None) != 0:
        raise RuntimeError(f"{label} must be a scalar")
    text = str(array.item())
    if not text:
        raise RuntimeError(f"{label} is empty")
    return text


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _under(root: Path, path: Path) -> bool:
    root, path = root.resolve(), path.resolve()
    return path != root and root in path.parents


def _record_check(checks: dict[str, bool], name: str, passed: bool) -> None:
    checks[name] = bool(passed)


def _load_raw(input_dir: Path, np: Any) -> dict[str, Any]:
    path = input_dir / "teacher_bias_raw.npz"
    if not path.is_file():
        raise RuntimeError(f"raw NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def _shape_checks(data: dict[str, Any], np: Any) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    expected = {
        "target_visual": (6, 2, None, 384),
        "pred_visual": (6, 3, 2, None, 384),
        "target_proprio": (6, 2, 10),
        "raw_target_proprio": (6, 2, 2),
        "pred_proprio": (6, 3, 2, 10),
        "initial_z": (6, None, 404),
        "initial_visual": (6, None, 384),
        "action_blocks": (6, 5, 10),
        "valid_local_indices": (6,),
        "trajectory_ids": (6,),
        "completed": (6,),
        "arm_names": (3,),
        "target_frame_indices": (2,),
        "schema": (),
    }
    required = set(expected)
    _record_check(checks, "raw_required_fields", required.issubset(data))
    if not checks["raw_required_fields"]:
        details["missing_fields"] = sorted(required - set(data))
        return checks, details
    target = data["target_visual"]
    patch_count = int(target.shape[2]) if target.ndim >= 3 else -1
    expected["target_visual"] = (6, 2, patch_count, 384)
    expected["pred_visual"] = (6, 3, 2, patch_count, 384)
    expected["initial_z"] = (6, patch_count, 404)
    expected["initial_visual"] = (6, patch_count, 384)
    for name, shape in expected.items():
        _record_check(checks, f"shape_{name}", tuple(data[name].shape) == shape)
        details[f"{name}_shape"] = list(data[name].shape)
    numeric = (
        "target_visual", "pred_visual", "target_proprio", "raw_target_proprio",
        "pred_proprio", "initial_z", "initial_visual", "action_blocks",
    )
    _record_check(checks, "raw_numeric_finite", all(np.isfinite(data[name]).all() for name in numeric))
    _record_check(checks, "raw_schema", _scalar_text(data["schema"], "schema") == RAW_SCHEMA)
    _record_check(checks, "indices_exact", np.array_equal(data["valid_local_indices"], np.asarray(VALID_INDICES)))
    _record_check(checks, "arms_exact", tuple(str(x) for x in data["arm_names"]) == ARM_NAMES)
    _record_check(checks, "target_frames_exact", np.array_equal(data["target_frame_indices"], np.asarray(TARGET_FRAMES)))
    _record_check(checks, "completed_exact", bool(data["completed"].dtype == np.bool_) and bool(data["completed"].all()))
    if checks.get("shape_initial_z") and checks.get("shape_initial_visual"):
        _record_check(
            checks,
            "initial_feature_allclose",
            bool(np.allclose(data["initial_z"][..., :384], data["initial_visual"], atol=1e-6, rtol=0)),
        )
        details["initial_feature_max_abs"] = float(
            np.max(np.abs(data["initial_z"][..., :384].astype(np.float64)
                          - data["initial_visual"].astype(np.float64)))
        )
    return checks, details


def _manifest_checks(input_dir: Path, engineering: dict[str, Any], data: dict[str, Any], np: Any) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any] | None]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    record = engineering.get("input_manifest")
    if not isinstance(record, dict):
        _record_check(checks, "engineering_input_manifest_record", False)
        return checks, details, None
    manifest_path = _resolve_path(record.get("path", ""), input_dir)
    exists = manifest_path.is_file()
    _record_check(checks, "input_manifest_exists", exists)
    if not exists:
        return checks, details, None
    actual_hash = _sha256(manifest_path)
    _record_check(checks, "input_manifest_hash", actual_hash == record.get("sha256"))
    details["input_manifest_path"] = str(manifest_path)
    details["input_manifest_sha256"] = actual_hash
    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:
        _record_check(checks, "input_manifest_json", False)
        details["manifest_error"] = f"{type(exc).__name__}: {exc}"[:400]
        return checks, details, None
    _record_check(checks, "manifest_schema", manifest.get("schema") == MANIFEST_SCHEMA)
    _record_check(checks, "manifest_raw_schema", manifest.get("raw_schema") == SAMPLE_SCHEMA)
    source_identity = manifest.get("source_identity")
    checkpoint = manifest.get("checkpoint")
    _record_check(checks, "manifest_source_identity", isinstance(source_identity, dict)
                 and source_identity.get("commit") == SOURCE_COMMIT)
    _record_check(checks, "manifest_checkpoint_identity", isinstance(checkpoint, dict)
                 and checkpoint.get("expected_sha256") == CHECKPOINT_SHA256
                 and checkpoint.get("expected_epoch") == CHECKPOINT_EPOCH
                 and checkpoint.get("dinov2_source_commit") == DINOV2_COMMIT)
    rows = manifest.get("samples")
    rows = rows if isinstance(rows, list) else []
    _record_check(checks, "manifest_six_samples", len(rows) == 6)
    try:
        rows = sorted(rows, key=lambda row: int(row.get("valid_local_index", -1))
                      if isinstance(row, dict) else -1)
    except Exception:
        _record_check(checks, "manifest_row_order", False)
    indices = []
    sample_hashes = []
    sample_ids = []
    sample_action_blocks = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("valid_local_index", -1))
        indices.append(idx)
        sample_ids.append(int(row.get("underlying_trajectory_id", -1)))
        sample_hashes.append(str(row.get("sha256", "")))
        sample_path = _resolve_path(row.get("sample_path", ""), manifest_path.parent)
        row_ok = (_under(manifest_path.parent, sample_path)
                  and sample_path.is_file() and _sha256(sample_path) == row.get("sha256"))
        _record_check(checks, f"sample_{idx}_hash", row_ok)
        if not row_ok:
            continue
        try:
            with np.load(sample_path, allow_pickle=False) as sample:
                meta = json.loads(_scalar_text(sample["metadata_json"], "metadata_json"))
                meta_ok = (
                    meta.get("schema") == SAMPLE_SCHEMA
                    and int(meta.get("valid_local_index", -1)) == idx
                    and int(meta.get("underlying_trajectory_id", -1)) == sample_ids[-1]
                    and list(meta.get("frame_indices", [])) == [0, 5, 25]
                    and tuple(sample["visual_0_5_25"].shape[:2]) == (3, 3)
                    and tuple(sample["proprio_0_5_25"].shape) == (3, 2)
                    and tuple(sample["model_actions_h5"].shape) == (5, 10)
                    and sample["visual_0_5_25"].dtype in (np.float32, np.float64)
                    and bool(np.isfinite(sample["visual_0_5_25"]).all())
                    and bool(np.isfinite(sample["proprio_0_5_25"]).all())
                    and bool(np.isfinite(sample["model_actions_h5"]).all())
                )
                if meta_ok:
                    sample_action_blocks.append(np.array(sample["model_actions_h5"], copy=True))
            _record_check(checks, f"sample_{idx}_contract", meta_ok)
        except Exception as exc:
            _record_check(checks, f"sample_{idx}_contract", False)
            details[f"sample_{idx}_error"] = f"{type(exc).__name__}: {exc}"[:300]
    _record_check(checks, "manifest_indices_exact", tuple(sorted(indices)) == VALID_INDICES)
    _record_check(checks, "manifest_source_ids_unique", len(sample_ids) == len(set(sample_ids)) == 6)
    engineering_samples = engineering.get("sample_identity")
    if isinstance(engineering_samples, list):
        expected = [(int(x.get("valid_local_index", -1)), str(x.get("sample_sha256", "")))
                    for x in engineering_samples if isinstance(x, dict)]
        _record_check(checks, "engineering_sample_identity", sorted(expected) == sorted(zip(indices, sample_hashes)))
    else:
        _record_check(checks, "engineering_sample_identity", False)
    if len(sample_action_blocks) == 6 and data.get("action_blocks") is not None:
        _record_check(checks, "raw_actions_match_manifest",
                     bool(np.array_equal(data["action_blocks"], np.stack(sample_action_blocks)))
                     )
    else:
        _record_check(checks, "raw_actions_match_manifest", False)
    meta_text = data.get("metadata_json")
    if meta_text is not None:
        try:
            metadata = json.loads(_scalar_text(meta_text, "metadata_json"))
            _record_check(checks, "raw_metadata_schema", metadata.get("schema") == RAW_SCHEMA)
            _record_check(checks, "raw_metadata_input_hash", metadata.get("input_manifest_sha256") == actual_hash)
            _record_check(checks, "raw_metadata_samples", metadata.get("sample_sha256") == sample_hashes)
            _record_check(checks, "raw_metadata_indices", metadata.get("trajectory_ids") == [f"WallDataset:{x}" for x in sample_ids])
        except Exception as exc:
            _record_check(checks, "raw_metadata_schema", False)
            details["raw_metadata_error"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        _record_check(checks, "raw_metadata_schema", False)
    if "trajectory_ids" in data and len(sample_ids) == 6:
        _record_check(checks, "raw_trajectory_ids", tuple(str(x) for x in data["trajectory_ids"])
                     == tuple(f"WallDataset:{x}" for x in sample_ids))
    else:
        _record_check(checks, "raw_trajectory_ids", False)
    return checks, details, manifest


def _git_head(source: Path) -> tuple[str, list[str]]:
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    modified = subprocess.run(
        ["git", "-C", str(source), "diff", "--name-only", "HEAD"],
        check=True, capture_output=True, text=True, timeout=20,
    ).stdout.splitlines()
    return head, modified


def _producer_checks(input_dir: Path, engineering: dict[str, Any], manifest: dict[str, Any] | None, np: Any) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    _record_check(checks, "engineering_schema", engineering.get("schema") == "teacher-bias-engineering-v1")
    _record_check(checks, "producer_status_complete", engineering.get("status") == "complete")
    _record_check(checks, "producer_completed_six", engineering.get("completed") == [True] * 6)
    allocation = engineering.get("allocation")
    if not isinstance(allocation, dict):
        allocation = {}
    job_id = str(allocation.get("job_id", ""))
    _record_check(checks, "producer_job_id_bound", job_id.isdigit() and job_id == input_dir.name
                  and allocation.get("verified") is True and allocation.get("partition") == "UGGPU-TC1")
    gpu_file = input_dir / "gpu_identity.txt"
    gpu_text = gpu_file.read_text(encoding="utf-8", errors="replace") if gpu_file.is_file() else ""
    _record_check(checks, "v100_identity_receipt", "V100" in gpu_text.upper())
    runtime_value = engineering.get("runtime_identity")
    runtime_value = runtime_value if isinstance(runtime_value, dict) else {}
    gpu_runtime = runtime_value.get("gpu_name", "")
    _record_check(checks, "v100_runtime_identity", "V100" in str(gpu_runtime).upper() or "V100" in gpu_text.upper())
    _record_check(checks, "cached_rollout_gate", engineering.get("no_op_official_vs_cached") is True)
    initial = engineering.get("initial_feature_evidence")
    _record_check(checks, "initial_feature_receipts", isinstance(initial, list) and len(initial) == 6
                 and all(isinstance(x, dict) and x.get("passed") is True for x in initial))
    runtime = runtime_value
    prepared = runtime.get("prepared_runtime_binding") or engineering.get("prepared_runtime_binding")
    _record_check(checks, "prepared_runtime_binding", isinstance(prepared, dict)
                 and prepared.get("prepared_source_config_checkpoint_verified") is True
                 and prepared.get("actual_git_commit") == SOURCE_COMMIT)
    _record_check(checks, "runtime_source_commit", runtime.get("source_commit") == SOURCE_COMMIT)
    _record_check(checks, "runtime_dinov2_commit", runtime.get("dinov2_source_commit") == DINOV2_COMMIT)
    _record_check(checks, "runtime_checkpoint_sha", runtime.get("checkpoint_sha256") == CHECKPOINT_SHA256)
    model_structure = engineering.get("model_structure")
    model_structure = model_structure if isinstance(model_structure, dict) else {}
    paths = model_structure.get("predictor_linear_paths", [])
    _record_check(checks, "predictor_linear_count", model_structure.get("predictor_linear_count") == 24
                 and isinstance(paths, list) and len(paths) == 24 and len(set(paths)) == 24)
    quant = engineering.get("quantization_evidence")
    quant = quant if isinstance(quant, dict) else {}
    qpaths = []
    for arm, bits in (("W4", 4), ("W8", 8)):
        item = quant.get(arm)
        ok = isinstance(item, dict) and item.get("bits") == bits and item.get("linear_count") == 24
        ok = ok and item.get("writeback_readback") is True
        _record_check(checks, f"{arm}_quantization_receipt", ok)
        if isinstance(item, dict):
            qpaths.append(str(item.get("params_path", "")))
    _record_check(checks, "quant_params_single_path", len(qpaths) == 2 and qpaths[0] == qpaths[1] and bool(qpaths[0]))
    restores = engineering.get("restore_evidence")
    restore_keys = {(int(x.get("sample", -1)), str(x.get("arm", ""))) for x in restores or [] if isinstance(x, dict)}
    _record_check(checks, "restore_receipts_18", len(restore_keys) == 18
                 and restore_keys == {(i, arm) for i in range(6) for arm in ARM_NAMES})
    _record_check(checks, "runner_hash", engineering.get("runner_sha256") == RUNNER_SHA256
                 and (input_dir / "teacher_bias_screen.py").is_file()
                 and _sha256(input_dir / "teacher_bias_screen.py") == RUNNER_SHA256)
    protocol_path = input_dir / "teacher_bias_protocol.zh.md"
    _record_check(checks, "protocol_hash", engineering.get("protocol_sha256") == PROTOCOL_SHA256
                 and protocol_path.is_file() and _sha256(protocol_path) == PROTOCOL_SHA256)
    freeze_path = input_dir / "teacher_bias_input_freeze.json"
    freeze_ok = False
    if freeze_path.is_file():
        try:
            freeze = _load_json(freeze_path)
            freeze_ok = freeze.get("manifest_sha256") == engineering.get("input_manifest", {}).get("sha256")
        except Exception:
            freeze_ok = False
    _record_check(checks, "external_input_freeze", freeze_ok)
    if manifest is not None:
        manifest_source = manifest.get("source_identity")
        manifest_source = manifest_source if isinstance(manifest_source, dict) else {}
        msource = manifest_source.get("files", {})
        runtime_files = runtime.get("source_file_sha256", {})
        runtime_root = _resolve_path(runtime.get("root", ""), input_dir)
        source_files_ok = isinstance(msource, dict) and isinstance(runtime_files, dict)
        if source_files_ok:
            for relative, record in msource.items():
                if not isinstance(record, dict) or runtime_files.get(str(relative)) != record.get("sha256"):
                    source_files_ok = False
                    continue
                actual = _resolve_path(record.get("path", ""), runtime_root)
                if not _under(runtime_root, actual) or not actual.is_file() or _sha256(actual) != record.get("sha256"):
                    source_files_ok = False
        _record_check(checks, "source_files_match_manifest", source_files_ok)
        manifest_checkpoint = manifest.get("checkpoint")
        manifest_checkpoint = manifest_checkpoint if isinstance(manifest_checkpoint, dict) else {}
        config = manifest_checkpoint.get("config", {})
        checkpoint = manifest_checkpoint.get("file", {})
        config_path = _resolve_path(config.get("path", ""), runtime_root)
        checkpoint_path = _resolve_path(checkpoint.get("path", ""), runtime_root)
        _record_check(checks, "checkpoint_files_match_manifest",
                     isinstance(config, dict) and isinstance(checkpoint, dict)
                     and _under(runtime_root, config_path) and _under(runtime_root, checkpoint_path)
                     and config_path.is_file() and checkpoint_path.is_file()
                     and _sha256(config_path) == config.get("sha256")
                     and _sha256(checkpoint_path) == checkpoint.get("sha256") == CHECKPOINT_SHA256)
        try:
            head, modified = _git_head(runtime_root / "source")
            _record_check(checks, "source_git_head", head == SOURCE_COMMIT)
            _record_check(checks, "source_patch_allowlist", not (set(modified) - {"models/dino.py", "env/__init__.py"}))
            details["source_modified_tracked_paths"] = modified
        except Exception as exc:
            _record_check(checks, "source_git_head", False)
            _record_check(checks, "source_patch_allowlist", False)
            details["source_git_error"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        for name in ("source_files_match_manifest", "checkpoint_files_match_manifest", "source_git_head", "source_patch_allowlist"):
            _record_check(checks, name, False)
    qparam_path = _resolve_path(qpaths[0], input_dir) if qpaths and qpaths[0] else input_dir / "quant_params.pt"
    qhash = engineering.get("quant_params_sha256")
    _record_check(checks, "quant_params_exists", qparam_path.is_file())
    if qparam_path.is_file():
        actual_qhash = _sha256(qparam_path)
        details["quant_params_sha256"] = actual_qhash
        _record_check(checks, "quant_params_hash", isinstance(qhash, str) and actual_qhash == qhash)
    else:
        _record_check(checks, "quant_params_hash", False)
    if qparam_path.is_file() and all(checks.get(f"{arm}_quantization_receipt", False) for arm in ("W4", "W8")):
        try:
            import torch
            try:
                payload = torch.load(qparam_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(qparam_path, map_location="cpu")
            arms = payload.get("arms") if isinstance(payload, dict) else None
            qok = isinstance(arms, dict)
            qsummary: dict[str, int] = {}
            for arm, bits in (("W4", 4), ("W8", 8)):
                records = arms.get(arm) if isinstance(arms, dict) else None
                paths_seen = []
                if not isinstance(records, list):
                    qok = False
                    continue
                for rec in records:
                    if not isinstance(rec, dict) or rec.get("bits") != bits:
                        qok = False
                        continue
                    if not all(key in rec for key in ("path", "shape", "fp_weight", "codes", "scale", "dequant_weight")):
                        qok = False
                        continue
                    try:
                        shape = tuple(int(x) for x in rec["shape"])
                        fp = rec["fp_weight"]
                        codes = rec["codes"]
                        scale = rec["scale"]
                        dequant = rec["dequant_weight"]
                        qmax = 2 ** (bits - 1) - 1
                        expected_scale_shape = (shape[0],)
                        dequant_scale = scale.reshape((-1,) + (1,) * (len(shape) - 1))
                        tensor_ok = (
                            tuple(fp.shape) == shape and tuple(codes.shape) == shape
                            and tuple(dequant.shape) == shape and tuple(scale.shape) == expected_scale_shape
                            and bool(torch.isfinite(fp).all()) and bool(torch.isfinite(codes).all())
                            and bool(torch.isfinite(scale).all()) and bool(torch.isfinite(dequant).all())
                            and bool((scale > 0).all())
                            and bool(torch.equal(codes, codes.round()))
                            and bool((codes.abs() <= qmax).all())
                            and bool(torch.allclose(dequant, codes * dequant_scale, atol=0, rtol=0))
                        )
                        qok = qok and tensor_ok
                    except Exception:
                        qok = False
                    paths_seen.append(str(rec.get("path", "")))
                qok = qok and len(records) == 24 and len(set(paths_seen)) == 24 and set(paths_seen) == set(paths)
                qsummary[arm] = len(records)
            details["quant_param_record_counts"] = qsummary
            _record_check(checks, "quant_param_records", qok)
        except Exception as exc:
            _record_check(checks, "quant_param_records", False)
            details["quant_param_error"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        _record_check(checks, "quant_param_records", False)
    return checks, details


def scientific_replay(data: dict[str, Any], np: Any) -> dict[str, Any]:
    target = data["target_visual"].astype(np.float64)
    predictions = data["pred_visual"].astype(np.float64)
    if target.shape[:2] != (6, 2) or target.ndim != 4 or target.shape[-1] != 384:
        raise ValueError("target_visual must have shape [6,2,patch,384]")
    if predictions.shape != (6, 3, 2, target.shape[2], 384):
        raise ValueError("pred_visual must have shape [6,3,2,patch,384]")
    if not np.isfinite(target).all() or not np.isfinite(predictions).all():
        raise ValueError("visual features contain non-finite values")
    teacher_error = predictions[:, 0] - target
    fp_mse = np.square(teacher_error).mean(axis=(-2, -1))
    target_variance = target.var(axis=(-2, -1))
    all_close = True
    states: list[dict[str, Any]] = []
    for state in range(6):
        horizons: list[dict[str, Any]] = []
        for hidx, horizon in enumerate((1, 5)):
            row: dict[str, Any] = {
                "horizon": horizon,
                "FP_target_MSE": float(fp_mse[state, hidx]),
                "target_feature_variance": float(target_variance[state, hidx]),
                "arms": {},
            }
            for arm_index, arm in ((1, "W4"), (2, "W8")):
                delta = predictions[state, arm_index, hidx] - predictions[state, 0, hidx]
                q = float(np.square(predictions[state, arm_index, hidx] - target[state, hidx]).mean())
                p = float(np.square(delta).mean())
                c = float(2 * (teacher_error[state, hidx] * delta).mean())
                residual = q - float(fp_mse[state, hidx]) - p - c
                close = bool(np.isclose(q, fp_mse[state, hidx] + p + c, atol=1e-10, rtol=1e-9))
                all_close &= close
                row["arms"][arm] = {
                    "target_MSE": q,
                    "FP_fidelity_MSE": p,
                    "cross_term": c,
                    "identity_residual": residual,
                    "identity_pass": close,
                    "gain": float(1 - q / fp_mse[state, hidx]) if fp_mse[state, hidx] > 0 else None,
                }
            horizons.append(row)
        h5 = horizons[1]
        nondegenerate = bool(fp_mse[state, 1] > 1e-12 and target_variance[state, 1] > 1e-12)
        gain = h5["arms"]["W4"]["gain"]
        primary_pass = bool(nondegenerate and gain is not None and gain >= 0.05)
        states.append({
            "state_index": state,
            "horizons": horizons,
            "nondegenerate": nondegenerate,
            "primary_pass": primary_pass,
        })
    all_nondegenerate = all(row["nondegenerate"] for row in states)
    if not all_close:
        decision = "implementation_inconclusive"
    elif not all_nondegenerate:
        decision = "inconclusive_degenerate"
    elif sum(row["primary_pass"] for row in states) >= 4:
        decision = "scope_limited_preliminary_go"
    else:
        decision = "mechanism_no_go"
    return {
        "numeric_decision": decision,
        "raw_science_checks": {"all_error_decompositions_close": all_close},
        "nondegenerate_count": sum(row["nondegenerate"] for row in states),
        "primary_pass_count": sum(row["primary_pass"] for row in states),
        "states": states,
        "mean_H5_FP_target_MSE": float(fp_mse[:, 1].mean()),
        "mean_H5_W4_target_MSE": float(np.square(predictions[:, 1, 1] - target[:, 1]).mean()),
        "mean_H5_W8_target_MSE": float(np.square(predictions[:, 2, 1] - target[:, 1]).mean()),
        "boundary": "Fixed encoder-feature prediction diagnostic; no physical truth or task-success claim",
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if temporary.stat().st_size >= MAX_REPORT_BYTES:
        temporary.unlink()
        raise RuntimeError(f"verification report exceeds {MAX_REPORT_BYTES} bytes")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input-dir", dest="input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-sha256")
    parser.add_argument("--max-seconds", type=int, default=270)
    return parser.parse_args()


def main() -> None:
    started = time.monotonic()
    args = _parse_args()
    guard = importlib.import_module("allocation_guard")
    allocation = guard.require_allocation()
    import numpy as np

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engineering_path = input_dir / "engineering.json"
    if not engineering_path.is_file():
        report = {
            "schema": "teacher-bias-verification-v1",
            "allocation": allocation,
            "input_dir": str(input_dir),
            "engineering_status": "missing",
            "science_ready": False,
            "numeric_decision": "not_evaluated_missing_engineering",
            "decision": "implementation_inconclusive",
            "note": "No engineering receipt; raw science is withheld.",
        }
        _write_report(output_dir / "verification.json", report)
        print(json.dumps({"decision": report["decision"], "science_ready": False}, sort_keys=True))
        return
    engineering = _load_json(engineering_path)
    if engineering.get("status") != "complete" or engineering.get("completed") != [True] * 6:
        decision = "inconclusive_budget" if engineering.get("status") == "running" else "implementation_inconclusive"
        report = {"schema": "teacher-bias-verification-v1", "allocation": allocation,
                  "input_dir": str(input_dir), "engineering_status": engineering.get("status"),
                  "completed": engineering.get("completed"), "science_ready": False,
                  "numeric_decision": "not_evaluated_incomplete_producer", "decision": decision,
                  "note": "Incomplete producer receipt; partial raw science arrays were not loaded."}
        _write_report(output_dir / "verification.json", report)
        print(json.dumps({"decision": decision, "science_ready": False}, sort_keys=True))
        return
    raw_path = input_dir / "teacher_bias_raw.npz"
    if not raw_path.is_file():
        report = {
            "schema": "teacher-bias-verification-v1",
            "allocation": allocation,
            "input_dir": str(input_dir),
            "engineering_status": engineering.get("status"),
            "science_ready": False,
            "numeric_decision": "not_evaluated_missing_raw",
            "decision": "implementation_inconclusive",
            "producer_checks": {"raw_npz_exists": False},
            "note": "Producer did not leave a raw NPZ; raw science is withheld.",
        }
        _write_report(output_dir / "verification.json", report)
        print(json.dumps({"decision": report["decision"], "science_ready": False}, sort_keys=True))
        return
    raw = _load_raw(input_dir, np)
    shape_checks, shape_details = _shape_checks(raw, np)
    manifest_checks, manifest_details, manifest = _manifest_checks(input_dir, engineering, raw, np)
    producer_checks, producer_details = _producer_checks(input_dir, engineering, manifest, np)
    expected_raw = engineering.get("raw_npz", {})
    _record_check(producer_checks, "raw_npz_hash",
                  isinstance(expected_raw, dict) and _sha256(raw_path) == expected_raw.get("sha256"))
    if args.input_sha256:
        _record_check(manifest_checks, "cli_input_manifest_hash",
                      manifest_details.get("input_manifest_sha256") == args.input_sha256)
    science_ready = all(shape_checks.values()) and all(manifest_checks.values()) and all(producer_checks.values())
    report: dict[str, Any] = {
        "schema": "teacher-bias-verification-v1",
        "allocation": allocation,
        "input_dir": str(input_dir),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "engineering_status": engineering.get("status"),
        "shape_checks": shape_checks,
        "shape_details": shape_details,
        "manifest_checks": manifest_checks,
        "manifest_details": manifest_details,
        "producer_checks": producer_checks,
        "producer_details": producer_details,
        "science_ready": science_ready,
        "scope": "six Wall validation-local indices 124..129; H1/H5 recorded-future encoder features",
    }
    if not science_ready:
        report["numeric_decision"] = "not_evaluated_producer_binding_failed"
        report["decision"] = "implementation_inconclusive"
        report["note"] = "Raw science metrics are withheld until every producer/input/identity receipt passes."
    elif time.monotonic() - started > args.max_seconds:
        report["numeric_decision"] = "not_evaluated_timeout"
        report["decision"] = "implementation_inconclusive"
    else:
        science = scientific_replay(raw, np)
        report["science"] = science
        report["numeric_decision"] = science["numeric_decision"]
        report["decision"] = science["numeric_decision"]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _write_report(output_dir / "verification.json", report)
    print(json.dumps({
        "decision": report["decision"],
        "science_ready": science_ready,
        "verification": str(output_dir / "verification.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

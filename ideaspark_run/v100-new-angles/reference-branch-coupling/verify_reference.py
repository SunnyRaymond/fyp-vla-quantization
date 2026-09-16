"""CPU verifier for the frozen reference-branch producer.

This verifier checks the producer's engineering receipts and independently
rebuilds the saved 4-bit encoder maps.  It does not load the DINO-WM model.
The scientific calculation is deliberately delegated to root's
``reference_replay.replay`` after all producer gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
from typing import Any


SOURCE_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CHECKPOINT_SHA = "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b"
INPUT_MANIFEST_SHA = "6dae677a0e7763896d735495500aea78a9254e59b67fa47817bb9bb9bb1f5c14"
TEACHER_HELPER_SHA = "6e475ff4c75dda269b6d916ac28f8d63879abe6aae6427da6955bb35d36ed559"
SMOKE_RUNNER_SHA = "de5c5eb19f4b26614e71f9e2af36db21e9fd5bcc65188f3191772552bc750af9"
SCREEN_RUNNER_SHA = "51c2463a92a3bf84eabae735eeaa9771a7202add0b2227fe06c7fe1f2bd69d19"
PROTOCOL_SHA = "076ceec559c9c2040185fd2c00d163d71d905620fa507397bf400a5a470bd795"
FREEZE_SHA = "8f15a288d3561d640068eeab647df68f8a8a65fb93e5f1d0657a5deee90a6546"
REPLAY_SHA = "258c41d8af9e991130db3df0f09935d560019b104e2cbb0d8ea2b9809bc2ee9c"
PRODUCER_SHA = "84959e7e9262c0839faae1e589b7ef60e88c03fd943a06848b4e015f1668f900"
CONTRACT_SHA = "f5e7a47cbf4f49fca2486844981d13975dba65a63dc3c2f7663972f27bc75e6d"
RAW_SCHEMA = "reference-branch-coupling-raw-v1"
ARMS = ["FP32", "encoder_W4_RTN", "encoder_W4_SR0", "encoder_W4_SR1", "encoder_W4_SR2"]
SEEDS = (2701, 2702, 2703)
VALID_INDICES = (124, 125, 126, 127, 128, 129)
TRAJECTORY_IDS = (1035, 1534, 1158, 203, 1837, 1095)
EXPECTED_TARGET_PATHS = tuple(sorted(
    f"encoder.base_model.blocks.{block}.{suffix}"
    for block in range(12)
    for suffix in ("attn.proj", "attn.qkv", "mlp.fc1", "mlp.fc2")))
RAW_SHAPES = {
    "pred_visual": (6, 5, 196, 384),
    "goal_visual": (6, 5, 196, 384),
    "current_visual": (6, 5, 196, 384),
    "fp_noop_pred": (6, 2, 196, 384),
    "fp_noop_goal": (6, 2, 196, 384),
    "fp_noop_current": (6, 2, 196, 384),
    "predictor_input": (6, 5, 196, 404),
    "completed": (6, 5),
    "trajectory_ids": (6,),
    "valid_indices": (6,),
    "arm_names": (5,),
    "schema": (),
}
RAW_DTYPES = {
    **{key: "float32" for key in (
        "pred_visual", "goal_visual", "current_visual", "fp_noop_pred",
        "fp_noop_goal", "fp_noop_current", "predictor_input")},
    "completed": "bool", "trajectory_ids": "int64", "valid_indices": "int64",
    "arm_names": "unicode", "schema": "unicode",
}


def _sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: pathlib.Path) -> Any:
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"missing or oversized JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _file(path: pathlib.Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing producer artifact: {path}")


def _sha_receipt(path: pathlib.Path, receipt: Any) -> bool:
    return (path.is_file() and isinstance(receipt, str) and
            bool(re.fullmatch(r"[0-9a-f]{64}", receipt)) and _sha(path) == receipt)


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise ValueError("verification.json exceeds 64 KiB")
    if path.exists():
        raise ValueError(f"refusing to overwrite verifier output: {path}")
    temporary = pathlib.Path(str(path) + ".writing")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _allocation_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    allocation = engineering.get("allocation")
    checks["allocation_receipt"] = isinstance(allocation, dict)
    if not isinstance(allocation, dict):
        return
    checks["allocation_verified"] = allocation.get("verified") is True
    checks["allocation_scheduler"] = str(allocation.get("scheduler", "")).casefold() == "slurm"
    checks["allocation_job"] = str(allocation.get("job_id", "")) == job.name and job.name.isdigit()
    host = str(allocation.get("hostname", "")).split(".", 1)[0].casefold()
    checks["allocation_hostname"] = bool(re.fullmatch(r"tc1n[0-9]{2}", host))
    checks["allocation_partition"] = allocation.get("partition") == "UGGPU-TC1"
    node = str(allocation.get("nodelist", "")).split("[", 1)[0].casefold()
    checks["allocation_node"] = bool(re.fullmatch(r"tc1n[0-9]{2}", node)) and node == host
    gpu = engineering.get("gpu")
    checks["v100"] = (isinstance(gpu, dict) and
                       "v100" in str(gpu.get("name", "")).casefold() and
                       gpu.get("compute_capability") == [7, 0])
    work = engineering.get("work_seconds")
    checks["producer_budget"] = isinstance(work, (int, float)) and 0 <= work <= 180


def _receipt_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    """Check the producer's explicit ``receipts`` dictionary.

    The producer intentionally records protocol/source files at their immutable
    source paths, while raw/runtime/snapshot are written in the job directory.
    Checking those paths and hashes avoids accepting a path label as identity.
    """
    receipts = engineering.get("receipts")
    checks["receipts_present"] = isinstance(receipts, dict)
    if not isinstance(receipts, dict):
        return
    required = {"producer", "teacher_helper", "protocol", "freeze", "runtime",
                "smoke_runner.py", "screen_runner.py", "raw", "quant_snapshot", "contract"}
    checks["required_receipts"] = required.issubset(receipts)
    for name in sorted(required):
        receipt = receipts.get(name)
        path = pathlib.Path(str(receipt.get("path", ""))) if isinstance(receipt, dict) else pathlib.Path("")
        digest = receipt.get("sha256") if isinstance(receipt, dict) else None
        checks[f"receipt_{name}"] = _sha_receipt(path, digest)
    checks["raw_receipt_path"] = (isinstance(receipts.get("raw"), dict) and
                                   pathlib.Path(str(receipts["raw"].get("path", ""))).resolve() == (job / "raw.npz").resolve())
    checks["snapshot_receipt_path"] = (isinstance(receipts.get("quant_snapshot"), dict) and
                                        pathlib.Path(str(receipts["quant_snapshot"].get("path", ""))).resolve() == (job / "quant_snapshot.pt").resolve())
    checks["runtime_receipt_path"] = (isinstance(receipts.get("runtime"), dict) and
                                       pathlib.Path(str(receipts["runtime"].get("path", ""))).resolve() == (job / "runtime.json").resolve())
    checks["contract_receipt_path"] = (isinstance(receipts.get("contract"), dict) and
                                       pathlib.Path(str(receipts["contract"].get("path", ""))).resolve() == (job / "RAW_CONTRACT.json").resolve())
    checks["teacher_helper_pin"] = isinstance(receipts.get("teacher_helper"), dict) and receipts["teacher_helper"].get("sha256") == TEACHER_HELPER_SHA
    checks["smoke_runner_pin"] = isinstance(receipts.get("smoke_runner.py"), dict) and receipts["smoke_runner.py"].get("sha256") == SMOKE_RUNNER_SHA
    checks["screen_runner_pin"] = isinstance(receipts.get("screen_runner.py"), dict) and receipts["screen_runner.py"].get("sha256") == SCREEN_RUNNER_SHA
    checks["protocol_pin"] = isinstance(receipts.get("protocol"), dict) and receipts["protocol"].get("sha256") == PROTOCOL_SHA
    checks["freeze_pin"] = isinstance(receipts.get("freeze"), dict) and receipts["freeze"].get("sha256") == FREEZE_SHA
    producer = receipts.get("producer")
    checks["producer_receipt"] = (isinstance(producer, dict) and
                                   _sha_receipt(pathlib.Path(str(producer.get("path", ""))), producer.get("sha256")) and
                                   producer.get("sha256") == PRODUCER_SHA)
    checks["contract_pin"] = (isinstance(receipts.get("contract"), dict) and
                               receipts["contract"].get("sha256") == CONTRACT_SHA)


def _identity_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    runtime_path = job / "runtime.json"
    if not runtime_path.is_file():
        checks["runtime_present"] = False
        return
    runtime = _json(runtime_path)
    checks["runtime_present"] = True
    identity = runtime.get("identity")
    checks["source_commit"] = isinstance(identity, dict) and identity.get("source_commit") == SOURCE_COMMIT
    checks["dinov2_commit"] = isinstance(identity, dict) and identity.get("dinov2_source_commit") == DINOV2_COMMIT
    checks["checkpoint_sha256"] = isinstance(identity, dict) and identity.get("checkpoint_sha256") == CHECKPOINT_SHA
    source_files = identity.get("source_file_sha256") if isinstance(identity, dict) else None
    checks["source_file_receipts"] = (isinstance(source_files, dict) and bool(source_files) and
                                       all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                                           for v in source_files.values()))
    # _source_identity stores relative paths plus the exact source checkout root.
    # Resolve and hash every recorded file; a path-only source label is rejected.
    source_root = pathlib.Path(str(identity.get("root", ""))) / "source" if isinstance(identity, dict) else pathlib.Path("")
    if checks["source_file_receipts"] and source_root.is_dir():
        checks["actual_source_files"] = all(
            (source_root / relative).is_file() and _sha(source_root / relative) == digest
            for relative, digest in source_files.items())
    else:
        checks["actual_source_files"] = False
    checks["runtime_sha_recorded"] = (isinstance(engineering.get("receipts"), dict) and
                                       isinstance(engineering["receipts"].get("runtime"), dict) and
                                       engineering["receipts"]["runtime"].get("sha256") == _sha(runtime_path))
    # The prepared binding is the helper's source/config/checkpoint verification,
    # not an inferred path convention.
    binding = identity.get("prepared_runtime_binding") if isinstance(identity, dict) else None
    checks["prepared_runtime_binding"] = (isinstance(binding, dict) and
                                           binding.get("prepared_source_config_checkpoint_verified") is True and
                                           isinstance(binding.get("checkpoint_config_sha256"), str))
    checks["runtime_target_paths"] = (isinstance(runtime.get("target_paths"), list) and
                                      len(runtime["target_paths"]) == 48 and
                                      len(set(runtime["target_paths"])) == 48 and
                                      runtime["target_paths"] == sorted(runtime["target_paths"]) and
                                      tuple(runtime["target_paths"]) == EXPECTED_TARGET_PATHS)
    checks["checkpoint_receipt"] = (isinstance(identity, dict) and
                                     identity.get("checkpoint_sha256") == CHECKPOINT_SHA)
    checks["source_receipt"] = (isinstance(identity, dict) and identity.get("source_commit") == SOURCE_COMMIT)
    helper_hashes = identity.get("helper_sha256") if isinstance(identity, dict) else None
    checks["runtime_helper_hashes"] = (isinstance(helper_hashes, dict) and
                                        helper_hashes.get("smoke_runner.py") == SMOKE_RUNNER_SHA and
                                        helper_hashes.get("screen_runner.py") == SCREEN_RUNNER_SHA)
    manifest = engineering.get("input_manifest")
    checks["input_manifest_receipt"] = (isinstance(manifest, dict) and
                                         manifest.get("sha256") == INPUT_MANIFEST_SHA and
                                         isinstance(manifest.get("path"), str) and
                                         pathlib.Path(manifest["path"]).is_file() and
                                         _sha(pathlib.Path(manifest["path"])) == INPUT_MANIFEST_SHA and
                                         isinstance(manifest.get("sample_sha256"), list) and
                                         len(manifest["sample_sha256"]) == 6 and
                                         all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                                             for v in manifest["sample_sha256"]))
    if checks["input_manifest_receipt"]:
        frozen = _json(pathlib.Path(manifest["path"]))
        dataset = frozen.get("dataset", {})
        loader = dataset.get("loader_settings", {})
        mapping = frozen.get("mapping", {})
        checkpoint = frozen.get("checkpoint", {})
        samples = frozen.get("samples")
        checks['source_matches_frozen_manifest'] = all(
            source_files.get(name) == record['sha256']
            for name, record in frozen['source_identity']['files'].items())
        checks['preprocessor_pin'] = source_files.get('preprocessor.py') == 'c5ad3949628bec91ecdc6f8e011d4cc9f0c8cc9231f2291a77e5daf7e4fd27f3'
        checks['prepared_config_pin'] = binding['checkpoint_config_sha256'] == checkpoint['config']['sha256']
        checks['runtime_fp32'] = identity.get('dtype') == 'float32' and identity.get('execution') == 'emulation_only'
        checks["manifest_source_commit"] = frozen.get("source_identity", {}).get("commit") == SOURCE_COMMIT
        checks["manifest_checkpoint"] = (checkpoint.get("expected_sha256") == CHECKPOINT_SHA and
                                           checkpoint.get("dinov2_source_commit") == DINOV2_COMMIT)
        checkpoint_file = checkpoint.get("file", {})
        config_file = checkpoint.get("config", {})
        checkpoint_path = pathlib.Path(str(checkpoint_file.get("path", "")))
        config_path = pathlib.Path(str(config_file.get("path", "")))
        # This is the one intentionally larger CPU read: prove the checkpoint
        # and Hydra config behind the manifest, rather than trusting hashes as
        # labels.  It happens only after the producer reports complete.
        checks["manifest_checkpoint_files"] = (
            checkpoint_file.get("sha256") == CHECKPOINT_SHA and checkpoint_path.is_file() and
            _sha(checkpoint_path) == CHECKPOINT_SHA and
            isinstance(config_file.get("sha256"), str) and config_path.is_file() and
            _sha(config_path) == config_file.get("sha256"))
        checks["manifest_loader_contract"] = (loader.get("frameskip") == 5 and
                                               loader.get("num_hist") == 1 and loader.get("num_pred") == 1 and
                                               loader.get("action_emb_dim") == 10 and
                                               loader.get("normalize_action") is True)
        checks["manifest_mapping"] = (mapping.get("valid_local_to_underlying") ==
                                       {str(k): v for k, v in zip(VALID_INDICES, TRAJECTORY_IDS)} and
                                       mapping.get("visual_key") == "visual" and
                                       mapping.get("proprio_key") == "proprio")
        checks["manifest_samples"] = (isinstance(samples, list) and len(samples) == 6 and
                                       [s.get("valid_local_index") for s in samples] == list(VALID_INDICES) and
                                       [s.get("underlying_trajectory_id") for s in samples] == list(TRAJECTORY_IDS) and
                                       [s.get("sha256") for s in samples] == manifest["sample_sha256"])
    else:
        for key in ("manifest_source_commit", "manifest_checkpoint", "manifest_loader_contract",
                    "manifest_mapping", "manifest_samples"):
            checks[key] = False


def _freeze_checks(job: pathlib.Path, contract_path: pathlib.Path,
                   engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    receipts = engineering.get("receipts", {})
    freeze_receipt = receipts.get("freeze", {}) if isinstance(receipts, dict) else {}
    freeze_path = pathlib.Path(str(freeze_receipt.get("path", "")))
    if not freeze_path.is_file() or not contract_path.is_file():
        checks["freeze_fields"] = False
        return
    freeze = _json(freeze_path)
    contract = _json(contract_path)
    checks["freeze_fields"] = (
        freeze.get("schema") == "reference-branch-freeze-v1" and
        freeze.get("protocol_sha256") == PROTOCOL_SHA and
        freeze.get("input_manifest_sha256") == INPUT_MANIFEST_SHA and
        freeze.get("sr_seeds") == list(SEEDS) and freeze.get("arms") == ["FP", "RTN", "SR0", "SR1", "SR2"] and
        freeze.get("primitive_frames") == [0, 5] and freeze.get("model_horizon") == 1)
    checks["contract_schema"] = contract.get("schema") == RAW_SCHEMA
    checks["contract_file"] = contract.get("file") == "raw.npz"
    checks["contract_ids"] = (contract.get("arrays", {}).get("valid_indices", {}).get("dtype") == "int64" and
                               contract.get("arrays", {}).get("trajectory_ids", {}).get("dtype") == "int64" and
                               contract.get("arrays", {}).get("valid_indices", {}).get("values") == list(VALID_INDICES) and
                               contract.get("arrays", {}).get("trajectory_ids", {}).get("values") == list(TRAJECTORY_IDS))
    checks["contract_arms"] = contract.get("arm_order") == ARMS
    checks["contract_quantizer"] = (
        contract.get("quantization", {}).get("bits") == 4 and
        contract.get("quantization", {}).get("qmin") == -7 and
        contract.get("quantization", {}).get("qmax") == 7 and
        contract.get("quantization", {}).get("seeds") == list(SEEDS))
    contract_receipt = engineering.get("receipts", {}).get("contract", {})
    expected_contract = contract_receipt.get("sha256") if isinstance(contract_receipt, dict) else None
    checks["contract_sha256"] = (expected_contract == CONTRACT_SHA and _sha(contract_path) == expected_contract)


def _raw_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]):
    import numpy as np

    raw_path = job / "raw.npz"
    _file(raw_path)
    with np.load(raw_path, allow_pickle=False) as packed:
        required = set(RAW_SHAPES)
        checks["raw_required_keys"] = required.issubset(set(packed.files))
        if not checks["raw_required_keys"]:
            raise ValueError("raw.npz does not contain the frozen key set")
        raw = {key: np.asarray(packed[key]).copy() for key in required}
    raw_receipt = engineering.get("receipts", {}).get("raw", {})
    expected_raw = raw_receipt.get("sha256") if isinstance(raw_receipt, dict) else None
    checks["raw_sha256"] = isinstance(expected_raw, str) and _sha(raw_path) == expected_raw
    for key, shape in RAW_SHAPES.items():
        checks[f"raw_{key}_shape"] = raw[key].shape == shape
        expected_dtype = RAW_DTYPES[key]
        checks[f"raw_{key}_dtype"] = (raw[key].dtype.kind == "U" if expected_dtype == "unicode"
                                       else str(raw[key].dtype) == expected_dtype)
    checks["raw_arm_order"] = raw["arm_names"].tolist() == ARMS
    checks["raw_schema"] = raw["schema"].item() == RAW_SCHEMA
    checks["raw_indices"] = np.array_equal(raw["valid_indices"], np.asarray(VALID_INDICES, dtype=np.int64))
    checks["raw_trajectories"] = np.array_equal(raw["trajectory_ids"], np.asarray(TRAJECTORY_IDS, dtype=np.int64))
    checks["raw_complete"] = raw["completed"].dtype == np.bool_ and bool(raw["completed"].all())
    checks["raw_finite"] = all(bool(np.isfinite(raw[key]).all()) for key in RAW_SHAPES
                                 if RAW_DTYPES[key] == "float32")
    checks["raw_receipt_matches"] = checks["raw_sha256"]
    return raw


def _as_cpu_tensor(value: Any, torch: Any):
    if not torch.is_tensor(value):
        raise ValueError("quant snapshot field is not a torch tensor")
    return value.detach().cpu().contiguous()


def _tensor_sha(value: Any, torch: Any) -> str:
    tensor = _as_cpu_tensor(value, torch)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _quant_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    import torch

    path = job / "quant_snapshot.pt"
    _file(path)
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    modules = snapshot.get("modules") if isinstance(snapshot, dict) else None
    checks["snapshot_modules"] = isinstance(modules, dict) and len(modules) == 48
    if not checks["snapshot_modules"]:
        raise ValueError("quant_snapshot.pt must contain exactly 48 modules")
    names = sorted(modules)
    checks["snapshot_name_order"] = list(modules) == names
    checks["snapshot_target_paths"] = engineering.get("target_paths") == names
    checks['snapshot_expected_paths'] = tuple(names) == EXPECTED_TARGET_PATHS
    checks["snapshot_schema"] = snapshot.get("schema") == "reference-branch-coupling-quant-v1"
    checks["snapshot_keying"] = snapshot.get("module_keying") == "complete_path"
    checks["snapshot_arms"] = snapshot.get("arms") == ["RTN", "SR0", "SR1", "SR2"]
    checks["quant_rebuild"] = True
    checks["quant_readback_sha"] = True
    checks["quant_restore_sha"] = True
    generator = {seed: torch.Generator(device="cpu").manual_seed(seed) for seed in SEEDS}
    qmax = torch.tensor(7.0, dtype=torch.float32)
    for name in names:
        row = modules[name]
        fp = _as_cpu_tensor(row.get("fp"), torch)
        scale = _as_cpu_tensor(row.get("scale"), torch)
        codes = _as_cpu_tensor(row.get("codes"), torch)
        if fp.dtype != torch.float32 or scale.dtype != torch.float32 or codes.dtype != torch.int8:
            checks["quant_rebuild"] = False
            continue
        checks["quant_rebuild"] &= row.get("shape") == list(fp.shape)
        rows = fp.reshape(fp.shape[0], -1)
        maximum = rows.abs().amax(dim=1)
        expected_scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / qmax)
        checks["quant_rebuild"] &= bool(scale.shape == (fp.shape[0],) and torch.equal(scale, expected_scale))
        ratio = rows / scale.reshape(-1, 1)
        rtn = torch.clamp(torch.round(ratio), -7, 7).to(torch.int8)
        expected_codes = [rtn]
        for seed in SEEDS:
            lower = torch.floor(ratio)
            fraction = ratio - lower
            uniform = torch.rand(ratio.shape, generator=generator[seed], dtype=torch.float32)
            code = torch.clamp(lower + (uniform < fraction).to(torch.float32), -7, 7).to(torch.int8)
            code = torch.where(maximum.reshape(-1, 1) == 0, torch.zeros_like(code), code)
            expected_codes.append(code)
        checks["quant_rebuild"] &= bool(codes.shape == (4,) + tuple(fp.shape) and
                                         all(torch.equal(codes[i], expected_codes[i].reshape(fp.shape))
                                             for i in range(4)))
        readbacks = row.get("readback_sha256")
        restore = row.get("restore_sha256")
        checks["quant_restore_sha"] &= isinstance(restore, str) and restore == _tensor_sha(fp, torch)
        checks["quant_readback_sha"] &= isinstance(readbacks, list) and len(readbacks) == 4
        if isinstance(readbacks, list) and len(readbacks) == 4:
            for i, code in enumerate(expected_codes):
                dequant = (code.to(torch.float32) * scale.reshape(-1, 1)).reshape_as(fp)
                checks["quant_readback_sha"] &= readbacks[i] == _tensor_sha(dequant, torch)
        checks["quant_rebuild"] &= bool(torch.isfinite(fp).all() and torch.isfinite(scale).all() and
                                         (scale > 0).all() and torch.isfinite(codes.to(torch.float32)).all())


def _gate_checks(engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    gates = engineering.get("checks")
    required = ("v100", "v100_capability", "eval_frozen", "final_restore",
                "all_completed", "finite_raw", "quant_receipts_complete",
                "predictor_input_readback", "fp32_execution", "nonvisual_input_unchanged")
    checks["producer_checks_present"] = isinstance(gates, dict)
    for key in required:
        checks[f"producer_{key}"] = isinstance(gates, dict) and gates.get(key) is True
    # Per-state/per-arm checks are created by reference_screen.py.  Requiring
    # every expected key prevents a short or hand-edited receipt from passing.
    per_state = [f"fp_copy_{state}" for state in range(6)] + [f"fp_restore_{state}" for state in range(6)]
    per_arm = [f"{kind}_{arm}" for arm in range(1, 5)
               for kind in ("pristine_before", "non_target", "restore")]
    for key in per_state + per_arm:
        checks[f"producer_{key}"] = isinstance(gates, dict) and gates.get(key) is True
    checks["target_count"] = engineering.get("target_count") == 48
    checks["target_path_allowlist"] = (engineering.get("target_paths") == list(EXPECTED_TARGET_PATHS))
    checks["completed_matrix"] = engineering.get("completed") == [[True] * 5 for _ in range(6)]
    checks["arm_receipt"] = engineering.get("arm_names") == ARMS
    checks["valid_receipt"] = engineering.get("valid_indices") == list(VALID_INDICES)
    checks["trajectory_receipt"] = engineering.get("trajectory_ids") == list(TRAJECTORY_IDS)
    checks["quantization_receipt"] = engineering.get("quantization") == {
        "bits": 4, "qmin": -7, "qmax": 7, "seeds": list(SEEDS),
        "traversal": "complete module path lexicographic; one float32 rand per module/draw"}
    before = engineering.get("state_before")
    after = engineering.get("state_after")
    non_before = engineering.get("non_target_state_before")
    non_after = engineering.get("non_target_state_after")
    checks["state_restore_digest"] = (isinstance(before, str) and len(before) == 64 and before == after)
    checks["non_target_restore_digest"] = (isinstance(non_before, str) and len(non_before) == 64 and non_before == non_after)


def _load_replay(path: pathlib.Path):
    _file(path)
    if _sha(path) != REPLAY_SHA:
        raise ValueError("reference_replay.py SHA-256 differs from frozen independent replay")
    spec = importlib.util.spec_from_file_location("reference_branch_independent_replay", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load independent replay")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "replay", None)):
        raise ValueError("independent replay has no replay(raw) function")
    return module


def verify(args: argparse.Namespace, allocation: dict[str, Any]) -> dict[str, Any]:
    job = args.input.resolve()
    checks: dict[str, bool] = {}
    report: dict[str, Any] = {
        "schema": "reference-branch-coupling-independent-verification-v1",
        "source_job_dir": str(job), "allocation": allocation, "checks": checks,
        "decision": "implementation_inconclusive",
    }
    engineering_path = job / "engineering.json"
    if not engineering_path.is_file():
        report["reason"] = "missing engineering.json"
        return report
    engineering = _json(engineering_path)
    report["producer_status"] = engineering.get("status")
    if engineering.get("status") != "complete":
        exit_path = job / "exit_status.json"
        exit_code = _json(exit_path).get("exit_code") if exit_path.is_file() else None
        report["exit_code"] = exit_code
        report["reason"] = engineering.get("error", "producer did not complete")
        report["decision"] = ("inconclusive_budget" if exit_code in (124, 137) or
                               engineering.get("status") in ("running", "timeout", "partial")
                               else "implementation_inconclusive")
        return report
    _allocation_checks(job, engineering, checks)
    _receipt_checks(job, engineering, checks)
    _identity_checks(job, engineering, checks)
    _freeze_checks(job, args.contract.resolve(), engineering, checks)
    _gate_checks(engineering, checks)
    if not all(checks.values()):
        report["decision"] = "implementation_inconclusive"
        report["failed_checks"] = [key for key, value in checks.items() if not value]
        return report
    raw = _raw_checks(job, engineering, checks)
    _quant_checks(job, engineering, checks)
    if not all(checks.values()):
        report["decision"] = "implementation_inconclusive"
        report["failed_checks"] = [key for key, value in checks.items() if not value]
        return report
    replay = _load_replay(args.replay)
    science = replay.replay(raw)
    report["scientific_replay"] = science
    report["decision"] = science.get("decision", "implementation_inconclusive")
    report["raw_sha256"] = _sha(job / "raw.npz")
    report["quant_snapshot_sha256"] = _sha(job / "quant_snapshot.pt")
    report["scope"] = "CPU checks rebuild the producer quantizer; scientific values come only from the frozen independent replay."
    return report


def main() -> None:
    # The guard intentionally precedes argument-dependent I/O and NumPy/Torch imports.
    from allocation_guard import require_allocation

    allocation = require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--replay", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = verify(args, allocation)
    except Exception as exc:  # preserve a compact, actionable failure receipt
        result = {
            "schema": "reference-branch-coupling-independent-verification-v1",
            "allocation": allocation, "source_job_dir": str(args.input.resolve()),
            "decision": "implementation_inconclusive", "error_type": type(exc).__name__,
            "error": str(exc),
        }
    _atomic_json(args.output / "verification.json", result)
    print(json.dumps({"decision": result["decision"],
                      "failed_checks": [k for k, v in result.get("checks", {}).items() if not v]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

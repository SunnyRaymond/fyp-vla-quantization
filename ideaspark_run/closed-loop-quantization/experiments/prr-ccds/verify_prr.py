"""Independent CPU verifier for the PRR CCDS-TC1 R1 artifact contract.
The allocation guard is the first artifact operation. Raw arrays, canonical
record pickles, and checkpoint hashes are read only inside a real SLURM CPU
allocation; this program never loads the model or connects to a cluster.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
ARMS = ("q_local", "q_recovery", "l_local", "l_recovery")
SEEDS = (1201, 1202, 1203)
EVALUATORS = tuple(f"{arm}:{seed}" for arm in ARMS for seed in SEEDS)
CAL_INDICES, DEV_INDICES = tuple(range(72, 78)), tuple(range(78, 84))
CAL_EPISODES = tuple(f"prr_cal:{i:03d}" for i in range(6))
DEV_EPISODES = tuple(f"prr_dev:{i:03d}" for i in range(6))
DONORS = ("q0", "clean_seed_1201")
EPS, REL_TOL, MAX_ERRORS = 1e-12, 1e-6, 24
SHA256 = re.compile(r"^[0-9a-f]{64}$")
def _errors(values: Sequence[str]) -> list[str]:
    out = [str(x)[:240] for x in values[:MAX_ERRORS]]
    return out + ([f"... {len(values) - MAX_ERRORS} more errors"] if len(values) > MAX_ERRORS else [])
def _json(path: Path, limit: int = 65536) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    if path.stat().st_size > limit:
        raise ValueError(f"{path.name} exceeds {limit} bytes")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value
def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(data.encode("utf-8")) > 65536:
        raise RuntimeError("verification summary exceeds 64 KiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
def _guard() -> Mapping[str, Any]:
    path = Path(__file__).with_name("allocation_guard.py")
    spec = importlib.util.spec_from_file_location("prr_allocation_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("allocation_guard.py is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.require_allocation()
def _finite(value: Any, positive: bool = False) -> bool:
    try:
        x = float(value)
        return math.isfinite(x) and (x > 0 if positive else True)
    except (TypeError, ValueError):
        return False
def _manifest_audit(m: Mapping[str, Any]) -> list[str]:
    e: list[str] = []
    if m.get("schema") != "prr-ccds-manifest-v1" or m.get("protocol_id") != "prr-ccds-v1":
        e.append("manifest schema/protocol mismatch")
    scope = m.get("research_scope")
    if not isinstance(scope, Mapping) or not isinstance(scope.get("engineering_only"), bool):
        e.append("research_scope.engineering_only must be boolean")
    if isinstance(scope, Mapping) and any(scope.get(k) is not False for k in ("r2", "r3", "test_opened")):
        e.append("R2/R3/TEST must remain closed")
    source = m.get("source_identity")
    expected = {"checkpoint_sha256": "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b", "checkpoint_epoch": 65,
                "source_commit": "0a9492fa12044b852ae9e001cc74604b79c8bb0c", "dinov2_source_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                "dtype": "float32", "decoder": None, "native_memory_claim": False}
    if not isinstance(source, Mapping):
        e.append("source_identity missing")
    else:
        e.extend(f"source_identity.{k} mismatch" for k, v in expected.items() if source.get(k) != v)
    cluster = m.get("cluster")
    if not isinstance(cluster, Mapping) or cluster.get("scheduler") != "slurm" or cluster.get("name") != "CCDS-TC1":
        e.append("cluster must be CCDS-TC1 SLURM")
    elif cluster.get("max_concurrent_gpu") != 1 or float(cluster.get("gpu_hours_cap", -1)) != 6.0:
        e.append("single-GPU six-hour cap mismatch")
    splits = m.get("splits")
    if not isinstance(splits, Mapping):
        e.append("splits missing")
    else:
        if splits.get("historical_reserved") != [0, 71]:
            e.append("historical reserve must be 0..71")
        if splits.get("test_locked") != list(range(84, 96)):
            e.append("TEST 84..95 must remain locked")
        for name, indices, episodes, env, cem in (("cal", CAL_INDICES, CAL_EPISODES, 1000000, 1010000), ("dev", DEV_INDICES, DEV_EPISODES, 1100000, 1110000)):
            node = splits.get(name)
            if not isinstance(node, Mapping):
                e.append(f"{name} split missing")
                continue
            if tuple(node.get("dataset_indices", ())) != indices or tuple(node.get("episode_ids", ())) != episodes:
                e.append(f"{name} dataset/episode IDs mismatch")
            if node.get("windows_per_episode") != 2 or node.get("record_count") != 12:
                e.append(f"{name} must have six episodes and two windows")
            if node.get("environment_namespace") != env or node.get("cem_namespace") != cem:
                e.append(f"{name} namespace mismatch")
    fit = m.get("fit")
    if not isinstance(fit, Mapping):
        e.append("fit section missing")
    else:
        checks = {"arms": list(ARMS), "seeds": list(SEEDS), "updates": 1000, "fit_updates": 1000, "batch_size": 2, "horizon": 2, "gamma": 1.0,
                  "optimizer": "Adam", "quantizer_lr": .01, "lora_lr": .001, "temperature": [2.0, .1], "rounding_regularization": [.01, .1]}
        e.extend(f"fit.{k} mismatch" for k, v in checks.items() if fit.get(k) != v)
    quant = m.get("quantizer")
    if not isinstance(quant, Mapping):
        e.append("quantizer section missing")
    else:
        checks = {"bits": 4, "signed": True, "q_min": -7, "q_max": 7, "scheme": "symmetric_per_output_channel_weight_only", "target_group_count": 24, "alpha_discarded": True, "float_bypass": False}
        e.extend(f"quantizer.{k} mismatch" for k, v in checks.items() if quant.get(k) != v)
        if quant.get("hardening") != "clip(floor(W_eff/s)+1[sigmoid(alpha)>=0.5], -7, 7), integer reload required":
            e.append("hardening contract mismatch")
    lora = m.get("lora")
    if not isinstance(lora, Mapping) or lora.get("rank") != 4 or lora.get("merge_before_quantization") is not True or lora.get("a_initialization") != "seeded_random_per_seed" or lora.get("b_initialization") != "zeros" or lora.get("both_zero_forbidden") is not True:
        e.append("LoRA rank/merge/initialization contract mismatch")
    contract = m.get("artifact_contract")
    if not isinstance(contract, Mapping) or contract.get("raw_final") != "raw_final.npz" or contract.get("summary") != "raw_final_summary.json":
        e.append("artifact filenames mismatch")
    elif contract.get("donor_ids") != list(DONORS) or contract.get("summary_max_bytes") != 65536 or contract.get("raw_arrays_stay_on_compute") is not True:
        e.append("artifact donor/size/location policy mismatch")
    gates = m.get("gates")
    if not isinstance(gates, Mapping) or float(gates.get("pair_ratio", -1)) != .95 or float(gates.get("clean_ratio", -1)) != 1.1:
        e.append("gate ratios mismatch")
    return _errors(e)
def _summary_audit(s: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    e: list[str] = []
    checks = {"schema": "prr-final-v1", "status": "complete", "formal": True, "scope": "R1_only", "arms": list(ARMS), "seeds": list(SEEDS), "fit_updates": 1000,
              "batch_size": 2, "gamma": 1.0, "horizon": 2, "cal_records": 12, "dev_records": 12, "donors": list(DONORS), "dev_tuning": False, "r2_opened": False, "test_opened": False, "execution": "fake_quant_emulation_only"}
    e.extend(f"summary.{k} mismatch" for k, v in checks.items() if s.get(k) != v)
    shapes = s.get("raw_shapes")
    prefixes = {"terminal_error": (12, 12, 1), "clean_error": (12, 12, 1), "donor_recovery_error": (12, 2, 12, 1), "fp_first": (12, 1), "fp_second": (12, 1), "fp_donor_local_target": (2, 12, 1), "fp_donor_recovery_target": (2, 12, 1)}
    if not isinstance(shapes, Mapping):
        e.append("summary.raw_shapes missing")
    else:
        for k, prefix in prefixes.items():
            shape = shapes.get(k)
            if not isinstance(shape, list) or tuple(shape[:len(prefix)]) != prefix:
                e.append(f"summary.raw_shapes.{k} mismatch")
    runs = s.get("fit_runs")
    seen: set[str] = set()
    if not isinstance(runs, list) or len(runs) != 12:
        e.append("summary.fit_runs must contain 12 compact references")
        runs = []
    for row in runs:
        if not isinstance(row, Mapping):
            e.append("summary fit run is not an object")
            continue
        key = f"{row.get('arm')}:{row.get('seed')}"
        seen.add(key)
        if key not in EVALUATORS:
            e.append(f"unexpected fit run {key}")
        if "hard_ledger" in row or "hard_payload" in row:
            e.append(f"{key} detailed ledger leaked into compact summary")
    if seen != set(EVALUATORS):
        e.append("summary fit run set is incomplete")
    return _errors(e), {"fit_runs": len(seen), "summary_schema": s.get("schema")}
def _strings(value: Any) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(value).reshape(-1).tolist()]
def _record_audit(rows: Any, split: str) -> list[str]:
    e: list[str] = []
    indices = CAL_INDICES if split == "cal" else DEV_INDICES
    episodes = CAL_EPISODES if split == "cal" else DEV_EPISODES
    if not isinstance(rows, list) or len(rows) != 12:
        return [f"metadata.{split} rows must contain 12 records"]
    got: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            e.append(f"{split} record is not an object")
            continue
        ep, idx = row.get("episode_id"), row.get("dataset_index")
        got.append(str(ep))
        if ep not in episodes or idx not in indices:
            e.append(f"{split} record has invalid episode/index")
            continue
        local = indices.index(idx)
        if ep != episodes[local] or row.get("env_seed") != (1000000 if split == "cal" else 1100000) + local or row.get("cem_seed") != (1010000 if split == "cal" else 1110000) + local:
            e.append(f"{split} namespace/index mismatch")
        if not _finite(row.get("record_index")) or not _finite(row.get("window_start")):
            e.append(f"{split} record/window index missing")
    if any(got.count(ep) != 2 for ep in episodes):
        e.append(f"{split} must contain two windows per episode")
    return _errors(e)
def _metadata_audit(meta: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    e: list[str] = []
    if meta.get("schema") != "prr-raw-final-v1" or meta.get("donors") != list(DONORS) or meta.get("arm_order") != list(EVALUATORS):
        e.append("raw metadata schema/donor/arm order mismatch")
    if meta.get("dev_tuning") is not False or meta.get("test_opened") is not False or meta.get("fp_batch_size") is not None:
        e.append("metadata tuning/TEST/FP batch policy mismatch")
    for k, v in (("gamma", 1.0), ("horizon", 2), ("target", "fp_second_on_same_full_DEV_batch"), ("clean_anchor", "fp_first_on_same_full_DEV_batch")):
        if meta.get(k) != v:
            e.append(f"metadata.{k} mismatch")
    e.extend(_record_audit(meta.get("record_rows"), "dev"))
    if "cal_record_rows" in meta:
        e.extend(_record_audit(meta.get("cal_record_rows"), "cal"))
    refs = meta.get("hard_checkpoint_ledger")
    if not isinstance(refs, list) or len(refs) != 12:
        e.append("metadata hard checkpoint references must contain 12 rows")
    else:
        seen = {f"{x.get('arm')}:{x.get('seed')}" for x in refs if isinstance(x, Mapping)}
        if seen != set(EVALUATORS):
            e.append("metadata checkpoint reference set is incomplete")
        if any(not isinstance(x, Mapping) or "hard_payload" in x for x in refs):
            e.append("metadata checkpoint references contain weight payload")
    runtime = meta.get("runtime_identity")
    source = manifest.get("source_identity", {})
    if isinstance(runtime, Mapping):
        for k in ("checkpoint_sha256", "checkpoint_epoch", "source_commit", "dinov2_source_commit", "dtype", "decoder", "native_memory_claim"):
            if runtime.get(k) != source.get(k):
                e.append(f"metadata.runtime_identity.{k} mismatch")
    return _errors(e)
def _resolve(value: Any, base: Path, default: Path) -> Path:
    if value is None or not str(value):
        return default
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()
def _load_pickle(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a mapping")
    return value
def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in ("episode_id", "record_index", "dataset_index", "window_start"))
def _canonical_audit(manifest: Mapping[str, Any], manifest_base: Path, artifact_dir: Path, meta: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    e: list[str] = []
    fingerprints: dict[str, set[str]] = {"episode_fingerprint": set(), "initial_state_fingerprint": set(), "initial_condition_fingerprint": set()}
    condition_owners: dict[str, str] = {}
    for split, expected_eps in (("cal", CAL_EPISODES), ("dev", DEV_EPISODES)):
        section = manifest.get("splits", {}).get(split, {}) if isinstance(manifest.get("splits"), Mapping) else {}
        path = _resolve(section.get("path") if isinstance(section, Mapping) else None, manifest_base, artifact_dir / f"{split}.pkl")
        try:
            payload = _load_pickle(path)
            rows = payload.get("records")
            if not isinstance(rows, list) or len(rows) != 12:
                raise ValueError("canonical record count is not 12")
            row_map = {_key(row): row for row in rows if isinstance(row, Mapping)}
            supplied = meta.get("cal_record_rows") if split == "cal" else meta.get("record_rows")
            if isinstance(supplied, list):
                for row in supplied:
                    if _key(row) not in row_map:
                        e.append(f"{split} metadata row absent from canonical records")
            else:
                supplied = rows
            for row in supplied:
                canonical = row_map.get(_key(row))
                if not isinstance(canonical, Mapping):
                    continue
                for name in fingerprints:
                    value = canonical.get(name)
                    if isinstance(value, str) and SHA256.fullmatch(value):
                        fingerprints[name].add(value)
                    else:
                        e.append(f"{split} canonical {name} missing")
                condition = canonical.get("initial_condition_fingerprint")
                owner = str(canonical.get("episode_id"))
                if condition in condition_owners and condition_owners[condition] != owner:
                    e.append("initial condition duplicated across canonical episodes")
                condition_owners[condition] = owner
        except Exception as exc:
            e.append(f"{split} canonical record audit: {type(exc).__name__}: {exc}")
    provenance_value = manifest.get("provenance_audit")
    provenance_path = _resolve(provenance_value, manifest_base, artifact_dir / "provenance_audit.json")
    try:
        provenance = _json(provenance_path, 4 * 1024 * 1024)
        if provenance.get("schema") != "prr-provenance-audit-v1" or provenance.get("status") not in ("complete", "complete_with_coverage_limitation") or provenance.get("test_opened") is not False:
            e.append("provenance audit is incomplete/rejected or TEST opened")
        if provenance.get("historical_valid_index_range") != [0, 71]:
            e.append("historical valid range mismatch")
        if provenance.get("errors"):
            e.append("provenance audit reports errors")
        new_rows = provenance.get("new_episode_rows")
        if not isinstance(new_rows, list) or len(new_rows) != 12:
            e.append("provenance audit must cover twelve new episodes")
        else:
            originals = [row.get("source_underlying_index") for row in new_rows]
            historical = set(provenance.get("historical_underlying_ids", {}).values())
            conditions = [row.get("initial_condition_fingerprint") for row in new_rows]
            if len(set(originals)) != 12 or set(originals) & historical:
                e.append("new underlying episodes duplicate or overlap history")
            if len(set(conditions)) != 12 or any(not isinstance(x, str) or not SHA256.fullmatch(x) for x in conditions):
                e.append("new initial conditions missing or duplicated")
            for row in new_rows:
                if not isinstance(row, Mapping):
                    continue
                for name in fingerprints:
                    value = row.get(name)
                    if isinstance(value, str) and SHA256.fullmatch(value):
                        fingerprints[name].add(value)
    except Exception as exc:
        e.append(f"provenance audit: {type(exc).__name__}: {exc}")
    return _errors(e), {"fingerprint_counts": {k: len(v) for k, v in fingerprints.items()}, "coverage": provenance.get("historical_fingerprint_registry", {}).get("coverage_complete") if 'provenance' in locals() else None}

def _runtime_audit(root: Path, manifest_path: Path, manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    e: list[str] = []
    progress = _json(root / "progress.json")
    identity = progress.get("identity", {})
    actual = identity.get("runtime_identity", {})
    if progress.get("status") != "complete" or set(progress.get("completed", [])) != set(EVALUATORS):
        e.append("formal progress is incomplete")
    if identity.get("manifest_sha256") != _sha256_file(manifest_path):
        e.append("formal manifest digest differs from runtime identity")
    for key, expected in manifest.get("runtime_identity", {}).items():
        if actual.get(key) != expected:
            e.append(f"runtime identity mismatch: {key}")
    if actual.get("checkpoint_sha256") != manifest.get("checkpoint_sha256") or "V100" not in str(actual.get("gpu")):
        e.append("FP checkpoint or V100 runtime mismatch")
    if actual.get("native_memory_claim") is not False:
        e.append("unsupported native deployment claim")
    engineering = _json(root / "engineering_summary.json")
    if engineering.get("engineering_pass") is not True or engineering.get("dev_opened") is not False:
        e.append("CAL engineering gate failed or used DEV")
    if not all(v.get("passed") is True for v in engineering.get("gates", {}).values()) or len(engineering.get("gates", {})) != 4:
        e.append("CAL engineering source/null/action/history checks incomplete")
    if summary.get("raw_final_sha256") != _sha256_file(root / "raw_final.npz"):
        e.append("raw NPZ digest differs from final summary")
    return e
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    return value
def _tensor_bytes(value: Any) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()
def _hard_digest(hard: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted(hard):
        digest.update(str(path).encode() + b"\0")
        digest.update(_tensor_bytes(hard[path]["integer"]))
        digest.update(_tensor_bytes(hard[path]["scale"]))
    return digest.hexdigest()
def _payload_audit(payload: Mapping[str, Any], arm: str, seed: int, fit: Mapping[str, Any]) -> tuple[list[str], str | None]:
    e: list[str] = []
    if payload.get("schema") != "prr-hard-checkpoint-v1" or payload.get("arm") != arm or payload.get("seed") != seed:
        e.append(f"{arm}:{seed} checkpoint schema/identity mismatch")
    for key in ("teacher_removed", "lora_removed", "alpha_removed"):
        if payload.get(key) is not True:
            e.append(f"{arm}:{seed} {key} flag is not true")
    state, hard, rows = payload.get("state_dict"), payload.get("hard_quant"), payload.get("hard_ledger")
    if not isinstance(state, Mapping) or not isinstance(hard, Mapping) or not isinstance(rows, list):
        return e + [f"{arm}:{seed} checkpoint state/hard payload missing"], None
    if len(hard) != 24 or len(rows) != 24:
        e.append(f"{arm}:{seed} hard target count is not 24")
    paths = {str(x) for x in hard}
    target_names = {f"{x}.weight" for x in paths}
    for key, value in state.items():
        name = str(key).lower()
        if str(key) in target_names or any(x in name for x in ("lora", "alpha", "scale_raw", "parametrizations")):
            e.append(f"{arm}:{seed} state_dict contains target/LoRA/alpha parameter {key}")
        if not hasattr(value, "dtype"):
            e.append(f"{arm}:{seed} state_dict value is not a tensor")
    by_path = {row.get("path"): row for row in rows if isinstance(row, Mapping)}
    if set(by_path) != paths:
        e.append(f"{arm}:{seed} hard ledger/payload path set mismatch")
    for path, item in hard.items():
        if not isinstance(item, Mapping) or set(item) != {"integer", "scale"}:
            e.append(f"{arm}:{seed} {path} has duplicate/extra hard payload fields")
            continue
        integer, scale = item.get("integer"), item.get("scale")
        if str(getattr(integer, "dtype", "")) != "torch.int8" or str(getattr(scale, "dtype", "")) != "torch.float32":
            e.append(f"{arm}:{seed} {path} integer/scale dtype mismatch")
            continue
        if integer.ndim < 2 or scale.ndim != 1 or integer.shape[0] != scale.shape[0] or int(integer.min()) < -7 or int(integer.max()) > 7:
            e.append(f"{arm}:{seed} {path} hard grid/shape invalid")
        if not np.isfinite(scale.numpy()).all() or bool((scale <= 0).any()):
            e.append(f"{arm}:{seed} {path} scale is not finite positive")
        row = by_path.get(path)
        if isinstance(row, Mapping):
            if row.get("q_min") != -7 or row.get("q_max") != 7 or row.get("integer_min") != int(integer.min()) or row.get("integer_max") != int(integer.max()):
                e.append(f"{arm}:{seed} {path} ledger extrema/q range mismatch")
            if row.get("integer_sha256") != hashlib.sha256(_tensor_bytes(integer)).hexdigest() or row.get("scale_sha256") != hashlib.sha256(_tensor_bytes(scale)).hexdigest():
                e.append(f"{arm}:{seed} {path} hard tensor hash mismatch")
            if row.get("use_lora") is not arm.startswith("l_"):
                e.append(f"{arm}:{seed} {path} LoRA usage mismatch")
    digest = _hard_digest(hard) if hard else None
    if fit.get("hard_map_sha256") is not None and fit.get("hard_map_sha256") != digest:
        e.append(f"{arm}:{seed} hard_map_sha256 mismatch")
    return e, digest
def _fit_audit(root: Path) -> tuple[list[str], dict[str, Any]]:
    e: list[str] = []
    schedules: dict[int, set[str]] = {seed: set() for seed in SEEDS}
    info: dict[str, Any] = {"completed": [], "hard_map_sha256": {}}
    methods = root / "methods"
    for arm in ARMS:
        for seed in SEEDS:
            key, json_path, pt_path = f"{arm}:{seed}", methods / f"{arm}_seed_{seed}.json", methods / f"{arm}_seed_{seed}.pt"
            try:
                fit = _json(json_path, 8 * 1024 * 1024)
            except Exception as exc:
                e.append(f"{key} fit ledger: {exc}")
                continue
            if fit.get("schema") != "prr-fit-v1" or fit.get("status") != "complete" or fit.get("arm") != arm or fit.get("seed") != seed:
                e.append(f"{key} fit identity/status mismatch")
            for field, value in (("fit_updates_completed", 1000), ("fit_updates_expected", 1000), ("batch_size", 2), ("gamma", 1.0), ("reload_equal", True), ("finite_grad", True)):
                if fit.get(field) != value:
                    e.append(f"{key} fit.{field} mismatch")
            schedule = fit.get("batch_schedule_hash")
            if not isinstance(schedule, str) or not SHA256.fullmatch(schedule):
                e.append(f"{key} batch schedule hash missing")
            else:
                schedules[seed].add(schedule)
            grad, updates, us = fit.get("gradient_summary"), fit.get("update_norms"), fit.get("update_summary")
            if not isinstance(grad, Mapping) or not isinstance(updates, Mapping) or not isinstance(us, Mapping):
                e.append(f"{key} gradient/update diagnostics missing")
            else:
                qgrad = bool(fit.get("quant_grad_after_step2")) and (_finite(grad.get("scale_raw_max_after_step2"), True) or _finite(grad.get("alpha_max_after_step2"), True))
                if not qgrad or not _finite(us.get("quantizer_max"), True):
                    e.append(f"{key} quantizer gradient/update ineffective")
                if arm.startswith("l_"):
                    ag = bool(fit.get("lora_A_grad_after_step2")) and _finite(grad.get("lora_A_max_after_step2"), True)
                    bg = bool(fit.get("lora_B_grad_after_step2")) and _finite(grad.get("lora_B_max_after_step2"), True)
                    if not ag or not bg or not _finite(us.get("lora_A_max"), True) or not _finite(us.get("lora_B_max"), True):
                        e.append(f"{key} LoRA A/B gradient/update ineffective")
            if not pt_path.is_file():
                e.append(f"{key} hard checkpoint missing")
                continue
            actual = _sha256_file(pt_path)
            if not isinstance(fit.get("checkpoint_sha256"), str) or fit.get("checkpoint_sha256") != actual:
                e.append(f"{key} actual checkpoint SHA256 mismatch/missing")
            try:
                payload = _torch_load(pt_path)
                pe, digest = _payload_audit(payload, arm, seed, fit)
                e.extend(pe)
                if digest:
                    info["hard_map_sha256"][key] = digest
            except Exception as exc:
                e.append(f"{key} checkpoint audit: {type(exc).__name__}: {exc}")
            info["completed"].append(key)
    for seed, values in schedules.items():
        if len(values) != 1:
            e.append(f"batch schedule differs across arms at seed {seed}")
    return _errors(e), info
def _raw_audit(path: Path) -> tuple[list[str], dict[str, Any]]:
    if not path.is_file():
        return ["raw_final.npz is missing"], {}
    required = {"arm_ids", "clean_error", "terminal_error", "donor_recovery_error", "fp_first", "fp_second", "fp_donor_local_target", "fp_donor_recovery_target", "wz", "action_mask", "metadata_json"}
    e: list[str] = []
    with np.load(path, allow_pickle=False) as z:
        missing = sorted(required - set(z.files))
        if missing:
            return [f"raw_final.npz missing {', '.join(missing)}"], {}
        if _strings(z["arm_ids"]) != list(EVALUATORS):
            e.append("raw evaluator order must be arm-major with seeds 1201..1203")
        terminal, clean, donor = (np.asarray(z[k]) for k in ("terminal_error", "clean_error", "donor_recovery_error"))
        if terminal.ndim != 5 or terminal.shape[:3] != (12, 12, 1):
            e.append("terminal_error shape must be [12,12,1,P,D]")
        if clean.shape != terminal.shape or donor.ndim != 6 or donor.shape[:4] != (12, 2, 12, 1) or donor.shape[-2:] != terminal.shape[-2:]:
            e.append("clean/donor error shape mismatch")
        wz, action = np.asarray(z["wz"]), np.asarray(z["action_mask"])
        if terminal.ndim == 5 and (wz.shape != terminal.shape[-2:] or action.shape != terminal.shape[-2:]):
            e.append("wz/action mask shape mismatch")
        if action.dtype != np.bool_ or not np.any(action) or not np.any(~action):
            e.append("action_mask must be boolean with action and active coordinates")
        if not np.issubdtype(wz.dtype, np.number) or not np.isfinite(wz).all() or np.any(wz[action] != 0) or np.any(wz[~action] <= 0):
            e.append("wz has invalid action/active values")
        for name, array in (("terminal_error", terminal), ("clean_error", clean), ("donor_recovery_error", donor)):
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                e.append(f"{name} is nonnumeric or nonfinite")
            elif action.shape == array.shape[-2:] and np.any(array[..., action] != 0):
                e.append(f"{name} has nonzero copied-action error")
        for name, prefix in (("fp_first", (12, 1)), ("fp_second", (12, 1)), ("fp_donor_local_target", (2, 12, 1)), ("fp_donor_recovery_target", (2, 12, 1))):
            array = np.asarray(z[name])
            if array.ndim < len(prefix) or tuple(array.shape[:len(prefix)]) != prefix or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                e.append(f"{name} shape or finite-value check failed")
        try:
            texts = _strings(z["metadata_json"])
            if len(texts) != 1:
                raise ValueError("metadata_json must be scalar")
            meta = json.loads(texts[0])
            if not isinstance(meta, Mapping):
                raise ValueError("metadata_json must decode to object")
        except Exception as exc:
            e.append(f"metadata_json invalid: {exc}")
            return _errors(e), {}
        if terminal.ndim == 5 and donor.ndim == 6 and wz.shape == terminal.shape[-2:]:
            metric = lambda x: np.mean((x.astype(np.float64) * wz) ** 2, axis=(-3, -2, -1))
            return _errors(e), {"terminal_mse": metric(terminal), "clean_mse": metric(clean), "donor_mse": np.mean((donor * wz) ** 2, axis=(-4, -3, -2, -1)), "metadata": meta, "episodes": [str(row.get("episode_id")) for row in meta.get("record_rows", []) if isinstance(row, Mapping)]}
    return _errors(e), {}
def _episode_matrix(metric: np.ndarray, episodes: Sequence[str], rows: Sequence[int]) -> dict[int, np.ndarray]:
    return {seed: np.asarray([float(np.mean(metric[row, [ep == name for ep in episodes]])) for name in DEV_EPISODES]) for seed, row in zip(SEEDS, rows)}
def _pair_gate(metric: np.ndarray, clean: np.ndarray, episodes: Sequence[str], local_arm: str, recovery_arm: str) -> dict[str, Any]:
    li = [EVALUATORS.index(f"{local_arm}:{seed}") for seed in SEEDS]
    ri = [EVALUATORS.index(f"{recovery_arm}:{seed}") for seed in SEEDS]
    local, recovery = _episode_matrix(metric, episodes, li), _episode_matrix(metric, episodes, ri)
    lc, rc = _episode_matrix(clean, episodes, li), _episode_matrix(clean, episodes, ri)
    gl, gr, gcl, gcr = (float(np.mean([np.mean(x) for x in v.values()])) for v in (local, recovery, lc, rc))
    rows: dict[str, Any] = {}
    passed, ambiguous = 0, False
    for seed in SEEDS:
        l, r, lclean, rclean = float(np.mean(local[seed])), float(np.mean(recovery[seed])), float(np.mean(lc[seed])), float(np.mean(rc[seed]))
        primary = r <= .95 * l + EPS and int(np.sum(recovery[seed] < local[seed])) >= 4
        clean_pass = rclean <= 1.10 * lclean + EPS
        near = abs(r - .95 * l - EPS) <= REL_TOL * max(abs(.95 * l), EPS) or abs(rclean - 1.10 * lclean - EPS) <= REL_TOL * max(abs(1.10 * lclean), EPS)
        ambiguous |= near
        passed += int(primary and clean_pass)
        rows[str(seed)] = {"local_h2": l, "recovery_h2": r, "h2_ratio": r / max(l, EPS), "episode_direction_count": int(np.sum(recovery[seed] < local[seed])), "clean_local": lclean, "clean_recovery": rclean, "primary_pass": primary, "clean_pass": clean_pass, "pass": primary and clean_pass, "ambiguous": near}
    gp, gc = gr <= .95 * gl + EPS, gcr <= 1.10 * gcl + EPS
    ambiguous |= abs(gr - .95 * gl - EPS) <= REL_TOL * max(abs(.95 * gl), EPS) or abs(gcr - 1.10 * gcl - EPS) <= REL_TOL * max(abs(1.10 * gcl), EPS)
    return {"local_arm": local_arm, "recovery_arm": recovery_arm, "global_local_h2": gl, "global_recovery_h2": gr, "global_h2_ratio": gr / max(gl, EPS), "global_primary_pass": gp, "global_clean_pass": gc, "seed_pass_count": passed, "required_seed_pass_count": 2, "seeds": rows, "pass": bool(gp and gc and passed >= 2 and not ambiguous), "ambiguous": ambiguous}
def _metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    terminal, clean, episodes = raw["terminal_mse"], raw["clean_mse"], raw["episodes"]
    q, l = _pair_gate(terminal, clean, episodes, "q_local", "q_recovery"), _pair_gate(terminal, clean, episodes, "l_local", "l_recovery")
    interaction = [{"seed": seed, "value": float(np.mean((terminal[9 + i] - terminal[6 + i]) - (terminal[3 + i] - terminal[i])))} for i, seed in enumerate(SEEDS)]
    donor = raw.get("donor_mse")
    secondary = {"clean_one_step_global": {arm: float(np.mean(clean[j:j + 3])) for arm, j in (("q_local", 0), ("q_recovery", 3), ("l_local", 6), ("l_recovery", 9))}, "donor_recovery_global": donor.mean(axis=0).tolist() if isinstance(donor, np.ndarray) else []}
    return {"pair_gates": {"q": q, "l": l}, "interaction": {"per_seed": interaction, "global_mean": float(np.mean([x["value"] for x in interaction]))}, "secondary": secondary}
def _sacct_budget(summary: Mapping[str, Any], metadata: Mapping[str, Any], manifest: Mapping[str, Any], artifact_dir: Path, cap: float) -> tuple[list[str], dict[str, Any]]:
    ids: set[str] = set()
    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                if k in ("gpu_job_id", "artifact_job_id", "collector_job_id", "job_id") and isinstance(v, (str, int)) and str(v):
                    ids.add(str(v))
                collect(v, str(k))
        elif isinstance(value, list):
            for v in value:
                collect(v, key)
    collect(summary); collect(metadata); collect(manifest)
    if not ids:
        for name in ("budget.json", "resource_usage.json", "slurm_usage.json"):
            try:
                companion = _json(artifact_dir / name)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                continue
            if _finite(companion.get("gpu_hours")):
                used = float(companion["gpu_hours"])
                return ([] if used <= cap + EPS else ["GPU budget exceeds six hours"], {"gpu_hours": used, "source": name})
    if not ids:
        return ["actual GPU job id/budget evidence missing"], {}
    seconds = 0.0
    for job_id in sorted(ids):
        try:
            result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job_id, "--format=ElapsedRaw,AllocTRES"], check=True, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"sacct budget query failed: {type(exc).__name__}"], {"job_ids": sorted(ids)}
        for line in result.stdout.splitlines():
            parts = line.strip().split("|", 1)
            match = re.search(r"(?:gres/)?gpu(?::[^=,]+)?=(\d+)", parts[1]) if len(parts) == 2 else None
            try:
                if match:
                    seconds += float(parts[0]) * int(match.group(1))
            except ValueError:
                pass
    used = seconds / 3600.0
    if seconds <= 0:
        return ["sacct returned no GPU allocation for artifact jobs"], {"job_ids": sorted(ids)}
    return ([] if used <= cap + EPS else ["GPU budget exceeds six hours"], {"job_ids": sorted(ids), "gpu_hours": used, "source": "sacct"})
def _self_test() -> None:
    errors = _manifest_audit(_json(Path(__file__).with_name("manifest_template.json")))
    if errors:
        raise SystemExit(json.dumps({"status": "failed", "errors": errors}, ensure_ascii=False))
    print(json.dumps({"status": "complete", "checks": ["manifest", "splits", "W4", "R1 gates", "actual runner contract"]}))
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); return
    if not args.artifact_dir or not args.manifest or not args.output:
        parser.error("--artifact-dir, --manifest and --output are required")
    allocation = _guard()
    errors: list[str] = []
    summary_info: dict[str, Any] = {}
    raw_info: dict[str, Any] = {}
    manifest: Mapping[str, Any] = {}
    artifact_dir, manifest_path = args.artifact_dir.resolve(), args.manifest.resolve()
    try:
        manifest = _json(manifest_path)
        errors.extend(_manifest_audit(manifest))
        contract = manifest.get("artifact_contract", {})
        summary_name = contract.get("summary", "raw_final_summary.json") if isinstance(contract, Mapping) else "raw_final_summary.json"
        raw_name = contract.get("raw_final", "raw_final.npz") if isinstance(contract, Mapping) else "raw_final.npz"
        summary = _json(artifact_dir / summary_name)
        se, summary_info = _summary_audit(summary); errors.extend(se)
        errors.extend(_runtime_audit(artifact_dir, manifest_path, manifest, summary))
        re_, raw_info = _raw_audit(artifact_dir / raw_name); errors.extend(re_)
        if raw_info:
            me = _metadata_audit(raw_info["metadata"], manifest); errors.extend(me)
            ce, cinfo = _canonical_audit(manifest, manifest_path.parent, artifact_dir, raw_info["metadata"]); errors.extend(ce)
            summary_info["canonical"] = cinfo
        fe, fit_info = _fit_audit(artifact_dir); errors.extend(fe)
        summary_info["fits"] = {"completed": len(fit_info["completed"]), "expected": 12}
        be, budget_info = _sacct_budget(summary, raw_info.get("metadata", {}), manifest, artifact_dir, 6.0); errors.extend(be)
        summary_info["budget"] = budget_info
    except Exception as exc:
        errors.append(f"verifier exception: {type(exc).__name__}: {exc}")
    engineering_pass = not errors
    metrics = _metrics(raw_info) if raw_info and not errors else {}
    q_pass = bool(metrics.get("pair_gates", {}).get("q", {}).get("pass")); l_pass = bool(metrics.get("pair_gates", {}).get("l", {}).get("pass"))
    ambiguous = any(bool(metrics.get("pair_gates", {}).get(n, {}).get("ambiguous")) for n in ("q", "l"))
    status = "inconclusive" if not engineering_pass else "conditional_signal" if (q_pass or l_pass) else "inconclusive" if ambiguous else "mechanism_no_go"
    result = {"schema": "prr-verification-v1", "status": status, "allocation": {k: allocation.get(k) for k in ("scheduler", "job_id", "hostname", "nodelist", "verified")}, "manifest": {"pass": not bool(_manifest_audit(manifest)) if manifest else False}, "engineering_pass": engineering_pass, "mechanism_gate_pass": bool(q_pass or l_pass), "family_gate_pass": {"q": q_pass, "l": l_pass}, "summary": summary_info, "metrics": metrics, "errors": _errors(errors)}
    _write_json(args.output.resolve(), result)
    print(json.dumps({"status": status, "engineering_pass": engineering_pass, "mechanism_gate_pass": bool(q_pass or l_pass)}, separators=(",", ":")))
if __name__ == "__main__":
    main()

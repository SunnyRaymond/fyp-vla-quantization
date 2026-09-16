"""Independent CPU audit for the bounded CEM-Update PTQ search.

This verifier deliberately does not import the model, smoke_runner, or
joint_search.  On a real run, allocation_guard is called before any workload
or raw-score file is opened.  It audits the independent CCDS V100 rerun.  The output is a compact audit summary; raw
arrays remain in the compute-side artifact directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


METHODS = ("CEM-Update", "MeanOnly", "ScoreError", "Rank")
OBJECTIVE_FOR = {
    "CEM-Update": "update",
    "MeanOnly": "mean_only",
    "ScoreError": "score_error",
    "Rank": "rank",
}
CHAIN_SEED = 810_000
CAL_TOL = 1e-10
STAT_TOL = 1e-6
GATE_REL = 0.05
TOPK = 30
POOL_SHAPE = (300, 5, 10)


def _load_guard() -> Any:
    try:
        from allocation_guard import require_allocation

        return require_allocation
    except ModuleNotFoundError:
        path = Path(__file__).with_name("allocation_guard.py")
        import importlib.util

        spec = importlib.util.spec_from_file_location("cem_verify_allocation_guard", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load allocation guard: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.require_allocation


def _torch() -> Any:
    import torch

    return torch


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _as_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def _map_key(bits: Mapping[str, Any]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((str(key), int(value)) for key, value in bits.items()))


def _map_id(bits: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(_map_key(bits)), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _bits(value: Mapping[str, Any]) -> Dict[str, int]:
    raw = value.get("bits", value.get("allocation"))
    if not isinstance(raw, Mapping):
        raise ValueError("map has no bits/allocation mapping")
    result = {str(key): int(item.get("bits") if isinstance(item, Mapping) else item) for key, item in raw.items()}
    if any(bit not in (4, 8) for bit in result.values()):
        raise ValueError("map bits must be 4 or 8")
    expected = {
        *(f"encoder.base_model.blocks.{index}" for index in range(12)),
        *(f"predictor.transformer.layers.{index}" for index in range(6)),
    }
    if set(result) != expected:
        raise ValueError("map does not specify exactly 12 encoder and 6 predictor groups")
    if sum(result[key] == 8 for key in result if key.startswith("encoder.")) != 3 or sum(result[key] == 8 for key in result if key.startswith("predictor.")) != 2:
        raise ValueError("map W8 quota is not encoder=3/predictor=2")
    return result


def _load_pickle(path: Path) -> Any:
    if path.is_dir():
        path = path / "workload.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def _planner_errors(planner: Any) -> List[str]:
    if not isinstance(planner, Mapping):
        return ["planner metadata missing"]
    errors: List[str] = []
    exact = {
        "horizon": 5,
        "execute_model_actions": 5,
        "frameskip": 5,
        "num_samples": 300,
        "elite_count": 30,
        "cem_iterations": 5,
        "stable_argsort": True,
        "inner_environment_evaluator": None,
        "quantization_execution": "emulation_only",
    }
    for key, expected in exact.items():
        if planner.get(key) != expected:
            errors.append(f"planner.{key}={planner.get(key)!r}, expected {expected!r}")
    # The collector's compact workload intentionally omits var_scale.  A
    # present value is still checked here; the missing-field case is proved
    # from the archived workspace helper by _var_scale_audit below.
    if "var_scale" in planner and planner.get("var_scale") != 1:
        errors.append(f"planner.var_scale={planner.get('var_scale')!r}, expected 1")
    objective = planner.get("objective")
    if not isinstance(objective, Mapping) or {
        key: objective.get(key) for key in ("mode", "alpha", "base")
    } != {"mode": "last", "alpha": 1, "base": 2}:
        errors.append("planner.objective is not last/alpha=1/base=2")
    reference = planner.get("reference")
    if not isinstance(reference, Mapping) or reference.get("dtype") != "float32" or reference.get("decoder") is not None or reference.get("activations") != "float32":
        errors.append("planner.reference is not FP32 activations with decoder=None")
    return errors


def _var_scale_audit(base: Path, search: Path, payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Prove var_scale=1 without requiring the compact workload to duplicate it."""
    declared = [payload.get("planner", {}).get("var_scale") for payload in payloads if isinstance(payload.get("planner"), Mapping) and "var_scale" in payload.get("planner", {})]
    if any(value != 1 for value in declared):
        return {"pass": False, "source": "workload planner", "declared": declared, "errors": ["explicit var_scale conflicts with frozen value 1"]}
    if declared and all(value == 1 for value in declared):
        return {"pass": True, "source": "workload planner", "declared": declared}
    candidates = [
        search / "smoke_runner.py",
        base / "smoke_runner.py",
        Path(__file__).with_name("smoke_runner.py"),
    ]
    checked: List[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        checked.append(path.name)
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^def _make_workspace\b(?P<body>.*?)(?=^def |\Z)", text, flags=re.MULTILINE | re.DOTALL)
        if match and re.search(r"[\"']var_scale[\"']\s*:\s*1\b", match.group("body")):
            return {"pass": True, "source": str(path) + "::_make_workspace", "declared": declared}
    return {"pass": False, "source": None, "declared": declared, "checked": checked, "errors": ["var_scale=1 is absent from compact planner and archived _make_workspace"]}


def _gpu_family(value: Any) -> str:
    text = str(value or "").strip()
    return "V100" if "v100" in text.casefold() else text


def _validate_ccds_workload(payload: Mapping[str, Any], label: str) -> None:
    identity = payload.get("runtime_identity")
    if not isinstance(identity, Mapping) or _gpu_family(identity.get("gpu")) != "V100":
        raise ValueError(f"{label} workload does not report a V100 runtime")
    provenance = payload.get("hardware_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("independent_rerun") is not True:
        raise ValueError(f"{label} workload lacks independent CCDS rerun provenance")
    if provenance.get("scheduler") != "slurm" or provenance.get("cluster") != "CCDS-TC1":
        raise ValueError(f"{label} workload has invalid CCDS scheduler provenance")
    if provenance.get("gpu_model") != identity.get("gpu"):
        raise ValueError(f"{label} workload GPU provenance disagrees with runtime identity")
    if provenance.get("candidate_score_pairing") != "same_fresh_fp32_collection" or provenance.get("historical_arrays_reused") is not False:
        raise ValueError(f"{label} workload may mix candidate/score arrays across hardware runs")


def _identity_signature(identity: Mapping[str, Any]) -> Tuple[Any, ...]:
    keys = (
        "directory", "checkpoint_file", "recorded_epoch", "checkpoint_size_bytes",
        "source_commit", "dinov2_source_commit", "dtype", "decoder",
        "execution", "native_memory_claim", "torch",
    )
    return tuple(identity.get(key) for key in keys) + (_gpu_family(identity.get("gpu")),)


def _summary_hardware_errors(summary: Mapping[str, Any], payloads: Sequence[Mapping[str, Any]]) -> List[str]:
    current = summary.get("checkpoint_identity")
    if not isinstance(current, Mapping):
        return ["search summary lacks checkpoint_identity for hardware audit"]
    errors: List[str] = []
    if _gpu_family(current.get("gpu")) != "V100":
        errors.append("search summary does not report a V100 runtime")
    for index, payload in enumerate(payloads):
        recorded = payload.get("runtime_identity")
        if not isinstance(recorded, Mapping) or _identity_signature(recorded) != _identity_signature(current):
            errors.append(f"search summary/current checkpoint disagrees with workload {index}")
    provenance = str(summary.get("hardware_provenance", ""))
    if "CCDS" not in provenance or "V100" not in provenance:
        errors.append("search summary lacks independent CCDS V100 provenance")
    return errors


def _load_workload(path: Path, label: str) -> Dict[str, Any]:
    payload = _load_pickle(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} workload is not a mapping")
    expected_split = "newcal" if label == "cal" else "newdev"
    if payload.get("schema") not in (None, "rankcal-wall-screen-workload-v1", "cem-update-ptq-ccds-workload-v1"):
        raise ValueError(f"{label} workload declares an unsupported schema")
    if not isinstance(payload.get("runtime_identity"), Mapping):
        raise ValueError(f"{label} workload lacks runtime identity")
    _validate_ccds_workload(payload, label)
    if str(payload.get("split")) != expected_split:
        raise ValueError(f"{label} workload split is not {expected_split}")
    planner_errors = _planner_errors(payload.get("planner"))
    roots = payload.get("cases")
    if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)):
        raise ValueError(f"{label} workload cases are missing")
    cases: List[Dict[str, Any]] = []
    for node in roots:
        if not isinstance(node, Mapping):
            raise ValueError(f"{label} contains a non-mapping case")
        candidate = np.asarray(node.get("candidates"), dtype=np.float32)
        if tuple(candidate.shape) != POOL_SHAPE:
            raise ValueError(f"{label} pool {node.get('pool_id')!r} has shape {candidate.shape}")
        reference = node.get("reference_scores")
        if reference is None:
            raise ValueError(f"{label} pool {node.get('pool_id')!r} has no reference_scores")
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if reference.shape != (POOL_SHAPE[0],) or not np.isfinite(reference).all():
            raise ValueError(f"{label} pool {node.get('pool_id')!r} has malformed reference_scores")
        episode = str(node.get("episode_id"))
        cases.append({
            "pool_id": str(node.get("pool_id")),
            "episode_id": episode,
            "dataset_index": int(node.get("dataset_index")),
            "env_seed": int(node.get("env_seed")),
            "cem_seed": int(node.get("cem_seed")),
            "local_index": int(re.search(r":(\d{3})$", episode).group(1)) if re.search(r":(\d{3})$", episode) else -1,
            "mpc_point": int(node.get("mpc_point", 0)),
            "cem_iteration": int(node.get("cem_iteration", 0)),
            "candidates": candidate,
            "reference_scores": reference,
        })
    if len(cases) != 8:
        raise ValueError(f"{label} workload must have 8 pools")
    if len({case["pool_id"] for case in cases}) != 8:
        raise ValueError(f"{label} workload has duplicate pool IDs")
    expected_prefix = expected_split
    expected_offset = 42 if label == "cal" else 46
    expected_namespace = 600000 if label == "cal" else 700000
    by_episode: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_episode[case["episode_id"]].append(case)
        local = case["local_index"]
        if case["mpc_point"] != 0 or case["dataset_index"] != expected_offset + local:
            raise ValueError(f"{label} pool {case['pool_id']!r} has invalid point/dataset metadata")
        if case["env_seed"] != expected_namespace + local or case["cem_seed"] != expected_namespace + 10000 + local:
            raise ValueError(f"{label} pool {case['pool_id']!r} has invalid seed metadata")
    if set(by_episode) != {f"{expected_prefix}:{index:03d}" for index in range(4)}:
        raise ValueError(f"{label} episode IDs are not the frozen 000..003 set")
    for episode, rows in by_episode.items():
        if len(rows) != 2 or sorted(row["cem_iteration"] for row in rows) != [1, 5]:
            raise ValueError(f"{label} episode {episode!r} is not exactly CEM iterations 1 and 5")
    return {"payload": payload, "cases": cases, "planner_errors": planner_errors}


def _target_audit(base: Path, old_fingerprint_path: Path | None = None) -> Dict[str, Any]:
    target_dir = base / "targets"
    fingerprints: Dict[str, set[str]] = {}
    details: Dict[str, Any] = {}
    errors: List[str] = []
    for split, expected_offset in (("newcal", 42), ("newdev", 46)):
        path = target_dir / f"episode_manifest_{split}.json"
        if not path.is_file():
            errors.append(f"missing {path.name}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = payload.get("episodes", [])
        actual = {str(row.get("target_fingerprint")) for row in episodes if isinstance(row, Mapping)}
        fingerprints[split] = actual
        if payload.get("schema") != "rankcal-wall-targets-v1" or payload.get("split") != split or len(episodes) != 4:
            errors.append(f"{split} target manifest schema/count mismatch")
        expected_indices = set(range(expected_offset, expected_offset + 4))
        actual_indices = {int(row.get("dataset_index", -1)) for row in episodes if isinstance(row, Mapping)}
        if actual_indices != expected_indices or len(actual) != 4 or "None" in actual:
            errors.append(f"{split} target indices/fingerprints mismatch")
        if len(actual) != len(episodes):
            errors.append(f"{split} target fingerprints are duplicated")
    old: set[str] = set()
    source = (old_fingerprint_path or Path(__file__).with_name("old_target_fingerprints.json")).resolve()
    if not source.is_file():
        errors.append(f"old fingerprint input is missing: {source}")
    else:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            records = payload.get("records")
            old = {str(row.get("target_fingerprint")) for row in records if isinstance(row, Mapping)}
            indices = {int(row.get("dataset_index", -1)) for row in records if isinstance(row, Mapping)}
            if payload.get("schema") != "cem-update-ptq-old-target-fingerprints-v1" or len(records) != 40 or len(old) != 40 or indices != set(range(2, 42)) or "None" in old:
                errors.append("old fingerprint input schema/count/index mismatch")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"cannot read old fingerprint input: {type(exc).__name__}")
    if len(old) != 40:
        errors.append(f"old fingerprint count is {len(old)}, expected 40")
    new = set().union(*fingerprints.values()) if fingerprints else set()
    overlap = new & old
    if overlap:
        errors.append(f"new/old target fingerprint overlap count={len(overlap)}")
    if fingerprints.get("newcal", set()) & fingerprints.get("newdev", set()):
        errors.append("new CAL/DEV target fingerprints overlap")
    details.update({"newcal": len(fingerprints.get("newcal", set())), "newdev": len(fingerprints.get("newdev", set())), "old": len(old), "old_overlap": len(overlap), "old_source": str(source)})
    return {"pass": not errors, "errors": errors, **details}


def _pool_ids(cases: Sequence[Mapping[str, Any]]) -> List[str]:
    return [str(case["pool_id"]) for case in cases]


def _load_scores(path: Path, cases: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    expected = _pool_ids(cases)
    with np.load(path, allow_pickle=False) as archive:
        if "pool_ids" not in archive.files:
            raise ValueError(f"score archive has no pool_ids: {path.name}")
        ids = [_as_id(value) for value in archive["pool_ids"].reshape(-1)]
        if ids != expected:
            raise ValueError(f"score archive pool order mismatch: {path.name}")
        result: Dict[str, np.ndarray] = {}
        for index, pool_id in enumerate(ids):
            key = f"pool_{index:04d}"
            if key not in archive.files:
                raise ValueError(f"score archive missing {key}: {path.name}")
            values = np.asarray(archive[key], dtype=np.float64).reshape(-1)
            if values.shape != (POOL_SHAPE[0],) or not np.isfinite(values).all():
                raise ValueError(f"malformed scores for {pool_id}: {path.name}")
            result[pool_id] = values.copy()
    return result


def _reference_score_errors(scores: Mapping[str, np.ndarray], cases: Sequence[Mapping[str, Any]], label: str) -> List[str]:
    """Link search reference archives back to the collected workload scores."""
    errors: List[str] = []
    for case in cases:
        pool_id = str(case["pool_id"])
        expected = np.asarray(case["reference_scores"], dtype=np.float64).reshape(-1)
        actual = scores.get(pool_id)
        if actual is None:
            errors.append(f"{label} reference archive is missing {pool_id}")
            continue
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-6, equal_nan=False):
            maximum = float(np.max(np.abs(actual - expected)))
            errors.append(f"{label} reference archive disagrees with workload at {pool_id}: max_abs={maximum:g}")
    return errors


def _array_consistency_errors(current: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray], label: str) -> List[str]:
    """Report missing or changed arrays that must be shared across aliases."""
    errors: List[str] = []
    if set(current) != set(baseline):
        errors.append(f"{label} pool IDs differ across method archives")
    for pool_id in sorted(set(current) & set(baseline)):
        if not np.array_equal(np.asarray(current[pool_id]), np.asarray(baseline[pool_id])):
            errors.append(f"{label} differs across method archives at {pool_id}")
    return errors


def _elite(values: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if scores.shape != (POOL_SHAPE[0],) or not np.isfinite(scores).all():
        raise ValueError("score vector is not finite 300-vector")
    return np.argsort(scores, kind="stable")[:TOPK]


def _stats(candidates: np.ndarray, elite: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    selected = torch.as_tensor(np.asarray(candidates, dtype=np.float32)[elite], dtype=torch.float32, device="cpu")
    mean = selected.mean(dim=0).numpy().astype(np.float32, copy=True)
    try:
        sigma = selected.std(dim=0, correction=1).numpy().astype(np.float32, copy=True)
    except TypeError:
        sigma = selected.std(dim=0, unbiased=True).numpy().astype(np.float32, copy=True)
    return mean, sigma


def _pairwise_rank(reference: np.ndarray, quantized: np.ndarray) -> float:
    i, j = np.triu_indices(reference.size, k=1)
    return float(np.mean(np.sign(reference[i] - reference[j]) != np.sign(quantized[i] - quantized[j])))


def _metric(reference: np.ndarray, quantized: np.ndarray, candidates: np.ndarray, iteration: int) -> Dict[str, Any]:
    ref_elite = _elite(reference)
    q_elite = _elite(quantized)
    ref_mu, ref_sigma = _stats(candidates, ref_elite)
    q_mu, q_sigma = _stats(candidates, q_elite)
    mu_delta = np.asarray(q_mu, dtype=np.float64) - np.asarray(ref_mu, dtype=np.float64)
    sigma_delta = np.asarray(q_sigma, dtype=np.float64) - np.asarray(ref_sigma, dtype=np.float64)
    l_mu = float(np.mean(mu_delta * mu_delta, dtype=np.float64))
    l_sigma = float(np.mean(sigma_delta * sigma_delta, dtype=np.float64))
    nmse = float(np.mean((quantized - reference) ** 2, dtype=np.float64) / (np.mean(reference ** 2, dtype=np.float64) + 1e-12))
    return {
        "mu": l_mu,
        "sigma": l_sigma,
        "update": l_mu + (0.0 if int(iteration) == 5 else l_sigma),
        "mean_only": l_mu,
        "score_error": nmse,
        "rank": _pairwise_rank(reference, quantized),
        "reference_elite_indices": ref_elite,
        "quantized_elite_indices": q_elite,
        "reference_mu": ref_mu,
        "reference_sigma": ref_sigma,
        "quantized_mu": q_mu,
        "quantized_sigma": q_sigma,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], fields: Sequence[str] = ("mu", "sigma", "update", "mean_only", "score_error", "rank")) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    by_episode: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
    episode_metrics: Dict[str, Dict[str, float]] = {}
    for episode, values in sorted(by_episode.items()):
        episode_metrics[episode] = {field: float(np.mean([float(value[field]) for value in values], dtype=np.float64)) for field in fields}
    overall = {field: float(np.mean([value[field] for value in episode_metrics.values()], dtype=np.float64)) for field in fields}
    return overall, episode_metrics


def _map_metrics(cases: Sequence[Mapping[str, Any]], references: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        pool_id = str(case["pool_id"])
        metric = _metric(references[pool_id], scores[pool_id], case["candidates"], int(case["cem_iteration"]))
        metric.update({"pool_id": pool_id, "episode_id": str(case["episode_id"])})
        rows.append(metric)
    overall, episodes = _aggregate(rows)
    return {"overall": overall, "episodes": episodes, "rows": rows}


def _neighbors(bits: Mapping[str, int]) -> List[Dict[str, int]]:
    families = {
        "encoder": sorted((key for key in bits if key.startswith("encoder.base_model.blocks.")), key=lambda key: int(key.split(".")[3])),
        "predictor": sorted((key for key in bits if key.startswith("predictor.transformer.layers.")), key=lambda key: int(key.split(".")[3])),
    }
    result: List[Dict[str, int]] = []
    for family in ("encoder", "predictor"):
        high = [key for key in families[family] if int(bits[key]) == 8]
        low = [key for key in families[family] if int(bits[key]) == 4]
        for source in high:
            for target in low:
                candidate = dict(bits)
                candidate[source], candidate[target] = 4, 8
                result.append(candidate)
    if len(result) != 35 or len({_map_id(item) for item in result}) != 35:
        raise ValueError(f"expected 35 unique same-family neighbors, got {len(result)}")
    return result


def _load_trace(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    configs: List[Dict[str, Any]] = []
    rounds: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "config_eval":
                configs.append(row)
            elif row.get("event") == "round_end":
                rounds.append(row)
            else:
                raise ValueError(f"unknown trace event at line {line_number}")
    return configs, rounds


def _close(a: float, b: float, tolerance: float = STAT_TOL) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= tolerance)


def _trace_audit(search: Path, cal: Sequence[Mapping[str, Any]], cal_ref: Mapping[str, np.ndarray], expected_start: Mapping[str, int], rounds_requested: int, declared_cost: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    configs, round_rows = _load_trace(search / "search_trace.jsonl")
    by_method: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    by_round: MutableMapping[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in configs:
        method = str(row.get("method"))
        if method not in METHODS:
            raise ValueError(f"trace contains unknown method {method!r}")
        bits = _bits(row)
        if str(row.get("map_id")) != _map_id(bits):
            raise ValueError(f"trace map ID mismatch for {method}")
        by_method[method].append(row)
        by_round[(method, int(row.get("round")))].append(row)
    round_end: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in round_rows:
        key = (str(row.get("method")), int(row.get("round")))
        if key in round_end:
            raise ValueError(f"duplicate round_end for {key}")
        round_end[key] = row
    results: Dict[str, Any] = {}
    errors: List[str] = []
    ambiguous = False
    shared_start: Dict[str, int] | None = None
    metric_cache: Dict[str, Dict[str, Any]] = {}

    def map_metric(map_id: str) -> Dict[str, Any]:
        if map_id not in metric_cache:
            score_path = search / "cal_scores" / f"map_{map_id}.npz"
            if not score_path.is_file():
                raise ValueError(f"missing CAL score archive {map_id}")
            metric_cache[map_id] = _map_metrics(cal, cal_ref, _load_scores(score_path, cal))
        return metric_cache[map_id]

    for method in METHODS:
        rows = by_method.get(method, [])
        method_errors: List[str] = []
        initial = [row for row in rows if int(row.get("round")) == -1]
        if len(initial) != 1:
            method_errors.append("missing or duplicate round -1 evaluation")
            results[method] = {"pass": False, "errors": method_errors}
            errors.extend(f"{method}: {item}" for item in method_errors)
            continue
        start = _bits(initial[0])
        if _map_key(start) != _map_key(expected_start):
            method_errors.append("method start map disagrees with summary shared_start")
        if shared_start is None:
            shared_start = start
        elif _map_key(start) != _map_key(shared_start):
            method_errors.append("method does not use the shared start map")
        current = dict(start)
        expected_index = 0
        method_ambiguous = False
        accepted_rounds = 0
        seen_maps: set[str] = set()
        for row in rows:
            if int(row.get("config_index")) != expected_index:
                method_errors.append("config_index is not contiguous")
            expected_index += 1
            map_id = str(row.get("map_id"))
            seen_maps.add(map_id)
            bits_row = _bits(row)
            try:
                metric = map_metric(map_id)
            except ValueError as exc:
                method_errors.append(str(exc))
                continue
            objective_name = OBJECTIVE_FOR[method]
            if row.get("objective_name") != objective_name or not _close(float(row.get("objective")), metric["overall"][objective_name]):
                method_errors.append(f"CAL metric mismatch at {map_id}")
            cost = row.get("cost")
            if not isinstance(cost, Mapping):
                method_errors.append(f"missing cost mapping at {map_id}")
            else:
                for field in ("logical_weight_bytes", "weight_bits"):
                    try:
                        if int(cost.get(field, -1)) < 0:
                            raise ValueError
                    except (TypeError, ValueError):
                        method_errors.append(f"invalid cost.{field} at {map_id}")
                # A runtime registry is deliberately not loaded here.  Check
                # only the shared start against the summary; swapped maps may
                # legitimately have different logical byte costs.
                if declared_cost is not None and map_id == _map_id(expected_start):
                    for field in ("logical_weight_bytes", "weight_bits"):
                        if field in declared_cost and int(cost.get(field, -1)) != int(declared_cost[field]):
                            method_errors.append(f"start cost mismatch at {field}")
        if expected_index > 71:
            method_errors.append(f"config budget exceeded: {expected_index}")
        for round_index in range(rounds_requested):
            candidates = by_round.get((method, round_index), [])
            if not candidates:
                break
            neighbors = {_map_id(item): item for item in _neighbors(current)}
            if len(candidates) != 35 or {str(row.get("map_id")) for row in candidates} != set(neighbors):
                method_errors.append(f"round {round_index} is not the complete 35-neighbor set")
                break
            values: Dict[str, float] = {}
            for row in candidates:
                map_id = str(row["map_id"])
                metric = map_metric(map_id)
                values[map_id] = metric["overall"][OBJECTIVE_FOR[method]]
            best_value = min(values.values())
            near = [map_id for map_id, value in values.items() if abs(value - best_value) <= STAT_TOL]
            round_ambiguous = len(near) > 1
            best_id = min(near, key=lambda map_id: _map_key(neighbors[map_id]))
            current_path = search / "cal_scores" / f"map_{_map_id(current)}.npz"
            current_metric = _map_metrics(cal, cal_ref, _load_scores(current_path, cal))["overall"][OBJECTIVE_FOR[method]]
            improvement = current_metric - best_value
            if abs(improvement - CAL_TOL) <= STAT_TOL:
                round_ambiguous = True
            expected_accept = improvement > CAL_TOL
            end = round_end.get((method, round_index))
            if end is None:
                method_errors.append(f"missing round_end {round_index}")
                break
            trace_best_id = str(end.get("best_neighbor_map_id"))
            if round_ambiguous and trace_best_id in near:
                best_id_for_state = trace_best_id
            else:
                best_id_for_state = best_id
            if trace_best_id != best_id_for_state or not _close(float(end.get("best_neighbor_objective")), best_value) or not _close(float(end.get("improvement")), improvement):
                method_errors.append(f"round {round_index} decision values mismatch")
            trace_accept = bool(end.get("accepted"))
            if not round_ambiguous and trace_accept != expected_accept:
                method_errors.append(f"round {round_index} accepted flag mismatch")
            method_ambiguous = method_ambiguous or round_ambiguous
            accept_for_state = trace_accept if round_ambiguous else expected_accept
            if accept_for_state:
                accepted_rounds += 1
                current = dict(neighbors[best_id_for_state])
            else:
                break
        attempted_rounds = {int(row.get("round")) for row in rows if int(row.get("round")) >= 0}
        if attempted_rounds and max(attempted_rounds) >= 0:
            # A no-improvement round is a terminal round.  Later rows would
            # indicate a trace that cannot be reproduced by best improvement.
            terminal = min(
                (round_index for round_index in attempted_rounds if (method, round_index) in round_end and not bool(round_end[(method, round_index)].get("accepted"))),
                default=None,
            )
            if terminal is not None and any(round_index > terminal for round_index in attempted_rounds):
                method_errors.append("trace contains evaluations after its first rejected round")
        selected_id = _map_id(current)
        if method_ambiguous:
            ambiguous = True
        selected = results.get(method, {})
        selected.update({
            "pass": not method_errors,
            "errors": method_errors,
            "evaluations": expected_index,
            "rounds_attempted": len(attempted_rounds),
            "accepted_rounds": accepted_rounds,
            "selected_map_id_recomputed": selected_id,
            "ambiguous": method_ambiguous,
            "unique_cal_maps": len(seen_maps),
        })
        results[method] = selected
        errors.extend(f"{method}: {item}" for item in method_errors)
    selected_path = search / "selected_maps.json"
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.is_file() else {}
    for method in METHODS:
        actual = selected_payload.get(method)
        if not isinstance(actual, Mapping):
            errors.append(f"selected_maps.json missing {method}")
            continue
        actual_bits = _bits(actual)
        actual_id = _map_id(actual_bits)
        results[method]["trace_selected_map_id"] = actual_id
        if actual_id != results[method].get("selected_map_id_recomputed"):
            results[method]["pass"] = False
            results[method]["errors"].append("selected map disagrees with trace reconstruction")
            errors.append(f"{method}: selected map disagrees with trace reconstruction")
    return {
        "pass": not errors,
        "errors": errors,
        "ambiguous": ambiguous,
        "shared_start_map_id": None if shared_start is None else _map_id(shared_start),
        "methods": results,
    }


def _shared_noise(seed: int, shape: Tuple[int, ...]) -> np.ndarray:
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).numpy()


def _array_close(a: np.ndarray, b: np.ndarray, tolerance: float = STAT_TOL) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=tolerance, equal_nan=False))


def _chain_audit(search: Path, dev: Sequence[Mapping[str, Any]], dev_ref: Mapping[str, np.ndarray], dev_scores: Mapping[str, Mapping[str, np.ndarray]], selected: Mapping[str, Mapping[str, int]], dev_metrics_file: Mapping[str, Any]) -> Dict[str, Any]:
    iter1 = [case for case in dev if int(case["cem_iteration"]) == 1]
    jsonl_path = search / "dev_chain.jsonl"
    chain_rows: List[Dict[str, Any]] = []
    if jsonl_path.is_file():
        with jsonl_path.open("r", encoding="utf-8") as stream:
            chain_rows = [json.loads(line) for line in stream if line.strip()]
    by_alias: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in chain_rows:
        by_alias[str(row.get("method"))].append(row)
    result: Dict[str, Any] = {}
    errors: List[str] = []
    reference_raw: Dict[str, np.ndarray] | None = None
    reference_scores: Dict[str, np.ndarray] | None = None
    for alias, bits in selected.items():
        path = search / "dev_chain" / f"{alias}.npz"
        alias_errors: List[str] = []
        if not path.is_file():
            errors.append(f"missing chain archive {alias}")
            result[alias] = {"pass": False, "errors": ["missing chain archive"]}
            continue
        rows = by_alias.get(alias, [])
        if len(rows) != 4:
            alias_errors.append(f"chain jsonl has {len(rows)} rows, expected 4")
        with np.load(path, allow_pickle=False) as archive:
            ids = [_as_id(value) for value in archive["pool_ids"].reshape(-1)] if "pool_ids" in archive.files else []
            expected_ids = _pool_ids(iter1)
            if ids != expected_ids:
                alias_errors.append("chain pool order mismatch")
            arrays = {key: np.asarray(archive[key]) for key in archive.files if key != "pool_ids"}
        final_rows: Dict[str, float] = {}
        max_error = 0.0
        ref_raw_for_compare: Dict[str, np.ndarray] = {}
        for index, case in enumerate(iter1):
            pool_id = str(case["pool_id"])
            suffix = f"{index:04d}"
            required = [f"u1_{suffix}", f"u2_ref_{suffix}", f"u2_q_{suffix}", f"noise_{suffix}", f"mu1_ref_{suffix}", f"sigma1_ref_{suffix}", f"mu1_q_{suffix}", f"sigma1_q_{suffix}", f"score2_ref_{suffix}", f"score2_q_{suffix}"]
            missing = [key for key in required if key not in arrays]
            if missing:
                alias_errors.append(f"{pool_id} missing chain arrays {missing}")
                continue
            u1 = np.asarray(arrays[f"u1_{suffix}"], dtype=np.float32)
            u2_ref = np.asarray(arrays[f"u2_ref_{suffix}"], dtype=np.float32)
            u2_q = np.asarray(arrays[f"u2_q_{suffix}"], dtype=np.float32)
            noise = np.asarray(arrays[f"noise_{suffix}"], dtype=np.float32)
            if u1.shape != POOL_SHAPE or u2_ref.shape != POOL_SHAPE or u2_q.shape != POOL_SHAPE or noise.shape != POOL_SHAPE:
                alias_errors.append(f"{pool_id} chain U/noise shape mismatch")
                continue
            expected_seed = CHAIN_SEED + int(case["local_index"])
            row = next((item for item in rows if str(item.get("pool_id")) == pool_id), None)
            if row is None or int(row.get("noise_seed", -1)) != expected_seed:
                alias_errors.append(f"{pool_id} noise seed mismatch")
            if row is not None and str(row.get("map_id")) != _map_id(bits):
                alias_errors.append(f"{pool_id} chain row map_id mismatch")
            if row is not None and (row.get("candidate_zero_is_mu") is not True or int(row.get("std_correction", -1)) != 1):
                alias_errors.append(f"{pool_id} chain semantic flags mismatch")
            expected_noise = _shared_noise(expected_seed, POOL_SHAPE)
            if not np.array_equal(noise, expected_noise):
                alias_errors.append(f"{pool_id} CPU torch noise mismatch")
            if not np.array_equal(u1, np.asarray(case["candidates"], dtype=np.float32)):
                alias_errors.append(f"{pool_id} u1 differs from DEV CEM1 pool")
            ref_elite = _elite(dev_ref[pool_id])
            q_elite = _elite(dev_scores[alias][pool_id])
            ref_mu_cpu, ref_sigma_cpu = _stats(u1, ref_elite)
            q_mu_cpu, q_sigma_cpu = _stats(u1, q_elite)
            for key, expected in ((f"mu1_ref_{suffix}", ref_mu_cpu), (f"sigma1_ref_{suffix}", ref_sigma_cpu), (f"mu1_q_{suffix}", q_mu_cpu), (f"sigma1_q_{suffix}", q_sigma_cpu)):
                actual = np.asarray(arrays[key], dtype=np.float32)
                if not _array_close(actual, expected):
                    alias_errors.append(f"{pool_id} {key} differs from CPU torch stats")
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            for u2, mu_key, sigma_key, branch in ((u2_ref, f"mu1_ref_{suffix}", f"sigma1_ref_{suffix}", "ref"), (u2_q, f"mu1_q_{suffix}", f"sigma1_q_{suffix}", "q")):
                mu = np.asarray(arrays[mu_key], dtype=np.float32)
                sigma = np.asarray(arrays[sigma_key], dtype=np.float32)
                expected = (noise * sigma + mu).astype(np.float32)
                expected[0] = mu
                if not _array_close(u2, expected):
                    alias_errors.append(f"{pool_id} u2_{branch} is inconsistent with saved noise/mu/sigma")
                if not np.array_equal(u2[0], mu):
                    alias_errors.append(f"{pool_id} u2_{branch}[0] is not mu")
            score2_ref = np.asarray(arrays[f"score2_ref_{suffix}"], dtype=np.float64).reshape(-1)
            score2_q = np.asarray(arrays[f"score2_q_{suffix}"], dtype=np.float64).reshape(-1)
            if score2_ref.shape != (300,) or score2_q.shape != (300,) or not np.isfinite(score2_ref).all() or not np.isfinite(score2_q).all():
                alias_errors.append(f"{pool_id} malformed score2 arrays")
                continue
            ref2_mu, ref2_sigma = _stats(u2_ref, _elite(score2_ref))
            q2_mu, q2_sigma = _stats(u2_q, _elite(score2_q))
            final_bias = float(np.mean((np.asarray(q2_mu, dtype=np.float64) - np.asarray(ref2_mu, dtype=np.float64)) ** 2, dtype=np.float64))
            final_rows[str(case["episode_id"])] = final_bias
            ref_raw_for_compare[pool_id] = u2_ref.copy()
            if row is not None:
                if not _close(float(row.get("final_mu_bias")), final_bias) or not _array_close(np.asarray(row.get("reference_mu_final"), dtype=np.float32), ref2_mu) or not _array_close(np.asarray(row.get("quantized_mu_final"), dtype=np.float32), q2_mu):
                    alias_errors.append(f"{pool_id} final mu row mismatch")
                if not np.array_equal(np.asarray(row.get("reference_elite_final"), dtype=np.int64), _elite(score2_ref)) or not np.array_equal(np.asarray(row.get("quantized_elite_final"), dtype=np.int64), _elite(score2_q)):
                    alias_errors.append(f"{pool_id} final elite indices mismatch")
        if reference_raw is None:
            reference_raw = ref_raw_for_compare
        else:
            alias_errors.extend(_array_consistency_errors(ref_raw_for_compare, reference_raw, "reference U2"))
        ref_scores_for_compare = {
            str(case["pool_id"]): np.asarray(arrays[f"score2_ref_{index:04d}"], dtype=np.float64).reshape(-1)
            for index, case in enumerate(iter1)
            if f"score2_ref_{index:04d}" in arrays
        }
        if reference_scores is None:
            reference_scores = ref_scores_for_compare
        else:
            alias_errors.extend(_array_consistency_errors(ref_scores_for_compare, reference_scores, "score2_ref"))
        overall = float(np.mean(list(final_rows.values()), dtype=np.float64)) if final_rows else float("nan")
        declared = dev_metrics_file.get(alias, {}).get("chain", {}).get("aggregate_final_mu_bias") if isinstance(dev_metrics_file.get(alias), Mapping) else None
        if declared is not None and not _close(float(declared), overall):
            alias_errors.append("chain aggregate differs from dev_metrics.json")
        result[alias] = {"pass": not alias_errors, "errors": alias_errors, "aggregate_final_mu_bias": overall, "episode_final_mu_bias": final_rows, "max_cpu_stat_abs_error": max_error, "score2_recomputed_from_raw": bool(final_rows)}
        errors.extend(f"{alias}: {item}" for item in alias_errors)
    return {"pass": not errors, "errors": errors, "methods": result, "score2_model_provenance": "raw score2 shape/finite checked; model-free verifier cannot rescore U2"}


def _rel_gate(value: float, baseline: float, name: str) -> Dict[str, Any]:
    threshold = float(baseline * (1.0 - GATE_REL))
    margin = float(value - threshold)
    return {"name": name, "value": float(value), "threshold": threshold, "pass": bool(margin <= 0.0), "ambiguous": bool(abs(margin) <= STAT_TOL)}


def _episode_gate(values: Mapping[str, float], baseline: Mapping[str, float], name: str) -> Dict[str, Any]:
    deltas = {episode: float(values[episode] - baseline[episode]) for episode in sorted(values)}
    ambiguous = [episode for episode, delta in deltas.items() if abs(delta + CAL_TOL) <= STAT_TOL]
    improved = sum(delta < -CAL_TOL for delta in deltas.values())
    return {"name": name, "improved_episodes": improved, "required_episodes": 3, "deltas": deltas, "pass": improved >= 3, "ambiguous_episodes": ambiguous, "ambiguous": bool(ambiguous)}


def _gates(dev_maps: Mapping[str, Any], chains: Mapping[str, Any], search: Mapping[str, Any], target: Mapping[str, Any], workloads: Mapping[str, Any]) -> Dict[str, Any]:
    gates: List[Dict[str, Any]] = []
    engineering_pass = bool(target.get("pass") and workloads.get("cal", {}).get("pass") and workloads.get("dev", {}).get("pass") and workloads.get("planner", {}).get("pass", True) and search.get("pass") and chains.get("pass"))
    gates.append({"name": "engineering", "pass": engineering_pass, "ambiguous": bool(search.get("ambiguous")), "restore_evidence": "search runner exact-weight assertion; verifier does not load model"})
    update_search = search.get("methods", {}).get("CEM-Update", {})
    gates.append({"name": "stable_CAL_improvement", "pass": int(update_search.get("accepted_rounds", 0)) > 0 and not bool(search.get("ambiguous")), "accepted_rounds": int(update_search.get("accepted_rounds", 0)), "ambiguous": bool(search.get("ambiguous"))})
    selected_ids = {name: str(value.get("map_id")) for name, value in dev_maps.items()}
    distinct = selected_ids.get("CEM-Update") != selected_ids.get("MeanOnly")
    gates.append({"name": "update_vs_mean_mapping_distinct", "pass": distinct, "map_ids": selected_ids})
    shared = dev_maps.get("shared_start", {}).get("overall", {})
    update = dev_maps.get("CEM-Update", {}).get("overall", {})
    update_chain = chains.get("methods", {}).get("CEM-Update", {})
    shared_chain = chains.get("methods", {}).get("shared_start", {})
    update_episodes = {episode: row["update"] for episode, row in dev_maps.get("CEM-Update", {}).get("episodes", {}).items()}
    shared_episodes = {episode: row["update"] for episode, row in dev_maps.get("shared_start", {}).get("episodes", {}).items()}
    if len(update_episodes) == 4 and set(update_episodes) == set(shared_episodes):
        gates.append(_episode_gate(update_episodes, shared_episodes, "CEM-Update vs shared start L_update per episode"))
    else:
        gates.append({"name": "CEM-Update vs shared start L_update per episode", "pass": False, "reason": "missing four matched episode values"})
    for field, label in (("update", "DEV L_update"),):
        gate = _rel_gate(update.get(field, float("nan")), shared.get(field, float("nan")), f"CEM-Update vs shared start {label}")
        gates.append(gate)
    if update_chain.get("episode_final_mu_bias") and shared_chain.get("episode_final_mu_bias"):
        gates.append(_rel_gate(float(update_chain.get("aggregate_final_mu_bias")), float(shared_chain.get("aggregate_final_mu_bias")), "CEM-Update vs shared start two-step final-mu MSE"))
        gates.append(_episode_gate(update_chain["episode_final_mu_bias"], shared_chain["episode_final_mu_bias"], "CEM-Update vs shared start final-mu per episode"))
    else:
        gates.append({"name": "CEM-Update vs shared start final-mu", "pass": False, "ambiguous": False, "reason": "missing chain rows"})
    for comparator in ("MeanOnly", "ScoreError", "Rank"):
        comp = dev_maps.get(comparator, {}).get("overall", {})
        gates.append(_rel_gate(update.get("update", float("nan")), comp.get("update", float("nan")), f"CEM-Update vs {comparator} DEV L_update"))
        comp_chain = chains.get("methods", {}).get(comparator, {})
        gates.append(_rel_gate(float(update_chain.get("aggregate_final_mu_bias", float("nan"))), float(comp_chain.get("aggregate_final_mu_bias", float("nan")),), f"CEM-Update vs {comparator} two-step final-mu MSE"))
    mean = dev_maps.get("MeanOnly", {}).get("overall", {})
    lmu_value = float(update.get("mu", float("nan")))
    lmu_threshold = float(mean.get("mu", float("nan")) * 1.05 + 1e-10)
    gates.append({"name": "L_mu non-inferiority vs MeanOnly", "value": lmu_value, "threshold": lmu_threshold, "pass": bool(lmu_value <= lmu_threshold), "ambiguous": bool(abs(lmu_value - lmu_threshold) <= STAT_TOL)})
    gates.append({"name": "resource_gate", "status": "not_assessed", "pass": True, "ambiguous": False, "reason": "SLURM/GPU accounting is an external resource decision, not a fidelity significance test"})
    mechanism = [gate for gate in gates if gate["name"] not in ("engineering", "resource_gate")]
    any_ambiguous = any(bool(gate.get("ambiguous")) for gate in mechanism)
    mechanism_pass = all(bool(gate.get("pass")) for gate in mechanism)
    mechanism_status = "ambiguous" if mechanism_pass and any_ambiguous else ("pass" if mechanism_pass else "fail")
    # Keep the resource entry visible for accounting review, but do not let its
    # placeholder pass value turn a hypothesis no-go into an engineering fail.
    status = "engineering_fail" if not engineering_pass else mechanism_status
    return {
        "status": status,
        "pass": bool(engineering_pass and mechanism_pass and not any_ambiguous),
        "engineering_pass": engineering_pass,
        "mechanism_gate_status": mechanism_status,
        "mechanism_gate_pass": bool(mechanism_pass and not any_ambiguous),
        "mechanism_ambiguous": any_ambiguous,
        "gates": gates,
    }


def _verify(args: argparse.Namespace) -> Dict[str, Any]:
    base = args.base.resolve()
    search = args.search.resolve()
    cal = _load_workload(base / "pools" / "newcal.pkl", "cal")
    dev = _load_workload(base / "pools" / "newdev.pkl", "dev")
    planner_audit = _var_scale_audit(base, search, [cal["payload"], dev["payload"]])
    workloads = {
        "cal": {"pass": not cal["planner_errors"], "pool_count": len(cal["cases"]), "planner_errors": cal["planner_errors"]},
        "dev": {"pass": not dev["planner_errors"], "pool_count": len(dev["cases"]), "planner_errors": dev["planner_errors"]},
        "planner": planner_audit,
    }
    target = _target_audit(base, args.old_fingerprints)
    if {case["dataset_index"] for case in cal["cases"]} & {case["dataset_index"] for case in dev["cases"]}:
        raise ValueError("CAL/DEV dataset indices overlap")
    search_summary_path = search / "summary.json"
    if not search_summary_path.is_file():
        raise ValueError("search summary.json is missing")
    search_summary = json.loads(search_summary_path.read_text(encoding="utf-8"))
    if search_summary.get("schema") != "cem-update-ptq-search-v1":
        raise ValueError("search summary is not a complete cem-update-ptq-search-v1 result")
    if search_summary.get("status") in {"budget_exhausted", "resource_incomplete"}:
        return {
            "schema": "cem-update-ptq-verification-v1",
            "status": "resource_incomplete",
            "engineering_pass": False,
            "mechanism_gate_pass": False,
            "resource_incomplete": True,
            "target_audit": target,
            "workloads": workloads,
            "search": {"status": search_summary.get("status"), "methods_completed": search_summary.get("methods_completed", [])},
            "research_status": "not_assessed",
            "limitations": ["The bounded search did not produce a complete result; no mechanism no-go conclusion is allowed."],
        }
    if search_summary.get("status") != "complete":
        raise ValueError("search summary is not a complete cem-update-ptq-search-v1 result")
    hardware_errors = _summary_hardware_errors(search_summary, [cal["payload"], dev["payload"]])
    if hardware_errors:
        raise ValueError("; ".join(hardware_errors))
    cal_ref = _load_scores(search / "cal_scores" / "reference.npz", cal["cases"])
    dev_ref = _load_scores(search / "dev_scores" / "reference.npz", dev["cases"])
    reference_errors = _reference_score_errors(cal_ref, cal["cases"], "CAL")
    reference_errors.extend(_reference_score_errors(dev_ref, dev["cases"], "DEV"))
    if reference_errors:
        raise ValueError("; ".join(reference_errors))
    initial = _bits(search_summary.get("shared_start", {}))
    trace = _trace_audit(search, cal["cases"], cal_ref, initial, int(search_summary.get("rounds_requested", 2)), search_summary.get("initial_cost"))
    for method in METHODS:
        declared = search_summary.get("methods", {}).get(method, {})
        observed = trace["methods"].get(method, {})
        for field in ("config_evaluations", "rounds_completed"):
            if isinstance(declared, Mapping) and field in declared:
                wanted = int(declared[field])
                actual = int(observed.get("evaluations" if field == "config_evaluations" else "rounds_attempted", -1))
                if wanted != actual:
                    message = f"summary {field}={wanted} disagrees with trace={actual}"
                    observed.setdefault("errors", []).append(message)
                    observed["pass"] = False
                    trace["errors"].append(f"{method}: {message}")
                    trace["pass"] = False
    selected_payload = json.loads((search / "selected_maps.json").read_text(encoding="utf-8"))
    selected = {name: _bits(selected_payload[name]) for name in ("shared_start",) + METHODS}
    for name, value in selected_payload.items():
        if isinstance(value, Mapping) and "bits" in value and "map_id" in value and str(value["map_id"]) != _map_id(_bits(value)):
            raise ValueError(f"selected map ID mismatch for {name}")
    for method in METHODS:
        declared = search_summary.get("methods", {}).get(method, {})
        if isinstance(declared, Mapping) and isinstance(declared.get("selected"), Mapping):
            if str(declared["selected"].get("map_id")) != _map_id(selected[method]):
                raise ValueError(f"summary selected map mismatch for {method}")
    dev_maps: Dict[str, Any] = {}
    dev_scores: Dict[str, Dict[str, np.ndarray]] = {}
    dev_metrics_file = json.loads((search / "dev_metrics.json").read_text(encoding="utf-8")) if (search / "dev_metrics.json").is_file() else {}
    for name, bits in selected.items():
        map_id = _map_id(bits)
        path = search / "dev_scores" / f"map_{map_id}.npz"
        values = _load_scores(path, dev["cases"])
        dev_scores[name] = values
        metric = _map_metrics(dev["cases"], dev_ref, values)
        dev_maps[name] = {"map_id": map_id, "overall": metric["overall"], "episodes": {episode: {field: value for field, value in row.items() if field in ("mu", "sigma", "update", "mean_only", "score_error", "rank")} for episode, row in metric["episodes"].items()}}
        declared = dev_metrics_file.get(name, {}).get("metrics", {}) if isinstance(dev_metrics_file.get(name), Mapping) else {}
        if any(field in declared and not _close(float(declared[field]), metric["overall"][field]) for field in metric["overall"]):
            workloads.setdefault("dev", {}).setdefault("errors", []).append(f"dev_metrics mismatch for {name}")
            workloads["dev"]["pass"] = False
    chains = _chain_audit(search, dev["cases"], dev_ref, dev_scores, selected, dev_metrics_file)
    gates = _gates(dev_maps, chains, trace, target, workloads)
    engineering_pass = bool(target.get("pass") and workloads.get("cal", {}).get("pass") and workloads.get("dev", {}).get("pass") and workloads.get("planner", {}).get("pass") and trace.get("pass") and chains.get("pass"))
    return {
        "schema": "cem-update-ptq-verification-v1",
        "status": gates["status"],
        "engineering_pass": engineering_pass,
        "mechanism_gate_pass": gates["mechanism_gate_pass"],
        "mechanism_gate_status": gates["mechanism_gate_status"],
        "target_audit": target,
        "workloads": workloads,
        "search": {"trace_pass": trace["pass"], "trace_ambiguous": trace["ambiguous"], "shared_start_map_id": trace["shared_start_map_id"], "methods": {name: {key: value for key, value in row.items() if key not in ("errors",)} for name, row in trace["methods"].items()}},
        "dev": {"maps": dev_maps, "chain": chains},
        "gates": gates,
        "limitations": ["No model is loaded; restore is accepted from the search runner's exact-weight assertion.", "score2 arrays are checked for shape/finite values and replay metrics are recomputed, but score2 cannot be independently rescored without the model.", "CPU torch float32 statistics are compared to recorded GPU statistics with absolute tolerance 1e-6; near frozen thresholds are reported ambiguous."],
    }


def _self_test() -> None:
    bits = {f"encoder.base_model.blocks.{i}": 4 for i in range(12)}
    bits.update({f"predictor.transformer.layers.{i}": 4 for i in range(6)})
    for index in (0, 1, 11):
        bits[f"encoder.base_model.blocks.{index}"] = 8
    for index in (4, 5):
        bits[f"predictor.transformer.layers.{index}"] = 8
    assert len(_neighbors(bits)) == 35
    assert _map_id(bits) == _map_id(dict(reversed(list(bits.items()))))
    candidates = np.arange(300 * 5 * 10, dtype=np.float32).reshape(300, 5, 10)
    scores = np.arange(300, dtype=np.float64)
    metric = _metric(scores, scores, candidates, 1)
    assert metric["update"] == 0.0 and np.array_equal(metric["reference_elite_indices"], metric["quantized_elite_indices"])
    first = _shared_noise(CHAIN_SEED, POOL_SHAPE)
    assert np.array_equal(first, _shared_noise(CHAIN_SEED, POOL_SHAPE))
    reference_cases = [{"pool_id": "p", "reference_scores": np.zeros(300, dtype=np.float64)}]
    reference_scores = {"p": np.zeros(300, dtype=np.float64)}
    assert not _reference_score_errors(reference_scores, reference_cases, "CAL")
    near = reference_scores["p"].copy()
    near[0] = 5e-7
    assert not _reference_score_errors({"p": near}, reference_cases, "CAL")
    far = reference_scores["p"].copy()
    far[0] = 2e-6
    assert _reference_score_errors({"p": far}, reference_cases, "CAL")
    assert not _array_consistency_errors(reference_scores, {"p": reference_scores["p"].copy()}, "score2_ref")
    changed = reference_scores["p"].copy()
    changed[0] = 1.0
    assert _array_consistency_errors({"p": changed}, reference_scores, "score2_ref")
    gates = _gates({}, {"pass": True}, {"pass": True, "ambiguous": False}, {"pass": True}, {"cal": {"pass": True}, "dev": {"pass": True}, "planner": {"pass": True}})
    assert gates["engineering_pass"] and not gates["mechanism_gate_pass"] and gates["status"] == "fail"
    print(json.dumps({"status": "complete", "checks": ["35 neighbors", "stable map id", "stable elite", "CPU torch noise", "engineering/mechanism separation"]}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path)
    parser.add_argument("--search", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--old-fingerprints", type=Path, help="40 legacy target fingerprints JSON")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (args.base is None or args.search is None or args.output is None):
        parser.error("--base, --search and --output are required")
    return args


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    # This is intentionally the first operation that can inspect the remote
    # run context.  All workload/NPZ/model-independent heavy reads follow it.
    _load_guard()()
    try:
        result = _verify(args)
    except Exception as exc:
        result = {"schema": "cem-update-ptq-verification-v1", "status": "engineering_fail", "engineering_pass": False, "error": f"{type(exc).__name__}: {exc}"}
    _write_json(args.output.resolve(), result)
    if args.output.resolve().stat().st_size > 65536:
        raise RuntimeError("verification summary exceeds 64 KiB")
    print(json.dumps({"status": result.get("status"), "engineering_pass": result.get("engineering_pass", False), "mechanism_gate_pass": result.get("mechanism_gate_pass", False)}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

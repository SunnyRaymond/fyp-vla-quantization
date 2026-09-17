"""CCDS SLURM CEM-Update PTQ search; fake-quant emulation only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


METHODS = ("CEM-Update", "MeanOnly", "ScoreError", "Rank")
OBJECTIVE_FOR = {
    "CEM-Update": "update",
    "MeanOnly": "mean_only",
    "ScoreError": "score_error",
    "Rank": "rank",
}
EXPECTED_FAMILY_COUNTS = {"encoder": 12, "predictor": 6}
W8_QUOTA = {"encoder": 3, "predictor": 2}
BITS = (4, 8)
CAL_TOL = 1e-10
NMSE_EPS = 1e-12
CHAIN_SEED = 810_000


class DeadlineExceeded(RuntimeError):
    """Raised before starting another bounded operation."""


class Deadline:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        if self.elapsed >= self.seconds:
            raise DeadlineExceeded(f"max-seconds reached ({self.seconds:g})")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_smoke() -> Any:
    try:
        import smoke_runner as smoke

        return smoke
    except ModuleNotFoundError:
        return _load_local_module("cem_update_smoke_runner", Path(__file__).with_name("smoke_runner.py"))


def _require_allocation() -> Mapping[str, Any]:
    try:
        from allocation_guard import require_allocation
    except ModuleNotFoundError:
        guard = _load_local_module("cem_update_allocation_guard", Path(__file__).with_name("allocation_guard.py"))
        require_allocation = guard.require_allocation
    return require_allocation()


def _read_payload(path: Path) -> Any:
    path = path.resolve()
    if path.is_dir():
        for name in ("workload.pkl", "pools.pkl", "workload.json"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"no workload file under {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        return pickle.load(stream)


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _case_candidates(node: Mapping[str, Any]) -> Any:
    return _first(node, "candidates", "candidate_actions", "actions")


def _case_reference(node: Mapping[str, Any]) -> Any:
    direct = _first(node, "reference_scores", "fp32_scores", "scores_fp32", "reference_score")
    if direct is not None:
        return direct
    scores = node.get("scores")
    if isinstance(scores, Mapping):
        return _first(scores, "FP32", "fp32", "reference", "REF")
    return None


def _case_obs(node: Mapping[str, Any], name: str) -> Any:
    aliases = (name, "obs0" if name == "obs_0" else "obsg", "current_obs" if name == "obs_0" else "goal_obs")
    value = _first(node, *aliases)
    if value is not None:
        return value
    target = node.get("target")
    if isinstance(target, Mapping):
        return _first(target, *aliases)
    return None


def _episode_local_index(value: Any) -> int | None:
    match = re.search(r":(\d{3})$", str(value))
    return None if match is None else int(match.group(1))


def _validate_new_split(cases: Sequence[Mapping[str, Any]], label: str, expected_split: str) -> None:
    """Validate the fresh 4-episode/8-pool newcal/newdev contract."""
    if len(cases) != 8:
        raise ValueError(f"{label} workload must contain exactly 8 pools; got {len(cases)}")
    by_episode: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        if tuple(np.asarray(case["candidates"]).shape) != (300, 5, 10):
            raise ValueError(f"{label} pool {case['pool_id']!r} must have candidate shape (300,5,10)")
        by_episode[str(case["episode_id"])].append(case)
    if set(by_episode) != {f"{expected_split}:{index:03d}" for index in range(4)}:
        raise ValueError(f"{label} must contain episode IDs {expected_split}:000..003")
    expected_offset = 42 if expected_split == "newcal" else 46
    for episode_id, episode_cases in sorted(by_episode.items()):
        local_index = _episode_local_index(episode_id)
        if local_index is None or len(episode_cases) != 2:
            raise ValueError(f"{label} episode {episode_id!r} must contain exactly two pools")
        if {int(case["mpc_point"]) for case in episode_cases} != {0}:
            raise ValueError(f"{label} episode {episode_id!r} must use MPC point 0")
        if sorted(int(case["cem_iteration"]) for case in episode_cases) != [1, 5]:
            raise ValueError(f"{label} episode {episode_id!r} must contain CEM iterations 1 and 5")
        datasets = {case.get("dataset_index") for case in episode_cases}
        if datasets != {expected_offset + local_index}:
            raise ValueError(f"{label} episode {episode_id!r} has invalid dataset_index metadata: {datasets}")


def _validate_split_disjoint(cal: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]]) -> None:
    cal_indices = {int(case["dataset_index"]) for case in cal}
    dev_indices = {int(case["dataset_index"]) for case in dev}
    if cal_indices & dev_indices:
        raise ValueError(f"newcal/newdev dataset indices overlap: {sorted(cal_indices & dev_indices)}")
    if cal_indices != set(range(42, 46)) or dev_indices != set(range(46, 50)):
        raise ValueError(f"newcal/newdev dataset indices must be 42..45 and 46..49, got {sorted(cal_indices)}/{sorted(dev_indices)}")


def _stable_runtime_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: identity.get(key)
        for key in (
            "directory", "checkpoint_file", "recorded_epoch", "checkpoint_size_bytes",
            "source_commit", "dinov2_source_commit", "dtype", "decoder",
            "execution", "native_memory_claim", "torch",
        )
    }


def _gpu_family(value: Any) -> str:
    text = str(value or "").strip()
    if "v100" in text.casefold():
        return "V100"
    return text


def _validate_runtime_identity(cases: Sequence[Mapping[str, Any]], actual: Mapping[str, Any], label: str) -> None:
    supplied = [case.get("runtime_identity") for case in cases]
    if not supplied or any(not isinstance(item, Mapping) for item in supplied):
        raise ValueError(f"{label} workload is missing runtime_identity metadata")
    expected = _stable_runtime_identity(actual)
    for identity in supplied:
        if _stable_runtime_identity(identity) != expected:
            raise ValueError(f"{label} workload runtime_identity does not match loaded checkpoint")
    supplied_gpu = {_gpu_family(identity.get("gpu")) for identity in supplied}
    actual_gpu = _gpu_family(actual.get("gpu"))
    if supplied_gpu != {"V100"} or actual_gpu != "V100" or supplied_gpu != {actual_gpu}:
        raise ValueError(f"{label} candidate/reference workload is not paired with the reported CCDS V100")


def _validate_hardware_provenance(payload: Mapping[str, Any], label: str) -> None:
    provenance = payload.get("hardware_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("independent_rerun") is not True:
        raise ValueError(f"{label} workload lacks independent CCDS rerun provenance")
    if provenance.get("scheduler") != "slurm" or provenance.get("cluster") != "CCDS-TC1":
        raise ValueError(f"{label} workload has invalid CCDS scheduler provenance")
    if provenance.get("candidate_score_pairing") != "same_fresh_fp32_collection":
        raise ValueError(f"{label} workload does not pair fresh candidates and FP32 scores")
    if provenance.get("historical_arrays_reused") is not False:
        raise ValueError(f"{label} workload may mix historical candidate/score arrays")
    if _gpu_family(provenance.get("gpu_model")) != "V100":
        raise ValueError(f"{label} workload does not report a V100 reference GPU")


def _validate_planner(planner: Any, label: str) -> None:
    if not isinstance(planner, Mapping):
        raise ValueError(f"{label} workload is missing planner protocol metadata")
    expected = {
        "horizon": 5, "execute_model_actions": 5, "frameskip": 5,
        "num_samples": 300, "elite_count": 30, "cem_iterations": 5,
        "outer_mpc_round_limit": 12, "environment_step_limit": 300,
        "inner_environment_evaluator": None, "stable_argsort": True,
        "objective": {"mode": "last", "alpha": 1, "base": 2},
        "reference": {"dtype": "float32", "decoder": None, "activations": "float32"},
        "quantization_execution": "emulation_only",
    }
    for key, value in expected.items():
        if planner.get(key) != value:
            raise ValueError(f"{label} planner field {key!r} does not match frozen screen protocol")


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_value_equal(left[key], right[key]) for key in left)
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception:
        return left == right


def _validate_target_manifest(workload_path: Path, cases: List[Dict[str, Any]], expected_split: str, label: str) -> None:
    targets_dir = workload_path.resolve().parent.parent / "targets"
    manifest_path = targets_dir / f"episode_manifest_{expected_split}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{label} target manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != expected_split or manifest.get("episode_count") != 4:
        raise ValueError(f"{label} target manifest split/count is invalid")
    namespace = 600000 if expected_split == "newcal" else 700000
    offset = 42 if expected_split == "newcal" else 46
    entries = manifest.get("episodes")
    if not isinstance(entries, Sequence) or len(entries) != 4:
        raise ValueError(f"{label} target manifest must contain four episodes")
    by_episode = {str(entry.get("episode_id")): entry for entry in entries if isinstance(entry, Mapping)}
    for case in cases:
        episode_id = str(case["episode_id"])
        index = _episode_local_index(episode_id)
        entry = by_episode.get(episode_id)
        if index is None or entry is None:
            raise ValueError(f"{label} pool {case['pool_id']!r} has no matching target manifest entry")
        expected = {
            "local_index": index, "dataset_index": offset + index,
            "env_seed": namespace + index, "cem_seed": namespace + 10000 + index,
        }
        for key, value in expected.items():
            if int(entry.get(key, -1)) != value or int(case.get(key, -1)) != value:
                raise ValueError(f"{label} {episode_id} {key} metadata mismatch")
        target_path = targets_dir / str(entry.get("target_path", ""))
        if not target_path.resolve().is_relative_to(targets_dir.resolve()) or not target_path.is_file():
            raise ValueError(f"{label} target path is invalid: {target_path}")
        with target_path.open("rb") as stream:
            target = pickle.load(stream)
        fingerprint = str(entry.get("target_fingerprint", ""))
        if str(target.get("target_fingerprint")) != fingerprint:
            raise ValueError(f"{label} target fingerprint mismatch for {episode_id}")
        if case.get("target_fingerprint") not in (None, fingerprint):
            raise ValueError(f"{label} pool {case['pool_id']!r} target fingerprint disagrees")
        if not _value_equal(case["obs_0"], target["obs_0"]) or not _value_equal(case["obs_g"], target["obs_g"]):
            raise ValueError(f"{label} pool {case['pool_id']!r} raw obs_0/obs_g do not match target")
        case["target_fingerprint"] = fingerprint


def _load_cases(path: Path, label: str) -> List[Dict[str, Any]]:
    """Load direct ``cases`` records without requiring the historical 8-episode schema."""
    workload_path = path.resolve()
    if workload_path.is_dir():
        for name in ("workload.pkl", "pools.pkl", "workload.json"):
            candidate = workload_path / name
            if candidate.is_file():
                workload_path = candidate
                break
    payload = _read_payload(workload_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} workload must contain a mapping")
    expected_split = "newcal" if label == "cal" else "newdev"
    if str(payload.get("split", "")).lower() != expected_split:
        raise ValueError(f"{label} workload must declare split={expected_split!r}")
    _validate_hardware_provenance(payload, label)
    _validate_planner(payload.get("planner"), label)
    roots = payload.get("cases", payload.get("pools"))
    if roots is None:
        roots = [payload] if _case_candidates(payload) is not None else None
    if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)):
        raise ValueError(f"{label} workload must contain a sequence under cases")

    cases: List[Dict[str, Any]] = []

    def visit(node: Any, episode_hint: Any = None) -> None:
        if not isinstance(node, Mapping):
            raise TypeError(f"{label} case must be a mapping")
        candidates = _case_candidates(node)
        if candidates is None:
            nested = node.get("cases", node.get("pools"))
            if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes)):
                raise ValueError(f"{label} entry has neither candidates nor nested cases")
            hint = _first(node, "episode_id", "episode", default=episode_hint)
            for child in nested:
                visit(child, hint)
            return

        pool_id = _first(node, "pool_id", "case_id", "record_id", default=f"{label}-pool-{len(cases):04d}")
        episode_id = _first(node, "episode_id", "episode", default=episode_hint)
        if episode_id is None:
            episode_id = pool_id
        episode_index = _episode_local_index(episode_id)
        declared_local_index = _first(node, "local_index", default=None)
        if episode_index is not None and declared_local_index is not None and int(declared_local_index) != episode_index:
            raise ValueError(f"{label} pool {pool_id!r} local_index disagrees with episode_id")
        local_index = episode_index if episode_index is not None else int(declared_local_index if declared_local_index is not None else len(cases))
        candidate_array = np.asarray(candidates, dtype=np.float32)
        if candidate_array.ndim < 2 or candidate_array.shape[0] < 2:
            raise ValueError(f"{label} pool {pool_id!r} needs candidates shaped (N,...), N>=2")
        if candidate_array.ndim != 3:
            raise ValueError(f"{label} pool {pool_id!r} candidates must be 3D (N,H,A)")
        reference = _case_reference(node)
        if reference is None:
            raise ValueError(f"{label} pool {pool_id!r} is missing reference_scores")
        reference_array = np.asarray(reference, dtype=np.float64).reshape(-1)
        if reference_array is not None and reference_array.shape != (candidate_array.shape[0],):
            raise ValueError(f"{label} pool {pool_id!r} reference_scores length does not match candidates")
        obs_0 = _case_obs(node, "obs_0")
        obs_g = _case_obs(node, "obs_g")
        if obs_0 is None or obs_g is None:
            raise ValueError(f"{label} pool {pool_id!r} must provide obs_0 and obs_g (or target fields)")
        cases.append(
            {
                "pool_id": str(pool_id),
                "episode_id": str(episode_id),
                "local_index": int(local_index),
                "dataset_index": _first(node, "dataset_index", default=None),
                "env_seed": _first(node, "env_seed", default=None),
                "cem_seed": _first(node, "cem_seed", default=None),
                "target_fingerprint": _first(node, "target_fingerprint", default=None),
                "runtime_identity": payload.get("runtime_identity"),
                "hardware_provenance": payload.get("hardware_provenance"),
                "mpc_point": int(_first(node, "mpc_point", "mpc_round", "round", default=0)),
                "cem_iteration": int(_first(node, "cem_iteration", "iteration", default=0)),
                "obs_0": obs_0,
                "obs_g": obs_g,
                "candidates": candidate_array,
                "reference_scores": reference_array,
                "ids": _first(node, "ids", "id", default=None),
            }
        )

    for root in roots:
        visit(root)
    if not cases:
        raise ValueError(f"{label} workload contains no scoring cases")
    ids = [case["pool_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} workload contains duplicate pool IDs")
    for case in cases:
        if case["dataset_index"] is None:
            raise ValueError(f"{label} pool {case['pool_id']!r} is missing dataset_index metadata")
    _validate_new_split(cases, label, expected_split)
    _validate_target_manifest(workload_path, cases, expected_split, label)
    return cases


def _group_cost(group: Mapping[str, Any], bits: int) -> int:
    key = f"logical_weight_bytes_W{bits}"
    if key in group:
        return int(group[key])
    return int(math.ceil(int(group["numel"]) * bits / 8) + int(group.get("scale_count", 0)) * 4)


def _validate_groups(smoke: Any, runtime: Mapping[str, Any]) -> List[Dict[str, Any]]:
    groups = [dict(group) for group in smoke._linear_groups(runtime["model"])]
    counts = {family: sum(str(group.get("family")) == family for group in groups) for family in EXPECTED_FAMILY_COUNTS}
    if counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"runtime groups must be 12 encoder/6 predictor, got {counts}")
    group_ids = [str(group.get("group_id")) for group in groups]
    if len(group_ids) != len(set(group_ids)) or any(not value for value in group_ids):
        raise ValueError("runtime group IDs must be unique and non-empty")
    for family in W8_QUOTA:
        signatures = {
            (int(group["numel"]), int(group.get("scale_count", 0)), _group_cost(group, 4), _group_cost(group, 8))
            for group in groups if str(group["family"]) == family
        }
        if len(signatures) != 1:
            raise ValueError(f"{family} groups do not have equal W4/W8 logical cost")
    return sorted(groups, key=lambda group: (str(group["family"]), int(group["index"]), str(group["group_id"])))


def _allocation_cost(groups: Sequence[Mapping[str, Any]], bits: Mapping[str, int]) -> Dict[str, Any]:
    by_family: Dict[str, Dict[str, int]] = {}
    total_bytes = 0
    total_weight_bits = 0
    for family in W8_QUOTA:
        family_groups = [group for group in groups if str(group["family"]) == family]
        family_bytes = sum(_group_cost(group, int(bits[str(group["group_id"])])) for group in family_groups)
        family_weight_bits = sum(int(group["numel"]) * int(bits[str(group["group_id"])]) for group in family_groups)
        by_family[family] = {
            "w8_count": sum(int(bits[str(group["group_id"])]) == 8 for group in family_groups),
            "logical_weight_bytes": int(family_bytes),
            "weight_bits": int(family_weight_bits),
        }
        total_bytes += family_bytes
        total_weight_bits += family_weight_bits
    return {"logical_weight_bytes": int(total_bytes), "weight_bits": int(total_weight_bits), "by_family": by_family}


def _validate_map(groups: Sequence[Mapping[str, Any]], bits: Mapping[str, Any]) -> Dict[str, int]:
    ids = [str(group["group_id"]) for group in groups]
    result = {str(key): int(value) for key, value in bits.items()}
    if set(result) != set(ids):
        raise ValueError(f"map must specify every runtime group; missing={sorted(set(ids)-set(result))}")
    if any(value not in BITS for value in result.values()):
        raise ValueError("map bit widths must be 4 or 8")
    counts = {
        family: sum(result[str(group["group_id"])] == 8 for group in groups if str(group["family"]) == family)
        for family in W8_QUOTA
    }
    if counts != W8_QUOTA:
        raise ValueError(f"map W8 quotas must be {W8_QUOTA}, got {counts}")
    return {group_id: result[group_id] for group_id in ids}


def _load_initial(path: Path, groups: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    payload = _read_payload(path)
    if not isinstance(payload, Mapping):
        raise ValueError("initial allocation must contain a mapping")
    if str(payload.get("method", "")) != "ScoreError":
        raise ValueError("initial allocation must be the frozen ScoreError map")
    raw = payload.get("allocation", payload.get("bits"))
    if not isinstance(raw, Mapping):
        raise ValueError("initial allocation needs allocation/bits mapping")
    bits: Dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            value = value.get("bits")
        bits[str(key)] = int(value)
    result = _validate_map(groups, bits)
    if payload.get("map_id") is not None and str(payload["map_id"]) != _map_id(result):
        raise ValueError("initial declared map_id does not match its allocation")
    declared_cost = payload.get("cost")
    if isinstance(declared_cost, Mapping):
        actual_cost = _allocation_cost(groups, result)
        for field in ("logical_weight_bytes", "weight_bits"):
            if field in declared_cost and int(declared_cost[field]) != int(actual_cost[field]):
                raise ValueError(
                    f"initial allocation {field}={declared_cost[field]} does not match runtime registry "
                    f"{actual_cost[field]}"
                )
    return result


def _map_key(bits: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((str(key), int(value)) for key, value in bits.items()))


def _map_id(bits: Mapping[str, int]) -> str:
    encoded = json.dumps(dict(_map_key(bits)), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _public_map(groups: Sequence[Mapping[str, Any]], bits: Mapping[str, int]) -> Dict[str, Any]:
    ordered = {str(group["group_id"]): int(bits[str(group["group_id"])]) for group in groups}
    return {
        "map_id": _map_id(ordered),
        "bits": ordered,
        "selected_sites": [key for key, value in ordered.items() if value == 8],
    }


def _neighbors(groups: Sequence[Mapping[str, Any]], bits: Mapping[str, int]) -> List[Dict[str, int]]:
    """Enumerate stable same-family W8/W4 swaps: 3*9 + 2*4 = 35."""
    result: List[Dict[str, int]] = []
    for family in ("encoder", "predictor"):
        family_groups = sorted(
            (group for group in groups if str(group["family"]) == family),
            key=lambda group: (int(group["index"]), str(group["group_id"])),
        )
        high = [group for group in family_groups if int(bits[str(group["group_id"])]) == 8]
        low = [group for group in family_groups if int(bits[str(group["group_id"])]) == 4]
        for source in high:
            for target in low:
                candidate = dict(bits)
                candidate[str(source["group_id"])] = 4
                candidate[str(target["group_id"])] = 8
                result.append(candidate)
    keys = [_map_key(item) for item in result]
    if len(keys) != len(set(keys)):
        raise RuntimeError("neighbor generator produced duplicate maps")
    return result


def _stable_elite(scores: Sequence[float], candidates: np.ndarray, topk: int, device: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape[0] != candidates.shape[0] or not np.isfinite(values).all():
        raise ValueError("score vector is non-finite or does not match candidate count")
    if values.shape[0] < max(2, topk):
        raise ValueError(f"candidate pool needs at least topk={topk} and two candidates")
    elite = np.argsort(values, kind="stable")[:topk]
    import torch

    selected = torch.as_tensor(np.asarray(candidates, dtype=np.float32)[elite], dtype=torch.float32, device=device or "cpu")
    mean = selected.mean(dim=0)
    try:
        sigma = selected.std(dim=0, correction=1)
    except TypeError:  # torch versions before the correction keyword
        sigma = selected.std(dim=0, unbiased=True)
    return mean.detach().cpu().numpy().astype(np.float32, copy=True), sigma.detach().cpu().numpy().astype(np.float32, copy=True), elite


def _pairwise_rank(reference: Sequence[float], quantized: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    quant = np.asarray(quantized, dtype=np.float64).reshape(-1)
    if ref.shape != quant.shape or ref.ndim != 1 or ref.size < 2:
        raise ValueError("score vectors must have the same one-dimensional shape")
    i, j = np.triu_indices(ref.size, k=1)
    return float(np.mean(np.sign(ref[i] - ref[j]) != np.sign(quant[i] - quant[j])))


def _score_nmse(reference: Sequence[float], quantized: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    quant = np.asarray(quantized, dtype=np.float64).reshape(-1)
    return float(np.mean((quant - ref) ** 2) / (np.mean(ref ** 2) + NMSE_EPS))


def _pool_metrics(reference: np.ndarray, quantized: np.ndarray, candidates: np.ndarray, topk: int, iteration: int, device: Any = None) -> Dict[str, Any]:
    ref_mu, ref_sigma, ref_elite = _stable_elite(reference, candidates, topk, device=device)
    q_mu, q_sigma, q_elite = _stable_elite(quantized, candidates, topk, device=device)
    mu_delta = np.asarray(q_mu, dtype=np.float64) - np.asarray(ref_mu, dtype=np.float64)
    sigma_delta = np.asarray(q_sigma, dtype=np.float64) - np.asarray(ref_sigma, dtype=np.float64)
    l_mu = float(np.mean(mu_delta * mu_delta, dtype=np.float64))
    l_sigma = float(np.mean(sigma_delta * sigma_delta, dtype=np.float64))
    l_update = l_mu + (0.0 if int(iteration) == 5 else l_sigma)
    return {
        "mu": l_mu,
        "sigma": l_sigma,
        "update": float(l_update),
        "mean_only": l_mu,
        "score_error": _score_nmse(reference, quantized),
        "rank": _pairwise_rank(reference, quantized),
        "reference_elite_indices": ref_elite,
        "quantized_elite_indices": q_elite,
        "reference_mu": ref_mu,
        "reference_sigma": ref_sigma,
        "quantized_mu": q_mu,
        "quantized_sigma": q_sigma,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    by_episode: MutableMapping[str, List[float]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(float(row[field]))
    if not by_episode:
        return float("nan")
    return float(np.mean([np.mean(values) for _, values in sorted(by_episode.items())]))


def _metrics_for_scores(cases: Sequence[Mapping[str, Any]], references: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray], topk: int, device: Any = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        pool_id = str(case["pool_id"])
        row = _pool_metrics(
            np.asarray(references[pool_id], dtype=np.float64),
            np.asarray(scores[pool_id], dtype=np.float64),
            np.asarray(case["candidates"], dtype=np.float64),
            topk,
            int(case["cem_iteration"]),
            device=device,
        )
        row.update({"pool_id": pool_id, "episode_id": str(case["episode_id"]), "cem_iteration": int(case["cem_iteration"])})
        rows.append(row)
    aggregate = {field: _aggregate(rows, field) for field in ("mu", "sigma", "update", "mean_only", "score_error", "rank")}
    return {"aggregate": aggregate, "pools": rows}


def _reference_scores(smoke: Any, runtime: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], preprocessor: Any, objective: Any, snapshot: Mapping[str, Any], deadline: Deadline, split: str) -> Dict[str, np.ndarray]:
    references: Dict[str, np.ndarray] = {}
    model = runtime["model"]
    for case in cases:
        deadline.check()
        smoke._restore_weights(model, snapshot)
        score = smoke._score_pool(
            model, preprocessor, objective,
            {"obs_0": case["obs_0"], "obs_g": case["obs_g"]}, case["candidates"],
        ).detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
        if not np.isfinite(score).all():
            raise FloatingPointError(f"non-finite FP32 scores for {split}/{case['pool_id']}")
        supplied = case.get("reference_scores")
        maximum = None
        top30_match = None
        if supplied is not None:
            supplied_array = np.asarray(supplied, dtype=np.float64).reshape(-1)
            maximum = float(np.max(np.abs(supplied_array - score)))
            if not np.allclose(supplied_array, score, rtol=0.0, atol=1e-6):
                raise ValueError(f"FP32 reference mismatch for {split}/{case['pool_id']}: max_abs={maximum:g}")
            top30_match = bool(np.array_equal(np.argsort(supplied_array, kind="stable")[:30], np.argsort(score, kind="stable")[:30]))
            if not top30_match:
                raise ValueError(f"FP32 reference stable top30 mismatch for {split}/{case['pool_id']}")
        case["reference_max_abs"] = maximum
        case["reference_top30_match"] = top30_match
        references[str(case["pool_id"])] = score
    smoke._restore_weights(model, snapshot)
    _assert_restored(model, snapshot)
    deadline.check()
    first = cases[0]
    check = smoke._score_pool(
        model, preprocessor, objective,
        {"obs_0": first["obs_0"], "obs_g": first["obs_g"]}, first["candidates"],
    ).detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    if not np.array_equal(check, references[str(first["pool_id"])]):
        raise RuntimeError(f"FP32 restore score is not exact for {split}/{first['pool_id']}")
    return references


def _reference_audit(cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{"pool_id": str(case["pool_id"]), "max_abs": case.get("reference_max_abs"), "stable_top30_match": case.get("reference_top30_match")} for case in cases]


def _assert_restored(model: Any, snapshot: Mapping[str, Any]) -> None:
    for key, original in snapshot.items():
        family, index, relative = key.split(".", 2)
        block = model.encoder.base_model.blocks[int(index)] if family == "encoder" else model.predictor.transformer.layers[int(index)]
        current = dict(block.named_modules())[relative].weight.detach()
        if not np.array_equal(current.cpu().numpy(), original.detach().cpu().numpy()):
            raise RuntimeError(f"weight restore mismatch at {key}")


def _score_map(
    smoke: Any,
    runtime: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    bits: Mapping[str, int],
    cases: Sequence[Mapping[str, Any]],
    preprocessor: Any,
    objective: Any,
    cache: MutableMapping[Tuple[str, str], Dict[str, Any]],
    split: str,
    output: Path,
    deadline: Deadline,
    artifact_suffix: str = "",
) -> Dict[str, Any]:
    key = (split, _map_id(bits))
    if key in cache:
        hit = dict(cache[key])
        hit["cache_hit"] = True
        hit["elapsed_seconds"] = 0.0
        hit["apply_seconds"] = 0.0
        hit["forward_seconds"] = 0.0
        hit["forward_count"] = 0
        return hit

    model = runtime["model"]
    started = time.monotonic()
    apply_started = time.monotonic()
    smoke._restore_weights(model, snapshot)
    quantization = []
    try:
        for group in groups:
            quantization.append(smoke._quantize_group(model, group, int(bits[str(group["group_id"])])))
        apply_seconds = time.monotonic() - apply_started
        forward_started = time.monotonic()
        scores: Dict[str, np.ndarray] = {}
        for case in cases:
            deadline.check()
            value = smoke._score_pool(
                model, preprocessor, objective,
                {"obs_0": case["obs_0"], "obs_g": case["obs_g"]}, case["candidates"],
            ).detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
            if not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite {split} scores for map {_map_id(bits)} pool {case['pool_id']}")
            scores[str(case["pool_id"])] = value
        forward_seconds = time.monotonic() - forward_started
    finally:
        smoke._restore_weights(model, snapshot)
        _assert_restored(model, snapshot)
    record = {
        "map_id": _map_id(bits),
        "bits": dict(_map_key(bits)),
        "scores": scores,
        "apply_seconds": float(apply_seconds),
        "forward_seconds": float(forward_seconds),
        "elapsed_seconds": float(time.monotonic() - started),
        "forward_count": int(len(cases)),
        "quantization": quantization,
        "cache_hit": False,
    }
    cache[key] = record
    arrays = {"pool_ids": np.asarray([str(case["pool_id"]) for case in cases])}
    arrays.update({f"pool_{index:04d}": scores[str(case["pool_id"])] for index, case in enumerate(cases)})
    stem = f"map_{_map_id(bits)}{artifact_suffix}"
    _atomic_npz(output / f"{split}_scores" / f"{stem}.npz", arrays)
    _atomic_json(output / f"{split}_scores" / f"{stem}.json", {"map_id": _map_id(bits), "bits": dict(_map_key(bits)), "pool_ids": arrays["pool_ids"].tolist(), "apply_seconds": record["apply_seconds"], "forward_seconds": record["forward_seconds"], "forward_count": record["forward_count"], "artifact_suffix": artifact_suffix})
    return record


def _write_reference_npz(output: Path, split: str, cases: Sequence[Mapping[str, Any]], references: Mapping[str, np.ndarray]) -> None:
    arrays = {"pool_ids": np.asarray([str(case["pool_id"]) for case in cases])}
    arrays.update({f"pool_{index:04d}": references[str(case["pool_id"])] for index, case in enumerate(cases)})
    _atomic_npz(output / f"{split}_scores" / "reference.npz", arrays)


def _trace_append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=_json_default, separators=(",", ":")) + "\n")


def _evaluate_config(
    method: str,
    round_index: int,
    bits: Mapping[str, int],
    groups: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    references: Mapping[str, np.ndarray],
    score_record: Mapping[str, Any],
    cost: Mapping[str, Any],
    trace_path: Path,
    config_index: int,
    device: Any = None,
) -> Dict[str, Any]:
    metrics = _metrics_for_scores(cases, references, score_record["scores"], 30, device=device)
    aggregate = metrics["aggregate"]
    objective_name = OBJECTIVE_FOR[method]
    row = {
        "event": "config_eval",
        "method": method,
        "round": int(round_index),
        "config_index": int(config_index),
        "map_id": str(score_record["map_id"]),
        "bits": dict(_map_key(bits)),
        "cost": cost,
        "objective_name": objective_name,
        "objective": float(aggregate[objective_name]),
        "metrics": {key: float(value) for key, value in aggregate.items()},
        "cache_hit": bool(score_record["cache_hit"]),
        "apply_seconds": float(score_record["apply_seconds"]),
        "forward_seconds": float(score_record["forward_seconds"]),
        "elapsed_seconds": float(score_record["elapsed_seconds"]),
        "forward_count": int(score_record["forward_count"]),
        "pool_count": len(cases),
    }
    _trace_append(trace_path, row)
    return row


def _search_method(
    method: str,
    start: Mapping[str, int],
    groups: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    references: Mapping[str, np.ndarray],
    smoke: Any,
    runtime: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    preprocessor: Any,
    objective: Any,
    cache: MutableMapping[Tuple[str, str], Dict[str, Any]],
    output: Path,
    trace_path: Path,
    deadline: Deadline,
    rounds: int,
    method_index: int,
) -> Dict[str, Any]:
    current = dict(start)
    evaluations: List[Dict[str, Any]] = []
    current_row: Dict[str, Any] | None = None
    current_cost = _allocation_cost(groups, current)
    for round_index in range(-1, rounds):
        if round_index == -1:
            deadline.check()
            record = _score_map(smoke, runtime, groups, snapshot, current, cases, preprocessor, objective, cache, "cal", output, deadline)
            current_row = _evaluate_config(method, -1, current, groups, cases, references, record, current_cost, trace_path, len(evaluations), device=runtime.get("device"))
            evaluations.append(current_row)
            continue
        neighbors = _neighbors(groups, current)
        if len(neighbors) != 35:
            raise RuntimeError(f"expected 35 same-family neighbors, got {len(neighbors)}")
        best_row: Dict[str, Any] | None = None
        best_bits: Dict[str, int] | None = None
        for neighbor in neighbors:
            deadline.check()
            record = _score_map(smoke, runtime, groups, snapshot, neighbor, cases, preprocessor, objective, cache, "cal", output, deadline)
            row = _evaluate_config(method, round_index, neighbor, groups, cases, references, record, _allocation_cost(groups, neighbor), trace_path, len(evaluations), device=runtime.get("device"))
            evaluations.append(row)
            if best_row is None:
                best_row, best_bits = row, neighbor
            elif float(row["objective"]) < float(best_row["objective"]) - CAL_TOL or (
                abs(float(row["objective"]) - float(best_row["objective"])) <= CAL_TOL
                and best_bits is not None
                and _map_key(neighbor) < _map_key(best_bits)
            ):
                best_row, best_bits = row, neighbor
        if current_row is None or best_row is None or best_bits is None:
            raise RuntimeError("search state was not initialized")
        improvement = float(current_row["objective"]) - float(best_row["objective"])
        accepted = improvement > CAL_TOL
        if accepted:
            current = dict(best_bits)
            current_row = best_row
        _trace_append(trace_path, {
            "event": "round_end",
            "method": method,
            "round": int(round_index),
            "current_map_id": str(current_row["map_id"]),
            "best_neighbor_map_id": str(best_row["map_id"]),
            "current_objective": float(current_row["objective"]),
            "best_neighbor_objective": float(best_row["objective"]),
            "improvement": float(improvement),
            "accepted": bool(accepted),
            "tie_tolerance": CAL_TOL,
        })
        if not accepted:
            break
    return {
        "method": method,
        "objective_name": OBJECTIVE_FOR[method],
        "start": _public_map(groups, start),
        "selected": _public_map(groups, current),
        "selected_objective": float(current_row["objective"] if current_row is not None else float("nan")),
        "selected_metrics": dict(current_row["metrics"] if current_row is not None else {}),
        "config_evaluations": len(evaluations),
        "rounds_completed": max(0, len({int(row["round"]) for row in evaluations if int(row["round"]) >= 0})),
        "evaluations": evaluations,
    }


def _score_many(
    smoke: Any,
    runtime: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    bits: Mapping[str, int] | None,
    pairs: Sequence[Tuple[Mapping[str, Any], np.ndarray]],
    preprocessor: Any,
    objective: Any,
    deadline: Deadline,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Score generated second-step pools after one complete map application."""
    model = runtime["model"]
    apply_started = time.monotonic()
    smoke._restore_weights(model, snapshot)
    try:
        quantization = [] if bits is None else [smoke._quantize_group(model, group, int(bits[str(group["group_id"])])) for group in groups]
        apply_seconds = time.monotonic() - apply_started
        forward_started = time.monotonic()
        scores: Dict[str, np.ndarray] = {}
        for case, candidates in pairs:
            deadline.check()
            value = smoke._score_pool(model, preprocessor, objective, {"obs_0": case["obs_0"], "obs_g": case["obs_g"]}, candidates).detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
            if not np.isfinite(value).all():
                raise FloatingPointError("non-finite generated chain score")
            scores[str(case["pool_id"])] = value
        forward_seconds = time.monotonic() - forward_started
    finally:
        smoke._restore_weights(model, snapshot)
        _assert_restored(model, snapshot)
    return scores, {"apply_seconds": float(apply_seconds), "forward_seconds": float(forward_seconds), "forward_count": len(pairs), "quantization": quantization}


def _shared_noise(seed: int, shape: Tuple[int, ...]) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)


def _sample_cem_candidates(mu: np.ndarray, sigma: np.ndarray, shape: Tuple[int, ...], seed: int, device: Any) -> np.ndarray:
    """Generate float32 next-pool actions on the runtime device from shared CPU noise."""
    import torch

    runtime_device = device or "cpu"
    noise_cpu = _shared_noise(seed, shape)
    noise = noise_cpu.to(runtime_device)
    mean = torch.as_tensor(mu, dtype=torch.float32, device=runtime_device)
    std = torch.as_tensor(sigma, dtype=torch.float32, device=runtime_device)
    candidates = noise * std + mean
    candidates[0] = mean
    return candidates.detach().cpu().numpy().astype(np.float32, copy=True)


def _chain_dev(
    method: str,
    bits: Mapping[str, int],
    groups: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    references: Mapping[str, np.ndarray],
    initial_scores: Mapping[str, np.ndarray],
    smoke: Any,
    runtime: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    preprocessor: Any,
    objective: Any,
    chain_reference_cache: MutableMapping[str, Dict[str, Any]],
    deadline: Deadline,
    output: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    iter1 = [case for case in cases if int(case["cem_iteration"]) == 1]
    if not iter1:
        _atomic_npz(output / "dev_chain" / f"{method}.npz", {"pool_ids": np.asarray([], dtype="U1")})
        return {"method": method, "map_id": _map_id(bits), "status": "no_iter1_pools", "pools": [], "aggregate_final_mu_bias": None, "reference_chain_forward_count": 0, "quantized_chain_forward_count": 0}
    q_pairs: List[Tuple[Mapping[str, Any], np.ndarray]] = []
    q_initial: Dict[str, Dict[str, Any]] = {}
    ref_initial: Dict[str, Dict[str, Any]] = {}
    noises: Dict[str, np.ndarray] = {}
    stats_device = runtime.get("device")
    for chain_index, case in enumerate(iter1):
        deadline.check()
        pool_id = str(case["pool_id"])
        ref_mu, ref_sigma, ref_elite = _stable_elite(references[pool_id], case["candidates"], 30, device=stats_device)
        q_mu, q_sigma, q_elite = _stable_elite(initial_scores[pool_id], case["candidates"], 30, device=stats_device)
        noise_seed = CHAIN_SEED + int(case.get("local_index", chain_index))
        ref_candidates = _sample_cem_candidates(ref_mu, ref_sigma, tuple(case["candidates"].shape), noise_seed, stats_device)
        q_candidates = _sample_cem_candidates(q_mu, q_sigma, tuple(case["candidates"].shape), noise_seed, stats_device)
        noises[pool_id] = _shared_noise(noise_seed, tuple(case["candidates"].shape)).numpy().astype(np.float32, copy=True)
        q_pairs.append((case, q_candidates))
        ref_initial[pool_id] = {"mu": ref_mu, "sigma": ref_sigma, "candidates": ref_candidates, "elite": ref_elite, "noise_seed": noise_seed}
        q_initial[pool_id] = {"mu": q_mu, "sigma": q_sigma, "candidates": q_candidates, "elite": q_elite, "noise_seed": noise_seed}

    uncached_ref = [(case, values["candidates"].astype(np.float32)) for case, values in ((case, ref_initial[str(case["pool_id"])]) for case in iter1) if str(case["pool_id"]) not in chain_reference_cache]
    if uncached_ref:
        generated, timing = _score_many(smoke, runtime, groups, snapshot, None, uncached_ref, preprocessor, objective, deadline)
        for case, _ in uncached_ref:
            pool_id = str(case["pool_id"])
            ref_values = generated[pool_id]
            mu2, sigma2, elite2 = _stable_elite(ref_values, ref_initial[pool_id]["candidates"], 30, device=stats_device)
            chain_reference_cache[pool_id] = {"scores": ref_values, "mu": mu2, "sigma": sigma2, "elite": elite2, "timing": timing}
    generated_q, q_timing = _score_many(smoke, runtime, groups, snapshot, bits, q_pairs, preprocessor, objective, deadline)
    for case in iter1:
        pool_id = str(case["pool_id"])
        ref2 = chain_reference_cache[pool_id]
        q_values = generated_q[pool_id]
        q_mu2, q_sigma2, q_elite2 = _stable_elite(q_values, q_initial[pool_id]["candidates"], 30, device=stats_device)
        ref_mu2 = np.asarray(ref2["mu"], dtype=np.float64)
        final_delta = np.asarray(q_mu2, dtype=np.float64) - ref_mu2
        initial_mu_delta = np.asarray(q_initial[pool_id]["mu"], dtype=np.float64) - np.asarray(ref_initial[pool_id]["mu"], dtype=np.float64)
        initial_sigma_delta = np.asarray(q_initial[pool_id]["sigma"], dtype=np.float64) - np.asarray(ref_initial[pool_id]["sigma"], dtype=np.float64)
        final_bias = float(np.mean(final_delta * final_delta, dtype=np.float64))
        initial_mu_bias = float(np.mean(initial_mu_delta * initial_mu_delta, dtype=np.float64))
        initial_sigma_bias = float(np.mean(initial_sigma_delta * initial_sigma_delta, dtype=np.float64))
        rows.append({
            "method": method,
            "map_id": _map_id(bits),
            "pool_id": pool_id,
            "episode_id": str(case["episode_id"]),
            "noise_seed": int(q_initial[pool_id]["noise_seed"]),
            "candidate_zero_is_mu": True,
            "std_correction": 1,
            "initial_mu_bias": initial_mu_bias,
            "initial_sigma_bias": initial_sigma_bias,
            "final_mu_bias": final_bias,
            "reference_mu_final": ref_mu2,
            "quantized_mu_final": q_mu2,
            "reference_sigma_final": np.asarray(ref2["sigma"]),
            "quantized_sigma_final": q_sigma2,
            "reference_elite_final": ref2["elite"],
            "quantized_elite_final": q_elite2,
            "quantized_chain_apply_seconds": q_timing["apply_seconds"],
            "quantized_chain_forward_seconds": q_timing["forward_seconds"],
            "quantized_chain_forward_count": q_timing["forward_count"],
        })
    arrays: Dict[str, Any] = {"pool_ids": np.asarray([str(case["pool_id"]) for case in iter1])}
    for index, case in enumerate(iter1):
        pool_id = str(case["pool_id"])
        arrays[f"u1_{index:04d}"] = np.asarray(case["candidates"], dtype=np.float32)
        arrays[f"u2_ref_{index:04d}"] = np.asarray(ref_initial[pool_id]["candidates"], dtype=np.float32)
        arrays[f"u2_q_{index:04d}"] = np.asarray(q_initial[pool_id]["candidates"], dtype=np.float32)
        arrays[f"noise_{index:04d}"] = noises[pool_id]
        arrays[f"mu1_ref_{index:04d}"] = np.asarray(ref_initial[pool_id]["mu"], dtype=np.float32)
        arrays[f"sigma1_ref_{index:04d}"] = np.asarray(ref_initial[pool_id]["sigma"], dtype=np.float32)
        arrays[f"mu1_q_{index:04d}"] = np.asarray(q_initial[pool_id]["mu"], dtype=np.float32)
        arrays[f"sigma1_q_{index:04d}"] = np.asarray(q_initial[pool_id]["sigma"], dtype=np.float32)
        arrays[f"score2_ref_{index:04d}"] = np.asarray(chain_reference_cache[pool_id]["scores"], dtype=np.float64)
        arrays[f"score2_q_{index:04d}"] = np.asarray(generated_q[pool_id], dtype=np.float64)
    _atomic_npz(output / "dev_chain" / f"{method}.npz", arrays)
    return {
        "method": method,
        "map_id": _map_id(bits),
        "status": "complete",
        "pools": rows,
        "aggregate_final_mu_bias": _aggregate(rows, "final_mu_bias"),
        "aggregate_initial_mu_bias": _aggregate(rows, "initial_mu_bias"),
        "aggregate_initial_sigma_bias": _aggregate(rows, "initial_sigma_bias"),
        "reference_chain_forward_count": int(len(uncached_ref)),
        "quantized_chain_forward_count": int(len(q_pairs)),
    }


def _benchmark(args: argparse.Namespace, deadline: Deadline, smoke: Any, runtime: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], initial: Mapping[str, int], output: Path, model_load_seconds: float) -> Dict[str, Any]:
    all_cal = _load_cases(args.cal, "cal")
    runtime_identity = smoke._checkpoint_identity(runtime)
    _validate_runtime_identity(all_cal, runtime_identity, "cal")
    cal = all_cal[:2]
    preprocessor = smoke._preprocessor(runtime)
    objective = smoke._objective()
    snapshot = smoke._snapshot_weights(runtime["model"], groups)
    references = _reference_scores(smoke, runtime, cal, preprocessor, objective, snapshot, deadline, "cal")
    _write_reference_npz(output, "cal", cal, references)
    first_scores: Dict[str, np.ndarray] | None = None
    repetitions = 2
    records = []
    for repetition in range(repetitions):
        deadline.check()
        record = _score_map(smoke, runtime, groups, snapshot, initial, cal, preprocessor, objective, {}, "cal", output, deadline, artifact_suffix=f"_rep{repetition + 1:02d}")
        if repetition == 0:
            first_scores = {pool_id: np.asarray(value).copy() for pool_id, value in record["scores"].items()}
        records.append({"repetition": repetition + 1, "map_id": record["map_id"], "cache_hit": record["cache_hit"], "apply_seconds": record["apply_seconds"], "forward_seconds": record["forward_seconds"], "elapsed_seconds": record["elapsed_seconds"], "forward_count": record["forward_count"]})
        if repetition == 1:
            if first_scores is None:
                raise RuntimeError("benchmark did not retain first repetition scores")
            for pool_id in first_scores:
                if not np.array_equal(first_scores[pool_id], record["scores"][pool_id]):
                    raise RuntimeError(f"benchmark repeated score mismatch for {pool_id}")
    result = {
        "schema": "cem-update-ptq-benchmark-v1",
        "status": "complete",
        "benchmark_only": True,
        "model_load_seconds": float(model_load_seconds),
        "checkpoint_identity": runtime_identity,
        "hardware_provenance": "independent CCDS-TC1 SLURM V100 rerun; fresh candidate/reference pairing",
        "helper_source": "same-dir archived Wall helpers; independent CCDS SLURM rerun",
        "cal_pool_count": len(cal),
        "dev_loaded_for_schema_check": False,
        "repetitions": records,
        "initial": _public_map(groups, initial),
        "cost": _allocation_cost(groups, initial),
        "reference_check": "recomputed FP32 per pool; supplied reference allclose(rtol=0, atol=1e-6); restore exact",
        "reference_audit": _reference_audit(cal),
        "forward_counts": {
            "fp32_cal_reference": len(cal),
            "fp32_cal_restore_recheck": 1,
            "quantized_benchmark_pool_scores": int(sum(row["forward_count"] for row in records)),
            "total": int(len(cal) + 1 + sum(row["forward_count"] for row in records)),
        },
        "search_performed": False,
        "elapsed_seconds": deadline.elapsed,
        "unresolved": ["fake quantization uses ordinary FP32 operators; no native kernel claim"],
    }
    _atomic_json(output / "benchmark_summary.json", result)
    _atomic_json(output / "summary.json", {"schema": result["schema"], "status": result["status"], "benchmark_only": True, "model_load_seconds": result["model_load_seconds"], "checkpoint_identity": result["checkpoint_identity"], "cal_pool_count": result["cal_pool_count"], "search_performed": False, "repetitions": records, "initial_map_id": result["initial"]["map_id"], "forward_counts": result["forward_counts"], "cache_scope": "none; benchmark repetitions intentionally use fresh caches"})
    return result


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    deadline = Deadline(args.max_seconds)
    allocation = _require_allocation()
    load_started = time.monotonic()
    smoke = _load_smoke()
    runtime = smoke._runtime(args.root)
    model_load_seconds = time.monotonic() - load_started
    groups = _validate_groups(smoke, runtime)
    initial = _load_initial(args.initial, groups)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.benchmark_only:
        return _benchmark(args, deadline, smoke, runtime, groups, initial, output, model_load_seconds)

    initial_cost = _allocation_cost(groups, initial)
    cal = _load_cases(args.cal, "cal")
    dev = _load_cases(args.dev, "dev")
    _validate_split_disjoint(cal, dev)
    runtime_identity = smoke._checkpoint_identity(runtime)
    _validate_runtime_identity(cal, runtime_identity, "cal")
    _validate_runtime_identity(dev, runtime_identity, "dev")

    preprocessor = smoke._preprocessor(runtime)
    objective = smoke._objective()
    snapshot = smoke._snapshot_weights(runtime["model"], groups)
    references = _reference_scores(smoke, runtime, cal, preprocessor, objective, snapshot, deadline, "cal")
    _write_reference_npz(output, "cal", cal, references)
    trace_path = output / "search_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    search_results: Dict[str, Any] = {}
    try:
        for method_index, method in enumerate(METHODS):
            search_results[method] = _search_method(method, initial, groups, cal, references, smoke, runtime, snapshot, preprocessor, objective, cache, output, trace_path, deadline, args.rounds, method_index)
    except DeadlineExceeded as exc:
        _atomic_json(output / "summary.json", {"schema": "cem-update-ptq-search-v1", "status": "budget_exhausted", "error": str(exc), "methods_completed": list(search_results), "elapsed_seconds": deadline.elapsed})
        return {"status": "budget_exhausted", "methods_completed": list(search_results), "error": str(exc)}

    selected_by_method = {method: search_results[method]["selected"]["bits"] for method in METHODS}
    selected_maps: Dict[str, Dict[str, int]] = {"shared_start": dict(initial)}
    selected_maps.update({method: dict(bits) for method, bits in selected_by_method.items()})
    _atomic_json(output / "selected_maps.json", {name: _public_map(groups, bits) for name, bits in selected_maps.items()})
    dev_references = _reference_scores(smoke, runtime, dev, preprocessor, objective, snapshot, deadline, "dev")
    _write_reference_npz(output, "dev", dev, dev_references)
    dev_metrics: Dict[str, Any] = {}
    chain_reference_cache: Dict[str, Dict[str, Any]] = {}
    chain_rows: List[Mapping[str, Any]] = []
    chain_forward_counts: List[Mapping[str, int]] = []
    for name, bits in selected_maps.items():
        deadline.check()
        record = _score_map(smoke, runtime, groups, snapshot, bits, dev, preprocessor, objective, cache, "dev", output, deadline)
        metrics = _metrics_for_scores(dev, dev_references, record["scores"], 30, device=runtime.get("device"))
        dev_metrics[name] = {"map_id": record["map_id"], "metrics": metrics["aggregate"], "pool_count": len(dev), "cache_hit": record["cache_hit"], "forward_count": record["forward_count"]}
        if name == "shared_start":
            chain_name = "shared_start"
        else:
            chain_name = name
        chain = _chain_dev(chain_name, bits, groups, dev, dev_references, record["scores"], smoke, runtime, snapshot, preprocessor, objective, chain_reference_cache, deadline, output)
        chain_rows.extend(chain.get("pools", []))
        chain_forward_counts.append({
            "reference_chain_forward_count": int(chain.get("reference_chain_forward_count", 0)),
            "quantized_chain_forward_count": int(chain.get("quantized_chain_forward_count", 0)),
        })
        dev_metrics[name]["chain"] = {key: value for key, value in chain.items() if key != "pools"}
    _atomic_json(output / "dev_metrics.json", dev_metrics)
    with (output / "dev_chain.jsonl").open("w", encoding="utf-8") as stream:
        for row in chain_rows:
            stream.write(json.dumps(row, default=_json_default, separators=(",", ":")) + "\n")

    unique_cal_maps = {key[1] for key in cache if key[0] == "cal"}
    unique_dev_maps = {key[1] for key in cache if key[0] == "dev"}
    map_forward_count = sum(int(value.get("forward_count", 0)) for value in cache.values())
    reference_forward_count = len(cal) + 1 + len(dev) + 1
    reference_chain_forward_count = sum(int(row["reference_chain_forward_count"]) for row in chain_forward_counts)
    quantized_chain_forward_count = sum(int(row["quantized_chain_forward_count"]) for row in chain_forward_counts)
    total_forward_count = reference_forward_count + map_forward_count + reference_chain_forward_count + quantized_chain_forward_count
    summary = {
        "schema": "cem-update-ptq-search-v1",
        "status": "complete",
        "allocation": allocation,
        "benchmark_only": False,
        "model_load_seconds": float(model_load_seconds),
        "checkpoint_identity": runtime_identity,
        "hardware_provenance": "independent CCDS-TC1 SLURM V100 rerun; fresh candidate/reference pairing",
        "helper_source": "same-dir archived Wall helpers; independent CCDS SLURM rerun",
        "reference_audit": {"cal": _reference_audit(cal), "dev": _reference_audit(dev)},
        "cal_pool_count": len(cal),
        "dev_pool_count": len(dev),
        "rounds_requested": int(args.rounds),
        "max_seconds": float(args.max_seconds),
        "shared_start": _public_map(groups, initial),
        "initial_cost": initial_cost,
        "methods": {
            method: {
                "selected": search_results[method]["selected"],
                "objective_name": search_results[method]["objective_name"],
                "selected_objective": search_results[method]["selected_objective"],
                "selected_metrics": search_results[method]["selected_metrics"],
                "config_evaluations": search_results[method]["config_evaluations"],
                "rounds_completed": search_results[method]["rounds_completed"],
                "dev": dev_metrics.get(method),
            }
            for method in METHODS
        },
        "cache": {"scope": "process-local; no cross-run restore", "unique_cal_maps": len(unique_cal_maps), "unique_dev_maps": len(unique_dev_maps), "map_forward_count": int(map_forward_count)},
        "forward_counts": {
            "fp32_cal_reference": len(cal),
            "fp32_cal_restore_recheck": 1,
            "fp32_dev_reference": len(dev),
            "fp32_dev_restore_recheck": 1,
            "quantized_map_pool_scores": int(map_forward_count),
            "fp32_chain_second_step": int(reference_chain_forward_count),
            "quantized_chain_second_step": int(quantized_chain_forward_count),
            "total": int(total_forward_count),
        },
        "artifacts": {"trace": str(trace_path), "selected_maps": str(output / "selected_maps.json"), "dev_metrics": str(output / "dev_metrics.json"), "dev_chain": str(output / "dev_chain.jsonl"), "raw_scores": "cal_scores/ and dev_scores/ NPZ files"},
        "tie_rule": "stable candidate-index argsort; stable lexicographic map_id among objective ties",
        "calibration_tolerance": CAL_TOL,
        "std_correction": 1,
        "normalization_scale": 1.0,
        "unresolved": ["fake quantization executes ordinary FP32 operators and cannot establish native low-bit deployment gains", "DEV update and replay fidelity do not establish closed-loop success"],
        "elapsed_seconds": deadline.elapsed,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def _self_test() -> None:
    candidates = np.arange(60, dtype=np.float64).reshape(30, 2)
    reference = np.arange(30, dtype=np.float64)
    metrics = _pool_metrics(reference, reference, candidates, 30, 1)
    assert metrics["mu"] == 0.0 and metrics["sigma"] == 0.0 and metrics["update"] == 0.0
    assert _pairwise_rank([0.0, 1.0, 2.0], [0.0, 1.0, 1.0]) == 1.0 / 3.0
    assert _pairwise_rank([0.0, 0.0, 1.0], [0.0, 0.0, 2.0]) == 0.0
    assert _score_nmse([0.0, 1.0], [0.0, 1.0]) == 0.0
    groups = [{"group_id": f"encoder.base_model.blocks.{i}", "family": "encoder", "index": i, "numel": 10, "scale_count": 2, "logical_weight_bytes_W4": 9, "logical_weight_bytes_W8": 14} for i in range(12)]
    groups += [{"group_id": f"predictor.transformer.layers.{i}", "family": "predictor", "index": i, "numel": 10, "scale_count": 2, "logical_weight_bytes_W4": 9, "logical_weight_bytes_W8": 14} for i in range(6)]
    bits = {group["group_id"]: 4 for group in groups}
    for group in groups[:3] + groups[12:14]:
        bits[group["group_id"]] = 8
    assert len(_neighbors(groups, bits)) == 35
    assert _map_id(bits) == _map_id(dict(reversed(list(bits.items()))))
    print(json.dumps({"status": "complete", "checks": ["stable ties", "std correction=1", "35 neighbors", "map hash stability"]}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=False, help="DINO-WM Wall runtime root")
    parser.add_argument("--cal", type=Path, required=False, help="new CAL workload.pkl")
    parser.add_argument("--dev", type=Path, required=False, help="new DEV workload.pkl")
    parser.add_argument("--initial", type=Path, required=False, help="frozen ScoreError map JSON (logical map only)")
    parser.add_argument("--output", type=Path, required=False, help="compute-side output directory")
    parser.add_argument("--rounds", type=int, default=2, help="maximum best-improvement rounds (0-2; default: 2)")
    parser.add_argument("--max-seconds", type=float, required=False, default=600.0, help="hard wall-clock budget")
    parser.add_argument("--benchmark-only", action="store_true", help="benchmark two CAL pools and never search")
    parser.add_argument("--self-test", action="store_true", help="run pure-CPU numerical checks without SLURM/model loading")
    args = parser.parse_args()
    if args.self_test:
        return args
    for name in ("root", "cal", "dev", "initial", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    if args.rounds < 0 or args.rounds > 2:
        parser.error("--rounds must be between 0 and 2")
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    try:
        result = _run(args)
    except DeadlineExceeded as exc:
        result = {"status": "budget_exhausted", "error": str(exc)}
        _atomic_json(args.output.resolve() / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("status", "benchmark_only", "cal_pool_count", "dev_pool_count") if key in result}, default=_json_default, indent=2), flush=True)


if __name__ == "__main__":
    main()




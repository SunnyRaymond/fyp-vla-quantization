"""Complete joint-model fidelity on frozen development candidate pools.

The runner consumes the FP32 development ``workload.pkl`` emitted by
``screen_runner.py`` and five frozen calibration allocation JSON files.  It
does not fit or alter allocations from development data and never calls the
real environment.  Every method scores the same candidate pools with the
same raw observations; weights are restored to the original FP32 snapshot
before each method is applied.

This is numerical fake quantization: the shared smoke quantizer dequantizes
weights and ordinary FP32 operators execute the model.  The output therefore
does not support native low-bit kernel, memory, or speed claims.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
try:
    import smoke_runner as smoke
except ModuleNotFoundError:
    smoke = _load_local_module("rankcal_wall_smoke_runner", _HERE / "smoke_runner.py")
try:
    import probe_runner as probe
except ModuleNotFoundError:
    probe = _load_local_module("rankcal_wall_probe_runner", _HERE / "probe_runner.py")


EXPECTED_ALLOCATION_METHODS = ("RankCal", "LocalMSE", "ScoreError", "Random1", "Random2")
SCREEN_METHODS = ("FP32", "all_W4", "all_W8") + EXPECTED_ALLOCATION_METHODS
EXPECTED_DEV_EPISODES = 8
EXPECTED_POOLS_PER_EPISODE = 4
EXPECTED_CANDIDATES = 300
W8_QUOTA = {"encoder": 3, "predictor": 2}
JOINT_SCHEMA = "rankcal-wall-joint-fidelity-v1"
POOL_SCHEMA = "rankcal-wall-joint-fidelity-pool-v1"
ALLOCATION_MANIFEST_SCHEMA = "rankcal-wall-joint-fidelity-allocations-v1"


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
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=_json_default) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "pool"


def _resolve_workload(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    return probe._resolve_workload(path, "dev")


def _load_workload(path: Path) -> List[Dict[str, Any]]:
    workload_path = _resolve_workload(path)
    payload = probe._load_payload(workload_path)
    pools = probe._flatten_pools(payload, "dev")
    probe._validate_pool_budget(pools, EXPECTED_POOLS_PER_EPISODE)
    episode_ids = {str(pool["episode_id"]) for pool in pools}
    if len(episode_ids) != EXPECTED_DEV_EPISODES:
        raise ValueError(
            f"development workload must contain exactly {EXPECTED_DEV_EPISODES} episodes; got {len(episode_ids)}"
        )
    pool_ids = [str(pool["pool_id"]) for pool in pools]
    if len(pool_ids) != len(set(pool_ids)):
        raise ValueError("development workload contains duplicate pool IDs")
    for pool in pools:
        candidates = np.asarray(pool["candidates"], dtype=np.float32)
        if candidates.shape[0] != EXPECTED_CANDIDATES:
            raise ValueError(f"pool {pool['pool_id']!r} must contain exactly 300 candidates")
        if int(pool["mpc_point"]) not in (0, 1) or int(pool["cem_iteration"]) not in (1, 5):
            raise ValueError(f"pool {pool['pool_id']!r} is outside the frozen MPC/CEM pool design")
        pool["candidates"] = candidates
    return [dict(pool) for pool in pools]


def _direct_method_payload(path: Path, method: str) -> Mapping[str, Any] | None:
    candidates = [path / f"{method}.json", path / f"{method}.pkl"]
    for candidate in candidates:
        if candidate.is_file():
            payload = probe._load_payload(candidate)
            if not isinstance(payload, Mapping):
                raise ValueError(f"allocation file must contain an object: {candidate}")
            return payload
    return None


def _central_method_payload(payload: Mapping[str, Any], method: str) -> Mapping[str, Any] | None:
    methods = payload.get("methods")
    if isinstance(methods, Mapping) and method in methods and isinstance(methods[method], Mapping):
        return methods[method]
    allocations = payload.get("allocations")
    if isinstance(allocations, Sequence) and not isinstance(allocations, (str, bytes)):
        for item in allocations:
            if isinstance(item, Mapping) and str(item.get("method", item.get("name", ""))) == method:
                return item
    if str(payload.get("method", payload.get("name", ""))) == method:
        return payload
    return None


def _bits_mapping(payload: Mapping[str, Any], group_ids: Sequence[str]) -> Dict[str, int]:
    raw = payload.get("allocation", payload.get("bits"))
    if not isinstance(raw, Mapping):
        raise ValueError("allocation payload needs an allocation/bits mapping")
    known = set(group_ids)
    result: Dict[str, int] = {}
    for key, value in raw.items():
        key = str(key)
        if key not in known:
            raise ValueError(f"allocation contains unknown group {key!r}")
        if isinstance(value, Mapping):
            value = value.get("bits")
        bits = int(value)
        if bits not in (4, 8):
            raise ValueError(f"allocation bit width must be 4 or 8: {key}={bits}")
        result[key] = bits
    missing = sorted(known - set(result))
    if missing or set(result) != known:
        raise ValueError(f"allocation must specify every runtime group; missing={missing}")
    return result


def _allocation_signature(groups: Sequence[Mapping[str, Any]], bits: Mapping[str, int]) -> Dict[str, Any]:
    by_family: Dict[str, int] = {}
    for family in W8_QUOTA:
        by_family[family] = sum(
            int(bits[str(group["group_id"])]) == 8
            for group in groups
            if str(group["family"]) == family
        )
    if by_family != W8_QUOTA:
        raise ValueError(f"allocation W8 quotas must be {W8_QUOTA}, got {by_family}")
    cost = probe._allocation_cost(groups, bits)
    return {"w8_counts": by_family, "cost": cost}


def _load_allocations(allocation_dir: Path, groups: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    allocation_dir = allocation_dir.resolve()
    if not allocation_dir.is_dir():
        raise FileNotFoundError(f"allocation directory not found: {allocation_dir}")
    group_ids = [str(group["group_id"]) for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("runtime group IDs are not unique")
    central_payload: Mapping[str, Any] | None = None
    for filename in ("allocations.json", "allocations.pkl"):
        candidate = allocation_dir / filename
        if candidate.is_file():
            loaded = probe._load_payload(candidate)
            if not isinstance(loaded, Mapping):
                raise ValueError(f"allocation bundle must contain an object: {candidate}")
            central_payload = loaded
            break

    result: Dict[str, Dict[str, Any]] = {}
    fingerprints = set()
    for method in EXPECTED_ALLOCATION_METHODS:
        payload = _direct_method_payload(allocation_dir, method)
        source = None
        if payload is not None:
            source = next(
                candidate for candidate in (allocation_dir / f"{method}.json", allocation_dir / f"{method}.pkl")
                if candidate.is_file()
            )
        elif central_payload is not None:
            payload = _central_method_payload(central_payload, method)
            source = allocation_dir / "allocations.json"
        if payload is None:
            raise FileNotFoundError(f"missing frozen allocation for {method} under {allocation_dir}")
        if payload.get("calibration_only") is False:
            raise ValueError(f"allocation {method} is not marked calibration-only")
        fit_split = str(payload.get("calibration_split", payload.get("fit_split", "cal"))).lower()
        if fit_split not in {"cal", "calibration"}:
            raise ValueError(f"allocation {method} was not fit from calibration: {fit_split!r}")
        bits = _bits_mapping(payload, group_ids)
        signature = _allocation_signature(groups, bits)
        fingerprint = tuple(sorted(bits.items()))
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate allocation map for {method}")
        fingerprints.add(fingerprint)
        result[method] = {
            "method": method,
            "bits": bits,
            "selected_sites": sorted(key for key, value in bits.items() if value == 8),
            "source": str(source.resolve()),
            "signal": payload.get("signal"),
            "calibration_split": payload.get("calibration_split", "cal"),
            **signature,
        }
    signatures = {
        (entry["cost"]["logical_weight_bytes"], entry["cost"]["weight_bits"])
        for entry in result.values()
    }
    if len(signatures) != 1:
        raise ValueError("the five frozen allocation maps do not have the same logical cost")
    return result


def _score_array(runtime: Mapping[str, Any], preprocessor: Any, objective_fn: Any, pool: Mapping[str, Any]) -> np.ndarray:
    scores = smoke._score_pool(
        runtime["model"],
        preprocessor,
        objective_fn,
        {"obs_0": pool["obs_0"], "obs_g": pool["obs_g"]},
        pool["candidates"],
    )
    torch = runtime["torch"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    array = scores.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    if array.shape != (EXPECTED_CANDIDATES,) or not np.isfinite(array).all():
        raise FloatingPointError(f"non-finite or malformed score vector for pool {pool['pool_id']!r}")
    return array


def _pool_metrics(reference: np.ndarray, quantized: np.ndarray, candidates: np.ndarray) -> Dict[str, Any]:
    reference_elite = np.argsort(reference, kind="stable")[: smoke.TOPK]
    quantized_elite = np.argsort(quantized, kind="stable")[: smoke.TOPK]
    overlap_count = int(np.intersect1d(reference_elite, quantized_elite).size)
    reference_mean = candidates[reference_elite].mean(axis=0)
    quantized_mean = candidates[quantized_elite].mean(axis=0)
    return {
        "e2_pairwise_disagreement": float(probe.pairwise_order_disagreement(reference, quantized)),
        "stable30_elite_overlap": overlap_count,
        "stable30_elite_overlap_fraction": overlap_count / float(smoke.TOPK),
        "elite_mean_normalized_action_mse": float(np.mean((quantized_mean - reference_mean) ** 2)),
        "score_nmse": float(probe.planner_score_nmse(reference, quantized)),
        "reference_elite_indices": reference_elite,
        "quantized_elite_indices": quantized_elite,
    }


def _configure_method(runtime: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any], method: str, bits: Mapping[str, int] | None) -> Dict[str, Any]:
    smoke._restore_weights(runtime["model"], snapshot)
    quantization: List[Dict[str, Any]] = []
    if method == "FP32":
        return {"method": method, "execution": "fp32_reference", "quantization": quantization}
    if method == "all_W4":
        mapping = {str(group["group_id"]): 4 for group in groups}
    elif method == "all_W8":
        mapping = {str(group["group_id"]): 8 for group in groups}
    else:
        if bits is None:
            raise ValueError(f"missing allocation bits for {method}")
        mapping = {str(key): int(value) for key, value in bits.items()}
    for group in groups:
        quantization.append(smoke._quantize_group(runtime["model"], group, mapping[str(group["group_id"])]))
    return {
        "method": method,
        "execution": "emulation_only",
        "bits": mapping,
        "quantization": quantization,
    }


def _aggregate_episode_metrics(pool_metrics: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in pool_metrics:
        by_episode[str(row["episode_id"])].append(row)
    metric_names = (
        "e2_pairwise_disagreement",
        "stable30_elite_overlap",
        "stable30_elite_overlap_fraction",
        "elite_mean_normalized_action_mse",
        "score_nmse",
    )
    episode_rows: List[Dict[str, Any]] = []
    for episode_id, rows in sorted(by_episode.items()):
        methods: Dict[str, Dict[str, float]] = {}
        for method in SCREEN_METHODS:
            method_rows = [row["methods"][method] for row in rows]
            methods[method] = {
                name: float(np.mean([float(item[name]) for item in method_rows]))
                for name in metric_names
            }
        episode_rows.append({"episode_id": episode_id, "pool_count": len(rows), "methods": methods})
    overall: Dict[str, Dict[str, float]] = {}
    for method in SCREEN_METHODS:
        overall[method] = {
            name: float(np.mean([row["methods"][method][name] for row in episode_rows]))
            for name in metric_names
        }
    return episode_rows, overall


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.monotonic()
    runtime = smoke._runtime(args.root.resolve(), device=args.device)
    groups = probe._validate_groups(probe._runtime_groups(smoke, runtime))
    allocations = _load_allocations(args.allocations_dir, groups)
    pools = _load_workload(args.workload)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pool_output = output / "pools"

    preprocessor = smoke._preprocessor(runtime)
    objective_fn = smoke._objective()
    model = runtime["model"]
    snapshot = smoke._snapshot_weights(model, groups)
    references: Dict[str, np.ndarray] = {}
    try:
        # Recompute every FP32 reference from this exact runtime.  Supplied
        # workload references are checked evidence, never substituted scores.
        for pool in pools:
            smoke._restore_weights(model, snapshot)
            reference = _score_array(runtime, preprocessor, objective_fn, pool)
            supplied = pool.get("reference_scores")
            if supplied is not None:
                supplied_array = np.asarray(supplied, dtype=np.float64).reshape(-1)
                if supplied_array.shape != reference.shape or not np.allclose(supplied_array, reference, rtol=0.0, atol=1e-6):
                    max_abs = float(np.max(np.abs(supplied_array - reference))) if supplied_array.shape == reference.shape else float("inf")
                    raise ValueError(f"FP32 reference mismatch for pool {pool['pool_id']!r}: max_abs={max_abs:g}")
            pool_id = str(pool["pool_id"])
            references[pool_id] = reference

        all_pool_scores: Dict[str, Dict[str, np.ndarray]] = {
            str(pool["pool_id"]): {"FP32": references[str(pool["pool_id"])]} for pool in pools
        }
        all_pool_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {
            str(pool["pool_id"]): {} for pool in pools
        }
        mode_metadata: Dict[str, Dict[str, Any]] = {}
        method_bits: Dict[str, Mapping[str, int] | None] = {
            "FP32": None,
            "all_W4": None,
            "all_W8": None,
            **{method: value["bits"] for method, value in allocations.items()},
        }
        for method in SCREEN_METHODS:
            metadata = _configure_method(runtime, groups, snapshot, method, method_bits[method])
            mode_metadata[method] = metadata
            for pool in pools:
                pool_id = str(pool["pool_id"])
                score = references[pool_id] if method == "FP32" else _score_array(runtime, preprocessor, objective_fn, pool)
                all_pool_scores[pool_id][method] = score
                all_pool_metrics[pool_id]["methods"] = all_pool_metrics[pool_id].get("methods", {})
                all_pool_metrics[pool_id]["methods"][method] = _pool_metrics(
                    references[pool_id], score, np.asarray(pool["candidates"], dtype=np.float64)
                )
            # No quantized values carry into the next complete joint model.
            smoke._restore_weights(model, snapshot)

        first_pool = pools[0]
        smoke._restore_weights(model, snapshot)
        restored = _score_array(runtime, preprocessor, objective_fn, first_pool)
        if not np.array_equal(restored, references[str(first_pool["pool_id"])]):
            raise RuntimeError(f"FP32 restore score mismatch for pool {first_pool['pool_id']!r}")

        pool_rows: List[Dict[str, Any]] = []
        for pool in pools:
            pool_id = str(pool["pool_id"])
            scores = all_pool_scores[pool_id]
            metrics = all_pool_metrics[pool_id]["methods"]
            score_arrays = {method: np.asarray(scores[method], dtype=np.float32) for method in SCREEN_METHODS}
            score_arrays["reference_scores"] = score_arrays["FP32"]
            for method in SCREEN_METHODS:
                score_arrays[f"{method}_elite_indices"] = np.asarray(metrics[method]["quantized_elite_indices"], dtype=np.int64)
            score_path = pool_output / f"pool_{_safe_id(pool_id)}.scores.npz"
            _write_npz(score_path, score_arrays)
            pool_rows.append({
                "schema": POOL_SCHEMA,
                "pool_id": pool_id,
                "episode_id": str(pool["episode_id"]),
                "mpc_point": int(pool["mpc_point"]),
                "cem_iteration": int(pool["cem_iteration"]),
                "candidate_count": EXPECTED_CANDIDATES,
                "score_file": str(score_path.resolve()),
                "methods": {
                    method: {
                        key: value
                        for key, value in metrics[method].items()
                        if key not in {"reference_elite_indices", "quantized_elite_indices"}
                    }
                    for method in SCREEN_METHODS
                },
            })
        episode_rows, overall = _aggregate_episode_metrics(pool_rows)

        allocation_manifest = {
            "schema": ALLOCATION_MANIFEST_SCHEMA,
            "source_directory": str(args.allocations_dir.resolve()),
            "methods": allocations,
            "common_cost": next(iter(allocations.values()))["cost"],
            "w8_quota": W8_QUOTA,
            "fit_split": "calibration",
            "development_maps_refit": False,
        }
        _write_json(output / "allocation_manifest.json", allocation_manifest)
        _write_jsonl(output / "pool_metrics.jsonl", pool_rows)
        _write_json(output / "episode_metrics.json", {
            "schema": f"{JOINT_SCHEMA}-episode-summary-v1",
            "split": "development",
            "episode_count": len(episode_rows),
            "aggregation": "pool mean within episode, then equal-weight episode mean",
            "episodes": episode_rows,
            "overall_equal_episode_weight": overall,
        })
        summary = {
            "schema": JOINT_SCHEMA,
            "status": "complete",
            "split": "development",
            "workload": str(_resolve_workload(args.workload)),
            "pool_count": len(pools),
            "episode_count": len(episode_rows),
            "methods": list(SCREEN_METHODS),
            "planner": {
                "candidate_count": EXPECTED_CANDIDATES,
                "elite_count": smoke.TOPK,
                "stable_argsort": True,
                "reference": "recomputed FP32 score vectors",
            },
            "runtime": smoke._checkpoint_identity(runtime),
            "mode_metadata": mode_metadata,
            "allocation_manifest": str((output / "allocation_manifest.json").resolve()),
            "pool_metrics": str((output / "pool_metrics.jsonl").resolve()),
            "episode_metrics": str((output / "episode_metrics.json").resolve()),
            "overall_equal_episode_weight": overall,
            "fp32_restore_exact_checked": True,
            "execution": "emulation_only",
            "elapsed_seconds": time.monotonic() - started,
            "unresolved": [
                "fake quantization executes FP32 operators; native low-bit kernel/memory/speed claims are out of scope",
                "fidelity on frozen development pools does not establish closed-loop success or calibration generalization",
            ],
        }
        _write_json(output / "summary.json", summary)
        print(json.dumps({"status": "complete", "split": "development", "pool_count": len(pools), "episode_count": len(episode_rows), "methods": list(SCREEN_METHODS)}, indent=2), flush=True)
        return summary
    except Exception as exc:
        _write_json(args.output.resolve() / "failure.json", {
            "schema": JOINT_SCHEMA,
            "status": "failed",
            "error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise
    finally:
        smoke._restore_weights(model, snapshot)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM Wall runtime root")
    parser.add_argument("--workload", type=Path, required=True, help="FP32 development workload.pkl (or containing directory)")
    parser.add_argument("--allocations-dir", type=Path, required=True, help="directory containing RankCal/LocalMSE/ScoreError/Random1/Random2 JSON files")
    parser.add_argument("--output", type=Path, required=True, help="fidelity artifact directory")
    parser.add_argument("--device", default=None, help="torch device, normally cuda:0")
    return parser.parse_args()


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()

"""Calibration probes and fixed-budget allocations for the Wall fast screen.

The runner is deliberately independent from the closed-loop screen runner.  It
consumes a small ``workload.pkl`` whose ``cases`` are calibration/development
pools and reuses the numerical quantizer and scoring helpers from
``smoke_runner.py``.  A pool is a fixed candidate set at one episode/MPC/CEM
point; all 36 site/bit probes use that same pool.

The probe artifact has two useful layers:

* ``records`` contains one row per pool, site and bit.
* ``site_metrics`` first averages pools within each episode and then averages
  episodes with equal weight.  Allocation is performed only from this layer.

Rank disagreement follows the E2 definition exactly.  For every unordered
candidate pair, it is one when the two signs differ.  A tie has sign zero; a
tie on only one side therefore counts as disagreement, while two exact ties
agree.  Local output NMSE uses FP32 teacher inputs: encoder current and goal
branches are equally weighted, and predictor rollout calls are equally
weighted.  No quantized activation is fed into a later block for this signal.

The execution is numerical fake quantization: integer values and scales are
recorded by the shared helper, then dequantized weights run in ordinary FP32
operators.  This file does not submit jobs or claim native INT4/INT8 speed or
memory savings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


EXPECTED_ENCODER_GROUPS = 12
EXPECTED_PREDICTOR_GROUPS = 6
EXPECTED_GROUPS = EXPECTED_ENCODER_GROUPS + EXPECTED_PREDICTOR_GROUPS
BITS = (4, 8)
W8_QUOTA = {"encoder": 3, "predictor": 2}
MAX_POOLS = 32
RANDOM_SEEDS = (71001, 71002)
NMSE_EPS = 1e-12
PROBE_SCHEMA = "rankcal-wall-calibration-probes-v1"
ALLOCATION_SCHEMA = "rankcal-wall-allocations-v1"
SPLIT_ALIASES = {"cal": "cal", "calibration": "cal", "dev": "dev", "development": "dev"}
SCREEN_WORKLOAD_SCHEMA = "rankcal-wall-screen-workload-v1"
SCREEN_EPISODE_COUNT = 8
SCREEN_CANDIDATE_SHAPE = (300, 5, 10)
SCREEN_PAIR_COUNT = SCREEN_CANDIDATE_SHAPE[0] * (SCREEN_CANDIDATE_SHAPE[0] - 1) // 2


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
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


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _load_payload(path: Path) -> Any:
    path = path.resolve()
    if path.is_dir():
        candidates = [path / "probes.pkl", path / "calibration_probes.pkl", path / "workload.pkl"]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"no workload/probe artifact under {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        return pickle.load(stream)


def _resolve_workload(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = [
        path / f"{split}_workload.pkl",
        path / f"{split}.pkl",
        path / {"cal": "calibration", "dev": "development"}[split] / "workload.pkl",
        path / "workload.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {split} workload under {path}")


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _reference_scores(item: Mapping[str, Any]) -> Any:
    direct = _first(item, "reference_scores", "fp32_scores", "scores_fp32", "reference_score")
    if direct is not None:
        return direct
    scores = item.get("scores")
    if isinstance(scores, Mapping):
        return _first(scores, "FP32", "fp32", "reference", "REF")
    return None


def _pool_fields(item: Mapping[str, Any]) -> bool:
    return _first(item, "candidates", "candidate_actions", "actions") is not None and (
        _first(item, "obs_0", "obs0", "current_obs") is not None
        or _first(item, "obs_g", "obsg", "goal_obs") is not None
    )


def _flatten_pools(payload: Any, split: str) -> List[Dict[str, Any]]:
    """Normalize smoke-style cases and screen-runner pool containers."""
    if not isinstance(payload, Mapping):
        raise TypeError("workload must be a mapping")
    declared_split = payload.get("split")
    declared_canonical = SPLIT_ALIASES.get(str(declared_split).lower()) if declared_split is not None else None
    if declared_split is not None and declared_canonical != split:
        raise ValueError(f"workload split={declared_split!r} does not match --split {split!r}")
    roots = payload.get("pools")
    if roots is None:
        roots = payload.get("cases")
    if roots is None:
        roots = payload.get("records")
    if roots is None:
        roots = [payload] if _pool_fields(payload) else None
    if roots is None or not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)):
        raise ValueError("workload must contain a sequence under pools, cases, or records")

    flattened: List[Dict[str, Any]] = []

    def visit(node: Any, episode_hint: Any = None, inherited_case_id: Any = None) -> None:
        if not isinstance(node, Mapping):
            raise TypeError(f"pool entry must be a mapping, got {type(node)!r}")
        if _pool_fields(node):
            obs_0 = _first(node, "obs_0", "obs0", "current_obs")
            obs_g = _first(node, "obs_g", "obsg", "goal_obs")
            candidates = _first(node, "candidates", "candidate_actions", "actions")
            episode_id = _first(node, "episode_id", "episode", default=episode_hint)
            if episode_id is None:
                episode_id = _first(node, "case_id", default=inherited_case_id)
            pool_id = _first(node, "pool_id", "record_id", default=None)
            if pool_id is None:
                pool_id = _first(node, "case_id", default=None)
            if pool_id is None:
                pool_id = f"episode-{episode_id}-pool-{len(flattened):04d}"
            try:
                candidate_array = np.asarray(candidates, dtype=np.float32)
            except Exception as exc:
                raise TypeError(f"cannot convert candidates for pool {pool_id!r}") from exc
            if candidate_array.ndim < 2 or candidate_array.shape[0] < 2:
                raise ValueError(f"pool {pool_id!r} candidates must have shape (N,...), N>=2")
            reference = _reference_scores(node)
            reference_array = None if reference is None else np.asarray(reference, dtype=np.float32).reshape(-1)
            reference_elite = _first(node, "reference_elite_indices", "fp32_elite_indices")
            if reference_elite is not None:
                reference_elite = np.asarray(reference_elite, dtype=np.int64).reshape(-1)
            if reference_array is not None and reference_array.shape[0] != candidate_array.shape[0]:
                raise ValueError(
                    f"pool {pool_id!r}: reference score count {reference_array.shape[0]} "
                    f"does not match candidates {candidate_array.shape[0]}"
                )
            flattened.append(
                {
                    "pool_id": str(pool_id),
                    "episode_id": str(episode_id if episode_id is not None else pool_id),
                    "mpc_point": int(_first(node, "mpc_point", "mpc_round", "round", default=0)),
                    "cem_iteration": int(_first(node, "cem_iteration", "iteration", default=0)),
                    "obs_0": obs_0,
                    "obs_g": obs_g,
                    "candidates": candidate_array,
                    "reference_scores": reference_array,
                    "reference_elite_indices": reference_elite,
                    "source": {
                        key: node[key]
                        for key in ("env_seed", "cem_seed", "layout", "dataset_index")
                        if key in node
                    },
                }
            )
            return
        nested = node.get("pools")
        if nested is None:
            nested = node.get("cases")
        if nested is None:
            nested = node.get("records")
        if nested is None or not isinstance(nested, Sequence) or isinstance(nested, (str, bytes)):
            raise ValueError("each case must contain a pool or nested pools/cases")
        next_episode = _first(node, "episode_id", "episode", default=episode_hint)
        case_id = _first(node, "case_id", default=inherited_case_id)
        for child in nested:
            visit(child, episode_hint=next_episode if next_episode is not None else case_id, inherited_case_id=case_id)

    for root in roots:
        visit(root)
    if not flattened:
        raise ValueError("workload contains no scoring pools")
    return flattened


def _validate_pool_budget(pools: Sequence[Mapping[str, Any]], max_pools: int) -> None:
    by_episode: MutableMapping[str, List[str]] = defaultdict(list)
    for pool in pools:
        by_episode[str(pool["episode_id"])].append(str(pool["pool_id"]))
    too_many = {episode: ids for episode, ids in by_episode.items() if len(ids) > max_pools}
    if too_many:
        detail = ", ".join(f"{episode}: {len(ids)}" for episode, ids in sorted(too_many.items()))
        raise ValueError(f"episode exceeds pool budget {max_pools}: {detail}")


def pairwise_order_disagreement(reference: Sequence[float], quantized: Sequence[float]) -> float:
    """Return E2 mismatch rate over all unordered pairs, retaining exact ties."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    quant = np.asarray(quantized, dtype=np.float64).reshape(-1)
    if ref.shape != quant.shape or ref.ndim != 1 or ref.size < 2:
        raise ValueError("reference and quantized scores must be equal one-dimensional arrays with N>=2")
    i, j = np.triu_indices(ref.size, k=1)
    ref_sign = np.sign(ref[i] - ref[j])
    quant_sign = np.sign(quant[i] - quant[j])
    return float(np.mean(ref_sign != quant_sign))


def planner_score_nmse(reference: Sequence[float], quantized: Sequence[float], eps: float = NMSE_EPS) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    quant = np.asarray(quantized, dtype=np.float64).reshape(-1)
    if ref.shape != quant.shape or ref.ndim != 1:
        raise ValueError("reference and quantized scores must have the same shape")
    return float(np.mean((quant - ref) ** 2) / (np.mean(ref ** 2) + eps))


def _self_test_metrics() -> None:
    # Three pairs: both tied -> 0, one-side tie -> 1, reversed ordering -> 1.
    reference = np.asarray([0.0, 1.0, 2.0])
    tied = np.asarray([0.0, 1.0, 1.0])
    assert pairwise_order_disagreement(reference, tied) == 1.0 / 3.0
    exact_ties_reference = np.asarray([0.0, 0.0, 1.0])
    exact_ties_quantized = np.asarray([0.0, 0.0, 2.0])
    assert pairwise_order_disagreement(exact_ties_reference, exact_ties_quantized) == 0.0
    reversed_pair = np.asarray([2.0, 1.0, 0.0])
    assert pairwise_order_disagreement(reference, reversed_pair) == 1.0


def _load_smoke_runner() -> Any:
    path = Path(__file__).with_name("smoke_runner.py")
    spec = importlib.util.spec_from_file_location("rankcal_wall_smoke_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load smoke helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_screen_runner() -> Any:
    """Load the screen adapter only for its single source of protocol truth."""
    path = Path(__file__).with_name("screen_runner.py")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("rankcal_wall_screen_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load screen protocol from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_split(value: Any) -> str | None:
    return SPLIT_ALIASES.get(str(value).lower()) if value is not None else None


def _stable_runtime_identity(runtime: Mapping[str, Any], smoke: Any) -> Dict[str, Any]:
    identity = smoke._checkpoint_identity(runtime)
    keys = (
        "directory", "checkpoint_file", "recorded_epoch", "checkpoint_size_bytes",
        "source_commit", "dinov2_source_commit", "dtype", "decoder", "execution", "torch",
    )
    return {key: identity.get(key) for key in keys}


def _hash_array(hasher: Any, value: Any) -> None:
    array = np.asarray(value)
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(repr(tuple(array.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())


def _pool_fingerprint(pool: Mapping[str, Any]) -> str:
    """Fingerprint pool identity and all data that affects its probes."""
    hasher = hashlib.blake2b(digest_size=20)
    metadata = {
        "pool_id": str(pool["pool_id"]),
        "episode_id": str(pool["episode_id"]),
        "mpc_point": int(pool["mpc_point"]),
        "cem_iteration": int(pool["cem_iteration"]),
    }
    hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    _hash_array(hasher, pool["candidates"])
    for observation_name in ("obs_0", "obs_g"):
        observation = pool[observation_name]
        if not isinstance(observation, Mapping):
            raise TypeError(f"{observation_name} must be a mapping for pool fingerprint")
        for field in sorted(observation):
            hasher.update(str(field).encode("utf-8"))
            _hash_array(hasher, observation[field])
    reference = pool.get("reference_scores")
    if reference is None:
        hasher.update(b"reference:none")
    else:
        hasher.update(b"reference:")
        _hash_array(hasher, reference)
    elite = pool.get("reference_elite_indices")
    if elite is not None:
        hasher.update(b"elite:")
        _hash_array(hasher, elite)
    return hasher.hexdigest()


def _validate_screen_workload(payload: Mapping[str, Any], split: str, pools: Sequence[Mapping[str, Any]]) -> None:
    """Reject ad-hoc/smoke inputs before any GPU runtime is occupied."""
    if payload.get("schema") != SCREEN_WORKLOAD_SCHEMA:
        raise ValueError(
            f"probe fitting requires {SCREEN_WORKLOAD_SCHEMA}; got {payload.get('schema')!r}"
        )
    declared_split = _canonical_split(payload.get("split"))
    screen_split = _canonical_split(payload.get("screen_split"))
    if declared_split != split or screen_split != split:
        raise ValueError(
            f"screen workload split mismatch: split={payload.get('split')!r}, "
            f"screen_split={payload.get('screen_split')!r}, expected {split!r}"
        )
    if payload.get("reference_score_field") != "reference_scores":
        raise ValueError("screen workload must declare reference_score_field='reference_scores'")
    expected_protocol = _load_screen_runner()._protocol_config()
    if payload.get("planner") != expected_protocol:
        raise ValueError("screen workload planner does not exactly match screen_runner protocol")
    expected_prefix = {"cal": "calibration", "dev": "development"}[split]
    expected_episode_ids = {f"{expected_prefix}:{index:03d}" for index in range(SCREEN_EPISODE_COUNT)}
    episode_ids = {str(pool["episode_id"]) for pool in pools}
    if episode_ids != expected_episode_ids:
        raise ValueError(
            f"screen workload must contain exactly 8 {expected_prefix} episode IDs; "
            f"got {sorted(episode_ids)}"
        )
    _validate_pool_structure(pools, expected_episode_ids, strict_screen=True)


def _validate_pool_structure(
    pools: Sequence[Mapping[str, Any]],
    expected_episode_ids: set[str] | None = None,
    strict_screen: bool = False,
) -> None:
    if not pools:
        raise ValueError("workload contains no pools")
    pool_ids = [str(pool["pool_id"]) for pool in pools]
    if len(pool_ids) != len(set(pool_ids)):
        raise ValueError("pool_id values must be unique")
    if strict_screen:
        for pool in pools:
            for observation_name in ("obs_0", "obs_g"):
                if not isinstance(pool.get(observation_name), Mapping):
                    raise ValueError(
                        f"screen pool {pool['pool_id']!r} is missing mapping {observation_name}"
                    )
            shape = tuple(np.asarray(pool["candidates"]).shape)
            if shape != SCREEN_CANDIDATE_SHAPE:
                raise ValueError(
                    f"pool {pool['pool_id']!r} candidates must have shape {SCREEN_CANDIDATE_SHAPE}, got {shape}"
                )
            if pool.get("reference_scores") is None:
                raise ValueError(f"screen pool {pool['pool_id']!r} is missing reference_scores")
            if np.asarray(pool["reference_scores"]).shape != (SCREEN_CANDIDATE_SHAPE[0],):
                raise ValueError(f"screen pool {pool['pool_id']!r} reference_scores must have shape (300,)")
            if int(pool["mpc_point"]) not in (0, 1):
                raise ValueError(f"screen pool {pool['pool_id']!r} has invalid mpc_point")
            if int(pool["cem_iteration"]) not in (1, 5):
                raise ValueError(f"screen pool {pool['pool_id']!r} must be CEM iteration 1 or 5")
        if expected_episode_ids is not None:
            observed = {str(pool["episode_id"]) for pool in pools}
            if observed != expected_episode_ids:
                raise ValueError("screen workload episode ID set changed during pool validation")
        by_episode: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for pool in pools:
            by_episode[str(pool["episode_id"])].append(pool)
        for episode_id, episode_pools in sorted(by_episode.items()):
            if len(episode_pools) not in (2, 4):
                raise ValueError(f"episode {episode_id} must contain exactly 2 or 4 pools")
            points = {int(pool["mpc_point"]) for pool in episode_pools}
            if points not in ({0}, {0, 1}):
                raise ValueError(f"episode {episode_id} visited points must be {0} or {0,1}")
            point_iterations: MutableMapping[int, List[int]] = defaultdict(list)
            for pool in episode_pools:
                point_iterations[int(pool["mpc_point"])].append(int(pool["cem_iteration"]))
            for point, iterations in point_iterations.items():
                if sorted(iterations) != [1, 5]:
                    raise ValueError(f"episode {episode_id} point {point} must contain exactly CEM iterations 1 and 5")
            keys = [(int(pool["mpc_point"]), int(pool["cem_iteration"])) for pool in episode_pools]
            if len(keys) != len(set(keys)):
                raise ValueError(f"episode {episode_id} contains duplicate MPC/CEM pool coordinates")


def _tensor_call_args(args: Sequence[Any]) -> Tuple[Any, ...]:
    import torch

    copied = []
    for value in args:
        copied.append(value.detach().clone() if isinstance(value, torch.Tensor) else value)
    return tuple(copied)


def _tensor_output(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for child in value:
            if isinstance(child, torch.Tensor):
                return child
    if isinstance(value, Mapping):
        for child in value.values():
            if isinstance(child, torch.Tensor):
                return child
    raise TypeError(f"module output does not contain a tensor: {type(value)!r}")


def _call_module(module: Any, args: Sequence[Any]) -> Any:
    return _tensor_output(module(*args))


def _predictor_block(layer: Any, x: Any) -> Any:
    """Replay one ViT predictor block (ModuleList has no forward method)."""
    first = _tensor_output(layer[0](x))
    first = first + x
    second = _tensor_output(layer[1](first))
    return second + first


def _move_obs(preprocessor: Any, target: Mapping[str, Any], device: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trans_0 = preprocessor.transform_obs(target["obs_0"])
    trans_g = preprocessor.transform_obs(target["obs_g"])
    return (
        {key: value.to(device) for key, value in trans_0.items()},
        {key: value.to(device) for key, value in trans_g.items()},
    )


def _repeat_batch(obs: Mapping[str, Any], count: int) -> Dict[str, Any]:
    return {
        key: value[0:1].repeat((count,) + (1,) * (value.ndim - 1))
        for key, value in obs.items()
    }


def _teacher_cache(model: Any, preprocessor: Any, target: Mapping[str, Any]) -> Dict[str, Any]:
    """Capture FP32 inputs/outputs for all groups once for one pool."""
    import torch

    device = next(model.parameters()).device
    trans_0, trans_g = _move_obs(preprocessor, target, device)
    encoder_blocks = model.encoder.base_model.blocks
    encoder_inputs: Dict[int, Dict[str, Tuple[Any, ...]]] = {
        index: {} for index in range(len(encoder_blocks))
    }
    handles = []
    for index, block in enumerate(encoder_blocks):
        def capture_encoder(_module: Any, args: Tuple[Any, ...], *, index: int = index) -> None:
            encoder_inputs[index]["current"] = _tensor_call_args(args)

        handles.append(block.register_forward_pre_hook(capture_encoder))
    with torch.no_grad():
        model.encode_obs(trans_0)
    for handle in handles:
        handle.remove()
    handles = []
    for index, block in enumerate(encoder_blocks):
        def capture_goal(_module: Any, args: Tuple[Any, ...], *, index: int = index) -> None:
            encoder_inputs[index]["goal"] = _tensor_call_args(args)

        handles.append(block.register_forward_pre_hook(capture_goal))
    with torch.no_grad():
        model.encode_obs(trans_g)
    for handle in handles:
        handle.remove()

    encoder_cache: Dict[int, Dict[str, Any]] = {}
    for index, block in enumerate(encoder_blocks):
        branch_cache: Dict[str, Any] = {}
        for branch in ("current", "goal"):
            args = encoder_inputs[index].get(branch)
            if args is None:
                raise RuntimeError(f"failed to capture encoder block {index} {branch} input")
            branch_cache[branch] = {
                "args": args,
                "output": _call_module(block, args).detach().clone(),
            }
        encoder_cache[index] = branch_cache

    candidates = np.asarray(target["candidates"], dtype=np.float32)
    actions = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    trans_0_batch = _repeat_batch(trans_0, int(candidates.shape[0]))
    predictor_layers = model.predictor.transformer.layers
    predictor_inputs: Dict[int, List[Any]] = {index: [] for index in range(len(predictor_layers))}
    handles = []
    for index, layer in enumerate(predictor_layers):
        def capture_predictor(_module: Any, args: Tuple[Any, ...], *, index: int = index) -> None:
            if not args:
                raise RuntimeError(f"predictor block {index} hook received no input")
            predictor_inputs[index].append(args[0].detach().clone())

        # Hook attention, the first child of the ModuleList, because ModuleList
        # intentionally has no forward implementation.
        handles.append(layer[0].register_forward_pre_hook(capture_predictor))
    with torch.no_grad():
        model.rollout(obs_0=trans_0_batch, act=actions)
    for handle in handles:
        handle.remove()

    predictor_cache: Dict[int, List[Dict[str, Any]]] = {}
    for index, layer in enumerate(predictor_layers):
        if not predictor_inputs[index]:
            raise RuntimeError(f"failed to capture predictor block {index} inputs")
        predictor_cache[index] = [
            {"input": value, "output": _predictor_block(layer, value).detach().clone()}
            for value in predictor_inputs[index]
        ]
    return {"encoder": encoder_cache, "predictor": predictor_cache}


def _local_group_nmse(model: Any, group: Mapping[str, Any], cache: Mapping[str, Any], eps: float = NMSE_EPS) -> float:
    import torch

    index = int(group["index"])
    values: List[float] = []
    if group["family"] == "encoder":
        for branch in ("current", "goal"):
            item = cache["encoder"][index][branch]
            quantized = _call_module(
                model.encoder.base_model.blocks[index], item["args"]
            )
            reference = item["output"]
            values.append(float(torch.mean((quantized - reference) ** 2).item()) / float(torch.mean(reference ** 2).item() + eps))
    else:
        layer = model.predictor.transformer.layers[index]
        for item in cache["predictor"][index]:
            quantized = _predictor_block(layer, item["input"])
            reference = item["output"]
            values.append(float(torch.mean((quantized - reference) ** 2).item()) / float(torch.mean(reference ** 2).item() + eps))
    if not values or not all(math.isfinite(value) for value in values):
        return float("nan")
    return float(np.mean(values))


def _runtime_groups(smoke: Any, runtime: Mapping[str, Any]) -> List[Dict[str, Any]]:
    groups = smoke._linear_groups(runtime["model"])
    enc_count = sum(group["family"] == "encoder" for group in groups)
    pred_count = sum(group["family"] == "predictor" for group in groups)
    if (enc_count, pred_count) != (EXPECTED_ENCODER_GROUPS, EXPECTED_PREDICTOR_GROUPS):
        raise RuntimeError(f"expected 12 encoder and 6 predictor groups, got {enc_count}/{pred_count}")
    return groups


def _probe_records(
    runtime: Mapping[str, Any],
    smoke: Any,
    pools: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    on_pool_complete: Any = None,
) -> List[Dict[str, Any]]:
    import torch

    model = runtime["model"]
    preprocessor = smoke._preprocessor(runtime)
    objective_fn = smoke._objective()
    snapshot = smoke._snapshot_weights(model, groups)
    rows: List[Dict[str, Any]] = []
    for pool_index, pool in enumerate(pools):
        pool_started = time.monotonic()
        pool_rows: List[Dict[str, Any]] = []
        target = {"obs_0": pool["obs_0"], "obs_g": pool["obs_g"]}
        candidates = pool["candidates"]
        pool_fingerprint = _pool_fingerprint(pool)
        smoke._restore_weights(model, snapshot)
        # Always recompute FP32.  An input reference is a validation target,
        # not a substitute for scoring with this exact runtime/checkpoint.
        recomputed_reference = smoke._score_pool(model, preprocessor, objective_fn, target, candidates).detach().cpu().numpy()
        reference = np.asarray(recomputed_reference, dtype=np.float64).reshape(-1)
        if reference.shape[0] != candidates.shape[0] or not np.isfinite(reference).all():
            raise FloatingPointError(f"non-finite or malformed FP32 scores for pool {pool['pool_id']!r}")
        supplied_reference = pool.get("reference_scores")
        reference_origin = "computed_fp32"
        if supplied_reference is not None:
            supplied = np.asarray(supplied_reference, dtype=np.float64).reshape(-1)
            if supplied.shape != reference.shape or not np.allclose(supplied, reference, rtol=0.0, atol=1e-6):
                max_abs = float(np.max(np.abs(supplied - reference))) if supplied.shape == reference.shape else float("inf")
                raise ValueError(f"FP32 reference mismatch for pool {pool['pool_id']!r}: max_abs={max_abs:g}")
            reference_origin = "verified_input"
            supplied_elite = pool.get("reference_elite_indices")
            if supplied_elite is not None:
                actual_elite = np.argsort(reference, kind="stable")[: smoke.TOPK]
                if not np.array_equal(np.asarray(supplied_elite, dtype=np.int64).reshape(-1), actual_elite):
                    raise ValueError(f"FP32 elite-index mismatch for pool {pool['pool_id']!r}")
        teacher = _teacher_cache(model, preprocessor, target | {"candidates": candidates})
        # _teacher_cache removes all hooks before returning.  The first pool
        # checks this no-op path explicitly, then checks FP32 restoration below.
        noop_reference = None
        if pool_index == 0:
            noop_reference = smoke._score_pool(model, preprocessor, objective_fn, target, candidates).detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            if not np.array_equal(reference, noop_reference):
                raise RuntimeError(f"FP32 no-op score mismatch after teacher hooks for pool {pool['pool_id']!r}")
        pool_scores: Dict[str, np.ndarray] = {"reference_scores": reference.astype(np.float32)}
        for group in groups:
            for bits in BITS:
                smoke._restore_weights(model, snapshot)
                quantization = smoke._quantize_group(model, group, bits)
                started = time.monotonic()
                scores = smoke._score_pool(model, preprocessor, objective_fn, target, candidates)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                score_array = scores.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
                score_finite = bool(np.isfinite(score_array).all())
                if score_finite:
                    disagreement = pairwise_order_disagreement(reference, score_array)
                    score_nmse = planner_score_nmse(reference, score_array)
                    local_nmse = _local_group_nmse(model, group, teacher)
                else:
                    disagreement = float("nan")
                    score_nmse = float("nan")
                    local_nmse = float("nan")
                signal_finite = all(
                    math.isfinite(value)
                    for value in (disagreement, local_nmse, score_nmse)
                )
                finite = signal_finite
                if not signal_finite:
                    smoke._restore_weights(model, snapshot)
                    raise FloatingPointError(
                        f"non-finite probe signal for {pool['pool_id']!r}/"
                        f"{group['group_id']}/W{bits}"
                    )
                score_key = f"g{int(group['index']):02d}_{group['family']}_W{bits}"
                pool_scores[score_key] = score_array.astype(np.float32, copy=True)
                pool_rows.append(
                    {
                        "pool_id": str(pool["pool_id"]),
                        "pool_fingerprint": pool_fingerprint,
                        "episode_id": str(pool["episode_id"]),
                        "mpc_point": int(pool["mpc_point"]),
                        "cem_iteration": int(pool["cem_iteration"]),
                        "pool_index": pool_index,
                        "group_id": str(group["group_id"]),
                        "family": str(group["family"]),
                        "site_index": int(group["index"]),
                        "bits": bits,
                        "finite": finite,
                        "rank_disagreement": disagreement,
                        "local_block_output_nmse": local_nmse,
                        "planner_score_nmse": score_nmse,
                        "reference_score_origin": reference_origin,
                        "candidate_count": int(candidates.shape[0]),
                        "pair_count": int(candidates.shape[0] * (candidates.shape[0] - 1) // 2),
                        "score_vector_key": score_key,
                        "fp32_noop_exact": (bool(np.array_equal(reference, noop_reference)) if noop_reference is not None else None),
                        "elapsed_seconds": elapsed,
                        "quantization": quantization,
                    }
                )
        smoke._restore_weights(model, snapshot)
        if pool_index == 0:
            restored_scores = smoke._score_pool(model, preprocessor, objective_fn, target, candidates).detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            if not np.array_equal(reference, restored_scores):
                raise RuntimeError(f"FP32 restore score mismatch for pool {pool['pool_id']!r}")
        pool_elapsed = time.monotonic() - pool_started
        for row in pool_rows:
            row["pool_elapsed_seconds"] = pool_elapsed
        rows.extend(pool_rows)
        if on_pool_complete is not None:
            on_pool_complete(pool, pool_rows, pool_scores, pool_elapsed)
        smoke._restore_weights(model, snapshot)
    return rows


SIGNALS = ("rank_disagreement", "local_block_output_nmse", "planner_score_nmse")


def _aggregate_site_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_site_bit_episode: MutableMapping[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_site_bit_episode[(str(row["group_id"]), int(row["bits"]), str(row["episode_id"]))].append(row)
    episode_values: MutableMapping[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for (group_id, bits, episode_id), group_rows in sorted(by_site_bit_episode.items()):
        value: Dict[str, Any] = {"group_id": group_id, "bits": bits, "episode_id": episode_id, "pool_count": len(group_rows)}
        for signal in SIGNALS:
            values = [float(row[signal]) for row in group_rows]
            value[signal] = float(np.mean(values)) if values and all(math.isfinite(item) for item in values) else float("nan")
        episode_values[(group_id, bits)].append(value)

    result: List[Dict[str, Any]] = []
    for (group_id, bits), values in sorted(episode_values.items()):
        summary: Dict[str, Any] = {
            "group_id": group_id,
            "bits": bits,
            "episode_count": len(values),
            "pool_count": int(sum(item["pool_count"] for item in values)),
            "episodes": values,
        }
        for signal in SIGNALS:
            signal_values = [float(item[signal]) for item in values]
            summary[signal] = float(np.mean(signal_values)) if signal_values and all(math.isfinite(item) for item in signal_values) else float("nan")
        result.append(summary)
    return result


def _metric_definitions() -> Dict[str, Any]:
    return {
        "rank_disagreement": {
            "formula": "mean_{i<j}[sign(r_i-r_j) != sign(q_i-q_j)]",
            "pairs": "all unordered candidate pairs",
            "ties": "exact zero sign is retained; one-side tie is disagreement, two-side tie agrees",
            "aggregation": "pool mean, then episode mean; episodes have equal weight",
        },
        "local_block_output_nmse": {
            "formula": "mean((q_block-r_block)^2)/(mean(r_block^2)+1e-12)",
            "teacher_inputs": "FP32 target-block inputs; no cumulative quantized activation drift",
            "aggregation": "encoder current/goal branches equally weighted; predictor rollout calls equally weighted; then pool/episode means",
        },
        "planner_score_nmse": {
            "formula": "mean((q_score-r_score)^2)/(mean(r_score^2)+1e-12)",
            "denominator": "the FP32 score pool denominator is fixed for each site/bit comparison",
            "aggregation": "pool mean, then episode mean; episodes have equal weight",
        },
    }


def _run_probes(args: argparse.Namespace, workload_path: Path, output: Path) -> Dict[str, Any]:
    smoke = _load_smoke_runner()
    payload = _load_payload(workload_path)
    pools = _flatten_pools(payload, args.split)
    _validate_screen_workload(payload, args.split, pools)
    runtime = smoke._runtime(args.root.resolve(), device=args.device)
    groups = _runtime_groups(smoke, runtime)
    runtime_identity = _stable_runtime_identity(runtime, smoke)
    pool_fingerprints = {str(pool["pool_id"]): _pool_fingerprint(pool) for pool in pools}
    _validate_pool_budget(pools, args.max_pools_per_episode)
    if len(pools) > args.max_pools:
        raise ValueError(f"workload has {len(pools)} pools; max allowed is {args.max_pools}")
    output.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(pool: Mapping[str, Any]) -> Path:
        safe_id = str(pool["pool_id"]).replace("/", "_").replace("\\", "_")
        return output / "pool_checkpoints" / f"pool_{safe_id}.pkl"

    resumed_rows: List[Dict[str, Any]] = []
    remaining_pools = list(pools)
    if args.resume:
        remaining_pools = []
        for pool in pools:
            checkpoint = checkpoint_path(pool)
            if not checkpoint.is_file():
                remaining_pools.append(pool)
                continue
            checkpoint_payload = _load_payload(checkpoint)
            if not isinstance(checkpoint_payload, Mapping) or checkpoint_payload.get("split") != args.split:
                remaining_pools.append(pool)
                continue
            if checkpoint_payload.get("runtime_identity") != runtime_identity:
                remaining_pools.append(pool)
                continue
            if checkpoint_payload.get("pool_fingerprint") != pool_fingerprints[str(pool["pool_id"])] :
                remaining_pools.append(pool)
                continue
            checkpoint_pool = checkpoint_payload.get("pool", {})
            if str(checkpoint_pool.get("pool_id")) != str(pool["pool_id"]):
                remaining_pools.append(pool)
                continue
            checkpoint_rows = checkpoint_payload.get("records", [])
            if not isinstance(checkpoint_rows, Sequence) or len(checkpoint_rows) != EXPECTED_GROUPS * len(BITS):
                remaining_pools.append(pool)
                continue
            if not checkpoint_path(pool).with_suffix(".scores.npz").is_file():
                remaining_pools.append(pool)
                continue
            resumed_rows.extend(dict(row) for row in checkpoint_rows)

    started = time.monotonic()

    def checkpoint_pool(pool: Mapping[str, Any], pool_rows: Sequence[Mapping[str, Any]], pool_scores: Mapping[str, np.ndarray], elapsed: float) -> None:
        checkpoint = checkpoint_path(pool)
        score_path = checkpoint.with_suffix(".scores.npz")
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(score_path, **dict(pool_scores))
        _write_pickle(
            checkpoint,
            {
                "schema": "rankcal-wall-probe-pool-checkpoint-v1",
                "split": args.split,
                "pool": {key: pool[key] for key in ("pool_id", "episode_id", "mpc_point", "cem_iteration")},
                "pool_fingerprint": pool_fingerprints[str(pool["pool_id"])],
                "runtime_identity": runtime_identity,
                "records": list(pool_rows),
                "score_vectors": str(score_path),
                "elapsed_seconds": elapsed,
            },
        )
        print(json.dumps({"status": "pool_complete", "split": args.split, "pool_id": str(pool["pool_id"]), "elapsed_seconds": elapsed}, default=_json_default), flush=True)

    rows = resumed_rows + _probe_records(runtime, smoke, remaining_pools, groups, on_pool_complete=checkpoint_pool)
    site_metrics = _aggregate_site_metrics(rows)
    artifact: Dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "status": "complete",
        "split": args.split,
        "input_workload": str(workload_path.resolve()),
        "source_workload_schema": SCREEN_WORKLOAD_SCHEMA,
        "source_screen_split": payload.get("screen_split"),
        "source_planner": payload.get("planner"),
        "pool_count": len(pools),
        "episode_count": len({str(pool["episode_id"]) for pool in pools}),
        "pools": [
            {
                "pool_id": str(pool["pool_id"]),
                "episode_id": str(pool["episode_id"]),
                "mpc_point": int(pool["mpc_point"]),
                "cem_iteration": int(pool["cem_iteration"]),
                "candidate_shape": list(pool["candidates"].shape),
                "pool_fingerprint": pool_fingerprints[str(pool["pool_id"])],
            }
            for pool in pools
        ],
        "max_pools_per_episode": args.max_pools_per_episode,
        "expected_probe_count": EXPECTED_GROUPS * len(BITS),
        "probe_record_count": len(rows),
        "resumed_pool_count": len(pools) - len(remaining_pools),
        "groups": groups,
        "records": rows,
        "site_metrics": site_metrics,
        "score_vector_storage": "pool_checkpoints/pool_<pool_id>.scores.npz",
        "fp32_reference_recomputed": True,
        "fp32_reference_check": "input reference_scores checked against recomputed FP32 with rtol=0, atol=1e-6; first-pool no-op and restore require exact equality",
        "metric_definitions": _metric_definitions(),
        "cost_rule": "all allocations use exactly 3 encoder and 2 predictor W8 groups; W4/W8 logical costs are recorded per group",
        "execution": "emulation_only",
        "checkpoint_identity": runtime.get("checkpoint") and smoke._checkpoint_identity(runtime),
        "stable_runtime_identity": runtime_identity,
        "planner": {
            "horizon": smoke.HORIZON,
            "num_samples": smoke.NUM_SAMPLES,
            "topk": smoke.TOPK,
            "cem_iterations": smoke.CEM_STEPS,
            "stable_argsort": True,
            "inner_environment_evaluator": None,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_pickle(output / "probes.pkl", artifact)
    _write_json(output / "probe_summary.json", artifact)
    with (output / "probe_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=_json_default) + "\n")
    return artifact


def _group_cost(group: Mapping[str, Any], bits: int) -> int:
    key = f"logical_weight_bytes_W{bits}"
    if key in group:
        return int(group[key])
    numel = int(group["numel"])
    scale_count = int(group.get("scale_count", 0))
    return int(math.ceil(numel * bits / 8) + scale_count * 4)


def _validate_groups(groups: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    copied = [dict(group) for group in groups]
    if len(copied) != EXPECTED_GROUPS:
        raise ValueError(f"allocation needs {EXPECTED_GROUPS} groups, got {len(copied)}")
    counts = {family: sum(group.get("family") == family for group in copied) for family in W8_QUOTA}
    if counts != {"encoder": EXPECTED_ENCODER_GROUPS, "predictor": EXPECTED_PREDICTOR_GROUPS}:
        raise ValueError(f"allocation group counts are {counts}")
    for family in W8_QUOTA:
        signatures = {
            (int(group["numel"]), int(group.get("scale_count", 0)), _group_cost(group, 4), _group_cost(group, 8))
            for group in copied if group["family"] == family
        }
        if len(signatures) != 1:
            raise ValueError(f"{family} groups have unequal logical shapes/costs; revise strata before allocation")
    return sorted(copied, key=lambda group: (str(group["family"]), int(group["index"]), str(group["group_id"])))


def _metric_lookup(probe: Mapping[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows = probe.get("site_metrics")
    if rows is None:
        rows = _aggregate_site_metrics(probe.get("records", []))
    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["group_id"]), int(row["bits"]))
        if key in lookup:
            raise ValueError(f"duplicate site metric {key}")
        lookup[key] = dict(row)
    return lookup


def _validate_probe_artifact(probe: Mapping[str, Any]) -> None:
    """Re-audit calibration completeness immediately before allocation."""
    if probe.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"allocation needs {PROBE_SCHEMA}; got {probe.get('schema')!r}")
    if probe.get("split") != "cal" or not probe.get("source_workload_schema") == SCREEN_WORKLOAD_SCHEMA:
        raise ValueError("allocation accepts only completed calibration probes from screen workload v1")
    if _canonical_split(probe.get("source_screen_split")) != "cal":
        raise ValueError("probe source screen_split is not calibration")
    if probe.get("source_planner") != _load_screen_runner()._protocol_config():
        raise ValueError("probe source planner does not exactly match screen_runner protocol")
    pools = probe.get("pools")
    if not isinstance(pools, Sequence) or isinstance(pools, (str, bytes)):
        raise ValueError("probe artifact is missing its pool manifest")
    pool_meta = [dict(pool) for pool in pools]
    expected_episode_ids = {f"calibration:{index:03d}" for index in range(SCREEN_EPISODE_COUNT)}
    pool_ids = [str(pool.get("pool_id")) for pool in pool_meta]
    if len(pool_ids) != len(set(pool_ids)):
        raise ValueError("probe pool manifest contains duplicate pool IDs")
    if {str(pool.get("episode_id")) for pool in pool_meta} != expected_episode_ids:
        raise ValueError("probe pool manifest must contain exactly 8 calibration episode IDs")
    for pool in pool_meta:
        if tuple(pool.get("candidate_shape", ())) != SCREEN_CANDIDATE_SHAPE:
            raise ValueError(f"probe pool {pool.get('pool_id')!r} has invalid candidate shape")
    by_episode: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for pool in pool_meta:
        by_episode[str(pool["episode_id"])].append(pool)
    for episode_id, episode_pools in sorted(by_episode.items()):
        if len(episode_pools) not in (2, 4):
            raise ValueError(f"probe episode {episode_id} must contain exactly 2 or 4 pools")
        points = {int(pool["mpc_point"]) for pool in episode_pools}
        if points not in ({0}, {0, 1}):
            raise ValueError(f"probe episode {episode_id} visited points must be 0 or 0/1")
        for point in points:
            iterations = sorted(int(pool["cem_iteration"]) for pool in episode_pools if int(pool["mpc_point"]) == point)
            if iterations != [1, 5]:
                raise ValueError(f"probe episode {episode_id} point {point} must contain CEM iterations 1 and 5")
    groups = _validate_groups(probe.get("groups", []))
    group_ids = {str(group["group_id"]): str(group["family"]) for group in groups}
    rows = probe.get("records")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("probe artifact is missing scalar probe records")
    if len(rows) != len(pool_meta) * EXPECTED_GROUPS * len(BITS):
        raise ValueError("probe artifact does not contain 36 rows for every pool")
    row_keys = []
    rows_by_pool: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("probe record is not a mapping")
        pool_id = str(row.get("pool_id"))
        group_id = str(row.get("group_id"))
        bits = int(row.get("bits", -1))
        if pool_id not in set(pool_ids) or group_id not in group_ids or bits not in BITS:
            raise ValueError("probe record refers to an unknown pool, group, or bit width")
        if int(row.get("candidate_count", -1)) != SCREEN_CANDIDATE_SHAPE[0]:
            raise ValueError(f"probe record for {pool_id} has invalid candidate_count")
        if int(row.get("pair_count", -1)) != SCREEN_PAIR_COUNT:
            raise ValueError(f"probe record for {pool_id} has invalid pair_count")
        signal_finite = all(
            math.isfinite(float(row.get(signal, float("nan"))))
            for signal in SIGNALS
        )
        if str(row.get("family")) != group_ids[group_id] or not bool(row.get("finite")) or not signal_finite:
            raise ValueError(f"probe record for {pool_id}/{group_id}/W{bits} failed finite/family validation")
        if row.get("pool_fingerprint") != next(pool["pool_fingerprint"] for pool in pool_meta if str(pool["pool_id"]) == pool_id):
            raise ValueError(f"probe record for {pool_id} has a pool fingerprint mismatch")
        key = (pool_id, group_id, bits)
        if key in row_keys:
            raise ValueError(f"duplicate probe row {key}")
        row_keys.append(key)
        rows_by_pool[pool_id].append(row)
    expected_keys = {(pool_id, group_id, bits) for pool_id in pool_ids for group_id in group_ids for bits in BITS}
    if set(row_keys) != expected_keys:
        raise ValueError("probe records do not cover every pool x group x bit combination")
    site_metrics = probe.get("site_metrics")
    if not isinstance(site_metrics, Sequence) or len(site_metrics) != EXPECTED_GROUPS * len(BITS):
        raise ValueError("probe artifact site_metrics must contain exactly 36 site/bit rows")
    metric_keys = {(str(row.get("group_id")), int(row.get("bits", -1))) for row in site_metrics}
    if metric_keys != {(group_id, bits) for group_id in group_ids for bits in BITS}:
        raise ValueError("probe site_metrics do not cover every group x bit combination")


def _allocation_cost(groups: Sequence[Mapping[str, Any]], bits_by_site: Mapping[str, int]) -> Dict[str, Any]:
    total_bytes = 0
    total_weight_bits = 0
    by_family: Dict[str, Any] = {}
    for family in W8_QUOTA:
        family_groups = [group for group in groups if group["family"] == family]
        family_bytes = 0
        family_weight_bits = 0
        for group in family_groups:
            bits = int(bits_by_site[str(group["group_id"])])
            family_bytes += _group_cost(group, bits)
            family_weight_bits += int(group["numel"]) * bits
        by_family[family] = {
            "w8_count": sum(int(bits_by_site[str(group["group_id"])]) == 8 for group in family_groups),
            "logical_weight_bytes": family_bytes,
            "weight_bits": family_weight_bits,
        }
        total_bytes += family_bytes
        total_weight_bits += family_weight_bits
    return {"logical_weight_bytes": total_bytes, "weight_bits": total_weight_bits, "by_family": by_family}


def _bits_map(groups: Sequence[Mapping[str, Any]], selected: Iterable[str]) -> Dict[str, int]:
    selected_set = {str(site) for site in selected}
    return {str(group["group_id"]): (8 if str(group["group_id"]) in selected_set else 4) for group in groups}


def _select_by_signal(groups: Sequence[Mapping[str, Any]], lookup: Mapping[Tuple[str, int], Mapping[str, Any]], signal: str) -> Tuple[List[str], Dict[str, Any]]:
    ranking = []
    detail: Dict[str, Any] = {}
    for group in groups:
        site = str(group["group_id"])
        w4 = lookup.get((site, 4))
        w8 = lookup.get((site, 8))
        if w4 is None or w8 is None:
            raise ValueError(f"missing W4/W8 metric for {site}")
        v4 = float(w4[signal])
        v8 = float(w8[signal])
        if not math.isfinite(v4) or not math.isfinite(v8):
            raise ValueError(f"non-finite {signal} metric for {site}; cannot fit allocation")
        benefit = v4 - v8
        detail[site] = {"W4": v4, "W8": v8, "benefit_W4_minus_W8": benefit}
        ranking.append((str(group["family"]), -benefit, site))
    selected: List[str] = []
    for family in W8_QUOTA:
        family_rank = sorted((item for item in ranking if item[0] == family), key=lambda item: (item[1], item[2]))
        selected.extend(item[2] for item in family_rank[: W8_QUOTA[family]])
    return sorted(selected), detail


def _random_selection(groups: Sequence[Mapping[str, Any]], seed: int, forbidden: Sequence[Iterable[str]]) -> Tuple[List[str], int]:
    rng = random.Random(seed)
    forbidden_fingerprints = {tuple(sorted(str(site) for site in selection)) for selection in forbidden}
    for attempt in range(1, 100000):
        selected = []
        for family in W8_QUOTA:
            family_sites = [str(group["group_id"]) for group in groups if group["family"] == family]
            selected.extend(rng.sample(family_sites, W8_QUOTA[family]))
        selected = sorted(selected)
        if tuple(selected) not in forbidden_fingerprints:
            return selected, attempt
    raise RuntimeError(f"could not draw a new random allocation for seed {seed}")


def _run_allocate(probe_path: Path, output: Path) -> Dict[str, Any]:
    probe = _load_payload(probe_path)
    if not isinstance(probe, Mapping):
        raise TypeError("probe artifact must be a mapping")
    _validate_probe_artifact(probe)
    groups = _validate_groups(probe.get("groups", []))
    lookup = _metric_lookup(probe)
    method_signals = {"RankCal": "rank_disagreement", "LocalMSE": "local_block_output_nmse", "ScoreError": "planner_score_nmse"}
    allocations: Dict[str, Any] = {}
    selected_for_exclusion: List[List[str]] = []
    for method, signal in method_signals.items():
        selected, detail = _select_by_signal(groups, lookup, signal)
        bits = _bits_map(groups, selected)
        allocations[method] = {
            "method": method,
            "signal": signal,
            "selected_sites": selected,
            "bits": bits,
            "allocation": bits,
            "family_w8_quota": dict(W8_QUOTA),
            "cost": _allocation_cost(groups, bits),
            "site_signal_values": detail,
            "tie_rule": "descending W4-W8 benefit; exact equal benefit breaks by lexicographic site ID within family",
            "calibration_split": "cal",
        }
        selected_for_exclusion.append(selected)
    for method, seed in zip(("Random1", "Random2"), RANDOM_SEEDS):
        selected, attempts = _random_selection(groups, seed, selected_for_exclusion)
        bits = _bits_map(groups, selected)
        allocations[method] = {
            "method": method,
            "signal": "uniform_random_mapping",
            "random_seed": seed,
            "draw_attempt": attempts,
            "selected_sites": selected,
            "bits": bits,
            "allocation": bits,
            "family_w8_quota": dict(W8_QUOTA),
            "cost": _allocation_cost(groups, bits),
            "tie_rule": "uniform random within each family; duplicate maps are redrawn from the same seeded stream",
            "calibration_split": "cal",
        }
        selected_for_exclusion.append(selected)

    costs = {method: value["cost"] for method, value in allocations.items()}
    common_cost = len({(cost["logical_weight_bytes"], cost["weight_bits"]) for cost in costs.values()}) == 1
    if not common_cost:
        raise ValueError("generated methods do not have the same logical cost; revise group strata before running")
    artifact: Dict[str, Any] = {
        "schema": ALLOCATION_SCHEMA,
        "status": "complete",
        "source_probe_artifact": str(probe_path.resolve()),
        "calibration_only": True,
        "methods": allocations,
        "allocations": list(allocations.values()),
        "method_order": list(allocations),
        "random_seeds": list(RANDOM_SEEDS),
        "family_w8_quota": dict(W8_QUOTA),
        "common_cost": common_cost,
        "cost": next(iter(costs.values())),
        "cost_definition": "logical packed W4/W8 weight bytes plus FP32 per-output-channel scales; execution remains FP32 emulation",
        "metric_definitions": _metric_definitions(),
        "allocation_tie_rule": "signal benefit is W4-W8 and larger is preferred; exact ties use site ID",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "allocations.json", artifact)
    # Each per-method file is directly consumable by screen_runner.py, whose
    # allocation mode accepts a top-level ``allocation``/``bits`` mapping.
    for method, method_payload in allocations.items():
        _write_json(
            output / f"{method}.json",
            {
                "schema": ALLOCATION_SCHEMA,
                "name": method,
                "method": method,
                "allocation": method_payload["bits"],
                "bits": method_payload["bits"],
                "selected_sites": method_payload["selected_sites"],
                "family_w8_quota": dict(W8_QUOTA),
                "cost": method_payload["cost"],
                "signal": method_payload["signal"],
                "random_seed": method_payload.get("random_seed"),
                "source_probe_artifact": str(probe_path.resolve()),
                "calibration_only": True,
            },
        )
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probes", "allocate", "all"), default=None)
    parser.add_argument("--root", type=Path, default=None, help="DINO-WM Wall runtime root; required for probes")
    parser.add_argument("--workload", "--input", dest="workload", type=Path, default=None, help="cal/dev workload.pkl or directory")
    parser.add_argument("--dev-workload", type=Path, default=None, help="alias used when --split dev")
    parser.add_argument("--probes", "--probe-file", dest="probes", type=Path, default=None, help="probe artifact for allocate")
    parser.add_argument("--output", type=Path, default=None, help="artifact output directory")
    parser.add_argument("--split", choices=("cal", "calibration", "dev", "development"), default="cal")
    parser.add_argument("--device", default=None, help="torch device, normally cuda:0")
    parser.add_argument("--max-pools", type=int, default=MAX_POOLS, help="maximum total pools in one invocation")
    parser.add_argument("--max-pools-per-episode", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="reuse completed per-pool checkpoints under --output")
    parser.add_argument("--self-test", action="store_true", help="run the three-pair E2 tie test and exit")
    return parser.parse_args()


def main() -> None:
    _self_test_metrics()
    args = _parse_args()
    args.split = SPLIT_ALIASES[args.split]
    if args.self_test:
        print(json.dumps({"status": "complete", "test": "pairwise_order_disagreement_three_pair_tie_cases"}, indent=2))
        return
    if args.mode is None:
        raise SystemExit("--mode is required unless --self-test is used")
    if args.output is None:
        raise SystemExit("--output is required")
    output = args.output.resolve()
    if args.mode in ("probes", "all"):
        if args.root is None:
            raise SystemExit("--root is required for probes")
        workload_arg = args.dev_workload if args.split == "dev" and args.dev_workload is not None else args.workload
        if workload_arg is None:
            raise SystemExit("--workload is required for probes")
        workload_path = _resolve_workload(workload_arg.resolve(), args.split)
        probe_artifact = _run_probes(args, workload_path, output)
        if args.mode == "probes" or args.split == "dev":
            print(json.dumps({"status": "complete", "mode": "probes", "split": args.split, "output": str(output)}, indent=2))
            return
    else:
        probe_artifact = None
    probe_path = output / "probes.pkl" if probe_artifact is not None else (args.probes or (output / "probes.pkl"))
    allocation = _run_allocate(probe_path.resolve(), output)
    print(json.dumps({"status": "complete", "mode": "allocate", "methods": list(allocation["methods"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

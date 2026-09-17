"""Independent raw-output verifier for the FRT CCDS-TC1 A/B pilot.

The verifier never imports the model and never opens an artifact before the
real SLURM allocation guard has passed. It audits arithmetic and metadata from
the raw NPZ/JSON outputs, then emits a compact summary. Raw arrays stay in the
compute-side artifact directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHODS = ("clean", "random_same_norm", "frt")
SEEDS = (1201, 1202, 1203)
BANKS = (
    "q0",
    "random_same_norm",
    "fresh:clean:1201",
    "fresh:clean:1202",
    "fresh:clean:1203",
    "fresh:random_same_norm:1201",
    "fresh:random_same_norm:1202",
    "fresh:random_same_norm:1203",
    "fresh:frt:1201",
    "fresh:frt:1202",
    "fresh:frt:1203",
)
EVALUATORS = (
    "fp32",
    "q0_rtn",
    "clean:1201",
    "clean:1202",
    "clean:1203",
    "random_same_norm:1201",
    "random_same_norm:1202",
    "random_same_norm:1203",
    "frt:1201",
    "frt:1202",
    "frt:1203",
)
Q0_BANK = 0
RANDOM_BANK = 1
FRESH_BANKS = tuple(range(2, 11))
FIT_EVALUATOR_INDEX = {
    f"{method}:{seed}": 2 + method_index * 3 + seed_index
    for method_index, method in enumerate(METHODS)
    for seed_index, seed in enumerate(SEEDS)
}
EXPECTED_TARGETS = {
    "frt_cal": tuple(range(60, 66)),
    "frt_dev": tuple(range(66, 72)),
}
EXPECTED_EPISODES = {
    "frt_cal": tuple(f"frt_cal:{index:03d}" for index in range(6)),
    "frt_dev": tuple(f"frt_dev:{index:03d}" for index in range(6)),
}
WZ_STD_FLOOR = 1.0e-3
EPS = 1.0e-12
REL_TOL = 1.0e-6
VECTOR_RTOL = 2.0e-5
VECTOR_ATOL = 2.0e-7
MAX_ERRORS = 24


def _compact_errors(errors: Sequence[str]) -> list[str]:
    result = [str(error)[:240] for error in errors[:MAX_ERRORS]]
    if len(errors) > MAX_ERRORS:
        result.append(f"... {len(errors) - MAX_ERRORS} more errors")
    return result


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)
    if path.stat().st_size > 65536:
        raise RuntimeError("verification summary exceeds 64 KiB")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(values: Any) -> list[str]:
    array = np.asarray(values)
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.reshape(-1).tolist()
    ]


def _finite(name: str, value: Any, errors: list[str]) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        errors.append(f"{name} is not numeric")
        return array
    if not np.isfinite(array).all():
        errors.append(f"{name} contains non-finite values")
    return array


def _load_guard() -> Any:
    candidates = (
        Path(__file__).with_name("allocation_guard.py"),
        Path(__file__).parents[1] / "cem-update-ptq-ccds" / "allocation_guard.py",
    )
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("frt_allocation_guard", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module.require_allocation
    raise FileNotFoundError("allocation_guard.py was not found beside FRT or old CCDS CEM")


def _runtime_errors(runtime: Any, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(runtime, Mapping):
        return ["runtime_identity is missing"]
    # The numerical runner writes the executable identity under
    # ``runtime_identity``.  Keep ``source_identity`` as the protocol-facing
    # alias so this verifier can audit both the frozen manifest and runner
    # summaries without importing the model.
    expected = manifest.get("runtime_identity") or manifest.get("source_identity", {})
    if not isinstance(expected, Mapping):
        expected = {}
    if "recorded_epoch" in expected and runtime.get("recorded_epoch") != expected["recorded_epoch"]:
        errors.append("runtime recorded_epoch disagrees with manifest")
    if runtime.get("source_commit") != expected.get("source_commit"):
        errors.append("runtime source_commit disagrees with manifest")
    wanted_dino = expected.get("dinov2_source_commit", expected.get("dinov2_commit"))
    if runtime.get("dinov2_source_commit") != wanted_dino:
        errors.append("runtime dinov2_source_commit disagrees with manifest")
    if runtime.get("dtype") != expected.get("dtype", "float32"):
        errors.append("runtime dtype disagrees with manifest")
    if runtime.get("decoder", "missing") is not None:
        errors.append("runtime decoder is not None")
    if runtime.get("execution") not in ("emulation_only", "weight_only_numerical_emulation"):
        errors.append("runtime execution is not emulation_only")
    if runtime.get("native_memory_claim") is not False:
        errors.append("runtime native_memory_claim must be false")
    gpu = str(runtime.get("gpu", ""))
    if "v100" not in gpu.casefold():
        errors.append("runtime GPU is not a V100")
    return errors


def _manifest_audit(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema") != "frt-ccds-manifest-v1":
        errors.append("manifest schema mismatch")
    if manifest.get("protocol_id") != "frt-ccds-v1":
        errors.append("protocol_id mismatch")
    scope = manifest.get("research_scope")
    if not isinstance(scope, Mapping) or scope.get("test_split") is not False or scope.get("stage_c") is not False:
        errors.append("manifest must declare no TEST and no Stage C")
    source = manifest.get("source_identity")
    if not isinstance(source, Mapping):
        errors.append("source_identity is missing")
    else:
        if source.get("recorded_epoch") != 65:
            errors.append("source epoch is not 65")
        if source.get("dtype") != "float32" or source.get("decoder") is not None:
            errors.append("source dtype/decoder mismatch")
        if source.get("native_memory_claim") is not False:
            errors.append("manifest enables native memory claim")
        if source.get("dinov2_source_commit") != "7764ea0f912e53c92e82eb78a2a1631e92725fc8":
            errors.append("manifest DINOv2 commit mismatch")
    quantizer = manifest.get("quantizer")
    expected_blocks = [f"predictor.transformer.layers.{index}" for index in range(6)]
    if (
        not isinstance(quantizer, Mapping)
        or quantizer.get("bits") != 4
        or quantizer.get("q_min") != -7
        or quantizer.get("q_max") != 7
        or quantizer.get("scheme") != "symmetric_per_output_channel_weight_only"
        or list(quantizer.get("target_blocks", [])) != expected_blocks
    ):
        errors.append("quantizer target/W4 contract mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        errors.append("splits are missing")
    else:
        if splits.get("historical_reserved_minimum") != [0, 49]:
            errors.append("historical reserve is not 0..49")
        if splits.get("safety_reserved") != [50, 59]:
            errors.append("safety reserve is not 50..59")
        if splits.get("frt_reserved") != [60, 71]:
            errors.append("FRT reserve is not 60..71")
        for split_name, expected_indices in (("cal", EXPECTED_TARGETS["frt_cal"]), ("dev", EXPECTED_TARGETS["frt_dev"])):
            node = splits.get(split_name)
            if not isinstance(node, Mapping) or tuple(node.get("dataset_indices", [])) != expected_indices:
                errors.append(f"{split_name} dataset index contract mismatch")
            if not isinstance(node, Mapping) or tuple(node.get("episode_ids", [])) != EXPECTED_EPISODES[node.get("split_id", "") if isinstance(node, Mapping) else "frt_cal"]:
                # The precise IDs are checked below; this branch only guards malformed split_id.
                if isinstance(node, Mapping):
                    errors.append(f"{split_name} episode ID contract mismatch")
    fit = manifest.get("fit")
    if not isinstance(fit, Mapping):
        errors.append("fit section is missing")
    else:
        if tuple(fit.get("methods", [])) != METHODS:
            errors.append("method order mismatch")
        if tuple(fit.get("seeds", [])) != SEEDS:
            errors.append("fit seed order mismatch")
        updates = fit.get("fit_updates")
        if updates is not None and (not isinstance(updates, int) or updates <= 0):
            errors.append("fit_updates must be null or a positive integer")
        policy = fit.get("fit_update_policy", {})
        if not isinstance(policy, Mapping) or tuple(policy.get("candidate_values", [])) != (256, 512, 1000):
            errors.append("fit update candidate policy mismatch")
        batch = fit.get("batch_size_policy", {})
        if not isinstance(batch, Mapping) or tuple(batch.get("candidate_values", [])) != (1, 2, 4):
            errors.append("batch-size candidate policy mismatch")
    loss = manifest.get("loss")
    wz = loss.get("wz") if isinstance(loss, Mapping) else None
    if not isinstance(wz, Mapping) or float(wz.get("std_floor", -1)) != WZ_STD_FLOOR:
        errors.append("Wz std floor mismatch")
    lambda_rule = loss.get("lambda_rule") if isinstance(loss, Mapping) else None
    if not isinstance(lambda_rule, Mapping) or tuple(lambda_rule.get("clip", [])) != (0.25, 4.0):
        errors.append("lambda CAL clipping rule is missing or changed")
    random_control = manifest.get("random_control")
    if not isinstance(random_control, Mapping) or int(random_control.get("seed_base", -1)) != 940000:
        errors.append("random control seed base mismatch")
    gates = manifest.get("gates")
    if not isinstance(gates, Mapping):
        errors.append("gates are missing")
    else:
        clean = gates.get("clean_tolerance", {})
        transport = gates.get("transport_improvement", {})
        if not isinstance(clean, Mapping) or clean.get("per_seed_episode_requirement") != 4 or clean.get("seed_requirement") != "at_least_2_of_3":
            errors.append("clean gate policy mismatch")
        if not isinstance(transport, Mapping) or tuple(transport.get("views", [])) != ("q0", "fresh_union") or tuple(transport.get("comparators", [])) != ("clean", "random_same_norm"):
            errors.append("transport gate policy mismatch")
    return {"pass": not errors, "errors": _compact_errors(errors)}


def _record_audit(meta: Mapping[str, Any], arrays: Mapping[str, Any], split: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    expected_indices = EXPECTED_TARGETS[split]
    expected_episodes = EXPECTED_EPISODES[split]
    for key in ("record_keys", "episode_ids", "dataset_indices", "env_seeds", "cem_seeds"):
        if key not in arrays:
            errors.append(f"missing {key}")
    if errors:
        return errors, {"record_count": 0}
    keys = _strings(arrays["record_keys"])
    episodes = _strings(arrays["episode_ids"])
    indices = np.asarray(arrays["dataset_indices"]).reshape(-1)
    env_seeds = np.asarray(arrays["env_seeds"]).reshape(-1)
    cem_seeds = np.asarray(arrays["cem_seeds"]).reshape(-1)
    count = len(keys)
    if len(set(keys)) != count:
        errors.append("record_keys are duplicated")
    if not (len(episodes) == len(indices) == len(env_seeds) == len(cem_seeds) == count):
        errors.append("record metadata lengths disagree")
        return errors, {"record_count": count}
    if set(episodes) - set(expected_episodes):
        errors.append("unexpected episode ID in record arrays")
    if set(episodes) != set(expected_episodes):
        errors.append("record arrays do not cover exactly six frozen episodes")
    if set(indices.tolist()) - set(expected_indices):
        errors.append("historical or out-of-range dataset index in record arrays")
    # Multiple records may belong to one episode, so dataset indices need not
    # be unique at the flat-record level.  The real invariant is a stable
    # episode -> dataset-index mapping; conflicting mappings are invalid.
    episode_to_index: dict[str, int] = {}
    for episode, value in zip(episodes, indices.tolist()):
        current = int(value)
        prior = episode_to_index.setdefault(episode, current)
        if prior != current:
            errors.append(f"episode {episode} maps to multiple dataset indices")
    episode_counts: dict[str, int] = {}
    for episode in episodes:
        episode_counts[episode] = episode_counts.get(episode, 0) + 1
    if any(episode_counts.get(episode, 0) != 2 for episode in expected_episodes):
        errors.append("each frozen episode must contribute exactly two nested records")
    for row, episode in enumerate(episodes):
        # Seeds identify an episode's initial state.  Each episode has two
        # nested records, so record-000/record-001 must not advance the seed
        # namespace.  Derive the local index from the episode ID itself and
        # compare both rows against namespace + episode-local-index.
        match = re.search(r":([0-9]{3})$", episode)
        local = int(match.group(1)) if match else -1
        if local < 0 or local >= 6:
            errors.append(f"invalid local index in {episode}")
            continue
        if int(indices[row]) != expected_indices[local]:
            errors.append(f"{episode} dataset index mismatch")
        expected_env = 800000 if split == "frt_cal" else 900000
        expected_cem = 810000 if split == "frt_cal" else 910000
        if int(env_seeds[row]) != expected_env + local or int(cem_seeds[row]) != expected_cem + local:
            errors.append(f"{episode} seed metadata mismatch")
    records = meta.get("records")
    if records is None and isinstance(meta.get("record_keys"), Sequence) and not isinstance(meta.get("record_keys"), (str, bytes)):
        # The frozen Stage B runner stores compact parallel metadata arrays
        # rather than duplicating each record object in JSON.
        records = meta.get("record_keys")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        errors.append("JSON record metadata is missing")
    elif len(records) != count:
        errors.append("JSON and NPZ record counts disagree")
    return errors, {
        "record_count": count,
        "episode_count": len(set(episodes)),
        "episodes": sorted(set(episodes)),
        "dataset_indices": sorted(set(int(item) for item in indices.tolist())),
    }


def _weighted_mse(error: np.ndarray, wz: np.ndarray, active: np.ndarray) -> np.ndarray:
    weighted = np.where(active, error * wz, 0.0)
    # Match frt_core.weighted_mse exactly: square then mean over the complete
    # P*D slot.  Wz[action]=0 excludes action values while preserving the
    # runner's denominator; dividing by active-count would change every gate.
    denominator = max(int(np.asarray(active, dtype=bool).size), 1)
    return np.sum(weighted * weighted, axis=(-2, -1), dtype=np.float64) / denominator


def _vector_close(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=VECTOR_RTOL, atol=VECTOR_ATOL, equal_nan=False))


def _scalar_json(value: Any) -> Any:
    """Extract a scalar string from an NPZ metadata field without pickle."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("metadata_json must be a scalar NPZ field")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return json.loads(str(item))


def _runner_bank_meta(arrays: Any) -> Mapping[str, Any]:
    if "metadata_json" not in arrays:
        raise ValueError("runner bank is missing metadata_json")
    value = _scalar_json(arrays["metadata_json"])
    if not isinstance(value, Mapping):
        raise ValueError("runner bank metadata_json must be an object")
    return value


def _runner_record_audit(meta: Mapping[str, Any], arrays: Any, expected_split: str) -> tuple[list[str], dict[str, Any]]:
    """Audit the actual frt_runner bank contract (including repeated rows)."""
    errors: list[str] = []
    try:
        bank_meta = _runner_bank_meta(arrays)
    except Exception as exc:
        return [f"bank metadata parse failed: {type(exc).__name__}: {exc}"], {}
    split = str(bank_meta.get("split", ""))
    if split not in {expected_split, "stage_a"}:
        errors.append(f"bank split is {split!r}, expected {expected_split!r}")
    rows = bank_meta.get("metadata")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        errors.append("bank metadata list is missing")
        return errors, {}
    episodes: list[str] = []
    indices: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("bank metadata contains a non-object row")
            continue
        episode = str(row.get("episode_id", ""))
        if not episode:
            errors.append("bank metadata row has no episode_id")
            continue
        try:
            index = int(row["dataset_index"])
        except Exception:
            errors.append(f"{episode} has no numeric dataset_index")
            continue
        episodes.append(episode)
        indices.append(index)
    if not episodes:
        return errors + ["bank metadata has no records"], {"record_count": 0}
    expected_indices = set(EXPECTED_TARGETS["frt_cal" if expected_split == "stage_a" else "frt_dev"])
    expected_episode_prefix = "frt_cal:" if expected_split == "stage_a" else "frt_dev:"
    if set(indices) != expected_indices:
        errors.append(f"dataset index set {sorted(set(indices))} != {sorted(expected_indices)}")
    if any(not episode.startswith(expected_episode_prefix) for episode in episodes):
        errors.append("bank contains an unexpected episode namespace")
    episode_to_index: dict[str, int] = {}
    for episode, index in zip(episodes, indices):
        prior = episode_to_index.setdefault(episode, index)
        if prior != index:
            errors.append(f"episode {episode} maps to multiple dataset indices")
    if set(episode_to_index) != set(EXPECTED_EPISODES["frt_cal" if expected_split == "stage_a" else "frt_dev"]):
        errors.append("bank does not cover exactly six frozen episode identities")
    return errors, {
        "record_count": len(episodes),
        "episode_count": len(episode_to_index),
        "record_episode_ids": episodes,
        "episodes": sorted(episode_to_index),
        "dataset_indices": sorted(set(indices)),
        "metadata_split": split,
    }


def _runner_wz_view(wz: np.ndarray, slot_shape: tuple[int, int], errors: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Normalize the runner's [D] Wz or explicit [P,D] Wz+mask contract."""
    if wz.shape == (slot_shape[1],):
        # The current runner emits [D].  Exact zeros in the tail are the
        # action-coordinate mask; if no zeros are present, retain a visible
        # evidence limitation instead of guessing the action layout.
        if not np.isfinite(wz).all() or np.any(wz < 0):
            errors.append("runner Wz[D] is not finite and non-negative")
        zero_columns = np.asarray(wz == 0, dtype=bool)
        if np.any(zero_columns):
            expanded = np.broadcast_to(wz.reshape((1, slot_shape[1])), slot_shape).copy()
            return expanded, np.broadcast_to(zero_columns.reshape((1, slot_shape[1])), slot_shape).copy()
        return wz.reshape((1, slot_shape[1])), None
    if wz.shape == slot_shape:
        if not np.isfinite(wz).all() or np.any(wz < 0):
            errors.append("runner Wz[P,D] is not finite and non-negative")
        return wz, np.asarray(wz == 0, dtype=bool)
    errors.append(f"runner Wz shape {wz.shape} disagrees with slot {slot_shape}")
    return None, None


def _runner_stage_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    """Independent audit of frt_runner.py's stage_a_summary + bank_stage_a."""
    errors: list[str] = []
    summary_path = artifact / "stage_a_summary.json"
    bank_path = artifact / "bank_stage_a.npz"
    if not summary_path.is_file() or not bank_path.is_file():
        return {"pass": False, "complete": False, "resource": True, "errors": ["stage_a_summary.json or bank_stage_a.npz is missing"], "record_count": 0}
    summary = _json(summary_path)
    if summary.get("schema") != "frt-stage-a-v1":
        errors.append("runner Stage A schema mismatch")
    if summary.get("status") != "complete":
        errors.append("runner Stage A status is incomplete")
    errors.extend(_runtime_errors(summary.get("runtime_identity"), manifest))
    gates = summary.get("gates")
    if not isinstance(gates, Mapping):
        errors.append("runner Stage A gates are missing")
    else:
        required_gates = ("source_rollout_agreement", "physical_replay", "delta_zero", "fp_transport_null", "history_action_invariants", "hard_materialize_reload")
        for name in required_gates:
            gate = gates.get(name)
            if not isinstance(gate, Mapping) or gate.get("passed") is not True:
                errors.append(f"Stage A gate {name} did not pass")
    record_summary: dict[str, Any] = {}
    try:
        with np.load(bank_path, allow_pickle=False) as arrays:
            required = ("history", "x_history", "action", "next_action", "fp_current", "q0_current", "q0_transport", "delta", "fp_transport", "random_delta", "random_transport", "wz", "action_mask", "active_mask", "metadata_json")
            missing = [key for key in required if key not in arrays]
            if missing:
                errors.extend(f"missing runner Stage A bank array {key}" for key in missing)
                return {"pass": False, "complete": False, "resource": False, "errors": _compact_errors(errors), "record_count": 0}
            meta_errors, record_summary = _runner_record_audit(arrays["metadata_json"], arrays, "stage_a")
            errors.extend(meta_errors)
            history = _finite("history", arrays["history"], errors)
            x_history = _finite("x_history", arrays["x_history"], errors)
            action = _finite("action", arrays["action"], errors)
            next_action = _finite("next_action", arrays["next_action"], errors)
            fp_current = _finite("fp_current", arrays["fp_current"], errors)
            q0_current = _finite("q0_current", arrays["q0_current"], errors)
            q0_transport = _finite("q0_transport", arrays["q0_transport"], errors)
            delta = _finite("delta", arrays["delta"], errors)
            fp_transport = _finite("fp_transport", arrays["fp_transport"], errors)
            random_delta = _finite("random_delta", arrays["random_delta"], errors)
            random_transport = _finite("random_transport", arrays["random_transport"], errors)
            wz_raw = _finite("wz", arrays["wz"], errors)
            if history.ndim != 4 or x_history.shape != history.shape:
                errors.append("history/x_history are not matching [N,T,P,D]")
            if fp_current.ndim != 4:
                errors.append("fp_current is not [N,1,P,D]")
            slot_shape = tuple(fp_current.shape[-2:]) if fp_current.ndim == 4 else ()
            if slot_shape and any(array.shape != fp_current.shape for array in (q0_current, q0_transport, delta, fp_transport, random_delta, random_transport)):
                errors.append("one or more current/delta/transport arrays disagree with fp_current")
            if history.ndim == 4 and fp_current.ndim == 4 and history.shape[0] != fp_current.shape[0]:
                errors.append("history and fp_current record counts disagree")
            if action.ndim != 3 or next_action.shape != action.shape or action.shape[0] != fp_current.shape[0]:
                errors.append("action/next_action are not matching [N,1,A]")
            wz, inferred_action_mask = _runner_wz_view(wz_raw, slot_shape, errors) if slot_shape else (None, None)
            if wz is not None and delta.ndim == 4:
                if not _vector_close(q0_current - fp_current, delta):
                    errors.append("q0_current-fp_current does not equal delta")
                if history.ndim == 4 and x_history.shape == history.shape:
                    if history.shape[1] == 1:
                        if not _vector_close(x_history[:, -1], fp_current[:, -1]):
                            errors.append("x_history does not equal appended fp_current for num_hist=1")
                    elif not _vector_close(x_history[:, :-1], history[:, 1:]) or not _vector_close(x_history[:, -1], fp_current[:, -1]):
                        errors.append("x_history is not the shifted history with fp_current appended")
                wz_broadcast = wz.reshape((1, 1, 1, slot_shape[1])) if wz.shape == (1, slot_shape[1]) else wz.reshape((1, 1, slot_shape[0], slot_shape[1]))
                q0_norm = np.sqrt(np.sum((delta * wz_broadcast) ** 2, axis=(-2, -1)))
                random_norm = np.sqrt(np.sum((random_delta * wz_broadcast) ** 2, axis=(-2, -1)))
                if not np.isfinite(q0_norm).all() or not np.isfinite(random_norm).all():
                    errors.append("Q0/random weighted norms are non-finite")
                elif np.any(q0_norm <= EPS):
                    errors.append("one or more Q0 weighted residual norms are zero")
                elif not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append("random bank does not match Q0 weighted norm")
                action_mask = np.asarray(arrays["action_mask"], dtype=bool)
                active_mask = np.asarray(arrays["active_mask"], dtype=bool)
                if action_mask.shape != slot_shape or active_mask.shape != slot_shape:
                    errors.append("runner action_mask/active_mask shape disagrees with latent slot")
                elif np.any(action_mask & active_mask) or np.any(~(action_mask | active_mask)):
                    errors.append("runner action/active masks are not disjoint and covering")
                elif np.any(wz[action_mask] != 0) or np.any(wz[active_mask] <= 0):
                    errors.append("runner Wz violates action-zero/active-positive contract")
                elif inferred_action_mask is not None and not np.array_equal(action_mask, inferred_action_mask):
                    errors.append("runner action_mask disagrees with Wz zero coordinates")
                if np.any(delta[..., action_mask] != 0) or np.any(random_delta[..., action_mask] != 0):
                    errors.append("delta has nonzero action coordinates")
                clean_error = q0_current - fp_current
                transport_error = q0_transport - fp_transport
                clean_component = float(np.mean(_weighted_mse(clean_error, wz, active_mask), dtype=np.float64))
                transport_component = float(np.mean(_weighted_mse(transport_error, wz, active_mask), dtype=np.float64))
                lambda_raw = max(clean_component, EPS) / max(transport_component, EPS)
                lambda_expected = float(np.clip(lambda_raw, 0.25, 4.0))
                freeze = summary.get("calibration_freeze")
                if not isinstance(freeze, Mapping):
                    errors.append("Stage A calibration_freeze is missing")
                else:
                    for name, expected_value in (("mean_L_clean_Q0", clean_component), ("mean_L_transport_Q0", transport_component), ("lambda_raw", lambda_raw), ("lambda_T", lambda_expected)):
                        if name not in freeze or not math.isclose(float(freeze[name]), expected_value, rel_tol=VECTOR_RTOL, abs_tol=VECTOR_ATOL):
                            errors.append(f"calibration_freeze.{name} disagrees with raw Q0 vectors")
                    if bool(freeze.get("lambda_clamped")) != bool(lambda_expected != lambda_raw):
                        errors.append("calibration_freeze.lambda_clamped disagrees with raw ratio")
                record_summary.update({"mean_L_clean_Q0": clean_component, "mean_L_transport_Q0": transport_component, "lambda_raw": lambda_raw, "lambda_T": lambda_expected})
            record_summary.update({
                "history_shape": list(history.shape),
                "slot_shape": list(slot_shape),
                "wz_shape": list(wz_raw.shape),
                "wz_sha256": hashlib.sha256(np.asarray(wz_raw).tobytes()).hexdigest(),
                "mean_q0_weighted_norm": float(np.mean(q0_norm)) if "q0_norm" in locals() else None,
            })
    except Exception as exc:
        errors.append(f"runner Stage A bank audit exception: {type(exc).__name__}: {exc}")
    complete = summary.get("status") == "complete"
    return {"pass": bool(complete and not errors), "complete": bool(complete and not errors), "resource": False, "errors": _compact_errors(errors), **record_summary}


def _stage_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    errors: list[str] = []
    if (artifact / "stage_a_summary.json").is_file() or (artifact / "bank_stage_a.npz").is_file():
        return _runner_stage_audit(artifact, manifest, manifest_hash)
    meta_path = artifact / "stage_a.json"
    arrays_path = artifact / "stage_a.npz"
    if not meta_path.is_file() or not arrays_path.is_file():
        return {
            "pass": False,
            "complete": False,
            "resource": True,
            "errors": ["stage_a.json or stage_a.npz is missing"],
            "record_count": 0,
        }
    meta = _json(meta_path)
    if meta.get("schema") != "frt-stage-a-raw-v1":
        errors.append("Stage A schema mismatch")
    if meta.get("manifest_sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("manifest_sha256"))):
        errors.append("Stage A manifest_sha256 is malformed")
    if meta.get("manifest_sha256") not in (None, manifest_hash):
        # A manifest revision may be written after A and before B. Keep the
        # fact visible, but do not invalidate the already audited A arrays.
        errors.append("Stage A was produced under an earlier manifest revision")
    errors.extend(_runtime_errors(meta.get("runtime_identity"), manifest))
    fingerprints = meta.get("target_fingerprints", [])
    if isinstance(fingerprints, Mapping):
        fingerprints = list(fingerprints.values())
    if not isinstance(fingerprints, Sequence) or isinstance(fingerprints, (str, bytes)):
        errors.append("Stage A target_fingerprints are missing")
    elif len(fingerprints) != 6 or len(set(map(str, fingerprints))) != 6:
        errors.append("Stage A target fingerprints are not six unique values")
    contract = meta.get("history_contract")
    if not isinstance(contract, Mapping):
        errors.append("history_contract is missing")
    else:
        for key in ("full_history", "shift_concat_real", "same_action_rng_replay", "x_delta_legal", "action_replaced_before_delta"):
            if contract.get(key) is not True:
                errors.append(f"history_contract.{key} is not true")
    try:
        with np.load(arrays_path, allow_pickle=False) as arrays:
            required = (
                "record_keys", "episode_ids", "dataset_indices", "env_seeds", "cem_seeds",
                "history_fp", "history_q0_slot", "next_fp", "next_q0", "delta_q0",
                "delta_random", "active_mask", "action_mask", "fp_output_variance", "wz",
                "q0_clean_error", "q0_transport_error",
            )
            missing = [key for key in required if key not in arrays]
            if missing:
                errors.extend(f"missing Stage A array {key}" for key in missing)
                return {"pass": False, "complete": False, "resource": False, "errors": _compact_errors(errors), "record_count": 0}
            record_errors, record_summary = _record_audit(meta, arrays, "frt_cal")
            errors.extend(record_errors)
            history_fp = _finite("history_fp", arrays["history_fp"], errors)
            history_q0 = _finite("history_q0_slot", arrays["history_q0_slot"], errors)
            next_fp = _finite("next_fp", arrays["next_fp"], errors)
            next_q0 = _finite("next_q0", arrays["next_q0"], errors)
            delta_q0 = _finite("delta_q0", arrays["delta_q0"], errors)
            delta_random = _finite("delta_random", arrays["delta_random"], errors)
            variance = _finite("fp_output_variance", arrays["fp_output_variance"], errors)
            wz = _finite("wz", arrays["wz"], errors)
            active = np.asarray(arrays["active_mask"], dtype=bool)
            action = np.asarray(arrays["action_mask"], dtype=bool)
            shape = next_fp.shape[1:] if next_fp.ndim == 3 else ()
            if history_fp.ndim != 4 or history_q0.shape != history_fp.shape:
                errors.append("history arrays are not both [N,T,P,D]")
            if next_fp.ndim != 3 or next_q0.shape != next_fp.shape or delta_q0.shape != next_fp.shape or delta_random.shape != next_fp.shape:
                errors.append("next/delta arrays are not all [N,P,D]")
            if len(shape) != 2 or active.shape != shape or action.shape != shape or wz.shape != shape or variance.shape != shape:
                errors.append("mask/Wz/variance shape does not match [P,D]")
            if history_fp.ndim == 4 and next_fp.ndim == 3 and history_fp.shape[0] != next_fp.shape[0]:
                errors.append("history and next record counts disagree")
            if active.shape == action.shape and (np.any(active & action) or np.any(~(active | action))):
                errors.append("active/action masks are not disjoint and covering")
            if active.shape == action.shape and np.any(wz[action] != 0):
                errors.append("Wz action coordinates are not zero")
            if active.shape == action.shape and np.any(wz[active] <= 0):
                errors.append("Wz active coordinates are not positive")
            if delta_q0.shape == next_q0.shape and action.shape == delta_q0.shape[1:]:
                if not np.array_equal(delta_q0[:, action], np.zeros_like(delta_q0[:, action])):
                    errors.append("delta_q0 action coordinates are not exact zero")
                if not np.array_equal(delta_random[:, action], np.zeros_like(delta_random[:, action])):
                    errors.append("delta_random action coordinates are not exact zero")
                if not _vector_close(next_q0 - next_fp, delta_q0):
                    errors.append("next_q0-next_fp does not equal delta_q0")
            if history_fp.ndim == 4 and history_q0.shape == history_fp.shape and delta_q0.ndim == 3:
                history_diff = history_q0 - history_fp
                if history_fp.shape[1] < 1:
                    errors.append("history has no new slot")
                else:
                    if not np.allclose(history_diff[:, :-1], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                        errors.append("old history slots changed under delta injection")
                    if not _vector_close(history_diff[:, -1], delta_q0):
                        errors.append("delta was not injected into only the newest slot")
            if next_fp.ndim == 3 and variance.shape == next_fp.shape[1:]:
                computed_var = np.var(next_fp.astype(np.float64), axis=0)
                expected_wz = np.zeros_like(computed_var, dtype=np.float64)
                if active.shape == computed_var.shape:
                    expected_wz[active] = 1.0 / np.maximum(np.sqrt(np.maximum(computed_var[active], 0.0)), WZ_STD_FLOOR)
                    if not _vector_close(wz, expected_wz):
                        errors.append("Wz does not match frozen CAL population variance and floor")
            if delta_q0.ndim == 3 and delta_random.shape == delta_q0.shape and wz.shape == delta_q0.shape[1:]:
                q0_norm = np.sqrt(np.sum(np.where(active, delta_q0 * wz, 0.0) ** 2, axis=(-2, -1)))
                random_norm = np.sqrt(np.sum(np.where(active, delta_random * wz, 0.0) ** 2, axis=(-2, -1)))
                if not np.isfinite(q0_norm).all() or np.any(q0_norm <= EPS):
                    errors.append("one or more Q0 residual weighted norms are zero/invalid")
                elif not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append("random bank does not match Q0 weighted norm per record")
            clean_error = _finite("q0_clean_error", arrays["q0_clean_error"], errors)
            transport_error = _finite("q0_transport_error", arrays["q0_transport_error"], errors)
            if clean_error.shape != next_fp.shape or transport_error.shape != next_fp.shape:
                errors.append("Q0 error vectors do not match [N,P,D]")
            if clean_error.shape == next_fp.shape and transport_error.shape == next_fp.shape and wz.shape == next_fp.shape[1:]:
                clean_component = float(np.mean(_weighted_mse(clean_error, wz, active), dtype=np.float64))
                transport_component = float(np.mean(_weighted_mse(transport_error, wz, active), dtype=np.float64))
                lambda_raw = max(clean_component, EPS) / max(transport_component, EPS)
                lambda_expected = float(np.clip(lambda_raw, 0.25, 4.0))
                spec = meta.get("lambda_spec")
                if not isinstance(spec, Mapping):
                    errors.append("lambda_spec is missing")
                else:
                    for key, expected in (
                        ("mean_L_clean_Q0", clean_component),
                        ("mean_L_transport_Q0", transport_component),
                        ("lambda_raw", lambda_raw),
                        ("lambda_T", lambda_expected),
                    ):
                        if key not in spec or not math.isclose(float(spec[key]), expected, rel_tol=VECTOR_RTOL, abs_tol=VECTOR_ATOL):
                            errors.append(f"lambda_spec.{key} disagrees with raw A vectors")
                    if spec.get("clamped") is not (lambda_raw < 0.25 or lambda_raw > 4.0):
                        errors.append("lambda_spec.clamped flag disagrees with lambda_raw")
                record_summary.update({
                    "mean_L_clean_Q0": clean_component,
                    "mean_L_transport_Q0": transport_component,
                    "lambda_raw": lambda_raw,
                    "lambda_T": lambda_expected,
                })
    except Exception as exc:
        errors.append(f"Stage A array audit exception: {type(exc).__name__}: {exc}")
    return {
        "pass": not errors,
        "complete": bool(meta.get("complete") is True and not errors),
        "resource": False,
        "errors": _compact_errors(errors),
        **record_summary,
    }


def _fit_run_audit(meta: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    runs = meta.get("fit_runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or len(runs) != 9:
        return ["fit_runs must contain exactly nine method-seed rows"], {}
    expected_updates = None
    fit = manifest.get("fit")
    if isinstance(fit, Mapping) and isinstance(fit.get("fit_updates"), int):
        expected_updates = int(fit["fit_updates"])
    configs: dict[str, tuple[Any, ...]] = {}
    seen: set[tuple[str, int]] = set()
    for row in runs:
        if not isinstance(row, Mapping):
            errors.append("fit_runs contains a non-object")
            continue
        method = str(row.get("method"))
        try:
            seed = int(row.get("seed"))
        except Exception:
            seed = -1
        key = (method, seed)
        seen.add(key)
        if method not in METHODS or seed not in SEEDS:
            errors.append(f"unexpected fit run {method}:{seed}")
        if row.get("status") != "complete":
            errors.append(f"{method}:{seed} is not complete")
        try:
            completed = int(row.get("fit_updates_completed"))
            declared = int(row.get("fit_updates_expected"))
        except Exception:
            completed = declared = -1
            errors.append(f"{method}:{seed} lacks numeric fit update ledger")
        if declared <= 0 or completed != declared:
            errors.append(f"{method}:{seed} did not complete all fit updates")
        if expected_updates is not None and declared != expected_updates:
            errors.append(f"{method}:{seed} fit update count disagrees with manifest")
        for flag in ("binary_final", "alpha_discarded", "reload_equal", "exact_w4"):
            if row.get(flag) is not True:
                errors.append(f"{method}:{seed} {flag} is not true")
        for key_name in ("checkpoint_sha256", "hard_map_sha256", "batch_schedule_hash", "initialization_id", "soft_schedule_id"):
            if not str(row.get(key_name, "")):
                errors.append(f"{method}:{seed} missing {key_name}")
        if row.get("checkpoint_sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(row.get("checkpoint_sha256"))):
            errors.append(f"{method}:{seed} checkpoint hash malformed")
        if row.get("hard_map_sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(row.get("hard_map_sha256"))):
            errors.append(f"{method}:{seed} hard map hash malformed")
        try:
            batch_size = int(row.get("batch_size"))
            exposure_count = int(row.get("exposure_count"))
        except Exception:
            batch_size = exposure_count = -1
            errors.append(f"{method}:{seed} lacks batch/exposure ledger")
        if batch_size not in (1, 2, 4):
            errors.append(f"{method}:{seed} batch size is outside A candidates")
        if exposure_count <= 0:
            errors.append(f"{method}:{seed} exposure count is not positive")
        configs.setdefault(str(seed), tuple(row.get(key) for key in (
            "batch_size", "batch_schedule_hash", "exposure_count",
            "scale_learning_rate", "rounding_learning_rate",
            "initialization_id", "soft_schedule_id", "fit_updates_expected",
        )))
        if configs[str(seed)] != tuple(row.get(key) for key in (
            "batch_size", "batch_schedule_hash", "exposure_count",
            "scale_learning_rate", "rounding_learning_rate",
            "initialization_id", "soft_schedule_id", "fit_updates_expected",
        )):
            errors.append(f"fit schedule/optimizer differs across methods for seed {seed}")
    expected_keys = {(method, seed) for method in METHODS for seed in SEEDS}
    if seen != expected_keys:
        errors.append("fit_runs do not cover exactly methods x seeds")
    fit_config = meta.get("fit_config")
    if not isinstance(fit_config, Mapping):
        errors.append("fit_config is missing")
    else:
        for key in ("fit_updates", "batch_size", "scale_learning_rate", "rounding_learning_rate", "schedule_hash"):
            if key not in fit_config:
                errors.append(f"fit_config.{key} is missing")
        if expected_updates is not None and fit_config.get("fit_updates") != expected_updates:
            errors.append("fit_config.fit_updates disagrees with manifest")
        for seed in SEEDS:
            row = configs.get(str(seed))
            if row is not None and fit_config.get("batch_size") != row[0]:
                errors.append(f"fit_config batch size disagrees for seed {seed}")
    return errors, {"fit_run_count": len(runs), "fit_config": dict(fit_config) if isinstance(fit_config, Mapping) else None}


def _runner_fit_audit(artifact: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Audit the nine actual method ledgers before reading any score matrix."""
    errors: list[str] = []
    fit = manifest.get("fit", {}) if isinstance(manifest.get("fit"), Mapping) else {}
    summary_methods = summary.get("methods", fit.get("methods", []))
    aliases = {"random": "random_same_norm", "random_direction": "random_same_norm", "FRT": "frt"}
    methods = tuple(aliases.get(str(value), str(value)) for value in summary_methods) if isinstance(summary_methods, Sequence) and not isinstance(summary_methods, (str, bytes)) else ()
    if methods != METHODS:
        errors.append(f"Stage B method order is {methods}, expected {METHODS}")
    summary_seeds = summary.get("seeds", fit.get("seeds", []))
    try:
        seeds = tuple(int(value) for value in summary_seeds)
    except Exception:
        seeds = ()
    if seeds != SEEDS:
        errors.append(f"Stage B fit seed order is {seeds}, expected {SEEDS}")
    expected_updates = summary.get("fit_updates")
    if expected_updates is not None:
        try:
            expected_updates = int(expected_updates)
            if expected_updates <= 0:
                raise ValueError
        except Exception:
            errors.append("Stage B fit_updates is not a positive integer")
            expected_updates = None
    rows: list[Mapping[str, Any]] = []
    for method in METHODS:
        runner_method = "random" if method == "random_same_norm" else method
        for seed in SEEDS:
            path = artifact / "methods" / f"{runner_method}_seed_{seed}.json"
            if not path.is_file():
                # Accept an explicitly canonical random_same_norm filename if
                # a runner revision chooses to expose the protocol name.
                path = artifact / "methods" / f"{method}_seed_{seed}.json"
            if not path.is_file():
                errors.append(f"missing method ledger {method}:{seed}")
                continue
            try:
                row = _json(path)
            except Exception as exc:
                errors.append(f"cannot parse {path.name}: {type(exc).__name__}: {exc}")
                continue
            rows.append(row)
            actual_method = aliases.get(str(row.get("method")), str(row.get("method")))
            if actual_method != method or int(row.get("seed", -1)) != seed:
                errors.append(f"method ledger identity mismatch for {method}:{seed}")
            try:
                steps = int(row.get("steps"))
                batch = int(row.get("batch_size"))
                lr = float(row.get("lr"))
                lam = float(row.get("lambda_transport"))
            except Exception:
                errors.append(f"{method}:{seed} lacks numeric fit config")
                continue
            if steps <= 0 or (expected_updates is not None and steps != expected_updates):
                errors.append(f"{method}:{seed} does not use the frozen positive update count")
            if batch not in (1, 2, 4):
                errors.append(f"{method}:{seed} batch size is outside Stage A candidates")
            if not math.isfinite(lr) or lr <= 0:
                errors.append(f"{method}:{seed} learning rate is invalid")
            expected_lambda = 0.0 if method == "clean" else 1.0
            if not math.isclose(lam, expected_lambda, rel_tol=0.0, abs_tol=0.0):
                errors.append(f"{method}:{seed} lambda_transport differs from the frozen loss")
            trace = row.get("fit_trace")
            if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)) or not trace:
                errors.append(f"{method}:{seed} fit_trace is missing")
            else:
                try:
                    final_step = int(trace[-1].get("step"))
                except Exception:
                    final_step = -1
                if final_step != steps:
                    errors.append(f"{method}:{seed} fit_trace does not reach the final update")
            ledger = row.get("hard_ledger")
            if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)) or not ledger:
                errors.append(f"{method}:{seed} hard W4 ledger is missing")
            else:
                for item in ledger:
                    if not isinstance(item, Mapping) or item.get("hard") is not True or item.get("q_min") != -7 or item.get("q_max") != 7:
                        errors.append(f"{method}:{seed} hard ledger has a non-W4 row")
                        break
            checkpoint = row.get("checkpoint")
            if not checkpoint or not Path(str(checkpoint)).is_file():
                errors.append(f"{method}:{seed} hard checkpoint is missing")
    # The method loop gives nine independent files; compare the effective
    # schedule fields by seed so a method cannot silently buy extra updates.
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row.get("seed", -1)) == seed]
        if len(seed_rows) == 3:
            fields = [(row.get("steps"), row.get("batch_size"), row.get("lr"), row.get("rounding")) for row in seed_rows]
            if any(item != fields[0] for item in fields[1:]):
                errors.append(f"method optimizer/schedule differs within fit seed {seed}")
    return errors, {"fit_run_count": len(rows), "fit_updates": expected_updates}


def _runner_bank_array_audit(arrays: Any, expected_split: str, errors: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Validate a Stage B bank using the same raw bank schema as Stage A."""
    required = ("history", "x_history", "action", "next_action", "fp_current", "q0_current", "q0_transport", "delta", "fp_transport", "random_delta", "random_transport", "wz", "action_mask", "active_mask", "metadata_json")
    missing = [key for key in required if key not in arrays]
    if missing:
        errors.extend(f"missing runner {expected_split} bank array {key}" for key in missing)
        return {}, None
    meta_errors, record_summary = _runner_record_audit(arrays["metadata_json"], arrays, "stage_a" if expected_split == "stage_a" else "dev")
    errors.extend(meta_errors)
    fp = _finite(f"{expected_split}.fp_current", arrays["fp_current"], errors)
    q0 = _finite(f"{expected_split}.q0_current", arrays["q0_current"], errors)
    q0_transport = _finite(f"{expected_split}.q0_transport", arrays["q0_transport"], errors)
    delta = _finite(f"{expected_split}.delta", arrays["delta"], errors)
    random_delta = _finite(f"{expected_split}.random_delta", arrays["random_delta"], errors)
    if fp.ndim != 4 or q0.shape != fp.shape or q0_transport.shape != fp.shape or delta.shape != fp.shape or random_delta.shape != fp.shape:
        errors.append(f"{expected_split} current/delta arrays do not have one matching [N,1,P,D] shape")
        return record_summary, None
    slot_shape = tuple(fp.shape[-2:])
    wz_raw = _finite(f"{expected_split}.wz", arrays["wz"], errors)
    wz, inferred_action_mask = _runner_wz_view(wz_raw, slot_shape, errors)
    if wz is None:
        return record_summary, None
    wz_broadcast = wz.reshape((1, 1, 1, slot_shape[1])) if wz.shape == (1, slot_shape[1]) else wz.reshape((1, 1, slot_shape[0], slot_shape[1]))
    if not _vector_close(q0 - fp, delta):
        errors.append(f"{expected_split} q0_current-fp_current does not equal delta")
    action_mask = np.asarray(arrays["action_mask"], dtype=bool)
    active_mask = np.asarray(arrays["active_mask"], dtype=bool)
    if action_mask.shape != slot_shape or active_mask.shape != slot_shape:
        errors.append(f"{expected_split} action/active mask shape disagrees with slot")
    elif np.any(action_mask & active_mask) or np.any(~(action_mask | active_mask)):
        errors.append(f"{expected_split} action/active masks are not disjoint and covering")
    elif np.any(wz[action_mask] != 0) or np.any(wz[active_mask] <= 0):
        errors.append(f"{expected_split} Wz violates action-zero/active-positive contract")
    elif inferred_action_mask is not None and not np.array_equal(action_mask, inferred_action_mask):
        errors.append(f"{expected_split} action_mask disagrees with Wz zero coordinates")
    if np.any(delta[..., action_mask] != 0) or np.any(random_delta[..., action_mask] != 0):
        errors.append(f"{expected_split} bank has nonzero action-coordinate residuals")
    q0_norm = np.sqrt(np.sum((delta * wz_broadcast) ** 2, axis=(-2, -1)))
    random_norm = np.sqrt(np.sum((random_delta * wz_broadcast) ** 2, axis=(-2, -1)))
    if np.any(q0_norm <= EPS):
        errors.append(f"{expected_split} has a zero Q0 weighted residual norm")
    elif not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
        errors.append(f"{expected_split} random bank does not match Q0 weighted norms")
    record_summary.update({"slot_shape": list(slot_shape), "wz_shape": list(wz_raw.shape), "wz_sha256": hashlib.sha256(np.asarray(wz_raw).tobytes()).hexdigest()})
    return record_summary, {"fp": fp, "q0": q0, "delta": delta, "random_delta": random_delta, "wz": wz}


def _runner_stage_b_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str, stage_a: Mapping[str, Any]) -> dict[str, Any]:
    """Audit current runner ledgers and require the real common-bank matrix."""
    errors: list[str] = []
    resource = False
    summary_path = artifact / "stage_b_summary.json"
    if not summary_path.is_file():
        return {"pass": False, "complete": False, "engineering_pass": False, "mechanism_pass": False, "resource": True, "status": "resource_incomplete", "errors": ["stage_b_summary.json is missing"]}
    summary = _json(summary_path)
    if summary.get("schema") != "frt-stage-b-v1":
        errors.append("runner Stage B schema mismatch")
    if summary.get("status") != "complete":
        resource = True
        errors.append("runner Stage B status is incomplete")
    errors.extend(_runtime_errors(summary.get("runtime_identity"), manifest))
    fit_errors, fit_summary = _runner_fit_audit(artifact, summary, manifest)
    errors.extend(fit_errors)
    if len(fit_errors) and any("missing method" in item or "does not use" in item or "fit_trace" in item for item in fit_errors):
        resource = True
    bank_summary: dict[str, Any] = {}
    bank_dev_path = artifact / "bank_dev.npz"
    if not bank_dev_path.is_file():
        resource = True
        errors.append("bank_dev.npz is missing")
    else:
        try:
            with np.load(bank_dev_path, allow_pickle=False) as arrays:
                bank_summary, _ = _runner_bank_array_audit(arrays, "dev", errors)
        except Exception as exc:
            errors.append(f"bank_dev audit exception: {type(exc).__name__}: {exc}")
    common_path = artifact / "common_bank.npz"
    if not common_path.is_file():
        errors.append("common_bank.npz is missing; mechanism matrix is not auditable")
    # The common matrix is deliberately checked only when the runner emits it.
    # Until then the result is engineering-incomplete rather than a mechanism
    # no-go.  The old raw-v1 reader below remains available for compatibility.
    matrix_summary: dict[str, Any] = {"available": False}
    if common_path.is_file():
        try:
            with np.load(common_path, allow_pickle=False) as arrays:
                required = ("bank_ids", "evaluator_ids", "bank_deltas", "active_mask", "action_mask", "wz", "clean_error", "transport_error", "clean_mse", "transport_mse")
                missing = [key for key in required if key not in arrays]
                if missing:
                    errors.extend(f"missing common-bank array {key}" for key in missing)
                else:
                    matrix_errors: list[str] = []
                    bank_ids = tuple(_strings(arrays["bank_ids"]))
                    evaluator_ids = tuple(_strings(arrays["evaluator_ids"]))
                    expected_evaluators = EVALUATORS
                    if bank_ids != BANKS:
                        matrix_errors.append("common bank_ids disagree with frozen bank order")
                    if evaluator_ids != expected_evaluators:
                        matrix_errors.append("common evaluator_ids disagree with frozen evaluator order")
                    bank_deltas = _finite("common.bank_deltas", arrays["bank_deltas"], matrix_errors)
                    active = np.asarray(arrays["active_mask"], dtype=bool)
                    action = np.asarray(arrays["action_mask"], dtype=bool)
                    wz_raw = _finite("common.wz", arrays["wz"], matrix_errors)
                    clean_error = _finite("common.clean_error", arrays["clean_error"], matrix_errors)
                    transport_error = _finite("common.transport_error", arrays["transport_error"], matrix_errors)
                    clean_mse = _finite("common.clean_mse", arrays["clean_mse"], matrix_errors)
                    transport_mse = _finite("common.transport_mse", arrays["transport_mse"], matrix_errors)
                    if bank_deltas.ndim == 5 and bank_deltas.shape[2] == 1:
                        bank_deltas = bank_deltas[:, :, 0]
                    if clean_error.ndim == 5 and clean_error.shape[2] == 1:
                        clean_error = clean_error[:, :, 0]
                    if transport_error.ndim == 6 and transport_error.shape[3] == 1:
                        transport_error = transport_error[:, :, :, 0]
                    if wz_raw.ndim == 1 and active.ndim == 2 and wz_raw.shape == (active.shape[1],):
                        wz = np.broadcast_to(wz_raw.reshape((1, active.shape[1])), active.shape)
                    else:
                        wz = wz_raw
                    if active.shape != action.shape or wz.shape != active.shape:
                        matrix_errors.append("common active/action/Wz shapes disagree")
                    elif np.any(active & action) or np.any(~(active | action)):
                        matrix_errors.append("common active/action masks are not disjoint and covering")
                    elif np.any(wz[action] != 0) or np.any(wz[active] <= 0):
                        matrix_errors.append("common Wz violates action-zero/active-positive contract")
                    if bank_deltas.ndim != 4 or bank_deltas.shape[0] != len(BANKS):
                        matrix_errors.append("common bank_deltas shape is not [11,N,P,D]")
                    elif active.shape != bank_deltas.shape[-2:]:
                        matrix_errors.append("common bank_deltas slot shape disagrees with masks")
                    else:
                        q0_weighted = np.where(active, bank_deltas[Q0_BANK] * wz, 0.0)
                        random_weighted = np.where(active, bank_deltas[RANDOM_BANK] * wz, 0.0)
                        q0_norm = np.sqrt(np.sum(q0_weighted * q0_weighted, axis=(-2, -1)))
                        random_norm = np.sqrt(np.sum(random_weighted * random_weighted, axis=(-2, -1)))
                        if np.any(q0_norm <= EPS):
                            matrix_errors.append("common Q0 bank contains a zero weighted norm")
                        elif not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                            matrix_errors.append("common random bank does not match Q0 weighted norms")
                        if np.any(bank_deltas[:, :, action] != 0):
                            matrix_errors.append("common bank has nonzero action-coordinate residuals")
                    if clean_error.ndim != 4 or clean_error.shape[0] != len(EVALUATORS):
                        matrix_errors.append("common clean_error shape is not [E,N,P,D]")
                    if transport_error.ndim != 5 or transport_error.shape[:2] != (len(EVALUATORS), len(BANKS)):
                        matrix_errors.append("common transport_error shape is not [E,B,N,P,D]")
                    if clean_error.ndim == 4 and clean_mse.shape != clean_error.shape[:2]:
                        matrix_errors.append("common clean_mse shape disagrees with raw clean_error")
                    if transport_error.ndim == 5 and transport_mse.shape != transport_error.shape[:3]:
                        matrix_errors.append("common transport_mse shape disagrees with raw transport_error")
                    if bank_summary.get("record_count") and clean_error.ndim == 4 and clean_error.shape[1] != bank_summary["record_count"]:
                        matrix_errors.append("common clean_error record count disagrees with DEV bank")
                    if bank_summary.get("record_count") and transport_error.ndim == 5 and transport_error.shape[2] != bank_summary["record_count"]:
                        matrix_errors.append("common transport_error record count disagrees with DEV bank")
                    if clean_error.ndim == 4 and transport_error.ndim == 5 and active.shape == clean_error.shape[-2:] and transport_error.shape[-2:] == active.shape:
                        expected_clean = _weighted_mse(clean_error, wz, active)
                        expected_transport = _weighted_mse(transport_error, wz, active)
                        if not np.allclose(clean_mse, expected_clean, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                            matrix_errors.append("common clean_mse is not recomputed from raw vectors")
                        if not np.allclose(transport_mse, expected_transport, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                            matrix_errors.append("common transport_mse is not recomputed from raw vectors")
                        if not np.allclose(clean_mse[0], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                            matrix_errors.append("FP32 clean matrix is not zero")
                        if not np.allclose(transport_mse[0], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                            matrix_errors.append("FP32 transport matrix is not zero")
                    errors.extend(matrix_errors)
                    matrix_summary = {"available": not matrix_errors, "bank_ids": list(bank_ids), "evaluator_ids": list(evaluator_ids), "errors": _compact_errors(matrix_errors)}
        except Exception as exc:
            errors.append(f"common-bank audit exception: {type(exc).__name__}: {exc}")
    engineering_pass = not errors and not resource
    status = "resource_incomplete" if resource else ("engineering_fail" if not engineering_pass else "mechanism_no_go")
    return {"pass": False, "complete": bool(summary.get("status") == "complete" and engineering_pass), "engineering_pass": engineering_pass, "mechanism_pass": False, "resource": resource, "status": status, "errors": _compact_errors(errors), **fit_summary, "bank": bank_summary, "common_matrix": matrix_summary}


def _stage_b_audit(
    artifact: Path,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    stage_a: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    resource = False
    # Stage B's frozen raw contract is stage_b.json + stage_b.npz.  Prefer it
    # whenever present; bank_dev/summary are runner progress/compatibility
    # outputs and must not shadow the complete common matrix.
    if not ((artifact / "stage_b.json").is_file() and (artifact / "stage_b.npz").is_file()) and ((artifact / "stage_b_summary.json").is_file() or (artifact / "bank_dev.npz").is_file()):
        return _runner_stage_b_audit(artifact, manifest, manifest_hash, stage_a)
    meta_path = artifact / "stage_b.json"
    arrays_path = artifact / "stage_b.npz"
    if not meta_path.is_file() or not arrays_path.is_file():
        return {"pass": False, "complete": False, "engineering_pass": False, "resource": True, "errors": ["stage_b.json or stage_b.npz is missing"]}
    meta = _json(meta_path)
    if meta.get("schema") != "frt-stage-b-raw-v1":
        errors.append("Stage B schema mismatch")
    if meta.get("manifest_sha256") != manifest_hash:
        errors.append("Stage B manifest_sha256 does not match the active manifest revision")
    errors.extend(_runtime_errors(meta.get("runtime_identity"), manifest))
    if meta.get("completion_status") != "complete" or meta.get("complete") is not True:
        resource = True
        errors.append("Stage B completion_status is incomplete")
    fit_errors, fit_summary = _fit_run_audit(meta, manifest)
    errors.extend(fit_errors)
    if any("not complete" in item or "did not complete" in item or "missing fit" in item for item in fit_errors):
        resource = True
    bank_ids = tuple(str(item) for item in meta.get("bank_ids", []))
    evaluator_ids = tuple(str(item) for item in meta.get("evaluator_ids", []))
    if bank_ids != BANKS:
        errors.append("bank_ids do not match the frozen common-bank order")
    if evaluator_ids != EVALUATORS:
        errors.append("evaluator_ids do not match the frozen all-evaluator order")
    if not meta.get("common_bank_all_evaluators") is True:
        errors.append("common_bank_all_evaluators is not true")
    if stage_a.get("pass") and meta.get("stage_a_wz_sha256") and meta.get("stage_a_wz_sha256") != stage_a.get("wz_sha256"):
        errors.append("Stage B Wz identity disagrees with Stage A")
    record_summary: dict[str, Any] = {}
    metric_summary: dict[str, Any] = {}
    try:
        with np.load(arrays_path, allow_pickle=False) as arrays:
            required = (
                "record_keys", "episode_ids", "dataset_indices",
                "active_mask", "action_mask", "wz", "bank_ids", "evaluator_ids",
                "bank_deltas", "clean_error", "transport_error", "clean_mse", "transport_mse",
            )
            missing = [key for key in required if key not in arrays]
            if missing:
                errors.extend(f"missing Stage B array {key}" for key in missing)
                return {
                    "pass": False,
                    "complete": False,
                    "engineering_pass": False,
                    "mechanism_pass": False,
                    "resource": resource,
                    "errors": _compact_errors(errors),
                    **fit_summary,
                }
            record_errors, record_summary = _record_audit(meta, arrays, "frt_dev")
            errors.extend(record_errors)
            record_keys = _strings(arrays["record_keys"])
            episodes = _strings(arrays["episode_ids"])
            n = len(record_keys)
            active = np.asarray(arrays["active_mask"], dtype=bool)
            action = np.asarray(arrays["action_mask"], dtype=bool)
            wz = _finite("wz", arrays["wz"], errors)
            array_bank_ids = tuple(_strings(arrays["bank_ids"]))
            array_evaluator_ids = tuple(_strings(arrays["evaluator_ids"]))
            if array_bank_ids != BANKS:
                errors.append("Stage B array bank_ids do not match the frozen common-bank order")
            if array_evaluator_ids != EVALUATORS:
                errors.append("Stage B array evaluator_ids do not match the frozen evaluator order")
            bank_deltas = _finite("bank_deltas", arrays["bank_deltas"], errors)
            clean_error = _finite("clean_error", arrays["clean_error"], errors)
            transport_error = _finite("transport_error", arrays["transport_error"], errors)
            clean_mse = _finite("clean_mse", arrays["clean_mse"], errors)
            transport_mse = _finite("transport_mse", arrays["transport_mse"], errors)
            if bank_deltas.ndim != 4 or bank_deltas.shape[0] != len(BANKS) or bank_deltas.shape[1] != n:
                errors.append("bank_deltas shape is not [11,N,P,D]")
            if clean_error.ndim != 4 or clean_error.shape[0] != len(EVALUATORS) or clean_error.shape[1] != n:
                errors.append("clean_error shape is not [11,N,P,D]")
            if transport_error.ndim != 5 or transport_error.shape[:3] != (len(EVALUATORS), len(BANKS), n):
                errors.append("transport_error shape is not [11,11,N,P,D]")
            if clean_mse.shape != (len(EVALUATORS), n):
                errors.append("clean_mse shape is not [11,N]")
            if transport_mse.shape != (len(EVALUATORS), len(BANKS), n):
                errors.append("transport_mse shape is not [11,11,N]")
            if len(record_keys) != len(set(record_keys)):
                errors.append("DEV record_keys are duplicated")
            if active.shape != action.shape or wz.shape != active.shape:
                errors.append("DEV masks/Wz shape mismatch")
            elif np.any(active & action) or np.any(~(active | action)):
                errors.append("DEV active/action masks are not disjoint and covering")
            if active.shape == wz.shape:
                if np.any(wz[action] != 0) or np.any(wz[active] <= 0):
                    errors.append("DEV Wz action/active values violate contract")
            # Stage B stores Wz as [P,D] for the common bank, while Stage A's
            # identity is the underlying [D] vector.  Verify the hash from the
            # emitted array itself (first row after checking row equality),
            # then compare it with both JSON and the independent A result.
            stage_a_wz_hash = meta.get("stage_a_wz_sha256")
            actual_wz_hash = None
            if wz.ndim == 1:
                actual_wz_hash = hashlib.sha256(np.asarray(wz).reshape(-1).tobytes()).hexdigest()
            elif wz.ndim == 2 and wz.shape[0] > 0:
                if not np.array_equal(wz, np.broadcast_to(wz[0:1], wz.shape)):
                    errors.append("DEV Wz rows disagree; cannot establish Stage A vector identity")
                actual_wz_hash = hashlib.sha256(np.asarray(wz[0]).reshape(-1).tobytes()).hexdigest()
            else:
                errors.append("DEV Wz has no usable Stage A identity vector")
            if not isinstance(stage_a_wz_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", stage_a_wz_hash):
                errors.append("Stage B stage_a_wz_sha256 is missing or malformed")
            elif actual_wz_hash is not None and actual_wz_hash != stage_a_wz_hash:
                errors.append("Stage B Wz array disagrees with stage_a_wz_sha256")
            if stage_a.get("pass") and stage_a.get("wz_sha256") and actual_wz_hash is not None and actual_wz_hash != stage_a.get("wz_sha256"):
                errors.append("Stage B Wz array disagrees with independently audited Stage A Wz")
            if bank_deltas.ndim == 4 and active.shape == bank_deltas.shape[2:]:
                if np.any(bank_deltas * action.reshape((1, 1) + action.shape)):
                    errors.append("a DEV bank has nonzero action coordinates")
                q0_norm = np.sqrt(np.sum(np.where(active, bank_deltas[Q0_BANK] * wz, 0.0) ** 2, axis=(-2, -1)))
                random_norm = np.sqrt(np.sum(np.where(active, bank_deltas[RANDOM_BANK] * wz, 0.0) ** 2, axis=(-2, -1)))
                if np.any(q0_norm <= EPS) or not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append("DEV random bank does not match shared q0 norm or has zero q0 norm")
            if clean_error.ndim == 4 and clean_error.shape[0] == len(EVALUATORS) and clean_error.shape[1] == n and wz.shape == clean_error.shape[2:]:
                recomputed_clean = _weighted_mse(clean_error, wz, active)
                if not np.allclose(clean_mse, recomputed_clean, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append("clean_mse does not match clean_error vectors")
            if transport_error.ndim == 5 and transport_error.shape[:3] == (len(EVALUATORS), len(BANKS), n) and wz.shape == transport_error.shape[3:]:
                recomputed_transport = _weighted_mse(transport_error, wz, active)
                if not np.allclose(transport_mse, recomputed_transport, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append("transport_mse does not match transport_error vectors")
            if clean_mse.shape == (len(EVALUATORS), n) and transport_mse.shape == (len(EVALUATORS), len(BANKS), n):
                if not np.allclose(clean_mse[0], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                    errors.append("FP32 clean error is not zero")
                if not np.allclose(transport_mse[0], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                    errors.append("FP32 transport error is not zero")
                for evaluator in FIT_EVALUATOR_INDEX:
                    row = FIT_EVALUATOR_INDEX[evaluator]
                    if not np.isfinite(clean_mse[row]).all() or not np.isfinite(transport_mse[row]).all():
                        errors.append(f"{evaluator} has non-finite metrics")
                # Direction diagnostics use every fresh bank, not only each
                # evaluator's own bank. They are descriptive, while the
                # mechanism gates below use the complete common matrix.
                cosine_summary: dict[str, float] = {}
                q0 = bank_deltas[Q0_BANK]
                q0w = np.where(active, q0 * wz, 0.0)
                q0norm = np.sqrt(np.sum(q0w * q0w, axis=(-2, -1)))
                for bank_index, bank_id in zip(FRESH_BANKS, BANKS[2:]):
                    freshw = np.where(active, bank_deltas[bank_index] * wz, 0.0)
                    denom = q0norm * np.sqrt(np.sum(freshw * freshw, axis=(-2, -1)))
                    cos = np.sum(q0w * freshw, axis=(-2, -1)) / np.maximum(denom, EPS)
                    if not np.isfinite(cos).all():
                        errors.append(f"{bank_id} direction cosine is non-finite")
                    cosine_summary[bank_id] = float(np.mean(cos, dtype=np.float64))
                metric_summary["fresh_direction_cosine_mean"] = cosine_summary
                metric_summary["clean_mse_macro"] = {
                    method: float(np.mean([np.mean(clean_mse[FIT_EVALUATOR_INDEX[f"{method}:{seed}"]]) for seed in SEEDS]))
                    for method in METHODS
                }
                gate_metrics = _mechanism_gates(clean_mse, transport_mse, episodes)
                metric_summary.update(gate_metrics)
    except Exception as exc:
        errors.append(f"Stage B array audit exception: {type(exc).__name__}: {exc}")
    engineering_pass = not errors and not resource
    mechanism_pass = bool(engineering_pass and metric_summary.get("all_gates_pass", False))
    if resource:
        status = "resource_incomplete"
    elif not engineering_pass:
        status = "engineering_fail"
    elif metric_summary.get("ambiguous"):
        status = "ambiguous"
    elif mechanism_pass:
        status = "conditional_signal"
    else:
        status = "mechanism_no_go"
    return {
        "pass": engineering_pass and mechanism_pass,
        "complete": bool(meta.get("complete") is True and not resource),
        "engineering_pass": engineering_pass,
        "mechanism_pass": mechanism_pass,
        "resource": resource,
        "status": status,
        "errors": _compact_errors(errors),
        **record_summary,
        **fit_summary,
        "metrics": metric_summary,
    }


def _episode_seed_values(
    metric: np.ndarray,
    episodes: Sequence[str],
    evaluator: str,
    bank_indices: Sequence[int] | None,
) -> dict[int, np.ndarray]:
    row = FIT_EVALUATOR_INDEX[evaluator]
    result: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        method = evaluator.split(":", 1)[0]
        row = FIT_EVALUATOR_INDEX[f"{method}:{seed}"]
        values = []
        for episode in EXPECTED_EPISODES["frt_dev"]:
            mask = np.asarray([item == episode for item in episodes], dtype=bool)
            if bank_indices is None:
                values.append(float(np.mean(metric[row, mask], dtype=np.float64)))
            else:
                values.append(float(np.mean(metric[row, list(bank_indices)][:, mask], dtype=np.float64)))
        result[seed] = np.asarray(values, dtype=np.float64)
    return result


def _gate(
    values: dict[int, np.ndarray],
    comparator: dict[int, np.ndarray],
    ratio: float,
    name: str,
) -> dict[str, Any]:
    seed_rows: dict[str, Any] = {}
    macro_value = float(np.mean([np.mean(row, dtype=np.float64) for row in values.values()]))
    macro_comparator = float(np.mean([np.mean(row, dtype=np.float64) for row in comparator.values()]))
    macro_threshold = ratio * macro_comparator + EPS
    seed_passes = 0
    ambiguous = False
    for seed in SEEDS:
        left = values[seed]
        right = comparator[seed]
        threshold = ratio * right + EPS
        delta = left - threshold
        passed = delta <= 0.0
        near = np.abs(delta) <= REL_TOL * np.maximum(np.abs(threshold), EPS)
        if bool(np.any(near)):
            ambiguous = True
        seed_rows[str(seed)] = {
            "episode_pass_count": int(np.sum(passed)),
            "required_episode_pass_count": 4,
            "macro_value": float(np.mean(left)),
            "macro_comparator": float(np.mean(right)),
            "pass": bool(np.sum(passed) >= 4 and np.mean(left) <= np.mean(right) * ratio + EPS),
            "ambiguous": bool(np.any(near)),
        }
        if seed_rows[str(seed)]["pass"]:
            seed_passes += 1
    macro_pass = macro_value <= macro_threshold
    if abs(macro_value - macro_threshold) <= REL_TOL * max(abs(macro_threshold), EPS):
        ambiguous = True
    return {
        "name": name,
        "ratio": ratio,
        "macro_value": macro_value,
        "macro_comparator": macro_comparator,
        "macro_pass": bool(macro_pass),
        "seed_pass_count": seed_passes,
        "required_seed_pass_count": 2,
        "seed_pass": seed_passes >= 2,
        "pass": bool(macro_pass and seed_passes >= 2 and not ambiguous),
        "ambiguous": ambiguous,
        "per_seed": seed_rows,
    }


def _mechanism_gates(clean_mse: np.ndarray, transport_mse: np.ndarray, episodes: Sequence[str]) -> dict[str, Any]:
    clean: dict[str, dict[int, np.ndarray]] = {}
    transport_q0: dict[str, dict[int, np.ndarray]] = {}
    transport_fresh: dict[str, dict[int, np.ndarray]] = {}
    for method in METHODS:
        evaluator = f"{method}:1201"
        clean[method] = _episode_seed_values(clean_mse, episodes, evaluator, None)
        transport_q0[method] = _episode_seed_values(transport_mse, episodes, evaluator, (Q0_BANK,))
        transport_fresh[method] = _episode_seed_values(transport_mse, episodes, evaluator, FRESH_BANKS)
    gates = [
        _gate(clean["frt"], clean["clean"], 1.10, "clean_tolerance_frt_vs_clean"),
        _gate(transport_q0["frt"], transport_q0["clean"], 0.95, "transport_q0_frt_vs_clean"),
        _gate(transport_q0["frt"], transport_q0["random_same_norm"], 0.95, "transport_q0_frt_vs_random"),
        _gate(transport_fresh["frt"], transport_fresh["clean"], 0.95, "transport_fresh_union_frt_vs_clean"),
        _gate(transport_fresh["frt"], transport_fresh["random_same_norm"], 0.95, "transport_fresh_union_frt_vs_random"),
    ]
    return {
        "gates": gates,
        "all_gates_pass": bool(all(bool(item["pass"]) for item in gates)),
        "ambiguous": bool(any(bool(item["ambiguous"]) for item in gates)),
    }


def _self_test() -> None:
    # Metadata-only smoke: avoids model loading and intentionally does not
    # create or inspect experiment arrays.
    manifest = {
        "schema": "frt-ccds-manifest-v1",
        "protocol_id": "frt-ccds-v1",
        "research_scope": {"test_split": False, "stage_c": False},
        "source_identity": {
            "recorded_epoch": 65,
            "dinov2_source_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "dtype": "float32",
            "decoder": None,
            "native_memory_claim": False,
        },
        "quantizer": {
            "bits": 4, "q_min": -7, "q_max": 7,
            "scheme": "symmetric_per_output_channel_weight_only",
            "target_blocks": [f"predictor.transformer.layers.{index}" for index in range(6)],
        },
        "splits": {
            "historical_reserved_minimum": [0, 49],
            "safety_reserved": [50, 59],
            "frt_reserved": [60, 71],
            "cal": {"split_id": "frt_cal", "dataset_indices": list(range(60, 66)), "episode_ids": list(EXPECTED_EPISODES["frt_cal"])},
            "dev": {"split_id": "frt_dev", "dataset_indices": list(range(66, 72)), "episode_ids": list(EXPECTED_EPISODES["frt_dev"])},
        },
        "fit": {
            "methods": list(METHODS), "seeds": list(SEEDS),
            "fit_updates": None,
            "fit_update_policy": {"candidate_values": [256, 512, 1000]},
            "batch_size_policy": {"candidate_values": [1, 2, 4]},
        },
        "loss": {"lambda_T": 1.0, "lambda_rule": {"clip": [0.25, 4.0]}, "wz": {"std_floor": WZ_STD_FLOOR}},
        "random_control": {"seed_base": 940000},
        "gates": {
            "clean_tolerance": {"per_seed_episode_requirement": 4, "seed_requirement": "at_least_2_of_3"},
            "transport_improvement": {"views": ["q0", "fresh_union"], "comparators": list(METHODS[0:2])},
        },
    }
    result = _manifest_audit(manifest)
    if not result["pass"]:
        raise AssertionError(result)
    print(json.dumps({"status": "complete", "checks": ["manifest contract", "split reserve", "W4 target", "A/B gate policy"]}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b", "all"), default="all")
    parser.add_argument("--artifact-dir", "--run-dir", dest="artifact_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.artifact_dir is None or args.manifest is None or args.output is None:
        parser.error("--artifact-dir, --manifest and --output are required")
    return args


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    # This must be the first operation that can inspect the run context.  The
    # guard uses real SLURM_JOB_ID/scontrol evidence and fails closed.
    require_allocation = _load_guard()
    allocation = require_allocation()
    artifact = args.artifact_dir.resolve()
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    manifest_hash = _manifest_sha256(manifest_path)
    manifest_result = _manifest_audit(manifest)
    result: dict[str, Any] = {
        "schema": "frt-verification-v1",
        "status": "engineering_fail",
        "allocation": {key: allocation.get(key) for key in ("scheduler", "job_id", "hostname", "nodelist", "verified")},
        "manifest": {**manifest_result, "sha256": manifest_hash},
    }
    stage_a: dict[str, Any] = {"pass": False, "complete": False, "errors": ["not run"]}
    stage_b: dict[str, Any] = {"pass": False, "complete": False, "errors": ["not run"]}
    if args.stage in ("a", "all"):
        stage_a = _stage_audit(artifact, manifest, manifest_hash)
        # Save only a compact identity for an optional B cross-check.
        stage_a["wz_sha256"] = None
        a_arrays = artifact / "bank_stage_a.npz"
        if not a_arrays.is_file():
            a_arrays = artifact / "stage_a.npz"
        if a_arrays.is_file():
            try:
                with np.load(a_arrays, allow_pickle=False) as arrays:
                    if "wz" in arrays:
                        stage_a["wz_sha256"] = hashlib.sha256(np.asarray(arrays["wz"]).tobytes()).hexdigest()
            except Exception:
                pass
    if args.stage == "a":
        result["stage_a"] = stage_a
        result["status"] = "complete" if manifest_result["pass"] and stage_a.get("pass") else "engineering_fail"
    elif args.stage == "b":
        stage_a_identity = {"pass": False, "wz_sha256": None}
        stage_b = _stage_b_audit(artifact, manifest, manifest_hash, stage_a_identity)
        result["stage_b"] = stage_b
        result["status"] = stage_b.get("status", "engineering_fail")
    else:
        stage_b = _stage_b_audit(artifact, manifest, manifest_hash, stage_a)
        result["stage_a"] = stage_a
        result["stage_b"] = stage_b
        if not manifest_result["pass"] or not stage_a.get("pass"):
            result["status"] = "engineering_fail"
        else:
            result["status"] = stage_b.get("status", "engineering_fail")
    result["engineering_pass"] = bool(
        manifest_result["pass"]
        and (stage_a.get("pass") if args.stage in ("a", "all") else True)
        and (stage_b.get("engineering_pass") if args.stage in ("b", "all") else True)
    )
    result["mechanism_gate_pass"] = bool(stage_b.get("mechanism_pass", False)) if args.stage in ("b", "all") else False
    _write_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "engineering_pass": result["engineering_pass"],
        "mechanism_gate_pass": result["mechanism_gate_pass"],
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

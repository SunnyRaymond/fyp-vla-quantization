"""Collect Paired-Rollout Recovery records on an allocated CCDS node.

The source model, dataset and all array hashing happen only after the existing
SLURM allocation guard succeeds.  This module deliberately reuses the old FRT
row builder so the PRR runner receives the same adapter/history/action and
source-bridge fields.  PRR changes the split and provenance audit; it does not
open the reserved held-out split or compute evaluation metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_NUM_HIST = 1
DEFAULT_FRAMESKIP = 5
DEFAULT_RECORDS_PER_EPISODE = 2
EPISODES_PER_SPLIT = 6
HISTORICAL_VALID_COUNT = 72
CAL_INDICES = tuple(range(72, 78))
DEV_INDICES = tuple(range(78, 84))
CAL_ENV_NAMESPACE = 1_000_000
CAL_CEM_NAMESPACE = 1_010_000
DEV_ENV_NAMESPACE = 1_100_000
DEV_CEM_NAMESPACE = 1_110_000
MAX_MANIFEST_BYTES = 65_536


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_pickle(path: Path, payload: Any) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _guard_path(helper_dir: Path | None) -> Path:
    here = Path(__file__).resolve()
    candidates = []
    if helper_dir is not None:
        candidates.extend((helper_dir / "allocation_guard.py", helper_dir.parent / "allocation_guard.py"))
    candidates.extend((
        here.with_name("allocation_guard.py"),
        here.parent.parent / "frt-ccds" / "allocation_guard.py",
    ))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("PRR allocation_guard.py was not found")


def _require_allocation(helper_dir: Path | None) -> Mapping[str, Any]:
    """Require RUNNING/UserId/TC1N/NodeList and a visible allocated GPU."""
    guard = _load_module(_guard_path(helper_dir), "prr_allocation_guard")
    allocation = dict(guard.require_allocation())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is missing inside the allocation")
    job_id = str(allocation.get("job_id", "")).strip()
    if not job_id:
        raise RuntimeError("allocation guard returned no job id")
    try:
        record = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id], check=True,
            capture_output=True, text=True, timeout=30,
        ).stdout
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not verify the allocated GPU") from exc
    fields = dict(item.split("=", 1) for item in record.split() if "=" in item)
    if fields.get("JobState") != "RUNNING":
        raise RuntimeError(f"SLURM job is not RUNNING: {fields.get('JobState')!r}")
    if not re.search(r"gres|gpu", record, re.IGNORECASE) or not gpu:
        raise RuntimeError("the current SLURM allocation does not expose a GPU")
    allocation.update({"cuda_visible_devices": visible, "gpu_query": gpu, "gpu_verified": True})
    return allocation


def _find_helper(root: Path, helper_dir: Path | None) -> Path:
    here = Path(__file__).resolve()
    candidates = []
    if helper_dir is not None:
        candidates.extend((helper_dir / "smoke_runner.py", helper_dir))
    candidates.extend((
        root / "control" / "smoke_runner.py",
        root / "smoke_runner.py",
        here.parents[4] / "world-model-quantization" / "experiments" / "dino-wm-wall" / "smoke_runner.py",
        here.parents[4] / "world-model-quantization" / "experiments" / "cem-update-ptq-ccds" / "smoke_runner.py",
    ))
    for path in candidates:
        if path.is_file() and path.name == "smoke_runner.py":
            return path
    raise FileNotFoundError("existing smoke_runner.py was not found; pass --helper-dir")


def _load_helper(path: Path) -> Any:
    module = _load_module(path, "prr_existing_smoke_runner")
    for name in ("_runtime", "_preprocessor"):
        if not hasattr(module, name):
            raise RuntimeError(f"existing helper lacks {name}: {path}")
    return module


def _load_old_collector(root: Path, helper_dir: Path | None) -> tuple[Any, Any]:
    """Locate the prior collector in local or known remote FRT control roots."""
    here = Path(__file__).resolve()
    home = root.parent.parent
    candidates = [
        here.parent.parent / "frt-ccds" / "frt_records.py",
        home / "frt_ccds" / "control" / "frt_records.py",
        home / "frt_ccds" / "artifacts" / "64694" / "frt_records.py",
        home / "frt_ccds" / "artifacts" / "64688" / "frt_records.py",
    ]
    if helper_dir is not None:
        candidates.extend((helper_dir / "frt_records.py", helper_dir.parent / "frt_records.py"))
    candidates.extend(Path(entry) / "frt_records.py" for entry in sys.path if entry)
    old_path = next((path for path in candidates if path.is_file()), None)
    if old_path is None:
        raise FileNotFoundError(
            "old frt_records.py is unavailable; provide the verified FRT control root via --helper-dir"
        )
    old = _load_module(old_path, "prr_old_frt_records")
    try:
        core = old._load_core()
    except (FileNotFoundError, ImportError):
        core_candidates = [path.with_name("frt_core.py") for path in candidates]
        core_path = next((path for path in core_candidates if path.is_file()), None)
        if core_path is None:
            raise
        core = _load_module(core_path, "prr_old_frt_core")
    return old, core


def _split_specs(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """Parse only CAL/DEV; this function has no held-out split access path."""
    defaults = {
        "cal": (CAL_INDICES, "prr_cal", CAL_ENV_NAMESPACE, CAL_CEM_NAMESPACE),
        "dev": (DEV_INDICES, "prr_dev", DEV_ENV_NAMESPACE, DEV_CEM_NAMESPACE),
    }
    indices_default, prefix, env_default, cem_default = defaults[split]
    splits = manifest.get("splits")
    section = splits.get(split) if isinstance(splits, Mapping) else None
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"manifest {split!r} section must be an object")
    raw_rows = section.get("episodes")
    if raw_rows is None:
        values = section.get("dataset_indices", section.get("indices", list(indices_default)))
        ids = section.get("episode_ids", section.get("ids"))
        locals_ = section.get("local_indices")
        raw_rows = []
        for position, index in enumerate(values):
            row = {"dataset_index": index}
            if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
                row["episode_id"] = ids[position]
            if isinstance(locals_, Sequence) and not isinstance(locals_, (str, bytes)):
                row["local_index"] = locals_[position]
            raw_rows.append(row)
    if isinstance(raw_rows, Mapping):
        raw_rows = [raw_rows]
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError(f"manifest {split!r} episodes must be a sequence")
    env_namespace = section.get("environment_namespace", env_default)
    cem_namespace = section.get("cem_namespace", cem_default)
    result = []
    for position, raw in enumerate(raw_rows):
        if isinstance(raw, Mapping):
            index = raw.get("dataset_index", raw.get("subset_index", raw.get("index")))
            episode_id = raw.get("episode_id", raw.get("id"))
            local = raw.get("local_index", raw.get("local", position))
            env_seed = raw.get("env_seed")
            cem_seed = raw.get("cem_seed")
        else:
            index, episode_id, local, env_seed, cem_seed = raw, None, position, None, None
        if index is None:
            raise ValueError(f"{split} episode {position} lacks dataset_index")
        expected_id = f"{prefix}:{position:03d}"
        if episode_id is None:
            episode_id = expected_id
        if str(episode_id) != expected_id:
            raise ValueError(f"{split} episode identity must remain {expected_id!r}")
        if env_seed is None:
            env_seed = int(env_namespace) + int(local)
        if cem_seed is None:
            cem_seed = int(cem_namespace) + int(local)
        result.append({
            "episode_id": expected_id,
            "dataset_index": int(index),
            "local_index": int(local),
            "env_seed": int(env_seed),
            "cem_seed": int(cem_seed),
        })
    if len(result) != EPISODES_PER_SPLIT:
        raise ValueError(f"PRR requires exactly six {split.upper()} episodes")
    if len({row["dataset_index"] for row in result}) != len(result):
        raise ValueError(f"{split.upper()} dataset indices must be distinct")
    return result


def _subset_mapping(dset: Any, position: int) -> tuple[int, list[int]]:
    current = dset
    current_position = int(position)
    chain: list[int] = []
    while hasattr(current, "indices"):
        indices = getattr(current, "indices")
        if current_position < 0 or current_position >= len(indices):
            raise IndexError(f"valid subset index {position} is outside the dataset")
        current_position = int(indices[current_position])
        chain.append(current_position)
        current = getattr(current, "dataset", None)
        if current is None:
            break
    return current_position, chain


def _resolve_position(dset: Any, requested: int, kind: str) -> tuple[int, int, list[int]]:
    if kind == "valid_subset":
        position = int(requested)
        if position < 0 or position >= len(dset):
            raise IndexError(f"valid subset index {requested} is outside the dataset")
    elif kind == "original":
        position = None
        for candidate in range(len(dset)):
            underlying, _ = _subset_mapping(dset, candidate)
            if underlying == int(requested):
                position = candidate
                break
        if position is None:
            raise ValueError(f"original dataset index {requested} is absent from the valid subset")
    else:
        raise ValueError(f"unsupported dataset_index_kind={kind!r}")
    underlying, chain = _subset_mapping(dset, position)
    return position, underlying, chain


def _as_numpy(value: Any, dtype: Any = None):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def _digest(value: Any) -> str:
    """Hash shape, dtype and bytes; called only on the compute node."""
    import numpy as np

    h = hashlib.sha256()
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            h.update(str(key).encode("utf-8"))
            h.update(_digest(value[key]).encode("ascii"))
        return h.hexdigest()
    if isinstance(value, (list, tuple)):
        for item in value:
            h.update(_digest(item).encode("ascii"))
        return h.hexdigest()
    try:
        arr = _as_numpy(value)
    except Exception:
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
    if arr.dtype == object:
        return hashlib.sha256(repr(arr.tolist()).encode("utf-8")).hexdigest()
    arr = np.ascontiguousarray(arr)
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _layout(info: Any) -> Mapping[str, Any]:
    if not isinstance(info, Mapping):
        return {}
    return {
        str(key): _as_numpy(value).reshape(-1).tolist()
        for key, value in info.items()
        if str(key).startswith("fix_") or str(key).startswith("layout")
    }


def _item_fingerprints(item: Any, num_hist: int, frameskip: int, start: int = 0) -> Mapping[str, Any]:
    import numpy as np

    if not isinstance(item, (tuple, list)) or len(item) < 4:
        raise ValueError("Wall dataset item must contain obs, actions, states and info")
    obs, actions, states, info = item[:4]
    if not isinstance(obs, Mapping) or "visual" not in obs or "proprio" not in obs:
        raise ValueError("Wall dataset item lacks visual/proprio observations")
    action_count = (int(num_hist) + 2) * int(frameskip)
    visual = _as_numpy(obs["visual"])
    proprio = _as_numpy(obs["proprio"])
    action_array = _as_numpy(actions)
    state_array = _as_numpy(states)
    if start < 0 or start + action_count > len(action_array) or start >= len(state_array):
        raise ValueError("source trajectory is too short for the PRR action window")
    if start >= len(visual) or start >= len(proprio):
        raise ValueError("source trajectory is too short for the PRR initial observation")
    initial_state = state_array[start]
    initial_observation = {"visual": visual[start], "proprio": proprio[start]}
    action_window = action_array[start : start + action_count]
    state_fp = _digest(initial_state)
    observation_fp = _digest(initial_observation)
    action_fp = _digest(action_window)
    layout = _layout(info)
    layout_fp = _digest(layout)
    condition_fp = _digest({
        "initial_state": state_fp,
        "initial_observation": observation_fp,
        "layout": layout_fp,
    })
    episode_fp = _digest({
        "initial_condition": condition_fp,
        "action_window": action_fp,
    })
    return {
        "initial_state_fingerprint": state_fp,
        "initial_observation_fingerprint": observation_fp,
        "action_window_fingerprint": action_fp,
        "layout_fingerprint": layout_fp,
        "initial_condition_fingerprint": condition_fp,
        "episode_fingerprint": episode_fp,
        "initial_state": np.array(initial_state, dtype=np.float32, copy=True),
        "action_window": np.array(action_window, dtype=np.float32, copy=True),
        "source_lengths": {
            "visual": int(len(visual)), "proprio": int(len(proprio)),
            "actions": int(len(action_array)), "states": int(len(state_array)),
        },
    }


def _metadata_registry(manifest: Mapping[str, Any], dset: Any, base: Path) -> tuple[set[str], set[str], set[str], Mapping[str, Any]]:
    """Read only registered historical fingerprints or lightweight dataset metadata."""
    source: Any = None
    source_name = None
    for key in ("historical_fingerprints", "historical_episode_fingerprints", "historical_valid_metadata"):
        if key in manifest:
            source, source_name = manifest[key], f"manifest:{key}"
            break
    if source is None:
        for key in ("historical_fingerprint_manifest", "historical_metadata_path"):
            raw = manifest.get(key)
            if raw is None:
                continue
            path = Path(str(raw))
            if not path.is_absolute():
                path = base / path
            if path.is_file():
                source, source_name = _load_json(path), str(path)
                break
    if source is None:
        for key in ("episode_metadata", "metadata", "_metadata", "infos", "_infos"):
            candidate = getattr(dset, key, None)
            if candidate is not None and not callable(candidate):
                source, source_name = candidate, f"dataset:{key}"
                break
    if isinstance(source, Mapping) and "fingerprints" in source:
        source = source["fingerprints"]
    rows: list[tuple[int, Any]] = []
    if isinstance(source, Mapping):
        for key, value in source.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            rows.append((index, value))
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        rows = list(enumerate(source))
    episode_fps: set[str] = set()
    state_fps: set[str] = set()
    condition_fps: set[str] = set()
    covered: list[int] = []
    for index, value in rows:
        if index < 0 or index >= HISTORICAL_VALID_COUNT:
            continue
        row = value if isinstance(value, Mapping) else {"episode_fingerprint": value}
        episode = row.get("episode_fingerprint")
        state = row.get("initial_state_fingerprint")
        condition = row.get("initial_condition_fingerprint")
        if episode is None and "initial_state" in row:
            episode = _digest(row["initial_state"])
        if state is None and "initial_state" in row:
            state = _digest(row["initial_state"])
        if condition:
            condition_fps.add(str(condition))
        if episode:
            episode_fps.add(str(episode))
        if state:
            state_fps.add(str(state))
        if episode or state or condition:
            covered.append(index)
    return episode_fps, state_fps, condition_fps, {
        "source": source_name,
        "covered_valid_indices": sorted(set(covered)),
        "coverage_count": len(set(covered)),
        "coverage_complete": len(set(covered)) == HISTORICAL_VALID_COUNT,
    }


def _audit_provenance(
    manifest: Mapping[str, Any], dset: Any, specs: Mapping[str, Sequence[Mapping[str, Any]]],
    dataset_kind: str, output: Path, manifest_base: Path,
) -> Mapping[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    historical_ids: dict[int, int] = {}
    for valid_index in range(min(HISTORICAL_VALID_COUNT, len(dset))):
        underlying, _ = _subset_mapping(dset, valid_index)
        historical_ids[valid_index] = underlying
    episode_fps, state_fps, condition_fps, registry = _metadata_registry(manifest, dset, manifest_base)
    rows = [dict(row, split=split) for split, values in specs.items() for row in values]
    audit_rows = []
    errors: list[str] = []
    seen_underlying: dict[int, str] = {}
    seen_episode: dict[str, str] = {}
    seen_condition: dict[str, str] = {}
    for row in rows:
        position, underlying, chain = _resolve_position(dset, int(row["dataset_index"]), dataset_kind)
        item = dset[position]
        fp = _item_fingerprints(item, DEFAULT_NUM_HIST, DEFAULT_FRAMESKIP)
        action_count = (DEFAULT_NUM_HIST + 2) * DEFAULT_FRAMESKIP
        lengths = fp["source_lengths"]
        max_start = min(lengths.values()) - action_count
        starts = [0] if max_start <= 0 else [0, max_start]
        window_fps = [fp]
        for extra_start in starts[1:]:
            window_fps.append(_item_fingerprints(item, DEFAULT_NUM_HIST, DEFAULT_FRAMESKIP, start=extra_start))
        episode_id = str(row["episode_id"])
        if underlying in set(historical_ids.values()):
            errors.append(f"{episode_id}: underlying episode {underlying} overlaps historical valid 0..71")
        if underlying in seen_underlying:
            errors.append(f"{episode_id}: underlying episode {underlying} duplicates {seen_underlying[underlying]}")
        else:
            seen_underlying[underlying] = episode_id
        if fp["episode_fingerprint"] in seen_episode:
            errors.append(f"{episode_id}: episode fingerprint duplicates {seen_episode[fp['episode_fingerprint']]}")
        else:
            seen_episode[str(fp["episode_fingerprint"])] = episode_id
        if fp["episode_fingerprint"] in episode_fps:
            errors.append(f"{episode_id}: episode fingerprint overlaps historical registry")
        if fp["initial_state_fingerprint"] in state_fps:
            errors.append(f"{episode_id}: initial-state fingerprint overlaps historical registry")
        for window_fp in window_fps:
            condition = str(window_fp["initial_condition_fingerprint"])
            if condition in seen_condition and seen_condition[condition] != episode_id:
                errors.append(f"{episode_id}: initial-condition fingerprint duplicates {seen_condition[condition]}")
            else:
                seen_condition[condition] = episode_id
            if condition in condition_fps:
                errors.append(f"{episode_id}: initial-condition fingerprint overlaps historical registry")
        audit_rows.append({
            "split": row["split"], "episode_id": episode_id,
            "requested_dataset_index": int(row["dataset_index"]),
            "valid_subset_index": int(position), "source_underlying_index": int(underlying),
            "source_underlying_index_id": f"valid[{position}] -> original[{underlying}]",
            "subset_chain": chain,
            "initial_state_fingerprint": fp["initial_state_fingerprint"],
            "initial_observation_fingerprint": fp["initial_observation_fingerprint"],
            "action_window_fingerprint": fp["action_window_fingerprint"],
            "layout_fingerprint": fp["layout_fingerprint"],
            "initial_condition_fingerprint": fp["initial_condition_fingerprint"],
            "episode_fingerprint": fp["episode_fingerprint"],
            "window_starts_audited": starts,
            "window_condition_fingerprints": [item["initial_condition_fingerprint"] for item in window_fps],
        })
    report = {
        "schema": "prr-provenance-audit-v1",
        "status": "rejected" if errors else ("complete" if registry["coverage_complete"] else "complete_with_coverage_limitation"),
        "historical_valid_index_range": [0, HISTORICAL_VALID_COUNT - 1],
        "historical_underlying_ids": historical_ids,
        "historical_fingerprint_registry": registry,
        "new_episode_rows": audit_rows,
        "new_episode_count": len(audit_rows),
        "new_underlying_indices_distinct": len(seen_underlying) == len(audit_rows),
        "new_episode_fingerprints_distinct": len(seen_episode) == len(audit_rows),
        "new_initial_condition_fingerprints_distinct_across_episodes": len(set(seen_condition)) == len(seen_condition),
        "dataset_index_is_not_independence": True,
        "independence_claim": "none; this is an episode/initial-state fingerprint audit",
        "errors": errors,
        "test_opened": False,
    }
    _atomic_json(output / "provenance_audit.json", report)
    if errors:
        raise RuntimeError(f"PRR provenance audit rejected the requested records; see {output / 'provenance_audit.json'}")
    return report


def _candidate_path(raw: Any, root: Path, helper_dir: Path | None, base: Path, defaults: Sequence[Path]) -> Path | None:
    values = []
    if raw is not None:
        values.append(Path(str(raw)))
    values.extend(defaults)
    for value in values:
        if value.is_absolute():
            candidates = [value]
        else:
            candidates = [base / value, root / value, root.parent / value, root.parent.parent / value]
            if helper_dir is not None:
                candidates.append(helper_dir / value)
        for path in candidates:
            if path.is_file():
                return path
    return None


def _find_clean_sha(value: Any, context: str = "") -> str | None:
    if isinstance(value, Mapping):
        marker = (context + " " + " ".join(str(k) for k in value.keys())).lower()
        method = str(value.get("method_seed", value.get("method", ""))).lower()
        seed = str(value.get("seed", "")).lower()
        marker += " " + method + " " + seed
        if "clean" in marker and ("1201" in marker or "clean_seed_1201" in marker):
            for key in ("checkpoint_sha256", "sha256", "sha"):
                candidate = value.get(key)
                if candidate and re.fullmatch(r"[0-9a-fA-F]{64}", str(candidate)):
                    return str(candidate).lower()
        for key, child in value.items():
            found = _find_clean_sha(child, marker + " " + str(key))
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _find_clean_sha(child, context)
            if found:
                return found
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_donor(manifest: Mapping[str, Any], root: Path, helper_dir: Path | None, output: Path, base: Path) -> Mapping[str, Any]:
    config = None
    donors = manifest.get("donors")
    if isinstance(donors, Mapping) and isinstance(donors.get("clean_seed_1201"), Mapping):
        config = donors["clean_seed_1201"]
    for key in ("old_clean_donor", "clean_donor", "donor"):
        if config is not None:
            break
        if isinstance(manifest.get(key), Mapping):
            config = manifest[key]
            break
        if isinstance(manifest.get(key), (str, Path)):
            config = {"path": manifest[key]}
            break
    config = dict(config or {})
    default_dir = root.parent.parent
    default_checkpoint = default_dir / "frt_ccds" / "artifacts" / "64694" / "checkpoints" / "clean_seed_1201.pt"
    default_checkpoint_alt = default_dir / "frt" / "artifacts" / "64694" / "checkpoints" / "clean_seed_1201.pt"
    default_meta = default_dir / "frt_ccds" / "artifacts" / "64694" / "stage_b.json"
    default_meta_alt = default_dir / "frt" / "artifacts" / "64694" / "stage_b.json"
    checkpoint = _candidate_path(config.get("path", config.get("checkpoint")), root, helper_dir, base, [default_checkpoint, default_checkpoint_alt])
    meta = _candidate_path(config.get("verification_meta", config.get("stage_b_json")), root, helper_dir, base, [default_meta, default_meta_alt])
    metadata: Mapping[str, Any] = {}
    meta_error = None
    if meta is not None:
        try:
            metadata = _load_json(meta)
        except (OSError, ValueError) as exc:
            meta_error = str(exc)
    expected = config.get("expected_sha256", config.get("sha256"))
    if expected is None and metadata:
        expected = _find_clean_sha(metadata)
    expected = str(expected).lower() if expected else None
    actual = _sha256_file(checkpoint) if checkpoint is not None else None
    fit_scope = str(config.get("fit_scope", "cal_only")).lower()
    method_seed = str(config.get("method_seed", "clean:1201")).lower()
    if "cal_only" in config and not bool(config["cal_only"]):
        fit_scope = "ambiguous"
    if "method" in config or "seed" in config:
        method_seed = f"{config.get('method', 'clean')}:{config.get('seed', 1201)}".lower()
    fit_cal_only = fit_scope in {"cal", "cal_only", "calibration", "fit_cal_only"} and "dev" not in fit_scope and "test" not in fit_scope
    selected_from_dev = "dev" in str(config.get("selection_source", "predeclared_clean_seed_1201")).lower()
    sha_match = bool(actual and expected and actual == expected)
    metadata_present = meta is not None and meta_error is None
    usable = bool(checkpoint and metadata_present and sha_match and fit_cal_only and method_seed == "clean:1201" and not selected_from_dev)
    report = {
        "schema": "prr-donor-audit-v1",
        "requested": True,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "verification_meta": str(meta.resolve()) if meta else None,
        "checkpoint_exists": checkpoint is not None,
        "verification_meta_exists": metadata_present,
        "metadata_error": meta_error,
        "expected_sha256_from_trusted_stage_b": expected,
        "actual_sha256": actual,
        "sha_matches_trusted_metadata": sha_match,
        "method_seed": method_seed,
        "fit_scope": fit_scope,
        "fit_cal_only": fit_cal_only,
        "fit_cal_only_source": "old frt runner _fit_method(cal_bank); predeclared clean:1201; no DEV selection",
        "selected_from_dev": selected_from_dev,
        "usable_as_shared_donor": usable,
        "provenance_verified": usable,
        "q0_fallback_allowed": True,
        "fallback_policy": "If donor provenance or hash is unavailable/ambiguous, every PRR cell uses the shared Q0 donor; no silent donor substitution.",
        "model_loaded": False,
    }
    _atomic_json(output / "donor_audit.json", report)
    return report


def _augment_episode(episode: Mapping[str, Any], dset: Any, subset_index: int, underlying: int, chain: Sequence[int], num_hist: int, frameskip: int) -> Mapping[str, Any]:
    import numpy as np

    out = dict(episode)
    item = dset[subset_index]
    baseline = _item_fingerprints(item, num_hist, frameskip, start=0)
    rows = []
    for old_row in episode["records"]:
        row = dict(old_row)
        start = int(row["window_start"])
        fp = _item_fingerprints(item, num_hist, frameskip, start=start)
        row.update({
            "requested_dataset_index": int(subset_index),
            "source_underlying_index": int(underlying),
            "source_underlying_index_id": f"valid[{subset_index}] -> original[{underlying}]",
            "source_subset_chain": list(chain),
            "initial_state": fp["initial_state"],
            "action_window": fp["action_window"],
            "source_action_window": fp["action_window"],
            "initial_state_fingerprint": fp["initial_state_fingerprint"],
            "initial_observation_fingerprint": fp["initial_observation_fingerprint"],
            "action_window_fingerprint": fp["action_window_fingerprint"],
            "layout_fingerprint": fp["layout_fingerprint"],
            "initial_condition_fingerprint": fp["initial_condition_fingerprint"],
            "episode_fingerprint": fp["episode_fingerprint"],
            "history_fp": row["history"],
            "next_fp": row["fp_reference_slot"],
        })
        rows.append(row)
    out.update({
        "source_underlying_index": int(underlying),
        "source_underlying_index_id": f"valid[{subset_index}] -> original[{underlying}]",
        "source_subset_chain": list(chain),
        "initial_state_fingerprint": baseline["initial_state_fingerprint"],
        "initial_observation_fingerprint": baseline["initial_observation_fingerprint"],
        "action_window_fingerprint": baseline["action_window_fingerprint"],
        "layout_fingerprint": baseline["layout_fingerprint"],
        "initial_condition_fingerprint": baseline["initial_condition_fingerprint"],
        "episode_fingerprint": baseline["episode_fingerprint"],
        "record_count": len(rows),
        "records": rows,
        "window_policy": "two nested start windows; episode is the independent unit",
    })
    if len(rows) != DEFAULT_RECORDS_PER_EPISODE:
        raise RuntimeError("PRR requires exactly two windows nested under every episode")
    return out


def _collect_split(old: Any, core: Any, helper: Any, runtime: Mapping[str, Any], preprocessor: Any, dset: Any, specs: Sequence[Mapping[str, Any]], dataset_kind: str, num_hist: int, frameskip: int) -> Mapping[str, Any]:
    adapter = core.SourceHistoryAdapter(runtime["model"], num_hist=num_hist)
    episodes = []
    for spec in specs:
        position, underlying, chain = _resolve_position(dset, int(spec["dataset_index"]), dataset_kind)
        episode = old._episode_records(
            helper, core, runtime, preprocessor, dset, spec["episode_id"], position, underlying,
            int(spec["env_seed"]), int(spec["cem_seed"]), num_hist, frameskip,
            DEFAULT_RECORDS_PER_EPISODE, False, adapter,
        )
        episodes.append(_augment_episode(episode, dset, position, underlying, chain, num_hist, frameskip))
    flat = [row for episode in episodes for row in episode["records"]]
    return {
        "schema": "prr-ccds-records-v1", "split": str(specs[0]["episode_id"]).split("_", 1)[1].split(":", 1)[0],
        "source_distribution": "offline_contiguous_wall_dataset",
        "num_hist": num_hist, "frameskip": frameskip,
        "episode_count": len(episodes), "record_count": len(flat),
        "episode_unit": "episode_initial_state", "episodes": episodes, "records": flat,
        "physical_replay_included": False, "test_opened": False,
    }


def collect(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.stage not in {"smoke", "r1"}:
        raise ValueError("stage must be smoke or r1")
    helper_dir = args.helper_dir.resolve() if args.helper_dir else None
    allocation = _require_allocation(helper_dir)
    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    root = args.root.resolve()
    output = args.output.resolve()
    requested_hist = manifest.get("num_hist", DEFAULT_NUM_HIST)
    frameskip = int(manifest.get("frameskip", DEFAULT_FRAMESKIP))
    if int(requested_hist) != DEFAULT_NUM_HIST or frameskip != DEFAULT_FRAMESKIP:
        raise RuntimeError("PRR is frozen to checkpoint num_hist=1 and frameskip=5")
    records_per_episode = int(manifest.get("records_per_episode", DEFAULT_RECORDS_PER_EPISODE))
    if records_per_episode != DEFAULT_RECORDS_PER_EPISODE:
        raise RuntimeError("PRR is frozen to two start windows per episode")
    dataset_kind = str(manifest.get("dataset_index_kind", "valid_subset"))
    helper = _load_helper(_find_helper(root, helper_dir))
    old, core = _load_old_collector(root, helper_dir)
    runtime = helper._runtime(root, device="cuda:0")
    model = runtime["model"]
    actual_hist = int(getattr(model, "num_hist", -1))
    if actual_hist != DEFAULT_NUM_HIST:
        raise RuntimeError(f"loaded source checkpoint has num_hist={actual_hist}, expected 1")
    if bool(getattr(model, "training", False)):
        model.eval()
    preprocessor = helper._preprocessor(runtime)
    dset = runtime["dset"]
    specs = {"cal": _split_specs(manifest, "cal")}
    if args.stage == "r1":
        specs["dev"] = _split_specs(manifest, "dev")
    audit = _audit_provenance(manifest, dset, specs, dataset_kind, output, manifest_path.parent)
    donor = _audit_donor(manifest, root, helper_dir, output, manifest_path.parent)
    collected = {
        split: _collect_split(old, core, helper, runtime, preprocessor, dset, rows, dataset_kind, DEFAULT_NUM_HIST, frameskip)
        for split, rows in specs.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for split, payload in collected.items():
        path = output / f"{split}.pkl"
        _atomic_pickle(path, payload)
        split_paths[split] = str(path.resolve())
    combined_records = [row for payload in collected.values() for row in payload["records"]]
    records_path = output / "records.pkl"
    _atomic_pickle(records_path, {
        "schema": "prr-ccds-records-v1", "stage": args.stage,
        "num_hist": DEFAULT_NUM_HIST, "frameskip": frameskip,
        "episode_count": sum(payload["episode_count"] for payload in collected.values()),
        "record_count": len(combined_records), "records": combined_records,
        "splits": collected, "test_opened": False,
    })
    identity = {}
    checkpoint_identity = getattr(helper, "_checkpoint_identity", None)
    if callable(checkpoint_identity):
        identity = dict(checkpoint_identity(runtime))
    amended_splits = {}
    input_splits = manifest.get("splits") if isinstance(manifest.get("splits"), Mapping) else {}
    for split, payload in collected.items():
        original = input_splits.get(split) if isinstance(input_splits, Mapping) else {}
        row = dict(original) if isinstance(original, Mapping) else {}
        row.update({
            "split_id": f"prr_{split}", "episode_count": payload["episode_count"],
            "record_count": payload["record_count"], "path": split_paths[split],
            "episode_ids": [episode["episode_id"] for episode in payload["episodes"]],
            "dataset_indices": [int(spec["dataset_index"]) for spec in specs[split]],
            "environment_namespace": int(specs[split][0]["env_seed"]) - int(specs[split][0]["local_index"]),
            "cem_namespace": int(specs[split][0]["cem_seed"]) - int(specs[split][0]["local_index"]),
            "seed_rule": "namespace + local_index",
        })
        amended_splits[split] = row
    if args.stage == "smoke" and isinstance(input_splits, Mapping) and isinstance(input_splits.get("dev"), Mapping):
        deferred_dev = dict(input_splits["dev"])
        deferred_dev.pop("path", None)
        deferred_dev.pop("record_count", None)
        deferred_dev.pop("episode_count", None)
        deferred_dev["deferred_until"] = "r1"
        amended_splits["dev"] = deferred_dev
    out_manifest = copy.deepcopy(dict(manifest))
    canonical_donor = {
        "provenance_verified": bool(donor["provenance_verified"]),
        "checkpoint": donor["checkpoint"],
        "sha256": donor["actual_sha256"],
        "method": "clean",
        "seed": 1201,
        "cal_only": bool(donor["fit_cal_only"]),
        "selected_from_dev": bool(donor["selected_from_dev"]),
    }
    canonical_donors = out_manifest.get("donors")
    canonical_donors = dict(canonical_donors) if isinstance(canonical_donors, Mapping) else {}
    canonical_donors["clean_seed_1201"] = canonical_donor
    out_manifest.update({
        "schema": "prr-ccds-manifest-v1", "collection_schema": "prr-ccds-record-manifest-v1",
        "stage": args.stage, "num_hist": DEFAULT_NUM_HIST, "runtime_num_hist": actual_hist,
        "frameskip": frameskip, "records_per_episode": DEFAULT_RECORDS_PER_EPISODE,
        "records_path": str(records_path.resolve()), "splits": {**dict(input_splits), **amended_splits},
        "provenance_audit": str((output / "provenance_audit.json").resolve()),
        "donor_audit": str((output / "donor_audit.json").resolve()),
        "donors": canonical_donors,
        "checkpoint_identity": identity, "hardware_provenance": dict(allocation),
        "test_opened": False, "test_policy": "reserved held-out split is never opened by this collector",
        "collection": {
            "episode_ids": {split: [episode["episode_id"] for episode in payload["episodes"]] for split, payload in collected.items()},
            "dataset_index_kind": dataset_kind,
            "independent_unit": "episode_initial_state; two start windows remain nested",
            "fingerprint_audit": "actual underlying mapping plus registered historical fingerprints",
            "dataset_index_is_not_independence": True,
            "q0_fallback_allowed": bool(donor["q0_fallback_allowed"]),
        },
    })
    manifest_bytes = json.dumps(out_manifest, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise RuntimeError(f"PRR manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    _atomic_json(output / "manifest.json", out_manifest)
    _atomic_json(output / "summary.json", {
        "schema": "prr-ccds-record-summary-v1", "status": "complete",
        "stage": args.stage, "split_episode_counts": {k: v["episode_count"] for k, v in collected.items()},
        "split_record_counts": {k: v["record_count"] for k, v in collected.items()},
        "records_path": str(records_path.resolve()), "provenance_audit": str((output / "provenance_audit.json").resolve()),
        "donor_audit": str((output / "donor_audit.json").resolve()), "test_opened": False,
        "donor_usable": bool(donor["usable_as_shared_donor"]), "q0_fallback_allowed": True,
        "historical_registry_coverage": audit["historical_fingerprint_registry"],
        "hardware_provenance": dict(allocation),
    })
    return out_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="verified DINO-WM modelroot")
    parser.add_argument("--manifest", type=Path, required=True, help="small PRR protocol manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, required=True, help="directory containing the existing helper")
    parser.add_argument("--stage", choices=("smoke", "r1"), required=True)
    return parser.parse_args()


def main() -> None:
    result = collect(_parse_args())
    print(json.dumps({
        "status": "complete", "stage": result["stage"],
        "records_path": result["records_path"], "test_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Collect bounded FRT CAL/DEV records from the official Wall runtime.

This collector is intentionally an offline, contiguous-dataset collector.  It
does not fabricate histories: each ``history`` is encoded from the source
Wall observations and their preceding action chunks.  It must run on a
real CCDS SLURM compute allocation; importing this module is safe for syntax
checks because the allocation guard and the model loader are called only by
``main``.

The output is ``records.pkl`` plus ``manifest.json`` under ``--output``.  The
manifest uses the small FRT protocol understood by ``frt_runner.py``:
``splits.cal/dev.path`` points to a payload containing episode records.  The
payload keeps NumPy arrays rather than JSON lists so large latent histories do
not get needlessly expanded.  The loaded checkpoint determines ``num_hist``;
the collector never pads a shorter source history with repeated frames.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_FRAMESKIP = 5
DEFAULT_RECORDS_PER_EPISODE = 2
DEFAULT_CAL_IDS = tuple(range(60, 66))
DEFAULT_DEV_IDS = tuple(range(66, 72))
MAX_RECORDS_PER_EPISODE = 4
MAX_EPISODES_PER_SPLIT = 6


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
    raise TypeError(f"cannot encode {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
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
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def _require_allocation() -> Mapping[str, Any]:
    """Require a real TC1N SLURM allocation before model or dataset I/O."""
    from allocation_guard import require_allocation

    allocation = dict(require_allocation())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is missing inside the allocation")
    try:
        job = subprocess.run(
            ["scontrol", "show", "job", "-o", str(allocation["job_id"])],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not verify the SLURM GPU allocation") from exc
    normalized = job.replace(" ", "").lower()
    if "gres/gpu=1" not in normalized and "gres=gpu:1" not in normalized:
        raise RuntimeError("the current SLURM job does not report one GPU")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi could not confirm the allocated GPU") from exc
    if "v100" not in gpu.casefold():
        raise RuntimeError(f"FRT CCDS collector requires a V100, got {gpu!r}")
    allocation["cuda_visible_devices"] = visible
    allocation["gpu_query"] = gpu
    allocation["gpu_verified"] = True
    return allocation


def _find_helper(root: Path, helper_dir: Path | None) -> Path:
    """Find the existing smoke runner without assuming a particular remote layout."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    if helper_dir is not None:
        candidates.append(helper_dir / "smoke_runner.py")
    candidates.extend(
        [
            root / "control" / "smoke_runner.py",
            root / "smoke_runner.py",
            here.parents[4] / "experiment" / "idea-validation" / "world-model-quantization" / "dino-wm-wall" / "smoke_runner.py",
            here.parents[4] / "experiment" / "idea-validation" / "world-model-quantization" / "cem-update-ptq-ccds" / "smoke_runner.py",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "existing CCDS smoke_runner.py was not found; pass --helper-dir pointing to the verified helper"
    )


def _load_helper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frt_existing_smoke_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import smoke runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("_runtime", "_linear_groups", "_module_for_path", "_preprocessor", "_new_wall_env"):
        if not hasattr(module, name):
            raise RuntimeError(f"existing smoke runner lacks required helper {name}: {path}")
    return module


def _load_core() -> Any:
    path = Path(__file__).resolve().with_name("frt_core.py")
    spec = importlib.util.spec_from_file_location("frt_local_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import FRT core {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scalar(value: Any) -> float | int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return value


def _numpy(value: Any, dtype: Any = None):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return np.array(value, dtype=dtype, copy=True)


def _episode_pairs(payload: Mapping[str, Any], split: str) -> list[tuple[Any, int, int | None, int | None]]:
    """Return (episode_id, valid-subset dataset index, env_seed, cem_seed)."""
    source: Any = payload.get("splits", {}).get(split) if isinstance(payload.get("splits"), Mapping) else None
    if source is None:
        source = payload.get(split)
    if source is None:
        source = DEFAULT_CAL_IDS if split == "cal" else DEFAULT_DEV_IDS
    namespace = None
    env_namespace = None
    cem_namespace = None
    local_values = None
    namespaces = payload.get("environment_namespace")
    if isinstance(namespaces, Mapping):
        namespace = namespaces.get(split)
    elif isinstance(namespaces, str):
        namespace = namespaces
    if isinstance(source, Mapping):
        namespace = source.get("environment_namespace", source.get("namespace", namespace))
        env_namespace = source.get("environment_namespace")
        cem_namespace = source.get("cem_namespace")
        local_values = source.get("local_indices")
        if "episodes" in source:
            rows = source["episodes"]
        else:
            dataset_values = source.get("dataset_indices", source.get("indices"))
            episode_values = source.get("episode_ids", source.get("ids"))
            if dataset_values is not None and episode_values is not None:
                if len(dataset_values) != len(episode_values):
                    raise ValueError(f"manifest {split!r} dataset_indices/episode_ids length mismatch")
                rows = []
                for position, (dataset_index, episode_id) in enumerate(zip(dataset_values, episode_values)):
                    row = {"dataset_index": dataset_index, "episode_id": episode_id}
                    if isinstance(local_values, Sequence) and not isinstance(local_values, (str, bytes)):
                        row["local_index"] = local_values[position]
                    rows.append(row)
            elif dataset_values is not None:
                rows = dataset_values
            elif episode_values is not None:
                rows = episode_values
            else:
                rows = source
        if isinstance(rows, Mapping):
            rows = [rows]
    else:
        rows = source
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"manifest split {split!r} must be a sequence of episodes")
    pairs: list[tuple[Any, int, int | None, int | None]] = []
    for position, item in enumerate(rows):
        if isinstance(item, Mapping):
            index = item.get("dataset_index", item.get("subset_index", item.get("index")))
            episode_id = item.get("episode_id", item.get("id", index))
            env_seed = item.get("env_seed")
            cem_seed = item.get("cem_seed")
            local_index = item.get("local_index", item.get("local", position))
        else:
            index = item
            episode_id = item
            env_seed = None
            cem_seed = None
            local_index = position
        if index is None:
            raise ValueError(f"manifest split {split!r} has an episode without dataset_index")
        if namespace is not None and not isinstance(episode_id, str):
            episode_id = f"{namespace}:{int(episode_id):03d}"
        if env_seed is None and isinstance(env_namespace, (int, float)):
            env_seed = int(env_namespace) + int(local_index)
        if cem_seed is None and isinstance(cem_namespace, (int, float)):
            cem_seed = int(cem_namespace) + int(local_index)
        pairs.append(
            (
                episode_id,
                int(index),
                None if env_seed is None else int(env_seed),
                None if cem_seed is None else int(cem_seed),
            )
        )
    if len(pairs) != MAX_EPISODES_PER_SPLIT:
        raise ValueError(f"FRT pilot requires exactly 6 {split.upper()} episodes, got {len(pairs)}")
    if len({pair[1] for pair in pairs}) != len(pairs):
        raise ValueError(f"{split.upper()} dataset indices must be distinct")
    return pairs


def _valid_subset_index(dset: Any, requested: int, kind: str) -> tuple[int, int | None]:
    indices = getattr(dset, "indices", None)
    if indices is None:
        return requested, requested
    original = [int(item) for item in indices]
    if kind == "original":
        try:
            subset = original.index(requested)
        except ValueError as exc:
            raise ValueError(f"original dataset index {requested} is absent from the valid subset") from exc
        return subset, requested
    if requested < 0 or requested >= len(original):
        raise IndexError(f"valid subset index {requested} outside 0..{len(original) - 1}")
    return requested, original[requested]


def _model_observation(obs: Mapping[str, Any], preprocessor: Any, device: Any) -> Mapping[str, Any]:
    """Use WallDataset's transformed CHW tensors without a second transform."""
    import torch

    visual = obs["visual"]
    if getattr(visual, "ndim", 0) == 4 and int(visual.shape[-1]) == 3:
        transformed = preprocessor.transform_obs(obs)
        return {key: value.to(device=device, dtype=torch.float32) for key, value in transformed.items()}
    visual_tensor = torch.as_tensor(visual, dtype=torch.float32, device=device)
    proprio_tensor = torch.as_tensor(obs["proprio"], dtype=torch.float32, device=device)
    if visual_tensor.ndim != 4 or visual_tensor.shape[1] != 3:
        raise ValueError(f"Wall visual must be [T,3,H,W] after dataset transform, got {tuple(visual_tensor.shape)}")
    return {"visual": visual_tensor.unsqueeze(0), "proprio": proprio_tensor.unsqueeze(0)}


def _layout(info: Mapping[str, Any], torch: Any) -> Mapping[str, Any]:
    if "fix_door_location" not in info or "fix_wall_location" not in info:
        raise ValueError("Wall dataset item lacks fixed door/wall layout metadata")
    return {
        "fix_door_location": torch.tensor(float(_scalar(info["fix_door_location"])), dtype=torch.float32),
        "fix_wall_location": torch.tensor(float(_scalar(info["fix_wall_location"])), dtype=torch.float32),
    }


def _close_env(helper: Any, env: Any) -> None:
    close = getattr(helper, "_close_env", None)
    if callable(close):
        close(env)
        return
    for child in getattr(env, "envs", []):
        if callable(getattr(child, "close", None)):
            child.close()


def _normalise_states(value: Any):
    import numpy as np

    result = _numpy(value, np.float32)
    if result.ndim == 3:
        if result.shape[0] == 1:
            result = result[0]
        elif result.shape[1] == 1:
            result = result[:, 0]
    if result.ndim != 2:
        raise ValueError(f"environment states must reduce to [time,state_dim], got {result.shape}")
    return result


def _physical_replay(
    helper: Any,
    runtime: Mapping[str, Any],
    preprocessor: Any,
    info: Mapping[str, Any],
    init_state: Any,
    action_chunks: Any,
    frameskip: int,
    goal_xy: Any,
    env_seed: int,
) -> Mapping[str, Any]:
    """Replay the same denormalized primitive actions twice for A evidence."""
    import numpy as np
    import torch

    action_norm = torch.as_tensor(action_chunks, dtype=torch.float32)
    primitive = action_norm.reshape(-1, int(action_norm.shape[-1] // frameskip))
    primitive = preprocessor.denormalize_actions(primitive).detach().cpu().numpy().astype(np.float32)
    state = _numpy(init_state, np.float32).reshape(-1)
    state_for_env = state[:2] if state.size > 2 else state
    layout = _layout(info, torch)
    rows = []
    for _ in range(2):
        env = helper._new_wall_env(runtime, count=1)
        try:
            env.update_env([layout])
            _, states = env.rollout([int(env_seed)], np.asarray([state_for_env]), np.asarray([primitive]))
            rows.append(_normalise_states(states))
        finally:
            _close_env(helper, env)
    states_a, states_b = rows
    if states_a.shape != states_b.shape:
        raise RuntimeError(f"deterministic replay shape mismatch: {states_a.shape} vs {states_b.shape}")
    max_abs = float(np.max(np.abs(states_a - states_b)))
    if max_abs > 1e-5:
        raise RuntimeError(f"Wall deterministic replay failed, max state difference={max_abs:g}")
    return {
        "states_a": states_a,
        "states_b": states_b,
        "xy_a": states_a[:, :2],
        "xy_b": states_b[:, :2],
        "goal_xy": _numpy(goal_xy, np.float32).reshape(-1)[:2],
        "initial_xy": states_a[0, :2],
        "endpoint_xy": states_a[-1, :2],
        "deterministic_max_abs": max_abs,
        "replay_steps": int(primitive.shape[0]),
        "replay_seed": int(env_seed),
        "same_seed": True,
        "state_projection": "first_two_coordinates" if state.size > 2 else "none",
    }


def _record_offsets(length: int, required: int, count: int) -> list[int]:
    import numpy as np

    max_start = int(length) - int(required)
    if max_start < 0:
        raise ValueError(f"trajectory length {length} cannot provide {required} contiguous raw frames")
    if count < 1 or count > MAX_RECORDS_PER_EPISODE:
        raise ValueError(f"records per episode must be in 1..{MAX_RECORDS_PER_EPISODE}")
    if max_start + 1 < count:
        raise ValueError(f"trajectory has only {max_start + 1} valid windows, need fixed count {count}")
    values = np.linspace(0, max_start, count, dtype=np.int64).tolist()
    if len(set(values)) != count:
        raise ValueError("record offsets collapsed to duplicates")
    return [int(value) for value in values]


def _episode_records(
    helper: Any,
    core: Any,
    runtime: Mapping[str, Any],
    preprocessor: Any,
    dset: Any,
    episode_id: Any,
    requested_index: int,
    original_index: int | None,
    env_seed: int,
    cem_seed: int,
    num_hist: int,
    frameskip: int,
    count: int,
    include_physical: bool,
    adapter: Any,
) -> Mapping[str, Any]:
    import numpy as np
    import torch

    obs, stored_actions, states, info = dset[requested_index]
    if not isinstance(obs, Mapping) or "visual" not in obs or "proprio" not in obs:
        raise ValueError(f"episode {episode_id!r} has incomplete observation mapping")
    visual_len = int(obs["visual"].shape[0])
    action_len = int(stored_actions.shape[0])
    state_len = int(states.shape[0])
    required_raw = (num_hist + 2) * frameskip + 1
    offsets = _record_offsets(min(visual_len, action_len, state_len), required_raw, count)
    model = runtime["model"]
    device = runtime["device"]
    records: list[Mapping[str, Any]] = []
    source_bridge_rows = []
    for record_index, start in enumerate(offsets):
        frame_indices = list(range(start, start + num_hist * frameskip, frameskip))
        action_end = start + (num_hist + 2) * frameskip
        obs_window = {key: value[frame_indices] for key, value in obs.items()}
        stored = torch.as_tensor(stored_actions[start:action_end], dtype=torch.float32)
        raw_actions = preprocessor.denormalize_actions(stored)
        normalized = preprocessor.normalize_actions(raw_actions)
        roundtrip_error = float((normalized - stored).abs().max().item())
        if roundtrip_error > 1e-5:
            raise RuntimeError(f"source action normalization roundtrip failed for episode {episode_id!r}")
        primitive_dim = int(normalized.shape[-1])
        if normalized.shape[0] != (num_hist + 2) * frameskip:
            raise RuntimeError("action window length does not match source num_hist/frameskip")
        chunks = normalized.reshape(num_hist + 2, frameskip * primitive_dim)
        history_actions = chunks[:num_hist].unsqueeze(0).to(device)
        current_action = chunks[num_hist : num_hist + 1].to(device)
        next_action = chunks[num_hist + 1 : num_hist + 2].to(device)
        model_obs = _model_observation(obs_window, preprocessor, device)
        with torch.no_grad():
            encoded = model.encode(model_obs, history_actions)
            if encoded.ndim != 4 or int(encoded.shape[1]) != num_hist:
                raise RuntimeError(f"source encoded history is not [B,{num_hist},P,D]: {tuple(encoded.shape)}")
            history = encoded[0].detach().clone()
            if num_hist > 1 and float((history[1:] - history[:-1]).abs().max().item()) <= 1e-8:
                raise RuntimeError(f"episode {episode_id!r} produced a repeated fake history")
            fp_slot = adapter.one_step(encoded, current_action)
            all_actions = chunks.unsqueeze(0).to(device)
            bridge_expected, bridge_actual = adapter.source_rollout_one_step(model_obs, all_actions)
            bridge_error = float((bridge_expected - bridge_actual).abs().max().item())
            if bridge_error > 1e-5:
                raise RuntimeError(f"source rollout bridge mismatch for episode {episode_id!r}: {bridge_error:g}")
        history_np = _numpy(history, np.float32)
        fp_np = _numpy(fp_slot[0, 0], np.float32)
        action_np = _numpy(chunks[num_hist], np.float32)
        next_np = _numpy(chunks[num_hist + 1], np.float32)
        bridge = {
            "max_abs_adapter_vs_source_first_slot": bridge_error,
            "indices": {"history_action_chunks": list(range(num_hist)), "action": num_hist, "next_action": num_hist + 1},
            "rollout_action_count": int(chunks.shape[0]),
            "num_hist": num_hist,
            "frameskip": frameskip,
        }
        record: dict[str, Any] = {
            "record_index": int(record_index),
            "episode_id": episode_id,
            "dataset_index": int(requested_index),
            "subset_index": int(requested_index),
            "original_dataset_index": original_index,
            # Keep protocol identity fields on every flat record.  The runner
            # intentionally accepts flat payloads without inheriting episode
            # metadata, so these must not live only on the episode wrapper.
            "env_seed": int(env_seed),
            "cem_seed": int(cem_seed),
            "cem_seed_usage": "declared_identity_only; CEM not run by collector",
            "replay_seed": int(env_seed + record_index),
            "window_start": int(start),
            "history": history_np,
            "full_history": history_np,
            "history_shape": list(history_np.shape),
            "fp_reference_slot": fp_np,
            "fp_next_slot": fp_np,
            "fp_reference": fp_np,
            "action": action_np,
            "current_action": action_np,
            "next_action": next_np,
            "action_next": next_np,
            "actions": _numpy(chunks, np.float32),
            "initial_history_actions": _numpy(chunks[:num_hist], np.float32),
            "action_space": "source_preprocessor_normalized",
            "action_normalization": {
                "roundtrip_max_abs_error": roundtrip_error,
                "primitive_action_dim": primitive_dim,
                "chunk_action_dim": int(chunks.shape[-1]),
            },
            "source_bridge": bridge,
        }
        if record_index == 0:
            record["obs_0"] = {key: _numpy(value, np.float32) for key, value in model_obs.items()}
            record["obs_0_layout"] = "model_input_chw"
            record["source_bridge_obs_0"] = record["obs_0"]
            record["source_bridge_actions"] = _numpy(chunks, np.float32)
        if include_physical:
            goal_state = _numpy(states[start + (num_hist + 2) * frameskip], np.float32)
            record_seed = int(env_seed + record_index)
            record["physical_replay"] = _physical_replay(
                helper,
                runtime,
                preprocessor,
                info,
                _numpy(states[start], np.float32),
                chunks,
                frameskip,
                goal_state[:2],
                record_seed,
            )
        records.append(record)
        source_bridge_rows.append({"record_index": int(record_index), "window_start": int(start), "max_abs": bridge_error})
    return {
        "episode_id": episode_id,
        "dataset_index": int(requested_index),
        "subset_index": int(requested_index),
        "original_dataset_index": original_index,
        "env_seed": int(env_seed),
        "cem_seed": int(cem_seed),
        "record_count": len(records),
        "records": records,
        "source_bridge": {"records": source_bridge_rows, "distribution": "offline_contiguous_wall_dataset"},
        "layout": {key: float(_scalar(value)) for key, value in _layout(info, runtime["torch"]).items()},
    }


def _collect_split(
    helper: Any,
    core: Any,
    runtime: Mapping[str, Any],
    preprocessor: Any,
    dset: Any,
    pairs: Iterable[tuple[Any, int, int | None, int | None]],
    split: str,
    num_hist: int,
    frameskip: int,
    count: int,
    include_physical: bool,
    dataset_index_kind: str,
) -> Mapping[str, Any]:
    adapter = core.SourceHistoryAdapter(runtime["model"], num_hist=num_hist)
    episodes = []
    for episode_id, requested, requested_seed, requested_cem_seed in pairs:
        subset_index, original_index = _valid_subset_index(dset, requested, dataset_index_kind)
        seed = requested_seed if requested_seed is not None else 700000 + int(requested)
        cem_seed = requested_cem_seed if requested_cem_seed is not None else 800000 + int(requested)
        episodes.append(
            _episode_records(
                helper,
                core,
                runtime,
                preprocessor,
                dset,
                episode_id,
                subset_index,
                original_index,
                seed,
                cem_seed,
                num_hist,
                frameskip,
                count,
                include_physical,
                adapter,
            )
        )
    flat = [record for episode in episodes for record in episode["records"]]
    return {
        "schema": "frt-ccds-records-v1",
        "split": split,
        "source_distribution": "offline_contiguous_wall_dataset",
        "num_hist": num_hist,
        "frameskip": frameskip,
        "episode_count": len(episodes),
        "record_count": len(flat),
        "episodes": episodes,
        "records": flat,
        "physical_replay_included": bool(include_physical),
    }


def collect(args: argparse.Namespace) -> Mapping[str, Any]:
    allocation = _require_allocation()
    manifest = _load_json(args.manifest)
    root = args.root.resolve()
    output = args.output.resolve()
    requested_num_hist = manifest.get("num_hist")
    frameskip = int(manifest.get("frameskip", DEFAULT_FRAMESKIP))
    count = int(manifest.get("records_per_episode", DEFAULT_RECORDS_PER_EPISODE))
    if frameskip < 1 or count < 1 or count > MAX_RECORDS_PER_EPISODE:
        raise ValueError("invalid frameskip or bounded records_per_episode")
    helper = _load_helper(_find_helper(root, args.helper_dir.resolve() if args.helper_dir else None))
    core = _load_core()
    runtime = helper._runtime(root, device="cuda:0")
    model = runtime["model"]
    num_hist = int(getattr(model, "num_hist", -1))
    if num_hist < 1:
        raise RuntimeError(f"loaded source model has invalid num_hist={num_hist}")
    if requested_num_hist is not None and int(requested_num_hist) != num_hist:
        raise RuntimeError(
            f"manifest num_hist={int(requested_num_hist)} disagrees with the loaded source model num_hist={num_hist}"
        )
    if bool(getattr(model, "training", False)):
        model.eval()
    preprocessor = helper._preprocessor(runtime)
    dset = runtime["dset"]
    split_map = {"cal": _episode_pairs(manifest, "cal"), "dev": _episode_pairs(manifest, "dev")}
    dataset_kind = str(manifest.get("dataset_index_kind", "valid_subset"))
    if dataset_kind not in {"valid_subset", "original"}:
        raise ValueError(f"unsupported dataset_index_kind={dataset_kind!r}")
    collected = {}
    for split, pairs in split_map.items():
        collected[split] = _collect_split(
            helper,
            core,
            runtime,
            preprocessor,
            dset,
            pairs,
            split,
            num_hist,
            frameskip,
            count,
            include_physical=args.stage == "a",
            dataset_index_kind=dataset_kind,
        )
    # Build one predictor-only Q0 pass over the already encoded FP histories.
    import torch

    groups = helper._linear_groups(model)
    modules = []
    for group in groups:
        if group.get("family") != "predictor":
            continue
        for linear in group["linear"]:
            modules.append((linear["path"], helper._module_for_path(model, "predictor", group["index"], linear["relative"])))
    if not modules:
        raise RuntimeError("no predictor target blocks found for Q0 residual collection")
    snapshot = core.snapshot_modules(modules, cpu=True)
    q0_meta = core.materialize_rtn(modules)
    adapter = core.SourceHistoryAdapter(model, num_hist=num_hist)
    max_action_delta = 0.0
    try:
        with torch.no_grad():
            for payload in collected.values():
                for episode in payload["episodes"]:
                    for record in episode["records"]:
                        history = torch.as_tensor(record["history"], dtype=torch.float32, device=runtime["device"]).unsqueeze(0)
                        action = torch.as_tensor(record["current_action"], dtype=torch.float32, device=runtime["device"]).reshape(1, 1, -1)
                        q0_slot = adapter.one_step(history, action)
                        fp_slot = torch.as_tensor(record["fp_reference_slot"], dtype=torch.float32, device=runtime["device"]).unsqueeze(0).unsqueeze(0)
                        raw_delta = q0_slot - fp_slot
                        masked = core.mask_action_delta(raw_delta, adapter.action_mask(raw_delta))
                        action_mask = adapter.action_mask(raw_delta)
                        max_action_delta = max(max_action_delta, float(raw_delta.masked_select(action_mask).abs().max().item()))
                        record["q0_reference_slot"] = _numpy(q0_slot[0, 0])
                        record["q0_next_slot"] = record["q0_reference_slot"]
                        record["q0_delta_raw"] = _numpy(raw_delta[0, 0])
                        record["delta"] = _numpy(masked[0, 0])
                        record["residual"] = record["delta"]
                        record["delta_action_max_abs"] = float(raw_delta.masked_select(action_mask).abs().max().item())
    finally:
        core.restore_modules(modules, snapshot)
    if max_action_delta > 1e-5:
        raise RuntimeError(f"Q0 residual changed known action coordinates: max_abs={max_action_delta:g}")
    flat_all = [record for payload in collected.values() for record in payload["records"]]
    combined = {
        "schema": "frt-ccds-records-v1",
        "stage": args.stage,
        "source_distribution": "offline_contiguous_wall_dataset",
        "num_hist": num_hist,
        "frameskip": frameskip,
        "q0_scope": "predictor_target_blocks_only",
        "episodes_per_split": 6,
        "record_count": len(flat_all),
        "records": flat_all,
        "splits": collected,
    }
    output.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for split, payload in collected.items():
        path = output / f"{split}.pkl"
        _atomic_pickle(path, payload)
        split_paths[split] = str(path)
    records_path = output / "records.pkl"
    _atomic_pickle(records_path, combined)
    identity = helper._checkpoint_identity(runtime) if hasattr(helper, "_checkpoint_identity") else {}
    import copy

    q0_ledger_path = output / "q0_ledger.pkl"
    _atomic_pickle(
        q0_ledger_path,
        {
            "schema": "frt-ccds-q0-ledger-v1",
            "scope": "predictor_target_blocks_only",
            "groups": q0_meta,
            "num_hist": num_hist,
            "record_count": len(flat_all),
        },
    )
    out_manifest = copy.deepcopy(dict(manifest))
    old_splits = out_manifest.get("splits")
    if not isinstance(old_splits, Mapping):
        old_splits = {}
    # Preserve protocol reservation and identity keys such as fit/loss and
    # historical reserve splits while amending only CAL/DEV collection paths.
    amended_splits = dict(old_splits)
    for split, path in split_paths.items():
        original = old_splits.get(split)
        if isinstance(original, Mapping):
            amended = dict(original)
        elif original is None:
            amended = {}
        else:
            amended = {"episodes": original}
        amended.update({"path": path, "episode_count": 6, "record_count": collected[split]["record_count"]})
        amended_splits[split] = amended
    existing_stage_a = out_manifest.get("stage_a")
    stage_a_manifest = dict(existing_stage_a) if isinstance(existing_stage_a, Mapping) else {}
    if args.stage == "a":
        # Stage A's Wz fit must see CAL rows only; records.pkl remains a
        # combined archival payload for Stage B and later diagnostics.
        stage_a_manifest["records_path"] = split_paths["cal"]
    out_manifest.update({
        # Keep the canonical protocol schema from the input manifest.  The
        # collector payload has its own explicit schema below.
        "collection_schema": "frt-ccds-record-manifest-v1",
        "stage": args.stage,
        "source_distribution": "offline_contiguous_wall_dataset",
        "runtime_num_hist": num_hist,
        "num_hist": num_hist,
        "frameskip": frameskip,
        "records_per_episode": count,
        "records_path": str(records_path),
        "stage_a": stage_a_manifest,
        "splits": amended_splits,
        "hardware_provenance": dict(allocation),
        "checkpoint_identity": identity,
        "q0": {
            "scope": "predictor_target_blocks_only",
            "ledger_path": str(q0_ledger_path),
            "group_count": len(q0_meta),
            "action_delta_max_abs": max_action_delta,
        },
        "collection": {
            "dataset_index_kind": dataset_kind,
            "episode_ids": {split: [episode["episode_id"] for episode in payload["episodes"]] for split, payload in collected.items()},
            "physical_replay_included": args.stage == "a",
        },
    })
    _atomic_json(output / "manifest.json", out_manifest)
    _atomic_json(
        output / "summary.json",
        {
            "schema": "frt-ccds-record-summary-v1",
            "status": "complete",
            "stage": args.stage,
            "split_episode_counts": {split: payload["episode_count"] for split, payload in collected.items()},
            "split_record_counts": {split: payload["record_count"] for split, payload in collected.items()},
            "history_shape": sorted({tuple(record["history"].shape) for record in flat_all}),
            "physical_replay_included": args.stage == "a",
            "q0_scope": "predictor_target_blocks_only",
            "hardware_provenance": dict(allocation),
        },
    )
    return out_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="verified DINO-WM modelroot")
    parser.add_argument("--manifest", type=Path, required=True, help="small FRT protocol manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, default=None, help="directory containing existing smoke_runner.py")
    return parser.parse_args()


def main() -> None:
    result = collect(_parse_args())
    print(json.dumps({"status": "complete", "records_path": result["records_path"], "stage": result["stage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

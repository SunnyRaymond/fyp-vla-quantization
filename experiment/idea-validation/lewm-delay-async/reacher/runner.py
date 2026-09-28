"""Vanilla LeWM Reacher manifest, asset checks, and paired N=1 runner.

The selected starts and synchronous evaluator follow LeWM's pinned ``eval.py``
and ``config/eval/reacher.yaml``. This file intentionally imports the large
model/environment stack only after the PBS compute-allocation guard passes.
Fixed-delay and true-async modes reuse the shared scheduler in ``../pusht``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


LEWM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
MODEL_REVISION = "62adae4b71dc474ddf8f794c476ebfe737a743ca"
DATASET_REVISION = "e70a080d0d04c6072123c9ebd343acf7fff28dbf"
DATASET_NAME = "dmc/reacher_random"
CHECKPOINT_REL = Path("reacher/lewm_object.ckpt")
DATASET_REL = Path("dmc/reacher_random.h5")
MANIFEST_SCHEMA = "lewm-reacher-seed42-v1"
DELAY_STEPS = (1, 2, 4, 8, 16)


def require_pbs_compute_allocation() -> dict[str, str]:
    """Fail closed unless this process runs on a host allocated by PBS."""
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile_value = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile_value:
        raise RuntimeError("this mode reads Reacher data or runs evaluation; submit it in a PBS allocation")
    nodefile = Path(nodefile_value)
    if not nodefile.is_file():
        raise RuntimeError("PBS_NODEFILE is missing or not a file")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing non-compute host: {host}")
    allocated = {
        line.strip().split(".", 1)[0].lower()
        for line in nodefile.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    }
    if host not in allocated:
        raise RuntimeError(f"current host {host!r} is absent from PBS_NODEFILE")
    return {"pbs_job_id": job_id, "host": host, "nodefile": str(nodefile)}


def _field(rows: Any, key: str) -> Any:
    try:
        return rows[key]
    except (TypeError, KeyError, IndexError):
        return getattr(rows, key)


def sample_upstream_tasks(dataset: Any, *, seed: int = 42, num_eval: int = 50, goal_offset_steps: int = 25) -> list[dict[str, int]]:
    """Copy the pinned upstream eval.py valid-row sampling, including ordering."""
    import numpy as np

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = np.asarray(dataset.get_col_data(col_name))
    step_indices = np.asarray(dataset.get_col_data("step_idx"))
    ep_indices, _ = np.unique(episode_ids, return_index=True)

    episode_lengths = []
    for episode_id in ep_indices:
        episode_lengths.append(np.max(step_indices[episode_ids == episode_id]) + 1)
    max_start_idx = np.asarray(episode_lengths) - int(goal_offset_steps) - 1
    max_start_by_episode = {episode_id: max_start_idx[index] for index, episode_id in enumerate(ep_indices)}
    max_start_per_row = np.asarray([max_start_by_episode[episode_id] for episode_id in episode_ids])
    valid_indices = np.nonzero(step_indices <= max_start_per_row)[0]
    if len(valid_indices) <= num_eval:
        raise ValueError(f"only {len(valid_indices)} valid starts; upstream sampling requires more than {num_eval}")

    # Keep the original len(valid_indices)-1 upper bound and the final sort.
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    row_indices = np.sort(valid_indices[chosen])
    rows = dataset.get_row_data(row_indices)
    selected_episodes = np.asarray(_field(rows, col_name)).reshape(-1)
    selected_steps = np.asarray(_field(rows, "step_idx")).reshape(-1)
    if len(row_indices) != num_eval or len(selected_episodes) != num_eval or len(selected_steps) != num_eval:
        raise RuntimeError("upstream task sampling returned an unexpected number of rows")

    return [
        {
            "manifest_order": index,
            "row_index": int(row_index),
            "episode_idx": int(episode_id),
            "start_step": int(start_step),
            "goal_step": int(start_step) + int(goal_offset_steps),
        }
        for index, (row_index, episode_id, start_step) in enumerate(zip(row_indices, selected_episodes, selected_steps))
    ]


def make_manifest(dataset: Any) -> dict[str, Any]:
    tasks = sample_upstream_tasks(dataset)
    return {
        "schema": MANIFEST_SCHEMA,
        "source": f"lucas-maes/le-wm@{LEWM_COMMIT}",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "dataset_name": DATASET_NAME,
        "seed": 42,
        "num_eval": 50,
        "goal_offset_steps": 25,
        "sampling": "pinned eval.py valid rows; default_rng(42).choice(len(valid_indices)-1, 50, replace=False); sort selected row indices",
        "tasks": tasks,
    }


def task_slice(manifest: dict[str, Any], start: int, count: int) -> list[dict[str, int]]:
    tasks = manifest.get("tasks")
    if manifest.get("schema") != MANIFEST_SCHEMA or not isinstance(tasks, list) or len(tasks) != 50:
        raise ValueError("expected a frozen 50-row vanilla Reacher manifest")
    if manifest.get("source") != f"lucas-maes/le-wm@{LEWM_COMMIT}" or manifest.get("model_revision") != MODEL_REVISION or manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("manifest source or Hugging Face revision does not match the pinned Reacher assets")
    if start < 0 or count <= 0 or start + count > len(tasks):
        raise ValueError("task slice is outside the frozen manifest")
    selected = tasks[start : start + count]
    if [int(task["manifest_order"]) for task in selected] != list(range(start, start + count)):
        raise ValueError("manifest task order is incomplete or altered")
    return selected


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen task manifest: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = task_slice(manifest, 0, 50)
    row_ids = [task["row_index"] for task in tasks]
    if len(set(row_ids)) != 50:
        raise ValueError("upstream sampling should produce 50 distinct dataset rows")
    for task in tasks:
        if task["goal_step"] != task["start_step"] + 25:
            raise ValueError("goal step does not match the pinned 25-step offset")
    return manifest


def _task_identity(task: dict[str, Any]) -> tuple[int, int, int]:
    return (int(task["row_index"]), int(task["episode_idx"]), int(task["start_step"]))


def validate_batch50_gate(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Require the complete official synchronous K0 run on this manifest."""
    gate = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = [_task_identity(task) for task in manifest["tasks"]]
    observed = [_task_identity(task) for task in gate.get("tasks", [])]
    if gate.get("protocol") != "vanilla_lewm_reacher_sync_k0":
        raise RuntimeError("batch50 gate is not a vanilla official synchronous K0 result")
    if gate.get("manifest_schema") != manifest["schema"] or int(gate.get("task_count", -1)) != 50:
        raise RuntimeError("batch50 gate must contain all 50 tasks from this manifest")
    if observed != expected:
        raise RuntimeError("batch50 gate task identities/order differ from the frozen manifest")
    if "metrics" not in gate:
        raise RuntimeError("batch50 gate is missing synchronous evaluation metrics")
    return {"path": str(path.resolve()), "task_count": 50, "manifest_schema": manifest["schema"]}


def validate_n1_k0_gates(paths: Sequence[Path], manifest: dict[str, Any]) -> dict[str, Any]:
    """Require complete N1 K0 task results, accepting multiple PBS slice summaries."""
    if not paths:
        raise RuntimeError("fixed/async arms require at least one complete N1 K0 gate summary")
    expected = {int(task["manifest_order"]): _task_identity(task) for task in manifest["tasks"]}
    observed: dict[int, tuple[int, int, int]] = {}
    used_summaries = []
    for summary_path in paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if summary.get("protocol") != "vanilla_lewm_reacher_n1_sync_k0":
            raise RuntimeError(f"N1 K0 gate summary has wrong protocol: {summary_path}")
        if summary.get("manifest_schema") != manifest["schema"]:
            raise RuntimeError(f"N1 K0 gate manifest mismatch: {summary_path}")
        files = summary.get("task_result_files", [])
        if len(files) != int(summary.get("task_count", -1)):
            raise RuntimeError(f"N1 K0 summary task count differs from its result list: {summary_path}")
        for relative in files:
            task_path = summary_path.parent / relative
            if not task_path.is_file():
                raise FileNotFoundError(f"N1 K0 task result is missing: {task_path}")
            row = json.loads(task_path.read_text(encoding="utf-8-sig"))
            task = row.get("task", {})
            order = int(task.get("manifest_order", -1))
            if row.get("protocol") != "vanilla_lewm_reacher_n1_sync_k0" or row.get("manifest_schema") != manifest["schema"]:
                raise RuntimeError(f"N1 K0 task result has wrong protocol/manifest: {task_path}")
            if order not in expected or _task_identity(task) != expected[order]:
                raise RuntimeError(f"N1 K0 task identity differs from the frozen manifest: {task_path}")
            if row.get("planner_seed") != 42 + order or "metrics" not in row:
                raise RuntimeError(f"N1 K0 task result is missing its paired seed or metrics: {task_path}")
            if order in observed:
                raise RuntimeError(f"duplicate N1 K0 task {order} across gate summaries")
            observed[order] = _task_identity(task)
        used_summaries.append(str(summary_path.resolve()))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        raise RuntimeError(f"N1 K0 gate must cover all 50 manifest tasks; missing orders={missing}")
    return {"summary_paths": used_summaries, "task_count": 50, "manifest_schema": manifest["schema"]}


def _source_revision(lewm_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"expected": LEWM_COMMIT, "verified": False, "actual": None}
    if (lewm_root / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(lewm_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            actual = proc.stdout.strip()
            result.update(actual=actual, verified=(actual == LEWM_COMMIT))
            if actual != LEWM_COMMIT:
                raise RuntimeError(f"LeWM source revision mismatch: expected {LEWM_COMMIT}, found {actual}")
    return result


def _compose_config(lewm_root: Path, *, num_envs: int, policy: str = "reacher/lewm") -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = (lewm_root / "config" / "eval").resolve(strict=True)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="reacher",
            overrides=[f"eval.num_eval={num_envs}", f"policy={policy}"],
        )
    OmegaConf.resolve(cfg)
    _validate_config(cfg, num_envs, policy)
    return cfg


def _validate_config(cfg: Any, num_envs: int, policy: str) -> None:
    expected = (
        ("world.env_name", cfg.world.env_name, "swm/ReacherDMControl-v0"),
        ("world.task", cfg.world.task, "qpos_match"),
        ("seed", cfg.seed, 42),
        ("eval.num_eval", cfg.eval.num_eval, num_envs),
        ("eval.goal_offset_steps", cfg.eval.goal_offset_steps, 25),
        ("eval.eval_budget", cfg.eval.eval_budget, 50),
        ("eval.img_size", cfg.eval.img_size, 224),
        ("eval.dataset_name", cfg.eval.dataset_name, DATASET_NAME),
        ("policy", cfg.policy, policy),
        ("plan_config.horizon", cfg.plan_config.horizon, 5),
        ("plan_config.receding_horizon", cfg.plan_config.receding_horizon, 5),
        ("plan_config.action_block", cfg.plan_config.action_block, 5),
        ("world.num_envs", cfg.world.num_envs, num_envs),
        ("solver.num_samples", cfg.solver.num_samples, 300),
        ("solver.topk", cfg.solver.topk, 30),
        ("solver.n_steps", cfg.solver.n_steps, 30),
        ("solver.batch_size", cfg.solver.batch_size, 1),
        ("solver.seed", cfg.solver.seed, 42),
    )
    mismatches = [f"{name}: expected {wanted!r}, got {actual!r}" for name, actual, wanted in expected if actual != wanted]
    if mismatches:
        raise RuntimeError("vanilla Reacher config differs from pinned protocol: " + "; ".join(mismatches))
    if cfg.solver.get("_target_", "").endswith("CEMSolver") is False:
        raise RuntimeError("pinned evaluator requires CEMSolver")
    expected_callables = [
        {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
        {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
    ]
    actual_callables = [
        {"method": item.method, "args": {key: {"value": value.value} for key, value in item.args.items()}}
        for item in cfg.eval.callables
    ]
    if actual_callables != expected_callables:
        raise RuntimeError("pinned Reacher reset/goal callables changed")


def check_assets(lewm_root: Path) -> dict[str, Any]:
    """Verify staged file paths and cache resolution without loading weights."""
    home_value = os.environ.get("STABLEWM_HOME", "").strip()
    if not home_value:
        raise RuntimeError("STABLEWM_HOME must point at the prepared stable-worldmodel asset root")
    home = Path(home_value).resolve(strict=True)
    checkpoint = home / CHECKPOINT_REL
    dataset = home / DATASET_REL
    for label, path in (("checkpoint", checkpoint), ("dataset", dataset)):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"prepared Reacher {label} is missing or empty: {path}")

    sys.path.insert(0, str(lewm_root))
    import stable_worldmodel as swm

    cache_root = Path(swm.data.utils.get_cache_dir()).resolve(strict=True)
    if cache_root != home:
        raise RuntimeError(f"stable-worldmodel cache resolves to {cache_root}, but STABLEWM_HOME is {home}")
    cached_dataset = cache_root / "datasets" / DATASET_REL
    cached_dataset.parent.mkdir(parents=True, exist_ok=True)
    staged_target = dataset.resolve(strict=True)
    if cached_dataset.is_symlink():
        if cached_dataset.resolve(strict=True) != staged_target:
            raise RuntimeError(f"dataset cache symlink points elsewhere: {cached_dataset} -> {cached_dataset.resolve()}")
    elif cached_dataset.exists():
        if not cached_dataset.is_file() or not os.path.samefile(cached_dataset, staged_target):
            raise RuntimeError(f"dataset cache path is occupied by a different file: {cached_dataset}")
    else:
        try:
            os.symlink(staged_target, cached_dataset)
        except FileExistsError:
            if not cached_dataset.exists() or not os.path.samefile(cached_dataset, staged_target):
                raise RuntimeError(f"concurrent cache-link creation produced an unexpected target: {cached_dataset}")
    import dm_control  # noqa: F401
    import mujoco  # noqa: F401

    revision = _source_revision(lewm_root)
    return {
        "stablewm_home": str(home),
        "cache_root": str(cache_root),
        "checkpoint": str(checkpoint.resolve(strict=True)),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "dataset": str(dataset.resolve(strict=True)),
        "dataset_bytes": dataset.stat().st_size,
        "dataset_cache_path": str(cached_dataset),
        "dataset_cache_target": str(cached_dataset.resolve(strict=True)),
        "source_revision": revision,
    }


def _load_dataset(cfg: Any, lewm_root: Path) -> Any:
    sys.path.insert(0, str(lewm_root))
    import stable_worldmodel as swm

    cache_dir = cfg.get("cache_dir") or swm.data.utils.get_cache_dir()
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )


def _load_model_and_transforms(cfg: Any, dataset: Any, lewm_root: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import hydra
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    sys.path.insert(0, str(lewm_root))
    process: dict[str, Any] = {}
    for column in cfg.dataset.keys_to_cache:
        if column == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        values = np.asarray(dataset.get_col_data(column))
        values = values[~np.isnan(values).any(axis=1)]
        scaler.fit(values)
        process[column] = scaler
        if column != "action":
            process[f"goal_{column}"] = scaler

    def image_transform():
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                transforms.Resize(size=cfg.eval.img_size),
            ]
        )

    transform = {"pixels": image_transform(), "goal": image_transform()}
    checkpoint = Path(os.environ["STABLEWM_HOME"]).resolve(strict=True) / CHECKPOINT_REL
    # The pinned HF conversion emits the historical full-module *_object.ckpt.
    # The stable-worldmodel HF loader expects a state_dict .pt + config.json,
    # so load this local serialized model directly and do not trigger a download.
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, process, transform


def _make_policy(cfg: Any, model: Any, process: dict[str, Any], transform: dict[str, Any], *, planner_seed: int | None = None) -> Any:
    import hydra
    import stable_worldmodel as swm

    if planner_seed is not None:
        # `solver.seed` is an interpolation of `seed` in the pinned Hydra YAML;
        # set both after compose because this runner resolves interpolation.
        cfg.seed = int(planner_seed)
        cfg.solver.seed = int(planner_seed)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    plan_config = swm.PlanConfig(**cfg.plan_config)
    return swm.policy.WorldModelPolicy(solver=solver, config=plan_config, process=process, transform=transform)


def _walk_env_objects(root: Any, max_depth: int = 8) -> list[Any]:
    attrs = ("env", "_env", "unwrapped", "environment", "_environment", "dm_env", "_dm_env", "wrapped_env", "_wrapped_env")
    seen: set[int] = set()
    queue: list[tuple[Any, int]] = [(root, 0)]
    found: list[Any] = []
    while queue:
        current, depth = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        found.append(current)
        if depth >= max_depth:
            continue
        for name in attrs:
            try:
                child = getattr(current, name)
            except Exception:
                continue
            if child is not current and child is not None:
                queue.append((child, depth + 1))
    return found


def world_tick_seconds(world: Any, *, validated_wrapper_repeat: int | None = None) -> dict[str, Any]:
    """Read DMC control timestep and wrapper repeat; fail if either is unknown.

    A DMC environment timestep is not necessarily one Reacher world tick. The
    wrapper repeat must come from runtime metadata or an explicitly frozen
    value validated against the pinned wrapper source (vanilla source repeats
    each action twice).
    """
    pool = getattr(world, "env_pool", None) or getattr(world, "envs", None)
    envs = getattr(pool, "envs", None)
    if not envs:
        raise RuntimeError("cannot inspect Reacher environment pool for timing metadata")
    nodes = _walk_env_objects(envs[0])
    control_seconds = None
    control_source = None
    for node in nodes:
        try:
            value = getattr(node, "control_timestep")
            value = value() if callable(value) else value
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value) and value > 0:
            control_seconds = value
            control_source = f"{type(node).__module__}.{type(node).__name__}.control_timestep"
            break

    repeat = None
    repeat_source = None
    repeat_attrs = ("action_repeat", "_action_repeat", "frame_skip", "_frame_skip", "frameskip", "_frameskip")
    for node in nodes:
        for name in repeat_attrs:
            try:
                value = int(getattr(node, name))
            except Exception:
                continue
            if value > 0:
                repeat = value
                repeat_source = f"{type(node).__module__}.{type(node).__name__}.{name}"
                break
        if repeat is not None:
            break
    if repeat is None and validated_wrapper_repeat is not None:
        if validated_wrapper_repeat <= 0:
            raise ValueError("validated_wrapper_repeat must be positive")
        repeat = int(validated_wrapper_repeat)
        repeat_source = "explicit_pinned_wrapper_override"
    if control_seconds is None:
        raise RuntimeError("pinned DMC control_timestep() is unavailable; true-async frequency is not measurable")
    if repeat is None:
        raise RuntimeError("Reacher action repeat is not exposed; freeze a verified wrapper-repeat value before true-async")
    return {
        "dmc_control_timestep_seconds": control_seconds,
        "wrapper_action_repeat": repeat,
        "wrapper_repeat_source": repeat_source,
        "control_timestep_source": control_source,
        "world_tick_seconds": control_seconds * repeat,
    }


def adapter_tick_metadata(timing: dict[str, Any]) -> dict[str, Any]:
    """Adapt the pinned DMC timing record to the shared scheduler interface."""
    seconds = float(timing["world_tick_seconds"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("Reacher world tick must be finite and positive")
    return {
        **timing,
        "source": "stable_worldmodel Reacher DMC control_timestep x wrapper action_repeat",
        "control_tick_seconds": seconds,
    }


def zero_midpoint_action(world: Any) -> tuple[Any, dict[str, Any]]:
    """Read the live Reacher action bounds and require a zero neutral midpoint."""
    import numpy as np

    pool = getattr(world, "envs", None)
    envs = getattr(pool, "envs", None)
    if not envs or len(envs) != 1:
        raise RuntimeError("paired Reacher arms require one live environment")
    env = envs[0]
    unwrapped = getattr(env, "unwrapped", env)
    space = getattr(unwrapped, "action_space", None)
    if space is None:
        raise RuntimeError("cannot inspect the live Reacher action space")
    low = np.asarray(space.low)
    high = np.asarray(space.high)
    if low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise RuntimeError("Reacher neutral action requires finite matching action bounds")
    midpoint = (low.astype(np.float64) + high.astype(np.float64)) / 2.0
    if not np.allclose(midpoint, 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Reacher action bounds midpoint is not zero: {midpoint.tolist()}")
    dtype = np.dtype(getattr(space, "dtype", np.float32))
    action = midpoint.astype(dtype, copy=True)
    return action, {
        "action_space_source": f"{type(unwrapped).__module__}.{type(unwrapped).__name__}.action_space",
        "action_shape": list(low.shape),
        "action_dtype": str(dtype),
        "action_low": low.tolist(),
        "action_high": high.tolist(),
        "neutral_action": action.tolist(),
        "neutral_rule": "verified_action_bounds_midpoint_zero",
    }


def _load_shared_adapter() -> Any:
    """Load the single shared delay scheduler beside the PushT adapter."""
    import importlib.util

    adapter_path = Path(__file__).resolve().parent.parent / "pusht" / "adapter.py"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"shared delay scheduler is missing: {adapter_path}")
    spec = importlib.util.spec_from_file_location("_lewm_shared_delay_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared delay scheduler: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DelayAsyncAdapter


def scheduler_event_summary(events: Sequence[dict[str, Any]], tick_seconds: float) -> dict[str, Any]:
    """Summarize first-call latency and per-plan simulated overlap from events."""
    ready = sorted(
        (event for event in events if event.get("event") == "inference_ready"),
        key=lambda event: int(event.get("plan_id", -1)),
    )
    outputs = [event for event in events if event.get("event") == "action_output"]
    plans = []
    for event in ready:
        plan_id = int(event["plan_id"])
        observed_step = int(event.get("observed_step", 0))
        available_step = int(event.get("available_step", observed_step))
        plan_outputs = [item for item in outputs if item.get("plan_id") == plan_id]
        holds = [
            item for item in plan_outputs
            if item.get("action_source") in {"startup_neutral", "fixed_delay_hold", "async_hold"}
        ]
        native_buffer_actions = sum(item.get("action_source") == "native_buffer" for item in plan_outputs)
        applied = [item for item in plan_outputs if item.get("action_source") in {"fixed_delay_release", "async_ready"}]
        plans.append({
            "plan_id": plan_id,
            "kind": event.get("kind"),
            "planner_seconds": float(event.get("inference_seconds", 0.0)),
            "observation_to_ready_seconds": max(0, int(event.get("finished_ns", 0)) - int(event.get("observed_ns", 0))) / 1e9,
            "observed_control_step": observed_step,
            "available_control_step": available_step,
            "simulated_overlap_steps": max(0, available_step - observed_step),
            "simulated_overlap_seconds": max(0, available_step - observed_step) * tick_seconds,
            "hold_action_outputs": len(holds),
            "plan_application_outputs": len(applied),
            "subsequent_native_buffer_action_outputs": native_buffer_actions,
        })
    return {
        "first_call": plans[0] if plans else None,
        "later_plans": plans[1:],
        "inference_plan_count": len(plans),
    }


def episode_success_bool(metrics: Any) -> bool:
    """Extract exactly one boolean from the pinned evaluator metric."""
    import numpy as np

    if not isinstance(metrics, dict) or "episode_successes" not in metrics:
        raise RuntimeError("evaluation metrics must contain episode_successes")
    value = metrics["episode_successes"]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    values = np.asarray(value, dtype=object).reshape(-1)
    if values.size != 1:
        raise RuntimeError(f"expected exactly one episode_successes value, got {values.size}")
    item = values[0]
    if not isinstance(item, (bool, np.bool_)):
        raise RuntimeError(f"episode_successes must contain a boolean, got {type(item).__name__}")
    return bool(item)


def n1_outcome_fields(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return ordered task identities and the aligned boolean outcome vector."""
    task_ids = []
    successes = []
    for row in rows:
        task = row["task"]
        success = row.get("success")
        if not isinstance(success, bool):
            raise RuntimeError("every N=1 task row must contain one boolean success value")
        task_ids.append({
            key: int(task[key])
            for key in ("manifest_order", "row_index", "episode_idx", "start_step", "goal_step")
        })
        successes.append(success)
    return {
        "ordered_task_ids": task_ids,
        "ordered_successes": successes,
        "success_count": sum(successes),
    }


def run_n1_scheduler_slice(
    lewm_root: Path,
    manifest: dict[str, Any],
    tasks: Sequence[dict[str, int]],
    *,
    output_dir: Path,
    mode: str,
    delay_steps: int = 0,
    validated_wrapper_repeat: int | None = None,
) -> dict[str, Any]:
    """Run fixed-delay or strict true-async Reacher tasks using shared scheduler."""
    if mode not in {"fixed_steps", "true_async"}:
        raise ValueError("scheduler mode must be fixed_steps or true_async")
    if not tasks or len(tasks) > 50:
        raise ValueError("N=1 scheduler task slice must contain between 1 and 50 tasks")
    if mode == "fixed_steps" and delay_steps not in DELAY_STEPS:
        raise ValueError(f"fixed_steps K must be one of {DELAY_STEPS}")
    if mode == "true_async" and delay_steps != 0:
        raise ValueError("true_async does not use a fixed delay_steps value")
    arm_name = f"{mode}_k{delay_steps}" if mode == "fixed_steps" else "true_async"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [output_dir / f"task_{int(task['manifest_order']):03d}.json" for task in tasks]
    existing = [str(path) for path in expected_paths if path.exists()]
    if existing or (output_dir / "summary.json").exists():
        raise FileExistsError("refusing to overwrite scheduler outputs: " + ", ".join(existing or [str(output_dir / 'summary.json')]))

    sys.path.insert(0, str(lewm_root))
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf

    Adapter = _load_shared_adapter()
    cfg = _compose_config(lewm_root, num_envs=1)
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    action_block = int(cfg.plan_config.receding_horizon) * int(cfg.plan_config.action_block)
    dataset = _load_dataset(cfg, lewm_root)
    model, process, transform = _load_model_and_transforms(cfg, dataset, lewm_root)
    world = swm.World(**cfg.world, image_shape=(224, 224))
    rows: list[dict[str, Any]] = []
    try:
        tick = world_tick_seconds(world, validated_wrapper_repeat=validated_wrapper_repeat)
        tick_for_adapter = adapter_tick_metadata(tick)
        neutral_action, action_space_meta = zero_midpoint_action(world)
        for task in tasks:
            order = int(task["manifest_order"])
            planner_seed = 42 + order
            task_policy = _make_policy(cfg, model, process, transform, planner_seed=planner_seed)
            world.set_policy(task_policy)
            adapter = Adapter(
                task_policy,
                world.envs,
                task_ids=[order],
                mode=mode,
                delay_steps=delay_steps,
                tick_seconds=float(tick_for_adapter["control_tick_seconds"]),
                tick_metadata=tick_for_adapter,
                neutral_env_action=neutral_action,
                action_block=action_block,
                synchronize=torch.cuda.synchronize,
            )
            try:
                metrics = world.evaluate(
                    dataset=dataset,
                    start_steps=[int(task["start_step"])],
                    goal_offset=int(cfg.eval.goal_offset_steps),
                    eval_budget=int(cfg.eval.eval_budget),
                    episodes_idx=[int(task["episode_idx"])],
                    callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
                    video=None,
                )
            finally:
                adapter.close()
            events = list(adapter.events)
            timing_summary = scheduler_event_summary(events, float(tick["world_tick_seconds"]))
            row = {
                "protocol": f"vanilla_lewm_reacher_n1_{arm_name}",
                "source_commit_expected": LEWM_COMMIT,
                "model_revision": manifest["model_revision"],
                "dataset_revision": manifest["dataset_revision"],
                "manifest_schema": manifest["schema"],
                "task": dict(task),
                "planner_seed": planner_seed,
                "mode": mode,
                "delay_steps": delay_steps if mode == "fixed_steps" else None,
                "action_block": action_block,
                "tick_metadata": tick_for_adapter,
                "neutral_action_metadata": action_space_meta,
                "first_call_and_buffer_overlap": timing_summary,
                "scheduler_events": events,
                "success": episode_success_bool(metrics),
                "metrics": _jsonable(metrics),
            }
            task_path = output_dir / f"task_{order:03d}.json"
            task_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
        summary = {
            "protocol": f"vanilla_lewm_reacher_n1_{arm_name}",
            "manifest_schema": manifest["schema"],
            "task_start": int(tasks[0]["manifest_order"]),
            "task_count": len(tasks),
            "planner_seed_rule": "42 + manifest_order",
            "mode": mode,
            "delay_steps": delay_steps if mode == "fixed_steps" else None,
            "action_block": action_block,
            "tick_metadata": tick_for_adapter,
            "neutral_action_metadata": action_space_meta,
            "task_result_files": [f"task_{int(task['manifest_order']):03d}.json" for task in tasks],
            **n1_outcome_fields(rows),
            "first_call_latencies_seconds": [
                row["first_call_and_buffer_overlap"]["first_call"]["observation_to_ready_seconds"]
                if row["first_call_and_buffer_overlap"]["first_call"] is not None else None
                for row in rows
            ],
            "later_plan_overlap_steps_by_task": {
                str(int(row["task"]["manifest_order"])): [
                    plan["simulated_overlap_steps"]
                    for plan in row["first_call_and_buffer_overlap"]["later_plans"]
                ]
                for row in rows
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        close = getattr(world, "close", None)
        if callable(close):
            close()


def run_sync(
    lewm_root: Path,
    manifest: dict[str, Any],
    tasks: Sequence[dict[str, int]],
    *,
    output_dir: Path,
    device: str = "cuda",
    validated_wrapper_repeat: int | None = None,
) -> dict[str, Any]:
    """Run the unmodified vanilla World.evaluate path as K=0 for a task slice."""
    sys.path.insert(0, str(lewm_root))
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    if device != "cuda":
        raise ValueError("pinned CEM Reacher runner is configured for CUDA")
    if not tasks:
        raise ValueError("K=0 runner requires at least one manifest task")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite result: {output_dir / 'result.json'}")
    cfg = _compose_config(lewm_root, num_envs=len(tasks))
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)  # same assignment as pinned eval.py
    dataset = _load_dataset(cfg, lewm_root)
    model, process, transform = _load_model_and_transforms(cfg, dataset, lewm_root)
    policy = _make_policy(cfg, model, process, transform)
    world = swm.World(**cfg.world, image_shape=(224, 224))
    tick_meta = None
    try:
        try:
            tick_meta = world_tick_seconds(world, validated_wrapper_repeat=validated_wrapper_repeat)
        except RuntimeError:
            # Sync baseline can still validate task outcomes; true-async entry
            # points must require this metadata and are fail-closed.
            tick_meta = None
        world.set_policy(policy)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=[int(task["start_step"]) for task in tasks],
            goal_offset=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=[int(task["episode_idx"]) for task in tasks],
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
            video=None,
        )
        result = {
            "protocol": "vanilla_lewm_reacher_sync_k0",
            "source_commit_expected": LEWM_COMMIT,
            "model_revision": manifest["model_revision"],
            "dataset_revision": manifest["dataset_revision"],
            "manifest_schema": manifest["schema"],
            "manifest_order_start": tasks[0]["manifest_order"],
            "tasks": list(tasks),
            "seed": int(cfg.seed),
            "task_count": len(tasks),
            "tick_metadata": tick_meta,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metrics": _jsonable(metrics),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        close = getattr(world, "close", None)
        if callable(close):
            close()


def run_n1_sync_slice(
    lewm_root: Path,
    manifest: dict[str, Any],
    tasks: Sequence[dict[str, int]],
    *,
    output_dir: Path,
    validated_wrapper_repeat: int | None = None,
) -> dict[str, Any]:
    """Run paired N=1 synchronous tasks while loading the model/data once.

    Each task gets a fresh WorldModelPolicy/CEM solver on the same one-env
    World, seeded with ``42 + manifest_order``. ``World.evaluate`` resets the
    environment and reapplies the fixed start/goal callables for each row.
    """
    if not tasks or len(tasks) > 50:
        raise ValueError("N=1 task slice must contain between 1 and 50 tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [output_dir / f"task_{int(task['manifest_order']):03d}.json" for task in tasks]
    existing = [str(path) for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite task outputs: " + ", ".join(existing))
    sys.path.insert(0, str(lewm_root))
    import stable_worldmodel as swm
    from omegaconf import OmegaConf

    cfg = _compose_config(lewm_root, num_envs=1)
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    dataset = _load_dataset(cfg, lewm_root)
    model, process, transform = _load_model_and_transforms(cfg, dataset, lewm_root)
    world = swm.World(**cfg.world, image_shape=(224, 224))
    try:
        try:
            tick_meta = world_tick_seconds(world, validated_wrapper_repeat=validated_wrapper_repeat)
        except RuntimeError:
            # Sync parity does not depend on pacing metadata. Any async caller
            # must call world_tick_seconds strictly and stop if it cannot resolve.
            tick_meta = None
        rows: list[dict[str, Any]] = []
        for task in tasks:
            order = int(task["manifest_order"])
            planner_seed = 42 + order
            task_policy = _make_policy(cfg, model, process, transform, planner_seed=planner_seed)
            # set_policy calls set_env and _set_seed, resetting policy buffers
            # and CEM's native generator before this independent task.
            world.set_policy(task_policy)
            metrics = world.evaluate(
                dataset=dataset,
                start_steps=[int(task["start_step"])],
                goal_offset=int(cfg.eval.goal_offset_steps),
                eval_budget=int(cfg.eval.eval_budget),
                episodes_idx=[int(task["episode_idx"])],
                callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
                video=None,
            )
            row = {
                "protocol": "vanilla_lewm_reacher_n1_sync_k0",
                "source_commit_expected": LEWM_COMMIT,
                "model_revision": manifest["model_revision"],
                "dataset_revision": manifest["dataset_revision"],
                "manifest_schema": manifest["schema"],
                "task": dict(task),
                "planner_seed": planner_seed,
                "task_count": 1,
                "tick_metadata": tick_meta,
                "success": episode_success_bool(metrics),
                "metrics": _jsonable(metrics),
            }
            task_path = output_dir / f"task_{order:03d}.json"
            task_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
        summary = {
            "protocol": "vanilla_lewm_reacher_n1_sync_k0",
            "manifest_schema": manifest["schema"],
            "task_start": int(tasks[0]["manifest_order"]),
            "task_count": len(tasks),
            "planner_seed_rule": "42 + manifest_order",
            "tick_metadata": tick_meta,
            "task_result_files": [f"task_{int(task['manifest_order']):03d}.json" for task in tasks],
            **n1_outcome_fields(rows),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        close = getattr(world, "close", None)
        if callable(close):
            close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _self_check() -> None:
    import numpy as np

    class TinyDataset:
        column_names = ["episode_idx", "step_idx"]

        def __init__(self) -> None:
            self.episodes = np.repeat(np.arange(3), 8)
            self.steps = np.tile(np.arange(8), 3)

        def get_col_data(self, key: str) -> Any:
            return self.episodes if key == "episode_idx" else self.steps

        def get_row_data(self, indices: Sequence[int]) -> dict[str, Any]:
            return {"episode_idx": self.episodes[indices], "step_idx": self.steps[indices]}

    sampled = sample_upstream_tasks(TinyDataset(), num_eval=3, goal_offset_steps=2)
    assert len(sampled) == 3
    assert [task["manifest_order"] for task in sampled] == [0, 1, 2]
    assert all(task["goal_step"] == task["start_step"] + 2 for task in sampled)
    valid_rows = np.concatenate((np.arange(0, 6), np.arange(8, 14), np.arange(16, 22)))
    expected_rows = np.sort(valid_rows[np.random.default_rng(42).choice(len(valid_rows) - 1, size=3, replace=False)])
    assert [task["row_index"] for task in sampled] == expected_rows.tolist()
    # Slice boundary logic is tested independently of file I/O and assets.
    tasks = [
        {
            "manifest_order": index,
            "row_index": index,
            "episode_idx": index // 2,
            "start_step": index,
            "goal_step": index + 25,
        }
        for index in range(50)
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": f"lucas-maes/le-wm@{LEWM_COMMIT}",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "tasks": tasks,
    }
    assert [item["manifest_order"] for item in task_slice(manifest, 17, 4)] == [17, 18, 19, 20]
    try:
        task_slice(manifest, 49, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range task slice should fail")

    class FakeDmc:
        def control_timestep(self) -> float:
            return 0.0075

    class FakeReacherWrapper:
        action_repeat = 2
        env = FakeDmc()

    class FakeEnvPool:
        envs = [FakeReacherWrapper()]

    class FakeWorld:
        envs = FakeEnvPool()

    timing = world_tick_seconds(FakeWorld())
    assert timing["world_tick_seconds"] == 0.015
    assert timing["wrapper_action_repeat"] == 2

    class FakeActionSpace:
        low = np.asarray([-1.0, -2.0], dtype=np.float32)
        high = np.asarray([1.0, 2.0], dtype=np.float32)
        dtype = np.float32
        shape = (2,)

    class FakeActionEnv:
        unwrapped = None
        action_space = FakeActionSpace()

    FakeActionEnv.unwrapped = FakeActionEnv

    class FakeActionPool:
        envs = [FakeActionEnv()]

    class FakeActionWorld:
        envs = FakeActionPool()

    neutral, action_meta = zero_midpoint_action(FakeActionWorld())
    assert neutral.tolist() == [0.0, 0.0]
    assert action_meta["neutral_rule"] == "verified_action_bounds_midpoint_zero"
    assert adapter_tick_metadata(timing)["control_tick_seconds"] == 0.015

    class OffCenterActionSpace(FakeActionSpace):
        low = np.asarray([0.0, -1.0], dtype=np.float32)
        high = np.asarray([2.0, 1.0], dtype=np.float32)

    class OffCenterEnv:
        unwrapped = None
        action_space = OffCenterActionSpace()

    OffCenterEnv.unwrapped = OffCenterEnv

    class OffCenterPool:
        envs = [OffCenterEnv()]

    class OffCenterWorld:
        envs = OffCenterPool()

    try:
        zero_midpoint_action(OffCenterWorld())
    except RuntimeError:
        pass
    else:
        raise AssertionError("nonzero action-space midpoint should fail closed")

    import inspect
    import tempfile

    adapter_parameters = inspect.signature(_load_shared_adapter()).parameters
    assert "tick_metadata" in adapter_parameters and "neutral_env_action" in adapter_parameters
    assert episode_success_bool({"episode_successes": np.asarray([True])}) is True
    assert episode_success_bool({"episode_successes": [False]}) is False
    for malformed in (
        {},
        {"episode_successes": [True, False]},
        {"episode_successes": [1]},
    ):
        try:
            episode_success_bool(malformed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("malformed episode_successes must fail closed")
    mini_rows = [
        {"task": {"manifest_order": 0, "row_index": 12, "episode_idx": 2, "start_step": 3, "goal_step": 28}, "success": True},
        {"task": {"manifest_order": 1, "row_index": 19, "episode_idx": 4, "start_step": 5, "goal_step": 30}, "success": False},
    ]
    mini_outcomes = n1_outcome_fields(mini_rows)
    assert mini_outcomes["ordered_successes"] == [True, False]
    assert mini_outcomes["success_count"] == 1
    assert [task["manifest_order"] for task in mini_outcomes["ordered_task_ids"]] == [0, 1]
    event_summary = scheduler_event_summary(
        [
            {"event": "inference_ready", "plan_id": 0, "kind": "startup", "inference_seconds": 0.4,
             "observed_ns": 10, "finished_ns": 500000010, "observed_step": 0, "available_step": 3},
            {"event": "action_output", "plan_id": 0, "action_source": "startup_neutral"},
        ],
        0.015,
    )
    assert event_summary["first_call"]["observation_to_ready_seconds"] == 0.5
    assert event_summary["first_call"]["simulated_overlap_steps"] == 3

    fake_tasks = [
        {"manifest_order": index, "row_index": index + 100, "episode_idx": index // 2, "start_step": index + 3}
        for index in range(50)
    ]
    fake_manifest = {"schema": MANIFEST_SCHEMA, "tasks": fake_tasks}
    with tempfile.TemporaryDirectory(prefix="reacher-runner-self-check-") as temp:
        root = Path(temp)
        batch_path = root / "batch50.json"
        batch_path.write_text(json.dumps({
            "protocol": "vanilla_lewm_reacher_sync_k0",
            "manifest_schema": MANIFEST_SCHEMA,
            "task_count": 50,
            "tasks": fake_tasks,
            "metrics": {"success": 0.5},
        }), encoding="utf-8")
        assert validate_batch50_gate(batch_path, fake_manifest)["task_count"] == 50
        slice_summaries = []
        for start in (0, 25):
            slice_dir = root / f"slice_{start:02d}"
            slice_dir.mkdir()
            task_files = []
            for task in fake_tasks[start:start + 25]:
                order = task["manifest_order"]
                task_file = f"task_{order:03d}.json"
                (slice_dir / task_file).write_text(json.dumps({
                    "protocol": "vanilla_lewm_reacher_n1_sync_k0",
                    "manifest_schema": MANIFEST_SCHEMA,
                    "task": task,
                    "planner_seed": 42 + order,
                    "metrics": {"success": True},
                }), encoding="utf-8")
                task_files.append(task_file)
            summary_path = slice_dir / "summary.json"
            summary_path.write_text(json.dumps({
                "protocol": "vanilla_lewm_reacher_n1_sync_k0",
                "manifest_schema": MANIFEST_SCHEMA,
                "task_count": 25,
                "task_result_files": task_files,
            }), encoding="utf-8")
            slice_summaries.append(summary_path)
        assert validate_n1_k0_gates(slice_summaries, fake_manifest)["task_count"] == 50
    print("reacher_runner_self_check=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-check", "check-assets", "freeze-manifest", "batch50-k0", "n1-k0", "fixed_steps", "true_async"), required=True)
    parser.add_argument("--lewm-root", type=Path, default=Path("/scratch/users/ntu/yguo017/lewm-pusht-iteration/le-wm"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--validated-wrapper-repeat", type=int, default=None)
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument("--batch50-gate", type=Path)
    parser.add_argument("--n1-k0-gate", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.mode == "self-check":
        _self_check()
        return

    allocation = require_pbs_compute_allocation()
    lewm_root = args.lewm_root.resolve(strict=True)
    if args.mode == "check-assets":
        print(json.dumps({"allocation": allocation, "assets": check_assets(lewm_root)}, indent=2))
        return

    assets = check_assets(lewm_root)
    if args.mode == "freeze-manifest":
        if args.output is None:
            raise ValueError("--output is required for freeze-manifest")
        cfg = _compose_config(lewm_root, num_envs=50, policy="random")
        dataset = _load_dataset(cfg, lewm_root)
        manifest = make_manifest(dataset)
        manifest["asset_paths"] = assets
        write_manifest(args.output, manifest)
        print(f"frozen_manifest={args.output} task_count={len(manifest['tasks'])}")
        return

    if args.manifest is None or args.output is None:
        raise ValueError("--manifest and --output are required for evaluation modes")
    manifest = read_manifest(args.manifest)
    gate_metadata: dict[str, Any] = {}
    if args.mode == "batch50-k0":
        if args.task_start != 0 or args.task_count != 50:
            raise ValueError("batch50-k0 requires the complete ordered 50-task manifest")
        tasks = task_slice(manifest, 0, 50)
    else:
        if not 1 <= args.task_count <= 50:
            raise ValueError("N=1 task-count must be between 1 and 50")
        tasks = task_slice(manifest, args.task_start, args.task_count)
        if args.mode in {"n1-k0", "fixed_steps", "true_async"}:
            if args.batch50_gate is None or not args.batch50_gate.is_file():
                raise RuntimeError(f"{args.mode} requires --batch50-gate pointing to a complete official K0 result")
            gate_metadata["batch50_k0_gate"] = validate_batch50_gate(args.batch50_gate, manifest)
        if args.mode in {"fixed_steps", "true_async"}:
            gate_metadata["n1_k0_gate"] = validate_n1_k0_gates(args.n1_k0_gate, manifest)
    if args.mode == "batch50-k0":
        result = run_sync(
            lewm_root,
            manifest,
            tasks,
            output_dir=args.output,
            validated_wrapper_repeat=args.validated_wrapper_repeat,
        )
        result_file = args.output / "result.json"
    elif args.mode == "n1-k0":
        result = run_n1_sync_slice(
            lewm_root,
            manifest,
            tasks,
            output_dir=args.output,
            validated_wrapper_repeat=args.validated_wrapper_repeat,
        )
        result_file = args.output / "summary.json"
    else:
        if args.mode == "fixed_steps" and args.delay_steps not in DELAY_STEPS:
            raise ValueError(f"fixed_steps requires --delay-steps from {DELAY_STEPS}")
        if args.mode == "true_async" and args.delay_steps != 0:
            raise ValueError("true_async does not accept a fixed --delay-steps value")
        result = run_n1_scheduler_slice(
            lewm_root,
            manifest,
            tasks,
            output_dir=args.output,
            mode=args.mode,
            delay_steps=args.delay_steps,
            validated_wrapper_repeat=args.validated_wrapper_repeat,
        )
        result_file = args.output / "summary.json"
    if gate_metadata:
        gate_path = args.output / "gate_inputs.json"
        gate_path.write_text(json.dumps(gate_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"allocation": allocation, "assets": assets, "result_file": str(result_file), "task_count": result["task_count"], "gates": gate_metadata}, indent=2))


if __name__ == "__main__":
    main()

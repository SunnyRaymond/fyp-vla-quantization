#!/usr/bin/env python3
"""Run one frozen, upstream-protocol-aligned LeWM PushT teacher baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--staged-home", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> str:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("runner requires PBS_JOBID and a real PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    allocated = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in allocated:
        raise RuntimeError(f"host {host} is not present in PBS_NODEFILE")
    return host


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def to_jsonable(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return False


def compose_pinned_config(lewm_root: Path) -> Any:
    """Compose the pinned upstream PushT eval config shared by both stages."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(lewm_root / "config" / "eval")):
        cfg = compose(config_name="pusht")
    cfg.policy = "pusht/lewm"
    expected = (
        int(cfg.seed) == 42
        and int(cfg.eval.num_eval) == 50
        and str(cfg.eval.dataset_name) == "pusht_expert_train"
        and int(cfg.eval.goal_offset_steps) == 25
        and int(cfg.eval.eval_budget) == 50
        and int(cfg.eval.img_size) == 224
        and list(cfg.dataset.keys_to_cache) == ["action", "proprio", "state"]
        and int(cfg.plan_config.horizon) == 5
        and int(cfg.plan_config.receding_horizon) == 5
        and int(cfg.plan_config.action_block) == 5
        and int(cfg.solver.batch_size) == 1
        and int(cfg.solver.num_samples) == 300
        and int(cfg.solver.n_steps) == 30
        and int(cfg.solver.topk) == 30
        and float(cfg.solver.var_scale) == 1.0
        and str(cfg.solver.device) == "cuda"
        and int(cfg.solver.seed) == 42
        and str(cfg.solver._target_) == "stable_worldmodel.solver.CEMSolver"
        and str(cfg.world.env_name) == "swm/PushT-v1"
        and int(cfg.world.num_envs) == 50
    )
    if not expected:
        raise RuntimeError("pinned eval config differs from frozen protocol")
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    expected_callables = [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
    ]
    if OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) != expected_callables:
        raise RuntimeError("pinned state/goal callables differ from frozen protocol")
    return cfg


def make_image_transform(cfg: Any, spt: Any, transforms: Any, torch: Any) -> Any:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=int(cfg.eval.img_size)),
        ]
    )


def fit_dataset_process(dataset: Any, cfg: Any, preprocessing: Any, np: Any) -> dict[str, Any]:
    process: dict[str, Any] = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[str(col)] = processor
        if col != "action":
            process[f"goal_{col}"] = processor
    return process


def make_pinned_world(swm: Any, cfg: Any) -> Any:
    return swm.World(**cfg.world, image_shape=(224, 224))


def evaluate_selected_tasks(world: Any, dataset: Any, tasks: list[dict[str, Any]], cfg: Any, OmegaConf: Any) -> Any:
    return world.evaluate(
        dataset=dataset,
        start_steps=[int(row["start_step"]) for row in tasks],
        goal_offset=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=[int(row["episode_idx"]) for row in tasks],
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=None,
    )


def main() -> None:
    args = args_parser()
    host = require_compute_node()
    out = args.out.resolve()
    cache_root = args.cache_root.resolve()
    if not out.is_dir() or out == Path("/"):
        raise RuntimeError("PBS wrapper must create the unique output directory first")
    if any((out / name).exists() for name in ("summary.json", "selected_tasks.json")):
        raise FileExistsError("refusing to overwrite baseline result files")

    staged_h5 = args.staged_home / "pusht_expert_train.h5"
    staged_ckpt = args.staged_home / "pusht" / "lewm_object.ckpt"
    cached_h5 = cache_root / "datasets" / "pusht_expert_train.h5"
    cached_ckpt = cache_root / "pusht" / "lewm_object.ckpt"
    for source in (staged_h5, staged_ckpt, cached_h5, cached_ckpt):
        if not source.is_file():
            raise FileNotFoundError(source)
    if cached_h5.resolve(strict=True) != staged_h5.resolve(strict=True):
        raise RuntimeError("dataset cache link does not resolve to the frozen staged HDF5")
    if cached_ckpt.resolve(strict=True) != staged_ckpt.resolve(strict=True):
        raise RuntimeError("checkpoint cache link does not resolve to the frozen official object")
    if staged_h5.stat().st_size != 46300921856 or staged_ckpt.stat().st_size != 72345781:
        raise RuntimeError("staged dataset/checkpoint metadata differs from frozen preflight")

    lewm_root = args.lewm_root.resolve(strict=True)
    stablewm_root = args.stablewm_root.resolve(strict=True)
    control_root = args.control_root.resolve(strict=True)
    sys.path[:0] = [str(lewm_root), str(stablewm_root), str(control_root)]
    os.environ["STABLEWM_HOME"] = str(cache_root)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Heavy imports and any dataset/model reads occur only after the PBS guard.
    import hydra
    import hdf5plugin  # Register filters required by the compressed pinned HDF5 dataset.
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    from run_lewm_recurrent_student import load_official_checkpoint

    def require_under(path: str, root: Path, label: str) -> None:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"{label} imported outside staged root: {resolved}")

    require_under(swm.__file__, stablewm_root, "stable_worldmodel")
    expected_venv_root = (stablewm_root.parent / "venv").resolve(strict=True)
    actual_python_prefix = Path(sys.prefix).resolve(strict=True)
    if actual_python_prefix != expected_venv_root:
        raise RuntimeError(
            f"Python sys.prefix differs from the staged venv: {actual_python_prefix}"
        )
    require_under(spt.__file__, expected_venv_root, "stable_pretraining")
    helper_path = control_root / "run_lewm_recurrent_student.py"
    if not helper_path.is_file():
        raise FileNotFoundError(helper_path)

    cfg = compose_pinned_config(lewm_root)

    def get_episodes_length(dataset: Any, episodes: Any, col_name: str) -> Any:
        episode_ids = dataset.get_col_data(col_name)
        step_idx = dataset.get_col_data("step_idx")
        return np.asarray([np.max(step_idx[episode_ids == ep_id]) + 1 for ep_id in episodes])

    start_total = time.monotonic()
    world = make_pinned_world(swm, cfg)
    transform = {
        "pixels": make_image_transform(cfg, spt, transforms, torch),
        "goal": make_image_transform(cfg, spt, transforms, torch),
    }

    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
        cache_dir=cache_root,
    )
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = fit_dataset_process(stats_dataset, cfg, preprocessing, np)

    model = load_official_checkpoint(cache_root)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    episode_len = get_episodes_length(dataset, ep_indices, col_name)
    max_start_idx = episode_len - int(cfg.eval.goal_offset_steps) - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    row_episode_ids = dataset.get_col_data(col_name)
    max_start_per_row = np.asarray([max_start_idx_dict[ep_id] for ep_id in row_episode_ids])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < int(cfg.eval.num_eval) + 1:
        raise RuntimeError("upstream selection cannot draw the frozen 50 tasks")
    rng = np.random.default_rng(int(cfg.seed))
    selected_rows = np.sort(
        valid_indices[
            rng.choice(len(valid_indices) - 1, size=int(cfg.eval.num_eval), replace=False)
        ]
    )
    selected = dataset.get_row_data(selected_rows)
    task_rows = []
    for i, row_index in enumerate(selected_rows.tolist()):
        task_rows.append(
            {
                "row_index": int(row_index),
                "episode_idx": int(selected[col_name][i]),
                "start_step": int(selected["step_idx"][i]),
            }
        )
    if len(task_rows) != 50 or len({row["row_index"] for row in task_rows}) != 50:
        raise RuntimeError("selected task rows are not exactly 50 unique rows")
    task_episode_ids = [row["episode_idx"] for row in task_rows]
    source_metadata = {
        "dataset_path": str(staged_h5.resolve(strict=True)),
        "dataset_size_bytes": int(staged_h5.stat().st_size),
        "checkpoint_path": str(staged_ckpt.resolve(strict=True)),
        "checkpoint_size_bytes": int(staged_ckpt.stat().st_size),
    }
    atomic_json(
        out / "selected_tasks.json",
        {
            "pbs_job_id": os.environ["PBS_JOBID"],
            "selection_rng_seed": 42,
            "valid_start_rows": int(len(valid_indices)),
            "unique_source_episodes": int(len(set(task_episode_ids))),
            "tasks": task_rows,
            "source_metadata": source_metadata,
        },
    )

    world.set_policy(policy)
    evaluation_start = time.monotonic()
    metrics = evaluate_selected_tasks(world, dataset, task_rows, cfg, OmegaConf)
    evaluation_seconds = time.monotonic() - evaluation_start
    json_metrics = to_jsonable(metrics)
    if not isinstance(json_metrics, dict) or "episode_successes" not in json_metrics:
        raise RuntimeError("World.evaluate omitted per-episode episode_successes")
    successes = json_metrics["episode_successes"]
    if not isinstance(successes, list) or len(successes) != 50:
        raise RuntimeError("World.evaluate did not return exactly 50 episode outcomes")
    successes = [bool(value) for value in successes]
    if not all_finite(json_metrics):
        raise RuntimeError("World.evaluate returned a non-finite numeric metric")
    success_count = sum(successes)
    if "success_rate" in json_metrics:
        reported_success_rate = float(json_metrics["success_rate"])
    else:
        reported_success_rate = 100.0 * success_count / 50.0
    if abs(reported_success_rate - 100.0 * success_count / 50.0) > 1e-9:
        raise RuntimeError("reported success_rate disagrees with episode_successes")

    # Preserve eval.py's small text result artifact at its usual job-private path.
    results_path = Path(swm.data.utils.get_cache_dir(), str(cfg.policy)).parent
    if not results_path.resolve().is_relative_to(cache_root):
        raise RuntimeError(f"upstream result path escaped job-private cache: {results_path}")
    results_path.mkdir(parents=True, exist_ok=True)
    with (results_path / str(cfg.output.filename)).open("a", encoding="utf-8") as handle:
        handle.write("\n==== CONFIG ====\n")
        handle.write(OmegaConf.to_yaml(cfg))
        handle.write("\n==== RESULTS ====\n")
        handle.write(f"metrics: {metrics}\n")
        handle.write(f"evaluation_time: {evaluation_seconds} seconds\n")

    summary = {
        "status": "COMPLETED",
        "experiment_id": "official-lewm-pusht-dataset-teacher-baseline",
        "benchmark_label": "pinned-upstream-protocol-aligned teacher baseline",
        "verbatim_upstream_eval_py": False,
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "validity": "PASS",
        "num_tasks": 50,
        "successes": int(success_count),
        "success_rate_percent": reported_success_rate,
        "engineering_continuation_gate": "ELIGIBLE_FOR_SEPARATE_REVIEW" if success_count >= 5 else "STOP_AND_DIAGNOSE",
        "engineering_gate_threshold": "at least 5/50; engineering floor only, not a statistical test",
        "evaluation_seconds": evaluation_seconds,
        "total_runner_seconds": time.monotonic() - start_total,
        "source_episode_ids_unique": int(len(set(task_episode_ids))),
        "loader_exception": "run_lewm_recurrent_student.load_official_checkpoint loads existing pusht/lewm_object.ckpt",
        "video": None,
        "source_metadata": source_metadata,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": json_metrics,
        "episode_successes": successes,
    }
    atomic_json(out / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "status", "pbs_job_id", "validity", "num_tasks", "successes",
        "success_rate_percent", "engineering_continuation_gate", "evaluation_seconds"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

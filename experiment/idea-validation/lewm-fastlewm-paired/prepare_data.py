#!/usr/bin/env python3
"""Fit the shared official dataset scalers once, inside a verified PBS allocation."""

import argparse
import importlib.metadata
import json
import os
import platform
import pickle
import shutil
import sys
from pathlib import Path


def require_allocation():
    host = platform.node().split(".")[0].lower()
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    if not os.environ.get("PBS_JOBID") or not nodefile.is_file():
        raise RuntimeError("A real PBS compute allocation is required")
    nodes = {line.strip().split(".")[0].lower() for line in nodefile.read_text().splitlines()}
    if host not in nodes or any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-allocation/control host: {host}")
    return host


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    args = parser.parse_args()
    host = require_allocation()
    stage = args.stage.resolve(strict=True)
    backend_root = args.backend.resolve(strict=True)
    out = args.out.resolve(strict=True)
    task_path = args.tasks.resolve(strict=True)
    dataset_path = stage / "stablewm_home" / "pusht_expert_train.h5"
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if any((out / name).exists() for name in ("process.pkl", "prepared.json", "selected_tasks.json")):
        raise FileExistsError("Preparation outputs already exist; refusing to overwrite")

    manifest = json.loads(task_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise RuntimeError("Frozen task manifest must contain exactly 50 tasks")
    if [int(row.get("task_index", -1)) for row in tasks] != list(range(50)):
        raise RuntimeError("Frozen task indices must be exactly 0..49 in manifest order")
    if len({int(row["row_index"]) for row in tasks}) != 50 or len({int(row["episode_idx"]) for row in tasks}) != 50:
        raise RuntimeError("Frozen rows and source episodes must each be unique")

    sys.path.insert(0, str(backend_root))
    os.environ["STABLEWM_HOME"] = str(stage / "stablewm_home")
    import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters
    import numpy as np
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler

    backend = Path(swm.__file__).resolve()
    if not backend.is_relative_to(backend_root):
        raise RuntimeError(f"Unexpected stable_worldmodel import: {backend}")
    if importlib.metadata.version("stable-worldmodel") != "0.0.6":
        raise RuntimeError("The frozen runtime requires stable-worldmodel==0.0.6")
    if Path(sys.prefix).resolve() != (stage / "venv").resolve():
        raise RuntimeError(f"Unexpected Python environment: {sys.prefix}")
    dataset = swm.data.HDF5Dataset(
        "pusht_expert_train", keys_to_cache=[], cache_dir=dataset_path.parent
    )
    episode_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    required = {episode_col, "step_idx", "action", "proprio", "state"}
    if not required.issubset(set(dataset.column_names)):
        raise RuntimeError(f"Dataset lacks frozen columns: {sorted(required - set(dataset.column_names))}")

    row_ids = np.asarray([int(row["row_index"]) for row in tasks], dtype=np.int64)
    selected = dataset.get_row_data(row_ids)
    for i, task in enumerate(tasks):
        actual_episode = int(selected[episode_col][i])
        actual_step = int(selected["step_idx"][i])
        if actual_episode != int(task["episode_idx"]) or actual_step != int(task["start_step"]):
            raise RuntimeError(f"Frozen row identity mismatch at task {i}: {(actual_episode, actual_step)}")

    offset = int(manifest.get("goal_offset_steps", 25))
    episode_values = dataset.get_col_data(episode_col)
    step_values = dataset.get_col_data("step_idx")
    for i, task in enumerate(tasks):
        ep, step = int(task["episode_idx"]), int(task["start_step"])
        ep_steps = step_values[episode_values == ep]
        if ep_steps.size == 0 or not np.any(ep_steps == step) or not np.any(ep_steps == step + offset):
            raise RuntimeError(f"Task {i} lacks its frozen start or goal row at offset {offset}")

    process = {}
    scaler_rows = {}
    for key in ("action", "proprio", "state"):
        values = np.asarray(dataset.get_col_data(key))
        values = values.reshape(len(values), -1)
        not_nan = ~np.isnan(values).any(axis=1)
        if not not_nan.any():
            raise RuntimeError(f"No finite rows available for scaler {key}")
        if np.isinf(values[not_nan]).any():
            raise RuntimeError(f"Infinite values found in non-NaN scaler rows for {key}; official fit would fail")
        scaler = StandardScaler().fit(values[not_nan])
        process[key] = scaler
        if key != "action":
            process[f"goal_{key}"] = scaler
        scaler_rows[key] = int(not_nan.sum())

    with (out / "process.pkl").open("xb") as handle:
        pickle.dump(process, handle, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copyfile(task_path, out / "selected_tasks.json")
    write_json(out / "prepared.json", {
        "status": "PASS", "pbs_job_id": os.environ["PBS_JOBID"], "compute_host": host,
        "dataset_path": str(dataset_path), "stable_worldmodel_path": str(backend),
        "task_count": len(tasks), "goal_offset_steps": offset,
        "scaler_finite_rows": scaler_rows, "scaler_keys": list(process),
        "dataset_cache_keys": [], "python": sys.executable,
        "note": "Dataset-wide action/proprio/state StandardScaler fit once; goal scalers alias proprio/state scalers as in the official evaluator.",
    })
    print(json.dumps({"status": "PASS", "tasks": len(tasks), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Trace reset action-space seed/sample behavior without loading a model."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any


def require_compute_allocation() -> dict[str, Any]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.gethostname().split(".", 1)[0].lower()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_JOBID and a valid PBS_NODEFILE are required")
    if any(marker in host for marker in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    nodes = sorted({line.split(".", 1)[0].lower() for line in Path(nodefile).read_text().splitlines() if line.strip()})
    if host not in nodes:
        raise RuntimeError(f"host {host} is not listed in PBS_NODEFILE: {nodes}")
    return {"pbs_job_id": job_id, "hostname": host, "pbs_nodes": nodes}


def plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def sample_summary(value: Any) -> dict[str, Any]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size <= 32:
        result["values"] = array.tolist()
    elif np.issubdtype(array.dtype, np.number):
        result["min"] = float(np.nanmin(array))
        result["max"] = float(np.nanmax(array))
        result["first_eight"] = array.reshape(-1)[:8].tolist()
    else:
        result["first_eight"] = array.reshape(-1)[:8].tolist()
    return result


def install_space_trace(events: list[dict[str, Any]]):
    spaces = importlib.import_module("gymnasium.spaces")
    base = getattr(spaces, "Space")
    restore: list[tuple[type, str, Any]] = []
    instrumented: list[str] = []
    seen_classes: set[int] = set()

    for class_name, cls in vars(spaces).items():
        if not isinstance(cls, type) or id(cls) in seen_classes:
            continue
        seen_classes.add(id(cls))
        try:
            if not issubclass(cls, base):
                continue
        except TypeError:
            continue
        for method_name in ("seed", "sample"):
            original = cls.__dict__.get(method_name)
            if not callable(original):
                continue

            def traced(self, *args, _original=original, _method=method_name,
                       _class_name=class_name, **kwargs):
                try:
                    result = _original(self, *args, **kwargs)
                except Exception as exc:
                    events.append({
                        "kind": _method,
                        "space_id": id(self),
                        "space_type": f"{type(self).__module__}.{type(self).__qualname__}",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise
                entry: dict[str, Any] = {
                    "kind": _method,
                    "space_id": id(self),
                    "space_type": f"{type(self).__module__}.{type(self).__qualname__}",
                    "instrumented_class": _class_name,
                }
                if _method == "seed":
                    entry["seed_arg"] = plain(args[0] if args else kwargs.get("seed"))
                    entry["return"] = plain(result)
                else:
                    entry["sample"] = sample_summary(result)
                events.append(entry)
                return result

            restore.append((cls, method_name, original))
            setattr(cls, method_name, traced)
            instrumented.append(f"{class_name}.{method_name}")

    if not instrumented:
        raise RuntimeError("no Gymnasium Space.seed/sample methods could be instrumented")
    return restore, instrumented


def restore_space_trace(restore: list[tuple[type, str, Any]]) -> None:
    for cls, method_name, original in reversed(restore):
        setattr(cls, method_name, original)


def wrapper_chain(concrete_env: Any) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current = concrete_env
    depth = 0
    while current is not None and id(current) not in seen and depth < 16:
        seen.add(id(current))
        try:
            space = getattr(current, "action_space")
        except Exception:
            space = None
        entry = {
            "depth": depth,
            "wrapper_id": id(current),
            "wrapper_type": f"{type(current).__module__}.{type(current).__qualname__}",
            "action_space_id": id(space) if space is not None else None,
            "action_space_type": (
                f"{type(space).__module__}.{type(space).__qualname__}" if space is not None else None
            ),
            "action_space_shape": plain(getattr(space, "shape", None)) if space is not None else None,
        }
        chain.append(entry)
        child = getattr(current, "env", None)
        if child is current:
            break
        current = child
        depth += 1
    return chain


def arrays_equal(left: Any, right: Any) -> bool:
    import numpy as np

    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
    except TypeError:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def action_matches_sample(action: Any, sample_event: dict[str, Any]) -> bool:
    import numpy as np

    summary = sample_event.get("sample", {})
    if "values" not in summary:
        return False
    sample = np.asarray(summary["values"])
    action_array = np.asarray(action)
    if arrays_equal(action_array, sample):
        return True
    if sample.ndim == 1 and sample.size and action_array.size % sample.size == 0:
        return any(arrays_equal(sample, row) for row in action_array.reshape(-1, sample.size))
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stablewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    allocation = require_compute_allocation()
    args = parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "diagnostic.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {result_path}")

    os.environ["STABLEWM_HOME"] = str(args.stablewm_home.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.stablewm_root.resolve()))

    events: list[dict[str, Any]] = []
    restore, instrumented = install_space_trace(events)
    rows: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    try:
        import stable_worldmodel as swm

        source_world = getattr(swm, "World", None)
        if source_world is None:
            raise RuntimeError("runtime stable_worldmodel import does not expose World")

        for block_index, seed in enumerate(freeze["scope"]["simulator_reset_seeds"]):
            main_paths = list(freeze["scope"]["main_path_labels"])
            random.Random(int(freeze["scope"]["arm_order_seed_base"]) + block_index).shuffle(main_paths)
            fidelity_paths = list(freeze["scope"]["fidelity_path_labels"])
            random.Random(int(freeze["scope"]["fidelity_order_seed_base"]) + block_index).shuffle(fidelity_paths)
            order = main_paths + fidelity_paths
            block_rows: list[dict[str, Any]] = []

            for path_label in order:
                event_start = len(events)
                world = source_world("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50)
                try:
                    concrete_envs = getattr(getattr(world, "envs", None), "envs", None)
                    if not isinstance(concrete_envs, list) or len(concrete_envs) != 1:
                        raise RuntimeError("expected one concrete EnvPool environment")
                    concrete = concrete_envs[0]
                    chain_before = wrapper_chain(concrete)
                    selected_space = getattr(concrete, "action_space", None)
                    if selected_space is None or not callable(getattr(selected_space, "seed", None)):
                        raise RuntimeError("concrete EnvPool wrapper has no callable action_space.seed")
                    selected_space_id = id(selected_space)
                    seed_return = selected_space.seed(int(seed))
                    world.reset(seed=int(seed))
                    chain_after = wrapper_chain(concrete)
                    infos = getattr(world, "infos", {})
                    action_value = plain(infos.get("action")) if isinstance(infos, dict) else None
                    reset_events = [
                        event for event in events[event_start:]
                        if event.get("space_id") in {entry["action_space_id"] for entry in chain_before if entry["action_space_id"] is not None}
                    ]
                    sample_events = [event for event in reset_events if event.get("kind") == "sample"]
                    seed_events = [event for event in reset_events if event.get("kind") == "seed"]
                    chosen_space_samples = [event for event in sample_events if event.get("space_id") == selected_space_id]
                    matched_sample = any(action_matches_sample(action_value, event) for event in sample_events)
                    row = {
                        "block_index": block_index,
                        "task_seed": int(seed),
                        "path_label": path_label,
                        "execution_order": order,
                        "wrapper_chain_before_reset": chain_before,
                        "wrapper_chain_after_reset": chain_after,
                        "selected_seeded_action_space_id": selected_space_id,
                        "selected_seeded_action_space_type": f"{type(selected_space).__module__}.{type(selected_space).__qualname__}",
                        "explicit_seed_return": plain(seed_return),
                        "space_seed_events": seed_events,
                        "space_sample_events": sample_events,
                        "selected_space_sample_call_count": len(chosen_space_samples),
                        "action_matches_a_recorded_reset_sample": bool(matched_sample),
                        "world_infos_action": sample_summary(action_value),
                    }
                    rows.append(row)
                    block_rows.append(row)
                finally:
                    world.close()

            baseline = block_rows[0]
            for row in block_rows:
                pair_results.append({
                    "task_seed": int(seed),
                    "path_label": row["path_label"],
                    "baseline_path_label": baseline["path_label"],
                    "same_reset_action_as_baseline": arrays_equal(
                        baseline["world_infos_action"].get("values"),
                        row["world_infos_action"].get("values"),
                    ),
                })

        report = {
            "schema": freeze["schema"],
            "status": "COMPLETED",
            "parent_stage1_job": freeze["parent_stage1_job"],
            "parent_stage1_status": freeze["parent_stage1_status"],
            "freeze_name": freeze["name"],
            "allocation": allocation,
            "runtime_import_root": str(Path(swm.__file__).resolve()),
            "runtime_world_api": f"{source_world.__module__}.{source_world.__qualname__}",
            "instrumented_space_methods": instrumented,
            "total_space_events": len(events),
            "reset_rows": rows,
            "same_seed_action_pairing": pair_results,
            "all_same_seed_reset_actions_equal": all(item["same_reset_action_as_baseline"] for item in pair_results),
        }
        temporary = result_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, result_path)
        print(json.dumps({
            "status": report["status"],
            "parent_stage1_job": report["parent_stage1_job"],
            "runtime_import_root": report["runtime_import_root"],
            "reset_rows": len(rows),
            "space_events": len(events),
            "all_same_seed_reset_actions_equal": report["all_same_seed_reset_actions_equal"],
            "diagnostic_path": str(result_path),
        }, ensure_ascii=False))
        return 0
    finally:
        restore_space_trace(restore)


if __name__ == "__main__":
    raise SystemExit(main())

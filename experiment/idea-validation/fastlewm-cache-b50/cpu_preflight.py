#!/usr/bin/env python3
"""Allocation-only source/runtime preflight; never loads a checkpoint or dataset."""

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import sys
from pathlib import Path

from cache_core import identity_self_check


def require_cpu_allocation():
    host = platform.node().split(".")[0].lower()
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    if not os.environ.get("PBS_JOBID") or not nodefile.is_file():
        raise RuntimeError("A real PBS allocation is required")
    nodes = {line.strip().split(".")[0].lower() for line in nodefile.read_text().splitlines()}
    if host not in nodes or any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-allocation/control host: {host}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise RuntimeError("CPU preflight must not receive a GPU allocation")
    return host


def source_text(obj):
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return "<source unavailable>\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source-reference", type=Path, required=True)
    args = parser.parse_args()
    host = require_cpu_allocation()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    settings = freeze["settings"]
    if settings["num_eval"] != 50 or settings["solver_seeds"] != [42, 43, 44]:
        raise RuntimeError("Frozen evaluation/task scope changed")
    if (settings["num_samples"], settings["topk"], settings["iterations"], settings["batch_size"]) != (300, 30, 30, 1):
        raise RuntimeError("Frozen CEM configuration changed")
    main_root = Path(freeze["source_roots"]["paired"])
    selected = json.loads((main_root / "source" / "selected_tasks.json").read_text(encoding="utf-8"))
    if selected.get("seed") != 42 or len(selected.get("tasks", [])) != 50:
        raise RuntimeError("Selected task manifest is not the frozen ordered 50-row manifest")
    prepared_path = Path(freeze["main_preparation"])
    prepared = json.loads((prepared_path / "prepared.json").read_text(encoding="utf-8"))
    if prepared.get("status") != "PASS" or not (prepared_path / "process.pkl").is_file():
        raise RuntimeError("Reusable preparation is incomplete")

    fast_root = Path(freeze["source_roots"]["fast"])
    source = fast_root / "upstream"
    backend = fast_root / "runtime_overlay"
    identity = json.loads((fast_root / "source_identity.json").read_text(encoding="utf-8"))
    if identity.get("source_sha") != freeze["source_commit"]:
        raise RuntimeError("Fast source identity does not match the frozen commit")
    revision_file = source / ".fastlewm-revision"
    if revision_file.is_file() and revision_file.read_text(encoding="utf-8").strip() != freeze["source_commit"]:
        raise RuntimeError("Fast source revision marker changed")
    if not (fast_root / "checkpoints" / "Fast-lewm_pusht_object.ckpt").is_file():
        raise RuntimeError("Pinned checkpoint is missing")

    sys.path[:0] = [str(source), str(backend)]
    import stable_worldmodel as swm
    import torch
    from stable_worldmodel.solver import CEMSolver
    import jepa
    if not Path(swm.__file__).resolve().is_relative_to(backend.resolve()):
        raise RuntimeError(f"Wrong stable-worldmodel import: {swm.__file__}")
    if importlib.metadata.version("stable-worldmodel") != "0.0.6":
        raise RuntimeError("Wrong stable-worldmodel version")

    candidate_classes = []
    for value in vars(jepa).values():
        if (inspect.isclass(value) and value.__module__ == jepa.__name__
                and callable(getattr(value, "encode", None))
                and callable(getattr(value, "get_cost", None))):
            candidate_classes.append(value)
    if len(candidate_classes) != 1 or candidate_classes[0].__name__ != "JEPA":
        raise RuntimeError("Could not identify the native Fast-LeWM encode/get_cost class")
    args.source_reference.mkdir(parents=True, exist_ok=True)
    references = {
        "cem_solve.py.txt": CEMSolver.solve,
        "model_encode.py.txt": candidate_classes[0].encode,
        "model_get_cost.py.txt": candidate_classes[0].get_cost,
        "model_rollout.py.txt": getattr(candidate_classes[0], "rollout", None),
    }
    for name, obj in references.items():
        if obj is not None:
            (args.source_reference / name).write_text(source_text(obj), encoding="utf-8")

    solve_src = source_text(CEMSolver.solve)
    cost_src = source_text(candidate_classes[0].get_cost)
    rollout_obj = getattr(candidate_classes[0], "rollout", None)
    rollout_src = source_text(rollout_obj) if rollout_obj is not None else ""
    if "get_cost" not in solve_src or "encode" not in cost_src or "rollout" not in cost_src or "encode" not in rollout_src:
        raise RuntimeError("Native CEM/get_cost/current-versus-goal source path could not be confirmed")

    import numpy as np
    report = {
        "status": "PASS",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "stable_worldmodel_version": importlib.metadata.version("stable-worldmodel"),
        "stable_worldmodel_file": str(Path(swm.__file__).resolve()),
        "fast_source": str(source.resolve()),
        "source_identity_commit": identity["source_sha"],
        "candidate_jepa_classes": [f"{cls.__module__}.{cls.__name__}" for cls in candidate_classes],
        "cem_solve_signature": str(inspect.signature(CEMSolver.solve)),
        "model_encode_signature": str(inspect.signature(candidate_classes[0].encode)),
        "model_get_cost_signature": str(inspect.signature(candidate_classes[0].get_cost)),
        "selected_task_count": len(selected["tasks"]),
        "prepared_status": prepared["status"],
        "dummy_cache_identity": identity_self_check(torch),
        "source_reference_files": sorted(references),
        "interface_note": "Source references are for root review. This preflight did not instantiate a model, open a dataset, load a checkpoint, or run a benchmark.",
        "numpy": np.__version__,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (args.source_reference.parent / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Small CPU-only LeWM interface and PushT dataset preflight.

This script intentionally does not train, run a model forward pass, or alter
the staged checkpoint.  It records only the object/module schema needed to
adapt the recurrent student to LeWM.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lewm-root", type=Path, required=True)
    p.add_argument("--stablewm-home", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def require_cpu_compute() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only preflight requires CUDA to be unavailable")


def type_name(value: Any) -> str | None:
    return None if value is None else f"{type(value).__module__}.{type(value).__name__}"


def shape(value: Any) -> list[int] | None:
    result = getattr(value, "shape", None)
    if result is None:
        return None
    try:
        return [int(item) for item in result]
    except (TypeError, ValueError):
        return None


def module_info(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    result: dict[str, Any] = {"type": type_name(value)}
    params = getattr(value, "parameters", None)
    if callable(params):
        entries = []
        total = 0
        for name, parameter in value.named_parameters():
            count = int(parameter.numel())
            total += count
            entries.append({"name": name, "shape": shape(parameter), "dtype": str(parameter.dtype)})
        result["parameter_count"] = total
        result["parameters"] = entries[:256]
    return result


def object_attr(model: Any, names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        try:
            value = getattr(model, name)
        except Exception as exc:  # pragma: no cover - defensive for custom objects
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        info = module_info(value)
        if info is not None:
            result[name] = info
        else:
            result[name] = {"type": type_name(value), "shape": shape(value), "repr": repr(value)[:300]}
    return result


def collect_hdf5(path: Path) -> dict[str, Any]:
    import h5py

    result: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size, "groups": [], "datasets": []}
    with h5py.File(path, "r") as handle:
        def visit(name: str, item: Any) -> None:
            if isinstance(item, h5py.Group):
                result["groups"].append(name)
            elif isinstance(item, h5py.Dataset):
                entry: dict[str, Any] = {
                    "name": name,
                    "shape": [int(v) for v in item.shape],
                    "dtype": str(item.dtype),
                    "chunks": None if item.chunks is None else [int(v) for v in item.chunks],
                }
                if item.shape:
                    try:
                        first = item[0]
                        entry["first_sample_shape"] = shape(first)
                        if len(item.shape) == 1:
                            preview = item[: min(8, int(item.shape[0]))].tolist()
                            entry["first_values"] = preview
                    except Exception as exc:
                        entry["first_sample_error"] = f"{type(exc).__name__}: {exc}"
                result["datasets"].append(entry)
        handle.visititems(visit)
        result["root_attrs"] = {str(k): repr(v)[:300] for k, v in handle.attrs.items()}
        for entry in result["datasets"]:
            if entry["name"] in {"ep_len", "ep_offset"}:
                dataset = handle[entry["name"]]
                if len(dataset.shape) == 1 and dataset.shape[0]:
                    values = dataset[:]
                    entry["min"] = int(values.min())
                    entry["max"] = int(values.max())
                    entry["count"] = int(values.shape[0])
    return result


def try_dataset_loader(repo_root: Path, data_path: Path) -> dict[str, Any]:
    """Best-effort loader probe, without requiring the loader for the preflight."""
    candidates = [
        ("lewm.data", "PushTDataset"),
        ("lewm.dataset", "PushTDataset"),
        ("dataset", "PushTDataset"),
        ("jepa", "PushTDataset"),
    ]
    result: dict[str, Any] = {"attempted": candidates, "found": None}
    old = os.getcwd()
    try:
        os.chdir(repo_root)
        for module_name, class_name in candidates:
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except Exception:
                continue
            result["found"] = {"module": module_name, "class": class_name, "type": type_name(cls)}
            try:
                signature = str(inspect.signature(cls))
            except (TypeError, ValueError):
                signature = None
            result["signature"] = signature
            for kwargs in ({"path": str(data_path)}, {"data_path": str(data_path)}, {"filename": str(data_path)}):
                try:
                    dataset = cls(**kwargs)
                    result["dataset_type"] = type_name(dataset)
                    result["length"] = int(len(dataset)) if hasattr(dataset, "__len__") else None
                    sample = dataset[0]
                    result["sample"] = summarize_value(sample)
                    return result
                except Exception as exc:
                    result.setdefault("constructor_errors", []).append(f"{kwargs}: {type(exc).__name__}: {exc}")
            return result
    finally:
        os.chdir(old)
    result["note"] = "No known official PushT dataset loader was importable; HDF5 metadata remains authoritative."
    return result


def summarize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): summarize_value(v) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [summarize_value(v) for v in list(value)[:32]]
    result = {"type": type_name(value), "shape": shape(value)}
    if result["shape"] is None:
        result["repr"] = repr(value)[:300]
    return result


def main() -> int:
    args = parse_args()
    require_cpu_compute()
    import torch

    checkpoint = args.stablewm_home / "pusht" / "lewm_object.ckpt"
    dataset_path = args.stablewm_home / "pusht_expert_train.h5"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    os.environ["STABLEWM_HOME"] = str(args.stablewm_home.resolve())
    os.environ.setdefault("WANDB_MODE", "disabled")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = model.state_dict() if hasattr(model, "state_dict") else {}
    state_shapes = {str(k): shape(v) for k, v in list(state_dict.items())[:512]}
    model_info = {
        "type": type_name(model),
        "parameter_count": sum(int(p.numel()) for p in model.parameters()) if hasattr(model, "parameters") else None,
        "attributes": object_attr(model, ["predictor", "action_encoder", "projector", "pred_proj", "encoder", "criterion", "latent_dim", "action_dim"]),
        "state_dict_shapes": state_shapes,
    }
    result = {
        "schema": "lewm-pusht-student-preflight",
        "schema_version": 1,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "model": model_info,
        "hdf5": collect_hdf5(dataset_path),
        "dataset_loader": try_dataset_loader(args.lewm_root, dataset_path),
        "torch": torch.__version__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight_summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "datasets": len(result["hdf5"]["datasets"]), "summary": str(args.output / "preflight_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

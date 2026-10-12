"""Collect signed Fast-WAM activations and weights for the frozen Prism view."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from typing import Any


PROTOCOL_NAME = "fastwam-prism-visualization-v1"
ARMS = ("bf16", "spinquant_native_w4a4", "smooth_scalar_native_w4a4",
        "row_scale_vq_a4", "micro_scale_vq_a4")
PAIRED_TRANSFORMS = ("raw", "learned_spinquant", "smooth05_hadamard")
EXPECTED_SEEDS = {1004: 911004000, 1014: 911014000}
SELECTED_SUFFIXES = ("self_attn.q", "ffn.2")


class CollectorStageTagger:
    """Stage interface used by the frozen scheduler wrappers."""

    def __init__(self) -> None:
        self.stage, self.step = "conditioning", -1

    def set_stage(self, stage: str, step: int) -> None:
        if stage in ("conditioning", "video_conditioning_prefill"):
            if step != -1:
                raise ValueError(f"{stage} must use step -1, got {step}")
        elif stage in ("video", "action"):
            if type(step) is not int or not 0 <= step < 10:
                raise ValueError(f"{stage} step must be in 0..9, got {step}")
        else:
            raise ValueError(f"unknown scheduler stage: {stage!r}")
        self.stage, self.step = stage, step


def guard() -> None:
    """Refuse all collector entry points outside their PBS compute allocation."""
    host = socket.gethostname().split(".", 1)[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("Approved PBS compute allocation required")
    nodes = {line.split(".", 1)[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")


def window_bounds(length: int, start: int, count: int) -> tuple[int, int]:
    if length < 0 or start < 0 or count < 0:
        raise ValueError("window length, start, and count must be nonnegative")
    begin = min(start, length)
    return begin, min(begin + count, length)


def selected_modules(module_names: list[str], blocks: list[int], streams: list[str],
                     suffixes: list[str]) -> list[tuple[int, str, str, int]]:
    """Return exact stream/block/suffix matches with their global Linear indices."""
    wanted = {(stream, block, suffix) for stream in streams for block in blocks for suffix in suffixes}
    found: list[tuple[int, str, str, int]] = []
    pattern = re.compile(r"^(video_expert|action_expert)\.blocks\.(\d+)\.(.+)$")
    for index, name in enumerate(module_names):
        match = pattern.fullmatch(name)
        if not match:
            continue
        stream, block, suffix = match.group(1), int(match.group(2)), match.group(3)
        if (stream, block, suffix) in wanted:
            found.append((index, name, stream, block))
    got = {(stream, block, match.group(3)) for _index, name, stream, block in found
           if (match := pattern.fullmatch(name))}
    if len(found) != len(wanted) or got != wanted:
        raise ValueError(f"selected module coverage differs: expected {len(wanted)}, found {len(found)}")
    return found


def aligned_window(input_shape: tuple[int, ...] | list[int], output_shape: tuple[int, ...] | list[int],
                   start: int, count: int, in_features: int | None = None,
                   out_features: int | None = None, batch_index: int = 0) -> tuple[int, int]:
    """Validate Linear alignment and clamp a token window within one leading branch."""
    left, right = tuple(input_shape), tuple(output_shape)
    if not left or not right or left[:-1] != right[:-1]:
        raise ValueError(f"Linear input/output leading dimensions differ: {left} vs {right}")
    if in_features is not None and left[-1] != in_features:
        raise ValueError(f"Linear input width {left[-1]} != {in_features}")
    if out_features is not None and right[-1] != out_features:
        raise ValueError(f"Linear output width {right[-1]} != {out_features}")
    if len(left) >= 3:
        if not 0 <= batch_index < left[0]:
            raise ValueError(f"batch index {batch_index} outside leading dimension {left[0]}")
        rows = math.prod(left[1:-1])
    else:
        rows = math.prod(left[:-1]) if len(left) > 1 else 1
    return window_bounds(rows, start, count)


def should_capture(stream: str, stage: str, step: int, snapshot_steps: list[int]) -> bool:
    if stage in ("conditioning", "video_conditioning_prefill"):
        return step == -1
    expected_stage = "video" if stream == "video_expert" else "action"
    return stage == expected_stage and step in snapshot_steps


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_npz(np: Any, path: Path, arrays: dict[str, Any], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_protocol(p: dict[str, Any]) -> None:
    if p.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"unexpected protocol {p.get('protocol')!r}")
    fixed = {
        "cases": [1004, 1014], "seed_index": 0, "arms": list(ARMS),
        "paired_transforms": list(PAIRED_TRANSFORMS), "blocks": [0, 10, 20, 29],
        "streams": ["video_expert", "action_expert"], "module_suffixes": list(SELECTED_SUFFIXES),
        "snapshot_steps": [0, 4, 9], "token_start": 0, "capture_tokens": 128,
        "plot_tokens": 64, "channel_start": 0, "plot_channels": 512,
        "weight_output_start": 0, "weight_output_count": 128, "walltime": "02:00:00",
        "soft_deadline_seconds": 6600, "max_queries": 10, "evaluation_episodes": 0,
        "refit": False, "training": False,
    }
    for key, expected in fixed.items():
        if p.get(key) != expected:
            raise ValueError(f"protocol {key} must be {expected!r}, got {p.get(key)!r}")
    for key in ("source_root", "micro_root", "rotation_root", "scalar_root", "spinquant_bank", "scalar_bank"):
        if not isinstance(p.get(key), str) or not p[key]:
            raise ValueError(f"protocol must name {key}")
    for key in ("base", "pilot", "old_inputs", "old_banks"):
        if not isinstance(p.get("roots", {}).get(key), str) or not p["roots"][key]:
            raise ValueError(f"protocol roots must name {key}")
    if p.get("batch_index", 0) != 0:
        raise ValueError("collector captures only batch_index 0 / the first CFG branch")
    if not isinstance(p.get("source_full_job"), str) or not p["source_full_job"]:
        raise ValueError("protocol must pin source_full_job")


def _atomic_npz(torch: Any, np: Any, path: Path, arrays: dict[str, Any], metadata: dict[str, Any]) -> None:
    converted = {}
    for key, value in arrays.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()
        converted[key] = value
    _save_npz(np, path, converted, metadata)


def _cpu_transform(torch: Any, transform: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in transform.items()}


def _device_transform(torch: Any, transform: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device=device, dtype=torch.float32) if torch.is_tensor(value) else value
            for key, value in transform.items()}


def _load_source_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _verify_sources(p: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    source_root = Path(p["source_root"])
    micro_root = Path(p["micro_root"])
    source_summary_path = source_root / "results/full/summary.json"
    micro_summary_path = micro_root / "results/full/summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    micro_summary = json.loads(micro_summary_path.read_text(encoding="utf-8"))
    source_protocol = json.loads((source_root / "protocol.json").read_text(encoding="utf-8"))
    micro_protocol = json.loads((micro_root / "protocol.json").read_text(encoding="utf-8"))
    if (source_summary.get("status") != "complete" or source_summary.get("pbs_jobid") != p["source_full_job"]
            or source_protocol.get("protocol") != "fastwam-vq-action-aware-v1"):
        raise RuntimeError("declared action-aware source run is not complete or does not match the frozen job")
    if (micro_summary.get("status") != "complete" or micro_summary.get("source_full_job") != p["source_full_job"]
            or micro_protocol.get("protocol") != "fastwam-vq-microscale-v1"):
        raise RuntimeError("declared micro-VQ source run is not complete or does not match the frozen action-aware job")
    spin_root = Path(p["spinquant_bank"])
    spin_manifest = json.loads((spin_root / "manifest.json").read_text(encoding="utf-8"))
    training = spin_manifest.get("experiment_metadata", {}).get("training", {})
    if (spin_manifest.get("arm") != "spinquant_adapted_w4a4" or spin_manifest.get("rotation_mode") != "learned"
            or spin_manifest.get("module_count") != 614 or training.get("selected_step") != 200):
        raise RuntimeError("learned SpinQuant bank identity differs from the frozen R128 bank")
    scalar_root = Path(p["scalar_bank"])
    scalar_receipt = json.loads((scalar_root / "receipt.json").read_text(encoding="utf-8"))
    scalar_run_root = scalar_root.parent.parent
    scalar_protocol = json.loads((scalar_run_root / "protocol.json").read_text(encoding="utf-8"))
    scalar_winners = json.loads((scalar_run_root / "frozen_winners.json").read_text(encoding="utf-8"))
    if (scalar_receipt.get("config") != "smooth05_hadamard" or scalar_receipt.get("kind") != "scalar"
            or scalar_winners.get("winners", {}).get("scalar") != "smooth05_hadamard"
            or scalar_protocol.get("protocol") != "fastwam-smooth-vq-phases12-v1"):
        raise RuntimeError("Smooth0.5/H128 scalar bank identity differs from the frozen source")
    return {
        "action_aware": {"root": str(source_root), "job_id": p["source_full_job"], "protocol": source_protocol["protocol"]},
        "micro_vq": {"root": str(micro_root), "source_job_id": micro_summary.get("job_id"), "protocol": micro_protocol["protocol"]},
        "spinquant": {"root": str(spin_root), "manifest_arm": spin_manifest["arm"],
                      "rotation_mode": spin_manifest["rotation_mode"], "module_count": spin_manifest["module_count"],
                      "selected_step": training["selected_step"]},
        "smooth_scalar": {"root": str(scalar_root), "source_job_id": scalar_protocol.get("source_full_job"),
                          "config": scalar_receipt["config"], "kind": scalar_receipt["kind"]},
    }


def _source_paths(p: dict[str, Any]) -> list[Path]:
    return [Path(p["roots"]["old_inputs"]), Path(p["source_root"]), Path(p["roots"]["old_banks"]),
            Path(p["roots"]["pilot"]), Path(p["rotation_root"]), Path(p["scalar_root"]), Path(p["micro_root"])]


def _prepare_imports(p: dict[str, Any], artifact: Path) -> tuple[Any, Any, Any]:
    old_inputs, source_root, old_banks, pilot_root, rotation_root, scalar_root, micro_root = _source_paths(p)
    paths = [old_inputs, source_root, old_banks, pilot_root, rotation_root, scalar_root, micro_root, artifact]
    sys.path[:] = [str(path) for path in paths if path.is_dir()] + [value for value in sys.path if value not in {str(path) for path in paths}]
    source = _load_source_module(old_inputs / "run_diagnostic.py", "_fastwam_prism_run_diagnostic")
    pilot = source.load_pilot_runner()
    import smooth_vq as core
    return source, pilot, core


def _load_transform(torch: Any, bank_path: Path, name: str, index: int, device: Any) -> dict[str, Any]:
    bank = torch.load(bank_path / f"{index:04d}.pt", map_location="cpu", weights_only=True)
    if bank.get("module") != name or not isinstance(bank.get("transform"), dict):
        raise RuntimeError(f"saved transform bank identity differs for {name}: {bank_path}")
    transform = _device_transform(torch, bank["transform"], device)
    del bank
    return transform


def _transform_artifact(np: Any, torch: Any, out: Path, transform_id: str, source_bank: str,
                        transform: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    relative = Path("raw/transforms") / f"{transform_id}.npz"
    path = out / relative
    if not path.is_file():
        arrays = {key: value for key, value in transform.items() if torch.is_tensor(value)}
        _atomic_npz(torch, np, path, arrays, {"transform_id": transform_id, "source_bank": source_bank, **metadata})
    return {"transform_id": transform_id, "source_bank": source_bank,
            "file": relative.as_posix(), "metadata": metadata}


def _flat(torch: Any, value: Any, width: int, batch_index: int = 0) -> Any:
    if not torch.is_tensor(value) or value.shape[-1] != width:
        raise ValueError(f"expected tensor with final dimension {width}")
    branch = value[batch_index] if value.ndim >= 3 else value
    return branch.detach().reshape(-1, width)


def _native_qdq(torch: Any, native: Any, value: Any) -> Any:
    codes, scales = native._quantize_activation(value, 4)
    flat = value.detach().reshape(-1, value.shape[-1])
    return (codes.float() * scales[:, None]).reshape_as(flat).reshape(value.shape)


def _smooth_qdq(torch: Any, value: Any) -> Any:
    rows = value.float().reshape(-1, value.shape[-1])
    maxima = rows.abs().amax(dim=1, keepdim=True)
    scales = maxima / 7
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    return (torch.round(rows / safe).clamp(-7, 7) * scales).to(torch.bfloat16).reshape(value.shape)


def _array_window(value: Any, start: int, count: int, batch_index: int = 0) -> tuple[Any, Any]:
    branch = value[batch_index] if value.ndim >= 3 else value
    flat = branch.detach().reshape(-1, value.shape[-1])
    begin, end = window_bounds(int(flat.shape[0]), start, count)
    return flat[begin:end], (begin, end)


def _original_row_ids(torch: Any, shape: tuple[int, ...], begin: int, end: int,
                      batch_index: int = 0, device: Any = "cpu") -> Any:
    branch_rows = math.prod(shape[1:-1]) if len(shape) >= 3 else (math.prod(shape[:-1]) if len(shape) > 1 else 1)
    base = batch_index * branch_rows if len(shape) >= 3 else 0
    return torch.arange(base + begin, base + end, dtype=torch.int32, device=device)


def _tensor_metadata(value: Any) -> dict[str, Any]:
    return {"shape": [int(dim) for dim in value.shape], "dtype": str(value.dtype)}


def _native_weight(torch: Any, native: Any, module: Any) -> Any:
    if module.weight is not None:
        return module.weight.detach().float()
    packed = module._quant_packed_weight
    padded_k = int(module._quant_padded_k)
    codes = native._unpack_nibbles(packed, padded_k).float()
    scales = module._quant_weight_scales.float().repeat_interleave(128, dim=1)
    return (codes * scales)[:, :module.in_features]


def _decode_bank_weights(torch: Any, modules: dict[str, Any], p: dict[str, Any], arm: str,
                          device: Any, core: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Install one frozen decoded-BF16 bank without retaining a CPU copy of all model weights."""
    import codebook

    bank_root = (Path(p["source_root"]) / "results/full/banks/row_scale_vq_a4" if arm == "row_scale_vq_a4"
                 else Path(p["micro_root"]) / "results/full/banks/micro_scale_vq_a4")
    decoder = codebook.decode_vq if arm == "row_scale_vq_a4" else None
    micro = None
    if decoder is None:
        import micro_codebook as micro
        decoder = micro.decode_vq
    transforms: dict[str, dict[str, Any]] = {}
    for index, (name, module) in enumerate(modules.items()):
        bank = torch.load(bank_root / f"{index:04d}.pt", map_location="cpu", weights_only=True)
        encoded = bank.get("encoded")
        if bank.get("module") != name or not isinstance(encoded, dict) or tuple(encoded.get("shape", ())) != tuple(module.weight.shape):
            raise RuntimeError(f"decoded-VQ saved bank identity/shape differs at module {index}: {name}")
        transform = bank.get("transform")
        if not isinstance(transform, dict):
            raise RuntimeError(f"decoded-VQ bank is missing its frozen transform: {name}")
        device_transform = _device_transform(torch, transform, device)
        transforms[name] = device_transform
        if arm == "row_scale_vq_a4":
            encoded_device = {key: value.to(device=device) if torch.is_tensor(value) else value
                             for key, value in encoded.items()}
            dense = decoder(encoded_device)
        else:
            encoded_device = {key: value.to(device=device) if torch.is_tensor(value) else value
                             for key, value in encoded.items()}
            dense = decoder(encoded_device)
        module.weight.data = dense
        del bank, encoded, encoded_device, dense

        def forward(value: Any, _name: str = name, _module: Any = module) -> Any:
            z = core.transform_input(value, transforms[_name]).float()
            return torch.nn.functional.linear(_smooth_qdq(torch, z), _module.weight, _module.bias)

        module.forward = forward
        if index % 100 == 99 or index == len(modules) - 1:
            print(f"DECODED_VQ_BANK_LOADED {arm} {index + 1}/{len(modules)}", flush=True)
    return transforms, {"root": str(bank_root), "format": "decoded BF16 additive VQ", "native_kernel": False}


def _cached_action(p: dict[str, Any], arm: str, case: int, seed_index: int, sampler_seed: int) -> tuple[Any, str] | None:
    candidates = [Path(p["source_root"]) / "results/full/actions.jsonl",
                  Path(p["micro_root"]) / "results/full/actions.jsonl"]
    for path in candidates:
        if not path.is_file():
            continue
        found = None
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (row.get("arm"), row.get("case_id"), row.get("seed_index")) == (arm, case, seed_index):
                    if found is not None:
                        raise RuntimeError(f"duplicate cached same-arm query for {arm}/{case}/{seed_index}")
                    found = row
        if found is not None:
            if found.get("sampler_seed") != sampler_seed:
                raise RuntimeError(f"cached sampler seed differs for {arm}/{case}/{seed_index}")
            action = found.get("action")
            if not isinstance(action, list) or len(action) != 32 or any(len(row) != 7 for row in action):
                raise RuntimeError(f"cached action shape differs for {arm}/{case}/{seed_index}")
            import numpy as np
            value = np.asarray(action, dtype=np.float32)
            if not np.isfinite(value).all():
                raise RuntimeError(f"cached action is nonfinite for {arm}/{case}/{seed_index}")
            return value, str(path)
    return None


def _run_arm(args: argparse.Namespace, p: dict[str, Any], arm: str) -> None:
    guard()
    import numpy as np
    import torch

    out = args.out.resolve()
    artifact = (args.artifact or Path(os.environ.get("ARTIFACTS", out / "artifacts"))).resolve()
    source, pilot, core = _prepare_imports(p, artifact)
    import quantization as native
    import rotation
    import scalar_native

    source_root = Path(p["source_root"])
    old_inputs = Path(p["roots"]["old_inputs"])
    plan = json.loads((source_root / "plan.json").read_text(encoding="utf-8"))
    if not isinstance(plan.get("inputs"), list):
        raise RuntimeError("frozen action-aware plan is missing its inputs")
    plan_rows = {row["case_id"]: row for row in plan["inputs"]}
    for case in p["cases"]:
        row = plan_rows.get(case)
        if not isinstance(row, dict) or row.get("case_id") != case:
            raise RuntimeError(f"frozen plan does not contain case {case}")
        seed_index = int(p["seed_index"])
        if row.get("sampler_seeds", [None])[seed_index] != EXPECTED_SEEDS[case]:
            raise RuntimeError(f"case {case} sampler seed differs from the frozen protocol")

    completed = []
    for case in p["cases"]:
        checkpoint = out / "checkpoints" / f"{arm}_{case}.json"
        if checkpoint.is_file():
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            if value.get("protocol") != p["protocol"] or value.get("arm") != arm or value.get("case_id") != case:
                raise RuntimeError(f"existing query checkpoint identity differs: {checkpoint}")
            completed.append(case)
    pending_cases = [case for case in p["cases"] if case not in completed]
    if not pending_cases:
        print(f"ARM_ALREADY_COMPLETE {arm}", flush=True)
        return
    case_rows = {}
    for case in pending_cases:
        row = plan_rows[case]
        observation, metadata = source.load_case(source_root, row)
        case_rows[case] = (row, observation, metadata, int(row["sampler_seeds"][int(p["seed_index"])]))
    deadline = float(os.environ.get("FW_JOB_START_EPOCH", time.time())) + int(p["soft_deadline_seconds"])
    if deadline - time.time() < 600:
        _save_json(out / "checkpoints" / f"{arm}_progress.json", {"protocol": p["protocol"], "arm": arm,
                    "status": "checkpointed", "completed_cases": completed, "pending_cases": pending_cases})
        return

    runtime = scheduler_handles = conditioning_handles = None
    collector_hooks: list[Any] = []
    original_forwards: dict[str, Any] = {}
    stage = "model_loading"
    try:
        runtime_dir = out / "runtime" / arm
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime = pilot.Runtime(runtime_dir)
        runtime.arm("bf16")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one PBS-allocated CUDA GPU is required")
        cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        props = torch.cuda.get_device_properties(0)
        if not cuda_mask or str(props.uuid).removeprefix("GPU-").lower() != cuda_mask.removeprefix("GPU-").lower():
            raise RuntimeError("CUDA GPU UUID does not match the visible PBS allocation")
        modules = runtime.q.modules
        module_items = list(modules.items())
        if len(module_items) != 614:
            raise RuntimeError(f"expected 614 target Linears, found {len(module_items)}")
        original_forwards = {name: module.forward for name, module in module_items}
        selected = selected_modules([name for name, _module in module_items], p["blocks"], p["streams"], p["module_suffixes"])
        selected_by_name = {name: (index, stream, block) for index, name, stream, block in selected}
        selected_indices = {index for index, _name, _stream, _block in selected}
        selected_original_weights = {name: modules[name].weight.detach().cpu().clone() for _index, name, _stream, _block in selected}
        device = next(iter(modules.values())).weight.device
        smooth_bank = Path(p["scalar_bank"])
        smooth_transforms: dict[str, dict[str, Any]] = {}
        smooth_transform_records = []
        raw_transform_records = []
        for index, name, stream, block in selected:
            transform = _load_transform(torch, smooth_bank, name, index, device)
            smooth_transforms[name] = transform
            smooth_transform_records.append(_transform_artifact(np, torch, out, f"smooth05_hadamard_m{index:04d}",
                str(smooth_bank), transform, {"basis": "smooth05_hadamard", "block": 128,
                "hadamard": True, "input_dim": int(modules[name].in_features), "stream": stream,
                "block_index": block, "module": name}))

        spin_root = Path(p["spinquant_bank"])
        spin_manifest = json.loads((spin_root / "manifest.json").read_text(encoding="utf-8"))
        rotation_payload = torch.load(spin_root / spin_manifest["rotation_file"], map_location="cpu", weights_only=True)
        spin_transforms = {stream: {key: value.to(device=device, dtype=torch.float32) for key, value in rotation_payload["streams"][rotation._stream_for_name(stream)].items()}
                           for stream in p["streams"]}
        spin_transform_records = []
        for stream in p["streams"]:
            spin_transform_records.append(_transform_artifact(np, torch, out, f"learned_spinquant_{stream}",
                str(spin_root), spin_transforms[stream], {"basis": "learned_spinquant", "rotation_mode": "learned",
                "block": 128, "stream": stream, "rotation_file": spin_manifest["rotation_file"]}))

        transforms_all: dict[str, dict[str, Any]] = {}
        source_bank_identity = _verify_sources(p, args.protocol)
        stage = "arm_setup"
        arm_install_metadata: dict[str, Any] = {"arm": arm, "native_kernel": False}
        if arm == "bf16":
            runtime.arm("bf16")
        elif arm == "spinquant_native_w4a4":
            spin_manifest = rotation.apply_native_bank(runtime.q, spin_root)
            runtime.q.enable("w4a4")
            arm_install_metadata = {"arm": arm, "native_kernel": True, "bank": str(spin_root),
                                    "manifest_arm": spin_manifest.get("arm"), "rotation_mode": spin_manifest.get("rotation_mode"),
                                    "module_count": spin_manifest.get("module_count"), "kv_cache_dtype": spin_manifest.get("kv_cache_dtype")}
            transforms_all = smooth_transforms
        elif arm == "smooth_scalar_native_w4a4":
            arm_install_metadata = scalar_native.install(runtime.q, smooth_bank)
            runtime.q.enable("w4a4")
            transforms_all = smooth_transforms
        elif arm in ("row_scale_vq_a4", "micro_scale_vq_a4"):
            transforms_all, arm_install_metadata = _decode_bank_weights(torch, modules, p, arm, device, core)
        else:
            raise ValueError(f"unknown arm: {arm}")
        runtime.current_arm = arm

        transform_by_module = transforms_all or smooth_transforms
        for index, name, stream, block in selected:
            if arm == "spinquant_native_w4a4":
                transform_id = f"learned_spinquant_{stream}"
            elif arm in ("smooth_scalar_native_w4a4", "row_scale_vq_a4", "micro_scale_vq_a4"):
                transform_id = f"smooth05_hadamard_m{index:04d}"
            else:
                transform_id = f"raw_identity_m{index:04d}"
            if arm == "bf16":
                identity = {"scale": torch.ones(modules[name].in_features, device=device),
                            "signs": torch.ones(modules[name].in_features, device=device),
                            "hadamard": False, "block": 128, "input_dim": int(modules[name].in_features)}
                raw_transform_records.append(_transform_artifact(np, torch, out, f"raw_identity_m{index:04d}",
                    "original BF16 model checkpoint", identity,
                    {"basis": "raw", "module": name, "input_dim": int(modules[name].in_features)}))
            else:
                _transform_artifact(np, torch, out, transform_id, str(spin_root if arm == "spinquant_native_w4a4" else smooth_bank),
                    spin_transforms[stream] if arm == "spinquant_native_w4a4" else transform_by_module[name],
                    {"basis": "learned_spinquant" if arm == "spinquant_native_w4a4" else "smooth05_hadamard",
                     "stream": stream, "module": name, "block_index": block})

        stage = "weight_snapshots"
        weight_records = []
        for index, name, stream, block in selected:
            module = modules[name]
            if arm in ("spinquant_native_w4a4", "smooth_scalar_native_w4a4"):
                actual_weight = _native_weight(torch, native, module)
            else:
                actual_weight = module.weight.detach().float()
            original = selected_original_weights[name].to(device=device, dtype=torch.float32)
            if arm == "bf16":
                reference = original
                basis, transform_id, transform_meta = "raw", f"raw_identity_m{index:04d}", {"type": "identity"}
            elif arm == "spinquant_native_w4a4":
                transform = spin_transforms[stream]
                reference = rotation._rotate_with_matrix(original, transform["matrix"], transform["signs"])
                basis, transform_id = "learned_spinquant", f"learned_spinquant_{stream}"
                transform_meta = {"type": "learned R128", "block": 128, "rotation_mode": "learned", "stream": stream}
            else:
                transform = transform_by_module[name]
                reference = core.transform_weight(original, transform)
                basis, transform_id = "smooth05_hadamard", f"smooth05_hadamard_m{index:04d}"
                transform_meta = {"type": "Smooth+H128", "block": 128, "hadamard": True,
                                  "scale_min": float(transform["scale"].min().item()),
                                  "scale_max": float(transform["scale"].max().item())}
            output_begin, output_end = window_bounds(int(module.out_features), int(p["weight_output_start"]), int(p["weight_output_count"]))
            input_begin, input_end = window_bounds(int(module.in_features), int(p["channel_start"]), int(p["plot_channels"]))
            weight_window = actual_weight[output_begin:output_end, input_begin:input_end]
            reference_window = reference[output_begin:output_end, input_begin:input_end]
            relative = Path("raw/weights") / arm / f"m{index:04d}.npz"
            record_id = f"weight:{arm}:{name}"
            metadata = {"kind": "weight", "record_id": record_id, "arm": arm, "module": name, "module_index": index,
                        "stream": stream, "block_index": block, "weight_shape": [int(module.out_features), int(module.in_features)],
                        "window_shape": [output_end - output_begin, input_end - input_begin],
                        "axes": ["output_channel", "input_channel"], "output_rows": [output_begin, output_end],
                        "input_channels": [input_begin, input_end], "basis": basis, "transform_id": transform_id,
                        "stage": "weight", "step": -1, "call_index": None,
                        "transform_metadata": transform_meta, "source_bank": arm_install_metadata,
                        "reference_source": "original BF16 weight transformed with the same frozen basis"}
            _atomic_npz(torch, np, out / relative, {"weight": weight_window, "reference": reference_window}, metadata)
            weight_records.append({"kind": "weight", "record_id": record_id, "module": name,
                "module_index": index, "block_index": block, "stream": stream, "arm": arm, "transform": basis,
                "case_id": None, "seed_index": None, "sampler_seed": None, "stage": "weight", "step": -1,
                "call_index": None, "shape": [output_end - output_begin, input_end - input_begin],
                "weight_shape": [int(module.out_features), int(module.in_features)], "axes": ["output_channel", "input_channel"],
                "file": relative.as_posix(), "keys": ["weight", "reference"], "source_bank": arm_install_metadata,
                "transform_id": transform_id, "transform_file": f"raw/transforms/{transform_id}.npz", "metadata": metadata})
        weights_checkpoint = out / "checkpoints" / f"weights_{arm}.json"
        _save_json(weights_checkpoint, {"protocol": p["protocol"], "arm": arm, "complete": True,
                    "records": weight_records, "source_bank": arm_install_metadata})
        transform_path = out / "transform_records.json"
        old_transform_rows = json.loads(transform_path.read_text(encoding="utf-8")) if transform_path.is_file() else []
        transform_by_id = {row["transform_id"]: row for row in old_transform_rows}
        for row in raw_transform_records + smooth_transform_records + spin_transform_records:
            transform_by_id[row["transform_id"]] = row
        _save_json(transform_path, list(transform_by_id.values()))

        tagger = CollectorStageTagger()
        scheduler_handles = source.install_stage_schedulers(runtime.model, tagger)
        conditioning_handles = source.install_video_conditioning_wrappers(runtime.model, tagger, scheduler_handles[0][3])
        active: dict[str, Any] = {}
        observed_all: set[str] = set()
        observed_selected: dict[str, set[tuple[str, int]]] = defaultdict(set)

        def pre_hook(module: Any, inputs: tuple[Any, ...], _name: str, _index: int) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                if _name in selected_by_name:
                    raise RuntimeError(f"selected Linear hook did not receive a tensor: {_name}")
                return
            stage_name, step_number = tagger.stage, tagger.step
            call_index = int(active["global_call_index"])
            active["global_call_index"] += 1
            observed_all.add(_name)
            if _name not in selected_by_name:
                return
            selected_index, stream, block = selected_by_name[_name]
            value = inputs[0].detach()
            original_shape = tuple(int(v) for v in value.shape)
            if value.shape[-1] != module.in_features:
                raise RuntimeError(f"selected input width differs for {_name}")
            if not should_capture(stream, stage_name, step_number, p["snapshot_steps"]):
                return
            batch_index = int(p.get("batch_index", 0))
            if arm == "bf16":
                pre_quant = value
                post_quant = value
                pre_scope, a4_applied = "raw BF16 Linear input; no A4 operation", False
            elif arm == "spinquant_native_w4a4":
                # apply_native_bank registered its rotation pre-hook first; value is already R128 transformed.
                pre_quant = value
                post_quant = _native_qdq(torch, native, pre_quant)
                pre_scope, a4_applied = "after native learned rotation, immediately before native A4", True
            elif arm == "smooth_scalar_native_w4a4":
                transform = smooth_transforms[_name]
                pre_quant = scalar_native.transform_input(value, transform["scale"], transform["signs"])
                post_quant = _native_qdq(torch, native, pre_quant)
                pre_scope, a4_applied = "after Smooth+H128, immediately before native A4", True
            else:
                transform = transforms_all[_name]
                pre_quant = core.transform_input(value, transform)
                post_quant = _smooth_qdq(torch, pre_quant)
                pre_scope, a4_applied = "after saved Smooth+H128, immediately before decoded-BF16 A4", True
            input_flat = _flat(torch, pre_quant, int(module.in_features), batch_index)
            p_start, p_end = window_bounds(int(input_flat.shape[0]), int(p["token_start"]), int(p["capture_tokens"]))
            active.setdefault("pending", defaultdict(list))[_name].append({
                "call_index": call_index, "stage": stage_name, "step": int(step_number),
                "raw_input_shape": list(original_shape), "pre_quant": input_flat[p_start:p_end],
                "post_quant": _flat(torch, post_quant, int(module.in_features), batch_index)[p_start:p_end],
                "token_row_ids": torch.arange(p_start, p_end, dtype=torch.int32),
                "original_flat_row_ids": _original_row_ids(torch, original_shape, p_start, p_end, batch_index),
                "batch_index": batch_index,
                "stream": stream, "block_index": block, "a4_applied": a4_applied,
                "pre_quant_scope": pre_scope, "raw_input": value,
            })
            observed_selected[_name].add((stage_name, int(step_number)))

        def output_hook(module: Any, inputs: tuple[Any, ...], output: Any, _name: str, _index: int) -> None:
            pending = active.get("pending", {}).get(_name)
            if not pending:
                return
            event = pending.pop()
            if not torch.is_tensor(output):
                raise RuntimeError(f"selected Linear output is not a tensor: {_name}")
            event["output_shape"] = [int(v) for v in output.shape]
            event["output_tensor"] = output.detach()
            begin, end = aligned_window(event["raw_input_shape"], event["output_shape"],
                                        int(p["token_start"]), int(p["capture_tokens"]),
                                        int(module.in_features), int(module.out_features),
                                        int(event["batch_index"]))
            if (begin, end) != (int(event["token_row_ids"][0]) if event["token_row_ids"].numel() else begin,
                                int(event["token_row_ids"][-1]) + 1 if event["token_row_ids"].numel() else end):
                raise RuntimeError(f"input/output token window alignment differs for {_name}")
            case = int(active["case_id"])
            sampler_seed = int(active["sampler_seed"])
            actual_relative = Path("raw/activations") / str(case) / arm / (
                f"m{_index:04d}_{event['stage']}_s{event['step']}_c{event['call_index']}.npz")
            actual_path = out / actual_relative
            output_flat = _flat(torch, event["output_tensor"], int(module.out_features), int(event["batch_index"]))
            out_begin, out_end = window_bounds(int(output_flat.shape[0]), begin, end - begin)
            output_channel_end = min(int(module.out_features), int(p["plot_channels"]))
            output_window = output_flat[out_begin:out_end, :output_channel_end]
            actual_record_id = f"activation:{arm}:{case}:{_name}:{event['stage']}:{event['step']}:{event['call_index']}"
            actual_metadata = {"protocol": p["protocol"], "kind": "activation", "arm": arm,
                "record_id": actual_record_id,
                "case_id": case, "seed_index": int(p["seed_index"]), "sampler_seed": sampler_seed,
                "module": _name, "module_index": _index, "stream": event["stream"],
                "block_index": event["block_index"], "stage": event["stage"], "step": event["step"],
                "call_index": event["call_index"], "input_shape": event["raw_input_shape"],
                "output_shape": event["output_shape"], "token_rows": [begin, end],
                "token_order": p.get("token_order", "flattened axes after batch"),
                "token_count": end - begin, "input_width": int(module.in_features),
                "output_width": int(module.out_features), "output_window_channels": [0, output_channel_end],
                "batch_index": event["batch_index"], "token_order": p.get("token_order", "flattened axes after batch"),
                "branch_identity": f"batch_index={event['batch_index']} (first sample/CFG branch)",
                "a4_applied": event["a4_applied"],
                "pre_quant_scope": event["pre_quant_scope"],
                "post_quant_scope": "native code*scale dequantization" if arm in ("spinquant_native_w4a4", "smooth_scalar_native_w4a4")
                                    else "decoded BF16 A4 output" if event["a4_applied"] else "identity copy of BF16 input",
                "source_bank": arm_install_metadata, "transform_id": (f"learned_spinquant_{event['stream']}" if arm == "spinquant_native_w4a4"
                    else f"smooth05_hadamard_m{_index:04d}" if arm in ("smooth_scalar_native_w4a4", "row_scale_vq_a4", "micro_scale_vq_a4")
                    else f"raw_identity_m{_index:04d}")}
            _atomic_npz(torch, np, actual_path, {"pre_quant": event["pre_quant"],
                "post_quant": event["post_quant"], "output": output_window,
                "token_row_ids": event["token_row_ids"], "original_flat_row_ids": event["original_flat_row_ids"]}, actual_metadata)
            actual_record = {"kind": "activation", "record_id": actual_record_id,
                "module": _name, "module_index": _index, "block_index": event["block_index"], "stream": event["stream"],
                "arm": arm, "case_id": case, "seed_index": int(p["seed_index"]), "sampler_seed": sampler_seed,
                "stage": event["stage"], "step": event["step"], "call_index": event["call_index"],
                "batch_index": event["batch_index"],
                "shape": [end - begin, int(module.in_features)], "input_shape": event["raw_input_shape"],
                "output_shape": event["output_shape"], "token_count": end - begin,
                "file": actual_relative.as_posix(), "keys": ["pre_quant", "post_quant", "output", "token_row_ids", "original_flat_row_ids"],
                "source_bank": arm_install_metadata, "transform_id": actual_metadata["transform_id"],
                "transform_file": f"raw/transforms/{actual_metadata['transform_id']}.npz", "metadata": actual_metadata}
            active["records"].append(actual_record)

            if arm == "bf16":
                raw = _flat(torch, event["raw_input"], int(module.in_features), int(event["batch_index"]))
                paired_values = {"paired_raw": raw, "paired_learned_spinquant": rotation._rotate_with_matrix(
                                    event["raw_input"], spin_transforms[event["stream"]]["matrix"], spin_transforms[event["stream"]]["signs"]),
                                 "paired_smooth05_hadamard": core.transform_input(event["raw_input"], smooth_transforms[_name])}
                paired_arrays = {"token_row_ids": event["token_row_ids"],
                                 "original_flat_row_ids": event["original_flat_row_ids"]}
                for key, value in paired_values.items():
                    window, bounds = _array_window(value, int(p["token_start"]), int(p["capture_tokens"]), int(event["batch_index"]))
                    if bounds != (begin, end):
                        raise RuntimeError(f"paired basis token alignment differs for {_name}")
                    paired_arrays[key] = window
                    paired_arrays[key + "_qdq"] = _native_qdq(torch, native, window)
                pair_relative = Path("raw/paired_local") / str(case) / (
                    f"m{_index:04d}_{event['stage']}_s{event['step']}_c{event['call_index']}.npz")
                pair_record_id = f"paired:{case}:{_name}:{event['stage']}:{event['step']}:{event['call_index']}"
                pair_metadata = {"protocol": p["protocol"], "kind": "paired_local", "record_id": pair_record_id, "case_id": case,
                    "seed_index": int(p["seed_index"]), "sampler_seed": sampler_seed, "module": _name,
                    "module_index": _index, "stream": event["stream"], "block_index": event["block_index"],
                    "stage": event["stage"], "step": event["step"], "call_index": event["call_index"],
                    "shape": [end - begin, int(module.in_features)], "input_shape": event["raw_input_shape"],
                    "output_shape": event["output_shape"], "token_rows": [begin, end],
                    "batch_index": event["batch_index"], "token_order": p.get("token_order", "flattened axes after batch"),
                    "branch_identity": f"batch_index={event['batch_index']} (first sample/CFG branch)",
                    "transforms": {"paired_raw": f"raw_identity_m{_index:04d}", "paired_learned_spinquant": f"learned_spinquant_{event['stream']}",
                                   "paired_smooth05_hadamard": f"smooth05_hadamard_m{_index:04d}"},
                    "source_checkpoint": "original BF16 model; all bases receive this same captured Linear input",
                    "candidate_qdq": "per-row signed A4 QDQ computed after each full-channel transform; visualization aid, not model execution"}
                _atomic_npz(torch, np, out / pair_relative, paired_arrays, pair_metadata)
                active["records"].append({"kind": "paired_local", "record_id": pair_record_id,
                    "module": _name, "module_index": _index, "block_index": event["block_index"], "stream": event["stream"],
                    "transform": "paired_local", "case_id": case, "seed_index": int(p["seed_index"]),
                    "sampler_seed": sampler_seed, "stage": event["stage"], "step": event["step"],
                    "call_index": event["call_index"], "batch_index": event["batch_index"],
                    "shape": [end - begin, int(module.in_features)],
                    "file": pair_relative.as_posix(), "keys": list(paired_arrays), "source_bank": "same BF16 input",
                    "transform_ids": pair_metadata["transforms"], "metadata": pair_metadata})
            event.clear()

        stage = "hook_install"
        selected_ids = {id(mod) for name, mod in module_items if name in selected_by_name}
        for index, (name, module) in enumerate(module_items):
            collector_hooks.append(module.register_forward_pre_hook(lambda mod, inputs, n=name, i=index: pre_hook(mod, inputs, n, i)))
            if id(module) in selected_ids:
                collector_hooks.append(module.register_forward_hook(lambda mod, inputs, output, n=name, i=index: output_hook(mod, inputs, output, n, i)))

        for case in pending_cases:
            if deadline - time.time() < 300:
                _save_json(out / "checkpoints" / f"{arm}_progress.json", {"protocol": p["protocol"], "arm": arm,
                            "status": "checkpointed", "completed_cases": completed, "pending_cases": pending_cases})
                return
            row, observation, metadata, sampler_seed = case_rows[case]
            datum = runtime.datum(observation)
            runtime.description = metadata["description"]
            runtime.current_arm = arm
            active.clear()
            active.update(case_id=case, sampler_seed=sampler_seed, global_call_index=0,
                          pending=defaultdict(list), records=[])
            observed_all.clear()
            observed_selected.clear()
            stage = "query"
            source.reset_stage_calls(scheduler_handles)
            tagger.set_stage("conditioning", -1)
            before_native = int(getattr(runtime.q, "_native_int4_gemm_calls", 0))
            started = time.monotonic()
            with torch.inference_mode():
                action, _ = runtime.infer(datum, sampler_seed, measure=False)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            stage_coverage = source.validate_stage_calls(scheduler_handles)
            if observed_all != set(modules):
                raise RuntimeError(f"{arm}/{case} observed {len(observed_all)}/614 target Linears")
            for index, name, stream, _block in selected:
                wanted_stage = "video" if stream == "video_expert" else "action"
                if not all((wanted_stage, step) in observed_selected[name] for step in p["snapshot_steps"]):
                    raise RuntimeError(f"{arm}/{case} selected module lacks denoising steps: {name} {sorted(observed_selected[name])}")
            native_calls = int(getattr(runtime.q, "_native_int4_gemm_calls", 0)) - before_native
            native_verified = bool(getattr(runtime.q, "_native_int4_ptx_verified", False))
            if arm in ("spinquant_native_w4a4", "smooth_scalar_native_w4a4") and (native_calls <= 0 or not native_verified):
                raise RuntimeError(f"{arm}/{case} did not verify native INT4 kernel calls")
            action_np = action.detach().float().cpu().numpy()
            reference = _cached_action(p, arm, case, int(p["seed_index"]), sampler_seed)
            if reference is None:
                reference_action = None
                reference_source = "parity_reference_absent: no frozen same-arm same-seed cached action; no decoded-BF16 substitution"
            else:
                reference_action, reference_path = reference
                reference_source = f"same-arm same-seed cached action: {reference_path}"
            if reference_action is None:
                comparison = {"status": "parity_reference_absent", "passive_hook_contract": "pre-hook and forward-hook return None; no collector replacement of inputs, output or module forward"}
            else:
                action_delta = np.asarray(action_np, dtype=np.float32) - np.asarray(reference_action, dtype=np.float32)
                max_abs = float(np.abs(action_delta).max())
                rmse = float(np.sqrt(np.mean(action_delta ** 2)))
                if max_abs > 1e-6:
                    raise RuntimeError(f"passive hook/action replay drift for {arm}/{case}: {max_abs}")
                comparison = {"status": "compared", "max_abs": max_abs, "rmse": rmse, "tolerance": 1e-6,
                              "passive_hooks_preserved_action": max_abs <= 1e-6}
            query_relative = Path("raw/queries") / arm / f"case_{case}_seed_{p['seed_index']}.json"
            query_record = {"protocol": p["protocol"], "query_id": f"{arm}:case{case}:seed{p['seed_index']}",
                "arm": arm, "case_id": case, "seed_index": int(p["seed_index"]), "sampler_seed": sampler_seed,
                "domain": row.get("domain"), "dimension": row.get("dimension"), "variant_id": row.get("variant_id"),
                "action": action_np.tolist(), "reference_source": reference_source,
                "reference_comparison": comparison, "actual_inference_calls_this_query": 1,
                "collector_hooks_observer_only": True,
                "target_module_count": len(observed_all), "selected_modules": len(observed_selected),
                "scheduler_coverage": stage_coverage, "native_int4_calls": native_calls,
                "native_int4_ptx_verified": native_verified, "elapsed_seconds_diagnostic_only": elapsed,
                "episodes": 0, "predicted_actions_executed": 0, "source_bank": arm_install_metadata}
            _save_json(out / query_relative, query_record)
            query_file = query_relative.as_posix()
            activation_records = active["records"]
            expected_records = set()
            for _index, name, stream, _block in selected:
                stage_name = "video" if stream == "video_expert" else "action"
                for step in p["snapshot_steps"]:
                    expected_records.add((name, stage_name, int(step)))
            actual_record_keys = {(r["module"], r["stage"], r["step"]) for r in activation_records if r["kind"] == "activation"}
            if not expected_records.issubset(actual_record_keys):
                raise RuntimeError(f"activation records miss expected module/stage/step coverage for {arm}/{case}")
            checkpoint = out / "checkpoints" / f"{arm}_{case}.json"
            _save_json(checkpoint, {"protocol": p["protocol"], "arm": arm, "case_id": case,
                "seed_index": int(p["seed_index"]), "sampler_seed": sampler_seed, "complete": True,
                "query": query_record, "query_file": query_file, "records": activation_records,
                "scheduler_coverage": stage_coverage, "source_bank": arm_install_metadata,
                "observed_module_count": len(observed_all)})
            completed.append(case)
            print("PRISM_VISUALIZATION_QUERY_COMPLETE " + json.dumps({"arm": arm, "case_id": case,
                  "seed_index": p["seed_index"], "records": len(activation_records),
                  "native_int4_calls": native_calls}), flush=True)
        _save_json(out / "checkpoints" / f"{arm}_progress.json", {"protocol": p["protocol"], "arm": arm,
                    "status": "complete" if len(completed) == len(p["cases"]) else "checkpointed",
                    "completed_cases": completed, "pending_cases": [case for case in p["cases"] if case not in completed]})
    except BaseException as exc:
        _save_json(out / "checkpoints" / f"{arm}_failure.json", {"protocol": p["protocol"], "arm": arm,
                    "stage": stage, "type": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
        raise
    finally:
        for hook in collector_hooks:
            hook.remove()
        if runtime is not None:
            try:
                if scheduler_handles:
                    source.restore_stage_schedulers(scheduler_handles)
                if conditioning_handles:
                    source.restore_methods(conditioning_handles)
                if arm == "smooth_scalar_native_w4a4":
                    scalar_native.close(runtime.q)
                elif arm == "spinquant_native_w4a4" and hasattr(runtime.q, "_rotation_native_state"):
                    rotation.discard_native_bank(runtime.q)
                if arm in ("row_scale_vq_a4", "micro_scale_vq_a4"):
                    for name, module in modules.items():
                        module.forward = original_forwards[name]
                if runtime.env is not None:
                    runtime.env.close()
                    runtime.env = None
                runtime.q.close()
            except Exception:
                pass


def _read_checkpoints(out: Path, p: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    queries, records, weights = [], [], []
    for arm in ARMS:
        weight_path = out / "checkpoints" / f"weights_{arm}.json"
        if weight_path.is_file():
            weight_value = json.loads(weight_path.read_text(encoding="utf-8"))
            if weight_value.get("protocol") == p["protocol"] and weight_value.get("complete"):
                weights.extend(weight_value.get("records", []))
        for case in p["cases"]:
            path = out / "checkpoints" / f"{arm}_{case}.json"
            if not path.is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("protocol") != p["protocol"] or value.get("complete") is not True:
                continue
            queries.append({"query_id": value["query"]["query_id"], "arm": arm, "case_id": case,
                            "seed_index": value["seed_index"], "sampler_seed": value["sampler_seed"],
                            "query_file": value["query_file"], "reference_comparison": value["query"]["reference_comparison"],
                            "reference_source": value["query"]["reference_source"],
                            "actual_inference_calls_this_query": value["query"]["actual_inference_calls_this_query"],
                            "collector_hooks_observer_only": value["query"]["collector_hooks_observer_only"],
                            "native_int4_calls": value["query"]["native_int4_calls"],
                            "native_int4_ptx_verified": value["query"]["native_int4_ptx_verified"],
                            "observed_module_count": value["observed_module_count"],
                            "scheduler_coverage": value["scheduler_coverage"]})
            records.extend(value.get("records", []))
    transform_path = out / "transform_records.json"
    transforms = json.loads(transform_path.read_text(encoding="utf-8")) if transform_path.is_file() else []
    return queries, records, weights + transforms


def _publish_manifest(out: Path, p: dict[str, Any], source_identities: dict[str, Any] | None = None) -> dict[str, Any]:
    queries, activation_records, weight_and_transforms = _read_checkpoints(out, p)
    weight_records = [row for row in weight_and_transforms if row.get("file", "").startswith("raw/weights/")]
    transform_records = [row for row in weight_and_transforms if row.get("file", "").startswith("raw/transforms/")]
    expected_query_ids = {f"{arm}:case{case}:seed{p['seed_index']}" for arm in ARMS for case in p["cases"]}
    actual_query_ids = {row["query_id"] for row in queries}
    complete = (actual_query_ids == expected_query_ids and len(weight_records) == len(ARMS) * 16)
    query_map = {row["query_id"]: row for row in queries}
    record_rows = activation_records + weight_records
    manifest = {"protocol": p["protocol"], "status": "complete" if complete else "checkpointed",
        "protocol_file": "protocol.json", "cases": p["cases"], "seed_indices": [p["seed_index"]],
        "sampler_seeds": {str(case): EXPECTED_SEEDS[case] for case in p["cases"]}, "arms": list(ARMS),
        "paired_transforms": list(PAIRED_TRANSFORMS), "queries_expected": int(p["max_queries"]),
        "query_count": len(queries), "runtime_inference_calls": sum(
            row["actual_inference_calls_this_query"] for row in queries),
        "parity_reference_absent_count": sum(
            row["reference_comparison"].get("status") == "parity_reference_absent" for row in queries),
        "evaluation_episodes": 0, "training": False, "refit": False,
        "selected_module_count": 16, "blocks": p["blocks"], "streams": p["streams"],
        "module_suffixes": p["module_suffixes"], "snapshot_steps": p["snapshot_steps"],
        "batch_index": int(p.get("batch_index", 0)),
        "token_order": p.get("token_order", "flattened axes after batch"),
        "branch_identity": "batch_index 0; first sample/CFG branch",
        "token_row_id_convention": "branch-local row index within the selected batch/CFG branch; original_flat_row_ids index the original flattened leading dimensions",
        "stage_step_convention": {"conditioning": -1, "video_conditioning_prefill": -1,
                                  "video_expert": "video", "action_expert": "action"},
        "primary_activation_record_rule": "lowest call_index for (case_id,seed_index,arm,module,stage,step); retain all calls in records",
        "records": record_rows, "query_records": [query_map[key] for key in sorted(query_map)],
        "transform_records": transform_records,
        "source_identities": source_identities or {},
        "activation_arrays": ["pre_quant", "post_quant", "output", "token_row_ids", "original_flat_row_ids"],
        "paired_local_arrays": ["paired_raw", "paired_raw_qdq", "paired_learned_spinquant",
            "paired_learned_spinquant_qdq", "paired_smooth05_hadamard", "paired_smooth05_hadamard_qdq", "token_row_ids", "original_flat_row_ids"],
        "weight_arrays": ["weight", "reference"],
        "raw_file_contract": "NPZ metadata_json is a NumPy Unicode scalar; activation and paired-local windows preserve signed full-channel values; weights are [output_channel,input_channel].",
        "plot_contract": {"activation_height": "absolute value", "activation_plot_window": [p["plot_tokens"], p["plot_channels"]],
                          "weight_window": [p["weight_output_count"], p["plot_channels"]],
                          "no_pooling_or_stride": True, "paired_local_is_separate_from_end_to_end": True},
        "evidence_boundary": {"native_arms": ["spinquant_native_w4a4", "smooth_scalar_native_w4a4"],
            "decoded_bf16_vq_arms": ["row_scale_vq_a4", "micro_scale_vq_a4"],
            "native_kernel_required_per_query": True}}
    _save_json(out / "raw/manifest.json", manifest)
    return manifest


def _write_summary(out: Path, status: str, manifest: dict[str, Any], identities: dict[str, Any]) -> None:
    rows = manifest["records"]
    _save_json(out / "summary.json", {"protocol": manifest["protocol"], "status": status,
        "query_count": manifest["query_count"], "runtime_inference_calls": manifest["runtime_inference_calls"],
        "weight_records": sum(row.get("kind") == "weight" for row in rows),
        "activation_records": sum(row.get("kind") == "activation" for row in rows),
        "paired_local_records": sum(row.get("kind") == "paired_local" for row in rows),
        "parity_reference_absent_count": manifest["parity_reference_absent_count"],
        "source_identities": identities, "pbs_jobid": os.environ["PBS_JOBID"]})


def _driver(args: argparse.Namespace, p: dict[str, Any]) -> None:
    guard()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / "protocol.json"
    if frozen_path.is_file() and json.loads(frozen_path.read_text(encoding="utf-8")) != p:
        raise RuntimeError("cannot resume with a changed frozen protocol")
    _save_json(frozen_path, p)
    identity_path = out / "source_identities.json"
    identities = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else _verify_sources(p, args.protocol)
    _save_json(identity_path, identities)
    driver_deadline = float(os.environ.get("FW_JOB_START_EPOCH", time.time())) + int(p["soft_deadline_seconds"])
    for arm in ARMS:
        expected = [out / "checkpoints" / f"{arm}_{case}.json" for case in p["cases"]]
        if all(path.is_file() for path in expected):
            _publish_manifest(out, p, identities)
            continue
        if driver_deadline - time.time() < 600:
            _save_json(out / "progress.json", {"protocol": p["protocol"], "status": "checkpointed",
                "completed_queries": len(_read_checkpoints(out, p)[0]), "expected_queries": p["max_queries"],
                "next_arm": arm, "reason": "soft deadline reserve"})
            manifest = _publish_manifest(out, p, identities)
            _write_summary(out, "checkpointed", manifest, identities)
            return
        command = [sys.executable, str(Path(__file__).resolve()), "--protocol", str(args.protocol.resolve()),
                   "--out", str(out), "--artifact", str((args.artifact or Path(os.environ.get("ARTIFACTS", out / "artifacts"))).resolve()),
                   "--worker-arm", arm]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            _publish_manifest(out, p, identities)
            raise RuntimeError(f"collector arm worker failed: {arm} exit={result.returncode}")
        manifest = _publish_manifest(out, p, identities)
        if not all((out / "checkpoints" / f"{arm}_{case}.json").is_file() for case in p["cases"]):
            _save_json(out / "progress.json", {"protocol": p["protocol"], "status": "checkpointed",
                "completed_queries": manifest["query_count"], "expected_queries": p["max_queries"],
                "next_arm": arm, "reason": "arm reached soft deadline"})
            _write_summary(out, "checkpointed", manifest, identities)
            return
    manifest = _publish_manifest(out, p, identities)
    if manifest["status"] != "complete":
        raise RuntimeError("collector reached the end without exact 10-query and 80-weight coverage")
    _write_summary(out, "complete", manifest, identities)
    _save_json(out / "progress.json", {"protocol": p["protocol"], "status": "complete",
        "completed_queries": manifest["query_count"], "expected_queries": p["max_queries"]})
    print(f"PRISM_VISUALIZATION_COMPLETE queries={manifest['query_count']} weights=80 records={len(manifest['records'])}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--worker-arm", choices=ARMS)
    args = parser.parse_args()
    guard()
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    _validate_protocol(p)
    if args.worker_arm:
        _run_arm(args, p, args.worker_arm)
    else:
        _driver(args, p)


if __name__ == "__main__":
    main()

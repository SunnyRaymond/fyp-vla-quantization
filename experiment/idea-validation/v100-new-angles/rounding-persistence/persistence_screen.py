"""Matched temporal-coupling screen for SmolVLA expert-only W4 SR snapshots.

This runner is intentionally a thin adapter around the already audited
``flow_screen.py`` loader and raw-sample helpers.  It must run inside a real
CCDS SLURM GPU allocation; it never downloads, trains, runs an environment, or
computes the scientific verdict.  The CPU verifier owns the S/cross-term and
endpoint metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "rounding-persistence-screen-v1"
RAW_SCHEMA = "rounding-persistence-raw-v1"
RAW_INPUT_SCHEMA = "rounding-persistence-raw-input-manifest-v1"
MODEL_REPO = "lerobot/smolvla_libero"
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
IDENTITY_SCHEMA = "smolvla-cpu-preparation-identity-v1"
HORIZON = 50
PHYSICAL_ACTION_DIM = 7
PADDED_ACTION_DIM = 32
STATE_DIM = 8
VELOCITY_STEPS = 10
DT = -0.1
BITS = 4
QMAX = 7
SAMPLE_COUNT = 6
TASKS = tuple(range(6))
NOISE_SEEDS = (2201, 2202)
NOISE_SHAPE = (1, HORIZON, PADDED_ACTION_DIM)
EXPERT_LINEAR_COUNT = 112
MAX_WORKLOAD_SECONDS = 540.0
TOLERANCE = 1e-6
EXPECTED_MANIFEST_SHA256 = "71243c83702ada092481abb2772787ebfc01774d8b92d63c4a5812e1771a04ef"
EXPECTED_BASE_IDENTITY_SHA256 = "be4a49ebe588a49a29bd26ed98b8a01a648a247e66d45e12a935ac7d8d0c4e64"
EXPECTED_FLOW_HELPER_SHA256 = "ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9"
EXPECTED_PROTOCOL_SHA256 = "0b9c9129ac8c81ff01fc26a304bc9ebd57ea25eeb0999505557198db713ac931"
ARM_NAMES = ("FP", "RTN", "F0", "F1", "F2", "C0", "C1", "C2")
DRAW_NAMES = ("F0", "F1", "F2")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _deadline(started: float, label: str) -> None:
    if time.monotonic() - started >= MAX_WORKLOAD_SECONDS:
        raise TimeoutError(f"workload deadline reached at {label}")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.casefold())


def _load_flow_helper(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"fixed flow helper is missing: {path}")
    helper_sha256 = _sha256_file(path)
    if helper_sha256 is None or helper_sha256.casefold() != EXPECTED_FLOW_HELPER_SHA256:
        raise RuntimeError(f"flow helper SHA disagrees with frozen pin: {path}")
    spec = importlib.util.spec_from_file_location("fixed_flow_screen_for_persistence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fixed flow helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "_load_raw_sample",
        "_prepare_batch",
        "_load_preprocessor",
        "_load_runtime",
        "_eligible_modules",
        "_state_subset_digest",
        "_gpu_evidence",
        "_clone_batch",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"fixed flow helper lacks required functions: {missing}")
    return module


def _load_protocol_identity(path: Path) -> Dict[str, Any]:
    identity = _file_identity(path.resolve())
    if identity["sha256"] is None or str(identity["sha256"]).casefold() != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"persistence protocol SHA disagrees with frozen pin: {path}")
    return identity


def _load_manifest(path: Path) -> Dict[str, Any]:
    resolved_path = path.resolve()
    manifest_sha256 = _sha256_file(resolved_path)
    if manifest_sha256 is None or manifest_sha256.casefold() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("persistence input manifest SHA disagrees with frozen pin")
    raw = _load_json(resolved_path, "persistence input manifest")
    if raw.get("schema") != RAW_INPUT_SCHEMA:
        raise RuntimeError(f"unsupported persistence manifest schema: {raw.get('schema')!r}")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) != SAMPLE_COUNT:
        raise RuntimeError("persistence manifest must contain exactly six samples")
    base_path_value = raw.get("base_identity_path")
    base_hash = raw.get("base_identity_sha256")
    if not isinstance(base_path_value, str) or not _is_sha(base_hash):
        raise RuntimeError("persistence manifest lacks base identity path/SHA")
    base_path = Path(base_path_value).expanduser().resolve()
    actual_base_hash = _sha256_file(base_path)
    if actual_base_hash is None or actual_base_hash.casefold() != str(base_hash).casefold():
        raise RuntimeError("base identity SHA disagrees with persistence manifest")
    if actual_base_hash.casefold() != EXPECTED_BASE_IDENTITY_SHA256:
        raise RuntimeError("base identity SHA disagrees with frozen pin")
    identity = dict(_load_json(base_path, "base SmolVLA identity"))
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise RuntimeError("base identity schema mismatch")
    checkpoint = identity.get("checkpoint")
    dataset = identity.get("dataset")
    if not isinstance(checkpoint, dict) or checkpoint.get("repo") != MODEL_REPO or checkpoint.get("revision") != MODEL_REVISION:
        raise RuntimeError("base identity checkpoint mismatch")
    if not isinstance(dataset, dict) or dataset.get("repo") != DATASET_REPO or dataset.get("revision") != DATASET_REVISION:
        raise RuntimeError("base identity dataset mismatch")
    extension_path = Path(str(raw.get("extension_identity_path", ""))).resolve()
    extension_sha = "f1d55051d660d25d7edfb01522870cdef08650be7752bbe7b52942741d1119ee"
    if raw.get("extension_identity_sha256") != extension_sha or _sha256_file(extension_path) != extension_sha:
        raise RuntimeError("extension identity differs from approved CPU preparation")
    extension = _load_json(extension_path, "conditional extension identity")
    parent = extension.get("bounded_extension", {})
    if (Path(str(parent.get("parent_identity_path", ""))).resolve() != base_path
            or parent.get("parent_identity_sha256") != EXPECTED_BASE_IDENTITY_SHA256):
        raise RuntimeError("extension parent chain mismatch")
    mapping = raw.get("mapping", {})
    if (mapping.get("state_shape") != [8] or mapping.get("physical_action_dim") != 7
            or mapping.get("padded_action_dim") != 32):
        raise RuntimeError("approved raw state/action dimensions mismatch")

    root = path.resolve().parent
    entries: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()
    for ordinal, item in enumerate(samples):
        if not isinstance(item, dict):
            raise RuntimeError(f"manifest sample {ordinal} is not an object")
        try:
            task = int(item["task_index"])
            episode = int(item.get("dataset_episode_index", item["episode_index"]))
            length = int(item["length"])
            frame = int(item["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"manifest sample {ordinal} lacks selection metadata") from exc
        if task not in TASKS or length <= 0 or frame != length // 2:
            raise RuntimeError(f"manifest sample {ordinal} violates frozen middle-frame rule")
        if (task, episode) in seen:
            raise RuntimeError(f"duplicate task/episode pair: {(task, episode)}")
        seen.add((task, episode))
        sample_rel = item.get("sample_path")
        recorded_hash = item.get("sha256")
        cameras = item.get("camera_keys")
        state_key = item.get("state_key")
        if item.get("state_shape") != [8] or item.get("action_shape") != [7]:
            raise RuntimeError("raw sample state/action dimension mismatch")
        if not isinstance(sample_rel, str) or not _is_sha(recorded_hash):
            raise RuntimeError(f"manifest sample {ordinal} lacks path/SHA")
        if cameras != ["observation.images.image", "observation.images.image2"] or state_key != "observation.state":
            raise RuntimeError(f"manifest sample {ordinal} violates fixed camera/state mapping")
        sample_path = (root / sample_rel).resolve()
        if root not in sample_path.parents or not sample_path.is_file():
            raise RuntimeError(f"manifest sample escapes asset root or is missing: {sample_path}")
        actual_hash = _sha256_file(sample_path)
        if actual_hash is None or actual_hash.casefold() != str(recorded_hash).casefold():
            raise RuntimeError(f"manifest sample SHA mismatch: {sample_path}")
        entry = dict(item)
        entry.update({
            "task_index": task,
            "episode_index": episode,
            "dataset_episode_index": episode,
            "length": length,
            "frame_index": frame,
            "sample_path": str(sample_path),
            "sha256": str(recorded_hash).casefold(),
        })
        entries.append(entry)
    entries.sort(key=lambda item: (item["task_index"], item["dataset_episode_index"]))
    if [item["task_index"] for item in entries] != list(TASKS):
        raise RuntimeError("persistence manifest must contain task0..task5 in order")
    return {
        "raw": raw,
        "identity": identity,
        "identity_path": str(base_path),
        "identity_sha256": actual_base_hash,
        "entries": entries,
        "path": str(path.resolve()),
        "sha256": manifest_sha256,
    }


def _load_runtime(flow: Any, args: argparse.Namespace, torch: Any, identity: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    runtime_args = argparse.Namespace(
        model_path=args.model_path.resolve(),
        checkpoint=args.checkpoint.resolve(),
        vlm_path=args.vlm_path.resolve(),
        lerobot_source=args.lerobot_source.resolve() if args.lerobot_source is not None else None,
        manifest_data={"identity": identity},
    )
    policy, _, runtime_identity = flow._load_runtime(runtime_args, torch)
    if runtime_identity.get("checkpoint", {}).get("sha256") is None:
        raise RuntimeError("runtime checkpoint SHA is unavailable")
    return policy, runtime_identity


def _noise_tensors(torch: Any, device: Any) -> Tuple[List[Any], np.ndarray]:
    values: List[Any] = []
    for seed in NOISE_SEEDS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        value = torch.randn(NOISE_SHAPE, dtype=torch.float32, generator=generator).to(device)
        values.append(value)
    return values, np.stack([value.detach().cpu().numpy()[0] for value in values]).astype(np.float32)


def _module_bindings(flow: Any, policy: Any, torch: Any) -> Tuple[List[Tuple[str, Any]], set[str], Dict[str, Any]]:
    groups = flow._eligible_modules(policy, torch)
    expert = sorted(groups.get("expert_W4", ()), key=lambda item: item[0])
    if len(expert) != EXPERT_LINEAR_COUNT:
        raise RuntimeError(f"expert W4 allowlist must contain exactly {EXPERT_LINEAR_COUNT} Linear modules, got {len(expert)}")
    try:
        named_parameter_items = policy.named_parameters(remove_duplicate=False)
    except TypeError as exc:
        raise RuntimeError("runtime cannot enumerate named parameter aliases") from exc
    aliases_by_id: Dict[int, List[str]] = {}
    for parameter_name, parameter in named_parameter_items:
        aliases_by_id.setdefault(id(parameter), []).append(str(parameter_name))
    state_names = set(policy.state_dict().keys())
    expert_state_names: set[str] = set()
    alias_report: Dict[str, Any] = {}
    for conceptual, module in expert:
        if not isinstance(module, torch.nn.Linear) or module.weight.ndim != 2 or module.weight.dtype != torch.float32:
            raise RuntimeError(f"expert allowlist contains a non-float32 Linear: {conceptual}")
        parameter_aliases = sorted(aliases_by_id.get(id(module.weight), ()))
        state_aliases = sorted(name for name in parameter_aliases if name in state_names)
        if not state_aliases:
            raise RuntimeError(f"expert module is not bound to one actual module.weight state entry: {conceptual}")
        expert_state_names.update(state_aliases)
        alias_report[conceptual] = {
            "parameter_aliases": parameter_aliases,
            "state_aliases": state_aliases,
            "shared_weight_alias": len(state_aliases) > 1,
        }
    return expert, expert_state_names, alias_report


def _tensor_map_digest(values: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tensor_bytes_sha256(value: Any, torch: Any) -> str:
    """Hash the exact contiguous float32 bytes captured at a denoise call."""
    if not torch.is_tensor(value):
        raise RuntimeError("cannot hash a non-tensor denoise input")
    array = value.detach().cpu().contiguous().numpy()
    if array.dtype != np.dtype(np.float32):
        raise RuntimeError(f"denoise input hash requires float32, got {array.dtype}")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _restore_fp(modules: Sequence[Tuple[str, Any]], fp: Mapping[str, Any], torch: Any) -> None:
    with torch.no_grad():
        for name, module in modules:
            module.weight.copy_(fp[name])
            if not torch.equal(module.weight, fp[name]):
                raise RuntimeError(f"FP restore mismatch: {name}")


def _apply_map(modules: Sequence[Tuple[str, Any]], values: Mapping[str, Any], torch: Any) -> None:
    with torch.no_grad():
        for name, module in modules:
            if name not in values:
                raise RuntimeError(f"snapshot lacks actual module weight: {name}")
            module.weight.copy_(values[name])
            if not torch.equal(module.weight, values[name]):
                raise RuntimeError(f"quantized weight readback mismatch: {name}")


def _build_quant_snapshots(
    modules: Sequence[Tuple[str, Any]], fp: Mapping[str, Any], torch: Any
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Create RTN and three SR maps using one independent CUDA Generator per seed."""
    device = next(iter(fp.values())).device
    generators = {seed: torch.Generator(device="cuda") for seed in (2101, 2102, 2103)}
    for seed, generator in generators.items():
        generator.manual_seed(seed)
    maps: Dict[str, Dict[str, Any]] = {"RTN": {}, "F0": {}, "F1": {}, "F2": {}}
    records: Dict[str, Any] = {"schema": "rounding-persistence-snapshot-v1", "bits": BITS, "qmax": QMAX, "seeds": [2101, 2102, 2103], "modules": {}}
    serialized_modules: Dict[str, Any] = {}
    weight_owners: Dict[int, str] = {}
    for name, module in sorted(modules, key=lambda item: item[0]):
        weight = fp[name]
        owner = weight_owners.get(id(module.weight))
        if owner is not None:
            for label in maps:
                maps[label][name] = maps[label][owner]
            serialized_modules[name] = {"alias_of": owner, "shape": list(weight.shape)}
            continue
        weight_owners[id(module.weight)] = name
        rows = weight.reshape(weight.shape[0], -1)
        maximum = rows.abs().amax(dim=1, keepdim=True)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum * (1.0 / float(QMAX)))
        reciprocal = torch.reciprocal(scale)
        ratio = rows * reciprocal
        rtn_code = torch.clamp(torch.round(ratio), -QMAX, QMAX).to(dtype=torch.int8)
        rtn_code = torch.where(maximum == 0, torch.zeros_like(rtn_code), rtn_code)
        maps["RTN"][name] = (rtn_code.to(dtype=weight.dtype) * scale).reshape_as(weight).clone()
        draw_codes: Dict[str, Any] = {}
        draw_scales: Dict[str, Any] = {}
        for draw_index, seed in enumerate((2101, 2102, 2103)):
            lower = torch.floor(ratio)
            fraction = ratio - lower
            random = torch.rand(ratio.shape, dtype=weight.dtype, device=device, generator=generators[seed])
            code = torch.clamp(lower + (random < fraction).to(dtype=weight.dtype), -QMAX, QMAX).to(dtype=torch.int8)
            code = torch.where(maximum == 0, torch.zeros_like(code), code)
            label = f"F{draw_index}"
            maps[label][name] = (code.to(dtype=weight.dtype) * scale).reshape_as(weight).clone()
            draw_codes[label] = code.detach().cpu()
            draw_scales[label] = scale.detach().cpu()
        serialized_modules[name] = {
            "shape": list(weight.shape),
            "fp_weight": weight.detach().cpu(),
            "rtn_codes": rtn_code.detach().cpu(),
            "scales": draw_scales["F0"],
            "draw_codes": draw_codes,
        }
    records["modules"] = serialized_modules
    records["map_digests"] = {label: _tensor_map_digest(values, torch) for label, values in maps.items()}
    records["module_count"] = len(modules)
    records["unique_weight_count"] = len(weight_owners)
    return maps, records


def _nested_tensor_digest(value: Any, torch: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, path: str) -> None:
        digest.update(path.encode("utf-8"))
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.map[{key!s}]")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        # Transformers cache objects in this runtime expose key_cache/value_cache.
        if hasattr(item, "key_cache") and hasattr(item, "value_cache"):
            visit(getattr(item, "key_cache"), f"{path}.key_cache")
            visit(getattr(item, "value_cache"), f"{path}.value_cache")
            return
        raise RuntimeError(f"unsupported prefix cache object at {path}: {type(item)!r}")

    visit(value, "root")
    return digest.hexdigest()


def _set_x_argument(args: List[Any], kwargs: Dict[str, Any], parameter_names: Sequence[str], value: Any) -> None:
    if "x_t" in kwargs:
        kwargs["x_t"] = value
        return
    try:
        index = list(parameter_names).index("x_t")
    except ValueError as exc:
        raise RuntimeError("official denoise_step signature has no x_t argument") from exc
    if index >= len(args):
        raise RuntimeError("denoise_step call did not bind x_t")
    args[index] = value


def _run_schedule(
    flow: Any,
    policy: Any,
    batch: Mapping[str, Any],
    noise: Any,
    schedule: Sequence[str],
    maps: Mapping[str, Mapping[str, Any]],
    fp: Mapping[str, Any],
    modules: Sequence[Tuple[str, Any]],
    torch: Any,
    *,
    reload_each_step: bool,
    force_x: Any | None = None,
) -> Dict[str, Any]:
    if len(schedule) != VELOCITY_STEPS:
        raise RuntimeError("schedule must contain exactly ten draw labels")
    if schedule[0] == "FP":
        _restore_fp(modules, fp, torch)
    else:
        if schedule[0] not in maps:
            raise RuntimeError(f"unknown initial schedule label: {schedule[0]}")
        _apply_map(modules, maps[schedule[0]], torch)
    model = policy.model
    original = model.denoise_step
    signature = inspect.signature(original)
    if tuple(signature.parameters) != ("prefix_pad_masks", "past_key_values", "x_t", "timestep"):
        raise RuntimeError(f"official v0.4.4 denoise_step signature drifted: {tuple(signature.parameters)}")
    parameter_names = tuple(signature.parameters)
    velocities: List[Any] = []
    x_inputs: List[Any] = []
    x_input_sha256: List[str] = []
    labels: List[str] = []
    actual_timesteps: List[float] = []
    cache_digest: str | None = None
    cache_end_digest: str | None = None
    cache_object_id: int | None = None
    last_cache: Any | None = None

    def wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal cache_digest, cache_object_id, last_cache
        step = len(velocities)
        if step >= VELOCITY_STEPS:
            raise RuntimeError("denoise_step called more than ten times")
        bound = signature.bind_partial(*call_args, **call_kwargs)
        if "x_t" not in bound.arguments or "past_key_values" not in bound.arguments:
            raise RuntimeError("denoise_step call lacks official x_t/past_key_values arguments")
        current_x = bound.arguments["x_t"]
        if force_x is not None:
            fixed = force_x[step].to(device=current_x.device, dtype=current_x.dtype)
            call_args_list = list(call_args)
            call_kwargs = dict(call_kwargs)
            _set_x_argument(call_args_list, call_kwargs, parameter_names, fixed)
            current_x = fixed
        if not torch.is_tensor(current_x) or tuple(current_x.shape) != (1, HORIZON, PADDED_ACTION_DIM):
            raise RuntimeError(f"denoise_step x_t shape is not [1,50,32]: {getattr(current_x, 'shape', None)}")
        cache = bound.arguments["past_key_values"]
        last_cache = cache
        timestep = bound.arguments["timestep"]
        expected_timestep = 1.0 + float(DT) * step
        if not torch.is_tensor(timestep) or tuple(timestep.shape) != (1,) or timestep.dtype != torch.float32:
            raise RuntimeError(f"denoise_step timestep shape/dtype drifted: {getattr(timestep, 'shape', None)}, {getattr(timestep, 'dtype', None)}")
        if not torch.allclose(timestep, torch.full_like(timestep, expected_timestep), atol=TOLERANCE, rtol=0):
            raise RuntimeError(f"denoise_step timestep differs from frozen grid at step {step}")
        if step == 0:
            cache_digest = _nested_tensor_digest(cache, torch)
            cache_object_id = id(cache)
        elif id(cache) != cache_object_id:
            raise RuntimeError("prefix past_key_values object changed between denoising calls")
        label = str(schedule[step])
        if label == "FP":
            _restore_fp(modules, fp, torch)
        elif label in maps:
            if reload_each_step or step == 0:
                _apply_map(modules, maps[label], torch)
            else:
                # A frozen schedule intentionally keeps one snapshot resident.
                # Read it back at every call so an accidental mutation fails
                # closed instead of silently changing the schedule.
                for name, module in modules:
                    if not torch.equal(module.weight, maps[label][name]):
                        raise RuntimeError(f"resident snapshot changed before denoise_step: {name}")
        else:
            raise RuntimeError(f"unknown draw schedule label at step {step}: {label}")
        x_input_sha256.append(_tensor_bytes_sha256(current_x, torch))
        value = original(*(call_args_list if force_x is not None else call_args), **(call_kwargs if force_x is not None else call_kwargs))
        if not torch.is_tensor(value) or tuple(value.shape) != (1, HORIZON, PADDED_ACTION_DIM):
            raise RuntimeError(f"denoise_step velocity shape is not [1,50,32]: {getattr(value, 'shape', None)}")
        x_inputs.append(current_x.detach().clone())
        velocities.append(value.detach().clone())
        labels.append(label)
        actual_timesteps.append(float(timestep.detach().cpu().reshape(-1)[0].item()))
        return value

    model.denoise_step = wrapped
    try:
        policy.reset()
        output = policy.predict_action_chunk(flow._clone_batch(batch, torch), noise=noise)
    finally:
        model.denoise_step = original
    if getattr(model.denoise_step, "__func__", None) is not getattr(original, "__func__", None):
        raise RuntimeError("denoise_step wrapper was not restored")
    if len(velocities) != VELOCITY_STEPS:
        raise RuntimeError(f"expected ten denoise calls, got {len(velocities)}")
    if last_cache is None or cache_digest is None:
        raise RuntimeError("denoise trajectory did not expose a prefix cache")
    cache_end_digest = _nested_tensor_digest(last_cache, torch)
    if cache_end_digest != cache_digest:
        raise RuntimeError("prefix cache content changed during denoising trajectory")
    velocity_tensor = torch.stack(velocities, dim=0)[:, 0]
    x_input_tensor = torch.stack(x_inputs, dim=0)[:, 0]
    next_tensor = x_input_tensor[-1] + float(DT) * velocity_tensor[-1]
    x_states = torch.cat([x_input_tensor, next_tensor.unsqueeze(0)], dim=0)
    if tuple(output.shape) != (1, HORIZON, PHYSICAL_ACTION_DIM):
        raise RuntimeError(f"physical action output shape is not [1,50,7]: {tuple(output.shape)}")
    # In a common-FP-path run the wrapper intentionally replaces every sampler
    # input with the saved FP x_t.  The sampler's own accumulated output is
    # then only a transport value, so it cannot equal fixed-path x_9+dt*v_Q9.
    # Free-running arms must satisfy this reconstruction gate.
    if force_x is None and not torch.allclose(output[0], x_states[-1, :, :PHYSICAL_ACTION_DIM], atol=TOLERANCE, rtol=0):
        raise RuntimeError("recorded x states do not reproduce official physical action output")
    return {
        "action": output.detach(),
        "velocities": velocity_tensor.detach().cpu().numpy().astype(np.float32),
        "x_states": x_states.detach().cpu().numpy().astype(np.float32),
        "cache_digest": cache_digest,
        "cache_end_digest": cache_end_digest,
        "cache_object_id": cache_object_id,
        "labels": labels,
        "actual_timesteps": np.asarray(actual_timesteps, dtype=np.float32),
        "x_input_sha256": x_input_sha256,
    }


def _manual_fp_action_unwrapped(policy: Any, batch: Mapping[str, Any], noise: Any, torch: Any) -> Any:
    """A direct v0.4.4 Euler call used only for the first manual-vs-official gate."""
    model = policy.model
    copied = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    images, masks = policy.prepare_images(copied)
    state = policy.prepare_state(copied)
    lang_tokens = copied["observation.language.tokens"]
    lang_masks = copied["observation.language.attention_mask"]
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, masks, lang_tokens, lang_masks, state=state)
    prefix_att = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks) if hasattr(model, "make_att_2d_masks") else None
    if prefix_att is None:
        prefix_att = importlib.import_module("lerobot.policies.smolvla.modeling_smolvla").make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past = model.vlm_with_expert.forward(
        attention_mask=prefix_att,
        position_ids=prefix_positions,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )
    x_t = noise.clone()
    for step in range(VELOCITY_STEPS):
        scalar_time = 1.0 + step * float(DT)
        timestep = torch.tensor(scalar_time, dtype=torch.float32, device=x_t.device).expand(x_t.shape[0])
        value = model.denoise_step(
            x_t=x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past,
            timestep=timestep,
        )
        if tuple(value.shape) != (1, HORIZON, PADDED_ACTION_DIM):
            raise RuntimeError("manual FP denoise velocity shape mismatch")
        x_t = x_t + float(DT) * value
    return x_t[:, :, :PHYSICAL_ACTION_DIM]


def _manual_fp_action(policy: Any, batch: Mapping[str, Any], noise: Any, torch: Any) -> Any:
    # The official policy entry point is @torch.no_grad, while this direct
    # Euler implementation is used as an independent no-op gate and therefore
    # needs its own explicit no-grad scope.
    with torch.no_grad():
        return _manual_fp_action_unwrapped(policy, batch, noise, torch)


def _schedule_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frozen = np.asarray([[draw] * VELOCITY_STEPS for draw in range(3)], dtype=np.int64)
    cyclic = np.asarray([[(draw + step) % 3 for step in range(VELOCITY_STEPS)] for draw in range(3)], dtype=np.int64)
    return np.stack([frozen, cyclic], axis=0), np.asarray([1.0 + DT * step for step in range(VELOCITY_STEPS)], dtype=np.float32), np.full(VELOCITY_STEPS, DT, dtype=np.float32)


def _labels_to_draw_indices(labels: Sequence[str]) -> np.ndarray:
    indices = np.full(len(labels), -1, dtype=np.int8)
    for index, label in enumerate(labels):
        if label in DRAW_NAMES:
            indices[index] = DRAW_NAMES.index(label)
        elif label not in {"FP", "RTN"}:
            raise RuntimeError(f"unexpected actual schedule label: {label}")
    return indices


def _write_raw(
    output: Path,
    raw_x: np.ndarray,
    raw_v: np.ndarray,
    actions: np.ndarray,
    common_q_v: np.ndarray,
    common_x_sha256: np.ndarray,
    actual_times: np.ndarray,
    noise: np.ndarray,
    completed: np.ndarray,
    episode_ids: Sequence[str],
    schedule: np.ndarray,
    times: np.ndarray,
    dt: np.ndarray,
    actual_schedule: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    _atomic_npz(
        output / "raw_persistence.npz",
        schema=np.asarray(RAW_SCHEMA),
        arm_names=np.asarray(ARM_NAMES),
        raw_x=raw_x,
        raw_v=raw_v,
        actions=actions,
        common_q_v=common_q_v,
        common_x_sha256=common_x_sha256,
        actual_times=actual_times,
        noise=noise,
        times=times,
        dt=dt,
        schedule=schedule,
        actual_schedule=actual_schedule,
        completed=completed,
        episode_ids=np.asarray(episode_ids),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def _write_summary(output: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB: {len(encoded)} bytes")
    temporary = output / "summary.json.writing"
    temporary.write_bytes(encoded)
    temporary.replace(output / "summary.json")


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engineering: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "allocation": dict(allocation),
        "parameters": {
            "arms": list(ARM_NAMES),
            "draw_seeds": [2101, 2102, 2103],
            "noise_seeds": list(NOISE_SEEDS),
            "noise_shape": list(NOISE_SHAPE),
            "raw_x_shape": [SAMPLE_COUNT, len(NOISE_SEEDS), len(ARM_NAMES), VELOCITY_STEPS + 1, HORIZON, PADDED_ACTION_DIM],
            "raw_v_shape": [SAMPLE_COUNT, len(NOISE_SEEDS), len(ARM_NAMES), VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM],
            "action_shape": [SAMPLE_COUNT, len(NOISE_SEEDS), len(ARM_NAMES), HORIZON, PHYSICAL_ACTION_DIM],
            "common_q_v_shape": [SAMPLE_COUNT, len(NOISE_SEEDS), 3, VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM],
            "common_x_sha256_shape": [SAMPLE_COUNT, len(NOISE_SEEDS), 3, VELOCITY_STEPS],
            "actual_times_shape": [SAMPLE_COUNT, len(NOISE_SEEDS), len(ARM_NAMES), VELOCITY_STEPS],
            "expert_linear_count": EXPERT_LINEAR_COUNT,
            "dt": DT,
            "bits": BITS,
            "qmax": QMAX,
            "execution": "FP32 fake-quantized dequantized weights; no native low-bit claim",
        },
    }
    _atomic_json(output / "engineering.json", engineering)
    try:
        manifest = _load_manifest(args.input_manifest)
        engineering["protocol_identity"] = _load_protocol_identity(args.protocol_copy)
        engineering["input_manifest"] = {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "schema": manifest["raw"]["schema"],
            "identity_path": manifest["identity_path"],
            "identity_sha256": manifest["identity_sha256"],
            "episode_ids": [f"task{entry['task_index']}:episode{entry['episode_index']}:frame{entry['frame_index']}" for entry in manifest["entries"]],
            "sample_sha256": [entry["sha256"] for entry in manifest["entries"]],
        }
        helper = _load_flow_helper(args.flow_helper.resolve())
        engineering["flow_helper"] = _file_identity(args.flow_helper.resolve())
        _deadline(started, "manifest and helper")

        import torch

        policy, runtime_identity = _load_runtime(helper, args, torch, manifest["identity"])
        gpu_identity = helper._gpu_evidence(torch)
        engineering["runtime_identity"] = runtime_identity
        engineering["gpu"] = gpu_identity
        engineering["source_identity"] = runtime_identity.get("source_identity")
        engineering["dataset_parent_identity"] = {
            "base_identity_path": manifest["identity_path"],
            "base_identity_sha256": manifest["identity_sha256"],
            "extension_identity_path": manifest["raw"].get("extension_identity_path"),
            "extension_identity_sha256": manifest["raw"].get("extension_identity_sha256"),
        }
        _atomic_json(output / "engineering.json", engineering)
        _deadline(started, "runtime load")

        preprocessor, preprocessor_identity = helper._load_preprocessor(args.model_path.resolve(), args.vlm_path.resolve())
        engineering["preprocessor"] = preprocessor_identity
        batches: List[Mapping[str, Any]] = []
        for entry in manifest["entries"]:
            raw = helper._load_raw_sample(Path(entry["sample_path"]), entry, torch)
            prepared, _ = helper._prepare_batch(raw, preprocessor, torch)
            batches.append(prepared)
        device = torch.device("cuda:0")
        batches = [helper._move_to_device(batch, device, torch) if hasattr(helper, "_move_to_device") else {key: value.to(device=device) if torch.is_tensor(value) else value for key, value in batch.items()} for batch in batches]
        noises, noise_np = _noise_tensors(torch, device)
        modules, expert_state_names, alias_report = _module_bindings(helper, policy, torch)
        fp = {name: module.weight.detach().clone() for name, module in modules}
        bypass_digest = helper._state_subset_digest(policy, expert_state_names, torch)
        maps, snapshot_records = _build_quant_snapshots(modules, fp, torch)
        snapshot_path = output / "rounding_snapshots.pt"
        torch.save(snapshot_records, snapshot_path)
        engineering["expert_binding"] = {
            "count": len(modules),
            "unique_weight_count": len({id(module.weight) for _, module in modules}),
            "weight_binding": "actual module.weight; no state_dict key rewrite",
            "parameter_aliases": alias_report,
            "nonexpert_state_digest_before": bypass_digest,
        }
        engineering["snapshots"] = {
            "path": str(snapshot_path),
            "sha256": _sha256_file(snapshot_path),
            "map_digests": snapshot_records["map_digests"],
            "scale_code_storage": "rounding_snapshots.pt; per-module codes/scales retained",
        }
        _atomic_json(output / "engineering.json", engineering)
        schedule_array, times, dt_array = _schedule_arrays()

        raw_x = np.full((SAMPLE_COUNT, 2, len(ARM_NAMES), VELOCITY_STEPS + 1, HORIZON, PADDED_ACTION_DIM), np.nan, dtype=np.float32)
        raw_v = np.full((SAMPLE_COUNT, 2, len(ARM_NAMES), VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM), np.nan, dtype=np.float32)
        actions = np.full((SAMPLE_COUNT, 2, len(ARM_NAMES), HORIZON, PHYSICAL_ACTION_DIM), np.nan, dtype=np.float32)
        common_q_v = np.full((SAMPLE_COUNT, 2, 3, VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM), np.nan, dtype=np.float32)
        common_x_sha256 = np.full((SAMPLE_COUNT, 2, 3, VELOCITY_STEPS), "", dtype="<U64")
        actual_times = np.full((SAMPLE_COUNT, 2, len(ARM_NAMES), VELOCITY_STEPS), np.nan, dtype=np.float32)
        completed = np.zeros((SAMPLE_COUNT, 2), dtype=np.bool_)
        actual_schedule = np.full((SAMPLE_COUNT, 2, len(ARM_NAMES), VELOCITY_STEPS), -1, dtype=np.int8)
        episode_ids = engineering["input_manifest"]["episode_ids"]
        prefix_cache_digests: Dict[str, str] = {}
        no_op = _run_schedule(helper, policy, batches[0], noises[0], ["FP"] * VELOCITY_STEPS, maps, fp, modules, torch, reload_each_step=False)
        manual_action = _manual_fp_action(policy, batches[0], noises[0], torch)
        if not torch.allclose(no_op["action"], manual_action, atol=TOLERANCE, rtol=0):
            raise RuntimeError("manual FP Euler action disagrees with official sample_actions")
        engineering["manual_fp_gate"] = {"allclose_atol_1e-6_rtol_0": True, "official_shape": list(no_op["action"].shape)}
        if no_op["cache_digest"] is None:
            raise RuntimeError("FP trajectory did not expose prefix cache digest")
        rtn_reuse = _run_schedule(helper, policy, batches[0], noises[0], ["RTN"] * VELOCITY_STEPS, maps, fp, modules, torch, reload_each_step=False)
        rtn_reload = _run_schedule(helper, policy, batches[0], noises[0], ["RTN"] * VELOCITY_STEPS, maps, fp, modules, torch, reload_each_step=True)
        if not np.allclose(rtn_reuse["velocities"], rtn_reload["velocities"], atol=TOLERANCE, rtol=0) or not torch.allclose(rtn_reuse["action"], rtn_reload["action"], atol=TOLERANCE, rtol=0):
            raise RuntimeError("RTN reuse/reload negative control is not identical")
        engineering["rtn_schedule_negative_control"] = {"allclose_atol_1e-6_rtol_0": True}
        _atomic_json(output / "engineering.json", engineering)

        map_by_arm: Dict[str, Mapping[str, Any]] = {"FP": fp, "RTN": maps["RTN"], "F0": maps["F0"], "F1": maps["F1"], "F2": maps["F2"], "C0": maps["F0"], "C1": maps["F1"], "C2": maps["F2"]}
        for state_index, batch in enumerate(batches):
            for noise_index, noise in enumerate(noises):
                fp_result: Dict[str, Any] | None = None
                condition_cache_digest: str | None = None
                for arm_index, arm in enumerate(ARM_NAMES):
                    if arm.startswith("C"):
                        draw = int(arm[1:])
                        arm_schedule = [DRAW_NAMES[(draw + step) % 3] for step in range(VELOCITY_STEPS)]
                        reload = True
                    elif arm in DRAW_NAMES:
                        arm_schedule = [arm] * VELOCITY_STEPS
                        reload = False
                    else:
                        arm_schedule = [arm] * VELOCITY_STEPS
                        reload = False
                    result = _run_schedule(helper, policy, batch, noise, arm_schedule, map_by_arm, fp, modules, torch, reload_each_step=reload)
                    if result["cache_digest"] is None:
                        raise RuntimeError(f"prefix cache digest missing for arm {arm}")
                    if arm == "FP":
                        condition_cache_digest = result["cache_digest"]
                        if state_index == 0 and noise_index == 0 and condition_cache_digest != no_op["cache_digest"]:
                            raise RuntimeError("first FP condition differs from manual no-op prefix cache")
                    elif condition_cache_digest is None or result["cache_digest"] != condition_cache_digest:
                        raise RuntimeError(f"prefix cache digest changed for arm {arm}")
                    raw_x[state_index, noise_index, arm_index] = result["x_states"]
                    raw_v[state_index, noise_index, arm_index] = result["velocities"]
                    actions[state_index, noise_index, arm_index] = result["action"].cpu().numpy()[0]
                    actual_schedule[state_index, noise_index, arm_index] = _labels_to_draw_indices(result["labels"])
                    actual_times[state_index, noise_index, arm_index] = result["actual_timesteps"]
                    if arm == "FP":
                        fp_result = result
                    if not np.isfinite(raw_x[state_index, noise_index, arm_index]).all() or not np.isfinite(raw_v[state_index, noise_index, arm_index]).all() or not np.isfinite(actions[state_index, noise_index, arm_index]).all():
                        raise FloatingPointError(f"non-finite raw output at state={state_index}, noise={noise_index}, arm={arm}")
                    _restore_fp(modules, fp, torch)
                    if helper._state_subset_digest(policy, expert_state_names, torch) != bypass_digest:
                        raise RuntimeError(f"nonexpert state digest changed after arm {arm}")
                if fp_result is None:
                    raise RuntimeError("FP result missing")
                fp_x = torch.from_numpy(fp_result["x_states"][:-1]).to(device=device, dtype=torch.float32).unsqueeze(1)
                for draw_index, label in enumerate(DRAW_NAMES):
                    common = _run_schedule(helper, policy, batch, noise, [label] * VELOCITY_STEPS, map_by_arm, fp, modules, torch, reload_each_step=False, force_x=fp_x)
                    if common["cache_digest"] != condition_cache_digest:
                        raise RuntimeError(f"prefix cache digest changed in common FP path for {label}")
                    common_q_v[state_index, noise_index, draw_index] = common["velocities"]
                    common_x_sha256[state_index, noise_index, draw_index] = np.asarray(common["x_input_sha256"], dtype="<U64")
                    _restore_fp(modules, fp, torch)
                    if helper._state_subset_digest(policy, expert_state_names, torch) != bypass_digest:
                        raise RuntimeError(f"nonexpert state digest changed after common path {label}")
                if condition_cache_digest is None:
                    raise RuntimeError("condition FP prefix cache digest was not established")
                prefix_cache_digests[f"state{state_index}:noise{noise_index}"] = condition_cache_digest
                completed[state_index, noise_index] = True
                _write_raw(
                    output,
                    raw_x,
                    raw_v,
                    actions,
                    common_q_v,
                    common_x_sha256,
                    actual_times,
                    noise_np,
                    completed,
                    episode_ids,
                    schedule_array,
                    times,
                    dt_array,
                    actual_schedule,
                    {
                        "schema": RAW_SCHEMA,
                        "manifest_path": manifest["path"],
                        "manifest_sha256": manifest["sha256"],
                        "identity_sha256": manifest["identity_sha256"],
                        "source_identity": runtime_identity.get("source_identity"),
                        "schedule_definition": "frozen[d,t]=d; cyclic[d,t]=(d+t)%3",
                        "physical_readout": "first 7 action coordinates; common path retains padded 32D velocity",
                        "common_x_sha256_formula": "sha256(contiguous float32 bytes of exact x_t captured immediately before each denoise_step)",
                    },
                )
                _deadline(started, f"state {state_index} noise {noise_index}")
                engineering["completed"] = completed.tolist()
                engineering["prefix_cache_digests"] = prefix_cache_digests
                engineering["last_completed"] = {"state": state_index, "noise": noise_index}
                _atomic_json(output / "engineering.json", engineering)
        _restore_fp(modules, fp, torch)
        if helper._state_subset_digest(policy, expert_state_names, torch) != bypass_digest:
            raise RuntimeError("nonexpert state digest changed after final FP restoration")
        if not bool(completed.all()):
            raise RuntimeError("not all state/noise conditions completed")
        engineering.update({"status": "complete", "completed": completed.tolist(), "elapsed_seconds": time.monotonic() - started, "weight_restore": "exact after every arm and final FP restore", "scientific_metrics": "delegated to independent CPU verifier"})
        _atomic_json(output / "engineering.json", engineering)
        _write_summary(output, {
            "schema": SCHEMA,
            "status": "complete",
            "raw_schema": RAW_SCHEMA,
            "raw_npz": str((output / "raw_persistence.npz").resolve()),
            "raw_sha256": _sha256_file(output / "raw_persistence.npz"),
            "engineering_json": str((output / "engineering.json").resolve()),
            "snapshot_path": str(snapshot_path),
            "completed_shape": list(completed.shape),
            "completed_count": int(completed.sum()),
            "arm_names": list(ARM_NAMES),
            "source_identity_recorded": True,
            "scientific_verdict": "independent CPU verifier required",
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"status": "complete", "output": str(output), "completed": int(completed.sum())}, sort_keys=True), flush=True)
    except Exception as exc:
        status = "implementation_failure" if isinstance(exc, (RuntimeError, FileNotFoundError, ImportError, ValueError)) else "inconclusive"
        engineering.update({"status": status, "error": f"{type(exc).__name__}: {exc}", "completed": engineering.get("completed", [[False, False] for _ in range(SAMPLE_COUNT)]), "elapsed_seconds": time.monotonic() - started})
        _atomic_json(output / "engineering.json", engineering)
        _write_summary(output, {
            "schema": SCHEMA,
            "status": status,
            "error": engineering["error"],
            "raw_npz": str((output / "raw_persistence.npz").resolve()),
            "engineering_json": str((output / "engineering.json").resolve()),
            "scientific_verdict": "not computed",
            "elapsed_seconds": engineering["elapsed_seconds"],
        })
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--flow-helper", type=Path, required=True)
    parser.add_argument("--protocol-copy", type=Path, required=True)
    parser.add_argument("--lerobot-source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_WORKLOAD_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_WORKLOAD_SECONDS:g}]")
    return args


def main() -> None:
    args = _parse_args()
    # This import and call are intentionally the first workload action.  The
    # copied campaign guard checks the real SLURM job/owner/partition/node.
    require_allocation = importlib.import_module("allocation_guard").require_allocation
    allocation = require_allocation()
    _run(args, allocation)


if __name__ == "__main__":
    main()

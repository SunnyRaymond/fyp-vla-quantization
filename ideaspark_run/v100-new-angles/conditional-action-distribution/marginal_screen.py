"""Run the bounded conditional-action marginal screen for SmolVLA.

This runner owns only the two-arm raw action collection.  It reuses the pinned
Flow screen helpers for runtime loading, preprocessing, raw sample conversion,
module discovery and the expert-only W4 quantizer, but does not call that
screen's old manifest loader, runner or science metrics.  The root CPU
verifier owns all marginal-distance and mapping gates.

The allocation guard is called before loading the pinned helper, torch, the
manifest samples, or the checkpoint.  The script is intended to run only from
the copied job snapshot on a real CCDS SLURM V100 allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


TOP_DEFAULT = Path("/tc1home/UG/yguo017/v100_newangles_ccds")
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
MODEL_REPO = "lerobot/smolvla_libero"
BASE_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
IDENTITY_SCHEMA = "smolvla-cpu-preparation-identity-v1"
INPUT_SCHEMA = "conditional-marginal-raw-input-manifest-v1"
RAW_SCHEMA = "conditional-marginal-raw-v1"
RAW_SAMPLE_SCHEMA = "smolvla-raw-input-sample-v1"
ARM_NAMES = ("FP32", "expert_W4")
NOISE_SEEDS = (1901, 1902)
NOISE_BLOCK_SIZE = 32
NOISE_COUNT = 64
HORIZON = 50
PADDED_ACTION_DIM = 32
PHYSICAL_ACTION_DIM = 7
VELOCITY_STEPS = 10
MICRO_BATCH = 4
CONDITION_COUNT = 8
BITS = 4
QMAX = 7
FLOW_SCREEN_SHA256 = "ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9"
MAX_WORKLOAD_SECONDS = 840.0
BATCH_TOLERANCE = 1e-5


class BatchImplementationInconclusive(RuntimeError):
    """The fixed batch-vs-individual implementation gate failed."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, np: Any, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_small_summary(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB ({len(encoded)} bytes)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _resolve_under(root: Path, value: str, label: str) -> Path:
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"{label} escapes asset root: {path}")
    return path


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.casefold())


def _load_input_manifest(path: Path, asset_root: Path) -> Dict[str, Any]:
    """Validate the eight-sample manifest without selecting or replacing samples."""
    raw = _load_json(path.resolve(), "conditional marginal input manifest")
    if raw.get("schema") != INPUT_SCHEMA:
        raise RuntimeError(f"unsupported input manifest schema: {raw.get('schema')!r}")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) != CONDITION_COUNT:
        raise RuntimeError("conditional marginal manifest must contain exactly eight samples")

    identity_value = raw.get("base_identity_path")
    identity_hash = raw.get("base_identity_sha256")
    if not isinstance(identity_value, str) or not _is_sha256(identity_hash):
        raise RuntimeError("manifest lacks base_identity_path/base_identity_sha256")
    identity_path = _resolve_under(asset_root, identity_value, "base_identity_path")
    actual_identity_hash = _sha256_file(identity_path)
    if actual_identity_hash is None or actual_identity_hash.casefold() != identity_hash.casefold():
        raise RuntimeError("base identity SHA-256 mismatch")
    identity = _load_json(identity_path, "base identity")
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise RuntimeError("base identity schema mismatch")
    checkpoint = identity.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("repo") != MODEL_REPO or checkpoint.get("revision") != MODEL_REVISION:
        raise RuntimeError("base identity checkpoint pin mismatch")
    base_vlm = identity.get("base_vlm")
    if not isinstance(base_vlm, dict) or base_vlm.get("repo") != BASE_VLM_REPO or base_vlm.get("weights_downloaded") is not False:
        raise RuntimeError("base identity VLM pin mismatch")
    dataset = identity.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("repo") != DATASET_REPO or dataset.get("revision") != DATASET_REVISION:
        raise RuntimeError("base identity dataset pin mismatch")
    mapping = identity.get("mapping")
    if not isinstance(mapping, dict) or mapping.get("state_key") != "observation.state":
        raise RuntimeError("base identity state mapping is not exact")
    if mapping.get("dataset_state_shape") != [8] or mapping.get("action_shape") != [7]:
        raise RuntimeError("base identity does not pin state8/action7")
    identity_cameras = mapping.get("present_image_keys")
    if not isinstance(identity_cameras, list) or len(identity_cameras) != 2 or len(set(identity_cameras)) != 2:
        raise RuntimeError("base identity does not pin exactly two present cameras")

    normalized: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[int, int]] = set()
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise RuntimeError(f"sample {index} is not a mapping")
        try:
            task_index = int(item["task_index"])
            episode_value = item.get("dataset_episode_index")
            if episode_value is None:
                episode_value = item["episode_index"]
            episode_index = int(episode_value)
            length = int(item["length"])
            frame_index = int(item["frame_index"])
            task_text = str(item["task_text"])
            sample_value = str(item["sample_path"])
            sample_hash = item.get("sha256", item.get("sample_sha256"))
            camera_keys = [str(value) for value in item["camera_keys"]]
            state_key = str(item["state_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"sample {index} has incomplete metadata") from exc
        if not _is_sha256(sample_hash):
            raise RuntimeError(f"sample {index} has no valid SHA-256")
        if task_index not in range(4) or episode_index < 0 or length <= 0 or frame_index != length // 4:
            raise RuntimeError(f"sample {index} violates frozen task/frame metadata")
        if not task_text.strip() or camera_keys != list(identity_cameras) or state_key != "observation.state":
            raise RuntimeError(f"sample {index} feature mapping differs from base identity")
        pair = (task_index, episode_index)
        if pair in seen_pairs:
            raise RuntimeError(f"duplicate task/episode pair: {pair}")
        seen_pairs.add(pair)
        sample_path = _resolve_under(path.parent, sample_value, f"sample {index}")
        if asset_root.resolve() not in sample_path.parents:
            raise RuntimeError(f"sample {index} is outside asset root: {sample_path}")
        actual_sample_hash = _sha256_file(sample_path)
        if actual_sample_hash is None or actual_sample_hash.casefold() != str(sample_hash).casefold():
            raise RuntimeError(f"sample {index} SHA-256 mismatch")
        normalized.append({
            "episode_index": index,
            "dataset_episode_index": episode_index,
            "task_index": task_index,
            "task_id": str(task_index),
            "task_text": task_text,
            "length": length,
            "frame_index": frame_index,
            "sample_path": str(sample_path),
            "sample_sha256": str(sample_hash).casefold(),
            "camera_keys": camera_keys,
            "state_key": state_key,
        })

    pairs = [(item["task_index"], item["dataset_episode_index"]) for item in normalized]
    if pairs != sorted(pairs) or len(set(pairs)) != CONDITION_COUNT:
        raise RuntimeError("samples must be sorted by task_index then episode_index")
    if any(sum(item["task_index"] == task for item in normalized) != 2 for task in range(4)):
        raise RuntimeError("manifest must contain two conditions for each task 0..3")
    return {
        "raw": raw,
        "identity": identity,
        "identity_path": str(identity_path),
        "identity_sha256": actual_identity_hash,
        "entries": normalized,
    }


def _load_flow(helper_path: Path) -> Any:
    helper_path = helper_path.resolve()
    actual = _sha256_file(helper_path)
    if actual != FLOW_SCREEN_SHA256:
        raise RuntimeError(f"flow_screen.py identity mismatch: {actual!r} != {FLOW_SCREEN_SHA256!r}")
    spec = importlib.util.spec_from_file_location("pinned_flow_screen_for_marginal", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_deadline(started: float, max_seconds: float, label: str) -> None:
    if time.monotonic() - started >= max_seconds:
        raise TimeoutError(f"max-seconds reached at {label}")


def _load_guard() -> Any:
    import importlib

    return importlib.import_module("allocation_guard").require_allocation


def _expert_state_names(policy: Any, expert_modules: Sequence[Tuple[str, Any]]) -> set[str]:
    try:
        named = policy.named_parameters(remove_duplicate=False)
    except TypeError:
        named = policy.named_parameters()
    by_parameter: Dict[int, List[str]] = {}
    for name, parameter in named:
        by_parameter.setdefault(id(parameter), []).append(name)
    state_names = set(policy.state_dict().keys())
    result: set[str] = set()
    for conceptual_name, module in expert_modules:
        matched = by_parameter.get(id(module.weight), [])
        if not matched:
            raise RuntimeError(f"expert module has no state_dict binding: {conceptual_name}")
        result.update(matched)
    if not result or not result.issubset(state_names):
        raise RuntimeError("expert state binding is not an exact state_dict subset")
    return result


def _make_noises(torch: Any, device: Any, np: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    blocks = []
    for seed in NOISE_SEEDS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        blocks.append(torch.randn((NOISE_BLOCK_SIZE, HORIZON, PADDED_ACTION_DIM), dtype=torch.float32, generator=generator))
    cpu_noise = torch.cat(blocks, dim=0).contiguous()
    if tuple(cpu_noise.shape) != (NOISE_COUNT, HORIZON, PADDED_ACTION_DIM) or cpu_noise.dtype != torch.float32:
        raise RuntimeError("noise shape/dtype contract failed")
    noise_np = cpu_noise.numpy().copy()
    return cpu_noise.to(device=device), noise_np, {
        "seeds": list(NOISE_SEEDS),
        "block_size": NOISE_BLOCK_SIZE,
        "count": NOISE_COUNT,
        "shape": list(noise_np.shape),
        "dtype": str(noise_np.dtype),
        "sha256": _sha256_bytes(noise_np.tobytes()),
    }


def _repeat_batch(batch: Mapping[str, Any], count: int, torch: Any) -> Dict[str, Any]:
    if count <= 0:
        raise RuntimeError("batch repeat count must be positive")
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != 1:
            raise RuntimeError(f"prepared batch cannot be repeated safely: {key}")
        result[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
    return result


def _predict_action(policy: Any, batch: Mapping[str, Any], noise: Any, torch: Any, label: str) -> Any:
    """One explicit-noise call; reset and clone are mandatory for every call."""
    with torch.no_grad():
        policy.reset()
        output = policy.predict_action_chunk(flow_clone_batch(batch, torch), noise=noise.clone())
    expected = (int(noise.shape[0]), HORIZON, PHYSICAL_ACTION_DIM)
    if tuple(output.shape) != expected:
        raise RuntimeError(f"{label} action shape must be {expected}, got {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all().item()):
        raise FloatingPointError(f"{label} action contains non-finite values")
    return output.detach()


def flow_clone_batch(value: Any, torch: Any) -> Any:
    """Local clone to keep the permitted Flow helper surface explicit."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: flow_clone_batch(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [flow_clone_batch(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(flow_clone_batch(item, torch) for item in value)
    if isinstance(value, str):
        return value
    raise RuntimeError(f"unsupported batch value: {type(value).__name__}")


def _batch_check(policy: Any, batch: Mapping[str, Any], noise: Any, torch: Any, np: Any, label: str) -> Dict[str, Any]:
    first = noise[:MICRO_BATCH]
    batch4 = _repeat_batch(batch, MICRO_BATCH, torch)
    batched = _predict_action(policy, batch4, first, torch, f"{label} batch4")
    individual = np.stack([
        _predict_action(policy, batch, first[index:index + 1], torch, f"{label} individual{index}").cpu().numpy()[0]
        for index in range(MICRO_BATCH)
    ], axis=0).astype(np.float32)
    batched_np = batched.cpu().numpy().astype(np.float32)
    max_abs = float(np.max(np.abs(batched_np.astype(np.float64) - individual.astype(np.float64))))
    return {"batched": batched_np, "individual": individual, "max_abs": max_abs, "pass": max_abs <= BATCH_TOLERANCE}


def _write_raw(output: Path, np: Any, actions: Any, noise: Any, completed: Any, episode_ids: Sequence[str], batch_batched: Any, batch_individual: Any) -> None:
    _atomic_npz(
        output / "raw_marginal.npz",
        np,
        schema=np.asarray(RAW_SCHEMA),
        arm_names=np.asarray(ARM_NAMES),
        actions=actions,
        noise=noise,
        completed=completed,
        episode_ids=np.asarray(episode_ids),
        batch_check_batched=batch_batched,
        batch_check_individual=batch_individual,
    )


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    import numpy as np

    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engineering: Dict[str, Any] = {
        "schema": "conditional-marginal-engineering-v1",
        "status": "running",
        "allocation": dict(allocation),
        "hostname": socket.gethostname(),
        "parameters": {
            "arms": list(ARM_NAMES),
            "conditions": CONDITION_COUNT,
            "draws_per_condition_per_arm": NOISE_COUNT,
            "noise_blocks": list(NOISE_SEEDS),
            "noise_block_size": NOISE_BLOCK_SIZE,
            "micro_batch": MICRO_BATCH,
            "num_steps": VELOCITY_STEPS,
            "horizon": HORIZON,
            "padded_action_dim": PADDED_ACTION_DIM,
            "physical_action_dim": PHYSICAL_ACTION_DIM,
            "bits": BITS,
            "qmax": QMAX,
            "dtype": "float32 runtime with FP32 fake-quantized dequantized weights",
            "science_gate": "deferred_to_root_cpu_verifier",
        },
        "raw_npz": str((output / "raw_marginal.npz").resolve()),
    }
    _atomic_json(output / "engineering.json", engineering)
    try:
        manifest_data = _load_input_manifest(args.input_manifest, args.asset_root)
        entries = manifest_data["entries"]
        episode_ids = [f"task{entry['task_index']}:episode{entry['dataset_episode_index']}:frame{entry['frame_index']}" for entry in entries]
        engineering["input_manifest"] = {
            "path": str(args.input_manifest.resolve()),
            "sha256": _sha256_file(args.input_manifest.resolve()),
            "schema": manifest_data["raw"].get("schema"),
            "sample_count": len(entries),
            "base_identity_path": manifest_data["identity_path"],
            "base_identity_sha256": manifest_data["identity_sha256"],
            "sample_sha256": [entry["sample_sha256"] for entry in entries],
            "actual_episode_ids": episode_ids,
            "selection": manifest_data["raw"].get("selection"),
        }
        engineering["base_identity"] = manifest_data["identity"]
        _atomic_json(output / "engineering.json", engineering)
        _check_deadline(started, args.max_seconds, "input manifest")

        flow = _load_flow(args.helper_path)
        args.manifest_data = {"identity": manifest_data["identity"]}
        import torch

        policy, torch, runtime_identity = flow._load_runtime(args, torch)
        gpu_identity = flow._gpu_evidence(torch)
        if runtime_identity.get("checkpoint", {}).get("sha256") is None:
            raise RuntimeError("checkpoint hash is unavailable")
        preprocessor, preprocessor_identity = flow._load_preprocessor(args.model_path.resolve(), args.vlm_path.resolve())
        if getattr(policy.config, "compile_model", True) or getattr(policy.config, "rtc_config", None) is not None:
            raise RuntimeError("compile_model/rtc_config must remain disabled")
        if bool(getattr(policy.config, "load_vlm_weights", True)):
            raise RuntimeError("load_vlm_weights must remain False")
        engineering.update({
            "flow_helper": {"path": str(args.helper_path.resolve()), "sha256": FLOW_SCREEN_SHA256},
            "runtime_identity": runtime_identity,
            "preprocessor_identity": preprocessor_identity,
            "gpu": gpu_identity,
            "runtime_identity_saved_before_inference": True,
        })
        _atomic_json(output / "runtime_identity.json", runtime_identity)
        _atomic_json(output / "engineering.json", engineering)
        _check_deadline(started, args.max_seconds, "runtime identity")

        batches: List[Dict[str, Any]] = []
        batch_auxiliary: List[Dict[str, Any]] = []
        for entry in entries:
            raw_batch = flow._load_raw_sample(Path(entry["sample_path"]), entry, torch)
            prepared, auxiliary = flow._prepare_batch(raw_batch, preprocessor, torch)
            batches.append(flow._move_to_device(prepared, torch.device("cuda:0"), torch))
            batch_auxiliary.append(auxiliary)
        engineering["prepared_batches"] = {
            "count": len(batches),
            "keys": [sorted(str(key) for key in batch) for batch in batches],
            "shapes": [{str(key): list(value.shape) for key, value in batch.items()} for batch in batches],
            "auxiliary": batch_auxiliary,
            "format": "official_saved_policy_preprocessor_v1",
        }
        noise_device, noise_np, noise_identity = _make_noises(torch, torch.device("cuda:0"), np)
        engineering["noise"] = noise_identity
        _atomic_json(output / "engineering.json", engineering)
        actions = np.full((CONDITION_COUNT, len(ARM_NAMES), NOISE_COUNT, HORIZON, PHYSICAL_ACTION_DIM), np.nan, dtype=np.float32)
        completed = np.zeros((CONDITION_COUNT, len(ARM_NAMES)), dtype=np.bool_)
        batch_batched = np.full((len(ARM_NAMES), MICRO_BATCH, HORIZON, PHYSICAL_ACTION_DIM), np.nan, dtype=np.float32)
        batch_individual = np.full_like(batch_batched, np.nan)
        _write_raw(output, np, actions, noise_np, completed, episode_ids, batch_batched, batch_individual)
        _check_deadline(started, args.max_seconds, "prepared inputs")

        modules = flow._eligible_modules(policy, torch)
        expert_modules = modules.get("expert_W4", [])
        if not expert_modules:
            raise RuntimeError("expert W4 allowlist is empty")
        expert_state_names = _expert_state_names(policy, expert_modules)
        snapshot = flow._snapshot_weights(modules, torch)
        bypass_before = flow._state_subset_digest(policy, expert_state_names, torch)
        engineering["quantizer"] = {
            "scope": "expert transformer Linear modules only",
            "module_count": len(expert_modules),
            "expert_state_names": sorted(expert_state_names),
            "bits": BITS,
            "qmin": -QMAX,
            "qmax": QMAX,
            "scheme": "symmetric_per_output_channel_RTN_dequantized_FP32",
            "bypassed_state_digest_before": bypass_before,
        }
        _atomic_json(output / "engineering.json", engineering)

        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        batch_checks: Dict[str, Any] = {}
        for arm_index, arm in enumerate(ARM_NAMES):
            flow._restore_weights(snapshot, modules, torch)
            quantizer_record = None
            if arm == "expert_W4":
                quantizer_record = flow._quantize_locus(expert_modules, torch)
            arm_bypass_before = flow._state_subset_digest(policy, expert_state_names, torch)
            if arm_bypass_before != bypass_before:
                raise RuntimeError(f"bypassed state changed before arm {arm}")
            engineering.setdefault("arms", {})[arm] = {
                "restore_before_exact": True,
                "quantizer_record": quantizer_record,
                "bypassed_state_digest_before": arm_bypass_before,
            }
            if arm == "FP32" or arm == "expert_W4":
                check = _batch_check(policy, batches[0], noise_device, torch, np, arm)
                batch_batched[arm_index] = check["batched"]
                batch_individual[arm_index] = check["individual"]
                batch_checks[arm] = {"max_abs": check["max_abs"], "pass": check["pass"], "draw_indices": [0, 1, 2, 3]}
                engineering["arms"][arm]["batch_check"] = batch_checks[arm]
                _write_raw(output, np, actions, noise_np, completed, episode_ids, batch_batched, batch_individual)
                _atomic_json(output / "engineering.json", engineering)
                if not check["pass"]:
                    raise BatchImplementationInconclusive(f"{arm} batch4-vs-individual max_abs={check['max_abs']:.9g} > {BATCH_TOLERANCE:g}")

            arm_bypass_after_check = flow._state_subset_digest(policy, expert_state_names, torch)
            if arm_bypass_after_check != bypass_before:
                raise RuntimeError(f"bypassed state changed after batch check for arm {arm}")
            for condition_index, (entry, batch) in enumerate(zip(entries, batches)):
                for start_index in range(0, NOISE_COUNT, MICRO_BATCH):
                    stop_index = start_index + MICRO_BATCH
                    noise_batch = noise_device[start_index:stop_index].clone()
                    output_batch = _predict_action(policy, _repeat_batch(batch, MICRO_BATCH, torch), noise_batch, torch, f"{arm} condition{condition_index} draws{start_index}:{stop_index}")
                    actions[condition_index, arm_index, start_index:stop_index] = output_batch.cpu().numpy().astype(np.float32)
                    _check_deadline(started, args.max_seconds, f"{arm} condition {condition_index} draws {start_index}")
                if not np.isfinite(actions[condition_index, arm_index]).all():
                    raise FloatingPointError(f"non-finite actions at arm={arm}, condition={condition_index}")
                completed[condition_index, arm_index] = True
                _write_raw(output, np, actions, noise_np, completed, episode_ids, batch_batched, batch_individual)
                engineering["completed"] = completed.tolist()
                engineering["arms"][arm]["last_completed_condition"] = condition_index
                _atomic_json(output / "engineering.json", engineering)
            arm_bypass_after = flow._state_subset_digest(policy, expert_state_names, torch)
            if arm_bypass_after != bypass_before:
                raise RuntimeError(f"bypassed state changed after arm {arm}")
            engineering["arms"][arm]["bypassed_state_digest_after"] = arm_bypass_after
            engineering["arms"][arm]["all_conditions_complete"] = True
            _atomic_json(output / "engineering.json", engineering)

        flow._restore_weights(snapshot, modules, torch)
        final_bypass = flow._state_subset_digest(policy, expert_state_names, torch)
        if final_bypass != bypass_before:
            raise RuntimeError("bypassed state changed after final exact restore")
        if not bool(completed.all()) or not np.isfinite(actions).all():
            raise RuntimeError("raw action collection is incomplete or non-finite")
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        peak_memory = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        engineering.update({
            "status": "complete",
            "completed": completed.tolist(),
            "batch_checks": batch_checks,
            "weight_restore": {"method": "flow._restore_weights", "exact_elementwise": True, "bypassed_state_digest_final": final_bypass},
            "torch_peak_memory": peak_memory,
            "elapsed_seconds": time.monotonic() - started,
            "science_gate_deferred_to_root_cpu_verifier": True,
        })
        _write_raw(output, np, actions, noise_np, completed, episode_ids, batch_batched, batch_individual)
        _atomic_json(output / "engineering.json", engineering)
        summary = {
            "schema": "conditional-marginal-screen-v1",
            "status": "complete",
            "allocation": dict(allocation),
            "input_manifest": engineering["input_manifest"],
            "source_identity": runtime_identity.get("source_identity"),
            "flow_helper": engineering["flow_helper"],
            "runtime_identity_path": str((output / "runtime_identity.json").resolve()),
            "checkpoint": runtime_identity.get("checkpoint"),
            "preprocessor": preprocessor_identity,
            "gpu": gpu_identity,
            "parameters": engineering["parameters"],
            "noise": noise_identity,
            "actions_shape": list(actions.shape),
            "completed": completed.tolist(),
            "batch_checks": batch_checks,
            "torch_peak_memory": peak_memory,
            "raw_npz": str((output / "raw_marginal.npz").resolve()),
            "engineering_json": str((output / "engineering.json").resolve()),
            "science_gate_deferred_to_root_cpu_verifier": True,
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_small_summary(output / "summary.json", summary)
        print(json.dumps({"status": "complete", "raw": str(output / "raw_marginal.npz"), "completed": int(completed.sum())}, indent=2), flush=True)
    except Exception:
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", "--manifest", dest="input_manifest", type=Path, default=TOP_DEFAULT / "conditional_marginal" / "manifest.json")
    parser.add_argument("--asset-root", type=Path, default=TOP_DEFAULT)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--helper-path", type=Path, required=True)
    parser.add_argument("--lerobot-source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_WORKLOAD_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_WORKLOAD_SECONDS:g}]")
    return args


def main() -> None:
    # Deliberately the first workload action.  The shell also sources the
    # guard before any snapshot, hash, model or sample operation.
    require_allocation = _load_guard()
    allocation = require_allocation()
    args = _parse_args()
    output = args.output.resolve()
    try:
        _run(args, allocation)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        engineering_path = output / "engineering.json"
        if engineering_path.is_file():
            try:
                engineering = dict(_load_json(engineering_path, "engineering"))
                engineering.update({
                    "status": "implementation_inconclusive" if isinstance(exc, BatchImplementationInconclusive) else "implementation_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": engineering.get("elapsed_seconds"),
                })
                _atomic_json(engineering_path, engineering)
            except Exception:
                pass
        status = "implementation_inconclusive" if isinstance(exc, BatchImplementationInconclusive) else "implementation_failure"
        try:
            _write_small_summary(output / "summary.json", {
                "schema": "conditional-marginal-screen-v1",
                "status": status,
                "allocation": dict(allocation),
                "error": f"{type(exc).__name__}: {exc}",
                "raw_npz": str((output / "raw_marginal.npz").resolve()),
                "engineering_json": str((output / "engineering.json").resolve()),
                "science_gate_deferred_to_root_cpu_verifier": True,
            })
        finally:
            raise


if __name__ == "__main__":
    main()

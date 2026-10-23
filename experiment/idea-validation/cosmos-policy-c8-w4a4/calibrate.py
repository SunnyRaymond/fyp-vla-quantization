"""Independent, resumable single-GPU C8 calibration for SpinQuant or micro-VQ.

This entry point is intentionally stage-exclusive.  A SpinQuant job writes only
``results/spinquant`` and a VQ job writes only ``results/vq``; both read the
same frozen C8 observations and reconstruct their own BF16 teacher caches.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import socket
import time
from pathlib import Path
from typing import Any


SEED = 20261015
SAVE_RESERVE_DEFAULT = 120


def _torch():
    import torch

    return torch


def _atomic_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload):
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cpu_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().to("cpu", memory_format=torch.preserve_format).clone(
            memory_format=torch.preserve_format
        )
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch) for item in value)
    return value


def _optimizer_to_device(optimizer, device, torch):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _rng_state(torch):
    return {
        "python": random.getstate(), "numpy": __import__("numpy").random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state, torch):
    random.setstate(state["python"])
    __import__("numpy").random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _run_endpoint(runtime, context, protocol, **kwargs):
    import endpoint

    torch = _torch()
    caller_rng = _rng_state(torch)
    try:
        _restore_rng(context["pre_forward_rng_state"], torch)
        return endpoint.run_endpoint(runtime.model, context["data_batch"], context["seed"],
                                     protocol, **kwargs)
    finally:
        _restore_rng(caller_rng, torch)


def _elapsed(start_epoch: float) -> float:
    return max(0.0, time.time() - float(start_epoch))


def _reserve_seconds() -> int:
    return max(30, int(os.environ.get("COSMOS_SAVE_RESERVE_SECONDS", SAVE_RESERVE_DEFAULT)))


def _has_time(start_epoch: float, soft_deadline: int, estimate_seconds: int = 0) -> bool:
    return _elapsed(start_epoch) + _reserve_seconds() + max(0, estimate_seconds) < soft_deadline


def _load_tensor_file(path: Path, torch):
    return torch.load(path, map_location="cpu", weights_only=False)


def _saved_completed_updates(out: Path, expected: int, arm: str, torch) -> int:
    """Preserve the exact update count in status summaries before training resumes."""
    path = out / "checkpoint.pt"
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != arm or checkpoint.get("expected_updates") != expected:
        raise RuntimeError("Existing calibration checkpoint does not match this frozen arm recipe")
    completed = int(checkpoint.get("completed_updates", -1))
    if not 0 <= completed <= expected:
        raise RuntimeError("Existing calibration checkpoint has an invalid completed-update count")
    return completed


def _parity_cache_is_current(parity: dict[str, Any] | None, version: int) -> bool:
    if (not isinstance(parity, dict) or parity.get("status") != "passed"
            or parity.get("endpoint_version") != version):
        return False
    checks = parity.get("checks")
    if not isinstance(checks, dict):
        return False
    inference, differentiable = checks.get("inference"), checks.get("differentiable")
    if (not isinstance(inference, dict) or inference.get("status") != "passed"
            or not isinstance(differentiable, dict) or differentiable.get("status") != "passed"):
        return False
    gradient = differentiable.get("gradient", {})
    return all(gradient.get(key) for key in (
        "generated_latent_requires_grad", "generated_latent_grad_fn",
        "action_requires_grad", "action_grad_fn",
    ))


def _tensor_parity_metrics(candidate, reference, torch, *, rtol=2e-3, atol=2e-3):
    if candidate is None or reference is None:
        return {"available": False, "passed": False}
    candidate_dtype, reference_dtype = str(candidate.dtype), str(reference.dtype)
    candidate = candidate.detach().to(device="cpu", dtype=torch.float32)
    reference = reference.detach().to(device="cpu", dtype=torch.float32)
    metrics = {
        "available": True, "candidate_dtype": candidate_dtype,
        "reference_dtype": reference_dtype,
        "candidate_shape": list(candidate.shape), "reference_shape": list(reference.shape),
        "passed": False,
    }
    if candidate.shape != reference.shape:
        return metrics
    delta = (candidate - reference).abs()
    metrics.update({
        "max_abs": float(delta.max()),
        "rmse": float(delta.square().mean().sqrt()),
        "passed": bool(torch.allclose(candidate, reference, rtol=rtol, atol=atol)),
    })
    return metrics


def _endpoint_parity_check(result, official_generated, official_original, official_actions,
                           torch, *, require_grad):
    generated = result.generated_latent
    action = result.action
    generated_metrics = _tensor_parity_metrics(generated, official_generated, torch)
    if official_actions.ndim == 2:
        official_actions = official_actions.unsqueeze(0)
    action_metrics = _tensor_parity_metrics(action, official_actions, torch)
    original_metrics = _tensor_parity_metrics(
        result.orig_clean_latent_frames, official_original, torch
    )
    slot_metrics = {}
    if (generated.ndim == 5 and official_generated is not None
            and official_generated.ndim == 5 and generated.shape == official_generated.shape):
        for slot in range(generated.shape[2]):
            slot_metrics[str(slot)] = _tensor_parity_metrics(
                generated[:, :, slot], official_generated[:, :, slot], torch
            )
    gradient = {
        "generated_latent_requires_grad": bool(generated.requires_grad),
        "generated_latent_grad_fn": (type(generated.grad_fn).__name__ if generated.grad_fn else None),
        "action_requires_grad": bool(action.requires_grad),
        "action_grad_fn": (type(action.grad_fn).__name__ if action.grad_fn else None),
    }
    gradient_ok = (not require_grad or all(gradient[key] for key in (
        "generated_latent_requires_grad", "generated_latent_grad_fn",
        "action_requires_grad", "action_grad_fn",
    )))
    passed = generated_metrics.get("passed", False) and action_metrics.get("passed", False) and gradient_ok
    return {
        "status": "passed" if passed else "failed",
        "generated_latent": generated_metrics, "action": action_metrics,
        "orig_clean_latent_frames": original_metrics,
        "generated_latent_slot_errors": slot_metrics,
        "input_video_layout": getattr(result, "input_video_layout", {}),
        "sampler_evaluations": int(result.sampler_evaluations),
        "denoiser_calls": int(result.denoiser_calls), "gradient": gradient,
        "gradient_flow_passed": bool(gradient_ok),
    }


def _input_capture_metrics(pre_batch, official_batch, torch):
    if not isinstance(pre_batch, dict) or not isinstance(official_batch, dict):
        return {"available": False}
    before, after = pre_batch.get("video"), official_batch.get("video")
    if not torch.is_tensor(before) or not torch.is_tensor(after):
        return {"available": False}
    metrics = {
        "available": True,
        "pre_sampler_dtype": str(before.dtype), "official_captured_dtype": str(after.dtype),
        "pre_sampler_shape": list(before.shape), "official_captured_shape": list(after.shape),
        "pre_sampler_stride": list(before.stride()),
        "official_captured_stride": list(after.stride()),
        "pre_sampler_is_contiguous": bool(before.is_contiguous()),
        "official_captured_is_contiguous": bool(after.is_contiguous()),
        "pre_sampler_channels_last_3d": bool(before.ndim == 5 and before.is_contiguous(
            memory_format=torch.channels_last_3d
        )),
        "official_captured_channels_last_3d": bool(after.ndim == 5 and after.is_contiguous(
            memory_format=torch.channels_last_3d
        )),
        "pre_sampler_preprocessed_marker": bool(pre_batch.get("is_preprocessed", False)),
        "official_preprocessed_marker": bool(official_batch.get("is_preprocessed", False)),
    }
    if before.shape != after.shape:
        metrics["expected_preprocessing_matches"] = False
        return metrics
    # Official get_action captures its batch after generate_samples_from_batch has
    # converted uint8 video to the model dtype and normalized it in-place.
    if before.dtype == torch.uint8:
        expected = before.detach().cpu().to(dtype=after.dtype) / 127.5 - 1.0
        actual = after.detach().cpu()
        error = (expected.float() - actual.float()).abs()
        metrics.update({
            "expected_preprocessing_max_abs": float(error.max()),
            "expected_preprocessing_rmse": float(error.square().mean().sqrt()),
            "expected_preprocessing_matches": bool(torch.allclose(expected.float(), actual.float(),
                                                                    rtol=0.0, atol=2e-3)),
        })
    return metrics


def _enable_target_weight_grads(modules):
    states, seen = [], set()
    for module in modules.values():
        weight = getattr(module, "weight", None)
        if weight is not None and hasattr(weight, "requires_grad_") and id(weight) not in seen:
            seen.add(id(weight))
            states.append((weight, bool(weight.requires_grad)))
            weight.requires_grad_(True)
    return states


def _restore_parameter_grad_states(states):
    for parameter, requires_grad in states:
        parameter.requires_grad_(requires_grad)


def _calibration_contexts(runtime, protocol, out: Path, start_epoch: float, soft_deadline: int,
                          modules, need_packets: bool):
    """Cache frozen C8 batches and BF16 endpoint/denoiser references by case."""
    import endpoint

    torch = _torch()
    cache = out / "calibration_cache"
    cache.mkdir(parents=True, exist_ok=True)
    contexts_path = cache / "contexts.pt"
    targets_path = cache / "teacher_targets.pt"
    packets_path = cache / "teacher_packets.pt"
    if contexts_path.exists():
        contexts = _load_tensor_file(contexts_path, torch)
    else:
        contexts = {}

    parity_path = cache / "endpoint_parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.exists() else None
    case_ids = [int(case_id) for case_id in protocol["calibration"]["case_ids"]]
    seed_index = int(protocol["calibration"]["seed_index"])
    for position, case_id in enumerate(case_ids):
        verify_parity = position == 0 and not _parity_cache_is_current(parity, endpoint.PARITY_VERSION)
        cached_context = contexts.get(case_id)
        if (cached_context is not None
                and cached_context.get("endpoint_version") == endpoint.PARITY_VERSION
                and "pre_forward_rng_state" in cached_context and not verify_parity):
            continue
        if not _has_time(start_epoch, soft_deadline, estimate_seconds=120):
            expected = 200 if need_packets else 16
            arm = "spinquant_w4a4" if need_packets else "action_latent_micro_vq_a4"
            completed = _saved_completed_updates(out, expected, arm, torch)
            _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected,
                                     "c8_contexts", elapsed=_elapsed(start_epoch))
            return None
        latent_indices = None
        pre_forward_rng_state = None
        if verify_parity:
            official = runtime.c8_query(case_id, seed=None, capture=True)
            # get_action normalizes data_batch['video'] in-place. Rebuild the
            # same pre-sampler batch so the endpoint applies official prep once.
            batch, _, _ = runtime.c8_data_batch(case_id)
            pre_forward_rng_state = _rng_state(torch)
            row = official.get("plan_row", {})
            metadata = official.get("metadata", {})
            official_generated = official.get("generated_latent")
            latent_indices = official.get("latent_indices")
            if official_generated is None:
                captured = official.get("captured", {})
                official_generated = captured.get("generated_latent")
            if latent_indices is None:
                latent_indices = official.get("captured", {}).get("latent_indices")
            seed = (int(official["seed"]) if official.get("seed") is not None
                    else int(row["sampler_seeds"][seed_index]))
        else:
            result = runtime.c8_data_batch(case_id)
            pre_forward_rng_state = _rng_state(torch)
            if isinstance(result, tuple):
                batch = result[0]
                metadata = result[1] if len(result) > 1 else {}
                row = result[2] if len(result) > 2 else {}
                latent_indices = result[3] if len(result) > 3 else None
            elif isinstance(result, dict) and "data_batch" in result:
                batch, metadata, row = (result["data_batch"], result.get("metadata", {}),
                                        result.get("plan_row", {}))
                latent_indices = result.get("latent_indices")
                if latent_indices is None:
                    latent_indices = result.get("captured", {}).get("latent_indices")
            else:
                batch, metadata, row = result, {}, {}
                latent_indices = None
            if latent_indices is None and isinstance(batch, dict):
                latent_indices = batch.get("latent_indices")
            seeds = row.get("sampler_seeds", [])
            if not seeds:
                _, _, row = runtime.c8_raw_observation(case_id)
                seeds = row.get("sampler_seeds", [])
            if len(seeds) <= seed_index:
                raise RuntimeError(f"Frozen C8 case {case_id} has no seed index {seed_index}")
            seed = int(seeds[seed_index])
            official_generated = None

        with torch.inference_mode(False):
            cpu_batch = _cpu_tree(batch, torch)
            cpu_latent_indices = _cpu_tree(latent_indices, torch) if latent_indices is not None else None
        contexts[case_id] = {"data_batch": cpu_batch, "metadata": metadata, "plan_row": row,
                             "seed": int(seed), "latent_indices": cpu_latent_indices,
                             "pre_forward_rng_state": pre_forward_rng_state,
                             "endpoint_version": endpoint.PARITY_VERSION}
        _atomic_torch(contexts_path, contexts)

        if verify_parity:
            if official_generated is None:
                raise RuntimeError("Pinned official C8 query did not expose generated_latent for sampler parity")
            official_actions = torch.as_tensor(official["actions"], dtype=torch.float32)
            captured = official.get("captured", {})
            official_original = official.get("orig_clean_latent_frames")
            if official_original is None:
                official_original = captured.get("orig_clean_latent_frames")
            official_batch = official.get("data_batch", captured.get("data_batch"))
            parity = {
                "status": "pending", "case_id": case_id,
                "endpoint_version": endpoint.PARITY_VERSION,
                "pbs_job_id": os.environ.get("PBS_JOBID"),
                "seed": int(seed), "official_sampler_calls": int(protocol["inference"]["denoising_steps"]),
                "rtol": 2e-3, "atol": 2e-3,
                "checks": {"inference": {"status": "pending"},
                           "differentiable": {"status": "pending"}},
                "input_capture": _input_capture_metrics(batch, official_batch, torch),
                "execution_scope": "official_inference_and_checkpointed_differentiable_BF16_forward_parity",
            }
            _atomic_json(parity_path, parity)
            try:
                endpoint_result = _run_endpoint(
                    runtime, contexts[case_id], protocol,
                    checkpoint_denoiser=False, collect_packets=False, grad_enabled=False,
                    dataset_stats=runtime.dataset_stats,
                    latent_indices=cpu_latent_indices,
                )
                parity["checks"]["inference"] = _endpoint_parity_check(
                    endpoint_result, official_generated, official_original,
                    official_actions, torch, require_grad=False,
                )
            except Exception as exc:
                parity["checks"]["inference"] = {
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}"
                }
                parity["status"] = "failed"
                _atomic_json(parity_path, parity)
                raise RuntimeError("Official-inference BF16 endpoint parity could not be verified") from exc
            finally:
                if "endpoint_result" in locals():
                    del endpoint_result
            _atomic_json(parity_path, parity)

            grad_states, endpoint_result = [], None
            try:
                grad_states = _enable_target_weight_grads(modules)
                endpoint_result = _run_endpoint(
                    runtime, contexts[case_id], protocol,
                    checkpoint_denoiser=True, collect_packets=False, grad_enabled=True,
                    dataset_stats=runtime.dataset_stats,
                    latent_indices=cpu_latent_indices,
                )
                parity["checks"]["differentiable"] = _endpoint_parity_check(
                    endpoint_result, official_generated, official_original,
                    official_actions, torch, require_grad=True,
                )
                parity["checks"]["differentiable"]["enabled_target_parameter_count"] = len(grad_states)
            except Exception as exc:
                parity["checks"]["differentiable"] = {
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "enabled_target_parameter_count": len(grad_states),
                }
            finally:
                _restore_parameter_grad_states(grad_states)
                if endpoint_result is not None:
                    del endpoint_result

            parity["status"] = (
                "passed" if all(parity["checks"][path]["status"] == "passed"
                                for path in ("inference", "differentiable")) else "failed"
            )
            _atomic_json(parity_path, parity)
            if parity["status"] != "passed":
                raise RuntimeError(
                    "Five-step BF16 parity gate failed: inference and checkpointed differentiable paths "
                    "must both match the pinned official oracle at rtol=atol=2e-3"
                )

    activation_path = cache / "activation_absmax.pt"
    partial_activation_path = cache / "activation_absmax_partial.pt"
    needs_activation_capture = not need_packets and not activation_path.exists()
    if (not targets_path.exists() or (need_packets and not packets_path.exists())
            or needs_activation_capture):
        targets, packet_groups, raw_act_max = {}, {}, {}
        if partial_activation_path.exists():
            raw_act_max = _load_tensor_file(partial_activation_path, torch)
        else:
            for name, module in modules.items():
                raw_act_max[name] = torch.zeros(module.in_features, dtype=torch.float32)
        handles = []
        capture_activations = (not need_packets and not activation_path.exists())
        if capture_activations:
            def make_hook(name, module):
                def hook(_module, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return
                    value = inputs[0].detach().float().reshape(-1, module.in_features)
                    if value.numel():
                        raw_act_max[name] = torch.maximum(raw_act_max[name], value.abs().amax(0).cpu())
                return hook
            for name, module in modules.items():
                handles.append(module.register_forward_pre_hook(make_hook(name, module)))
        try:
            for position, case_id in enumerate(case_ids):
                target_exists = targets_path.exists() and case_id in _load_tensor_file(targets_path, torch)
                packet_exists = (not need_packets or (packets_path.exists()
                                and case_id in _load_tensor_file(packets_path, torch)))
                if target_exists and packet_exists and not capture_activations:
                    continue
                if not _has_time(start_epoch, soft_deadline, estimate_seconds=180):
                    expected = 200 if need_packets else 16
                    arm = "spinquant_w4a4" if need_packets else "action_latent_micro_vq_a4"
                    completed = _saved_completed_updates(out, expected, arm, torch)
                    _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected,
                                             "bf16_teacher_endpoints", elapsed=_elapsed(start_epoch))
                    return None
                context = contexts[case_id]
                endpoint_result = _run_endpoint(
                    runtime, context, protocol,
                    checkpoint_denoiser=False, collect_packets=need_packets, grad_enabled=False,
                    dataset_stats=runtime.dataset_stats,
                    latent_indices=context.get("latent_indices"),
                )
                targets[case_id] = {
                    "action": endpoint_result.action.detach().cpu(),
                    "future_image_slots": {key: value.detach().cpu()
                                           for key, value in endpoint_result.future_image_slots.items()},
                    "seed": int(context["seed"]),
                }
                if need_packets:
                    packet_groups[case_id] = endpoint_result.packets
                if targets:
                    previous = (_load_tensor_file(targets_path, torch) if targets_path.exists() else {})
                    previous.update(targets)
                    _atomic_torch(targets_path, previous)
                    targets = {}
                if need_packets and packet_groups:
                    previous = (_load_tensor_file(packets_path, torch) if packets_path.exists() else {})
                    previous.update(packet_groups)
                    _atomic_torch(packets_path, previous)
                    packet_groups = {}
                if capture_activations:
                    _atomic_torch(partial_activation_path, raw_act_max)
                print(f"BF16_ENDPOINT case={case_id} evaluations={endpoint_result.sampler_evaluations} "
                      f"denoiser_calls={endpoint_result.denoiser_calls}", flush=True)
        finally:
            for handle in handles:
                handle.remove()

        if not targets_path.exists() or (need_packets and not packets_path.exists()):
            return None
        if capture_activations and not activation_path.exists():
            partial = _load_tensor_file(partial_activation_path, torch)
            if any(not bool(torch.isfinite(value).all()) for value in partial.values()):
                raise RuntimeError("Nonfinite BF16 activation absmax was captured")
            _atomic_torch(activation_path, partial)
    targets = _load_tensor_file(targets_path, torch)
    packets = _load_tensor_file(packets_path, torch) if need_packets else {}
    act_max = (_load_tensor_file(activation_path, torch) if activation_path.exists() else {})
    if need_packets and set(packets) != set(case_ids):
        raise RuntimeError("SpinQuant teacher packet cache is missing one or more frozen C8 contexts")
    if set(targets) != set(case_ids):
        raise RuntimeError("BF16 endpoint teacher cache is missing one or more frozen C8 contexts")
    return contexts, targets, packets, act_max, parity


def _write_checkpoint_summary(out: Path, protocol, status: str, completed: int, expected: int,
                              phase: str, *, elapsed: float, size_report=None, parity=None,
                              message=None, checkpoint_strategy=None, gpu_memory=None):
    payload = {
        "protocol": protocol["protocol"], "status": status,
        "arm": "spinquant_w4a4" if expected == 200 else "action_latent_micro_vq_a4",
        "phase": phase, "expected_updates": expected, "completed_updates": completed,
        "training_updates": {"expected": expected, "completed": completed},
        "elapsed_seconds_from_allocation_start": float(elapsed),
        "soft_deadline_seconds": int(os.environ.get("COSMOS_SOFT_DEADLINE_SECONDS", "6600")),
        "save_reserve_seconds": _reserve_seconds(),
        "size_report": size_report, "parity": parity,
    }
    if message:
        payload["message"] = str(message)
    if checkpoint_strategy is not None:
        payload["checkpoint_strategy"] = checkpoint_strategy
    if gpu_memory is not None:
        payload["gpu_memory"] = gpu_memory
    _atomic_json(out / "summary.json", payload)


def _module_size_report(modules, model):
    import quantization

    return quantization.effective_bpw(modules, model)


def _spin_checkpoint_strategy(model):
    blocks = list(getattr(getattr(model, "net", None), "blocks", ()))
    wrapped = sum(hasattr(block, "_checkpoint_wrapped_module") for block in blocks)
    if not blocks or wrapped != len(blocks):
        raise RuntimeError(
            "SpinQuant requires the pinned Cosmos per-block selective checkpoint wrappers; "
            f"found {wrapped}/{len(blocks)} wrapped blocks"
        )
    context_labels = []
    for block in blocks:
        checkpoint_fn = getattr(block, "checkpoint_fn", None)
        keywords = getattr(checkpoint_fn, "keywords", None) or {}
        context_fn = keywords.get("context_fn")
        while hasattr(context_fn, "func") and hasattr(context_fn, "keywords"):
            context_fn = context_fn.func
        if context_fn is None:
            context_labels.append("unavailable")
        else:
            module = getattr(context_fn, "__module__", "")
            name = getattr(context_fn, "__qualname__", getattr(context_fn, "__name__", type(context_fn).__name__))
            context_labels.append(f"{module}.{name}" if module else name)
    unique_contexts = sorted(set(context_labels))
    if len(unique_contexts) != 1 or unique_contexts[0] == "unavailable":
        raise RuntimeError(f"Cosmos block checkpoint contexts differ or are not inspectable: {unique_contexts}")
    torch = _torch()
    try:
        autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
        autocast_dtype = str(torch.get_autocast_dtype("cuda"))
    except (AttributeError, TypeError):
        autocast_enabled = bool(torch.is_autocast_enabled())
        autocast_dtype = str(torch.get_autocast_gpu_dtype())
    return {
        "native_policy_context_fn": unique_contexts[0],
        "native_policy_context_source": "runtime block checkpoint_fn.context_fn introspection",
        "native_checkpoint_scope": "per-transformer-block selective activation checkpoint",
        "native_block_count": len(blocks),
        "native_wrapped_block_count": wrapped,
        "rotation_per_linear_checkpoint": False,
        "rotation_matrix_prewarm": "outside SAC before training forward and after each optimizer refresh",
        "rotation_matrix_prewarm_autocast_enabled": autocast_enabled,
        "rotation_matrix_prewarm_autocast_dtype": autocast_dtype,
        "reason": "avoid nested checkpoints and lazy Cayley mm construction changing native SAC replay order",
    }


def _spin_gpu_memory(torch, device):
    if getattr(device, "type", None) != "cuda":
        return {"available": False, "scope": "spin_update_phase"}
    gib = float(1024 ** 3)
    return {
        "available": True,
        "scope": "spin_update_phase_since_peak_reset",
        "current_allocated_gib": float(torch.cuda.memory_allocated(device) / gib),
        "current_reserved_gib": float(torch.cuda.memory_reserved(device) / gib),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / gib),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / gib),
    }


def _spin_calibrate(runtime, protocol, root, out, start_epoch, soft_deadline,
                    modules, target_data, packet_data, size_report, parity):
    import endpoint
    import quantization

    torch = _torch()
    rotation, _, _, _, _ = quantization._helpers()
    case_ids = [int(value) for value in protocol["calibration"]["case_ids"]]
    # Canonical 8 contexts x 5 solver evaluations; multiple CFG denoiser calls
    # at a solver evaluation are averaged as one packet group.
    groups = []
    for case_id in case_ids:
        packets = packet_data[case_id]
        by_step = {step: [] for step in range(5)}
        for packet in packets:
            step = int(packet["sampler_evaluation"])
            if step not in by_step:
                raise RuntimeError(f"SpinQuant packet has out-of-range sampler evaluation {step}")
            by_step[step].append(packet)
        if any(not by_step[step] for step in range(5)):
            raise RuntimeError(f"SpinQuant C8 case {case_id} has an empty teacher step packet")
        groups.extend({"case_id": case_id, "step": step, "packets": by_step[step]}
                      for step in range(5))
    expected, lr = 200, 1e-3
    checkpoint_path = out / "checkpoint.pt"
    checkpoint_strategy = _spin_checkpoint_strategy(runtime.model)
    state, handle, aliases = quantization.build_spin_training(modules, seed=SEED)
    optimizer = rotation.build_rotation_optimizer(state, lr=lr)
    device = next(iter(modules.values())).weight.device
    completed = 0
    if checkpoint_path.exists():
        state_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state_payload.get("arm") != quantization.SPIN_ARM or state_payload.get("expected_updates") != expected:
            raise RuntimeError("SpinQuant resume checkpoint does not match the frozen recipe")
        state.load_state_dict({key: value.to(device) for key, value in state_payload["rotation_state"].items()})
        optimizer.load_state_dict(state_payload["optimizer_state"])
        _optimizer_to_device(optimizer, device, torch)
        completed = int(state_payload["completed_updates"])
        _restore_rng(state_payload["rng_state"], torch)
        # Loading replaces rotation parameters; discard the matrix built during
        # construction and rebuild its graph before the next checkpointed pass.
        state.refresh()
        quantization.prewarm_spin_rotations(state)
    if getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        while completed < expected:
            if not _has_time(start_epoch, soft_deadline, estimate_seconds=10):
                _spin_checkpoint(out, state, optimizer, completed, expected, torch, protocol,
                                 size_report, parity, start_epoch, "spin_updates",
                                 checkpoint_strategy=checkpoint_strategy,
                                 gpu_memory=_spin_gpu_memory(torch, device))
                return "checkpointed"
            group = groups[completed % len(groups)]
            optimizer.zero_grad(set_to_none=True)
            losses = [endpoint.denoiser_nmse(runtime.model, packet, device)
                      for packet in group["packets"]]
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"SpinQuant produced nonfinite packet loss at update {completed + 1}")
            loss.backward()
            parameters = [parameter for parameter in state.parameters() if parameter.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            handle.optimizer_step(optimizer)
            # optimizer_step refreshes RotationState's lazy matrix cache. Build
            # the next Cayley matrix outside the native block SAC region.
            quantization.prewarm_spin_rotations(state)
            completed += 1
            _spin_checkpoint(out, state, optimizer, completed, expected, torch, protocol,
                             size_report, parity, start_epoch, "spin_updates",
                             last_loss=float(loss.detach()), last_grad_norm=float(norm),
                             checkpoint_strategy=checkpoint_strategy,
                             gpu_memory=_spin_gpu_memory(torch, device))
            if completed % 10 == 0 or completed == expected:
                print(f"SPIN_UPDATE {completed}/{expected} loss={float(loss.detach()):.7g} "
                      f"grad_norm={float(norm):.6g} elapsed={_elapsed(start_epoch):.1f}s", flush=True)

        if not _has_time(start_epoch, soft_deadline, estimate_seconds=60):
            return "checkpointed"
        metadata = quantization.save_spin_deployment(
            out, modules, state, aliases, size_report, seed=SEED,
            metadata={"training_objective": "balanced_full_joint_denoiser_output_nmse",
                      "teacher_source": "independent_BF16_8context_x_5sampler_evaluations",
                      "expected_updates": expected, "completed_updates": completed,
                      "learning_rate": lr, "gradient_clip_norm": 1.0,
                      "checkpoint_selection": "fixed_final_update_200",
                      "checkpoint_strategy": checkpoint_strategy,
                      "gpu_memory": _spin_gpu_memory(torch, device),
                      "parity_status": parity["status"]},
        )
        metadata["bpw_match_within_5_percent"] = size_report["same_module_within_5_percent"]
        summary = {
            "protocol": protocol["protocol"], "status": "complete", "arm": quantization.SPIN_ARM,
            "phase": "complete", "expected_updates": expected, "completed_updates": completed,
            "training_updates": {"expected": expected, "completed": completed},
            "size_report": size_report, "effective_bpw": metadata["same_module_effective_bpw"],
            "whole_dit_effective_bpw": metadata["whole_dit_effective_bpw"],
            "bpw_match_within_5_percent": size_report["same_module_within_5_percent"],
            "parity": parity, "install_metadata": metadata,
            "checkpoint_strategy": checkpoint_strategy,
            "gpu_memory": _spin_gpu_memory(torch, device),
            "elapsed_seconds_from_allocation_start": _elapsed(start_epoch),
            "training_objective": "full joint denoiser-output NMSE vs BF16 on fixed C8 x 5 packets",
            "vq_endpoint_objective_used": False,
        }
        _atomic_json(out / "summary.json", summary)
        return "complete"
    finally:
        rotation.restore_original_linears(handle)


def _spin_checkpoint(out, state, optimizer, completed, expected, torch, protocol,
                     size_report, parity, start_epoch, phase, last_loss=None,
                     last_grad_norm=None, checkpoint_strategy=None, gpu_memory=None):
    checkpoint = {
        "arm": "spinquant_w4a4", "expected_updates": expected,
        "completed_updates": completed,
        "rotation_state": {key: value.detach().cpu() for key, value in state.state_dict().items()},
        "optimizer_state": _cpu_tree(optimizer.state_dict(), torch), "rng_state": _rng_state(torch),
        "checkpoint_strategy": checkpoint_strategy,
        "gpu_memory": gpu_memory,
    }
    _atomic_torch(out / "checkpoint.pt", checkpoint)
    _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected, phase,
                              elapsed=_elapsed(start_epoch), size_report=size_report, parity=parity,
                              checkpoint_strategy=checkpoint_strategy, gpu_memory=gpu_memory,
                              message=None if last_loss is None else
                              f"last_loss={last_loss}; last_grad_norm={last_grad_norm}")


def _capture_transformed_second_moment(runtime, protocol, contexts, modules, transforms,
                                       out, start_epoch, soft_deadline):
    import endpoint
    import quantization

    torch = _torch()
    _, _, batched, _, _ = quantization._helpers()
    case_ids = [int(value) for value in protocol["calibration"]["case_ids"]]
    path = out / "calibration_cache" / "transformed_second_moment.pt"
    if path.exists():
        return _load_tensor_file(path, torch)
    sums = {name: torch.zeros(module.in_features, dtype=torch.float64)
            for name, module in modules.items()}
    counts = {name: 0 for name in modules}
    handles = []
    def make_hook(name):
        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            z = batched.transform_input(inputs[0].detach(), transforms[name]).float()
            z = z.reshape(-1, z.shape[-1]).double()
            sums[name].add_(z.square().sum(0).cpu())
            counts[name] += z.shape[0]
        return hook
    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        for case_id in case_ids:
            if not _has_time(start_epoch, soft_deadline, estimate_seconds=180):
                return None
            context = contexts[case_id]
            result = _run_endpoint(runtime, context, protocol,
                                   checkpoint_denoiser=False, collect_packets=False,
                                   grad_enabled=False, dataset_stats=runtime.dataset_stats,
                                   latent_indices=context.get("latent_indices"))
            print(f"VQ_TRANSFORM_STATS case={case_id} calls={result.sampler_evaluations}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(value <= 0 for value in counts.values()):
        missing = [name for name, value in counts.items() if value <= 0]
        raise RuntimeError(f"No transformed A4 activation rows for target Linears: {missing[:8]}")
    moments = {name: (sums[name] / counts[name]).float() for name in modules}
    _atomic_torch(path, moments)
    return moments


def _vq_prepare_banks(runtime, protocol, contexts, act_max, modules, out, start_epoch, soft_deadline):
    import quantization

    torch = _torch()
    _, micro, _, smooth, _ = quantization._helpers()
    cache = out / "calibration_cache"
    transform_path = cache / "transforms.pt"
    if transform_path.exists():
        transforms = _load_tensor_file(transform_path, torch)
        transforms = {name: {key: value.to(modules[name].weight.device) if torch.is_tensor(value) else value
                             for key, value in transform.items()}
                      for name, transform in transforms.items()}
    else:
        transforms = {}
        for index, (name, module) in enumerate(modules.items()):
            if not _has_time(start_epoch, soft_deadline, estimate_seconds=20):
                return None
            transforms[name] = smooth.make_transform(
                act_max[name].to(module.weight.device), module.weight,
                alpha=float(protocol["quantization"]["vq"]["smooth_alpha"]),
                hadamard=True, block=int(protocol["quantization"]["vq"]["hadamard_block"]),
                seed=SEED + index,
            )
        _atomic_torch(transform_path, _cpu_tree(transforms, torch))
    moments = _capture_transformed_second_moment(runtime, protocol, contexts, modules,
                                                transforms, out, start_epoch, soft_deadline)
    if moments is None:
        return None

    enc_dir = cache / "vq_banks"
    enc_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, module) in enumerate(modules.items()):
        filename = f"linear_{index:04d}.pt"
        path = enc_dir / filename
        if path.exists():
            continue
        if not _has_time(start_epoch, soft_deadline, estimate_seconds=30):
            return None
        weight = smooth.transform_weight(module.weight.detach(), transforms[name])
        encoding = micro.fit_vq(
            weight, second_moment=moments[name].to(module.weight.device),
            seed=SEED + index, sample_limit=8192, iterations=8,
        )
        micro.check_encoding(encoding)
        _atomic_torch(path, {"name": name, "encoding": _cpu_tree(encoding, torch),
                             "transform": _cpu_tree(transforms[name], torch)})
        if index % 20 == 0 or index + 1 == len(modules):
            print(f"VQ_BANK_FIT {index + 1}/{len(modules)} name={name}", flush=True)
    encodings, stored_transforms = {}, {}
    for index, name in enumerate(modules):
        payload = _load_tensor_file(enc_dir / f"linear_{index:04d}.pt", torch)
        if payload["name"] != name:
            raise RuntimeError("VQ bank file order/name differs from frozen target manifest")
        encodings[name] = {key: value.to(modules[name].weight.device) if torch.is_tensor(value) else value
                           for key, value in payload["encoding"].items()}
        stored_transforms[name] = {key: value.to(modules[name].weight.device) if torch.is_tensor(value) else value
                                   for key, value in payload["transform"].items()}
    return encodings, stored_transforms


def _vq_calibrate(runtime, protocol, root, out, start_epoch, soft_deadline,
                  modules, contexts, teacher_targets, act_max, size_report, parity):
    import endpoint
    import quantization

    torch = _torch()
    case_ids = [int(value) for value in protocol["calibration"]["case_ids"]]
    checkpoint_path = out / "checkpoint.pt"
    expected, completed = 16, 0
    saved_checkpoint = None
    if checkpoint_path.exists():
        saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved_checkpoint.get("arm") != quantization.VQ_ARM or saved_checkpoint.get("expected_updates") != expected:
            raise RuntimeError("VQ resume checkpoint does not match the frozen recipe")
        completed = int(saved_checkpoint["completed_updates"])
        if not 0 <= completed <= expected:
            raise RuntimeError("VQ resume checkpoint has an invalid completed-update count")
    prepared = _vq_prepare_banks(runtime, protocol, contexts, act_max, modules,
                                 out, start_epoch, soft_deadline)
    if prepared is None:
        _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected,
                                 "vq_bank_fit_or_activation_stats", elapsed=_elapsed(start_epoch),
                                 size_report=size_report, parity=parity)
        return "checkpointed"
    encodings, transforms = prepared
    bank_state = quantization.vq_train_state(modules, encodings, transforms, trainable=True)
    parameters = bank_state.parameters()
    optimizer = torch.optim.Adam(parameters, lr=5e-4)
    latent_weight_path = out / "calibration_cache" / "latent_weight.json"
    latent_weight = None
    initial_forward_seconds = []
    if latent_weight_path.exists():
        latent_payload = json.loads(latent_weight_path.read_text(encoding="utf-8"))
        expected_motor_dimensions = int(protocol["quantization"]["vq"]["action_motor_dimensions"])
        if latent_payload.get("action_motor_dimensions") == expected_motor_dimensions:
            latent_weight = float(latent_payload["lambda"])
            initial_forward_seconds = list(latent_payload.get("endpoint_forward_seconds", []))
    if latent_weight is None:
        if not _has_time(start_epoch, soft_deadline, estimate_seconds=180):
            bank_state.restore_forwards()
            _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected, "vq_initial_loss_ratio",
                                     elapsed=_elapsed(start_epoch), size_report=size_report, parity=parity)
            return "checkpointed"
        action0, future0, gripper0 = [], [], []
        for case_id in case_ids:
            context = contexts[case_id]
            forward_started = time.perf_counter()
            output = _run_endpoint(runtime, context, protocol,
                                   checkpoint_denoiser=False, collect_packets=False,
                                   grad_enabled=False, dataset_stats=runtime.dataset_stats,
                                   latent_indices=context.get("latent_indices"))
            initial_forward_seconds.append(time.perf_counter() - forward_started)
            reference = endpoint.EndpointResult(
                generated_latent=None, orig_clean_latent_frames=None,
                action=teacher_targets[case_id]["action"].to(next(iter(modules.values())).weight.device),
                future_image_slots={key: value.to(next(iter(modules.values())).weight.device)
                                    for key, value in teacher_targets[case_id]["future_image_slots"].items()},
                sampler_evaluations=5, denoiser_calls=0, packets=[],
            )
            action_loss, future_loss = endpoint.endpoint_losses(
                output, reference, action_width=16 * 7,
                action_motor_dimensions=int(protocol["quantization"]["vq"]["action_motor_dimensions"]))
            gripper_mse = endpoint.endpoint_gripper_mse(output, reference, action_width=16 * 7)
            action0.append(float(action_loss.detach()))
            future0.append(float(future_loss.detach()))
            gripper0.append(float(gripper_mse.detach()))
            if not _has_time(start_epoch, soft_deadline, estimate_seconds=15):
                bank_state.restore_forwards()
                _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected, "vq_initial_loss_ratio",
                                         elapsed=_elapsed(start_epoch), size_report=size_report, parity=parity)
                return "checkpointed"
        a0, z0 = sum(action0) / len(action0), sum(future0) / len(future0)
        rho = float(protocol["quantization"]["vq"]["rho"])
        if not math.isfinite(a0) or not math.isfinite(z0) or z0 <= 0:
            bank_state.restore_forwards()
            raise RuntimeError(f"Invalid frozen initial endpoint losses A0={a0}, Z0={z0}")
        latent_weight = rho * a0 / z0
        _atomic_json(latent_weight_path, {"rho": rho, "A0_mean_motor_action_mse": a0,
                                         "Z0_mean_future_image_nmse": z0,
                                         "action_motor_dimensions": int(protocol["quantization"]["vq"]["action_motor_dimensions"]),
                                         "initial_gripper_continuous_mse": sum(gripper0) / len(gripper0),
                                         "lambda": latent_weight,
                                         "endpoint_forward_seconds": initial_forward_seconds,
                                         "frozen_before_update_1": True})

    if saved_checkpoint is not None:
        bank_state.restore(saved_checkpoint["bank_masters"])
        optimizer.load_state_dict(saved_checkpoint["optimizer_state"])
        _optimizer_to_device(optimizer, next(iter(modules.values())).weight.device, torch)
        _restore_rng(saved_checkpoint["rng_state"], torch)

    device = next(iter(modules.values())).weight.device
    loss_rows = []
    if saved_checkpoint is not None:
        # Restore the diagnostic log so the fixed 16-update run remains auditable.
        saved_loss_rows = saved_checkpoint.get("loss_rows", [])
        loss_rows = list(saved_loss_rows)
    prior_update_seconds = (float(loss_rows[-1].get("update_seconds", 0.0)) if loss_rows else 0.0)
    if prior_update_seconds > 0:
        next_update_estimate = max(30, int(prior_update_seconds * 1.5))
    elif initial_forward_seconds:
        next_update_estimate = max(90, int(sorted(initial_forward_seconds)[len(initial_forward_seconds) // 2] * 2.0))
    else:
        next_update_estimate = 240
    try:
        while completed < expected:
            if not _has_time(start_epoch, soft_deadline,
                             estimate_seconds=next_update_estimate + 30):
                _vq_checkpoint(out, bank_state, optimizer, completed, expected, torch,
                               protocol, size_report, parity, start_epoch, latent_weight, loss_rows,
                               "vq_endpoint_updates")
                return "checkpointed"
            update_index = completed
            case_id = case_ids[update_index % len(case_ids)]
            context = contexts[case_id]
            reference = endpoint.EndpointResult(
                generated_latent=None, orig_clean_latent_frames=None,
                action=teacher_targets[case_id]["action"].to(device),
                future_image_slots={key: value.to(device)
                                    for key, value in teacher_targets[case_id]["future_image_slots"].items()},
                sampler_evaluations=5, denoiser_calls=0, packets=[],
            )
            optimizer.zero_grad(set_to_none=True)
            update_started = time.perf_counter()
            student = _run_endpoint(runtime, context, protocol,
                                    checkpoint_denoiser=True, collect_packets=False,
                                    grad_enabled=True, dataset_stats=runtime.dataset_stats,
                                    latent_indices=context.get("latent_indices"))
            action_loss, future_loss = endpoint.endpoint_losses(
                student, reference, action_width=16 * 7,
                action_motor_dimensions=int(protocol["quantization"]["vq"]["action_motor_dimensions"]))
            gripper_mse = endpoint.endpoint_gripper_mse(student, reference, action_width=16 * 7)
            total_loss = action_loss + latent_weight * future_loss
            if not bool(torch.isfinite(total_loss)):
                raise RuntimeError(f"VQ endpoint loss is nonfinite at update {update_index + 1}")
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            bank_state.refresh()
            update_seconds = time.perf_counter() - update_started
            completed += 1
            loss_rows.append({"update": completed, "case_id": case_id,
                              "action_mse": float(action_loss.detach()),
                              "gripper_continuous_mse": float(gripper_mse.detach()),
                              "future_image_nmse": float(future_loss.detach()),
                              "lambda": float(latent_weight),
                              "total": float(total_loss.detach()),
                              "grad_norm": float(grad_norm),
                              "update_seconds": float(update_seconds)})
            next_update_estimate = max(30, int(update_seconds * 1.5))
            _vq_checkpoint(out, bank_state, optimizer, completed, expected, torch,
                           protocol, size_report, parity, start_epoch, latent_weight,
                           loss_rows, "vq_endpoint_updates")
            if completed % 2 == 0 or completed == expected:
                print(f"VQ_UPDATE {completed}/{expected} case={case_id} "
                      f"action={float(action_loss.detach()):.7g} future={float(future_loss.detach()):.7g} "
                      f"lambda={latent_weight:.7g} elapsed={_elapsed(start_epoch):.1f}s", flush=True)

        if not _has_time(start_epoch, soft_deadline, estimate_seconds=60):
            return "checkpointed"
        metadata = quantization.save_vq_deployment(
            out, modules, bank_state, size_report,
            metadata={"training_objective": "full_five_evaluation_endpoint_action_plus_future_slots_6_7",
                      "action_motor_dimensions": int(protocol["quantization"]["vq"]["action_motor_dimensions"]),
                      "rho": float(protocol["quantization"]["vq"]["rho"]),
                      "lambda": float(latent_weight), "expected_updates": expected,
                      "completed_updates": completed, "learning_rate": 5e-4,
                      "gradient_clip_norm": 1.0, "checkpoint_selection": "fixed_final_update_16",
                      "parity_status": parity["status"]},
        )
        metadata["bpw_match_within_5_percent"] = size_report["same_module_within_5_percent"]
        _atomic_json(out / "training_losses.json", {"updates": loss_rows,
                                                       "latent_weight": latent_weight})
        summary = {
            "protocol": protocol["protocol"], "status": "complete", "arm": quantization.VQ_ARM,
            "phase": "complete", "expected_updates": expected, "completed_updates": completed,
            "training_updates": {"expected": expected, "completed": completed},
            "size_report": size_report, "effective_bpw": metadata["same_module_effective_bpw"],
            "whole_dit_effective_bpw": metadata["whole_dit_effective_bpw"],
            "bpw_match_within_5_percent": size_report["same_module_within_5_percent"],
            "parity": parity, "install_metadata": metadata,
            "elapsed_seconds_from_allocation_start": _elapsed(start_epoch),
            "rho": float(protocol["quantization"]["vq"]["rho"]),
            "lambda": float(latent_weight),
            "training_objective": "five-step sampler endpoint unnormalized 16x6 motor-action MSE + lambda * future-image latent NMSE at slots 6 and 7",
            "gripper_training_loss_used": False,
            "spin_denoiser_objective_used": False,
        }
        _atomic_json(out / "summary.json", summary)
        return "complete"
    finally:
        bank_state.restore_forwards()


def _vq_checkpoint(out, bank_state, optimizer, completed, expected, torch,
                   protocol, size_report, parity, start_epoch, latent_weight, loss_rows, phase):
    _atomic_torch(out / "checkpoint.pt", {
        "arm": "action_latent_micro_vq_a4", "expected_updates": expected,
        "completed_updates": completed, "bank_masters": bank_state.snapshot(),
        "optimizer_state": _cpu_tree(optimizer.state_dict(), torch),
        "rng_state": _rng_state(torch), "latent_weight": float(latent_weight),
        "loss_rows": list(loss_rows),
    })
    _atomic_json(out / "training_losses.json", {"updates": loss_rows,
                                                "latent_weight": float(latent_weight)})
    _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected, phase,
                              elapsed=_elapsed(start_epoch), size_report=size_report, parity=parity)


def calibration_embedding_cache(protocol, root):
    """Use original C8 embeddings without waiting for held-out Plus text preparation."""
    root = Path(root)
    requirement = json.loads((root / "language_requirements.json").read_text(encoding="utf-8"))
    path = Path(requirement["source_cache"])
    missing = set(requirement["missing"])
    inputs = Path(protocol["calibration"]["source_root"]) / "data" / "inputs"
    for case_id in protocol["calibration"]["case_ids"]:
        metadata = json.loads((inputs / f"case_{case_id}.json").read_text(encoding="utf-8"))
        if metadata["description"] in missing:
            raise RuntimeError(f"Original C8 instruction embedding is missing for case {case_id}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("spinquant", "vq"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--soft-deadline-seconds", type=int, default=6600)
    args = parser.parse_args(argv)

    start_raw = os.environ.get("COSMOS_JOB_START_EPOCH")
    if not start_raw:
        raise RuntimeError("COSMOS_JOB_START_EPOCH must include model loading in the allocation budget")
    start_epoch = float(start_raw)
    if args.soft_deadline_seconds <= _reserve_seconds() + 30:
        raise ValueError("Soft deadline must leave time for a checkpoint and final save reserve")
    if not os.environ.get("PBS_JOBID") or not os.environ.get("PBS_NODEFILE"):
        raise RuntimeError("C8 calibration requires a real PBS compute allocation")
    if "login" in socket.gethostname().lower():
        raise RuntimeError("Calibration/model work is forbidden on a login node")

    torch = _torch()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SpinQuant/VQ calibration requires one allocated GPU")
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    expected = 200 if args.arm == "spinquant" else 16
    stage_arm = "spinquant_w4a4" if args.arm == "spinquant" else "action_latent_micro_vq_a4"
    print(f"CALIBRATION_START arm={stage_arm} start_epoch={start_epoch:.0f} "
          f"soft_deadline={args.soft_deadline_seconds} save_reserve={_reserve_seconds()}", flush=True)

    import quantization
    runtime_module = __import__("cosmos_runtime")
    embeddings_path = calibration_embedding_cache(protocol, root)
    runtime = runtime_module.CosmosRuntime(protocol, root, out, arm="bf16",
                                          text_embeddings_path=embeddings_path)
    try:
        modules = quantization.select_target_linears(runtime.model, protocol)
        target_rows = quantization.target_manifest(modules)
        size_report = _module_size_report(modules, runtime.model)
        _atomic_json(out / "target_modules.json", {
            "protocol": protocol["protocol"], "target_scope": protocol["quantization"]["target_scope"],
            "target_modules": target_rows, "size_report": size_report,
        })
        need_packets = args.arm == "spinquant"
        cached = _calibration_contexts(runtime, protocol, out, start_epoch,
                                       args.soft_deadline_seconds, modules, need_packets)
        if cached is None:
            return 0
        contexts, targets, packets, act_max, parity = cached
        if not _has_time(start_epoch, args.soft_deadline_seconds, estimate_seconds=60):
            arm = "spinquant_w4a4" if args.arm == "spinquant" else "action_latent_micro_vq_a4"
            completed = _saved_completed_updates(out, expected, arm, torch)
            _write_checkpoint_summary(out, protocol, "checkpointed", completed, expected,
                                     "after_bf16_reference_cache", elapsed=_elapsed(start_epoch),
                                     size_report=size_report, parity=parity)
            return 0
        if args.arm == "spinquant":
            status = _spin_calibrate(runtime, protocol, root, out, start_epoch,
                                     args.soft_deadline_seconds, modules, targets,
                                     packets, size_report, parity)
        else:
            status = _vq_calibrate(runtime, protocol, root, out, start_epoch,
                                   args.soft_deadline_seconds, modules, contexts,
                                   targets, act_max, size_report, parity)
        if status == "checkpointed":
            print(f"CALIBRATION_CHECKPOINTED arm={stage_arm} "
                  f"elapsed={_elapsed(start_epoch):.1f}s", flush=True)
        else:
            print(f"CALIBRATION_COMPLETE arm={stage_arm} "
                  f"elapsed={_elapsed(start_epoch):.1f}s", flush=True)
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())


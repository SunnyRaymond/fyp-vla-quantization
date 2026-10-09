"""Final-action VJPs for the fixed Fast-WAM Optional-IDM pilot query.

This module patches only the live model instance for a single PBS query. It
does not modify the vendored FastWAM source. Each random action probe gets a
fresh checkpointed denoising replay, so graph memory stays bounded by one
denoising call plus the retained latent/KV state.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
import inspect
import os
from pathlib import Path
import socket
import time
from typing import Callable, Iterable


PARITY_MAX_ABS_TOL = 2e-3
PARITY_RMSE_TOL = 1e-3
PREFLIGHT_MODULES = (
    "video_expert.blocks.17.cross_attn.o",
    "action_expert.blocks.16.cross_attn.o",
    "action_expert.blocks.21.cross_attn.o",
)


def _require_compute_allocation() -> None:
    host = socket.gethostname().split(".")[0]
    jobid = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not jobid or not nodefile or "login" in host.lower():
        raise RuntimeError("Approved PBS compute allocation required for sensitivity inference")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")


def _raw_idm_infer(model):
    """Find the undecorated IDM implementation, skipping the Optional-IDM router."""
    for cls in type(model).__mro__:
        method = cls.__dict__.get("infer_action")
        if method is None:
            continue
        parameters = inspect.signature(method).parameters
        if "num_video_frames" in parameters and "action_infer_mode" not in parameters:
            return inspect.unwrap(method)
    raise TypeError("Expected a FastWAMIDM-compatible infer_action method")


def _query_kwargs(runtime, datum: dict, seed: int) -> dict:
    if not isinstance(getattr(runtime, "description", None), str):
        raise ValueError("Runtime must have the fixed task description set by runtime.task()")
    return {
        "prompt": runtime.ev.DEFAULT_PROMPT.format(task=runtime.description),
        "input_image": datum["image"],
        "proprio": datum["proprio"],
        "action_horizon": 32,
        "num_video_frames": 9,
        "num_inference_steps": 10,
        "sigma_shift": 1.0,
        "seed": int(seed),
        "rand_device": "cpu",
        "tiled": False,
        "compile_action_infer": False,
    }


def _regular_inputs(torch, datum: dict) -> dict:
    # A caller may have prepared tensors while inference_mode was active.
    out = {}
    with torch.inference_mode(False):
        for key in ("image", "proprio"):
            value = datum[key]
            out[key] = value.clone() if value.is_inference() else value
    return out


@contextmanager
def _capture_phase(state: dict, phase: str):
    previous = state["phase"]
    state["phase"] = phase
    try:
        yield
    finally:
        state["phase"] = previous


def _copy_group_to_cpu(torch, tensors: list):
    if not tensors:
        return []
    if all(tuple(t.shape) == tuple(tensors[0].shape) for t in tensors):
        return list(torch.stack(tensors, dim=0).cpu().unbind(0))
    return [tensor.cpu() for tensor in tensors]


def _move_probe_to_cpu(torch, records: dict, probe_index: int, first_probe: bool) -> None:
    """Transfer compact sampled rows in module-sized batches, not hook-sized copies."""
    for calls in records.values():
        if first_probe:
            for field in ("inputs", "teacher_y"):
                values = [call.pop(f"_{field}_gpu") for call in calls]
                for call, value in zip(calls, _copy_group_to_cpu(torch, values)):
                    call[field] = value.float()
        connected_indices = []
        gpu_values = []
        cpu_values = [None] * len(calls)
        for index, call in enumerate(calls):
            value = call.pop("_adjoint_gpu", None)
            connected = value is not None
            call["adjoint_connected"][probe_index] = connected
            if connected:
                connected_indices.append(index)
                gpu_values.append(value)
            else:
                # A target may be called yet be structurally disconnected from
                # this action probe (for example, a no-grad conditioning branch).
                # Keep its row/output shape explicit and its task contribution zero.
                cpu_values[index] = torch.zeros(
                    (int(call["row_ids"].numel()), int(call["output_shape"][-1])),
                    dtype=torch.float32,
                    device="cpu",
                )
        for index, value in zip(connected_indices, _copy_group_to_cpu(torch, gpu_values)):
            cpu_values[index] = value.float()
        for call, value in zip(calls, cpu_values):
            call["adjoints"][probe_index] = value


def _stack_probe_rows(torch, records: dict, probes: int) -> None:
    for calls in records.values():
        for call in calls:
            call["adjoints"] = torch.stack(call["adjoints"], dim=0)
            if call["adjoints"].shape[0] != probes:
                raise RuntimeError("Sensitivity probe count differs from the requested count")
            if any(value is None for value in call["adjoint_connected"]):
                raise RuntimeError("A call is missing connected/disconnected status for a probe")


def capture_action_projections(
    runtime,
    datum: dict,
    seed: int,
    module_names: Iterable[str] | None,
    probes: int,
    probe_seed: int,
    rows_per_call: int = 2,
    artifact_path: str | Path | None = None,
) -> dict:
    """Capture sampled Linear outputs and their final-action probe adjoints.

    Each adjoint is d(probe dot final_action[:10, :6])/d(Linear output) for
    one actual module invocation. For a shared weight, callers must sum all
    of its per-call dot products before squaring the sampled local first-order
    projected proxy. Passing ``module_names=None`` selects every target in the
    runtime quantizer's canonical ``q.modules`` table.
    """
    _require_compute_allocation()
    if probes <= 0 or rows_per_call <= 0:
        raise ValueError("probes and rows_per_call must be positive")
    import torch
    from torch.nn import Linear
    from torch.utils.checkpoint import checkpoint

    model = runtime.model
    if getattr(runtime, "current_arm", None) != "bf16":
        raise RuntimeError("Arm BF16 before collecting the teacher action sensitivities")
    if bool(getattr(runtime, "weight_converted", False)) or bool(
        getattr(getattr(runtime, "q", None), "weights_converted", False)
    ):
        raise RuntimeError("Sensitivity capture requires the unconverted BF16 teacher weights")

    # Quantization.modules is the canonical, object-deduplicated target table
    # used by the pilot quantizer (video/action experts plus proprio_encoder).
    # Keep it authoritative so a full call can cover all 614 target Linears.
    q_module_table = getattr(getattr(runtime, "q", None), "modules", None)
    target_modules = dict(q_module_table) if isinstance(q_module_table, Mapping) else {}
    try:
        modules = dict(model.named_modules(remove_duplicate=False))
    except TypeError:  # compatibility with older torch; pilot quantizer uses the same API
        modules = dict(model.named_modules())
    modules.update(target_modules)
    if module_names is None:
        if not target_modules:
            raise ValueError("module_names=None requires runtime.q.modules as the target table")
        names = tuple(target_modules)
    else:
        names = tuple(module_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("module_names must be a nonempty list of unique module paths")
    missing = [name for name in names if name not in modules]
    if missing:
        raise KeyError(f"Unknown Linear modules: {missing[:8]}")
    bad = [name for name in names if not isinstance(modules[name], Linear)]
    if bad:
        raise TypeError(f"Sensitivity targets must be nn.Linear modules: {bad[:8]}")
    if len({id(modules[name]) for name in names}) != len(names):
        raise ValueError("module_names contains multiple aliases for the same Linear object")
    target_name_set = set(target_modules)
    selected_name_set = set(names)
    covered_target_names = target_name_set & selected_name_set

    query_datum = _regular_inputs(torch, datum)
    kwargs = _query_kwargs(runtime, query_datum, seed)
    baseline_kwargs = dict(kwargs, action_infer_mode="idm")
    baseline_start = time.perf_counter()
    print("SENSITIVITY baseline query start", flush=True)
    with torch.inference_mode():
        baseline_action = model.infer_action(**baseline_kwargs)["action"].detach().float().cpu()
    print(
        f"SENSITIVITY baseline query complete seconds={time.perf_counter() - baseline_start:.1f}",
        flush=True,
    )
    if tuple(baseline_action.shape) != (32, 7) or not bool(torch.isfinite(baseline_action).all()):
        raise RuntimeError(f"Expected finite normalized [32,7] baseline action, got {tuple(baseline_action.shape)}")

    probe_generator = torch.Generator(device="cpu").manual_seed(int(probe_seed))
    probe_vectors = torch.randint(
        0, 2, (probes, 60), generator=probe_generator, dtype=torch.int64
    ).mul_(2).sub_(1).float().div_(60**0.5).reshape(probes, 10, 6)

    state = {
        "phase": "capture",
        "probe": 0,
        "stage": "raw_query_conditioning",
        "step": -1,
        "context": "raw_idm_preprocessing",
        "video_step": 0,
        "action_step": 0,
        "action_steps": 0,
        "action": None,
        "call_counts": {name: 0 for name in names},
    }
    records = {name: [] for name in names}
    module_order = {name: index for index, name in enumerate(names)}

    def checkpoint_contexts():
        return _capture_phase(state, "capture"), _capture_phase(state, "recompute")

    handles = []
    for name in names:
        module = modules[name]

        def capture(module, args, output, _name=name):
            phase = state["phase"]
            if phase not in ("capture", "recompute"):
                return
            if not isinstance(output, torch.Tensor) or not args or not isinstance(args[0], torch.Tensor):
                raise TypeError(f"Expected Tensor input/output from Linear {_name}")
            leaf_injected = not output.requires_grad
            if leaf_injected:
                if output.is_inference():
                    raise RuntimeError(
                        f"Cannot make inference tensor output differentiable for {_name}; "
                        "the raw query must run inside inference_mode(False)"
                    )
                try:
                    # Preserve an existing graph. If this target was evaluated
                    # with frozen weights and a non-grad input, make its output a
                    # leaf so downstream enabled-grad conditioning can still be
                    # differentiated with respect to this Linear's output.
                    output.requires_grad_(True)
                except RuntimeError as exc:
                    raise RuntimeError(f"Cannot inject a grad leaf at {_name} output") from exc
            if phase == "recompute":
                # Match the original checkpoint forward's requires-grad state
                # so recomputation saves/rebuilds the same backward structure.
                # Never add records or hooks during replay.
                return
            call_id = state["call_counts"][_name]
            state["call_counts"][_name] += 1
            x, y = args[0], output
            in_width, out_width = int(x.shape[-1]), int(y.shape[-1])
            total_rows = int(y.numel() // out_width)
            if int(x.numel() // in_width) != total_rows:
                raise RuntimeError(f"Linear row count changed at {_name} call {call_id}")

            if state["probe"] == 0:
                count = min(rows_per_call, total_rows)
                # ponytail: uniform row sampling bounds artifacts; the loss helper applies N/n scaling.
                generator = torch.Generator(device="cpu").manual_seed(
                    int(probe_seed) + (module_order[_name] + 1) * 1_000_003 + call_id * 9_176
                )
                row_ids = torch.randperm(total_rows, generator=generator)[:count].sort().values
                record = {
                    "module": _name,
                    "call_id": call_id,
                    "stage": state["stage"],
                    "step": state["step"],
                    "context": state["context"],
                    "input_shape": tuple(x.shape),
                    "output_shape": tuple(y.shape),
                    "leaf_injected": leaf_injected,
                    "row_ids": row_ids,
                    "total_rows": total_rows,
                    "adjoints": [None] * probes,
                    "adjoint_connected": [None] * probes,
                    "_inputs_gpu": x.reshape(-1, in_width).index_select(
                        0, row_ids.to(device=x.device)
                    ).detach().float(),
                    "_teacher_y_gpu": y.reshape(-1, out_width).index_select(
                        0, row_ids.to(device=y.device)
                    ).detach().float(),
                }
                records[_name].append(record)
            else:
                calls = records[_name]
                if call_id >= len(calls):
                    raise RuntimeError(f"Extra replay call {_name}#{call_id} on probe {state['probe']}")
                record = calls[call_id]
                if record["input_shape"] != tuple(x.shape) or record["output_shape"] != tuple(y.shape):
                    raise RuntimeError(f"Tensor shape changed at {_name} call {call_id}")
                if (
                    record["stage"] != state["stage"]
                    or record["step"] != state["step"]
                    or record["context"] != state["context"]
                ):
                    raise RuntimeError(f"Denoising call order changed at {_name} call {call_id}")
                if record["leaf_injected"] != leaf_injected:
                    raise RuntimeError(f"Grad-leaf behavior changed at {_name} call {call_id}")
            rows = record["row_ids"].to(device=y.device)

            def save_adjoint(grad, _record=record, _rows=rows, _width=out_width):
                if _record.get("_adjoint_gpu") is not None:
                    raise RuntimeError("Linear output gradient hook fired more than once")
                _record["_adjoint_gpu"] = grad.reshape(-1, _width).index_select(0, _rows).detach().float()
                return grad

            output.register_hook(save_adjoint)

        handles.append(module.register_forward_hook(capture))

    params = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    for parameter, _requires_grad in params:
        parameter.requires_grad_(False)

    patches = []

    def patch(obj, name, replacement):
        existed = name in getattr(obj, "__dict__", {})
        previous = getattr(obj, "__dict__", {}).get(name)
        patches.append((obj, name, existed, previous))
        setattr(obj, name, replacement)

    original_video = model._denoise_video

    def denoise_video(*, latents_video, timestep_video, context, context_mask,
                      video_self_attn_mask, fuse_vae_embedding_in_latents):
        state.update(stage="video", step=state["video_step"], context="video_denoising")
        state["video_step"] += 1
        if not latents_video.requires_grad:
            latents_video = latents_video.detach().requires_grad_(True)

        def run(x, timestep, prompt_context, prompt_mask):
            return original_video(
                latents_video=x, timestep_video=timestep, context=prompt_context,
                context_mask=prompt_mask, video_self_attn_mask=video_self_attn_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )

        return checkpoint(
            run, latents_video, timestep_video, context, context_mask,
            use_reentrant=False, context_fn=checkpoint_contexts,
        )

    patch(model, "_denoise_video", denoise_video)

    original_action = model._denoise_action_with_video_cache

    def denoise_action(*, latents_action, timestep_action, context, context_mask,
                       video_cache_k, video_cache_v, action_attention_mask):
        state.update(stage="action", step=state["action_step"], context="action_with_video_kv")
        state["action_step"] += 1
        if not latents_action.requires_grad:
            latents_action = latents_action.detach().requires_grad_(True)
        layer_count = len(video_cache_k)

        def run(x, timestep, prompt_context, prompt_mask, *flat_cache):
            keys = list(flat_cache[:layer_count])
            values = list(flat_cache[layer_count:])
            return original_action(
                latents_action=x, timestep_action=timestep, context=prompt_context,
                context_mask=prompt_mask, video_cache_k=keys, video_cache_v=values,
                action_attention_mask=action_attention_mask,
            )

        return checkpoint(
            run, latents_action, timestep_action, context, context_mask,
            *video_cache_k, *video_cache_v,
            use_reentrant=False, context_fn=checkpoint_contexts,
        )

    patch(model, "_denoise_action_with_video_cache", denoise_action)

    original_prefill = model.mot.prefill_video_cache_tensor
    layer_count = len(model.video_expert.blocks)

    def prefill_video_cache(*, video_tokens, video_freqs, video_t_mod, video_context,
                            video_context_mask, video_attention_mask):
        state.update(stage="video_conditioning_prefill", step=-1, context="video_kv_prefill")

        def run(tokens, freqs, t_mod, prompt_context, prompt_mask, attention_mask):
            keys, values = original_prefill(
                video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod,
                video_context=prompt_context, video_context_mask=prompt_mask,
                video_attention_mask=attention_mask,
            )
            return tuple(keys) + tuple(values)

        cache = checkpoint(
            run, video_tokens, video_freqs, video_t_mod, video_context,
            video_context_mask, video_attention_mask,
            use_reentrant=False, context_fn=checkpoint_contexts,
        )
        return list(cache[:layer_count]), list(cache[layer_count:])

    patch(model.mot, "prefill_video_cache_tensor", prefill_video_cache)

    original_step = model.infer_action_scheduler.step

    def action_step(model_output, delta, sample):
        result = original_step(model_output, delta, sample)
        state["action"] = result[0]
        state["action_steps"] += 1
        return result

    patch(model.infer_action_scheduler, "step", action_step)

    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_infer = _raw_idm_infer(model)
    probe_action_gradients = []
    try:
        for probe_index in range(probes):
            state.update(
                phase="capture", probe=probe_index,
                stage="raw_query_conditioning", step=-1, context="raw_idm_preprocessing",
                video_step=0, action_step=0, action_steps=0, action=None,
                call_counts={name: 0 for name in names},
            )
            probe_start = time.perf_counter()
            print(f"SENSITIVITY probe {probe_index + 1}/{probes} forward start", flush=True)
            with torch.inference_mode(False), torch.enable_grad():
                result = raw_infer(model, **kwargs)
                action = state["action"]
                if action is None or not action.requires_grad:
                    raise RuntimeError("The final action lost its autograd graph")
                if tuple(action.shape) != (32, 7):
                    raise RuntimeError(f"Expected final normalized [32,7] action, got {tuple(action.shape)}")
                action.retain_grad()
                if state["video_step"] != 10 or state["action_steps"] != 10:
                    raise RuntimeError(
                        f"Expected 10 video and 10 action steps, got {state['video_step']} and {state['action_steps']}"
                    )
                if probe_index == 0:
                    differentiable_action = action.detach().float().cpu()
                probe = probe_vectors[probe_index].to(device=action.device)
                print(
                    f"SENSITIVITY probe {probe_index + 1}/{probes} forward complete "
                    f"seconds={time.perf_counter() - probe_start:.1f}",
                    flush=True,
                )
                print(
                    f"SENSITIVITY probe {probe_index + 1}/{probes} backward start",
                    flush=True,
                )
                backward_start = time.perf_counter()
                ((action[:10, :6].float() * probe).sum()).backward()
                action_grad = action.grad
                if (
                    action_grad is None
                    or tuple(action_grad.shape) != (32, 7)
                    or not bool(torch.isfinite(action_grad).all())
                    or not bool(action_grad.abs().sum() > 0)
                ):
                    raise RuntimeError("The full [32,7] final-action tensor did not receive a finite nonzero probe gradient")
                probe_action_gradients.append(action_grad.detach().float().cpu())
                print(
                    f"SENSITIVITY probe {probe_index + 1}/{probes} backward complete "
                    f"seconds={time.perf_counter() - backward_start:.1f}; "
                    f"total_seconds={time.perf_counter() - probe_start:.1f}",
                    flush=True,
                )
            for name in names:
                if state["call_counts"][name] != len(records[name]):
                    raise RuntimeError(f"Unexpected call coverage for {name}")
            _move_probe_to_cpu(torch, records, probe_index, first_probe=(probe_index == 0))
            del result, action
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        _stack_probe_rows(torch, records, probes)
        parity_delta = differentiable_action - baseline_action
        parity = {
            "max_abs": float(parity_delta.abs().max()),
            "rmse": float(parity_delta.square().mean().sqrt()),
            "max_abs_tolerance": PARITY_MAX_ABS_TOL,
            "rmse_tolerance": PARITY_RMSE_TOL,
        }
        parity["passed"] = (
            parity["max_abs"] <= PARITY_MAX_ABS_TOL and parity["rmse"] <= PARITY_RMSE_TOL
        )
        gradient_norms = {
            name: (
                float(torch.stack([call["adjoints"].square().sum() for call in calls]).sum().sqrt())
                if calls else 0.0
            )
            for name, calls in records.items()
        }
        connected_by_probe = [
            sum(bool(call["adjoint_connected"][probe]) for calls in records.values() for call in calls)
            for probe in range(probes)
        ]
        disconnected_by_probe = [
            sum(not bool(call["adjoint_connected"][probe]) for calls in records.values() for call in calls)
            for probe in range(probes)
        ]
        fully_disconnected_by_module = {
            name: sum(not any(call["adjoint_connected"]) for call in calls)
            for name, calls in records.items()
        }
        partially_disconnected_by_module = {
            name: sum(any(call["adjoint_connected"]) and not all(call["adjoint_connected"]) for call in calls)
            for name, calls in records.items()
        }
        leaf_injected_by_module = {
            name: sum(bool(call["leaf_injected"]) for call in calls)
            for name, calls in records.items()
        }
        uncalled_modules = [name for name, calls in records.items() if not calls]
        summary = {
            "query": "FastWAMOptionalIDM.infer_action(action_infer_mode='idm')",
            "derivative_point": "unquantized BF16 teacher query",
            "objective": "sampled local first-order projected proxy on randomized final-action probes",
            "target_module_count": len(names),
            "quantizer_target_module_count": len(target_name_set) if target_modules else None,
            "quantizer_target_modules_captured": len(covered_target_names) if target_modules else None,
            "quantizer_target_coverage_fraction": (
                len(covered_target_names) / len(target_name_set) if target_name_set else None
            ),
            "full_quantizer_target_coverage": bool(target_modules) and covered_target_names == target_name_set,
            "uncalled_target_modules": uncalled_modules,
            "video_steps": 10,
            "action_steps": 10,
            "action_shape": list(differentiable_action.shape),
            "action_slice": ["first10", "motor6"],
            "gradient_passes": probes,
            "rows_per_call": rows_per_call,
            "final_action_gradient_norm_by_probe": [
                float(gradient.norm()) for gradient in probe_action_gradients
            ],
            "checkpointed": ["video_denoise_step", "video_kv_prefill", "action_denoise_step"],
            "parity": parity,
            "module_call_counts": {name: len(calls) for name, calls in records.items()},
            "adjoint_norm_by_module": gradient_norms,
            "connected_call_count_by_probe": connected_by_probe,
            "disconnected_call_count_by_probe": disconnected_by_probe,
            "fully_disconnected_call_count_by_module": fully_disconnected_by_module,
            "partially_disconnected_call_count_by_module": partially_disconnected_by_module,
            "leaf_injected_call_count_by_module": leaf_injected_by_module,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        }
        payload = {
            "summary": summary,
            "action": differentiable_action,
            "baseline_action": baseline_action,
            "probe_vectors": probe_vectors,
            "probe_action_gradients": probe_action_gradients,
            "layers": records,
        }
        if artifact_path is not None:
            target = Path(artifact_path)
            if target.exists():
                raise FileExistsError(f"Refusing to replace sensitivity artifact {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".tmp")
            torch.save(payload, temp)
            temp.replace(target)
        return payload
    finally:
        for handle in handles:
            handle.remove()
        for obj, name, existed, previous in reversed(patches):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        for parameter, requires_grad in params:
            parameter.requires_grad_(requires_grad)


def projected_action_task_loss(capture: dict, reconstruct_output: Callable[[str, dict, object], object]):
    """Return the sampled local first-order projected action-loss proxy.

    `reconstruct_output(module_name, call, call['inputs'])` must return the
    candidate quantized Linear output for those exact sampled input rows.
    Per-call responses are summed across all steps and modules before the
    square, preserving shared-weight temporal cross terms.
    """
    import torch

    probe_count = int(capture["summary"]["gradient_passes"])
    projected = None
    for module_name, calls in capture["layers"].items():
        for call in calls:
            candidate = reconstruct_output(module_name, call, call["inputs"])
            if not isinstance(candidate, torch.Tensor) or candidate.ndim != 2:
                raise TypeError("reconstruct_output must return [sampled_rows, output_width] Tensor")
            if tuple(candidate.shape) != tuple(call["teacher_y"].shape):
                raise ValueError(f"Candidate output shape changed for {module_name} call {call['call_id']}")
            target = call["teacher_y"].to(device=candidate.device, dtype=torch.float32)
            adjoints = call["adjoints"].to(device=candidate.device, dtype=torch.float32)
            scale = float(call["total_rows"]) / float(call["row_ids"].numel())
            contribution = (adjoints * (candidate.float() - target).unsqueeze(0)).sum(dim=(1, 2)) * scale
            projected = contribution if projected is None else projected + contribution
    if projected is None or projected.numel() != probe_count:
        raise ValueError("Capture contains no complete Linear-output adjoints")
    return 0.5 * projected.square().mean(), projected


def validate_preflight(capture: dict, required_modules: Iterable[str] = PREFLIGHT_MODULES) -> None:
    """PBS-side assertion for an IDM query and the requested action path."""
    _require_compute_allocation()
    action = capture["action"]
    if tuple(action.shape) != (32, 7):
        raise AssertionError(f"Expected normalized action shape [32,7], got {tuple(action.shape)}")
    action_gradients = capture.get("probe_action_gradients", ())
    if not action_gradients or any(
        tuple(gradient.shape) != (32, 7)
        or not bool(gradient.isfinite().all())
        or not bool(gradient.abs().sum() > 0)
        for gradient in action_gradients
    ):
        raise AssertionError("Expected finite nonzero full [32,7] final-action probe gradients")
    if not capture["summary"]["parity"]["passed"]:
        raise AssertionError(f"Differentiable-query parity failed: {capture['summary']['parity']}")
    required = tuple(dict.fromkeys((*PREFLIGHT_MODULES, *required_modules)))
    for name in required:
        calls = capture["layers"].get(name)
        if not calls:
            raise AssertionError(f"No captured calls for required module {name}")
        norm = sum(float(call["adjoints"].square().sum()) for call in calls) ** 0.5
        if not norm > 0.0:
            raise AssertionError(f"Final-action adjoint is zero for {name}")

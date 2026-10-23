"""Differentiable, official five-evaluation Cosmos Policy endpoint sampler.

Latent indices address modality slots in the generated sequence.  They are
never treated as time steps.  The sampler and denoiser wrappers below only
remove the official ``no_grad`` boundaries; equations, sigmas, seed, and solver
remain the pinned Cosmos implementation.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import types
from dataclasses import dataclass
from typing import Any, Mapping


PARITY_VERSION = 4


@dataclass
class EndpointResult:
    generated_latent: Any
    orig_clean_latent_frames: Any
    action: Any
    future_image_slots: dict[str, Any]
    sampler_evaluations: int
    denoiser_calls: int
    packets: list[dict[str, Any]]
    input_video_layout: dict[str, Any] = dataclasses.field(default_factory=dict)


def _torch():
    import torch

    return torch


def _tree_copy_to_device(value, device, torch):
    if torch.is_tensor(value):
        # Runtime snapshots may have been created in inference_mode.  Cloning
        # after disabling inference mode converts them to ordinary tensors.
        return value.detach().to(device=device, memory_format=torch.preserve_format).clone(
            memory_format=torch.preserve_format
        )
    if isinstance(value, dict):
        return {key: _tree_copy_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_copy_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_copy_to_device(item, device, torch) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        for field in dataclasses.fields(value):
            try:
                object.__setattr__(result, field.name,
                                   _tree_copy_to_device(getattr(value, field.name), device, torch))
            except (AttributeError, TypeError):
                pass
        return result
    if hasattr(value, "__dict__") and not isinstance(value, (type, torch.nn.Module)):
        result = copy.copy(value)
        for key, item in vars(value).items():
            try:
                setattr(result, key, _tree_copy_to_device(item, device, torch))
            except (AttributeError, TypeError):
                pass
        return result
    return value


def _preserve_video_layout(data_batch, torch):
    video = data_batch.get("video")
    if torch.is_tensor(video) and video.ndim == 5:
        data_batch["video"] = video.to(memory_format=torch.channels_last_3d)
    return data_batch


def _tensor_layout(tensor, torch):
    if not torch.is_tensor(tensor):
        return None
    return {
        "dtype": str(tensor.dtype), "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "channels_last_3d": bool(tensor.ndim == 5 and tensor.is_contiguous(
            memory_format=torch.channels_last_3d
        )),
    }


def _cpu_tensor_preserve_format(tensor, torch):
    with torch.inference_mode(False):
        return tensor.detach().to(device="cpu", memory_format=torch.preserve_format).clone(
            memory_format=torch.preserve_format
        )


def _first_parameter_device(model):
    for parameter in model.parameters():
        return parameter.device
    raise RuntimeError("Cosmos model has no parameters from which to infer its device")


def _tensor_index(batch: dict[str, Any], name: str, batch_size: int, device, torch,
                  latent_indices: Mapping[str, Any] | None = None):
    aliases = {
        "action_latent_idx": ("action", "action_idx"),
        "future_wrist_image_latent_idx": ("future_wrist_latent_idx", "future_wrist_idx", "future_wrist"),
        "future_image_latent_idx": ("future_primary_image_latent_idx", "future_image_idx",
                                     "future_primary_image", "future_image"),
    }
    value = None
    for source in (batch, latent_indices or {}):
        for key in (name, *aliases.get(name, ())):
            if key in source:
                value = source[key]
                break
        if value is not None:
            break
    if value is None:
        raise KeyError(f"Official Cosmos data_batch is missing latent-slot index {name!r}")
    value = torch.as_tensor(value, device=device, dtype=torch.long).reshape(-1)
    if value.numel() == 1 and batch_size > 1:
        value = value.expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(f"{name} has {value.numel()} entries for batch size {batch_size}")
    if bool((value < 0).any()):
        raise ValueError(f"{name} contains a negative or absent modality slot")
    return value


def _gather_slot(generated, indices, batch_size, torch):
    if generated.ndim != 5 or generated.shape[0] != batch_size:
        raise ValueError(f"Expected generated latent [B,C,T,H,W], got {tuple(generated.shape)}")
    if bool((indices >= generated.shape[2]).any()):
        raise IndexError("A modality slot index exceeds the generated latent slot axis")
    rows = torch.arange(batch_size, device=generated.device)
    return generated[rows, :, indices, :, :]


def _denoised_tensor(prediction):
    if hasattr(prediction, "x0"):
        return prediction.x0
    if isinstance(prediction, dict) and "x0" in prediction:
        return prediction["x0"]
    if isinstance(prediction, (tuple, list)) and prediction:
        return prediction[0]
    if _torch().is_tensor(prediction):
        return prediction
    raise TypeError(f"Cannot extract the Cosmos denoised output from {type(prediction)!r}")


def _rebuild_prediction(template, x0, eps=None):
    if _torch().is_tensor(template):
        return x0
    if isinstance(template, dict):
        result = dict(template)
        result["x0"] = x0
        if "eps" in result and eps is not None:
            result["eps"] = eps
        return result
    if isinstance(template, tuple):
        return (x0, eps, *template[2:]) if len(template) >= 2 else (x0,)
    if isinstance(template, list):
        return [x0, eps, *template[2:]] if len(template) >= 2 else [x0]
    result = copy.copy(template)
    try:
        object.__setattr__(result, "x0", x0)
        if hasattr(result, "eps") and eps is not None:
            object.__setattr__(result, "eps", eps)
        return result
    except Exception as exc:
        raise TypeError(f"Cannot reconstruct Cosmos denoise prediction {type(template)!r}") from exc


def _sampler_module(model):
    candidates = [module for module in model.modules()
                  if callable(getattr(module, "forward", None))
                  and callable(getattr(module, "_forward_impl", None))
                  and "sampler" in type(module).__name__.lower()]
    if not candidates:
        candidate = getattr(model, "sampler", None)
        if candidate is not None and callable(getattr(candidate, "_forward_impl", None)):
            candidates = [candidate]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Cosmos sampler with _forward_impl, found {len(candidates)}")
    return candidates[0]


def run_endpoint(model, data_batch: dict[str, Any], seed: int, protocol: dict[str, Any], *,
                 checkpoint_denoiser: bool = False, collect_packets: bool = False,
                 grad_enabled: bool = True, dataset_stats: dict[str, Any] | None = None,
                 latent_indices: Mapping[str, Any] | None = None) -> EndpointResult:
    """Run the pinned official sampler with all five NFE calls and gradients.

    When ``collect_packets`` is true, every shared-DiT ``denoise`` invocation
    records its fixed noisy latent, sigma, conditioning object, full joint x0
    output, and the enclosing sampler evaluation number.  Guidance branches at
    one evaluation are grouped together downstream by ``sampler_evaluation``.
    """
    torch = _torch()
    from torch.utils.checkpoint import checkpoint

    device = _first_parameter_device(model)
    with torch.inference_mode(False):
        batch = _tree_copy_to_device(data_batch, device, torch)
        if "video" not in batch:
            raise KeyError("Cosmos endpoint requires the official prepared data_batch['video']")
        _preserve_video_layout(batch, torch)
        pre_sampler_video_layout = _tensor_layout(batch["video"], torch)
    sampler = _sampler_module(model)
    sampler_forward = sampler.forward
    sampler_impl = sampler._forward_impl
    sampler_forward_raw = inspect.unwrap(type(sampler).forward)
    sampler_impl_raw = inspect.unwrap(type(sampler)._forward_impl)
    generate_bound = model.generate_samples_from_batch
    generate_raw = inspect.unwrap(type(model).generate_samples_from_batch)
    denoise_bound = model.denoise
    denoise_raw = inspect.unwrap(type(model).denoise)
    old_grad_enabled = torch.is_grad_enabled()
    state = {"sampler_evaluations": 0, "denoiser_calls": 0, "packets": []}

    def differentiable_denoise(_model, x, sigma, condition, *args, **kwargs):
        step = max(0, state["sampler_evaluations"] - 1)
        state["denoiser_calls"] += 1

        def invoke(x_value, sigma_value):
            grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
            with grad_context:
                return denoise_raw(_model, x_value, sigma_value, condition, *args, **kwargs)

        if checkpoint_denoiser and grad_enabled:
            template_holder = {}

            def checkpointed(x_value, sigma_value):
                output = invoke(x_value, sigma_value)
                template_holder["output"] = output
                x0 = _denoised_tensor(output)
                eps = getattr(output, "eps", None)
                if eps is None and isinstance(output, dict):
                    eps = output.get("eps")
                if eps is None:
                    return (x0,)
                return x0, eps

            tensors = checkpoint(checkpointed, x, sigma, use_reentrant=False)
            if not isinstance(tensors, tuple):
                tensors = (tensors,)
            template = template_holder["output"]
            output = _rebuild_prediction(template, tensors[0], tensors[1] if len(tensors) > 1 else None)
        else:
            output = invoke(x, sigma)

        if collect_packets:
            target = _denoised_tensor(output).detach()
            state["packets"].append({
                "sampler_evaluation": int(step),
                "denoiser_call": int(state["denoiser_calls"] - 1),
                "x": _cpu_tensor_preserve_format(x, torch),
                "sigma": _cpu_tensor_preserve_format(sigma, torch),
                "condition": _tree_copy_to_device(condition, torch.device("cpu"), torch),
                "target": _cpu_tensor_preserve_format(target, torch),
            })
        return output

    def differentiable_sampler_forward(_sampler, x0_fn, x_sigma_max, *args, **kwargs):
        raw_x0_fn = inspect.unwrap(x0_fn)

        def counted_x0_fn(x, sigma):
            state["sampler_evaluations"] += 1
            grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
            with grad_context:
                return raw_x0_fn(x, sigma)

        return sampler_forward_raw(_sampler, counted_x0_fn, x_sigma_max, *args, **kwargs)

    def differentiable_sampler_impl(_sampler, *args, **kwargs):
        return sampler_impl_raw(_sampler, *args, **kwargs)

    forward_mode = torch.inference_mode() if not grad_enabled else torch.inference_mode(False)
    try:
        sampler.forward = types.MethodType(differentiable_sampler_forward, sampler)
        sampler._forward_impl = types.MethodType(differentiable_sampler_impl, sampler)
        model.denoise = types.MethodType(differentiable_denoise, model)
        grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with forward_mode, grad_context:
            generated, original = generate_raw(
                model, batch,
                seed=int(seed), n_sample=int(batch["video"].shape[0]),
                num_steps=int(protocol["inference"]["denoising_steps"]),
                is_negative_prompt=False,
                use_variance_scale=bool(protocol["inference"]["use_variance_scale"]),
                return_orig_clean_latent_frames=True,
            )
    finally:
        sampler.forward, sampler._forward_impl, model.denoise = (
            sampler_forward, sampler_impl, denoise_bound
        )
        torch.set_grad_enabled(old_grad_enabled)

    expected_evaluations = int(protocol["inference"]["denoising_steps"])
    if state["sampler_evaluations"] != expected_evaluations:
        raise RuntimeError(
            f"Pinned sampler made {state['sampler_evaluations']} evaluations, expected {expected_evaluations}"
        )
    batch_size = generated.shape[0]
    action_idx = _tensor_index(batch, "action_latent_idx", batch_size, generated.device, torch,
                               latent_indices)
    action_slot = _gather_slot(generated, action_idx, batch_size, torch).reshape(batch_size, -1)
    chunk_size = int(protocol["inference"]["chunk_size"])
    action_width = chunk_size * 7
    if action_slot.shape[1] < action_width:
        raise ValueError(f"Action slot has {action_slot.shape[1]} values; expected at least {action_width}")
    chunk_count = action_slot.shape[1] // action_width
    action = action_slot[:, :chunk_count * action_width].reshape(batch_size, chunk_count,
                                                                  chunk_size, 7).mean(dim=1)
    if bool(protocol["inference"].get("unnormalize_actions", True)):
        if dataset_stats is None or "actions_min" not in dataset_stats or "actions_max" not in dataset_stats:
            raise ValueError("Official motor-space action loss requires frozen Cosmos action min/max statistics")
        action_min = torch.as_tensor(dataset_stats["actions_min"], device=action.device, dtype=torch.float32)
        action_max = torch.as_tensor(dataset_stats["actions_max"], device=action.device, dtype=torch.float32)
        if action_min.numel() != action.shape[-1] or action_max.numel() != action.shape[-1]:
            raise ValueError("Frozen action statistics do not match the official action dimension")
        action = 0.5 * (action.float() + 1.0) * (action_max - action_min) + action_min

    future = {}
    for output_name, batch_key in (
        ("future_wrist", "future_wrist_image_latent_idx"),
        ("future_image", "future_image_latent_idx"),
    ):
        index = _tensor_index(batch, batch_key, batch_size, generated.device, torch,
                              latent_indices)
        future[output_name] = _gather_slot(generated, index, batch_size, torch)

    input_video_layout = {
        "pre_sampler": pre_sampler_video_layout,
        "post_sampler": _tensor_layout(batch["video"], torch),
    }

    return EndpointResult(
        generated_latent=generated,
        orig_clean_latent_frames=original,
        action=action,
        future_image_slots=future,
        sampler_evaluations=state["sampler_evaluations"],
        denoiser_calls=state["denoiser_calls"],
        packets=state["packets"],
        input_video_layout=input_video_layout,
    )


def endpoint_losses(student: EndpointResult, teacher: EndpointResult, *, action_width: int = 16 * 7,
                    action_motor_dimensions: int = 6):
    """Return motor-only endpoint MSE and mean future-image latent NMSE.

    The two future image views are slots 6 and 7.  No temporal slicing or
    one-step denoiser proxy is used for the VQ training objective.  The
    seventh action dimension (gripper) is excluded from motor loss and can be
    inspected independently with :func:`endpoint_gripper_mse`.
    """
    torch = _torch()
    if set(student.future_image_slots) != set(teacher.future_image_slots):
        raise ValueError("Student and teacher future image slot sets differ")
    if action_width <= 0 or action_width % 7:
        raise ValueError("action_width must describe a positive number of seven-dimensional actions")
    if not 1 <= action_motor_dimensions <= 6:
        raise ValueError("action_motor_dimensions must select motor dimensions only (1..6)")
    action_student = student.action.reshape(student.action.shape[0], -1)[:, :action_width]
    action_teacher = teacher.action.reshape(teacher.action.shape[0], -1)[:, :action_width]
    if action_student.shape != action_teacher.shape:
        raise ValueError("Student and teacher action-prefix shapes differ")
    action_student = action_student.reshape(action_student.shape[0], -1, 7).float()
    action_teacher = action_teacher.reshape(action_teacher.shape[0], -1, 7).float()
    action_mse = (action_student[..., :action_motor_dimensions]
                  - action_teacher[..., :action_motor_dimensions]).square().mean()
    future_losses = []
    for name in ("future_wrist", "future_image"):
        estimate = student.future_image_slots[name].float()
        target = teacher.future_image_slots[name].float()
        if estimate.shape != target.shape:
            raise ValueError(f"Student and teacher {name} latent shapes differ")
        denominator = target.square().mean().clamp_min(torch.finfo(torch.float32).eps)
        future_losses.append((estimate - target).square().mean() / denominator)
    return action_mse, torch.stack(future_losses).mean()


def endpoint_gripper_mse(student: EndpointResult, teacher: EndpointResult, *,
                         action_width: int = 16 * 7):
    """Report continuous gripper error separately; it never enters VQ loss."""
    torch = _torch()
    if action_width <= 0 or action_width % 7:
        raise ValueError("action_width must describe a positive number of seven-dimensional actions")
    action_student = student.action.reshape(student.action.shape[0], -1)[:, :action_width]
    action_teacher = teacher.action.reshape(teacher.action.shape[0], -1)[:, :action_width]
    if action_student.shape != action_teacher.shape:
        raise ValueError("Student and teacher action-prefix shapes differ")
    action_student = action_student.reshape(action_student.shape[0], -1, 7).float()
    action_teacher = action_teacher.reshape(action_teacher.shape[0], -1, 7).float()
    return (action_student[..., 6] - action_teacher[..., 6]).square().mean()


def denoiser_nmse(model, packet: dict[str, Any], device):
    """Full shared-DiT x0 NMSE for one frozen noisy input/sigma/condition packet."""
    torch = _torch()
    x = packet["x"].to(device=device)
    sigma = packet["sigma"].to(device=device)
    condition = _tree_copy_to_device(packet["condition"], device, torch)
    target = packet["target"].to(device=device, dtype=torch.float32)
    raw_denoise = inspect.unwrap(type(model).denoise)
    output = raw_denoise(model, x, sigma, condition)
    estimate = _denoised_tensor(output).float()
    if estimate.shape != target.shape:
        raise ValueError("Current denoiser output differs from saved BF16 teacher packet shape")
    return (estimate - target).square().mean() / target.square().mean().clamp_min(
        torch.finfo(torch.float32).eps
    )


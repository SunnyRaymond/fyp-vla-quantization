"""Differentiable future-to-action sampling with the original BF16 reader."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import inspect
import os
from pathlib import Path
import socket
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


ACTION_HORIZON = 32
VIDEO_FRAMES = 9
INFERENCE_STEPS = 10
FUTURE_SHAPE = (1, 48, 3, 14, 28)


def _require_compute_allocation() -> None:
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("Approved PBS compute allocation required for BF16 reader sampling")
    nodes = {
        line.split(".")[0]
        for line in Path(nodefile).read_text(encoding="utf-8").split()
    }
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")


def _raw_idm_infer(model):
    """Return FastWAM's undecorated sampler so its inference guard is bypassed."""
    for cls in type(model).__mro__:
        method = cls.__dict__.get("infer_action")
        if method is None:
            continue
        parameters = inspect.signature(method).parameters
        if "num_video_frames" in parameters and "action_infer_mode" not in parameters:
            return inspect.unwrap(method)
    raise TypeError("Expected a FastWAMIDM-compatible infer_action method")


def _regular_inputs(datum: dict) -> dict:
    """Clone tensors made under inference_mode before using them with autograd."""
    with torch.inference_mode(False):
        return {
            key: value.clone() if torch.is_inference(value) else value
            for key, value in (("image", datum["image"]), ("proprio", datum["proprio"]))
        }


def _inference_kwargs(runtime, datum: dict, seed: int) -> dict:
    return {
        "prompt": runtime.ev.DEFAULT_PROMPT.format(task=runtime.description),
        "input_image": datum["image"],
        "proprio": datum["proprio"],
        "action_horizon": ACTION_HORIZON,
        "num_video_frames": VIDEO_FRAMES,
        "num_inference_steps": INFERENCE_STEPS,
        "sigma_shift": 1.0,
        "seed": int(seed),
        "rand_device": "cpu",
        "tiled": False,
        "compile_action_infer": False,
    }


def _patch(obj, name, replacement, patches) -> None:
    namespace = getattr(obj, "__dict__", {})
    existed = name in namespace
    previous = namespace.get(name)
    patches.append((obj, name, existed, previous))
    setattr(obj, name, replacement)


def _restore(patches) -> None:
    for obj, name, existed, previous in reversed(patches):
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


@contextmanager
def _bf16_forward_context(modules, original_weights):
    """Temporarily route target Linear calls through retained BF16 tensors."""
    patches = []
    try:
        for name, module in modules.items():
            weight = original_weights[name]

            def forward(x, _weight=weight, _bias=module.bias):
                return torch.nn.functional.linear(x, _weight, _bias)

            _patch(module, "forward", forward, patches)
        yield
    finally:
        _restore(patches)


@contextmanager
def _recompute_bf16_context(modules, original_weights, state):
    state["recompute_depth"] += 1
    try:
        with _bf16_forward_context(modules, original_weights):
            yield
    finally:
        state["recompute_depth"] -= 1


def _checkpoint_context_fn(modules, original_weights, state):
    return lambda: (
        nullcontext(),
        _recompute_bf16_context(modules, original_weights, state),
    )


class _FutureInjector:
    """Clamp frame zero from the observation and inject a connected future."""

    def __init__(self, future):
        self.future = future
        self.calls = 0
        self.observed_first_frame = None
        self.last_output = None

    def __call__(self, _prediction, _delta, sample):
        if self.observed_first_frame is None:
            self.observed_first_frame = sample[:, :, 0:1].detach().clone()
        injected = torch.cat((sample[:, :, 0:1], self.future[:, :, 1:]), dim=2)
        self.calls += 1
        self.last_output = injected
        return injected


class BF16Reader:
    """Run the 10-step IDM action reader using retained, decoded BF16 weights."""

    def __init__(self, runtime, original_weights):
        self.runtime = runtime
        self.modules = runtime.q.modules
        if len(self.modules) != 614:
            raise RuntimeError(f"Expected 614 target Linears, got {len(self.modules)}")
        if set(original_weights) != set(self.modules):
            raise RuntimeError("Original BF16 weight names differ from target Linear coverage")
        self.original_weights = dict(original_weights)
        for name, module in self.modules.items():
            weight = self.original_weights[name]
            if (not isinstance(weight, torch.Tensor) or weight.requires_grad or weight.grad_fn is not None
                    or weight.device.type != "cuda" or weight.dtype != torch.bfloat16
                    or tuple(weight.shape) != tuple(module.weight.shape)):
                raise ValueError(f"{name}: expected a detached GPU BF16 weight with the module shape")
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, runtime, datum, seed, future, checkpointed: bool = True):
        _require_compute_allocation()
        if runtime is not self.runtime:
            raise ValueError("BF16Reader must be used with the runtime it was constructed for")
        if not isinstance(getattr(runtime, "description", None), str):
            raise ValueError("Runtime must have the fixed task description set by runtime.task()")
        use_checkpoint = bool(checkpointed)
        if (not isinstance(future, torch.Tensor) or tuple(future.shape) != FUTURE_SHAPE
                or (use_checkpoint and (not future.requires_grad or torch.is_inference(future)))):
            requirement = "grad-connected regular tensor" if use_checkpoint else "tensor"
            raise ValueError(f"future must be a {requirement} with shape {FUTURE_SHAPE}")

        model = runtime.model
        device = getattr(model, "device", self.original_weights[next(iter(self.modules))].device)
        dtype = getattr(model, "torch_dtype", torch.bfloat16)
        query_datum = _regular_inputs(datum)
        kwargs = _inference_kwargs(runtime, query_datum, seed)
        grad_mode = torch.enable_grad if use_checkpoint else torch.no_grad
        with torch.inference_mode(False), grad_mode():
            future_on_device = future.to(device=device, dtype=dtype)
        state = {
            "action": None,
            "action_initial_noise": None,
            "action_conditioning_latents": None,
            "action_steps": 0,
            "action_dit_calls": 0,
            "video_loop_calls": 0,
            "prefill_calls": 0,
            "recompute_depth": 0,
        }
        injector = _FutureInjector(future_on_device)
        patches = []
        contexts = _checkpoint_context_fn(self.modules, self.original_weights, state)

        def skip_video_dit(*, latents_video, **_kwargs):
            state["video_loop_calls"] += 1
            return None

        _patch(model, "_denoise_video", skip_video_dit, patches)
        _patch(model.infer_video_scheduler, "step", injector, patches)

        original_action = model._denoise_action_with_video_cache
        def denoise_action(*, latents_action, timestep_action, context, context_mask,
                           video_cache_k, video_cache_v, action_attention_mask):
            if state["recompute_depth"] == 0:
                if state["action_initial_noise"] is None:
                    state["action_initial_noise"] = latents_action.detach().to(
                        device="cpu", dtype=torch.float32
                    ).clone()
                state["action_dit_calls"] += 1

            def run(x, timestep, prompt_context, prompt_mask, *flat_cache):
                return original_action(
                    latents_action=x,
                    timestep_action=timestep,
                    context=prompt_context,
                    context_mask=prompt_mask,
                    video_cache_k=list(flat_cache[:layer_count]),
                    video_cache_v=list(flat_cache[layer_count:]),
                    action_attention_mask=action_attention_mask,
                )

            args = (latents_action, timestep_action, context, context_mask,
                    *video_cache_k, *video_cache_v)
            return checkpoint(run, *args, use_reentrant=False, context_fn=contexts) if use_checkpoint else run(*args)

        _patch(model, "_denoise_action_with_video_cache", denoise_action, patches)

        original_prefill = model.mot.prefill_video_cache_tensor
        layer_count = len(model.video_expert.blocks)

        def prefill_video_cache(*, video_tokens, video_freqs, video_t_mod, video_context,
                                video_context_mask, video_attention_mask):
            state["prefill_calls"] += 1

            def run(tokens, freqs, t_mod, prompt_context, prompt_mask, attention_mask):
                keys, values = original_prefill(
                    video_tokens=tokens,
                    video_freqs=freqs,
                    video_t_mod=t_mod,
                    video_context=prompt_context,
                    video_context_mask=prompt_mask,
                    video_attention_mask=attention_mask,
                )
                return tuple(keys) + tuple(values)

            args = (video_tokens, video_freqs, video_t_mod, video_context,
                    video_context_mask, video_attention_mask)
            cache = checkpoint(run, *args, use_reentrant=False, context_fn=contexts) if use_checkpoint else run(*args)
            return list(cache[:layer_count]), list(cache[layer_count:])

        _patch(model.mot, "prefill_video_cache_tensor", prefill_video_cache, patches)

        original_prepare = model.video_expert.prepare

        def capture_action_condition(*args, **prepare_kwargs):
            result = original_prepare(*args, **prepare_kwargs)
            conditioning = prepare_kwargs.get("x", args[0] if args else None)
            if conditioning is None:
                raise RuntimeError("Could not capture action-conditioning video latents")
            state["action_conditioning_latents"] = conditioning.detach().clone()
            return result

        _patch(model.video_expert, "prepare", capture_action_condition, patches)

        original_action_step = model.infer_action_scheduler.step

        def action_step(model_output, delta, sample):
            result = original_action_step(model_output, delta, sample)
            state["action_steps"] += 1
            if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, 32, 7):
                raise RuntimeError("Expected action scheduler output with shape [1,32,7]")
            state["action"] = result[0]
            return result

        _patch(model.infer_action_scheduler, "step", action_step, patches)

        if getattr(model, "proprio_encoder", None) is not None:
            bound = getattr(model, "_append_proprio_to_context", None)
            function = getattr(bound, "__func__", bound)
            if function is not None and hasattr(function, "__wrapped__"):
                _patch(
                    model,
                    "_append_proprio_to_context",
                    MethodType(inspect.unwrap(function), model),
                    patches,
                )

        try:
            raw_infer = _raw_idm_infer(model)
            with torch.inference_mode(False), grad_mode(), _bf16_forward_context(
                self.modules, self.original_weights
            ):
                raw_infer(model, **kwargs)
                action = state["action"]
                if action is None or tuple(action.shape) != (ACTION_HORIZON, 7):
                    raise RuntimeError("Expected final normalized action with shape [32,7]")
                if use_checkpoint and not action.requires_grad:
                    raise RuntimeError("Final action lost its autograd graph")
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("Final action contains non-finite values")
                if (injector.calls != INFERENCE_STEPS or state["video_loop_calls"] != INFERENCE_STEPS
                        or state["action_steps"] != INFERENCE_STEPS
                        or state["action_dit_calls"] != INFERENCE_STEPS
                        or state["prefill_calls"] != 1):
                    raise RuntimeError(
                        "Unexpected IDM call counts: "
                        f"video_steps={injector.calls}, video_loops={state['video_loop_calls']}, "
                        f"action_steps={state['action_steps']}, action_dit={state['action_dit_calls']}, "
                        f"prefills={state['prefill_calls']}"
                    )
                if state["action_initial_noise"] is None:
                    raise RuntimeError("Could not capture the initial action noise for parity")
                if state["action_conditioning_latents"] is None:
                    raise RuntimeError("Could not capture action-conditioning video latents")

                output = injector.last_output
                conditioning = state["action_conditioning_latents"].to(
                    device=future_on_device.device, dtype=future_on_device.dtype
                )
                tail_matches = torch.equal(output[:, :, 1:], future_on_device[:, :, 1:])
                frame_zero_clamped = torch.equal(
                    output[:, :, 0:1], injector.observed_first_frame
                )
                conditioning_tail_matches = torch.equal(
                    conditioning[:, :, 1:], future_on_device[:, :, 1:]
                )
                conditioning_frame_zero_clamped = torch.equal(
                    conditioning[:, :, 0:1], injector.observed_first_frame
                )
                supplied_first_frame_matches = torch.equal(
                    future_on_device[:, :, 0:1], injector.observed_first_frame
                )
                if not all((tail_matches, frame_zero_clamped, conditioning_tail_matches,
                            conditioning_frame_zero_clamped, supplied_first_frame_matches)):
                    raise RuntimeError("Future injection or observation-frame clamp verification failed")

            counts = {
                "video_dit_calls": 0,
                "video_denoise_loop_calls": state["video_loop_calls"],
                "video_scheduler_steps": injector.calls,
                "action_dit_calls": state["action_dit_calls"],
                "action_scheduler_steps": state["action_steps"],
                "video_prefill_calls": state["prefill_calls"],
                "teacher_future_injection_steps": injector.calls,
                "teacher_future_injection_verified": True,
                "future_tail_matches": tail_matches,
                "first_frame_clamp_verified": frame_zero_clamped,
                "action_conditioning_future_matches": conditioning_tail_matches,
                "action_conditioning_first_frame_clamped": conditioning_frame_zero_clamped,
                "future_first_frame_matches_observation": supplied_first_frame_matches,
                "action_initial_noise": state["action_initial_noise"],
                "action_conditioning_latents": conditioning.detach().to(
                    device="cpu", dtype=torch.float32
                ).clone(),
            }
            return action, counts
        finally:
            _restore(patches)


def capture_original_weights(runtime):
    """Capture detached references before installing ChannelBanks or another wrapper."""
    return {name: module.weight.detach() for name, module in runtime.q.modules.items()}

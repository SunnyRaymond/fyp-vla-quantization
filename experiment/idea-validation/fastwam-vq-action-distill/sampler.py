"""Differentiable Fast-WAM Optional-IDM action sampler for PBS experiments."""
from __future__ import annotations

from contextlib import nullcontext
import inspect
import os
from pathlib import Path
import socket
from types import MethodType


def _raw_idm_infer(model):
    """Return the undecorated FastWAMIDM sampler, bypassing inference no-grad."""
    for cls in type(model).__mro__:
        method = cls.__dict__.get("infer_action")
        if method is None:
            continue
        parameters = inspect.signature(method).parameters
        if "num_video_frames" in parameters and "action_infer_mode" not in parameters:
            return inspect.unwrap(method)
    raise TypeError("Expected a FastWAMIDM-compatible infer_action method")


def _regular_inputs(torch, datum: dict) -> dict:
    """Clone tensors created under inference_mode before entering autograd."""
    with torch.inference_mode(False):
        return {
            key: value.clone() if value.is_inference() else value
            for key, value in (("image", datum["image"]), ("proprio", datum["proprio"]))
        }


def _require_compute_allocation() -> None:
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("Approved PBS compute allocation required for differentiable sampling")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")


def differentiable_action(runtime, datum: dict, seed: int, *, checkpointed: bool = True):
    """Run the fixed 10+10-step IDM sampler and return its connected [32,7] action.

    With ``checkpointed=True``, non-reentrant checkpoints bound denoising
    activation memory and the returned action remains connected for caller
    backward. ``checkpointed=False`` runs the same forward under no-grad for
    parity checks. Returns ``(action, {"video_steps": 10, "action_steps": 10})``.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    _require_compute_allocation()
    model = runtime.model
    if not isinstance(getattr(runtime, "description", None), str):
        raise ValueError("Runtime must have the fixed task description set by runtime.task()")

    query_datum = _regular_inputs(torch, datum)
    kwargs = {
        "prompt": runtime.ev.DEFAULT_PROMPT.format(task=runtime.description),
        "input_image": query_datum["image"],
        "proprio": query_datum["proprio"],
        "action_horizon": 32,
        "num_video_frames": 9,
        "num_inference_steps": 10,
        "sigma_shift": 1.0,
        "seed": int(seed),
        "rand_device": "cpu",
        "tiled": False,
        "compile_action_infer": False,
    }

    state = {"action": None, "video_steps": 0, "action_steps": 0}
    patches = []

    def patch(obj, name, replacement):
        namespace = getattr(obj, "__dict__", {})
        existed = name in namespace
        previous = namespace.get(name)
        patches.append((obj, name, existed, previous))
        setattr(obj, name, replacement)

    use_checkpoint = bool(checkpointed)
    contexts = lambda: (nullcontext(), nullcontext())

    original_video = model._denoise_video

    def denoise_video(*, latents_video, timestep_video, context, context_mask,
                      video_self_attn_mask, fuse_vae_embedding_in_latents):
        def run(x, timestep, prompt_context, prompt_mask):
            return original_video(
                latents_video=x,
                timestep_video=timestep,
                context=prompt_context,
                context_mask=prompt_mask,
                video_self_attn_mask=video_self_attn_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )

        if use_checkpoint:
            return checkpoint(
                run, latents_video, timestep_video, context, context_mask,
                use_reentrant=False, context_fn=contexts,
            )
        return run(latents_video, timestep_video, context, context_mask)

    patch(model, "_denoise_video", denoise_video)

    original_action = model._denoise_action_with_video_cache

    def denoise_action(*, latents_action, timestep_action, context, context_mask,
                       video_cache_k, video_cache_v, action_attention_mask):
        layer_count = len(video_cache_k)

        def run(x, timestep, prompt_context, prompt_mask, *flat_cache):
            keys = list(flat_cache[:layer_count])
            values = list(flat_cache[layer_count:])
            return original_action(
                latents_action=x,
                timestep_action=timestep,
                context=prompt_context,
                context_mask=prompt_mask,
                video_cache_k=keys,
                video_cache_v=values,
                action_attention_mask=action_attention_mask,
            )

        if use_checkpoint:
            return checkpoint(
                run, latents_action, timestep_action, context, context_mask,
                *video_cache_k, *video_cache_v,
                use_reentrant=False, context_fn=contexts,
            )
        return run(latents_action, timestep_action, context, context_mask,
                   *video_cache_k, *video_cache_v)

    patch(model, "_denoise_action_with_video_cache", denoise_action)

    original_prefill = model.mot.prefill_video_cache_tensor
    layer_count = len(model.video_expert.blocks)

    def prefill_video_cache(*, video_tokens, video_freqs, video_t_mod, video_context,
                            video_context_mask, video_attention_mask):
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

        cache = checkpoint(
            run, video_tokens, video_freqs, video_t_mod, video_context,
            video_context_mask, video_attention_mask,
            use_reentrant=False, context_fn=contexts,
        ) if use_checkpoint else run(
            video_tokens, video_freqs, video_t_mod, video_context,
            video_context_mask, video_attention_mask,
        )
        return list(cache[:layer_count]), list(cache[layer_count:])

    patch(model.mot, "prefill_video_cache_tensor", prefill_video_cache)

    # This method currently has no no-grad decorator, but it runs the target
    # proprio_encoder Linear; unwrap it if the vendored revision adds one.
    if getattr(model, "proprio_encoder", None) is not None:
        bound = getattr(model, "_append_proprio_to_context", None)
        function = getattr(bound, "__func__", bound)
        if function is not None and hasattr(function, "__wrapped__"):
            patch(model, "_append_proprio_to_context", MethodType(inspect.unwrap(function), model))

    original_video_step = model.infer_video_scheduler.step

    def video_step(model_output, delta, sample):
        result = original_video_step(model_output, delta, sample)
        state["video_steps"] += 1
        return result

    patch(model.infer_video_scheduler, "step", video_step)

    original_action_step = model.infer_action_scheduler.step

    def action_step(model_output, delta, sample):
        result = original_action_step(model_output, delta, sample)
        state["action_steps"] += 1
        if not isinstance(result, torch.Tensor) or result.ndim != 3 or result.shape[0] != 1:
            raise RuntimeError("Expected batched action scheduler output with shape [1,32,7]")
        state["action"] = result[0]
        return result

    patch(model.infer_action_scheduler, "step", action_step)

    try:
        raw_infer = _raw_idm_infer(model)
        grad_context = torch.enable_grad() if use_checkpoint else torch.no_grad()
        with torch.inference_mode(False), grad_context:
            raw_infer(model, **kwargs)
            action = state["action"]
            if action is None or tuple(action.shape) != (32, 7):
                raise RuntimeError("Expected final normalized action with shape [32,7]")
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError("Final action contains non-finite values")
            if state["video_steps"] != 10 or state["action_steps"] != 10:
                raise RuntimeError(
                    f"Expected 10 video and 10 action steps, got "
                    f"{state['video_steps']} and {state['action_steps']}"
                )
            if use_checkpoint and not action.requires_grad:
                raise RuntimeError("Final action lost its autograd graph")
        return action, {"video_steps": state["video_steps"], "action_steps": state["action_steps"]}
    finally:
        for obj, name, existed, previous in reversed(patches):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)

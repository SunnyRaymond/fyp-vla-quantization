#!/usr/bin/env python3
"""Measure the official FastWAM LIBERO single-episode path without source edits.

This runner must execute inside an approved PBS compute allocation. It imports
the read-only source checkout selected by FASTWAM_SOURCE and passes the normal
Hydra overrides through to experiments/libero/eval_libero_single.py.
"""

from __future__ import annotations

# Guard before importing torch, Hydra, or any model code. Login-node use is
# rejected even when the user starts this script through a background shell.
import os
import socket
import sys
import time
from pathlib import Path


EXPECTED_REVISION = "7faa71108368fbb3b6885649f112af607427a2d4"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"required PBS/profile environment variable is empty: {name}")
    return value


def _allocation_context() -> tuple[str, str, Path, Path]:
    job_id = _required_env("PBS_JOBID")
    nodefile = Path(_required_env("PBS_NODEFILE"))
    hostname = socket.gethostname()
    if "login" in hostname.lower():
        raise SystemExit(f"refusing model work on login host: {hostname}")
    if not nodefile.is_file():
        raise SystemExit(f"PBS_NODEFILE is not a readable allocation file: {nodefile}")
    source = Path(_required_env("FASTWAM_SOURCE")).resolve()
    artifacts = Path(_required_env("ARTIFACTS")).resolve()
    if not (source / "experiments/libero/eval_libero_single.py").is_file():
        raise SystemExit(f"FASTWAM_SOURCE does not contain the official LIBERO evaluator: {source}")
    # The prepared source is the official pinned archive, with its existing marker.
    revision = (source / ".fastwam-revision").read_text(encoding="utf-8").strip()
    if revision != EXPECTED_REVISION:
        raise SystemExit(f"FastWAM source revision mismatch: expected {EXPECTED_REVISION}, got {revision}")
    return job_id, hostname, source, artifacts


PBS_JOB_ID, HOSTNAME, FASTWAM_SOURCE, ARTIFACTS = _allocation_context()
sys.path.insert(0, str(FASTWAM_SOURCE / "experiments/libero"))
sys.path.insert(0, str(FASTWAM_SOURCE))

import copy
import functools
import json
import random
import statistics
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.libero import eval_libero_single as official_eval
from fastwam.models.wan22 import mot as mot_module


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise SystemExit(f"{name} must be >= {minimum}, got {value}")
    return value


WARMUPS = _env_int("FW_PROFILE_WARMUPS", 3)
REPEATS = _env_int("FW_PROFILE_REPEATS", 4, minimum=1)
CONTEXTS = _env_int("FW_PROFILE_CONTEXTS", 3, minimum=3)
TRACE_CALLS = _env_int("FW_PROFILE_TRACE_CALLS", 1)
COARSE_REPEATS = _env_int("FW_PROFILE_COARSE_REPEATS", 2, minimum=1)


def _tensor_shapes(value: Any, limit: int = 8) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if len(found) >= limit:
            return
        if torch.is_tensor(item):
            found.append(
                {
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "device": str(item.device),
                }
            )
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return found


def _snapshot_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(child) for child in value)
    if isinstance(value, list):
        return [_snapshot_tree(child) for child in value]
    return value


def _compare_tree(left: Any, right: Any, path: str = "output") -> dict[str, Any]:
    if torch.is_tensor(left):
        left = left.detach().cpu().numpy()
    if torch.is_tensor(right):
        right = right.detach().cpu().numpy()
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape:
            return {"path": path, "allclose": False, "reason": "shape_mismatch", "left": list(a.shape), "right": list(b.shape)}
        if a.dtype.kind in "fciub" and b.dtype.kind in "fciub":
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            return {
                "path": path,
                "allclose": finite and bool(np.allclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=False)),
                "finite": finite,
                "max_abs_diff": float(delta.max()) if delta.size else 0.0,
                "shape": list(a.shape),
            }
        return {"path": path, "allclose": bool(np.array_equal(a, b)), "shape": list(a.shape)}
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return {"path": path, "allclose": False, "reason": "key_mismatch"}
        children = [_compare_tree(left[key], right[key], f"{path}.{key}") for key in left]
        return {"path": path, "allclose": all(row["allclose"] for row in children), "children": children}
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return {"path": path, "allclose": False, "reason": "length_mismatch"}
        children = [_compare_tree(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right))]
        return {"path": path, "allclose": all(row["allclose"] for row in children), "children": children}
    return {"path": path, "allclose": left == right}


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "min": None, "max": None, "stdev": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


class Runner:
    def __init__(self, model: torch.nn.Module, cfg: DictConfig, bootstrap: list[dict[str, Any]]):
        self.model = model
        self.cfg = cfg
        self.bootstrap = bootstrap
        self.phase = "episode"
        self.stage = "episode"
        self.context_index: int | None = None
        self.pair_index: int | None = None
        self.denoise_step: int | None = None
        self.video_seq_len: int | None = None
        self.rows: list[dict[str, Any]] = []
        self.pending_events: list[tuple[dict[str, Any], Any, Any]] = []
        self.episode_calls: list[dict[str, Any]] = []
        self.context_rows: list[dict[str, Any]] = []
        self.native_pairs: list[dict[str, Any]] = []
        self.coarse_pairs: list[dict[str, Any]] = []
        self.attention_calls: list[dict[str, Any]] = []
        self.coarse_attention_calls: list[dict[str, Any]] = []
        self.qkv_shapes: list[dict[str, Any]] = []
        self.parity: list[dict[str, Any]] = []
        self.module_handles: list[Any] = []
        self.patches: list[tuple[Any, str, Any]] = []
        self.fine_patch_start: int | None = None
        self.call_counter = 0
        self.infer_output: Any = None
        self.trace_active = False
        self._original_predict = official_eval._predict_action_chunk
        self._original_get_env = official_eval.get_libero_env
        self._original_save_rollout = official_eval.save_rollout_video
        self._original_obs_input = official_eval._obs_to_model_input
        self._original_denorm = official_eval._denormalize_action
        self._original_invert_gripper = official_eval.invert_gripper_action
        self._original_infer = model.infer_action
        self._original_encode_prompt = model.encode_prompt
        self._original_vae_input_encode = model._encode_input_image_latents_tensor
        self._original_video_prefill = model.mot.prefill_video_cache_tensor
        self._original_action_denoise = model._denoise_action_with_video_cache

    def _record(self, name: str, start_ns: int, *, detail: dict[str, Any] | None = None, cuda_start=None, cuda_end=None) -> None:
        row: dict[str, Any] = {
            "name": name,
            "pass": self.phase,
            "stage": self.stage,
            "context_index": self.context_index,
            "pair_index": self.pair_index,
            "denoise_step": self.denoise_step,
            "cpu_wall_ms": (time.perf_counter_ns() - start_ns) / 1e6,
            "inclusive": True,
        }
        if detail:
            row.update(detail)
        if cuda_start is not None:
            self.pending_events.append((row, cuda_start, cuda_end))
        self.rows.append(row)

    @contextmanager
    def span(self, name: str, *, detail: dict[str, Any] | None = None, cuda: bool = True):
        active = self.phase in {"replay_instrumented", "replay_coarse_instrumented", "replay_native", "trace"}
        detailed = self.phase in {"replay_instrumented", "replay_coarse_instrumented"}
        if not active or (not detailed and not cuda and not self.trace_active):
            yield
            return
        start_ns = time.perf_counter_ns()
        use_cuda = cuda and torch.cuda.is_available() and detailed
        event_start = event_end = None
        if use_cuda:
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
        record_scope = None
        if self.trace_active:
            record_scope = torch.profiler.record_function(f"FW/{name}")
            record_scope.__enter__()
        try:
            yield
        finally:
            if record_scope is not None:
                record_scope.__exit__(None, None, None)
            if event_end is not None:
                event_end.record()
            if detailed or name == "model.infer_action":
                self._record(name, start_ns, detail=detail, cuda_start=event_start, cuda_end=event_end)

    @contextmanager
    def _stage(self, stage: str, denoise_step: int | None = None):
        old_stage, old_step = self.stage, self.denoise_step
        self.stage, self.denoise_step = stage, denoise_step
        try:
            yield
        finally:
            self.stage, self.denoise_step = old_stage, old_step

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def install_coarse(self) -> None:
        runner = self

        @functools.wraps(self._original_predict)
        def predict(*args, **kwargs):
            if runner.phase == "episode":
                capture_started = time.perf_counter_ns()
                saved_obs = copy.deepcopy(kwargs["obs"])
                capture_ms = (time.perf_counter_ns() - capture_started) / 1e6
                started = time.perf_counter_ns()
                result = runner._original_predict(*args, **kwargs)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                runner.episode_calls.append(
                    {
                        "replan_index": len(runner.episode_calls),
                        "cpu_wall_ms": elapsed,
                        "capture_copy_ms_excluded": capture_ms,
                        "obs": saved_obs,
                        "action": np.asarray(result[0]).copy(),
                    }
                )
                return result
            started = time.perf_counter_ns()
            result = runner._original_predict(*args, **kwargs)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            runner.last_predict_ms = elapsed
            return result

        @functools.wraps(self._original_obs_input)
        def obs_input(*args, **kwargs):
            if runner.phase != "episode":
                with runner.span("preprocess.obs_to_model_input", cuda=False):
                    return runner._original_obs_input(*args, **kwargs)
            start = time.perf_counter_ns()
            result = runner._original_obs_input(*args, **kwargs)
            runner._record("preprocess.obs_to_model_input", start)
            return result

        @functools.wraps(self._original_denorm)
        def denorm(*args, **kwargs):
            if runner.phase != "episode":
                with runner.span("postprocess.denormalize_action", cuda=False):
                    return runner._original_denorm(*args, **kwargs)
            start = time.perf_counter_ns()
            result = runner._original_denorm(*args, **kwargs)
            runner._record("postprocess.denormalize_action", start)
            return result

        @functools.wraps(self._original_invert_gripper)
        def invert_gripper(*args, **kwargs):
            if runner.phase != "episode":
                with runner.span("postprocess.invert_gripper", cuda=False):
                    return runner._original_invert_gripper(*args, **kwargs)
            start = time.perf_counter_ns()
            result = runner._original_invert_gripper(*args, **kwargs)
            runner._record("postprocess.invert_gripper", start)
            return result

        @functools.wraps(self._original_save_rollout)
        def save_rollout(*args, **kwargs):
            start = time.perf_counter_ns()
            result = runner._original_save_rollout(*args, **kwargs)
            runner._record("video_output.save_rollout_video", start)
            return result

        @functools.wraps(self._original_get_env)
        def get_env(*args, **kwargs):
            start = time.perf_counter_ns()
            env, description = runner._original_get_env(*args, **kwargs)
            runner._record("environment.create", start)
            for method_name in ("reset", "set_init_state", "close"):
                if not hasattr(env, method_name):
                    continue
                original = getattr(env, method_name)

                @functools.wraps(original)
                def timed_env_method(*m_args, __original=original, __name=method_name, **m_kwargs):
                    tick = time.perf_counter_ns()
                    value = __original(*m_args, **m_kwargs)
                    runner._record(f"environment.{__name}", tick)
                    return value

                runner._patch(env, method_name, timed_env_method)
            if hasattr(env, "step"):
                original_step = env.step

                @functools.wraps(original_step)
                def timed_step(*s_args, **s_kwargs):
                    tick = time.perf_counter_ns()
                    value = original_step(*s_args, **s_kwargs)
                    runner._record("environment.step", tick)
                    return value

                runner._patch(env, "step", timed_step)
            render_candidates = []
            current = env
            for _ in range(5):
                if current is None:
                    break
                sim = getattr(current, "sim", None)
                if sim is not None and callable(getattr(sim, "render", None)):
                    render_candidates.append(sim)
                current = getattr(current, "env", None)
            self_seen: set[int] = set()
            runner.render_hook_count = 0
            for sim in render_candidates:
                if id(sim) in self_seen:
                    continue
                self_seen.add(id(sim))
                original_render = sim.render

                @functools.wraps(original_render)
                def timed_render(*r_args, __original=original_render, **r_kwargs):
                    camera = r_kwargs.get("camera_name", r_args[0] if r_args else "default")
                    tick = time.perf_counter_ns()
                    value = __original(*r_args, **r_kwargs)
                    runner._record("environment.sim.render", tick, detail={"camera_name": str(camera)})
                    return value

                try:
                    runner._patch(sim, "render", timed_render)
                    runner.render_hook_count += 1
                except (AttributeError, TypeError):
                    continue
            return env, description

        @functools.wraps(self._original_infer)
        def infer_action(*args, **kwargs):
            if runner.phase == "episode":
                runner.call_counter = 0
                start = time.perf_counter_ns()
                with runner._stage("infer_action"):
                    result = runner._original_infer(*args, **kwargs)
                    runner._record("model.infer_action", start)
                return result
            if runner.phase == "replay_native":
                runner.call_counter = 0
                with runner._stage("infer_action"):
                    with runner.span("model.infer_action", cuda=True):
                        result = runner._original_infer(*args, **kwargs)
                runner.infer_output = _snapshot_tree(result)
                return result
            if runner.phase == "replay_instrumented":
                with runner.span("model.infer_action", cuda=True):
                    result = runner._original_infer(*args, **kwargs)
                runner.infer_output = _snapshot_tree(result)
                return result
            with runner.span("model.infer_action", cuda=False):
                result = runner._original_infer(*args, **kwargs)
            runner.infer_output = _snapshot_tree(result)
            return result

        def coarse_method(owner: Any, name: str, label: str, stage: str, *, step: bool = False, original_override=None):
            original = original_override or getattr(owner, name)

            @functools.wraps(original)
            def timed(*args, **kwargs):
                if runner.phase != "episode":
                    return original(*args, **kwargs)
                denoise_step = None
                if step:
                    denoise_step = runner.call_counter
                    runner.call_counter += 1
                start = time.perf_counter_ns()
                with runner._stage(stage, denoise_step if step else None):
                    result = original(*args, **kwargs)
                    runner._record(label, start, detail={"input_shapes": _tensor_shapes((args, kwargs))})
                return result

            runner._patch(owner, name, timed)

        self._patch(official_eval, "_predict_action_chunk", predict)
        self._patch(official_eval, "_obs_to_model_input", obs_input)
        self._patch(official_eval, "_denormalize_action", denorm)
        self._patch(official_eval, "invert_gripper_action", invert_gripper)
        self._patch(official_eval, "save_rollout_video", save_rollout)
        self._patch(official_eval, "get_libero_env", get_env)
        self._patch(self.model, "infer_action", infer_action)
        coarse_method(self.model, "encode_prompt", "text.encode_prompt", "prompt_encode", original_override=self._original_encode_prompt)
        coarse_method(self.model, "_encode_input_image_latents_tensor", "VAE.input_encode", "vae_encode", original_override=self._original_vae_input_encode)
        coarse_method(self.model.mot, "prefill_video_cache_tensor", "video.cache_prefill", "video_cache_prefill", original_override=self._original_video_prefill)
        coarse_method(self.model, "_denoise_action_with_video_cache", "action.denoise_step", "action_denoise", step=True, original_override=self._original_action_denoise)

    def install_fine(self) -> None:
        if self.fine_patch_start is not None:
            return
        self.fine_patch_start = len(self.patches)
        runner = self
        self.block_ids: dict[int, tuple[str, int]] = {}
        for expert_name in ("video", "action"):
            expert = getattr(self.model, f"{expert_name}_expert")
            for layer_idx, block in enumerate(expert.blocks):
                self.block_ids[id(block)] = (expert_name, layer_idx)

        def method(
            owner: Any,
            name: str,
            label: str,
            stage: str | None = None,
            step: bool = False,
            original_override: Callable[..., Any] | None = None,
        ):
            original = original_override or getattr(owner, name)

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                if runner.phase not in {"replay_instrumented", "replay_native", "trace"}:
                    return original(*args, **kwargs)
                this_step = None
                if step:
                    this_step = runner.call_counter
                    runner.call_counter += 1
                target_stage = stage or runner.stage
                if label == "video.cache_prefill":
                    video_tokens = kwargs.get("video_tokens", args[0] if args else None)
                    if torch.is_tensor(video_tokens):
                        runner.video_seq_len = int(video_tokens.shape[1])
                with runner._stage(target_stage, this_step if step else runner.denoise_step):
                    with runner.span(label, detail={"input_shapes": _tensor_shapes((args, kwargs))}):
                        result = original(*args, **kwargs)
                if label == "model.infer_action":
                    runner.infer_output = _snapshot_tree(result)
                return result

            runner._patch(owner, name, wrapped)

        # The inference entry keeps functools metadata so official inspect.signature checks
        # for action_infer_mode and compile_action_infer continue to see source parameters.
        method(
            self.model,
            "infer_action",
            "model.infer_action",
            stage="infer_action",
            original_override=self._original_infer,
        )
        method(self.model, "_encode_input_image_latents_tensor", "VAE.input_encode", stage="vae_encode", original_override=self._original_vae_input_encode)
        method(self.model, "encode_prompt", "text.encode_prompt", stage="prompt_encode", original_override=self._original_encode_prompt)
        method(self.model, "_denoise_action_with_video_cache", "action.denoise_step", stage="action_denoise", step=True, original_override=self._original_action_denoise)
        method(self.model.mot, "prefill_video_cache_tensor", "video.cache_prefill", stage="video_cache_prefill", original_override=self._original_video_prefill)
        method(self.model.video_expert, "prepare", "video.prepare", stage="video_prepare")
        method(self.model.action_expert, "prepare", "action.prepare", stage="action_prepare")
        method(self.model.action_expert, "post", "action.post", stage="action_post")
        method(self.model.infer_action_scheduler, "step", "action.scheduler_step", stage="action_scheduler_step")

        original_build = self.model.mot._build_expert_attention_io

        @functools.wraps(original_build)
        def build_attention_io(*args, **kwargs):
            expert = kwargs.get("expert", args[0] if args else None)
            block = kwargs.get("block", args[1] if len(args) > 1 else None)
            expert_name, layer_idx = self.block_ids.get(id(block), ("unknown", -1))
            label = f"MoT.{expert_name}.layer{layer_idx}.qkv_rope"
            with self.span(label, detail={"layer": layer_idx, "expert": expert_name}):
                result = original_build(*args, **kwargs)
            if self.phase == "replay_instrumented":
                self.qkv_shapes.append(
                    {
                        "context_index": self.context_index,
                        "pair_index": self.pair_index,
                        "stage": self.stage,
                        "expert": expert_name,
                        "layer": layer_idx,
                        "q": list(result[0].shape),
                        "k": list(result[1].shape),
                        "v": list(result[2].shape),
                        "source": "first_frame_observation" if expert_name == "video" else "noisy_action_tokens",
                    }
                )
            return result

        self._patch(self.model.mot, "_build_expert_attention_io", build_attention_io)

        original_flash = mot_module.flash_attention

        @functools.wraps(original_flash)
        def flash_attention(*args, **kwargs):
            q = kwargs.get("q", args[0] if args else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            v = kwargs.get("v", args[2] if len(args) > 2 else None)
            heads = int(kwargs.get("num_heads", args[3] if len(args) > 3 else self.model.mot.num_heads))
            mask = kwargs.get("ctx_mask", args[4] if len(args) > 4 else None)
            q_shape, k_shape, v_shape = list(q.shape), list(k.shape), list(v.shape)
            stage = self.stage
            query_tokens = int(q_shape[1])
            key_tokens = int(k_shape[1])
            head_dim = int(q_shape[-1]) // heads
            batch = int(q_shape[0])
            detail: dict[str, Any] = {
                "attention_stage": stage,
                "q_shape": q_shape,
                "k_shape": k_shape,
                "v_shape": v_shape,
                "mask_shape": list(mask.shape) if torch.is_tensor(mask) else None,
                "heads": heads,
                "head_dim": head_dim,
                "attention_backend": "torch.nn.functional.scaled_dot_product_attention; concrete backend is trace-only",
            }
            if stage == "action_denoise" and self.video_seq_len is not None:
                video_keys = min(self.video_seq_len, key_tokens)
                action_keys = max(0, key_tokens - video_keys)
                obs_pairs = query_tokens * video_keys
                action_pairs = query_tokens * action_keys
                detail.update(
                    {
                        "key_groups": {"first_frame_observation_cache": video_keys, "current_action_tokens": action_keys},
                        "flops_qk_plus_av_by_group_mac2": {
                            "observation_cache": 4 * batch * heads * head_dim * obs_pairs,
                            "action_tokens": 4 * batch * heads * head_dim * action_pairs,
                        },
                        "flops_qk_plus_av_dense_mac2": 4 * batch * heads * head_dim * query_tokens * key_tokens,
                "timing_scope_note": "SDPA mixed attention is timed as one call; observation-only time is not directly measured.",
                    }
                )
            if self.phase in {"replay_instrumented", "replay_coarse_instrumented"}:
                row = {
                    "context_index": self.context_index,
                    "pair_index": self.pair_index,
                    "denoise_step": self.denoise_step,
                    **detail,
                }
                if self.phase == "replay_instrumented":
                    self.attention_calls.append(row)
                else:
                    self.coarse_attention_calls.append(row)
            with self.span(f"MoT.{stage}.flash_attention", detail=detail, cuda=True):
                return original_flash(*args, **kwargs)

        self._patch(mot_module, "flash_attention", flash_attention)

        # Observe real projection, attention output, text cross-attention, and FFN modules.
        for name, module in self.model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            relevant = (
                ".self_attn.q" in name
                or ".self_attn.k" in name
                or ".self_attn.v" in name
                or ".self_attn.o" in name
                or ".cross_attn" in name
                or ".ffn" in name
                or name == "text_encoder"
            )
            if not relevant:
                continue
            module_detail = {}
            if ".cross_attn" in name:
                module_detail["attention_kind"] = "text_context_cross_attention"
                module_detail["expert"] = "action" if "action_expert" in name else "video"
            starts: list[tuple[int, Any, Any]] = []

            def pre_hook(mod, inputs, __name=name, __starts=starts):
                if runner.phase not in {"replay_instrumented", "trace"}:
                    return
                label = f"module.{__name}"
                start_ns = time.perf_counter_ns()
                use_cuda = runner.phase == "replay_instrumented" and torch.cuda.is_available()
                event_start = torch.cuda.Event(enable_timing=True) if use_cuda else None
                event_end = torch.cuda.Event(enable_timing=True) if use_cuda else None
                if event_start is not None:
                    event_start.record()
                record_scope = None
                if runner.trace_active:
                    record_scope = torch.profiler.record_function(f"FW/{label}")
                    record_scope.__enter__()
                __starts.append((start_ns, event_start, event_end, record_scope, _tensor_shapes(inputs)))

            def post_hook(mod, inputs, output, __name=name, __starts=starts, __module_detail=module_detail):
                if runner.phase not in {"replay_instrumented", "trace"} or not __starts:
                    return
                start_ns, event_start, event_end, record_scope, input_shapes = __starts.pop()
                if record_scope is not None:
                    record_scope.__exit__(None, None, None)
                if event_end is not None:
                    event_end.record()
                if runner.phase == "replay_instrumented":
                    runner._record(
                        f"module.{__name}",
                        start_ns,
                        detail={**__module_detail, "input_shapes": input_shapes, "output_shapes": _tensor_shapes(output)},
                        cuda_start=event_start,
                        cuda_end=event_end,
                    )

            self.module_handles.append(module.register_forward_pre_hook(pre_hook))
            self.module_handles.append(module.register_forward_hook(post_hook))

    def install_thin(self) -> None:
        """Low-overhead same-algorithm pass: phase and SDPA boundaries only."""
        if self.fine_patch_start is not None:
            return
        self.fine_patch_start = len(self.patches)
        runner = self

        def timed_method(owner: Any, name: str, label: str, stage: str, *, step: bool = False, original=None):
            source = original or getattr(owner, name)

            @functools.wraps(source)
            def wrapped(*args, **kwargs):
                if runner.phase != "replay_coarse_instrumented":
                    return source(*args, **kwargs)
                denoise_step = None
                if step:
                    denoise_step = runner.call_counter
                    runner.call_counter += 1
                if label == "video.cache_prefill":
                    tokens = kwargs.get("video_tokens", args[0] if args else None)
                    if torch.is_tensor(tokens):
                        runner.video_seq_len = int(tokens.shape[1])
                with runner._stage(stage, denoise_step if step else None):
                    with runner.span(label, cuda=True):
                        result = source(*args, **kwargs)
                if label == "model.infer_action":
                    runner.infer_output = _snapshot_tree(result)
                return result

            runner._patch(owner, name, wrapped)

        timed_method(self.model, "infer_action", "model.infer_action", "infer_action", original=self._original_infer)
        timed_method(self.model, "_encode_input_image_latents_tensor", "VAE.input_encode", "vae_encode", original=self._original_vae_input_encode)
        timed_method(self.model, "encode_prompt", "text.encode_prompt", "prompt_encode", original=self._original_encode_prompt)
        timed_method(self.model.mot, "prefill_video_cache_tensor", "video.cache_prefill", "video_cache_prefill", original=self._original_video_prefill)
        timed_method(self.model, "_denoise_action_with_video_cache", "action.denoise_step", "action_denoise", step=True, original=self._original_action_denoise)

        original_flash = mot_module.flash_attention

        @functools.wraps(original_flash)
        def timed_flash(*args, **kwargs):
            q = kwargs.get("q", args[0] if args else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            v = kwargs.get("v", args[2] if len(args) > 2 else None)
            heads = int(kwargs.get("num_heads", args[3] if len(args) > 3 else runner.model.mot.num_heads))
            mask = kwargs.get("ctx_mask", args[4] if len(args) > 4 else None)
            q_shape, k_shape, v_shape = list(q.shape), list(k.shape), list(v.shape)
            batch, query_tokens, key_tokens = int(q_shape[0]), int(q_shape[1]), int(k_shape[1])
            head_dim = int(q_shape[-1]) // heads
            detail: dict[str, Any] = {
                "attention_stage": runner.stage,
                "q_shape": q_shape,
                "k_shape": k_shape,
                "v_shape": v_shape,
                "mask_shape": list(mask.shape) if torch.is_tensor(mask) else None,
                "heads": heads,
                "head_dim": head_dim,
                "timing_scope_note": "SDPA mixed attention is timed as one call; observation-only time is not directly measured.",
            }
            if runner.stage == "action_denoise" and runner.video_seq_len is not None:
                obs_keys = min(runner.video_seq_len, key_tokens)
                action_keys = max(0, key_tokens - obs_keys)
                detail.update(
                    {
                        "key_groups": {"first_frame_observation_cache": obs_keys, "current_action_tokens": action_keys},
                        "flops_qk_plus_av_by_group_mac2": {
                            "observation_cache": 4 * batch * heads * head_dim * query_tokens * obs_keys,
                            "action_tokens": 4 * batch * heads * head_dim * query_tokens * action_keys,
                        },
                        "flops_qk_plus_av_dense_mac2": 4 * batch * heads * head_dim * query_tokens * key_tokens,
                    }
                )
            row = {"context_index": runner.context_index, "pair_index": runner.pair_index, "denoise_step": runner.denoise_step, **detail}
            runner.coarse_attention_calls.append(row)
            with runner.span(f"MoT.{runner.stage}.flash_attention", detail=detail, cuda=True):
                return original_flash(*args, **kwargs)

        self._patch(mot_module, "flash_attention", timed_flash)

    def remove_fine(self) -> None:
        for handle in self.module_handles:
            handle.remove()
        self.module_handles.clear()
        if self.fine_patch_start is not None:
            while len(self.patches) > self.fine_patch_start:
                owner, name, original = self.patches.pop()
                setattr(owner, name, original)
            self.fine_patch_start = None

    def restore_patches(self) -> None:
        while self.patches:
            owner, name, original = self.patches.pop()
            setattr(owner, name, original)

    def _select_contexts(self) -> list[dict[str, Any]]:
        count = len(self.episode_calls)
        if count < 3:
            raise RuntimeError(f"official episode produced only {count} action contexts; need first/middle/last")
        indices = sorted({0, count // 2, count - 1})
        if len(indices) < 3:
            raise RuntimeError(f"could not select 3 distinct contexts from {count} action contexts")
        selected = []
        for context_index in indices[:CONTEXTS]:
            row = self.episode_calls[context_index]
            selected.append(
                {
                    "context_index": context_index,
                    "replan_index": row["replan_index"],
                    "obs": row["obs"],
                    "episode_action": row["action"],
                    "obs_shapes": {
                        key: list(value.shape) if isinstance(value, np.ndarray) else type(value).__name__
                        for key, value in row["obs"].items()
                    },
                }
            )
        self.context_rows = selected
        return selected

    def _replay(self, context: dict[str, Any], kind: str, pair_index: int | None = None) -> dict[str, Any]:
        self.phase = kind
        self.context_index = context["context_index"]
        self.pair_index = pair_index
        self.call_counter = 0
        self.infer_output = None
        call = {
            "obs": copy.deepcopy(context["obs"]),
            "task_description": self.context_rows[0].get("task_description"),
            "model": self.model,
            "processor": self.processor,
            "cfg": self.cfg,
            "action_horizon": self.action_horizon,
            "input_w": self.input_w,
            "input_h": self.input_h,
            "model_device": self.model_device,
        }
        # task_description is recorded from the actual official call and is identical
        # for this single-task episode.
        if context.get("task_description") is not None:
            call["task_description"] = context["task_description"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        top_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        top_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if top_start is not None:
            top_start.record()
        started = time.perf_counter_ns()
        result = self._original_predict(**call)
        if top_end is not None:
            top_end.record()
            torch.cuda.synchronize()
        predict_ms = (time.perf_counter_ns() - started) / 1e6
        predict_cuda_ms = float(top_start.elapsed_time(top_end)) if top_start is not None else None
        return {
            "kind": kind,
            "context_index": context["context_index"],
            "pair_index": pair_index,
            "predict_cpu_wall_ms": predict_ms,
            "predict_cuda_event_ms": predict_cuda_ms,
            "action": np.asarray(result[0]).copy(),
            "raw_model_output": self.infer_output,
        }

    def run_replays(self) -> None:
        contexts = self._select_contexts()
        if not hasattr(self, "processor"):
            raise RuntimeError("processor was not attached to the official task call")
        rng_after_episode = _capture_rng()
        try:
            for context in contexts:
                context["task_description"] = self.task_description
                for warmup_idx in range(WARMUPS):
                    _restore_rng(rng_after_episode)
                    self._replay(context, "replay_warmup")
                for repeat_idx in range(REPEATS):
                    order = ("replay_native", "replay_instrumented") if repeat_idx % 2 == 0 else ("replay_instrumented", "replay_native")
                    pair: dict[str, Any] = {
                        "context_index": context["context_index"],
                        "repetition": repeat_idx,
                        "order": list(order),
                        "results": {},
                    }
                    for kind in order:
                        if kind == "replay_native":
                            self.remove_fine()
                        else:
                            self.install_fine()
                        _restore_rng(rng_after_episode)
                        replay = self._replay(context, kind, repeat_idx)
                        pair["results"][kind] = replay
                    native = pair["results"]["replay_native"]
                    instrumented = pair["results"]["replay_instrumented"]
                    action_parity = _compare_tree(native["action"], instrumented["action"], "action_chunk")
                    raw_parity = _compare_tree(native["raw_model_output"], instrumented["raw_model_output"], "infer_action_output")
                    episode_parity = _compare_tree(context["episode_action"], native["action"], "episode_vs_native_replay")
                    pair["parity"] = {
                        "native_vs_instrumented_postprocessed": action_parity,
                        "native_vs_instrumented_raw_model_output": raw_parity,
                        "episode_vs_native_replay": episode_parity,
                    }
                    if not (action_parity["allclose"] and raw_parity["allclose"] and episode_parity["allclose"]):
                        raise RuntimeError(f"same-observation parity failed for context {context['context_index']} repeat {repeat_idx}")
                    self.parity.append(pair)
                    self.native_pairs.append(
                        {
                            "context_index": context["context_index"],
                            "repetition": repeat_idx,
                            "order": list(order),
                            "native_predict_cpu_wall_ms": native["predict_cpu_wall_ms"],
                            "instrumented_predict_cpu_wall_ms": instrumented["predict_cpu_wall_ms"],
                            "native_predict_cuda_event_ms": native["predict_cuda_event_ms"],
                            "instrumented_predict_cuda_event_ms": instrumented["predict_cuda_event_ms"],
                            "allclose": True,
                        }
                    )
                    pair["results"] = {
                        key: {
                            "predict_cpu_wall_ms": value["predict_cpu_wall_ms"],
                            "predict_cuda_event_ms": value["predict_cuda_event_ms"],
                        }
                        for key, value in pair["results"].items()
                    }
                    self.parity[-1] = {
                        "context_index": pair["context_index"],
                        "repetition": repeat_idx,
                        "order": list(order),
                        "timings": pair["results"],
                        "parity": pair["parity"],
                    }
            for context in contexts:
                for repeat_idx in range(COARSE_REPEATS):
                    order = ("replay_native", "replay_coarse_instrumented") if repeat_idx % 2 == 0 else ("replay_coarse_instrumented", "replay_native")
                    results: dict[str, dict[str, Any]] = {}
                    for kind in order:
                        if kind == "replay_native":
                            self.remove_fine()
                        else:
                            self.install_thin()
                        _restore_rng(rng_after_episode)
                        results[kind] = self._replay(context, kind, repeat_idx)
                    native = results["replay_native"]
                    coarse = results["replay_coarse_instrumented"]
                    action_parity = _compare_tree(native["action"], coarse["action"], "action_chunk")
                    raw_parity = _compare_tree(native["raw_model_output"], coarse["raw_model_output"], "infer_action_output")
                    episode_parity = _compare_tree(context["episode_action"], native["action"], "episode_vs_native_replay")
                    pair = {
                        "context_index": context["context_index"],
                        "repetition": repeat_idx,
                        "order": list(order),
                        "native_vs_coarse_postprocessed": action_parity,
                        "native_vs_coarse_raw_model_output": raw_parity,
                        "episode_vs_native_replay": episode_parity,
                    }
                    if not (action_parity["allclose"] and raw_parity["allclose"] and episode_parity["allclose"]):
                        raise RuntimeError(f"thin-pass same-observation parity failed for context {context['context_index']} repeat {repeat_idx}")
                    self.coarse_pairs.append(
                        {
                            "context_index": context["context_index"],
                            "repetition": repeat_idx,
                            "order": list(order),
                            "native_predict_cpu_wall_ms": native["predict_cpu_wall_ms"],
                            "coarse_instrumented_predict_cpu_wall_ms": coarse["predict_cpu_wall_ms"],
                            "native_predict_cuda_event_ms": native["predict_cuda_event_ms"],
                            "coarse_instrumented_predict_cuda_event_ms": coarse["predict_cuda_event_ms"],
                            "parity": pair,
                        }
                    )
            self.remove_fine()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for row, event_start, event_end in self.pending_events:
                row["cuda_event_ms"] = float(event_start.elapsed_time(event_end))
            self.pending_events.clear()
            if TRACE_CALLS > 0:
                self.install_fine()
                self._run_profiler_trace(contexts[0])
                self.remove_fine()
        finally:
            self.remove_fine()
            _restore_rng(rng_after_episode)
            self.phase = "done"

    def _run_profiler_trace(self, context: dict[str, Any]) -> None:
        if TRACE_CALLS <= 0:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.phase = "trace"
        self.context_index = context["context_index"]
        self.trace_active = True
        self.call_counter = 0
        trace_rng = _capture_rng()
        try:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                for trace_idx in range(TRACE_CALLS):
                    _restore_rng(trace_rng)
                    self._original_predict(
                        obs=copy.deepcopy(context["obs"]),
                        task_description=context["task_description"],
                        model=self.model,
                        processor=self.processor,
                        cfg=self.cfg,
                        action_horizon=self.action_horizon,
                        input_w=self.input_w,
                        input_h=self.input_h,
                        model_device=self.model_device,
                    )
                    prof.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            trace_path = ARTIFACTS / "torch_profiler_trace.json"
            prof.export_chrome_trace(str(trace_path))
            op_groups: dict[tuple[str, str], dict[str, Any]] = {}
            for event in prof.events():
                key = str(getattr(event, "name", ""))
                if not (
                    key.startswith("FW/")
                    or "scaled_dot_product" in key
                    or key in {"aten::to", "aten::_to_copy", "aten::copy_"}
                ):
                    continue
                ancestors = []
                parent = getattr(event, "cpu_parent", None)
                while parent is not None and len(ancestors) < 12:
                    ancestors.append(str(getattr(parent, "name", "unknown")))
                    parent = getattr(parent, "cpu_parent", None)
                scope = next((label for label in ancestors if label.startswith("FW/")), "")
                group_key = (key, scope)
                row = op_groups.setdefault(
                    group_key,
                    {
                        "operator_or_record_function": key,
                        "scope_label": scope,
                        "count": 0,
                        "cpu_total_us": 0.0,
                        "cpu_self_us": 0.0,
                        "cuda_total_us": 0.0,
                        "cuda_self_us": 0.0,
                        "ancestor_labels": ancestors,
                        "input_shapes": getattr(event, "input_shapes", None),
                    },
                )
                row["count"] += int(getattr(event, "count", 1))
                row["cpu_total_us"] += float(getattr(event, "cpu_time_total", 0.0))
                row["cpu_self_us"] += float(getattr(event, "self_cpu_time_total", 0.0))
                row["cuda_total_us"] += float(getattr(event, "device_time_total", 0.0))
                row["cuda_self_us"] += float(getattr(event, "self_device_time_total", 0.0))
            self.trace_ops = list(op_groups.values())
        finally:
            self.trace_active = False

    def _event_summaries(self) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str, int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            key = (row["name"], row["pass"], row["stage"], row["context_index"], row.get("denoise_step"))
            groups[key].append(row)
        result = []
        for (name, phase, stage, context_index, denoise_step), rows in groups.items():
            entry: dict[str, Any] = {
                "name": name,
                "pass": phase,
                "stage": stage,
                "context_index": context_index,
                "denoise_step": denoise_step,
                "cpu_wall_ms": _stats([row["cpu_wall_ms"] for row in rows]),
            }
            cuda_values = [row["cuda_event_ms"] for row in rows if "cuda_event_ms" in row]
            if cuda_values:
                entry["cuda_event_ms"] = _stats(cuda_values)
            result.append(entry)
        return result

    def _deduplicated_shapes(self, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            key_data = {field: row.get(field) for field in fields}
            key = json.dumps(key_data, sort_keys=True, default=str)
            group = groups.setdefault(key, {**key_data, "count": 0, "contexts": set(), "denoise_steps": set()})
            group["count"] += 1
            group["contexts"].add(row.get("context_index"))
            if row.get("denoise_step") is not None:
                group["denoise_steps"].add(row["denoise_step"])
        output = []
        for group in groups.values():
            group["contexts"] = sorted(value for value in group["contexts"] if value is not None)
            group["denoise_steps"] = sorted(group["denoise_steps"])
            output.append(group)
        return output

    def _episode_stage_summaries(self) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in self.rows:
            if row["pass"] != "episode":
                continue
            camera = str(row.get("camera_name", "")) if row["name"] == "environment.sim.render" else ""
            groups[(row["name"], camera)].append(float(row["cpu_wall_ms"]))
        return [
            {
                "name": name,
                "camera_name": camera or None,
                "cpu_wall_ms": _stats(values),
            }
            for (name, camera), values in groups.items()
        ]

    def write_summary(self, bootstrap: list[dict[str, Any]], config_yaml: str, error: str | None = None) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "resolved_config.yaml").write_text(config_yaml, encoding="utf-8")
        episode_infer = [row["cpu_wall_ms"] for row in self.rows if row["name"] == "model.infer_action" and row["pass"] == "episode"]
        episode_predict = [row["cpu_wall_ms"] for row in self.episode_calls]
        capture_copy_ms = [row["capture_copy_ms_excluded"] for row in self.episode_calls]
        context_summary = [
            {
                "context_index": row["context_index"],
                "episode_replan_index": row["replan_index"],
                "observation_shapes": row["obs_shapes"],
            }
            for row in self.context_rows
        ]
        native_wall = [row["native_predict_cpu_wall_ms"] for row in self.native_pairs]
        instr_wall = [row["instrumented_predict_cpu_wall_ms"] for row in self.native_pairs]
        native_cuda = [row["native_predict_cuda_event_ms"] for row in self.native_pairs]
        instr_cuda = [row["instrumented_predict_cuda_event_ms"] for row in self.native_pairs]
        raw_spans_path = ARTIFACTS / "raw_spans.jsonl"
        with raw_spans_path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
        span_summaries = self._event_summaries()
        denoise_step_timings = [
            row for row in span_summaries
            if row["name"] == "action.denoise_step"
            and row["pass"] in {"episode", "replay_native", "replay_instrumented", "replay_coarse_instrumented"}
        ]
        coarse_native_wall = [row["native_predict_cpu_wall_ms"] for row in self.coarse_pairs]
        coarse_instrumented_wall = [row["coarse_instrumented_predict_cpu_wall_ms"] for row in self.coarse_pairs]
        coarse_native_cuda = [row["native_predict_cuda_event_ms"] for row in self.coarse_pairs]
        coarse_instrumented_cuda = [row["coarse_instrumented_predict_cuda_event_ms"] for row in self.coarse_pairs]
        summary = {
            "status": "error" if error else "complete",
            "error": error,
            "claim_label": "FastWAMOptionalIDM action_infer_mode=first_frame (direct action head); not FastWAM-Joint",
            "allocation": {"pbs_job_id": PBS_JOB_ID, "hostname": HOSTNAME},
            "source": {"path": str(FASTWAM_SOURCE), "revision": EXPECTED_REVISION, "read_only": True},
            "protocol": {
                "model_class": type(self.model).__name__,
                "checkpoint": str(self.cfg.ckpt),
                "task_suite": str(self.cfg.EVALUATION.task_suite_name),
                "task_id": int(self.cfg.EVALUATION.task_id),
                "num_trials": int(self.cfg.EVALUATION.num_trials),
                "seed": int(self.cfg.seed),
                "action_infer_mode": str(self.cfg.EVALUATION.get("action_infer_mode")),
                "sigma_shift": float(self.cfg.EVALUATION.sigma_shift),
                "compile_action_infer": bool(self.cfg.EVALUATION.compile_action_infer),
                "num_inference_steps": int(self.cfg.EVALUATION.num_inference_steps),
                "num_steps_wait": int(self.cfg.EVALUATION.num_steps_wait),
                "replan_steps": int(self.cfg.EVALUATION.replan_steps),
                "text_cfg_scale": float(self.cfg.EVALUATION.text_cfg_scale),
                "dtype": str(self.model.torch_dtype),
                "device": str(self.model.device),
                "eval_mode": not self.model.training,
                "visible_future_video": bool(self.cfg.EVALUATION.get("visualize_future_video", False)),
            },
            "bootstrap_cpu_stages": bootstrap,
            "episode": {
                "action_context_count": len(self.episode_calls),
                "first_chunk_cpu_wall_ms": episode_predict[0] if episode_predict else None,
                "steady_chunk_cpu_wall_ms": _stats(episode_predict[1:]),
                "observation_copy_overhead_excluded_ms": _stats(capture_copy_ms),
                "first_infer_action_cpu_wall_ms": episode_infer[0] if episode_infer else None,
                "steady_infer_action_cpu_wall_ms": _stats(episode_infer[1:]),
                "native_episode_cpu_stages": self._episode_stage_summaries(),
                "sim_render_hook_count": getattr(self, "render_hook_count", 0),
            },
            "replay": {
                "context_selection": "first, middle, and last actual replan observations from the official episode",
                "contexts": context_summary,
                "warmups_per_context": WARMUPS,
                "paired_repeats_per_context": REPEATS,
                "pair_order": "alternating AB/BA, native vs instrumented",
                "trace_calls": TRACE_CALLS,
                "native_predict_cpu_wall_ms": _stats(native_wall),
                "instrumented_predict_cpu_wall_ms": _stats(instr_wall),
                "instrumentation_cpu_overhead_ms": _stats([b - a for a, b in zip(native_wall, instr_wall)]),
                "instrumentation_cpu_overhead_fraction": (
                    (statistics.fmean(instr_wall) - statistics.fmean(native_wall)) / statistics.fmean(native_wall)
                    if native_wall and statistics.fmean(native_wall) != 0 else None
                ),
                "instrumentation_cuda_event_overhead_ms": _stats([b - a for a, b in zip(native_cuda, instr_cuda)]),
                "instrumentation_cuda_event_overhead_fraction": (
                    (statistics.fmean(instr_cuda) - statistics.fmean(native_cuda)) / statistics.fmean(native_cuda)
                    if native_cuda and statistics.fmean(native_cuda) != 0 else None
                ),
                "pairs": self.native_pairs,
                "output_parity": self.parity,
                "instrumentation_spans": span_summaries,
                "denoise_step_timings": denoise_step_timings,
                "attention_geometry_counts": self._deduplicated_shapes(
                    self.attention_calls,
                    ("attention_stage", "q_shape", "k_shape", "v_shape", "mask_shape", "heads", "head_dim", "key_groups", "flops_qk_plus_av_by_group_mac2", "flops_qk_plus_av_dense_mac2"),
                ),
                "attention_geometry_count": len(self.attention_calls),
                "attention_geometry_source_pass": "replay_instrumented",
                "low_overhead_pass": {
                    "paired_repeats_per_context": COARSE_REPEATS,
                    "native_predict_cpu_wall_ms": _stats(coarse_native_wall),
                    "instrumented_predict_cpu_wall_ms": _stats(coarse_instrumented_wall),
                    "instrumentation_cpu_overhead_ms": _stats([b - a for a, b in zip(coarse_native_wall, coarse_instrumented_wall)]),
                    "instrumentation_cpu_overhead_fraction": (
                        (statistics.fmean(coarse_instrumented_wall) - statistics.fmean(coarse_native_wall)) / statistics.fmean(coarse_native_wall)
                        if coarse_native_wall and statistics.fmean(coarse_native_wall) != 0 else None
                    ),
                    "native_predict_cuda_event_ms": _stats(coarse_native_cuda),
                    "instrumented_predict_cuda_event_ms": _stats(coarse_instrumented_cuda),
                    "instrumentation_cuda_event_overhead_ms": _stats([b - a for a, b in zip(coarse_native_cuda, coarse_instrumented_cuda)]),
                    "pairs": self.coarse_pairs,
                    "stage_spans": [row for row in span_summaries if row["pass"] == "replay_coarse_instrumented"],
                    "action_attention_geometry_counts": self._deduplicated_shapes(
                        self.coarse_attention_calls,
                        ("attention_stage", "q_shape", "k_shape", "v_shape", "mask_shape", "heads", "head_dim", "key_groups", "flops_qk_plus_av_by_group_mac2", "flops_qk_plus_av_dense_mac2"),
                    ),
                    "action_attention_call_count": len(self.coarse_attention_calls),
                    "scope_note": "Only VAE encode, prompt encode, video cache prefill, each action denoise step, model infer_action, and each SDPA call are timed; no per-layer QKV/FFN hooks.",
                },
                "qkv_shape_counts": self._deduplicated_shapes(
                    self.qkv_shapes,
                    ("stage", "expert", "layer", "q", "k", "v", "source"),
                ),
                "profiler_operator_summary": getattr(self, "trace_ops", []),
                "profiler_trace": str(ARTIFACTS / "torch_profiler_trace.json") if TRACE_CALLS else None,
                "raw_spans_jsonl": str(raw_spans_path),
                "raw_span_count": len(self.rows),
            },
            "interpretation": {
                "nested_spans": "All spans are inclusive; nested times overlap and must not be summed.",
                "cpu_wall": "CPU wall around a CUDA call includes Python launch and any synchronization/D2H in that call.",
                "cuda_event": "Paired events are queued asynchronously and resolved after replay calls; spans include stream gaps.",
                "attention": "MoT imports flash_attention, which calls scaled_dot_product_attention. Action-to-observation QK/AV has no native standalone timer; report the SDPA mixed-attention interval plus mask/shape FLOP estimates.",
                "text_cross_attention": "Action block text cross-attention is separately timed as module.*cross_attn and is not the action-to-observation self-attention target.",
                "h2d": "Native H2D sub-operations are identified only by profiler aten::to/copy and CUDA memcpy events; those trace timings are perturbative and are not native latency.",
                "video": "first_frame path VAE-encodes the observation and prefills video K/V once per infer_action; it does not denoise a future video.",
                "episode_env": "environment.step and sim.render are nested inclusive wall spans; do not add them. A zero render hook count means sim.render was unavailable or could not be patched; environment.step remains inclusive.",
            },
        }
        if hasattr(self, "official_result"):
            summary["official_episode_result"] = {
                "successes": self.official_result.get("successes"),
                "success_episodes": self.official_result.get("success_episodes"),
                "failure_episodes": self.official_result.get("failure_episodes"),
                "task_description": self.official_result.get("task_description"),
            }
        (ARTIFACTS / "profile_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def _validate_protocol(cfg: DictConfig, model: torch.nn.Module) -> None:
    required = {
        "task_suite_name": (str(cfg.EVALUATION.task_suite_name), "libero_goal"),
        "task_id": (int(cfg.EVALUATION.task_id), 0),
        "num_trials": (int(cfg.EVALUATION.num_trials), 1),
        "seed": (int(cfg.seed), 42),
        "action_infer_mode": (str(cfg.EVALUATION.get("action_infer_mode", "")), "first_frame"),
        "sigma_shift": (float(cfg.EVALUATION.sigma_shift), 1.0),
        "compile_action_infer": (bool(cfg.EVALUATION.compile_action_infer), False),
        "num_inference_steps": (int(cfg.EVALUATION.num_inference_steps), 10),
        "num_steps_wait": (int(cfg.EVALUATION.num_steps_wait), 30),
        "replan_steps": (int(cfg.EVALUATION.replan_steps), 10),
        "text_cfg_scale": (float(cfg.EVALUATION.text_cfg_scale), 1.0),
    }
    mismatches = [f"{name}={got!r} expected {want!r}" for name, (got, want) in required.items() if got != want]
    target = str(cfg.model.get("_target_", ""))
    if "optional_idm" not in target.lower() or type(model).__name__ != "FastWAMOptionalIDM":
        mismatches.append(f"model must be FastWAMOptionalIDM, got target={target!r} class={type(model).__name__!r}")
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        mismatches.append("visualize_future_video must remain false for the fixed baseline")
    if mismatches:
        raise ValueError("fixed profiling protocol mismatch: " + "; ".join(mismatches))


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bootstrap: list[dict[str, Any]] = []
    invocation_started = time.perf_counter_ns()

    original_instantiate = official_eval.instantiate

    @functools.wraps(original_instantiate)
    def timed_instantiate(config, *args, **kwargs):
        target = str(config.get("_target_", "unknown")) if isinstance(config, DictConfig) else "unknown"
        started = time.perf_counter_ns()
        value = original_instantiate(config, *args, **kwargs)
        bootstrap.append({"name": f"hydra.instantiate:{target}", "cpu_wall_ms": (time.perf_counter_ns() - started) / 1e6})
        return value

    original_load = official_eval._load_model_checkpoint

    @functools.wraps(original_load)
    def timed_load(model, ckpt):
        started = time.perf_counter_ns()
        value = original_load(model, ckpt)
        bootstrap.append({"name": "checkpoint.load", "cpu_wall_ms": (time.perf_counter_ns() - started) / 1e6})
        return value

    original_stats = official_eval.load_dataset_stats_from_json

    @functools.wraps(original_stats)
    def timed_stats(path):
        started = time.perf_counter_ns()
        value = original_stats(path)
        bootstrap.append({"name": "dataset_stats.load", "cpu_wall_ms": (time.perf_counter_ns() - started) / 1e6})
        return value

    original_task = official_eval._run_task_to_file

    @functools.wraps(original_task)
    def profiled_task(**kwargs):
        cfg = kwargs["cfg"]
        model = kwargs["model"]
        _validate_protocol(cfg, model)
        runner = Runner(model, cfg, bootstrap)
        runner.processor = kwargs["processor"]
        runner.action_horizon = kwargs["action_horizon"]
        runner.input_w = kwargs["input_w"]
        runner.input_h = kwargs["input_h"]
        runner.model_device = kwargs["model_device"]
        runner.task_description = None
        runner.config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        runner.install_coarse()
        start_ms = (time.perf_counter_ns() - invocation_started) / 1e6
        bootstrap.append({"name": "runner_start_to_official_episode", "cpu_wall_ms": start_ms})
        try:
            output_file, results = original_task(**kwargs)
            runner.official_result = results
            runner.episode_result_path = str(output_file)
            # task text is returned by run_single_task and applies to this task.
            runner.task_description = results.get("task_description")
            runner.install_fine()
            runner.run_replays()
            runner.write_summary(bootstrap, runner.config_yaml)
            return output_file, results
        except BaseException as exc:
            try:
                runner.write_summary(bootstrap, runner.config_yaml, error=repr(exc))
            except Exception:
                pass
            raise
        finally:
            runner.restore_patches()

    for owner, name, replacement in (
        (official_eval, "instantiate", timed_instantiate),
        (official_eval, "_load_model_checkpoint", timed_load),
        (official_eval, "load_dataset_stats_from_json", timed_stats),
        (official_eval, "_run_task_to_file", profiled_task),
    ):
        setattr(owner, name, replacement)

    try:
        # Keep Hydra's own CLI parsing and official config composition intact.
        import hydra

        official_main = hydra.main(
            version_base="1.3",
            config_path=str(FASTWAM_SOURCE / "configs"),
            config_name="sim_libero.yaml",
        )(official_eval.eval_single_process.__wrapped__)
        official_main()
    finally:
        official_eval.instantiate = original_instantiate
        official_eval._load_model_checkpoint = original_load
        official_eval.load_dataset_stats_from_json = original_stats
        official_eval._run_task_to_file = original_task


if __name__ == "__main__":
    main()

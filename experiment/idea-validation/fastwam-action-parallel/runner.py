#!/usr/bin/env python3
"""Bounded, paired Fast-WAM action experiments; approved PBS compute only."""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

JOB = os.environ.get("PBS_JOBID", "")
NODEFILE = Path(os.environ.get("PBS_NODEFILE", "/missing"))
HOST = socket.gethostname().split(".")[0]
if not JOB or "login" in HOST.lower() or not NODEFILE.is_file():
    raise SystemExit("A real non-login PBS allocation is required")
if HOST not in {line.split(".")[0] for line in NODEFILE.read_text().splitlines()}:
    raise SystemExit("Host is not in PBS_NODEFILE")
SOURCE = Path(os.environ["FASTWAM_SOURCE"])
if (SOURCE / ".fastwam-revision").read_text().strip() != "7faa71108368fbb3b6885649f112af607427a2d4":
    raise SystemExit("Source revision mismatch")
ARTIFACTS = Path(os.environ["ARTIFACTS"])
LINE = os.environ["FW_LINE"]
sys.path[:0] = [str(SOURCE), str(SOURCE / "src"), str(SOURCE / "experiments/libero")]

import csv
import functools
import gc
import inspect
import json
import statistics
import time
from contextlib import contextmanager

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from experiments.libero import eval_libero_single as official

# One task per PBS process; preserve completed rows if a later candidate fails.
MEASUREMENTS = []
SELF_CHECKS = {}


def tree(value, *, cpu=False, device=None):
    if isinstance(value, torch.Tensor):
        target = "cpu" if cpu else device or value.device
        return value.detach().to(target).clone()
    if isinstance(value, dict):
        return {key: tree(item, cpu=cpu, device=device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tree(item, cpu=cpu, device=device) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def measured(fn):
    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    return result, {"wall_ms": (time.perf_counter() - started) * 1000, "cuda_ms": start_event.elapsed_time(end_event)}


def peak(fn):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = {"baseline_allocated": torch.cuda.memory_allocated(), "baseline_reserved": torch.cuda.memory_reserved()}
    result = fn()
    torch.cuda.synchronize()
    return {**before, "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}


def errors(left, right):
    a, b = left.detach().float().cpu(), right.detach().float().cpu()
    diff = a - b
    return {"finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs": float(diff.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-12)),
            "strict_allclose": bool(torch.allclose(a, b, atol=1e-5, rtol=1e-5))}


def postprocessed(action, processor, cfg):
    result = official._denormalize_action(action[0].float().cpu(), processor)[0]
    result[..., -1] = result[..., -1] * 2 - 1
    result = official.invert_gripper_action(result)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        result[..., -1] = np.sign(result[..., -1])
    return result


def quality(candidate, reference, processor, cfg):
    report = errors(candidate, reference)
    report["first10_max_abs"] = errors(candidate[:, :10], reference[:, :10])["max_abs"]
    report["gripper_sign_mismatches"] = int(np.count_nonzero(np.sign(postprocessed(candidate, processor, cfg)[..., -1]) != np.sign(postprocessed(reference, processor, cfg)[..., -1])))
    report["pilot_gate"] = report["finite"] and report["max_abs"] <= 1e-3 and report["relative_l2"] <= 1e-3
    report["action_execution_gate"] = report["pilot_gate"] and report["gripper_sign_mismatches"] == 0
    return report


def pair(native, candidate, *, label, context_id, scope, rows):
    for _ in range(3):
        native()
        candidate()
    ratios = []
    last = None
    for repetition in range(4):
        order = ["native", "candidate"] if repetition % 2 == 0 else ["candidate", "native"]
        sample = {}
        for arm in order:
            output, timing = measured(native if arm == "native" else candidate)
            sample[arm] = timing
            rows.append({"context": context_id, "candidate": label, "scope": scope, "pair": repetition, "order": "/".join(order), "arm": arm, **timing})
            if arm == "candidate":
                last = output
        ratios.append(sample["native"]["wall_ms"] / sample["candidate"]["wall_ms"])
    selected = [row for row in rows if row["context"] == context_id and row["candidate"] == label and row["scope"] == scope]
    return last, {"native_median_ms": statistics.median(row["wall_ms"] for row in selected if row["arm"] == "native"),
                  "candidate_median_ms": statistics.median(row["wall_ms"] for row in selected if row["arm"] == "candidate"),
                  "paired_speedup_median": statistics.median(ratios), "paired_speedup_samples": ratios}


def constants(core):
    return {key: value for key, value in core.items() if key not in {"latents_action", "timestep_action"}}


def batched_core(raw, core, action, timestep):
    width = action.shape[0]
    kwargs = constants(core)
    kwargs["context"] = kwargs["context"].expand(width, -1, -1)
    kwargs["context_mask"] = kwargs["context_mask"].expand(width, -1)
    for key in ("video_cache_k", "video_cache_v"):
        kwargs[key] = [value.expand(width, -1, -1) for value in kwargs[key]]
    return raw(latents_action=action, timestep_action=timestep, **kwargs)


def serial(model, raw, data, *, trace=False):
    action = data["core"]["latents_action"].clone()
    states = [action]
    velocities = []
    for timestep, delta in zip(data["timesteps"], data["deltas"]):
        velocity = raw(latents_action=action, timestep_action=timestep.unsqueeze(0), **constants(data["core"]))
        action = model.infer_action_scheduler.step(velocity, delta, action)
        if trace:
            states.append(action)
            velocities.append(velocity)
    return (action, states, velocities) if trace else action


@contextmanager
def replaced(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def capture_contexts(model, task_kwargs):
    calls = []
    original_infer = model.infer_action

    @functools.wraps(original_infer)
    def capture_infer(*args, **kwargs):
        if args:
            raise ValueError("Expected keyword-only official infer invocation")
        saved = tree(kwargs, cpu=True)
        output = original_infer(**kwargs)
        calls.append({"infer_kwargs": saved, "episode_output": tree(output, cpu=True)})
        return output

    with replaced(model, "infer_action", capture_infer):
        output_file, episode = ORIGINAL_TASK(**task_kwargs)
    if len(calls) < 3:
        raise RuntimeError(f"Need three real replan contexts, found {len(calls)}")
    indices = [0, 6, 12] if len(calls) >= 13 else sorted({0, len(calls) // 2, len(calls) - 1})
    raw = model._denoise_action_with_video_cache
    signature = inspect.signature(raw)
    selected = []
    for index in indices:
        saved = calls[index]
        core = None
        states, velocities = [], []

        @functools.wraps(raw)
        def capture_core(*args, **kwargs):
            nonlocal core
            bound = signature.bind(*args, **kwargs).arguments
            if core is None:
                core = tree(dict(bound))
            states.append(bound["latents_action"].detach().clone())
            result = raw(*args, **kwargs)
            velocities.append(result.detach().clone())
            return result

        with replaced(model, "_denoise_action_with_video_cache", capture_core):
            replay = model.infer_action(**tree(saved["infer_kwargs"], device=model.device))
        assert len(states) == 10, len(states)
        identity = errors(replay["action"], saved["episode_output"]["action"])
        if not identity["finite"] or not identity["strict_allclose"]:
            raise RuntimeError(f"Native episode replay mismatch at {index}: {identity}")
        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(num_inference_steps=10, device=model.device, dtype=core["latents_action"].dtype, shift_override=1.0)
        item = {"replan_index": index, "core": core, "states": states, "velocities": velocities,
                "timesteps": timesteps, "deltas": deltas, "infer_kwargs": saved["infer_kwargs"],
                "reference": replay["action"].to(model.device, dtype=core["latents_action"].dtype).unsqueeze(0), "replay_identity": identity}
        core_ref = serial(model, raw, item)
        if not errors(core_ref, item["reference"])["strict_allclose"]:
            raise RuntimeError("Serial harness did not reproduce native action")
        selected.append(item)
    torch.save(tree(selected, cpu=True), ARTIFACTS / "frozen_contexts.pt")
    return selected, episode, output_file


def context_line(model, cfg, processor, contexts, rows):
    from context_cache import ContextCache
    raw = model._denoise_action_with_video_cache
    results = MEASUREMENTS
    for mode in ["embedding", "kv", "batched_kv"]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        construction_baseline = torch.cuda.memory_allocated()
        started = time.perf_counter()
        cache = ContextCache(model, mode)
        torch.cuda.synchronize()
        construction_ms = (time.perf_counter() - started) * 1000
        construction_memory = {"baseline_allocated": construction_baseline, "after_allocated": torch.cuda.memory_allocated(), "peak_allocated": torch.cuda.max_memory_allocated()}
        for data in contexts:
            cid = data["replan_index"]
            def candidate(trace=False):
                with cache.activate():
                    cache.prepare(data["core"]["context"])
                    return serial(model, raw, data, trace=trace)
            output, timing = pair(lambda: serial(model, raw, data), candidate, label=mode, context_id=cid, scope="action_phase_including_cache_prepare", rows=rows)
            parity = quality(output, data["reference"], processor, cfg)
            _, native_states, native_velocities = serial(model, raw, data, trace=True)
            _, cached_states, cached_velocities = candidate(trace=True)
            step_errors = [errors(a, b) for a, b in zip(cached_states, native_states)]
            velocity_errors = [errors(a, b) for a, b in zip(cached_velocities, native_velocities)]
            cache.clear()
            memory = peak(candidate)
            entry = {"candidate": mode, "context": cid, "scope": "action_phase_including_cache_prepare", **timing,
                     "parity": parity, "step_errors": step_errors, "velocity_errors": velocity_errors,
                     "semantic_gate": all(item["finite"] and item["strict_allclose"] for item in step_errors + velocity_errors),
                     "memory": memory, "construction_ms": construction_ms, "construction_memory": construction_memory,
                     "packed_weight_bytes": cache.packed_weight_bytes, "cache_bytes": cache.cache_bytes}
            results.append(entry)
            print("MEASUREMENT", json.dumps({key: entry[key] for key in ["candidate", "context", "scope", "paired_speedup_median", "semantic_gate"]}), flush=True)
            if entry["semantic_gate"]:
                def full_candidate():
                    prepared = False
                    def cached_core(*args, **kwargs):
                        nonlocal prepared
                        bound = inspect.signature(raw).bind(*args, **kwargs).arguments
                        if not prepared:
                            cache.prepare(bound["context"])
                            prepared = True
                        return raw(*args, **kwargs)
                    with cache.activate(), replaced(model, "_denoise_action_with_video_cache", cached_core):
                        return model.infer_action(**tree(data["infer_kwargs"], device=model.device))["action"]
                full, full_timing = pair(lambda: model.infer_action(**tree(data["infer_kwargs"], device=model.device))["action"], full_candidate, label=mode, context_id=cid, scope="full_infer_action", rows=rows)
                cache.clear()
                results.append({"candidate": mode, "context": cid, "scope": "full_infer_action", **full_timing,
                                "parity": errors(full, data["reference"][0]), "memory": peak(full_candidate)})
            cache.clear()
        del cache
        gc.collect()
    # Direct ablations distinguish K/V caching from batching its preparation.
    for reference_mode, candidate_mode in [("embedding", "kv"), ("kv", "batched_kv")]:
        reference_cache = ContextCache(model, reference_mode)
        candidate_cache = ContextCache(model, candidate_mode)
        for data in contexts:
            def cached_phase(cache):
                with cache.activate():
                    cache.prepare(data["core"]["context"])
                    return serial(model, raw, data)
            label = f"{candidate_mode}_vs_{reference_mode}"
            output, timing = pair(lambda: cached_phase(reference_cache), lambda: cached_phase(candidate_cache), label=label,
                                  context_id=data["replan_index"], scope="action_phase_direct_ablation", rows=rows)
            results.append({"candidate": label, "reference": reference_mode, "context": data["replan_index"], "scope": "action_phase_direct_ablation", **timing,
                            "parity": errors(output, cached_phase(reference_cache))})
            _, preparation = pair(lambda: reference_cache.prepare(data["core"]["context"]), lambda: candidate_cache.prepare(data["core"]["context"]),
                                  label=label, context_id=data["replan_index"], scope="fixed_context_prepare_only", rows=rows)
            results.append({"candidate": label, "reference": reference_mode, "context": data["replan_index"], "scope": "fixed_context_prepare_only", **preparation})
            reference_cache.clear()
            candidate_cache.clear()
        del reference_cache, candidate_cache
        gc.collect()
    return results


class ActionSolved(Exception):
    def __init__(self, action):
        self.action = action


def time_line(model, cfg, processor, contexts, rows):
    from time_parallel import picard_windowed
    raw = model._denoise_action_with_video_cache
    results = MEASUREMENTS
    specs = [(width, iterations) for width, counts in [(2, [1, 2]), (5, [1, 2, 3, 5]), (10, [1, 2, 3, 5, 10])] for iterations in counts]
    for data in contexts:
        cid = data["replan_index"]
        denoise = lambda action, timestep: batched_core(raw, data["core"], action, timestep)
        for width in [2, 5, 10]:
            action = torch.cat(data["states"][:width], dim=0)
            reference = torch.cat(data["velocities"][:width], dim=0)
            def singles():
                return torch.cat([denoise(action[index:index + 1], data["timesteps"][index:index + 1]) for index in range(width)], dim=0)
            candidate = lambda: denoise(action, data["timesteps"][:width])
            output, timing = pair(singles, candidate, label=f"batch_probe_W{width}", context_id=cid, scope="oracle_batch_forward_diagnostic", rows=rows)
            results.append({"candidate": f"batch_probe_W{width}", "context": cid, "scope": "oracle_batch_forward_diagnostic", **timing, "parity": errors(output, reference), "memory": peak(candidate), "deployable_speedup": False})
        for width, iterations in specs:
            label = f"prefix_W{width}_R{iterations}"
            metadata = {}
            def candidate():
                nonlocal metadata
                output, metadata = picard_windowed(denoise, data["core"]["latents_action"][0], data["timesteps"], data["deltas"], width, iterations)
                return output.unsqueeze(0)
            output, timing = pair(lambda: serial(model, raw, data), candidate, label=label, context_id=cid, scope="action_phase_parallel_solver", rows=rows)
            entry = {"candidate": label, "window": width, "iterations": iterations, "context": cid, "scope": "action_phase_parallel_solver", **timing,
                     "parity": quality(output, data["reference"], processor, cfg), "solver": metadata, "memory": peak(candidate)}
            results.append(entry)
            print("MEASUREMENT", json.dumps({key: entry[key] for key in ["candidate", "context", "paired_speedup_median", "parity"]}), flush=True)
        # Control: preserve native BF16 update arithmetic in a triangular iteration.
        output, metadata = picard_windowed(denoise, data["core"]["latents_action"][0], data["timesteps"], data["deltas"], 10, 10, method="triangular")
        output = output.unsqueeze(0)
        results.append({"candidate": "triangular_W10_R10_control", "context": cid, "scope": "native_rounding_control", "parity": quality(output, data["reference"], processor, cfg), "solver": metadata})
    eligible = []
    for width, iterations in specs:
        label = f"prefix_W{width}_R{iterations}"
        entries = [entry for entry in results if entry["candidate"] == label]
        if len(entries) == len(contexts) and all(entry["parity"]["action_execution_gate"] for entry in entries):
            eligible.append((statistics.median(entry["paired_speedup_median"] for entry in entries), width, iterations))
    if eligible:
        _, width, iterations = max(eligible)
        label = f"prefix_W{width}_R{iterations}"
        for data in contexts:
            def full_candidate():
                def solved_core(*args, **kwargs):
                    core = inspect.signature(raw).bind(*args, **kwargs).arguments
                    denoise = lambda action, timestep: batched_core(raw, core, action, timestep)
                    output, _ = picard_windowed(denoise, core["latents_action"][0], data["timesteps"], data["deltas"], width, iterations)
                    raise ActionSolved(output.unsqueeze(0))
                with replaced(model, "_denoise_action_with_video_cache", solved_core):
                    try:
                        model.infer_action(**tree(data["infer_kwargs"], device=model.device))
                    except ActionSolved as result:
                        return result.action[0].detach().float().cpu()
                raise RuntimeError("Parallel solver did not intercept action phase")
            output, timing = pair(lambda: model.infer_action(**tree(data["infer_kwargs"], device=model.device))["action"], full_candidate, label=label, context_id=data["replan_index"], scope="full_infer_action_selected_pilot", rows=rows)
            results.append({"candidate": label, "context": data["replan_index"], "scope": "full_infer_action_selected_pilot", **timing, "parity": quality(output.unsqueeze(0), data["reference"], processor, cfg), "memory": peak(full_candidate), "selection_scope": "Same three pilot contexts; not independent validation"})
    return results


ORIGINAL_TASK = official._run_task_to_file


def experiment_task(**kwargs):
    model, cfg = kwargs["model"], kwargs["cfg"]
    frozen = {"task_suite_name": "libero_goal", "task_id": 0, "num_trials": 1, "action_infer_mode": "first_frame", "sigma_shift": 1.0,
              "compile_action_infer": False, "num_inference_steps": 10, "num_steps_wait": 30, "replan_steps": 10, "text_cfg_scale": 1.0}
    for key, expected in frozen.items():
        assert cfg.EVALUATION.get(key) == expected, (key, cfg.EVALUATION.get(key), expected)
    assert cfg.seed == 42 and type(model).__name__ == "FastWAMOptionalIDM"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    rows = []
    summary = {"line": LINE, "job_id": JOB, "host": HOST, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "source_revision": "7faa71108368fbb3b6885649f112af607427a2d4", "scope": "Fixed-observation mechanism pilot; candidate closed-loop success not measured", "measurements": MEASUREMENTS, "self_checks": SELF_CHECKS}
    try:
        with torch.no_grad():
            contexts, episode, output_file = capture_contexts(model, kwargs)
            summary.update({"episode": episode, "episode_result_path": str(output_file),
                            "episode_videos": [{"path": str(path), "bytes": path.stat().st_size} for path in Path(cfg.EVALUATION.output_dir).rglob("*.mp4")],
                            "contexts": [{"replan_index": item["replan_index"], "replay_identity": item["replay_identity"], "action_shape": list(item["core"]["latents_action"].shape), "context_shape": list(item["core"]["context"].shape)} for item in contexts]})
            raw = model._denoise_action_with_video_cache
            summary["native_memory"] = [{"context": item["replan_index"], **peak(lambda: serial(model, raw, item))} for item in contexts]
            summary["measurements"] = (context_line if LINE == "context" else time_line)(model, cfg, kwargs["processor"], contexts, rows)
            summary["pipeline_complete"] = True
        (ARTIFACTS / "PIPELINE_COMPLETE").touch()
        return output_file, episode
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        (ARTIFACTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if rows:
            with (ARTIFACTS / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    if LINE == "context":
        from context_cache import self_check
    else:
        from time_parallel import self_check
    SELF_CHECKS[LINE] = self_check()
    print("SELF_CHECK", json.dumps(SELF_CHECKS[LINE]), flush=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "self_check.json").write_text(json.dumps(SELF_CHECKS), encoding="utf-8")
    with replaced(official, "_run_task_to_file", experiment_task):
        main = hydra.main(version_base="1.3", config_path=str(SOURCE / "configs"), config_name="sim_libero.yaml")(official.eval_single_process.__wrapped__)
        main()

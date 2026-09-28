#!/usr/bin/env python3
"""Frozen B=50 fixed-solve and closed-loop exact-cache experiment."""

import argparse
import copy
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cache_core import PerEnvironmentExactCache, identity_self_check


class CapturedContext(Exception):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_allocation(require_gpu=True):
    host = platform.node().split(".")[0].lower()
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    if not os.environ.get("PBS_JOBID") or not nodefile.is_file():
        raise RuntimeError("A real PBS allocation is required")
    nodes = {line.strip().split(".")[0].lower() for line in nodefile.read_text().splitlines()}
    if host not in nodes or any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-allocation/control host: {host}")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
    if require_gpu and not visible_gpu:
        raise RuntimeError("A visible allocated GPU is required")
    if not require_gpu and visible_gpu:
        raise RuntimeError("The self-check is CPU-only")
    return host


def self_check(freeze, main_freeze, torch):
    settings = freeze["settings"]
    fixed = freeze["fixed_context_stage"]
    if settings["num_eval"] != 50 or settings["solver_seeds"] != [42, 43, 44]:
        raise RuntimeError("Frozen task or seed scope changed")
    if settings["task_indices"] != list(range(50)) or fixed["required_active_environments"] != 50:
        raise RuntimeError("Frozen ordered task rows or fixed active batch changed")
    if (settings["num_samples"], settings["topk"], settings["iterations"], settings["batch_size"]) != (300, 30, 30, 1):
        raise RuntimeError("Frozen CEM configuration changed")
    if (settings["fixed_context_warmups_per_arm_seed"],
            settings["fixed_context_timed_repetitions_per_arm_seed"],
            settings["fixed_context_trace_pairs_per_seed"]) != (2, 6, 1):
        raise RuntimeError("Fixed-context repetition schedule changed")
    if settings["closed_loop_order_by_seed"] != {"42": ["native", "cache"],
                                                 "43": ["cache", "native"],
                                                 "44": ["native", "cache"]}:
        raise RuntimeError("Closed-loop pair order changed")
    if freeze["source_commit"] != main_freeze["fast_source_commit"]:
        raise RuntimeError("Fast-LeWM source identity changed")
    if freeze["checkpoint_revision"] != main_freeze["fast_checkpoint_revision"]:
        raise RuntimeError("Fast-LeWM checkpoint identity changed")
    if freeze["backend"] != "stable-worldmodel==0.0.6 isolated runtime_overlay":
        raise RuntimeError("Backend identity changed")
    expected = freeze["prior_native_outcomes"]
    if any(len(expected[str(seed)]) != settings["num_eval"] for seed in settings["solver_seeds"]):
        raise RuntimeError("Frozen prior native outcome vectors do not cover all task rows")
    cache_check = identity_self_check(torch)
    if cache_check["status"] != "PASS" or cache_check["maximum_resident_entries"] != 2:
        raise RuntimeError("Streaming cache dummy identity check failed")
    solver_identity_check()
    return {"status": "PASS", "cache_identity": cache_check,
            "closed_loop_solver_identity": "PASS", "settings": settings}


def is_official_eval_solver(solver):
    wrapper = solver.__dict__.get("solve")
    if wrapper is None:
        return False
    closure = inspect.getclosurevars(wrapper).nonlocals
    captured_solver = closure.get("solver")
    if captured_solver is None:
        raise RuntimeError("Could not identify run_paired.wrapped_solve's original solver closure")
    return captured_solver is solver


def solver_identity_check():
    class DummySolver:
        pass

    solver = DummySolver()

    def wrapped_solve():
        return solver

    solver.solve = wrapped_solve
    cloned_solver = copy.deepcopy(solver)
    if not is_official_eval_solver(solver) or is_official_eval_solver(cloned_solver):
        raise RuntimeError("Closed-loop wrapper identity check failed to separate benchmark clones")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve(strict=True))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clone(value, torch, np):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone(item, torch, np) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item, torch, np) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item, torch, np) for item in value)
    return copy.deepcopy(value)


def rng_state(torch, np):
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(state, torch, np):
    python_state, numpy_state, cpu_state, cuda_states = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)


def tensor_bytes_equal(left, right, torch):
    if (not torch.is_tensor(left) or not torch.is_tensor(right)
            or left.shape != right.shape or left.dtype != right.dtype or left.device != right.device):
        return False
    left_bytes = left.contiguous().view(torch.uint8)
    right_bytes = right.contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def cpu_tensor_bytes_equal(left, right, torch):
    if (not torch.is_tensor(left) or not torch.is_tensor(right)
            or left.shape != right.shape or left.dtype != right.dtype
            or left.device.type != "cpu" or right.device.type != "cpu"):
        return False
    return bool(torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def float_lists_equal(left, right):
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    try:
        return all(struct.pack("!d", float(a)) == struct.pack("!d", float(b)) for a, b in zip(left, right))
    except (TypeError, ValueError, OverflowError):
        return False


def finite(value, torch):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item, torch) for item in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


def capture_context(args, seed, torch, np):
    """Use the official evaluator setup and intercept its first B=50 solve."""
    stage, fast_root = args.stage.resolve(strict=True), args.fast_root.resolve(strict=True)
    source, backend = fast_root / "upstream", fast_root / "runtime_overlay"
    sys.path[:0] = [str(source), str(backend)]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"
    import stable_worldmodel as swm
    from stable_worldmodel.solver import CEMSolver

    if not Path(swm.__file__).resolve().is_relative_to(backend.resolve()):
        raise RuntimeError(f"Wrong stable_worldmodel backend: {swm.__file__}")
    if importlib.metadata.version("stable-worldmodel") != "0.0.6":
        raise RuntimeError("The frozen runtime requires stable-worldmodel==0.0.6")
    paired = load_module("cache_b50_paired_runner", args.paired_runner)
    captured = {}

    def intercept(solver, info, *pos, **kwargs):
        if captured:
            raise RuntimeError("Unexpected second native solve during context capture")
        init_action = kwargs.get("init_action", pos[0] if pos else None)
        captured.update(solver=solver, model=solver.model, info=clone(info, torch, np),
                        init_action=clone(init_action, torch, np))
        raise CapturedContext()

    native_solve = CEMSolver.solve
    arm_dir = args.out / "capture" / f"seed{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    child = argparse.Namespace(
        stage=stage, fast_root=fast_root, out=args.out, freeze=args.main_freeze,
        prepared=args.prepared, model="fastlewm", iterations=30, seed=seed,
        arm_id=f"fastlewm_i30_s{seed}", arm_dir=arm_dir)
    try:
        CEMSolver.solve = intercept
        try:
            paired.run_child(child)
        except CapturedContext:
            pass
        else:
            raise RuntimeError("Official evaluation did not reach a native CEM solve")
    finally:
        CEMSolver.solve = native_solve
    if not captured:
        raise RuntimeError("No first-solve context was captured")

    solver, model = captured["solver"], captured["model"]
    solver.__dict__.pop("solve", None)
    counted_cost = model.get_cost
    native_cost = inspect.getclosurevars(counted_cost).nonlocals.get("original_cost")
    if not inspect.ismethod(native_cost):
        raise RuntimeError("Could not restore the original bound native get_cost")
    model.get_cost = native_cost
    if int(solver.n_envs) != 50 or int(solver.batch_size) != 1:
        raise RuntimeError(f"Captured solver drifted: n_envs={solver.n_envs}, batch_size={solver.batch_size}")
    info = captured["info"]
    if not torch.is_tensor(info.get("pixels")) or int(info["pixels"].shape[0]) != 50:
        raise RuntimeError("Captured first-solve context is not the frozen ordered B=50 observation")
    if not torch.is_tensor(info.get("goal")) or int(info["goal"].shape[0]) != 50:
        raise RuntimeError("Captured first-solve context has no matching B=50 goal tensor")
    provenance = {"seed": seed, "context": "first official evaluate_from_dataset solver.solve before sampling",
                  "pbs_job_id": os.environ["PBS_JOBID"], "compute_host": platform.node().split(".")[0],
                  "backend_file": str(Path(swm.__file__).resolve()), "model_source": str(source),
                  "source_commit": args.freeze_data["source_commit"],
                  "checkpoint_revision": args.freeze_data["checkpoint_revision"],
                  "solver_class": type(solver).__module__ + "." + type(solver).__name__,
                  "active_environments": int(info["pixels"].shape[0]),
                  "batch_size": int(solver.batch_size), "solver_environments": int(solver.n_envs)}
    return solver, model, native_cost, info, captured["init_action"], provenance


def solve_once(pristine, model, native_cost, context_info, init_action, cached, torch, np,
               iterations, candidates, trace=False, trace_reference=None):
    solver = copy.deepcopy(pristine, {id(model): model})
    solver.__dict__.pop("solve", None)
    info = clone(context_info, torch, np)
    initial_action = clone(init_action, torch, np)
    original_encode, original_cost = model.encode, model.get_cost
    counts = {"cost_calls": 0, "candidate_scores": 0, "encode_calls": 0,
              "computed": 0, "hits": 0, "resident_entries_peak": 0,
              "payload_peak_bytes": 0, "trace_action_matches": 0, "trace_cost_matches": 0,
              "trace_mismatches": []}
    reference_actions = [] if trace and trace_reference is None else None
    reference_costs = [] if trace and trace_reference is None else None
    cache_holder = {"cache": None, "active_slot": None}

    def counted_cost(local_info, actions):
        call_index = counts["cost_calls"]
        if actions.ndim < 2 or int(actions.shape[0]) != 1 or int(actions.shape[1]) != candidates:
            raise RuntimeError(f"Unexpected cost candidate shape: {tuple(actions.shape)}")
        counts["cost_calls"] += 1
        counts["candidate_scores"] += int(actions.shape[0]) * int(actions.shape[1])
        if trace:
            action_cpu = actions.detach().contiguous().cpu()
            if trace_reference is None:
                reference_actions.append(action_cpu.clone())
            elif call_index >= len(trace_reference["actions"]) or not cpu_tensor_bytes_equal(
                    trace_reference["actions"][call_index], action_cpu, torch):
                counts["trace_mismatches"].append({"cost_call": call_index, "tensor": "candidate_actions"})
            else:
                counts["trace_action_matches"] += 1

        if cached:
            slot = call_index // iterations
            iteration = call_index % iterations
            if slot >= int(context_info["pixels"].shape[0]):
                raise RuntimeError("CEM cost-call schedule exceeded the active environment rows")
            cache = cache_holder["cache"]
            if cache is None:
                cache = cache_holder["cache"] = PerEnvironmentExactCache(torch)
            if cache_holder["active_slot"] != slot:
                if iteration != 0:
                    raise RuntimeError("CEM changed environment slot outside its 30-round boundary")
                cache.activate_environment(slot)
                cache_holder["active_slot"] = slot

            role_index = {"value": 0}
            role_order = ("goal", "current")

            def validated_encode(local_encode_info):
                role_idx = role_index["value"]
                role_index["value"] += 1
                if role_idx >= len(role_order):
                    raise RuntimeError("Unexpected extra native encode call inside get_cost")
                role = role_order[role_idx]
                if "action" in local_encode_info or not torch.is_tensor(local_encode_info.get("pixels")):
                    raise RuntimeError("Cache path received an action-conditioned or malformed encode")
                pixels = local_encode_info["pixels"]
                key = (slot, role)
                had_entry = key in cache.entries
                embedding = cache.lookup(slot, role, pixels)
                if embedding is not None:
                    counts["hits"] += 1
                    local_encode_info["emb"] = embedding
                    return local_encode_info
                if had_entry:
                    raise RuntimeError(f"Exact input changed within environment slot {slot}, role {role}")
                context_pixels = context_info["goal" if role == "goal" else "pixels"][slot:slot + 1]
                context_pixels = context_pixels.to(device=pixels.device)
                if not tensor_bytes_equal(pixels, context_pixels, torch):
                    raise RuntimeError(f"CEM slot {slot} {role} pixels differ from solve-start observation/goal")
                result = original_encode(local_encode_info)
                if "emb" not in result:
                    raise RuntimeError("Native encode returned no embedding")
                cache.store(slot, role, pixels, result["emb"])
                counts["computed"] += 1
                counts["resident_entries_peak"] = max(counts["resident_entries_peak"], len(cache.entries))
                counts["payload_peak_bytes"] = max(counts["payload_peak_bytes"], cache.payload_bytes())
                return result

            model.encode = validated_encode
        if cached:
            counts["encode_calls"] += 2
        result = native_cost(local_info, actions)
        if cached and role_index["value"] != 2:
            raise RuntimeError(f"Native get_cost made {role_index['value']} encodes; expected goal then current")
        if trace:
            costs_cpu = result.detach().contiguous().cpu()
            if trace_reference is None:
                reference_costs.append(costs_cpu.clone())
            elif call_index >= len(trace_reference["costs"]) or not cpu_tensor_bytes_equal(
                    trace_reference["costs"][call_index], costs_cpu, torch):
                counts["trace_mismatches"].append({"cost_call": call_index, "tensor": "candidate_costs"})
            else:
                counts["trace_cost_matches"] += 1
        return result

    model.get_cost = counted_cost
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = int(torch.cuda.memory_allocated())
    started = time.perf_counter()
    try:
        with torch.no_grad():
            output = type(solver).solve(solver, info, init_action=initial_action)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated())
        if counts["cost_calls"] != int(context_info["pixels"].shape[0]) * iterations:
            raise RuntimeError(f"Expected {context_info['pixels'].shape[0] * iterations} get_cost calls; got {counts['cost_calls']}")
        if cached and (counts["computed"] != 2 * int(context_info["pixels"].shape[0])
                       or counts["hits"] != 2 * (int(context_info["pixels"].shape[0]) * iterations
                                                  - int(context_info["pixels"].shape[0]))
                       or counts["resident_entries_peak"] > 2):
            raise RuntimeError(f"Streaming cache contract failed: {counts}")
        if not finite(output, torch):
            raise RuntimeError("Non-finite solver output")
    finally:
        model.get_cost, model.encode = original_cost, original_encode

    result = {"elapsed_seconds": elapsed, "peak_allocated_bytes": peak,
              "peak_increment_bytes": max(0, peak - allocated_before),
              "active_environments": int(context_info["pixels"].shape[0]),
              "cost_calls": counts["cost_calls"], "candidate_scores": counts["candidate_scores"],
              "encode_calls": counts["encode_calls"] if cached else None,
              "cache_computed": counts["computed"],
              "cache_hits": counts["hits"], "cache_resident_entries_peak": counts["resident_entries_peak"],
              "cache_payload_peak_bytes": counts["payload_peak_bytes"], "finite": True,
              "output": {"actions": output["actions"], "costs": output["costs"]}}
    if trace:
        result["trace_reference"] = {"actions": reference_actions, "costs": reference_costs} if trace_reference is None else None
        result["trace_comparison"] = {"candidate_actions_compared": counts["trace_action_matches"],
                                      "candidate_costs_compared": counts["trace_cost_matches"],
                                      "mismatches": counts["trace_mismatches"]}
    return result


def trace_pair(seed, pristine, model, native_cost, info, init_action, settings, torch, np):
    state = rng_state(torch, np)
    try:
        restore_rng(state, torch, np)
        native = solve_once(pristine, model, native_cost, info, init_action, False, torch, np,
                            settings["iterations"], settings["num_samples"], trace=True)
        reference = native["trace_reference"]
        restore_rng(state, torch, np)
        cache = solve_once(pristine, model, native_cost, info, init_action, True, torch, np,
                           settings["iterations"], settings["num_samples"], trace=True,
                           trace_reference=reference)
        actions_equal = tensor_bytes_equal(native["output"]["actions"], cache["output"]["actions"], torch)
        costs_equal = float_lists_equal(native["output"]["costs"], cache["output"]["costs"])
        trace_cmp = cache["trace_comparison"]
        all_trace_equal = (len(reference["actions"]) == len(reference["costs"]) == native["cost_calls"]
                           and trace_cmp["candidate_actions_compared"] == native["cost_calls"]
                           and trace_cmp["candidate_costs_compared"] == native["cost_calls"]
                           and not trace_cmp["mismatches"])
        finite_trace = all(finite(tensor, torch) for tensor in reference["actions"] + reference["costs"])
        return {"seed": seed, "active_environments": int(info["pixels"].shape[0]),
                "native_cost_calls": native["cost_calls"], "cache_cost_calls": cache["cost_calls"],
                "native_candidate_scores": native["candidate_scores"], "cache_candidate_scores": cache["candidate_scores"],
                "candidate_action_tensors_compared": trace_cmp["candidate_actions_compared"],
                "candidate_cost_tensors_compared": trace_cmp["candidate_costs_compared"],
                "candidate_shapes": {"actions": list(reference["actions"][0].shape),
                                     "costs": list(reference["costs"][0].shape)},
                "returned_actions_bitwise_equal": actions_equal, "returned_costs_bitwise_equal": costs_equal,
                "full_candidate_trace_bitwise_equal": all_trace_equal,
                "finite_native_and_trace": native["finite"] and cache["finite"] and finite_trace,
                "cache_contract": {key: cache[key] for key in ("cache_computed", "cache_hits",
                                                                  "cache_resident_entries_peak", "cache_payload_peak_bytes")}}
    finally:
        restore_rng(state, torch, np)


def fixed_seed(args):
    host = require_allocation(True)
    import numpy as np
    import torch
    settings = args.freeze_data["settings"]
    seed = args.seed
    pristine, model, native_cost, info, init_action, provenance = capture_context(args, seed, torch, np)
    if int(info["pixels"].shape[0]) != settings["num_eval"]:
        raise RuntimeError("Captured fixed context is not an active B=50 observation")
    warmups, timed = [], []
    fixed_settings = args.freeze_data["settings"]
    for rep in range(fixed_settings["fixed_context_warmups_per_arm_seed"]):
        order = [False, True] if (seed + rep) % 2 == 0 else [True, False]
        state = rng_state(torch, np)
        try:
            for cached in order:
                restore_rng(state, torch, np)
                item = solve_once(pristine, model, native_cost, info, init_action, cached, torch, np,
                                  settings["iterations"], settings["num_samples"])
                expected_scores = settings["num_eval"] * settings["iterations"] * settings["num_samples"]
                if item["candidate_scores"] != expected_scores:
                    raise RuntimeError("Warmup candidate score count differs from the frozen B=50 contract")
                warmups.append({"rep": rep, "arm": "cache" if cached else "native",
                                **{key: value for key, value in item.items() if key != "output"}})
        finally:
            restore_rng(state, torch, np)

    repetitions = fixed_settings["fixed_context_timed_repetitions_per_arm_seed"]
    for rep in range(repetitions):
        order = [False, True] if (seed + rep) % 2 == 0 else [True, False]
        state = rng_state(torch, np)
        try:
            for cached in order:
                restore_rng(state, torch, np)
                item = solve_once(pristine, model, native_cost, info, init_action, cached, torch, np,
                                  settings["iterations"], settings["num_samples"])
                expected_scores = settings["num_eval"] * settings["iterations"] * settings["num_samples"]
                if item["candidate_scores"] != expected_scores:
                    raise RuntimeError("Timed candidate score count differs from the frozen B=50 contract")
                timed.append({"rep": rep, "arm": "cache" if cached else "native",
                              **{key: value for key, value in item.items() if key != "output"}})
        finally:
            restore_rng(state, torch, np)

    trace = trace_pair(seed, pristine, model, native_cost, info, init_action, settings, torch, np)
    native = [item["elapsed_seconds"] for item in timed if item["arm"] == "native"]
    cached = [item["elapsed_seconds"] for item in timed if item["arm"] == "cache"]
    native_by_rep = {item["rep"]: item["elapsed_seconds"] for item in timed if item["arm"] == "native"}
    cache_by_rep = {item["rep"]: item["elapsed_seconds"] for item in timed if item["arm"] == "cache"}
    paired_reductions = [1.0 - cache_by_rep[index] / native_by_rep[index] for index in range(repetitions)]
    memory_native = max(item["peak_allocated_bytes"] for item in warmups + timed if item["arm"] == "native")
    memory_cache = max(item["peak_allocated_bytes"] for item in warmups + timed if item["arm"] == "cache")
    reduction = float(np.median(paired_reductions))
    memory_ratio = float(memory_cache / memory_native)
    gates = args.freeze_data["gates"]["fixed_context"]
    fidelity_pass = bool(trace["full_candidate_trace_bitwise_equal"]
                         and trace["returned_actions_bitwise_equal"] and trace["returned_costs_bitwise_equal"]
                         and trace["native_cost_calls"] == gates["expected_cost_calls_per_b50_solve"]
                         and trace["cache_cost_calls"] == gates["expected_cost_calls_per_b50_solve"]
                         and trace["native_candidate_scores"] == gates["expected_candidate_scores_per_b50_solve"]
                         and trace["cache_candidate_scores"] == gates["expected_candidate_scores_per_b50_solve"])
    finite_pass = bool(trace["finite_native_and_trace"]
                       and all(item["finite"] for item in warmups + timed))
    result = {"status": "PASS" if fidelity_pass and finite_pass else "NO_GO",
              "protocol": args.freeze_data["protocol"], "seed": seed,
              "pbs_job_id": os.environ["PBS_JOBID"], "compute_host": host,
              "provenance": provenance, "warmups": warmups, "timed": timed, "trace": trace,
              "summary": {"native_median_seconds": float(np.median(native)),
                          "cache_median_seconds": float(np.median(cached)),
                          "paired_latency_reductions": paired_reductions,
                          "median_paired_latency_reduction_fraction": reduction,
                          "peak_allocated_memory_ratio": memory_ratio,
                          "native_peak_allocated_bytes": memory_native,
                          "cache_peak_allocated_bytes": memory_cache,
                          "cache_payload_peak_bytes": max(item["cache_payload_peak_bytes"] for item in warmups + timed if item["arm"] == "cache"),
                          "fidelity_pass": fidelity_pass, "finite_pass": finite_pass,
                          "performance_pass": reduction >= gates["minimum_median_paired_latency_reduction_fraction"],
                          "memory_pass": memory_ratio <= gates["maximum_peak_allocated_memory_ratio"]}}
    write_json(args.out / "fixed" / f"seed{seed}" / "result.json", result)
    print(json.dumps({"event": "fixed_seed_complete", "seed": seed, **result["summary"]}), flush=True)
    return 0 if fidelity_pass and finite_pass else 4


def closed_loop_child(args):
    host = require_allocation(True)
    import numpy as np
    import torch
    stage, fast_root = args.stage.resolve(strict=True), args.fast_root.resolve(strict=True)
    fast_source, backend = fast_root / "upstream", fast_root / "runtime_overlay"
    sys.path[:0] = [str(fast_source), str(backend)]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"
    import stable_worldmodel as swm
    from stable_worldmodel.solver import CEMSolver
    if not Path(swm.__file__).resolve().is_relative_to(backend.resolve()):
        raise RuntimeError(f"Wrong stable_worldmodel backend: {swm.__file__}")
    if importlib.metadata.version("stable-worldmodel") != "0.0.6":
        raise RuntimeError("The frozen runtime requires stable-worldmodel==0.0.6")
    paired = load_module("cache_b50_paired_runner", args.paired_runner)
    original_solve = CEMSolver.solve
    eval_records = []
    settings = args.freeze_data["settings"]
    cache_arm = args.arm == "cache"

    def instrumented_solve(solver, info, init_action=None):
        is_evaluation_solve = is_official_eval_solver(solver)
        if not cache_arm:
            output = original_solve(solver, info, init_action=init_action)
            if is_evaluation_solve:
                eval_records.append({"output": output, "active_environments": int(info["pixels"].shape[0]),
                                    "cost_calls": None, "candidate_scores": None,
                                    "cache": {"enabled": False}})
            return output

        active = int(info["pixels"].shape[0])
        if active != int(solver.n_envs) or int(solver.batch_size) != 1:
            raise RuntimeError(f"Unexpected active CEM batch: active={active}, n_envs={solver.n_envs}, batch={solver.batch_size}")
        model = solver.model
        original_encode, original_cost = model.encode, model.get_cost
        cache = PerEnvironmentExactCache(torch)
        state = {"calls": 0, "scores": 0, "encodes": 0, "computed": 0, "hits": 0,
                 "active_slot": None, "resident_peak": 0, "payload_peak": 0,
                 "eval_cost_calls": 0, "eval_scores": 0}

        def cached_cost(local_info, actions):
            call_index = state["calls"]
            slot, iteration = divmod(call_index, settings["iterations"])
            if slot >= active or int(actions.shape[0]) != 1 or int(actions.shape[1]) != settings["num_samples"]:
                raise RuntimeError("Cached CEM environment/candidate schedule differs from the frozen source")
            if state["active_slot"] != slot:
                if iteration != 0:
                    raise RuntimeError("Native CEM did not transition environments at a 30-round boundary")
                cache.activate_environment(slot)
                state["active_slot"] = slot
            state["calls"] += 1
            state["scores"] += int(actions.shape[0]) * int(actions.shape[1])
            state["eval_cost_calls"] += 1
            state["eval_scores"] += int(actions.shape[0]) * int(actions.shape[1])
            phase = {"count": 0}

            def cached_encode(local_encode_info):
                phase_index = phase["count"]
                phase["count"] += 1
                if phase_index >= 2 or "action" in local_encode_info:
                    raise RuntimeError("Unexpected action-conditioned/extra model.encode call")
                role = ("goal", "current")[phase_index]
                pixels = local_encode_info.get("pixels")
                if not torch.is_tensor(pixels):
                    raise RuntimeError(f"Native {role} encode received non-tensor pixels")
                key = (slot, role)
                had_entry = key in cache.entries
                embedding = cache.lookup(slot, role, pixels)
                if embedding is not None:
                    state["hits"] += 1
                    local_encode_info["emb"] = embedding
                    return local_encode_info
                if had_entry:
                    raise RuntimeError(f"Cached {role} pixels changed within environment {slot}")
                reference = info.get("goal" if role == "goal" else "pixels")
                if slot >= int(reference.shape[0]) or not tensor_bytes_equal(
                        pixels, reference[slot:slot + 1].to(device=pixels.device), torch):
                    raise RuntimeError(f"Native CEM slot {slot} {role} differs from its solve-start input")
                result = original_encode(local_encode_info)
                if "emb" not in result:
                    raise RuntimeError("Native encoder returned no embedding")
                cache.store(slot, role, pixels, result["emb"])
                state["computed"] += 1
                state["resident_peak"] = max(state["resident_peak"], len(cache.entries))
                state["payload_peak"] = max(state["payload_peak"], cache.payload_bytes())
                return result

            model.encode = cached_encode
            result = original_cost(local_info, actions)
            if phase["count"] != 2:
                raise RuntimeError("Native cost path no longer performs goal then current encoding")
            return result

        model.get_cost = cached_cost
        try:
            output = original_solve(solver, info, init_action=init_action)
            if is_evaluation_solve:
                if state["calls"] != active * settings["iterations"]:
                    raise RuntimeError("Closed-loop solve cost-call count is incomplete")
                if state["computed"] != 2 * active or state["hits"] != 2 * active * (settings["iterations"] - 1):
                    raise RuntimeError(f"Closed-loop cache counters violate the frozen contract: {state}")
                eval_records.append({"output": output, "active_environments": active,
                                    "cost_calls": state["calls"], "candidate_scores": state["scores"],
                                    "cache": {"computed": state["computed"], "hits": state["hits"],
                                              "resident_entries_peak": state["resident_peak"],
                                              "payload_peak_bytes": state["payload_peak"]}})
            return output
        finally:
            model.get_cost, model.encode = original_cost, original_encode

    CEMSolver.solve = instrumented_solve
    arm_dir = args.arm_dir.resolve()
    arm_dir.mkdir(parents=True, exist_ok=True)
    child = argparse.Namespace(
        stage=stage, fast_root=fast_root, out=args.out, freeze=args.main_freeze,
        prepared=args.prepared, model="fastlewm", iterations=settings["iterations"],
        seed=args.seed, arm_id=f"fastlewm_i30_s{args.seed}", arm_dir=arm_dir)
    try:
        status = paired.run_child(child)
    finally:
        CEMSolver.solve = original_solve
    result_path = arm_dir / "result.json"
    result = read_json(result_path)
    if status != 0 or result.get("status") != "PASS":
        raise RuntimeError(f"Official evaluation child did not pass: {status}")
    if len(eval_records) != len(result.get("solve_records", [])):
        raise RuntimeError("Instrumented solve outputs do not match official evaluation solve records")
    for recorded, official_record in zip(eval_records, result["solve_records"]):
        actions = recorded["output"]["actions"].detach().contiguous().cpu()
        recorded["actions"] = {"shape": list(actions.shape), "dtype": str(actions.dtype),
                                "values": actions.tolist()}
        recorded["costs"] = list(recorded["output"]["costs"])
        del recorded["output"]
        recorded["elapsed_seconds"] = official_record["elapsed_seconds"]
        recorded["cost_calls"] = official_record["cost_calls"] if recorded["cost_calls"] is None else recorded["cost_calls"]
        recorded["candidate_scores"] = (official_record["candidate_scores"] if recorded["candidate_scores"] is None
                                         else recorded["candidate_scores"])
        if recorded["active_environments"] != official_record["active_environments"]:
            raise RuntimeError("Instrumented and official active-environment counts diverged")
    result["cache_solve_records"] = eval_records
    result["raw_evaluation_timing"] = {
        "evaluation_wall_seconds_including_benchmark": result["evaluation_wall_seconds_including_benchmark"],
        "measured_fixed_benchmark_overhead_seconds": result["benchmark_overhead_seconds"],
        "runner_reported_evaluation_seconds": result["evaluation_seconds"],
        "note": "Raw wall and separately measured benchmark overhead are both retained; no estimated subtraction is used for deployment claims."}
    result["cache_arm"] = args.arm
    result["compute_host_verified"] = host
    write_json(result_path, result)
    print(json.dumps({"event": "closed_loop_arm_complete", "seed": args.seed, "arm": args.arm,
                      "successes": result["successes"], "solve_count": len(eval_records),
                      "raw_evaluation_wall_seconds": result["evaluation_wall_seconds_including_benchmark"]}), flush=True)
    return 0


def float_tree_equal(left, right):
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(float_tree_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return struct.pack("!d", float(left)) == struct.pack("!d", float(right))
    return left == right


def compare_closed_loop(native, cached):
    records_a, records_b = native["cache_solve_records"], cached["cache_solve_records"]
    solve_equal = len(records_a) == len(records_b)
    pairs = []
    for index, (left, right) in enumerate(zip(records_a, records_b)):
        action_equal = (left["actions"]["shape"] == right["actions"]["shape"]
                        and left["actions"]["dtype"] == right["actions"]["dtype"]
                        and float_tree_equal(left["actions"]["values"], right["actions"]["values"]))
        cost_equal = float_lists_equal(left["costs"], right["costs"])
        record = {"ordinal": index, "active_environments": [left["active_environments"], right["active_environments"]],
                  "cost_calls": [left["cost_calls"], right["cost_calls"]],
                  "candidate_scores": [left["candidate_scores"], right["candidate_scores"]],
                  "actions_bitwise_equal": action_equal, "costs_bitwise_equal": cost_equal,
                  "native_actions": left["actions"], "cache_actions": right["actions"],
                  "native_costs": left["costs"], "cache_costs": right["costs"],
                  "cache_contract": right["cache"]}
        pairs.append(record)
        solve_equal = solve_equal and action_equal and cost_equal
    outcomes_equal = native["episode_successes"] == cached["episode_successes"]
    count_shape_equal = all(record["cost_calls"][0] == record["cost_calls"][1]
                            and record["candidate_scores"][0] == record["candidate_scores"][1]
                            and record["active_environments"][0] == record["active_environments"][1]
                            for record in pairs)
    return {"seed": native["seed"], "native_successes": native["successes"],
            "cache_successes": cached["successes"], "ordered_episode_outcomes_equal": outcomes_equal,
            "all_solve_actions_and_costs_bitwise_equal": solve_equal,
            "all_solve_counts_equal": count_shape_equal, "native_raw_e2e_seconds": native["raw_evaluation_timing"],
            "cache_raw_e2e_seconds": cached["raw_evaluation_timing"], "solve_pairs": pairs}


def run_all(args):
    require_allocation(True)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty experiment output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    try:
        entry_check = self_check(args.freeze_data, args.main_data, torch)
        entry_check.update({"pbs_job_id": os.environ["PBS_JOBID"],
                            "compute_host": platform.node().split(".")[0],
                            "phase": "before model/checkpoint/dataset loading"})
        write_json(out / "entry_self_check.json", entry_check)
        print(json.dumps({"event": "entry_self_check", "status": entry_check["status"],
                          "phase": entry_check["phase"]}), flush=True)
    except Exception as exc:
        write_json(out / "entry_self_check.json", {"status": "FAILED", "error": repr(exc),
                                                    "traceback": traceback.format_exc(),
                                                    "pbs_job_id": os.environ["PBS_JOBID"]})
        write_json(out / "run_summary.json", {"status": "FAILED", "stop_reason": "entry_self_check_failed",
                                               "pbs_job_id": os.environ["PBS_JOBID"]})
        return 4
    fixed_results = []
    for seed in args.freeze_data["settings"]["solver_seeds"]:
        command = [sys.executable, str(Path(__file__).resolve()), "--fixed-seed", str(seed),
                   "--freeze", str(args.freeze), "--main-freeze", str(args.main_freeze),
                   "--paired-runner", str(args.paired_runner), "--stage", str(args.stage),
                   "--fast-root", str(args.fast_root), "--prepared", str(args.prepared), "--out", str(out)]
        returned = subprocess.run(command, check=False)
        path = out / "fixed" / f"seed{seed}" / "result.json"
        if returned.returncode != 0 or not path.is_file():
            failed_fixed = read_json(path) if path.is_file() else None
            science_no_go = failed_fixed is not None and failed_fixed.get("status") == "NO_GO"
            write_json(out / "run_summary.json", {"status": "NO_GO" if science_no_go else "FAILED",
                                                    "stop_reason": "fixed_fidelity_gate_failed" if science_no_go else "fixed_technical_failure",
                                                    "fixed_seeds_completed": len(fixed_results), "failed_seed": seed,
                                                    "failed_seed_result": failed_fixed,
                                                    "pbs_job_id": os.environ["PBS_JOBID"]})
            return returned.returncode or 4
        fixed = read_json(path)
        fixed_results.append(fixed)

    fidelity_pass = all(item["summary"]["fidelity_pass"] and item["summary"]["finite_pass"] for item in fixed_results)
    fixed_gates = {"fidelity_pass": fidelity_pass,
                   "performance_pass": all(item["summary"]["performance_pass"] for item in fixed_results),
                   "memory_pass": all(item["summary"]["memory_pass"] for item in fixed_results)}
    summary = {"protocol": args.freeze_data["protocol"], "pbs_job_id": os.environ["PBS_JOBID"],
               "fixed_context": {"gates": fixed_gates,
                                 "seeds": [{"seed": item["seed"], **item["summary"], "trace": item["trace"]}
                                           for item in fixed_results]},
               "closed_loop": {"status": "NOT_RUN"}, "status": "RUNNING"}
    write_json(out / "run_summary.json", summary)
    if not fidelity_pass:
        summary.update(status="NO_GO", stop_reason="fixed_trace_or_finite_gate_failed")
        summary["closed_loop"] = {"status": "STOPPED_BY_FROZEN_STAGE_RULE"}
        write_json(out / "run_summary.json", summary)
        return 4

    comparisons = []
    arm_order = {int(seed): tuple(order) for seed, order in args.freeze_data["settings"]["closed_loop_order_by_seed"].items()}
    for seed in args.freeze_data["settings"]["solver_seeds"]:
        results = {}
        for arm in arm_order[seed]:
            arm_dir = out / "closed_loop" / f"seed{seed}" / arm
            command = [sys.executable, str(Path(__file__).resolve()), "--closed-loop-child",
                       "--arm", arm, "--seed", str(seed), "--arm-dir", str(arm_dir),
                       "--freeze", str(args.freeze), "--main-freeze", str(args.main_freeze),
                       "--paired-runner", str(args.paired_runner), "--stage", str(args.stage),
                       "--fast-root", str(args.fast_root), "--prepared", str(args.prepared),
                       "--out", str(out)]
            returned = subprocess.run(command, check=False)
            path = arm_dir / "result.json"
            if returned.returncode != 0 or not path.is_file():
                summary.update(status="FAILED", stop_reason=f"closed_loop_technical_failure_seed{seed}_{arm}")
                summary["closed_loop"] = {"status": "INCOMPLETE", "completed_seed_pairs": comparisons}
                write_json(out / "run_summary.json", summary)
                return returned.returncode or 5
            results[arm] = read_json(path)
        native, cached = results["native"], results["cache"]
        comparison = compare_closed_loop(native, cached)
        comparison["native_prior_control_exact"] = native["episode_successes"] == args.freeze_data["prior_native_outcomes"][str(seed)]
        comparisons.append(comparison)
        summary["closed_loop"] = {"status": "IN_PROGRESS", "seed_pairs": comparisons}
        write_json(out / "run_summary.json", summary)

    closed_loop_pass = all(item["ordered_episode_outcomes_equal"]
                           and item["all_solve_actions_and_costs_bitwise_equal"]
                           and item["all_solve_counts_equal"]
                           and item["native_prior_control_exact"] is not False for item in comparisons)
    all_science_gates = (fixed_gates["performance_pass"] and fixed_gates["memory_pass"] and closed_loop_pass)
    summary.update(status="PASS" if all_science_gates else "NO_GO",
                   fixed_context={**summary["fixed_context"],
                   "median_paired_latency_reduction_fraction": float(statistics.median(
                                      item["summary"]["median_paired_latency_reduction_fraction"] for item in fixed_results)),
                                  "maximum_peak_allocated_memory_ratio": max(item["summary"]["peak_allocated_memory_ratio"] for item in fixed_results)},
                   closed_loop={"status": "PASS" if closed_loop_pass else "NO_GO", "seed_pairs": comparisons},
                   claim_boundary=args.freeze_data["claim_limit"])
    write_json(out / "run_summary.json", summary)
    print(json.dumps({"event": "experiment_complete", "status": summary["status"],
                      "fixed_gates": fixed_gates, "closed_loop_pass": closed_loop_pass}), flush=True)
    return 0 if summary["status"] == "PASS" else 4


def main():
    parser = argparse.ArgumentParser()
    for name in ("freeze", "main-freeze", "paired-runner", "stage", "fast-root", "prepared", "out", "arm-dir"):
        parser.add_argument("--" + name, type=Path, required=name in ("freeze", "main-freeze", "paired-runner"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=("native", "cache"))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--fixed-seed", type=int)
    parser.add_argument("--closed-loop-child", action="store_true")
    args = parser.parse_args()
    args.freeze_data = read_json(args.freeze)
    args.main_data = read_json(args.main_freeze)
    if args.self_check:
        require_allocation(False)
        import torch
        print(json.dumps(self_check(args.freeze_data, args.main_data, torch), indent=2, ensure_ascii=False))
        return 0
    if not all(getattr(args, name.replace("-", "_")) is not None
               for name in ("stage", "fast-root", "prepared", "out")):
        parser.error("runtime paths are required")
    if args.run_all:
        return run_all(args)
    if args.fixed_seed is not None:
        args.seed = args.fixed_seed
        try:
            return fixed_seed(args)
        except Exception as exc:
            write_json(args.out / "fixed" / f"seed{args.seed}" / "result.json",
                       {"status": "FAILED", "seed": args.seed, "error": repr(exc),
                        "traceback": traceback.format_exc(), "pbs_job_id": os.environ.get("PBS_JOBID")})
            raise
    if args.closed_loop_child:
        if args.seed is None or args.arm is None or args.arm_dir is None:
            parser.error("closed-loop child requires --seed --arm --arm-dir")
        try:
            return closed_loop_child(args)
        except Exception as exc:
            if args.arm_dir:
                write_json(args.arm_dir / "runner_error.json", {"status": "FAILED", "seed": args.seed,
                           "arm": args.arm, "error": repr(exc), "traceback": traceback.format_exc(),
                           "pbs_job_id": os.environ.get("PBS_JOBID")})
            raise
    parser.error("choose --self-check, --run-all, --fixed-seed, or --closed-loop-child")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fixed-context exact encoder-cache pilot for Fast-LeWM."""

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
import struct
import sys
import time
import traceback
from pathlib import Path


class CapturedContext(Exception):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_allocation():
    host = platform.node().split(".")[0].lower()
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    if not os.environ.get("PBS_JOBID") or not nodefile.is_file():
        raise RuntimeError("A real PBS compute allocation is required")
    nodes = {line.strip().split(".")[0].lower() for line in nodefile.read_text().splitlines()}
    if host not in nodes or any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-allocation/control host: {host}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
        raise RuntimeError("A visible allocated GPU is required")
    return host


def validate_freezes(pilot, main):
    if pilot.get("protocol") != "fastlewm-exact-iteration-cache-v1":
        raise RuntimeError("Unexpected pilot freeze")
    if (pilot.get("model"), pilot.get("source_commit"), pilot.get("checkpoint_revision"), pilot.get("backend")) != (
        "fastlewm", main.get("fast_source_commit"), main.get("fast_checkpoint_revision"), "stable-worldmodel==0.0.6"
    ):
        raise RuntimeError("Pilot source/checkpoint/backend differs from the main freeze")
    required = {"solver_seed": 42, "iterations": 30, "num_samples": 300, "topk": 30, "batch_size": 1,
                "actual_active_environments": 1, "horizon": 1, "action_block": 25, "receding_horizon": 1,
                "history_len": 1, "rollout_consistency_weight": 0.0,
                "warmups_per_arm_per_context": 2, "timed_repetitions_per_arm_per_context": 5,
                "trace_pairs_per_context": 1}
    if any(pilot.get("settings", {}).get(k) != v for k, v in required.items()):
        raise RuntimeError("Pilot settings differ from the freeze")
    if pilot.get("task_indices") != list(range(8)) or pilot.get("arms") != ["native", "exact_action_free_encode_cache"]:
        raise RuntimeError("Pilot task or arm schedule differs from the freeze")
    if pilot.get("gates", {}).get("cost_calls_per_solve") != 30 or pilot["gates"].get("candidate_scores_per_solve") != 9000:
        raise RuntimeError("Frozen CEM call contract is malformed")
    plan = main.get("plans", {}).get("fastlewm")
    if plan != {"horizon": 1, "receding_horizon": 1, "action_block": 25, "history_len": 1, "warm_start": True}:
        raise RuntimeError("Main Fast-LeWM plan differs from the pilot contract")


def self_check(pilot, main):
    validate_freezes(pilot, main)
    if not scalar_lists_equal([1.25, -0.0], [1.25, -0.0]) or scalar_lists_equal([-0.0], [0.0]):
        raise RuntimeError("Scalar-cost bitwise self-check failed")
    return {"status": "PASS", "contexts": 8, "arms": pilot["arms"]}


def clone(value, torch, np):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: clone(v, torch, np) for k, v in value.items()}
    if isinstance(value, list):
        return [clone(v, torch, np) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v, torch, np) for v in value)
    return copy.deepcopy(value)


def one_row(value, index, batch, torch, np):
    if torch.is_tensor(value):
        return value[index:index + 1].clone() if value.ndim and value.shape[0] == batch else value.clone()
    if isinstance(value, np.ndarray):
        return value[index:index + 1].copy() if value.ndim and value.shape[0] == batch else value.copy()
    if isinstance(value, dict):
        return {k: one_row(v, index, batch, torch, np) for k, v in value.items()}
    if isinstance(value, list):
        return [one_row(v, index, batch, torch, np) for v in value]
    if isinstance(value, tuple):
        return tuple(one_row(v, index, batch, torch, np) for v in value)
    return copy.deepcopy(value)


def save_rng(torch, np):
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def load_rng(state, torch, np):
    py, nps, cpu, cuda = state
    random.setstate(py)
    np.random.set_state(nps)
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)


def finite(value, torch):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite(v, torch) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v, torch) for v in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


def scalar_lists_equal(left, right):
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    if any(type(value) is not float for value in left + right):
        return False
    return all(struct.pack("!d", a) == struct.pack("!d", b) for a, b in zip(left, right))


def tensor_bits_equal(left, right, torch):
    if not torch.is_tensor(left) or not torch.is_tensor(right) or left.dtype != right.dtype or left.shape != right.shape:
        return False
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def capture_context(args, pilot, main, host, torch, np):
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

    spec = importlib.util.spec_from_file_location("frozen_main_paired_runner", args.paired_runner.resolve(strict=True))
    paired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired)
    captured = {}

    def intercept(solver, info, *pos, **kwargs):
        if captured:
            raise RuntimeError("Unexpected second native solve during bootstrap")
        init_action = kwargs.get("init_action", pos[0] if pos else None)
        captured.update(solver=solver, model=solver.model, info=clone(info, torch, np),
                        init_action=clone(init_action, torch, np))
        raise CapturedContext()

    native_solve = CEMSolver.solve
    try:
        CEMSolver.solve = intercept
        child = argparse.Namespace(stage=stage, fast_root=fast_root, out=args.out, freeze=args.main_freeze,
                                   prepared=args.prepared, model="fastlewm", iterations=30, seed=42,
                                   arm_id="fastlewm_i30_s42", arm_dir=args.out / "bootstrap")
        child.arm_dir.mkdir(parents=True, exist_ok=True)
        try:
            paired.run_child(child)
        except CapturedContext:
            pass
        else:
            raise RuntimeError("Bootstrap did not reach the native solver")
    finally:
        CEMSolver.solve = native_solve
    if not captured:
        raise RuntimeError("No first-solve context was captured")

    solver, model = captured["solver"], captured["model"]
    solver.__dict__.pop("solve", None)
    wrapped_cost = model.get_cost
    original_cost = inspect.getclosurevars(wrapped_cost).nonlocals.get("original_cost")
    if original_cost is None or not inspect.ismethod(original_cost):
        raise RuntimeError("Could not restore bound native get_cost from run_child's counted_cost")
    model.get_cost = original_cost

    batch = int(main["settings"]["num_eval"])
    info = captured["info"]
    if not torch.is_tensor(info.get("pixels")) or int(info["pixels"].shape[0]) != batch:
        raise RuntimeError("Captured first-solve context does not have the frozen 50-row batch")
    contexts = [{"task_index": int(i), "info": one_row(info, int(i), batch, torch, np),
                 "init_action": one_row(captured["init_action"], int(i), batch, torch, np)}
                for i in pilot["task_indices"]]
    if any(int(context["info"]["pixels"].shape[0]) != 1 for context in contexts):
        raise RuntimeError("A fixed pilot context does not have actual batch size one")
    solver.configure(action_space=solver._action_space, n_envs=1, config=solver._config)
    provenance = {"bootstrap": "paired.run_child first solve intercepted before native CEM sampling",
                  "pbs_job_id": os.environ["PBS_JOBID"], "compute_host": host,
                  "backend_file": str(Path(swm.__file__).resolve()), "model_source": str(source),
                  "jepa_source": str(Path(sys.modules["jepa"].__file__).resolve()),
                  "module_source": str(Path(sys.modules["module"].__file__).resolve()),
                  "prepared": str(args.prepared.resolve()), "source_commit": pilot["source_commit"],
                  "checkpoint_revision": pilot["checkpoint_revision"],
                  "solver_class": type(solver).__module__ + "." + type(solver).__name__,
                  "captured_batch": batch, "contexts": len(contexts)}
    return paired, solver, model, original_cost, contexts, provenance


def solve_once(pristine, model, original_cost, context, cached, torch, np, trace=False):
    solver = copy.deepcopy(pristine, {id(model): model})
    solver.__dict__.pop("solve", None)
    info, init_action = clone(context["info"], torch, np), clone(context["init_action"], torch, np)
    original_encode = model.encode
    cache_state = {"calls": 0, "computed": 0, "hits": 0}
    embeddings, shapes = {}, {}

    if cached:
        def cached_encode(local_info):
            if "action" in local_info or not hasattr(local_info.get("pixels"), "shape"):
                raise RuntimeError("Unexpected action-conditioned or malformed encode")
            role = "goal" if cache_state["calls"] % 2 == 0 else "current"
            shape = tuple(int(x) for x in local_info["pixels"].shape)
            if role in shapes and shapes[role] != shape:
                raise RuntimeError(f"Unexpected {role} encode shape: {shape}")
            shapes[role] = shape
            cache_state["calls"] += 1
            if role not in embeddings:
                result = original_encode(local_info)
                if "emb" not in result:
                    raise RuntimeError("Native encode returned no emb")
                embeddings[role] = result["emb"]
                cache_state["computed"] += 1
                return result
            local_info["emb"] = embeddings[role]
            cache_state["hits"] += 1
            return local_info
        model.encode = cached_encode

    counts = {"calls": 0, "scores": 0, "actions": [], "costs": []}
    def counted_cost(local_info, actions):
        counts["calls"] += 1
        if actions.ndim >= 2:
            counts["scores"] += int(actions.shape[0]) * int(actions.shape[1])
        if trace:
            counts["actions"].append(actions.detach().cpu().clone())
        result = original_cost(local_info, actions)
        if trace:
            counts["costs"].append(result.detach().cpu().clone())
        return result

    model.get_cost = counted_cost
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = int(torch.cuda.memory_allocated())
    start = time.perf_counter()
    try:
        with torch.no_grad():
            output = type(solver).solve(solver, info, init_action=init_action)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = int(torch.cuda.max_memory_allocated())
    finally:
        model.get_cost, model.encode = original_cost, original_encode

    cache_ok = not cached or (cache_state["calls"] == 60 and cache_state["computed"] == 2 and cache_state["hits"] == 58)
    result = {"elapsed_seconds": elapsed, "peak_allocated_bytes": peak,
              "peak_increment_bytes": max(0, peak - allocated), "cost_calls": counts["calls"],
              "candidate_scores": counts["scores"], "finite": finite(output, torch),
              "cache_contract": cache_ok, "output": {key: output.get(key) for key in ("actions", "costs")}}
    if trace:
        result["trace"] = {"actions": counts["actions"], "costs": counts["costs"]}
        result["finite"] = result["finite"] and finite(result["trace"], torch)
    return result


def same_trace(native, cached, torch):
    if not tensor_bits_equal(native["output"].get("actions"), cached["output"].get("actions"), torch):
        return False
    if not scalar_lists_equal(native["output"].get("costs"), cached["output"].get("costs")):
        return False
    for key in ("actions", "costs"):
        left, right = native["trace"][key], cached["trace"][key]
        if len(left) != len(right) or any(not tensor_bits_equal(a, b, torch) for a, b in zip(left, right)):
            return False
    return True


def run(args):
    host = require_allocation()
    pilot, main = read_json(args.freeze), read_json(args.main_freeze)
    validate_freezes(pilot, main)
    prepared = args.prepared.resolve(strict=True)
    if prepared != Path(pilot["main_preparation"]).resolve(strict=True):
        raise RuntimeError("Prepared path differs from the freeze")
    tasks = read_json(args.tasks).get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 50 or [int(t["task_index"]) for t in tasks] != list(range(50)):
        raise RuntimeError("Source task manifest differs from the frozen 50 rows")

    import numpy as np
    import torch
    _, pristine, model, original_cost, contexts, provenance = capture_context(args, pilot, main, host, torch, np)
    settings, gate = pilot["settings"], pilot["gates"]
    result_contexts = []
    for context in contexts:
        row = {"task_index": context["task_index"], "warmups": [], "timed": []}
        for rep in range(int(settings["warmups_per_arm_per_context"])):
            order = [False, True] if (context["task_index"] + rep) % 2 == 0 else [True, False]
            paired_rng = save_rng(torch, np)
            try:
                for cached in order:
                    load_rng(paired_rng, torch, np)
                    item = solve_once(pristine, model, original_cost, context, cached, torch, np)
                    row["warmups"].append({"arm": "cache" if cached else "native", **{k: v for k, v in item.items() if k != "output"}})
            finally:
                load_rng(paired_rng, torch, np)
        for rep in range(int(settings["timed_repetitions_per_arm_per_context"])):
            order = [False, True] if (context["task_index"] + rep) % 2 == 0 else [True, False]
            paired_rng = save_rng(torch, np)
            try:
                for cached in order:
                    load_rng(paired_rng, torch, np)
                    item = solve_once(pristine, model, original_cost, context, cached, torch, np)
                    if item["cost_calls"] != gate["cost_calls_per_solve"] or item["candidate_scores"] != gate["candidate_scores_per_solve"] or not item["cache_contract"]:
                        raise RuntimeError(f"Frozen solve contract failed at task {context['task_index']}")
                    row["timed"].append({"rep": rep, "arm": "cache" if cached else "native",
                                         **{k: v for k, v in item.items() if k != "output"}})
            finally:
                load_rng(paired_rng, torch, np)

        paired_rng = save_rng(torch, np)
        try:
            traces = {}
            order = [False, True] if context["task_index"] % 2 == 0 else [True, False]
            for cached in order:
                load_rng(paired_rng, torch, np)
                traces["cache" if cached else "native"] = solve_once(
                    pristine, model, original_cost, context, cached, torch, np, trace=True)
            a, b = traces["native"], traces["cache"]
            row["trace"] = {"bitwise_equal": same_trace(a, b, torch), "cache_contract": b["cache_contract"],
                             "finite": a["finite"] and b["finite"],
                             "cost_calls": [a["cost_calls"], b["cost_calls"]],
                             "candidate_scores": [a["candidate_scores"], b["candidate_scores"]],
                             "candidate_shapes": [list(x.shape) for x in a["trace"]["actions"]],
                             "cost_shapes": [list(x.shape) for x in a["trace"]["costs"]]}
            row["fidelity_pass"] = (row["trace"]["bitwise_equal"] and row["trace"]["cache_contract"]
                                    and row["trace"]["finite"] and row["trace"]["cost_calls"] == [30, 30]
                                    and row["trace"]["candidate_scores"] == [9000, 9000])
        finally:
            load_rng(paired_rng, torch, np)

        native = [x for x in row["timed"] if x["arm"] == "native"]
        cached = [x for x in row["timed"] if x["arm"] == "cache"]
        row["finite_solve_outputs"] = all(x["finite"] for x in row["warmups"] + row["timed"])
        native_median, cache_median = float(np.median([x["elapsed_seconds"] for x in native])), float(np.median([x["elapsed_seconds"] for x in cached]))
        native_by_rep = {int(x["rep"]): float(x["elapsed_seconds"]) for x in native}
        cache_by_rep = {int(x["rep"]): float(x["elapsed_seconds"]) for x in cached}
        paired_reductions = [1.0 - cache_by_rep[i] / native_by_rep[i] for i in sorted(native_by_rep)]
        row["summary"] = {"native_median_seconds": native_median, "cache_median_seconds": cache_median,
                           "paired_latency_reduction_fractions": paired_reductions,
                           "latency_reduction_fraction": float(np.median(paired_reductions)),
                           "peak_memory_ratio": max(x["peak_allocated_bytes"] for x in cached) /
                                                max(x["peak_allocated_bytes"] for x in native)}
        result_contexts.append(row)
        write_json(args.out / "result.json", {"status": "RUNNING", "contexts": result_contexts, "provenance": provenance})
        print(json.dumps({"event": "context_complete", "task_index": context["task_index"],
                          "fidelity_pass": row["fidelity_pass"], **row["summary"]}), flush=True)

    median_reduction = float(np.median([x["summary"]["latency_reduction_fraction"] for x in result_contexts]))
    max_memory_ratio = max(x["summary"]["peak_memory_ratio"] for x in result_contexts)
    fidelity = all(x["fidelity_pass"] for x in result_contexts)
    finite_outputs = all(x["finite_solve_outputs"] for x in result_contexts)
    performance = median_reduction >= float(gate["minimum_median_context_latency_reduction_fraction"])
    memory = max_memory_ratio <= float(gate["maximum_peak_memory_ratio"])
    final = {"status": "PASS" if fidelity and finite_outputs and performance and memory else "NO_GO",
             "protocol": pilot["protocol"], "pbs_job_id": os.environ["PBS_JOBID"], "compute_host": host,
             "contexts": result_contexts,
             "summary": {"context_count": len(result_contexts), "median_context_latency_reduction_fraction": median_reduction,
                         "maximum_context_peak_memory_ratio": max_memory_ratio, "fidelity_pass": fidelity,
                         "finite_solve_outputs": finite_outputs,
                         "performance_pass": performance, "memory_pass": memory},
             "provenance": provenance,
             "claim_boundary": "Eight fixed B=1 observations; bitwise planner trace and latency only; no closed-loop or deployment claim."}
    write_json(args.out / "result.json", final)
    print(json.dumps({"event": "pilot_complete", "status": final["status"], **final["summary"]}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    for name in ("freeze", "main-freeze", "stage", "fast-root", "out", "prepared", "tasks", "paired-runner"):
        parser.add_argument("--" + name, type=Path, required=name in ("freeze", "main-freeze"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    pilot, main_freeze = read_json(args.freeze), read_json(args.main_freeze)
    if args.self_check:
        print(json.dumps(self_check(pilot, main_freeze), indent=2, ensure_ascii=False))
        return 0
    if any(getattr(args, name.replace("-", "_")) is None for name in ("stage", "fast-root", "out", "prepared", "tasks", "paired-runner")):
        parser.error("runtime paths are required")
    try:
        return run(args)
    except Exception as exc:
        if args.out and args.out.is_dir():
            prior_path = args.out / "result.json"
            prior = read_json(prior_path) if prior_path.is_file() else None
            write_json(prior_path, {"status": "FAILED", "error": repr(exc),
                       "traceback": traceback.format_exc(), "pbs_job_id": os.environ.get("PBS_JOBID"),
                       "partial_result": prior})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

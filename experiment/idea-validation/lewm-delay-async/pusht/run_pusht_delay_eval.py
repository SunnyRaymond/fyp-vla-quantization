#!/usr/bin/env python3
"""Run frozen PushT K0 parity, paired fixed-K, or strict true-async tasks."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from adapter import (
    K_STEPS,
    DelayAsyncAdapter,
    paired_task_seed,
    pusht_control_tick,
    read_reference,
    validate_k0_exactness,
)

BASELINE_JOB = "25534994.pbs101"
HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch50_official_k0", "paired_k0", "fixed_steps", "true_async"))
    parser.add_argument("--task-start", type=int, default=0, help="inclusive index in the frozen 50-task list")
    parser.add_argument("--task-stop", type=int, default=50, help="exclusive index in the frozen 50-task list")
    parser.add_argument("--delay-steps", type=int, default=0, choices=K_STEPS, help="fixed simulation ticks K")
    parser.add_argument("--k0-gate", type=Path, help="PASS batch50_official_k0.json required for N=1 arms")
    for name in ("lewm-root", "stablewm-root", "control-root", "staged-home", "cache-root", "freeze", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--self-check", action="store_true", help="run the stdlib adapter checks without loading model code")
    return parser.parse_args()


def require_compute_node() -> str:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("runner requires PBS_JOBID and a real PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    allocated = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in allocated:
        raise RuntimeError(f"host {host} is not present in PBS_NODEFILE")
    return host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def jsonable(value: Any, np: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item, np, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item, np, torch) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def check_frozen_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, int]], dict[str, Any]]:
    freeze = read_json(args.freeze.resolve(strict=True))
    protocol = freeze.get("protocol", {})
    if (
        freeze.get("schema") != "lewm-pusht-cem-budget.freeze"
        or protocol.get("stable_worldmodel_commit_reference") != "10c26dbd5677083fa31dba69eb738b973845e9a4"
        or protocol.get("lewm_commit_reference") != "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
        or protocol.get("solver_fixed") != {"target": "stable_worldmodel.solver.CEMSolver", "batch_size": 1, "iterations": 30, "var_scale": 1.0, "device": "cuda", "seed": 42}
        or protocol.get("solver_fixed", {}).get("num_samples", 300) != 300
        or protocol.get("solver_fixed", {}).get("topk", 30) != 30
        or protocol.get("plan_config") != {"horizon": 5, "receding_horizon": 5, "action_block": 5}
    ):
        raise RuntimeError("frozen LeWM/StableWorldModel/CEM protocol identity mismatch")
    tasks, reference = read_reference(HERE / "selected_tasks_25534994.json", HERE / "reference_outcomes.json")
    if len(tasks) != 50 or reference["source_job"] != BASELINE_JOB:
        raise RuntimeError("expected the frozen 50-task 25534994.pbs101 PushT list")
    if args.task_start < 0 or args.task_stop > 50 or args.task_start >= args.task_stop:
        raise ValueError("task slice must satisfy 0 <= task-start < task-stop <= 50")
    if args.mode == "batch50_official_k0" and (args.task_start, args.task_stop) != (0, 50):
        raise ValueError("batch50_official_k0 always evaluates the complete ordered 50-task gate")
    if args.mode == "fixed_steps" and args.delay_steps not in K_STEPS:
        raise ValueError(f"fixed_steps K must be one of {K_STEPS}")
    if args.mode in {"paired_k0", "true_async"} and args.delay_steps != 0:
        raise ValueError(f"{args.mode} uses delay_steps=0; K applies only to fixed_steps")
    if args.mode != "batch50_official_k0":
        if args.k0_gate is None:
            raise ValueError("N=1 arms require --k0-gate pointing to the completed batch50_official_k0.json")
        gate = read_json(args.k0_gate.resolve(strict=True))
        if gate.get("mode") != "batch50_official_k0" or gate.get("gate_status") != "PASS" or not gate.get("success_vector_matches_reference"):
            raise RuntimeError("N=1 arm refused because the batch-50 K0 exactness gate did not pass")
    return freeze, tasks, reference


def check_stage_paths(args: argparse.Namespace, protocol: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    lewm = args.lewm_root.resolve(strict=True)
    stablewm = args.stablewm_root.resolve(strict=True)
    control = args.control_root.resolve(strict=True)
    staged = args.staged_home.resolve(strict=True)
    cache = args.cache_root.resolve(strict=True)
    helper = control / "official-lewm-dataset-teacher-baseline"
    h5 = staged / "pusht_expert_train.h5"
    ckpt = staged / "pusht" / "lewm_object.ckpt"
    if not (lewm / "config" / "eval" / "pusht.yaml").is_file() or not (stablewm / "stable_worldmodel" / "policy.py").is_file():
        raise FileNotFoundError("pinned LeWM or stable-worldmodel source tree is incomplete")
    if not (helper / "run_dataset_teacher_baseline.py").is_file() or not (control / "run_lewm_recurrent_student.py").is_file():
        raise FileNotFoundError("existing PushT evaluator helper/checkpoint loader is missing")
    if not h5.is_file() or not ckpt.is_file():
        raise FileNotFoundError("staged PushT dataset or official checkpoint is missing")
    if str(h5.resolve(strict=True)) != protocol["dataset_path"] or str(ckpt.resolve(strict=True)) != protocol["checkpoint_path"]:
        raise RuntimeError("staged dataset/checkpoint paths differ from the frozen source identity")
    if h5.stat().st_size != int(protocol["dataset_size_bytes"]) or ckpt.stat().st_size != int(protocol["checkpoint_size_bytes"]):
        raise RuntimeError("staged dataset/checkpoint sizes differ from the frozen source identity")
    if (cache / "datasets" / h5.name).resolve(strict=True) != h5.resolve(strict=True):
        raise RuntimeError("job-private dataset cache link is not the frozen staged HDF5")
    if (cache / "pusht" / ckpt.name).resolve(strict=True) != ckpt.resolve(strict=True):
        raise RuntimeError("job-private checkpoint cache link is not the frozen staged checkpoint")
    return lewm, stablewm, control, staged, cache


def task_result_path(out: Path, task: dict[str, int]) -> Path:
    return out / "tasks" / f"task_{task['row_index']}.json"


def evaluate_one(
    *, mode: str, delay_steps: int, task_index: int, task: dict[str, int],
    cfg: Any, baseline: Any, swm: Any, dataset: Any, process: dict[str, Any],
    transforms: Any, torch: Any, spt: Any, teacher: Any, OmegaConf: Any,
    out: Path, gate_identity: str | None,
) -> dict[str, Any]:
    import hydra

    task_seed = paired_task_seed(task_index)
    cfg_i = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_i.world.num_envs = 1
    cfg_i.solver.seed = task_seed
    solver = hydra.utils.instantiate(cfg_i.solver, model=teacher)
    if int(solver.torch_gen.initial_seed()) != task_seed:
        raise RuntimeError("N=1 solver did not retain the frozen per-task paired seed")
    world = baseline.make_pinned_world(swm, cfg_i)
    image_transform = baseline.make_image_transform(cfg_i, spt, transforms, torch)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg_i.plan_config),
        process=dict(process),
        transform={"pixels": image_transform, "goal": baseline.make_image_transform(cfg_i, spt, transforms, torch)},
    )
    world.set_policy(policy)
    adapter_mode = "true_async" if mode == "true_async" else "fixed_steps"
    tick = pusht_control_tick(world.envs)
    adapter = DelayAsyncAdapter(
        policy,
        world.envs,
        task_ids=[task["row_index"]],
        mode=adapter_mode,
        delay_steps=delay_steps if mode == "fixed_steps" else 0,
        tick_seconds=tick["control_tick_seconds"] if mode == "true_async" else None,
        synchronize=torch.cuda.synchronize,
    )
    started = time.monotonic()
    try:
        metrics = baseline.evaluate_selected_tasks(world, dataset, [task], cfg_i, OmegaConf)
        torch.cuda.synchronize()
    finally:
        adapter.close()
        world.close()
    metrics = jsonable(metrics, __import__("numpy"), torch)
    successes = [bool(value) for value in metrics.get("episode_successes", [])]
    if len(successes) != 1:
        raise RuntimeError(f"World.evaluate returned {len(successes)} outcomes for one frozen task")
    events_path = out / "tasks" / f"task_{task['row_index']}.events.jsonl"
    write_events(events_path, adapter.events)
    return {
        "status": "COMPLETED",
        "mode": mode,
        "delay_steps": delay_steps if mode == "fixed_steps" else 0,
        "task_index": task_index,
        "task_id": int(task["row_index"]),
        "task": task,
        "task_seed_rule": "42 + frozen_task_index",
        "solver_seed": task_seed,
        "gate_reference": gate_identity,
        "success": successes[0],
        "final_task_status": adapter.status[0],
        "metrics": metrics,
        "control_tick": tick,
        "adapter_event_count": len(adapter.events),
        "adapter_events_file": events_path.name,
        "evaluation_wall_seconds": time.monotonic() - started,
        "rng_scope_note": "N=1 seed is paired across synchronous K0, fixed-K, and true-async arms; it is not RNG-identical to the official batch-50 sequence.",
    }


def run_self_check() -> dict[str, Any]:
    """Exercise fixed-K tick age and wall-clock pacing without ML packages."""
    from collections import deque

    frozen_tasks, reference = read_reference(HERE / "selected_tasks_25534994.json", HERE / "reference_outcomes.json")
    validate_k0_exactness(frozen_tasks, reference["ordered_episode_successes"], reference)

    class Space:
        low, high, dtype, shape = (-1.0,), (1.0,), float, (1,)

    class Env:
        unwrapped = None
        action_space = Space()
        dt = 0.01
        control_hz = 10.0
        metadata = {"render_fps": 10.0}

        def __init__(self) -> None:
            self.unwrapped = self

    class Pool:
        envs = [Env()]

        def step(self, action: Any) -> tuple[Any, Any, list[bool], list[bool], dict[str, Any]]:
            return None, None, [False], [False], {}

    class Policy:
        def __init__(self) -> None:
            self._action_buffer = [deque()]
            self._next_init = None
            self.solver = object()

        def get_action(self, info: Any) -> int:
            if not self._action_buffer[0]:
                self._action_buffer[0].extend(range(25))
            return self._action_buffer[0].popleft()

    class ProbeAdapter(DelayAsyncAdapter):
        def _neutral_action(self) -> int:
            return -1

    direct_policy, direct_pool = Policy(), Pool()
    direct_adapter = DelayAsyncAdapter(direct_policy, direct_pool, task_ids=[100], mode="fixed_steps", delay_steps=0)
    direct_action = direct_policy.get_action({})
    assert direct_action == 0 and len(direct_policy._action_buffer[0]) == 24
    direct_output = next(event for event in direct_adapter.events if event["event"] == "action_output")
    assert direct_output["action_source"] == "native_k0"
    direct_adapter.close()

    class BatchPolicy:
        def __init__(self, count: int) -> None:
            self._action_buffer = [deque(range(25)) for _ in range(count)]
            self._next_init = None
            self.solver = object()

        def get_action(self, info: Any) -> list[int]:
            return [queue.popleft() for queue in self._action_buffer]

    batch_count = 50
    batch_pool = Pool()
    batch_pool.envs = [Env() for _ in range(batch_count)]
    batch_policy = BatchPolicy(batch_count)
    batch_adapter = DelayAsyncAdapter(batch_policy, batch_pool, task_ids=list(range(batch_count)), mode="fixed_steps", delay_steps=0)
    native_actions = batch_policy.get_action({})
    assert native_actions == [0] * batch_count
    batch_adapter.finish()
    batch_drops = [event for event in batch_adapter.events if event["event"] == "terminal_drop"]
    assert len(batch_drops) == batch_count
    assert all(event["buffered_actions"] == 24 and len(event["task_ids"]) == 1 for event in batch_drops)
    assert batch_adapter.status == ["budget_exhausted"] * batch_count
    batch_adapter.close()

    delays: dict[str, int] = {}
    for k in (1, 2, 4, 8, 16):
        policy, pool = Policy(), Pool()
        adapter = ProbeAdapter(policy, pool, task_ids=[100], mode="fixed_steps", delay_steps=k)
        first_action = policy.get_action({})
        pool.step(first_action)
        while True:
            action = policy.get_action({})
            release = next(event for event in reversed(adapter.events) if event["event"] == "action_output")
            if release["action_source"] == "fixed_delay_release":
                apply_step = adapter.control_step
                break
            pool.step(action)
        delays[str(k)] = apply_step
        assert apply_step == k, (k, apply_step)
        adapter.close()

    class FakeClock:
        now = 1_000
        sleeps: list[float] = []

        def clock(self) -> int:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += int(seconds * 1e9)

    clock = FakeClock()
    pacing = DelayAsyncAdapter.__new__(DelayAsyncAdapter)
    pacing.mode = "true_async"
    pacing.tick_seconds = 0.0000001
    pacing._period_ns = 100
    pacing._next_tick_ns = None
    pacing._last_scheduled_tick_ns = None
    pacing._clock_ns = clock.clock
    pacing._sleep = clock.sleep
    pacing.control_step = 0
    pacing.events = []
    pacing._event_id = 0
    pacing._lock = __import__("threading").Lock()
    first, second, third = pacing._pace_step(), pacing._pace_step(), pacing._pace_step()
    assert [first, second, third] == [1_000, 1_100, 1_200]
    assert clock.sleeps == [1e-7, 1e-7]
    return {"status": "PASS", "checks": ["frozen_k0_vector", "native_k0_passthrough", "batch50_finish_drop_accounting", "fixed_k_exact_tick_age", "true_async_wall_pacing"], "batch50_task_drops": len(batch_drops), "fixed_k_apply_steps": delays, "async_tick_starts_ns": [first, second, third], "mock_sleep_count": len(clock.sleeps)}


def main() -> int:
    args = parse_args()
    if args.self_check:
        print(json.dumps(run_self_check(), ensure_ascii=False))
        return 0
    if args.mode is None:
        raise ValueError("--mode is required outside --self-check")

    for name in ("lewm_root", "stablewm_root", "control_root", "staged_home", "cache_root", "freeze", "out"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} is required outside --self-check")

    host = require_compute_node()
    freeze, tasks, reference = check_frozen_inputs(args)
    out = args.out.resolve(strict=True)
    if out == Path("/") or not out.is_dir():
        raise RuntimeError("PBS wrapper must create the unique arm output directory first")
    if (out / "batch50_official_k0.json").exists() and args.mode == "batch50_official_k0":
        raise FileExistsError("refusing to overwrite batch50 gate output")
    gate_identity = str(args.k0_gate.resolve()) if args.k0_gate else None
    task_slice = tasks[args.task_start:args.task_stop]
    if args.mode == "batch50_official_k0":
        task_slice = tasks

    lewm_root, stablewm_root, control_root, staged_home, cache_root = check_stage_paths(args, freeze["protocol"])
    baseline_dir = control_root / "official-lewm-dataset-teacher-baseline"
    sys.path[:0] = [str(lewm_root), str(stablewm_root), str(control_root), str(baseline_dir)]
    os.environ["STABLEWM_HOME"] = str(cache_root)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Import data/model runtimes only after PBS allocation and source/path guards.
    import hdf5plugin  # noqa: F401
    import hydra
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    import run_dataset_teacher_baseline as baseline
    from run_lewm_recurrent_student import load_official_checkpoint
    import jepa

    if not Path(swm.__file__).resolve().is_relative_to(stablewm_root):
        raise RuntimeError("stable_worldmodel imported outside the frozen source root")
    venv_root = (stablewm_root.parent / "venv").resolve(strict=True)
    if Path(spt.__file__).resolve().is_relative_to(venv_root) is False or Path(sys.prefix).resolve(strict=True) != venv_root:
        raise RuntimeError("stable_pretraining or Python runtime is outside the staged virtual environment")
    if not Path(jepa.__file__).resolve().is_relative_to(lewm_root):
        raise RuntimeError("LeWM jepa module imported outside the frozen source root")
    if not Path(baseline.__file__).resolve().is_relative_to(baseline_dir):
        raise RuntimeError("PushT baseline helper imported outside the control source tree")
    if not Path(inspect.getsourcefile(load_official_checkpoint)).resolve().is_relative_to(control_root):
        raise RuntimeError("official LeWM checkpoint loader imported outside the control source tree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pinned LeWM CEM evaluation")
    cfg = baseline.compose_pinned_config(lewm_root)
    cfg.world.num_envs = 50 if args.mode == "batch50_official_k0" else 1
    cfg.solver.seed = 42
    if int(cfg.eval.eval_budget) != 50 or int(cfg.world.max_episode_steps) != 100:
        raise RuntimeError("pinned evaluation budgets changed")
    dataset = swm.data.HDF5Dataset(str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=cache_root)
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    selected = dataset.get_row_data(np.asarray([row["row_index"] for row in task_slice], dtype=np.int64))
    for i, task in enumerate(task_slice):
        if int(selected[episode_column][i]) != task["episode_idx"] or int(selected["step_idx"][i]) != task["start_step"]:
            raise RuntimeError(f"frozen PushT task differs from staged HDF5 at frozen index {args.task_start + i}")
    teacher = load_official_checkpoint(cache_root).to("cuda").eval()
    teacher.requires_grad_(False)
    teacher.interpolate_pos_encoding = True

    if args.mode == "batch50_official_k0":
        cfg.world.num_envs = 50
        solver = hydra.utils.instantiate(cfg.solver, model=teacher)
        if int(solver.torch_gen.initial_seed()) != 42:
            raise RuntimeError("batch-50 K0 solver seed differs from official seed 42")
        world = baseline.make_pinned_world(swm, cfg)
        image_transform = baseline.make_image_transform(cfg, spt, transforms, torch)
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(**cfg.plan_config),
            process=dict(process),
            transform={"pixels": image_transform, "goal": baseline.make_image_transform(cfg, spt, transforms, torch)},
        )
        world.set_policy(policy)
        adapter = DelayAsyncAdapter(policy, world.envs, task_ids=[row["row_index"] for row in tasks], mode="fixed_steps", delay_steps=0, synchronize=torch.cuda.synchronize)
        try:
            metrics = baseline.evaluate_selected_tasks(world, dataset, tasks, cfg, OmegaConf)
            torch.cuda.synchronize()
        finally:
            adapter.close()
            world.close()
        metrics = jsonable(metrics, np, torch)
        successes = [bool(x) for x in metrics.get("episode_successes", [])]
        observed_tasks = [{key: int(row[key]) for key in ("row_index", "episode_idx", "start_step")} for row in tasks]
        gate_status = "PASS"
        gate_error = None
        try:
            validate_k0_exactness(observed_tasks, successes, reference)
        except Exception as exc:
            gate_status, gate_error = "STOP", f"{type(exc).__name__}: {exc}"
        write_events(out / "batch50_events.jsonl", adapter.events)
        result = {
            "status": "COMPLETED" if len(successes) == 50 else "INVALID",
            "mode": args.mode,
            "gate_status": gate_status if len(successes) == 50 else "STOP",
            "success_vector_matches_reference": gate_status == "PASS" and len(successes) == 50,
            "gate_error": gate_error,
            "task_count": len(successes),
            "successes": sum(successes),
            "episode_successes": successes,
            "final_task_statuses": list(adapter.status),
            "tasks": tasks,
            "reference_job": BASELINE_JOB,
            "solver_seed": 42,
            "metrics": metrics,
            "control_tick": adapter.tick,
            "adapter_event_count": len(adapter.events),
            "adapter_events_file": "batch50_events.jsonl",
            "pbs_job_id": os.environ["PBS_JOBID"],
            "compute_host": host,
            "claim_boundary": "Batch-50 native synchronous K0 parity/reference only; not the N=1 paired control.",
        }
        write_json(out / "batch50_official_k0.json", result)
        print(json.dumps({"gate_status": result["gate_status"], "successes": result["successes"], "task_count": result["task_count"]}), flush=True)
        return 0 if result["gate_status"] == "PASS" else 2

    task_dir = out / "tasks"
    task_dir.mkdir(exist_ok=True)
    results: list[dict[str, Any]] = []
    for task_index in range(args.task_start, args.task_stop):
        task = tasks[task_index]
        result_path = task_result_path(out, task)
        signature = {"mode": args.mode, "delay_steps": args.delay_steps if args.mode == "fixed_steps" else 0, "task_index": task_index, "task_id": task["row_index"], "solver_seed": paired_task_seed(task_index)}
        if result_path.exists():
            existing = read_json(result_path)
            events_path = task_dir / f"task_{task['row_index']}.events.jsonl"
            if existing.get("status") == "COMPLETED" and events_path.is_file() and all(existing.get(key) == value for key, value in signature.items()):
                print(json.dumps({"event": "skip_completed_task", **signature}), flush=True)
                results.append(existing)
                continue
            raise FileExistsError(f"existing per-task result does not match this arm: {result_path}")
        lock_path = result_path.with_suffix(".running")
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(os.environ["PBS_JOBID"])
        try:
            print(json.dumps({"event": "task_start", **signature}), flush=True)
            result = evaluate_one(
                mode=args.mode, delay_steps=args.delay_steps, task_index=task_index, task=task,
                cfg=cfg, baseline=baseline, swm=swm, dataset=dataset, process=process,
                transforms=transforms, torch=torch, spt=spt, teacher=teacher,
                OmegaConf=OmegaConf, out=out, gate_identity=gate_identity,
            )
            result.update({"pbs_job_id": os.environ["PBS_JOBID"], "compute_host": host})
            write_json(result_path, result)
            results.append(result)
        finally:
            lock_path.unlink(missing_ok=True)

    summary_path = out / f"run_summary_{args.mode}_k{args.delay_steps}_{args.task_start}_{args.task_stop}.json"
    write_json(summary_path, {
        "status": "COMPLETED",
        "mode": args.mode,
        "delay_steps": args.delay_steps if args.mode == "fixed_steps" else 0,
        "task_slice_half_open": [args.task_start, args.task_stop],
        "task_count": len(results),
        "successes": sum(bool(row["success"]) for row in results),
        "episode_successes": [bool(row["success"]) for row in results],
        "task_ids": [row["task_id"] for row in results],
        "task_seed_rule": "42 + frozen_task_index; same rule across all N=1 arms",
        "batch50_reference": "25534994.pbs101 parity gate is separate; N=1 task RNG consumption differs from batch-50.",
        "k0_gate": gate_identity,
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
    })
    print(json.dumps({"event": "slice_completed", "mode": args.mode, "successes": sum(bool(row["success"]) for row in results), "tasks": len(results)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

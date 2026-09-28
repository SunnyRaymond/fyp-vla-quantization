"""Delay and strict-async hooks for pinned stable-worldmodel PushT."""

from __future__ import annotations

import copy
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

K_STEPS = (0, 1, 2, 4, 8, 16)
ACTION_BLOCK = 25  # receding_horizon=5 * action_block=5
PAIRED_TASK_SEED_BASE = 42


def _queue_length(policy: Any) -> int:
    """Return the single task's native queue length without modifying it."""
    buffers = getattr(policy, "_action_buffer", None)
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise RuntimeError("delay/async arms require the pinned per-task action-buffer list (num_envs=1)")
    return len(buffers[0])


def _queue_lengths(policy: Any, count: int) -> list[int]:
    """Return native per-environment buffer lengths for terminal accounting."""
    buffers = getattr(policy, "_action_buffer", None)
    if not isinstance(buffers, list) or len(buffers) != count:
        raise RuntimeError(f"native action buffer does not match {count} environment tasks")
    return [len(queue) for queue in buffers]


def _copy_state(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().clone()
    return copy.deepcopy(value)


def _bool_vector(value: Any, count: int) -> list[bool]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    result = [bool(item) for item in value]
    if len(result) != count:
        raise RuntimeError(f"step status has {len(result)} rows for {count} tasks")
    return result


def pusht_control_tick(env_pool: Any) -> dict[str, float | int | str]:
    """Read and validate one pinned PushT control tick from the live env."""
    envs = getattr(env_pool, "envs", None)
    if not envs:
        raise RuntimeError("cannot read the pinned PushT control tick from EnvPool")
    env = envs[0].unwrapped
    dt = float(env.dt)
    control_hz = float(env.control_hz)
    render_fps = float(env.metadata["render_fps"])
    physics_steps = int(1 / (dt * control_hz))
    seconds = physics_steps * dt
    if not (dt > 0 and control_hz == render_fps == 10.0 and physics_steps == 10 and abs(seconds - 0.1) < 1e-12):
        raise RuntimeError(
            "pinned PushT tick changed: expected dt=0.01, control_hz=render_fps=10, 10 physics steps"
        )
    return {
        "source": "stable_worldmodel.envs.pusht.env:PushT.step",
        "dt_seconds": dt,
        "control_hz": control_hz,
        "physics_steps_per_control_tick": physics_steps,
        "control_tick_seconds": seconds,
    }


def paired_task_seed(task_index: int, base: int = PAIRED_TASK_SEED_BASE) -> int:
    if task_index < 0:
        raise ValueError("task_index must be non-negative")
    return int(base + task_index)


def validate_k0_exactness(
    actual_tasks: Sequence[dict[str, Any]],
    actual_successes: Sequence[bool],
    reference: dict[str, Any],
) -> None:
    """Require the ordered batch-50 K0 vector to match baseline 25534994."""
    fields = ("row_index", "episode_idx", "start_step")
    expected_tasks = [
        dict(zip(fields, map(int, values)))
        for values in zip(
            reference["ordered_row_indices"],
            reference["ordered_episode_ids"],
            reference["ordered_start_steps"],
        )
    ]
    observed_tasks = [{field: int(row[field]) for field in fields} for row in actual_tasks]
    expected = [bool(value) for value in reference["ordered_episode_successes"]]
    observed = [bool(value) for value in actual_successes]
    if observed_tasks != expected_tasks:
        raise RuntimeError("batch-50 K0 task order/identity differs from 25534994.pbs101")
    if len(expected) != 50 or sum(expected) != 49 or observed != expected:
        raise RuntimeError("batch-50 K0 outcome vector is not itemwise equal to frozen 49/50 reference")


def read_reference(task_path: str | Path, outcomes_path: str | Path) -> tuple[list[dict[str, int]], dict[str, Any]]:
    import json

    task_doc = json.loads(Path(task_path).read_text(encoding="utf-8-sig"))
    reference = json.loads(Path(outcomes_path).read_text(encoding="utf-8-sig"))
    tasks = [{key: int(row[key]) for key in ("row_index", "episode_idx", "start_step")} for row in task_doc["tasks"]]
    if task_doc.get("pbs_job_id") != "25534994.pbs101" or len(tasks) != 50:
        raise RuntimeError("task manifest must be the frozen 50-task 25534994.pbs101 list")
    if reference.get("source_job") != "25534994.pbs101" or reference.get("task_count") != 50:
        raise RuntimeError("outcome vector must be the frozen 25534994.pbs101 reference")
    if len(reference.get("ordered_episode_successes", [])) != 50 or sum(map(bool, reference["ordered_episode_successes"])) != 49:
        raise RuntimeError("reference must contain exactly 49 successes among 50 ordered outcomes")
    identity = [(row["row_index"], row["episode_idx"], row["start_step"]) for row in tasks]
    captured = list(zip(reference["ordered_row_indices"], reference["ordered_episode_ids"], reference["ordered_start_steps"]))
    if identity != [tuple(map(int, row)) for row in captured]:
        raise RuntimeError("task manifest and ordered success vector are not aligned")
    return tasks, reference


class DelayAsyncAdapter:
    """Keep World and the native action buffer on the caller thread.

    Fixed-step mode runs native planning synchronously, then holds the previous
    action for K actual EnvPool steps before releasing the first planned action.
    Strict true-async prepares and snapshots solver input on the caller thread;
    while a solve is in flight the worker exclusively owns the pinned solver.
    The caller thread owns the live policy buffer and installs the returned
    plan. The default tick reader is pinned PushT; other environments can
    inject verified ``tick_metadata`` and an environment-space neutral action.
    True-async is single-task only.
    """

    def __init__(
        self,
        policy: Any,
        env_pool: Any,
        *,
        task_ids: Sequence[Any],
        mode: str,
        delay_steps: int = 0,
        tick_seconds: float | None = None,
        tick_metadata: dict[str, Any] | None = None,
        neutral_env_action: Any = None,
        action_block: int = ACTION_BLOCK,
        synchronize: Callable[[], None] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if mode not in {"fixed_steps", "true_async"}:
            raise ValueError("mode must be 'fixed_steps' or 'true_async'")
        if mode == "fixed_steps" and delay_steps not in K_STEPS:
            raise ValueError(f"delay_steps must be one of {K_STEPS}")
        if action_block != ACTION_BLOCK:
            raise ValueError(f"pinned action block is {ACTION_BLOCK}, got {action_block}")
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be non-empty and unique")
        if mode == "true_async" and len(task_ids) != 1:
            raise RuntimeError("strict true_async runs one frozen task per World (num_envs=1)")
        if mode == "fixed_steps" and delay_steps and len(task_ids) != 1:
            raise RuntimeError("fixed K>0 paired arms run one frozen task per World (num_envs=1)")
        if not hasattr(policy, "_action_buffer") or not hasattr(policy, "_next_init"):
            raise RuntimeError("pinned WorldModelPolicy action buffer and warm-start state are required")
        if mode == "true_async" and getattr(policy.solver, "callbacks", []):
            raise RuntimeError("true_async requires the frozen CEMSolver with no mutable callbacks")

        self.policy = policy
        self.env_pool = env_pool
        self.task_ids = list(task_ids)
        self.mode = mode
        self.delay_steps = int(delay_steps)
        self.action_block = action_block
        self.tick = dict(tick_metadata) if tick_metadata is not None else pusht_control_tick(env_pool)
        try:
            measured_tick = float(self.tick["control_tick_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("tick_metadata must include control_tick_seconds") from exc
        if not math.isfinite(measured_tick) or measured_tick <= 0:
            raise ValueError("tick_metadata.control_tick_seconds must be finite and positive")
        self.tick["control_tick_seconds"] = measured_tick
        self.tick.setdefault("source", "injected_tick_metadata" if tick_metadata is not None else "pinned_PushT")
        if tick_seconds is not None and (not math.isfinite(float(tick_seconds)) or abs(float(tick_seconds) - measured_tick) > 1e-12):
            raise RuntimeError("tick_seconds differs from the supplied/derived control tick metadata")
        self.tick_seconds = measured_tick if mode == "true_async" else None
        self.synchronize = synchronize or (lambda: None)
        self._clock_ns = clock_ns
        self._sleep = sleeper
        self._provided_neutral_env_action = copy.deepcopy(neutral_env_action)

        self.events: list[dict[str, Any]] = []
        self.status = ["active"] * len(self.task_ids)
        self.control_step = 0
        self._event_id = 0
        self._plan_id = 0
        self._active_plan_id: int | None = None
        self._last_action: Any = None
        self._delayed_action: Any = None
        self._delayed_plan_id: int | None = None
        self._delayed_timing: dict[str, Any] | None = None
        self._delay_remaining = 0
        self._future: Future[tuple[Any, dict[str, Any]]] | None = None
        self._future_plan_id: int | None = None
        self._future_started_step: int | None = None
        self._terminal_drop_plans: set[tuple[int | None, str]] = set()
        self._terminal_recorded_tasks: set[int] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lewm-pusht-infer") if mode == "true_async" else None
        self._lock = threading.Lock()
        self._next_tick_ns: int | None = None
        self._last_scheduled_tick_ns: int | None = None
        self._period_ns = int(self.tick_seconds * 1e9) if self.tick_seconds is not None else 0
        self._last_step_started_ns: int | None = None
        self._last_action_event_id: int | None = None
        self._finished = False
        self._closed = False

        self._native_get_action = policy.get_action
        self._native_step = env_pool.step
        self._original_policy_method = policy.get_action
        self._original_step_method = env_pool.step
        self.policy.get_action = self._get_action
        self.env_pool.step = self._step
        self._initial_action = self._neutral_action() if mode == "true_async" or (mode == "fixed_steps" and delay_steps > 0) else None

    def _neutral_action(self) -> Any:
        import numpy as np

        envs = getattr(self.env_pool, "envs", None)
        if not envs:
            raise RuntimeError("true_async startup requires env_pool.envs[0].action_space")
        space = envs[0].unwrapped.action_space
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise RuntimeError("true_async startup requires finite action bounds")
        if self._provided_neutral_env_action is None:
            action = (low + high) / 2
        else:
            action = np.asarray(self._provided_neutral_env_action, dtype=space.dtype)
            if action.shape == tuple(space.shape):
                action = action.reshape((1, *space.shape))
            elif action.shape != (1, *space.shape):
                raise ValueError(f"neutral_env_action must have shape {tuple(space.shape)} or (1, *space.shape)")
        if not np.isfinite(action).all() or np.any(action < low) or np.any(action > high):
            raise ValueError("neutral_env_action must be finite and within the pinned environment action bounds")
        return np.broadcast_to(action, (1, *space.shape)).astype(space.dtype, copy=True)

    def _record(self, name: str, **fields: Any) -> int:
        with self._lock:
            event_id = self._event_id
            self._event_id += 1
            self.events.append({"event_id": event_id, "event": name, **fields})
        return event_id

    def _emit(self, action: Any, source: str, observation_id: int, plan_id: int | None) -> Any:
        if source not in {"startup_neutral", "fixed_delay_hold", "async_hold"}:
            self._last_action = _copy_state(action)
            if plan_id is not None:
                self._active_plan_id = plan_id
        event_id = self._record(
            "action_output",
            observation_event_id=observation_id,
            control_step=self.control_step,
            action_source=source,
            plan_id=plan_id,
            returned_ns=self._clock_ns(),
            native_buffer_after=_queue_length(self.policy) if self.mode == "fixed_steps" and self.delay_steps > 0 else None,
        )
        self._last_action_event_id = event_id
        ended = [task for task, state in zip(self.task_ids, self.status) if state != "active"]
        if ended:
            self._record("terminal_action_drop", plan_id=plan_id, control_step=self.control_step, task_ids=ended)
        return action

    def _hold(self, source: str, observation_id: int, plan_id: int | None) -> Any:
        action = self._initial_action if self._last_action is None else self._last_action
        if action is None:
            raise RuntimeError("no previous or neutral action is available while waiting")
        returned = _copy_state(action)
        self._emit(returned, source, observation_id, plan_id)
        return returned

    def _run_sync_plan(self, info: Any, plan_id: int, observed_step: int, observed_ns: int, observation_id: int) -> tuple[Any, dict[str, Any]]:
        started_ns = self._clock_ns()
        self.synchronize()
        self._record("inference_start", plan_id=plan_id, kind="startup" if plan_id == 0 else "steady_state", observation_event_id=observation_id, observed_step=observed_step, observed_ns=observed_ns, started_ns=started_ns)
        action = self._native_get_action(info)
        self.synchronize()
        buffered = _queue_length(self.policy)
        if buffered != self.action_block - 1:
            raise RuntimeError(f"native plan left {buffered} buffered actions, expected {self.action_block - 1}")
        finished_ns = self._clock_ns()
        timing = {"plan_id": plan_id, "kind": "startup" if plan_id == 0 else "steady_state", "observation_event_id": observation_id, "observed_step": observed_step, "observed_ns": observed_ns, "started_ns": started_ns, "finished_ns": finished_ns, "inference_seconds": (finished_ns - started_ns) / 1e9, "available_step": self.control_step, "buffered_actions": buffered}
        self._record("inference_ready", **timing)
        return action, timing

    def _run_worker_plan(self, solver: Any, sliced: Any, plan_id: int, observed_step: int, observed_ns: int, observation_id: int) -> tuple[Any, dict[str, Any]]:
        started_ns = self._clock_ns()
        self.synchronize()
        self._record("inference_start", plan_id=plan_id, kind="startup" if plan_id == 0 else "steady_state", observation_event_id=observation_id, observed_step=observed_step, observed_ns=observed_ns, started_ns=started_ns)
        # This is the only operation performed on the worker: the caller has
        # already prepared, selected, and snapshotted the single-task inputs.
        # No caller-thread path touches solver/RNG state until it completes.
        outputs = solver.solve(sliced[0], init_action=sliced[1])
        self.synchronize()
        finished_ns = self._clock_ns()
        with self._lock:
            available_step = self.control_step
        timing = {"plan_id": plan_id, "kind": "startup" if plan_id == 0 else "steady_state", "observation_event_id": observation_id, "observed_step": observed_step, "observed_ns": observed_ns, "started_ns": started_ns, "finished_ns": finished_ns, "inference_seconds": (finished_ns - started_ns) / 1e9, "available_step": available_step}
        self._record("inference_ready", **timing)
        return outputs, timing

    def _install_worker_plan(self, outputs: Any, info: Any) -> Any:
        import torch

        if _queue_length(self.policy) != 0:
            raise RuntimeError("live action buffer must be empty before a worker plan is installed")
        actions = outputs.get("actions")
        if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[0] != 1:
            raise RuntimeError("pinned CEMSolver returned an unexpected one-task action plan")
        keep_horizon = int(self.policy.cfg.receding_horizon)
        if keep_horizon * int(self.policy.cfg.action_block) != self.action_block:
            raise RuntimeError("WorldModelPolicy receding/action block no longer matches 25 steps")
        plan, rest = actions[:, :keep_horizon], actions[:, keep_horizon:]
        if self.policy.cfg.warm_start and rest.shape[1] > 0:
            if self.policy._next_init is None:
                self.policy._next_init = torch.zeros(1, rest.shape[1], rest.shape[2], dtype=rest.dtype)
            self.policy._next_init[0] = rest[0]
        elif not self.policy.cfg.warm_start:
            self.policy._next_init = None
        flat_plan = plan.reshape(1, self.action_block, -1)
        self.policy._action_buffer[0].extend(flat_plan[0])
        action = self._native_get_action(info)
        buffered = _queue_length(self.policy)
        if buffered != self.action_block - 1:
            raise RuntimeError(f"installed native plan left {buffered} buffered actions, expected {self.action_block - 1}")
        return action

    def _slice_solver_inputs(self, info: Any) -> tuple[Any, Any]:
        """Mirror pinned policy preprocessing and replan slicing for N=1."""
        import numpy as np
        import torch

        if _queue_length(self.policy) != 0:
            raise RuntimeError("async solve may start only after the native action buffer drains")
        prepared = self.policy._prepare_info(info)
        needs_flush = prepared.pop("_needs_flush", None)
        if needs_flush is not None and bool(np.asarray(needs_flush, dtype=bool).reshape(-1)[0]):
            self.policy._action_buffer[0].clear()
            if self.policy._next_init is not None:
                self.policy._next_init[0] = 0
        terminated = prepared.get("terminated")
        if terminated is not None and bool(np.asarray(terminated, dtype=bool).reshape(-1)[0]):
            raise RuntimeError("async planning observation is already terminal")

        index = torch.as_tensor([0], dtype=torch.long)
        sliced: dict[str, Any] = {}
        for key, value in prepared.items():
            if torch.is_tensor(value):
                sliced[key] = value[index]
            elif isinstance(value, np.ndarray):
                sliced[key] = value[[0]]
            elif isinstance(value, list):
                sliced[key] = [value[0]]
            else:
                sliced[key] = value
        init_action = self.policy._next_init[index] if self.policy._next_init is not None else None
        # Clone tensors and arrays now: EnvPool mutates its stacked info arrays
        # in place on the next step while the worker may still be solving.
        return copy.deepcopy(sliced), _copy_state(init_action)

    def _get_action(self, info: Any, *args: Any, **kwargs: Any) -> Any:
        observed_ns = self._clock_ns()
        observed_step = self.control_step
        observation_id = self._record("observation", control_step=observed_step, observed_ns=observed_ns, task_ids=self.task_ids)

        if self.mode == "fixed_steps" and self.delay_steps == 0:
            # Transparent wrapper for both official batch-50 K0 and the N=1
            # per-task synchronous control. No buffer/RNG/policy state is read.
            action = self._native_get_action(info, *args, **kwargs)
            return self._emit(action, "native_k0", observation_id, None)

        if self.mode == "fixed_steps":
            if self._delayed_action is not None:
                if self._delay_remaining:
                    self._delay_remaining -= 1
                    return self._hold("fixed_delay_hold", observation_id, self._delayed_plan_id)
                action, plan_id = self._delayed_action, self._delayed_plan_id
                self._delayed_action = None
                timing = self._delayed_timing or {}
                self._delayed_timing = None
                now = self._clock_ns()
                observed_ns_for_plan = int(timing.get("observed_ns", now))
                self._record("action_ready_wait", plan_id=plan_id, observation_event_id=timing.get("observation_event_id"), observed_ns=observed_ns_for_plan, available_ns=timing.get("finished_ns"), applied_ns=now, applied_control_step=self.control_step, simulation_age_steps=max(0, self.control_step - int(timing.get("observed_step", self.control_step))), fixed_delay_steps=self.delay_steps, fixed_delay_simulated_seconds=self.delay_steps * float(self.tick["control_tick_seconds"]), wall_observation_age_seconds=(now - observed_ns_for_plan) / 1e9)
                return self._emit(action, "fixed_delay_release", observation_id, plan_id)
            if _queue_length(self.policy):
                action = self._native_get_action(info, *args, **kwargs)
                return self._emit(action, "native_buffer", observation_id, self._active_plan_id)
            plan_id = self._plan_id
            action, timing = self._run_sync_plan(info, plan_id, observed_step, observed_ns, observation_id)
            self._plan_id += 1
            if self._last_action is None:
                self._delayed_action = action
                self._delayed_plan_id = plan_id
                self._delayed_timing = timing
                self._delay_remaining = self.delay_steps - 1
                return self._hold("fixed_delay_hold", observation_id, plan_id)
            self._delayed_action = action
            self._delayed_plan_id = plan_id
            self._delayed_timing = timing
            self._delay_remaining = self.delay_steps - 1
            return self._hold("fixed_delay_hold", observation_id, plan_id)

        if self._future is not None:
            if not self._future.done():
                return self._hold("async_hold", observation_id, self._future_plan_id)
            outputs, timing = self._future.result()
            action = self._install_worker_plan(outputs, info)
            plan_id = int(timing["plan_id"])
            self._future = None
            self._future_plan_id = None
            self._future_started_step = None
            self._record("action_ready_wait", plan_id=plan_id, observation_event_id=timing["observation_event_id"], observed_ns=timing["observed_ns"], available_ns=timing["finished_ns"], applied_ns=self._clock_ns(), applied_control_step=self.control_step, wait_steps=max(0, self.control_step - int(timing["available_step"])), simulation_age_steps=max(0, self.control_step - int(timing["observed_step"])), wall_observation_age_seconds=(self._clock_ns() - int(timing["observed_ns"])) / 1e9)
            return self._emit(action, "async_ready", observation_id, plan_id)

        if _queue_length(self.policy):
            action = self._native_get_action(info, *args, **kwargs)
            return self._emit(action, "native_buffer", observation_id, self._active_plan_id)

        plan_id = self._plan_id
        self._plan_id += 1
        self._future_plan_id = plan_id
        self._future_started_step = observed_step
        sliced = self._slice_solver_inputs(info)
        self._future = self._executor.submit(self._run_worker_plan, self.policy.solver, sliced, plan_id, observed_step, observed_ns, observation_id)
        return self._hold("startup_neutral" if plan_id == 0 else "async_hold", observation_id, plan_id)

    def _pace_step(self) -> int:
        if self.mode != "true_async":
            return self._clock_ns()
        now = self._clock_ns()
        target = now if self._next_tick_ns is None else self._next_tick_ns
        if now < target:
            self._sleep((target - now) / 1e9)
            now = self._clock_ns()
        lateness = max(0, now - target)
        if lateness:
            self._record("wall_clock_tick_overrun", control_step=self.control_step, scheduled_tick_ns=target, actual_start_ns=now, lateness_seconds=lateness / 1e9, missed_deadlines=int(lateness // self._period_ns))
        self._last_scheduled_tick_ns = target
        self._next_tick_ns = target + self._period_ns
        return now

    def _record_terminal_drop(self, plan_id: int | None, kind: str, reason: str, **fields: Any) -> None:
        key = (plan_id, kind)
        if key in self._terminal_drop_plans:
            return
        self._terminal_drop_plans.add(key)
        task_ids = fields.pop("task_ids", self.task_ids)
        self._record("terminal_drop", plan_id=plan_id, control_step=self.control_step, reason=reason, task_ids=task_ids, **fields)

    def _step(self, actions: Any, *args: Any, **kwargs: Any) -> Any:
        started_ns = self._pace_step()
        result = self._native_step(actions, *args, **kwargs)
        finished_ns = self._clock_ns()
        if not isinstance(result, tuple) or len(result) < 5:
            raise RuntimeError("pinned EnvPool.step must return the Gymnasium five-tuple")
        terminated = _bool_vector(result[2], len(self.task_ids))
        truncated = _bool_vector(result[3], len(self.task_ids))
        for i, (term, trunc) in enumerate(zip(terminated, truncated)):
            if term:
                self.status[i] = "terminated"
            elif trunc:
                self.status[i] = "truncated"
        previous_started = self._last_step_started_ns
        wall_period = None if previous_started is None else (started_ns - previous_started) / 1e9
        step_wall_seconds = (finished_ns - started_ns) / 1e9
        step_record = {
            "control_step": self.control_step,
            "action_event_id": self._last_action_event_id,
            "step_started_ns": started_ns,
            "step_finished_ns": finished_ns,
            "step_wall_seconds_including_render": step_wall_seconds,
            "step_execution_overrun_seconds": max(0.0, step_wall_seconds - float(self.tick["control_tick_seconds"])) if self.tick else None,
            "scheduled_tick_ns": self._last_scheduled_tick_ns,
            "task_ids": self.task_ids,
            "statuses": list(self.status),
            "terminated": terminated,
            "truncated": truncated,
            "simulated_seconds": self.tick["control_tick_seconds"] if self.tick else None,
            "actual_wall_period_seconds": wall_period,
            "real_time_factor": (self.tick["control_tick_seconds"] / wall_period) if wall_period and wall_period > 0 else None,
            "pacing_mode": "wall_clock_target" if self.mode == "true_async" else "native_unpaced_simulation",
        }
        self._record("environment_step", **step_record)
        self._last_step_started_ns = started_ns
        with self._lock:
            self.control_step += 1
        for i, state in enumerate(self.status):
            if state != "active":
                self._record_stopped_task(i)
        return result

    def _record_stopped_task(self, index: int) -> None:
        if index in self._terminal_recorded_tasks:
            return
        self._terminal_recorded_tasks.add(index)
        task_id = self.task_ids[index]
        status = self.status[index]
        self._record("task_status_final", control_step=self.control_step, task_id=task_id, status=status)
        buffered = _queue_lengths(self.policy, len(self.task_ids))[index]
        if self._future is not None and self.mode == "true_async":
            self._record_terminal_drop(self._future_plan_id, "async", "task_stopped_before_async_plan_was_applied", future_done=self._future.done(), task_ids=[task_id], statuses=[status], buffered_actions=buffered)
        elif self._delayed_action is not None and self.mode == "fixed_steps":
            self._record_terminal_drop(self._delayed_plan_id, "fixed", "task_stopped_during_fixed_step_delay", delay_steps_remaining=self._delay_remaining, task_ids=[task_id], statuses=[status], buffered_actions=buffered)
        elif buffered:
            self._record("terminal_drop", plan_id=self._active_plan_id, kind="native_buffer", reason="task_stopped_with_native_actions_buffered", control_step=self.control_step, task_ids=[task_id], statuses=[status], buffered_actions=buffered)

    def finish(self) -> None:
        """Mark budget-stopped tasks and record plans/actions left unapplied."""
        if self._finished:
            return
        self._finished = True
        for i, state in enumerate(self.status):
            if state == "active":
                self.status[i] = "budget_exhausted"
        self._record("evaluation_stop", control_step=self.control_step, statuses=list(self.status), task_ids=self.task_ids)
        for i in range(len(self.status)):
            self._record_stopped_task(i)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.finish()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if self._future is not None and all(state != "active" for state in self.status):
                try:
                    _, timing = self._future.result()
                    self._record_terminal_drop(int(timing["plan_id"]), "async", "async_plan_finished_after_task_stop", future_done=True, statuses=list(self.status))
                except BaseException as exc:
                    self._record("terminal_drop_error", plan_id=self._future_plan_id, error=type(exc).__name__)
        self.policy.get_action = self._original_policy_method
        self.env_pool.step = self._original_step_method

    def __enter__(self) -> "DelayAsyncAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

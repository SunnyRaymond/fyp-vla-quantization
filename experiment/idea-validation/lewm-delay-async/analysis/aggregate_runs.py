#!/usr/bin/env python3
"""Validate complete PushT/Reacher arms and emit analyzer-ready episode rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "lewm-delay-async-artifact-map-v1"
OUTPUT_SCHEMA = "lewm-delay-async-episodes-v1"
GATE_CONDITION = "batch50_official_k0"
BASELINE_CONDITION = "n1_sync_k0"
METRICS = ("latency_ms", "observation_age_ticks", "rtf", "applied_plan_count", "missed_deadline_ticks", "max_tick_lateness_ms")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot parse JSON {path}: {exc}") from exc


def strict_bools(values: Any, count: int, label: str) -> list[bool]:
    if not isinstance(values, list) or len(values) != count or any(type(value) is not bool for value in values):
        raise ValueError(f"{label} must be exactly {count} JSON booleans")
    return values


def task_identity(raw: dict[str, Any], order: int) -> dict[str, int]:
    try:
        result = {
            "manifest_order": int(raw.get("manifest_order", order)),
            "row_index": int(raw["row_index"]),
            "episode_idx": int(raw["episode_idx"]),
            "start_step": int(raw["start_step"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid task identity at order {order}: {raw!r}") from exc
    if "goal_step" in raw:
        result["goal_step"] = int(raw["goal_step"])
    return result


def identity_key(task: dict[str, int]) -> tuple[int, int, int]:
    return task["row_index"], task["episode_idx"], task["start_step"]


def analyzer_task_id(task: dict[str, int]) -> str:
    return f"row{task['row_index']}:episode{task['episode_idx']}:start{task['start_step']}"


def load_manifest(path: Path, expected_n: int) -> tuple[list[dict[str, int]], str | None]:
    data = read_json(path)
    raw_tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(raw_tasks, list) or len(raw_tasks) != expected_n:
        raise ValueError(f"frozen manifest must contain exactly {expected_n} tasks: {path}")
    tasks = [task_identity(item, order) for order, item in enumerate(raw_tasks)]
    keys = [identity_key(task) for task in tasks]
    row_ids = [task["row_index"] for task in tasks]
    if len(set(keys)) != expected_n or len(set(row_ids)) != expected_n:
        raise ValueError(f"frozen manifest does not contain {expected_n} unique row/task IDs: {path}")
    return tasks, data.get("schema") if isinstance(data, dict) else None


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing completion metadata: {path}")
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def validate_completed_job(arm: dict[str, Any], summary_path: Path, kind: str, summary: dict[str, Any]) -> Path:
    job_id = str(arm.get("job_id", "")).strip()
    if not job_id:
        raise ValueError("every arm mapping must include an explicit job_id")
    job_dir = summary_path.parent if kind.startswith("pusht_") else summary_path.parent.parent
    identity = parse_key_value_file(job_dir / "execution_identity.txt")
    if identity.get("job_id") != job_id:
        raise ValueError(f"mapped job_id {job_id!r} does not match {job_dir / 'execution_identity.txt'}")
    status = parse_key_value_file(job_dir / "job_status")
    runner_status = parse_key_value_file(job_dir / "final_exit_status.txt")
    if status.get("EXIT_STATUS") != "0" or runner_status.get("RUNNER_EXIT_STATUS") != "0":
        raise ValueError(f"job {job_id} did not complete successfully")
    if kind.startswith("pusht_"):
        if str(summary.get("pbs_job_id", "")) != job_id:
            raise ValueError(f"PushT summary job ID does not match {job_id}")
    return job_dir


def check_identities(actual: list[dict[str, int]], expected: list[dict[str, int]], label: str) -> None:
    actual_keys = [identity_key(task) for task in actual]
    expected_keys = [identity_key(task) for task in expected]
    if len(actual_keys) != len(expected_keys) or len(set(actual_keys)) != len(expected_keys):
        raise ValueError(f"{label} must contain exactly {len(expected)} unique task IDs")
    if actual_keys != expected_keys:
        raise ValueError(f"{label} task identities/order differ from the frozen manifest")


def load_jsonl_events(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception as exc:
            raise ValueError(f"invalid event JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"event JSONL row must be an object: {path}:{line_number}")
        events.append(event)
    return events


def finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def summarize_events(events: list[dict[str, Any]] | None, condition: str) -> dict[str, float | None]:
    if condition in {GATE_CONDITION, BASELINE_CONDITION}:
        age = 0.0
    else:
        age_values = [
            value
            for event in (events or [])
            if event.get("event") == "action_ready_wait"
            for value in [finite_nonnegative(event.get("simulation_age_steps"))]
            if value is not None
        ]
        age = statistics.median(age_values) if age_values else None

    latency_values = [
        value
        for event in (events or [])
        if event.get("event") == "inference_ready"
        for value in [finite_nonnegative(event.get("inference_seconds"))]
        if value is not None
    ]
    latency_ms = statistics.median(latency_values) * 1000.0 if latency_values else None

    steps = [event for event in (events or []) if event.get("event") == "environment_step"]
    applied_plan_count = None if events is None or condition in {GATE_CONDITION, BASELINE_CONDITION} else sum(
        event.get("event") == "action_ready_wait" for event in events
    )
    missed_deadline_ticks = None
    max_tick_lateness_ms = None
    if condition == "true_async" and steps:
        late = [event for event in events or [] if event.get("event") == "wall_clock_tick_overrun"]
        missed = [event.get("missed_deadlines", 0) for event in late]
        if any(type(value) is not int or value < 0 for value in missed):
            raise ValueError("wall_clock_tick_overrun.missed_deadlines must be a nonnegative integer")
        lateness = [finite_nonnegative(event.get("lateness_seconds")) for event in late]
        if any(value is None for value in lateness):
            raise ValueError("wall_clock_tick_overrun.lateness_seconds must be finite and nonnegative")
        missed_deadline_ticks = sum(missed)
        max_tick_lateness_ms = max(lateness, default=0.0) * 1000.0
    rtf = None
    if steps:
        first_ns = steps[0].get("step_started_ns")
        last_ns = steps[-1].get("step_finished_ns")
        simulated = [finite_nonnegative(event.get("simulated_seconds")) for event in steps]
        if (
            isinstance(first_ns, int) and not isinstance(first_ns, bool)
            and isinstance(last_ns, int) and not isinstance(last_ns, bool)
            and last_ns > first_ns and all(value is not None for value in simulated)
        ):
            elapsed = (last_ns - first_ns) / 1e9
            rtf = sum(value for value in simulated if value is not None) / elapsed
    return {
        "latency_ms": latency_ms,
        "observation_age_ticks": age,
        "rtf": rtf,
        "applied_plan_count": applied_plan_count,
        "missed_deadline_ticks": missed_deadline_ticks,
        "max_tick_lateness_ms": max_tick_lateness_ms,
    }


def check_condition_kind(condition: str, kind: str) -> None:
    if not isinstance(kind, str):
        raise ValueError("every arm mapping must include a supported artifact kind")
    if condition == GATE_CONDITION and not kind.endswith("_gate"):
        raise ValueError(f"{GATE_CONDITION} must map to a gate artifact")
    if condition != GATE_CONDITION and kind.endswith("_gate"):
        raise ValueError("a batch50 gate artifact cannot be used as a paired N=1 condition")
    if condition == BASELINE_CONDITION and kind.endswith("_gate"):
        raise ValueError("batch50 gate is never the paired N=1 baseline")


def check_condition_summary(condition: str, kind: str, summary: dict[str, Any]) -> None:
    if condition == GATE_CONDITION:
        return
    if kind == "pusht_n1":
        mode = summary.get("mode")
        if mode == "paired_k0":
            actual_condition = BASELINE_CONDITION
        elif mode == "fixed_steps":
            actual_condition = f"fixed_k{int(summary.get('delay_steps', -1))}"
        elif mode == "true_async":
            actual_condition = "true_async"
        else:
            raise ValueError(f"unsupported PushT N1 mode {mode!r}")
    elif kind == "reacher_n1":
        if summary.get("protocol") == "vanilla_lewm_reacher_n1_sync_k0":
            actual_condition = BASELINE_CONDITION
        elif summary.get("mode") == "fixed_steps":
            actual_condition = f"fixed_k{int(summary.get('delay_steps', -1))}"
        elif summary.get("mode") == "true_async":
            actual_condition = "true_async"
        else:
            raise ValueError("unsupported Reacher N1 protocol/mode")
    else:
        return
    if condition != actual_condition:
        raise ValueError(f"mapped condition {condition!r} does not match run mode {actual_condition!r}")


def load_push_arm(
    arm: dict[str, Any], kind: str, summary_path: Path, expected: list[dict[str, int]], condition: str
) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"PushT summary must be an object: {summary_path}")
    job_dir = validate_completed_job(arm, summary_path, kind, summary)
    check_condition_summary(condition, kind, summary)
    if kind == "pusht_gate":
        if (
            summary.get("status") != "COMPLETED"
            or summary.get("mode") != "batch50_official_k0"
            or summary.get("gate_status") != "PASS"
            or summary.get("success_vector_matches_reference") is not True
            or int(summary.get("task_count", -1)) != len(expected)
        ):
            raise ValueError("PushT batch50 gate is not a completed PASS artifact")
        actual = [task_identity(task, i) for i, task in enumerate(summary.get("tasks", []))]
        check_identities(actual, expected, "PushT batch50 gate")
        successes = strict_bools(summary.get("episode_successes"), len(expected), "PushT gate episode_successes")
        return [{"task": task, "success": success, "events": None, "source": str(summary_path)} for task, success in zip(expected, successes)]

    if kind != "pusht_n1":
        raise ValueError(f"unsupported PushT artifact kind: {kind}")
    if summary.get("status") != "COMPLETED" or int(summary.get("task_count", -1)) != len(expected):
        raise ValueError(f"PushT arm summary is incomplete: {summary_path}")
    if summary.get("task_slice_half_open") != [0, len(expected)]:
        raise ValueError("PushT summary must cover task slice [0,50)")
    actual_rows = summary.get("task_ids")
    expected_rows = [task["row_index"] for task in expected]
    if actual_rows != expected_rows:
        raise ValueError("PushT N1 summary task IDs/order differ from frozen manifest")
    summary_success = strict_bools(summary.get("episode_successes"), len(expected), "PushT N1 episode_successes")
    tasks_dir = summary_path.parent / "tasks"
    results = []
    for order, (task, summary_outcome) in enumerate(zip(expected, summary_success)):
        row_path = tasks_dir / f"task_{task['row_index']}.json"
        row = read_json(row_path)
        if row.get("status") != "COMPLETED" or int(row.get("task_index", -1)) != order:
            raise ValueError(f"PushT task result is incomplete or out of order: {row_path}")
        row_task = task_identity(row.get("task", {}), order)
        check_identities([row_task], [task], str(row_path))
        if row.get("success") is not summary_outcome or int(row.get("task_id", -1)) != task["row_index"]:
            raise ValueError(f"PushT task success/ID differs from run summary: {row_path}")
        event_name = row.get("adapter_events_file")
        events_path = tasks_dir / str(event_name) if isinstance(event_name, str) else Path()
        if isinstance(event_name, str) and events_path.parent.resolve() != tasks_dir.resolve():
            raise ValueError(f"PushT event path escapes tasks directory: {event_name}")
        events = load_jsonl_events(events_path) if isinstance(event_name, str) else None
        if events is not None and int(row.get("adapter_event_count", len(events))) != len(events):
            raise ValueError(f"PushT event count differs from task result: {events_path}")
        results.append({"task": task, "success": summary_outcome, "events": events, "source": str(row_path)})
    return results


def load_reacher_arm(
    arm: dict[str, Any], kind: str, summary_path: Path, expected: list[dict[str, int]], manifest_schema: str | None, condition: str
) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Reacher summary must be an object: {summary_path}")
    validate_completed_job(arm, summary_path, kind, summary)
    check_condition_summary(condition, kind, summary)
    if summary.get("manifest_schema") != manifest_schema:
        raise ValueError("Reacher result manifest schema differs from frozen manifest")

    if kind == "reacher_gate":
        if summary.get("protocol") != "vanilla_lewm_reacher_sync_k0" or int(summary.get("task_count", -1)) != len(expected):
            raise ValueError("Reacher batch50 gate is incomplete or has the wrong protocol")
        actual = [task_identity(task, i) for i, task in enumerate(summary.get("tasks", []))]
        check_identities(actual, expected, "Reacher batch50 gate")
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Reacher batch50 gate has no metrics object")
        successes = strict_bools(metrics.get("episode_successes"), len(expected), "Reacher gate episode_successes")
        return [{"task": task, "success": success, "events": None, "source": str(summary_path)} for task, success in zip(expected, successes)]

    if kind != "reacher_n1":
        raise ValueError(f"unsupported Reacher artifact kind: {kind}")
    if int(summary.get("task_count", -1)) != len(expected):
        raise ValueError(f"Reacher N1 summary is incomplete: {summary_path}")
    actual = [task_identity(task, i) for i, task in enumerate(summary.get("ordered_task_ids", []))]
    check_identities(actual, expected, "Reacher N1 summary")
    summary_successes = strict_bools(summary.get("ordered_successes"), len(expected), "Reacher ordered_successes")
    files = summary.get("task_result_files")
    if not isinstance(files, list) or len(files) != len(expected):
        raise ValueError("Reacher N1 summary must identify all per-task JSON files")
    if condition == BASELINE_CONDITION and summary.get("protocol") != "vanilla_lewm_reacher_n1_sync_k0":
        raise ValueError("Reacher N1 paired baseline must be synchronous K0")
    results = []
    for order, (task, outcome, relative) in enumerate(zip(expected, summary_successes, files)):
        row_path = summary_path.parent / str(relative)
        if row_path.parent.resolve() != summary_path.parent.resolve():
            raise ValueError(f"Reacher task path escapes summary directory: {relative}")
        row = read_json(row_path)
        row_task = task_identity(row.get("task", {}), order)
        check_identities([row_task], [task], str(row_path))
        if row.get("success") is not outcome:
            raise ValueError(f"Reacher task success differs from summary: {row_path}")
        events = row.get("scheduler_events", [])
        if events is None:
            events = []
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError(f"Reacher scheduler_events must be an array of objects: {row_path}")
        results.append({"task": task, "success": outcome, "events": events, "source": str(row_path)})
    return results


def aggregate(mapping: dict[str, Any], expected_n: int = 50) -> dict[str, Any]:
    if mapping.get("schema") != INPUT_SCHEMA or not isinstance(mapping.get("benchmarks"), list):
        raise ValueError(f"mapping schema must be {INPUT_SCHEMA}")
    episodes: list[dict[str, Any]] = []
    sources = []
    missing_counts: dict[str, dict[str, dict[str, int]]] = {}
    benchmarks_seen = set()
    conditions_by_benchmark: dict[str, set[str]] = {}
    for benchmark in mapping["benchmarks"]:
        task_name = benchmark.get("task")
        if task_name not in {"PushT", "Reacher"} or task_name in benchmarks_seen:
            raise ValueError("mapping must have one unique benchmark entry each for PushT and/or Reacher")
        benchmarks_seen.add(task_name)
        expected, manifest_schema = load_manifest(Path(benchmark["manifest"]), expected_n)
        arms = benchmark.get("arms")
        if not isinstance(arms, list) or not arms:
            raise ValueError(f"{task_name} mapping must include completed arm entries")
        condition_names = [arm.get("condition") for arm in arms]
        if any(not isinstance(value, str) or not value for value in condition_names) or len(set(condition_names)) != len(condition_names):
            raise ValueError(f"{task_name} arm conditions must be unique non-empty names")
        if GATE_CONDITION not in condition_names or BASELINE_CONDITION not in condition_names:
            raise ValueError(f"{task_name} requires separate batch50 gate and N1 K0 paired baseline")
        conditions_by_benchmark[task_name] = set(condition_names)
        if GATE_CONDITION == BASELINE_CONDITION:
            raise ValueError("batch50 gate can never be used as the paired baseline")
        missing_counts[task_name] = {}
        for arm in arms:
            condition = arm["condition"]
            kind = arm.get("kind")
            check_condition_kind(condition, kind)
            expected_prefix = "pusht_" if task_name == "PushT" else "reacher_"
            if not isinstance(kind, str) or not kind.startswith(expected_prefix):
                raise ValueError(f"artifact kind {kind!r} does not match task {task_name}")
            summary_path = Path(arm["summary"]).resolve(strict=True)
            if task_name == "PushT":
                records = load_push_arm(arm, kind, summary_path, expected, condition)
            else:
                records = load_reacher_arm(arm, kind, summary_path, expected, manifest_schema, condition)
            metric_rows = []
            for record in records:
                metrics = summarize_events(record["events"], condition)
                missing = [name for name in METRICS if metrics[name] is None]
                metric_rows.append({
                    "task": task_name,
                    "condition": condition,
                    "task_id": analyzer_task_id(record["task"]),
                    "success": record["success"],
                    **metrics,
                    "job_id": str(arm["job_id"]),
                    "task_identity": record["task"],
                    "missing_metrics": missing,
                })
            if len(metric_rows) != expected_n or len({row["task_id"] for row in metric_rows}) != expected_n:
                raise ValueError(f"{task_name}/{condition} does not contain {expected_n} unique task IDs")
            missing_counts[task_name][condition] = {name: sum(row[name] is None for row in metric_rows) for name in METRICS}
            episodes.extend(metric_rows)
            sources.append({
                "task": task_name,
                "condition": condition,
                "kind": kind,
                "job_id": str(arm["job_id"]),
                "summary": str(summary_path),
            })
    if benchmarks_seen != {"PushT", "Reacher"}:
        raise ValueError("mapping must include complete PushT and Reacher artifacts")
    if conditions_by_benchmark["PushT"] != conditions_by_benchmark["Reacher"]:
        raise ValueError("PushT and Reacher mappings must include the same complete condition set")
    return {
        "schema": OUTPUT_SCHEMA,
        "expected_n": expected_n,
        "baseline_condition": BASELINE_CONDITION,
        "gate_condition": GATE_CONDITION,
        "episodes": episodes,
        "missing_counts": missing_counts,
        "sources": sources,
    }


def self_check() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="lewm-aggregate-check-") as temp:
        root = Path(temp)
        mapping = {"schema": INPUT_SCHEMA, "benchmarks": []}

        def job_files(job_dir: Path, job_id: str) -> None:
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "execution_identity.txt").write_text(f"job_id={job_id}\n", encoding="utf-8")
            (job_dir / "job_status").write_text("EXIT_STATUS=0\n", encoding="utf-8")
            (job_dir / "final_exit_status.txt").write_text("RUNNER_EXIT_STATUS=0\n", encoding="utf-8")

        push_tasks = [
            {"row_index": 10 + i, "episode_idx": 20 + i, "start_step": 2 + i}
            for i in range(2)
        ]
        push_manifest = root / "push_manifest.json"
        push_manifest.write_text(json.dumps({"tasks": push_tasks}), encoding="utf-8")
        push_arms = []
        for condition, kind, successes, mode in (
            (GATE_CONDITION, "pusht_gate", [True, False], "gate"),
            (BASELINE_CONDITION, "pusht_n1", [True, False], "paired_k0"),
            ("true_async", "pusht_n1", [True, True], "true_async"),
        ):
            job_id = f"push-{mode}"
            job_dir = root / job_id
            job_files(job_dir, job_id)
            if kind == "pusht_gate":
                data = {
                    "status": "COMPLETED", "mode": "batch50_official_k0", "gate_status": "PASS",
                    "success_vector_matches_reference": True, "task_count": 2, "pbs_job_id": job_id,
                    "tasks": push_tasks, "episode_successes": successes,
                }
                summary_path = job_dir / "batch50_official_k0.json"
            else:
                task_dir = job_dir / "tasks"
                task_dir.mkdir()
                for i, (task, success) in enumerate(zip(push_tasks, successes)):
                    events = [
                        {"event": "inference_ready", "inference_seconds": 0.2},
                        {"event": "inference_ready", "inference_seconds": 0.4},
                        {"event": "action_ready_wait", "simulation_age_steps": 1 + i},
                        {"event": "action_ready_wait", "simulation_age_steps": 2 + i},
                        {"event": "environment_step", "step_started_ns": 1_000_000_000, "step_finished_ns": 1_100_000_000, "simulated_seconds": 0.1},
                        {"event": "environment_step", "step_started_ns": 1_100_000_000, "step_finished_ns": 1_200_000_000, "simulated_seconds": 0.1},
                        {"event": "wall_clock_tick_overrun", "missed_deadlines": 2, "lateness_seconds": 0.04},
                    ]
                    events_name = f"task_{task['row_index']}.events.jsonl"
                    (task_dir / events_name).write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
                    (task_dir / f"task_{task['row_index']}.json").write_text(json.dumps({
                        "status": "COMPLETED", "mode": mode, "task_index": i, "task_id": task["row_index"],
                        "task": task, "success": success, "adapter_events_file": events_name,
                        "adapter_event_count": len(events),
                    }), encoding="utf-8")
                data = {
                    "status": "COMPLETED", "mode": mode, "delay_steps": 1 if mode == "fixed_steps" else 0,
                    "task_slice_half_open": [0, 2], "task_count": 2, "pbs_job_id": job_id,
                    "task_ids": [task["row_index"] for task in push_tasks], "episode_successes": successes,
                }
                summary_path = job_dir / f"run_summary_{mode}_k1_0_2.json"
            summary_path.write_text(json.dumps(data), encoding="utf-8")
            push_arms.append({"condition": condition, "kind": kind, "job_id": job_id, "summary": str(summary_path)})
        mapping["benchmarks"].append({"task": "PushT", "manifest": str(push_manifest), "arms": push_arms})

        reacher_tasks = [
            {"manifest_order": i, "row_index": 30 + i, "episode_idx": 40 + i, "start_step": i, "goal_step": i + 25}
            for i in range(2)
        ]
        reacher_manifest = root / "reacher_manifest.json"
        reacher_manifest.write_text(json.dumps({"schema": "synthetic-v1", "tasks": reacher_tasks}), encoding="utf-8")
        reacher_arms = []
        for condition, kind, successes in (
            (GATE_CONDITION, "reacher_gate", [True, False]),
            (BASELINE_CONDITION, "reacher_n1", [True, False]),
            ("true_async", "reacher_n1", [False, True]),
        ):
            job_id = f"reacher-{condition}"
            job_dir = root / job_id
            results_dir = job_dir / "results"
            results_dir.mkdir(parents=True)
            job_files(job_dir, job_id)
            if kind == "reacher_gate":
                result = {
                    "protocol": "vanilla_lewm_reacher_sync_k0", "manifest_schema": "synthetic-v1",
                    "task_count": 2, "tasks": reacher_tasks, "metrics": {"episode_successes": successes},
                }
                summary_path = results_dir / "result.json"
            else:
                task_files = []
                for i, (task, success) in enumerate(zip(reacher_tasks, successes)):
                    relative = f"task_{i:03d}.json"
                    events = [] if condition == BASELINE_CONDITION else [
                        {"event": "inference_ready", "inference_seconds": 0.4},
                        {"event": "action_ready_wait", "simulation_age_steps": 2},
                        {"event": "environment_step", "step_started_ns": 2_000_000_000, "step_finished_ns": 2_100_000_000, "simulated_seconds": 0.02},
                        {"event": "environment_step", "step_started_ns": 2_100_000_000, "step_finished_ns": 2_200_000_000, "simulated_seconds": 0.02},
                    ]
                    (results_dir / relative).write_text(json.dumps({"task": task, "success": success, "scheduler_events": events}), encoding="utf-8")
                    task_files.append(relative)
                result = {
                    "protocol": "vanilla_lewm_reacher_n1_sync_k0" if condition == BASELINE_CONDITION else "vanilla_lewm_reacher_n1_true_async",
                    "manifest_schema": "synthetic-v1", "task_count": 2,
                    "mode": "sync_k0" if condition == BASELINE_CONDITION else "true_async",
                    "ordered_task_ids": reacher_tasks, "ordered_successes": successes,
                    "task_result_files": task_files,
                }
                summary_path = results_dir / "summary.json"
            summary_path.write_text(json.dumps(result), encoding="utf-8")
            reacher_arms.append({"condition": condition, "kind": kind, "job_id": job_id, "summary": str(summary_path)})
        mapping["benchmarks"].append({"task": "Reacher", "manifest": str(reacher_manifest), "arms": reacher_arms})

        output = aggregate(mapping, expected_n=2)
        assert len(output["episodes"]) == 12
        push_async = [row for row in output["episodes"] if row["task"] == "PushT" and row["condition"] == "true_async"][0]
        assert math.isclose(push_async["latency_ms"], 300.0) and push_async["observation_age_ticks"] == 1.5 and push_async["rtf"] == 1.0
        assert (push_async["applied_plan_count"], push_async["missed_deadline_ticks"], push_async["max_tick_lateness_ms"]) == (2, 2, 40.0)
        reacher_async = [row for row in output["episodes"] if row["task"] == "Reacher" and row["condition"] == "true_async"][0]
        assert reacher_async["latency_ms"] == 400.0 and reacher_async["observation_age_ticks"] == 2 and math.isclose(reacher_async["rtf"], 0.2)
        assert (reacher_async["applied_plan_count"], reacher_async["missed_deadline_ticks"], reacher_async["max_tick_lateness_ms"]) == (1, 0, 0.0)
        reacher_k0 = [row for row in output["episodes"] if row["task"] == "Reacher" and row["condition"] == BASELINE_CONDITION][0]
        assert reacher_k0["latency_ms"] is None and reacher_k0["observation_age_ticks"] == 0 and reacher_k0["rtf"] is None
        assert output["missing_counts"]["Reacher"][BASELINE_CONDITION] == {
            "latency_ms": 2, "observation_age_ticks": 0, "rtf": 2,
            "applied_plan_count": 2, "missed_deadline_ticks": 2, "max_tick_lateness_ms": 2,
        }
        check_condition_summary("fixed_k1", "pusht_n1", {"mode": "fixed_steps", "delay_steps": 1})
        print("aggregate self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="explicit JSON mapping of benchmark/condition/job/artifact paths")
    parser.add_argument("--output", type=Path, help="analyzer-ready JSON containing episodes and missing_counts")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required outside --self-check")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate output: {args.output}")
    mapping = read_json(args.input.resolve(strict=True))
    result = aggregate(mapping, expected_n=50)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temp_path.write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp_path.replace(args.output)
    print(json.dumps({"output": str(args.output), "episode_rows": len(result["episodes"]), "missing_counts": result["missing_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

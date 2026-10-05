#!/usr/bin/env python3
"""Summarize one completed FastWAM profile and time CPU-only LIBERO rendering."""

from __future__ import annotations

import os
import socket
from pathlib import Path
import sys


def allocation_guard() -> tuple[str, str, Path]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = Path(os.environ.get("PBS_NODEFILE", "").strip())
    hostname = socket.gethostname().split(".", 1)[0]
    if not job_id or not nodefile.is_file() or "login" in hostname.lower():
        raise SystemExit("refusing diagnostics outside a PBS compute allocation")
    allocated_hosts = {
        line.strip().split(".", 1)[0]
        for line in nodefile.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if hostname not in allocated_hosts:
        raise SystemExit(f"hostname {hostname} is absent from PBS_NODEFILE")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise SystemExit("CPU diagnostic must not see CUDA devices")
    return job_id, hostname, nodefile


JOB_ID, HOSTNAME, NODEFILE = allocation_guard()

import ctypes
import ctypes.util
import inspect
import json
import math
import re
import statistics
import threading
import time
from collections import defaultdict
from typing import Any

import torch
from libero.libero import benchmark, get_libero_path
from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_env


PROFILE_JOB_ID = os.environ.get("SOURCE_PROFILE_JOB_ID", "25663213.pbs101")
PROFILE_ARTIFACTS = Path(os.environ["SOURCE_PROFILE_ARTIFACTS"])
ARTIFACTS = Path(os.environ["ARTIFACTS"])
RESOLUTION = 256
SEED = 42
WARMUP_STEPS = 30
MEASURED_STEPS = 10
EXPECTED_CAMERAS = ("agentview", "robot0_eye_in_hand")
STAGE_NAMES = {
    "preprocess.obs_to_model_input",
    "VAE.input_encode",
    "text.encode_prompt",
    "video.prepare",
    "video.cache_prefill",
    "action.prepare",
    "action.post",
    "action.scheduler_step",
    "action.denoise_step",
    "model.infer_action",
    "postprocess.denormalize_action",
    "postprocess.invert_gripper",
}
LAYER_SCOPE_RE = re.compile(
    r"^FW/module\.(video_expert|action_expert)\.blocks\.(\d+)\."
    r"(self_attn\.[qkvo]|cross_attn|ffn)$"
)


def checked_ms(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid nonnegative finite time for {label}: {value!r}")
    return result


def stats(values: list[float]) -> dict[str, float | int]:
    cleaned = [checked_ms(value, "statistics") for value in values]
    if not cleaned:
        return {"count": 0, "sum_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(cleaned),
        "sum_ms": statistics.fmean(cleaned) * len(cleaned),
        "mean_ms": statistics.fmean(cleaned),
        "median_ms": statistics.median(cleaned),
        "min_ms": min(cleaned),
        "max_ms": max(cleaned),
    }


def raw_episode_chunks(raw_path: Path, expected_chunks: int, expected_denoise: int) -> dict[str, Any]:
    selected: list[tuple[int, dict[str, Any]]] = []
    raw_rows_read = 0
    with raw_path.open("r", encoding="utf-8") as stream:
        for source_line, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_rows_read += 1
            if row.get("pass") == "episode" and row.get("name") in STAGE_NAMES:
                selected.append((source_line, row))

    grouped: list[list[tuple[int, dict[str, Any]]]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    current_has_infer = False
    for source_line, row in selected:
        name = row["name"]
        if name == "preprocess.obs_to_model_input" and current_has_infer:
            grouped.append(current)
            current, current_has_infer = [], False
        elif name == "model.infer_action" and current_has_infer:
            grouped.append(current)
            current, current_has_infer = [], False
        current.append((source_line, row))
        if name == "model.infer_action":
            current_has_infer = True
    if current:
        grouped.append(current)

    name_occurrences: dict[str, int] = defaultdict(int)
    denoise_step_occurrences: dict[str, int] = defaultdict(int)
    chunks = []
    cold_calls: dict[str, dict[str, Any]] = {}
    for chunk_index, events in enumerate(grouped):
        span_rows = []
        for source_line, row in events:
            name = str(row["name"])
            cpu_ms = checked_ms(row.get("cpu_wall_ms"), f"raw line {source_line} {name}")
            first_for_name = name_occurrences[name] == 0
            name_occurrences[name] += 1
            denoise_index = row.get("denoise_step") if name == "action.denoise_step" else None
            first_for_denoise_index = False
            if denoise_index is not None:
                denoise_key = str(denoise_index)
                first_for_denoise_index = denoise_step_occurrences[denoise_key] == 0
                denoise_step_occurrences[denoise_key] += 1
            detail = {
                "source_line": source_line,
                "name": name,
                "stage": row.get("stage"),
                "denoise_step": denoise_index,
                "context_index": row.get("context_index"),
                "pair_index": row.get("pair_index"),
                "cpu_wall_ms": cpu_ms,
                "inclusive": bool(row.get("inclusive", True)),
                "cold_call_for_name": first_for_name,
            }
            if name == "action.denoise_step":
                detail["cold_call_for_denoise_step_index"] = first_for_denoise_index
            span_rows.append(detail)
            if first_for_name:
                cold_calls[name] = {
                    "chunk_index": chunk_index,
                    "source_line": source_line,
                    "cpu_wall_ms": cpu_ms,
                    "denoise_step": denoise_index,
                }

        infer_rows = [row for row in span_rows if row["name"] == "model.infer_action"]
        denoise_rows = [row for row in span_rows if row["name"] == "action.denoise_step"]
        if len(infer_rows) != 1 or len(denoise_rows) != expected_denoise:
            raise ValueError(
                f"chunk {chunk_index}: model.infer_action={len(infer_rows)}, "
                f"denoise={len(denoise_rows)}, expected {expected_denoise}"
            )
        if len({row["denoise_step"] for row in denoise_rows}) != expected_denoise:
            raise ValueError(f"chunk {chunk_index} has repeated or missing denoise_step indices")
        stage_values: dict[str, list[float]] = defaultdict(list)
        for row in span_rows:
            stage_values[row["name"]].append(row["cpu_wall_ms"])
        chunks.append({
            "chunk_index": chunk_index,
            "cold_chunk": chunk_index == 0,
            "inference_close_source_line": infer_rows[0]["source_line"],
            "span_count": len(span_rows),
            "denoise_step_count": len(denoise_rows),
            "stage_cpu_wall_ms": dict(stage_values),
            "denoise_steps": sorted(denoise_rows, key=lambda row: int(row["denoise_step"])),
            "spans": span_rows,
        })

    if len(chunks) != expected_chunks:
        raise ValueError(f"raw spans contain {len(chunks)} model chunks; summary records {expected_chunks}")
    return {
        "raw_span_file": str(raw_path),
        "raw_span_file_bytes": raw_path.stat().st_size,
        "raw_rows_read": raw_rows_read,
        "episode_stage_rows": len(selected),
        "chunk_boundary": "each model.infer_action closing span; following postprocess rows stay with that chunk until the next preprocess span",
        "cold_call_rule": "the first occurrence of each span name is marked cold; denoise steps also mark their first occurrence per step index",
        "chunk_count": len(chunks),
        "expected_denoise_steps_per_chunk": expected_denoise,
        "cold_calls_by_name": cold_calls,
        "chunks": chunks,
    }


def profiler_cpu_scope_summary(profile: dict[str, Any]) -> dict[str, Any]:
    operators = profile.get("replay", {}).get("profiler_operator_summary", [])
    by_scope: dict[str, dict[str, Any]] = {}
    sdpa: dict[tuple[str, str], dict[str, Any]] = {}
    excluded = {"fw_cuda_annotations_cpu_zero": 0, "empty_scope_cpu_zero": 0, "unscoped_rows": 0}

    for row in operators:
        operator = str(row.get("operator_or_record_function", ""))
        scope = str(row.get("scope_label") or "")
        try:
            cpu_us = float(row.get("cpu_total_us", 0.0) or 0.0)
            cuda_us = float(row.get("cuda_total_us", 0.0) or 0.0)
            count = int(row.get("count", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cpu_us) or not math.isfinite(cuda_us) or cpu_us < 0 or cuda_us < 0:
            raise ValueError(f"invalid profiler timing in operator row {operator!r}")
        if operator.startswith("FW/") and cpu_us == 0:
            excluded["fw_cuda_annotations_cpu_zero"] += 1
            continue
        if not scope and cpu_us == 0:
            excluded["empty_scope_cpu_zero"] += 1
            continue

        if "scaled_dot_product" in operator.lower() and cpu_us > 0:
            key = (operator, scope)
            item = sdpa.setdefault(key, {"operator": operator, "scope_label": scope, "count": 0, "cpu_total_us": 0.0, "cuda_total_us": 0.0})
            item["count"] += count
            item["cpu_total_us"] += cpu_us
            item["cuda_total_us"] += cuda_us

        scope_match = LAYER_SCOPE_RE.fullmatch(scope)
        if not scope_match and operator.startswith("FW/") and cpu_us > 0:
            scope_match = LAYER_SCOPE_RE.fullmatch(operator)
            if scope_match:
                scope = operator
        if cpu_us <= 0 or not scope_match or operator.startswith("FW/"):
            if not scope and cpu_us > 0:
                excluded["unscoped_rows"] += 1
            continue

        expert, layer, component = scope_match.groups()
        bucket = by_scope.setdefault(scope, {
            "expert": expert,
            "layer": int(layer),
            "component": component,
            "operator_rows": {},
        })
        op = bucket["operator_rows"].setdefault(operator, {
            "operator": operator,
            "count": 0,
            "cpu_total_us": 0.0,
            "cuda_total_us": 0.0,
        })
        op["count"] += count
        op["cpu_total_us"] += cpu_us
        op["cuda_total_us"] += cuda_us

    layers = []
    for scope, bucket in sorted(by_scope.items()):
        ops = sorted(bucket["operator_rows"].values(), key=lambda row: row["operator"])
        layers.append({
            "scope_label": scope,
            "expert": bucket["expert"],
            "layer": bucket["layer"],
            "component": bucket["component"],
            "cpu_scoped_operator_count": sum(row["count"] for row in ops),
            "cpu_scoped_cuda_total_us_sum": sum(row["cuda_total_us"] for row in ops),
            "operators": ops,
        })
    sdpa_rows = [dict(row) for _, row in sorted(sdpa.items())]
    return {
        "source": "profile_summary.json replay.profiler_operator_summary",
        "cpu_scope_filter": "CPU operator rows with cpu_total_us > 0 whose scope_label is the matching FW/module.<expert>.blocks.<layer>.<component> record scope; synthetic FW/ rows are excluded from CUDA accumulation",
        "layers": layers,
        "sdpa_backend_operator_rows": sdpa_rows,
        "excluded_rows": excluded,
        "interpretation": "cuda_total_us is the profiler operator-reported device-time sum under each CPU scope; nested operator totals can overlap and are not asserted to be unique kernel wall time. Exact SDPA backend operator names and their device-time sums are retained.",
    }


def profile_reference(profile: dict[str, Any]) -> dict[str, Any]:
    episode = profile.get("episode", {})
    protocol = profile.get("protocol", {})
    stages = episode.get("native_episode_cpu_stages", [])
    return {
        "profile_status": profile.get("status"),
        "profile_allocation": profile.get("allocation", {}),
        "protocol": {
            key: protocol.get(key)
            for key in ("task_suite", "task_id", "seed", "num_inference_steps", "num_steps_wait", "replan_steps", "action_infer_mode", "compile_action_infer")
        },
        "episode_action_context_count": episode.get("action_context_count"),
        "episode_first_chunk_cpu_wall_ms": episode.get("first_chunk_cpu_wall_ms"),
        "episode_steady_chunk_cpu_wall_ms": episode.get("steady_chunk_cpu_wall_ms"),
        "sim_render_hook_count": episode.get("sim_render_hook_count"),
        "native_episode_environment_stages": [
            row for row in stages if str(row.get("name", "")).startswith("environment.")
        ],
    }


def env_sims(root: Any) -> list[Any]:
    found = []
    seen = set()
    current = root
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        sim = getattr(current, "sim", None)
        if sim is not None and callable(getattr(sim, "render", None)):
            found.append(sim)
        current = getattr(current, "env", None)
    return found


def environment_diagnostic() -> dict[str, Any]:
    osmesa_library = ctypes.util.find_library("OSMesa")
    ctypes.CDLL("libOSMesa.so.8")
    runtime: dict[str, Any] = {
        "env": None,
        "phase": "create",
        "active_step": None,
        "render_rows": [],
        "patched_classes": {},
        "main_thread_id": threading.get_ident(),
    }

    def patch_render_class(sim: Any) -> None:
        sim_class = type(sim)
        if sim_class in runtime["patched_classes"]:
            return
        original = getattr(sim_class, "render", None)
        if not callable(original):
            raise TypeError(f"sim class has no patchable render method: {sim_class!r}")
        try:
            parameter_names = list(inspect.signature(original).parameters)
            camera_arg_index = parameter_names.index("camera_name") - 1 if "camera_name" in parameter_names else None
        except (TypeError, ValueError):
            camera_arg_index = None

        def timed_render(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not any(self is candidate for candidate in env_sims(runtime["env"])):
                return original(self, *args, **kwargs)
            camera = kwargs.get("camera_name")
            if camera is None and camera_arg_index is not None and camera_arg_index < len(args):
                camera = args[camera_arg_index]
            camera_name = str(camera) if camera is not None else "unknown"
            started = time.perf_counter_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                row = {
                    "phase": runtime["phase"],
                    "camera_name": camera_name,
                    "cpu_wall_ms": checked_ms(elapsed_ms, "sim.render"),
                    "thread_id": threading.get_ident(),
                }
                runtime["render_rows"].append(row)
                active = runtime["active_step"]
                if active is not None:
                    active["render_rows"].append(row)

        timed_render.__name__ = getattr(original, "__name__", "render")
        setattr(sim_class, "render", timed_render)
        runtime["patched_classes"][sim_class] = original

    task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = task_suite.get_task(0)
    started = time.perf_counter_ns()
    env, description = get_libero_env(task, RESOLUTION, SEED)
    create_ms = (time.perf_counter_ns() - started) / 1e6
    runtime["env"] = env
    initial_sims = env_sims(env)
    if not initial_sims:
        raise RuntimeError("official LIBERO env exposes no sim.render method")
    initial_sim_id = id(initial_sims[0])
    patch_render_class(initial_sims[0])

    phase_times: dict[str, float] = {}
    try:
        runtime["phase"] = "reset"
        started = time.perf_counter_ns()
        env.reset()
        phase_times["reset_ms"] = (time.perf_counter_ns() - started) / 1e6
        reset_sims = env_sims(env)
        if not reset_sims:
            raise RuntimeError("env.reset left no accessible sim")
        patch_render_class(reset_sims[0])
        after_reset_sim_id = id(reset_sims[0])

        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        started = time.perf_counter_ns()
        initial_states = torch.load(init_path, weights_only=False)
        load_init_state_ms = (time.perf_counter_ns() - started) / 1e6
        runtime["phase"] = "set_init_state"
        started = time.perf_counter_ns()
        obs = env.set_init_state(initial_states[0])
        phase_times["set_init_state_ms"] = (time.perf_counter_ns() - started) / 1e6
        init_sims = env_sims(env)
        if not init_sims:
            raise RuntimeError("env.set_init_state left no accessible sim")
        patch_render_class(init_sims[0])
        after_init_sim_id = id(init_sims[0])
        image_shapes = {
            key: list(getattr(obs.get(key), "shape", ()))
            for key in ("agentview_image", "robot0_eye_in_hand_image")
        }
        if any(shape != [RESOLUTION, RESOLUTION, 3] for shape in image_shapes.values()):
            raise RuntimeError(f"unexpected initial observation image shapes: {image_shapes}")

        dummy_action = get_libero_dummy_action()
        warmup_step_ms = []
        first_warmup_render_count = 0
        runtime["phase"] = "warmup_step"
        for index in range(WARMUP_STEPS):
            active = {"render_rows": []}
            runtime["active_step"] = active
            started = time.perf_counter_ns()
            obs, _, _, _ = env.step(dummy_action)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            warmup_step_ms.append(checked_ms(elapsed_ms, f"warmup step {index}"))
            if index == 0:
                first_warmup_render_count = len(active["render_rows"])
            runtime["active_step"] = None
        if first_warmup_render_count == 0:
            raise RuntimeError("class-level sim.render hook did not fire after reset/set_init_state")

        measured = []
        runtime["phase"] = "measured_step"
        for index in range(MEASURED_STEPS):
            active = {"render_rows": []}
            runtime["active_step"] = active
            started = time.perf_counter_ns()
            obs, _, _, _ = env.step(dummy_action)
            step_ms = checked_ms((time.perf_counter_ns() - started) / 1e6, f"measured step {index}")
            runtime["active_step"] = None
            by_camera: dict[str, list[float]] = defaultdict(list)
            for row in active["render_rows"]:
                by_camera[row["camera_name"]].append(row["cpu_wall_ms"])
            camera_ms = {camera: sum(values) for camera, values in by_camera.items()}
            render_ms = sum(camera_ms.values())
            residual_ms = step_ms - render_ms
            if residual_ms < -0.01:
                raise RuntimeError(f"hooked sim.render time exceeds env.step time at measured step {index}")
            measured.append({
                "step_index": index,
                "step_cpu_wall_ms": step_ms,
                "render_call_count": len(active["render_rows"]),
                "render_cpu_wall_ms_by_camera": camera_ms,
                "render_cpu_wall_ms_total": render_ms,
                "step_cpu_residual_excluding_hooked_render_ms": max(0.0, residual_ms),
            })

        counts = defaultdict(int)
        measured_camera_calls: dict[str, list[float]] = defaultdict(list)
        for row in runtime["render_rows"]:
            if row["phase"] == "measured_step":
                counts[row["camera_name"]] += 1
                measured_camera_calls[row["camera_name"]].append(row["cpu_wall_ms"])
        missing = [camera for camera in EXPECTED_CAMERAS if counts[camera] == 0]
        if missing:
            raise RuntimeError(f"measured step hooks did not capture expected cameras: {missing}; counts={dict(counts)}")
        render_threads = sorted({row["thread_id"] for row in runtime["render_rows"] if row["phase"] == "measured_step"})
        if render_threads != [runtime["main_thread_id"]]:
            raise RuntimeError(f"OSMesa renders did not remain on the diagnostic thread: {render_threads}")

        by_phase_camera: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in runtime["render_rows"]:
            by_phase_camera[(row["phase"], row["camera_name"])].append(row["cpu_wall_ms"])
        measured_total = [row["step_cpu_wall_ms"] for row in measured]
        measured_render = [row["render_cpu_wall_ms_total"] for row in measured]
        measured_residual = [row["step_cpu_residual_excluding_hooked_render_ms"] for row in measured]
        return {
            "status": "complete",
            "allocation": {"pbs_job_id": JOB_ID, "hostname": HOSTNAME, "nodefile": str(NODEFILE)},
            "environment": {
                "suite": "libero_goal",
                "task_id": 0,
                "task_description": description,
                "init_state_index": 0,
                "seed": SEED,
                "resolution": [RESOLUTION, RESOLUTION],
                "renderer": os.environ.get("MUJOCO_GL"),
                "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                "osmesa_library": osmesa_library,
                "thread_settings": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS")},
                "main_python_thread_id": runtime["main_thread_id"],
                "measured_render_thread_ids": render_threads,
                "sim_render_class": f"{type(init_sims[0]).__module__}.{type(init_sims[0]).__qualname__}",
                "class_render_hook": "patched sim class method before reset; rechecked after reset and set_init_state",
                "sim_instance_changed_on_reset": after_reset_sim_id != initial_sim_id,
                "sim_instance_changed_by_set_init_state": after_init_sim_id != after_reset_sim_id,
                "render_hook_valid_after_reset_and_init_state": first_warmup_render_count > 0,
                "camera_shapes": image_shapes,
                "environment_create_ms": checked_ms(create_ms, "env create"),
                "init_state_file_load_ms": checked_ms(load_init_state_ms, "init state file load"),
                "reset_ms": checked_ms(phase_times["reset_ms"], "env reset"),
                "set_init_state_ms": checked_ms(phase_times["set_init_state_ms"], "env set_init_state"),
            },
            "warmup": {
                "noop_steps": WARMUP_STEPS,
                "step_cpu_wall_ms": stats(warmup_step_ms),
                "render_call_count_by_phase_camera": {
                    f"{phase}:{camera}": len(values)
                    for (phase, camera), values in sorted(by_phase_camera.items())
                },
                "first_step_render_call_count": first_warmup_render_count,
            },
            "measured": {
                "noop_steps": MEASURED_STEPS,
                "step_cpu_wall_ms": stats(measured_total),
                "hooked_render_cpu_wall_ms_per_call_by_camera": {
                    camera: stats(values) for camera, values in sorted(measured_camera_calls.items())
                },
                "hooked_render_call_count_by_camera": dict(sorted(counts.items())),
                "hooked_render_cpu_wall_ms_total_per_step": stats(measured_render),
                "step_residual_excluding_hooked_render_ms": stats(measured_residual),
                "steps": measured,
                "direct_render_repeats": 0,
                "direct_render_note": "omitted because measured env.step hooks captured both expected cameras",
                "step_component_relation": "each step residual is step wall time minus synchronous hooked sim.render wall time; render and residual are compared per step, not added to other inclusive profile spans",
            },
            "scope_note": "Independent CPU OSMesa diagnostic; its environment.step/render timings do not exactly decompose the prior allocated-GPU episode's 23.3-second environment.step total.",
        }
    finally:
        runtime["active_step"] = None
        runtime["phase"] = "close"
        try:
            if runtime["env"] is not None:
                runtime["env"].close()
        finally:
            for sim_class, original in runtime["patched_classes"].items():
                setattr(sim_class, "render", original)


def main() -> None:
    torch.set_num_threads(8)
    raw_path = PROFILE_ARTIFACTS / "raw_spans.jsonl"
    summary_path = PROFILE_ARTIFACTS / "profile_summary.json"
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"expected completed profile inputs under {PROFILE_ARTIFACTS}")
    profile = json.loads(summary_path.read_text(encoding="utf-8"))
    if profile.get("status") != "complete":
        raise ValueError(f"source profile is not complete: {profile.get('status')!r}")
    protocol = profile.get("protocol", {})
    if protocol.get("task_suite") != "libero_goal" or int(protocol.get("task_id", -1)) != 0:
        raise ValueError(f"unexpected source profile task: {protocol}")
    expected_chunks = int(profile.get("episode", {}).get("action_context_count", 0))
    expected_denoise = int(protocol.get("num_inference_steps", 10))
    episode = raw_episode_chunks(raw_path, expected_chunks, expected_denoise)
    profiler = profiler_cpu_scope_summary(profile)
    print(f"PROFILE_RAW_SPANS_PARSED chunks={episode['chunk_count']} selected_rows={episode['episode_stage_rows']}", flush=True)

    env = environment_diagnostic()
    print(f"CPU_ENV_DIAGNOSTIC steps={MEASURED_STEPS} render_calls={sum(env['measured']['hooked_render_call_count_by_camera'].values())}", flush=True)
    result = {
        "status": "complete",
        "diagnostic_job_id": JOB_ID,
        "source_profile_job_id": PROFILE_JOB_ID,
        "source_profile_artifacts": str(PROFILE_ARTIFACTS),
        "profile_reference": profile_reference(profile),
        "episode_model_chunks": episode,
        "profiler_cpu_scoped_device_times": profiler,
        "independent_cpu_environment": env,
    }
    output_path = ARTIFACTS / "episode_detail.json"
    temp_path = ARTIFACTS / "episode_detail.json.tmp"
    temp_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temp_path.replace(output_path)
    print(f"CPU_DETAIL_COMPLETE path={output_path} bytes={output_path.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()

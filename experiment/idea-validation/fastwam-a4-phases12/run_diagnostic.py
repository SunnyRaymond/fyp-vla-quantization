"""Run one fixed-observation Fast-WAM A4 phase-diagnostic case in PBS."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import socket
import sys
import time


PROTOCOL = "fastwam-a4-phases12-v1"
PILOT_ROOT = Path("/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005")
DEFAULT_DIAG_ROOT = Path("/scratch/users/ntu/yguo017/fastwam-a4-phases12-20261006")
COUNTERS = (
    "integer_gemm_calls",
    "native_int4_gemm_calls",
    "kv_packed_prefills",
    "kv_layer_reads",
)
ALL_TRIPLES = tuple(itertools.product((4, 8), repeat=3))
MIXED_TRIPLES = tuple(bits for bits in ALL_TRIPLES if bits not in ((4, 4, 4), (8, 8, 8)))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def guard() -> None:
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    jobid = os.environ.get("PBS_JOBID")
    if not jobid or not nodefile or "login" in host.lower():
        raise RuntimeError("Approved PBS compute allocation required")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")


def validate_plan(plan: dict, case_id: int) -> dict:
    if plan.get("protocol") != PROTOCOL or not isinstance(plan.get("inputs"), list):
        raise ValueError(f"Expected plan protocol {PROTOCOL}")
    if plan.get("input_count") != 22 or len(plan["inputs"]) != 22:
        raise ValueError("Frozen phase12 plan must contain exactly 22 inputs")
    if [row.get("case_id") for row in plan["inputs"]] != list(range(22)):
        raise ValueError("Frozen phase12 plan must list case IDs 0..21 in order")
    rows = [row for row in plan["inputs"] if type(row.get("case_id")) is int and row["case_id"] == case_id]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one input for case_id={case_id}; found {len(rows)}")
    row = rows[0]
    if row.get("domain") not in ("original", "plus"):
        raise ValueError(f"Invalid domain in case {case_id}: {row.get('domain')!r}")
    for key in ("dimension", "original_task", "task_id", "environment_seed", "observation_file", "metadata_file"):
        if key not in row:
            raise ValueError(f"Case {case_id} is missing {key}")
    seeds = row.get("sampler_seeds")
    if (not isinstance(seeds, list) or len(seeds) != 2
            or any(type(seed) is not int for seed in seeds) or seeds[0] == seeds[1]):
        raise ValueError(f"Case {case_id} needs two distinct integer sampler_seeds")
    if row["domain"] == "plus" and ("source_index" not in row or type(row["source_index"]) is not int):
        raise ValueError(f"Plus case {case_id} needs an integer source_index")
    return row


def rooted_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"Input path must be relative to DIAG_ROOT: {relative}")
    resolved = (root / path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def load_case(diag_root: Path, row: dict) -> tuple[dict, dict]:
    import numpy as np

    with np.load(rooted_path(diag_root, row["observation_file"]), allow_pickle=False) as bundle:
        observation = {key: bundle[key] for key in bundle.files}
    if not observation:
        raise ValueError(f"Empty observation archive for case {row['case_id']}")
    required = {"agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                "robot0_eef_quat", "robot0_gripper_qpos"}
    if not required.issubset(observation):
        raise ValueError(f"Case {row['case_id']} is missing raw observation keys: {sorted(required - set(observation))}")
    for key, value in observation.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"Raw observation {key} must contain finite numeric values, got {value.dtype}")

    metadata = json.loads(rooted_path(diag_root, row["metadata_file"]).read_text(encoding="utf-8"))
    if not isinstance(metadata.get("description"), str) or not metadata["description"].strip():
        raise ValueError(f"Case {row['case_id']} has no task description")
    for key in ("case_id", "domain", "suite", "dimension", "variant", "variant_id", "original_task",
                "task_name", "task_id", "source_index", "state_id", "environment_seed"):
        if metadata.get(key) != row.get(key):
            raise ValueError(f"Case {row['case_id']} {key} differs between plan and metadata")
    if row.get("description") and metadata["description"] != row["description"]:
        raise ValueError(f"Case {row['case_id']} description differs between plan and metadata")
    if metadata.get("case_id") != row["case_id"] or metadata.get("settling_steps") != 30:
        raise ValueError(f"Case {row['case_id']} has mismatched identity or settling_steps")
    if metadata.get("evaluation_episodes") != 0 or not metadata.get("statepath"):
        raise ValueError(f"Case {row['case_id']} must identify its fixed state and have zero episodes")
    if not isinstance(metadata.get("dataset_import_paths"), dict):
        raise ValueError(f"Case {row['case_id']} is missing source import provenance")
    return observation, metadata


def mixed_order(case_id: int, seed_index: int) -> tuple[tuple[int, int, int], ...]:
    offset = (case_id + seed_index) % len(MIXED_TRIPLES)
    return MIXED_TRIPLES[offset:] + MIXED_TRIPLES[:offset]


def query_specs(case_id: int, seed_index: int) -> list[tuple[str, tuple[int, int, int], bool, bool]]:
    specs = [
        ("all_a8", (8, 8, 8), False, False),
        ("all_a4", (4, 4, 4), False, True),
        ("independent_reference", (4, 4, 4), True, False),
    ]
    specs.extend((f"mixed_{bits_label(bits)}", bits, False, False)
                 for bits in mixed_order(case_id, seed_index))
    return specs


def query_id(label: str, sampler_seed: int) -> str:
    return f"{label}_seed{sampler_seed}"


def bits_label(bits: tuple[int, int, int]) -> str:
    return "v{}a{}p{}".format(*bits)


def action_metrics(torch, action, reference, compared_to="bf16") -> dict:
    delta = action.float() - reference.float()
    first10 = delta[:10]
    return {
        f"motor_rmse_first10_vs_{compared_to}": float(first10[:, :6].square().mean().sqrt()),
        f"gripper_rmse_first10_vs_{compared_to}": float(first10[:, 6].square().mean().sqrt()),
        f"motor_rmse_32_vs_{compared_to}": float(delta[:, :6].square().mean().sqrt()),
        f"gripper_rmse_32_vs_{compared_to}": float(delta[:, 6].square().mean().sqrt()),
        f"max_abs_first10_vs_{compared_to}": float(first10.abs().max()),
        f"max_abs_32_vs_{compared_to}": float(delta.abs().max()),
    }


def counters(runtime) -> dict:
    summary = runtime.q.summary()
    return {name: int(summary.get(name, 0) or 0) for name in COUNTERS}


class _TaggedSequence:
    def __init__(self, values, stage: str, quantizer, state: dict):
        self.values, self.stage, self.quantizer, self.state = values, stage, quantizer, state

    def __iter__(self):
        for index, value in enumerate(self.values):
            if self.state["last_tagged_step"] != index:
                self.quantizer.set_stage(self.stage, index)
                self.state["last_tagged_step"] = index
                self.state["observed_steps"].add(index)
            yield value


def install_stage_schedulers(model, quantizer):
    """Tag actual scheduler loop indices before the denoiser executes each step."""
    handles = []
    for name, stage in (("infer_video_scheduler", "video"), ("infer_action_scheduler", "action")):
        scheduler = getattr(model, name, None)
        build = getattr(scheduler, "build_inference_schedule", None)
        step = getattr(scheduler, "step", None)
        if scheduler is None or not callable(build) or not callable(step):
            raise RuntimeError(f"Missing expected scheduler object or methods: model.{name}")
        state = {"build_calls": 0, "step_calls": 0, "last_tagged_step": None, "observed_steps": set()}

        def build_wrapped(*args, _build=build, _stage=stage, _state=state, **kwargs):
            _state.update(build_calls=_state["build_calls"] + 1, step_calls=0,
                          last_tagged_step=None, observed_steps=set())
            result = _build(*args, **kwargs)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise RuntimeError(f"{_stage} scheduler returned an unexpected schedule")
            return tuple(_TaggedSequence(values, _stage, quantizer, _state) for values in result)

        def step_wrapped(*args, _step=step, _stage=stage, _state=state, **kwargs):
            expected = _state["step_calls"]
            if _state["last_tagged_step"] != expected:
                raise RuntimeError(
                    f"{_stage} scheduler step {expected} had no matching pre-denoiser stage tag"
                )
            result = _step(*args, **kwargs)
            _state["step_calls"] += 1
            return result

        setattr(scheduler, "build_inference_schedule", build_wrapped)
        setattr(scheduler, "step", step_wrapped)
        handles.append((scheduler, build, step, state, stage))
    return handles


def reset_stage_calls(handles) -> None:
    for _scheduler, _build, _step, state, _stage in handles:
        state.update(build_calls=0, step_calls=0, last_tagged_step=None, observed_steps=set())


def install_video_conditioning_wrappers(model, quantizer, video_state):
    """Label the real post-video conditioning prep and KV prefill between schedulers."""
    video = getattr(model, "video_expert", None)
    mot = getattr(model, "mot", None)
    prepare = getattr(video, "prepare", None)
    prefill = getattr(mot, "prefill_video_cache_tensor", None)
    if video is None or not callable(prepare) or mot is None or not callable(prefill):
        raise RuntimeError("Pinned IDM video conditioning prepare/prefill path is unavailable")

    def prepare_wrapped(*args, **kwargs):
        if video_state["step_calls"] == 10:
            quantizer.set_stage("video_conditioning_prefill", -1)
        return prepare(*args, **kwargs)

    def prefill_wrapped(*args, **kwargs):
        quantizer.set_stage("video_conditioning_prefill", -1)
        return prefill(*args, **kwargs)

    video.prepare = prepare_wrapped
    mot.prefill_video_cache_tensor = prefill_wrapped
    return ((video, "prepare", prepare), (mot, "prefill_video_cache_tensor", prefill))


def restore_methods(handles) -> None:
    for target, name, original in handles:
        setattr(target, name, original)


def restore_stage_schedulers(handles) -> None:
    for scheduler, build, step, _state, _stage in handles:
        scheduler.build_inference_schedule = build
        scheduler.step = step


def validate_stage_calls(handles, expected_steps: int = 10) -> dict:
    report = {}
    for _scheduler, _build, _step, state, stage in handles:
        steps = sorted(state["observed_steps"])
        report[stage] = {"schedule_calls": state["build_calls"], "step_calls": state["step_calls"],
                         "observed_steps": steps}
        if state["build_calls"] != 1 or state["step_calls"] != expected_steps or steps != list(range(expected_steps)):
            raise RuntimeError(f"Incomplete {stage} stage coverage: {report[stage]}")
    return report


def trace_coverage(trace_result: dict, expected_steps: int = 10) -> dict:
    if trace_result.get("completed") is not True:
        raise RuntimeError("A4 trace did not finalize as complete")
    coverage = trace_result.get("coverage")
    if not isinstance(coverage, list):
        raise RuntimeError("A4 trace did not return coverage records")
    stages = {stage: set() for stage in ("video", "action")}
    modules = {}
    scopes = {}
    for record in coverage:
        module = record.get("module")
        scope = record.get("scope")
        stage = record.get("stage")
        step = record.get("step")
        if module:
            modules[module] = modules.get(module, 0) + int(record.get("calls", 0))
        if scope:
            scopes.setdefault(scope, set()).add(module)
        if stage in stages and type(step) is int and int(record.get("calls", 0)) > 0:
            stages[stage].add(step)
    wanted = set(range(expected_steps))
    missing = {stage: sorted(wanted - seen) for stage, seen in stages.items() if seen != wanted}
    if missing:
        raise RuntimeError(f"A4 trace missed real denoising stages or steps: {missing}")
    return {
        "trace_calls": trace_result.get("calls"),
        "coverage_records": len(coverage),
        "modules": modules,
        "scopes": {scope: sorted(name for name in names if name) for scope, names in scopes.items()},
        "stage_steps": {stage: sorted(steps) for stage, steps in stages.items()},
        "trace_path": trace_result.get("trace_path"),
    }


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def load_pilot_runner():
    artifacts = os.environ.get("ARTIFACTS")
    artifact_runner = Path(artifacts) / "runner.py" if artifacts else None
    if artifact_runner is not None and artifact_runner.is_file():
        pilot_dir = artifact_runner.parent
    else:
        pilot_dir = Path(__file__).resolve().parents[1] / "fastwam-libero-plus-pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    import runner as pilot_runner
    return pilot_runner


def run_case(case_id: int, diag_root: Path, out: Path, plan_path: Path) -> dict:
    if case_id < 0 or case_id > 21:
        raise ValueError("case_id must be in [0, 21]")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    row = validate_plan(plan, case_id)
    row = {**row, "comparison_scope": plan["comparison_scope"], "claim_scope": plan["claim_scope"]}
    observation, metadata = load_case(diag_root, row)
    out.mkdir(parents=True, exist_ok=True)
    query_file = out / "queries.jsonl"
    if query_file.exists() and query_file.stat().st_size:
        raise FileExistsError(f"Refusing to append a second run to {query_file}")

    os.environ["ROOT"] = str(PILOT_ROOT)
    os.environ["DIAG_ROOT"] = str(diag_root)
    pilot = load_pilot_runner()
    runtime = pilot.Runtime(out)
    from stage_quantization import StageQuantization

    handles = []
    conditioning_handles = []
    current = {"case_id": case_id, "stage": "runtime_ready"}
    try:
        from stage_quantization import StageQuantization

        quantizer = StageQuantization(runtime.q)
        runtime.description = metadata["description"]
        datum = runtime.datum(observation)
        handles = install_stage_schedulers(runtime.model, quantizer)
        conditioning_handles = install_video_conditioning_wrappers(
            runtime.model, quantizer, handles[0][3]
        )
        previous_tf32 = runtime.torch.backends.cuda.matmul.allow_tf32
        runtime.torch.backends.cuda.matmul.allow_tf32 = False
    except BaseException as exc:
        try:
            atomic_json(out / "diagnostic_failure.json", {"protocol": PROTOCOL, **current,
                                                           "error": f"{type(exc).__name__}: {exc}"})
            with (out / "diagnostic.log").open("a", encoding="utf-8") as stream:
                stream.write(f"ERROR {type(exc).__name__}: {exc}\n")
        finally:
            try:
                restore_methods(conditioning_handles)
                restore_stage_schedulers(handles)
                if runtime.env is not None:
                    runtime.env.close()
            finally:
                runtime.q.close()
        raise
    actions = {}
    query_ids = {}
    query_count = 0

    def execute(label: str, sampler_seed: int, bits=None, reference=False, trace=False):
        nonlocal query_count
        current.update(query_id=query_id(label, sampler_seed), query_type=label)
        runtime.current_arm = label
        reset_stage_calls(handles)
        if bits is not None:
            quantizer.configure(bits)
        quantizer.set_stage("conditioning", -1)
        before = counters(runtime)
        runtime.torch.cuda.synchronize()
        started = time.perf_counter()
        trace_info = None
        observed_modules = set()
        observed_call_count = 0
        forward_hooks = []
        original_linear = None
        try:
            if bits is None:
                for module_id, module in ((id(module), module) for module in runtime.q.modules.values()):
                    def record_forward(_module, _args, _output, _module_id=module_id):
                        nonlocal observed_call_count
                        observed_call_count += 1
                        observed_modules.add(_module_id)
                    forward_hooks.append(module.register_forward_hook(record_forward))
            else:
                original_linear = runtime.q._linear

                def observe_linear(module, x):
                    nonlocal observed_call_count
                    observed_call_count += 1
                    observed_modules.add(id(module))
                    return original_linear(module, x)

                runtime.q._linear = observe_linear

            if reference:
                with quantizer.reference_mode():
                    action, _ = runtime.infer(datum, sampler_seed, measure=False)
            elif trace:
                trace_dir = out / "traces" / query_id(label, sampler_seed)
                with quantizer.trace_mode(trace_dir, {
                    "protocol": PROTOCOL, "case_id": case_id, "query_id": query_id(label, sampler_seed),
                    "domain": row["domain"], "bits": list(bits), "sampler_seed": sampler_seed,
                }) as trace_context:
                    action, _ = runtime.infer(datum, sampler_seed, measure=False)
                trace_result = trace_context.finish_trace()
                trace_info = trace_coverage(trace_result)
            else:
                action, _ = runtime.infer(datum, sampler_seed, measure=False)
        finally:
            if original_linear is not None:
                runtime.q._linear = original_linear
            for hook in forward_hooks:
                hook.remove()
        runtime.torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        after = counters(runtime)
        delta = {key: after[key] - before[key] for key in COUNTERS}
        if trace and trace_info["trace_calls"] != delta["integer_gemm_calls"]:
            raise RuntimeError(
                "A4 trace calls do not cover every native Linear GEMM: "
                f"trace={trace_info['trace_calls']} integer_gemm_delta={delta['integer_gemm_calls']}"
            )
        if delta["kv_packed_prefills"] or delta["kv_layer_reads"]:
            raise RuntimeError(f"KV path changed in query {query_id(label, sampler_seed)}: {delta}")
        if reference or label == "bf16":
            if any(delta.values()):
                raise RuntimeError(f"Floating reference/BF16 used quantized counters: {delta}")
        else:
            if delta["integer_gemm_calls"] <= 0:
                raise RuntimeError(f"Quantized query did not execute integer GEMM: {delta}")
            if bits == (8, 8, 8) and delta["native_int4_gemm_calls"] != 0:
                raise RuntimeError(f"All-A8 query executed native A4: {delta}")
            if 4 in bits and delta["native_int4_gemm_calls"] <= 0:
                raise RuntimeError(f"A4 path did not execute native INT4 GEMM: {delta}")

        if tuple(action.shape) != (32, 7) or not bool(runtime.torch.isfinite(action).all()):
            raise RuntimeError(f"Invalid normalized action in {query_id(label, sampler_seed)}")
        action_cpu = action.detach().float().cpu()
        commands = runtime.command(action)
        actions[label, sampler_seed] = action_cpu
        query_ids[label, sampler_seed] = query_id(label, sampler_seed)
        reference_action = actions["bf16", sampler_seed]
        observed_names = sorted(quantizer._name_by_id[module_id] for module_id in observed_modules)
        observed_scopes = {}
        for module_id in observed_modules:
            scope = quantizer._scope_by_id[module_id]
            observed_scopes.setdefault(scope, []).append(quantizer._name_by_id[module_id])
        observed_scopes = {scope: sorted(names) for scope, names in observed_scopes.items()}
        a4_reference_metrics = None
        if label == "independent_reference":
            native_a4 = actions["all_a4", sampler_seed]
            a4_reference_metrics = action_metrics(
                runtime.torch, native_a4, action_cpu, compared_to="independent_reference"
            )
        record = {
            "protocol": PROTOCOL,
            "case_id": case_id,
            "query_id": query_id(label, sampler_seed),
            "query_type": label,
            "domain": row["domain"],
            "suite": row.get("suite"),
            "dimension": row["dimension"],
            "variant": row.get("variant"),
            "variant_id": row.get("variant_id"),
            "original_task": row["original_task"],
            "task_name": row.get("task_name"),
            "task_id": row["task_id"],
            "state_id": row.get("state_id"),
            "source_index": row.get("source_index"),
            "comparison_scope": row.get("comparison_scope"),
            "claim_scope": row.get("claim_scope"),
            "environment_seed": row["environment_seed"],
            "sampler_seed": sampler_seed,
            "bits": list(bits) if bits is not None else None,
            "query_count": query_count + 1,
            "context_id": f"case_{case_id:02d}_seed{sampler_seed}",
            "raw_normalized_action_32x7": action_cpu.tolist(),
            "actual_gripper_first10": commands[:10, -1].tolist(),
            "action_metrics_vs_bf16": ({} if label == "bf16" else
                                        action_metrics(runtime.torch, action_cpu, reference_action)),
            "native_all_a4_vs_independent_reference": a4_reference_metrics,
            "runtime_counters": {"before": before, "after": after, "delta": delta},
            "quantization_scope": runtime.q.summary().get("unquantized_scope"),
            "quantized_module_count": runtime.q.summary().get("packed_weight_module_count"),
            "module_names": runtime.q.summary().get("packed_weight_modules"),
            "observed_linear_call_count": observed_call_count,
            "observed_linear_modules": observed_names,
            "observed_linear_scopes": observed_scopes,
            "stage_scheduler_coverage": validate_stage_calls(handles),
            "trace_coverage": trace_info,
            "query_wall_seconds_diagnostic_only": wall_seconds,
            "timing_scope": "diagnostic tracing/reference comparison only; not a formal latency metric",
            "episodes": 0,
            "post_query_env_steps": 0,
        }
        append_jsonl(query_file, record)
        query_count += 1
        current.update(last_completed_query=record["query_id"], query_count=query_count, stage="query_saved")
        atomic_json(out / "progress.json", {"case_id": case_id, "status": "running", "query_count": query_count,
                                              "expected_queries": 20, "last_query_id": record["query_id"]})
        return record

    try:
        # Both BF16 contexts run before the one irreversible shared W4 conversion.
        for sampler_seed in row["sampler_seeds"]:
            runtime.arm("bf16")
            execute("bf16", sampler_seed)

        runtime.arm("w4a8")
        if runtime.q.group_size != 128:
            raise RuntimeError("Phase diagnostic requires the pilot's G128 packed-weight recipe")

        seed_ids = list(row["sampler_seeds"])
        for seed_index, sampler_seed in enumerate(seed_ids):
            for label, bits, reference, trace in query_specs(case_id, seed_index):
                execute(label, sampler_seed, bits=bits, reference=reference, trace=trace)

        factorial = []
        for sampler_seed in seed_ids:
            for bits in ALL_TRIPLES:
                label = ("all_a4" if bits == (4, 4, 4) else "all_a8" if bits == (8, 8, 8)
                         else f"mixed_{bits_label(bits)}")
                factorial.append({"sampler_seed": sampler_seed, "bits": list(bits),
                                  "query_id": query_ids[label, sampler_seed], "reused_query": True})
        if query_count != 20 or runtime.calls != 20:
            raise RuntimeError(f"Expected 20 actual queries, got query_count={query_count}, Runtime.calls={runtime.calls}")
        summary = {
            "protocol": PROTOCOL,
            "case_id": case_id,
            "pbs_jobid": os.environ["PBS_JOBID"],
            "host": socket.gethostname(),
            "domain": row["domain"],
            "suite": row.get("suite"),
            "dimension": row["dimension"],
            "variant": row.get("variant"),
            "variant_id": row.get("variant_id"),
            "original_task": row["original_task"],
            "task_name": row.get("task_name"),
            "task_id": row["task_id"],
            "state_id": row.get("state_id"),
            "source_index": row.get("source_index"),
            "comparison_scope": row.get("comparison_scope"),
            "claim_scope": row.get("claim_scope"),
            "environment_seed": row["environment_seed"],
            "sampler_seeds": seed_ids,
            "input": {"observation_file": row["observation_file"], "metadata_file": row["metadata_file"],
                      "statepath": metadata.get("statepath"), "settling_steps": metadata.get("settling_steps"),
                      "dataset_import_paths": metadata.get("dataset_import_paths"),
                      "description": metadata["description"], "raw_observation_keys": sorted(observation)},
            "query_count": query_count,
            "context_count": len(seed_ids),
            "factorial_cell_count": len(factorial),
            "factorial_cells": factorial,
            "episodes": 0,
            "post_query_env_steps": 0,
            "query_wall_scope": "diagnostic tracing/reference only; not formal latency metrics",
            "quantization_runtime": runtime.q.summary(),
            "status": "complete",
        }
        atomic_json(out / "case_summary.json", summary)
        atomic_json(out / "progress.json", {"case_id": case_id, "status": "complete", "query_count": query_count,
                                              "expected_queries": 20})
        return summary
    except Exception as exc:
        atomic_json(out / "diagnostic_failure.json", {"protocol": PROTOCOL, **current,
                                                       "error": f"{type(exc).__name__}: {exc}"})
        with (out / "diagnostic.log").open("a", encoding="utf-8") as stream:
            stream.write(f"ERROR {type(exc).__name__}: {exc}\n")
        raise
    finally:
        runtime.torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        restore_methods(conditioning_handles)
        restore_stage_schedulers(handles)
        if runtime.env is not None:
            runtime.env.close()
        runtime.q.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=int(os.environ["PBS_ARRAY_INDEX"])
                        if os.environ.get("PBS_ARRAY_INDEX", "").isdigit() else None)
    parser.add_argument("--diag-root", type=Path, default=Path(os.environ.get("DIAG_ROOT", DEFAULT_DIAG_ROOT)))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.case_id is None:
        parser.error("--case-id is required outside a PBS array")
    guard()
    plan_path = args.plan or args.diag_root / "plan.json"
    out = args.out or args.diag_root / "results" / f"case_{args.case_id:02d}"
    run_case(args.case_id, args.diag_root, out, plan_path)
    print(f"DIAGNOSTIC_CASE_COMPLETE case_id={args.case_id} out={out}", flush=True)


if __name__ == "__main__":
    main()

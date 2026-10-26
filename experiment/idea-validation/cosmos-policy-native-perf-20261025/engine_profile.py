"""Four-query, inference-only Cosmos engine profiler for an approved PBS GPU job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

try:
    import resource
except ImportError:  # Windows stdlib tests do not expose Linux process accounting.
    resource = None


FORMAT = "cosmos_engine_profile_v1"
MAX_PROFILE_BYTES = 256 * 1024
DIAGNOSTIC_QUERIES = ((3000, 0), (3009, 1), (3020, 0), (3039, 1))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostic_queries(protocol: dict) -> list[tuple[int, int]]:
    performance = protocol.get("performance", {})
    available = {(int(case), int(seed)) for case in performance.get("case_ids", [])
                 for seed in performance.get("seed_indices", [])}
    missing = [key for key in DIAGNOSTIC_QUERIES if key not in available]
    if missing:
        raise ValueError(f"Frozen protocol lacks diagnostic queries: {missing}")
    return list(DIAGNOSTIC_QUERIES)


def formal_panel_binding(protocol: dict) -> dict:
    performance = protocol.get("performance", {})
    cases, seeds = performance.get("case_ids"), performance.get("seed_indices")
    queries = performance.get("queries")
    blocks, repetitions = performance.get("blocks"), performance.get("repetitions")
    if (not isinstance(cases, list) or len(cases) != 40 or cases != list(range(3000, 3040))
            or seeds != [0, 1] or queries != 80 or blocks != 3 or repetitions != 3):
        raise ValueError("The inherited formal panel must remain the frozen 80-query, 720-sample design")
    return {"case_ids": cases, "seed_indices": seeds, "queries": queries,
            "blocks": blocks, "repetitions": repetitions,
            "samples_per_arm": queries * blocks * repetitions,
            "latency_scope": performance.get("latency_scope")}


def validate_output_paths(old_root: Path, root: Path, out: Path) -> tuple[Path, Path, Path]:
    old_root, root, out = (path.expanduser().resolve() for path in (old_root, root, out))
    if old_root == root or old_root in root.parents or root in old_root.parents:
        raise ValueError("--root must be isolated from the read-only --old-root")
    if out != root and root not in out.parents:
        raise ValueError("--out must be inside the new --root")
    if old_root == out or old_root in out.parents:
        raise ValueError("--out cannot write into the read-only --old-root")
    return old_root, root, out


def bounded_json_bytes(value: dict, limit: int = MAX_PROFILE_BYTES) -> bytes:
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > limit:
        raise ValueError(f"Compact profile exceeds {limit} bytes: {len(payload)}")
    return payload


def write_profile(path: Path, value: dict) -> int:
    payload = bounded_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(payload)


def _time_value(event: Any, names: tuple[str, ...]) -> float:
    for name in names:
        value = getattr(event, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def profiler_event_row(event: Any) -> dict:
    return {"name": str(getattr(event, "key", "")),
            "calls": int(getattr(event, "count", 0)),
            "self_cpu_us": _time_value(event, ("self_cpu_time_total",)),
            "inclusive_cpu_us": _time_value(event, ("cpu_time_total",)),
            "self_cuda_us": _time_value(event, ("self_device_time_total", "self_cuda_time_total")),
            "inclusive_cuda_us": _time_value(event, ("device_time_total", "cuda_time_total"))}


def profiler_top_ops(profiler: Any, limit: int = 24) -> dict:
    rows = [profiler_event_row(event) for event in profiler.key_averages()]
    rows.sort(key=lambda row: (row["self_cuda_us"], row["self_cpu_us"]), reverse=True)
    named = {row["name"]: row for row in rows}
    spans = [named[name] for name in ("engine._Banks._operand", "engine._Banks.decode",
                                      "engine.fp32_U_matmul") if name in named]
    return {"sort": "self CUDA time, then self CPU time",
            "nested_times": "Inclusive and self values are separate; nested categories must not be summed.",
            "instrumented_spans": spans,
            "top_ops": rows[:limit]}


def _rss_bytes() -> dict:
    result = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _unit = line.split()
                result["rss_current_bytes" if key == "VmRSS:" else "rss_peak_bytes"] = int(value) * 1024
    except (OSError, ValueError):
        pass
    if "rss_peak_bytes" not in result and resource is not None:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["rss_peak_bytes"] = value if sys.platform == "darwin" else value * 1024
    return result


def _cuda_memory(torch, device) -> dict:
    return {"allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def _graph_counts(engine) -> dict:
    graphs = getattr(engine, "graphs", None)
    if graphs is None:
        return {"captures": 0, "replays": 0, "denoise_calls": 0}
    calls = graphs.calls
    return {"captures": int(calls.get("graph_captures", 0)),
            "replays": int(calls.get("graph_replays", 0)),
            "denoise_calls": int(calls.get("denoise_calls", 0))}


def _query_context_path(old_root: Path, case_id: int, seed_index: int) -> Path:
    path = old_root / "results" / "teacher" / "contexts" / f"{case_id}_{seed_index}.pt"
    fallback = path.with_name(f"{case_id}.pt")
    if path.is_file():
        return path
    if seed_index == 0 and fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Frozen teacher context is missing (old root stays read-only): {path}")


def _verified_ours_install(runtime, protocol: dict, old_root: Path, out: Path,
                           receipt_binding: dict) -> dict:
    """Install the existing engine and reuse read-only proof files; never start full80 validation here."""
    import engine
    import validation
    import common

    manifest_path = old_root / "results" / "ours" / "deployment" / "manifest.json"
    manifest = read_json(manifest_path)
    device = next(runtime.model.parameters()).device
    torch = common.torch_module()
    props = torch.cuda.get_device_properties(device)
    device_info = {"name": props.name, "capability": [int(props.major), int(props.minor)]}
    key = re.sub(r"[^A-Za-z0-9_-]", "_", props.name) + f"_sm{props.major}{props.minor}"
    expected = validation.identity(protocol, manifest, device_info)
    validation_dir = old_root / "results" / "ours" / "engine_validation" / key
    receipt_path = validation_dir / "summary.json"
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    if not validation.valid_receipt(receipt, expected):
        raise RuntimeError("A matching passed full80 engine receipt is required before the bounded profile")
    receipt_binding.update({"full80_receipt": str(receipt_path),
                            "full80_queries": receipt["checked_queries"],
                            "device": device_info, "device_key": key})

    source_engine_dir = old_root / "results" / "ours" / "engine" / key
    has_decoder_receipt = (source_engine_dir / "decoder_tuning.json").is_file()
    install_out = source_engine_dir if has_decoder_receipt else out / "engine"
    metadata = engine.install(runtime, protocol, old_root / "results" / "ours" / "deployment",
                              old_root, install_out)
    if has_decoder_receipt:
        engine.preflight(runtime, protocol, old_root / "results" / "ours" / "deployment",
                         old_root, source_engine_dir)
        metadata = runtime.install_metadata
    runtime.current_arm = "ours_vqwam"
    runtime.quantization_evidence = metadata
    return metadata


def _install_arm(runtime, protocol: dict, old_root: Path, out: Path, arm: str,
                 receipt_binding: dict) -> dict:
    from runtime import install_arm

    storage = old_root / "results" / "ours" / "inference_bundle" / "storage.json"
    if not storage.is_file():
        raise FileNotFoundError(f"Read-only deployment storage receipt is missing: {storage}")
    if protocol.get("protocol") != read_json(old_root / "protocol.json").get("protocol"):
        raise ValueError("The supplied protocol is not bound to the read-only source root")

    if arm == "ours_vqwam":
        import validation
        original = validation.install_validated

        def reuse_full80(runtime_arg, protocol_arg, root_arg, artifact_dir):
            if Path(root_arg).resolve() != old_root:
                raise RuntimeError("Unexpected ours install root")
            return _verified_ours_install(runtime_arg, protocol_arg, old_root, out,
                                          receipt_binding)

        validation.install_validated = reuse_full80
        try:
            metadata = install_arm(runtime, protocol, old_root, "ours_vqwam", optimized=True)
        finally:
            validation.install_validated = original
    else:
        metadata = install_arm(runtime, protocol, old_root, "bf16", optimized=True)
    return metadata


def _install_query_instrumentation(runtime, engine, torch, state: dict) -> dict:
    model = runtime.model
    original_net_forward = model.net.forward

    def counted_net_forward(*args, **kwargs):
        step = state.get("current_nfe")
        row = state["nfe_rows"][step] if isinstance(step, int) and step < len(state["nfe_rows"]) else None
        before = _graph_counts(engine)
        result = original_net_forward(*args, **kwargs)
        after = _graph_counts(engine)
        if row is not None:
            row["net_forward_calls"] += 1
            replay_delta, capture_delta = after["replays"] - before["replays"], after["captures"] - before["captures"]
            row["graph_replays"] += replay_delta
            row["graph_captures"] += capture_delta
            row["paths"].append("cuda_graph_replay" if replay_delta else
                                "cuda_graph_capture" if capture_delta else "eager_forward")
        return result

    model.net.forward = counted_net_forward
    had_statistics = "statistics_step" in runtime.__dict__
    old_statistics = runtime.__dict__.get("statistics_step")

    def statistics_step(index):
        state["current_nfe"] = int(index)
        state["nfe_rows"].append({"nfe_index": int(index), "net_forward_calls": 0,
                                   "graph_captures": 0, "graph_replays": 0, "paths": []})

    runtime.statistics_step = statistics_step
    return {"original_net_forward": original_net_forward,
            "had_statistics": had_statistics, "old_statistics": old_statistics}


def _restore_query_instrumentation(runtime, instrumentation) -> None:
    runtime.model.net.forward = instrumentation["original_net_forward"]
    if instrumentation["had_statistics"]:
        runtime.statistics_step = instrumentation["old_statistics"]
    else:
        runtime.__dict__.pop("statistics_step", None)


def _run_query(common, runtime, context: dict, torch, device, *, profiled: bool = False) -> dict:
    import endpoint as legacy_endpoint

    engine = getattr(runtime, "ours_vqwam_engine", None)
    state = {"current_nfe": None, "nfe_rows": []}
    instrumentation = _install_query_instrumentation(runtime, engine, torch, state)
    sampler = legacy_endpoint._sampler_module(runtime.model)
    original_sampler_forward = sampler.forward
    interval = {"calls": 0, "wall_seconds": None, "cuda_event_ms": None}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def timed_sampler_forward(*args, **kwargs):
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        interval["calls"] += 1
        start_event.record()
        try:
            result = original_sampler_forward(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize(device)
            interval["cuda_event_ms"] = float(start_event.elapsed_time(end_event))
            return result
        finally:
            interval["wall_seconds"] = time.perf_counter() - wall_start

    sampler.forward = timed_sampler_forward
    try:
        torch.cuda.synchronize(device)
        endpoint_start = time.perf_counter()
        result = common.endpoint(runtime, context)
        torch.cuda.synchronize(device)
        endpoint_wall = time.perf_counter() - endpoint_start
    finally:
        sampler.forward = original_sampler_forward
        _restore_query_instrumentation(runtime, instrumentation)

    expected_nfe = int(runtime.protocol["inference"]["denoising_steps"])
    if interval["calls"] != 1 or result.sampler_evaluations != expected_nfe or result.denoiser_calls != expected_nfe:
        raise RuntimeError("The query did not follow the frozen single-call, five-NFE sampler path")
    if len(state["nfe_rows"]) != expected_nfe or any(row["net_forward_calls"] < 1 for row in state["nfe_rows"]):
        raise RuntimeError("Per-NFE DiT forward instrumentation did not cover the complete sampler")
    for row in state["nfe_rows"]:
        if not row["paths"]:
            row["paths"] = ["eager_forward"]
        row["path"] = row["paths"][0] if len(set(row["paths"])) == 1 else "mixed"
        del row["paths"]
    return {"sampler_forward_calls": interval["calls"],
            "sampler_evaluations": int(result.sampler_evaluations),
            "denoiser_calls": int(result.denoiser_calls),
            "synchronized_sampler_wall_seconds": interval["wall_seconds"],
            "cuda_event_dit_range_ms": interval["cuda_event_ms"],
            "complete_endpoint_wall_seconds": endpoint_wall,
            "profiled_repeat": bool(profiled),
            "per_nfe_execution": state["nfe_rows"],
            "cuda_graph_after_query": (engine.graphs.metadata() if engine is not None else None),
            "condition_cache_after_query": (engine.cache.metadata() if engine is not None else None)}


def _install_profile_spans(runtime, torch):
    engine = getattr(runtime, "ours_vqwam_engine", None)
    if engine is None:
        return None
    banks = engine.banks
    original_operand, original_decode = banks._operand, banks.decode
    rotation = banks.rotation
    from torch.utils._python_dispatch import TorchDispatchMode
    counters = {"fp32_U_matmul": 0}

    class TagFp32UMatmul(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            op_name = str(function)
            operand = args[0] if args else None
            matrix = args[1] if len(args) > 1 else None
            same_rotation_storage = False
            if hasattr(matrix, "untyped_storage"):
                try:
                    same_rotation_storage = (matrix.untyped_storage().data_ptr()
                                             == rotation.untyped_storage().data_ptr())
                except (RuntimeError, AttributeError):
                    same_rotation_storage = matrix is rotation
            is_u = (op_name.startswith(("aten.matmul", "aten.mm", "aten.bmm"))
                    and same_rotation_storage and getattr(operand, "dtype", None) == torch.float32
                    and getattr(matrix, "dtype", None) == torch.float32
                    and tuple(matrix.shape[-2:]) == (128, 128))
            if is_u:
                counters["fp32_U_matmul"] += 1
                with torch.profiler.record_function("engine.fp32_U_matmul"):
                    return function(*args, **kwargs)
            return function(*args, **kwargs)

    def profiled_operand(value, name):
        with torch.profiler.record_function("engine._Banks._operand"):
            with TagFp32UMatmul():
                return original_operand(value, name)

    def profiled_decode(prepared, scratch):
        with torch.profiler.record_function("engine._Banks.decode"):
            return original_decode(prepared, scratch)

    banks._operand, banks.decode = profiled_operand, profiled_decode
    return banks, original_operand, original_decode, counters


def _restore_profile_spans(instrumentation) -> None:
    if instrumentation is not None:
        banks, original_operand, original_decode, _counters = instrumentation
        banks._operand, banks.decode = original_operand, original_decode


def _profile_repeat(common, runtime, context, torch, device, trace_path: Path,
                    *, force_eager: bool = False) -> dict:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    engine = getattr(runtime, "ours_vqwam_engine", None)
    spans = _install_profile_spans(runtime, torch) if force_eager else None
    original_net_forward = runtime.model.net.forward
    if force_eager and engine is not None:
        # Profile the actual wrapped linears; the four timed samples keep graph/eager behavior unchanged.
        runtime.model.net.forward = engine.graphs.original_net_forward
    graph_before = _graph_counts(engine)
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                   profile_memory=False, with_stack=False) as profiler:
            sample = _run_query(common, runtime, context, torch, device, profiled=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
        counters = spans[3] if spans is not None else {"fp32_U_matmul": 0}
        eager_nfes = sum(row["path"] == "eager_forward" for row in sample["per_nfe_execution"])
        if force_eager and engine is not None and (counters["fp32_U_matmul"] <= 0 or eager_nfes < 2):
            raise RuntimeError("Eager profiler repeat did not tag FP32 U matmul across at least two NFE")
        return {"query_key": [context["case_id"], context["seed_index"]],
                "timing_includes_profiler_overhead": True,
                "forced_eager_profile": bool(force_eager and engine is not None),
                "fp32_U_matmul_tagged_calls": int(counters["fp32_U_matmul"]),
                "graph_counter_delta": {key: value - graph_before[key]
                                         for key, value in _graph_counts(engine).items()},
                "profiled_repeat_sampler_wall_seconds": sample["synchronized_sampler_wall_seconds"],
                "profiled_repeat_cuda_event_dit_range_ms": sample["cuda_event_dit_range_ms"],
                "per_nfe_execution": sample["per_nfe_execution"],
                "trace_path": str(trace_path),
                "trace_bytes": trace_path.stat().st_size,
                "profiler": profiler_top_ops(profiler)}
    finally:
        runtime.model.net.forward = original_net_forward
        _restore_profile_spans(spans)


def _validate_readonly_inputs(old_root: Path, protocol: dict, arm: str) -> None:
    if protocol.get("protocol") != "cosmos-policy-vqwam-v1":
        raise ValueError("Unexpected Cosmos protocol ID")
    if arm not in ("bf16", "ours_vqwam") or arm not in protocol.get("arms", []):
        raise ValueError(f"Unsupported profile arm: {arm}")
    for path in (old_root / "results/language_cache/embeddings.pkl",
                 old_root / "results/language_cache/summary.json",
                 old_root / "protocol.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Read-only source input is missing: {path}")
    if arm == "ours_vqwam":
        for relative in ("results/ours/deployment/manifest.json",
                         "results/ours/inference_bundle/storage.json"):
            if not (old_root / relative).is_file():
                raise FileNotFoundError(f"Read-only ours artifact is missing: {old_root / relative}")
    diagnostic_queries(protocol)
    formal_panel_binding(protocol)


def run_profile(args, old_root: Path, root: Path, out: Path, protocol: dict, common) -> int:
    if not args.profile_only:
        raise ValueError("This worker only supports --profile-only; it never runs a formal panel")
    _validate_readonly_inputs(old_root, protocol, args.arm)
    for case_id, seed_index in DIAGNOSTIC_QUERIES:
        _query_context_path(old_root, case_id, seed_index)

    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rss_before_model = _rss_bytes()
    runtime = common.prepare_runtime(protocol, old_root, out)
    torch = common.torch_module()
    device = next(runtime.model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA model in the approved PBS job, found {device}")
    props = torch.cuda.get_device_properties(device)
    hardware = {"gpu_name": props.name, "total_memory_bytes": int(props.total_memory),
                "capability": [int(props.major), int(props.minor)],
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    rss_after_model = _rss_bytes()

    receipt_binding = {}
    metadata = _install_arm(runtime, protocol, old_root, out, args.arm, receipt_binding)
    if args.arm != "bf16" and metadata.get("native_int4_kernel_used") is not False:
        raise RuntimeError("Expected decoded-BF16 VQ execution with no native INT4 kernel")
    common.save_json(out / "install_metadata.json", metadata)
    engine = getattr(runtime, "ours_vqwam_engine", None)
    engine_initial_graph = engine.graphs.metadata() if engine is not None else None
    contexts = []
    for case_id, seed_index in DIAGNOSTIC_QUERIES:
        context = common.get_context(runtime, case_id, seed_index=seed_index)
        if (int(context.get("case_id", -1)), int(context.get("seed_index", -1))) != (case_id, seed_index):
            raise RuntimeError("Context key differs from the bounded diagnostic panel")
        contexts.append(context)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_queries = _cuda_memory(torch, device)
    samples = []
    memory_after_timed_queries = None
    rss_after_timed_queries = None
    try:
        for context in contexts:
            torch.cuda.reset_peak_memory_stats(device)
            sample = _run_query(common, runtime, context, torch, device)
            sample.update({"case_id": int(context["case_id"]),
                           "seed_index": int(context["seed_index"]),
                           "process_memory": _rss_bytes(),
                           "cuda_memory": _cuda_memory(torch, device)})
            samples.append(sample)
        memory_after_timed_queries = _cuda_memory(torch, device)
        rss_after_timed_queries = _rss_bytes()
        trace_stem = f"{args.arm}_{os.environ.get('PBS_JOBID', 'local')}_case3039_seed1"
        graph_trace = _profile_repeat(common, runtime, contexts[-1], torch, device,
                                      out / "traces" / f"{trace_stem}_graph_path.json")
        if engine is not None:
            eager_trace = _profile_repeat(common, runtime, contexts[-1], torch, device,
                                          out / "traces" / f"{trace_stem}_eager_ops.json",
                                          force_eager=True)
        else:
            eager_trace = None
        profiled_repeat = {"query_key": [3039, 1], "graph_path": graph_trace,
                           "eager_operator_attribution": eager_trace}
    finally:
        restore = getattr(runtime, "close", None)
        if callable(restore):
            restore()

    profile = {"format": FORMAT, "status": "complete", "protocol": protocol["protocol"],
               "arm": args.arm, "pbs_job_id": os.environ["PBS_JOBID"],
               "process_id": os.getpid(),
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "profile_only": True, "new_training_updates": 0, "closed_loop_episodes": 0,
               "diagnostic_panel": {"purpose": "bounded engine diagnosis; does not replace the formal panel",
                                    "queries": [{"case_id": case, "seed_index": seed}
                                                for case, seed in DIAGNOSTIC_QUERIES],
                                    "timed_query_count": len(samples),
                                    "profiled_repeat_count": 1 if engine is None else 2,
                                    "profiled_repeat_key": [3039, 1]},
               "formal_panel_binding": formal_panel_binding(protocol),
               "execution_boundary": {"sampler_wall": "synchronized sampler.forward; includes all five NFE",
                                      "cuda_event_dit_range": "CUDA event interval around sampler.forward",
                                      "full_endpoint_wall": "common.endpoint including endpoint setup; context load excluded",
                                      "low_bit": "decoded BF16 weights and BF16 GEMM; no native INT4 kernel"},
               "hardware": hardware,
               "resident_memory": {"before_model_load": rss_before_model,
                                   "after_model_load": rss_after_model,
                                   "before_queries_cuda": memory_before_queries,
                                   "after_timed_queries_cuda": memory_after_timed_queries,
                                   "after_timed_queries_process": rss_after_timed_queries,
                                   "after_profiled_repeats_cuda": _cuda_memory(torch, device),
                                   "after_profiled_repeats_process": _rss_bytes()},
               "engine_install": {"metadata": metadata,
                                  "reused_full80_receipt": receipt_binding or None,
                                  "cuda_graph_initial": engine_initial_graph},
               "timed_queries": samples,
               "profiler_repeat": profiled_repeat,
               "training_or_closed_loop": {"training_updates": 0, "closed_loop_episodes": 0}}
    size = write_profile(out / "profile.json", profile)
    print(json.dumps({"status": "complete", "arm": args.arm, "profile_path": str(out / "profile.json"),
                      "profile_bytes": size,
                      "trace_paths": [item["trace_path"] for item in
                                      (graph_trace, eager_trace) if item is not None],
                      "diagnostic_queries": 4, "formal_queries": 80, "formal_samples_per_arm": 720},
                     ensure_ascii=False, allow_nan=False), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True,
                        help="Read-only source results root for the frozen protocol and artifacts")
    parser.add_argument("--root", type=Path, required=True,
                        help="Isolated new experiment root")
    parser.add_argument("--out", type=Path, required=True,
                        help="Profile output directory inside --root")
    parser.add_argument("--arm", choices=("bf16", "ours_vqwam"), required=True)
    parser.add_argument("--profile-only", action="store_true",
                        help="Required explicit bound; this worker never runs the formal panel")
    parser.add_argument("--protocol", type=Path,
                        help="Optional protocol file; defaults to --old-root/protocol.json")
    parser.add_argument("--old-code-root", type=Path,
                        help="Optional directory containing the read-only legacy common.py/runtime.py/engine.py")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.profile_only:
        raise ValueError("Pass --profile-only to enable this bounded diagnostic worker")
    old_root, root, out = validate_output_paths(args.old_root, args.root, args.out)
    protocol_path = args.protocol or old_root / "protocol.json"
    protocol = read_json(protocol_path)

    code_root = (args.old_code_root or
                 (Path(os.environ["COSMOS_CODE_ROOT"]) if os.environ.get("COSMOS_CODE_ROOT") else None) or
                 (Path(__file__).resolve().parent.parent / "cosmos-policy-vqwam-20261024"))
    code_root = code_root.expanduser().resolve()
    if not (code_root / "common.py").is_file():
        raise FileNotFoundError(f"Cannot locate read-only legacy runtime source: {code_root}")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root))
    import common

    # This guard runs before importing the model runtime or Torch and before any tensor or artifact I/O.
    common.guard()
    return run_profile(args, old_root, root, out, protocol, common)


if __name__ == "__main__":
    raise SystemExit(main())

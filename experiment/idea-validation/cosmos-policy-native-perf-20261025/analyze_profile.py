"""Summarize CUDA profiler Chrome traces on an approved PBS CPU allocation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import tempfile


MAX_PROFILE_BYTES = 256 * 1024
CPU_CATEGORIES = {"cpu_op", "cuda_runtime", "user_annotation"}
KERNEL_CATEGORIES = {"kernel", "gpu_kernel"}
FOCUS_NAMES = ("aten::equal", "aten::_local_scalar_dense", "aten::item",
               "aten::copy_", "aten::mm", "cudaLaunchKernel",
               "cudaStreamSynchronize", "cudaDeviceSynchronize")
TOP_LIMIT = 40


def validate_allocation(job_id: str | None, nodefile: str | None, hostname: str,
                        visible_devices: str | None = None,
                        node_names: set[str] | None = None) -> None:
    """Require PBS identity and a non-login host listed in this allocation."""
    host = hostname.split(".", 1)[0]
    if not job_id or not nodefile or "login" in host.lower():
        raise RuntimeError("Trace analysis requires a genuine PBS compute allocation")
    if visible_devices not in (None, "", "-1", "NoDevFiles"):
        raise RuntimeError("Trace analysis must run in a CPU-only allocation")
    if node_names is None:
        try:
            node_names = {line.split(".", 1)[0] for line in Path(nodefile).read_text().splitlines() if line.strip()}
        except OSError as exc:
            raise RuntimeError("PBS nodefile is unavailable") from exc
    if host not in node_names:
        raise RuntimeError("Current host is outside the PBS allocation")


def guard_real_pbs_cpu() -> None:
    validate_allocation(os.environ.get("PBS_JOBID"), os.environ.get("PBS_NODEFILE"),
                        socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    if os.environ.get("PBS_NGPUS", "0").strip() not in ("", "0"):
        raise RuntimeError("Trace analysis requires a PBS allocation with zero GPUs")


def _event_key(event: dict) -> tuple[str, str]:
    return str(event.get("cat", "")), str(event.get("name", ""))


def _is_focus(name: str, target: str) -> bool:
    return name == target or name.startswith(target + "_")


def analyze_events(events: list[dict], trace_path: str = "<memory>") -> dict:
    """Aggregate CPU intervals with direct-child subtraction and GPU kernels separately."""
    inclusive = defaultdict(lambda: {"count": 0, "inclusive_us": 0.0})
    kernels = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "max_duration_us": 0.0})
    thread_events = defaultdict(list)
    category_counts = defaultdict(int)
    complete_events = 0

    for event in events:
        if event.get("ph") != "X":
            continue
        category = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        try:
            start, duration = float(event["ts"]), float(event["dur"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            continue
        complete_events += 1
        category_counts[category] += 1
        key = (category, name)
        if category in CPU_CATEGORIES:
            row = inclusive[key]
            row["count"] += 1
            row["inclusive_us"] += duration
            thread_events[(str(event.get("pid", "")), str(event.get("tid", "")))].append(
                {"start": start, "end": start + duration, "duration": duration,
                 "key": key, "child_us": 0.0, "overlap": False})
        elif category in KERNEL_CATEGORIES:
            row = kernels[key]
            row["count"] += 1
            row["duration_us"] += duration
            row["max_duration_us"] = max(row["max_duration_us"], duration)

    valid_exclusive = defaultdict(lambda: {"exclusive_us": 0.0, "valid_count": 0, "overlap_count": 0})
    overlap_events = 0
    overlap_threads = 0
    for records in thread_events.values():
        records.sort(key=lambda row: (row["start"], -row["end"], row["key"]))
        active = []
        thread_overlapped = False
        for event in records:
            start, end = event["start"], event["end"]
            active = [prior for prior in active if prior["end"] > start]
            equal_interval = [prior for prior in active
                              if prior["start"] == start and prior["end"] == end]
            crossing = [prior for prior in active if end > prior["end"]]
            if equal_interval or crossing:
                event["overlap"] = True
                thread_overlapped = True
                # In a crossing interval, direct-child accounting is ambiguous for every
                # active scope that overlaps it, including otherwise containing ancestors.
                for prior in active:
                    if start < prior["end"] and prior["start"] < end:
                        prior["overlap"] = True
            else:
                containers = [prior for prior in active if end <= prior["end"]]
                if containers:
                    parent = max(containers, key=lambda prior: (prior["start"], -prior["end"]))
                    parent["child_us"] += event["duration"]
            active.append(event)

        for event in records:
            row = valid_exclusive[event["key"]]
            if event["overlap"] or event["child_us"] > event["duration"] + 1e-6:
                row["overlap_count"] += 1
                overlap_events += 1
                thread_overlapped = True
            else:
                row["valid_count"] += 1
                row["exclusive_us"] += max(0.0, event["duration"] - event["child_us"])
        if thread_overlapped:
            overlap_threads += 1

    cpu_rows = []
    for key, totals in inclusive.items():
        exclusive = valid_exclusive[key]
        count = totals["count"]
        overlap_count = exclusive["overlap_count"]
        valid_count = exclusive["valid_count"]
        status = ("overlap_non_nested" if overlap_count and not valid_count else
                  "mixed_overlap" if overlap_count else "strict")
        cpu_rows.append({"category": key[0], "name": key[1], "calls": count,
                         "inclusive_us": totals["inclusive_us"],
                         "exclusive_us": exclusive["exclusive_us"],
                         "exclusive_valid_calls": valid_count,
                         "overlap_non_nested_calls": overlap_count,
                         "exclusive_status": status})
    cpu_inclusive = sorted(cpu_rows, key=lambda row: row["inclusive_us"], reverse=True)
    cpu_exclusive = sorted(cpu_rows, key=lambda row: row["exclusive_us"], reverse=True)
    kernel_rows = [{"category": key[0], "name": key[1], "calls": row["count"],
                    "duration_us": row["duration_us"],
                    "mean_duration_us": row["duration_us"] / row["count"],
                    "max_duration_us": row["max_duration_us"]}
                   for key, row in kernels.items()]
    kernel_rows.sort(key=lambda row: row["duration_us"], reverse=True)
    focus = {target: [row for row in cpu_rows if _is_focus(row["name"], target)]
             for target in FOCUS_NAMES}
    return {"trace_path": trace_path, "complete_event_count": complete_events,
            "category_event_counts": dict(sorted(category_counts.items())),
            "cpu_inclusive_top": cpu_inclusive[:TOP_LIMIT],
            "cpu_exclusive_top": cpu_exclusive[:TOP_LIMIT],
            "focus_cpu_events": focus,
            "engine_user_annotations": [row for row in cpu_rows
                                        if row["category"] == "user_annotation"
                                        and row["name"].startswith("engine.")],
            "gpu_kernel_top": kernel_rows[:TOP_LIMIT],
            "exclusive_quality": {"non_nested_overlap_events": overlap_events,
                                  "threads_with_non_nested_overlap": overlap_threads,
                                  "strict_exclusive_sum_caveat": "Rows with overlap flags include only valid-event exclusive time; nested category totals are not additive."}}


def analyze_trace(path: Path) -> dict:
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"Empty Chrome trace: {path}")
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"Chrome trace lacks a traceEvents array: {path}")
    result = analyze_events(events, str(path))
    result["trace_bytes"] = size
    return result


def bounded_json_bytes(value: dict, limit: int = MAX_PROFILE_BYTES) -> bytes:
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > limit:
        raise ValueError(f"Trace summary exceeds {limit} bytes: {len(payload)}")
    return payload


def write_summary(path: Path, value: dict) -> int:
    payload = bounded_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(payload)


def _read_profile(path: Path) -> dict:
    size = path.stat().st_size
    if size > MAX_PROFILE_BYTES:
        raise ValueError(f"Profile metadata exceeds {MAX_PROFILE_BYTES} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise RuntimeError(f"Profile is not complete: {path}")
    return value


def _profile_path(root: Path, arm: str) -> Path:
    candidates = [root / "results" / "profile_initial" / arm / "profile.json"]
    if arm == "ours_vqwam":
        candidates.append(root / "results" / "profile_initial" / "ours" / "profile.json")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Profile JSON is missing for {arm}: {candidates[0]}")


def _trace_binding(root: Path, trace_path: str) -> Path:
    path = Path(trace_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Trace path escapes --root: {path}")
    return path


def _profile_trace(root: Path, profile: dict, trace_kind: str, item: dict) -> dict:
    trace_path = _trace_binding(root, item["trace_path"])
    analysis = analyze_trace(trace_path)
    return {"kind": trace_kind, "query_key": item.get("query_key"),
            "forced_eager_profile": bool(item.get("forced_eager_profile", False)),
            "timing_includes_profiler_overhead": bool(item.get("timing_includes_profiler_overhead", True)),
            "fp32_U_matmul_tagged_calls": item.get("fp32_U_matmul_tagged_calls", 0),
            "graph_counter_delta": item.get("graph_counter_delta"),
            "profiled_repeat_sampler_wall_seconds": item.get("profiled_repeat_sampler_wall_seconds"),
            "profiled_repeat_cuda_event_dit_range_ms": item.get("profiled_repeat_cuda_event_dit_range_ms"),
            "trace": analysis}


def analyze_profiles(root: Path) -> dict:
    root = root.expanduser().resolve()
    arms = {}
    for arm in ("bf16", "ours_vqwam"):
        profile_path = _profile_path(root, arm)
        profile = _read_profile(profile_path)
        if profile.get("protocol") != "cosmos-policy-vqwam-v1" or profile.get("arm") != arm:
            raise ValueError(f"Profile protocol/arm binding differs: {profile_path}")
        repeat = profile.get("profiler_repeat", {})
        traces = []
        graph = repeat.get("graph_path")
        if not isinstance(graph, dict) or not graph.get("trace_path"):
            raise ValueError(f"Graph-path trace binding is missing: {profile_path}")
        traces.append(_profile_trace(root, profile, "graph_path", graph))
        eager = repeat.get("eager_operator_attribution")
        if eager is not None:
            if not isinstance(eager, dict) or not eager.get("trace_path"):
                raise ValueError(f"Eager trace binding is invalid: {profile_path}")
            traces.append(_profile_trace(root, profile, "eager_operator_attribution", eager))
        arms[arm] = {"profile_json": str(profile_path), "profile_created_utc": profile.get("created_utc"),
                     "hardware": profile.get("hardware"),
                     "diagnostic_queries": profile.get("diagnostic_panel", {}).get("queries"),
                     "traces": traces}
        print(f"TRACE_ANALYZED arm={arm} trace_count={len(traces)}", flush=True)
    return {"format": "cosmos_profile_trace_analysis_v1", "status": "complete",
            "protocol": "cosmos-policy-vqwam-v1", "pbs_job_id": os.environ.get("PBS_JOBID"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": {"hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "gpu_count": int(os.environ.get("PBS_NGPUS", "0") or "0")},
            "semantics": {"cpu_categories": sorted(CPU_CATEGORIES),
                          "gpu_kernel_categories": sorted(KERNEL_CATEGORIES),
                          "cpu_children": "exclusive duration subtracts direct tracked children within each pid/tid interval tree",
                          "overlaps": "non-nested overlaps are flagged; those event durations are excluded from strict exclusive sums",
                          "kernel_separation": "GPU kernel durations are reported separately from CPU operator/runtime totals; do not add them to aten rows",
                          "trace_input": "original profile traces are read in place and are not copied"},
            "arms": arms}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    guard_real_pbs_cpu()
    summary = analyze_profiles(args.root)
    path = args.root / "results" / "profile_analysis" / "summary.json"
    size = write_summary(path, summary)
    print(json.dumps({"status": "complete", "summary": str(path), "bytes": size},
                     ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

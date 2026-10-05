#!/usr/bin/env python3
"""Reduce the remote profile to small, readable metrics inside its allocation."""
import json
import os
import re
import socket
import statistics
from collections import defaultdict
from pathlib import Path

if not os.environ.get("PBS_JOBID") or "login" in socket.gethostname().lower():
    raise SystemExit("summarization requires the PBS compute allocation")
artifacts = Path(os.environ["ARTIFACTS"])
data = json.loads((artifacts / "profile_summary.json").read_text())
assert data["status"] == "complete", data.get("error")
replay = data["replay"]
spans = replay["instrumentation_spans"]

# Every detailed cell is a repeated inclusive interval. Aggregate matching
# siblings only, dividing by the number of calls represented in each phase.
groups = defaultdict(lambda: defaultdict(float))
layer_groups = defaultdict(lambda: defaultdict(float))
context_ids = defaultdict(set)
phase_repetitions = {}
for row in spans:
    phase = row["pass"]
    if row.get("context_index") is not None:
        context_ids[phase].add(row["context_index"])
    name = row["name"]
    cpu = row.get("cpu_wall_ms", {})
    gpu = row.get("cuda_event_ms", {})
    layer_match = re.fullmatch(r"module\.(video_expert|action_expert)\.blocks\.(\d+)\.(self_attn\.[qkvo]|cross_attn|ffn)", name)
    if layer_match:
        expert, layer, component = layer_match.groups()
        cell = layer_groups[(phase, expert, int(layer), component)]
        cell["cpu_sum_ms"] += (cpu.get("mean") or 0) * cpu.get("count", 0)
        cell["cuda_sum_ms"] += (gpu.get("mean") or 0) * gpu.get("count", 0)
        cell["count"] += cpu.get("count", 0)
    bucket = None
    if name in {"VAE.input_encode", "text.encode_prompt", "video.prepare", "video.cache_prefill", "FastWAM.infer_action", "model.infer_action", "action.prepare", "action.post", "action.denoise_step", "action.scheduler_step", "preprocess.obs_to_model_input", "postprocess.denormalize_action", "postprocess.invert_gripper"}:
        bucket = name
    elif "flash_attention" in name:
        bucket = name
    elif name.startswith("module."):
        expert = "video" if ".video." in name or "video_expert." in name else "action" if ".action." in name or "action_expert." in name else None
        if expert and re.search(r"\.ffn$", name):
            bucket = f"{expert}.FFN.parents"
        elif expert and re.search(r"\.cross_attn$", name):
            bucket = f"{expert}.text_cross_attention.parents"
        elif expert and re.search(r"\.self_attn\.[qkvo]$", name):
            bucket = f"{expert}.projection.{name[-1]}"
    if bucket:
        cell = groups[(phase, bucket)]
        cell["cpu_sum_ms"] += (cpu.get("mean") or 0) * cpu.get("count", 0)
        cell["cuda_sum_ms"] += (gpu.get("mean") or 0) * gpu.get("count", 0)
        cell["count"] += cpu.get("count", 0)
    if name == "action.denoise_step":
        cell = groups[(phase, f"denoise.step{row.get('denoise_step')}")]
        cell["cpu_sum_ms"] += (cpu.get("mean") or 0) * cpu.get("count", 0)
        cell["cuda_sum_ms"] += (gpu.get("mean") or 0) * gpu.get("count", 0)
        cell["count"] += cpu.get("count", 0)
    if name in {"FastWAM.infer_action", "model.infer_action"}:
        phase_repetitions[phase] = max(phase_repetitions.get(phase, 0), cpu.get("count", 0))

stage_rows = []
for (phase, name), values in sorted(groups.items()):
    per_context = phase_repetitions.get(phase, replay["paired_repeats_per_context"])
    calls = len(context_ids[phase]) * per_context
    stage_rows.append({"phase": phase, "name": name, "represented_calls": calls,
                       "events": int(values["count"]),
                       "cpu_inclusive_ms_per_call": values["cpu_sum_ms"] / calls if calls else None,
                       "cuda_inclusive_ms_per_call": values["cuda_sum_ms"] / calls if calls else None})

environment_rows = []
for row in data["episode"]["native_episode_cpu_stages"]:
    if row["name"].startswith("environment.") or row["name"].startswith("video_output."):
        key = row["name"]
        if key == "environment.sim.render":
            key += ":" + row.get("camera_name", "unknown")
        values = row["cpu_wall_ms"]
        environment_rows.append({"name": key, "count": values["count"],
                                 "sum_ms": values["mean"] * values["count"],
                                 "mean_ms": values["mean"], "median_ms": values["median"]})

layer_rows = []
for (phase, expert, layer, component), values in sorted(layer_groups.items()):
    calls = len(context_ids[phase]) * phase_repetitions.get(phase, replay["paired_repeats_per_context"])
    layer_rows.append({"phase": phase, "expert": expert, "layer": layer, "component": component,
                       "represented_calls": calls, "events": int(values["count"]),
                       "cpu_inclusive_ms_per_call": values["cpu_sum_ms"] / calls if calls else None,
                       "cuda_inclusive_ms_per_call": values["cuda_sum_ms"] / calls if calls else None})

metrics = {"status": data["status"], "allocation": data["allocation"], "source": data["source"],
           "protocol": data["protocol"], "bootstrap": data["bootstrap_cpu_stages"],
           "official_episode_result": data.get("official_episode_result"),
           "episode_chunk_count": data["episode"]["action_context_count"],
           "episode_first_chunk_ms": data["episode"]["first_chunk_cpu_wall_ms"],
           "episode_steady_chunk_ms": data["episode"]["steady_chunk_cpu_wall_ms"],
           "native_predict_ms": replay["native_predict_cpu_wall_ms"],
           "detailed_predict_ms": replay["instrumented_predict_cpu_wall_ms"],
           "detailed_overhead_fraction": replay["instrumentation_cpu_overhead_fraction"],
           "contexts": replay["contexts"], "pairs": replay["pairs"],
           "parity": replay["output_parity"], "stages": stage_rows, "layer_stages": layer_rows,
           "environment": environment_rows,
           "attention_geometry": replay.get("attention_geometry_counts", []),
           "qkv_shapes": replay.get("qkv_shape_counts", []),
           "profiler_operator_summary": replay.get("profiler_operator_summary", []),
           "coarse_replay": replay.get("low_overhead_pass", {}),
           "interpretation": data["interpretation"]}
(artifacts / "report_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("PROFILE_METRICS_COMPLETE", {"stages": len(stage_rows), "pairs": len(replay["pairs"]), "path": str(artifacts / "report_metrics.json")})

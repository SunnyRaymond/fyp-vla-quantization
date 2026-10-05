#!/usr/bin/env python3
"""Write the compact IDM metrics inside its guarded PBS allocation."""
import json
import os
import re
import socket
from collections import defaultdict
from pathlib import Path

if not os.environ.get("PBS_JOBID") or "login" in socket.gethostname().lower():
    raise SystemExit("IDM profile summarization requires a PBS compute allocation")
artifacts = Path(os.environ["ARTIFACTS"])
data = json.loads((artifacts / "profile_summary.json").read_text(encoding="utf-8"))
if data.get("status") != "complete":
    raise SystemExit(f"profile summary is incomplete: {data.get('error')}")
replay = data["replay"]
paired_modes = replay["paired_mode_timings"]
contexts = replay["contexts"]

def all_parity_ok(row):
    if isinstance(row, dict):
        if "allclose" in row and not row["allclose"]:
            return False
        return all(all_parity_ok(value) for value in row.values())
    if isinstance(row, list):
        return all(all_parity_ok(value) for value in row)
    return True

if len(contexts) != 3 or len(paired_modes["pairs"]) != 12:
    raise SystemExit("expected three actual contexts and 12 paired first_frame/idm comparisons")
if not all(all_parity_ok(row) for row in replay["output_parity"]):
    raise SystemExit("IDM native/fine parity gate failed")
if not all(all_parity_ok(row) for row in paired_modes["same_mode_repeat_parity"]):
    raise SystemExit("fixed-mode repeat parity gate failed")
if len(paired_modes["instrumentation_parity"]) != 3 or not all(all_parity_ok(row) for row in paired_modes["instrumentation_parity"]):
    raise SystemExit("first_frame native/fine parity gate failed")
if len(replay["low_overhead_pass"]["pairs"]) != 6 or not all(all_parity_ok(row["parity"]) for row in replay["low_overhead_pass"]["pairs"]):
    raise SystemExit("IDM native/thin parity gate failed")

geometry = replay["video_prepare_geometry"]
video_rows = [row for row in geometry if row["pass"] == "replay_instrumented" and row["prepare_role"] == "noisy_video_step_input"]
cache_rows = [row for row in geometry if row["pass"] == "replay_instrumented" and row["prepare_role"] == "frozen_cond_video_cache_prefill"]
if not video_rows or not cache_rows or any(row["tokens_per_frame"] != 98 or row["video_seq_len"] != 294 for row in video_rows + cache_rows):
    raise SystemExit("runtime video.prepare geometry did not confirm 98 tokens/frame and 294 video tokens")
all_action_attention = [row for row in replay["attention_geometry_counts"] if row.get("attention_stage") == "action_denoise"]
action_attention = [row for row in all_action_attention if row.get("pass") == "replay_instrumented"]
first_frame_attention = [row for row in all_action_attention if row.get("pass") == "mode_first_frame_instrumented"]
expected_groups = {"current_observation_frame": 98, "generated_future_video_frames": 196, "current_action_tokens": 32}
if not action_attention or any(row.get("key_groups") != expected_groups for row in action_attention):
    raise SystemExit("action mixed-attention K/V partition did not confirm 98/196/32 runtime tokens")
expected_first_frame_groups = {"current_observation_frame": 98, "generated_future_video_frames": 0, "current_action_tokens": 32}
if not first_frame_attention or any(row.get("key_groups") != expected_first_frame_groups for row in first_frame_attention):
    raise SystemExit("first_frame action mixed-attention geometry did not confirm 98/0/32 runtime tokens")
if not replay["video_attention_geometry_counts"]:
    raise SystemExit("video self/text-cross attention geometry is missing")

# Keep each layer's stage separate: the video expert runs repeatedly for IDM
# denoising and once per cache prefill. Counts describe module-forward events.
layer_groups = defaultdict(lambda: {"events": 0, "cpu_sum_ms": 0.0, "cuda_sum_ms": 0.0})
stage_rows = []
for row in replay["instrumentation_spans"]:
    cpu, cuda = row.get("cpu_wall_ms", {}), row.get("cuda_event_ms", {})
    count = int(cpu.get("count", 0))
    name, stage = row["name"], row.get("stage")
    m = re.fullmatch(r"module\.(video_expert|action_expert)\.blocks\.(\d+)\.(self_attn\.[qkvo]|cross_attn|ffn)", name)
    if m:
        expert = "video" if m.group(1) == "video_expert" else "action"
        key = (row["pass"], stage, expert, int(m.group(2)), m.group(3))
        cell = layer_groups[key]
        cell["events"] += count
        cell["cpu_sum_ms"] += (cpu.get("mean") or 0.0) * count
        cell["cuda_sum_ms"] += (cuda.get("mean") or 0.0) * int(cuda.get("count", 0))
    is_step_or_stage = name in {
        "model.infer_action", "video.denoise_step", "video.scheduler_step",
        "video.cache_prefill", "action.denoise_step", "action.scheduler_step",
    }
    is_attention = name.startswith(("MoT.", "video.")) and name.endswith(".flash_attention")
    if is_step_or_stage or is_attention:
        stage_rows.append({
        "pass": row["pass"], "stage": stage, "name": name,
        "context_index": row.get("context_index"), "denoise_step": row.get("denoise_step"),
        "cpu_wall_ms": cpu, "cuda_event_ms": cuda or None,
        })
layer_rows = [
    {"pass": phase, "stage": stage, "expert": expert, "layer": layer, "component": component,
     **values,
     "cpu_inclusive_mean_ms_per_event": values["cpu_sum_ms"] / values["events"] if values["events"] else None,
     "cuda_inclusive_mean_ms_per_event": values["cuda_sum_ms"] / values["events"] if values["events"] else None}
    for (phase, stage, expert, layer, component), values in sorted(layer_groups.items())
]

# Keep actual operator/kernel rows separate from profiler scope totals. FW/
# scopes are retained only when they have CPU duration and a valid parent chain.
profile_rows = replay.get("profiler_operator_summary", [])
ops = [row for row in profile_rows if not str(row.get("operator_or_record_function", "")).startswith("FW/")]
ops.sort(key=lambda row: (float(row.get("cuda_self_us", 0.0)), float(row.get("cuda_total_us", 0.0))), reverse=True)
ops = ops[:80]
scoped_cuda = [
    row for row in profile_rows
    if str(row.get("operator_or_record_function", "")).startswith("FW/")
    and float(row.get("cpu_total_us", 0.0)) > 0.0
    and row.get("ancestor_labels")
]

metrics = {
    "status": data["status"], "claim_label": data["claim_label"],
    "allocation": data["allocation"], "source": data["source"], "protocol": data["protocol"],
    "official_episode_result": data.get("official_episode_result"),
    "episode": {
        "action_context_count": data["episode"]["action_context_count"],
        "first_chunk_cpu_wall_ms": data["episode"]["first_chunk_cpu_wall_ms"],
        "steady_chunk_cpu_wall_ms": data["episode"]["steady_chunk_cpu_wall_ms"],
        "native_cpu_stages": data["episode"]["native_episode_cpu_stages"],
    },
    "idm_native_fine": {
        "native_predict_cpu_wall_ms": replay["native_predict_cpu_wall_ms"],
        "instrumented_predict_cpu_wall_ms": replay["instrumented_predict_cpu_wall_ms"],
        "cpu_overhead_fraction": replay["instrumentation_cpu_overhead_fraction"],
        "cuda_event_overhead_ms": replay["instrumentation_cuda_event_overhead_ms"],
        "pairs": replay["pairs"], "parity": replay["output_parity"],
    },
    "idm_native_thin": replay["low_overhead_pass"],
    "paired_mode_timings": paired_modes,
    "contexts": contexts,
    "stage_timings": stage_rows,
    "layer_timings": layer_rows,
    "action_mixed_attention_geometry": action_attention,
    "first_frame_action_attention_geometry": first_frame_attention,
    "video_attention_geometry": replay["video_attention_geometry_counts"],
    "video_prepare_runtime_geometry": geometry,
    "qkv_shapes": replay["qkv_shape_counts"],
    "profiler_operator_kernel_summary": ops,
    "profiler_scoped_cuda_summary": scoped_cuda,
    "profiler_trace_remote": replay.get("profiler_trace"),
    "interpretation": data["interpretation"],
}
out = artifacts / "report_metrics.json"
out.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
print("IDM_PROFILE_METRICS_COMPLETE", {"contexts": len(contexts), "mode_pairs": len(paired_modes["pairs"]), "bytes": out.stat().st_size, "path": str(out)}, flush=True)

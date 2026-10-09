"""Render bounded diagnostic plots from frozen Fast-WAM collector outputs.

Run only inside an approved PBS compute allocation. This script reads exported
records and snapshots; it does not load a model, refit VQ, or issue queries.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from io import BytesIO
import json
import math
import os
import sys
from pathlib import Path


CONFIGS = ("identity", "hadamard", "smooth05_hadamard", "smooth1_hadamard")
WEIGHT_CONFIGS = ("smooth05_hadamard", "smooth1_hadamard")
CASES = (1, 13, 21)
MAX_PNG_BYTES = 220 * 1024


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_data_file(data_dir: Path, snapshots_dir: Path, value) -> Path:
    if not value:
        raise ValueError("Missing channel_stats_file path in activation record")
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [data_dir / raw, snapshots_dir / raw,
                                                 snapshots_dir / raw.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"channel_stats_file does not resolve under collector output: {value}")


def require_fields(row: dict, names: tuple[str, ...], where: str) -> None:
    missing = [name for name in names if name not in row]
    if missing:
        raise ValueError(f"{where} is missing required fields: {', '.join(missing)}")


def finite(value, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value for {where}, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Expected finite value for {where}, got {value!r}")
    return number


def stream_bucket(row: dict) -> str | None:
    label = f"{row.get('stream', '')} {row.get('module', '')}".lower().replace("-", "_")
    if "video" in label:
        return "video_expert"
    if "action" in label:
        return "action_expert"
    if "proprio" in label:
        return "proprio_encoder"
    return None


def aggregate_energy(rows: list[dict], group_fields: tuple[str, ...]) -> dict[tuple, dict]:
    """Combine calls as sqrt(sum(error energy)/sum(signal energy))."""
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_fields)
        bucket = grouped.setdefault(key, {"error_squared": 0.0, "signal_squared": 0.0, "calls": 0,
                                          "zero_count": 0, "rows": 0})
        bucket["error_squared"] += finite(row["error_squared"], "error_squared")
        bucket["signal_squared"] += finite(row["signal_squared"], "signal_squared")
        bucket["calls"] += 1
        bucket["zero_count"] += int(row.get("zero_count", 0))
        bucket["rows"] += int(row.get("rows", 0))
    for bucket in grouped.values():
        bucket["rrmse"] = math.sqrt(bucket["error_squared"] / max(bucket["signal_squared"], 1e-30))
    return grouped


def activation_channel_expected_groups(coverage_per_query: list[dict], selected: list[dict],
                                       configs: tuple[str, ...]) -> tuple[set[tuple], list[dict]]:
    selected_by_name = {row["module"]: int(row["module_index"]) for row in selected}
    if len(selected_by_name) != len(selected):
        raise ValueError("selected modules must have unique names")
    active_by_case = defaultdict(set)
    observed_by_case_module = defaultdict(list)
    seen_cases = set()
    for query_index, query in enumerate(coverage_per_query):
        if not isinstance(query, dict) or "case_id" not in query or not isinstance(query.get("modules"), dict):
            raise ValueError(f"Invalid selected-module coverage query at index {query_index}")
        case_id = int(query["case_id"])
        if case_id not in CASES:
            raise ValueError(f"Unexpected selected-module coverage case: {case_id}")
        seen_cases.add(case_id)
        modules = query["modules"]
        absent = sorted(set(selected_by_name) - set(modules))
        if absent:
            raise ValueError(f"Selected-module coverage is missing modules for case {case_id}: {absent}")
        for module_name, module_index in selected_by_name.items():
            events = modules[module_name]
            if not isinstance(events, list):
                raise ValueError(f"Invalid stage/step coverage for case {case_id}, module {module_name}")
            for event in events:
                if not isinstance(event, dict) or not {"stage", "step", "calls"} <= set(event):
                    raise ValueError(f"Invalid stage/step event for case {case_id}, module {module_name}")
                stage, step, calls = str(event["stage"]), int(event["step"]), int(event["calls"])
                if calls < 0:
                    raise ValueError(f"Negative call count for case {case_id}, module {module_name}")
                if calls:
                    observed_by_case_module[case_id, module_name].append(
                        {"stage": stage, "step": step, "calls": calls})
                    if stage in ("video", "action") and 0 <= step <= 9:
                        active_by_case[case_id].add(module_index)
    if seen_cases != set(CASES):
        raise ValueError(f"Selected-module coverage is missing cases: {sorted(set(CASES) - seen_cases)}")

    expected = {(case_id, config, module_index) for case_id, module_indices in active_by_case.items()
                for config in configs for module_index in module_indices}
    exclusions = []
    conditioning_stages = {"conditioning", "video_conditioning_prefill"}
    for case_id in CASES:
        for module_name, module_index in selected_by_name.items():
            if module_index in active_by_case[case_id]:
                continue
            events = observed_by_case_module[case_id, module_name]
            conditioning_only = bool(events) and all(
                event["stage"] in conditioning_stages or event["step"] == -1 for event in events)
            if conditioning_only:
                reason = "conditioning-only calls observed; no video/action scheduler calls at steps 0..9"
            elif events:
                reason = "no video/action scheduler calls at steps 0..9"
            else:
                reason = "no observed forward calls for this case"
            exclusions.append({"case_id": case_id, "module": module_name, "module_index": module_index,
                               "classification": "conditioning_only" if conditioning_only else "no_scheduler_calls",
                               "reason": reason, "observed_stage_steps": events})
    return expected, exclusions


def aggregation_selfcheck() -> None:
    demo = [{"module_index": 7, "config": "toy", "error_squared": 9.0, "signal_squared": 1.0},
            {"module_index": 7, "config": "toy", "error_squared": 1.0, "signal_squared": 99.0}]
    grouped = aggregate_energy(demo, ("module_index", "config"))
    combined = grouped[(7, "toy")]["rrmse"]
    mean_of_ratios = sum(math.sqrt(r["error_squared"] / r["signal_squared"]) for r in demo) / len(demo)
    assert grouped[(7, "toy")]["calls"] == 2
    assert abs(combined - math.sqrt(0.1)) < 1e-12
    assert abs(combined - mean_of_ratios) > 0.1
    demo_selected = [{"module": "root.proprio", "module_index": 3},
                     {"module": "action.linear", "module_index": 7}]
    demo_coverage = [{"case_id": case_id, "modules": {
        "root.proprio": [{"stage": "conditioning", "step": -1, "calls": 1}],
        "action.linear": [{"stage": "action", "step": 0, "calls": 1}]}}
        for case_id in CASES]
    expected, exclusions = activation_channel_expected_groups(demo_coverage, demo_selected, CONFIGS)
    assert expected == {(case_id, config, 7) for case_id in CASES for config in CONFIGS}
    assert all(item["classification"] == "conditioning_only" and item["module_index"] == 3
               for item in exclusions)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_validation
    run_validation.guard()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
    from PIL import Image

    aggregation_selfcheck()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="collector output directory")
    parser.add_argument("--full", type=Path, required=True, help="completed full validation output directory")
    parser.add_argument("--out", type=Path, required=True, help="new, empty plot output directory")
    args = parser.parse_args()
    data_dir, full_dir, out_dir = (p.resolve() for p in (args.data, args.full, args.out))
    if not data_dir.is_dir() or not full_dir.is_dir():
        raise FileNotFoundError("--data and --full must both be existing directories")
    if out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            raise FileExistsError(f"--out must be a fresh or empty directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True)

    job_id = os.environ["PBS_JOBID"]
    full_summary = read_json(full_dir / "summary.json")
    data_summary_path = data_dir / "summary.json"
    data_summary = read_json(data_summary_path)
    if full_summary.get("phase") != "full" or full_summary.get("status") != "complete":
        raise ValueError("--full/summary.json must describe a completed full phase")
    if not full_summary.get("pbs_jobid"):
        raise ValueError("--full/summary.json has no source pbs_jobid")
    collector_pbsid = (data_summary.get("pbs_jobid") or data_summary.get("collector_pbsid") or
                       data_summary.get("source_pbsid"))
    if not collector_pbsid and not data_summary.get("source_pbsids"):
        raise ValueError("--data/summary.json must identify the collector/source PBS job")

    summary = {
        "status": "running", "stage": "input_load", "pbs_jobid": job_id,
        "source_pbsids": {"full": full_summary["pbs_jobid"],
                           "collector": collector_pbsid,
                           "collector_sources": data_summary.get("source_pbsids", {})},
        "counts": {}, "figure_paths": [], "figure_render_info": [],
        "missing_coverage_assert": {"passed": False, "missing": []},
        "notes_zh": ["所有量化结果来自输入文件中的实际导出记录；本脚本不运行模型或重新拟合 VQ。",
                     "Activation relative RMSE 按同一 case、stage、step、layer 汇总各调用的平方误差能量和信号能量后计算。",
                     "Scheduler heatmap 排除 step=-1 的 conditioning 记录；它们仍计入输入覆盖统计。",
                     "Hadamard/Smooth 变换会改变 activation channel 坐标；不同配置的横向 channel 位置不能直接对应。",
                     "Weight 矩阵热图使用不超过 64×64 的 block maximum pooling；显示的是峰值分布，不是逐元素矩阵。",
                     "Action error 是固定输入上相对同 case、同 seed BF16 action 的差异，不代表闭环成功率。"]
    }
    write_json(out_dir / "summary.json", summary)

    print("[plot] loading collector records", flush=True)
    weights = jsonl(data_dir / "weights.jsonl")
    selected_path = data_dir / "selected.json"
    if not selected_path.is_file():
        raise FileNotFoundError(f"Required input is missing: {selected_path}")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    activations = jsonl(data_dir / "activations.jsonl")
    actions = jsonl(full_dir / "actions.jsonl")
    if not isinstance(selected, list) or len(selected) != 6:
        raise ValueError("selected.json must contain exactly six selected module objects")
    for index, row in enumerate(selected):
        require_fields(row, ("module", "module_index", "stream", "reason"), f"selected.json[{index}]")
    summary["counts"].update({"weight_rows": len(weights), "selected_modules": len(selected),
                              "activation_rows": len(activations), "full_action_rows": len(actions)})

    weight_by_config = {name: {} for name in WEIGHT_CONFIGS}
    for row_index, row in enumerate(weights):
        require_fields(row, ("config", "module", "module_index", "stream", "shape", "weight_rms",
                             "scalar_rmse", "vq_rmse", "scalar_rrmse", "vq_rrmse", "ratio",
                             "scalar_peak_error", "vq_peak_error", "scalar_peak_position", "vq_peak_position"),
                       f"weights.jsonl row {row_index + 1}")
        config = row["config"]
        if config not in weight_by_config:
            raise ValueError(f"Unexpected weights config {config!r}")
        module_index = int(row["module_index"])
        if module_index in weight_by_config[config]:
            raise ValueError(f"Duplicate weight row for {config}, module_index={module_index}")
        weight_by_config[config][module_index] = row
    for config, rows in weight_by_config.items():
        if len(rows) != 614:
            raise ValueError(f"Expected 614 weight rows for {config}, found {len(rows)}")
    module_ids = set(weight_by_config[WEIGHT_CONFIGS[0]])
    if any(set(rows) != module_ids for rows in weight_by_config.values()):
        raise ValueError("Weight module_index coverage differs between Smooth configurations")
    selected_indices = [int(row["module_index"]) for row in selected]
    if len(set(selected_indices)) != 6 or not set(selected_indices) <= module_ids:
        raise ValueError("selected.json module indices must be six distinct members of the 614 layers")

    print("[plot] checking activation coverage and aggregating call energies", flush=True)
    activation_fields = ("case_id", "seed_index", "sampler_seed", "stage", "step", "call_index", "module",
                         "module_index", "stream", "rows", "input_shape", "config", "activation_rrmse",
                         "zero_fraction", "absmax", "rms", "outlier_ratio", "error_squared", "signal_squared",
                         "zero_count")
    observed_coverage = set()
    activation_rows = []
    activation_max_error_call = {}
    for row_index, row in enumerate(activations):
        require_fields(row, activation_fields, f"activations.jsonl row {row_index + 1}")
        if row["config"] not in CONFIGS:
            raise ValueError(f"Unexpected activation config {row['config']!r}")
        if int(row["case_id"]) not in CASES:
            continue
        if int(row["seed_index"]) != 0:
            continue
        activation_rows.append(row)
        if int(row["step"]) >= 0:
            if int(row["step"]) > 9:
                raise ValueError(f"Unexpected scheduler step {row['step']} in activation record {row_index + 1}")
            require_fields(row, ("error_absmax", "error_peak_position"),
                           f"scheduler activation record {row_index + 1}")
            observed_coverage.add((int(row["case_id"]), row["stage"], int(row["step"]), row["config"]))
            activation_key = (int(row["case_id"]), row["stage"], int(row["step"]),
                              int(row["module_index"]), row["config"])
            peak_value = finite(row["error_absmax"], "error_absmax")
            if activation_key not in activation_max_error_call or peak_value > activation_max_error_call[activation_key][0]:
                activation_max_error_call[activation_key] = (peak_value, row)
    if not activation_rows:
        raise ValueError("No activation rows found for cases 1, 13, 21 at seed_index=0")
    scheduler_stages = sorted({row["stage"] for row in activation_rows if int(row["step"]) >= 0})
    scheduler_rows = [row for row in activation_rows if int(row["step"]) >= 0]
    modules_for_stage = defaultdict(set)
    modules_for_step = defaultdict(set)
    for row in scheduler_rows:
        case_id, stage, config = int(row["case_id"]), row["stage"], row["config"]
        module_index, step = int(row["module_index"]), int(row["step"])
        modules_for_stage[case_id, stage].add(module_index)
        modules_for_step[case_id, stage, config, step].add(module_index)
    missing = []
    for case_id in CASES:
        for stage in scheduler_stages:
            for config in CONFIGS:
                expected_modules = modules_for_stage.get((case_id, stage), set())
                for step in range(10):
                    if (case_id, stage, step, config) not in observed_coverage:
                        missing.append({"case_id": case_id, "stage": stage, "step": step, "config": config})
                        continue
                    actual_modules = modules_for_step.get((case_id, stage, config, step), set())
                    for module_index in sorted(expected_modules - actual_modules):
                        missing.append({"case_id": case_id, "stage": stage, "step": step,
                                        "config": config, "module_index": module_index})
    summary["missing_coverage_assert"] = {"passed": not missing, "missing": missing,
        "required": {"cases": list(CASES), "seed_index": 0, "configs": list(CONFIGS),
                     "scheduler_steps": list(range(10)), "stages": scheduler_stages,
                     "conditioning_step_minus_one": "excluded from scheduler heatmaps"}}
    if missing:
        raise ValueError(f"Activation scheduler coverage is incomplete ({len(missing)} missing case/stage/step/config cells)")
    act_group = aggregate_energy(scheduler_rows,
                                 ("case_id", "stage", "step", "module_index", "config"))
    modules_by_stage = {}
    for stage in scheduler_stages:
        modules_by_stage[stage] = sorted({int(row["module_index"]) for row in scheduler_rows
                                          if row["stage"] == stage and int(row["step"]) >= 0})
    summary["counts"].update({"activation_target_rows": len(activation_rows),
                              "activation_scheduler_rows": len(scheduler_rows),
                              "activation_conditioning_rows_excluded": sum(int(row["step"]) == -1 for row in activation_rows),
                              "scheduler_stages": scheduler_stages,
                              "aggregated_activation_cells": len(act_group)})

    print("[plot] loading bounded selected snapshots", flush=True)
    snapshots_dir = data_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(f"Required snapshot directory is missing: {snapshots_dir}")
    weight_arrays = {}
    for config in WEIGHT_CONFIGS:
        for selected_row in selected:
            module_index = int(selected_row["module_index"])
            path = snapshots_dir / f"weight_{module_index:04d}_{config}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing selected weight snapshot: {path}")
            with np.load(path, allow_pickle=False) as archive:
                needed = ("weight_absmax", "scalar_error_absmax", "vq_error_absmax", "row_rms",
                          "row_scalar_rrmse", "row_vq_rrmse", "scalar_column_energy", "vq_column_energy")
                missing_arrays = [name for name in needed if name not in archive]
                if missing_arrays:
                    raise ValueError(f"{path} is missing arrays: {', '.join(missing_arrays)}")
                arrays = {name: np.asarray(archive[name], dtype=np.float32).copy() for name in needed}
                if not all(np.all(np.isfinite(array)) for array in arrays.values()):
                    raise ValueError(f"Non-finite values in selected weight snapshot: {path}")
                if any(arrays[name].ndim != 2 or max(arrays[name].shape) > 64
                       for name in ("weight_absmax", "scalar_error_absmax", "vq_error_absmax")):
                    raise ValueError(f"Expected <=64x64 pooled maps in {path}")
                if arrays["scalar_error_absmax"].shape != arrays["vq_error_absmax"].shape:
                    raise ValueError(f"Scalar/VQ error map shape mismatch in {path}")
                weight_arrays[config, module_index] = arrays

    # Build a compact set of matched before/after/Q4 examples from existing snapshots.
    sample_targets = {1: 4, 13: 9, 21: 9}
    activation_examples = []
    chosen_activation_module = None
    chosen_stage_by_case = {}
    for selected_row in selected:
        module_index = int(selected_row["module_index"])
        candidate = []
        for case_id, step in sample_targets.items():
            common = None
            paths_by_stage = {}
            for config in CONFIGS:
                matches = list(snapshots_dir.glob(
                    f"act_c{case_id}_m{module_index:04d}_{config}_*_s{step}_call*.npz"))
                by_stage = defaultdict(list)
                for path in matches:
                    name = path.name
                    prefix = f"act_c{case_id}_m{module_index:04d}_{config}_"
                    suffix = f"_s{step}_call"
                    if not name.startswith(prefix) or suffix not in name:
                        continue
                    stage = name[len(prefix):name.index(suffix)]
                    call_token = name[name.index(suffix) + len(suffix):-4]
                    try:
                        call_index = int(call_token)
                    except ValueError:
                        continue
                    by_stage[stage].append((call_index, path))
                paths_by_stage[config] = by_stage
            common_stages = set.intersection(*(set(paths_by_stage[c]) for c in CONFIGS)) if CONFIGS else set()
            if not common_stages:
                candidate = []
                break
            stage = sorted(common_stages)[0]
            chosen_stage_by_case[case_id] = stage
            files = {}
            for config in CONFIGS:
                files[config] = min(paths_by_stage[config][stage], key=lambda item: item[0])[1]
            candidate.append({"case_id": case_id, "step": step, "stage": stage, "files": files})
        if candidate and len(candidate) == len(sample_targets):
            chosen_activation_module = module_index
            activation_examples = candidate
            break
    if chosen_activation_module is None:
        raise ValueError("No selected module has matched activation snapshots for cases 1/13/21, steps 4/9, and all four configs")
    activation_snapshots = {}
    for example in activation_examples:
        for config, path in example["files"].items():
            with np.load(path, allow_pickle=False) as archive:
                needed = ("original_absmax", "transformed_absmax", "quantized_absmax", "error_absmax")
                missing_arrays = [name for name in needed if name not in archive]
                if missing_arrays:
                    raise ValueError(f"{path} is missing arrays: {', '.join(missing_arrays)}")
                arrays = {name: np.asarray(archive[name], dtype=np.float32).copy() for name in needed}
                if not all(np.all(np.isfinite(array)) for array in arrays.values()):
                    raise ValueError(f"Non-finite values in selected activation snapshot: {path}")
                if any(array.ndim != 2 or max(array.shape) > 64 for array in arrays.values()):
                    raise ValueError(f"Expected <=64x64 pooled activation maps in {path}")
                activation_snapshots[example["case_id"], config] = arrays
                activation_snapshots[example["case_id"], config]["path"] = str(path.relative_to(data_dir))

    print("[plot] matching raw actions to BF16 references", flush=True)
    action_map = {}
    for row_index, row in enumerate(actions):
        require_fields(row, ("case_id", "seed_index", "arm", "action"), f"actions.jsonl row {row_index + 1}")
        key = (int(row["case_id"]), int(row["seed_index"]), row["arm"])
        if key in action_map:
            raise ValueError(f"Duplicate action record for {key}")
        action_map[key] = row["action"]
    action_pairs = {}
    action_configs = (("smooth05_hadamard_scalar_a4", "S0.5H Scalar W4A4"),
                      ("smooth05_hadamard_vq_a4", "S0.5H additive VQ+A4"),
                      ("smooth1_hadamard_scalar_a4", "S1H Scalar W4A4"),
                      ("smooth1_hadamard_vq_a4", "S1H additive VQ+A4"))
    for case_id in CASES:
        reference = action_map.get((case_id, 0, "bf16"))
        if reference is None:
            raise ValueError(f"Missing BF16 action reference for case={case_id}, seed_index=0")
        reference = np.asarray(reference, dtype=np.float32)
        for arm, label in action_configs:
            actual = action_map.get((case_id, 0, arm))
            if actual is None:
                raise ValueError(f"Missing action record for case={case_id}, seed_index=0, arm={arm}")
            actual = np.asarray(actual, dtype=np.float32)
            if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] < 7:
                raise ValueError(f"Action shape mismatch/unsupported for case={case_id}, arm={arm}: {actual.shape}")
            action_pairs[case_id, arm] = np.abs(actual - reference)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "figure.titlesize": 10, "savefig.facecolor": "white"})
    figure_paths = []

    def save_figure(fig, name: str) -> None:
        target = out_dir / name
        render_info = None
        for dpi in (180, 160, 140, 120, 100, 90, 75, 60, 48):
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight",
                        metadata={"Software": "plot_visualization.py"})
            buffer.seek(0)
            with Image.open(buffer) as image:
                dimensions = image.size
                palette_image = image.convert("RGB").quantize(
                    colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                palette_image.save(target, format="PNG", optimize=True, dpi=(dpi, dpi))
            if target.stat().st_size <= MAX_PNG_BYTES:
                render_info = {"file": name, "dpi": dpi, "width_px": dimensions[0],
                               "height_px": dimensions[1], "bytes": target.stat().st_size,
                               "palette_colors": 256, "dither": "none"}
                break
        plt.close(fig)
        if render_info is None:
            target.unlink(missing_ok=True)
            raise ValueError(f"PNG exceeded 220 KiB after palette compression at 48 dpi: {name}")
        figure_paths.append(str(target.relative_to(out_dir)))
        summary["figure_render_info"].append(render_info)

    def stage_update(stage: str) -> None:
        summary["stage"] = stage
        write_json(out_dir / "summary.json", summary)
        print(f"[plot] {stage}", flush=True)

    # 1. Per-layer error curves make selected module indices directly locatable.
    stage_update("weight_error_overview")
    stream_names = ("video_expert", "action_expert", "proprio_encoder")
    stream_labels = ("Video", "Action", "Proprio")
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharey="row")
    for row_index, (stream, stream_label) in enumerate(zip(stream_names, stream_labels)):
        for col_index, config in enumerate(WEIGHT_CONFIGS):
            axis = axes[row_index, col_index]
            module_rows = sorted((weight_by_config[config][idx] for idx in module_ids
                                  if stream_bucket(weight_by_config[config][idx]) == stream),
                                 key=lambda row: int(row["module_index"]))
            if not module_rows:
                raise ValueError(f"No weight rows found for stream {stream}")
            x = [int(row["module_index"]) for row in module_rows]
            for method, label, color, field in (("scalar", "Scalar W4", "#2563eb", "scalar_rrmse"),
                                                ("vq", "Additive VQ", "#ea580c", "vq_rrmse")):
                y = [finite(row[field], field) for row in module_rows]
                axis.plot(x, y, label=label, color=color, linewidth=.75, alpha=.85)
                selected_rows = [row for row in module_rows if int(row["module_index"]) in selected_indices]
                if selected_rows:
                    selected_x = [int(row["module_index"]) for row in selected_rows]
                    selected_y = [finite(row[field], field) for row in selected_rows]
                    if stream == "proprio_encoder":
                        axis.scatter(selected_x, selected_y, s=18, color=color, zorder=5)
                    axis.scatter(selected_x, selected_y, s=42, facecolors="none", edgecolors="black",
                                 linewidths=1.0, zorder=4)
                    for sx, sy in zip(selected_x, selected_y):
                        axis.annotate(str(sx), (sx, sy), xytext=(2, 3), textcoords="offset points",
                                      fontsize=7, color="black")
            axis.set_yscale("symlog", linthresh=1e-5)
            axis.grid(alpha=.25)
            axis.set_title(f"{stream_label} · {config.replace('_hadamard', 'H')}")
            axis.set_xlabel("module_index")
            axis.text(.98, .95, f"n={len(module_rows)} layers" if len(module_rows) > 1 else "single layer",
                      transform=axis.transAxes, ha="right", va="top", fontsize=7)
            if col_index == 0:
                axis.set_ylabel("Per-layer normalized weight RMSE")
            if row_index == 0 and col_index == 0:
                axis.scatter([], [], s=36, facecolors="none", edgecolors="black", label="Selected layer")
                axis.legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Full-layer normalized weight error · each layer weighted equally")
    fig.tight_layout()
    save_figure(fig, "01_weight_rrmse_by_stream.png")

    # 2. Paired case × stage activation heatmaps; no ratio-of-call-ratios averaging.
    stage_update("activation_layer_step_heatmaps")
    stage_names = scheduler_stages
    fig, axes = plt.subplots(len(CASES) * len(stage_names), len(CONFIGS),
                             figsize=(12, max(8, 1.5 * len(CASES) * len(stage_names))), squeeze=False)
    all_errors = [bucket["rrmse"] for bucket in act_group.values()]
    vmax = max(all_errors) if all_errors else 1.0
    vmax = max(vmax, 1e-8)
    bins = np.linspace(0.0, vmax, 65)
    heat_cmap = ListedColormap(plt.get_cmap("magma")(np.linspace(0, 1, 64)))
    cmap64 = {name: ListedColormap(plt.get_cmap(name)(np.linspace(0, 1, 64)))
              for name in ("viridis", "magma")}
    heat_norm = BoundaryNorm(bins, heat_cmap.N, clip=True)
    for case_row, case_id in enumerate(CASES):
        for stage_index, stage in enumerate(stage_names):
            axis_row = case_row * len(stage_names) + stage_index
            indices = modules_by_stage[stage]
            for config_index, config in enumerate(CONFIGS):
                axis = axes[axis_row, config_index]
                grid = np.full((len(indices), 10), np.nan, dtype=np.float32)
                row_index_by_module = {module_index: i for i, module_index in enumerate(indices)}
                for step in range(10):
                    for module_index, target_row in row_index_by_module.items():
                        bucket = act_group.get((case_id, stage, step, module_index, config))
                        if bucket is not None:
                            grid[target_row, step] = bucket["rrmse"]
                axis.imshow(np.ma.masked_invalid(grid), aspect="auto", interpolation="nearest",
                            cmap=heat_cmap, norm=heat_norm, origin="upper")
                axis.set_title(f"case {case_id} · {stage} · {config}")
                axis.set_xticks(range(10), labels=range(10))
                if config_index == 0:
                    axis.set_ylabel(f"{len(indices)} layers\n(layer index order)")
                else:
                    axis.set_yticks([])
                if axis_row == len(CASES) * len(stage_names) - 1:
                    axis.set_xlabel("Scheduler step")
    fig.suptitle("Activation relative RMSE · calls pooled by energy · conditioning step −1 excluded\n"
                 "One fixed color scale across all cases, stages, and configurations")
    fig.subplots_adjust(right=.91, top=.92, hspace=.6, wspace=.18)
    color_axis = fig.add_axes([.93, .20, .015, .62])
    scalar = plt.cm.ScalarMappable(norm=heat_norm, cmap=heat_cmap)
    fig.colorbar(scalar, cax=color_axis, label="sqrt(sum(error²) / sum(signal²))")
    save_figure(fig, "02_activation_layer_step_heatmaps.png")

    # 3. Sum local-output energies from the per-call activation records.
    stage_update("local_output_error_overview")
    local_error_energies = ("weight_error_energy", "activation_error_energy", "cross_error_energy", "total_error_energy")
    local_groups = {}
    for row in activation_rows:
        config = row["config"]
        if config not in WEIGHT_CONFIGS:
            continue
        local = row.get("local_output")
        if not isinstance(local, dict):
            raise ValueError(f"Missing per-call local_output for {config}, module={row['module']}")
        for method in ("scalar", "vq"):
            if method not in local or not isinstance(local[method], dict):
                raise ValueError(f"Missing local_output.{method} for {config}, module={row['module']}")
            entry = local[method]
            fields = ("reference_output_energy", *local_error_energies)
            require_fields(entry, fields, f"local_output.{method} in activation record")
            key = (config, int(row["module_index"]), method)
            bucket = local_groups.setdefault(key, {"reference_output_energy": 0.0, "calls": 0,
                                                   **{field: 0.0 for field in local_error_energies}})
            for field in fields:
                value = finite(entry[field], f"local_output.{method}.{field}")
                if value < 0:
                    raise ValueError(f"Energy must be non-negative: {field}={value}")
                bucket[field] += value
            bucket["calls"] += 1
    local_records = {}
    local_rrmse_fields = {"weight_error_energy": "weight_rrmse", "activation_error_energy": "activation_rrmse",
                          "cross_error_energy": "cross_rrmse", "total_error_energy": "total_rrmse"}
    for key, bucket in local_groups.items():
        reference_energy = bucket["reference_output_energy"]
        if reference_energy <= 0:
            raise ValueError(f"Non-positive reference_output_energy for local-output group {key}")
        bucket.update({metric: math.sqrt(bucket[energy] / reference_energy)
                       for energy, metric in local_rrmse_fields.items()})
        local_records[key] = bucket
    expected_local = {(config, index, method) for config in WEIGHT_CONFIGS for index in module_ids
                      for method in ("scalar", "vq")}
    if set(local_records) != expected_local:
        missing_local = sorted(expected_local - set(local_records))[:20]
        extra_local = sorted(set(local_records) - expected_local)[:20]
        raise ValueError(f"Incomplete local-output layer coverage; missing={missing_local}, extra={extra_local}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    local_metric_fields = ("weight_rrmse", "activation_rrmse", "cross_rrmse", "total_rrmse")
    config_colors = {"smooth05_hadamard": "#2563eb", "smooth1_hadamard": "#ea580c"}
    for axis, metric in zip(axes.flat, local_metric_fields):
        for config in WEIGHT_CONFIGS:
            for method, linestyle in (("scalar", "-"), ("vq", "--")):
                indices = sorted(module_ids)
                values = [local_records[config, index, method][metric] for index in indices]
                line, = axis.plot(indices, values, color=config_colors[config], linestyle=linestyle,
                                  linewidth=.8, alpha=.85,
                                  label=f"{config.replace('_hadamard', 'H')} {method}")
                selected_x = [index for index in selected_indices if index in module_ids]
                selected_y = [local_records[config, index, method][metric] for index in selected_x]
                axis.scatter(selected_x, selected_y, s=26, facecolors="none", edgecolors="black",
                             linewidths=.8, zorder=4)
        axis.set_title(metric.replace("_", " "))
        axis.set_ylabel("Aggregated local normalized RMSE")
        axis.set_xlabel("module_index")
        axis.set_yscale("symlog", linthresh=1e-5)
        axis.grid(alpha=.25)
        axis.scatter([], [], s=26, facecolors="none", edgecolors="black", label="Selected layer")
    axes[0, 0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("Full-layer local output proxy · per-call energies summed before normalization\n"
                 "Sampled 16-row full-covariance projection; components: weight, activation, cross, total")
    fig.tight_layout()
    save_figure(fig, "03_local_output_error_components.png")

    # 4. Six selected matrices, max-pooled to <=64x64; paired per-row scales.
    stage_update("selected_weight_heatmaps")
    weight_norms, weight_error_norms = {}, {}
    for selected_row in selected:
        module_index = int(selected_row["module_index"])
        paired_arrays = [weight_arrays[config, module_index] for config in WEIGHT_CONFIGS]
        weight_vmax = max(max(float(np.max(arrays["weight_absmax"])) for arrays in paired_arrays), 1e-12)
        error_vmax = max(max(float(np.max(arrays[key])) for arrays in paired_arrays
                             for key in ("scalar_error_absmax", "vq_error_absmax")), 1e-12)
        weight_norms[module_index] = PowerNorm(gamma=.5, vmin=0.0, vmax=weight_vmax)
        weight_error_norms[module_index] = PowerNorm(gamma=.5, vmin=0.0, vmax=error_vmax)

    def short_module_name(module: str) -> str:
        parts = module.split(".")
        block = next((part for part in parts if part.lower().startswith("block") and
                      part[5:].isdigit()), None)
        if block is None:
            block = next((f"block{parts[i + 1]}" for i, part in enumerate(parts[:-1])
                          if part.lower() in ("blocks", "layers") and parts[i + 1].isdigit()), None)
        short = ".".join(parts[-2:])
        return f"{block} {short}" if block and not short.lower().startswith(block.lower()) else short

    for config in WEIGHT_CONFIGS:
        fig, axes = plt.subplots(6, 3, figsize=(12, 10), squeeze=False)
        fig.subplots_adjust(left=.24, right=.88, top=.91, bottom=.05, wspace=.20, hspace=.34)
        for row_index, selected_row in enumerate(selected):
            module_index = int(selected_row["module_index"])
            arrays = weight_arrays[config, module_index]
            weight_map = arrays["weight_absmax"]
            scalar_map, vq_map = arrays["scalar_error_absmax"], arrays["vq_error_absmax"]
            if weight_map.ndim != 2 or scalar_map.ndim != 2 or vq_map.ndim != 2:
                raise ValueError(f"Expected 2D pooled weight maps for module_index={module_index}, config={config}")
            images = []
            for column, (matrix, title, cmap_name) in enumerate(((weight_map, "B absmax", "viridis"),
                                                                 (scalar_map, "Scalar error absmax", "magma"),
                                                                 (vq_map, "VQ error absmax", "magma"))):
                axis = axes[row_index, column]
                norm = weight_norms[module_index] if column == 0 else weight_error_norms[module_index]
                images.append(axis.imshow(matrix, aspect="auto", interpolation="nearest",
                                          cmap=cmap64[cmap_name], norm=norm))
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(title)
                if column == 0:
                    axis.set_ylabel(f"{module_index} · {selected_row['stream']}\n"
                                    f"{short_module_name(str(selected_row['module']))}", fontsize=8)
            weight_bar = fig.colorbar(images[0], ax=axes[row_index, 0], fraction=.035, pad=.025)
            error_bar = fig.colorbar(images[1], ax=axes[row_index, 1:3], fraction=.025, pad=.035)
            weight_bar.ax.tick_params(labelsize=6)
            error_bar.ax.tick_params(labelsize=6)
        fig.suptitle(f"Selected layer weight maps · {config}\n"
                     "Block-max pooled to ≤64×64. Each row has its own scales; the same module shares scales across α. "
                     "Do not compare absolute color values across rows.")
        save_figure(fig, f"04_selected_weight_maps_{config}.png")

    # 5. Full selected-row amplitude/error relation; draw deterministically bounded points.
    stage_update("selected_row_amplitude_scatter")
    max_points = 4000
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    scatter_display = {}
    for config_index, config in enumerate(WEIGHT_CONFIGS):
        for method_index, method in enumerate(("scalar", "vq")):
            axis = axes[config_index, method_index]
            amplitudes, errors = [], []
            for selected_row in selected:
                arrays = weight_arrays[config, int(selected_row["module_index"])]
                amp = np.asarray(arrays["row_rms"], dtype=np.float64).reshape(-1)
                err_key = "row_scalar_rrmse" if method == "scalar" else "row_vq_rrmse"
                err = np.asarray(arrays[err_key], dtype=np.float64).reshape(-1)
                if amp.shape != err.shape:
                    raise ValueError(f"Row arrays differ for {config}/{method}/module={selected_row['module_index']}")
                if not np.all(np.isfinite(amp)) or not np.all(np.isfinite(err)):
                    raise ValueError(f"Non-finite row values for {config}/{method}/module={selected_row['module_index']}")
                amplitudes.extend(amp.tolist())
                errors.extend(err.tolist())
            x, y = np.asarray(amplitudes), np.asarray(errors)
            order = np.argsort(x, kind="stable")
            take = np.linspace(0, max(len(order) - 1, 0), min(len(order), max_points), dtype=np.int64)
            shown = order[take] if len(order) else order
            axis.scatter(x[shown], y[shown], s=2, alpha=.28, rasterized=True)
            axis.set_xscale("symlog", linthresh=1e-6)
            axis.set_yscale("symlog", linthresh=1e-6)
            axis.set_title(f"{config} · {method} · plotted {len(shown):,}/{len(x):,} rows")
            axis.set_xlabel("BF16 row RMS")
            axis.set_ylabel("Row relative RMSE")
            axis.grid(alpha=.2)
            scatter_display[f"{config}/{method}"] = {
                "full_rows_loaded_from_selected_npz": int(len(x)), "points_plotted": int(len(shown)),
                "point_cap": max_points,
                "selected_module_indices": selected_indices,
                "source_snapshots": [f"snapshots/weight_{int(row['module_index']):04d}_{config}.npz" for row in selected]}
    fig.suptitle(f"Selected layer row amplitude vs relative error · deterministic sorted downsample, at most {max_points} points/panel")
    fig.tight_layout()
    save_figure(fig, "05_selected_row_amplitude_vs_error.png")
    summary["row_scatter_display"] = scatter_display
    row_amplitude_stats = []
    for config in WEIGHT_CONFIGS:
        for selected_row in selected:
            module_index = int(selected_row["module_index"])
            arrays = weight_arrays[config, module_index]
            amplitude = np.asarray(arrays["row_rms"], dtype=np.float64).reshape(-1)
            for method, error_key in (("scalar", "row_scalar_rrmse"), ("vq", "row_vq_rrmse")):
                relative_error = np.asarray(arrays[error_key], dtype=np.float64).reshape(-1)
                valid = np.isfinite(amplitude) & np.isfinite(relative_error)
                amplitude, relative_error = amplitude[valid], relative_error[valid]
                if not len(amplitude):
                    raise ValueError(f"No valid row samples for {config}/{module_index}/{method}")
                q25, q75 = np.quantile(amplitude, (0.25, 0.75))
                low = relative_error[amplitude <= q25]
                high = relative_error[amplitude >= q75]
                positive = (amplitude > 0) & (relative_error > 0) & np.isfinite(amplitude) & np.isfinite(relative_error)
                log_amp, log_error = np.log(amplitude[positive]), np.log(relative_error[positive])
                correlation = None
                if len(log_amp) >= 3 and np.ptp(log_amp) > 0 and np.ptp(log_error) > 0:
                    correlation = float(np.corrcoef(log_amp, log_error)[0, 1])
                    if not math.isfinite(correlation):
                        correlation = None
                row_amplitude_stats.append({
                    "config": config, "method": method, "module": selected_row["module"],
                    "module_index": module_index, "stream": selected_row["stream"],
                    "source_npz": f"snapshots/weight_{module_index:04d}_{config}.npz",
                    "valid_row_count": int(len(amplitude)),
                    "low_amplitude_q25": float(q25), "low_amplitude_rows": int(len(low)),
                    "low_amplitude_mean_relative_error": float(np.mean(low)) if len(low) else None,
                    "high_amplitude_q75": float(q75), "high_amplitude_rows": int(len(high)),
                    "high_amplitude_mean_relative_error": float(np.mean(high)) if len(high) else None,
                    "positive_log_pair_count": int(np.sum(positive)),
                    "pearson_log_amplitude_log_relative_error": correlation})
    summary["selected_row_amplitude_stats"] = row_amplitude_stats

    # Aggregate per-call channel statistics for all six selected modules.
    stage_update("selected_activation_channel_concentration")
    selected_scheduler_rows = [row for row in scheduler_rows
                               if int(row["module_index"]) in selected_indices and 0 <= int(row["step"]) <= 9]
    conditioning_excluded_calls = sum(int(row["module_index"]) in selected_indices and int(row["step"]) == -1
                                      for row in activation_rows)
    channel_groups = {}
    seen_channel_calls = set()
    channel_stats_file_reads = 0
    activation_module_info = {}
    for row in selected_scheduler_rows:
        require_fields(row, ("channel_stats_file", "call_index"), "selected scheduler activation record")
        call_key = (int(row["case_id"]), row["config"], int(row["module_index"]), row["stage"],
                    int(row["step"]), int(row["call_index"]))
        if call_key in seen_channel_calls:
            raise ValueError(f"Duplicate selected activation call record: {call_key}")
        seen_channel_calls.add(call_key)
        path = resolve_data_file(data_dir, snapshots_dir, row["channel_stats_file"])
        row_count = int(row["rows"])
        if row_count <= 0:
            raise ValueError(f"Invalid activation row count in {call_key}: {row_count}")
        with np.load(path, allow_pickle=False) as archive:
            needed = ("channel_error_energy", "channel_rms")
            absent = [field for field in needed if field not in archive]
            if absent:
                raise ValueError(f"{path} is missing arrays: {', '.join(absent)}")
            error_energy = np.asarray(archive["channel_error_energy"], dtype=np.float64).reshape(-1)
            channel_rms = np.asarray(archive["channel_rms"], dtype=np.float64).reshape(-1)
        if not len(error_energy) or error_energy.shape != channel_rms.shape:
            raise ValueError(f"Channel array shape mismatch in {path}")
        if not np.all(np.isfinite(error_energy)) or not np.all(np.isfinite(channel_rms)):
            raise ValueError(f"Non-finite channel statistics in {path}")
        if np.any(error_energy < 0) or np.any(channel_rms < 0):
            raise ValueError(f"Negative channel energy/RMS in {path}")
        signal_energy = channel_rms ** 2 * row_count
        group_key = (int(row["case_id"]), row["config"], int(row["module_index"]))
        bucket = channel_groups.setdefault(group_key, {"channel_error_energy": np.zeros_like(error_energy),
            "channel_signal_energy": np.zeros_like(signal_energy), "rows": 0, "calls": 0,
            "steps": set(), "source_files": 0})
        if bucket["channel_error_energy"].shape != error_energy.shape:
            raise ValueError(f"Channel count changed within activation group {group_key}")
        bucket["channel_error_energy"] += error_energy
        bucket["channel_signal_energy"] += signal_energy
        bucket["rows"] += row_count
        bucket["calls"] += 1
        bucket["steps"].add(int(row["step"]))
        bucket["source_files"] += 1
        channel_stats_file_reads += 1
        activation_module_info.setdefault(int(row["module_index"]),
            {"module": row["module"], "stream": row["stream"]})
    expected_channel_groups, channel_group_exclusions = activation_channel_expected_groups(
        data_summary.get("selected_module_actual_stage_step_coverage_per_query"), selected, CONFIGS)
    if set(channel_groups) != expected_channel_groups:
        missing_channel_groups = sorted(expected_channel_groups - set(channel_groups))[:20]
        unexpected_channel_groups = sorted(set(channel_groups) - expected_channel_groups)[:20]
        raise ValueError(f"Selected activation channel stats do not match observed scheduler coverage: "
                         f"missing={missing_channel_groups}, unexpected={unexpected_channel_groups}")
    channel_concentration = []
    channel_hotspots = []
    for (case_id, config, module_index), bucket in sorted(channel_groups.items()):
        energy = bucket["channel_error_energy"]
        total = float(np.sum(energy))
        order = np.argsort(energy, kind="stable")[::-1]
        channel_count = len(energy)
        top1_count = max(1, int(math.ceil(channel_count * .01)))
        top16_count = min(16, channel_count)
        top4_indices = order[:min(4, channel_count)]
        concentration = {"case_id": case_id, "config": config, "module_index": module_index,
            "module": activation_module_info[module_index]["module"],
            "stream": activation_module_info[module_index]["stream"], "channel_count": channel_count,
            "top1pct_channel_count": top1_count,
            "top1pct_error_energy_share": float(np.sum(energy[order[:top1_count]]) / total) if total > 0 else None,
            "top16_error_energy_share": float(np.sum(energy[order[:top16_count]]) / total) if total > 0 else None,
            "largestchannel": int(order[0]), "largestchannel_error_energy": float(energy[order[0]]),
            "top4indices": [int(index) for index in top4_indices],
            "top4_error_energies": [float(energy[index]) for index in top4_indices],
            "total_channel_error_energy": total,
            "total_channel_signal_energy": float(np.sum(bucket["channel_signal_energy"])),
            "rows": bucket["rows"], "calls": bucket["calls"],
            "scheduler_steps": sorted(bucket["steps"]), "conditioning_included": False,
            "channel_stats_files_read": bucket["source_files"]}
        channel_concentration.append(concentration)
        for rank, channel in enumerate(top4_indices, 1):
            channel_hotspots.append({"rank": rank, "case_id": case_id, "config": config,
                "module": concentration["module"], "module_index": module_index, "stream": concentration["stream"],
                "channel": int(channel), "channel_error_energy": float(energy[channel]),
                "channel_error_energy_share": float(energy[channel] / total) if total > 0 else None,
                "group_error_energy_total": total, "calls": bucket["calls"], "rows": bucket["rows"]})
    summary["activation_channel_concentration"] = channel_concentration
    summary["activation_channel_concentration_scope"] = {
        "steps": "scheduler steps 0..9; conditioning step -1 excluded",
        "expected_groups_basis": "collector observed per-module stage/step calls per query; video/action steps 0..9",
        "excluded_case_module_groups": channel_group_exclusions,
        "conditioning_only_exclusions": [item for item in channel_group_exclusions
                                         if item["classification"] == "conditioning_only"],
        "conditioning_selected_calls_excluded": int(conditioning_excluded_calls),
        "channel_signal_energy_formula": "sum(channel_rms ** 2 * record.rows)",
        "top1pct_rule": "ceil(1 percent of channel_count), at least one channel",
        "top16_rule": "up to 16 channels", "npz_files_read": channel_stats_file_reads}

    # 6. One readable 4-config × 4-map activation figure per case.
    stage_update("selected_activation_examples")
    absmax_values = [activation_snapshots[case_id, config][key]
                     for case_id in CASES for config in CONFIGS
                     for key in ("original_absmax", "transformed_absmax", "quantized_absmax")]
    error_values = [activation_snapshots[case_id, config]["error_absmax"]
                    for case_id in CASES for config in CONFIGS]
    absmax_vmax = max(max(float(np.max(arr)) for arr in absmax_values), 1e-12)
    error_vmax = max(max(float(np.max(arr)) for arr in error_values), 1e-12)
    heat_titles = ("Original X", "Transformed Z", "Quantized QZ", "Error |QZ−Z|")
    snapshot_keys = ("original_absmax", "transformed_absmax", "quantized_absmax", "error_absmax")
    for example in activation_examples:
        case_id = example["case_id"]
        fig, axes = plt.subplots(len(CONFIGS), 4, figsize=(13, 8), squeeze=False)
        for config_index, config in enumerate(CONFIGS):
            arrays = activation_snapshots[case_id, config]
            for column, (key, title) in enumerate(zip(snapshot_keys, heat_titles)):
                axis = axes[config_index, column]
                matrix = arrays[key]
                if matrix.ndim != 2:
                    raise ValueError(f"Expected 2D pooled activation map: case={case_id}, config={config}, key={key}")
                is_error = key == "error_absmax"
                axis.imshow(matrix, aspect="auto", interpolation="nearest",
                            cmap=cmap64["magma" if is_error else "viridis"],
                            norm=PowerNorm(gamma=.5, vmin=0.0, vmax=error_vmax if is_error else absmax_vmax))
                axis.set_xticks([])
                axis.set_yticks([])
                if config_index == 0:
                    axis.set_title(title, fontsize=9)
                if column == 0:
                    axis.set_ylabel(config, fontsize=9)
        fig.suptitle(f"case {case_id} · {example['stage']} step {example['step']} · module_index {chosen_activation_module}\n"
                     "Block-max pooled to ≤64×64; X/Z/QZ and error each use one scale shared across all cases/configs. Channel bins are not matched.")
        fig.subplots_adjust(left=.08, right=.87, top=.89, bottom=.07, hspace=.24, wspace=.22)
        signal_color_axis = fig.add_axes([.90, .15, .018, .68])
        error_color_axis = fig.add_axes([.95, .15, .018, .68])
        signal_bar = fig.colorbar(plt.cm.ScalarMappable(
            norm=PowerNorm(gamma=.5, vmin=0.0, vmax=absmax_vmax), cmap=cmap64["viridis"]),
            cax=signal_color_axis, label="Signal absmax · shared X/Z/QZ")
        error_bar = fig.colorbar(plt.cm.ScalarMappable(
            norm=PowerNorm(gamma=.5, vmin=0.0, vmax=error_vmax), cmap=cmap64["magma"]),
            cax=error_color_axis, label="Error absmax · shared")
        signal_bar.ax.tick_params(labelsize=7)
        error_bar.ax.tick_params(labelsize=7)
        save_figure(fig, f"06_selected_activation_case{case_id}.png")

    # 7. Fixed-input action differences for seed 0, paired to the original BF16 action.
    stage_update("raw_action_error_maps")
    action_errors = [action_pairs[case_id, arm] for case_id in CASES for arm, _ in action_configs]
    action_vmax = max(max(float(np.max(arr)) for arr in action_errors), 1e-12)
    fig, axes = plt.subplots(len(CASES), len(action_configs), figsize=(13, 7), squeeze=False)
    for case_row, case_id in enumerate(CASES):
        for config_index, (arm, label) in enumerate(action_configs):
            axis = axes[case_row, config_index]
            matrix = action_pairs[case_id, arm]
            axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap64["magma"],
                        norm=PowerNorm(gamma=.5, vmin=0.0, vmax=action_vmax), origin="upper")
            axis.axvline(5.5, color="cyan", linewidth=.7)
            axis.axhline(9.5, color="cyan", linewidth=.8)
            axis.set_xticks(range(7), labels=("0", "1", "2", "3", "4", "5", "G"))
            axis.set_yticks((0, 9, 19, 29), labels=("0", "9", "19", "29"))
            if case_row == 0:
                axis.set_title(label, fontsize=9)
            if config_index == 0:
                axis.set_ylabel(f"case {case_id}\naction step")
            if config_index == 0 and case_row == 0:
                axis.text(.02, 9.5, "first 10: primary", transform=axis.get_yaxis_transform(),
                          ha="left", va="bottom", fontsize=7, color="white",
                          bbox={"facecolor": "black", "alpha": .55, "pad": 1, "edgecolor": "none"})
            if case_row == len(CASES) - 1:
                axis.set_xlabel("Action column")
    fig.suptitle("Absolute action difference from original BF16 · same case, seed 0 · fixed inputs only\n"
                 "Primary interval: action time positions 0–9 (line at 9.5); all 32 rows are action time positions, "
                 "not denoising steps.")
    fig.subplots_adjust(left=.08, right=.88, top=.83, bottom=.12, wspace=.25, hspace=.34)
    action_color_axis = fig.add_axes([.92, .20, .018, .58])
    action_bar = fig.colorbar(plt.cm.ScalarMappable(
        norm=PowerNorm(gamma=.5, vmin=0.0, vmax=action_vmax), cmap=cmap64["magma"]),
        cax=action_color_axis, label="Absolute action difference")
    action_bar.ax.tick_params(labelsize=7)
    save_figure(fig, "07_action_step_coordinate_absolute_error.png")

    # Evidence-derived hotspots: no estimated or synthetic values.
    stage_update("hotspot_extraction")
    weight_hotspots = []
    for config in WEIGHT_CONFIGS:
        for method in ("scalar", "vq"):
            field = f"{method}_rrmse"
            ranked = sorted(weight_by_config[config].values(),
                            key=lambda row: finite(row[field], field), reverse=True)[:20]
            for rank, row in enumerate(ranked, 1):
                weight_hotspots.append({"rank": rank, "config": config, "method": method,
                    "module": row["module"], "module_index": int(row["module_index"]), "stream": row["stream"],
                    "rrmse": finite(row[field], field), "weight_rms": finite(row["weight_rms"], "weight_rms"),
                    "ratio": finite(row["ratio"], "ratio"), "peak_error": finite(row[f"{method}_peak_error"], f"{method}_peak_error"),
                    "peak_position": row[f"{method}_peak_position"]})
    local_output_hotspots = []
    for config in WEIGHT_CONFIGS:
        for method in ("scalar", "vq"):
            ranked = sorted(((index, local_records[config, index, method]) for index in module_ids),
                            key=lambda item: item[1]["total_rrmse"], reverse=True)[:20]
            for rank, (module_index, record) in enumerate(ranked, 1):
                module_row = weight_by_config[config][module_index]
                local_output_hotspots.append({"rank": rank, "config": config, "method": method,
                    "module": module_row["module"], "module_index": module_index, "stream": module_row["stream"],
                    "calls_aggregated": record["calls"], "reference_output_energy": record["reference_output_energy"],
                    "weight_rrmse": record["weight_rrmse"], "activation_rrmse": record["activation_rrmse"],
                    "cross_rrmse": record["cross_rrmse"], "total_rrmse": record["total_rrmse"],
                    "weight_error_energy": record["weight_error_energy"],
                    "activation_error_energy": record["activation_error_energy"],
                    "cross_error_energy": record["cross_error_energy"],
                    "total_error_energy": record["total_error_energy"]})
    activation_hotspots = []
    for (case_id, stage, step, module_index, config), bucket in sorted(
            act_group.items(), key=lambda item: item[1]["rrmse"], reverse=True)[:100]:
        peak_value, peak_row = activation_max_error_call[(case_id, stage, step, module_index, config)]
        activation_hotspots.append({"case_id": case_id, "stage": stage, "step": step, "config": config,
            "module": peak_row["module"], "module_index": module_index, "rrmse": bucket["rrmse"],
            "error_squared": bucket["error_squared"], "signal_squared": bucket["signal_squared"],
            "calls": bucket["calls"], "zero_count": bucket["zero_count"],
            "error_absmax": peak_value, "error_peak_position": peak_row["error_peak_position"],
            "call_index": int(peak_row["call_index"])})
    hotspots = {"weight_top_layers": weight_hotspots, "local_output_top_layers_by_total_rrmse": local_output_hotspots,
                "activation_top_layer_cells": activation_hotspots, "activation_channel_peaks": channel_hotspots}
    write_json(out_dir / "hotspots.json", hotspots)
    with (out_dir / "hotspots.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ("kind", "rank", "config", "method", "case_id", "stage", "step", "module", "module_index",
                   "stream", "channel", "rrmse", "value", "calls", "calls_aggregated", "rows", "error_squared",
                   "signal_squared", "peak_error", "peak_position", "error_absmax", "error_peak_position", "call_index",
                   "channel_error_energy", "channel_error_energy_share", "group_error_energy_total", "snapshot", "reference_output_energy",
                   "weight_rrmse", "activation_rrmse", "cross_rrmse", "total_rrmse", "weight_error_energy",
                   "activation_error_energy", "cross_error_energy", "total_error_energy")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in weight_hotspots:
            writer.writerow({"kind": "weight_layer", **row})
        for row in local_output_hotspots:
            writer.writerow({"kind": "local_output_layer", **row})
        for rank, row in enumerate(activation_hotspots, 1):
            writer.writerow({"kind": "activation_layer_cell", "rank": rank, **row})
        for rank, row in enumerate(channel_hotspots, 1):
            writer.writerow({"kind": "activation_channel", "rank": rank, "value": row["channel_error_energy"], **row})

    summary["counts"].update({"weight_snapshot_count": len(weight_arrays),
        "activation_example_count": len(activation_examples), "activation_example_module_index": chosen_activation_module,
        "weight_hotspot_rows": len(weight_hotspots), "local_output_hotspot_rows": len(local_output_hotspots),
        "activation_hotspot_rows": len(activation_hotspots),
        "activation_channel_concentration_groups": len(channel_concentration),
        "activation_channel_peak_rows": len(channel_hotspots),
        "channel_stats_npz_files_read": channel_stats_file_reads})
    summary["figure_paths"] = figure_paths
    summary["hotspots_path"] = "hotspots.json"
    summary["missing_coverage_assert"]["passed"] = True
    summary["missing_coverage_assert"]["missing"] = []
    summary["source_pbsids"] = {"full": full_summary["pbs_jobid"], "collector": collector_pbsid,
                                 "collector_sources": data_summary.get("source_pbsids", {})}
    summary["local_output_aggregation"] = {
        "source": "activations.jsonl local_output per-call records",
        "grouping": "config, module_index, method",
        "formula": "sqrt(sum(error_energy) / sum(reference_output_energy))",
        "activation_record_scope": "cases 1, 13, 21; seed_index=0; all available calls including conditioning records",
        "projection": "sampled full-covariance local Linear-output proxy using the same deterministic 16 BF16 input rows",
        "module_method_groups": len(local_records), "top20_path": "hotspots.json"}
    summary["notes_zh"].extend([
        "全层 weight 图按 module_index 绘制 Scalar/VQ 曲线并圈出 selected6；热点文件给出具体层名和数值。",
        "Local output 从 activations.jsonl 的逐调用能量按 config、module_index、method 累加后归一化；这是基于确定性 16 行输入的 sampled full-covariance local Linear proxy，不是最终 action sensitivity。",
        "Diagonal 仅描述 VQ 的训练目标；本图的 local output 投影保留样本通道相关性。",
        "Scatter 点来自 selected weight NPZ 中的完整 row_rms 和 row relative RMSE 数组；summary 只记录显示点数和 NPZ 路径，完整行数据/统计保留在 collector NPZ。",
        "Selected row 另外按振幅 Q1/Q4 计算 mean relative error，并在至少 3 个正值配对且 log 两侧都不恒定时计算 Pearson(log amplitude, log relative error)；低/高组计数显式保存。",
        "Activation channel concentration 汇总 selected6 的每调用 NPZ，累计 channel_error_energy；仅用 scheduler steps 0–9，conditioning step -1 的调用数单独说明并排除。",
        "activation_top_layer_cells 中的 error_absmax、error_peak_position 与 call_index 来自该 cell 内最大 error_absmax 的真实调用。",
        "Weight reconstruction 与 activation reconstruction 的交互图是机制诊断，不能解释为对最终 action error 的因果归因。",
        f"Activation 示例分为 cases 1/13/21 三张图，每张 4 configs × 4 maps；三张图和所有配置使用共同色标，module_index={chosen_activation_module}。",
        "Every PNG is capped at 220 KiB using a 256-color no-dither palette; heatmaps use 64 fixed color bins where applicable.",
        "Selected weight-map rows use independent PowerNorm(gamma=0.5) scales computed across both Smooth α configs; compare α within the same module row, not absolute colors across rows."])
    summary["stage"] = "complete"
    summary["status"] = "complete"
    write_json(out_dir / "summary.json", summary)
    print(f"[plot] complete: {len(figure_paths)} figures in {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        try:
            output_arg = next((sys.argv[i + 1] for i, value in enumerate(sys.argv[:-1]) if value == "--out"), None)
            if output_arg:
                output = Path(output_arg)
                if output.is_dir() and (not any(output.iterdir()) or (output / "summary.json").is_file()):
                    failed = {}
                    if (output / "summary.json").is_file():
                        try:
                            failed = read_json(output / "summary.json")
                        except Exception:
                            failed = {}
                    failed["last_stage"] = failed.get("stage")
                    failed.update(status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")
                    write_json(output / "summary.json", failed)
        except Exception:
            pass
        raise

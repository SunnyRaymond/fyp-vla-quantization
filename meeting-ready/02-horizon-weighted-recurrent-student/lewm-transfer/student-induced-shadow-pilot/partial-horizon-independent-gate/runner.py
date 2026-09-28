#!/usr/bin/env python3
"""Collect and score the frozen eight-episode partial-horizon independent gate."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
PILOT_DIR = EXPERIMENT_DIR
COLLECTOR_PATH = PILOT_DIR / "onpolicy-fullbank-ranker" / "collect_fullbank.py"
HYBRID_RUNNER_PATH = PILOT_DIR / "onpolicy-fullbank-ranker" / "partial-horizon-teacher-hybrid" / "runner.py"
ROUNDS = (10, 20, 30)
ARMS = (0, 1, 2, 5)


def require_compute_node() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("runner requires PBS_JOBID and a real PBS_NODEFILE")
    host = platform.node().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "submit", "head")):
        raise RuntimeError(f"refusing login/submit host {host}")
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes or any(any(token in node for token in ("login", "submit", "head")) for node in nodes):
        raise RuntimeError(f"host {host} is not on a clean PBS compute allocation")
    return job_id, host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_freeze(freeze: dict[str, Any], job_id: str, output: Path) -> list[dict[str, int]]:
    if freeze.get("schema") != "lewm-pusht-partial-horizon-independent-gate-freeze" or freeze.get("schema_version") != 1:
        raise RuntimeError("independent-gate freeze identity mismatch")
    if freeze.get("status") != "frozen_before_execution":
        raise RuntimeError("independent-gate freeze is not frozen before execution")
    if output.name != job_id:
        raise RuntimeError("artifact directory basename must equal PBS_JOBID")
    tasks = freeze.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 8:
        raise RuntimeError("freeze must contain exactly eight independent tasks")
    normalized = []
    for index, task in enumerate(tasks, start=24):
        row = {key: int(task[key]) for key in ("selection_order", "episode_idx", "row_index", "start_step")}
        if row["selection_order"] != index:
            raise RuntimeError("frozen task order must be selection orders 24 through 31")
        normalized.append(row)
    if len({row["episode_idx"] for row in normalized}) != 8 or len({row["row_index"] for row in normalized}) != 8:
        raise RuntimeError("frozen independent tasks must have unique episodes and rows")
    if int(freeze["collection"]["reset_seed"]) != 42 or list(freeze["collection"]["capture_rounds"]) != list(ROUNDS):
        raise RuntimeError("frozen CEM seed or capture rounds drifted")
    if list(freeze["arms"]["k_values"]) != list(ARMS):
        raise RuntimeError("frozen arms must be k=0,1,2,5")
    if int(freeze["metrics"]["timing"]["warmups_per_arm_per_bank"]) < 3 or int(freeze["metrics"]["timing"]["repeats_per_arm_per_bank"]) < 5:
        raise RuntimeError("frozen timing protocol must have at least 3 warmups and 5 repeats")
    return normalized


def load_source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup_runtime(freeze: dict[str, Any], output: Path) -> tuple[dict[str, Any], Any, Any]:
    # These imports, dataset reads, and model loads happen only after require_compute_node().
    collector = load_source_module("partial_gate_fullbank_collector", COLLECTOR_PATH)
    pilot = collector.pilot
    runtime_paths = freeze["runtime"]
    roots = {
        "lewm_root": Path(runtime_paths["lewm_root"]),
        "stablewm_root": Path(runtime_paths["stablewm_root"]),
        "stablewm_home": Path(runtime_paths["stablewm_home"]),
        "control_root": Path(runtime_paths["control_root"]),
        "baseline_dir": Path(runtime_paths["baseline_dir"]),
        "official_cem_dir": Path(runtime_paths["official_cem_dir"]),
        "adaptive_dir": Path(runtime_paths["adaptive_dir"]),
        "interface_probe": Path(runtime_paths["interface_probe_path"]),
        "cache_root": output / "cache",
    }
    pilot_args = argparse.Namespace(**roots)
    dataset_path = Path(runtime_paths["dataset_path"]).resolve(strict=True)
    teacher_path = Path(freeze["models"]["teacher_checkpoint_path"]).resolve(strict=True)
    if dataset_path != (roots["stablewm_home"] / "pusht_expert_train.h5").resolve(strict=True):
        raise RuntimeError("frozen dataset path differs from the staged official training dataset")
    if teacher_path != (roots["stablewm_home"] / "pusht" / "lewm_object.ckpt").resolve(strict=True):
        raise RuntimeError("frozen teacher path differs from the staged official LeWM checkpoint")
    if (roots["cache_root"] / "datasets" / dataset_path.name).resolve(strict=True) != dataset_path:
        raise RuntimeError("PBS dataset cache link differs from the frozen training dataset")
    if (roots["cache_root"] / "pusht" / teacher_path.name).resolve(strict=True) != teacher_path:
        raise RuntimeError("PBS teacher cache link differs from the frozen official checkpoint")
    np, spt, swm, torch, transforms, modules = pilot.load_modules(pilot_args)
    baseline, adaptive, old_router, load_official_checkpoint, preprocessing, instantiate = modules
    cfg = baseline.compose_pinned_config(roots["lewm_root"].resolve(strict=True))
    collector.validate_runtime_config(cfg)
    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=roots["cache_root"].resolve(strict=True)
    )
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    schedule, _, reference, _ = old_router.load_modules(roots["control_root"].resolve(strict=True), roots["lewm_root"].resolve(strict=True))
    schedule.validate_interface(reference, roots["interface_probe"].resolve(strict=True))
    official = load_official_checkpoint(roots["cache_root"].resolve(strict=True))
    official.interpolate_pos_encoding = True
    official.eval()
    official.requires_grad_(False)
    student_path = Path(freeze["models"]["student_checkpoint_path"]).resolve(strict=True)
    student, metadata = adaptive.load_main_student(reference, student_path)
    provenance = metadata.get("provenance", {})
    for key in ("source", "arm", "extra_updates"):
        if provenance.get(key) != freeze["models"]["student_provenance"].get(key):
            raise RuntimeError(f"student checkpoint provenance mismatch: {key}")
    if sum(parameter.numel() for parameter in student.parameters()) != 775872:
        raise RuntimeError("treatment_step1000 h256 student parameter count mismatch")
    student.eval()
    student.requires_grad_(False)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    if episode_column not in dataset.column_names or "step_idx" not in dataset.column_names:
        raise RuntimeError("official dataset lacks the frozen task identity columns")
    runtime = {
        "np": np, "torch": torch, "spt": spt, "swm": swm, "transforms": transforms,
        "baseline": baseline, "cfg": cfg, "dataset": dataset, "process": process,
        "official": official, "reference": reference, "student": student,
        "old_router": old_router, "instantiate": instantiate,
        # run_episode does not read command-line options; its seeded reset is explicit below.
        "pilot_args": argparse.Namespace(),
    }
    return runtime, collector, pilot


def validate_task_rows(runtime: dict[str, Any], tasks: list[dict[str, int]]) -> None:
    np = runtime["np"]
    dataset = runtime["dataset"]
    rows_by_index = sorted(tasks, key=lambda row: row["row_index"])
    rows = dataset.get_row_data(np.asarray([row["row_index"] for row in rows_by_index], dtype=np.int64))
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    for index, task in enumerate(rows_by_index):
        if int(rows[episode_column][index]) != task["episode_idx"] or int(rows["step_idx"][index]) != task["start_step"]:
            raise RuntimeError(f"frozen episode/row identity mismatch at order {task['selection_order']}")


def collect_independent_banks(runtime: dict[str, Any], collector: Any, tasks: list[dict[str, int]], freeze: dict[str, Any], output: Path, job_id: str, host: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate = {
        "schema": "lewm-pusht-partial-horizon-independent-noninterference-v1",
        "pbs_job_id": job_id, "compute_host": host, "status": "RUNNING",
        "required_pairs": 4, "passed_pairs": 0, "pairs": [], "remaining_collection_started": False,
    }
    gate_path = output / "noninterference_gate.json"
    write_json(gate_path, gate)
    validate_task_rows(runtime, tasks[:4])
    banks: list[dict[str, Any]] = []
    episode_rows = []
    for task in tasks[:4]:
        pair_row: dict[str, Any] = {"task": dict(task)}
        try:
            control = collector.run_captured_episode(runtime, task, False)
            shadow = collector.run_captured_episode(runtime, task, True)
            comparison = collector.compare_gate_pair(control, shadow, runtime["torch"])
            pair_row["comparison"] = comparison
            pair_row["t25_status_control"] = control["t25_status"]
            pair_row["t25_status_shadow"] = shadow["t25_status"]
            pair_row["control_dataset_reset_seed"] = control.get("dataset_reset_seed")
            pair_row["shadow_dataset_reset_seed"] = shadow.get("dataset_reset_seed")
            if comparison["pass"]:
                task_banks = collector.bank_records(shadow, "independent_gate", task, runtime["torch"])
                banks.extend(task_banks)
                episode_rows.append(collector.episode_manifest_row("independent_gate", task, shadow, len(task_banks)))
            del control, shadow
        except Exception as exc:
            pair_row["comparison"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        gate["pairs"].append(pair_row)
        gate["passed_pairs"] = sum(bool(row.get("comparison", {}).get("pass")) for row in gate["pairs"])
        write_json(gate_path, gate)
        gc.collect()
        if not pair_row["comparison"]["pass"]:
            gate["status"] = "FAIL_CLOSED"
            gate["failure_rule"] = "stop before evaluating the remaining four episodes or reporting quality metrics"
            write_json(gate_path, gate)
            raise RuntimeError(f"exact noninterference pair failed at selection order {task['selection_order']}")
    if gate["passed_pairs"] != 4:
        raise RuntimeError("all four exact noninterference pairs are required")
    gate["status"] = "PASS"
    gate["remaining_collection_started"] = True
    write_json(gate_path, gate)

    validate_task_rows(runtime, tasks[4:])
    for task in tasks[4:]:
        run = collector.run_captured_episode(runtime, task, True)
        task_banks = collector.bank_records(run, "independent_gate", task, runtime["torch"])
        banks.extend(task_banks)
        episode_rows.append(collector.episode_manifest_row("independent_gate", task, run, len(task_banks)))
        del run
        gc.collect()

    expected_ids = {task["episode_idx"] for task in tasks}
    observed_ids = {int(bank["episode_idx"]) for bank in banks}
    if observed_ids != expected_ids:
        raise RuntimeError(f"new independent banks do not cover the frozen eight episodes: {sorted(observed_ids)}")
    if any(bank["split"] != "independent_gate" for bank in banks):
        raise RuntimeError("independent-gate banks have an unexpected split label")
    episode_rows.sort(key=lambda row: int(row["task"]["selection_order"]))
    t25_ids = {int(row["task"]["episode_idx"]) for row in episode_rows if row["t25_status"] == "reached"}
    collector.save_banks(output / "independent_banks.pt", banks, runtime["torch"])
    collection = {
        "schema": "lewm-pusht-partial-horizon-independent-bank-manifest-v1",
        "status": "COMPLETE", "pbs_job_id": job_id, "compute_host": host,
        "banks_path": "independent_banks.pt", "bank_count": len(banks),
        "episode_count": len(episode_rows), "t25_episode_count": len(t25_ids),
        "minimum_t25_episode_coverage": int(freeze["validity"]["minimum_t25_episodes"]),
        "t25_coverage_sufficient": len(t25_ids) >= int(freeze["validity"]["minimum_t25_episodes"]),
        "episodes": episode_rows,
        "student_only_cem": True, "teacher_shadow_fed_to_cem": False,
    }
    write_json(output / "collection_manifest.json", collection)
    return banks, collection


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def move_bank_to_cuda(bank: dict[str, Any], torch: Any) -> dict[str, Any]:
    return {
        **bank,
        "candidates": bank["candidates"].to(device="cuda", dtype=torch.float32),
        "initial_emb": bank["initial_emb"].to(device="cuda", dtype=torch.float32),
        "goal_emb": bank["goal_emb"].to(device="cuda", dtype=torch.float32),
    }


def score_and_time(banks: list[dict[str, Any]], freeze: dict[str, Any], output: Path, runtime: dict[str, Any], helper: Any, job_id: str) -> dict[str, Any]:
    torch = runtime["torch"]
    base = sys.modules.get("run_lewm_recurrent_student")
    if base is None:
        raise RuntimeError("pinned LeWM scoring module is not loaded")
    student, teacher = runtime["student"], runtime["official"]
    prepared = [move_bank_to_cuda(bank, torch) for bank in banks]
    threshold = freeze["validity"]
    alignment = []
    costs_by_bank: list[dict[int, Any]] = []
    for index, (bank, gpu_bank) in enumerate(zip(banks, prepared, strict=True)):
        costs = {}
        for k, label, expected_key in ((0, "k0", "student_costs"), (5, "k5", "teacher_costs")):
            actual = helper.predict_and_score(base, student, teacher, gpu_bank, k, torch)
            expected = bank[expected_key].to(device=actual.device, dtype=actual.dtype)
            difference = (actual - expected).abs()
            max_abs = float(difference.max().item())
            is_close = bool(torch.allclose(actual, expected, rtol=float(threshold["rtol"]), atol=float(threshold["atol"])))
            alignment.append({
                "bank_index": index, "episode_idx": int(bank["episode_idx"]),
                "replan_step": int(bank["replan_step"]), "cem_round": int(bank["cem_round"]),
                "arm": label, "allclose": is_close, "max_abs_error": max_abs,
            })
            if not is_close:
                write_json(output / "control_alignment_failure.json", {
                    "status": "FAIL_CLOSED", "pbs_job_id": job_id,
                    "failed_control": alignment[-1],
                    "controls_checked": alignment,
                    "quality_metrics_written": False,
                })
                raise RuntimeError(f"{label} cost alignment failed on bank {index}: max abs error {max_abs:.9g}")
            costs[k] = actual
        costs_by_bank.append(costs)
    write_json(output / "control_alignment.json", {
        "status": "PASS", "pbs_job_id": job_id,
        "all_controls_checked": len(alignment), "required_controls": 2 * len(banks),
        "rtol": float(threshold["rtol"]), "atol": float(threshold["atol"]),
        "controls": alignment, "quality_metrics_may_be_reported": True,
    })

    warmups = int(freeze["metrics"]["timing"]["warmups_per_arm_per_bank"])
    repeats = int(freeze["metrics"]["timing"]["repeats_per_arm_per_bank"])
    result_rows: list[dict[str, Any]] = []
    for bank_index, (bank, gpu_bank, costs) in enumerate(zip(banks, prepared, costs_by_bank, strict=True)):
        samples: dict[int, list[float]] = {k: [] for k in ARMS}
        for cycle in range(warmups):
            order = [ARMS[(offset + bank_index + cycle) % len(ARMS)] for offset in range(len(ARMS))]
            for k in order:
                helper.timed_score(base, student, teacher, gpu_bank, k, torch)
        for cycle in range(repeats):
            order = [ARMS[(offset + bank_index + cycle) % len(ARMS)] for offset in range(len(ARMS))]
            for k in order:
                samples[k].append(helper.timed_score(base, student, teacher, gpu_bank, k, torch))
        for k in ARMS:
            arm_costs = costs.get(k)
            if arm_costs is None:
                arm_costs = helper.predict_and_score(base, student, teacher, gpu_bank, k, torch)
            regret, recall = helper.bank_metrics(arm_costs, bank["teacher_costs"].to(device="cuda", dtype=torch.float32), torch)
            result_rows.append({
                "bank_index": bank_index, "episode_idx": int(bank["episode_idx"]),
                "selection_order": int(bank["selection_order"]),
                "replan_step": int(bank["replan_step"]), "stratum": "t0" if int(bank["replan_step"]) == 0 else "t25",
                "cem_round": int(bank["cem_round"]), "k": k,
                "normalized_elite_regret": regret, "top30_recall": recall,
                "latency_ms_per_300": median(samples[k]),
                "latency_samples_ms": samples[k],
            })
        torch.cuda.synchronize()
        if bank_index % 6 == 5 or bank_index + 1 == len(banks):
            print(json.dumps({"timed_banks": bank_index + 1, "total_banks": len(banks)}), flush=True)

    summary = summarize_gate(result_rows, freeze)
    status = summary["decision"]
    write_json(output / "bank_metrics.json", {
        "schema": "lewm-pusht-partial-horizon-independent-bank-metrics-v1",
        "pbs_job_id": job_id, "status": "COMPLETE", "rows": result_rows,
    })
    write_json(output / "gate_summary.json", {
        "schema": "lewm-pusht-partial-horizon-independent-gate-result-v1",
        "pbs_job_id": job_id, "decision": status,
        "freeze_path": str((HERE / "FREEZE.json").resolve()),
        "banks_path": "independent_banks.pt", "control_alignment": "PASS",
        "noninterference_gate": "PASS", **summary,
    })
    return summary


def summarize_gate(rows: list[dict[str, Any]], freeze: dict[str, Any]) -> dict[str, Any]:
    episodes = [int(task["episode_idx"]) for task in freeze["tasks"]]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["episode_idx"]), int(row["k"])), []).append(row)
    per_arm: dict[str, Any] = {}
    all_passes = {}
    for k in (1, 2):
        episode_delta = {}
        episode_latency = {}
        round_delta = {str(round_number): {} for round_number in (20, 30)}
        for episode in episodes:
            hybrid = grouped[(episode, k)]
            baseline = grouped[(episode, 0)]
            baseline_by_key = {(int(row["replan_step"]), int(row["cem_round"])): row for row in baseline}
            pairs = [
                (row, baseline_by_key[(int(row["replan_step"]), int(row["cem_round"]))])
                for row in hybrid
            ]
            episode_delta[episode] = median([left["normalized_elite_regret"] - right["normalized_elite_regret"] for left, right in pairs])
            k5_by_key = {
                (int(row["replan_step"]), int(row["cem_round"])): row
                for row in grouped[(episode, 5)]
            }
            episode_latency[episode] = median([
                1.0 - left["latency_ms_per_300"] / k5_by_key[(int(left["replan_step"]), int(left["cem_round"]))]["latency_ms_per_300"]
                for left, _ in pairs
            ])
            for round_number in (20, 30):
                values = [
                    left["normalized_elite_regret"] - right["normalized_elite_regret"]
                    for left, right in pairs if int(left["cem_round"]) == round_number
                ]
                if values:
                    round_delta[str(round_number)][episode] = median(values)
        median_delta = median(list(episode_delta.values()))
        improved = sum(value < 0.0 for value in episode_delta.values())
        latency_reduction = median(list(episode_latency.values()))
        round_medians = {key: median(list(value.values())) for key, value in round_delta.items()}
        passes = (
            improved >= int(freeze["gate"]["episodes_with_strict_regret_improvement_min"])
            and median_delta <= float(freeze["gate"]["median_episode_regret_delta_max"])
            and latency_reduction >= float(freeze["gate"]["median_episode_latency_reduction_vs_k5_min"])
            and all(value <= float(freeze["gate"]["round20_and_round30_median_episode_regret_delta_max"]) for value in round_medians.values())
        )
        all_passes[f"k{k}"] = passes
        per_arm[f"k{k}"] = {
            "episodes_improved": improved, "episode_regret_deltas": {str(key): value for key, value in episode_delta.items()},
            "median_episode_regret_delta": median_delta,
            "median_episode_latency_reduction_vs_k5": latency_reduction,
            "round20_median_episode_regret_delta": round_medians["20"],
            "round30_median_episode_regret_delta": round_medians["30"],
            "episode_latency_reductions": {str(key): value for key, value in episode_latency.items()},
            "passes_all_thresholds": passes,
        }

    strata: dict[str, Any] = {}
    for stratum in ("t0", "t25"):
        for round_number in ROUNDS:
            key = f"{stratum}_round{round_number}"
            bank_rows = [row for row in rows if row["stratum"] == stratum and int(row["cem_round"]) == round_number]
            if not bank_rows:
                strata[key] = {"episode_count": 0, "bank_count": 0, "arms": {}}
                continue
            arm_summary = {}
            for k in ARMS:
                selected = [row for row in bank_rows if int(row["k"]) == k]
                by_episode: dict[int, list[dict[str, Any]]] = {}
                for row in selected:
                    by_episode.setdefault(int(row["episode_idx"]), []).append(row)
                arm_summary[f"k{k}"] = {
                    "median_episode_normalized_elite_regret": median([median([row["normalized_elite_regret"] for row in values]) for values in by_episode.values()]),
                    "median_episode_top30_recall": median([median([row["top30_recall"] for row in values]) for values in by_episode.values()]),
                    "median_episode_latency_ms_per_300": median([median([row["latency_ms_per_300"] for row in values]) for values in by_episode.values()]),
                }
            strata[key] = {"episode_count": len({int(row["episode_idx"]) for row in bank_rows}), "bank_count": len(bank_rows) // len(ARMS), "arms": arm_summary}
    t25_count = len({int(row["episode_idx"]) for row in rows if row["stratum"] == "t25" and int(row["k"]) == 0})
    if t25_count < int(freeze["validity"]["minimum_t25_episodes"]):
        decision = "INCONCLUSIVE"
    elif any(all_passes.values()):
        decision = "GO_TO_SEPARATELY_FROZEN_ADAPTIVE_CEM_GATE"
    else:
        decision = "NO-GO"
    return {
        "independent_episode_count": 8, "t25_episode_count": t25_count,
        "episodes_are_statistical_unit": True,
        "arm_gate": per_arm, "strata": strata,
        "exploration_transfer_limit": "GO authorizes only a separately frozen adaptive-CEM gate; no deployment or closed-loop evaluation.",
        "decision": decision,
    }


def main() -> int:
    job_id, host = require_compute_node()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.name != job_id:
        raise RuntimeError("output directory basename must equal PBS_JOBID")
    output.mkdir(parents=True, exist_ok=True)
    for artifact in ("noninterference_gate.json", "independent_banks.pt", "gate_summary.json", "failure.json"):
        if (output / artifact).exists():
            raise FileExistsError(f"refusing to overwrite {artifact}")
    freeze = read_json(args.freeze.resolve(strict=True))
    tasks = validate_freeze(freeze, job_id, output)
    runtime, collector, _pilot = setup_runtime(freeze, output)
    banks, collection = collect_independent_banks(runtime, collector, tasks, freeze, output, job_id, host)
    helper = load_source_module("partial_gate_hybrid_scoring", HYBRID_RUNNER_PATH)
    summary = score_and_time(banks, freeze, output, runtime, helper, job_id)
    print(json.dumps({"status": summary["decision"], "banks": len(banks), "t25_episodes": collection["t25_episode_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        try:
            job_id = os.environ.get("PBS_JOBID", "").strip()
            if job_id:
                output = Path(os.environ.get("PARTIAL_GATE_OUTPUT", ""))
                if output.name == job_id and output.is_dir() and not (output / "gate_summary.json").exists():
                    write_json(output / "failure.json", {"status": "FAIL_CLOSED", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        raise

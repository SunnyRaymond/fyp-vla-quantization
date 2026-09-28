#!/usr/bin/env python3
"""Decompose frozen student terminal errors on existing LeWM PushT train banks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "latent-score-sensitivity-posthoc" / "runner.py"


def require_compute_node() -> str:
    job = os.environ.get("PBS_JOBID", "").strip()
    nodefile = Path(os.environ.get("PBS_NODEFILE", ""))
    host = platform.node().split(".", 1)[0].lower()
    if not job or not nodefile.is_file() or any(x in host for x in ("login", "head", "submit")):
        raise RuntimeError("requires a real PBS compute allocation")
    nodes = {x.split(".", 1)[0].lower() for x in nodefile.read_text().splitlines() if x.strip()}
    if host not in nodes:
        raise RuntimeError("compute host is absent from PBS_NODEFILE")
    return host


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bank_row(bank, student_pred, teacher_pred, student_cost, teacher_cost, source, torch):
    zt = teacher_pred[:, -1, :].to(torch.float64)
    zs = student_pred[:, -1, :].to(torch.float64)
    goal = bank["goal_emb"].to(device=zt.device, dtype=torch.float64).reshape(1, 192)
    error = zs - zt
    common = error.mean(dim=0, keepdim=True)
    contrast = error - common
    residual = zt - goal
    scale = teacher_cost.std(unbiased=False).to(torch.float64).clamp_min(1e-6)

    teacher_manual = residual.square().sum(dim=-1)
    student_manual = (zs - goal).square().sum(dim=-1)
    if not torch.allclose(teacher_manual, teacher_cost.to(torch.float64), rtol=1e-5, atol=1e-4):
        raise RuntimeError("teacher cost differs from terminal squared-distance identity")
    if not torch.allclose(student_manual, student_cost.to(torch.float64), rtol=1e-5, atol=1e-4):
        raise RuntimeError("student cost differs from terminal squared-distance identity")

    common_score = 2 * (residual * common).sum(dim=-1)
    contrast_score = (2 * residual * contrast + 2 * common * contrast + contrast.square()).sum(dim=-1)
    score_error = student_manual - teacher_manual
    closure = score_error - (common_score + common.square().sum() + contrast_score)
    if float(closure.abs().max()) > 1e-8:
        raise RuntimeError("cost decomposition did not close")
    error_energy = error.square().sum(dim=-1).mean()
    energy_closure = error_energy - (common.square().sum() + contrast.square().sum(dim=-1).mean())
    if float(energy_closure.abs()) > 1e-8:
        raise RuntimeError("latent decomposition did not close")

    def centered_rms(value):
        return float(((value - value.mean()).square().mean().sqrt() / scale).item())

    def metrics(cost):
        return source.bank_metrics(cost.to(torch.float32), teacher_cost, torch)

    student_regret, student_recall = metrics(student_cost)
    no_common_regret, no_common_recall = metrics((zs - common - goal).square().sum(dim=-1))
    no_contrast_regret, no_contrast_recall = metrics((zs - contrast - goal).square().sum(dim=-1))
    teacher_order = torch.argsort(teacher_cost)
    upper, lower = teacher_order[20:30], teacher_order[30:40]
    boundary_inversion = (student_cost[upper, None] >= student_cost[None, lower]).to(torch.float32).mean()
    return {
        "episode_idx": int(bank["episode_idx"]),
        "replan_step": int(bank["replan_step"]),
        "cem_round": int(bank["cem_round"]),
        "student_error_contrast_energy_fraction": float((contrast.square().sum(dim=-1).mean() / error_energy.clamp_min(1e-12)).item()),
        "common_score_centered_rms_teacher_std": centered_rms(common_score),
        "contrast_score_centered_rms_teacher_std": centered_rms(contrast_score),
        "total_score_centered_rms_teacher_std": centered_rms(score_error),
        "teacher_boundary_30_31_gap_teacher_std": float(((teacher_cost[teacher_order[30]] - teacher_cost[teacher_order[29]]) / scale).item()),
        "student_boundary_inversion_20_40": float(boundary_inversion.item()),
        "student_regret": student_regret,
        "student_top30_recall": student_recall,
        "remove_common_oracle_regret": no_common_regret,
        "remove_common_oracle_top30_recall": no_common_recall,
        "remove_contrast_oracle_regret": no_contrast_regret,
        "remove_contrast_oracle_top30_recall": no_contrast_recall,
    }


def summarize(rows):
    metrics = [key for key in rows[0] if key not in ("episode_idx", "replan_step", "cem_round")]

    def one_set(selected):
        by_episode = defaultdict(list)
        for row in selected:
            by_episode[row["episode_idx"]].append(row)
        result = {"bank_count": len(selected), "episode_count": len(by_episode), "metrics": {}}
        for metric in metrics:
            episode_values = {str(ep): statistics.median(item[metric] for item in banks) for ep, banks in by_episode.items()}
            result["metrics"][metric] = {
                "episode_median": statistics.median(episode_values.values()),
                "episode_values": episode_values,
            }
        for oracle in ("remove_common", "remove_contrast"):
            deltas = [
                statistics.median(item[f"{oracle}_oracle_regret"] - item["student_regret"] for item in banks)
                for banks in by_episode.values()
            ]
            result[f"{oracle}_episodes_with_regret_improvement"] = sum(delta < 0 for delta in deltas)
            result[f"{oracle}_episode_median_regret_delta"] = statistics.median(deltas)
        return result

    return {
        "overall": one_set(rows),
        "by_replan_step": {str(step): one_set([r for r in rows if r["replan_step"] == step]) for step in (0, 25)},
        "by_cem_round": {str(round_): one_set([r for r in rows if r["cem_round"] == round_]) for round_ in (10, 20, 30)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    host = require_compute_node()  # before reading model, bank, or freeze artifacts
    sensitivity = load_module(SOURCE, "latent_score_sensitivity_source")
    source = sensitivity.load_source_runner(sensitivity.SOURCE_RUNNER_PATH)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != "lewm-pusht-action-response-decomposition-posthoc-freeze" or freeze.get("status") != "frozen_before_execution":
        raise ValueError("freeze identity mismatch")
    source_freeze = source.read_json(Path(freeze["source"]["source_freeze_path"]))
    if Path(freeze["source"]["train_banks_path"]).resolve() != Path(source_freeze["source"]["train_banks_path"]).resolve():
        raise ValueError("train bank source differs from existing frozen source")
    for key in ("student_checkpoint_path", "teacher_checkpoint_path"):
        if Path(freeze["models"][key]).resolve() != Path(source_freeze["models"][key]).resolve():
            raise ValueError(f"{key} differs from existing frozen source")

    import torch

    banks = torch.load(freeze["source"]["train_banks_path"], map_location="cpu", weights_only=False)
    source.validate_banks(banks, source_freeze, torch)
    base, student, teacher = source.load_models(source_freeze, torch)
    rows, control_errors = [], []
    for bank in banks:
        sp, tp, sc, tc = sensitivity.predict_both(base, student, teacher, bank, torch)
        key = (int(bank["episode_idx"]), int(bank["replan_step"]), int(bank["cem_round"]))
        control_errors.append(source.compare_costs("k0", sc, bank["student_costs"], source_freeze, torch, key))
        control_errors.append(source.compare_costs("k5", tc, bank["teacher_costs"], source_freeze, torch, key))
        rows.append(bank_row(bank, sp, tp, sc, tc, source, torch))
    if len(rows) != 93 or len(control_errors) != 186:
        raise RuntimeError("frozen bank/control coverage mismatch")
    output = {
        "schema": freeze["schema"] + ".result",
        "status": "READ_ONLY_DESCRIPTIVE_POSTHOC_COMPLETE",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "source_collection_job": freeze["source"]["collection_job"],
        "bank_count": len(rows),
        "control_alignment": {"comparisons": len(control_errors), "max_abs_error": max(x["max_abs_error"] for x in control_errors)},
        "claim_boundary": freeze["claim_boundary"],
        "bank_rows": rows,
        "summary": summarize(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "action_response_decomposition_summary.json"
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {target}; 93 train banks, 186 exact cost controls; descriptive only")


if __name__ == "__main__":
    main()

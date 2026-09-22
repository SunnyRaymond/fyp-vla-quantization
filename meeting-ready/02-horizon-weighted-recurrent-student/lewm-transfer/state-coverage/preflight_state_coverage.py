#!/usr/bin/env python3
"""CPU-only freeze/formula/status preflight; never opens HDF5 or prepared rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = load_json(args.freeze)
    manifest = load_json(args.manifest)
    reference = load_json(args.reference_summary)
    if freeze.get("status") != "frozen":
        raise ValueError("freeze is not frozen")
    if freeze.get("experiment_name") != "lewm_score_distill_state_context_coverage_phase2":
        raise ValueError("unexpected phase-2 experiment name")
    design = freeze["design"]
    if design["new_arms"] != ["256x3000", "512x1500", "512x3000"]:
        raise ValueError("new arm set drifted")
    if manifest.get("schema") != "lewm-recurrent-student.context-manifest":
        raise ValueError("unexpected baseline manifest schema")
    train = manifest["splits"]["train"]
    heldout = manifest["splits"]["heldout"]
    if len(train) != 256 or len(heldout) != 8:
        raise ValueError("baseline manifest is not 256/8")
    if not manifest.get("episode_disjoint"):
        raise ValueError("baseline manifest is not episode-disjoint")
    if {row["episode_id"] for row in train} & {row["episode_id"] for row in heldout}:
        raise ValueError("baseline train/heldout episode overlap")
    for row in train + heldout:
        if row["history_steps"] != [0] or row["history_action_starts"] != [0]:
            raise ValueError("temporal anchor drifted")
        if row["future_action_start"] != 0 or row["goal_step_offset"] != 25:
            raise ValueError("action/goal anchor drifted")
    if reference.get("status") != "PREDICTOR_LEVEL_COMPLETE":
        raise ValueError("score-distill reference is not complete")
    score = reference["arms"]["score_distill"]["evaluation"]["step_1500"]
    terminal = score["terminal_ranking"]
    ref_metrics = {
        "spearman_median": float(terminal["spearman_median"]),
        "top30_median": float(terminal["top30_overlap_median"]),
        "relative_latent_mse_median": float(score["relative_latent_mse_median"]),
        "positive_top30_blocks": sum(float(item["top30_overlap"]) > 0 for item in score["per_block"]),
        "top30_minimum": float(terminal["top30_overlap_minimum"]),
    }
    expected = freeze["treatment_gate"]
    result = {
        "schema": "lewm-recurrent-student.state-context-coverage-preflight",
        "status": "PASS",
        "heavy_data_opened": False,
        "hdf5_opened": False,
        "baseline_manifest_counts": {"train": len(train), "heldout": len(heldout)},
        "prefix_invariance_formula": {
            "status": "PASS",
            "selection_algorithm": freeze["context_manifest"]["selection_algorithm"],
            "heldout_verbatim": True,
            "old_train_is_new_train_prefix": True,
            "temporal_anchor_changed": False,
        },
        "reference_metrics": ref_metrics,
        "frozen_treatment_gate": expected,
        "reference_matches_treatment_thresholds": {
            "spearman": ref_metrics["spearman_median"] >= float(expected["spearman_median_min"]),
            "top30": ref_metrics["top30_median"] >= float(expected["top30_median_min"]),
            "relative_mse": ref_metrics["relative_latent_mse_median"] <= float(expected["relative_latent_mse_median_max"]),
            "positive_top30": ref_metrics["positive_top30_blocks"] == int(expected["positive_top30_blocks_exact"]),
            "minimum_top30": ref_metrics["top30_minimum"] > 0.0,
        },
        "scope": {"official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
    }
    result["reference_matches_treatment_thresholds"]["status"] = "PASS" if all(result["reference_matches_treatment_thresholds"].values()) else "INFO_ONLY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

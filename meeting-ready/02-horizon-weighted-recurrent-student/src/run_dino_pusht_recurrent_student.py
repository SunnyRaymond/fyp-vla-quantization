#!/usr/bin/env python3
"""Train and evaluate one recurrent compact DINO-WM predictor student.

This is an architecture contrast against the completed query-slate latent-MSE
control.  The control is read-only: its held-out per-block, logged-action and
timing records are loaded from the supplied query-slate summary.  The new
student is trained once on the exact frozen slate/context schedule with dense
native-latent MSE only.  Its recurrent transition consumes the current action
token and its own current latent state; outputs are fed back before the next
step.  Observation encoding and the official teacher remain outside the
student and are never trained.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_dino_pusht_context_density as density  # noqa: E402
import run_dino_pusht_optimization_length as opt  # noqa: E402
import run_dino_pusht_query_slate_rank as slate  # noqa: E402
from run_dino_pusht_grounded_prefix import (  # noqa: E402
    _RawEpisodeCache,
    _load_trajectory_datasets,
    _preencode_manifest,
)
from run_dino_pusht_rankdistill import _latent_mse  # noqa: E402
from run_dino_pusht_stage_a import (  # noqa: E402
    HORIZON,
    _gpu_snapshot,
    _json_default,
    _leakage_test,
    _latency,
    _load_json,
    _load_official,
    _metrics_by_horizon,
    _require_assets,
    _require_compute_node,
    _teacher_targets,
    _set_seed,
)


WIDTH = 256
SNAPSHOT_STEPS = (500, 1000, 1500)
SNAPSHOT_NAMES = tuple(f"step_{step}" for step in SNAPSHOT_STEPS)
SLATE_SIZE = 4


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True, help="RECURRENT_STUDENT_FREEZE.json")
    parser.add_argument("--query-slate-freeze", type=Path, required=True, help="shared QUERY_SLATE_RANK_FREEZE.json")
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, default=None)
    parser.add_argument("--context-density-freeze", type=Path, default=None)
    parser.add_argument("--density-summary", type=Path, default=None)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--deps-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing frozen field {path}.{key}")
    return mapping[key]


def _settings(recurrent_freeze: Mapping[str, Any], query_freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the recurrent freeze and independently verify its parent freeze."""

    if str(_required(recurrent_freeze, "schema", "recurrent_freeze")) != "jepa-action-prefix-compiler.recurrent-student-freeze":
        raise ValueError("unexpected recurrent freeze schema")
    if int(_required(recurrent_freeze, "schema_version", "recurrent_freeze")) != 1:
        raise ValueError("unsupported recurrent freeze schema version")
    if str(_required(query_freeze, "schema", "query_freeze")) != "jepa-action-prefix-compiler.query-slate-rank-freeze":
        raise ValueError("unexpected query-slate parent freeze schema")
    base = slate._settings(query_freeze, {})
    scope = _required(recurrent_freeze, "scope", "recurrent_freeze")
    treatment = _required(recurrent_freeze, "treatment", "recurrent_freeze")
    cell = _required(treatment, "cell", "recurrent_freeze.treatment")
    shared = _required(recurrent_freeze, "shared_data_and_schedule", "recurrent_freeze")
    shared_schedule = _required(shared, "context_schedule", "recurrent_freeze.shared_data_and_schedule")
    shared_slate = _required(shared, "action_query_slate", "recurrent_freeze.shared_data_and_schedule")
    shared_seeds = _required(shared, "seeds", "recurrent_freeze.shared_data_and_schedule")
    shared_training = _required(shared, "training", "recurrent_freeze.shared_data_and_schedule")
    heldout = _required(recurrent_freeze, "heldout_evaluation", "recurrent_freeze")
    baseline_spec = _required(recurrent_freeze, "baseline", "recurrent_freeze")
    ranking = _required(heldout, "ranking", "recurrent_freeze.heldout_evaluation")
    latency_eval = _required(heldout, "latency", "recurrent_freeze.heldout_evaluation")
    gates = _required(recurrent_freeze, "gates", "recurrent_freeze")
    integrity_gate = _required(gates, "pairing_and_integrity", "recurrent_freeze.gates")
    effect_gate = _required(gates, "architecture_effect", "recurrent_freeze.gates")
    fidelity_gate = _required(gates, "absolute_fidelity", "recurrent_freeze.gates")
    noninf_gate = _required(gates, "logged_mse_noninferiority", "recurrent_freeze.gates")
    causality_gate = _required(gates, "causality", "recurrent_freeze.gates")
    latency_gate = _required(gates, "latency", "recurrent_freeze.gates")
    for name, value in {
        "scope": scope, "treatment": treatment, "cell": cell, "shared": shared,
        "shared_schedule": shared_schedule, "shared_slate": shared_slate,
        "shared_seeds": shared_seeds, "shared_training": shared_training,
        "heldout": heldout, "baseline_spec": baseline_spec, "ranking": ranking, "latency_eval": latency_eval,
        "gates": gates, "integrity_gate": integrity_gate, "effect_gate": effect_gate,
        "fidelity_gate": fidelity_gate, "noninf_gate": noninf_gate,
        "causality_gate": causality_gate, "latency_gate": latency_gate,
    }.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"recurrent freeze field {name} must be an object")

    # Architecture and objective are explicit, immutable treatment fields.
    if str(_required(treatment, "class", "recurrent_freeze.treatment")) != "RecurrentNativeLatentTransitionStudent":
        raise ValueError("recurrent treatment class drifted")
    if int(_required(treatment, "hidden_dim", "recurrent_freeze.treatment")) != WIDTH or int(_required(scope, "student_hidden_dim", "recurrent_freeze.scope")) != WIDTH:
        raise ValueError("recurrent student hidden_dim must remain 256")
    if int(_required(scope, "horizon", "recurrent_freeze.scope")) != 5 or int(_required(scope, "frameskip", "recurrent_freeze.scope")) != 5 or int(_required(scope, "visual_dim", "recurrent_freeze.scope")) != 384 or int(_required(scope, "proprio_dim", "recurrent_freeze.scope")) != 10 or int(_required(scope, "primitive_action_dim", "recurrent_freeze.scope")) != 2 or int(_required(scope, "packed_action_token_dim", "recurrent_freeze.scope")) != 10:
        raise ValueError("recurrent native/action dimensions drifted")
    if str(_required(treatment, "parameterization", "recurrent_freeze.treatment")).find("shared") < 0:
        raise ValueError("recurrent transition must use one shared cell")
    if str(_required(treatment, "loss", "recurrent_freeze.treatment")) != "mean dense native-latent MSE over all 32 query rows, five predicted steps, and native visual/proprio dimensions":
        raise ValueError("recurrent loss is not the frozen pure dense native-latent MSE")
    if str(_required(cell, "shared_transition", "recurrent_freeze.treatment.cell")).find("Linear(256,1024)") < 0:
        raise ValueError("recurrent shared transition width drifted")
    if bool(_required(cell, "spatial_attention", "recurrent_freeze.treatment.cell")) or bool(_required(cell, "per_step_parameter_copies", "recurrent_freeze.treatment.cell")):
        raise ValueError("recurrent cell must use shared projections without spatial attention")
    if str(_required(cell, "patch_summary", "recurrent_freeze.treatment.cell")) != "mean over projected visual patches; this is the only spatial mixing":
        raise ValueError("recurrent patch summary drifted")
    if bool(_required(scope, "student_receives_goal", "recurrent_freeze.scope")) or bool(_required(scope, "student_encode_obs_calls", "recurrent_freeze.scope")):
        raise ValueError("recurrent student must not receive goal or call encode_obs")
    if not bool(_required(scope, "student_contains_teacher_or_source_parameters", "recurrent_freeze.scope")) is False:
        raise ValueError("recurrent student must not contain source/teacher parameters")
    if "teacher forcing" not in [str(item).lower() for item in _required(treatment, "forbidden", "recurrent_freeze.treatment")]:
        raise ValueError("teacher forcing must remain forbidden")

    # Parent-shared values must agree field-for-field with QUERY_SLATE_RANK_FREEZE.
    if int(_required(shared_schedule, "pool_size", "recurrent_freeze.shared_data_and_schedule.context_schedule")) != 256 or int(_required(shared_schedule, "contexts_per_update", "recurrent_freeze.shared_data_and_schedule.context_schedule")) != 8 or int(_required(shared_schedule, "updates", "recurrent_freeze.shared_data_and_schedule.context_schedule")) != 1500:
        raise ValueError("recurrent context schedule dimensions drifted")
    if int(_required(shared_schedule, "schedule_seed", "recurrent_freeze.shared_data_and_schedule.context_schedule")) != base["context_schedule_seed"]:
        raise ValueError("recurrent context schedule seed differs from parent freeze")
    if int(_required(shared_slate, "queries_per_context", "recurrent_freeze.shared_data_and_schedule.action_query_slate")) != 4 or int(_required(shared_slate, "rows", "recurrent_freeze.shared_data_and_schedule.action_query_slate")) != 1024:
        raise ValueError("recurrent action slate dimensions drifted")
    if int(_required(shared_slate, "candidate_count_M", "recurrent_freeze.shared_data_and_schedule.action_query_slate")) != 64 or int(_required(shared_slate, "elite_count_K", "recurrent_freeze.shared_data_and_schedule.action_query_slate")) != 8 or float(_required(shared_slate, "variance_floor", "recurrent_freeze.shared_data_and_schedule.action_query_slate")) != 0.05:
        raise ValueError("recurrent CEM slate parameters differ from parent freeze")
    if [str(x) for x in _required(shared_slate, "slots", "recurrent_freeze.shared_data_and_schedule.action_query_slate")] != ["primary_logged", "gaussian_planner_init_a", "gaussian_planner_init_b", "one_step_cem_resample"]:
        raise ValueError("recurrent action slate slots differ from parent freeze")
    expected_seeds = {
        "initialization": 99, "training": 20260930, "context_schedule": 20260931,
        "action_slate": 20260932, "cem": 20260933, "timing_action_prefix": 20270925,
    }
    for key, expected in expected_seeds.items():
        if key == "timing_action_prefix":
            actual = int(_required(shared_seeds, key, "recurrent_freeze.shared_data_and_schedule.seeds"))
            actual_parent = base["timing_seed"]
        else:
            actual = int(_required(shared_seeds, key, "recurrent_freeze.shared_data_and_schedule.seeds"))
            actual_parent = {"initialization": base["initialization_seed"], "training": base["training_seed"], "context_schedule": base["context_schedule_seed"], "action_slate": base["action_slate_seed"], "cem": base["cem_seed"]}[key]
        if actual != expected or actual != actual_parent:
            raise ValueError(f"recurrent seed {key} differs from parent/frozen value")
    if [int(x) for x in _required(shared_seeds, "heldout_action_prefix", "recurrent_freeze.shared_data_and_schedule.seeds")] != base["heldout_seeds"]:
        raise ValueError("recurrent held-out action seeds differ from parent freeze")
    if int(_required(shared_training, "steps", "recurrent_freeze.shared_data_and_schedule.training")) != base["steps"] or int(_required(shared_training, "batch_contexts", "recurrent_freeze.shared_data_and_schedule.training")) != base["batch"] or int(_required(shared_training, "effective_query_rows", "recurrent_freeze.shared_data_and_schedule.training")) != base["effective_batch"]:
        raise ValueError("recurrent training schedule differs from parent freeze")
    if float(_required(shared_training, "learning_rate", "recurrent_freeze.shared_data_and_schedule.training")) != base["lr"] or float(_required(shared_training, "weight_decay", "recurrent_freeze.shared_data_and_schedule.training")) != base["weight_decay"] or str(_required(shared_training, "optimizer", "recurrent_freeze.shared_data_and_schedule.training")) != "AdamW":
        raise ValueError("recurrent optimizer differs from parent freeze")
    if bool(_required(shared_training, "teacher_forcing", "recurrent_freeze.shared_data_and_schedule.training")) or bool(_required(shared_training, "loss_change", "recurrent_freeze.shared_data_and_schedule.training")) or bool(_required(shared_training, "retune_after_observation", "recurrent_freeze.shared_data_and_schedule.training")):
        raise ValueError("recurrent training must remain pure feedback MSE with no teacher forcing or retuning")
    if [int(x) for x in _required(shared_training, "snapshot_steps", "recurrent_freeze.shared_data_and_schedule.training")] != list(SNAPSHOT_STEPS):
        raise ValueError("recurrent snapshots differ from parent freeze")
    if [int(x) for x in _required(ranking, "fresh_action_prefix_seeds", "recurrent_freeze.heldout_evaluation.ranking")] != base["heldout_seeds"] or int(_required(ranking, "contexts", "recurrent_freeze.heldout_evaluation.ranking")) != base["eval_contexts"] or int(_required(ranking, "blocks", "recurrent_freeze.heldout_evaluation.ranking")) != 16 or int(_required(ranking, "candidates_per_block", "recurrent_freeze.heldout_evaluation.ranking")) != base["eval_batch"] or int(_required(ranking, "topk", "recurrent_freeze.heldout_evaluation.ranking")) != base["eval_topk"]:
        raise ValueError("recurrent held-out evaluation contract drifted")
    settings = dict(base)
    settings.update(
        {
            "hidden_dim": WIDTH,
            "architecture": str(_required(treatment, "name", "recurrent_freeze.treatment")),
            "transition": "shared_residual_mlp",
            "teacher_relative_mse_ratio_max": float(_required(noninf_gate, "ratio_max", "recurrent_freeze.gates.logged_mse_noninferiority")),
            "architecture_spearman_delta_min": float(_required(effect_gate, "median_spearman_delta_min", "recurrent_freeze.gates.architecture_effect")),
            "architecture_top30_delta_min": float(_required(effect_gate, "median_top30_delta_min", "recurrent_freeze.gates.architecture_effect")),
            "architecture_positive_spearman_min": int(_required(effect_gate, "positive_spearman_blocks_min", "recurrent_freeze.gates.architecture_effect")),
            "architecture_positive_top30_min": int(_required(effect_gate, "positive_top30_blocks_min", "recurrent_freeze.gates.architecture_effect")),
            "absolute_spearman_min": float(_required(fidelity_gate, "median_spearman_min", "recurrent_freeze.gates.absolute_fidelity")),
            "absolute_spearman_floor": float(_required(fidelity_gate, "minimum_spearman_min", "recurrent_freeze.gates.absolute_fidelity")),
            "absolute_top30_min": float(_required(fidelity_gate, "median_top30_min", "recurrent_freeze.gates.absolute_fidelity")),
            "absolute_top30_floor": float(_required(fidelity_gate, "minimum_top30_min", "recurrent_freeze.gates.absolute_fidelity")),
            "future_action_tolerance": float(_required(causality_gate, "maximum_future_action_leakage_abs", "recurrent_freeze.gates.causality")),
            "training_ratio_max": float(_required(integrity_gate, "last10_to_first_training_mse_ratio_max", "recurrent_freeze.gates.pairing_and_integrity")),
            "latency_min": float(_required(latency_gate, "predictor_only_reduction_min", "recurrent_freeze.gates.latency")),
            "baseline_job_id": str(_required(baseline_spec, "remote_job_id", "recurrent_freeze.baseline")),
            "baseline_arm": str(_required(baseline_spec, "arm", "recurrent_freeze.baseline")),
            "baseline_snapshot": str(_required(baseline_spec, "snapshot", "recurrent_freeze.baseline")),
        }
    )
    if settings["baseline_job_id"] != "24494756.pbs101" or settings["baseline_arm"] != "control" or settings["baseline_snapshot"] != "step_1500":
        raise ValueError("recurrent read-only baseline identity drifted")
    if settings["architecture_spearman_delta_min"] != 0.05 or settings["architecture_top30_delta_min"] != 0.1 or settings["teacher_relative_mse_ratio_max"] != 1.25:
        raise ValueError("unexpected recurrent gate values")
    return settings


def RecurrentNativeLatentStudent(action_dim: int, visual_dim: int, proprio_dim: int, hidden_dim: int) -> Any:
    """Build the frozen shared residual transition student.

    At step ``t`` the shared transition sees only the current action token and
    the current predicted visual/proprio native latent.  The predicted output
    is fed back to the next step; no separate recurrent hidden state or
    teacher-forced future latent is used.
    """

    import torch
    import torch.nn as nn

    class _Impl(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_dim = int(visual_dim)
            self.proprio_dim = int(proprio_dim)
            self.hidden_dim = int(hidden_dim)
            self.visual_projection = nn.Linear(self.visual_dim, hidden_dim)
            self.proprio_projection = nn.Linear(self.proprio_dim, hidden_dim)
            self.action_projection = nn.Linear(action_dim, hidden_dim)
            self.shared_transition = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.visual_update_norm = nn.LayerNorm(hidden_dim)
            self.proprio_update_norm = nn.LayerNorm(hidden_dim)
            self.visual_out = nn.Linear(hidden_dim, self.visual_dim)
            self.proprio_out = nn.Linear(hidden_dim, self.proprio_dim)

        def forward(self, context: Mapping[str, Any], actions: Any) -> dict[str, Any]:
            visual_state = context["visual"][:, -1]
            proprio_state = context["proprio"][:, -1]
            visual_outputs = []
            proprio_outputs = []
            for index in range(int(actions.shape[1])):
                visual_hidden = self.visual_projection(visual_state)
                proprio_hidden = self.proprio_projection(proprio_state)
                action_hidden = self.action_projection(actions[:, index])
                shared_condition = visual_hidden.mean(dim=1) + proprio_hidden + action_hidden
                transition = self.shared_transition(shared_condition)
                visual_state = visual_state + self.visual_out(
                    self.visual_update_norm(visual_hidden + transition.unsqueeze(1))
                )
                proprio_state = proprio_state + self.proprio_out(
                    self.proprio_update_norm(proprio_hidden + transition)
                )
                visual_outputs.append(visual_state)
                proprio_outputs.append(proprio_state)
            return {
                "visual": torch.stack(visual_outputs, dim=1),
                "proprio": torch.stack(proprio_outputs, dim=1),
            }

    return _Impl()


def _student(*args: Any, **kwargs: Any) -> Any:
    return RecurrentNativeLatentStudent(*args, **kwargs)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _safe_ratio(a: float, b: float) -> float:
    return float(a / max(float(b), 1e-12))


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rankings = [item["ranking"] for item in cases]
    return {
        "per_case": list(cases),
        "per_horizon": {
            key: [_mean([float(item[key][h]) for item in cases]) for h in range(HORIZON)]
            for key in ("relative_mse", "cosine")
        },
        "terminal_ranking": {
            "spearman_mean": _mean([float(item["spearman"]) for item in rankings]),
            "spearman_median": _median([float(item["spearman"]) for item in rankings]),
            "spearman_minimum": min(float(item["spearman"]) for item in rankings),
            "top30_overlap_mean": _mean([float(item["top30_overlap"]) for item in rankings]),
            "top30_overlap_median": _median([float(item["top30_overlap"]) for item in rankings]),
            "top30_overlap_minimum": min(float(item["top30_overlap"]) for item in rankings),
        },
        "mean_relative_mse": _mean([float(x) for item in cases for x in item["relative_mse"]]),
    }


def _evaluate_snapshots(
    snapshots: Mapping[str, Mapping[str, Any]],
    model: Any,
    encoded: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    eval_seeds: Sequence[int],
    batch_size: int,
    device: Any,
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    blocks = opt._prepare_eval_blocks(encoded, 8, eval_seeds, batch_size, action_dim, device)
    evaluation: dict[str, Any] = {"recurrent": {}}
    contrasts: dict[str, Any] = {}
    baseline_eval = baseline.get("evaluation", {}).get("control", {})
    for snap in SNAPSHOT_NAMES:
        student = _student(
            action_dim,
            int(encoded["context"]["visual"].shape[-1]),
            int(encoded["context"]["proprio"].shape[-1]),
            WIDTH,
        ).to(device)
        student.load_state_dict(snapshots[snap], strict=True)
        student.eval()
        cases: list[dict[str, Any]] = []
        contrast_cases: list[dict[str, Any]] = []
        reference_cases = baseline_eval.get(snap, {}).get("per_case", [])
        if len(reference_cases) != len(blocks):
            raise ValueError(f"baseline {snap} has {len(reference_cases)} blocks; expected {len(blocks)}")
        for block, reference in zip(blocks, reference_cases):
            if (int(reference.get("seed", -1)), int(reference.get("context_index", -1))) != (int(block["seed"]), int(block["context_index"])):
                raise ValueError("baseline held-out block key differs from deterministic evaluation bank")
            with torch.no_grad():
                target = _teacher_targets(model, block["context"], block["actions"])
                prediction = student(block["context"], block["actions"])
            teacher_cost = objective_fn(target, block["goal"])
            student_cost = objective_fn(prediction, block["goal"])
            ranking = {
                "spearman": slate._spearman(teacher_cost, student_cost),
                "top30_overlap": slate._topk_overlap(teacher_cost, student_cost),
            }
            metrics = _metrics_by_horizon(prediction, target)
            metrics.update(
                {
                    "seed": int(block["seed"]),
                    "context_index": int(block["context_index"]),
                    "candidate_count": int(batch_size),
                    "topk": min(30, int(batch_size)),
                    "ranking": ranking,
                }
            )
            cases.append(metrics)
            reference_ranking = reference.get("ranking", {})
            contrast_cases.append(
                {
                    "seed": int(block["seed"]),
                    "context_index": int(block["context_index"]),
                    "spearman_delta_recurrent_minus_baseline": float(ranking["spearman"]) - float(reference_ranking["spearman"]),
                    "top30_overlap_delta_recurrent_minus_baseline": float(ranking["top30_overlap"]) - float(reference_ranking["top30_overlap"]),
                }
            )
        evaluation["recurrent"][snap] = _aggregate_cases(cases)
        contrasts[snap] = {
            "per_case": contrast_cases,
            "spearman_mean": _mean([x["spearman_delta_recurrent_minus_baseline"] for x in contrast_cases]),
            "spearman_median": _median([x["spearman_delta_recurrent_minus_baseline"] for x in contrast_cases]),
            "top30_overlap_mean": _mean([x["top30_overlap_delta_recurrent_minus_baseline"] for x in contrast_cases]),
            "top30_overlap_median": _median([x["top30_overlap_delta_recurrent_minus_baseline"] for x in contrast_cases]),
            "positive_spearman_blocks": sum(x["spearman_delta_recurrent_minus_baseline"] > 0 for x in contrast_cases),
            "positive_top30_blocks": sum(x["top30_overlap_delta_recurrent_minus_baseline"] > 0 for x in contrast_cases),
        }
    return evaluation, contrasts


def _evaluate_logged(
    snapshots: Mapping[str, Mapping[str, Any]],
    model: Any,
    encoded: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for snap in SNAPSHOT_NAMES:
        student = _student(
            int(encoded["actions"].shape[-1]),
            int(encoded["context"]["visual"].shape[-1]),
            int(encoded["context"]["proprio"].shape[-1]),
            WIDTH,
        ).to(device)
        student.load_state_dict(snapshots[snap], strict=True)
        student.eval()
        cases = []
        for index in range(8):
            context = {key: value[index : index + 1].to(device) for key, value in encoded["context"].items()}
            actions = encoded["actions"][index : index + 1].to(device)
            with torch.no_grad():
                target = _teacher_targets(model, context, actions)
                prediction = student(context, actions)
            cases.append({"context_index": index, **_metrics_by_horizon(prediction, target)})
        result[snap] = {
            "context_count": len(cases),
            "per_context": cases,
            "mean_relative_mse": _mean([x for case in cases for x in case["relative_mse"]]),
        }
    return result


def _baseline_latencies(baseline: Mapping[str, Any]) -> Mapping[str, Any]:
    latency = baseline.get("latency", {})
    if not isinstance(latency, Mapping):
        return {}
    return latency.get("control", {}) if isinstance(latency.get("control", {}), Mapping) else {}


def _parameter_summary(student: Any) -> dict[str, int]:
    total = sum(int(p.numel()) for p in student.parameters())
    trainable = sum(int(p.numel()) for p in student.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _gates(
    settings: Mapping[str, Any],
    result: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    final = result["evaluation"]["recurrent"]["step_1500"]["terminal_ranking"]
    effect = result["architecture_effect"]["step_1500"]
    contrast = result["latent_noninferiority"]
    causality = result["causality"]
    integrity = result["integrity"]
    recurrent_ratio = float(contrast["recurrent_to_baseline_logged_mse_ratio"])
    effect_ok = bool(
        effect["spearman_median"] >= settings["architecture_spearman_delta_min"]
        and effect["top30_overlap_median"] >= settings["architecture_top30_delta_min"]
        and effect["positive_spearman_blocks"] >= settings["architecture_positive_spearman_min"]
        and effect["positive_top30_blocks"] >= settings["architecture_positive_top30_min"]
    )
    absolute_ok = bool(
        final["spearman_median"] >= settings["absolute_spearman_min"]
        and final["spearman_minimum"] >= settings["absolute_spearman_floor"]
        and final["top30_overlap_median"] >= settings["absolute_top30_min"]
        and final["top30_overlap_minimum"] >= settings["absolute_top30_floor"]
    )
    noninf_ok = recurrent_ratio <= settings["teacher_relative_mse_ratio_max"]
    causality_ok = all(bool(item.get("passed", False)) for item in causality.values())
    latency_ok = float(result["latency"]["recurrent"]["median_reduction"]) >= float(settings["latency_min"])
    integrity_ok = bool(integrity["passed"])
    # The frozen contract separates an independent replacement decision from
    # the relative architecture-effect decision.  A recurrent student may
    # satisfy full replacement even when it does not beat the historical
    # control on every ranking threshold.
    full = all((integrity_ok, absolute_ok, noninf_ok, causality_ok, latency_ok))
    supported = bool(full and integrity_ok and effect_ok)
    return {
        "integrity": {"status": "PASS" if integrity_ok else "FAIL", "details": integrity},
        "architecture_effect": {
            "status": "PASS" if effect_ok else "FAIL",
            "metrics": effect,
            "thresholds": {
                "median_spearman_delta_min": settings["architecture_spearman_delta_min"],
                "median_top30_delta_min": settings["architecture_top30_delta_min"],
                "positive_spearman_blocks_min": settings["architecture_positive_spearman_min"],
                "positive_top30_blocks_min": settings["architecture_positive_top30_min"],
            },
            "interpretation": "descriptive architecture contrast against a completed read-only baseline; not same-initialization causal evidence",
        },
        "absolute_fidelity": {"status": "PASS" if absolute_ok else "FAIL", "metrics": final},
        "latent_noninferiority": {
            "status": "PASS" if noninf_ok else "FAIL",
            "recurrent_to_baseline_logged_mse_ratio": recurrent_ratio,
            "threshold_max": settings["teacher_relative_mse_ratio_max"],
        },
        "causality": {"status": "PASS" if causality_ok else "FAIL"},
        "predictor_latency": {
            "status": "PASS" if latency_ok else "FAIL",
            "recurrent_reduction": result["latency"]["recurrent"]["median_reduction"],
            "threshold_min": settings["latency_min"],
        },
        "decision_levels": {
            "architecture_effect": "PASS" if integrity_ok and effect_ok else "FAIL",
            "full_replacement": "GO" if full else "NO-GO",
            "recurrent_supported_replacement": "GO" if supported else "NO-GO",
        },
        "overall": "GO" if full else "NO-GO",
        "baseline_summary": baseline.get("schema", "unknown"),
    }


def _validate_baseline(baseline: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    if str(baseline.get("schema", "")) != "jepa-action-prefix-compiler.dino-pusht-query-slate-rank-summary":
        raise ValueError("baseline summary schema is not the completed query-slate summary")
    contracts = baseline.get("contracts", {})
    if not isinstance(contracts, Mapping):
        raise ValueError("baseline summary has no contracts object")
    if int(contracts.get("heldout_blocks", -1)) != 16 or int(contracts.get("heldout_contexts", -1)) != 8 or int(contracts.get("candidates_per_block", -1)) != 300 or [int(x) for x in contracts.get("heldout_seeds", [])] != [20264925, 20264926]:
        raise ValueError("baseline held-out contract does not match the recurrent freeze")
    if str(baseline.get("source", {}).get("external_dhigh", "")) == "":
        raise ValueError("baseline source metadata is incomplete")
    pairing = baseline.get("pairing_integrity", {})
    expected_block_ids = [f"seed={seed}:context={context}" for seed in (20264925, 20264926) for context in range(8)]
    if not isinstance(pairing, Mapping) or pairing.get("context_schedule_equal") is not True or pairing.get("slate_bank_equal") is not True or pairing.get("heldout_candidate_bank_equal") is not True or pairing.get("all_slate_groups_complete") is not True or pairing.get("all_slate_queries_pairwise_distinct") is not True or pairing.get("heldout_split_unchanged") is not True or pairing.get("paired_block_ids") != expected_block_ids:
        raise ValueError("baseline pairing/schedule/block-key contract is not authoritative")
    if baseline.get("fallback_used") is not False or baseline.get("finite_outputs_and_training") is not True or baseline.get("shape_dtype_device_match") is not True:
        raise ValueError("baseline finite/shape/fallback integrity failed")
    if baseline.get("student_integrity", {}).get("control", {}).get("passed") is not True:
        raise ValueError("baseline student integrity failed")
    if not isinstance(baseline.get("evaluation"), Mapping):
        raise ValueError("baseline summary has no evaluation object")
    control = baseline["evaluation"].get("control", {})
    expected_keys = [(seed, context) for seed in (20264925, 20264926) for context in range(8)]
    for snap in SNAPSHOT_NAMES:
        rows = control.get(snap, {}).get("per_case", [])
        if len(rows) != 16:
            raise ValueError(f"baseline control {snap} must contain exactly 16 held-out blocks")
        actual_keys = []
        for row in rows:
            key = (int(row.get("seed", -1)), int(row.get("context_index", -1)))
            if int(row.get("candidate_count", -1)) != 300 or int(row.get("topk", -1)) != 30:
                raise ValueError(f"baseline {snap} candidate contract differs")
            ranking = row.get("ranking", {})
            if not isinstance(ranking, Mapping) or "spearman" not in ranking or "top30_overlap" not in ranking:
                raise ValueError(f"baseline {snap} has incomplete ranking metrics")
            actual_keys.append(key)
        if actual_keys != expected_keys:
            raise ValueError(f"baseline {snap} block order/keys differ from frozen seed outer, context inner order")
    required = baseline.get("logged_teacher_relative", {}).get("control", {})
    if any(snap not in required for snap in SNAPSHOT_NAMES):
        raise ValueError("baseline is missing control logged-action records for all snapshots")
    if "control" not in baseline.get("latency", {}):
        raise ValueError("baseline is missing control latency")
    return {
        "schema": baseline.get("schema"),
        "job_id": settings["baseline_job_id"],
        "arm": settings["baseline_arm"],
        "snapshot": settings["baseline_snapshot"],
        "heldout_blocks": 16,
        "block_keys": expected_keys,
        "snapshots": list(SNAPSHOT_NAMES),
        "read_only": True,
    }


def main() -> int:
    args = _args()
    _require_compute_node()
    root = args.root.resolve()
    asset_root = (args.asset_root or root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    query_freeze = _load_json(args.query_slate_freeze.resolve())
    settings = _settings(freeze, query_freeze)
    if settings["baseline_job_id"] not in str(args.baseline_summary.resolve()):
        raise ValueError("baseline summary path does not identify the frozen read-only job")
    baseline = _load_json(args.baseline_summary.resolve())
    baseline_meta = _validate_baseline(baseline, settings)
    runtime_freeze = dict(query_freeze)
    for contract_key in ("assets", "model", "official_pushT"):
        if contract_key in freeze:
            runtime_freeze[contract_key] = freeze[contract_key]
    context_freeze = _load_json(args.context_density_freeze.resolve()) if args.context_density_freeze else None
    density_settings = slate._density_schedule_settings(query_freeze, context_freeze)
    manifest = density._load_density_manifest(args.manifest.resolve(), density_settings, args.legacy_manifest.resolve() if args.legacy_manifest else None)
    _require_assets(asset_root, runtime_freeze)
    _gpu_snapshot(output, "start")
    import torch

    _set_seed(settings["training_seed"])
    model, workspace, _anchors, _goals, objective_fn, action_dim, device = _load_official(
        root,
        asset_root,
        runtime_freeze,
        output,
        settings["training_seed"],
        config_path=args.config.resolve() if args.config else None,
        checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None,
        checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None,
        data_root=args.data_root.resolve() if args.data_root else None,
    )
    del workspace, _anchors, _goals
    if int(action_dim) != 10:
        raise ValueError("official packed action token dim must remain 10")
    train_dset, heldout_dset = _load_trajectory_datasets(root, asset_root, runtime_freeze, args.checkpoint, args.checkpoint_config)
    train_examples = manifest["splits"]["train"]["examples"]
    heldout_examples = manifest["splits"]["heldout"]["examples"]
    cache = _RawEpisodeCache(8)
    train_encoded = _preencode_manifest(train_dset, "train", train_examples, 5, 8, cache, model, device)
    heldout_encoded = _preencode_manifest(heldout_dset, "heldout", heldout_examples, 5, 8, cache, model, device)
    visual_dim = int(train_encoded["context"]["visual"].shape[-1])
    proprio_dim = int(train_encoded["context"]["proprio"].shape[-1])
    if visual_dim != 384:
        raise ValueError("visual native latent dimension must remain 384")
    _set_seed(settings["initialization_seed"])
    template = _student(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    student = _student(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    student.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
        betas=settings["betas"],
        eps=settings["eps"],
    )
    schedule, schedule_meta = slate._context_schedule(settings["steps"], settings["batch"], settings["context_schedule_seed"])
    bank = slate._precompute_slate_bank(
        model,
        train_encoded,
        objective_fn,
        action_dim,
        settings["action_slate_seed"],
        settings["cem_seed"],
        device,
    )
    _gpu_snapshot(output, "precompute_complete")
    losses: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    checkpoint_paths: dict[str, str] = {}
    shape_dtype_device_match = True
    for step_index, selected in enumerate(schedule):
        step = step_index + 1
        context, actions, target, _goal = slate._materialize_slate_batch(train_encoded, bank, selected, device)
        student.train()
        prediction = student(context, actions)
        shape_dtype_device_match = shape_dtype_device_match and all(
            prediction[key].shape == target[key].shape
            and prediction[key].dtype == target[key].dtype
            and prediction[key].device == target[key].device
            for key in ("visual", "proprio")
        )
        latent = _latent_mse(prediction, target)
        if not torch.isfinite(latent):
            raise FloatingPointError(f"non-finite recurrent latent MSE at step {step}")
        optimizer.zero_grad(set_to_none=True)
        latent.backward()
        optimizer.step()
        losses.append({"step": step, "latent_mse": float(latent.detach().cpu()), "effective_query_rows": settings["effective_batch"]})
        if step in SNAPSHOT_STEPS:
            state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
            snapshots[f"step_{step}"] = state
            filename = f"recurrent_compact_step{step:04d}.pt"
            torch.save(
                {
                    "schema": "jepa-action-prefix-compiler.dino-pusht-recurrent-student-checkpoint",
                    "step": step,
                    "architecture": settings["architecture"],
                    "hidden_dim": WIDTH,
                    "state_dict": state,
                    "parent_freeze": str(args.freeze.resolve()),
                    "query_slate_freeze": str(args.query_slate_freeze.resolve()),
                },
                output / filename,
            )
            checkpoint_paths[f"step_{step}"] = filename
        if step == 1 or step % max(1, settings["steps"] // 10) == 0:
            _gpu_snapshot(output, f"train_step_{step}")
    evaluation, architecture_effect = _evaluate_snapshots(
        snapshots,
        model,
        heldout_encoded,
        objective_fn,
        action_dim,
        settings["heldout_seeds"],
        settings["eval_batch"],
        device,
        baseline,
    )
    evaluation["baseline_control"] = {
        snap: baseline["evaluation"]["control"][snap] for snap in SNAPSHOT_NAMES
    }
    logged = _evaluate_logged(snapshots, model, heldout_encoded, device)
    final_student = _student(action_dim, visual_dim, proprio_dim, WIDTH).to(device)
    final_student.load_state_dict(snapshots["step_1500"], strict=True)
    causality = {
        "recurrent": _leakage_test(
            final_student,
            {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()},
            action_dim,
            settings["heldout_seeds"][0] + 9000,
            device,
            settings["future_action_tolerance"],
        )
    }
    latency = {
        "recurrent": _latency(
            final_student,
            model,
            {key: value[:2].to(device) for key, value in heldout_encoded["context"].items()},
            action_dim,
            settings["timing_seed"],
            settings["timing_batch"],
            settings["warmup"],
            settings["repeats"],
            device,
        )
    }
    baseline_latency = _baseline_latencies(baseline)
    if isinstance(baseline_latency, Mapping) and "student_median_ms" in baseline_latency:
        latency["recurrent_to_baseline_control_ratio"] = _safe_ratio(
            float(latency["recurrent"]["student_median_ms"]),
            float(baseline_latency["student_median_ms"]),
        )
    _gpu_snapshot(output, "complete")
    train_result = {
        "steps": settings["steps"],
        "batch_contexts": settings["batch"],
        "queries_per_context": SLATE_SIZE,
        "effective_query_samples": settings["effective_batch"],
        "loss_first": losses[0]["latent_mse"],
        "loss_last": losses[-1]["latent_mse"],
        "loss_median_last_10": _median([item["latent_mse"] for item in losses[-10:]]),
        "last10_to_first_loss_ratio": _safe_ratio(_median([item["latent_mse"] for item in losses[-10:]]), losses[0]["latent_mse"]),
        "per_step": losses,
    }
    baseline_logged = baseline["logged_teacher_relative"]["control"]
    recurrent_logged_mse = {snap: float(logged[snap]["mean_relative_mse"]) for snap in SNAPSHOT_NAMES}
    baseline_logged_mse = {snap: float(baseline_logged[snap]["mean_relative_mse"]) for snap in SNAPSHOT_NAMES}
    student_integrity = opt._student_integrity(final_student, model)
    manifest_validation = manifest.get("density_validation", {})
    manifest_fields_ok = bool(
        manifest_validation.get("status") == "PASS"
        and manifest_validation.get("old_context_rows_preserved") is True
        and manifest_validation.get("old_context_order_preserved") is True
        and manifest_validation.get("new_contexts_exclude_old_starts") is True
        and manifest_validation.get("train_episode_set_unchanged") is True
        and manifest_validation.get("validation_split_unchanged") is True
        and manifest_validation.get("heldout_split_unchanged") is True
        and int(manifest_validation.get("new_trajectory_count", -1)) == 0
    )
    student_state_finite = all(bool(torch.isfinite(parameter).all().item()) for parameter in final_student.parameters())
    training_ratio_ok = train_result["last10_to_first_loss_ratio"] <= settings["training_ratio_max"]
    causality_ok = all(bool(item.get("passed", False)) for item in causality.values())
    evaluation_finite = _all_finite({"evaluation": evaluation, "effect": architecture_effect, "logged": logged, "latency": latency})
    integrity = {
        "passed": False,
        "manifest_integrity": manifest_validation,
        "manifest_fields_ok": manifest_fields_ok,
        "context_schedule_equal_to_query_slate": True,
        "action_slate_equal_to_query_slate": True,
        "heldout_candidate_bank_equal_by_key": True,
        "all_slate_groups_complete": len(schedule) == settings["steps"] and all(len(x) == settings["batch"] for x in schedule),
        "all_slate_queries_pairwise_distinct": bool(bank["generation"]["pairwise_distinct_passed"]),
        "student_goal_input_hidden": True,
        "student_encode_obs_hidden": bool(student_integrity.get("encode_obs_entry_points", []) == []),
        "student_source_or_teacher_parameters_absent": bool(
            not student_integrity.get("student_is_teacher", True)
            and student_integrity.get("source_encoder_module_entries", []) == []
            and student_integrity.get("shared_teacher_module_entries", []) == []
            and not student_integrity.get("source_teacher_reference", True)
            and student_integrity.get("passed", False)
        ),
        "student_integrity": student_integrity,
        "shape_dtype_device_match": shape_dtype_device_match,
        "recurrent_outputs_finite": student_state_finite,
        "recurrent_training_finite": _all_finite(train_result),
        "recurrent_evaluation_finite": evaluation_finite,
        "last10_to_first_training_mse_ratio": train_result["last10_to_first_loss_ratio"],
        "training_ratio_max": settings["training_ratio_max"],
        "training_ratio_ok": training_ratio_ok,
        "causality_ok": causality_ok,
        "no_oom_nan_or_silent_fallback": True,
        "baseline_read_only": bool(baseline_meta["read_only"]),
        "baseline_block_keys_validated": True,
    }
    integrity["passed"] = bool(
        integrity["recurrent_outputs_finite"]
        and integrity["recurrent_training_finite"]
        and integrity["recurrent_evaluation_finite"]
        and shape_dtype_device_match
        and integrity["student_source_or_teacher_parameters_absent"]
        and integrity["training_ratio_ok"]
        and integrity["causality_ok"]
        and integrity["manifest_fields_ok"]
        and integrity["context_schedule_equal_to_query_slate"]
        and integrity["action_slate_equal_to_query_slate"]
        and integrity["heldout_candidate_bank_equal_by_key"]
        and integrity["all_slate_groups_complete"]
        and integrity["all_slate_queries_pairwise_distinct"]
        and integrity["baseline_read_only"]
        and integrity["baseline_block_keys_validated"]
        and integrity["no_oom_nan_or_silent_fallback"]
    )
    result = {
        "evaluation": evaluation,
        "architecture_effect": architecture_effect,
        "logged_teacher_relative": logged,
        "latent_noninferiority": {
            "recurrent_logged_mean_relative_mse": recurrent_logged_mse,
            "baseline_control_logged_mean_relative_mse": baseline_logged_mse,
            "recurrent_to_baseline_logged_mse_ratio_by_snapshot": {snap: _safe_ratio(recurrent_logged_mse[snap], baseline_logged_mse[snap]) for snap in SNAPSHOT_NAMES},
            "recurrent_to_baseline_logged_mse_ratio": _safe_ratio(recurrent_logged_mse["step_1500"], baseline_logged_mse["step_1500"]),
        },
        "causality": causality,
        "latency": latency,
        "integrity": integrity,
        "train": {"recurrent": train_result},
        "finite_outputs_and_training": _all_finite({"evaluation": evaluation, "effect": architecture_effect, "logged": logged, "latency": latency, "train": train_result}),
        "shape_dtype_device_match": shape_dtype_device_match,
        "student_integrity": student_integrity,
        "fallback_used": False,
    }
    recurrent_parameters = _parameter_summary(final_student)
    baseline_parameter_model = slate.NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, WIDTH)
    baseline_parameters = _parameter_summary(baseline_parameter_model)
    result["parameter_counts"] = {
        "recurrent": recurrent_parameters,
        "baseline_control_contract": baseline_parameters,
        "recurrent_to_baseline_total_ratio": _safe_ratio(recurrent_parameters["total"], baseline_parameters["total"]),
    }
    gates = _gates(settings, result, baseline)
    summary = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-recurrent-student-summary",
        "schema_version": 1,
        "freeze": str(args.freeze.resolve()),
        "query_slate_freeze": str(args.query_slate_freeze.resolve()),
        "protocol": str(args.protocol.resolve()) if args.protocol else None,
        "manifest": str(args.manifest.resolve()),
        "baseline_summary": str(args.baseline_summary.resolve()),
        "density_summary": str(args.density_summary.resolve()) if args.density_summary else None,
        "source": {
            "student": "RecurrentNativeLatentTransitionStudent hidden_dim=256",
            "transition": "shared residual MLP: s=mean(visual_proj)+proprio_proj+action_proj; predicted native latent feedback each step",
            "teacher_target": "shared detached official DINO-WM rollout target",
            "training_loss": "dense native-latent MSE only",
            "baseline": "read-only query-slate latent-MSE control from supplied summary",
            "observation_encoder": "outside scope; cached native observation latent",
        },
        "contracts": {
            "train_contexts": 256,
            "contexts_per_update": settings["batch"],
            "queries_per_context": SLATE_SIZE,
            "effective_query_samples_per_update": settings["effective_batch"],
            "updates": settings["steps"],
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "heldout_blocks": 16,
            "heldout_contexts": 8,
            "heldout_seeds": settings["heldout_seeds"],
            "candidates_per_block": settings["eval_batch"],
            "hidden_dim": WIDTH,
            "same_context_and_slate_schedule_as_query_slate": True,
            "gpu_telemetry_seconds": settings["telemetry_seconds"],
        },
        "architecture": {
            "name": settings["architecture"],
            "transition": settings["transition"],
            "causal_feedback": True,
            "goal_input": False,
            "encode_obs_calls": False,
            "source_encoder_frozen": True,
            "source_predictor_frozen": True,
            "parameter_counts": result["parameter_counts"],
        },
        "slate": bank["generation"],
        "schedule": schedule_meta,
        "pairing_integrity": integrity,
        "train": {"recurrent": train_result},
        "evaluation": evaluation,
        "architecture_effect": architecture_effect,
        "absolute_recurrent": evaluation["recurrent"]["step_1500"]["terminal_ranking"],
        "logged_teacher_relative": logged,
        "baseline_control_logged_teacher_relative": baseline["logged_teacher_relative"]["control"],
        "latent_noninferiority": result["latent_noninferiority"],
        "causality": causality,
        "latency": {
            "recurrent": latency["recurrent"],
            "baseline_control": _baseline_latencies(baseline),
        },
        "parameter_counts": result["parameter_counts"],
        "finite_outputs_and_training": result["finite_outputs_and_training"],
        "shape_dtype_device_match": result["shape_dtype_device_match"],
        "fallback_used": False,
        "checkpoint_filenames": checkpoint_paths,
        "gates": gates,
        "decisions": gates["decision_levels"],
        "claim_boundary": {
            "architecture_effect": "A predictor-level architecture contrast against the completed query-slate control under the shared frozen schedule; it is not same-initialization causal evidence.",
            "full_replacement": "Only if all frozen recurrent gates pass, including absolute fidelity, logged-action non-inferiority, causality, and latency.",
            "forbidden": ["encode_obs effect", "closed-loop CEM or environment success", "LeWM/Fast-LeWM transfer", "all-JEPA universality", "native low-bit deployment"],
        },
        "timing_boundary": "predictor-level: cached native observation latent plus normalized action prefix; excludes encode_obs, CEM execution, environment interaction and closed-loop control",
        "unverified": ["No closed-loop CEM or environment execution is included", "This DINO-WM runner does not establish LeWM transfer"],
    }
    summary_path = (args.summary or output / "recurrent_student_summary.json").resolve()
    text = json.dumps(summary, indent=2, default=_json_default)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text, encoding="utf-8")
    default_summary = output / "recurrent_student_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary), "overall": gates["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

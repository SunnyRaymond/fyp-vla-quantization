#!/usr/bin/env python3
"""Paired LeWM PushT terminal action-response-loss predictor experiment.

The runner is intentionally self-contained at the experiment layer: it imports
the frozen LeWM model, scoring, and fresh-row adapters without modifying them.
All model, HDF5, checkpoint, and bank I/O is guarded to a live PBS compute node.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
REFERENCE_DIR = TRANSFER / "state-action-prefix-gru"
SCHEMA = "lewm-recurrent-student.terminal-response-loss-freeze"
SOURCE_JOB = "25213164.pbs101"
STEPS = 3000
TRAIN_CONTEXTS = 512
BATCH_CONTEXTS = 8
TRAIN_CANDIDATES = 64
FRESH_EPISODES = 8
FRESH_ANCHORS = ("early", "middle", "late")
NUM_CANDIDATES = 300
ELITE_COUNT = 30
INIT_SEED = 20300901
TRAINING_SEED = 20300902
CONTEXT_SCHEDULE_SEED = 20300904
SCORE_LOSS_WEIGHT = 0.1
RESPONSE_LOSS_WEIGHT = 0.01
RESPONSE_DENOMINATOR_FLOOR = 1e-6
VALID_SELECTION_SEED = 20300903
VALID_SELECTION_START = 600
HORIZON = 5
LATENT_DIM = 192
ACTION_DIM = 10
HORIZON_PRIMITIVES = HORIZON * (ACTION_DIM // 2)
LAST10_TO_FIRST_MAX = 0.8
EXPECTED_PARAMETER_COUNT = 775872


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> str:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile_path = os.environ.get("PBS_NODEFILE", "").strip()
    host = platform.node().split(".", 1)[0].lower()
    if not job_id or not nodefile_path:
        raise RuntimeError("requires a live PBS allocation (PBS_JOBID and PBS_NODEFILE)")
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    nodefile = Path(nodefile_path)
    if not nodefile.is_file():
        raise RuntimeError("PBS_NODEFILE is missing")
    nodes = {line.strip().split(".", 1)[0].lower() for line in nodefile.read_text().splitlines() if line.strip()}
    if host not in nodes:
        raise RuntimeError(f"compute host {host!r} is absent from PBS_NODEFILE")
    return host


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_reference_modules() -> Any:
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    return importlib.import_module("run_lewm_state_action_prefix_gru")


def _close(actual: Any, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"freeze contract drifted for {label}: {actual} != {expected}")


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != SCHEMA or freeze.get("status") != "frozen_before_execution":
        raise ValueError("terminal-response freeze schema/status mismatch")
    source = freeze.get("source", {})
    training = freeze.get("training", {})
    evaluation = freeze.get("evaluation", {})
    used = source.get("used_episode_manifests", {})
    required_paths = (
        "lewm_root",
        "stablewm_home",
        "dataset_path",
        "teacher_checkpoint_path",
        "interface_probe_path",
        "phase2_manifest_path",
        "balanced_rows_path",
        "balanced_base_reference_checkpoint_path",
    )
    for key in required_paths:
        if not source.get(key):
            raise ValueError(f"freeze source.{key} is required")
    for key in ("seeded_pilot32", "real_observation80", "closed_loop50"):
        if not used.get(key):
            raise ValueError(f"freeze source.used_episode_manifests.{key} is required")
    required_eval = (
        "official_action_low",
        "official_action_high",
        "gaussian_std",
        "action_prefix_seeds",
        "selection_start",
        "candidates_per_block",
    )
    for key in required_eval:
        if key not in evaluation:
            raise ValueError(f"freeze evaluation.{key} is required")
    action_prefix_seeds = [int(value) for value in evaluation["action_prefix_seeds"]]
    if action_prefix_seeds != [20301007, 20301008]:
        raise ValueError("evaluation action-prefix seeds drifted")
    if evaluation["official_action_low"] != [-1.0, -1.0] or evaluation["official_action_high"] != [1.0, 1.0]:
        raise ValueError("evaluation action bounds drifted")
    _close(evaluation["gaussian_std"], 0.22360679774997896, "evaluation gaussian_std")

    training_seeds = training.get("seeds", {})
    response_loss = training.get("response_loss", {})
    optimizer = training.get("optimizer", {})
    objective = training.get("balanced_base_objective", {})
    fixed = {
        "initialization_seed": (training_seeds.get("initialization_seed"), INIT_SEED),
        "training_seed": (training_seeds.get("training_seed"), TRAINING_SEED),
        "context_schedule_seed": (training_seeds.get("context_schedule_seed"), CONTEXT_SCHEDULE_SEED),
        "candidate_slate_seed": (training_seeds.get("candidate_slate_seed"), 20300905),
        "updates": (training.get("updates"), STEPS),
        "batch_contexts": (training.get("batch_contexts"), BATCH_CONTEXTS),
        "candidate_slates_per_context": (training.get("candidate_slates_per_context"), TRAIN_CANDIDATES),
        "response_loss.lambda": (response_loss.get("lambda"), RESPONSE_LOSS_WEIGHT),
    }
    for key, (value, expected) in fixed.items():
        if value is None:
            raise ValueError(f"freeze training.{key} is required")
        if isinstance(expected, float):
            _close(value, expected, f"training.{key}")
        elif int(value) != expected:
            raise ValueError(f"freeze contract drifted for training.{key}")
    expected_horizon_weights = (1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3)
    if len(objective.get("horizon_weights", ())) != HORIZON:
        raise ValueError("freeze balanced_base horizon weights are required")
    for index, (actual, expected) in enumerate(zip(objective["horizon_weights"], expected_horizon_weights)):
        _close(actual, expected, f"training.balanced_base_objective.horizon_weights[{index}]")
    expected_score_description = "0.1 * context-normalized teacher-score SmoothL1 mean; top-12 weight 2.0, remaining candidate weight 1.0, std floor 1e-6, beta 1.0."
    if objective.get("score_loss") != expected_score_description:
        raise ValueError("freeze balanced_base score objective drifted")
    expected_normalizer = "Teacher bank-centered terminal latent MSE; floor fixed at 1e-6. Compute the mean across the 64 fixed training candidates and latent dimensions for each context."
    if response_loss.get("normalizer") != expected_normalizer:
        raise ValueError("freeze response-loss normalizer drifted")
    if int(response_loss.get("terminal_horizon_index", -1)) != HORIZON - 1:
        raise ValueError("freeze response loss must use the terminal horizon index")
    if not str(training.get("teacher_targets", "")).startswith("Reuse the frozen teacher targets stored"):
        raise ValueError("freeze requires direct reuse of cached teacher targets")
    expected_optimizer = {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8}
    if optimizer != expected_optimizer:
        raise ValueError("freeze optimizer differs from the reconstructed balanced_base contract")
    if int(evaluation.get("selection_seed", -1)) != VALID_SELECTION_SEED:
        raise ValueError("fresh episode shuffle seed must remain 20300903")
    if int(evaluation.get("selection_start", -1)) != VALID_SELECTION_START:
        raise ValueError("fresh episode selection must start at valid[600:]")
    if int(evaluation.get("episodes", -1)) != FRESH_EPISODES:
        raise ValueError("fresh evaluation must use exactly eight episodes")
    if int(evaluation.get("candidates_per_block", -1)) != NUM_CANDIDATES:
        raise ValueError("fresh evaluation must use 300 candidates per block")
    if int(evaluation.get("blocks", -1)) != FRESH_EPISODES * len(FRESH_ANCHORS) * len(action_prefix_seeds):
        raise ValueError("fresh evaluation must contain exactly 48 paired blocks")
    if tuple(evaluation.get("anchors", ())) != FRESH_ANCHORS:
        raise ValueError("fresh evaluation anchors must be early/middle/late")
    gates = freeze.get("gates", {})
    if not isinstance(gates.get("absolute_predictor"), Mapping):
        raise ValueError("freeze gates.absolute_predictor is required")
    if not isinstance(gates.get("stratum_protection"), Mapping):
        raise ValueError("freeze gates.stratum_protection is required")
    expected_absolute = {
        "median_spearman_min": 0.95,
        "minimum_spearman_min": 0.80,
        "median_top30_min": 0.75,
        "minimum_top30_min": 0.50,
        "median_relative_latent_mse_max": 0.25,
        "positive_spearman_blocks_min": 36,
        "positive_top30_blocks_min": 36,
        "latency_reduction_min": 0.20,
    }
    for key, expected in expected_absolute.items():
        actual = gates["absolute_predictor"].get(key)
        if actual is None:
            raise ValueError(f"freeze gates.absolute_predictor.{key} is required")
        if isinstance(expected, float):
            _close(actual, expected, f"absolute predictor threshold {key}")
        elif int(actual) != expected:
            raise ValueError(f"absolute predictor threshold {key} drifted")
    expected_stratum = {
        "median_spearman_min": 0.95,
        "median_top30_min": 0.75,
        "positive_spearman_blocks_min": 12,
        "positive_top30_blocks_min": 12,
    }
    if gates["stratum_protection"].get("strata") != list(FRESH_ANCHORS) or int(gates["stratum_protection"].get("blocks_per_stratum", -1)) != 16:
        raise ValueError("stratum protection coverage drifted")
    for key, expected in expected_stratum.items():
        actual = gates["stratum_protection"].get(key)
        if actual is None:
            raise ValueError(f"freeze gates.stratum_protection.{key} is required")
        if isinstance(expected, float):
            _close(actual, expected, f"stratum protection threshold {key}")
        elif int(actual) != expected:
            raise ValueError(f"stratum protection threshold {key} drifted")
    mechanism = gates.get("relative_mechanism", {})
    for key in (
        "response_normalized_mse_episode_median_delta_max",
        "standardized_elite_regret_episode_median_delta_max",
        "episodes_with_strict_regret_improvement_min",
        "top30_recall_episode_median_delta_min",
    ):
        if key not in mechanism:
            raise ValueError(f"freeze gates.relative_mechanism.{key} is required")
    _close(mechanism["response_normalized_mse_episode_median_delta_max"], -0.05, "mechanism response MSE threshold")
    _close(mechanism["standardized_elite_regret_episode_median_delta_max"], -0.05, "mechanism regret threshold")
    if int(mechanism["episodes_with_strict_regret_improvement_min"]) != 5:
        raise ValueError("mechanism regret improvement count must be five")
    _close(mechanism["top30_recall_episode_median_delta_min"], 0.0, "mechanism top30 threshold")
    if int(mechanism.get("response_floor_active_blocks_max", -1)) != 0:
        raise ValueError("mechanism gate requires zero floor-active evaluation blocks")
    if not bool(gates["absolute_predictor"]["integrity_convergence_causality_required"]):
        raise ValueError("absolute predictor integrity/convergence/causality gate must remain required")
    latency = evaluation.get("latency", {})
    if int(latency.get("warmup", -1)) != 3 or int(latency.get("repeats", -1)) != 10:
        raise ValueError("freeze evaluation latency must retain 3 warmups and 10 repeats")
    if latency.get("student_teacher_comparison_required") is not True:
        raise ValueError("freeze requires the student/teacher latency comparison")
    if float(evaluation.get("causality_tolerance", -1)) != 1e-6:
        raise ValueError("causality tolerance must remain at the inherited 1e-6 threshold")
    return {"source": source, "training": training, "evaluation": evaluation, "gates": gates, "action_prefix_seeds": action_prefix_seeds}


def _manifest_task_ids(value: Any, label: str) -> tuple[list[int], list[str] | None]:
    if isinstance(value, list):
        tasks = value
    elif isinstance(value, Mapping):
        tasks = value.get("tasks")
        if not isinstance(tasks, list):
            for key in ("selected_tasks", "selected", "episode_ids"):
                candidate = value.get(key)
                if isinstance(candidate, list):
                    tasks = candidate
                    break
    else:
        tasks = None
    if not isinstance(tasks, list):
        raise ValueError(f"{label} has no task or episode-id list")
    ids: list[int] = []
    splits: list[str] | None = []
    for item in tasks:
        if isinstance(item, Mapping):
            episode = item.get("episode_idx", item.get("episode_id"))
            split = item.get("split", item.get("role"))
            if split is None:
                splits = None
        else:
            episode, split = item, None
            splits = None
        if episode is None:
            raise ValueError(f"{label} task lacks episode_idx/episode_id")
        ids.append(int(episode))
        if splits is not None:
            splits.append(str(split))
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate episode IDs")
    return ids, splits


def read_used_episode_ids(paths: Mapping[str, Any]) -> tuple[dict[str, list[int]], dict[str, Any]]:
    pilot_path = Path(paths["seeded_pilot32"]).resolve()
    real_path = Path(paths["real_observation80"]).resolve()
    closed_path = Path(paths["closed_loop50"]).resolve()
    pilot = load_json(pilot_path)
    real = load_json(real_path)
    closed = load_json(closed_path)

    if pilot.get("schema") != "lewm-pusht-student-induced-shadow-episode-selection-result":
        raise ValueError("seeded-pilot selection manifest schema mismatch")
    pilot_ids, pilot_roles = _manifest_task_ids(pilot, "seeded-pilot selection manifest")
    if len(pilot_ids) != 32 or pilot_roles is None:
        raise ValueError("seeded-pilot manifest must list 32 split-labeled episodes")
    if {role: pilot_roles.count(role) for role in set(pilot_roles)} != {"collection": 16, "reserved_holdout": 16}:
        raise ValueError("seeded-pilot split/ID counts drifted")

    if real.get("schema") != "lewm-pusht-student-driven-real-observation-training-selection-manifest-v1":
        raise ValueError("real-observation selection manifest schema mismatch")
    if int(real.get("selected_episode_count", -1)) != 80:
        raise ValueError("real-observation manifest must select 80 episodes")
    real_ids, real_splits = _manifest_task_ids(real, "real-observation selection manifest")
    if len(real_ids) != 80 or real_splits is None:
        raise ValueError("real-observation manifest must list 80 split-labeled episodes")
    expected_real_splits = {"collection_train": 64, "collection_validation": 16}
    if {split: real_splits.count(split) for split in set(real_splits)} != expected_real_splits:
        raise ValueError("real-observation split/ID counts drifted")

    closed_ids, _ = _manifest_task_ids(closed, "closed-loop-50 selection manifest")
    if len(closed_ids) != 50:
        raise ValueError("closed-loop manifest must list exactly 50 unique episodes")
    exclusions = pilot.get("selection_freeze", {}).get("legacy_exclusions", {})
    frozen_closed_ids = exclusions.get("closed_loop_50", {}).get("episode_ids")
    if not isinstance(frozen_closed_ids, list) or {int(value) for value in frozen_closed_ids} != set(closed_ids):
        raise ValueError("closed-loop task IDs do not match the frozen 50-ID exclusion")

    ids = {
        "seeded_pilot32": pilot_ids,
        "real_observation80": real_ids,
        "closed_loop50": closed_ids,
    }
    metadata = {
        "manifest_paths": {"seeded_pilot32": str(pilot_path), "real_observation80": str(real_path), "closed_loop50": str(closed_path)},
        "counts": {key: len(value) for key, value in ids.items()},
        "split_counts": {
            "seeded_pilot32": {role: pilot_roles.count(role) for role in sorted(set(pilot_roles))},
            "real_observation80": {split: real_splits.count(split) for split in sorted(set(real_splits))},
        },
        "closed_loop_ids_match_seeded_freeze": True,
    }
    return ids, metadata


def select_fresh_episode_ids(
    dataset_path: Path,
    phase2_manifest_path: Path,
    seeded_pilot_manifest_path: Path,
    used_ids: Mapping[str, Sequence[int]],
    selection_seed: int = VALID_SELECTION_SEED,
) -> tuple[list[int], dict[str, Any]]:
    """Select eight fresh episodes after valid[600:], excluding every frozen used set."""
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        lengths = [int(value) for value in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON_PRIMITIVES + 1]
    random.Random(int(selection_seed)).shuffle(valid)
    phase2 = load_json(phase2_manifest_path)
    splits = phase2.get("splits", {})
    heldout = [int(row["episode_id"]) for row in splits.get("heldout", [])]
    train = [int(row["episode_id"]) for row in splits.get("train", [])]
    if len(heldout) != 8 or len(train) != TRAIN_CONTEXTS:
        raise ValueError("Phase2 manifest must contain the frozen 8 heldout and 512 train episodes")
    if valid[:520] != heldout + train:
        raise ValueError("fixed valid shuffle does not reproduce the Phase2 manifest prefix")

    pilot = load_json(seeded_pilot_manifest_path)
    selection_freeze = pilot.get("selection_freeze", {})
    prior = pilot.get("legacy_exclusions", {}).get("prior_valid_episode_ids")
    prior_meta = selection_freeze.get("legacy_exclusions", {}).get("prior_valid_prefix", {})
    if int(prior_meta.get("shuffle_seed", -1)) != selection_seed or int(prior_meta.get("slice_stop_exclusive", -1)) != VALID_SELECTION_START:
        raise ValueError("seeded-pilot freeze does not certify the required valid[0:600] shuffle prefix")
    if not isinstance(prior, list) or [int(value) for value in prior] != valid[:VALID_SELECTION_START]:
        raise ValueError("fixed valid shuffle differs from the frozen first-600 exclusion list")

    used = set(heldout + train)
    for values in used_ids.values():
        used.update(int(value) for value in values)
    valid_set = set(valid)
    if not used <= valid_set:
        raise ValueError("a frozen used episode ID is absent from the valid episode pool")
    candidates = [episode for episode in valid[VALID_SELECTION_START:] if episode not in used]
    fresh = candidates[:FRESH_EPISODES]
    if len(fresh) != FRESH_EPISODES or len(set(fresh)) != FRESH_EPISODES or set(fresh) & used:
        raise ValueError("could not select eight distinct unused fresh episodes")
    return fresh, {
        "selection_seed": selection_seed,
        "valid_episode_count": len(valid),
        "selection_start": VALID_SELECTION_START,
        "candidate_count_after_exclusions": len(candidates),
        "phase2_heldout_count": len(heldout),
        "phase2_train_count": len(train),
        "fresh_episode_ids": fresh,
        "selection_slice": "valid[600:] filtered by frozen used episode IDs, first 8",
        "result_dependent_selection": False,
    }


def validate_training_rows(rows: Sequence[Mapping[str, Any]], phase2_manifest_path: Path) -> list[Mapping[str, Any]]:
    manifest = load_json(phase2_manifest_path)
    train_manifest = list(manifest.get("splits", {}).get("train", []))
    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) != TRAIN_CONTEXTS or len(train_manifest) != TRAIN_CONTEXTS:
        raise ValueError("balanced training data must contain the exact 512 Phase2 train contexts")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("reconstructed balanced rows must contain only train-split rows")
    strata: dict[str, int] = defaultdict(int)
    seen_contexts: set[str] = set()
    for ordinal, (row, manifest_row) in enumerate(zip(train_rows, train_manifest)):
        if int(row.get("ordinal", -1)) != ordinal:
            raise ValueError("balanced rows are not in frozen Phase2 ordinal order")
        if int(row.get("episode_id", -1)) != int(manifest_row["episode_id"]):
            raise ValueError(f"balanced row episode ID differs from Phase2 train ordinal {ordinal}")
        context_id = str(row.get("context_id", ""))
        if not context_id or context_id in seen_contexts:
            raise ValueError("balanced row context IDs must be present and unique")
        seen_contexts.add(context_id)
        strata[str(row.get("stratum", ""))] += 1
        if tuple(row["future_actions"].shape) != (TRAIN_CANDIDATES, HORIZON, ACTION_DIM):
            raise ValueError(f"balanced row {ordinal} action shape drifted")
        if tuple(row["teacher_targets"].shape) != (TRAIN_CANDIDATES, HORIZON, LATENT_DIM):
            raise ValueError(f"balanced row {ordinal} teacher target shape drifted")
        if tuple(row["latent_history"].shape[-2:]) != (1, LATENT_DIM):
            raise ValueError(f"balanced row {ordinal} latent-history shape drifted")
    if strata != {"early": 171, "middle": 171, "late": 170}:
        raise ValueError(f"balanced temporal strata drifted: {dict(strata)}")
    return train_rows


def bank_centered_terminal_response_loss(
    prediction: Any,
    target: Any,
    batch_contexts: int,
    candidates: int = TRAIN_CANDIDATES,
    denominator_floor: float = RESPONSE_DENOMINATOR_FLOOR,
) -> tuple[Any, Any]:
    """Return mean per-bank response MSE and its per-bank values.

    Inputs are flattened candidate rows [batch_contexts*candidates, horizon, dim].
    Centering is always within each candidate bank, never across different states.
    """
    import torch

    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("response loss expects aligned [banks*candidates,horizon,latent] tensors")
    if prediction.shape[0] != int(batch_contexts) * int(candidates):
        raise ValueError("flattened response rows do not match bank/candidate counts")
    if prediction.shape[1:] != (HORIZON, LATENT_DIM):
        raise ValueError("response loss expects five 192-D terminal prediction steps")
    student_terminal = prediction.reshape(batch_contexts, candidates, HORIZON, LATENT_DIM)[:, :, -1, :]
    teacher_terminal = target.detach().reshape(batch_contexts, candidates, HORIZON, LATENT_DIM)[:, :, -1, :]
    student_centered = student_terminal - student_terminal.mean(dim=1, keepdim=True)
    teacher_centered = teacher_terminal - teacher_terminal.mean(dim=1, keepdim=True)
    numerator = (student_centered - teacher_centered).square().mean(dim=(1, 2))
    denominator = teacher_centered.square().mean(dim=(1, 2)).clamp_min(float(denominator_floor))
    per_bank = numerator / denominator
    return per_bank.mean(), per_bank


def train_arm(
    reference: Any,
    rows: Sequence[Mapping[str, Any]],
    initial_state: Mapping[str, Any],
    official_model: Any,
    arm: str,
) -> dict[str, Any]:
    """Train one baseline h256 arm with matched seeds and context schedule."""
    import torch

    if arm not in ("balanced_base", "terminal_response"):
        raise ValueError(f"unknown training arm: {arm}")
    train_rows = list(rows)
    torch.manual_seed(INIT_SEED)
    student = reference.instantiate_student("balanced_base").to("cuda")
    reference.load_full_state(student, initial_state)
    torch.manual_seed(TRAINING_SEED)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    schedule = torch.Generator(device="cpu").manual_seed(CONTEXT_SCHEDULE_SEED)
    history: list[dict[str, Any]] = []
    for step in range(1, STEPS + 1):
        indices = torch.randperm(len(train_rows), generator=schedule)[:BATCH_CONTEXTS]
        contexts = torch.cat([reference.base._tensor(train_rows[int(index)]["latent_history"]).expand(TRAIN_CANDIDATES, -1, -1) for index in indices], dim=0).to("cuda")
        actions = torch.cat([reference.base._tensor(train_rows[int(index)]["future_actions"]) for index in indices], dim=0).to("cuda")
        targets = torch.cat([reference.base._tensor(train_rows[int(index)]["teacher_targets"]) for index in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        latent_loss, per_horizon = reference.base.recurrent_loss(prediction, targets)
        student_cost = reference.score._student_costs_with_gradient(official_model, train_rows, indices, prediction)
        teacher_cost = reference.score._effective_teacher_costs(train_rows, indices, "score_distill")
        score_loss = reference.score.score_distill_loss(student_cost, teacher_cost)
        if arm == "terminal_response":
            response_loss, response_by_bank = bank_centered_terminal_response_loss(
                prediction, targets, BATCH_CONTEXTS, TRAIN_CANDIDATES, RESPONSE_DENOMINATOR_FLOOR
            )
            total_loss = latent_loss + SCORE_LOSS_WEIGHT * score_loss + RESPONSE_LOSS_WEIGHT * response_loss
        else:
            response_loss = prediction.new_zeros(())
            response_by_bank = prediction.new_zeros((BATCH_CONTEXTS,))
            total_loss = latent_loss + SCORE_LOSS_WEIGHT * score_loss
        finite = bool(
            torch.isfinite(total_loss).item()
            and torch.isfinite(per_horizon).all().item()
            and torch.isfinite(score_loss).item()
            and torch.isfinite(response_loss).item()
            and torch.isfinite(response_by_bank).all().item()
        )
        if not finite:
            raise FloatingPointError(f"non-finite loss at step {step} in {arm}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        history.append(
            {
                "step": step,
                "weighted_latent_mse": float(latent_loss.detach().cpu()),
                "score_loss": float(score_loss.detach().cpu()),
                "normalized_response_mse": float(response_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "per_horizon_mse": [float(value) for value in per_horizon.detach().cpu()],
                "finite": finite,
            }
        )
    first = float(history[0]["weighted_latent_mse"])
    last10 = float(statistics.median(row["weighted_latent_mse"] for row in history[-10:]))
    terminal_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    return {
        "state_dict": terminal_state,
        "parameter_count": int(sum(parameter.numel() for parameter in student.parameters())),
        "last10_to_first_ratio": last10 / max(first, 1e-12),
        "trace": {
            "first": history[0],
            "step_500": history[499],
            "step_1000": history[999],
            "step_1500": history[1499],
            "step_2000": history[1999],
            "step_2500": history[2499],
            "terminal": history[-1],
            "last10_weighted_latent_mse_median": last10,
            "finite_updates": sum(bool(item["finite"]) for item in history),
        },
        "finite": all(bool(item["finite"]) for item in history),
        "arm": arm,
    }


def save_terminal_checkpoint(path: Path, arm: str, state: Mapping[str, Any], freeze_path: Path, rows_path: Path) -> None:
    import torch

    torch.save(
        {
            "schema": SCHEMA + ".checkpoint",
            "arm": arm,
            "step": STEPS,
            "architecture": "LeWMCompactRecurrentTransitionStudent h256",
            "state_dict": dict(state),
            "provenance": {
                "freeze": str(freeze_path.resolve()),
                "training_rows": str(rows_path.resolve()),
                "training_rows_source_job": SOURCE_JOB,
                "initialization_seed": INIT_SEED,
                "training_seed": TRAINING_SEED,
                "context_schedule_seed": CONTEXT_SCHEDULE_SEED,
                "candidate_slate_seed": 20300905,
                "teacher_targets_reused_from_cached_rows": True,
                "training_arm": arm,
                "response_loss_weight": RESPONSE_LOSS_WEIGHT if arm == "terminal_response" else 0.0,
                "response_denominator_floor": RESPONSE_DENOMINATOR_FLOOR if arm == "terminal_response" else None,
            },
        },
        path,
    )


def compare_reconstructed_checkpoint(state: Mapping[str, Any], path: Path) -> dict[str, Any]:
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or raw.get("provenance", {}).get("reconstructed") is not True:
        raise ValueError("25213164 balanced_base checkpoint must carry reconstructed provenance")
    original = raw.get("state_dict")
    if not isinstance(original, Mapping) or set(original) != set(state):
        raise ValueError("reconstructed balanced_base checkpoint parameter keys drifted")
    differences = []
    for key, value in state.items():
        if tuple(value.shape) != tuple(original[key].shape):
            raise ValueError(f"reconstructed checkpoint parameter shape drifted: {key}")
        differences.append(float((value.to(torch.float32) - original[key].to(torch.float32)).abs().max().item()))
    return {
        "reference_job": SOURCE_JOB,
        "reference_checkpoint": str(path.resolve()),
        "reference_is_reconstructed": True,
        "max_abs_parameter_difference": max(differences, default=0.0),
        "diagnostic_only": True,
        "hard_gate": False,
    }


def build_fresh_rows(reference: Any, official_model: Any, freeze: Mapping[str, Any], dataset_path: Path, episode_ids: Sequence[int]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    old_seeds = tuple(reference.ema.FRESH_SEEDS)
    old_anchors = tuple(reference.ema.FRESH_ANCHORS)
    reference.ema.FRESH_SEEDS = tuple(int(value) for value in freeze["evaluation"]["action_prefix_seeds"])
    reference.ema.FRESH_ANCHORS = FRESH_ANCHORS
    try:
        rows, metadata = reference.ema.build_fresh_rows(official_model, dataset_path, episode_ids, freeze)
    finally:
        reference.ema.FRESH_SEEDS = old_seeds
        reference.ema.FRESH_ANCHORS = old_anchors
    expected_rows = FRESH_EPISODES * len(FRESH_ANCHORS)
    if len(rows) != expected_rows or int(metadata.get("blocks", -1)) != expected_rows * 2:
        raise ValueError("fresh-row builder returned unexpected episode/anchor/seed coverage")
    expected_ids = set(int(value) for value in episode_ids)
    for row in rows:
        if row.get("split") != "fresh_eval" or int(row.get("episode_id", -1)) not in expected_ids:
            raise ValueError("fresh evaluation row split/episode ID drifted")
        if tuple(row["future_actions"].shape) != (2, NUM_CANDIDATES, HORIZON, ACTION_DIM):
            raise ValueError("fresh candidate bank shape drifted")
        if tuple(row["teacher_targets"].shape) != (2, NUM_CANDIDATES, HORIZON, LATENT_DIM):
            raise ValueError("fresh teacher target shape drifted")
    return rows, metadata


def evaluate_block(reference: Any, official_model: Any, student: Any, row: Mapping[str, Any], block: int, seed: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
    target = reference.base._tensor(row["teacher_targets"][block], dtype=torch.float32).to("cuda")
    context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda").expand(NUM_CANDIDATES, -1, -1)
    with torch.no_grad():
        prediction = student(context, actions)
        student_cost = reference.base._official_objective(official_model, row, prediction)
    teacher_cost = reference.base._tensor(row["teacher_objective"][block], dtype=torch.float32).to("cuda")
    teacher_order = torch.argsort(teacher_cost)
    student_order = torch.argsort(student_cost)
    teacher_top = teacher_order[:ELITE_COUNT]
    student_top = student_order[:ELITE_COUNT]
    recall = float(torch.isin(teacher_top, student_top).sum().item()) / ELITE_COUNT
    teacher_std = teacher_cost.std(unbiased=False).clamp_min(1e-6)
    regret = (teacher_cost[student_top].mean() - teacher_cost[teacher_top].mean()) / teacher_std
    centered_student = prediction[:, -1, :] - prediction[:, -1, :].mean(dim=0, keepdim=True)
    centered_teacher = target[:, -1, :] - target[:, -1, :].mean(dim=0, keepdim=True)
    response_delta = centered_student - centered_teacher
    teacher_contrast_mse = centered_teacher.square().mean()
    response_floor_active = bool(teacher_contrast_mse < RESPONSE_DENOMINATOR_FLOOR)
    teacher_energy = teacher_contrast_mse.clamp_min(RESPONSE_DENOMINATOR_FLOOR)
    response_mse = response_delta.square().mean() / teacher_energy
    student_norm = torch.linalg.vector_norm(centered_student.reshape(-1))
    teacher_norm = torch.linalg.vector_norm(centered_teacher.reshape(-1))
    response_cosine = F.cosine_similarity(centered_student.reshape(1, -1), centered_teacher.reshape(1, -1), dim=1, eps=1e-8)[0]
    response_scale = student_norm / teacher_norm.clamp_min(1e-8)
    relative_latent_mse = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)
    spearman = reference.base._spearman(teacher_cost, student_cost)
    finite = bool(
        torch.isfinite(prediction).all().item()
        and torch.isfinite(student_cost).all().item()
        and torch.isfinite(teacher_cost).all().item()
        and torch.isfinite(regret).item()
        and torch.isfinite(teacher_contrast_mse).item()
        and torch.isfinite(response_mse).item()
        and torch.isfinite(response_cosine).item()
        and torch.isfinite(response_scale).item()
        and math.isfinite(float(spearman))
    )
    if not finite:
        raise FloatingPointError("non-finite fresh evaluation metric")
    return {
        "episode_id": int(row["episode_id"]),
        "context_id": str(row["context_id"]),
        "stratum": str(row["stratum"]),
        "anchor": int(row["anchor"]),
        "action_prefix_seed": int(seed),
        "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}",
        "spearman": float(spearman),
        "top30_recall": recall,
        "standardized_elite_regret": float(regret.detach().cpu()),
        "relative_latent_mse": float(relative_latent_mse.detach().cpu()),
        "normalized_response_mse": float(response_mse.detach().cpu()),
        "response_cosine": float(response_cosine.detach().cpu()),
        "response_scale": float(response_scale.detach().cpu()),
        "teacher_terminal_contrast_mse": float(teacher_contrast_mse.detach().cpu()),
        "response_floor_active": response_floor_active,
        "finite": finite,
    }


METRICS = (
    "spearman",
    "top30_recall",
    "standardized_elite_regret",
    "relative_latent_mse",
    "normalized_response_mse",
    "response_cosine",
    "response_scale",
)


def _summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot summarize an empty block set")
    result: dict[str, Any] = {"blocks": len(items), "finite": all(bool(item["finite"]) for item in items)}
    for metric in METRICS:
        values = [float(item[metric]) for item in items]
        result[f"{metric}_median"] = float(statistics.median(values))
        result[f"{metric}_minimum"] = min(values)
        result[f"{metric}_maximum"] = max(values)
    result["positive_spearman_blocks"] = sum(float(item["spearman"]) > 0 for item in items)
    result["positive_top30_blocks"] = sum(float(item["top30_recall"]) > 0 for item in items)
    result["response_floor_active_blocks"] = sum(bool(item["response_floor_active"]) for item in items)
    result["teacher_terminal_contrast_mse_median"] = float(statistics.median(float(item["teacher_terminal_contrast_mse"]) for item in items))
    result["worst_blocks"] = {
        "minimum_spearman": _block_identity(min(items, key=lambda item: float(item["spearman"])), "spearman"),
        "minimum_top30_recall": _block_identity(min(items, key=lambda item: float(item["top30_recall"])), "top30_recall"),
        "maximum_standardized_elite_regret": _block_identity(max(items, key=lambda item: float(item["standardized_elite_regret"])), "standardized_elite_regret"),
        "maximum_relative_latent_mse": _block_identity(max(items, key=lambda item: float(item["relative_latent_mse"])), "relative_latent_mse"),
        "maximum_normalized_response_mse": _block_identity(max(items, key=lambda item: float(item["normalized_response_mse"])), "normalized_response_mse"),
    }
    return result


def _block_identity(item: Mapping[str, Any], metric: str) -> dict[str, Any]:
    return {
        "episode_id": int(item["episode_id"]),
        "context_id": str(item["context_id"]),
        "stratum": str(item["stratum"]),
        "action_prefix_seed": int(item["action_prefix_seed"]),
        "metric": float(item[metric]),
    }


def summarize_evaluation(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stratum = {
        stratum: _summarize([item for item in blocks if item["stratum"] == stratum])
        for stratum in FRESH_ANCHORS
    }
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in blocks:
        by_episode[int(item["episode_id"])].append(item)
    episodes: dict[str, Any] = {}
    for episode, items in sorted(by_episode.items()):
        if len(items) != len(FRESH_ANCHORS) * 2 or {str(item["stratum"]) for item in items} != set(FRESH_ANCHORS):
            raise ValueError(f"episode {episode} must contain 3 anchors x 2 seeds")
        episodes[str(episode)] = {
            "blocks": len(items),
            **{metric: float(statistics.median(float(item[metric]) for item in items)) for metric in METRICS},
        }
    if len(episodes) != FRESH_EPISODES:
        raise ValueError("fresh evaluation did not cover exactly eight episodes")
    episode_medians = {
        metric: float(statistics.median(float(item[metric]) for item in episodes.values()))
        for metric in METRICS
    }
    return {"overall": _summarize(blocks), "by_stratum": by_stratum, "episodes": episodes, "episode_medians": episode_medians}


def evaluate_arm(reference: Any, official_model: Any, terminal_state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], arm: str, action_seeds: Sequence[int]) -> dict[str, Any]:
    import torch

    student = reference.instantiate_student("balanced_base").to("cuda")
    reference.load_full_state(student, terminal_state)
    student.eval()
    blocks: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            for block, seed in enumerate(action_seeds):
                item = evaluate_block(reference, official_model, student, row, block, int(seed))
                item["arm"] = arm
                blocks.append(item)
    result = {"blocks": blocks, **summarize_evaluation(blocks)}
    del student
    torch.cuda.empty_cache()
    return result


def measure_causality_and_latency(
    reference: Any,
    official_model: Any,
    terminal_state: Mapping[str, Any],
    row: Mapping[str, Any],
    action_seed: int,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    student = reference.instantiate_student("balanced_base").to("cuda")
    reference.load_full_state(student, terminal_state)
    student.eval()
    causality = reference.base.causality_test(
        student, row, int(action_seed), float(evaluation["causality_tolerance"])
    )
    latency_config = evaluation["latency"]
    latency = reference.base.predictor_latency(
        official_model,
        student,
        row,
        warmup=int(latency_config["warmup"]),
        repeats=int(latency_config["repeats"]),
    )
    del student
    torch.cuda.empty_cache()
    return {"causality": causality, "predictor_latency": latency}


def paired_episode_deltas(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    control_blocks = {str(item["pairing_key"]): item for item in control["blocks"]}
    treatment_blocks = {str(item["pairing_key"]): item for item in treatment["blocks"]}
    if len(control_blocks) != 48 or len(treatment_blocks) != 48 or set(control_blocks) != set(treatment_blocks):
        raise ValueError("control/treatment fresh blocks do not pair exactly")
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key, left in control_blocks.items():
        right = treatment_blocks[key]
        if int(left["episode_id"]) != int(right["episode_id"]):
            raise ValueError("paired block episode IDs disagree")
        by_episode[int(left["episode_id"])].append({
            "pairing_key": key,
            **{metric: float(right[metric]) - float(left[metric]) for metric in METRICS},
        })
    if len(by_episode) != FRESH_EPISODES or any(len(items) != 6 for items in by_episode.values()):
        raise ValueError("paired blocks must cover eight episodes with six banks each")
    per_episode = []
    for episode, items in sorted(by_episode.items()):
        per_episode.append({
            "episode_id": episode,
            "paired_blocks": items,
            **{metric: float(statistics.median(item[metric] for item in items)) for metric in METRICS},
        })
    medians = {
        metric: float(statistics.median(item[metric] for item in per_episode))
        for metric in METRICS
    }
    return {
        "paired_unit": "episode after median of six matched block deltas (3 anchors x 2 action-prefix seeds)",
        "per_episode": per_episode,
        "median_delta": medians,
        "strictly_improved_regret_episodes": sum(item["standardized_elite_regret"] < 0 for item in per_episode),
    }


def absolute_predictor_gate(
    evaluation: Mapping[str, Any],
    training: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = evaluation["overall"]
    required = (
        "median_spearman_min",
        "minimum_spearman_min",
        "median_top30_min",
        "minimum_top30_min",
        "median_relative_latent_mse_max",
        "positive_spearman_blocks_min",
        "positive_top30_blocks_min",
        "latency_reduction_min",
    )
    missing = [key for key in required if key not in thresholds]
    if missing:
        raise ValueError(f"absolute predictor gate is missing thresholds: {missing}")
    conditions = {
        "median_spearman_min": metrics["spearman_median"] >= float(thresholds["median_spearman_min"]),
        "minimum_spearman_min": metrics["spearman_minimum"] >= float(thresholds["minimum_spearman_min"]),
        "median_top30_min": metrics["top30_recall_median"] >= float(thresholds["median_top30_min"]),
        "minimum_top30_min": metrics["top30_recall_minimum"] >= float(thresholds["minimum_top30_min"]),
        "median_relative_latent_mse_max": metrics["relative_latent_mse_median"] <= float(thresholds["median_relative_latent_mse_max"]),
        "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(thresholds["positive_spearman_blocks_min"]),
        "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(thresholds["positive_top30_blocks_min"]),
        "training_converged": float(training["last10_to_first_ratio"]) <= LAST10_TO_FIRST_MAX,
        "all_finite": bool(metrics["finite"] and training["finite"]),
        "causality": all(bool(item["passed"]) for item in integrity["causality"].values()),
        "latency_reduction_min": float(integrity["predictor_latency"]["reduction"]) >= float(thresholds["latency_reduction_min"]),
    }
    return {
        "status": "GO" if all(conditions.values()) else "NO-GO",
        "conditions": conditions,
        "metrics": metrics,
        "training_last10_to_first_ratio": float(training["last10_to_first_ratio"]),
        "training_ratio_max": LAST10_TO_FIRST_MAX,
        "causality": integrity["causality"],
        "predictor_latency": integrity["predictor_latency"],
    }


def stratum_protection_gate(evaluation: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    required = ("median_spearman_min", "median_top30_min", "positive_spearman_blocks_min", "positive_top30_blocks_min")
    missing = [key for key in required if key not in thresholds]
    if missing:
        raise ValueError(f"stratum gate is missing thresholds: {missing}")
    conditions: dict[str, Any] = {}
    for stratum, metrics in evaluation["by_stratum"].items():
        conditions[stratum] = {
            "median_spearman_min": metrics["spearman_median"] >= float(thresholds["median_spearman_min"]),
            "median_top30_min": metrics["top30_recall_median"] >= float(thresholds["median_top30_min"]),
            "positive_spearman_blocks_min": metrics["positive_spearman_blocks"] >= int(thresholds["positive_spearman_blocks_min"]),
            "positive_top30_blocks_min": metrics["positive_top30_blocks"] >= int(thresholds["positive_top30_blocks_min"]),
        }
    passed = all(all(values.values()) for values in conditions.values())
    return {"status": "GO" if passed else "NO-GO", "conditions": conditions, "minimum_thresholds_repeated": False}


def mechanism_gate(paired: Mapping[str, Any], thresholds: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    medians = paired["median_delta"]
    conditions = {
        "response_normalized_mse_episode_median_delta_max": medians["normalized_response_mse"] <= float(thresholds["response_normalized_mse_episode_median_delta_max"]),
        "standardized_elite_regret_episode_median_delta_max": medians["standardized_elite_regret"] <= float(thresholds["standardized_elite_regret_episode_median_delta_max"]),
        "episodes_with_strict_regret_improvement_min": int(paired["strictly_improved_regret_episodes"]) >= int(thresholds["episodes_with_strict_regret_improvement_min"]),
        "top30_recall_episode_median_delta_min": medians["top30_recall"] >= float(thresholds["top30_recall_episode_median_delta_min"]),
        "response_floor_active_blocks_max": int(treatment["overall"]["response_floor_active_blocks"]) <= int(thresholds["response_floor_active_blocks_max"]),
    }
    floor_active = not conditions["response_floor_active_blocks_max"]
    return {"status": "INCONCLUSIVE" if floor_active else ("GO" if all(conditions.values()) else "NO-GO"), "conditions": conditions, "paired_median_deltas": medians, "strictly_improved_regret_episodes": paired["strictly_improved_regret_episodes"], "response_floor_active_blocks": int(treatment["overall"]["response_floor_active_blocks"]), "teacher_terminal_contrast_mse_median": float(treatment["overall"]["teacher_terminal_contrast_mse_median"])}


def validate_input_paths(source: Mapping[str, Any]) -> dict[str, Path]:
    paths = {
        "lewm_root": Path(source["lewm_root"]).resolve(),
        "stablewm_home": Path(source["stablewm_home"]).resolve(),
        "dataset": Path(source["dataset_path"]).resolve(),
        "official_checkpoint": Path(source["teacher_checkpoint_path"]).resolve(),
        "interface_probe": Path(source["interface_probe_path"]).resolve(),
        "phase2_manifest": Path(source["phase2_manifest_path"]).resolve(),
        "balanced_rows": Path(source["balanced_rows_path"]).resolve(),
        "reconstructed_checkpoint": Path(source["balanced_base_reference_checkpoint_path"]).resolve(),
    }
    rows = paths["balanced_rows"]
    if rows.name != "prepared_balanced_rows_reconstructed.pt" or SOURCE_JOB not in rows.parts:
        raise ValueError("balanced rows must come from teacher-screening job25213164")
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"required {label} path does not exist: {path}")
    for key, value in source["used_episode_manifests"].items():
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"used episode manifest {key} does not exist: {path}")
    derived_checkpoint = (paths["stablewm_home"] / "pusht" / "lewm_object.ckpt").resolve()
    if not paths["official_checkpoint"].is_file() or paths["official_checkpoint"] != derived_checkpoint:
        raise ValueError("freeze teacher checkpoint must match stablewm_home/pusht/lewm_object.ckpt")
    return paths


def run(freeze_path: Path, output_dir: Path) -> dict[str, Any]:
    host = require_compute_node()
    freeze_path = freeze_path.resolve()
    freeze = load_json(freeze_path)
    contract = validate_freeze(freeze)
    source = contract["source"]
    paths = validate_input_paths(source)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference_modules()
    import torch

    rows_raw = torch.load(paths["balanced_rows"], map_location="cpu", weights_only=False)
    if not isinstance(rows_raw, Sequence):
        raise ValueError("reconstructed balanced rows must be a sequence")
    train_rows = validate_training_rows(rows_raw, paths["phase2_manifest"])
    phase_freeze = load_json(TRANSFER / "LEWM_RECURRENT_STUDENT_FREEZE.json")
    phase_freeze["shared_data_and_schedule"]["context_manifest"]["train_context_target"] = TRAIN_CONTEXTS
    contract_probe = reference.base.load_interface_contract(paths["interface_probe"])
    reference.base.validate_manifest(paths["phase2_manifest"], phase_freeze)

    official_model = reference.base.load_official_checkpoint(paths["stablewm_home"])
    official_model.requires_grad_(False)
    torch.manual_seed(INIT_SEED)
    template = reference.base.make_student("baseline")
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    if sum(value.numel() for value in initial_state.values()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("baseline h256 parameter count drifted")

    control = train_arm(reference, train_rows, initial_state, official_model, "balanced_base")
    control_checkpoint = output_dir / "balanced_base_step3000.pt"
    save_terminal_checkpoint(control_checkpoint, "balanced_base", control["state_dict"], freeze_path, paths["balanced_rows"])
    response = train_arm(reference, train_rows, initial_state, official_model, "terminal_response")
    response_checkpoint = output_dir / "terminal_response_step3000.pt"
    save_terminal_checkpoint(response_checkpoint, "terminal_response", response["state_dict"], freeze_path, paths["balanced_rows"])
    reconstructed_comparison = compare_reconstructed_checkpoint(control["state_dict"], paths["reconstructed_checkpoint"])

    # Freeze and persist selected identities before opening them for fresh inference/scoring.
    used_ids, used_metadata = read_used_episode_ids(source["used_episode_manifests"])
    fresh_ids, selection = select_fresh_episode_ids(
        paths["dataset"],
        paths["phase2_manifest"],
        Path(source["used_episode_manifests"]["seeded_pilot32"]),
        used_ids,
        int(contract["evaluation"]["selection_seed"]),
    )
    selection_path = output_dir / "selection.json"
    selection_record = {
        "schema": SCHEMA + ".fresh-selection",
        "selection": {**selection, "used_episode_manifests": used_metadata},
        "ordered_episode_ids": fresh_ids,
    }
    write_json(selection_path, selection_record)

    fresh_rows, fresh_meta = build_fresh_rows(reference, official_model, freeze, paths["dataset"], fresh_ids)
    action_seeds = contract["action_prefix_seeds"]
    control_eval = evaluate_arm(reference, official_model, control["state_dict"], fresh_rows, "balanced_base", action_seeds)
    response_eval = evaluate_arm(reference, official_model, response["state_dict"], fresh_rows, "terminal_response", action_seeds)
    control_integrity = measure_causality_and_latency(reference, official_model, control["state_dict"], fresh_rows[0], action_seeds[0], contract["evaluation"])
    response_integrity = measure_causality_and_latency(reference, official_model, response["state_dict"], fresh_rows[0], action_seeds[0], contract["evaluation"])
    paired = paired_episode_deltas(control_eval, response_eval)
    gates = contract["gates"]
    absolute_control = absolute_predictor_gate(control_eval, control, gates["absolute_predictor"], control_integrity)
    absolute_response = absolute_predictor_gate(response_eval, response, gates["absolute_predictor"], response_integrity)
    stratum_control = stratum_protection_gate(control_eval, gates["stratum_protection"])
    stratum_response = stratum_protection_gate(response_eval, gates["stratum_protection"])
    mechanism = mechanism_gate(paired, gates["relative_mechanism"], response_eval)
    primary_go = all(value["status"] == "GO" for value in (absolute_response, stratum_response, mechanism))

    result = {
        "schema": SCHEMA + ".result",
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "freeze": str(freeze_path),
        "source": {
            "balanced_rows": str(paths["balanced_rows"]),
            "balanced_rows_source_job": SOURCE_JOB,
            "phase2_manifest": str(paths["phase2_manifest"]),
            "training_teacher_targets_reused_from_cache": True,
            "reconstructed_control_checkpoint": reconstructed_comparison,
            "official_checkpoint": str(paths["official_checkpoint"]),
        },
        "interface_contract": dict(contract_probe.__dict__),
        "training": {
            "shared_contract": {
                "architecture": "LeWMCompactRecurrentTransitionStudent h256",
                "attention": False,
                "goal_input": False,
                "teacher_forcing": False,
                "updates": STEPS,
                "batch_contexts": BATCH_CONTEXTS,
                "candidates_per_context": TRAIN_CANDIDATES,
                "optimizer": dict(contract["training"]["optimizer"]),
                "seeds": dict(contract["training"]["seeds"]),
                "balanced_base_objective": dict(contract["training"]["balanced_base_objective"]),
                "same_initial_state": True,
                "same_training_seed": True,
                "same_context_schedule": True,
                "latent_loss": "frozen mean-normalized horizon-weighted MSE",
                "score_loss_weight": SCORE_LOSS_WEIGHT,
            },
            "response_treatment": {
                "weight": RESPONSE_LOSS_WEIGHT,
                "normalization": "per-bank teacher-centered terminal contrast MSE, clamp_min(floor)",
                "denominator_floor": RESPONSE_DENOMINATOR_FLOOR,
            },
            "control": {key: value for key, value in control.items() if key != "state_dict"},
            "treatment": {key: value for key, value in response.items() if key != "state_dict"},
            "checkpoints": {"balanced_base": str(control_checkpoint), "terminal_response": str(response_checkpoint)},
            "fresh_selection_path": str(selection_path),
        },
        "fresh_selection": {**selection, "used_episode_manifests": used_metadata},
        "fresh_evaluation": {
            "metadata": fresh_meta,
            "control": control_eval,
            "terminal_response": response_eval,
            "integrity": {"balanced_base": control_integrity, "terminal_response": response_integrity},
        },
        "paired_episode_deltas": paired,
        "gates": {
            "absolute_predictor_control": absolute_control,
            "absolute_predictor_treatment": absolute_response,
            "stratum_control": stratum_control,
            "stratum_treatment": stratum_response,
            "relative_mechanism": mechanism,
            "primary": {"status": "GO" if primary_go else "NO-GO", "requires_absolute_and_stratum_and_mechanism": True},
        },
        "scope": {
            "official_cem": "NOT_RUN_BY_SCOPE",
            "planner_viability": "NOT_RUN_BY_SCOPE",
            "closed_loop": "NOT_RUN_BY_SCOPE",
            "claim_boundary": "Fresh predictor-level comparison only; no CEM, planner, or closed-loop claim.",
        },
    }
    write_json(output_dir / "terminal_response_loss_summary.json", result)
    return result


def main() -> int:
    args = parse_args()
    result = run(args.freeze, args.output_dir)
    print(json.dumps({"status": result["status"], "primary_gate": result["gates"]["primary"]["status"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

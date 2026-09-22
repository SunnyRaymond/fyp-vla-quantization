#!/usr/bin/env python3
"""Fixed-observation LeWM CEM elite-boundary pairwise-ranking experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1] / "cem-distribution-distill"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_cem_distribution_distill as cem  # noqa: E402


SCHEMA = "lewm-recurrent-student.cem-boundary-ranking-runner"
TRAIN_CONTEXTS = 512
TRAIN_CANDIDATES = 64
BATCH_CONTEXTS = 8
EXTRA_UPDATES = 1000
TRAINING_SEED = 20300982
SCHEDULE_SEED = 20300984
PAIR_POSITIVE = (20, 30)
PAIR_NEGATIVE = (30, 40)
PAIR_COUNT = 100
PAIR_STD_FLOOR = 1e-6
PAIR_LAMBDA = 0.05
PAIR_TEMPERATURE = 1.0
FRESH_SLICE = (568, 576)
FRESH_SEEDS = (20301001, 20301002)
SELECTION_SEED = 20300903
CEM_CHECKPOINTS = (10, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--mixed-rows", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--temporal-freeze", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("status", "preflight", "run"), default="status")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "lewm-recurrent-student.cem-boundary-ranking-freeze":
        raise ValueError("unexpected boundary-ranking freeze schema")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("freeze must be frozen_before_results")
    training = freeze.get("training", {})
    expected = {
        "contexts": TRAIN_CONTEXTS,
        "batch_contexts": BATCH_CONTEXTS,
        "candidates_per_context": TRAIN_CANDIDATES,
        "extra_updates": EXTRA_UPDATES,
        "training_seed": TRAINING_SEED,
        "context_schedule_seed": SCHEDULE_SEED,
    }
    for key, expected_value in expected.items():
        if int(training.get(key, -1)) != expected_value:
            raise ValueError(f"training contract drifted: {key}")
    pair = freeze.get("boundary_pairwise", {})
    if pair.get("positive_rank_interval") != list(PAIR_POSITIVE) or pair.get("negative_rank_interval") != list(PAIR_NEGATIVE):
        raise ValueError("boundary rank intervals drifted")
    if int(pair.get("cross_pairs_per_context", -1)) != PAIR_COUNT:
        raise ValueError("pair count drifted")
    if not math.isclose(float(pair.get("lambda", -1.0)), PAIR_LAMBDA, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("pair lambda drifted")
    if not math.isclose(float(pair.get("temperature", -1.0)), PAIR_TEMPERATURE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("pair temperature drifted")
    if pair.get("degenerate_std_abort") is not True or pair.get("teacher_cost_detached") is not True:
        raise ValueError("pair std/detach guard drifted")
    cem_cfg = freeze.get("cem_collection", {})
    for key, expected_value in {
        "iterations": cem.CEM_ITERATIONS,
        "saved_checkpoints": list(CEM_CHECKPOINTS),
        "num_samples": cem.NUM_CANDIDATES,
        "topk": cem.ELITE_COUNT,
        "horizon": cem.HORIZON,
        "packed_action_dim": cem.ACTION_DIM,
    }.items():
        if cem_cfg.get(key) != expected_value:
            raise ValueError(f"CEM contract drifted: {key}")
    if cem_cfg.get("candidate_zero_is_pre_update_mu") is not True or cem_cfg.get("std_unbiased") is not True:
        raise ValueError("CEM candidate-zero/std contract drifted")
    if cem_cfg.get("selection_operator") != "torch.topk(cost, k=30, largest=False), matching pinned stable-worldmodel LeWM solver":
        raise ValueError("CEM selection operator drifted")
    fresh = freeze.get("fresh_evaluation", {})
    expected_fresh = {
        "selection_slice": "valid[568:576]",
        "excluded_valid_prefix": 568,
        "selection_seed": SELECTION_SEED,
        "blocks": 48,
        "action_prefix_seeds": list(FRESH_SEEDS),
    }
    for key, expected_value in expected_fresh.items():
        if fresh.get(key) != expected_value:
            raise ValueError(f"fresh contract drifted: {key}")
    if freeze.get("scope", {}).get("official_cem_deployment") != "NOT_RUN_BY_SCOPE":
        raise ValueError("official CEM scope drifted")
    historical = freeze.get("model_and_start", {})
    if historical.get("historical_control_job") != "25239551.pbs101" or historical.get("historical_control_arm") != "treatment":
        raise ValueError("historical control job/arm drifted")
    if historical.get("historical_control_checkpoint_filename") != "treatment_step1000.pt" or historical.get("historical_control_rows_filename") != "cem_distribution_train_rows.pt" or historical.get("historical_control_summary_filename") != "cem_distribution_distill_summary.json":
        raise ValueError("historical control filenames drifted")
    artifact_dir = str(historical.get("historical_control_artifact_dir", "")).replace("\\", "/").rstrip("/")
    for key, filename in (("historical_control_checkpoint_path", "treatment_step1000.pt"), ("historical_control_rows_path", "cem_distribution_train_rows.pt"), ("historical_control_summary_path", "cem_distribution_distill_summary.json")):
        expected_path = f"{artifact_dir}/{filename}"
        if str(historical.get(key, "")).replace("\\", "/").rstrip("/") != expected_path:
            raise ValueError(f"historical control path drifted: {key}")
    return dict(freeze)


def validate_protocol(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    required = ("valid[568:576]", "20300903", "pairwise", "softplus", "historical", "student-driven", "closed-loop")
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"protocol missing frozen terms: {missing}")


def validate_interface(reference: Any, probe: Path) -> dict[str, Any]:
    return cem.validate_interface(reference, probe)


def validate_prior_summary(summary: Mapping[str, Any], prior_checkpoint: Path, mixed_rows: Path, prior_summary: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "COMPLETE":
        raise ValueError("historical CEM summary is not COMPLETE")
    start = summary.get("start_checkpoint", {})
    provenance = start.get("provenance", {})
    if provenance.get("source") != "anchor_aligned_bank_treatment" or int(provenance.get("updates", -1)) != 3000:
        raise ValueError("historical summary start provenance is not frozen anchor step3000")
    historical = freeze["model_and_start"]
    expected_checkpoint = str(historical["historical_control_checkpoint_path"]).replace("\\", "/").rstrip("/")
    expected_rows = str(historical["historical_control_rows_path"]).replace("\\", "/").rstrip("/")
    expected_summary = str(historical["historical_control_summary_path"]).replace("\\", "/").rstrip("/")
    actual_checkpoint = str(prior_checkpoint.resolve()).replace("\\", "/").rstrip("/")
    actual_rows = str(mixed_rows.resolve()).replace("\\", "/").rstrip("/")
    actual_summary = str(prior_summary.resolve()).replace("\\", "/").rstrip("/")
    if actual_checkpoint != expected_checkpoint or actual_rows != expected_rows or actual_summary != expected_summary:
        raise ValueError("requested historical control paths are outside frozen 25239551 artifact directory")
    treatment = summary.get("training", {}).get("treatment", {})
    summary_checkpoint = str(treatment.get("checkpoint", "")).replace("\\", "/").rstrip("/")
    summary_rows = str(treatment.get("rows", "")).replace("\\", "/").rstrip("/")
    if summary_checkpoint != expected_checkpoint or summary_rows != expected_rows:
        raise ValueError("historical summary treatment checkpoint/rows path is not the frozen exact path")
    if int(treatment.get("trace", {}).get("updates", -1)) != EXTRA_UPDATES:
        raise ValueError("historical treatment trace update count drifted")
    bank = summary.get("bank_construction", {})
    if bank.get("teacher_did_not_select_or_update") is not True or int(bank.get("teacher_labelled_tail_candidates", -1)) != TRAIN_CONTEXTS * 32:
        raise ValueError("historical CEM provenance lacks teacher shadow-only marker")
    if bank.get("tail_counts") != {"round_10": 11, "round_20": 11, "round_30": 10}:
        raise ValueError("historical CEM tail counts drifted")
    return {
        "summary_status": summary.get("status"),
        "historical_arm": "treatment",
        "checkpoint_filename": prior_checkpoint.name,
        "rows_filename": mixed_rows.name,
        "start_provenance": provenance,
        "non_concurrent": True,
    }


def load_mixed_rows(path: Path) -> list[Mapping[str, Any]]:
    rows = cem.load_train_rows(path)
    expected_variant = "cem_distribution_mixed_original32_plus_round10_20_30_tail32"
    for ordinal, row in enumerate(rows):
        if row.get("bank_variant") != expected_variant:
            raise ValueError(f"mixed row bank variant drifted at {ordinal}")
        tail = row.get("cem_tail_provenance", {})
        if tail.get("teacher_shadow_only") is not True:
            raise ValueError(f"mixed row teacher provenance drifted at {ordinal}")
    return rows


def load_checkpoint(reference: Any, path: Path, expected_arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    raw = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("provenance"), Mapping):
        raise ValueError(f"checkpoint provenance missing: {path}")
    provenance = dict(raw["provenance"])
    required = {
        "source": "cem_distribution_distill",
        "arm": expected_arm,
        "reconstructed": False,
        "extra_updates": EXTRA_UPDATES,
        "training_seed": TRAINING_SEED,
        "context_schedule_seed": SCHEDULE_SEED,
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(f"checkpoint provenance drifted: {key}")
    start = provenance.get("start_checkpoint", {})
    if start.get("source") != "anchor_aligned_bank_treatment" or int(start.get("provenance", {}).get("updates", -1)) != 3000:
        raise ValueError("historical checkpoint start provenance drifted")
    state = raw.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint state_dict missing")
    copied = {str(key): value.detach().clone() for key, value in state.items()}
    reference.base.make_student("baseline").load_state_dict(copied, strict=True)
    return copied, {"path": str(path.resolve()), "provenance": provenance}


def boundary_pairwise_loss(student_cost: Any, teacher_cost: Any) -> tuple[Any, Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    if student_cost.ndim != 2 or teacher_cost.shape != student_cost.shape or student_cost.shape[1] != TRAIN_CANDIDATES:
        raise ValueError("boundary pair loss expects [batch_contexts,64] cost matrices")
    detached_teacher = teacher_cost.detach()
    raw_std = detached_teacher.std(dim=1, unbiased=False)
    if not bool(torch.isfinite(raw_std).all().item()) or bool((raw_std <= PAIR_STD_FLOOR).any().item()):
        raise FloatingPointError("degenerate or non-finite teacher cost std in boundary pair loss")
    order = torch.argsort(detached_teacher, dim=1, stable=True)
    positive_idx = order[:, PAIR_POSITIVE[0] : PAIR_POSITIVE[1]]
    negative_idx = order[:, PAIR_NEGATIVE[0] : PAIR_NEGATIVE[1]]
    positive = student_cost.gather(1, positive_idx)
    negative = student_cost.gather(1, negative_idx)
    margins = (positive.unsqueeze(2) - negative.unsqueeze(1)) / raw_std.view(-1, 1, 1) / PAIR_TEMPERATURE
    per_context = F.softplus(margins).mean(dim=(1, 2))
    loss = per_context.mean()
    inversion_rate = (margins >= 0.0).float().mean()
    teacher_positive = detached_teacher.gather(1, positive_idx)
    teacher_negative = detached_teacher.gather(1, negative_idx)
    teacher_gap = (teacher_negative.mean(dim=1) - teacher_positive.mean(dim=1)).mean()
    return loss, inversion_rate, teacher_gap, raw_std.min()


def train_treatment(reference: Any, rows: Sequence[Mapping[str, Any]], initial_state: Mapping[str, Any], official: Any) -> dict[str, Any]:
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) != TRAIN_CONTEXTS:
        raise ValueError("mixed rows must contain exactly 512 train contexts")
    student = reference.base.make_student("baseline").to("cuda")
    student.load_state_dict(copy.deepcopy(initial_state), strict=True)
    torch.manual_seed(TRAINING_SEED)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    schedule = torch.Generator(device="cpu").manual_seed(SCHEDULE_SEED)
    history: list[dict[str, Any]] = []
    for step in range(1, EXTRA_UPDATES + 1):
        indices = torch.randperm(len(train_rows), generator=schedule)[:BATCH_CONTEXTS]
        contexts = torch.cat([cem.tensor(train_rows[int(index)]["latent_history"], device="cpu").expand(TRAIN_CANDIDATES, -1, -1) for index in indices], dim=0).to("cuda")
        actions = torch.cat([cem.tensor(train_rows[int(index)]["future_actions"], device="cpu") for index in indices], dim=0).to("cuda")
        targets = torch.cat([cem.tensor(train_rows[int(index)]["teacher_targets"], device="cpu") for index in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        latent_loss, per_horizon = reference.base.recurrent_loss(prediction, targets)
        student_cost = reference.score._student_costs_with_gradient(official, train_rows, indices, prediction)
        teacher_cost = reference.score._effective_teacher_costs(train_rows, indices, "score_distill").detach()
        score_loss = reference.score.score_distill_loss(student_cost, teacher_cost)
        pair_loss, inversion_rate, teacher_gap, raw_std_min = boundary_pairwise_loss(student_cost, teacher_cost)
        total_loss = latent_loss + float(reference.ema.SCORE_LOSS_WEIGHT) * score_loss + PAIR_LAMBDA * pair_loss
        finite = bool(torch.isfinite(total_loss).item() and torch.isfinite(per_horizon).all().item() and torch.isfinite(score_loss).item() and torch.isfinite(pair_loss).item())
        if not finite:
            raise FloatingPointError(f"non-finite boundary-ranking loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        history.append({
            "step": step,
            "weighted_latent_mse": float(latent_loss.detach().cpu()),
            "score_loss": float(score_loss.detach().cpu()),
            "pairwise_loss": float(pair_loss.detach().cpu()),
            "pairwise_inversion_rate": float(inversion_rate.detach().cpu()),
            "teacher_boundary_gap": float(teacher_gap.detach().cpu()),
            "teacher_std_min": float(raw_std_min.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "finite": finite,
        })
        if step % 128 == 0:
            print(json.dumps({"phase": "boundary_pairwise_training", "updates_completed": step, "updates_total": EXTRA_UPDATES}), flush=True)
    last10 = statistics.median(item["weighted_latent_mse"] for item in history[-10:])
    return {
        "student": student,
        "state_dict": {key: value.detach().cpu().clone() for key, value in student.state_dict().items()},
        "trace": {
            "first": history[0],
            "terminal": history[-1],
            "last10_weighted_latent_mse_median": last10,
            "last10_to_first_ratio": float(last10 / max(history[0]["weighted_latent_mse"], 1e-12)),
            "updates": EXTRA_UPDATES,
        },
    }


def select_fresh(dataset: Path, manifest: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    with h5py.File(dataset.resolve(), "r") as handle:
        lengths = [int(value) for value in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= cem.HORIZON * 5 + 1]
    random.Random(SELECTION_SEED).shuffle(valid)
    prefix = [int(row["episode_id"]) for row in manifest["splits"]["heldout"] + manifest["splits"]["train"]]
    if valid[: len(prefix)] != prefix:
        raise ValueError("valid shuffle prefix cannot be reproduced")
    selected = valid[FRESH_SLICE[0] : FRESH_SLICE[1]]
    if len(selected) != 8 or set(selected) & set(valid[: FRESH_SLICE[0]]):
        raise ValueError("fresh valid[568:576] overlaps excluded prefix")
    return selected, {"selection_seed": SELECTION_SEED, "selection_slice": "valid[568:576]", "excluded_valid_prefix": 568, "fresh_episode_ids": selected, "result_dependent_selection": False}


def make_fresh_rows(reference: Any, dataset: Path, manifest: Mapping[str, Any], temporal_freeze: Mapping[str, Any], official: Any, fresh_ids: Sequence[int]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    cem.FRESH_SEEDS = FRESH_SEEDS
    cem.FRESH_SLICE = FRESH_SLICE
    cem.NUM_CANDIDATES = 300
    rows, metadata = cem.make_fresh_rows(reference, dataset, manifest, temporal_freeze, official, fresh_ids)
    return rows, {**metadata, "selection_slice": "valid[568:576]", "selection_seed": SELECTION_SEED, "action_prefix_seeds": list(FRESH_SEEDS)}


def boundary_metric(reference: Any, official: Any, student: Any, row: Mapping[str, Any], bank: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    actions = cem.tensor(bank["actions"])
    target = cem.tensor(bank["teacher_targets"])
    teacher_cost = cem.tensor(bank["teacher_cost"])
    latent = cem.tensor(row["latent_history"])
    with torch.no_grad():
        prediction = student(latent.expand(actions.shape[0], -1, -1), actions)
        student_cost = reference.base._official_objective(official, row, prediction)
    teacher_order = torch.topk(teacher_cost, k=cem.ELITE_COUNT, largest=False, sorted=True).indices
    student_order = torch.topk(student_cost, k=cem.ELITE_COUNT, largest=False, sorted=True).indices
    teacher_top = set(int(value) for value in teacher_order.detach().cpu())
    student_top = set(int(value) for value in student_order.detach().cpu())
    teacher_mean = teacher_cost.index_select(0, teacher_order).mean()
    student_teacher_mean = teacher_cost.index_select(0, student_order).mean()
    regret = student_teacher_mean - teacher_mean
    std = teacher_cost.std(unbiased=False)
    if float(std.detach().cpu()) <= PAIR_STD_FLOOR:
        raise FloatingPointError("degenerate fresh teacher cost std")
    order = torch.argsort(teacher_cost, stable=True)
    positive = order[PAIR_POSITIVE[0] : PAIR_POSITIVE[1]]
    negative = order[PAIR_NEGATIVE[0] : PAIR_NEGATIVE[1]]
    positive_student = student_cost.index_select(0, positive)
    negative_student = student_cost.index_select(0, negative)
    positive_teacher = teacher_cost.index_select(0, positive)
    negative_teacher = teacher_cost.index_select(0, negative)
    margins = (positive_student[:, None] - negative_student[None, :]) / std / PAIR_TEMPERATURE
    return {
        "spearman": float(reference.base._spearman(teacher_cost, student_cost)),
        "top30_overlap": float(len(teacher_top & student_top) / cem.ELITE_COUNT),
        "relative_latent_mse": float(((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).detach().cpu()),
        "teacher_cost_std_population": float(std.detach().cpu()),
        "raw_elite_mean_cost_regret": float(regret.detach().cpu()),
        "standardized_elite_mean_cost_regret": float((regret / std).detach().cpu()),
        "boundary_inversion_rate": float((margins >= 0.0).float().mean().detach().cpu()),
        "teacher_boundary_gap": float((negative_teacher.mean() - positive_teacher.mean()).detach().cpu()),
        "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(student_cost).all().item() and torch.isfinite(teacher_cost).all().item()),
        "recall_at_60": float(len(teacher_top & set(int(value) for value in torch.topk(student_cost, k=60, largest=False).indices.detach().cpu())) / cem.ELITE_COUNT),
        "recall_at_120": float(len(teacher_top & set(int(value) for value in torch.topk(student_cost, k=120, largest=False).indices.detach().cpu())) / cem.ELITE_COUNT),
        "full_elite_containment_at_60": bool(len(teacher_top & set(int(value) for value in torch.topk(student_cost, k=60, largest=False).indices.detach().cpu())) == cem.ELITE_COUNT),
        "full_elite_containment_at_120": bool(len(teacher_top & set(int(value) for value in torch.topk(student_cost, k=120, largest=False).indices.detach().cpu())) == cem.ELITE_COUNT),
    }


def boundary_summary(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {key: [float(item[key]) for item in blocks] for key in ("boundary_inversion_rate", "teacher_boundary_gap")}
    result: dict[str, Any] = {}
    for key, items in values.items():
        result[f"{key}_median"] = float(statistics.median(items))
        result[f"{key}_minimum"] = min(items)
        result[f"{key}_maximum"] = max(items)
    return result


def evaluate_cem(reference: Any, official: Any, student: Any, rows: Sequence[Mapping[str, Any]], banks: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for bank in banks:
        row = rows[int(bank["row_index"])]
        item = boundary_metric(reference, official, student, row, bank)
        item.update({"label": label, "context_id": row["context_id"], "episode_id": int(bank["episode_id"]), "anchor": int(bank["anchor"]), "stratum": bank["stratum"], "action_prefix_seed": int(bank["action_prefix_seed"]), "round": int(bank["round"]), "pairing_key": f"episode={int(bank['episode_id'])}:anchor={bank['stratum']}:seed={int(bank['action_prefix_seed'])}:round={int(bank['round'])}"})
        blocks.append(item)
    return {"label": label, "per_block": blocks, **cem.grouped_summary(blocks), "boundary_summary": boundary_summary(blocks)}


def evaluate_standard(reference: Any, official: Any, student: Any, rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for row in rows:
        for block, seed in enumerate(FRESH_SEEDS):
            bank = {"actions": row["future_actions"][block], "teacher_targets": row["teacher_targets"][block], "teacher_cost": row["teacher_objective"][block]}
            item = boundary_metric(reference, official, student, row, bank)
            item.update({"label": label, "context_id": row["context_id"], "episode_id": int(row["episode_id"]), "anchor": int(row["anchor"]), "stratum": row["stratum"], "action_prefix_seed": int(seed), "round": "standard", "pairing_key": f"episode={int(row['episode_id'])}:anchor={row['stratum']}:seed={int(seed)}"})
            blocks.append(item)
    return {"label": label, "per_block": blocks, **cem.grouped_summary(blocks), "boundary_summary": boundary_summary(blocks)}


def save_checkpoint(path: Path, state: Mapping[str, Any], start_meta: Mapping[str, Any]) -> None:
    import torch

    torch.save({"state_dict": dict(state), "provenance": {"source": "cem_boundary_ranking", "arm": "treatment", "reconstructed": False, "start_checkpoint": start_meta, "extra_updates": EXTRA_UPDATES, "training_seed": TRAINING_SEED, "context_schedule_seed": SCHEDULE_SEED, "pair_lambda": PAIR_LAMBDA, "pair_positive_ranks": list(PAIR_POSITIVE), "pair_negative_ranks": list(PAIR_NEGATIVE)}}, path)


def run(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any) -> dict[str, Any]:
    cem.require_compute_node()
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest_512.resolve())
    temporal_freeze = load_json(args.temporal_freeze.resolve())
    prior_summary = load_json(args.prior_summary.resolve())
    prior_provenance = validate_prior_summary(prior_summary, args.prior_checkpoint, args.mixed_rows, args.prior_summary, freeze)
    mixed_rows = load_mixed_rows(args.mixed_rows)
    start_state, start_meta = cem.load_start_state(reference, args.anchor_checkpoint)
    prior_state, prior_checkpoint_meta = load_checkpoint(reference, args.prior_checkpoint, "treatment")
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.requires_grad_(False)
    treatment = train_treatment(reference, mixed_rows, start_state, official)
    treatment_path = output / "treatment_boundary_step1000.pt"
    save_checkpoint(treatment_path, treatment["state_dict"], start_meta)
    control_student = reference.base.make_student("baseline").to("cuda")
    control_student.load_state_dict(copy.deepcopy(prior_state), strict=True)
    control_student.eval()
    treatment_student = treatment["student"].eval()
    fresh_ids, selection = select_fresh(args.dataset, manifest)
    fresh_rows, fresh_metadata = make_fresh_rows(reference, args.dataset, manifest, temporal_freeze, official, fresh_ids)
    driver = reference.base.make_student("baseline").to("cuda")
    driver.load_state_dict(copy.deepcopy(start_state), strict=True)
    driver.eval()
    fresh_banks = cem.collect_fresh_cem_banks(reference, official, driver, fresh_rows, freeze, output)
    control_cem = evaluate_cem(reference, official, control_student, fresh_rows, fresh_banks, "control_historical_cem_distill_treatment")
    treatment_cem = evaluate_cem(reference, official, treatment_student, fresh_rows, fresh_banks, "treatment_boundary_pairwise")
    control_standard = evaluate_standard(reference, official, control_student, fresh_rows, "control_historical_standard_current_anchor")
    treatment_standard = evaluate_standard(reference, official, treatment_student, fresh_rows, "treatment_boundary_standard_current_anchor")
    gates = cem.paired_gates(control_cem, treatment_cem, control_standard, treatment_standard, freeze)
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_contract": cem.validate_interface(reference, args.interface_probe.resolve()),
        "start_checkpoint": start_meta,
        "historical_control": {"concurrent_retraining": False, "checkpoint": prior_checkpoint_meta, "summary": prior_provenance, "trace": prior_summary["training"]["treatment"]["trace"], "rows": str(args.mixed_rows.resolve())},
        "treatment": {"checkpoint": str(treatment_path), "trace": treatment["trace"], "pairwise": {"lambda": PAIR_LAMBDA, "temperature": PAIR_TEMPERATURE, "positive_ranks": list(PAIR_POSITIVE), "negative_ranks": list(PAIR_NEGATIVE), "pairs_per_context": PAIR_COUNT, "teacher_std_floor": PAIR_STD_FLOOR, "teacher_detached": True}},
        "source": {"manifest_512": str(args.manifest_512.resolve()), "mixed_rows": str(args.mixed_rows.resolve()), "anchor_checkpoint": str(args.anchor_checkpoint.resolve()), "prior_summary": str(args.prior_summary.resolve()), "dataset": str(args.dataset.resolve()), "temporal_freeze": str(args.temporal_freeze.resolve())},
        "fresh_selection": selection,
        "fresh_evaluation": {**fresh_metadata, "action_prefix_seeds": list(FRESH_SEEDS), "cem_banks_shared": True, "rows_generated_on_compute_node": True, "rows_returned": False},
        "cem_trajectory_evaluation": {"control": control_cem, "treatment": treatment_cem},
        "standard_current_anchor_evaluation": {"control": control_standard, "treatment": treatment_standard},
        "gates": gates,
        "stage_b": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "Fixed-observation candidate-ranking evidence only; historical control is non-concurrent reuse; fresh banks are frozen-start student-driven actions with teacher labels, not student on-policy environment data.",
    }
    write_json(output / "cem_boundary_ranking_summary.json", jsonable(summary))
    return summary


def preflight(args: argparse.Namespace, freeze: Mapping[str, Any]) -> dict[str, Any]:
    reference = cem.load_reference()
    interface = validate_interface(reference, args.interface_probe.resolve())
    value = {"schema": SCHEMA, "status": "PASS", "model_work_started": False, "reference_import": "PASS", "interface": interface, "training_contexts": TRAIN_CONTEXTS, "training_candidates": TRAIN_CANDIDATES, "extra_updates": EXTRA_UPDATES, "historical_control_job": "25239551.pbs101", "pairwise": {"positive_ranks": list(PAIR_POSITIVE), "negative_ranks": list(PAIR_NEGATIVE), "pairs_per_context": PAIR_COUNT, "lambda": PAIR_LAMBDA, "temperature": PAIR_TEMPERATURE}, "fresh_selection": "valid[568:576]", "selection_seed": SELECTION_SEED, "fresh_blocks": 48, "pbs_compute_only": True, "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
    write_json(args.output.resolve() / "preflight_status.json", value)
    return value


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "historical_control": "25239551.pbs101 treatment checkpoint", "pairwise": {"positive_ranks": list(PAIR_POSITIVE), "negative_ranks": list(PAIR_NEGATIVE), "pairs_per_context": PAIR_COUNT, "lambda": PAIR_LAMBDA}, "fresh_selection": "valid[568:576]", "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output.resolve() / "run_status.json", value)
        print(json.dumps(value, ensure_ascii=False))
        return 0
    freeze = validate_freeze(load_json(args.freeze.resolve()))
    validate_protocol(args.protocol.resolve())
    if args.mode == "preflight":
        print(json.dumps(preflight(args, freeze), ensure_ascii=False))
        return 0
    reference = cem.load_reference()
    summary = run(args, freeze, reference)
    print(json.dumps({"status": summary["status"], "gates": summary["gates"], "output": str((args.output / "cem_boundary_ranking_summary.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

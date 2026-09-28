"""Fit the frozen full-bank residual ranker and run its one held-out gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn


SEED = 20260925
UPDATES = 1000
BANKS_PER_UPDATE = 8
TEMPERATURE = 0.5
ROUNDS = (10, 20, 30)
TASK_FIELDS = ("selection_order", "episode_idx", "row_index", "start_step")
GATE_CHECKS = (
    "task_identity",
    "initial_simulator_observation",
    "success",
    "actual_env_step_count",
    "env_step_call_count",
    "committed_actions_and_post_step_states",
    "state_trace",
    "solve_count",
    "every_solve_exact",
    "all_required",
)
SOLVE_CHECKS = (
    "rng_state_before_exact",
    "rng_state_after_exact",
    "solver_action_exact",
    "candidate_shape_equal",
    "batch_count_is_one_both",
    "round_arrays_exact",
    "pass",
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=default_root / "FREEZE.json")
    parser.add_argument("--collection-gate", type=Path, required=True)
    parser.add_argument("--train-banks", type=Path, required=True)
    parser.add_argument("--validation-banks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> str:
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
    if host not in nodes:
        raise RuntimeError(f"host {host} is not present in PBS_NODEFILE")
    if any(any(token in node for token in ("login", "submit", "head")) for node in nodes):
        raise RuntimeError("PBS_NODEFILE contains a probable login/submit host")
    return host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def task_tuple(task: dict[str, Any]) -> tuple[int, int, int, int]:
    try:
        return tuple(int(task[key]) for key in TASK_FIELDS)  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid task identity: {task}") from exc


def frozen_tasks(freeze: dict[str, Any]) -> dict[str, list[dict[str, int]]]:
    if freeze.get("schema") != "lewm-pusht-onpolicy-fullbank-ranker-freeze":
        raise ValueError("unexpected FREEZE.json schema")
    if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen_before_collection":
        raise ValueError("FREEZE.json is not the expected pre-collection freeze")
    split = freeze["task_split"]
    result = {name: split[name]["tasks"] for name in ("training", "validation", "untouched")}
    if tuple(map(len, result.values())) != (16, 8, 8):
        raise ValueError("frozen task counts differ from 16/8/8")
    all_ids = [task_tuple(task) for tasks in result.values() for task in tasks]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("task identities overlap across frozen splits")
    episode_ids = [task["episode_idx"] for tasks in result.values() for task in tasks]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode identities overlap across frozen splits")
    return result


def validate_collection_gate(path: Path, train_tasks: list[dict[str, int]]) -> dict[str, Any]:
    gate = read_json(path)
    if gate.get("status") != "PASS" or ("passed" in gate and gate["passed"] is not True):
        raise RuntimeError("first-four-task noninterference collection gate did not pass")
    pairs = gate.get("pairs")
    if gate.get("required_pairs") != 4 or not isinstance(pairs, list) or len(pairs) != 4:
        raise RuntimeError("collection gate must contain exactly four required task pairs")
    if "passed_pairs" in gate and gate["passed_pairs"] != 4:
        raise RuntimeError("collection gate does not report all four pairs passed")
    for index, pair in enumerate(pairs):
        expected_task = task_tuple(train_tasks[index])
        actual_task = task_tuple(pair.get("task", {}))
        if actual_task != expected_task:
            raise RuntimeError(f"collection gate pair {index} has the wrong frozen task")
        comparison = pair.get("comparison", {})
        checks = comparison.get("checks", {})
        if comparison.get("pass") is not True:
            raise RuntimeError(f"collection gate pair {index} did not pass")
        if not isinstance(checks, dict) or any(checks.get(name) is not True for name in GATE_CHECKS):
            raise RuntimeError(f"collection gate pair {index} has a missing or failed required check")
        per_solve = comparison.get("per_solve")
        if not isinstance(per_solve, list) or len(per_solve) != 2:
            raise RuntimeError(f"collection gate pair {index} must cover both expected solves")
        for solve in per_solve:
            if any(solve.get(name) is not True for name in SOLVE_CHECKS):
                raise RuntimeError(f"collection gate pair {index} has a failed solve comparison")
        if [solve.get("sim_steps_before_solve") for solve in per_solve] != [0, 25]:
            raise RuntimeError(f"collection gate pair {index} must compare t0 and t25 solves")
        for name in ("control_dataset_reset_seed", "shadow_dataset_reset_seed"):
            reset = pair.get(name)
            if not isinstance(reset, dict):
                raise RuntimeError(f"collection gate pair {index} is missing {name} evidence")
            effective_seed = reset.get("world_reset_effective_seed")
            if effective_seed not in (42, [42]) or reset.get("world_reset_call_count") != 1:
                raise RuntimeError(f"collection gate pair {index} has an unexpected reset seed/call count")
            if reset.get("dataset_seed_column_present") is not False:
                raise RuntimeError(f"collection gate pair {index} has unexpected dataset seed-column evidence")
            requested_seed = reset.get("world_reset_requested_seed")
            if requested_seed is not None and requested_seed != [None]:
                raise RuntimeError(f"collection gate pair {index} has unexpected requested reset seed")
    return gate


def field_alias(record: dict[str, Any], primary: str, alias: str) -> Any:
    if primary in record and alias in record:
        left, right = record[primary], record[alias]
        same = (torch.equal(torch.as_tensor(left), torch.as_tensor(right))
                if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor)
                else left == right)
        if not same:
            raise ValueError(f"conflicting aliases {primary!r} and {alias!r}")
    if primary in record:
        return record[primary]
    if alias in record:
        return record[alias]
    raise KeyError(primary)


def normalize_record(record: Any, expected_split: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("each bank must be a dictionary")
    if record.get("split") != expected_split:
        raise ValueError(f"bank split must be {expected_split!r}")

    if "task" in record:
        task = record["task"]
        if not isinstance(task, dict):
            raise TypeError("task must be a dictionary")
    else:
        task = {key: record[key] for key in TASK_FIELDS if key in record}
    identity = task_tuple(task)
    if any(key in record and int(record[key]) != identity[i] for i, key in enumerate(TASK_FIELDS)):
        raise ValueError("nested and top-level task identities disagree")

    steps = int(field_alias(record, "sim_steps_before_solve", "replan_step"))
    round_index = int(field_alias(record, "round", "cem_round"))
    if steps not in (0, 25) or round_index not in ROUNDS:
        raise ValueError("bank must use replan step 0/25 and CEM round 10/20/30")
    if "solver_seed" in record and int(record["solver_seed"]) != 42:
        raise ValueError("bank solver_seed must match the frozen per-episode seed 42")
    actions_value = field_alias(record, "actions", "candidates")
    initial = torch.as_tensor(record["initial_emb"], dtype=torch.float32).reshape(-1)
    goal = torch.as_tensor(record["goal_emb"], dtype=torch.float32).reshape(-1)
    actions = torch.as_tensor(actions_value, dtype=torch.float32)
    student = torch.as_tensor(record["student_costs"], dtype=torch.float32).reshape(-1)
    teacher = torch.as_tensor(record["teacher_costs"], dtype=torch.float32).reshape(-1)
    if tuple(initial.shape) != (192,) or tuple(goal.shape) != (192,):
        raise ValueError("initial_emb and goal_emb must each have shape [192]")
    if tuple(actions.shape) != (300, 5, 10):
        raise ValueError("candidate action tensor must have shape [300,5,10]")
    if tuple(student.shape) != (300,) or tuple(teacher.shape) != (300,):
        raise ValueError("student_costs and teacher_costs must each have shape [300]")
    for name, tensor in (("initial_emb", initial), ("goal_emb", goal), ("actions", actions),
                         ("student_costs", student), ("teacher_costs", teacher)):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite values in {name}")
    student_std = student.std(unbiased=False)
    teacher_std = teacher.std(unbiased=False)
    if not bool(torch.isfinite(student_std)) or float(student_std) <= 1e-6:
        raise ValueError("bank student-cost population std must be finite and > 1e-6")
    if not bool(torch.isfinite(teacher_std)) or float(teacher_std) <= 1e-6:
        raise ValueError("teacher-cost population std must be finite and > 1e-6 for regret")
    if "candidate_indices" in record:
        indices = torch.as_tensor(record["candidate_indices"], dtype=torch.int64).reshape(-1)
        if tuple(indices.shape) != (300,) or not torch.equal(indices, torch.arange(300)):
            raise ValueError("candidate_indices must preserve native order 0..299")

    student_z = (student - student.mean()) / student_std
    features = torch.cat((
        initial.expand(300, -1),
        goal.expand(300, -1),
        actions.reshape(300, 50),
        student_z[:, None],
    ), dim=1).contiguous()
    if tuple(features.shape) != (300, 435) or not bool(torch.isfinite(features).all()):
        raise ValueError("constructed 435-dimensional bank features are invalid")
    return {
        "split": expected_split,
        "task": {key: identity[i] for i, key in enumerate(TASK_FIELDS)},
        "task_id": identity,
        "episode_idx": identity[1],
        "steps": steps,
        "round": round_index,
        "features": features,
        "student": student,
        "teacher": teacher,
        "teacher_elite": torch.argsort(teacher)[:30],
        "teacher_std": teacher_std,
    }


def load_banks(path: Path, split: str) -> list[dict[str, Any]]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older cluster PyTorch.
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, list) or not raw:
        raise TypeError(f"{split} bank file must contain a non-empty list")
    banks = [normalize_record(record, split) for record in raw]
    return banks


def validate_bank_set(banks: list[dict[str, Any]], split: str,
                      allowed_tasks: list[dict[str, int]]) -> None:
    allowed = {task_tuple(task) for task in allowed_tasks}
    seen: set[tuple[int, int, int, int, int, int]] = set()
    present: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    task_seen: set[tuple[int, int, int, int]] = set()
    for bank in banks:
        task_id = bank["task_id"]
        if task_id not in allowed:
            raise ValueError(f"{split} bank contains a task outside its frozen split: {task_id}")
        bank_id = (*task_id, bank["steps"], bank["round"])
        if bank_id in seen:
            raise ValueError(f"duplicate bank identity: {bank_id}")
        seen.add(bank_id)
        task_seen.add(task_id)
        present[(*task_id, bank["steps"])].add(bank["round"])
    if task_seen != allowed:
        missing = sorted(allowed - task_seen)
        raise ValueError(f"{split} banks do not cover the exact frozen task set; missing={missing}")
    for task_id in allowed:
        if present.get((*task_id, 0)) != set(ROUNDS):
            raise ValueError(f"{split} task {task_id} must have t0 banks for rounds 10/20/30")
        t25 = present.get((*task_id, 25), set())
        if t25 and t25 != set(ROUNDS):
            raise ValueError(f"{split} task {task_id} has an incomplete t25 round set")


class ResidualRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(435, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def train(model: ResidualRanker, banks: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4,
                                  betas=(0.9, 0.999), eps=1e-8)
    sampler = torch.Generator(device="cpu").manual_seed(SEED)
    loss_sum = 0.0
    first_loss = math.nan
    for update in range(UPDATES):
        indices = torch.randint(len(banks), (BANKS_PER_UPDATE,), generator=sampler).tolist()
        features = torch.stack([banks[i]["features"] for i in indices]).to(device)
        teacher_elite = torch.stack([banks[i]["teacher_elite"] for i in indices]).to(device)
        residual = model(features)
        corrected = features[..., -1] + residual
        log_prob = torch.log_softmax(-corrected / TEMPERATURE, dim=1)
        elite_log_prob = log_prob.gather(1, teacher_elite)
        listwise_ce = -elite_log_prob.mean()
        loss = listwise_ce + 1e-3 * residual.square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at update {update + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if update == 0:
            first_loss = loss_value
        loss_sum += loss_value
        if (update + 1) % 100 == 0:
            print(f"training update {update + 1}/{UPDATES}: loss={loss_value:.7f}", flush=True)
    if update + 1 != UPDATES:
        raise RuntimeError("training did not complete exactly 1000 updates")
    return {"updates": UPDATES, "first_loss": first_loss,
            "mean_loss": loss_sum / UPDATES, "final_loss": loss_value}


@torch.no_grad()
def evaluate(model: ResidualRanker, banks: list[dict[str, Any]], device: torch.device,
             validation_tasks: list[dict[str, int]]) -> dict[str, Any]:
    model.eval()
    deltas_by_episode: dict[int, list[float]] = defaultdict(list)
    deltas_by_round: dict[int, dict[int, list[float]]] = {
        round_index: defaultdict(list) for round_index in ROUNDS
    }
    bank_rows: list[dict[str, Any]] = []
    t25_episodes: set[int] = set()
    for bank in banks:
        features = bank["features"].to(device)
        residual = model(features)
        control_score = features[:, -1]
        treatment_score = control_score + residual
        if not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(treatment_score).all()):
            raise FloatingPointError("non-finite held-out scores")
        control_idx = torch.argsort(control_score)[:30]
        treatment_idx = torch.argsort(treatment_score)[:30]
        teacher_idx = bank["teacher_elite"]
        teacher = bank["teacher"].to(device)
        teacher_std = bank["teacher_std"].to(device)
        teacher_best = teacher[teacher_idx].mean()
        control_regret = (teacher[control_idx].mean() - teacher_best) / teacher_std
        treatment_regret = (teacher[treatment_idx].mean() - teacher_best) / teacher_std
        delta = float((treatment_regret - control_regret).cpu())
        if not math.isfinite(delta):
            raise FloatingPointError("non-finite held-out regret delta")
        episode = bank["episode_idx"]
        round_index = bank["round"]
        deltas_by_episode[episode].append(delta)
        deltas_by_round[round_index][episode].append(delta)
        if bank["steps"] == 25:
            t25_episodes.add(episode)
        bank_rows.append({
            "task": bank["task"], "replan_step": bank["steps"], "round": round_index,
            "control_regret": float(control_regret.cpu()),
            "treatment_regret": float(treatment_regret.cpu()), "delta": delta,
        })

    if set(deltas_by_episode) != {task["episode_idx"] for task in validation_tasks}:
        raise ValueError("held-out metrics do not cover the exact eight frozen validation episodes")
    episode_rows = [
        {"episode_idx": episode, "median_delta": statistics.median(values)}
        for episode, values in sorted(deltas_by_episode.items())
    ]
    episode_deltas = [row["median_delta"] for row in episode_rows]
    primary_median = statistics.median(episode_deltas)
    improved = sum(delta < 0.0 for delta in episode_deltas)
    round_medians: dict[str, float] = {}
    for round_index in ROUNDS:
        episode_medians = [statistics.median(values)
                            for _, values in sorted(deltas_by_round[round_index].items())]
        if len(episode_medians) != len(validation_tasks):
            raise ValueError(f"round {round_index} does not cover all validation episodes")
        round_medians[str(round_index)] = statistics.median(episode_medians)

    coverage = len(t25_episodes)
    performance_pass = (
        primary_median <= -0.05
        and improved >= 5
        and all(value <= 0.0 for value in round_medians.values())
    )
    status = "INCONCLUSIVE" if coverage < 6 else ("GO" if performance_pass else "NO_GO")
    return {
        "status": status,
        "finite_and_paired": True,
        "validation_episode_count": len(validation_tasks),
        "t25_episode_count": coverage,
        "t25_episode_indices": sorted(t25_episodes),
        "primary_episode_median_delta": primary_median,
        "episodes_with_strict_improvement": improved,
        "round_median_delta": round_medians,
        "gates": {
            "coverage_pass": coverage >= 6,
            "primary_pass": primary_median <= -0.05,
            "episode_improvement_pass": improved >= 5,
            "rounds_pass": all(value <= 0.0 for value in round_medians.values()),
        },
        "episode_rows": episode_rows,
        "bank_rows": bank_rows,
    }


def main() -> None:
    args = parse_args()
    host = require_compute_node()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("frozen run requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")

    freeze = read_json(args.freeze)
    task_split = frozen_tasks(freeze)
    collection_gate = validate_collection_gate(args.collection_gate, task_split["training"])
    output_dir = args.output_dir.resolve()
    collection_root = args.collection_gate.resolve().parent
    if output_dir == collection_root or collection_root not in output_dir.parents:
        raise ValueError("--output-dir must be a new child directory under the collection output root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"ranker output directory must be new and empty: {output_dir}")
    train_banks = load_banks(args.train_banks, "train")
    validate_bank_set(train_banks, "train", task_split["training"])
    task_order = {task_tuple(task): index for index, task in enumerate(task_split["training"])}
    train_banks.sort(key=lambda bank: (task_order[bank["task_id"]], bank["steps"], bank["round"]))

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = ResidualRanker()
    training = train(model, train_banks, device)
    if training["updates"] != UPDATES:
        raise RuntimeError("validation cannot begin before exactly 1000 training updates")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "residual_ranker.pt"
    torch.save({
        "schema": "lewm-pusht-onpolicy-fullbank-residual-ranker-checkpoint",
        "schema_version": 1,
        "seed": SEED,
        "updates": UPDATES,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": [435, 256, 256, 1],
        "score": "bank_z_student_cost + residual",
    }, checkpoint_path)
    print("training complete; loading held-out validation banks once", flush=True)

    # Deliberately first touch the held-out file after the fixed training budget.
    validation_banks = load_banks(args.validation_banks, "validation")
    validate_bank_set(validation_banks, "validation", task_split["validation"])
    validation_order = {task_tuple(task): index for index, task in enumerate(task_split["validation"])}
    validation_banks.sort(key=lambda bank: (validation_order[bank["task_id"]], bank["steps"], bank["round"]))
    if {bank["task_id"] for bank in train_banks} & {bank["task_id"] for bank in validation_banks}:
        raise ValueError("train and validation task identities overlap")
    if {bank["task_id"] for bank in validation_banks} & {task_tuple(task) for task in task_split["untouched"]}:
        raise ValueError("validation file contains an untouched task")

    gate_result = evaluate(model, validation_banks, device, task_split["validation"])
    gate_result.update({
        "schema": "lewm-pusht-onpolicy-fullbank-residual-ranker-gate",
        "schema_version": 1,
        "training_updates": UPDATES,
        "collection_gate_status": collection_gate.get("status", "PASS"),
    })
    summary = {
        "schema": "lewm-pusht-onpolicy-fullbank-residual-ranker-result",
        "schema_version": 1,
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "status": gate_result["status"],
        "training": training,
        "train_bank_count": len(train_banks),
        "validation_bank_count": len(validation_banks),
        "validation_episode_count": gate_result["validation_episode_count"],
        "t25_episode_count": gate_result["t25_episode_count"],
        "primary_episode_median_delta": gate_result["primary_episode_median_delta"],
        "episodes_with_strict_improvement": gate_result["episodes_with_strict_improvement"],
        "round_median_delta": gate_result["round_median_delta"],
        "checkpoint": checkpoint_path.name,
        "claim_boundary": "predictor-level ranking on baseline-generated held-out full banks only",
    }
    atomic_json(output_dir / "gate.json", gate_result)
    atomic_json(output_dir / "ranker_summary.json", summary)
    print(f"held-out predictor gate: {gate_result['status']}", flush=True)


if __name__ == "__main__":
    main()

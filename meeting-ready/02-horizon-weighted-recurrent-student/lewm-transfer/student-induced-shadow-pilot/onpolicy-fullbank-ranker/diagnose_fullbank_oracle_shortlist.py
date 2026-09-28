"""Read-only, episode-aggregated oracle-shortlist diagnostics for frozen full banks."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


SOURCE_JOB_ID = "25538135.pbs101"
K_VALUES = (30, 60, 120, 300)
ROUNDS = (10, 20, 30)
TASK_FIELDS = ("selection_order", "episode_idx", "row_index", "start_step")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    source = root / "artifacts" / SOURCE_JOB_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=root / "FREEZE.json")
    parser.add_argument("--train-banks", type=Path, default=source / "train_banks.pt")
    parser.add_argument("--validation-banks", type=Path, default=source / "validation_banks.pt")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_cpu_compute_node() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("requires PBS_JOBID and a real PBS_NODEFILE")
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

    ngpus = os.environ.get("PBS_NGPUS", "0").strip()
    if ngpus.isdigit() and int(ngpus) > 0:
        raise RuntimeError(f"refusing GPU allocation PBS_NGPUS={ngpus}")
    gpufile = os.environ.get("PBS_GPUFILE", "").strip()
    if gpufile and Path(gpufile).is_file() and Path(gpufile).stat().st_size > 0:
        raise RuntimeError("refusing a non-empty PBS_GPUFILE")
    resources = os.environ.get("PBS_RESOURCE_LIST", "")
    if re.search(r"(?:^|[:,])\s*ngpus\s*=\s*[1-9]\d*", resources, flags=re.IGNORECASE):
        raise RuntimeError("refusing an ngpus PBS resource request")
    return job_id, host


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def task_id(task: dict[str, Any]) -> tuple[int, int, int, int]:
    try:
        return tuple(int(task[key]) for key in TASK_FIELDS)  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid task identity: {task}") from exc


def load_task_split(path: Path) -> dict[str, list[dict[str, int]]]:
    freeze = read_json(path)
    if freeze.get("schema") != "lewm-pusht-onpolicy-fullbank-ranker-freeze":
        raise ValueError("unexpected FREEZE.json schema")
    if freeze.get("schema_version") != 1:
        raise ValueError("unsupported freeze version")
    split = freeze["task_split"]
    tasks = {name: split[name]["tasks"] for name in ("training", "validation", "untouched")}
    if tuple(map(len, tasks.values())) != (16, 8, 8):
        raise ValueError("frozen task split must contain 16/8/8 tasks")
    if len({task_id(task) for group in tasks.values() for task in group}) != 32:
        raise ValueError("frozen task identities overlap")
    return tasks


def tensor_float(record: dict[str, Any], key: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = torch.as_tensor(record[key], dtype=torch.float32, device="cpu")
    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{key} must be finite with shape {shape}")
    return value


def bank_identity(record: dict[str, Any], expected_split: str) -> tuple[tuple[int, ...], int, int]:
    if record.get("split") != expected_split:
        raise ValueError(f"expected split {expected_split!r}")
    nested = record.get("task")
    task = nested if isinstance(nested, dict) else record
    identity = task_id(task)
    for index, key in enumerate(TASK_FIELDS):
        if key in record and int(record[key]) != identity[index]:
            raise ValueError("nested and top-level task identities disagree")
    steps = int(record.get("replan_step", record.get("sim_steps_before_solve", -1)))
    round_index = int(record.get("cem_round", record.get("round", -1)))
    if steps not in (0, 25) or round_index not in ROUNDS:
        raise ValueError("bank has an unexpected replan step or CEM round")
    if int(record.get("solver_seed", 42)) != 42:
        raise ValueError("bank solver seed differs from frozen seed 42")
    if "candidates" in record:
        candidates = record["candidates"]
    elif "actions" in record:
        candidates = record["actions"]
    else:
        raise KeyError("candidate action tensor is missing")
    tensor_float({"candidates": candidates}, "candidates", (300, 5, 10))
    tensor_float(record, "initial_emb", (192,))
    tensor_float(record, "goal_emb", (192,))
    student = tensor_float(record, "student_costs", (300,))
    teacher = tensor_float(record, "teacher_costs", (300,))
    if float(student.std(unbiased=False)) <= 1e-6 or float(teacher.std(unbiased=False)) <= 1e-6:
        raise ValueError("student and teacher population std must be > 1e-6")
    if "candidate_indices" in record:
        indices = torch.as_tensor(record["candidate_indices"], dtype=torch.int64, device="cpu").reshape(-1)
        if tuple(indices.shape) != (300,) or not torch.equal(indices, torch.arange(300)):
            raise ValueError("candidate_indices must be native order 0..299")
    return identity, steps, round_index


def load_split(path: Path, split: str, frozen_tasks: list[dict[str, int]]) -> list[dict[str, Any]]:
    try:
        records = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older cluster PyTorch compatibility.
        records = torch.load(path, map_location="cpu")
    if not isinstance(records, list) or not records:
        raise TypeError(f"{split} bank file must contain a non-empty list")
    expected = {task_id(task) for task in frozen_tasks}
    seen: set[tuple[tuple[int, ...], int, int]] = set()
    by_task_step: dict[tuple[tuple[int, ...], int], set[int]] = defaultdict(set)
    banks: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("every full bank must be a dictionary")
        identity, steps, round_index = bank_identity(record, split)
        if identity not in expected:
            raise ValueError(f"{split} bank task is outside the frozen split: {identity}")
        key = (identity, steps, round_index)
        if key in seen:
            raise ValueError(f"duplicate bank identity {key}")
        seen.add(key)
        by_task_step[(identity, steps)].add(round_index)
        banks.append({
            "split": split,
            "task": {name: identity[i] for i, name in enumerate(TASK_FIELDS)},
            "task_id": identity,
            "episode_idx": identity[1],
            "steps": steps,
            "round": round_index,
            "student": torch.as_tensor(record["student_costs"], dtype=torch.float32, device="cpu"),
            "teacher": torch.as_tensor(record["teacher_costs"], dtype=torch.float32, device="cpu"),
        })
    if {bank["task_id"] for bank in banks} != expected:
        raise ValueError(f"{split} banks do not cover exactly the frozen tasks")
    for identity in expected:
        if by_task_step.get((identity, 0)) != set(ROUNDS):
            raise ValueError(f"{split} task {identity} lacks complete t0 round banks")
        t25 = by_task_step.get((identity, 25), set())
        if t25 and t25 != set(ROUNDS):
            raise ValueError(f"{split} task {identity} has incomplete t25 rounds")
    return banks


def bank_metrics(bank: dict[str, Any]) -> dict[int, dict[str, float | int]]:
    student = bank["student"]
    teacher = bank["teacher"]
    student_order = torch.argsort(student)
    teacher_order = torch.argsort(teacher)
    teacher_elite = set(teacher_order[:30].tolist())
    best_teacher_cost = teacher[teacher_order[:30]].mean()
    teacher_std = teacher.std(unbiased=False)
    student_top30 = student_order[:30]
    student_regret = float((teacher[student_top30].mean() - best_teacher_cost) / teacher_std)
    output: dict[int, dict[str, float | int]] = {}
    for k in K_VALUES:
        shortlist = student_order[:k]
        shortlist_elite_order = torch.argsort(teacher[shortlist])[:30]
        oracle_top30 = shortlist[shortlist_elite_order]
        oracle_regret = float((teacher[oracle_top30].mean() - best_teacher_cost) / teacher_std)
        recall = len(teacher_elite.intersection(shortlist.tolist())) / 30.0
        delta = oracle_regret - student_regret
        if not all(math.isfinite(value) for value in (student_regret, oracle_regret, delta, recall)):
            raise FloatingPointError("non-finite oracle-shortlist metric")
        output[k] = {
            "student_top30_regret": student_regret,
            "oracle_shortlist_regret": oracle_regret,
            "delta_vs_student_top30": delta,
            "teacher_elite_recall_at_k": recall,
            "teacher_elite_recovered": len(teacher_elite.intersection(shortlist.tolist())),
        }
    return output


def median_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    keys = ("student_top30_regret", "oracle_shortlist_regret",
            "delta_vs_student_top30", "teacher_elite_recall_at_k")
    return {key: statistics.median(float(row[key]) for row in rows) for key in keys}


def summarize(banks: list[dict[str, Any]], frozen_tasks: dict[str, list[dict[str, int]]],
              job_id: str, host: str) -> dict[str, Any]:
    by_episode: dict[tuple[str, int, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    by_round_step: dict[tuple[str, int, int, int], dict[int, dict[str, float | int]]] = defaultdict(dict)
    bank_counts: dict[str, int] = defaultdict(int)

    for bank in banks:
        split, episode = bank["split"], bank["episode_idx"]
        bank_counts[split] += 1
        for k, metrics in bank_metrics(bank).items():
            by_episode[(split, k, bank["task_id"])].append(metrics)
            by_round_step[(split, k, bank["steps"], bank["round"])][episode] = metrics

    episode_rows: list[dict[str, Any]] = []
    for (split, k, task), rows in sorted(by_episode.items()):
        metrics = median_metrics(rows)
        episode_rows.append({
            "split": split, "source_episode_idx": task[1], "task": {
                name: task[i] for i, name in enumerate(TASK_FIELDS)
            }, "k": k, "bank_count": len(rows), **metrics,
        })

    round_step_rows: list[dict[str, Any]] = []
    for (split, k, steps, round_index), episode_map in sorted(by_round_step.items()):
        values = list(episode_map.values())
        round_step_rows.append({
            "split": split, "k": k, "replan_step": steps, "cem_round": round_index,
            "source_episode_count": len(episode_map), **median_metrics(values),
        })

    split_summaries: list[dict[str, Any]] = []
    for split, tasks in (("train", frozen_tasks["training"]),
                         ("validation", frozen_tasks["validation"])):
        expected_episode_count = len(tasks)
        for k in K_VALUES:
            rows = [row for row in episode_rows if row["split"] == split and row["k"] == k]
            if len(rows) != expected_episode_count:
                raise ValueError(f"{split} K={k} did not preserve all source episodes")
            split_summaries.append({
                "split": split, "k": k, "bank_count": bank_counts[split],
                "source_episode_count": len(rows),
                **median_metrics(rows),
            })

    return {
        "schema": "lewm-pusht-fullbank-oracle-shortlist-posthoc-v1",
        "report_kind": "POSTHOC_MECHANISM_DIAGNOSTIC",
        "pbs_job_id": job_id,
        "compute_host": host,
        "source_job_id": SOURCE_JOB_ID,
        "k_values": list(K_VALUES),
        "split_summary": split_summaries,
        "episode_summary": episode_rows,
        "round_step_summary": round_step_rows,
        "validation_use": (
            "The validation split was already used by the frozen residual-ranker gate. "
            "These results are descriptive mechanism evidence only; they are not an "
            "independent gate, a tuning signal, or a new held-out evaluation."
        ),
        "decision": None,
        "claim_boundary": (
            "Oracle best-30 reranking is a post-hoc shortlist ceiling on baseline-generated "
            "banks. It does not establish an implementable scorer, treatment CEM behavior, "
            "planner deployment, closed-loop success, or speedup."
        ),
    }


def main() -> None:
    args = parse_args()
    job_id, host = require_cpu_compute_node()
    freeze_tasks = load_task_split(args.freeze)
    train_path = args.train_banks.resolve()
    validation_path = args.validation_banks.resolve()
    output_path = args.output.resolve()
    source_root = train_path.parent
    if validation_path.parent != source_root or train_path == validation_path:
        raise ValueError("train and validation bank files must share one immutable source directory")
    if source_root == output_path.parent or source_root in output_path.parents:
        raise ValueError("posthoc output must be outside the source artifact directory")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite posthoc output: {output_path}")

    train_banks = load_split(train_path, "train", freeze_tasks["training"])
    validation_banks = load_split(validation_path, "validation", freeze_tasks["validation"])
    if {bank["task_id"] for bank in train_banks} & {bank["task_id"] for bank in validation_banks}:
        raise ValueError("training and validation task identities overlap")
    if {bank["task_id"] for bank in validation_banks} & {
        task_id(task) for task in freeze_tasks["untouched"]
    }:
        raise ValueError("validation banks contain an untouched task")

    report = summarize(train_banks + validation_banks, freeze_tasks, job_id, host)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output_path)
    print(f"wrote read-only posthoc diagnostic: {output_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect frozen-student full CEM banks for the on-policy ranker experiment.

This entry point is intended to run only inside the PBS compute allocation
provided by the experiment wrapper. It reuses the validated pilot episode
runner and adds a small read-only context capture around its cost router.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE.parent
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

import run_student_induced_shadow_pilot as pilot  # noqa: E402


ROUNDS = (10, 20, 30)
BANK_SCHEMA = "lewm-pusht-onpolicy-fullbank-bank-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    parser.add_argument("--pilot-freeze", type=Path, default=PILOT_DIR / "PILOT_FREEZE.json")
    parser.add_argument("--seeded-pilot-freeze", type=Path, default=PILOT_DIR / "SEEDED_PILOT_FREEZE.json")
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-job-id")
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--official-cem-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    return parser.parse_args()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_tasks(rows: Any) -> list[dict[str, int]]:
    if not isinstance(rows, list):
        raise TypeError("frozen task list is missing")
    return [pilot.normalize_task(row) for row in rows]


def validate_fullbank_freeze(
    freeze: dict[str, Any],
    seeded_freeze: dict[str, Any],
    pilot_freeze: dict[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, int]], list[dict[str, int]]]:
    if freeze.get("schema") != "lewm-pusht-onpolicy-fullbank-ranker-freeze" or freeze.get("schema_version") != 1:
        raise RuntimeError("full-bank freeze schema identity mismatch")
    if freeze.get("status") != "frozen_before_collection":
        raise RuntimeError("full-bank freeze is not frozen before collection")

    splits = freeze.get("task_split", {})
    training = normalize_tasks(splits.get("training", {}).get("tasks"))
    validation = normalize_tasks(splits.get("validation", {}).get("tasks"))
    untouched = normalize_tasks(splits.get("untouched", {}).get("tasks"))
    expected_train = normalize_tasks(seeded_freeze.get("collection_tasks", []))
    seeded_holdout = normalize_tasks(seeded_freeze.get("reserved_holdout_tasks", []))
    if len(training) != 16 or training != expected_train:
        raise RuntimeError("full-bank training tasks differ from seeded collection_tasks")
    if len(validation) != 8 or validation != seeded_holdout[:8]:
        raise RuntimeError("full-bank validation tasks are not the first eight reserved holdout tasks")
    if len(untouched) != 8 or untouched != seeded_holdout[8:]:
        raise RuntimeError("full-bank untouched tasks differ from the remaining reserved holdout tasks")
    if set(row["episode_idx"] for row in training + validation + untouched) != set(
        row["episode_idx"] for row in expected_train + seeded_holdout
    ):
        raise RuntimeError("full-bank task split has missing or extra source episodes")

    provenance = freeze.get("provenance", {})
    if provenance.get("parent_freeze_schema") != seeded_freeze.get("schema"):
        raise RuntimeError("full-bank parent freeze schema differs from the seeded pilot")
    if provenance.get("selection_manifest_job_id") != pilot_freeze.get("selection_manifest", {}).get("pbs_job_id"):
        raise RuntimeError("full-bank selection manifest provenance differs from PILOT_FREEZE")
    base_student = provenance.get("base_student", {})
    if base_student.get("checkpoint_path") != seeded_freeze.get("student_checkpoint", {}).get("path"):
        raise RuntimeError("full-bank student checkpoint differs from the seeded pilot checkpoint")
    if base_student.get("architecture") != "LeWMCompactRecurrentTransitionStudent h256":
        raise RuntimeError("full-bank student architecture drifted")

    collection = freeze.get("collection", {})
    reset = collection.get("reset", {})
    if (
        reset.get("requested_dataset_seed") is not None
        or reset.get("effective_seed") != 42
        or reset.get("world_reset_calls_per_episode") != 1
        or reset.get("native_solver_seed") != 42
    ):
        raise RuntimeError("full-bank reset/seed protocol drifted")
    env = collection.get("environment_protocol", {})
    if (
        env.get("goal_offset_steps") != 25
        or env.get("eval_budget") != 50
        or env.get("max_episode_steps") != 100
        or env.get("expected_replan_steps") != [0, 25]
        or env.get("receding_horizon") != 5
        or env.get("action_block") != 5
    ):
        raise RuntimeError("full-bank environment/replan protocol drifted")
    cem = collection.get("cem", {})
    if (
        cem.get("batch_size") != 1
        or cem.get("num_samples") != 300
        or cem.get("topk") != 30
        or cem.get("iterations") != 30
        or cem.get("captured_rounds_1_indexed") != list(ROUNDS)
    ):
        raise RuntimeError("full-bank CEM capture protocol drifted")
    if collection.get("policy") != "Frozen base student generates every candidate bank and all real actions; the residual MLP is absent during collection.":
        raise RuntimeError("full-bank collection policy drifted")
    return training, validation, untouched


def prepare_pilot_args(args: argparse.Namespace, selection_job_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        lewm_root=args.lewm_root,
        stablewm_root=args.stablewm_root,
        stablewm_home=args.stablewm_home,
        control_root=args.control_root,
        baseline_dir=args.baseline_dir,
        official_cem_dir=args.official_cem_dir,
        adaptive_dir=args.adaptive_dir,
        interface_probe=args.interface_probe,
        freeze=args.pilot_freeze,
        selection_manifest=args.selection_manifest,
        selection_job_id=selection_job_id,
        cache_root=args.cache_root,
        output=args.output_dir,
    )


def validate_runtime_config(cfg: Any) -> None:
    expected = {
        "max_episode_steps": 100,
        "world_num_envs": 50,
        "eval_budget": 50,
        "goal_offset_steps": 25,
        "num_samples": 300,
        "topk": 30,
        "n_steps": 30,
        "seed": 42,
        "batch_size": 1,
    }
    observed = {
        "max_episode_steps": int(cfg.world.max_episode_steps),
        "world_num_envs": int(cfg.world.num_envs),
        "eval_budget": int(cfg.eval.eval_budget),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "num_samples": int(cfg.solver.num_samples),
        "topk": int(cfg.solver.topk),
        "n_steps": int(cfg.solver.n_steps),
        "seed": int(cfg.solver.seed),
        "batch_size": int(cfg.solver.batch_size),
    }
    if observed != expected:
        raise RuntimeError(f"pinned evaluator/CEM config drifted: {observed}")


def setup_runtime(
    args: argparse.Namespace,
    freeze: dict[str, Any],
    seeded_freeze: dict[str, Any],
    pilot_freeze: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    selection_job_id = str(args.selection_job_id or manifest.get("pbs_job_id", ""))
    if not selection_job_id:
        raise RuntimeError("selection manifest lacks its frozen PBS job ID")
    pilot_args = prepare_pilot_args(args, selection_job_id)
    collection, gate_tasks, student_binding = pilot.validate_freeze_and_manifest(
        pilot_freeze, manifest, pilot_args
    )
    pilot.validate_seeded_pilot_freeze(seeded_freeze, pilot_freeze, manifest, collection, gate_tasks)
    training, validation, untouched = validate_fullbank_freeze(freeze, seeded_freeze, pilot_freeze)

    np, spt, swm, torch, transforms, modules = pilot.load_modules(pilot_args)
    baseline, adaptive, old_router, load_official_checkpoint, preprocessing, instantiate = modules
    cfg = baseline.compose_pinned_config(args.lewm_root.resolve(strict=True))
    validate_runtime_config(cfg)

    baseline_binding = pilot_freeze["upstream_baseline"]
    if args.lewm_root.resolve(strict=True) != Path(str(baseline_binding["lewm_root"])).resolve(strict=True):
        raise RuntimeError("LeWM source root differs from the frozen upstream baseline")
    if args.stablewm_root.resolve(strict=True) != Path(str(baseline_binding["stable_worldmodel_root"])).resolve(strict=True):
        raise RuntimeError("stable-worldmodel source root differs from the frozen upstream baseline")

    dataset_path = Path(str(baseline_binding["dataset_path"])).resolve(strict=True)
    staged_dataset = args.stablewm_home.resolve(strict=True) / "pusht_expert_train.h5"
    teacher_path = args.stablewm_home.resolve(strict=True) / "pusht" / "lewm_object.ckpt"
    if dataset_path != staged_dataset.resolve(strict=True):
        raise RuntimeError("dataset path differs from the frozen official LeWM dataset")
    if teacher_path.resolve(strict=True) != Path(str(baseline_binding["teacher_checkpoint_path"])).resolve(strict=True):
        raise RuntimeError("teacher checkpoint differs from the frozen official LeWM checkpoint")
    if dataset_path.stat().st_size != int(baseline_binding["dataset_size_bytes"]):
        raise RuntimeError("staged HDF5 size differs from the frozen baseline record")
    if teacher_path.stat().st_size != int(baseline_binding["teacher_checkpoint_size_bytes"]):
        raise RuntimeError("staged teacher size differs from the frozen baseline record")
    if (args.cache_root / "datasets" / staged_dataset.name).resolve(strict=True) != staged_dataset:
        raise RuntimeError("PBS dataset cache link does not resolve to the frozen HDF5")
    if (args.cache_root / "pusht" / teacher_path.name).resolve(strict=True) != teacher_path:
        raise RuntimeError("PBS teacher cache link does not resolve to the frozen checkpoint")

    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=args.cache_root.resolve()
    )
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    first_gate_tasks = sorted(training[:4], key=lambda row: row["row_index"])
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    gate_rows = dataset.get_row_data(np.asarray([row["row_index"] for row in first_gate_tasks], dtype=np.int64))
    for index, task in enumerate(first_gate_tasks):
        if int(gate_rows[episode_column][index]) != task["episode_idx"] or int(gate_rows["step_idx"][index]) != task["start_step"]:
            raise RuntimeError(f"frozen gate task row identity mismatch at selection order {task['selection_order']}")

    schedule, _, reference, _ = old_router.load_modules(
        args.control_root.resolve(strict=True), args.lewm_root.resolve(strict=True)
    )
    schedule.validate_interface(reference, args.interface_probe.resolve(strict=True))
    official = load_official_checkpoint(args.cache_root.resolve(strict=True))
    official.interpolate_pos_encoding = True
    official.eval()
    official.requires_grad_(False)
    student_checkpoint = Path(str(student_binding["path"])).resolve(strict=True)
    if not student_checkpoint.is_file():
        raise FileNotFoundError(student_checkpoint)
    if student_checkpoint != Path(str(freeze["provenance"]["base_student"]["checkpoint_path"])).resolve(strict=True):
        raise RuntimeError("loaded student checkpoint path differs from the full-bank freeze")
    student, student_metadata = adaptive.load_main_student(reference, student_checkpoint)
    for key in ("source", "arm", "extra_updates"):
        if student_metadata["provenance"].get(key) != student_binding["provenance"].get(key):
            raise RuntimeError(f"loaded student checkpoint provenance differs from freeze: {key}")
    if int(student_metadata["provenance"].get("extra_updates", -1)) != 1000:
        raise RuntimeError("loaded student is not the frozen treatment_step1000 checkpoint")
    if int(sum(parameter.numel() for parameter in student.parameters())) != int(
        freeze["provenance"]["base_student"]["parameter_count"]
    ):
        raise RuntimeError("loaded student parameter count differs from the full-bank freeze")
    student = student.to("cuda").eval()
    student.requires_grad_(False)

    return {
        "np": np,
        "torch": torch,
        "spt": spt,
        "swm": swm,
        "transforms": transforms,
        "baseline": baseline,
        "cfg": cfg,
        "dataset": dataset,
        "process": process,
        "official": official,
        "reference": reference,
        "student": student,
        "old_router": old_router,
        "instantiate": instantiate,
        "training": training,
        "validation": validation,
        "untouched": untouched,
        "pilot_args": pilot_args,
    }


def validate_task_rows(runtime: dict[str, Any], tasks: list[dict[str, int]]) -> None:
    np = runtime["np"]
    dataset = runtime["dataset"]
    row_order = sorted(tasks, key=lambda row: row["row_index"])
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    rows = dataset.get_row_data(np.asarray([row["row_index"] for row in row_order], dtype=np.int64))
    for index, task in enumerate(row_order):
        if int(rows[episode_column][index]) != task["episode_idx"] or int(rows["step_idx"][index]) != task["start_step"]:
            raise RuntimeError(f"frozen task row identity mismatch at selection order {task['selection_order']}")


def capture_vector(value: Any, torch: Any, label: str, last_history_item: bool) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"cost router did not expose tensor {label}")
    if value.shape[-1] != 192:
        raise RuntimeError(f"expected {label} feature dimension 192, got {tuple(value.shape)}")
    tensor = value.detach()
    if last_history_item:
        tensor = tensor[0, 0]
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1, 192)[-1]
    else:
        tensor = tensor.reshape(-1, 192)[0]
    tensor = tensor.reshape(192).to(device="cpu")
    if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{label} must be finite CPU float32 data")
    return tensor.clone().contiguous()


def run_captured_episode(runtime: dict[str, Any], task: dict[str, int], shadow_enabled: bool) -> dict[str, Any]:
    torch = runtime["torch"]
    captured_solves: list[dict[int, dict[str, Any]]] = []
    base_router = pilot.ShadowRouter

    class CapturingShadowRouter(base_router):
        def begin_solve(self) -> None:
            super().begin_solve()
            captured_solves.append({})

        def get_cost(self, info_dict: dict[str, Any], action_candidates: Any):
            result = super().get_cost(info_dict, action_candidates)
            round_number = self.round_index
            if round_number in ROUNDS:
                if not captured_solves:
                    raise RuntimeError("context capture occurred outside a CEM solve")
                captured_solves[-1][round_number] = {
                    "initial_emb": capture_vector(info_dict.get("emb"), torch, "initial_emb", True),
                    "goal_emb": capture_vector(info_dict.get("goal_emb"), torch, "goal_emb", False),
                }
            return result

    pilot.ShadowRouter = CapturingShadowRouter
    try:
        result = pilot.run_episode(
            task,
            shadow_enabled,
            True,
            runtime["pilot_args"],
            runtime["baseline"],
            runtime["swm"],
            runtime["cfg"],
            runtime["dataset"],
            runtime["process"],
            runtime["transforms"],
            torch,
            runtime["spt"],
            runtime["official"],
            runtime["reference"],
            runtime["student"],
            runtime["old_router"],
            runtime["instantiate"],
            runtime["np"],
            seeded_reset_override=42,
        )
    finally:
        pilot.ShadowRouter = base_router
    result["_captured_contexts"] = captured_solves
    return result


def seed_protocol_valid(run: dict[str, Any]) -> bool:
    seed = run.get("dataset_reset_seed", {})
    return bool(
        seed.get("dataset_seed_column_present") is False
        and seed.get("world_reset_requested_seed") == [None]
        and seed.get("world_reset_effective_seed") == [42]
        and seed.get("world_reset_call_count") == 1
        and run.get("episode_seed") == 42
        and run.get("native_solver_seed") == 42
    )


def compare_gate_pair(control: dict[str, Any], shadow: dict[str, Any], torch: Any) -> dict[str, Any]:
    base = pilot.compare_paired_runs(control, shadow, torch)
    left_contexts = control.get("_captured_contexts", [])
    right_contexts = shadow.get("_captured_contexts", [])
    contexts_exact = len(left_contexts) == len(right_contexts) and bool(left_contexts)
    if contexts_exact:
        for left, right in zip(left_contexts, right_contexts, strict=True):
            if left.keys() != right.keys() or len(left) != len(ROUNDS):
                contexts_exact = False
                break
            if any(
                not pilot.exact_equal(left[round_number][field], right[round_number][field])
                for round_number in ROUNDS
                for field in ("initial_emb", "goal_emb")
            ):
                contexts_exact = False
                break
    checks = dict(base["checks"])
    checks["seed_reset_protocol_control"] = seed_protocol_valid(control)
    checks["seed_reset_protocol_shadow"] = seed_protocol_valid(shadow)
    checks["capture_context_exact"] = contexts_exact
    checks["shadow_arm_identity"] = control.get("shadow_enabled") is False and shadow.get("shadow_enabled") is True
    control_steps = [int(row["sim_steps_before_solve"]) for row in control.get("_solves", [])]
    shadow_steps = [int(row["sim_steps_before_solve"]) for row in shadow.get("_solves", [])]
    checks["expected_replan_steps_0_25_exact"] = control_steps == shadow_steps == [0, 25]
    checks["all_required"] = all(checks.values())
    per_solve = []
    for left, row in zip(control.get("_solves", []), base["per_solve"], strict=True):
        per_solve.append({**row, "sim_steps_before_solve": int(left["sim_steps_before_solve"])})
    return {"pass": checks["all_required"], "checks": checks, "per_solve": per_solve}


def bank_records(
    run: dict[str, Any], split: str, task: dict[str, int], torch: Any
) -> list[dict[str, Any]]:
    solves = run.get("_solves", [])
    contexts_by_solve = run.get("_captured_contexts", [])
    if len(solves) != len(contexts_by_solve):
        raise RuntimeError("context capture and validated CEM solve counts differ")
    records: list[dict[str, Any]] = []
    for solve, contexts in zip(solves, contexts_by_solve, strict=True):
        if set(contexts) != set(ROUNDS):
            raise RuntimeError("a reached replan is missing a captured CEM context round")
        step = int(solve["sim_steps_before_solve"])
        for round_number in ROUNDS:
            score = solve["scores"][str(round_number)]
            pair_arrays = score.get("pair_arrays")
            if not isinstance(pair_arrays, dict):
                raise RuntimeError("pilot runner did not retain the complete CEM candidate arrays")
            candidates = pair_arrays["candidates"]
            student_costs = score["student_costs"].reshape(-1)
            teacher_costs = score["teacher_costs"]
            if candidates.shape != (1, 300, 5, 10):
                raise RuntimeError(f"candidate shape drifted: {tuple(candidates.shape)}")
            if teacher_costs is None:
                raise RuntimeError("teacher shadow labels are missing")
            candidates = candidates[0].detach().cpu().contiguous().clone()
            student_costs = student_costs.detach().cpu().contiguous().clone()
            teacher_costs = teacher_costs.reshape(-1).detach().cpu().contiguous().clone()
            initial_emb = contexts[round_number]["initial_emb"]
            goal_emb = contexts[round_number]["goal_emb"]
            if candidates.dtype != torch.float32 or candidates.shape != (300, 5, 10):
                raise RuntimeError("candidate actions must be CPU float32 [300,5,10]")
            if student_costs.dtype != torch.float32 or teacher_costs.dtype != torch.float32:
                raise RuntimeError("objective costs must be CPU float32")
            if student_costs.shape != (300,) or teacher_costs.shape != (300,):
                raise RuntimeError("objective cost vectors must have 300 entries")
            for label, value in (
                ("candidates", candidates),
                ("student_costs", student_costs),
                ("teacher_costs", teacher_costs),
            ):
                if not bool(torch.isfinite(value).all().item()):
                    raise FloatingPointError(f"non-finite values in {label}")
            student_std = student_costs.std(unbiased=False)
            teacher_std = teacher_costs.std(unbiased=False)
            if not bool(torch.isfinite(student_std).item()) or float(student_std.item()) <= 1e-6:
                raise FloatingPointError("student-cost population standard deviation must be finite and greater than 1e-6")
            if not bool(torch.isfinite(teacher_std).item()) or float(teacher_std.item()) <= 0.0:
                raise FloatingPointError("teacher-cost population standard deviation must be finite and positive")
            records.append({
                "schema": BANK_SCHEMA,
                "split": split,
                "selection_order": int(task["selection_order"]),
                "episode_idx": int(task["episode_idx"]),
                "row_index": int(task["row_index"]),
                "start_step": int(task["start_step"]),
                "replan_step": step,
                "cem_round": round_number,
                "solver_seed": 42,
                "candidate_indices": torch.arange(300, dtype=torch.int64),
                "initial_emb": initial_emb,
                "goal_emb": goal_emb,
                "candidates": candidates,
                "student_costs": student_costs,
                "teacher_costs": teacher_costs,
            })
    return records


def episode_manifest_row(split: str, task: dict[str, int], run: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        "split": split,
        "task": dict(task),
        "t25_status": run["t25_status"],
        "actual_env_step_count": int(run["actual_env_step_count"]),
        "success": bool(run["success"]),
        "bank_count": count,
    }


def save_banks(path: Path, banks: list[dict[str, Any]], torch: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(banks, temporary)
    temporary.replace(path)


def collect(args: argparse.Namespace) -> int:
    host = pilot.require_compute_node()
    job_id = os.environ["PBS_JOBID"].strip()
    if not job_id:
        raise RuntimeError("PBS_JOBID must be non-empty")
    output = args.output_dir.resolve()
    if output.name != job_id:
        raise RuntimeError("output directory basename must equal PBS_JOBID")
    output.mkdir(parents=True, exist_ok=True)
    existing_artifacts = (
        "collection_gate.json",
        "collection_manifest.json",
        "train_banks.pt",
        "validation_banks.pt",
    )
    if any((output / name).exists() for name in existing_artifacts):
        raise FileExistsError(f"refusing to overwrite collection artifacts in {output}")

    # The compute-node guard above precedes all freeze, dataset, and model reads.
    freeze = json_object(args.freeze.resolve(strict=True))
    pilot_freeze = json_object(args.pilot_freeze.resolve(strict=True))
    seeded_freeze = json_object(args.seeded_pilot_freeze.resolve(strict=True))
    manifest = json_object(args.selection_manifest.resolve(strict=True))
    runtime = setup_runtime(args, freeze, seeded_freeze, pilot_freeze, manifest)
    gate_path = output / "collection_gate.json"
    gate: dict[str, Any] = {
        "schema": "lewm-pusht-onpolicy-fullbank-collection-gate-v1",
        "pbs_job_id": job_id,
        "compute_host": host,
        "status": "RUNNING",
        "required_pairs": 4,
        "passed_pairs": 0,
        "pairs": [],
        "remaining_collection_started": False,
    }
    write_json(gate_path, gate)

    train_banks: list[dict[str, Any]] = []
    validation_banks: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for task in runtime["training"][:4]:
        pair_row: dict[str, Any] = {"task": dict(task)}
        try:
            control = run_captured_episode(runtime, task, False)
            shadow = run_captured_episode(runtime, task, True)
            comparison = compare_gate_pair(control, shadow, runtime["torch"])
            pair_row["comparison"] = comparison
            pair_row["t25_status_control"] = control["t25_status"]
            pair_row["t25_status_shadow"] = shadow["t25_status"]
            pair_row["control_dataset_reset_seed"] = control.get("dataset_reset_seed")
            pair_row["shadow_dataset_reset_seed"] = shadow.get("dataset_reset_seed")
            if comparison["pass"]:
                task_banks = bank_records(shadow, "train", task, runtime["torch"])
                train_banks.extend(task_banks)
                episode_rows.append(episode_manifest_row("train", task, shadow, len(task_banks)))
        except Exception as exc:
            pair_row["comparison"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        gate["pairs"].append(pair_row)
        gate["passed_pairs"] = sum(bool(row.get("comparison", {}).get("pass")) for row in gate["pairs"])
        write_json(gate_path, gate)
        if not pair_row["comparison"]["pass"]:
            gate["status"] = "FAIL"
            gate["failure_rule"] = "stop before any other task, training, or validation collection"
            write_json(gate_path, gate)
            write_json(output / "collection_manifest.json", {
                "schema": "lewm-pusht-onpolicy-fullbank-collection-manifest-v1",
                "status": "GATE_FAILED",
                "pbs_job_id": job_id,
                "train_banks_written": False,
                "validation_banks_written": False,
            })
            print(json.dumps({"status": "GATE_FAILED", "passed_pairs": gate["passed_pairs"]}), flush=True)
            return 2

    if gate["passed_pairs"] != 4:
        gate["status"] = "FAIL"
        write_json(gate_path, gate)
        raise RuntimeError("all four required first-task pairs did not pass")

    gate["status"] = "PASS"
    gate["remaining_collection_started"] = True
    write_json(gate_path, gate)

    # Do not inspect task-row metadata for the remaining training or validation
    # tasks until all four first-task paired checks have passed.
    validate_task_rows(runtime, runtime["training"][4:] + runtime["validation"])

    for task in runtime["training"][4:]:
        run = run_captured_episode(runtime, task, True)
        task_banks = bank_records(run, "train", task, runtime["torch"])
        train_banks.extend(task_banks)
        episode_rows.append(episode_manifest_row("train", task, run, len(task_banks)))

    for task in runtime["validation"]:
        run = run_captured_episode(runtime, task, True)
        task_banks = bank_records(run, "validation", task, runtime["torch"])
        validation_banks.extend(task_banks)
        episode_rows.append(episode_manifest_row("validation", task, run, len(task_banks)))

    expected_train_ids = {row["episode_idx"] for row in runtime["training"]}
    expected_validation_ids = {row["episode_idx"] for row in runtime["validation"]}
    if {row["episode_idx"] for row in runtime["untouched"]} & (
        {int(bank["episode_idx"]) for bank in train_banks + validation_banks}
    ):
        raise RuntimeError("untouched reserved tasks entered a bank")
    if {int(bank["episode_idx"]) for bank in train_banks} != expected_train_ids:
        raise RuntimeError("training bank task coverage differs from the frozen 16 tasks")
    if {int(bank["episode_idx"]) for bank in validation_banks} != expected_validation_ids:
        raise RuntimeError("validation bank task coverage differs from the frozen first eight holdout tasks")

    save_banks(output / "train_banks.pt", train_banks, runtime["torch"])
    save_banks(output / "validation_banks.pt", validation_banks, runtime["torch"])
    write_json(output / "collection_manifest.json", {
        "schema": "lewm-pusht-onpolicy-fullbank-collection-manifest-v1",
        "status": "COMPLETE",
        "pbs_job_id": job_id,
        "compute_host": host,
        "freeze": str(args.freeze.resolve()),
        "seeded_pilot_freeze": str(args.seeded_pilot_freeze.resolve()),
        "selection_manifest_job_id": str(args.selection_job_id or manifest["pbs_job_id"]),
        "bank_schema": BANK_SCHEMA,
        "train_banks": {"path": "train_banks.pt", "count": len(train_banks)},
        "validation_banks": {"path": "validation_banks.pt", "count": len(validation_banks)},
        "untouched_tasks": runtime["untouched"],
        "episodes": episode_rows,
        "validation_optimizer_access": False,
        "treatment_scoring_or_action": False,
    })
    print(json.dumps({
        "status": "COMPLETE",
        "train_banks": len(train_banks),
        "validation_banks": len(validation_banks),
        "validation_t25_episodes": sum(
            row["split"] == "validation" and row["t25_status"] == "reached" for row in episode_rows
        ),
    }), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())

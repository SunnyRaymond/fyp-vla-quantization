"""Extend real-observation latent alignment to the remaining 14 collection tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE.parent
sys.path.insert(0, str(PILOT_DIR))
import run_student_induced_shadow_pilot as pilot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--official-cem-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--pilot-freeze", type=Path, required=True)
    parser.add_argument("--seeded-freeze", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class RealStepCapture:
    """Capture only actual token boundaries and their corresponding raw actions."""

    def __init__(self, np: Any, torch: Any) -> None:
        self.np = np
        self.torch = torch
        self.active_steps = 0
        self.pending: dict[str, Any] | None = None
        self.actions: dict[int, Any] = {}
        self.post_pixels: dict[int, Any] = {}
        self.pre_action_pixels: dict[int, Any] = {}

    def _image(self, infos: Mapping[str, Any], where: str) -> Any:
        if "pixels" not in infos:
            raise RuntimeError(f"{where}: World.infos has no pixels key")
        value = infos["pixels"]
        if self.torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = self.np.asarray(value)
        while array.ndim > 3:
            if array.shape[0] != 1:
                raise RuntimeError(f"{where}: expected singleton env/history pixels, got {array.shape}")
            array = array[0]
        if array.ndim != 3 or array.shape[-1] != 3:
            raise RuntimeError(f"{where}: expected raw HWC RGB pixels, got {array.shape}")
        if not self.np.issubdtype(array.dtype, self.np.integer):
            raise RuntimeError(f"{where}: expected raw integer image pixels, got {array.dtype}")
        return self.np.ascontiguousarray(array).copy()

    def attach(self, world: Any) -> None:
        original_get_actions = world._get_actions

        def tracked_get_actions(*args: Any, **kwargs: Any):
            if self.active_steps in (0, 25) and self.active_steps not in self.pre_action_pixels:
                self.pre_action_pixels[self.active_steps] = self._image(
                    world.infos, f"pre-action observation at env step {self.active_steps}"
                )
            return original_get_actions(*args, **kwargs)

        world._get_actions = tracked_get_actions

        original_step = world.envs.step

        def tracked_step(actions: Any, *args: Any, **kwargs: Any):
            mask = kwargs.get("mask")
            if mask is None and args:
                mask = args[0]
            active = pilot.mask_is_active(mask, self.torch)
            action = pilot.first_env_value(actions, self.np, self.torch) if active else None
            self.pending = {"active": active, "action": action}
            return original_step(actions, *args, **kwargs)

        world.envs.step = tracked_step

        original_run_iter = world._run_iter

        def tracked_run_iter(*args: Any, **kwargs: Any):
            original_on_step = kwargs.get("on_step")

            def on_step(current_world: Any) -> None:
                if original_on_step is not None:
                    original_on_step(current_world)
                if self.pending is None:
                    raise RuntimeError("post-step callback has no matching envs.step")
                pending = self.pending
                self.pending = None
                if not pending["active"]:
                    return
                self.active_steps += 1
                action = pending["action"]
                action_array = action.detach().cpu().numpy() if self.torch.is_tensor(action) else self.np.asarray(action)
                action_array = self.np.asarray(action_array).reshape(-1)
                if action_array.shape != (2,) or not self.np.isfinite(action_array).all():
                    raise RuntimeError(f"env action at step {self.active_steps} is not a finite 2-D PushT action")
                self.actions[self.active_steps] = action_array.astype(self.np.float64, copy=True)
                if self.active_steps % 5 == 0:
                    self.post_pixels[self.active_steps] = self._image(
                        current_world.infos, f"post-step observation at env step {self.active_steps}"
                    )

            kwargs["on_step"] = on_step
            yield from original_run_iter(*args, **kwargs)

        world._run_iter = tracked_run_iter


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_freezes(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, int]]]:
    freeze = read_json(args.freeze.resolve(strict=True))
    seeded = read_json(args.seeded_freeze.resolve(strict=True))
    base = read_json(args.pilot_freeze.resolve(strict=True))
    if freeze.get("schema") != "lewm-pusht-real-observation-latent-alignment-14-task-extension-freeze" or freeze.get("schema_version") != 1:
        raise RuntimeError("probe freeze schema mismatch")
    if seeded.get("schema") != pilot.SEEDED_PILOT_FREEZE_SCHEMA or base.get("schema") != pilot.FREEZE_SCHEMA:
        raise RuntimeError("source pilot freeze schema mismatch")
    if freeze.get("protocol", {}).get("solve_starts_global_env_steps") != [0, 25]:
        raise RuntimeError("probe solve boundaries drifted")
    if freeze.get("protocol", {}).get("relative_observation_steps_per_solve") != [5, 10, 15, 20, 25]:
        raise RuntimeError("probe observation boundaries drifted")
    if freeze.get("protocol", {}).get("action_geometry") != {
        "raw_action_width": 2,
        "packed_token_width": 10,
        "raw_actions_per_token": 5,
        "forecast_horizon_tokens": 5,
        "receding_horizon_tokens": 5,
        "env_steps_per_token": 5,
        "env_steps_per_solve": 25,
    }:
        raise RuntimeError("packed action/token geometry drifted")
    tasks = freeze.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 14:
        raise RuntimeError("extension must freeze exactly 14 tasks")
    seeded_tasks = seeded.get("collection_tasks", [])
    expected_tasks = [dict(task) for task in seeded_tasks if 2 <= int(task["selection_order"]) <= 15]
    expected_tasks.sort(key=lambda task: int(task["selection_order"]))
    if len(seeded_tasks) != 16 or len(expected_tasks) != 14 or [dict(task) for task in tasks] != expected_tasks:
        raise RuntimeError("extension task rows differ from seeded collection selection_order 2..15")
    if {int(task["selection_order"]) for task in tasks} & {0, 1}:
        raise RuntimeError("extension must not rerun the two completed tasks")
    holdout_ids = {int(task["episode_idx"]) for task in seeded.get("reserved_holdout_tasks", [])}
    if {int(task["episode_idx"]) for task in tasks} & holdout_ids:
        raise RuntimeError("probe task list overlaps a reserved holdout episode")
    if freeze.get("assets", {}).get("student_checkpoint_path") != seeded.get("student_checkpoint", {}).get("path"):
        raise RuntimeError("probe student checkpoint differs from seeded pilot")
    if freeze.get("assets", {}).get("student_checkpoint_path") != base.get("student_checkpoint", {}).get("path"):
        raise RuntimeError("probe student checkpoint differs from base pilot")
    for key, frozen_key in (("teacher_checkpoint_path", "teacher_checkpoint_path"), ("dataset_path", "dataset_path")):
        if freeze.get("assets", {}).get(key) != seeded.get("upstream_baseline", {}).get(frozen_key):
            raise RuntimeError(f"probe {key} differs from seeded pilot")
    protocol = seeded.get("dataset_evaluator_protocol", {})
    if (protocol.get("eval_budget"), protocol.get("goal_offset_steps"), protocol.get("expected_replan_steps")) != (50, 25, [0, 25]):
        raise RuntimeError("seeded evaluator settings differ from the probe contract")
    if seeded.get("cem", {}).get("seed") != 42:
        raise RuntimeError("native solver seed differs from the seeded pilot contract")
    if not isinstance(seeded.get("reserved_holdout_tasks"), list):
        raise RuntimeError("seeded freeze has no reserved holdout identity list")
    normalized = [pilot.normalize_task(task) for task in tasks]
    return freeze, normalized


def image_tensor(raw: Any, np: Any, torch: Any, tv_tensors: Any, image_transform: Any) -> Any:
    array = np.asarray(raw)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError(f"raw observation image shape is invalid: {array.shape}")
    value = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    transformed = image_transform(tv_tensors.Image(value))
    if transformed.ndim != 3 or transformed.shape[0] != 3:
        raise RuntimeError(f"official pixel transform shape mismatch: {tuple(transformed.shape)}")
    return transformed.unsqueeze(0).unsqueeze(0).to("cuda", dtype=torch.float32)


def encode_h1(raw: Any, official: Any, image_transform: Any, np: Any, torch: Any, tv_tensors: Any) -> Any:
    pixels = image_tensor(raw, np, torch, tv_tensors, image_transform)
    with torch.inference_mode():
        latent = official.encode({"pixels": pixels})["emb"]
    if tuple(latent.shape) != (1, 1, 192):
        raise RuntimeError(f"official H=1 encode must return [1,1,192], got {tuple(latent.shape)}")
    if not bool(torch.isfinite(latent).all().item()):
        raise FloatingPointError("official H=1 observation latent is non-finite")
    return latent.detach()


def run_one_episode(
    task: Mapping[str, int], args: argparse.Namespace, baseline: Any, swm: Any,
    cfg: Any, dataset: Any, process: Mapping[str, Any], transforms: Any, spt: Any,
    torch: Any, np: Any, official: Any, reference: Any, student: Any,
    old_router: Any, instantiate: Any,
) -> tuple[dict[str, Any], RealStepCapture]:
    capture = RealStepCapture(np, torch)
    original_factory = baseline.make_pinned_world

    def captured_factory(*factory_args: Any, **factory_kwargs: Any):
        world = original_factory(*factory_args, **factory_kwargs)
        capture.attach(world)
        return world

    baseline.make_pinned_world = captured_factory
    try:
        result = pilot.run_episode(
            task, False, False, args, baseline, swm, cfg, dataset, process, transforms,
            torch, spt, official, reference, student, old_router, instantiate, np,
            seeded_reset_override=42,
        )
    finally:
        baseline.make_pinned_world = original_factory
    if result.get("shadow_enabled") is not False:
        raise RuntimeError("probe unexpectedly enabled teacher shadow scoring")
    if result.get("_dataset_seed_column_present") is not False:
        raise RuntimeError("seeded protocol requires the dataset seed column to be absent")
    if result.get("_reset_seed_observed") != [None] or result.get("_reset_seed_effective") != [42]:
        raise RuntimeError("probe did not perform exactly one dataset reset(None) -> reset(42)")
    if capture.active_steps != int(result["actual_env_step_count"]):
        raise RuntimeError("pixel/action callback count differs from the validated pilot commit count")
    if sorted(capture.actions) != list(range(1, capture.active_steps + 1)):
        raise RuntimeError("captured committed action steps are incomplete or duplicated")
    return result, capture


def compare_forecast(
    model_name: str, predicted: Any, observed: Any, horizon: int, torch: Any,
) -> dict[str, Any]:
    if predicted.ndim != 3 or predicted.shape[0] != 1 or predicted.shape[1] < horizon or predicted.shape[2] != 192 or tuple(observed.shape) != (1, 1, 192):
        raise RuntimeError("forecast/observed latent shape failed the frozen H=1 alignment gate")
    pred = predicted[:, horizon - 1]
    target = observed[:, 0]
    if not bool(torch.isfinite(pred).all().item() and torch.isfinite(target).all().item()):
        raise FloatingPointError(f"{model_name} or observed latent is non-finite at horizon {horizon}")
    mse = (pred - target).square().mean()
    denominator = target.square().mean().clamp_min(1e-8)
    return {
        "forecast_shape": list(predicted.shape),
        "compared_latent_shape": list(pred.shape),
        "finite": True,
        "mse": float(mse.detach().cpu()),
        "relative_mse": float((mse / denominator).detach().cpu()),
    }


def horizon5_comparison(solve_rows: list[dict[str, Any]], start_step: int) -> dict[str, Any] | None:
    solve = next((row for row in solve_rows if row["global_start_env_step"] == start_step), None)
    if solve is None:
        return None
    horizon = next(row for row in solve["horizons"] if row["horizon_index_1based"] == 5)
    if not horizon["available"]:
        return None
    student_relative_mse = float(horizon["student"]["relative_mse"])
    teacher_relative_mse = float(horizon["teacher"]["relative_mse"])
    delta = student_relative_mse - teacher_relative_mse
    return {
        "global_solve_start_env_step": start_step,
        "global_observation_env_step": int(horizon["global_env_step"]),
        "student_relative_mse": student_relative_mse,
        "teacher_relative_mse": teacher_relative_mse,
        "student_minus_teacher_relative_mse": delta,
        "student_higher_than_teacher": delta > 0.0,
        "student_delta_at_least_0_05": delta >= 0.05,
    }


def summarize_episode_first_h5(episodes: list[dict[str, Any]], expected_tasks: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "aggregation_unit": "one episode-first value per task and solve boundary",
        "delta_definition": "student_relative_mse - teacher_relative_mse",
        "support_threshold": 0.05,
        "minimum_supporting_tasks": 12,
        "by_solve": {},
    }
    for label, start_step in (("t0", 0), ("t25", 25)):
        rows = [episode["episode_first_h5"][label] for episode in episodes if episode["episode_first_h5"][label] is not None]
        deltas = [float(row["student_minus_teacher_relative_mse"]) for row in rows]
        result["by_solve"][label] = {
            "available_episode_count": len(rows),
            "missing_episode_count": expected_tasks - len(rows),
            "median_student_relative_mse": statistics.median(float(row["student_relative_mse"]) for row in rows) if rows else None,
            "median_teacher_relative_mse": statistics.median(float(row["teacher_relative_mse"]) for row in rows) if rows else None,
            "student_higher_than_teacher_episode_count": sum(delta > 0.0 for delta in deltas),
            "student_delta_at_least_0_05_episode_count": sum(delta >= 0.05 for delta in deltas),
            "combined_16_support_decision_computed": False,
        }
    result["missing_t25_horizon5_episode_count"] = result["by_solve"]["t25"]["missing_episode_count"]
    return result


def summarize_episode(
    task: Mapping[str, int], result: Mapping[str, Any], capture: RealStepCapture,
    official: Any, reference: Any, student: Any, process: Mapping[str, Any],
    image_transform: Any, np: Any, torch: Any, tv_tensors: Any,
) -> dict[str, Any]:
    steps = int(result["actual_env_step_count"])
    if capture.active_steps != steps:
        raise RuntimeError("actual simulator step count changed after capture")
    post = capture.post_pixels
    if any(step > steps for step in post):
        raise RuntimeError("captured post-step pixel lies beyond the episode's committed transition count")
    solves = result.get("solve_records", [])
    starts = [int(solve["sim_steps_before_solve"]) for solve in solves]
    if starts not in ([0], [0, 25]):
        raise RuntimeError(f"unexpected solve starts for selected episode: {starts}")
    if 25 in starts:
        if 25 not in capture.pre_action_pixels or 25 not in post:
            raise RuntimeError("t25 solve lacks both pre-solve and post-step-25 real pixels")
        if not np.array_equal(capture.pre_action_pixels[25], post[25]):
            raise RuntimeError("t25 solve pixels differ from the actual step-25 post-step pixels")
    if 0 not in capture.pre_action_pixels:
        raise RuntimeError("t0 dataset-start pixels were not captured before the first action")

    termination = {
        int(row["active_step"]): {"terminated": bool(row["terminated"]), "truncated": bool(row["truncated"])}
        for row in result.get("termination_signals", [])
    }
    solve_rows = []
    for solve in solves:
        start = int(solve["sim_steps_before_solve"])
        ordinal = int(solve["solve_ordinal"])
        plan = torch.as_tensor(solve["student_actions_from_solver"], dtype=torch.float32, device="cuda")
        if tuple(plan.shape) != (1, 5, 10) or not bool(torch.isfinite(plan).all().item()):
            raise RuntimeError(f"solve {ordinal} native plan must be finite [1,5,10], got {tuple(plan.shape)}")
        anchor_raw = capture.pre_action_pixels[start]
        if start == 25:
            anchor_raw = post[25]

        available_horizons = 0
        raw_token_rows = []
        for horizon in range(1, 6):
            boundary = start + 5 * horizon
            if boundary > steps:
                break
            if boundary not in post:
                raise RuntimeError(f"committed boundary {boundary} has no post-step pixels")
            action_ids = range(start + (horizon - 1) * 5 + 1, boundary + 1)
            if any(index not in capture.actions for index in action_ids):
                raise RuntimeError(f"committed actions for solve {ordinal} horizon {horizon} are incomplete")
            raw_actions = np.stack([capture.actions[index] for index in action_ids], axis=0)
            model_actions = process["action"].transform(raw_actions)
            if tuple(model_actions.shape) != (5, 2) or not np.isfinite(model_actions).all():
                raise RuntimeError("official action preprocessing did not preserve a finite five-step block")
            token = np.asarray(model_actions, dtype=np.float32).reshape(10)
            raw_token_rows.append(token)
            available_horizons += 1

        if available_horizons:
            actual_tokens = torch.as_tensor(np.stack(raw_token_rows), dtype=torch.float32, device="cuda")
            action_delta = (actual_tokens - plan[0, :available_horizons]).abs()
            max_action_delta = float(action_delta.max().detach().cpu())
            token_action_deltas = [float(value) for value in action_delta.amax(dim=-1).detach().cpu().tolist()]
            if not math.isfinite(max_action_delta) or max_action_delta > 1e-5:
                raise RuntimeError(
                    f"solve {ordinal} actual action replay does not match the native plan "
                    f"(max_abs_delta={max_action_delta:.8g})"
                )
            anchor = encode_h1(anchor_raw, official, image_transform, np, torch, tv_tensors)
            with torch.inference_mode():
                student_pred = student(anchor, actual_tokens.unsqueeze(0))
                teacher_pred = reference.base.official_teacher_targets(
                    official, anchor, actual_tokens.unsqueeze(0)
                )
            expected_shape = (1, available_horizons, 192)
            if tuple(student_pred.shape) != expected_shape or tuple(teacher_pred.shape) != expected_shape:
                raise RuntimeError(
                    f"solve {ordinal} forecasts must be {expected_shape}; got "
                    f"student={tuple(student_pred.shape)} teacher={tuple(teacher_pred.shape)}"
                )
        else:
            max_action_delta = None
            token_action_deltas = []
            student_pred = teacher_pred = None

        horizon_rows = []
        for horizon in range(1, 6):
            boundary = start + 5 * horizon
            if horizon > available_horizons:
                done = termination.get(steps, {"terminated": False, "truncated": False})
                if steps < boundary and not (done["terminated"] or done["truncated"]):
                    raise RuntimeError("episode ended before a token boundary without explicit termination evidence")
                horizon_rows.append({
                    "horizon_index_1based": horizon,
                    "global_env_step": boundary,
                    "available": False,
                    "reason": "terminal_before_complete_token_boundary" if steps < boundary else "not_reached",
                    "terminated_or_truncated_step": steps if done["terminated"] or done["truncated"] else None,
                    "student": None,
                    "teacher": None,
                    "action_alignment_verified": None,
                })
                continue
            observed = encode_h1(post[boundary], official, image_transform, np, torch, tv_tensors)
            horizon_rows.append({
                "horizon_index_1based": horizon,
                "global_env_step": boundary,
                "available": True,
                "observation_shape": list(observed.shape),
                "action_alignment_verified": True,
                "action_max_abs_delta": token_action_deltas[horizon - 1],
                "student": compare_forecast("student", student_pred, observed, horizon, torch),
                "teacher": compare_forecast("teacher", teacher_pred, observed, horizon, torch),
            })
        solve_rows.append({
            "solve_ordinal": ordinal,
            "global_start_env_step": start,
            "anchor_observation_shape": [1, 1, 192],
            "native_plan_shape": list(plan.shape),
            "executed_action_tokens_verified": available_horizons,
            "max_abs_action_delta": max_action_delta,
            "action_alignment_verified": available_horizons > 0,
            "horizons": horizon_rows,
        })
    episode_first_h5 = {
        "t0": horizon5_comparison(solve_rows, 0),
        "t25": horizon5_comparison(solve_rows, 25),
    }
    return {
        "task": dict(task),
        "t25_status": result["t25_status"],
        "actual_env_step_count": steps,
        "success": bool(result["success"]),
        "termination_at_final_step": termination.get(steps),
        "solve_count": len(solves),
        "solves": solve_rows,
        "episode_first_h5": episode_first_h5,
        "raw_pixels_or_action_trace_written": False,
    }


def main() -> int:
    args = parse_args()
    host = pilot.require_compute_node()
    if os.environ.get("PBS_NGPUS", "1").strip() in ("0", ""):
        raise RuntimeError("probe requires its requested GPU PBS allocation")
    out = args.output.resolve()
    if out == Path("/") or out.name != os.environ.get("PBS_JOBID") or out.parent.name != "real-observation-rollout-probe-14-task":
        raise RuntimeError("output must be artifacts/real-observation-rollout-probe-14-task/<PBS_JOBID>")
    if (out / "real_observation_latent_alignment_14_task.json").exists():
        raise FileExistsError(f"refusing to reuse existing result directory: {out}")
    freeze, tasks = validate_freezes(args)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema": "lewm-pusht-real-observation-latent-alignment-14-task-extension-result",
        "schema_version": 1,
        "status": "RUNNING",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "probe_freeze": str(args.freeze.resolve()),
        "seeded_pilot_freeze": str(args.seeded_freeze.resolve()),
        "pilot_freeze": str(args.pilot_freeze.resolve()),
        "tasks_frozen": tasks,
        "episodes": [],
        "reserved_holdout_evaluated_or_scored": False,
        "raw_images_or_action_traces_written": False,
        "claim_boundary": freeze["claim_boundary"],
    }
    try:
        np, spt, swm, torch, transforms, modules = pilot.load_modules(args)
        baseline, adaptive, old_router, load_official_checkpoint, preprocessing, instantiate = modules
        cfg = baseline.compose_pinned_config(args.lewm_root.resolve(strict=True))
        if (int(cfg.eval.eval_budget), int(cfg.eval.goal_offset_steps), int(cfg.world.max_episode_steps)) != (50, 25, 100):
            raise RuntimeError("pinned evaluation budget/goal offset/world horizon drifted")

        baseline_binding = read_json(args.seeded_freeze.resolve(strict=True))["upstream_baseline"]
        dataset_path = Path(str(baseline_binding["dataset_path"])).resolve(strict=True)
        teacher_path = Path(str(baseline_binding["teacher_checkpoint_path"])).resolve(strict=True)
        if dataset_path != (args.stablewm_home.resolve(strict=True) / "pusht_expert_train.h5").resolve(strict=True):
            raise RuntimeError("dataset path differs from the frozen staged HDF5")
        if teacher_path != (args.stablewm_home.resolve(strict=True) / "pusht" / "lewm_object.ckpt").resolve(strict=True):
            raise RuntimeError("teacher path differs from the frozen staged checkpoint")
        if dataset_path.stat().st_size != int(baseline_binding["dataset_size_bytes"]):
            raise RuntimeError("staged HDF5 size differs from its frozen provenance")
        if teacher_path.stat().st_size != int(baseline_binding["teacher_checkpoint_size_bytes"]):
            raise RuntimeError("staged teacher size differs from its frozen provenance")
        if (args.cache_root / "datasets" / dataset_path.name).resolve(strict=True) != dataset_path:
            raise RuntimeError("dataset cache symlink does not resolve to the frozen HDF5")
        if (args.cache_root / "pusht" / teacher_path.name).resolve(strict=True) != teacher_path:
            raise RuntimeError("teacher cache symlink does not resolve to the frozen checkpoint")

        dataset = swm.data.HDF5Dataset(
            str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache),
            cache_dir=args.cache_root.resolve(strict=True),
        )
        if "seed" in getattr(dataset, "column_names", []):
            raise RuntimeError("seeded pilot requires the dataset seed column to be absent")
        process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
        episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        ordered = sorted(tasks, key=lambda row: row["row_index"])
        rows = dataset.get_row_data(np.asarray([row["row_index"] for row in ordered], dtype=np.int64))
        for index, task in enumerate(ordered):
            if int(rows[episode_column][index]) != task["episode_idx"] or int(rows["step_idx"][index]) != task["start_step"]:
                raise RuntimeError("task row identity differs from frozen episode/start step")

        schedule, _, reference, _ = old_router.load_modules(
            args.control_root.resolve(strict=True), args.lewm_root.resolve(strict=True)
        )
        schedule.validate_interface(reference, args.interface_probe.resolve(strict=True))
        official = load_official_checkpoint(args.cache_root.resolve(strict=True))
        official.interpolate_pos_encoding = True
        official.eval()
        official.requires_grad_(False)
        checkpoint = Path(str(read_json(args.seeded_freeze.resolve(strict=True))["student_checkpoint"]["path"])).resolve(strict=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        student, metadata = adaptive.load_main_student(reference, checkpoint)
        expected_binding = read_json(args.seeded_freeze.resolve(strict=True))["student_checkpoint"]
        for key in ("source", "arm", "extra_updates"):
            if metadata.get("provenance", {}).get(key) != expected_binding.get("provenance", {}).get(key):
                raise RuntimeError(f"loaded student provenance differs from the seeded freeze: {key}")
        if int(metadata.get("provenance", {}).get("extra_updates", -1)) != 1000:
            raise RuntimeError("loaded student is not the frozen treatment step1000 checkpoint")
        parameter_count = sum(int(parameter.numel()) for parameter in student.parameters())
        if parameter_count != int(expected_binding.get("parameter_count", -1)):
            raise RuntimeError("loaded student parameter count differs from seeded freeze")
        student = student.to("cuda").eval()
        student.requires_grad_(False)
        image_transform = baseline.make_image_transform(cfg, spt, transforms, torch)
        from torchvision import tv_tensors

        for task in tasks:
            result, capture = run_one_episode(
                task, args, baseline, swm, cfg, dataset, process, transforms, spt,
                torch, np, official, reference, student, old_router, instantiate,
            )
            episode_summary = summarize_episode(
                task, result, capture, official, reference, student, process,
                image_transform, np, torch, tv_tensors,
            )
            summary["episodes"].append(episode_summary)
            pilot.atomic_json(out / "real_observation_latent_alignment_14_task.json", summary)

        summary["status"] = (
            "COMPLETED_ALL_REACHED_SOLVE_HORIZONS"
            if all(
                all(h["available"] for s in ep["solves"] for h in s["horizons"])
                for ep in summary["episodes"]
            )
            else "COMPLETED_WITH_TERMINAL_OR_BUDGET_LIMITED_HORIZONS"
        )
        summary["episode_first_h5_aggregate"] = summarize_episode_first_h5(summary["episodes"], expected_tasks=len(tasks))
        summary["combined_16_task_decision_computed"] = False
        summary["prior_two_task_evidence_to_merge_later"] = freeze["output"]["prior_two_task_evidence"]
        pilot.atomic_json(out / "real_observation_latent_alignment_14_task.json", summary)
        print(json.dumps({"status": summary["status"], "episodes": len(summary["episodes"]), "output": str(out / "real_observation_latent_alignment_14_task.json")}), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "FAIL_CLOSED_ALIGNMENT_OR_RUNTIME_ERROR"
        summary["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        pilot.atomic_json(out / "real_observation_latent_alignment_14_task.json", summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

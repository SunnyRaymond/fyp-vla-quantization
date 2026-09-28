#!/usr/bin/env python3
"""Collect student-driven real-observation latent training tuples from a frozen manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE.parent
PROBE14_PATH = PILOT_DIR / "real-observation-rollout-probe-14-task" / "run_real_observation_rollout_probe_14_task.py"
PROBE_SPEC = importlib.util.spec_from_file_location("real_observation_probe14_helpers", PROBE14_PATH)
if PROBE_SPEC is None or PROBE_SPEC.loader is None:
    raise RuntimeError(f"cannot load validated observation capture helpers: {PROBE14_PATH}")
probe = importlib.util.module_from_spec(PROBE_SPEC)
PROBE_SPEC.loader.exec_module(probe)
pilot = probe.pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "lewm-root", "stablewm-root", "stablewm-home", "control-root", "baseline-dir",
        "official-cem-dir", "adaptive-dir", "interface-probe", "pilot-freeze", "seeded-freeze",
        "freeze", "selection-manifest", "cache-root", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    freeze = read_json(args.freeze.resolve(strict=True))
    seeded = read_json(args.seeded_freeze.resolve(strict=True))
    manifest = read_json(args.selection_manifest.resolve(strict=True))
    if freeze.get("schema") != "lewm-pusht-student-driven-real-observation-training-data-freeze" or freeze.get("schema_version") != 1:
        raise RuntimeError("real-observation training-data freeze schema mismatch")
    if seeded.get("schema") != pilot.SEEDED_PILOT_FREEZE_SCHEMA or seeded.get("schema_version") != 1:
        raise RuntimeError("seeded-pilot freeze schema mismatch")
    if manifest.get("schema") != "lewm-pusht-student-driven-real-observation-training-selection-manifest-v1" or manifest.get("schema_version") != 1:
        raise RuntimeError("80-task selection manifest schema mismatch")
    if manifest.get("status") != "COMPLETE_METADATA_ONLY_SELECTION":
        raise RuntimeError("selection manifest is not complete")
    frozen_manifest_path = Path(str(freeze["episode_selection"].get("selection_manifest_path", ""))).resolve(strict=True)
    if args.selection_manifest.resolve(strict=True) != frozen_manifest_path:
        raise RuntimeError("collector only accepts the selection manifest path frozen before capture")
    if manifest.get("pbs_job_id") != freeze["episode_selection"].get("selection_manifest_job_id"):
        raise RuntimeError("selection manifest job ID differs from the exact frozen CPU selection job")
    if Path(str(manifest.get("freeze_path", ""))).resolve(strict=True) != args.freeze.resolve(strict=True):
        raise RuntimeError("selection manifest is bound to a different training-data freeze")
    if Path(str(manifest.get("dataset_path", ""))).resolve(strict=True) != Path(str(freeze["provenance"]["dataset_path"])).resolve(strict=True):
        raise RuntimeError("selection manifest is bound to a different HDF5 dataset")
    if manifest.get("selection_seed") != freeze["episode_selection"]["selection_seed"]:
        raise RuntimeError("selection manifest seed differs from the freeze")
    if manifest.get("selected_episode_count") != 80 or manifest.get("split_counts") != {"collection_train": 64, "collection_validation": 16}:
        raise RuntimeError("selection manifest must contain exactly 64 train and 16 validation tasks")
    if manifest.get("selected_overlap_with_exclusions") != 0 or manifest.get("ranker_validation_subset_of_reserved_assertion") is not True:
        raise RuntimeError("selection manifest exclusion assertions did not pass")
    expected_counts = {
        "seeded_collection": 16,
        "seeded_reserved_holdout": 16,
        "ranker_validation": 8,
        "ranker_untouched": 8,
        "closed_loop_50": 50,
        "prior_valid_prefix_600": 600,
    }
    if any(int(manifest.get("exclusion_counts", {}).get(key, -1)) != value for key, value in expected_counts.items()):
        raise RuntimeError("selection manifest exclusion source counts differ from the freeze")
    collection_ids = {int(row["episode_idx"]) for row in seeded["collection_tasks"]}
    reserved_ids = {int(row["episode_idx"]) for row in seeded["reserved_holdout_tasks"]}
    if len(collection_ids) != 16 or len(reserved_ids) != 16 or collection_ids & reserved_ids:
        raise RuntimeError("seeded collection/reserved identity lists are invalid")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 80:
        raise RuntimeError("selection manifest task list must contain 80 rows")
    if [int(task.get("selection_order", -1)) for task in tasks] != list(range(80)):
        raise RuntimeError("selection manifest order must be exactly 0 through 79")
    selected_ids = [int(task["episode_idx"]) for task in tasks]
    if len(set(selected_ids)) != 80 or set(selected_ids) & (collection_ids | reserved_ids):
        raise RuntimeError("selected tasks repeat or overlap an old 32-task diagnostic identity")
    if sum(task.get("split") == "collection_train" for task in tasks) != 64 or sum(task.get("split") == "collection_validation" for task in tasks) != 16:
        raise RuntimeError("selection manifest split membership differs from 64/16 freeze")
    return freeze, manifest, tasks


def add_solve_samples(
    task: dict[str, Any], result: dict[str, Any], capture: Any, process: dict[str, Any],
    official: Any, image_transform: Any, np: Any, torch: Any, tv_tensors: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_steps = int(result["actual_env_step_count"])
    if capture.active_steps != actual_steps or sorted(capture.actions) != list(range(1, actual_steps + 1)):
        raise RuntimeError("real action/pixel capture does not match committed simulator steps")
    post = capture.post_pixels
    if any(step > actual_steps for step in post):
        raise RuntimeError("captured pixels exceed the committed environment-step count")
    solve_records = result.get("solve_records", [])
    starts = [int(record["sim_steps_before_solve"]) for record in solve_records]
    if starts not in ([0], [0, 25]):
        raise RuntimeError(f"unexpected solve starts: {starts}")
    if 0 not in capture.pre_action_pixels:
        raise RuntimeError("dataset-start observation pixels were not captured")
    if 25 in starts and (25 not in post or not np.array_equal(post[25], capture.pre_action_pixels.get(25))):
        raise RuntimeError("t25 solve anchor is not the real post-step-25 observation")

    windows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for solve in solve_records:
        start = int(solve["sim_steps_before_solve"])
        ordinal = int(solve["solve_ordinal"])
        plan = torch.as_tensor(solve["student_actions_from_solver"], dtype=torch.float32, device="cuda")
        if tuple(plan.shape) != (1, 5, 10) or not bool(torch.isfinite(plan).all().item()):
            raise RuntimeError(f"solve {ordinal} native student plan must be finite [1,5,10]")
        horizon_count = min(5, max(0, (actual_steps - start) // 5))
        tokens: list[Any] = []
        target_pixels: list[Any] = []
        for horizon in range(1, horizon_count + 1):
            boundary = start + 5 * horizon
            if boundary not in post:
                raise RuntimeError(f"actual post-step pixels are missing at complete boundary {boundary}")
            action_ids = range(start + (horizon - 1) * 5 + 1, boundary + 1)
            raw_actions = np.stack([capture.actions[index] for index in action_ids], axis=0)
            transformed = process["action"].transform(raw_actions)
            if transformed.shape != (5, 2) or not np.isfinite(transformed).all():
                raise RuntimeError("official action transform must produce five finite 2-D actions")
            tokens.append(np.asarray(transformed, dtype=np.float32).reshape(10))
            target_pixels.append(post[boundary])
        max_action_delta: float | None = None
        if tokens:
            action_array = np.stack(tokens, axis=0)
            delta = np.abs(action_array - plan[0, :horizon_count].detach().cpu().numpy())
            max_action_delta = float(delta.max())
            if not math.isfinite(max_action_delta) or max_action_delta > 1e-5:
                raise RuntimeError(f"solve {ordinal} committed actions differ from native student plan (max_abs={max_action_delta:.8g})")
        window = {
            "solve_ordinal": ordinal,
            "solve_start_env_step": start,
            "complete_token_count": horizon_count,
            "full_window_available": horizon_count == 5,
            "max_plan_action_abs_error": max_action_delta,
            "sample_index": None,
        }
        windows.append(window)
        if horizon_count != 5:
            continue
        anchor_pixels = capture.pre_action_pixels[start]
        z_start = probe.encode_h1(anchor_pixels, official, image_transform, np, torch, tv_tensors)
        if tuple(z_start.shape) != (1, 1, 192) or not bool(torch.isfinite(z_start).all().item()):
            raise RuntimeError(f"solve {ordinal} H=1 start latent failed shape/finite checks")
        latent_rows = []
        for raw_pixels in target_pixels:
            latent = probe.encode_h1(raw_pixels, official, image_transform, np, torch, tv_tensors)
            if tuple(latent.shape) != (1, 1, 192) or not bool(torch.isfinite(latent).all().item()):
                raise RuntimeError(f"solve {ordinal} real post-step latent failed shape/finite checks")
            latent_rows.append(latent[0, 0].detach().cpu().numpy().astype(np.float32, copy=False))
        samples.append({
            "episode_idx": int(task["episode_idx"]),
            "selection_order": int(task["selection_order"]),
            "split": str(task["split"]),
            "solve_start_env_step": start,
            "reset_seed": 42,
            "native_cem_seed": 42,
            "relative_env_step_indices": [5, 10, 15, 20, 25],
            "z_start": z_start[0, 0].detach().cpu().numpy().astype(np.float32, copy=False),
            "packed_actions": action_array.astype(np.float32, copy=False),
            "poststep_latents": np.stack(latent_rows, axis=0),
            "max_plan_action_abs_error": max_action_delta,
        })
        window["sample_index"] = len(samples) - 1
    return samples, {
        "task": {key: task[key] for key in ("split", "selection_order", "episode_idx", "row_index", "start_step")},
        "actual_env_step_count": actual_steps,
        "success": bool(result["success"]),
        "t25_status": result["t25_status"],
        "solve_windows": windows,
    }


def save_dataset(out: Path, samples: list[dict[str, Any]], np: Any) -> None:
    count = len(samples)
    arrays = {
        "episode_idx": np.asarray([row["episode_idx"] for row in samples], dtype=np.int64),
        "selection_order": np.asarray([row["selection_order"] for row in samples], dtype=np.int32),
        "split": np.asarray([row["split"] for row in samples], dtype="U24"),
        "solve_start_env_step": np.asarray([row["solve_start_env_step"] for row in samples], dtype=np.int32),
        "reset_seed": np.asarray([row["reset_seed"] for row in samples], dtype=np.int32),
        "native_cem_seed": np.asarray([row["native_cem_seed"] for row in samples], dtype=np.int32),
        "relative_env_step_indices": np.asarray([row["relative_env_step_indices"] for row in samples], dtype=np.int32).reshape(count, 5),
        "z_start": np.asarray([row["z_start"] for row in samples], dtype=np.float32).reshape(count, 192),
        "packed_actions": np.asarray([row["packed_actions"] for row in samples], dtype=np.float32).reshape(count, 5, 10),
        "poststep_latents": np.asarray([row["poststep_latents"] for row in samples], dtype=np.float32).reshape(count, 5, 192),
        "max_plan_action_abs_error": np.asarray([row["max_plan_action_abs_error"] for row in samples], dtype=np.float32),
    }
    temporary = out / "real_observation_training_data.npz.tmp"
    final = out / "real_observation_training_data.npz"
    if temporary.exists() or final.exists():
        raise FileExistsError("refusing to overwrite an existing training-data archive")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(final)


def main() -> int:
    args = parse_args()
    host = pilot.require_compute_node()
    if os.environ.get("PBS_NGPUS", "1").strip() in ("0", ""):
        raise RuntimeError("collector requires its requested GPU PBS allocation")
    out = args.output.resolve()
    job_id = os.environ.get("PBS_JOBID", "")
    if out == Path("/") or out.name != job_id or out.parent.name != "capture":
        raise RuntimeError("output must be artifacts/real-observation-training-data/capture/<PBS_JOBID>")
    result_path = out / "collection_summary.json"
    data_path = out / "real_observation_training_data.npz"
    if result_path.exists() or data_path.exists():
        raise FileExistsError(f"refusing to reuse existing capture results: {out}")
    freeze, selection, tasks = validate_inputs(args)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema": "lewm-pusht-student-driven-real-observation-training-collection-result-v1",
        "schema_version": 1,
        "status": "RUNNING",
        "pbs_job_id": job_id,
        "compute_host": host,
        "freeze_path": str(args.freeze.resolve()),
        "selection_manifest_path": str(args.selection_manifest.resolve()),
        "selection_job_id": selection["pbs_job_id"],
        "tasks_frozen": len(tasks),
        "reserved_holdout_evaluated_or_scored": False,
        "teacher_forecasts_costs_or_shadow_calls": False,
        "raw_pixels_or_stepwise_action_traces_written": False,
        "episodes": [],
        "claim_boundary": "Collection of student-driven action-to-real-observation latent tuples only; no training, teacher target, CEM treatment, or closed-loop claim.",
    }
    pilot.atomic_json(result_path, summary)
    try:
        np, spt, swm, torch, transforms, modules = pilot.load_modules(args)
        baseline, adaptive, old_router, load_official_checkpoint, preprocessing, instantiate = modules
        cfg = baseline.compose_pinned_config(args.lewm_root.resolve(strict=True))
        if (int(cfg.eval.eval_budget), int(cfg.eval.goal_offset_steps), int(cfg.world.max_episode_steps)) != (50, 25, 100):
            raise RuntimeError("pinned evaluator budget/goal offset/world horizon drifted")
        seeded = read_json(args.seeded_freeze.resolve(strict=True))
        baseline_binding = seeded["upstream_baseline"]
        dataset_path = Path(str(baseline_binding["dataset_path"])).resolve(strict=True)
        teacher_path = Path(str(baseline_binding["teacher_checkpoint_path"])).resolve(strict=True)
        if dataset_path != (args.stablewm_home.resolve(strict=True) / "pusht_expert_train.h5").resolve(strict=True):
            raise RuntimeError("HDF5 path differs from the seeded evaluator freeze")
        if teacher_path != (args.stablewm_home.resolve(strict=True) / "pusht" / "lewm_object.ckpt").resolve(strict=True):
            raise RuntimeError("official H=1 encoder checkpoint path differs from the seeded freeze")
        if (args.cache_root / "datasets" / dataset_path.name).resolve(strict=True) != dataset_path:
            raise RuntimeError("HDF5 cache entry does not resolve to the frozen dataset")
        if (args.cache_root / "pusht" / teacher_path.name).resolve(strict=True) != teacher_path:
            raise RuntimeError("official encoder cache entry does not resolve to the frozen checkpoint")
        dataset = swm.data.HDF5Dataset(
            str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache),
            cache_dir=args.cache_root.resolve(strict=True),
        )
        if "seed" in getattr(dataset, "column_names", []):
            raise RuntimeError("seeded reset protocol requires no dataset seed column")
        process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
        episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        # h5py fancy indexing requires increasing indices; this identity read
        # does not change the frozen randomized rollout order in ``tasks``.
        sorted_tasks = sorted(tasks, key=lambda task: int(task["row_index"]))
        row_indices = np.asarray([int(task["row_index"]) for task in sorted_tasks], dtype=np.int64)
        rows = dataset.get_row_data(row_indices)
        for index, task in enumerate(sorted_tasks):
            if int(rows[episode_column][index]) != int(task["episode_idx"]) or int(rows["step_idx"][index]) != int(task["start_step"]):
                raise RuntimeError("selection-manifest task row identity differs from the staged dataset")

        schedule, _, reference, _ = old_router.load_modules(
            args.control_root.resolve(strict=True), args.lewm_root.resolve(strict=True)
        )
        schedule.validate_interface(reference, args.interface_probe.resolve(strict=True))
        official = load_official_checkpoint(args.cache_root.resolve(strict=True))
        official.interpolate_pos_encoding = True
        official.eval()
        official.requires_grad_(False)
        checkpoint = Path(str(freeze["provenance"]["student_checkpoint"]["path"])).resolve(strict=True)
        student, _metadata = adaptive.load_main_student(reference, checkpoint)
        student = student.to("cuda").eval()
        student.requires_grad_(False)
        image_transform = baseline.make_image_transform(cfg, spt, transforms, torch)
        from torchvision import tv_tensors

        all_samples: list[dict[str, Any]] = []
        for task in tasks:
            task_id = {key: task[key] for key in ("split", "selection_order", "episode_idx", "row_index", "start_step")}
            result, capture = probe.run_one_episode(
                task_id, args, baseline, swm, cfg, dataset, process, transforms, spt,
                torch, np, official, reference, student, old_router, instantiate,
            )
            samples, episode = add_solve_samples(task_id, result, capture, process, official, image_transform, np, torch, tv_tensors)
            for sample in samples:
                sample["sample_index"] = len(all_samples)
                all_samples.append(sample)
            for window in episode["solve_windows"]:
                if window["sample_index"] is not None:
                    window["sample_index"] += len(all_samples) - len(samples)
            summary["episodes"].append(episode)
            pilot.atomic_json(result_path, summary)

        save_dataset(out, all_samples, np)
        summary["status"] = "COMPLETE_WITH_ALL_TASKS_CAPTURED" if all(
            len(episode["solve_windows"]) == 2 and all(window["full_window_available"] for window in episode["solve_windows"])
            for episode in summary["episodes"]
        ) else "COMPLETE_WITH_TERMINAL_OR_BUDGET_LIMITED_WINDOWS"
        summary["sample_count"] = len(all_samples)
        summary["train_sample_count"] = sum(sample["split"] == "collection_train" for sample in all_samples)
        summary["validation_sample_count"] = sum(sample["split"] == "collection_validation" for sample in all_samples)
        summary["unavailable_window_count"] = sum(
            not window["full_window_available"]
            for episode in summary["episodes"] for window in episode["solve_windows"]
        )
        summary["archive"] = {
            "filename": data_path.name,
            "sample_schema": {"z_start": "float32[N,192]", "packed_actions": "float32[N,5,10]", "poststep_latents": "float32[N,5,192]"},
            "sample_count": len(all_samples),
        }
        pilot.atomic_json(result_path, summary)
        print(json.dumps({"status": summary["status"], "episodes": len(tasks), "samples": len(all_samples), "output": str(result_path)}), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "FAIL_CLOSED_COLLECTION_ERROR"
        summary["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        pilot.atomic_json(result_path, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

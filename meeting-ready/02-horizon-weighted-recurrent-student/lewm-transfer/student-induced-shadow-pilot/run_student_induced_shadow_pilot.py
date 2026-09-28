"""Closed-loop LeWM student evaluation with non-invasive teacher shadow scores."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA = "lewm-pusht-student-induced-shadow-pilot-result"
FREEZE_SCHEMA = "lewm-pusht-student-induced-shadow-pilot-freeze"
SELECTION_SCHEMA = "lewm-pusht-student-induced-shadow-episode-selection-result"
SELECTION_VERSION = 1
DIAGNOSTIC_FREEZE_SCHEMA = "lewm-pusht-student-induced-shadow-two-pair-diagnostic-freeze"
DIAGNOSTIC_RESULT_SCHEMA = "lewm-pusht-student-induced-shadow-two-pair-diagnostic-result"
REPRO_FREEZE_SCHEMA = "lewm-pusht-student-induced-shadow-control-control-repro-freeze"
REPRO_RESULT_SCHEMA = "lewm-pusht-student-induced-shadow-control-control-repro-result"
SEEDED_REPRO_FREEZE_SCHEMA = "lewm-pusht-student-induced-shadow-seeded-control-control-repro-freeze"
SEEDED_REPRO_RESULT_SCHEMA = "lewm-pusht-student-induced-shadow-seeded-control-control-repro-result"
SEEDED_PILOT_FREEZE_SCHEMA = "lewm-pusht-student-induced-shadow-seeded-pilot-freeze"
SEEDED_PILOT_RESULT_SCHEMA = "lewm-pusht-student-induced-shadow-seeded-pilot-result"
SOLVER_ROUNDS = (10, 20, 30)
OBS_KEYS = ("state", "goal_state")
REPRO_INPUT_KEYS = ("pixels", "goal", "proprio", "state", "goal_state", "action")


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
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-job-id", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-two-pair-only", action="store_true")
    parser.add_argument("--diagnostic-freeze", type=Path)
    parser.add_argument("--control-control-repro-only", action="store_true")
    parser.add_argument("--repro-freeze", type=Path)
    parser.add_argument("--seeded-control-control-repro-only", action="store_true")
    parser.add_argument("--seeded-repro-freeze", type=Path)
    parser.add_argument("--seeded-pilot-variant", action="store_true")
    parser.add_argument("--seeded-pilot-freeze", type=Path)
    args = parser.parse_args()
    selected_modes = sum((
        args.diagnostic_two_pair_only,
        args.control_control_repro_only,
        args.seeded_control_control_repro_only,
        args.seeded_pilot_variant,
    ))
    if selected_modes > 1:
        parser.error("diagnostic runner modes are mutually exclusive")
    if args.diagnostic_two_pair_only and args.diagnostic_freeze is None:
        parser.error("--diagnostic-two-pair-only requires --diagnostic-freeze")
    if args.diagnostic_freeze is not None and not args.diagnostic_two_pair_only:
        parser.error("--diagnostic-freeze is valid only with --diagnostic-two-pair-only")
    if args.control_control_repro_only and args.repro_freeze is None:
        parser.error("--control-control-repro-only requires --repro-freeze")
    if args.repro_freeze is not None and not args.control_control_repro_only:
        parser.error("--repro-freeze is valid only with --control-control-repro-only")
    if args.seeded_control_control_repro_only and args.seeded_repro_freeze is None:
        parser.error("--seeded-control-control-repro-only requires --seeded-repro-freeze")
    if args.seeded_repro_freeze is not None and not args.seeded_control_control_repro_only:
        parser.error("--seeded-repro-freeze is valid only with --seeded-control-control-repro-only")
    if args.seeded_pilot_variant and args.seeded_pilot_freeze is None:
        parser.error("--seeded-pilot-variant requires --seeded-pilot-freeze")
    if args.seeded_pilot_freeze is not None and not args.seeded_pilot_variant:
        parser.error("--seeded-pilot-freeze is valid only with --seeded-pilot-variant")
    return args


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def jsonable(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def clone_tree(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def exact_equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(exact_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(exact_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def finite_tree(value: Any) -> bool:
    import numpy as np
    import torch

    if value is None or isinstance(value, (str, bool)):
        return True
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.floating) else True
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def score_vector(value: Any, torch: Any) -> Any:
    scores = torch.as_tensor(value).detach().reshape(-1)
    if scores.numel() != 300 or not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError(f"expected 300 finite candidate scores, got {tuple(scores.shape)}")
    return scores


def candidate_metrics(student_costs: Any, teacher_costs: Any, torch: Any) -> dict[str, Any]:
    student = score_vector(student_costs, torch)
    teacher = score_vector(teacher_costs, torch)
    if student.shape != teacher.shape:
        raise RuntimeError("student and teacher shadow score vectors do not align")
    student_order = torch.argsort(student)
    teacher_order = torch.argsort(teacher)
    student_top120 = set(int(value) for value in student_order[:120].detach().cpu().tolist())
    teacher_top30 = set(int(value) for value in teacher_order[:30].detach().cpu().tolist())
    recall = len(student_top120 & teacher_top30) / 30.0
    raw_std = teacher.std(unbiased=False)
    if not bool(torch.isfinite(raw_std).item()):
        return {
            "recall_at_120": recall,
            "standardized_elite_regret": None,
            "teacher_population_std": float(raw_std.item()),
            "valid": False,
        }
    denominator = torch.clamp(raw_std, min=1e-6)
    student_top30 = student_order[:30]
    teacher_elite_mean = teacher[torch.as_tensor(sorted(teacher_top30), device=teacher.device)].mean()
    student_elite_teacher_mean = teacher[student_top30].mean()
    regret = (student_elite_teacher_mean - teacher_elite_mean) / denominator
    return {
        "recall_at_120": float(recall),
        "standardized_elite_regret": float(regret.item()),
        "teacher_population_std": float(raw_std.item()),
        "standardization_denominator": float(denominator.item()),
        "valid": bool(torch.isfinite(regret).item()),
    }


def normalize_task(row: Mapping[str, Any]) -> dict[str, int]:
    return {
        "selection_order": int(row["selection_order"]),
        "episode_idx": int(row["episode_idx"]),
        "row_index": int(row["row_index"]),
        "start_step": int(row["start_step"]),
    }


def validate_freeze_and_manifest(
    freeze: Mapping[str, Any], manifest: Mapping[str, Any], args: argparse.Namespace
) -> tuple[list[dict[str, int]], list[dict[str, int]], dict[str, Any]]:
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("schema_version") != 1:
        raise RuntimeError("pilot freeze schema identity mismatch")
    if freeze.get("status") != "frozen_before_results":
        raise RuntimeError("pilot freeze must be frozen_before_results")
    if manifest.get("schema") != SELECTION_SCHEMA or manifest.get("schema_version") != SELECTION_VERSION:
        raise RuntimeError("episode-selection manifest schema mismatch")
    if manifest.get("pbs_job_id") != args.selection_job_id:
        raise RuntimeError("selection manifest PBS job ID differs from the command line")

    manifest_contract = freeze.get("selection_manifest", {})
    if str(args.selection_manifest).replace("\\", "/") != str(manifest_contract.get("path", "")).replace("\\", "/"):
        raise RuntimeError("selection manifest path differs from the frozen contract")
    if str(manifest_contract.get("pbs_job_id")) != str(args.selection_job_id):
        raise RuntimeError("selection manifest job ID differs from the frozen contract")
    baseline = freeze.get("upstream_baseline", {})
    if manifest.get("dataset_path") != baseline.get("dataset_path"):
        raise RuntimeError("selection manifest dataset differs from the pilot freeze")
    if manifest.get("selection_freeze", {}).get("schema") != "lewm-pusht-student-induced-shadow-episode-selection":
        raise RuntimeError("selection manifest does not embed the expected CPU selection freeze")

    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, list) or len(manifest_tasks) != 32:
        raise RuntimeError("selection manifest must contain exactly 32 task metadata rows")
    collection_manifest = [normalize_task(row) for row in manifest_tasks if row.get("role") == "collection"]
    holdout_manifest = [normalize_task(row) for row in manifest_tasks if row.get("role") == "reserved_holdout"]
    if len(collection_manifest) != 16 or len(holdout_manifest) != 16:
        raise RuntimeError("selection manifest must contain 16 collection and 16 reserved holdout rows")
    if len({row["episode_idx"] for row in collection_manifest + holdout_manifest}) != 32:
        raise RuntimeError("selection manifest source episodes are not unique")

    collection = [normalize_task(row) for row in freeze.get("collection_tasks", [])]
    holdout = [normalize_task(row) for row in freeze.get("reserved_holdout_tasks", [])]
    gate = [normalize_task(row) for row in freeze.get("paired_gate_tasks", [])]
    if collection != collection_manifest or holdout != holdout_manifest:
        raise RuntimeError("pilot freeze task identities/order differ from the selection manifest")
    if gate != collection[:4] or len(gate) != 4:
        raise RuntimeError("paired gate must be the first four collection tasks in frozen order")
    if {row["episode_idx"] for row in collection} & {row["episode_idx"] for row in holdout}:
        raise RuntimeError("collection overlaps reserved holdout")

    cem = freeze.get("cem", {})
    expected_cem = {
        "batch_size": 1,
        "num_samples": 300,
        "topk": 30,
        "iterations": 30,
        "seed": 42,
        "shadow_rounds_1_indexed": list(SOLVER_ROUNDS),
    }
    for key, value in expected_cem.items():
        if cem.get(key) != value:
            raise RuntimeError(f"frozen CEM contract mismatch: {key}")
    protocol = freeze.get("protocol", {})
    expected_protocol = {"goal_offset_steps": 25, "eval_budget": 50, "max_episode_steps": 100}
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"frozen dataset-evaluator protocol mismatch: {key}")
    if protocol.get("expected_replan_steps") != [0, 25] or protocol.get("receding_horizon") != 5 or protocol.get("action_block") != 5:
        raise RuntimeError("frozen native policy replan schedule drifted")
    metrics = freeze.get("metrics", {})
    if int(metrics.get("recall_k", -1)) != 120 or int(metrics.get("elite_count", -1)) != 30:
        raise RuntimeError("frozen ranking metric dimensions drifted")
    if int(metrics.get("candidate_bank", -1)) != 300:
        raise RuntimeError("frozen candidate bank size drifted")
    formula = str(metrics.get("standardized_elite_regret_formula", "")).lower()
    if not all(token in formula for token in ("top30", "population", "teacher", "student", "1e-6")):
        raise RuntimeError("freeze must spell out the standardized regret formula")
    recall_formula = str(metrics.get("recall_at_120_formula", "")).lower()
    if "teacher top30" not in recall_formula or "student top120" not in recall_formula or "/ 30" not in recall_formula:
        raise RuntimeError("freeze recall@120 must count teacher Top30 recovered by student Top120")
    if int(freeze.get("validity", {}).get("minimum_matched_episodes", -1)) != 12:
        raise RuntimeError("frozen matched-episode minimum must be 12")
    if list(freeze.get("validity", {}).get("t25_status_values", [])) != ["reached", "terminal_before_t25"]:
        raise RuntimeError("frozen t25 status vocabulary drifted")
    go = freeze.get("diagnostic_go_rule", {})
    if float(go.get("median_delta_at_least", math.nan)) != 0.05 or float(go.get("positive_fraction_at_least", math.nan)) != 0.75:
        raise RuntimeError("frozen diagnostic GO thresholds drifted")

    student_checkpoint = freeze.get("student_checkpoint", {})
    if not student_checkpoint.get("path") or not isinstance(student_checkpoint.get("provenance"), Mapping):
        raise RuntimeError("freeze lacks exact student checkpoint path/provenance")
    return collection, gate, dict(student_checkpoint)


def load_modules(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any]:
    search_paths = (
        args.lewm_root,
        args.stablewm_root,
        args.control_root / "lewm-transfer",
        args.baseline_dir,
        args.adaptive_dir,
        args.official_cem_dir,
    )
    sys.path[:0] = [str(path.resolve(strict=True)) for path in search_paths]
    os.environ["STABLEWM_HOME"] = str(args.cache_root.resolve())
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # This function is called only after the real PBS allocation guard.
    import hdf5plugin  # noqa: F401
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import jepa
    import torch
    from hydra.utils import instantiate
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms
    import run_dataset_teacher_baseline as baseline
    import run_adaptive_teacher_schedule as adaptive
    import run_official_pusht_cem as old_router
    from run_lewm_recurrent_student import load_official_checkpoint

    if not Path(swm.__file__).resolve().is_relative_to(args.stablewm_root.resolve(strict=True)):
        raise RuntimeError("stable_worldmodel imported outside the pinned staged root")
    if not Path(jepa.__file__).resolve().is_relative_to(args.lewm_root.resolve(strict=True)):
        raise RuntimeError("LeWM JEPA imported outside the pinned staged root")
    expected_venv = args.stablewm_root.resolve(strict=True).parent / "venv"
    if Path(sys.prefix).resolve(strict=True) != expected_venv.resolve(strict=True):
        raise RuntimeError("Python sys.prefix differs from the staged venv")
    if not Path(spt.__file__).resolve().is_relative_to(expected_venv.resolve(strict=True)):
        raise RuntimeError("stable_pretraining imported outside the staged venv")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    return np, spt, swm, torch, transforms, (baseline, adaptive, old_router, load_official_checkpoint, preprocessing, instantiate)


def first_env_value(value: Any, np: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        result = value.detach().cpu()
    else:
        result = torch.as_tensor(np.asarray(value))
    if result.ndim >= 3:
        return result[0, -1].clone()
    if result.ndim >= 2:
        return result[0].clone()
    return result.clone()


def mask_is_active(mask: Any, torch: Any) -> bool:
    if mask is None:
        return True
    tensor = torch.as_tensor(mask).reshape(-1)
    if tensor.numel() == 0:
        return True
    return bool(tensor[0].item())


def compare_array_values(left: Any, right: Any, np: Any) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "available": False,
            "exact": False,
            "shape_equal": False,
            "dtype_equal": False,
            "max_abs_delta": None,
            "left_shape": None,
            "right_shape": None,
        }
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    shape_equal = left_array.shape == right_array.shape
    dtype_equal = left_array.dtype == right_array.dtype
    if not shape_equal:
        return {
            "available": True,
            "exact": False,
            "shape_equal": False,
            "dtype_equal": dtype_equal,
            "max_abs_delta": None,
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
        }
    numeric = np.issubdtype(left_array.dtype, np.number) or np.issubdtype(left_array.dtype, np.bool_)
    if not numeric or not (
        np.issubdtype(right_array.dtype, np.number) or np.issubdtype(right_array.dtype, np.bool_)
    ):
        exact = dtype_equal and bool(np.array_equal(left_array, right_array))
        return {
            "available": True,
            "exact": exact,
            "shape_equal": True,
            "dtype_equal": dtype_equal,
            "max_abs_delta": 0.0 if exact else None,
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
        }
    equal_elements = left_array == right_array
    if np.issubdtype(left_array.dtype, np.inexact) and np.issubdtype(right_array.dtype, np.inexact):
        equal_elements = equal_elements | (np.isnan(left_array) & np.isnan(right_array))
    exact = dtype_equal and bool(np.all(equal_elements))
    finite_pairs = np.isfinite(left_array) & np.isfinite(right_array)
    if bool(finite_pairs.any()):
        delta = float(np.max(np.abs(
            left_array[finite_pairs].astype(np.float64) - right_array[finite_pairs].astype(np.float64)
        )))
        max_abs_delta = delta if math.isfinite(delta) else None
    else:
        max_abs_delta = 0.0 if exact else None
    return {
        "available": True,
        "exact": exact,
        "shape_equal": True,
        "dtype_equal": dtype_equal,
        "max_abs_delta": max_abs_delta,
        "left_shape": list(left_array.shape),
        "right_shape": list(right_array.shape),
    }


def capture_simulator_signature(world: Any, jsonable_fn: Any) -> dict[str, Any]:
    try:
        env = world.envs.envs[0].unwrapped
        variation = env.variation_space
        signature = {
            "agent_position": [float(value) for value in env.agent.position],
            "agent_velocity": [float(value) for value in env.agent.velocity],
            "block_position": [float(value) for value in env.block.position],
            "block_velocity": [float(value) for value in env.block.velocity],
            "block_angle": float(env.block.angle),
            "agent_shape_variant": jsonable_fn(variation["agent"]["shape"].value),
            "agent_scale": jsonable_fn(variation["agent"]["scale"].value),
            "block_shape_variant": jsonable_fn(variation["block"]["shape"].value),
            "block_scale": jsonable_fn(variation["block"]["scale"].value),
        }
        return {"available": True, "api": "unwrapped PushT variation-space and agent/block body fields", **signature}
    except Exception as exc:
        return {"available": False, "missing_marker": type(exc).__name__}


class PilotCallback:
    output_key = "official_cem_trace"

    def __init__(self, base_class: Any, retain_pair_arrays: bool, retained_rounds: tuple[int, ...] = ()) -> None:
        self.base = base_class(retain_tensors=False)
        self.retain_pair_arrays = retain_pair_arrays
        self.retained_rounds = frozenset(retained_rounds)
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.batch_starts = 0
        self.candidates_finite = True
        self.candidate_shapes: list[list[int]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.base.history

    @property
    def cem_input_shapes(self) -> Any:
        return self.base.cem_input_shapes

    def __getattr__(self, name: str) -> Any:
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()
        self.snapshots.clear()
        self.batch_starts = 0
        self.candidates_finite = True
        self.candidate_shapes.clear()

    def start_batch(self) -> None:
        self.batch_starts += 1
        self.base.start_batch()

    def end_solve(self) -> None:
        self.base.end_solve()

    def __call__(self, **kwargs: Any) -> None:
        import torch

        self.base(**kwargs)
        self.candidates_finite = self.candidates_finite and bool(torch.isfinite(kwargs["candidates"]).all().item())
        round_number = int(kwargs["step"]) + 1
        self.candidate_shapes.append(list(kwargs["candidates"].shape))
        row = {
            "candidate_shape": list(kwargs["candidates"].shape),
            "cost_shape": list(kwargs["costs"].shape),
        }
        if self.retain_pair_arrays or round_number in self.retained_rounds:
            for key in ("candidates", "costs", "mean", "var"):
                row[key] = kwargs[key].detach().cpu().clone()
        if round_number in SOLVER_ROUNDS or self.retain_pair_arrays or round_number in self.retained_rounds:
            self.snapshots[round_number] = row


class ShadowRouter:
    """Return student costs unchanged; compute teacher costs on isolated copies."""

    def __init__(self, official: Any, student_router: Any, shadow_enabled: bool) -> None:
        self.official = official
        self.student_router = student_router
        self.shadow_enabled = shadow_enabled
        self.round_index = 0
        self.scores: dict[int, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        if name in {"official", "student_router"}:
            raise AttributeError(name)
        return getattr(self.student_router, name)

    def parameters(self):
        return self.official.parameters()

    def begin_solve(self) -> None:
        self.round_index = 0
        self.scores = {}
        self.student_router.begin_solve()

    def end_solve(self) -> None:
        self.student_router.end_solve()

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Any):
        import torch

        self.round_index += 1
        round_number = self.round_index
        shadow_inputs = None
        if self.shadow_enabled and round_number in SOLVER_ROUNDS:
            # student_get_cost mutates info_dict. The teacher must see its own
            # untouched copy of the identical pre-student inputs and candidate bank.
            shadow_inputs = (clone_tree(info_dict), action_candidates.detach().clone())
        student_costs = self.student_router.student_get_cost(info_dict, action_candidates)
        if round_number in SOLVER_ROUNDS:
            record: dict[str, Any] = {
                "student_costs": student_costs.detach().reshape(-1).cpu().clone(),
                "teacher_costs": None,
                "shadow_wall_s": 0.0,
            }
            if shadow_inputs is not None:
                info_copy, candidate_copy = shadow_inputs
                started = time.perf_counter()
                with torch.inference_mode():
                    teacher_costs = self.official.get_cost(info_copy, candidate_copy)
                record["shadow_wall_s"] = time.perf_counter() - started
                record["teacher_costs"] = teacher_costs.detach().reshape(-1).cpu().clone()
            self.scores[round_number] = record
        return student_costs


class EpisodeTrace:
    def __init__(
        self, world: Any, np: Any, torch: Any, capture_repro_inputs: bool = False,
        seeded_reset_override: int | None = None,
    ) -> None:
        self.world = world
        self.np = np
        self.torch = torch
        self.capture_repro_inputs = capture_repro_inputs
        self.active_steps = 0
        self.step_calls = 0
        self.pending: dict[str, Any] | None = None
        self.pending_repro_action: Any = None
        self.commits: list[dict[str, Any]] = []
        self.repro_commits: list[dict[str, Any]] = []
        self.state_trace: list[Any] = []
        self.termination_signals: list[dict[str, Any]] = []
        self.initial: dict[str, Any] | None = None
        self.repro_initial_info: dict[str, Any] | None = None
        self.simulator_signature: dict[str, Any] = {"available": False, "missing_marker": "not_requested"}
        self.reset_seed_observed: list[Any] = []
        self.reset_seed_effective: list[Any] = []
        self.solve_records: list[dict[str, Any]] = []
        self.error: str | None = None

        if capture_repro_inputs or seeded_reset_override is not None:
            original_reset = world.reset

            def tracked_reset(*args: Any, **kwargs: Any):
                if seeded_reset_override is not None and self.reset_seed_observed:
                    raise RuntimeError("seeded reproducibility requires exactly one dataset World.reset call")
                reset_seed = kwargs.get("seed", args[0] if args else None)
                self.reset_seed_observed.append(jsonable(reset_seed))
                effective_seed = reset_seed
                if seeded_reset_override is not None:
                    if reset_seed is not None:
                        raise RuntimeError("seeded reproducibility expected dataset World.reset(seed=None)")
                    effective_seed = seeded_reset_override
                    if "seed" in kwargs:
                        kwargs = {**kwargs, "seed": effective_seed}
                    elif args:
                        args = (effective_seed, *args[1:])
                    else:
                        kwargs = {**kwargs, "seed": effective_seed}
                self.reset_seed_effective.append(jsonable(effective_seed))
                return original_reset(*args, **kwargs)

            world.reset = tracked_reset

        original_get_actions = world._get_actions

        def tracked_get_actions(*args: Any, **kwargs: Any):
            if self.initial is None:
                infos = self.world.infos
                self.initial = {
                    key: jsonable(first_env_value(infos[key], self.np, self.torch))
                    for key in OBS_KEYS
                    if key in infos
                }
                if self.capture_repro_inputs:
                    self.repro_initial_info = {
                        key: clone_tree(first_env_value(infos[key], self.np, self.torch))
                        for key in REPRO_INPUT_KEYS
                        if key in infos
                    }
                    self.simulator_signature = capture_simulator_signature(self.world, jsonable)
            return original_get_actions(*args, **kwargs)

        world._get_actions = tracked_get_actions

        original_step = world.envs.step

        def tracked_step(actions: Any, *args: Any, **kwargs: Any):
            mask = kwargs.get("mask")
            if mask is None and args:
                mask = args[0]
            self.pending = {
                "action": jsonable(clone_tree(actions)),
                "mask": jsonable(clone_tree(mask)) if mask is not None else None,
                "active": mask_is_active(mask, self.torch),
            }
            if self.capture_repro_inputs:
                self.pending_repro_action = clone_tree(actions)
            self.step_calls += 1
            result = original_step(actions, *args, **kwargs)
            return result

        world.envs.step = tracked_step

        original_run_iter = world._run_iter

        def tracked_run_iter(*args: Any, **kwargs: Any):
            original_on_step = kwargs.get("on_step")

            def on_step(current_world: Any) -> None:
                if original_on_step is not None:
                    original_on_step(current_world)
                if self.pending is None:
                    raise RuntimeError("World._run_iter on_step had no matching real envs.step call")
                pending = self.pending
                self.pending = None
                state = None
                if "state" in current_world.infos:
                    state = first_env_value(current_world.infos["state"], self.np, self.torch)
                if pending["active"]:
                    self.active_steps += 1
                    terminateds = getattr(current_world, "terminateds", None)
                    truncateds = getattr(current_world, "truncateds", None)
                    if terminateds is None or truncateds is None:
                        raise RuntimeError("World step did not expose terminateds and truncateds")
                    terminated = bool(first_env_value(terminateds, self.np, self.torch).reshape(-1)[0].item())
                    truncated = bool(first_env_value(truncateds, self.np, self.torch).reshape(-1)[0].item())
                    self.termination_signals.append({
                        "active_step": self.active_steps,
                        "terminated": terminated,
                        "truncated": truncated,
                    })
                    self.commits.append({
                        "active_step": self.active_steps,
                        "action": pending["action"],
                        "mask": pending["mask"],
                        "state_after": jsonable(state) if state is not None else None,
                    })
                    if state is not None:
                        self.state_trace.append(jsonable(state))
                    if self.capture_repro_inputs:
                        self.repro_commits.append({
                            "active_step": self.active_steps,
                            "action": self.pending_repro_action,
                            "state_after": clone_tree(state),
                        })
                self.pending_repro_action = None

            kwargs["on_step"] = on_step
            yield from original_run_iter(*args, **kwargs)

        world._run_iter = tracked_run_iter


class PilotPolicyFactory:
    @staticmethod
    def build(policy_class: Any, solver: Any, plan_config: Any, process: Mapping[str, Any], transform: Mapping[str, Any]):
        class Policy(policy_class):
            def __init__(self) -> None:
                super().__init__(solver=solver, config=plan_config, process=dict(process), transform=dict(transform))

        return Policy()


def state_bytes_equal(left: Any, right: Any) -> bool:
    import torch

    return torch.equal(left, right)


def run_episode(
    task: Mapping[str, int],
    shadow_enabled: bool,
    retain_pair_arrays: bool,
    args: argparse.Namespace,
    baseline: Any,
    swm: Any,
    cfg: Any,
    dataset: Any,
    process: Mapping[str, Any],
    transforms: Any,
    torch: Any,
    spt: Any,
    official: Any,
    reference: Any,
    student: Any,
    old_router: Any,
    instantiate: Any,
    np: Any,
    capture_repro_inputs: bool = False,
    seeded_reset_override: int | None = None,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    episode_cfg = copy.deepcopy(cfg)
    episode_cfg.world.num_envs = 1
    world = baseline.make_pinned_world(swm, episode_cfg)
    image_transform = baseline.make_image_transform(episode_cfg, spt, transforms, torch)
    transform = {"pixels": image_transform, "goal": baseline.make_image_transform(episode_cfg, spt, transforms, torch)}
    student_router = old_router.RoutedCostModel(official, reference, student, "student_only")
    router = ShadowRouter(official, student_router, shadow_enabled)
    callback = PilotCallback(
        old_router.TraceCallback,
        retain_pair_arrays,
        retained_rounds=(1,) if capture_repro_inputs else (),
    )
    solver = instantiate(episode_cfg.solver, model=router, callbacks=[callback])
    if not hasattr(solver, "torch_gen") or int(solver.torch_gen.initial_seed()) != 42:
        world.close()
        raise RuntimeError("native CEM generator seed is not the frozen 42")
    if int(episode_cfg.world.num_envs) != 1 or int(episode_cfg.solver.batch_size) != 1:
        world.close()
        raise RuntimeError("episode evaluator must use a single environment and one CEM batch")

    plan_config = swm.PlanConfig(**episode_cfg.plan_config)
    policy = PilotPolicyFactory.build(swm.policy.WorldModelPolicy, solver, plan_config, process, transform)
    world.set_policy(policy)
    trace = EpisodeTrace(
        world, np, torch, capture_repro_inputs=capture_repro_inputs,
        seeded_reset_override=seeded_reset_override,
    )
    original_solve = solver.solve
    rng_states: list[dict[str, Any]] = []

    def tracked_solve(*solve_args: Any, **solve_kwargs: Any):
        ordinal = len(trace.solve_records)
        router.begin_solve()
        callback.reset()
        rng_before = solver.torch_gen.get_state().clone()
        pre_solve_state = first_env_value(world.infos["state"], np, torch) if "state" in world.infos else None
        started = time.perf_counter()
        result = original_solve(*solve_args, **solve_kwargs)
        elapsed = time.perf_counter() - started
        rng_after = solver.torch_gen.get_state().clone()
        router.end_solve()
        action_output = result.get("actions") if isinstance(result, Mapping) else None
        if action_output is None or not finite_tree(action_output):
            raise FloatingPointError("native CEM did not return finite executable actions")
        score_records = {}
        for round_number in SOLVER_ROUNDS:
            if round_number not in router.scores or round_number not in callback.snapshots:
                raise RuntimeError(f"CEM solve {ordinal} omitted frozen round {round_number}")
            score = router.scores[round_number]
            callback_row = callback.snapshots[round_number]
            if tuple(callback_row["candidate_shape"]) != (1, 300, 5, 10):
                raise RuntimeError(f"candidate bank shape drifted at CEM round {round_number}: {callback_row['candidate_shape']}")
            if shadow_enabled and score["teacher_costs"] is None:
                raise RuntimeError("teacher shadow cost missing at a frozen CEM round")
            if not math.isfinite(float(score["shadow_wall_s"])) or (shadow_enabled and score["shadow_wall_s"] <= 0.0):
                raise FloatingPointError(f"teacher shadow timing is invalid at round {round_number}")
            if retain_pair_arrays:
                if not torch.equal(score["student_costs"].reshape(1, 300), callback_row["costs"].reshape(1, 300).float()):
                    raise RuntimeError("student score capture differs from the actual CEM cost vector")
            metric_row = candidate_metrics(score["student_costs"], score["teacher_costs"], torch) if score["teacher_costs"] is not None else None
            if metric_row is not None and not metric_row["valid"]:
                raise FloatingPointError(f"invalid candidate ranking metrics at round {round_number}")
            score_records[str(round_number)] = {
                "student_costs": score["student_costs"],
                "teacher_costs": score["teacher_costs"],
                "shadow_wall_s": float(score["shadow_wall_s"]),
                "candidate_shape": callback_row["candidate_shape"],
                "callback_cost_shape": callback_row["cost_shape"],
                "metrics": metric_row,
            }
            if retain_pair_arrays:
                score_records[str(round_number)]["pair_arrays"] = callback_row
        record = {
            "solve_ordinal": ordinal,
            "task": dict(task),
            "sim_steps_before_solve": trace.active_steps,
            "batch_ordinal": 0,
            "batches_in_solve": callback.batch_starts,
            "solver_seed": int(solver.torch_gen.initial_seed()),
            "generator_state_advanced": not state_bytes_equal(rng_before, rng_after),
            "pre_solve_state": jsonable(pre_solve_state) if pre_solve_state is not None else None,
            "solve_seconds_including_shadow": float(elapsed),
            "student_actions_from_solver": jsonable(action_output),
            "candidate_shape": callback.cem_input_shapes.get("candidates") if callback.cem_input_shapes else None,
            "rounds_recorded": sorted(int(key) for key in score_records),
            "scores": score_records,
            "_rng_before": rng_before,
            "_rng_after": rng_after,
            "_round_states": callback.snapshots if retain_pair_arrays else None,
            "_candidate_shapes": callback.candidate_shapes.copy(),
            "_first_round_arrays": callback.snapshots.get(1) if capture_repro_inputs else None,
        }
        if callback.batch_starts != 1:
            raise RuntimeError(f"singleton CEM solve expected exactly one batch, got {callback.batch_starts}")
        if len(callback.history) != 30 or router.round_index != 30:
            raise RuntimeError("native CEM solve did not produce exactly 30 locally numbered cost rounds")
        if len(callback.candidate_shapes) != 30:
            raise RuntimeError("candidate callback did not observe all 30 rounds")
        if not callback.candidates_finite or not all(bool(row["finite_flags"].item()) for row in callback.history):
            raise FloatingPointError("candidate bank, score, mean, or variance was non-finite")
        if not math.isfinite(float(elapsed)) or elapsed <= 0.0:
            raise FloatingPointError("CEM solve duration is not finite and positive")
        trace.solve_records.append(record)
        rng_states.append({"before": rng_before, "after": rng_after})
        return result

    solver.solve = tracked_solve
    try:
        metrics = baseline.evaluate_selected_tasks(world, dataset, [dict(task)], episode_cfg, OmegaConf)
        torch.cuda.synchronize()
    finally:
        world.close()

    if seeded_reset_override is not None:
        if "seed" in getattr(dataset, "column_names", []):
            raise RuntimeError("seeded pilot requires the frozen dataset seed column to be absent")
        if trace.reset_seed_observed != [None] or trace.reset_seed_effective != [seeded_reset_override]:
            raise RuntimeError("seeded pilot requires exactly one requested None reset forwarded as seed 42")

    metric_json = jsonable(metrics)
    successes = metric_json.get("episode_successes") if isinstance(metric_json, Mapping) else None
    if not isinstance(successes, list) or len(successes) != 1:
        raise RuntimeError("singleton World.evaluate returned no single success outcome")
    solve_records = trace.solve_records
    if not solve_records or solve_records[0]["sim_steps_before_solve"] != 0:
        raise RuntimeError("dataset-start t0 CEM solve was not observed at simulator step zero")
    if any(item["sim_steps_before_solve"] not in (0, 25) for item in solve_records):
        raise RuntimeError("unexpected native CEM solve location; expected only t0 and t25")
    if len(solve_records) > 1:
        if len(solve_records) != 2 or solve_records[1]["sim_steps_before_solve"] != 25:
            raise RuntimeError("second native CEM solve did not occur after exactly 25 real transitions")
        if not state_bytes_equal(solve_records[0]["_rng_after"], solve_records[1]["_rng_before"]):
            raise RuntimeError("native solver RNG was reseeded or changed between t0 and t25")
        if not exact_equal(solve_records[1]["pre_solve_state"], trace.state_trace[24]):
            raise RuntimeError("t25 pre-solve state differs from the 25th committed env.step post-state")

    if trace.pending is not None:
        raise RuntimeError("World._run_iter ended with an env.step that lacked post-step evidence")
    if trace.step_calls < trace.active_steps or trace.active_steps != len(trace.commits):
        raise RuntimeError("actual env.step call count and committed transition trace disagree")
    if trace.initial is None or "state" not in trace.initial or "goal_state" not in trace.initial:
        raise RuntimeError("initial dataset simulator state/goal were not captured")
    if not finite_tree(trace.initial) or not finite_tree(trace.commits) or not finite_tree(trace.state_trace):
        raise FloatingPointError("initial simulator observations, committed actions, or post-step states are non-finite")
    if not exact_equal(solve_records[0]["pre_solve_state"], trace.initial["state"]):
        raise RuntimeError("t0 pre-solve state differs from the selected dataset-start state")
    for record in solve_records:
        record["rng_contiguous_from_previous_solve"] = True if record["solve_ordinal"] == 0 else state_bytes_equal(
            solve_records[record["solve_ordinal"] - 1]["_rng_after"], record["_rng_before"]
        )
    t25 = next((item for item in solve_records if item["sim_steps_before_solve"] == 25), None)
    terminal_after_step25 = any(
        row["active_step"] == 25 and (row["terminated"] or row["truncated"])
        for row in trace.termination_signals
    )
    if t25 is not None:
        reached_status = "reached"
    elif trace.active_steps < 25:
        reached_status = "terminal_before_t25"
    elif seeded_reset_override is not None and trace.active_steps == 25 and terminal_after_step25:
        reached_status = "terminal_at_t25_before_replan"
    else:
        raise RuntimeError(
            f"missing t25 CEM solve without a terminal/truncated signal at step 25 "
            f"(active_steps={trace.active_steps})"
        )

    public_solves = []
    for record in solve_records:
        public = {key: jsonable(value) for key, value in record.items() if not key.startswith("_") and key != "scores"}
        public["scores"] = {
            round_key: {
                key: jsonable(value)
                for key, value in score.items()
                if key not in ("student_costs", "teacher_costs", "pair_arrays")
            }
            for round_key, score in record["scores"].items()
        }
        public_solves.append(public)
    score_rows = []
    for record in solve_records:
        for round_key, score in record["scores"].items():
            score_rows.append({
                "task": dict(task),
                "solve_ordinal": record["solve_ordinal"],
                "sim_steps_before_solve": record["sim_steps_before_solve"],
                "round": int(round_key),
                "candidate_shape": score["candidate_shape"],
                "student_costs": jsonable(score["student_costs"]),
                "teacher_costs": jsonable(score["teacher_costs"]),
                "shadow_wall_s": score["shadow_wall_s"],
                "metrics": score["metrics"],
            })
    episode_result = {
        "task": dict(task),
        "shadow_enabled": shadow_enabled,
        "episode_seed": 42,
        "native_solver_seed": 42,
        "success": bool(successes[0]),
        "t25_status": reached_status,
        "actual_env_step_count": trace.active_steps,
        "termination_signals": trace.termination_signals,
        "env_step_call_count_including_masked_calls": trace.step_calls,
        "initial_simulator_observation": trace.initial,
        "committed_action_and_state_trace": trace.commits,
        "state_trace": trace.state_trace,
        "solve_records": public_solves,
        "score_rows": score_rows,
        "world_evaluate_metrics": metric_json,
        "_solves": solve_records,
        "_rng_states": rng_states,
        "_initial": trace.initial,
        "_commits": trace.commits,
        "_state_trace": trace.state_trace,
        "_repro_initial_info": trace.repro_initial_info,
        "_simulator_signature": trace.simulator_signature,
        "_reset_seed_observed": trace.reset_seed_observed,
        "_reset_seed_effective": trace.reset_seed_effective,
        "_dataset_seed_column_present": "seed" in getattr(dataset, "column_names", []),
        "_repro_commits": trace.repro_commits,
    }
    if seeded_reset_override is not None:
        episode_result["dataset_reset_seed"] = {
            "dataset_seed_column_present": False,
            "world_reset_requested_seed": trace.reset_seed_observed,
            "world_reset_effective_seed": trace.reset_seed_effective,
            "world_reset_call_count": len(trace.reset_seed_observed),
        }
    return episode_result


def compare_paired_runs(control: Mapping[str, Any], shadow: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "task_identity": control["task"] == shadow["task"],
        "initial_simulator_observation": exact_equal(control["_initial"], shadow["_initial"]),
        "success": control["success"] == shadow["success"],
        "actual_env_step_count": control["actual_env_step_count"] == shadow["actual_env_step_count"],
        "env_step_call_count": control["env_step_call_count_including_masked_calls"] == shadow["env_step_call_count_including_masked_calls"],
        "committed_actions_and_post_step_states": exact_equal(control["_commits"], shadow["_commits"]),
        "state_trace": exact_equal(control["_state_trace"], shadow["_state_trace"]),
        "solve_count": len(control["_solves"]) == len(shadow["_solves"]),
    }
    per_solve: list[dict[str, Any]] = []
    if len(control["_solves"]) == len(shadow["_solves"]):
        for left, right in zip(control["_solves"], shadow["_solves"], strict=True):
            local = {
                "solve_ordinal": left["solve_ordinal"],
                "task_active_and_sim_step": left["task"] == right["task"] and left["sim_steps_before_solve"] == right["sim_steps_before_solve"],
                "rng_state_before_exact": state_bytes_equal(left["_rng_before"], right["_rng_before"]),
                "rng_state_after_exact": state_bytes_equal(left["_rng_after"], right["_rng_after"]),
                "solver_action_exact": exact_equal(left["student_actions_from_solver"], right["student_actions_from_solver"]),
                "candidate_shape_equal": left["candidate_shape"] == right["candidate_shape"],
                "batch_count_is_one_both": left["batches_in_solve"] == right["batches_in_solve"] == 1,
                "round_arrays_exact": True,
            }
            left_rounds = left.get("_round_states")
            right_rounds = right.get("_round_states")
            if left.get("_candidate_shapes") != right.get("_candidate_shapes"):
                local["candidate_shape_equal"] = False
            if not isinstance(left_rounds, Mapping) or not isinstance(right_rounds, Mapping) or left_rounds.keys() != right_rounds.keys() or len(left_rounds) != 30:
                local["round_arrays_exact"] = False
            else:
                for round_key in left_rounds:
                    larr = left_rounds[round_key]
                    rarr = right_rounds[round_key]
                    for field in ("candidates", "costs", "mean", "var"):
                        if not exact_equal(larr[field], rarr[field]):
                            local["round_arrays_exact"] = False
                            break
                    if not local["round_arrays_exact"]:
                        break
            local["pass"] = all(value for key, value in local.items() if key not in ("solve_ordinal", "pass"))
            per_solve.append(local)
    checks["every_solve_exact"] = bool(per_solve) and all(row["pass"] for row in per_solve)
    checks["all_required"] = all(checks.values())
    return {"checks": checks, "per_solve": per_solve, "pass": checks["all_required"]}


def compare_trace_field(
    left: list[Mapping[str, Any]], right: list[Mapping[str, Any]], field: str, np: Any, with_delta: bool
) -> dict[str, Any]:
    first_mismatch = None
    first_shadow_mismatch = None
    exact = len(left) == len(right)
    max_abs_delta = 0.0 if with_delta else None
    delta_complete = True
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        same_step = left_row.get("active_step") == right_row.get("active_step")
        same_value = exact_equal(left_row.get(field), right_row.get(field))
        if not same_step or not same_value:
            exact = False
            if first_mismatch is None:
                first_mismatch = left_row.get("active_step", right_row.get("active_step", index + 1))
                first_shadow_mismatch = right_row.get("active_step", left_row.get("active_step", index + 1))
        if with_delta:
            left_value = np.asarray(left_row.get(field))
            right_value = np.asarray(right_row.get(field))
            if left_value.shape != right_value.shape or not (
                np.issubdtype(left_value.dtype, np.number) or np.issubdtype(left_value.dtype, np.bool_)
            ) or not (
                np.issubdtype(right_value.dtype, np.number) or np.issubdtype(right_value.dtype, np.bool_)
            ):
                delta_complete = False
            elif left_value.size:
                delta = float(np.max(np.abs(left_value.astype(np.float64) - right_value.astype(np.float64))))
                if not math.isfinite(delta):
                    delta_complete = False
                else:
                    max_abs_delta = max(max_abs_delta, delta)
    if len(left) != len(right) and first_mismatch is None:
        extra = left[len(right)] if len(left) > len(right) else right[len(left)]
        first_mismatch = extra.get("active_step", min(len(left), len(right)) + 1)
        first_shadow_mismatch = first_mismatch
    return {
        "exact": exact,
        "first_mismatch_active_step": first_mismatch,
        "first_mismatch_shadow_active_step": first_shadow_mismatch,
        "max_abs_delta": max_abs_delta if with_delta and delta_complete else None,
        "delta_complete": delta_complete if with_delta else None,
        "control_steps": len(left),
        "shadow_steps": len(right),
    }


def compare_execution_traces(control: Mapping[str, Any], shadow: Mapping[str, Any], np: Any) -> dict[str, Any]:
    left = control["_commits"]
    right = shadow["_commits"]
    actions = compare_trace_field(left, right, "action", np, with_delta=True)
    masks = compare_trace_field(left, right, "mask", np, with_delta=False)
    states = compare_trace_field(left, right, "state_after", np, with_delta=True)
    field_checks = {"action": actions, "mask": masks, "post_step_state": states}
    mismatches = [
        (field, check["first_mismatch_active_step"], check["first_mismatch_shadow_active_step"])
        for field, check in field_checks.items()
        if check["first_mismatch_active_step"] is not None
    ]
    first_step = min((int(row[1]) for row in mismatches), default=None)
    return {
        "first_execution_mismatch": {
            "control_active_step": first_step,
            "shadow_active_step": next((int(row[2]) for row in mismatches if int(row[1]) == first_step), None),
            "fields": [row[0] for row in mismatches if int(row[1]) == first_step],
        } if first_step is not None else None,
        "committed_action_and_mask_exact": actions["exact"] and masks["exact"],
        "committed_actions": actions,
        "commit_masks": masks,
        "post_step_state_after_exact": states["exact"],
        "post_step_states": states,
        "state_trace_exact": exact_equal(control["_state_trace"], shadow["_state_trace"]),
        "raw_step_traces_written": False,
    }


def compare_repro_inputs(control: Mapping[str, Any], repeat: Mapping[str, Any], np: Any) -> dict[str, Any]:
    left = control.get("_repro_initial_info")
    right = repeat.get("_repro_initial_info")
    fields: dict[str, Any] = {}
    for key in REPRO_INPUT_KEYS:
        left_value = left.get(key) if isinstance(left, Mapping) else None
        right_value = right.get(key) if isinstance(right, Mapping) else None
        if left_value is not None and hasattr(left_value, "detach"):
            left_value = left_value.detach().cpu().numpy()
        if right_value is not None and hasattr(right_value, "detach"):
            right_value = right_value.detach().cpu().numpy()
        fields[key] = compare_array_values(left_value, right_value, np)
    return {
        "exact": all(value["available"] and value["exact"] for value in fields.values()),
        "fields": fields,
        "raw_inputs_written": False,
    }


def compare_first_round_arrays(control: Mapping[str, Any], repeat: Mapping[str, Any], np: Any) -> dict[str, Any]:
    left_solves = control.get("_solves", [])
    right_solves = repeat.get("_solves", [])
    left = left_solves[0].get("_first_round_arrays") if left_solves else None
    right = right_solves[0].get("_first_round_arrays") if right_solves else None
    fields: dict[str, Any] = {}
    for key in ("candidates", "costs", "mean", "var"):
        left_value = left.get(key) if isinstance(left, Mapping) else None
        right_value = right.get(key) if isinstance(right, Mapping) else None
        if left_value is not None and hasattr(left_value, "detach"):
            left_value = left_value.detach().cpu().numpy()
        if right_value is not None and hasattr(right_value, "detach"):
            right_value = right_value.detach().cpu().numpy()
        fields[key] = compare_array_values(left_value, right_value, np)
    return {
        "available": all(value["available"] for value in fields.values()),
        "exact": all(value["available"] and value["exact"] for value in fields.values()),
        "fields": fields,
        "raw_round_arrays_written": False,
    }


def compare_first_execution(control: Mapping[str, Any], repeat: Mapping[str, Any], np: Any) -> dict[str, Any]:
    left_commits = control.get("_repro_commits", control.get("_commits", []))
    right_commits = repeat.get("_repro_commits", repeat.get("_commits", []))
    if not left_commits or not right_commits:
        return {
            "available": False,
            "missing_marker": "first_active_commit_missing",
            "committed_first_action_exact": False,
            "first_post_step_state_exact": False,
            "committed_first_action_max_abs_delta": None,
            "first_post_step_state_max_abs_delta": None,
        }
    left_action = np.asarray(left_commits[0]["action"])
    right_action = np.asarray(right_commits[0]["action"])
    left_state = np.asarray(left_commits[0]["state_after"])
    right_state = np.asarray(right_commits[0]["state_after"])
    action = compare_array_values(left_action, right_action, np)
    state = compare_array_values(left_state, right_state, np)
    return {
        "available": True,
        "active_step": left_commits[0].get("active_step"),
        "committed_first_action_exact": action["exact"],
        "committed_first_action_max_abs_delta": action["max_abs_delta"],
        "first_post_step_state_exact": state["exact"],
        "first_post_step_state_max_abs_delta": state["max_abs_delta"],
    }


def compare_simulator_signatures(control: Mapping[str, Any], repeat: Mapping[str, Any]) -> dict[str, Any]:
    left = control.get("_simulator_signature", {"available": False})
    right = repeat.get("_simulator_signature", {"available": False})
    available = bool(left.get("available")) and bool(right.get("available"))
    if not available:
        return {
            "available": False,
            "exact": False,
            "control": left,
            "repeat": right,
            "missing_marker": "signature_api_unavailable",
        }
    left_values = {key: value for key, value in left.items() if key not in ("available", "api")}
    right_values = {key: value for key, value in right.items() if key not in ("available", "api")}
    return {
        "available": True,
        "exact": exact_equal(left_values, right_values),
        "control": left_values,
        "repeat": right_values,
    }


def validate_diagnostic_freeze(
    diagnostic_freeze: Mapping[str, Any],
    pilot_freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    gate_tasks: list[dict[str, int]],
    args: argparse.Namespace,
) -> list[dict[str, int]]:
    if diagnostic_freeze.get("schema") != DIAGNOSTIC_FREEZE_SCHEMA or diagnostic_freeze.get("schema_version") != 1:
        raise RuntimeError("two-pair diagnostic freeze schema identity mismatch")
    if diagnostic_freeze.get("status") != "frozen_for_followup_before_diagnostic_run":
        raise RuntimeError("two-pair diagnostic freeze has an unexpected status")
    if Path(str(diagnostic_freeze.get("pilot_freeze_filename", ""))).name != args.freeze.name:
        raise RuntimeError("diagnostic freeze references a different pilot freeze")
    if diagnostic_freeze.get("pilot_freeze_schema") != pilot_freeze.get("schema"):
        raise RuntimeError("diagnostic freeze pilot schema reference differs")
    if diagnostic_freeze.get("source_pilot_job_id") != "25537036.pbs101":
        raise RuntimeError("diagnostic freeze source gate job differs")
    if diagnostic_freeze.get("selection_manifest_job_id") != manifest.get("pbs_job_id"):
        raise RuntimeError("diagnostic freeze selection job differs from the manifest")
    tasks_value = diagnostic_freeze.get("tasks")
    if not isinstance(tasks_value, list) or len(tasks_value) != 2:
        raise RuntimeError("diagnostic freeze must contain exactly two tasks")
    tasks = [normalize_task(row) for row in tasks_value]
    if [row["episode_idx"] for row in tasks] != [9136, 12704]:
        raise RuntimeError("diagnostic freeze must target only episodes 9136 and 12704")
    if tasks != gate_tasks[:2] or len(gate_tasks) != 4:
        raise RuntimeError("diagnostic tasks must match the first two frozen paired-gate tasks")
    if {row["episode_idx"] for row in tasks} & {
        int(row["episode_idx"]) for row in pilot_freeze.get("reserved_holdout_tasks", [])
    }:
        raise RuntimeError("diagnostic task overlaps the reserved holdout")
    return tasks


def validate_repro_freeze(
    repro_freeze: Mapping[str, Any],
    pilot_freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    collection: list[dict[str, int]],
    args: argparse.Namespace,
) -> dict[str, int]:
    if repro_freeze.get("schema") != REPRO_FREEZE_SCHEMA or repro_freeze.get("schema_version") != 1:
        raise RuntimeError("control-control reproducibility freeze schema identity mismatch")
    if repro_freeze.get("status") != "frozen_for_followup_before_reproducibility_run":
        raise RuntimeError("control-control reproducibility freeze has an unexpected status")
    if Path(str(repro_freeze.get("pilot_freeze_filename", ""))).name != args.freeze.name:
        raise RuntimeError("reproducibility freeze references a different pilot freeze")
    if repro_freeze.get("pilot_freeze_schema") != pilot_freeze.get("schema"):
        raise RuntimeError("reproducibility freeze pilot schema reference differs")
    if repro_freeze.get("selection_manifest_job_id") != manifest.get("pbs_job_id"):
        raise RuntimeError("reproducibility freeze selection job differs from the manifest")
    task_value = repro_freeze.get("task")
    if not isinstance(task_value, Mapping):
        raise RuntimeError("reproducibility freeze must contain one task")
    task = normalize_task(task_value)
    if task["episode_idx"] != 12704:
        raise RuntimeError("control-control reproducibility diagnostic must target episode 12704")
    if task not in collection:
        raise RuntimeError("reproducibility task does not match the frozen collection manifest")
    settings = repro_freeze.get("protocol", {})
    if (
        int(settings.get("repetitions", -1)) != 2
        or settings.get("shadow_enabled") is not False
        or int(settings.get("episode_seed", -1)) != 42
        or int(settings.get("native_solver_seed", -1)) != 42
    ):
        raise RuntimeError("control-control reproducibility protocol drifted")
    if repro_freeze.get("source_split_diagnostic_job_id") != "25537175.pbs101":
        raise RuntimeError("reproducibility freeze source split-diagnostic job differs")
    return task


def validate_seeded_repro_freeze(
    seeded_freeze: Mapping[str, Any],
    pilot_freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    collection: list[dict[str, int]],
    args: argparse.Namespace,
) -> dict[str, int]:
    if seeded_freeze.get("schema") != SEEDED_REPRO_FREEZE_SCHEMA or seeded_freeze.get("schema_version") != 1:
        raise RuntimeError("seeded reproducibility freeze schema identity mismatch")
    if seeded_freeze.get("status") != "frozen_for_seeded_followup_before_result":
        raise RuntimeError("seeded reproducibility freeze has an unexpected status")
    if Path(str(seeded_freeze.get("pilot_freeze_filename", ""))).name != args.freeze.name:
        raise RuntimeError("seeded reproducibility freeze references a different pilot freeze")
    if seeded_freeze.get("pilot_freeze_schema") != pilot_freeze.get("schema"):
        raise RuntimeError("seeded reproducibility freeze pilot schema reference differs")
    if seeded_freeze.get("selection_manifest_job_id") != manifest.get("pbs_job_id"):
        raise RuntimeError("seeded reproducibility freeze selection job differs from the manifest")
    task_value = seeded_freeze.get("task")
    if not isinstance(task_value, Mapping):
        raise RuntimeError("seeded reproducibility freeze must contain one task")
    task = normalize_task(task_value)
    if task["episode_idx"] != 12704 or task not in collection:
        raise RuntimeError("seeded reproducibility task does not match frozen collection episode 12704")
    protocol = seeded_freeze.get("protocol", {})
    if (
        int(protocol.get("repetitions", -1)) != 2
        or protocol.get("shadow_enabled") is not False
        or int(protocol.get("episode_seed", -1)) != 42
        or int(protocol.get("native_solver_seed", -1)) != 42
        or "requested_dataset_reset_seed" not in protocol
        or protocol.get("requested_dataset_reset_seed") is not None
        or protocol.get("dataset_seed_column_present") is not False
        or int(protocol.get("effective_dataset_reset_seed", -1)) != 42
        or int(protocol.get("expected_world_reset_calls_per_run", -1)) != 1
        or protocol.get("intercept_only_dataset_world_reset_seed_none") is not True
        or protocol.get("fresh_world_and_solver_each_repetition") is not True
        or protocol.get("compare_initial_world_info_keys") != list(REPRO_INPUT_KEYS)
        or protocol.get("compare_first_cem_round_fields") != ["candidates", "costs", "mean", "var"]
        or protocol.get("compare_first_committed_action_and_post_step_state") is not True
        or protocol.get("write_raw_images_or_cem_arrays") is not False
        or protocol.get("claim_boundary") != "controlled seeded protocol variant; not the official dataset-start baseline"
    ):
        raise RuntimeError("seeded reproducibility protocol drifted")
    scope = seeded_freeze.get("scope", {})
    if (
        scope.get("episode_ids_evaluated") != [12704]
        or scope.get("other_collection_tasks_evaluated") is not False
        or scope.get("reserved_holdout_evaluated_or_scored") is not False
        or seeded_freeze.get("output_filename") != "seeded_control_control_repro.json"
    ):
        raise RuntimeError("seeded reproducibility scope or output drifted")
    return task


def validate_seeded_pilot_freeze(
    seeded_freeze: Mapping[str, Any],
    pilot_freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    collection: list[dict[str, int]],
    gate_tasks: list[dict[str, int]],
) -> None:
    if seeded_freeze.get("schema") != SEEDED_PILOT_FREEZE_SCHEMA or seeded_freeze.get("schema_version") != 1:
        raise RuntimeError("seeded pilot freeze schema identity mismatch")
    if seeded_freeze.get("status") != "frozen_before_seeded_variant_results":
        raise RuntimeError("seeded pilot freeze must precede variant results")
    if seeded_freeze.get("base_pilot_freeze_filename") != "PILOT_FREEZE.json":
        raise RuntimeError("seeded pilot must inherit the frozen baseline pilot identity")
    if seeded_freeze.get("base_pilot_freeze_schema") != pilot_freeze.get("schema"):
        raise RuntimeError("seeded pilot base-freeze schema differs")
    if seeded_freeze.get("selection_manifest_job_id") != manifest.get("pbs_job_id"):
        raise RuntimeError("seeded pilot selection manifest identity differs")
    if [normalize_task(row) for row in seeded_freeze.get("collection_tasks", [])] != collection:
        raise RuntimeError("seeded pilot collection tasks differ from PILOT_FREEZE")
    if [normalize_task(row) for row in seeded_freeze.get("paired_gate_tasks", [])] != gate_tasks or gate_tasks != collection[:4]:
        raise RuntimeError("seeded pilot paired gate differs from the frozen first four tasks")
    if [normalize_task(row) for row in seeded_freeze.get("reserved_holdout_tasks", [])] != [
        normalize_task(row) for row in pilot_freeze.get("reserved_holdout_tasks", [])
    ]:
        raise RuntimeError("seeded pilot reserved holdout identities differ from PILOT_FREEZE")
    if seeded_freeze.get("cem") != pilot_freeze.get("cem"):
        raise RuntimeError("seeded pilot CEM settings differ from PILOT_FREEZE")
    if seeded_freeze.get("shadow_scoring_rounds") != list(SOLVER_ROUNDS):
        raise RuntimeError("seeded pilot shadow rounds differ from frozen rounds")
    for key in ("metrics", "upstream_baseline", "student_checkpoint"):
        if seeded_freeze.get(key) != pilot_freeze.get(key):
            raise RuntimeError(f"seeded pilot inherited freeze field drifted: {key}")
    expected_validity = copy.deepcopy(pilot_freeze["validity"])
    expected_validity["t25_status_values"] = [
        *expected_validity["t25_status_values"],
        "terminal_at_t25_before_replan",
    ]
    if seeded_freeze.get("validity") != expected_validity:
        raise RuntimeError("seeded pilot t25-status vocabulary drifted beyond the boundary status")
    expected_go_rule = copy.deepcopy(pilot_freeze["diagnostic_go_rule"])
    expected_go_rule["early_termination_rule"] = (
        "An episode ending before 25 actual env steps has t25_status=terminal_before_t25. "
        "If exactly 25 transitions were committed and terminateds or truncateds is true after step 25, "
        "classify terminal_at_t25_before_replan because World._run_iter stops before the next policy solve. "
        "Both statuses are nonmatched; do not extend, replace, or count them as matched primary episodes."
    )
    if seeded_freeze.get("diagnostic_go_rule") != expected_go_rule:
        raise RuntimeError("seeded pilot t25-boundary rule changed a GO threshold or unrelated GO field")
    if seeded_freeze.get("dataset_evaluator_protocol") != pilot_freeze.get("protocol"):
        raise RuntimeError("seeded pilot dataset-evaluator protocol differs from PILOT_FREEZE")
    protocol = seeded_freeze.get("seeded_variant_protocol", {})
    if (
        "requested_dataset_reset_seed" not in protocol
        or protocol.get("requested_dataset_reset_seed") is not None
        or protocol.get("effective_dataset_reset_seed") != 42
        or protocol.get("dataset_seed_column_present") is not False
        or protocol.get("expected_world_reset_calls_per_run") != 1
        or protocol.get("intercept_only_dataset_world_reset_seed_none") is not True
        or protocol.get("claim_boundary") != "controlled seeded protocol variant; not the official dataset-start baseline"
    ):
        raise RuntimeError("seeded pilot reset intervention drifted")
    if seeded_freeze.get("artifact_directory_suffix") != "artifacts/seeded-pilot/<PBS_JOBID>":
        raise RuntimeError("seeded pilot output directory must be unique per PBS job")


def aggregate_result(episodes: list[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, Any]:
    matched = []
    per_episode: list[dict[str, Any]] = []
    for episode in episodes:
        if episode["t25_status"] != "reached":
            signals = episode.get("termination_signals", [])
            last_signal = signals[-1] if signals else None
            per_episode.append({
                "task": episode["task"],
                "status": episode["t25_status"],
                "actual_env_step_count": episode["actual_env_step_count"],
                "terminal_signal_at_last_active_step": last_signal,
            })
            continue
        by_step = {int(row["sim_steps_before_solve"]): row for row in episode["solve_records"]}
        t0 = by_step[0]
        t25 = by_step[25]
        r0 = {int(row["round"]): row["metrics"] for row in episode["score_rows"] if row["sim_steps_before_solve"] == 0}
        r25 = {int(row["round"]): row["metrics"] for row in episode["score_rows"] if row["sim_steps_before_solve"] == 25}
        deltas = {}
        for round_number in SOLVER_ROUNDS:
            before = r0[round_number]["standardized_elite_regret"]
            after = r25[round_number]["standardized_elite_regret"]
            if before is None or after is None:
                raise RuntimeError(f"invalid standardized regret in matched episode {episode['task']['episode_idx']}")
            deltas[str(round_number)] = float(after - before)
        matched.append(deltas)
        per_episode.append({
            "task": episode["task"],
            "status": "matched_t0_t25",
            "t0_solve_ordinal": t0["solve_ordinal"],
            "t25_solve_ordinal": t25["solve_ordinal"],
            "t0_sim_steps": t0["sim_steps_before_solve"],
            "t25_sim_steps": t25["sim_steps_before_solve"],
            "standardized_regret_delta_by_round": deltas,
            "recall_at_120_by_round": {
                str(round_number): {
                    "t0": r0[round_number]["recall_at_120"],
                    "t25": r25[round_number]["recall_at_120"],
                }
                for round_number in SOLVER_ROUNDS
            },
        })
    primary_deltas = [row["30"] for row in matched]
    matched_count = len(primary_deltas)
    minimum = int(freeze["validity"]["minimum_matched_episodes"])
    go_rule = freeze["diagnostic_go_rule"]
    median_delta = statistics.median(primary_deltas) if primary_deltas else None
    positive_fraction = sum(value > 0 for value in primary_deltas) / matched_count if matched_count else None
    coverage_ok = matched_count >= minimum
    gate_pass = bool(
        coverage_ok
        and median_delta is not None
        and positive_fraction is not None
        and median_delta >= float(go_rule["median_delta_at_least"])
        and positive_fraction >= float(go_rule["positive_fraction_at_least"])
    )
    return {
        "matched_episode_count": matched_count,
        "required_matched_episode_count": minimum,
        "coverage_ok": coverage_ok,
        "primary_round": 30,
        "primary_metric": "paired episode standardized_elite_regret(t25) - standardized_elite_regret(t0)",
        "primary_median_delta": median_delta,
        "primary_positive_fraction": positive_fraction,
        "secondary_round_median_deltas": {
            str(round_number): statistics.median([row[str(round_number)] for row in matched]) if matched else None
            for round_number in (10, 20)
        },
        "episode_rows": per_episode,
        "status": "GO" if gate_pass else "INCONCLUSIVE_COVERAGE" if not coverage_ok else "NO_GO",
        "go_rule": {
            "median_delta_at_least": float(go_rule["median_delta_at_least"]),
            "positive_fraction_at_least": float(go_rule["positive_fraction_at_least"]),
            "observed_median_delta": median_delta,
            "observed_positive_fraction": positive_fraction,
        },
        "interpretation": "Primary delta captures combined student-induced simulator state, action-history, and naturally shifted candidate-distribution change; it is not a pure state-only effect.",
    }


def main() -> int:
    args = parse_args()
    host = require_compute_node()
    out = args.output.resolve(strict=True)
    existing_outputs = (
        "pilot_summary.json",
        "episodes.jsonl",
        "candidate_scores.jsonl",
        "diagnostic_two_pair.json",
        "control_control_repro.json",
        "seeded_control_control_repro.json",
        "seeded_pilot_summary.json",
    )
    if out == Path("/") or any((out / name).exists() for name in existing_outputs):
        raise FileExistsError("output directory is invalid or already contains pilot results")

    freeze = read_json(args.freeze.resolve(strict=True))
    manifest = read_json(args.selection_manifest.resolve(strict=True))
    collection, gate_tasks, student_binding = validate_freeze_and_manifest(freeze, manifest, args)
    diagnostic_tasks = []
    repro_task = None
    seeded_repro_task = None
    seeded_pilot_freeze = None
    if args.seeded_pilot_variant:
        if out.name != os.environ.get("PBS_JOBID") or out.parent.name != "seeded-pilot":
            raise RuntimeError("seeded pilot requires a unique artifacts/seeded-pilot/<PBS_JOBID> output directory")
        seeded_pilot_freeze = read_json(args.seeded_pilot_freeze.resolve(strict=True))
        validate_seeded_pilot_freeze(seeded_pilot_freeze, freeze, manifest, collection, gate_tasks)
    if args.diagnostic_two_pair_only:
        diagnostic_freeze = read_json(args.diagnostic_freeze.resolve(strict=True))
        diagnostic_tasks = validate_diagnostic_freeze(diagnostic_freeze, freeze, manifest, gate_tasks, args)
    if args.control_control_repro_only:
        repro_freeze = read_json(args.repro_freeze.resolve(strict=True))
        repro_task = validate_repro_freeze(repro_freeze, freeze, manifest, collection, args)
    if args.seeded_control_control_repro_only:
        seeded_freeze = read_json(args.seeded_repro_freeze.resolve(strict=True))
        seeded_repro_task = validate_seeded_repro_freeze(seeded_freeze, freeze, manifest, collection, args)
    if args.diagnostic_two_pair_only:
        evaluation_tasks = diagnostic_tasks
    elif args.control_control_repro_only:
        evaluation_tasks = [repro_task]
    elif args.seeded_control_control_repro_only:
        evaluation_tasks = [seeded_repro_task]
    else:
        evaluation_tasks = collection
    checkpoint = Path(str(student_binding["path"])).resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    np, spt, swm, torch, transforms, modules = load_modules(args)
    baseline, adaptive, old_router, load_official_checkpoint, preprocessing, instantiate = modules
    cfg = baseline.compose_pinned_config(args.lewm_root.resolve(strict=True))
    if (
        int(cfg.world.max_episode_steps) != 100
        or int(cfg.eval.eval_budget) != 50
        or int(cfg.eval.goal_offset_steps) != 25
        or int(cfg.world.num_envs) != 50
        or int(cfg.solver.num_samples) != 300
        or int(cfg.solver.topk) != 30
        or int(cfg.solver.n_steps) != 30
        or int(cfg.solver.seed) != 42
    ):
        raise RuntimeError("pinned official dataset-evaluator/CEM config drifted")

    baseline_binding = freeze["upstream_baseline"]
    if args.lewm_root.resolve(strict=True) != Path(str(baseline_binding["lewm_root"])).resolve(strict=True):
        raise RuntimeError("LeWM source root differs from the upstream baseline freeze")
    if args.stablewm_root.resolve(strict=True) != Path(str(baseline_binding["stable_worldmodel_root"])).resolve(strict=True):
        raise RuntimeError("stable-worldmodel source root differs from the upstream baseline freeze")
    dataset_path = Path(str(baseline_binding["dataset_path"])).resolve(strict=True)
    staged_dataset = args.stablewm_home.resolve(strict=True) / "pusht_expert_train.h5"
    if dataset_path != staged_dataset.resolve(strict=True):
        raise RuntimeError("pilot dataset path differs from the staged official LeWM dataset")
    if (args.cache_root / "datasets" / staged_dataset.name).resolve(strict=True) != staged_dataset:
        raise RuntimeError("PBS dataset cache symlink does not resolve to the frozen HDF5")
    teacher_path = args.stablewm_home.resolve(strict=True) / "pusht" / "lewm_object.ckpt"
    if teacher_path.resolve(strict=True) != Path(str(baseline_binding["teacher_checkpoint_path"])).resolve(strict=True):
        raise RuntimeError("official teacher checkpoint path differs from the upstream baseline freeze")
    if dataset_path.stat().st_size != int(baseline_binding["dataset_size_bytes"]):
        raise RuntimeError("staged HDF5 size differs from the upstream teacher-baseline freeze")
    if teacher_path.stat().st_size != int(baseline_binding["teacher_checkpoint_size_bytes"]):
        raise RuntimeError("staged official teacher size differs from the upstream teacher-baseline freeze")
    if (args.cache_root / "pusht" / teacher_path.name).resolve(strict=True) != teacher_path.resolve(strict=True):
        raise RuntimeError("PBS teacher cache symlink does not resolve to the official LeWM checkpoint")
    dataset = swm.data.HDF5Dataset(
        str(cfg.eval.dataset_name), keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=args.cache_root.resolve()
    )
    process = baseline.fit_dataset_process(dataset, cfg, preprocessing, np)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    # h5py fancy indexing requires increasing row indices. This metadata
    # check is independent of the frozen, randomized evaluation order.
    sorted_collection = sorted(evaluation_tasks, key=lambda row: row["row_index"])
    collection_rows = dataset.get_row_data(
        np.asarray([row["row_index"] for row in sorted_collection], dtype=np.int64)
    )
    for index, task in enumerate(sorted_collection):
        if int(collection_rows[episode_column][index]) != task["episode_idx"] or int(collection_rows["step_idx"][index]) != task["start_step"]:
            raise RuntimeError(f"collection row identity mismatch at selection order {task['selection_order']}")

    schedule, _, reference, _ = old_router.load_modules(args.control_root.resolve(strict=True), args.lewm_root.resolve(strict=True))
    schedule.validate_interface(reference, args.interface_probe.resolve(strict=True))
    official = load_official_checkpoint(args.cache_root.resolve(strict=True))
    official.interpolate_pos_encoding = True
    official.eval()
    official.requires_grad_(False)
    student, student_metadata = adaptive.load_main_student(reference, checkpoint)
    for key in ("source", "arm", "extra_updates"):
        if student_metadata["provenance"].get(key) != student_binding["provenance"].get(key):
            raise RuntimeError(f"loaded student checkpoint provenance differs from freeze: {key}")
    if int(student_metadata["provenance"].get("extra_updates", -1)) != 1000:
        raise RuntimeError("loaded student checkpoint is not treatment step1000")
    if int(sum(parameter.numel() for parameter in student.parameters())) != int(student_binding.get("parameter_count", -1)):
        raise RuntimeError("loaded student parameter count differs from freeze")
    student = student.to("cuda").eval()
    student.requires_grad_(False)

    if args.control_control_repro_only:
        if repro_task is None:
            raise RuntimeError("validated control-control task is missing")
        control = run_episode(
            repro_task, False, False, args, baseline, swm, cfg, dataset, process, transforms,
            torch, spt, official, reference, student, old_router, instantiate, np,
            capture_repro_inputs=True,
        )
        repeat = run_episode(
            repro_task, False, False, args, baseline, swm, cfg, dataset, process, transforms,
            torch, spt, official, reference, student, old_router, instantiate, np,
            capture_repro_inputs=True,
        )
        initial_inputs = compare_repro_inputs(control, repeat, np)
        first_round = compare_first_round_arrays(control, repeat, np)
        first_execution = compare_first_execution(control, repeat, np)
        simulator = compare_simulator_signatures(control, repeat)
        left_solves = control.get("_solves", [])
        right_solves = repeat.get("_solves", [])
        rng = {
            "available": bool(left_solves and right_solves),
            "before_exact": bool(left_solves and right_solves) and state_bytes_equal(
                left_solves[0]["_rng_before"], right_solves[0]["_rng_before"]
            ),
            "after_exact": bool(left_solves and right_solves) and state_bytes_equal(
                left_solves[0]["_rng_after"], right_solves[0]["_rng_after"]
            ),
        }
        seed_column_equal = control["_dataset_seed_column_present"] == repeat["_dataset_seed_column_present"]
        reset_seed_equal = exact_equal(control["_reset_seed_observed"], repeat["_reset_seed_observed"])
        seed_metadata = {
            "dataset_seed_column_present_equal": seed_column_equal,
            "dataset_seed_column_present": control["_dataset_seed_column_present"],
            "control_reset_seed_observed": control["_reset_seed_observed"],
            "repeat_reset_seed_observed": repeat["_reset_seed_observed"],
            "reset_seed_calls_exact": reset_seed_equal,
            "reset_seed_source": "dataset init_state['seed'] if present; otherwise World dataset reset receives None",
            "control_reset_call_count": len(control["_reset_seed_observed"]),
            "repeat_reset_call_count": len(repeat["_reset_seed_observed"]),
        }
        repeatability_pass = bool(
            initial_inputs["exact"]
            and first_round["exact"]
            and first_execution["available"]
            and first_execution["committed_first_action_exact"]
            and first_execution["first_post_step_state_exact"]
            and simulator["available"]
            and simulator["exact"]
            and rng["available"]
            and rng["before_exact"]
            and rng["after_exact"]
            and seed_column_equal
            and reset_seed_equal
            and len(control["_reset_seed_observed"]) == 1
            and len(repeat["_reset_seed_observed"]) == 1
        )
        atomic_json(out / "control_control_repro.json", {
            "schema": REPRO_RESULT_SCHEMA,
            "schema_version": 1,
            "status": "COMPLETED_EXACT_REPRODUCIBILITY" if repeatability_pass else "COMPLETED_DIVERGENCE_OR_MISSING_API",
            "pbs_job_id": os.environ["PBS_JOBID"],
            "compute_host": host,
            "pilot_freeze": str(args.freeze.resolve()),
            "repro_freeze": str(args.repro_freeze.resolve()),
            "selection_manifest_job_id": manifest["pbs_job_id"],
            "episode": dict(repro_task),
            "repetitions": 2,
            "shadow_enabled": False,
            "other_collection_tasks_evaluated": False,
            "reserved_holdout_evaluated_or_scored": False,
            "repeatability_pass": repeatability_pass,
            "dataset_reset_seed": seed_metadata,
            "initial_prepared_world_info": initial_inputs,
            "simulator_signature": simulator,
            "solver_rng": rng,
            "first_cem_round_arrays": first_round,
            "first_committed_action_and_post_step_state": first_execution,
            "control_success": control["success"],
            "repeat_success": repeat["success"],
            "control_active_env_steps": control["actual_env_step_count"],
            "repeat_active_env_steps": repeat["actual_env_step_count"],
            "raw_images_or_round_arrays_written": False,
        })
        print(json.dumps({
            "status": "COMPLETED_EXACT_REPRODUCIBILITY" if repeatability_pass else "COMPLETED_DIVERGENCE_OR_MISSING_API",
            "episode_idx": repro_task["episode_idx"],
            "prepared_inputs_exact": initial_inputs["exact"],
            "first_round_exact": first_round["exact"],
            "first_action_exact": first_execution["committed_first_action_exact"],
            "first_post_step_state_exact": first_execution["first_post_step_state_exact"],
            "simulator_signature_available": simulator["available"],
        }), flush=True)
        return 0

    if args.seeded_control_control_repro_only:
        if seeded_repro_task is None:
            raise RuntimeError("validated seeded control-control task is missing")
        control = run_episode(
            seeded_repro_task, False, False, args, baseline, swm, cfg, dataset, process, transforms,
            torch, spt, official, reference, student, old_router, instantiate, np,
            capture_repro_inputs=True, seeded_reset_override=42,
        )
        repeat = run_episode(
            seeded_repro_task, False, False, args, baseline, swm, cfg, dataset, process, transforms,
            torch, spt, official, reference, student, old_router, instantiate, np,
            capture_repro_inputs=True, seeded_reset_override=42,
        )
        initial_inputs = compare_repro_inputs(control, repeat, np)
        first_round = compare_first_round_arrays(control, repeat, np)
        first_execution = compare_first_execution(control, repeat, np)
        simulator = compare_simulator_signatures(control, repeat)
        left_solves = control.get("_solves", [])
        right_solves = repeat.get("_solves", [])
        rng = {
            "available": bool(left_solves and right_solves),
            "before_exact": bool(left_solves and right_solves) and state_bytes_equal(
                left_solves[0]["_rng_before"], right_solves[0]["_rng_before"]
            ),
            "after_exact": bool(left_solves and right_solves) and state_bytes_equal(
                left_solves[0]["_rng_after"], right_solves[0]["_rng_after"]
            ),
        }
        requested = control["_reset_seed_observed"]
        repeat_requested = repeat["_reset_seed_observed"]
        effective = control["_reset_seed_effective"]
        repeat_effective = repeat["_reset_seed_effective"]
        seed_column_absent = (
            control["_dataset_seed_column_present"] is False
            and repeat["_dataset_seed_column_present"] is False
        )
        reset_protocol_valid = (
            seed_column_absent
            and len(requested) == len(repeat_requested) == len(effective) == len(repeat_effective) == 1
            and requested == repeat_requested == [None]
            and effective == repeat_effective == [42]
        )
        seeded_reset = {
            "dataset_seed_column_present_control": control["_dataset_seed_column_present"],
            "dataset_seed_column_present_repeat": repeat["_dataset_seed_column_present"],
            "control_requested_reset_seed": requested,
            "repeat_requested_reset_seed": repeat_requested,
            "control_effective_reset_seed": effective,
            "repeat_effective_reset_seed": repeat_effective,
            "exactly_one_dataset_world_reset_per_run": len(requested) == len(repeat_requested) == 1,
            "requested_none_and_effective_42_both_runs": reset_protocol_valid,
            "protocol_variant": "dataset World.reset(seed=None) intercepted and forwarded as seed=42 before dataset _set_state",
        }
        repeatability_pass = bool(
            reset_protocol_valid
            and initial_inputs["exact"]
            and first_round["exact"]
            and first_execution["available"]
            and first_execution["committed_first_action_exact"]
            and first_execution["first_post_step_state_exact"]
            and simulator["available"]
            and simulator["exact"]
            and rng["available"]
            and rng["before_exact"]
            and rng["after_exact"]
        )
        atomic_json(out / "seeded_control_control_repro.json", {
            "schema": SEEDED_REPRO_RESULT_SCHEMA,
            "schema_version": 1,
            "status": "COMPLETED_EXACT_SEEDED_REPRODUCIBILITY" if repeatability_pass else "COMPLETED_DIVERGENCE_OR_MISSING_API",
            "pbs_job_id": os.environ["PBS_JOBID"],
            "compute_host": host,
            "pilot_freeze": str(args.freeze.resolve()),
            "seeded_repro_freeze": str(args.seeded_repro_freeze.resolve()),
            "selection_manifest_job_id": manifest["pbs_job_id"],
            "episode": dict(seeded_repro_task),
            "repetitions": 2,
            "shadow_enabled": False,
            "protocol_claim_boundary": "controlled seeded protocol variant; not the official dataset-start baseline",
            "other_collection_tasks_evaluated": False,
            "reserved_holdout_evaluated_or_scored": False,
            "repeatability_pass": repeatability_pass,
            "dataset_reset_seed": seeded_reset,
            "initial_prepared_world_info": initial_inputs,
            "simulator_signature": simulator,
            "solver_rng": rng,
            "first_cem_round_arrays": first_round,
            "first_committed_action_and_post_step_state": first_execution,
            "control_success": control["success"],
            "repeat_success": repeat["success"],
            "control_active_env_steps": control["actual_env_step_count"],
            "repeat_active_env_steps": repeat["actual_env_step_count"],
            "raw_images_or_round_arrays_written": False,
        })
        print(json.dumps({
            "status": "COMPLETED_EXACT_SEEDED_REPRODUCIBILITY" if repeatability_pass else "COMPLETED_DIVERGENCE_OR_MISSING_API",
            "episode_idx": seeded_repro_task["episode_idx"],
            "requested_reset_seed": requested,
            "effective_reset_seed": effective,
            "prepared_inputs_exact": initial_inputs["exact"],
            "first_round_exact": first_round["exact"],
            "first_action_exact": first_execution["committed_first_action_exact"],
            "first_post_step_state_exact": first_execution["first_post_step_state_exact"],
            "simulator_signature_available": simulator["available"],
        }), flush=True)
        return 0

    if args.diagnostic_two_pair_only:
        diagnostic_pairs = []
        for task in diagnostic_tasks:
            control = run_episode(
                task, False, True, args, baseline, swm, cfg, dataset, process, transforms,
                torch, spt, official, reference, student, old_router, instantiate, np,
            )
            shadow = run_episode(
                task, True, True, args, baseline, swm, cfg, dataset, process, transforms,
                torch, spt, official, reference, student, old_router, instantiate, np,
            )
            comparison = compare_paired_runs(control, shadow, torch)
            trace_comparison = compare_execution_traces(control, shadow, np)
            diagnostic_pairs.append({
                "task": dict(task),
                "original_gate_pass": comparison["pass"],
                "gate_checks": comparison["checks"],
                "per_solve_checks": comparison["per_solve"],
                "execution_trace": trace_comparison,
                "control_success": control["success"],
                "shadow_success": shadow["success"],
                "control_active_env_steps": control["actual_env_step_count"],
                "shadow_active_env_steps": shadow["actual_env_step_count"],
            })
            print(json.dumps({
                "diagnostic_episode": task["episode_idx"],
                "original_gate_pass": comparison["pass"],
                "actions_exact": trace_comparison["committed_actions"]["exact"],
                "masks_exact": trace_comparison["commit_masks"]["exact"],
                "post_step_states_exact": trace_comparison["post_step_states"]["exact"],
            }), flush=True)
        all_match = all(row["original_gate_pass"] for row in diagnostic_pairs)
        atomic_json(out / "diagnostic_two_pair.json", {
            "schema": DIAGNOSTIC_RESULT_SCHEMA,
            "schema_version": 1,
            "status": "COMPLETED_MATCH" if all_match else "COMPLETED_MISMATCH",
            "pbs_job_id": os.environ["PBS_JOBID"],
            "compute_host": host,
            "pilot_freeze": str(args.freeze.resolve()),
            "diagnostic_freeze": str(args.diagnostic_freeze.resolve()),
            "selection_manifest_job_id": manifest["pbs_job_id"],
            "episode_ids_evaluated": [row["episode_idx"] for row in diagnostic_tasks],
            "other_collection_tasks_evaluated": False,
            "reserved_holdout_evaluated_or_scored": False,
            "raw_action_or_state_traces_written": False,
            "original_exact_gate_retained": True,
            "pairs": diagnostic_pairs,
        })
        return 0

    gate_results = []
    reusable_shadow_runs: dict[int, dict[str, Any]] = {}
    gate_pass = True
    for task in gate_tasks:
        control = run_episode(
            task, False, True, args, baseline, swm, cfg, dataset, process, transforms,
            torch, spt, official, reference, student, old_router, instantiate, np,
            seeded_reset_override=42 if args.seeded_pilot_variant else None,
        )
        shadow = run_episode(
            task, True, True, args, baseline, swm, cfg, dataset, process, transforms,
            torch, spt, official, reference, student, old_router, instantiate, np,
            seeded_reset_override=42 if args.seeded_pilot_variant else None,
        )
        comparison = compare_paired_runs(control, shadow, torch)
        gate_pass = gate_pass and comparison["pass"]
        pair_result = {"task": dict(task), "comparison": comparison}
        if args.seeded_pilot_variant:
            pair_result["control_dataset_reset_seed"] = control["dataset_reset_seed"]
            pair_result["shadow_dataset_reset_seed"] = shadow["dataset_reset_seed"]
        gate_results.append(pair_result)
        reusable_shadow_runs[int(task["selection_order"])] = shadow
        print(json.dumps({"paired_gate_episode": task["episode_idx"], "pass": comparison["pass"]}), flush=True)

    if not gate_pass:
        atomic_json(out / "paired_gate.json", {"status": "FAIL_CLOSED", "pairs": gate_results})
        failed_summary = {
            "schema": SEEDED_PILOT_RESULT_SCHEMA if args.seeded_pilot_variant else SCHEMA,
            "schema_version": 1,
            "status": "FAIL_CLOSED_PAIRED_NONINTERFERENCE",
            "pbs_job_id": os.environ["PBS_JOBID"],
            "compute_host": host,
            "paired_gate": {"passed": False, "required_pairs": 4, "pairs": gate_results},
            "claim_boundary": "No collection metric is valid because shadow-on changed a student-only action, CEM state, RNG, or simulator trace.",
        }
        if args.seeded_pilot_variant:
            failed_summary["seeded_pilot_freeze"] = str(args.seeded_pilot_freeze.resolve())
            failed_summary["protocol_variant"] = "controlled seeded protocol variant; not the official dataset-start baseline"
        summary_name = "seeded_pilot_summary.json" if args.seeded_pilot_variant else "pilot_summary.json"
        atomic_json(out / summary_name, failed_summary)
        return 2

    episodes: list[dict[str, Any]] = []
    for task in collection:
        order = int(task["selection_order"])
        if order in reusable_shadow_runs:
            episode = reusable_shadow_runs[order]
        else:
            episode = run_episode(
                task, True, False, args, baseline, swm, cfg, dataset, process, transforms,
                torch, spt, official, reference, student, old_router, instantiate, np,
                seeded_reset_override=42 if args.seeded_pilot_variant else None,
            )
        episodes.append(episode)
        print(json.dumps({
            "collection_episode": task["episode_idx"],
            "t25_status": episode["t25_status"],
            "actual_env_step_count": episode["actual_env_step_count"],
            "success": episode["success"],
        }), flush=True)

    aggregate = aggregate_result(episodes, freeze)
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            public = {key: value for key, value in episode.items() if not key.startswith("_") and key != "score_rows"}
            handle.write(json.dumps(jsonable(public), ensure_ascii=False, allow_nan=False) + "\n")
    with (out / "candidate_scores.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            for row in episode["score_rows"]:
                handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
    atomic_json(out / "paired_gate.json", {"status": "PASS", "required_pairs": 4, "pairs": gate_results})
    summary = {
        "schema": SEEDED_PILOT_RESULT_SCHEMA if args.seeded_pilot_variant else SCHEMA,
        "schema_version": 1,
        "status": aggregate["status"],
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "freeze": str(args.freeze.resolve()),
        "selection_manifest": {
            "path": str(args.selection_manifest.resolve()),
            "pbs_job_id": manifest["pbs_job_id"],
            "schema": manifest["schema"],
            "collection_tasks_evaluated": len(collection),
            "reserved_holdout_evaluated": False,
            "reserved_holdout_scored": False,
            "reserved_holdout_used_for_decision": False,
            "official_dataset_wide_preprocessing_reads_action_proprio_state": True,
        },
        "source": {
            "dataset_path": str(dataset_path),
            "dataset_size_bytes": dataset_path.stat().st_size,
            "teacher_checkpoint_path": str(teacher_path.resolve()),
            "teacher_checkpoint_size_bytes": teacher_path.stat().st_size,
            "student_checkpoint": student_metadata,
            "lewm_root": str(args.lewm_root.resolve()),
            "stableworldmodel_root": str(args.stablewm_root.resolve()),
            "interface_probe": str(args.interface_probe.resolve()),
        },
        "protocol": {
            "world_evaluate": "official dataset-backed World.evaluate, one frozen collection episode per fresh World",
            "goal_offset_steps": 25,
            "eval_budget": 50,
            "max_episode_steps": 100,
            "native_solver_seed_per_episode": 42,
            "t0_to_t25_rng_reseed": False,
            "collection_order": [row["episode_idx"] for row in collection],
        },
        "paired_gate": {"passed": True, "required_pairs": 4, "pairs": gate_results},
        "episode_count": len(episodes),
        "successes": sum(bool(row["success"]) for row in episodes),
        "t25_reached_count": sum(row["t25_status"] == "reached" for row in episodes),
        "terminal_before_t25_count": sum(row["t25_status"] == "terminal_before_t25" for row in episodes),
        "terminal_at_t25_before_replan_count": sum(row["t25_status"] == "terminal_at_t25_before_replan" for row in episodes),
        "actual_env_step_counts": [int(row["actual_env_step_count"]) for row in episodes],
        "aggregate": aggregate,
        "output_files": {
            "episodes": "episodes.jsonl",
            "candidate_scores": "candidate_scores.jsonl",
            "paired_gate": "paired_gate.json",
        },
        "gpu_telemetry": {"source": "job.log direct nvidia-smi samples every 5 seconds", "required": True},
        "claim_boundary": "Only the 16 collection tasks are evaluated, shadow-scored, or used for this decision; reserved holdout task rows are not passed to World.evaluate or scored. Official dataset-wide preprocessing reads action/proprio/state columns, including holdout rows. The diagnostic measures combined student-induced simulator state, action-history, and candidate-distribution drift; no training is part of this runner.",
    }
    if args.seeded_pilot_variant:
        summary["seeded_pilot_freeze"] = str(args.seeded_pilot_freeze.resolve())
        summary["protocol_variant"] = "controlled seeded protocol variant; not the official dataset-start baseline"
        summary["protocol"]["dataset_reset_seed_override"] = {
            "dataset_seed_column_present": False,
            "requested_seed": None,
            "effective_seed": 42,
            "world_reset_calls_per_episode": 1,
        }
        summary["claim_boundary"] = "Controlled seeded protocol variant, not the official dataset-start baseline. Each collection evaluation intercepts the sole dataset World.reset(seed=None) and supplies seed 42 before applying the selected dataset init state. Task identities, CEM/shadow rounds, metrics, validity criteria, paired-gate rules, and GO thresholds are inherited unchanged from PILOT_FREEZE. Reserved holdout tasks are never passed to World.evaluate or scored; official dataset-wide preprocessing reads action/proprio/state columns, including holdout rows. No training is part of this runner."
    summary_name = "seeded_pilot_summary.json" if args.seeded_pilot_variant else "pilot_summary.json"
    atomic_json(out / summary_name, summary)
    print(json.dumps({"status": aggregate["status"], "matched": aggregate["matched_episode_count"], "median_delta": aggregate["primary_median_delta"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            args = parse_args()
            if args.output.exists() and args.output.is_dir():
                atomic_json(args.output / "failure.json", {"status": "FAIL_CLOSED", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

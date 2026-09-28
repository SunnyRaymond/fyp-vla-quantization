#!/usr/bin/env python3
"""Run the frozen official LeWM CEM Stage 1 gate or Stage 2 PushT episodes.

All model, simulator, and planner work is restricted to a validated PBS
compute allocation.  The pinned CEMSolver.solve and World.evaluate methods
remain the execution path; wrappers only route get_cost, collect trace data,
time solver calls, and reseed CEMSolver.torch_gen before paired calls.
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
import time
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
SCHEDULE_DIR = TRANSFER / "adaptive-teacher-schedule"
STUDENT_DIR = TRANSFER / "cem-distribution-distill"
SCHEMA = "lewm-recurrent-student.official-pusht-cem-runner"
FIELD_NAMES = (
    "candidates", "costs", "topk_vals", "topk_inds", "topk_candidates",
    "prev_mean", "prev_var", "mean", "var",
)
TEACHER_ROUNDS = {
    "student_only": frozenset(),
    "uniform_teacher7": frozenset((4, 8, 12, 16, 20, 24, 28)),
    "late_teacher7": frozenset((24, 25, 26, 27, 28, 29, 30)),
    "teacher_only": frozenset(range(1, 31)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--stablewm-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-summary", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def assert_fresh_result_slots(output: Path, stage: str) -> None:
    names = ("stage1_summary.json", "stage1_call_records.json") if stage == "stage1" else ("stage2_episodes.jsonl", "stage2_summary.json")
    existing = [name for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prior {stage} results: {', '.join(existing)}")


def plain(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def require_compute_node() -> str:
    if not os.environ.get("PBS_JOBID", "").strip():
        raise RuntimeError("PBS_JOBID is required")
    host = platform.node().lower()
    short_host = host.split(".", 1)[0]
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit node: {host}")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_NODEFILE must exist for allocation validation")
    nodes = {line.strip().lower() for line in Path(nodefile).read_text(encoding="utf-8").splitlines() if line.strip()}
    short_nodes = {node.split(".", 1)[0] for node in nodes}
    if not nodes or (host not in nodes and short_host not in short_nodes):
        raise RuntimeError(f"current host {host} is not a member of PBS_NODEFILE")
    return host


def validate_freeze(freeze: Mapping[str, Any]) -> None:
    if freeze.get("schema") != "lewm-recurrent-student.official-pusht-cem-freeze":
        raise ValueError("freeze schema mismatch")
    if freeze.get("status") != "frozen_before_results":
        raise ValueError("freeze must remain frozen_before_results")
    solver = freeze["official_planner"]["cem_solver"]
    expected = {"batch_size": 1, "num_samples": 300, "var_scale": 1.0, "n_steps": 30, "topk": 30, "device": "cuda"}
    if solver != expected:
        raise ValueError("official CEM solver settings drifted")
    if freeze["official_planner"]["late_teacher7_rounds_1_indexed"] != [24, 25, 26, 27, 28, 29, 30]:
        raise ValueError("late_teacher7 schedule drifted")
    if freeze["official_planner"]["uniform_teacher7_rounds_1_indexed"] != [4, 8, 12, 16, 20, 24, 28]:
        raise ValueError("uniform_teacher7 schedule drifted")
    if freeze["stage1"]["quality_estimand"].find("final_late_teacher7_plan) - official_teacher_cost(final_student_plan)") < 0:
        raise ValueError("quality estimand must be late minus student")


def load_models(args: argparse.Namespace, freeze: Mapping[str, Any]):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    expected_checkpoint = str(freeze["model"]["student_checkpoint_path"]).replace("\\", "/")
    if str(args.checkpoint).replace("\\", "/") != expected_checkpoint:
        raise ValueError("student checkpoint path does not match the frozen artifact")
    schedule, old, reference, _ = load_modules(args.control_root, args.lewm_root)
    import stable_worldmodel as swm
    import jepa
    stable_file = Path(swm.__file__).resolve()
    lewm_file = Path(jepa.__file__).resolve()
    if args.stablewm_root.resolve() not in stable_file.parents:
        raise RuntimeError(f"stable_worldmodel imported outside the staged pinned source: {stable_file}")
    if args.lewm_root.resolve() not in lewm_file.parents:
        raise RuntimeError(f"LeWM imported outside the staged pinned source: {lewm_file}")
    schedule.validate_interface(reference, args.probe.resolve())
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.interpolate_pos_encoding = True
    official.requires_grad_(False)
    student, checkpoint_meta = schedule.load_main_student(reference, args.checkpoint.resolve())
    return schedule, old, reference, official, student, checkpoint_meta


def load_modules(control_root: Path, lewm_root: Path):
    for path in (str(TRANSFER), str(SCHEDULE_DIR), str(STUDENT_DIR), str(control_root / "lewm-transfer")):
        if path not in sys.path:
            sys.path.insert(0, path)
    schedule = importlib.import_module("run_adaptive_teacher_schedule")
    old = schedule.load_old_runner()
    reference = schedule.load_reference(old)
    if str(lewm_root) not in sys.path:
        sys.path.insert(0, str(lewm_root))
    import torch
    return schedule, old, reference, torch


def policy_transform():
    import torch
    from torchvision.transforms import v2 as transforms

    def image_pipeline():
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(size=224),
        ])
    return {"pixels": image_pipeline(), "goal": image_pipeline()}


def as_cpu_tensor(value: Any):
    import torch
    return value.detach().to("cpu").clone() if torch.is_tensor(value) else value


def first_sample_cost_info(info_dict: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    return {
        key: value[:, :1] if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and value.ndim >= 2 else value
        for key, value in info_dict.items()
    }


def capture_first_cost_input(owner: Any, info_dict: Mapping[str, Any], action_candidates: Any) -> None:
    import numpy as np
    import torch
    if owner.first_cost_info is not None:
        return
    owner.first_cost_input_shapes = {
        key: list(value.shape) for key, value in info_dict.items()
        if torch.is_tensor(value) or isinstance(value, np.ndarray)
    }
    owner.first_cost_action_shape = list(action_candidates.shape)
    owner.first_cost_info = first_sample_cost_info(info_dict)
    owner.single_sample_cost_info_shapes = {
        key: list(value.shape) for key, value in owner.first_cost_info.items()
        if torch.is_tensor(value) or isinstance(value, np.ndarray)
    }


def reset_cost_capture(owner: Any) -> None:
    owner.first_cost_info = None
    owner.first_cost_input_shapes = None
    owner.first_cost_action_shape = None
    owner.single_sample_cost_info_shapes = None


def is_finite_numeric(value: Any) -> bool:
    import torch
    try:
        return bool(torch.isfinite(torch.as_tensor(value)).all().item())
    except (TypeError, ValueError, RuntimeError):
        return False


class TraceCallback:
    output_key = "official_cem_trace"

    def __init__(self, retain_tensors: bool, record_population_std: bool = False) -> None:
        self.retain_tensors = retain_tensors
        self.record_population_std = record_population_std
        self.round1_population_std = None
        self.history: list[dict[str, Any]] = []
        self.cem_input_shapes: dict[str, list[int]] | None = None

    def reset(self) -> None:
        self.history.clear()
        self.round1_population_std = None
        self.cem_input_shapes = None

    def start_batch(self) -> None:
        return None

    def __call__(self, **kwargs: Any) -> None:
        # Keep tensors only for the native-vs-routed fidelity pair. Other
        # paths record round numbers and lightweight GPU finite flags.
        import torch
        row = {"step": int(kwargs["step"]) + 1}
        if self.cem_input_shapes is None:
            self.cem_input_shapes = {
                "candidates": list(kwargs["candidates"].shape),
                "costs": list(kwargs["costs"].shape),
            }
        if self.retain_tensors:
            row["tensors"] = {key: kwargs[key].detach() for key in FIELD_NAMES}
            finite_keys = [key for key in FIELD_NAMES if kwargs[key].is_floating_point()]
        else:
            finite_keys = ["costs", "mean", "var"]
        row["finite_flags"] = torch.stack([torch.isfinite(kwargs[key]).all() for key in finite_keys]).all()
        if self.record_population_std and int(kwargs["step"]) == 0:
            self.round1_population_std = kwargs["costs"].std(unbiased=False).detach()
        self.history.append(row)

    def end_solve(self) -> None:
        return None

    def public_history(self) -> list[dict[str, Any]]:
        import torch
        flags = torch.stack([row["finite_flags"] for row in self.history]).detach().cpu().tolist()
        return [{"step": row["step"], "finite": bool(flags[i])} for i, row in enumerate(self.history)]

    def finite_trace(self) -> bool:
        import torch
        return all(bool(row["finite_flags"].item()) for row in self.history)


class RoutedCostModel:
    """Transparent LeWM cost router; all planning updates stay in CEMSolver."""

    def __init__(self, official: Any, reference: Any, student: Any, arm: str) -> None:
        self.official = official
        self.reference = reference
        self.student = student
        self.arm = arm
        self.round_index = 0
        self.teacher_rounds: list[int] = []
        self.solve_teacher_rounds: list[int] = []
        self.solve_round_choices: list[bool] = []
        self.active_solver_seed = None
        self.active_call_index = 0
        self.policy_reseeded = False
        reset_cost_capture(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.official, name)

    def parameters(self):
        return self.official.parameters()

    def begin_solve(self) -> None:
        self.round_index = 0
        self.solve_teacher_rounds = []
        self.solve_round_choices = []
        reset_cost_capture(self)

    def end_solve(self) -> None:
        self.teacher_rounds.extend(self.solve_teacher_rounds)

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Any):
        import torch

        self.round_index += 1
        capture_first_cost_input(self, info_dict, action_candidates)
        chosen = self.arm == "teacher_only" or self.round_index in TEACHER_ROUNDS[self.arm]
        self.solve_round_choices.append(chosen)
        if chosen:
            self.solve_teacher_rounds.append(self.round_index)
            return self.official.get_cost(info_dict, action_candidates)
        return self.student_get_cost(info_dict, action_candidates)

    def student_get_cost(self, info: dict[str, Any], candidates: Any):
        import torch

        # Match the official JEPA.get_cost/rollout input path, then replace
        # only future latent prediction with the frozen recurrent student.
        batch, samples, horizon = candidates.shape[:3]
        for key, value in list(info.items()):
            if torch.is_tensor(value):
                info[key] = value.to(device=next(self.official.parameters()).device)
        goal = {key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)}
        if "goal" not in goal:
            raise KeyError("official LeWM info lacks goal pixels")
        goal["pixels"] = goal["goal"]
        for key in list(goal):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action")
        goal_info = self.official.encode(goal)
        info["goal_emb"] = goal_info["emb"]

        history_len = int(info["pixels"].shape[2])
        act0, act_future = torch.split(candidates, [history_len, horizon - history_len], dim=2)
        info["action"] = act0
        initial_info = {key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)}
        initial_info = self.official.encode(initial_info)
        initial = initial_info["emb"]
        info["emb"] = initial.unsqueeze(1).expand(batch, samples, *initial.shape[1:])

        latent = initial[:, -1:, :].expand(batch, samples, -1, -1).reshape(batch * samples, 1, -1)
        actions = candidates.reshape(batch * samples, horizon, candidates.shape[-1])
        with torch.inference_mode():
            predictions = self.student(latent, actions)
        if tuple(predictions.shape) != (batch * samples, horizon, 192):
            raise RuntimeError(f"student prediction shape drifted: {tuple(predictions.shape)}")
        predicted_emb = torch.cat([latent, predictions], dim=1).reshape(batch, samples, horizon + 1, 192)
        info["predicted_emb"] = predicted_emb
        return self.official.criterion(info)


class NativeTeacherCostCapture:
    """Capture actual CEM cost input shapes and delegate unchanged to JEPA."""

    def __init__(self, official: Any) -> None:
        self.official = official
        self.round_index = 0
        self.teacher_rounds: list[int] = []
        self.solve_teacher_rounds: list[int] = []
        self.solve_round_choices: list[bool] = []
        self.active_solver_seed = None
        self.active_call_index = 0
        self.policy_reseeded = False
        reset_cost_capture(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.official, name)

    def parameters(self):
        return self.official.parameters()

    def begin_solve(self) -> None:
        self.round_index = 0
        self.solve_teacher_rounds = []
        self.solve_round_choices = []
        reset_cost_capture(self)

    def end_solve(self) -> None:
        self.teacher_rounds.extend(self.solve_teacher_rounds)

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Any):
        self.round_index += 1
        self.solve_teacher_rounds.append(self.round_index)
        self.solve_round_choices.append(True)
        capture_first_cost_input(self, info_dict, action_candidates)
        return self.official.get_cost(info_dict, action_candidates)


class InstrumentedPolicy:
    """Mixin factory for exact WorldModelPolicy lifecycle instrumentation."""

    @staticmethod
    def build(world_policy_cls: Any, solver: Any, cfg: Any, transform: Mapping[str, Any], route: Any, call_seed: Any = None, capture_prepared: bool = True):
        class Policy(world_policy_cls):
            def __init__(self):
                super().__init__(solver=solver, config=cfg, process={}, transform=dict(transform))
                self.initial_raw = None
                self.initial_prepared = None
                self.call_seed_fn = call_seed
                self.planner_call_index = 0
                self.capture_prepared = capture_prepared

            def _prepare_info(self, info_dict):
                value = super()._prepare_info(info_dict)
                if self.capture_prepared and self.initial_prepared is None:
                    self.initial_prepared = {
                        key: as_cpu_tensor(item)
                        for key, item in value.items()
                        if key in ("pixels", "goal", "state", "goal_state", "proprio", "action") and hasattr(item, "detach")
                    }
                return value

            def get_action(self, info_dict, **kwargs):
                if self.initial_raw is None:
                    self.initial_raw = {
                        key: plain(info_dict[key])
                        for key in ("state", "goal_state")
                        if key in info_dict
                    }
                if self.call_seed_fn is not None and self._action_buffer is not None:
                    terminated = info_dict.get("terminated")
                    if hasattr(terminated, "reshape"):
                        term_values = [bool(x) for x in terminated.reshape(-1)]
                    elif terminated is None:
                        term_values = [False] * len(self._action_buffer)
                    else:
                        term_values = [bool(x) for x in terminated]
                    needs_plan = any(not term_values[i] and len(buf) == 0 for i, buf in enumerate(self._action_buffer))
                    if needs_plan:
                        seed = int(self.call_seed_fn(self.planner_call_index))
                        solver.torch_gen.manual_seed(seed)
                        if route is not None:
                            route.active_solver_seed = seed
                            route.active_call_index = self.planner_call_index
                            route.policy_reseeded = True
                        self.planner_call_index += 1
                return super().get_action(info_dict, **kwargs)

        return Policy()


def make_solver_and_policy(world: Any, official: Any, reference: Any, student: Any, arm: str,
                           callbacks: list[Any], solver_seed: int, call_seed_fn: Any = None,
                           instrumented_solve_records: list[dict[str, Any]] | None = None,
                           capture_prepared: bool = True):
    import torch
    from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
    try:
        from stable_worldmodel.solver import CEMSolver
    except ImportError:
        from stable_worldmodel.planning import CEMSolver

    routed = NativeTeacherCostCapture(official) if arm == "native_teacher_reference" else RoutedCostModel(official, reference, student, arm)
    cost_model = routed
    settings = {"batch_size": 1, "num_samples": 300, "var_scale": 1.0, "n_steps": 30, "topk": 30, "device": "cuda"}
    solver = CEMSolver(model=cost_model, callbacks=callbacks, seed=int(solver_seed), **settings)
    config = PlanConfig(horizon=5, receding_horizon=5, history_len=1, action_block=5, warm_start=True)
    policy = InstrumentedPolicy.build(WorldModelPolicy, solver, config, policy_transform(), routed, call_seed_fn, capture_prepared)

    if instrumented_solve_records is not None:
        original_solve = solver.solve

        def timed_solve(info_dict, init_action=None):
            routed.begin_solve()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = original_solve(info_dict, init_action=init_action)
            torch.cuda.synchronize()
            solver.last_output = output
            elapsed = time.perf_counter() - start
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            output_finite = is_finite_numeric(output["actions"])
            routed.end_solve()
            instrumented_solve_records.append({
                "wall_s": float(elapsed),
                "peak_cuda_allocated_bytes": int(peak_alloc),
                "peak_cuda_reserved_bytes": int(peak_reserved),
                "solver_seed": int(solver.torch_gen.initial_seed()),
                "planner_call_index": int(getattr(routed, "active_call_index", 0)),
                "policy_reseeded": bool(getattr(routed, "policy_reseeded", False)),
                "teacher_rounds": list(routed.solve_teacher_rounds),
                "round_teacher_flags": list(routed.solve_round_choices),
                "cem_input_shapes": callbacks[0].cem_input_shapes,
                "cost_info_shapes": routed.first_cost_input_shapes,
                "cost_action_candidates_shape": routed.first_cost_action_shape,
                "finite_trace": callbacks[0].finite_trace(),
                "output_finite": output_finite,
            })
            return output

        solver.solve = timed_solve
    world.set_policy(policy)
    return solver, policy, routed


def make_world(stablewm_home: Path):
    os.environ["STABLEWM_HOME"] = str(stablewm_home.resolve())
    import stable_worldmodel as swm
    return swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=50)


def seed_reset_action_space(world: Any, seed: int) -> int:
    envs = getattr(world.envs, "envs", None)
    if not isinstance(envs, list) or len(envs) != 1:
        raise RuntimeError("paired PushT reset requires one concrete EnvPool env")
    action_space = getattr(envs[0], "action_space", None)
    seed_method = getattr(action_space, "seed", None)
    if not callable(seed_method):
        raise RuntimeError("cannot seed the concrete PushT reset action space")
    seed_method(int(seed))
    return int(seed)


def current_env_state(world: Any) -> dict[str, Any]:
    infos = world.infos
    return {key: plain(infos[key]) for key in ("state", "goal_state") if key in infos}


def observation_pair_details(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, bool]:
    import torch
    keys = ("pixels", "goal", "state", "goal_state", "proprio", "action")
    details = {}
    for key in keys:
        a, b = left.get(key), right.get(key)
        if a is None and b is None:
            details[key] = True
            continue
        equal = (
            a is not None and b is not None
            and torch.is_tensor(a) and torch.is_tensor(b)
            and a.shape == b.shape and a.dtype == b.dtype
        )
        if equal and key == "action" and a.is_floating_point():
            # EverythingToInfoWrapper samples an action-space value, then
            # intentionally replaces it with an all-NaN reset placeholder.
            # Preserve the key and require equal placeholder positions,
            # equal signed infinities, and exact equality of every finite
            # action value.
            masks_equal = all(
                torch.equal(mask_a, mask_b)
                for mask_a, mask_b in (
                    (torch.isnan(a), torch.isnan(b)),
                    (torch.isposinf(a), torch.isposinf(b)),
                    (torch.isneginf(a), torch.isneginf(b)),
                    (torch.isfinite(a), torch.isfinite(b)),
                )
            )
            equal = masks_equal and torch.equal(a[torch.isfinite(a)], b[torch.isfinite(b)])
        elif equal:
            equal = torch.equal(a, b)
        details[key] = bool(equal)
    return details


def observation_pair_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(observation_pair_details(left, right).values())


def callback_equal(left: TraceCallback, right: TraceCallback) -> tuple[bool, list[dict[str, Any]]]:
    import torch
    if len(left.history) != 30 or len(right.history) != 30:
        return False, []
    per_round = []
    same = True
    for lhs, rhs in zip(left.history, right.history, strict=True):
        field_equal = {}
        for key in FIELD_NAMES:
            equal = torch.equal(lhs["tensors"][key], rhs["tensors"][key])
            field_equal[key] = bool(equal)
            same = same and bool(equal)
        per_round.append({"round": lhs["step"], "field_equal": field_equal})
    return same, per_round


def score_final_plan(official: Any, route: Any, policy: Any, plan: Any) -> tuple[float, dict[str, Any]]:
    import torch
    if route.first_cost_info is None:
        raise RuntimeError("cannot audit the final plan without the actual solver cost-input snapshot")
    info = dict(route.first_cost_info)
    for key in ("pixels", "goal"):
        value = info.get(key)
        if not torch.is_tensor(value) or value.ndim != 6 or value.shape[1] != 1:
            shape = list(value.shape) if hasattr(value, "shape") else None
            raise ValueError(f"single-candidate official cost input {key} must have shape [B,1,T,C,H,W], got {shape}")
    actions = plan.detach().to("cuda", dtype=torch.float32).unsqueeze(1)
    if actions.ndim != 4 or actions.shape[1] != 1:
        raise ValueError(f"final plan action shape must be [B,1,T,A], got {tuple(actions.shape)}")
    with torch.inference_mode():
        cost = official.get_cost(info, actions)
    if cost.shape != (1, 1) or not torch.isfinite(cost).all():
        raise FloatingPointError("final plan audit cost is invalid")
    shapes = {
        "policy_prepared_observation": {key: list(value.shape) for key, value in (policy.initial_prepared or {}).items()},
        "first_solver_cost_info": route.first_cost_input_shapes,
        "single_sample_cost_info": route.single_sample_cost_info_shapes,
        "first_solver_action_candidates": route.first_cost_action_shape,
        "final_plan_action_candidates": list(actions.shape),
        "final_plan_cost": list(cost.shape),
    }
    return float(cost[0, 0].detach().cpu()), shapes


def stage1(args: argparse.Namespace, freeze: Mapping[str, Any], schedule: Any, old: Any,
           reference: Any, official: Any, student: Any, checkpoint_meta: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    config = freeze["stage1"]
    output = args.output.resolve()
    summary_path = output / "stage1_summary.json"
    assert_fresh_result_slots(output, "stage1")
    seeds = [int(value) for value in config["simulator_reset_seeds"]]
    paths = list(config["execution_paths"])
    fidelity_paths = list(config["native_equivalence_paths"])
    if set(paths) != set(TEACHER_ROUNDS) or len(paths) != 4:
        raise ValueError("Stage 1 execution paths drifted")
    if set(fidelity_paths) != {"native_teacher_reference", "routed_teacher_reference"}:
        raise ValueError("Stage 1 native-equivalence paths drifted")
    call_records: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    all_observation_equal = True
    all_finite = True
    all_schedule_exact = True
    all_native_equal = True
    round1_stds: list[float] = []

    for block_index, seed in enumerate(seeds):
        order = paths[:]
        random.Random(int(config["arm_order_seed_base"]) + block_index).shuffle(order)
        seed_solver = int(config["cem_seed_base"]) + block_index
        records_by_path: dict[str, dict[str, Any]] = {}
        for path_name in order:
            world = make_world(args.stablewm_home)
            callback = TraceCallback(retain_tensors=False, record_population_std=(path_name == "teacher_only"))
            solve_records: list[dict[str, Any]] = []
            solver, policy, route = make_solver_and_policy(
                world, official, reference, student, path_name, [callback], seed_solver,
                instrumented_solve_records=solve_records,
            )
            try:
                reset_action_space_seed = seed_reset_action_space(world, seed)
                world.reset(seed=seed)
                initial = current_env_state(world)
                returned_action = policy.get_action(world.infos)
                torch.cuda.synchronize()
                if len(solve_records) != 1 or len(callback.history) != 30:
                    raise RuntimeError(f"expected one native 30-round solve for {path_name}")
                plan = solver.last_output["actions"] if hasattr(solver, "last_output") else None
                # Preserve the exact final mean directly from CEM outputs. The
                # timing wrapper records it before returning to WorldModelPolicy.
                if plan is None:
                    raise RuntimeError("instrumented CEM solve did not retain output")
                audit_cost, audit_shapes = score_final_plan(official, route, policy, plan)
                processed = policy.initial_prepared or {}
                entry = {
                    "path": path_name,
                    "seed": seed,
                    "order": order,
                    "solver_seed": seed_solver,
                    "reset_action_space_seed": reset_action_space_seed,
                    "initial_state_goal": initial,
                    "processed_observation": processed,
                    "first_action": plain(returned_action),
                    "final_plan": plain(plan[0]),
                    "final_plan_teacher_cost": audit_cost,
                    "final_plan_audit_shapes": audit_shapes,
                    "trace": callback.public_history(),
                    "solve": solve_records[0],
                    "teacher_rounds": list(route.solve_teacher_rounds),
                    "teacher_call_count": len(route.solve_teacher_rounds),
                    "finite": bool(solve_records[0]["finite_trace"] and solve_records[0]["output_finite"]
                                   and is_finite_numeric(returned_action) and math.isfinite(audit_cost)),
                }
                if path_name == "teacher_only":
                    round1_stds.append(float(callback.round1_population_std.item()))
                records_by_path[path_name] = entry
                call_records.append({key: value for key, value in entry.items() if key not in ("processed_observation", "final_plan")})
            finally:
                world.close()
                del policy, solver, route, world
                torch.cuda.empty_cache()

        baseline_obs = records_by_path[order[0]]["processed_observation"]
        block_observation_equal = True
        observation_key_equal_to_baseline = {}
        for entry in records_by_path.values():
            details = observation_pair_details(baseline_obs, entry["processed_observation"])
            observation_key_equal_to_baseline[entry["path"]] = details
            equal = all(details.values())
            block_observation_equal = block_observation_equal and equal
            all_observation_equal = all_observation_equal and equal
            all_finite = all_finite and bool(entry["finite"] and math.isfinite(entry["solve"]["wall_s"]) and entry["solve"]["wall_s"] > 0)
            expected_rounds = sorted(TEACHER_ROUNDS[entry["path"]])
            all_schedule_exact = all_schedule_exact and entry["teacher_rounds"] == expected_rounds and entry["teacher_call_count"] == len(expected_rounds)

        fidelity_order = fidelity_paths[:]
        random.Random(int(config["native_equivalence_order_seed_base"]) + block_index).shuffle(fidelity_order)
        fidelity_records: dict[str, dict[str, Any]] = {}
        fidelity_callbacks: dict[str, TraceCallback] = {}
        for fidelity_path in fidelity_order:
            world = make_world(args.stablewm_home)
            callback = TraceCallback(retain_tensors=True)
            solve_records: list[dict[str, Any]] = []
            route_arm = "native_teacher_reference" if fidelity_path == "native_teacher_reference" else "teacher_only"
            solver, policy, route = make_solver_and_policy(
                world, official, reference, student, route_arm, [callback], seed_solver,
                instrumented_solve_records=solve_records,
            )
            try:
                reset_action_space_seed = seed_reset_action_space(world, seed)
                world.reset(seed=seed)
                initial = current_env_state(world)
                returned_action = policy.get_action(world.infos)
                torch.cuda.synchronize()
                if len(solve_records) != 1 or len(callback.history) != 30:
                    raise RuntimeError(f"expected one 30-round fidelity solve for {fidelity_path}")
                plan = solver.last_output["actions"]
                audit_cost, audit_shapes = score_final_plan(official, route, policy, plan)
                fidelity_records[fidelity_path] = {
                    "path": fidelity_path,
                    "reset_action_space_seed": reset_action_space_seed,
                    "initial_state_goal": initial,
                    "processed_observation": policy.initial_prepared or {},
                    "first_action": plain(returned_action),
                    "final_plan": plain(plan[0]),
                    "final_plan_teacher_cost": audit_cost,
                    "final_plan_audit_shapes": audit_shapes,
                    "finite": bool(solve_records[0]["finite_trace"] and solve_records[0]["output_finite"]
                                   and is_finite_numeric(returned_action) and math.isfinite(audit_cost)),
                    "solve": solve_records[0],
                    "teacher_rounds": list(route.solve_teacher_rounds),
                    "teacher_call_count": len(route.solve_teacher_rounds),
                }
                fidelity_callbacks[fidelity_path] = callback
                call_records.append({key: value for key, value in fidelity_records[fidelity_path].items() if key != "processed_observation"})
            finally:
                world.close()
                del policy, solver, route, world
                torch.cuda.empty_cache()

        for entry in fidelity_records.values():
            details = observation_pair_details(baseline_obs, entry["processed_observation"])
            observation_key_equal_to_baseline[entry["path"]] = details
            equal = all(details.values())
            block_observation_equal = block_observation_equal and equal
            all_observation_equal = all_observation_equal and equal
            all_finite = all_finite and bool(entry["finite"] and math.isfinite(entry["solve"]["wall_s"]) and entry["solve"]["wall_s"] > 0)
            all_schedule_exact = all_schedule_exact and entry["teacher_rounds"] == list(range(1, 31)) and entry["teacher_call_count"] == 30

        native = fidelity_records["native_teacher_reference"]
        routed = fidelity_records["routed_teacher_reference"]
        native_equal, detailed_trace = callback_equal(fidelity_callbacks["native_teacher_reference"], fidelity_callbacks["routed_teacher_reference"])
        final_plan_equal = torch.equal(torch.as_tensor(native["final_plan"]), torch.as_tensor(routed["final_plan"]))
        first_action_equal = torch.equal(torch.as_tensor(native["first_action"]), torch.as_tensor(routed["first_action"]))
        native_equal = native_equal and final_plan_equal and first_action_equal
        all_native_equal = all_native_equal and native_equal
        blocks.append({
            "seed": seed,
            "randomized_execution_order": order,
            "randomized_fidelity_order": fidelity_order,
            "solver_seed": seed_solver,
            "paired_observation_equal": block_observation_equal,
            "policy_observation_key_equal_to_baseline": observation_key_equal_to_baseline,
            "native_teacher_vs_routed_teacher_only": {
                "bitwise_equal": native_equal,
                "rounds": detailed_trace,
                "final_plan_equal": bool(final_plan_equal),
                "first_action_equal": bool(first_action_equal),
            },
            "arm_records": {name: {key: value for key, value in entry.items() if key != "processed_observation"} for name, entry in records_by_path.items()},
            "fidelity_path_records": {name: {key: value for key, value in entry.items() if key != "processed_observation"} for name, entry in fidelity_records.items()},
        })

    student_costs = [records_by_seed["student_only"]["final_plan_teacher_cost"] for records_by_seed in (block["arm_records"] for block in blocks)]
    late_costs = [records_by_seed["late_teacher7"]["final_plan_teacher_cost"] for records_by_seed in (block["arm_records"] for block in blocks)]
    stds = round1_stds
    if len(stds) != len(seeds) or any(not math.isfinite(value) or value <= 0 for value in stds):
        all_finite = False
        normalized = []
    else:
        normalized = [(late - student_cost) / sd for late, student_cost, sd in zip(late_costs, student_costs, stds, strict=True)]
    strict_better = sum(late < control for late, control in zip(late_costs, student_costs, strict=True))
    late_times = [block["arm_records"]["late_teacher7"]["solve"]["wall_s"] for block in blocks]
    teacher_times = [block["arm_records"]["teacher_only"]["solve"]["wall_s"] for block in blocks]
    latency_reduction = 1.0 - statistics.mean(late_times) / statistics.mean(teacher_times)
    gates = freeze["stage1"]["gates"]
    checks = {
        "native_equivalence": all_native_equal,
        "schedule": all_schedule_exact,
        "quality": bool(normalized and statistics.median(normalized) <= float(gates["quality"]["median_late_minus_student_standardized_teacher_cost_max"]) and strict_better >= int(gates["quality"]["strictly_improved_reset_seeds_min"])),
        "latency": bool(math.isfinite(latency_reduction) and latency_reduction >= float(gates["latency"]["late_teacher7_mean_solve_time_reduction_vs_teacher_min"])),
        "validity": bool(all_observation_equal and all_finite and len(blocks) == len(seeds)),
    }
    result = {
        "schema": SCHEMA,
        "stage": "stage1",
        "status": "COMPLETED",
        "overall": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "checks": checks,
        "freeze_experiment_name": freeze["experiment_name"],
        "source_provenance": freeze["source"],
        "checkpoint": plain(checkpoint_meta),
        "sampled_seeds": seeds,
        "quality": {
            "standardized_late_minus_student_by_seed": normalized,
            "median": statistics.median(normalized) if normalized else None,
            "strictly_improved_seeds": strict_better,
            "late_final_teacher_cost": late_costs,
            "student_final_teacher_cost": student_costs,
            "teacher_round1_population_std": stds,
        },
        "latency": {
            "late_teacher7_mean_solve_s": statistics.mean(late_times),
            "teacher_only_mean_solve_s": statistics.mean(teacher_times),
            "mean_reduction_fraction": latency_reduction,
            "late_teacher7_solve_s": late_times,
            "teacher_only_solve_s": teacher_times,
        },
        "blocks": blocks,
        "claim_boundary": "Official fixed-observation CEM integration gate only; no closed-loop outcome is claimed here.",
    }
    write_json(summary_path, result)
    write_json(output / "stage1_call_records.json", {"calls": call_records})
    return result


def official_step_metrics(world: Any) -> dict[str, Any]:
    import numpy as np
    env = world.envs.envs[0].unwrapped
    infos = world.infos
    state = np.asarray(infos["state"][0, -1], dtype=np.float64)
    goal = np.asarray(infos["goal_state"][0, -1], dtype=np.float64)
    position_error = float(np.linalg.norm(goal[:4] - state[:4]))
    block = env.shapes[int(env.variation_space["block"]["shape"].value)]
    symmetry = float(env.shape_symmetry_angles.get(block, 2 * np.pi))
    angle_error = float(abs(goal[4] - state[4]) % symmetry)
    angle_error = min(angle_error, symmetry - angle_error)
    _, state_distance = env.eval_state(goal, state)
    return {
        "step": int(np.asarray(infos["step_idx"])[0, -1]) if "step_idx" in infos else None,
        "state_distance": float(state_distance),
        "position_error": position_error,
        "angle_error_rad": angle_error,
        "reward": float(world.rewards[0]),
        "terminated": bool(world.terminateds[0]),
        "truncated": bool(world.truncateds[0]),
    }


def run_episode(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any,
                official: Any, student: Any, task_seed: int, block_index: int,
                arm: str, arm_order: list[str]):
    import torch

    episode_start = time.perf_counter()
    solve_records: list[dict[str, Any]] = []
    callback = TraceCallback(retain_tensors=False)
    world = make_world(args.stablewm_home)
    solver, policy, route = make_solver_and_policy(
        world, official, reference, student, arm, [callback],
        solver_seed=int(freeze["stage2"]["pairing"]["solver_seed_base"]) + 4 * block_index,
        call_seed_fn=lambda call_index: int(freeze["stage2"]["pairing"]["solver_seed_base"]) + 4 * block_index + call_index,
        instrumented_solve_records=solve_records,
        capture_prepared=True,
    )
    step_records: list[dict[str, Any]] = []
    original_run_iter = world._run_iter

    def recording_run_iter(*a, **kw):
        original_on_step = kw.get("on_step")
        def on_step(current_world):
            if original_on_step is not None:
                original_on_step(current_world)
            step_records.append(official_step_metrics(current_world))
        kw["on_step"] = on_step
        yield from original_run_iter(*a, **kw)

    world._run_iter = recording_run_iter
    try:
        reset_action_space_seed = seed_reset_action_space(world, task_seed)
        result = world.evaluate(episodes=1, seed=int(task_seed), reset_mode="wait", dataset=None)
        torch.cuda.synchronize()
        initial = policy.initial_raw or {}
        prepared = policy.initial_prepared or {}
        success = bool(result["episode_successes"][0])
        if len(step_records) < 1 or len(step_records) > 50:
            raise RuntimeError(f"episode length outside frozen bound: {len(step_records)}")
        if len(solve_records) > 2:
            raise RuntimeError(f"planner call count exceeded two per 50-step episode: {len(solve_records)}")
        teacher_calls = len(route.teacher_rounds) if route is not None else sum(len(item["teacher_rounds"]) for item in solve_records)
        expected_rounds = sorted(TEACHER_ROUNDS[arm])
        schedule_exact = all(item["teacher_rounds"] == expected_rounds for item in solve_records)
        solver_seed_base = int(freeze["stage2"]["pairing"]["solver_seed_base"])
        rng_exact = all(
            bool(item["policy_reseeded"])
            and item["planner_call_index"] == call_index
            and item["solver_seed"] == solver_seed_base + 4 * block_index + call_index
            for call_index, item in enumerate(solve_records)
        )
        finite = all(math.isfinite(float(row[k])) for row in step_records for k in ("state_distance", "position_error", "angle_error_rad", "reward"))
        finite = finite and all(item["finite_trace"] and item["output_finite"] and math.isfinite(item["wall_s"]) for item in solve_records)
        row = {
            "schema": SCHEMA,
            "stage": "stage2_episode",
            "task_seed": int(task_seed),
            "block_index": int(block_index),
            "arm": arm,
            "randomized_arm_order": arm_order,
            "reset_action_space_seed": reset_action_space_seed,
            "initial_state_goal": initial,
            "policy_prepared_observation_shapes": {key: list(value.shape) for key, value in prepared.items()},
            "success": success,
            "steps": len(step_records),
            "terminated": bool(step_records[-1]["terminated"]),
            "truncated": bool(step_records[-1]["truncated"]),
            "step_metrics": step_records,
            "episode_wall_s": float(time.perf_counter() - episode_start),
            "planner_call_count": len(solve_records),
            "planner_solve_wall_s": [item["wall_s"] for item in solve_records],
            "planner_total_wall_s": float(sum(item["wall_s"] for item in solve_records)),
            "planner_calls": solve_records,
            "teacher_call_count": int(teacher_calls),
            "schedule_exact": bool(schedule_exact and teacher_calls == len(expected_rounds) * len(solve_records)),
            "paired_innovation_seed_exact": bool(rng_exact),
            "teacher_rounds": list(route.teacher_rounds) if route is not None else list(range(1, 31)) * len(solve_records),
            "peak_cuda_allocated_bytes": max((item["peak_cuda_allocated_bytes"] for item in solve_records), default=0),
            "peak_cuda_reserved_bytes": max((item["peak_cuda_reserved_bytes"] for item in solve_records), default=0),
            "finite": bool(finite),
        }
        return row, prepared
    finally:
        world.close()
        del policy, solver, route, world
        torch.cuda.empty_cache()


def mcnemar_exact(success_late: list[bool], success_student: list[bool]) -> dict[str, Any]:
    b = sum(l and not s for l, s in zip(success_late, success_student, strict=True))
    c = sum(s and not l for l, s in zip(success_late, success_student, strict=True))
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)
    return {"late_success_student_failure": b, "student_success_late_failure": c, "discordant_pairs": n, "two_sided_exact_p": p_value}


def stage2(args: argparse.Namespace, freeze: Mapping[str, Any], reference: Any,
           official: Any, student: Any, checkpoint_meta: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    config = freeze["stage2"]
    if args.stage1_summary is None:
        raise ValueError("Stage 2 requires the Stage 1 summary path")
    first = read_json(args.stage1_summary.resolve())
    if first.get("overall") != "PASS" or first.get("freeze_experiment_name") != freeze["experiment_name"]:
        raise RuntimeError("Stage 1 is not PASS for this frozen experiment; Stage 2 is fail-closed")
    output = args.output.resolve()
    jsonl_path = output / "stage2_episodes.jsonl"
    summary_path = output / "stage2_summary.json"
    assert_fresh_result_slots(output, "stage2")
    output.mkdir(parents=True, exist_ok=True)
    first_seed = int(config["simulator_reset_seeds"]["first"])
    count = int(config["simulator_reset_seeds"]["count"])
    seeds = list(range(first_seed, first_seed + count))
    if seeds[-1] != int(config["simulator_reset_seeds"]["last"]):
        raise ValueError("frozen Stage 2 seed range drifted")
    arms = ["student_only", "late_teacher7", "teacher_only", "uniform_teacher7"]
    all_rows: list[dict[str, Any]] = []
    identity_by_seed: dict[int, dict[str, Any]] = {}
    for block_index, task_seed in enumerate(seeds):
        order = arms[:]
        random.Random(int(config["pairing"]["arm_order_seed_base"]) + block_index).shuffle(order)
        paired: dict[str, dict[str, Any]] = {}
        prepared_baseline = None
        prepared_reference_arm = order[0]
        prepared_observation_key_equal_by_arm = {}
        for arm in order:
            row, prepared = run_episode(args, freeze, reference, official, student, task_seed, block_index, arm, order)
            if prepared_baseline is None:
                prepared_baseline = prepared
            key_equal = observation_pair_details(prepared_baseline, prepared)
            prepared_observation_key_equal_by_arm[arm] = key_equal
            row["policy_prepared_observation_reference_arm"] = prepared_reference_arm
            row["policy_prepared_observation_key_equal"] = key_equal
            paired[arm] = row
            all_rows.append(row)
            with jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            write_json(output / "stage2_progress.json", {
                "completed_episodes": len(all_rows), "planned_episodes": len(seeds) * len(arms),
                "completed_blocks": block_index + (1 if len(paired) == len(arms) else 0),
                "last_task_seed": task_seed, "last_arm": arm, "randomized_arm_order": order,
            })
        states = [row["initial_state_goal"] for row in paired.values()]
        exact_pair = all(value == states[0] for value in states[1:])
        exact_prepared_pair = all(all(value.values()) for value in prepared_observation_key_equal_by_arm.values())
        rng_pair = all(row["paired_innovation_seed_exact"] for row in paired.values())
        schedule_pair = all(row["schedule_exact"] for row in paired.values())
        identity_by_seed[task_seed] = {
            "exact_initial_state_and_goal": exact_pair,
            "exact_policy_prepared_observation": exact_prepared_pair,
            "policy_prepared_observation_key_equal_by_arm": prepared_observation_key_equal_by_arm,
            "paired_solver_innovations_exact": rng_pair,
            "schedules_exact": schedule_pair,
            "arm_order": order,
        }

    by_arm = {arm: [row for row in all_rows if row["arm"] == arm] for arm in arms}
    successes = {arm: sum(bool(row["success"]) for row in rows) for arm, rows in by_arm.items()}
    by_seed_arm = {(int(row["task_seed"]), row["arm"]): row for row in all_rows}
    student_success = [bool(by_seed_arm[(seed, "student_only")]["success"]) for seed in seeds]
    late_success = [bool(by_seed_arm[(seed, "late_teacher7")]["success"]) for seed in seeds]
    teacher_success = [bool(by_seed_arm[(seed, "teacher_only")]["success"]) for seed in seeds]
    paired_success = {
        "late7_minus_student": successes["late_teacher7"] - successes["student_only"],
        "late7_minus_teacher": successes["late_teacher7"] - successes["teacher_only"],
        "late7_vs_student_mcnemar": mcnemar_exact(late_success, student_success),
    }
    pair_valid = all(value["exact_initial_state_and_goal"] and value["exact_policy_prepared_observation"] and value["paired_solver_innovations_exact"] and value["schedules_exact"] for value in identity_by_seed.values())
    finite_valid = all(bool(row["finite"]) for row in all_rows)
    complete = len(all_rows) == len(seeds) * len(arms) and all(len(rows) == len(seeds) for rows in by_arm.values())
    gates = config["decision_gate"]
    benefit = paired_success["late7_minus_student"] >= 5 and paired_success["late7_vs_student_mcnemar"]["two_sided_exact_p"] < 0.05
    teacher_margin = paired_success["late7_minus_teacher"] >= -5
    checks = {"complete": complete, "pairing": pair_valid, "finite": finite_valid, "late7_benefit_vs_student": benefit, "late7_practical_margin_vs_teacher": teacher_margin}
    overall = "PASS" if all(checks.values()) else ("FAIL" if complete and pair_valid and finite_valid else "INVALID")
    result = {
        "schema": SCHEMA,
        "stage": "stage2",
        "status": "COMPLETED",
        "overall": overall,
        "checks": checks,
        "freeze_experiment_name": freeze["experiment_name"],
        "source_provenance": freeze["source"],
        "stage1_summary": str(args.stage1_summary.resolve()),
        "checkpoint": plain(checkpoint_meta),
        "seeds": seeds,
        "arms": {arm: {"successes": successes[arm], "episodes": len(by_arm[arm]), "success_rate": successes[arm] / max(1, len(by_arm[arm])), "mean_episode_wall_s": statistics.mean(row["episode_wall_s"] for row in by_arm[arm]), "mean_total_planner_wall_s": statistics.mean(row["planner_total_wall_s"] for row in by_arm[arm]), "mean_teacher_calls": statistics.mean(row["teacher_call_count"] for row in by_arm[arm]), "max_peak_cuda_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in by_arm[arm])} for arm in arms},
        "paired_success": paired_success,
        "pairing_by_seed": identity_by_seed,
        "episodes_jsonl": str(jsonl_path),
        "validity": {"episode_rows": len(all_rows), "expected_episode_rows": len(seeds) * len(arms), "all_finite": finite_valid, "all_reset_pairs_exact": pair_valid},
        "claim_boundary": "Paired closed-loop outcome on fresh seeded episodes in the pinned stable-worldmodel PushT simulator; no formal powered non-inferiority claim.",
    }
    write_json(summary_path, result)
    return result


def main() -> int:
    args = parse_args()
    host = require_compute_node()
    freeze = read_json(args.freeze.resolve())
    validate_freeze(freeze)
    if args.output.exists() and not args.output.is_dir():
        raise NotADirectoryError(args.output)
    if args.stage == "stage2":
        if args.stage1_summary is None or not args.stage1_summary.is_file():
            raise FileNotFoundError("Stage 2 requires a completed Stage 1 summary")
        prior = read_json(args.stage1_summary.resolve())
        if prior.get("overall") != "PASS" or prior.get("freeze_experiment_name") != freeze["experiment_name"]:
            raise RuntimeError("Stage 1 is not PASS for this freeze; refusing to load models or start Stage 2")
    assert_fresh_result_slots(args.output.resolve(), args.stage)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["STABLEWM_HOME"] = str(args.stablewm_home.resolve())
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"
    schedule, old, reference, official, student, checkpoint_meta = load_models(args, freeze)
    import torch
    identity = {
        "job_id": os.environ["PBS_JOBID"], "hostname": host, "stage": args.stage,
        "experiment_name": freeze["experiment_name"], "freeze_status": freeze["status"],
        "source_provenance": freeze["source"],
        "started_unix": time.time(), "checkpoint": str(args.checkpoint),
        "probe": str(args.probe), "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "pbs_nodefile": os.environ["PBS_NODEFILE"],
    }
    write_json(args.output.resolve() / "execution_identity.json", identity)
    if args.stage == "stage1":
        result = stage1(args, freeze, schedule, old, reference, official, student, checkpoint_meta)
    else:
        result = stage2(args, freeze, reference, official, student, checkpoint_meta)
    print(json.dumps({"stage": args.stage, "overall": result["overall"], "summary": str((args.output.resolve() / f"{args.stage}_summary.json"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

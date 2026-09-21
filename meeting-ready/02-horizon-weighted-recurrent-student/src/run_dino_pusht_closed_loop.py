#!/usr/bin/env python3
"""Paired PushT closed-loop smoke for the trained recurrent student and teacher."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing execution outside PBS")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _set_seed(value: int) -> None:
    import numpy as np
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _student_module(action_dim: int, visual_dim: int, proprio_dim: int, hidden_dim: int) -> Any:
    import torch
    import torch.nn as nn

    class RecurrentNativeLatentTransitionStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_projection = nn.Linear(visual_dim, hidden_dim)
            self.proprio_projection = nn.Linear(proprio_dim, hidden_dim)
            self.action_projection = nn.Linear(action_dim, hidden_dim)
            self.shared_transition = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.visual_update_norm = nn.LayerNorm(hidden_dim)
            self.proprio_update_norm = nn.LayerNorm(hidden_dim)
            self.visual_out = nn.Linear(hidden_dim, visual_dim)
            self.proprio_out = nn.Linear(hidden_dim, proprio_dim)

        def forward(self, context: Mapping[str, Any], actions: Any) -> dict[str, Any]:
            visual_state = context["visual"][:, -1]
            proprio_state = context["proprio"][:, -1]
            visual_outputs = []
            proprio_outputs = []
            for index in range(int(actions.shape[1])):
                visual_hidden = self.visual_projection(visual_state)
                proprio_hidden = self.proprio_projection(proprio_state)
                action_hidden = self.action_projection(actions[:, index])
                transition = self.shared_transition(
                    visual_hidden.mean(dim=1) + proprio_hidden + action_hidden
                )
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

    return RecurrentNativeLatentTransitionStudent()


def _student_world_model(teacher: Any, student: Any) -> Any:
    import torch
    import torch.nn as nn

    class StudentWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.teacher_encoder = teacher
            self.student = student
            self.decoder = None

        def encode_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
            return self.teacher_encoder.encode_obs(obs)

        def rollout(self, obs_0: Mapping[str, Any], act: Any) -> tuple[dict[str, Any], None]:
            if act.ndim != 3 or int(act.shape[-1]) != 10:
                raise ValueError(f"expected packed actions [B,T,10], got {tuple(act.shape)}")
            encoded = self.encode_obs(obs_0)
            predicted = self.student(encoded, act)
            full = {
                key: torch.cat([encoded[key][:, -1:], predicted[key]], dim=1)
                for key in ("visual", "proprio")
            }
            return full, None

    return StudentWorldModel()


def _teacher_planning_world_model(teacher: Any) -> Any:
    """Expose the official teacher rollout while disabling diagnostic rendering."""

    import torch.nn as nn

    class TeacherPlanningWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.teacher = teacher
            self.decoder = None

        def encode_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
            return self.teacher.encode_obs(obs)

        def rollout(self, obs_0: Mapping[str, Any], act: Any) -> tuple[dict[str, Any], Any]:
            return self.teacher.rollout(obs_0=obs_0, act=act)

    return TeacherPlanningWorldModel()


def _validate_freeze(freeze: Mapping[str, Any], source: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if freeze.get("schema") != "horizon-weighted-recurrent-student.pusht-closed-loop-freeze":
        raise ValueError("unexpected closed-loop freeze schema")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unsupported closed-loop freeze schema version")
    settings = dict(freeze["official_pusht"])
    official = OmegaConf.load(source / "conf" / "plan_pusht.yaml")
    if official.planner._target_ != "planning.mpc.MPCPlanner":
        raise ValueError("official PushT main planner is no longer MPCPlanner")
    if official.planner.sub_planner.target != "planning.cem.CEMPlanner":
        raise ValueError("official PushT sub-planner is no longer CEMPlanner")
    checks = {
        "seed": official.seed,
        "goal_H": official.goal_H,
        "goal_source": official.goal_source,
        "n_taken_actions": official.planner.n_taken_actions,
    }
    for key, actual in checks.items():
        if str(actual) != str(settings[key]):
            raise ValueError(f"frozen {key} differs from official plan_pusht.yaml")
    for key, expected in settings["cem"].items():
        if float(official.planner.sub_planner[key]) != float(expected):
            raise ValueError(f"frozen CEM {key} differs from official plan_pusht.yaml")
    return settings


def _load_models(args: argparse.Namespace, freeze: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    import hydra
    import torch
    from omegaconf import OmegaConf

    root = args.root.resolve()
    source = root / "source"
    sys.path.insert(0, str(source))
    import plan

    config_path = (args.checkpoint_config or root / "checkpoints/outputs/pusht/hydra.yaml").resolve()
    checkpoint_path = (args.teacher_checkpoint or config_path.parent / "checkpoints/model_latest.pth").resolve()
    data_root = (args.data_root or root / "data/pusht_noise").resolve()
    for required in (config_path, checkpoint_path, data_root, args.student_checkpoint.resolve()):
        if not required.exists():
            raise FileNotFoundError(required)
    os.environ["DATASET_DIR"] = str(data_root.parent if data_root.name == "pusht_noise" else data_root)
    os.environ["WANDB_MODE"] = "disabled"
    model_cfg = OmegaConf.load(config_path)
    _, datasets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    teacher = plan.load_model(
        checkpoint_path,
        model_cfg,
        model_cfg.num_action_repeat,
        device=torch.device("cuda:0"),
    )
    # The pinned official VWorldModel.eval() predates the usual fluent return.
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    spec = freeze["student"]
    student = _student_module(
        int(spec["packed_action_dim"]),
        int(spec["visual_dim"]),
        int(spec["proprio_dim"]),
        int(spec["hidden_dim"]),
    ).to("cuda:0")
    payload = torch.load(args.student_checkpoint.resolve(), map_location="cpu")
    if payload.get("schema") != spec["checkpoint_schema"] or int(payload.get("step", -1)) != int(spec["step"]):
        raise ValueError("student checkpoint identity does not match the freeze")
    student.load_state_dict(payload["state_dict"], strict=True)
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return teacher, _student_world_model(teacher, student).eval(), model_cfg, datasets["valid"]


def _workspace(args: argparse.Namespace, freeze: Mapping[str, Any], settings: Mapping[str, Any], teacher: Any, model_cfg: Any, dataset: Any) -> Any:
    import plan
    from env.pusht.pusht_wrapper import PushTWrapper
    from env.serial_vector_env import SerialVectorEnv
    from omegaconf import OmegaConf

    env_kwargs = OmegaConf.to_container(model_cfg.env.kwargs, resolve=True)
    environment = SerialVectorEnv([PushTWrapper(**env_kwargs) for _ in range(int(settings["n_evals"]))])
    cfg = {
        "seed": int(settings["seed"]),
        "n_evals": int(settings["n_evals"]),
        "goal_source": settings["goal_source"],
        "goal_H": int(settings["goal_H"]),
        "n_plot_samples": 0,
        "debug_dset_init": False,
        "objective": {"_target_": "planning.objectives.create_objective_fn", **settings["objective"]},
        "planner": {
            "_target_": "planning.mpc.MPCPlanner",
            "max_iter": int(settings["max_mpc_rounds"]),
            "n_taken_actions": int(settings["n_taken_actions"]),
            "sub_planner": {"target": "planning.cem.CEMPlanner", **settings["cem"]},
            "name": "mpc_cem",
        },
        "saved_folder": str(args.output.resolve()),
        "wandb_logging": False,
    }
    _set_seed(int(settings["seed"]))
    os.chdir(args.output.resolve())
    return plan.PlanWorkspace(
        cfg_dict=cfg,
        wm=teacher,
        dset=dataset,
        env=environment,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=None,
    )


def _fresh_planner(workspace: Any, wm: Any, settings: Mapping[str, Any]) -> tuple[Any, Any]:
    from planning.evaluator import PlanEvaluator
    from planning.mpc import MPCPlanner

    evaluator = PlanEvaluator(
        obs_0=workspace.obs_0,
        obs_g=workspace.obs_g,
        state_0=workspace.state_0,
        state_g=workspace.state_g,
        env=workspace.env,
        wm=wm,
        frameskip=workspace.frameskip,
        seed=workspace.eval_seed,
        preprocessor=workspace.data_preprocessor,
        n_plot_samples=0,
    )
    planner = MPCPlanner(
        max_iter=int(settings["max_mpc_rounds"]),
        n_taken_actions=int(settings["n_taken_actions"]),
        sub_planner={"target": "planning.cem.CEMPlanner", **settings["cem"]},
        wm=wm,
        env=workspace.env,
        action_dim=workspace.action_dim,
        objective_fn=workspace.planner.objective_fn,
        preprocessor=workspace.data_preprocessor,
        evaluator=evaluator,
        wandb_run=workspace.wandb_run,
        logging_prefix="mpc",
        log_filename=None,
    )
    return planner, evaluator


def _run_arm(name: str, wm: Any, workspace: Any, settings: Mapping[str, Any], planner_seed: int, output: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    planner, evaluator = _fresh_planner(workspace, wm, settings)
    _set_seed(planner_seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        actions, action_len = planner.plan(workspace.obs_0, workspace.obs_g, actions=None)
    torch.cuda.synchronize()
    planner_seconds = time.perf_counter() - started
    _, successes, _, states = evaluator.eval_actions(actions.detach(), action_len, save_video=False, filename=f"{name}_final")
    final_indices = np.where(np.isinf(action_len), -1, action_len * workspace.frameskip).astype(int)
    final_states = states[np.arange(states.shape[0]), final_indices]
    eval_results = workspace.env.eval_state(workspace.state_g, final_states)
    episodes = []
    for index in range(len(successes)):
        model_actions = None if math.isinf(float(action_len[index])) else int(action_len[index])
        episodes.append({
            "case_index": index,
            "eval_seed": int(workspace.eval_seed[index]),
            "success": bool(successes[index]),
            "state_distance": float(eval_results["state_dist"][index]),
            "model_action_length": model_actions,
            "environment_steps": int(settings["max_mpc_rounds"] * settings["n_taken_actions"] * workspace.frameskip) if model_actions is None else int(model_actions * workspace.frameskip),
        })
    result = {
        "arm": name,
        "success_count": int(np.asarray(successes).sum()),
        "n_evals": len(episodes),
        "success_rate": float(np.asarray(successes, dtype=float).mean()),
        "planner_walltime_seconds": planner_seconds,
        "peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        "episodes": episodes,
    }
    (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = _args()
    _require_compute_node()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    source = args.root.resolve() / "source"
    settings = _validate_freeze(freeze, source)
    teacher, student_wm, model_cfg, dataset = _load_models(args, freeze)
    teacher_wm = _teacher_planning_world_model(teacher).eval()
    workspace = _workspace(args, freeze, settings, teacher, model_cfg, dataset)
    if int(workspace.action_dim) != int(freeze["student"]["packed_action_dim"]):
        raise ValueError("official packed action dimension differs from the student")
    target_manifest = {
        "seed": int(settings["seed"]),
        "eval_seeds": [int(value) for value in workspace.eval_seed],
        "n_evals": int(settings["n_evals"]),
        "goal_source": settings["goal_source"],
        "obs_shapes": {key: list(value.shape) for key, value in workspace.obs_0.items()},
        "goal_shapes": {key: list(value.shape) for key, value in workspace.obs_g.items()},
    }
    (output / "target_manifest.json").write_text(json.dumps(target_manifest, indent=2), encoding="utf-8")
    planner_seed = int(freeze["comparison"]["planner_seed_per_arm"])
    arms = {
        "official_dino_wm_teacher": _run_arm("official_dino_wm_teacher", teacher_wm, workspace, settings, planner_seed, output),
        "horizon_weighted_recurrent_student": _run_arm("horizon_weighted_recurrent_student", student_wm, workspace, settings, planner_seed, output),
    }
    paired = []
    for teacher_row, student_row in zip(arms["official_dino_wm_teacher"]["episodes"], arms["horizon_weighted_recurrent_student"]["episodes"], strict=True):
        paired.append({
            "case_index": teacher_row["case_index"],
            "eval_seed": teacher_row["eval_seed"],
            "teacher_success": teacher_row["success"],
            "student_success": student_row["success"],
            "student_minus_teacher_state_distance": student_row["state_distance"] - teacher_row["state_distance"],
        })
    table = {
        "both_success": sum(row["teacher_success"] and row["student_success"] for row in paired),
        "teacher_only": sum(row["teacher_success"] and not row["student_success"] for row in paired),
        "student_only": sum(not row["teacher_success"] and row["student_success"] for row in paired),
        "both_fail": sum(not row["teacher_success"] and not row["student_success"] for row in paired),
    }
    gate_spec = freeze["comparison"].get("progression_gate")
    progression = None
    if gate_spec is not None:
        teacher_count = int(arms["official_dino_wm_teacher"]["success_count"])
        student_count = int(arms["horizon_weighted_recurrent_student"]["success_count"])
        passed = (
            student_count >= int(gate_spec["student_success_count_min"])
            and teacher_count - student_count <= int(gate_spec["student_deficit_to_teacher_max"])
        )
        progression = {
            "status": "PASS" if passed else "FAIL",
            "teacher_success_count": teacher_count,
            "student_success_count": student_count,
            "student_deficit_to_teacher": teacher_count - student_count,
            "gate": gate_spec,
            "decision": gate_spec["next_stage_if_pass"] if passed else gate_spec["next_stage_if_fail"],
        }
    summary = {
        "schema": "horizon-weighted-recurrent-student.pusht-closed-loop-summary",
        "schema_version": 1,
        "stage": freeze["stage"],
        "freeze": str(args.freeze.resolve()),
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "teacher_checkpoint": str((args.teacher_checkpoint or args.root / "checkpoints/outputs/pusht/checkpoints/model_latest.pth").resolve()),
        "target_manifest": target_manifest,
        "official_settings": settings,
        "arms": arms,
        "paired_episodes": paired,
        "paired_success_table": table,
        "progression": progression,
        "claim_boundary": freeze["claim_boundary"],
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "paired_success_table": table, "success_rates": {name: value["success_rate"] for name, value in arms.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

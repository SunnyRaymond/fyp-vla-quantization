#!/usr/bin/env python3
"""Stage-A action-prefix compiler smoke for the official DINO-WM PushT model.

The runner deliberately starts after ``VWorldModel.encode_obs``.  It loads the
official PushT checkpoint through the existing ``plan.load_model`` path, keeps
the teacher frozen, and trains only a small native-geometry student:

    cached observation latent + raw normalized action prefix -> dense
    observation latents (visual/proprio), without action dimensions.

This is a predictor-level experiment.  It does not run CEM or closed-loop
control.  The two anchors and their goals are prepared by the official
``PlanWorkspace`` dset path so that the input contract remains tied to the
existing benchmark rather than a synthetic observation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn


HORIZON = 5
ANCHOR_COUNT = 2


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM reproduction root")
    parser.add_argument("--asset-root", type=Path, default=None, help="staged checkpoint/data root")
    parser.add_argument("--output", type=Path, required=True, help="Stage-A artifact directory")
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path(__file__).parents[1]
        / "dino-wm-shared-prefix-cache"
        / "pusht_transfer"
        / "PUSHT_TRANSFER_FREEZE.json",
        help="frozen official PushT transfer configuration",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("STAGE_A_FREEZE.json"),
        help="frozen predictor-level Stage-A protocol",
    )
    parser.add_argument("--config", type=Path, default=None, help="official plan_pusht.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="official model_latest.pth")
    parser.add_argument("--checkpoint-config", type=Path, default=None, help="checkpoint hydra.yaml")
    parser.add_argument("--data-root", type=Path, default=None, help="staged data/pusht_noise directory")
    parser.add_argument("--deps-root", type=Path, default=None, help="accepted for PBS contract; PYTHONPATH is prepared by wrapper")
    parser.add_argument("--summary", type=Path, default=None, help="optional alternate summary path")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=300)
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--latency-batch", type=int, default=300)
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--leakage-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing model loading outside a PBS allocation")
    if os.name != "posix":
        raise RuntimeError("Stage-A model loading requires a Linux PBS compute node")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _json_default(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _all_finite(mapping: Mapping[str, Any]) -> bool:
    import torch

    return all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in mapping.values())


def _require_assets(asset_root: Path, freeze: Mapping[str, Any]) -> None:
    assets = freeze.get(
        "assets",
        {
            "checkpoint_config": "checkpoints/outputs/pusht/hydra.yaml",
            "checkpoint_file": "checkpoints/outputs/pusht/checkpoints/model_latest.pth",
            "dataset_root": "data/pusht_noise",
            "dataset_splits": ["train", "val"],
            "required_dataset_files": [
                "states.pth",
                "rel_actions.pth",
                "seq_lengths.pkl",
                "velocities.pth",
                "obses/episode_000.mp4",
            ],
        },
    )
    required = [
        asset_root / assets["checkpoint_config"],
        asset_root / assets["checkpoint_file"],
    ]
    for split in assets["dataset_splits"]:
        split_root = asset_root / assets["dataset_root"] / split
        required.extend(split_root / item for item in assets["required_dataset_files"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "PushT assets are not staged; refusing download/extraction: " + ", ".join(missing)
        )


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gpu_snapshot(output: Path, label: str) -> None:
    """Append a small periodic utilization/memory sample to the job artifacts."""

    record: dict[str, Any] = {"label": label, "time": time.time()}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        record["query_returncode"] = result.returncode
        rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                rows.append(
                    {
                        "index": int(fields[0]),
                        "utilization_gpu_percent": float(fields[1]),
                        "memory_used_mib": float(fields[2]),
                        "memory_total_mib": float(fields[3]),
                    }
                )
        record["gpus"] = rows
    except (OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    with (output / "gpu_telemetry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class CausalPrefixEncoder(nn.Module):
    """Encode action prefixes with a causal mask and shared parameters."""

    def __init__(self, action_dim: int, context_dim: int, hidden_dim: int) -> None:
        import torch
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_summary: Any, actions: Any) -> Any:
        import torch

        prefix_length = actions.shape[1]
        state = self.context_proj(context_summary).unsqueeze(1)
        tokens = self.action_proj(actions) + state
        mask = torch.triu(
            torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=actions.device),
            diagonal=1,
        )
        return self.norm(self.transformer(tokens, mask=mask))


class NativeDinoPrefixStudent(nn.Module):
    """DINO-shaped student: causal prefixes plus shared per-patch residual heads."""

    def __init__(self, action_dim: int, visual_dim: int, proprio_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.proprio_dim = proprio_dim
        context_dim = visual_dim + proprio_dim
        self.prefix = CausalPrefixEncoder(action_dim, context_dim, hidden_dim)
        self.visual_in = nn.Linear(visual_dim, hidden_dim)
        self.proprio_in = nn.Linear(proprio_dim, hidden_dim)
        self.visual_residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.proprio_residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.visual_out = nn.Linear(hidden_dim, visual_dim)
        self.proprio_out = nn.Linear(hidden_dim, proprio_dim)

    def forward(self, context: Mapping[str, Any], actions: Any) -> dict[str, Any]:
        visual_anchor = context["visual"][:, -1]
        proprio_anchor = context["proprio"][:, -1]
        visual_summary = visual_anchor.mean(dim=1)
        context_summary = torch_cat((visual_summary, proprio_anchor), dim=-1)
        prefix_tokens = self.prefix(context_summary, actions)

        visual_base = self.visual_in(visual_anchor)
        visual_hidden = self.visual_residual(
            visual_base.unsqueeze(1) + prefix_tokens.unsqueeze(2)
        )
        visual = visual_anchor.unsqueeze(1) + self.visual_out(visual_hidden)

        proprio_base = self.proprio_in(proprio_anchor)
        proprio_hidden = self.proprio_residual(
            proprio_base.unsqueeze(1) + prefix_tokens
        )
        proprio = proprio_anchor.unsqueeze(1) + self.proprio_out(proprio_hidden)
        return {"visual": visual, "proprio": proprio}


def torch_cat(values: tuple[Any, ...], dim: int) -> Any:
    import torch

    return torch.cat(values, dim=dim)


def _select(mapping: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    return {
        key: value.index_select(0, indices.to(value.device))
        for key, value in mapping.items()
    }


def _repeat_anchor(mapping: Mapping[str, Any], batch: int) -> dict[str, Any]:
    return {key: value[:1].expand((batch,) + tuple(value.shape[1:])) for key, value in mapping.items()}


def _teacher_targets(model: Any, cached_obs: Mapping[str, Any], actions: Any) -> dict[str, Any]:
    """Run the frozen official teacher from native cached ``encode_obs`` output."""

    from cache_core import rollout_from_encoded_obs

    with torch_no_grad():
        predicted, _ = rollout_from_encoded_obs(model, cached_obs, actions)
    targets = {key: value[:, 1 : HORIZON + 1].detach() for key, value in predicted.items()}
    if any(value.shape[1] != HORIZON for value in targets.values()):
        raise RuntimeError("teacher did not produce exactly H=5 dense future targets")
    return targets


class torch_no_grad:
    def __enter__(self):
        import torch

        self._context = torch.no_grad()
        return self._context.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return self._context.__exit__(exc_type, exc_value, traceback)


def _sample_actions(generator: Any, batch: int, action_dim: int, device: Any) -> Any:
    import torch

    # CPU generation keeps held-out seeds independent of CUDA RNG state.
    return torch.randn((batch, HORIZON, action_dim), generator=generator).to(device)


def _metrics_by_horizon(pred: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, list[float]]:
    import torch
    import torch.nn.functional as F

    relative: list[float] = []
    cosine: list[float] = []
    for horizon_index in range(HORIZON):
        rel_fields = []
        cos_fields = []
        for key in ("visual", "proprio"):
            p = pred[key][:, horizon_index].flatten(start_dim=1)
            t = target[key][:, horizon_index].flatten(start_dim=1)
            mse = (p - t).square().mean()
            denom = t.square().mean().clamp_min(1e-8)
            rel_fields.append(float((mse / denom).detach().cpu()))
            cos_fields.append(float(F.cosine_similarity(p, t, dim=1).mean().detach().cpu()))
        relative.append(sum(rel_fields) / len(rel_fields))
        cosine.append(sum(cos_fields) / len(cos_fields))
    return {"relative_mse": relative, "cosine": cosine}


def _rank(values: Any) -> Any:
    import torch

    return torch.argsort(torch.argsort(values, stable=True), stable=True).float()


def _spearman(first: Any, second: Any) -> float:
    import torch

    a = _rank(first.detach().float().cpu())
    b = _rank(second.detach().float().cpu())
    a = a - a.mean()
    b = b - b.mean()
    denominator = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-12))
    return float((a * b).sum() / denominator)


def _objective_cost(objective_fn: Any, prediction: Mapping[str, Any], goal: Mapping[str, Any]) -> Any:
    return objective_fn(prediction, goal)


def _topk_overlap(first: Any, second: Any, k: int = 30) -> float:
    k = min(k, int(first.numel()), int(second.numel()))
    first_top = set(torch_argsort(first)[:k].detach().cpu().tolist())
    second_top = set(torch_argsort(second)[:k].detach().cpu().tolist())
    return float(len(first_top.intersection(second_top)) / max(k, 1))


def torch_argsort(values: Any) -> Any:
    import torch

    return torch.argsort(values)


def _evaluate_student(
    student: Any,
    model: Any,
    anchors: Mapping[str, Any],
    goals: Mapping[str, Any],
    objective_fn: Any,
    action_dim: int,
    eval_seeds: list[int],
    eval_batch: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    student.eval()
    all_metrics: list[dict[str, Any]] = []
    all_cost_metrics: list[dict[str, Any]] = []
    for eval_seed in eval_seeds:
        generator = torch.Generator(device="cpu").manual_seed(eval_seed)
        for anchor_index in range(ANCHOR_COUNT):
            context = _repeat_anchor(
                {key: value[anchor_index : anchor_index + 1] for key, value in anchors.items()},
                eval_batch,
            )
            goal = _repeat_anchor(
                {key: value[anchor_index : anchor_index + 1, :1] for key, value in goals.items()},
                eval_batch,
            )
            actions = _sample_actions(generator, eval_batch, action_dim, device)
            target = _teacher_targets(model, context, actions)
            with torch.no_grad():
                prediction = student(context, actions)
            metrics = _metrics_by_horizon(prediction, target)
            metrics.update({"seed": eval_seed, "anchor": anchor_index})
            all_metrics.append(metrics)
            teacher_cost = _objective_cost(objective_fn, target, goal)
            student_cost = _objective_cost(objective_fn, prediction, goal)
            all_cost_metrics.append(
                {
                    "seed": eval_seed,
                    "anchor": anchor_index,
                    "spearman": _spearman(teacher_cost, student_cost),
                    "top30_overlap": _topk_overlap(teacher_cost, student_cost),
                }
            )

    def _mean_series(key: str) -> list[float]:
        return [sum(item[key][index] for item in all_metrics) / len(all_metrics) for index in range(HORIZON)]

    return {
        "per_horizon": {"relative_mse": _mean_series("relative_mse"), "cosine": _mean_series("cosine")},
        "per_case": all_metrics,
        "terminal_ranking": {
            "spearman_mean": sum(item["spearman"] for item in all_cost_metrics) / len(all_cost_metrics),
            "top30_overlap_mean": sum(item["top30_overlap"] for item in all_cost_metrics) / len(all_cost_metrics),
            "per_case": all_cost_metrics,
        },
    }


def _leakage_test(student: Any, anchors: Mapping[str, Any], action_dim: int, seed: int, device: Any, tolerance: float) -> dict[str, Any]:
    import torch

    student.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context = {key: value[:ANCHOR_COUNT] for key, value in anchors.items()}
    actions_a = _sample_actions(generator, ANCHOR_COUNT, action_dim, device)
    suffix = torch.randn((ANCHOR_COUNT, HORIZON, action_dim), generator=generator).to(device)
    results = []
    with torch.no_grad():
        for cut in range(1, HORIZON):
            actions_b = actions_a.clone()
            actions_b[:, cut:] = suffix[:, cut:]
            pred_a = student(context, actions_a)
            pred_b = student(context, actions_b)
            visual_delta = (pred_a["visual"][:, :cut] - pred_b["visual"][:, :cut]).abs().max().item()
            proprio_delta = (pred_a["proprio"][:, :cut] - pred_b["proprio"][:, :cut]).abs().max().item()
            max_delta = max(visual_delta, proprio_delta)
            results.append({"unchanged_prefix_length": cut, "max_abs_delta": max_delta, "passed": max_delta <= tolerance})
    return {"tolerance": tolerance, "passed": all(item["passed"] for item in results), "cases": results}


def _latency(
    student: Any,
    model: Any,
    anchors: Mapping[str, Any],
    action_dim: int,
    seed: int,
    batch: int,
    warmup: int,
    repeats: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    context = _repeat_anchor({key: value[:1] for key, value in anchors.items()}, batch)
    actions = _sample_actions(generator, batch, action_dim, device)
    student.eval()
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _teacher_targets(model, context, actions)
            student(context, actions)
        teacher_times = []
        student_times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _teacher_targets(model, context, actions)
            torch.cuda.synchronize()
            teacher_times.append((time.perf_counter() - start) * 1000.0)

            torch.cuda.synchronize()
            start = time.perf_counter()
            student(context, actions)
            torch.cuda.synchronize()
            student_times.append((time.perf_counter() - start) * 1000.0)
    teacher_sorted = sorted(teacher_times)
    student_sorted = sorted(student_times)
    return {
        "batch": batch,
        "horizon": HORIZON,
        "warmup": warmup,
        "repeats": repeats,
        "teacher_ms": teacher_times,
        "student_ms": student_times,
        "teacher_median_ms": teacher_sorted[len(teacher_sorted) // 2],
        "student_median_ms": student_sorted[len(student_sorted) // 2],
        "median_speedup": teacher_sorted[len(teacher_sorted) // 2] / max(student_sorted[len(student_sorted) // 2], 1e-9),
        "median_reduction": 1.0
        - student_sorted[len(student_sorted) // 2]
        / max(teacher_sorted[len(teacher_sorted) // 2], 1e-9),
    }


def _check_official_config(source: Path, freeze: Mapping[str, Any], config_path: Path | None = None) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path or (source / "conf" / "plan_pusht.yaml"))
    official = _official_settings(freeze)
    if cfg.planner._target_ != "planning.mpc.MPCPlanner":
        raise ValueError("official PushT config no longer uses MPCPlanner")
    if cfg.planner.sub_planner.target != "planning.cem.CEMPlanner":
        raise ValueError("official PushT config no longer uses CEMPlanner")
    if int(cfg.goal_H) != int(official["goal_H"]):
        raise ValueError("frozen PushT goal_H does not match official config")
    for key, expected in official["cem_settings"].items():
        if float(cfg.planner.sub_planner[key]) != float(expected):
            raise ValueError(f"frozen CEM setting {key} does not match official config")


def _official_settings(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if "official_pushT" in freeze:
        return dict(freeze["official_pushT"])
    # STAGE_A_FREEZE intentionally omits the full-plan settings because this
    # runner measures only the predictor.  These values are still checked
    # against the official plan_pusht.yaml before target preparation.
    return {
        "seed": int(freeze.get("anchors_and_seeds", {}).get("timing_cem_seed", 99)),
        "goal_source": "dset",
        "goal_H": HORIZON,
        "objective": {"alpha": 1, "base": 2, "mode": "last"},
        "cem_settings": {
            "horizon": HORIZON,
            "topk": 30,
            "num_samples": 300,
            "var_scale": 1,
            "opt_steps": 30,
            "eval_every": 1,
        },
    }


def _load_official(
    root: Path,
    asset_root: Path,
    freeze: Mapping[str, Any],
    output: Path,
    seed: int,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_config: Path | None = None,
    data_root: Path | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any, int, Any]:
    import hydra
    import torch
    from omegaconf import OmegaConf

    source = root / "source"
    shared_cache = root / "idea-validation" / "dino-wm-shared-prefix-cache"
    sys.path.insert(0, str(shared_cache))
    sys.path.insert(0, str(source))
    _check_official_config(source, freeze, config_path=config_path)

    staged_data = (data_root or (asset_root / "data")).resolve()
    dataset_dir = staged_data.parent if staged_data.name == "pusht_noise" else staged_data
    os.environ["DATASET_DIR"] = str(dataset_dir)
    os.environ["WANDB_MODE"] = "disabled"
    import plan
    from env.pusht.pusht_wrapper import PushTWrapper
    from env.serial_vector_env import SerialVectorEnv
    from utils import move_to_device

    if checkpoint_config is not None:
        model_path = checkpoint_config.parent
    elif checkpoint_path is not None:
        model_path = checkpoint_path.parent.parent
    else:
        model_path = asset_root / "checkpoints" / "outputs" / freeze["model"]["model_name"]
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    if model_cfg.env.name != "pusht":
        raise ValueError(f"checkpoint config env must be pusht, got {model_cfg.env.name!r}")
    _, trajectory_datasets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    model = plan.load_model(
        checkpoint_path or (model_path / "checkpoints" / "model_latest.pth"),
        model_cfg,
        model_cfg.num_action_repeat,
        device=torch.device("cuda:0"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    env_kwargs = OmegaConf.to_container(model_cfg.env.kwargs, resolve=True)
    environment = SerialVectorEnv([PushTWrapper(**env_kwargs) for _ in range(ANCHOR_COUNT)])
    official = _official_settings(freeze)
    settings = official["cem_settings"]
    cfg = {
        "seed": int(official["seed"]),
        "n_evals": ANCHOR_COUNT,
        "goal_source": official["goal_source"],
        "goal_H": HORIZON,
        "n_plot_samples": 0,
        "debug_dset_init": False,
        "objective": {"_target_": "planning.objectives.create_objective_fn", **official["objective"]},
        "planner": {"_target_": "planning.cem.CEMPlanner", **settings, "name": "cem"},
        "saved_folder": str(output),
        "wandb_logging": False,
    }
    _set_seed(int(official["seed"]))
    os.chdir(output)
    workspace = plan.PlanWorkspace(
        cfg_dict=cfg,
        wm=model,
        dset=trajectory_datasets["valid"],
        env=environment,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=None,
    )
    transformed_anchors = move_to_device(
        workspace.data_preprocessor.transform_obs(workspace.obs_0), torch.device("cuda:0")
    )
    transformed_goals = move_to_device(
        workspace.data_preprocessor.transform_obs(workspace.obs_g), torch.device("cuda:0")
    )
    with torch.no_grad():
        anchors = model.encode_obs(transformed_anchors)
        goals = model.encode_obs(transformed_goals)
    return model, workspace, anchors, goals, workspace.planner.objective_fn, int(workspace.action_dim), torch.device("cuda:0")


def main() -> int:
    args = _args()
    _require_compute_node()
    if args.steps < 1 or args.batch < 1 or args.eval_batch < 30:
        raise ValueError("steps/batch must be positive and eval_batch must be at least 30")
    root = args.root.resolve()
    asset_root = (args.asset_root or root).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_json(args.freeze.resolve())
    protocol = _load_json(args.protocol.resolve())
    if protocol.get("schema") != "jepa-action-prefix-compiler.stage-a-freeze":
        raise ValueError("unexpected Stage-A protocol schema")
    frozen_training = protocol["training"]
    expected_training = {
        "steps": args.steps,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
    }
    for key, actual in expected_training.items():
        if float(frozen_training[key]) != float(actual):
            raise ValueError(f"runner {key}={actual} does not match frozen protocol")
    frozen_timing = protocol["timing"]
    expected_timing = {
        "batch_size": args.latency_batch,
        "warmup_repeats": args.latency_warmup,
        "technical_repeats": args.latency_repeats,
    }
    for key, actual in expected_timing.items():
        if int(frozen_timing[key]) != int(actual):
            raise ValueError(f"runner {key}={actual} does not match frozen protocol")
    _require_assets(asset_root, freeze)
    _gpu_snapshot(output, "start")

    import torch
    import torch.nn.functional as F

    protocol_seeds = protocol.get("anchors_and_seeds", {})
    training_seed = int(protocol_seeds.get("training_action_prefix_seed", args.seed))
    if training_seed != args.seed:
        raise ValueError("runner seed does not match frozen training_action_prefix_seed")
    training_seeds = [training_seed]
    eval_seeds = [int(value) for value in protocol_seeds.get("heldout_action_prefix_seeds", [])]
    if not eval_seeds:
        eval_seeds = [args.seed + 1001 + index for index in range(args.eval_seeds)]
    _set_seed(args.seed)
    model, workspace, anchors, goals, objective_fn, action_dim, device = _load_official(
        root,
        asset_root,
        freeze,
        output,
        args.seed,
        config_path=args.config.resolve() if args.config else None,
        checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None,
        checkpoint_config=args.checkpoint_config.resolve() if args.checkpoint_config else None,
        data_root=args.data_root.resolve() if args.data_root else None,
    )
    del workspace
    visual_dim = int(anchors["visual"].shape[-1])
    proprio_dim = int(anchors["proprio"].shape[-1])
    student = NativeDinoPrefixStudent(action_dim, visual_dim, proprio_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    train_generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in training_seeds]
    telemetry_period = max(1, args.steps // 10)
    train_losses: list[float] = []

    for step in range(args.steps):
        train_generator = train_generators[step % len(train_generators)]
        indices = torch.randint(0, ANCHOR_COUNT, (args.batch,), generator=train_generator)
        context = _select(anchors, indices).copy()
        actions = _sample_actions(train_generator, args.batch, action_dim, device)
        target = _teacher_targets(model, context, actions)
        prediction = student(context, actions)
        loss = 0.5 * (
            F.mse_loss(prediction["visual"], target["visual"])
            + F.mse_loss(prediction["proprio"], target["proprio"])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % telemetry_period == 0:
            _gpu_snapshot(output, f"train_step_{step + 1}")

    evaluation = _evaluate_student(
        student,
        model,
        anchors,
        goals,
        objective_fn,
        action_dim,
        eval_seeds,
        args.eval_batch,
        device,
    )
    leakage = _leakage_test(
        student,
        anchors,
        action_dim,
        eval_seeds[0] + 9000,
        device,
        args.leakage_tolerance,
    )
    latency = _latency(
        student,
        model,
        anchors,
        action_dim,
        int(protocol_seeds.get("timing_action_prefix_seed", args.seed + 10001)),
        args.latency_batch,
        args.latency_warmup,
        args.latency_repeats,
        device,
    )
    _gpu_snapshot(output, "complete")

    checkpoint = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-stage-a-checkpoint",
        "horizon": HORIZON,
        "native_output_fields": ["visual", "proprio"],
        "action_dim": action_dim,
        "visual_dim": visual_dim,
        "proprio_dim": proprio_dim,
        "hidden_dim": args.hidden_dim,
        "train_seed": args.seed,
        "state_dict": student.state_dict(),
    }
    torch.save(checkpoint, output / "stage_a_student.pt")

    gates = protocol.get("gates", {})
    capacity_gates = gates.get("capacity", {})
    fidelity_gates = gates.get("fidelity", {})
    latency_gates = gates.get("latency", {})
    spearman_threshold = float(fidelity_gates.get("median_objective_spearman_min", 0.99))
    overlap_threshold = float(fidelity_gates.get("median_top30_overlap_min", 0.95))
    spearman_value = float(evaluation["terminal_ranking"]["spearman_mean"])
    overlap_value = float(evaluation["terminal_ranking"]["top30_overlap_mean"])
    loss_ratio = float(
        sorted(train_losses[-10:])[len(train_losses[-10:]) // 2]
        / max(train_losses[0], 1e-12)
    )
    max_loss_ratio = float(
        capacity_gates.get("maximum_last10_to_first_train_loss_ratio", 0.8)
    )
    eval_values = (
        evaluation["per_horizon"]["relative_mse"]
        + evaluation["per_horizon"]["cosine"]
        + [spearman_value, overlap_value]
    )
    heldout_finite = all(math.isfinite(float(value)) for value in eval_values)
    capacity_pass = (
        bool(leakage["passed"])
        and all(math.isfinite(value) for value in train_losses)
        and heldout_finite
        and loss_ratio <= max_loss_ratio
    )
    fidelity_pass = spearman_value >= spearman_threshold and overlap_value >= overlap_threshold
    latency_threshold = float(
        latency_gates.get("minimum_paired_median_predictor_reduction", 0.20)
    )
    predictor_latency_pass = latency["median_reduction"] >= latency_threshold
    summary = {
        "schema": "jepa-action-prefix-compiler.dino-pusht-stage-a-summary",
        "schema_version": 1,
        "model": freeze.get(
            "model",
            {
                "model_name": "pusht",
                "checkpoint": "outputs/pusht/checkpoints/model_latest.pth",
                "dtype": "torch.float32",
            },
        ),
        "source": {
            "reproduction_root": str(root),
            "official_loader": "plan.load_model + PlanWorkspace dset target path",
            "cache_helper": "cache_core.rollout_from_encoded_obs",
            "protocol": str(args.protocol.resolve()),
        },
        "contracts": {
            "anchors": ANCHOR_COUNT,
            "anchor_ids": ["pusht_obs_00", "pusht_obs_01"],
            "horizon": HORIZON,
            "action_input": ["B", HORIZON, action_dim],
            "context_visual": list(anchors["visual"].shape),
            "context_proprio": list(anchors["proprio"].shape),
            "student_visual_output": ["B", HORIZON, int(anchors["visual"].shape[2]), visual_dim],
            "student_proprio_output": ["B", HORIZON, proprio_dim],
            "action_dims_in_student_output": False,
        },
        "train": {
            "steps": args.steps,
            "batch": args.batch,
            "seed": args.seed,
            "eval_seeds": eval_seeds,
            "loss_first": train_losses[0],
            "loss_last": train_losses[-1],
            "loss_median_last_10": sorted(train_losses[-10:])[len(train_losses[-10:]) // 2],
            "last10_to_first_loss_ratio": loss_ratio,
        },
        "future_action_leakage": leakage,
        "evaluation": evaluation,
        "latency": latency,
        "timing_boundary": "predictor-level: cached native observation latent + normalized action prefix through frozen teacher rollout or one student forward; excludes encoder, CEM, environment interaction, and closed-loop execution",
        "gates": {
            "capacity": {
                "status": "PASS" if capacity_pass else "FAIL",
                "future_action_leakage": leakage["passed"],
                "finite_training_trace": all(math.isfinite(value) for value in train_losses),
                "finite_heldout_metrics": heldout_finite,
                "last10_to_first_loss_ratio": loss_ratio,
                "maximum_loss_ratio": max_loss_ratio,
            },
            "fidelity": {
                "status": "PASS" if fidelity_pass else "FAIL",
                "spearman_mean": spearman_value,
                "spearman_minimum": spearman_threshold,
                "top30_overlap_mean": overlap_value,
                "top30_overlap_minimum": overlap_threshold,
            },
            "predictor_latency": {
                "status": "PASS" if predictor_latency_pass else "FAIL",
                "median_reduction": latency["median_reduction"],
                "minimum_reduction": latency_threshold,
                "full_plan_latency": "NOT_EVALUATED",
            },
            "overall": "PASS" if capacity_pass and fidelity_pass and predictor_latency_pass else "FAIL",
        },
        "unverified": [
            "No closed-loop CEM or environment execution is included in Stage-A.",
            "Student quality on LeWM compact-vector latents is not tested by this DINO-only runner.",
            "Teacher/student ranking agreement does not establish closed-loop task success.",
        ],
    }
    summary_text = json.dumps(summary, indent=2, default=_json_default)
    summary_path = (args.summary.resolve() if args.summary else output / "stage_a_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")
    default_summary = output / "stage_a_summary.json"
    if summary_path != default_summary:
        default_summary.write_text(summary_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(default_summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

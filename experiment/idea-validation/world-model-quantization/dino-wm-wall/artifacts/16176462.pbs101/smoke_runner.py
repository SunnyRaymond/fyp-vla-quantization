"""Bounded RankCal/Wall smoke and scoring benchmark.

This runner deliberately keeps the source checkout untouched.  It reuses the
official ``plan.load_model`` and ``PlanWorkspace`` for the closed-loop bridge,
while the fixed-pool scoring path is kept small and deterministic so that it
can also be used by the same-GPU one/two-worker throughput controller.

The quantizer is numerical fake quantization: integer values and scales are
recorded, then dequantized weights are used by the ordinary FP32 operators.
It is therefore an emulation-only smoke test, not a native INT4/INT8 claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# Two workers are intentionally CPU-light.  Set these before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np


SOURCE_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CHECKPOINT_DIR = "outputs/wall_single"
EXPECTED_ENCODER_GROUPS = 12
EXPECTED_PREDICTOR_GROUPS = 6
HORIZON = 5
NUM_SAMPLES = 300
TOPK = 30
CEM_STEPS = 5
FRAMESKIP = 5


def _json_default(value: Any) -> Any:
    """Make audit writing robust to NumPy, Torch and pathlib values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _insert_source(source_dir: Path) -> None:
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def _runtime(root: Path, device: str | None = None):
    """Load the source runtime, checkpoint model and validation dataset."""
    os.environ["DATASET_DIR"] = str(root / "data")
    _insert_source(root / "source")
    import hydra
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for Wall smoke/bench; refusing accidental CPU model execution")
    selected_device = torch.device(device or "cuda:0")
    if selected_device.type != "cuda":
        raise RuntimeError(f"GPU required, got device={selected_device}")
    import plan

    model_path = root / "checkpoints" / CHECKPOINT_DIR
    config_path = model_path / "hydra.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {config_path}")
    model_cfg = OmegaConf.load(config_path)
    checkpoint = _resolve_checkpoint(model_path)

    # This is the same loader used by reproduction/dino-wm-wall/evaluate_wall.py.
    _set_seed(100000)
    _, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dsets["valid"]
    load_capture: Dict[str, Any] = {}
    original_load_ckpt = plan.load_ckpt

    def capture_load_ckpt(snapshot_path, load_device):
        result = original_load_ckpt(snapshot_path, load_device)
        load_capture["epoch"] = int(result["epoch"])
        return result

    plan.load_ckpt = capture_load_ckpt
    try:
        model = plan.load_model(
            checkpoint,
            model_cfg,
            model_cfg.num_action_repeat,
            device=selected_device,
        )
    finally:
        plan.load_ckpt = original_load_ckpt
    model.eval()
    # Decoder is visualization-only and must not enter this smoke scoring path.
    model.decoder = None
    model.eval()
    return {
        "root": root,
        "torch": torch,
        "hydra": hydra,
        "OmegaConf": OmegaConf,
        "plan": plan,
        "model": model,
        "model_cfg": model_cfg,
        "dset": dset,
        "checkpoint": checkpoint,
        "checkpoint_epoch": load_capture.get("epoch"),
        "device": selected_device,
    }


def _resolve_checkpoint(model_path: Path) -> Path:
    preferred = model_path / "checkpoints" / "model_latest.pth"
    if preferred.is_file():
        return preferred
    candidates = list((model_path / "checkpoints").glob("model_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no model checkpoint under {model_path / 'checkpoints'}")

    def key(path: Path) -> Tuple[int, str]:
        stem = path.stem.rsplit("_", 1)[-1]
        return (int(stem) if stem.isdigit() else -1, path.name)

    return sorted(candidates, key=key)[-1]


def _new_wall_env(runtime: Mapping[str, Any], count: int = 1):
    import gym
    from env.serial_vector_env import SerialVectorEnv

    env_cfg = runtime["model_cfg"].env
    envs = [gym.make(env_cfg.name, *env_cfg.args, **env_cfg.kwargs) for _ in range(count)]
    return SerialVectorEnv(envs)


def _close_env(env: Any) -> None:
    for child in getattr(env, "envs", []):
        close = getattr(child, "close", None)
        if callable(close):
            close()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.copy()
    return value.detach().cpu().numpy().copy()


def _target_arrays(obs: Mapping[str, Any], state: Any) -> Dict[str, Any]:
    return {
        "obs": {key: np.expand_dims(_to_numpy(value), axis=1) for key, value in obs.items()},
        "state": _to_numpy(state),
    }


def _make_explicit_targets(runtime: Mapping[str, Any], out: Path) -> List[Dict[str, Any]]:
    """Create two explicit seed/layout targets and validate 25-step replay."""
    dset = runtime["dset"]
    case_specs = [
        {"case_id": 0, "dataset_index": 0, "env_seed": 100000, "cem_seed": 110000},
        {"case_id": 1, "dataset_index": 1, "env_seed": 100001, "cem_seed": 110001},
    ]
    cases: List[Dict[str, Any]] = []
    for spec in case_specs:
        if spec["dataset_index"] >= len(dset):
            raise RuntimeError(f"validation dataset has only {len(dset)} trajectories")
        env = _new_wall_env(runtime, count=1)
        try:
            _, _, _, env_info = dset[spec["dataset_index"]]
            info = env_info
            env.update_env([info])
            init_state, goal_state = env.sample_random_init_goal_states([spec["env_seed"]])
            obs_0, state_0 = env.prepare([spec["env_seed"]], init_state)
            obs_g, state_g = env.prepare([spec["env_seed"]], goal_state)
            target = {
                "obs_0": _target_arrays(obs_0, state_0)["obs"],
                "obs_g": _target_arrays(obs_g, state_g)["obs"],
                "state_0": _to_numpy(state_0),
                "state_g": _to_numpy(state_g),
                "gt_actions": None,
                "goal_H": HORIZON,
            }
            # A fixed non-zero primitive action exercises the actual wall
            # transition and still keeps replay independent of model RNG.
            replay_actions = np.tile(
                np.asarray([[0.2, -0.1]], dtype=np.float32),
                (1, HORIZON * FRAMESKIP, 1),
            )
            first_obs, first_states = env.rollout(
                [spec["env_seed"]], target["state_0"], replay_actions
            )
            second_obs, second_states = env.rollout(
                [spec["env_seed"]], target["state_0"], replay_actions
            )
            replay_equal = bool(
                np.array_equal(first_states, second_states)
                and all(np.array_equal(first_obs[key], second_obs[key]) for key in first_obs)
            )
            if not replay_equal:
                raise RuntimeError(f"deterministic replay failed for case {spec['case_id']}")

            layout = {
                "fix_door_location": int(info["fix_door_location"].item()),
                "fix_wall_location": int(info["fix_wall_location"].item()),
            }
            case_dir = out / f"case_{spec['case_id']:02d}"
            target_path = case_dir / "explicit_targets.pkl"
            _write_pickle(target_path, target)
            replay_path = case_dir / "replay_25steps.npz"
            np.savez_compressed(
                replay_path,
                actions=replay_actions,
                states=first_states,
                visual=first_obs["visual"],
                proprio=first_obs["proprio"],
            )
            cases.append(
                {
                    **spec,
                    "layout": layout,
                    "target_path": str(target_path),
                    "initial_state": target["state_0"].tolist(),
                    "goal_state": target["state_g"].tolist(),
                    "replay_env_steps": int(replay_actions.shape[1]),
                    "replay_deterministic": replay_equal,
                    "replay_path": str(replay_path),
                }
            )
        finally:
            _close_env(env)
    _write_json(out / "episode_manifest.json", {"cases": cases})
    return cases


def _linear_groups(model: Any) -> List[Dict[str, Any]]:
    """Enumerate runtime encoder/predictor block Linear groups and shapes."""
    import torch.nn as nn

    groups: List[Dict[str, Any]] = []
    encoder_blocks = getattr(getattr(model.encoder, "base_model", None), "blocks", None)
    if encoder_blocks is None:
        raise RuntimeError("encoder.base_model.blocks was not found at runtime")
    for index, block in enumerate(encoder_blocks):
        modules = []
        for relative, module in block.named_modules():
            if relative and isinstance(module, nn.Linear):
                modules.append((relative, module))
        if not modules:
            raise RuntimeError(f"encoder block {index} has no Linear weights")
        groups.append(_group_record("encoder", index, modules))

    predictor_layers = getattr(getattr(model.predictor, "transformer", None), "layers", None)
    if predictor_layers is None:
        raise RuntimeError("predictor.transformer.layers was not found at runtime")
    for index, layer in enumerate(predictor_layers):
        modules = []
        for relative, module in layer.named_modules():
            if relative and isinstance(module, nn.Linear):
                modules.append((relative, module))
        if not modules:
            raise RuntimeError(f"predictor layer {index} has no Linear weights")
        groups.append(_group_record("predictor", index, modules))

    enc_count = sum(record["family"] == "encoder" for record in groups)
    pred_count = sum(record["family"] == "predictor" for record in groups)
    if enc_count != EXPECTED_ENCODER_GROUPS or pred_count != EXPECTED_PREDICTOR_GROUPS:
        raise RuntimeError(
            f"runtime group count mismatch: encoder={enc_count}, predictor={pred_count}; "
            f"expected {EXPECTED_ENCODER_GROUPS}/{EXPECTED_PREDICTOR_GROUPS}"
        )
    return groups


def _group_record(family: str, index: int, modules: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    rows = []
    total = 0
    scale_count = 0
    for relative, module in modules:
        shape = list(module.weight.shape)
        numel = int(module.weight.numel())
        total += numel
        scale_count += int(shape[0])
        rows.append(
            {
                "path": f"{family}.block_{index:02d}.{relative}",
                "relative": relative,
                "weight_shape": shape,
                "numel": numel,
                "out_channels": int(shape[0]),
            }
        )
    return {
        "group_id": f"{family}.block_{index:02d}",
        "family": family,
        "index": index,
        "linear_count": len(modules),
        "numel": total,
        "scale_count": scale_count,
        "linear": rows,
        "logical_weight_bytes_W4": int(math.ceil(total * 4 / 8) + scale_count * 4),
        "logical_weight_bytes_W8": int(math.ceil(total * 8 / 8) + scale_count * 4),
    }


def _module_for_path(model: Any, family: str, index: int, relative: str) -> Any:
    block = model.encoder.base_model.blocks[index] if family == "encoder" else model.predictor.transformer.layers[index]
    return dict(block.named_modules())[relative]


def _quantize_group(model: Any, group: Mapping[str, Any], bits: int) -> Dict[str, Any]:
    """Apply symmetric per-output-channel RTN to one group in-place."""
    import torch

    if bits not in (4, 8):
        raise ValueError(bits)
    qmax = 2 ** (bits - 1) - 1
    integer_numel = 0
    scale_numel = 0
    max_abs_scale = 0.0
    with torch.no_grad():
        for linear in group["linear"]:
            module = _module_for_path(model, group["family"], group["index"], linear["relative"])
            weight = module.weight.detach()
            rows = weight.reshape(weight.shape[0], -1)
            max_abs = rows.abs().amax(dim=1, keepdim=True)
            scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
            q = torch.clamp(torch.round(rows / scale), -qmax, qmax)
            dequant = torch.where(max_abs == 0, torch.zeros_like(rows), q * scale)
            module.weight.copy_(dequant.reshape_as(weight))
            integer_numel += int(q.numel())
            scale_numel += int(scale.numel())
            max_abs_scale = max(max_abs_scale, float(scale.max().item()))
    return {
        "bits": bits,
        "integer_numel": integer_numel,
        "scale_numel": scale_numel,
        "max_scale": max_abs_scale,
        "scheme": "symmetric_per_output_channel_RTN",
        "execution": "emulation_only",
    }


def _snapshot_weights(model: Any, groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        f"{group['family']}.{group['index']}.{linear['relative']}": _module_for_path(
            model, group["family"], group["index"], linear["relative"]
        ).weight.detach().clone()
        for group in groups
        for linear in group["linear"]
    }


def _restore_weights(model: Any, snapshot: Mapping[str, Any]) -> None:
    with __import__("torch").no_grad():
        for key, weight in snapshot.items():
            family, index, relative = key.split(".", 2)
            _module_for_path(model, family, int(index), relative).weight.copy_(weight)


def _apply_all_quantized(model: Any, groups: Sequence[Mapping[str, Any]], bits: int) -> List[Dict[str, Any]]:
    return [_quantize_group(model, group, bits) for group in groups]


def _preprocessor(runtime: Mapping[str, Any]):
    from preprocessor import Preprocessor

    dset = runtime["dset"]
    return Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )


def _objective():
    from planning.objectives import create_objective_fn

    return create_objective_fn(alpha=1, base=2, mode="last")


def _score_pool(model: Any, preprocessor: Any, objective_fn: Any, target: Mapping[str, Any], candidates: Any) -> Any:
    """Score a fixed (300, H=5) normalized candidate pool without env calls."""
    import torch

    device = next(model.parameters()).device
    obs_0 = {key: value for key, value in target["obs_0"].items()}
    obs_g = {key: value for key, value in target["obs_g"].items()}
    trans_0 = preprocessor.transform_obs(obs_0)
    trans_g = preprocessor.transform_obs(obs_g)
    trans_0 = {key: value.to(device) for key, value in trans_0.items()}
    trans_g = {key: value.to(device) for key, value in trans_g.items()}
    n = int(candidates.shape[0])
    trans_0 = {
        key: value[0:1].repeat((n,) + (1,) * (value.ndim - 1))
        for key, value in trans_0.items()
    }
    z_goal = model.encode_obs(trans_g)
    z_goal = {
        key: value[0:1].repeat((n,) + (1,) * (value.ndim - 1))
        for key, value in z_goal.items()
    }
    actions = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_obses, _ = model.rollout(obs_0=trans_0, act=actions)
        scores = objective_fn(z_obses, z_goal)
    return scores.detach()


def _pool_record(scores: Any, candidates: np.ndarray) -> Dict[str, Any]:
    import torch

    finite = bool(torch.isfinite(scores).all().item())
    if not finite:
        return {"finite": False, "n": int(len(scores))}
    order = torch.argsort(scores, stable=True)
    elite = order[:TOPK].detach().cpu().numpy().astype(np.int64)
    elite_mean = candidates[elite].mean(axis=0)
    return {
        "finite": True,
        "n": int(len(scores)),
        "topk": TOPK,
        "elite_indices": elite,
        "elite_mean_action": elite_mean,
        "min_score": float(scores.min().item()),
        "mean_score": float(scores.mean().item()),
    }


def _adapter_cem_plan(self: Any, obs_0: Mapping[str, Any], obs_g: Mapping[str, Any], actions: Any = None):
    """Fixed CEM5 adapter: no inner env evaluator and stable candidate ordering."""
    import torch
    from einops import repeat
    from utils import move_to_device

    trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
    trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
    z_obs_g = self.wm.encode_obs(trans_obs_g)
    mu, sigma = self.init_mu_sigma(obs_0, actions)
    mu, sigma = mu.to(self.device), sigma.to(self.device)
    n_evals = mu.shape[0]
    for iteration in range(self.opt_steps):
        losses = []
        for traj in range(n_evals):
            cur_trans_obs_0 = {
                key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                for key, arr in trans_obs_0.items()
            }
            cur_z_obs_g = {
                key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                for key, arr in z_obs_g.items()
            }
            # Match the source CEM RNG stream (CPU randn followed by transfer).
            candidate = torch.randn(
                self.num_samples, self.horizon, self.action_dim
            ).to(self.device) * sigma[traj] + mu[traj]
            candidate[0] = mu[traj]
            with torch.no_grad():
                imagined, _ = self.wm.rollout(obs_0=cur_trans_obs_0, act=candidate)
                loss = self.objective_fn(imagined, cur_z_obs_g)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"non-finite CEM score at iteration {iteration + 1}")
            topk_idx = torch.argsort(loss, stable=True)[: self.topk]
            topk_action = candidate[topk_idx]
            losses.append(float(loss[topk_idx[0]].item()))
            mu[traj] = topk_action.mean(dim=0)
            sigma[traj] = topk_action.std(dim=0)
        self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": iteration + 1})
    return mu, np.full(n_evals, np.inf)


def _install_adapter(runtime: Mapping[str, Any]) -> Any:
    cem = __import__("planning.cem", fromlist=["CEMPlanner"])
    original = cem.CEMPlanner.plan
    cem.CEMPlanner.plan = _adapter_cem_plan
    return original


def _target_for_workspace(case: Mapping[str, Any]) -> Dict[str, Any]:
    with Path(case["target_path"]).open("rb") as stream:
        return pickle.load(stream)


def _run_cem_bridge(workspace: Any, case: Mapping[str, Any], original_cem: Any) -> Dict[str, Any]:
    """Compare adapter/original CEM5 on exactly the same target and RNG stream."""
    import torch

    sub = workspace.planner.sub_planner
    old_evaluator = sub.evaluator
    old_steps = sub.opt_steps
    sub.evaluator = None
    sub.opt_steps = CEM_STEPS
    try:
        _set_seed(case["cem_seed"])
        adapted, _ = _adapter_cem_plan(sub, workspace.obs_0, workspace.obs_g, actions=None)
        _set_seed(case["cem_seed"])
        original_actions, _ = original_cem(sub, workspace.obs_0, workspace.obs_g, actions=None)
        adapted_np = adapted.detach().cpu().numpy()
        original_np = original_actions.detach().cpu().numpy()
        diff = np.abs(adapted_np - original_np)
        return {
            "ran": True,
            "same_inputs": True,
            "cem_iterations": CEM_STEPS,
            "max_abs_action_diff": float(diff.max()),
            "mean_abs_action_diff": float(diff.mean()),
            "exact_equal": bool(np.array_equal(adapted_np, original_np)),
            "evaluator": None,
            "note": "original implementation called with evaluator=None; only stable-sort/adapter behavior is bridged",
        }
    finally:
        sub.evaluator = old_evaluator
        sub.opt_steps = old_steps


def _make_workspace(runtime: Mapping[str, Any], case: Mapping[str, Any], case_dir: Path):
    """Instantiate PlanWorkspace using explicit file targets and bounded MPC."""
    plan = runtime["plan"]
    model_cfg = runtime["model_cfg"]
    cfg = {
        "ckpt_base_path": str(runtime["root"] / "checkpoints"),
        "model_name": "wall_single",
        "model_epoch": "latest",
        "seed": int(case["env_seed"]),
        "n_evals": 1,
        "goal_source": "file",
        "goal_file_path": str(case["target_path"]),
        "goal_H": HORIZON,
        "n_plot_samples": 0,
        "debug_dset_init": False,
        "objective": {
            "_target_": "planning.objectives.create_objective_fn",
            "alpha": 1,
            "base": 2,
            "mode": "last",
        },
        "planner": {
            "_target_": "planning.mpc.MPCPlanner",
            "max_iter": 1,
            "n_taken_actions": HORIZON,
            "sub_planner": {
                "target": "planning.cem.CEMPlanner",
                "horizon": HORIZON,
                "topk": TOPK,
                "num_samples": NUM_SAMPLES,
                "var_scale": 1,
                "opt_steps": CEM_STEPS,
                "eval_every": 1,
            },
            "name": "mpc_cem_adapter",
        },
        "saved_folder": str(case_dir),
        "wandb_logging": False,
    }
    env = _new_wall_env(runtime, count=1)
    import torch

    env.update_env(
        [
            {
                "fix_door_location": torch.tensor(case["layout"]["fix_door_location"]),
                "fix_wall_location": torch.tensor(case["layout"]["fix_wall_location"]),
            }
        ]
    )
    # PlanWorkspace only needs dset statistics in file-target mode; use the same
    # validation split as plan.planning_main.
    workspace = plan.PlanWorkspace(
        cfg_dict=cfg,
        wm=runtime["model"],
        dset=runtime["dset"],
        env=env,
        env_name=model_cfg.env.name,
        frameskip=FRAMESKIP,
        wandb_run=None,
    )
    workspace.eval_seed = [int(case["env_seed"])]
    workspace.evaluator.seed = [int(case["env_seed"])]
    workspace.planner.max_iter = 1
    workspace.planner.sub_planner.opt_steps = CEM_STEPS
    return workspace, env, cfg


def _run_check(root: Path, out: Path, skip_original_bridge: bool = False) -> None:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runtime = _runtime(root)
    model = runtime["model"]
    if next(model.parameters()).dtype != __import__("torch").float32:
        raise RuntimeError("smoke requires FP32 reference model")
    groups = _linear_groups(model)
    _write_json(
        out / "site_manifest.json",
        {
            "schema": "rankcal-wall-site-manifest-v1",
            "execution": "emulation_only",
            "groups": groups,
            "encoder_group_count": EXPECTED_ENCODER_GROUPS,
            "predictor_group_count": EXPECTED_PREDICTOR_GROUPS,
            "quantized_scope": "encoder.base_model.blocks and predictor.transformer.layers Linear weights",
        },
    )
    cases = _make_explicit_targets(runtime, out)
    original_cem = _install_adapter(runtime)
    preprocessor = _preprocessor(runtime)
    objective_fn = _objective()
    snapshot = _snapshot_weights(model, groups)
    workload_cases = []
    shared_arrays: Dict[str, np.ndarray] = {}
    case_audits = []
    all_timing = []
    try:
        for case in cases:
            case_dir = out / f"case_{case['case_id']:02d}"
            target = _target_for_workspace(case)
            generator = __import__("torch").Generator().manual_seed(int(case["cem_seed"]))
            action_dim = int(runtime["dset"].action_dim * FRAMESKIP)
            candidates = __import__("torch").randn(
                NUM_SAMPLES, HORIZON, action_dim, generator=generator, dtype=__import__("torch").float32
            ).numpy()
            candidates[0] = 0.0
            modes: Dict[str, Dict[str, Any]] = {}
            mode_scores: Dict[str, np.ndarray] = {}
            for mode, bits in (("FP32", None), ("NOOP", None), ("all_W4", 4), ("all_W8", 8)):
                _restore_weights(model, snapshot)
                if bits is not None:
                    quant_records = _apply_all_quantized(model, groups, bits)
                else:
                    quant_records = []
                score_started = time.monotonic()
                scores = _score_pool(model, preprocessor, objective_fn, target, candidates)
                if next(model.parameters()).device.type == "cuda":
                    __import__("torch").cuda.synchronize()
                elapsed = time.monotonic() - score_started
                scores_np = scores.detach().cpu().numpy()
                mode_scores[mode] = scores_np
                record = _pool_record(scores, candidates)
                record.update({"mode": mode, "elapsed_seconds": elapsed, "quantization": quant_records})
                modes[mode] = record
            _restore_weights(model, snapshot)
            restored_scores = _score_pool(model, preprocessor, objective_fn, target, candidates)
            restored_np = restored_scores.detach().cpu().numpy()
            restored_exact = bool(np.array_equal(mode_scores["FP32"], restored_np))
            restored_max_abs = float(np.max(np.abs(mode_scores["FP32"] - restored_np)))
            if not restored_exact:
                raise RuntimeError(f"FP32 restore score mismatch for case {case['case_id']}")
            fp32 = mode_scores["FP32"]
            noop = mode_scores["NOOP"]
            noop_exact = bool(np.array_equal(fp32, noop))
            noop_max_abs = float(np.max(np.abs(fp32 - noop)))
            if not noop_exact:
                raise RuntimeError(f"FP32 no-op score mismatch for case {case['case_id']}")
            for mode in ("all_W4", "all_W8"):
                if not np.isfinite(mode_scores[mode]).all():
                    raise FloatingPointError(f"non-finite {mode} score for case {case['case_id']}")

            # Persist compact arrays; latent rollouts are intentionally not kept.
            shared_arrays[f"case_{case['case_id']:02d}_candidates"] = candidates
            for mode, scores in mode_scores.items():
                shared_arrays[f"case_{case['case_id']:02d}_{mode}_scores"] = scores
                shared_arrays[f"case_{case['case_id']:02d}_{mode}_elite_indices"] = modes[mode]["elite_indices"]
                shared_arrays[f"case_{case['case_id']:02d}_{mode}_elite_mean_action"] = modes[mode]["elite_mean_action"]
            workload_cases.append(
                {
                    "case_id": case["case_id"],
                    "env_seed": case["env_seed"],
                    "cem_seed": case["cem_seed"],
                    "layout": case["layout"],
                    "obs_0": target["obs_0"],
                    "obs_g": target["obs_g"],
                    "candidates": candidates,
                    "scores": {mode: scores for mode, scores in mode_scores.items()},
                }
            )

            workspace = None
            env = None
            bridge = {"ran": False, "reason": "skipped"}
            previous_cwd = os.getcwd()
            cem_started = time.monotonic()
            planning_status = "not_started"
            planning_error = None
            try:
                os.chdir(case_dir)
                if not skip_original_bridge:
                    workspace, env, cfg = _make_workspace(runtime, case, case_dir)
                    bridge = _run_cem_bridge(workspace, case, original_cem)
                    _write_json(case_dir / "cem_bridge.json", bridge)
                    _close_env(env)
                    workspace = None
                    env = None

                # The required end-to-end adapter bridge: two cases, one real MPC round.
                workspace, env, cfg = _make_workspace(runtime, case, case_dir)
                _set_seed(case["cem_seed"])
                if __import__("torch").cuda.is_available():
                    __import__("torch").cuda.reset_peak_memory_stats()
                cem_started = time.monotonic()
                try:
                    logs = workspace.perform_planning()
                    planning_status = "complete"
                    planning_error = None
                except Exception as exc:
                    logs = {}
                    planning_status = "failed"
                    planning_error = repr(exc)
                    raise
            finally:
                if __import__("torch").cuda.is_available():
                    __import__("torch").cuda.synchronize()
                elapsed = time.monotonic() - cem_started
                timing = {
                    "case_id": case["case_id"],
                    "stage": "fp32_adapter_cem5_closed_loop",
                    "seconds": elapsed,
                    "planning_status": planning_status,
                    "error": planning_error,
                    "cuda_peak_allocated_gib": float(__import__("torch").cuda.max_memory_allocated() / 2**30)
                    if __import__("torch").cuda.is_available() else None,
                    "cuda_peak_reserved_gib": float(__import__("torch").cuda.max_memory_reserved() / 2**30)
                    if __import__("torch").cuda.is_available() else None,
                }
                all_timing.append(timing)
                _write_json(case_dir / "timing.json", timing)
                with (out / "timing.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(timing, default=_json_default) + "\n")
                os.chdir(previous_cwd)
            _write_json(
                case_dir / "closed_loop.json",
                {
                    "case_id": case["case_id"],
                    "planner": cfg["planner"],
                    "checkpoint_identity": _checkpoint_identity(runtime),
                    "logs": logs,
                    "execution": "emulation_only",
                },
            )
            _close_env(env)
            case_audits.append(
                {
                    **case,
                    "shared_pool": modes,
                    "fp32_noop_exact": noop_exact,
                    "fp32_noop_max_abs_score_diff": noop_max_abs,
                    "fp32_restore_exact": restored_exact,
                    "fp32_restore_max_abs_score_diff": restored_max_abs,
                    "w4_finite": bool(np.isfinite(mode_scores["all_W4"]).all()),
                    "w8_finite": bool(np.isfinite(mode_scores["all_W8"]).all()),
                    "original_cem_bridge": bridge,
                    "closed_loop": "complete",
                }
            )
    finally:
        _restore_weights(model, snapshot)
    np.savez_compressed(out / "shared_pool_scores.npz", **shared_arrays)
    _write_pickle(
        out / "workload.pkl",
        {
            "schema": "rankcal-wall-smoke-workload-v1",
            "execution": "emulation_only",
            "checkpoint_identity": _checkpoint_identity(runtime),
            "planner": {
                "horizon": HORIZON,
                "num_samples": NUM_SAMPLES,
                "topk": TOPK,
                "cem_iterations": CEM_STEPS,
                "frameskip": FRAMESKIP,
                "inner_environment_evaluator": None,
                "stable_argsort": True,
            },
            "cases": workload_cases,
        },
    )
    _write_json(
        out / "quantizer_audit.json",
        {
            "execution": "emulation_only",
            "scheme": "symmetric_per_output_channel_RTN",
            "activation_dtype": "float32",
            "all_eligible_modes": ["all_W4", "all_W8"],
            "restore_fp32_after_each_mode": True,
            "group_count": len(groups),
            "groups": groups,
        },
    )
    _write_json(
        out / "check_audit.json",
        {
            "schema": "rankcal-wall-smoke-audit-v1",
            "status": "complete",
            "execution": "emulation_only",
            "checkpoint_identity": _checkpoint_identity(runtime),
            "case_count": len(case_audits),
            "cases": case_audits,
            "timing": all_timing,
            "elapsed_seconds": time.monotonic() - started,
            "outputs": ["workload.pkl", "shared_pool_scores.npz", "episode_manifest.json", "site_manifest.json"],
        },
    )
    print(json.dumps({"mode": "check", "status": "complete", "output": str(out)}, indent=2), flush=True)


def _checkpoint_identity(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "directory": CHECKPOINT_DIR,
        "checkpoint_file": str(runtime["checkpoint"]),
        "recorded_epoch": runtime.get("checkpoint_epoch", 65),
        "checkpoint_size_bytes": runtime["checkpoint"].stat().st_size,
        "source_commit": SOURCE_COMMIT,
        "dinov2_source_commit": DINOV2_COMMIT,
        "dtype": "float32",
        "decoder": None,
        "execution": "emulation_only",
        "native_memory_claim": False,
        "torch": __import__("torch").__version__,
        "gpu": __import__("torch").cuda.get_device_name() if __import__("torch").cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _wait_for_start(path: Path, timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"start barrier was not released within {timeout_seconds:g}s: {path}")
        time.sleep(0.05)


def _run_bench(root: Path, out: Path, workload_path: Path, repeats: int, warmup: int,
               ready_file: Path | None, start_file: Path | None, worker_id: int,
               workers: int, score_mode: str) -> None:
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    runtime = _runtime(root)
    with workload_path.open("rb") as stream:
        workload = pickle.load(stream)
    cases = workload["cases"]
    if workers < 1 or not 0 <= worker_id < workers:
        raise ValueError(f"invalid worker assignment {worker_id}/{workers}")
    selected = cases[worker_id::workers]
    if not selected:
        raise RuntimeError(f"worker {worker_id} received no workload cases")
    if score_mode not in {"FP32", "NOOP", "all_W4", "all_W8"}:
        raise ValueError(f"unknown score mode {score_mode}")
    model = runtime["model"]
    groups = _linear_groups(model)
    snapshot = _snapshot_weights(model, groups)
    preprocessor = _preprocessor(runtime)
    objective_fn = _objective()
    # Quantize once per worker before the barrier; repeated scoring measures
    # model throughput rather than weight patch/restore overhead.
    _restore_weights(model, snapshot)
    quantization = None
    if score_mode == "all_W4":
        quantization = _apply_all_quantized(model, groups, 4)
    elif score_mode == "all_W8":
        quantization = _apply_all_quantized(model, groups, 8)
    # Warmup is deliberately after the shared barrier and excluded from timing.
    for _ in range(warmup):
        for case in selected:
            scores = _score_pool(model, preprocessor, objective_fn, case, case["candidates"])
            if not bool(__import__("torch").isfinite(scores).all().item()):
                raise FloatingPointError(f"non-finite bench score in case {case['case_id']}")
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.synchronize()
    # Signal readiness only after loading, optional quantization, and warmup.
    if ready_file is not None:
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.touch()
    if start_file is not None:
        _wait_for_start(start_file, timeout_seconds=60.0)
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.reset_peak_memory_stats()
        __import__("torch").cuda.synchronize()
    start_mono = time.monotonic()
    start_wall = time.time()
    repeat_records = []
    for repeat in range(repeats):
        scores_record = []
        for case in selected:
            scores = _score_pool(model, preprocessor, objective_fn, case, case["candidates"])
            if not bool(__import__("torch").isfinite(scores).all().item()):
                raise FloatingPointError(f"non-finite bench score in case {case['case_id']}")
            scores_record.append({
                "case_id": int(case["case_id"]),
                "scores": scores.detach().cpu().tolist(),
                "score_min": float(scores.min().item()),
                "score_mean": float(scores.mean().item()),
            })
        repeat_records.append({"repeat": repeat, "cases": scores_record})
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.synchronize()
    end_mono = time.monotonic()
    end_wall = time.time()
    elapsed = end_mono - start_mono
    _restore_weights(model, snapshot)
    record = {
        "schema": "rankcal-wall-bench-v1",
        "status": "complete",
        "execution": workload.get("execution", "emulation_only"),
        "checkpoint_identity": workload.get("checkpoint_identity"),
        "worker_id": worker_id,
        "workers": workers,
        "score_mode": score_mode,
        "quantization": quantization,
        "warmup": warmup,
        "repeats": repeats,
        "case_ids": [int(case["case_id"]) for case in selected],
        "records": len(selected) * repeats,
        "start_monotonic": start_mono,
        "end_monotonic": end_mono,
        "start_wall_epoch": start_wall,
        "end_wall_epoch": end_wall,
        "elapsed_seconds": elapsed,
        "records_per_second": (len(selected) * repeats / elapsed) if elapsed > 0 else None,
        "cuda_peak_allocated_gib": float(__import__("torch").cuda.max_memory_allocated() / 2**30)
        if __import__("torch").cuda.is_available() else None,
        "cuda_peak_reserved_gib": float(__import__("torch").cuda.max_memory_reserved() / 2**30)
        if __import__("torch").cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scores": repeat_records,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / f"bench_worker_{worker_id}.json", record)
    _write_json(out / "bench.json", record)
    print(json.dumps({k: record[k] for k in ("status", "worker_id", "workers", "records", "elapsed_seconds", "records_per_second")}, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DINO-WM Wall runtime root")
    parser.add_argument("--output", type=Path, required=True, help="artifact directory")
    parser.add_argument("--mode", choices=("check", "bench"), required=True)
    parser.add_argument("--workload", type=Path, default=None, help="workload.pkl for bench")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--start-file", type=Path, default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--score-mode", "--bench-mode", dest="score_mode", default="FP32")
    parser.add_argument("--skip-original-bridge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root
    out = args.output
    if args.mode == "check":
        _run_check(root, out, skip_original_bridge=args.skip_original_bridge)
    else:
        workload = args.workload or (out / "workload.pkl")
        if not workload.is_file():
            raise FileNotFoundError(f"bench workload not found: {workload}")
        _run_bench(
            root,
            out,
            workload,
            repeats=args.repeats,
            warmup=args.warmup,
            ready_file=args.ready_file,
            start_file=args.start_file,
            worker_id=args.worker_id,
            workers=args.workers,
            score_mode=args.score_mode,
        )


if __name__ == "__main__":
    main()

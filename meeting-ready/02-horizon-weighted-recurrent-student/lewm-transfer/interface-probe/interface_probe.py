#!/usr/bin/env python3
"""Bounded GPU-only probe of the official LeWM PushT rollout interface.

This intentionally calls the already staged official checkpoint and official
``JEPA.rollout``/``predict`` methods without monkeypatching model behavior.
Only small interface metadata and shape results are written to ``--output``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--student-freeze", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing model work outside PBS")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def shape_dtype(value: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"type": type(value).__name__}
    if hasattr(value, "shape"):
        item["shape"] = [int(x) for x in value.shape]
    if hasattr(value, "dtype"):
        item["dtype"] = str(value.dtype)
    if hasattr(value, "device"):
        item["device"] = str(value.device)
    return item


def config_dict(value: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raw = getattr(value, "__dict__", {})
    return {str(k): json_value(v) for k, v in raw.items() if not str(k).startswith("_")}


def json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_checkpoint(stable_home: Path) -> Any:
    import torch

    checkpoint = stable_home / "pusht" / "lewm_object.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not hasattr(model, "rollout") or not hasattr(model, "predict"):
        raise TypeError(f"checkpoint is not an official JEPA-like model: {checkpoint}")
    return model


def move_info(value: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_info(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_info(item, device) for item in value]
    return copy.deepcopy(value)


def build_official_cem_info(info: dict[str, Any], samples: int, device: Any) -> dict[str, Any]:
    """Reproduce CEMSolver's candidate-axis expansion before ``get_cost``.

    ``JEPA.rollout`` receives this expanded view from the official solver; it
    does not itself expand the caller's ``info`` dictionary.
    """
    import torch

    expanded: dict[str, Any] = {}
    for key, value in info.items():
        if torch.is_tensor(value) and value.ndim >= 1:
            expanded[key] = (
                value.to(device=device)
                .unsqueeze(1)
                .expand(value.shape[0], samples, *value.shape[1:])
                .clone()
            )
        else:
            expanded[key] = copy.deepcopy(value)
    return expanded


def predict_probe(model: Any, initial_info: dict[str, Any], candidates: Any) -> dict[str, Any]:
    import torch

    pixels = initial_info["pixels"]
    h = int(pixels.shape[2])
    # Encode one real history once, then exercise only the official predict call
    # at sequence lengths 1/2/3.  No model method is replaced or patched.
    encoded = model.encode(
        {
            "pixels": pixels[:, 0],
            "action": candidates[:, 0, :h, :],
        }
    )
    latent = encoded["emb"]
    base_latent = latent[:, -1:, :]
    rows: dict[str, Any] = {}
    for length in (1, 2, 3):
        key = str(length)
        try:
            # Use a repeated real encoded latent when the policy reset has
            # H=1; this isolates sequence-length acceptance from policy history.
            emb = base_latent.expand(-1, length, -1).clone()
            raw_actions = candidates[:, 0, :1, :].expand(-1, length, -1).clone()
            act_emb = model.action_encoder(raw_actions)
            output = model.predict(emb, act_emb)
            rows[key] = {
                "accepted": True,
                "input_emb_shape": list(emb.shape),
                "input_action_shape": list(raw_actions.shape),
                "action_embedding_shape": list(act_emb.shape),
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
            }
        except Exception as exc:  # interface probe records, rather than hides, a rejection
            rows[key] = {
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
    return {"latent_history_shape": list(latent.shape), "lengths": rows}


def rolling_five_targets_checked(
    model: Any, latent: Any, history_actions: Any, future_actions: Any
) -> dict[str, Any]:
    """Trace five free-running predictor calls with aligned latest-3 windows."""
    import torch

    history_size = 3
    history = latent[:, -history_size:, :].clone()
    action_history = history_actions[:, -history_size:, :].clone()
    outputs: list[Any] = []
    try:
        with torch.no_grad():
            for step in range(5):
                current_action = future_actions[:, step : step + 1, :]
                action_history = torch.cat([action_history, current_action], dim=1)
                action_window = action_history[:, -history_size:, :]
                pred = model.predict(
                    history, model.action_encoder(action_window)
                )[:, -1:, :]
                outputs.append(pred)
                history = torch.cat([history[:, 1:, :], pred], dim=1)
        stacked = torch.cat(outputs, dim=1)
        return {
            "accepted": True,
            "future_target_count": len(outputs),
            "future_target_shapes": [list(item.shape) for item in outputs],
            "stacked_shape": list(stacked.shape),
            "latent_feedback": "student prediction is appended; no teacher forcing",
            "history_action_shape": list(history_actions.shape),
            "future_action_shape": list(future_actions.shape),
            "history_size": history_size,
            "error": None,
        }
    except Exception as exc:
        return {
            "accepted": False,
            "future_target_count": len(outputs),
            "future_target_shapes": [list(item.shape) for item in outputs],
            "stacked_shape": None,
            "latent_feedback": "student prediction is appended; no teacher forcing",
            "history_action_shape": list(history_actions.shape),
            "future_action_shape": list(future_actions.shape),
            "history_size": history_size,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    args = parse_args()
    require_compute_node()
    freeze = load_json(args.freeze)
    student_freeze = load_json(args.student_freeze)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["STABLEWM_HOME"] = str(args.stablewm_home.resolve())
    sys.path.insert(0, str(args.lewm_root.resolve()))
    sys.path.insert(0, str(args.freeze.parent.resolve()))

    import torch
    from stable_worldmodel.policy import PlanConfig

    # Reuse the verified official policy-preprocessing harness.
    from run_lewm_pusht_iteration import prepare_policy_info

    model = load_checkpoint(args.stablewm_home.resolve()).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    observations, action_space = prepare_policy_info(args.stablewm_home.resolve(), 1)
    observation_id, info_cpu = observations[0]
    info = move_info(info_cpu, torch.device("cuda"))
    info_shapes = {key: shape_dtype(value) for key, value in info_cpu.items()}
    plan_cfg = PlanConfig(**freeze["official_pusht"]["plan_config"])
    solver_settings = dict(freeze["official_pusht"]["cem_solver"])
    batch = int(solver_settings["batch_size"])
    samples = 2
    horizon = int(plan_cfg.horizon)
    action_dim = int(math.prod(action_space.shape[1:])) * int(plan_cfg.action_block)
    candidates = torch.zeros(
        (batch, samples, horizon, action_dim), dtype=torch.float32, device="cuda"
    )
    cem_info = build_official_cem_info(info, samples, torch.device("cuda"))
    # JEPA.rollout defines H on the solver-expanded layout (B,S,H,C,W).
    pixels_history = int(cem_info["pixels"].shape[2])

    rollout_trace: dict[str, Any] = {}
    original_rollout = model.rollout

    def traced_rollout(info_arg: dict[str, Any], action_arg: Any, history_size: int = 3) -> Any:
        result = original_rollout(info_arg, action_arg, history_size=history_size)
        rollout_trace.update(
            {
                "H": int(info_arg["pixels"].shape[2]),
                "T": int(action_arg.shape[2]),
                "n_steps": int(action_arg.shape[2] - info_arg["pixels"].shape[2]),
                "history_size": int(history_size),
                "predicted_emb_shape": list(result["predicted_emb"].shape),
                "predicted_emb_dtype": str(result["predicted_emb"].dtype),
            }
        )
        return result

    with torch.no_grad():
        model.rollout = traced_rollout
        try:
            # This is the official JEPA.get_cost path.  It calls JEPA.rollout
            # with the CEMSolver-expanded info and the fixed candidate tensor.
            costs = model.get_cost(cem_info, candidates)
        finally:
            model.rollout = original_rollout
        direct_predict = predict_probe(model, cem_info, candidates)
        encoded = model.encode(
            {
                "pixels": cem_info["pixels"][:, 0],
                "action": candidates[:, 0, :pixels_history, :],
            }
        )
        latent_seed = encoded["emb"]
        rolling_history_was_padded = int(latent_seed.shape[1]) < 3
        if rolling_history_was_padded:
            latent_seed = latent_seed[:, -1:, :].expand(-1, 3, -1).clone()
        history_actions = candidates[:, 0, :pixels_history, :]
        if history_actions.shape[1] < 3:
            history_actions = history_actions[:, -1:, :].expand(-1, 3, -1).clone()
        future_actions = torch.zeros(
            (batch, int(student_freeze["scope"]["horizon"]), action_dim),
            dtype=torch.float32,
            device="cuda",
        )
        five = rolling_five_targets_checked(
            model,
            latent_seed,
            history_actions,
            future_actions,
        )
        five["latent_history_seed_shape"] = list(latent_seed.shape)
        five["history_padding_from_policy_H"] = rolling_history_was_padded

    result = {
        "schema": "lewm-recurrent-student.interface-probe",
        "schema_version": 1,
        "status": "PASS" if five["accepted"] else "INTERFACE_ERROR",
        "job_id": os.environ.get("PBS_JOBID"),
        "observation_id": observation_id,
        "checkpoint": "pusht/lewm_object.ckpt",
        "prepare_policy_info": {
            "keys": sorted(info_shapes),
            "values": info_shapes,
            "pixels_history_H": pixels_history,
            "observation_count": len(observations),
        },
        "plan_config": {
            "class": "stable_worldmodel.policy.PlanConfig",
            "fields": config_dict(plan_cfg),
            "plan_len": int(plan_cfg.plan_len),
        },
        "solver": {
            "class": "stable_worldmodel.solver.CEMSolver",
            "frozen_settings": solver_settings,
            "action_space_shape": list(action_space.shape),
            "raw_action_dim": int(math.prod(action_space.shape[1:])),
        },
        "cem_candidate": {
            "shape": list(candidates.shape),
            "dtype": str(candidates.dtype),
            "packed_action": "each 10-D token packs action_block=5 consecutive raw 2-D PushT actions as [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]",
            "candidate_axis": "B=batch, S=num_samples, T=horizon, 10=packed action token",
        },
        "official_jepa_rollout": {
            "candidate_shape": list(candidates.shape),
            **rollout_trace,
            "future_predictions_beyond_initial_H": int(rollout_trace["predicted_emb_shape"][2] - rollout_trace["H"]),
            "cost_shape": list(costs.shape),
            "semantic_note": "trace captured from official JEPA.get_cost -> JEPA.rollout; CEMSolver supplies the candidate-axis-expanded info",
        },
        "model_predict": direct_predict,
        "rolling_latest3_five_targets": five,
        "training_and_stage_b_mapping": {
            "training_target_horizon_steps": int(student_freeze["scope"]["horizon"]),
            "reason": "frozen student protocol target is z(t+1)..z(t+5); observed official rollout produces future_predictions_beyond_initial_H from its T-H loops plus one final predict",
            "stage_b_wrapper_contract": "wrap student.get_cost-compatible predictor so it receives official candidate [B,S,5,10], preserves official info/goal criterion and CEM, and returns predicted_emb with the exact official predicted_emb shape observed in this probe",
            "semantics_equal_to_official_cem_rollout": bool(
                five["future_target_count"] == rollout_trace["predicted_emb_shape"][2] - rollout_trace["H"]
            ),
            "semantics_boundary": "rolling latest-3 latent + one 10-D action produced five future targets here; compare its count and shape to the traced official CEM rollout before claiming equivalence",
        },
        "runtime": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    output_path = args.output / "interface_probe.json"
    output_path.write_text(json.dumps(json_value(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_value(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

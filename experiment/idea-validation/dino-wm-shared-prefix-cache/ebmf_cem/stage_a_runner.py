"""Stage A calibration runner for EBMF-CEM.

This runner implements only the frozen calibration stage.  It evaluates the
same candidate tensor with ``cheap_L4`` and the complete six-layer predictor,
and updates CEM only from the full objective.  No held-out, system-timing, or
closed-loop stage is included here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


CANDIDATES = 300
HORIZON = 5
TOPK = 30
OPT_STEPS = 10
CALIBRATION_CASES = 6
CASE_IDS = tuple(f"wall_case_{index:02d}" for index in range(CALIBRATION_CASES))
CHEAP_LAYERS = 4
FULL_LAYERS = 6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-observations", type=int, default=CALIBRATION_CASES)
    return parser.parse_args()


def _guard_compute_node() -> None:
    """Fail closed before model/data access unless running in PBS allocation."""

    if not os.environ.get("PBS_JOBID", "").strip():
        raise RuntimeError("PBS_JOBID is empty; EBMF-CEM calibration requires PBS")
    host = socket.gethostname().lower()
    if any(token in host for token in ("login", "asp2a-login", "ntu-login")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, default=_json_default, allow_nan=False) + "\n"
            )


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _runtime(root: Path) -> Dict[str, Any]:
    """Load the existing DINO-WM runtime; all calls happen on a PBS node."""

    os.environ["DATASET_DIR"] = str(root / "data")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["PYTHONUNBUFFERED"] = "1"
    source = root / "source"
    sys.path.insert(0, str(source))

    import gym
    import hydra
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; refusing CPU EBMF-CEM calibration")

    checkpoint_dir = root / "checkpoints" / "outputs" / "wall_single"
    config_path = checkpoint_dir / "hydra.yaml"
    checkpoint_path = checkpoint_dir / "checkpoints" / "model_latest.pth"
    for required in (root / "data", config_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(required)

    model_cfg = OmegaConf.load(config_path)
    _, trajectory_datasets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    plan = importlib.import_module("plan")
    model = plan.load_model(
        checkpoint_path,
        model_cfg,
        model_cfg.num_action_repeat,
        device=torch.device("cuda:0"),
    )
    model.eval()
    model.decoder = None

    predictor = model.predictor
    layers = getattr(getattr(predictor, "transformer", None), "layers", None)
    if layers is None or len(layers) != FULL_LAYERS:
        raise RuntimeError(
            f"expected a six-layer ViT predictor, found {None if layers is None else len(layers)}"
        )

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    return {
        "torch": torch,
        "model": model,
        "model_cfg": model_cfg,
        "dset": trajectory_datasets["valid"],
        "env": env,
        "checkpoint": checkpoint_path,
        "predictor_layers": len(layers),
    }


def _make_targets(runtime: Mapping[str, Any]) -> List[Dict[str, Any]]:
    dset = runtime["dset"]
    env = runtime["env"]
    targets: List[Dict[str, Any]] = []
    for case_index, observation_id in enumerate(CASE_IDS):
        if case_index >= len(dset):
            raise RuntimeError(f"validation dataset has only {len(dset)} trajectories")
        _, _, _, env_info = dset[case_index]
        env.update_env(env_info)
        env_seed = 100000 + case_index
        init_state, goal_state = env.sample_random_init_goal_states(env_seed)
        obs_0, state_0 = env.prepare(env_seed, init_state)
        obs_g, state_g = env.prepare(env_seed, goal_state)
        obs_0 = {key: _to_numpy(value)[None, None, ...] for key, value in obs_0.items()}
        obs_g = {key: _to_numpy(value)[None, None, ...] for key, value in obs_g.items()}
        targets.append(
            {
                "observation_id": observation_id,
                "observation_seed": env_seed,
                "obs_0": obs_0,
                "obs_g": obs_g,
                "state_0": _to_numpy(state_0).tolist(),
                "state_g": _to_numpy(state_g).tolist(),
            }
        )
    return targets


def _make_preprocessor(runtime: Mapping[str, Any]) -> Any:
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


def _expand_prefix(prefix: Mapping[str, Any], repeats: int) -> Dict[str, Any]:
    return {
        key: value[:1].expand((repeats,) + tuple(value.shape[1:]))
        for key, value in prefix.items()
    }


def _encode_from_prefix(model: Any, prefix: Mapping[str, Any], act_0: Any) -> Any:
    """Match VWorldModel.encode after its observation prefix is encoded."""

    import torch

    act_emb = model.encode_act(act_0)
    visual = prefix["visual"]
    proprio = prefix["proprio"]
    if model.concat_dim == 0:
        return torch.cat(
            [visual, proprio.unsqueeze(2), act_emb.unsqueeze(2)], dim=2
        )
    if model.concat_dim == 1:
        num_patches = visual.shape[2]
        proprio_tiled = proprio.unsqueeze(2).expand(
            -1, -1, num_patches, -1
        )
        proprio_repeated = proprio_tiled.repeat(
            1, 1, 1, model.num_proprio_repeat
        )
        act_tiled = act_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
        act_repeated = act_tiled.repeat(1, 1, 1, model.num_action_repeat)
        return torch.cat([visual, proprio_repeated, act_repeated], dim=3)
    raise ValueError(f"unsupported model concat_dim={model.concat_dim!r}")


def _predict_full(model: Any, z: Any) -> Any:
    return model.predict(z)


def _predict_cheap_l4(model: Any, z: Any) -> Any:
    """Run pos embedding, dropout, first four blocks, and final LayerNorm."""

    predictor = model.predictor
    batch, time, patches, dim = z.shape
    x = z.reshape(batch, time * patches, dim)
    x = x + predictor.pos_embedding[:, : x.shape[1]]
    x = predictor.dropout(x)
    for attention, feed_forward in predictor.transformer.layers[:CHEAP_LAYERS]:
        x = attention(x) + x
        x = feed_forward(x) + x
    x = predictor.transformer.norm(x)
    return x.reshape(batch, time, patches, dim)


def _rollout_from_common_prefix(
    model: Any,
    encoded_prefix: Mapping[str, Any],
    actions: Any,
    *,
    cheap: bool,
) -> Dict[str, Any]:
    """Roll out from a common prefix with an independent autoregressive state."""

    import torch

    visual = encoded_prefix["visual"]
    num_obs_init = visual.shape[1]
    if actions.shape[1] < num_obs_init:
        raise ValueError("candidate actions must include initial observation actions")
    act_0 = actions[:, :num_obs_init]
    action_suffix = actions[:, num_obs_init:]
    z = _encode_from_prefix(model, encoded_prefix, act_0)
    predictor = _predict_cheap_l4 if cheap else _predict_full

    time_index = 0
    increment = 1
    while time_index < action_suffix.shape[1]:
        z_pred = predictor(model, z[:, -model.num_hist :])
        z_new = z_pred[:, -increment:, ...]
        z_new = model.replace_actions_from_z(
            z_new,
            action_suffix[:, time_index : time_index + increment, :],
        )
        z = torch.cat([z, z_new], dim=1)
        time_index += increment

    z_pred = predictor(model, z[:, -model.num_hist :])
    z_new = z_pred[:, -1:, ...]
    z = torch.cat([z, z_new], dim=1)
    z_obses, _ = model.separate_emb(z)
    return z_obses


def _stable_topk(values: Any, k: int) -> List[int]:
    flat = values.detach().cpu().tolist()
    return sorted(range(len(flat)), key=lambda index: (float(flat[index]), index))[:k]


def _round_pair(
    runtime: Mapping[str, Any],
    objective_fn: Any,
    common_prefix: Mapping[str, Any],
    goal_z: Mapping[str, Any],
    mu: Any,
    sigma: Any,
    observation_index: int,
    round_index: int,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Evaluate one shared candidate tensor and return the full-CEM update."""

    import torch

    model = runtime["model"]
    torch_module = runtime["torch"]
    action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
    candidate_seed = 1200000 + observation_index * 100 + round_index
    # The candidate population is the CEM noise schedule for this paired
    # round; record one seed for both names rather than inventing two streams.
    noise_seed = candidate_seed
    _set_seed(torch_module, noise_seed)
    # Match the existing CEMPlanner sampling order: draw on CPU, then move
    # the shared population to the model device.
    noise = torch.randn(CANDIDATES, HORIZON, action_dim)
    actions = noise.to(mu.device) * sigma + mu
    actions[0] = mu

    with torch.no_grad():
        # Cheap and full each construct their own rollout z from the shared
        # factorized_full prefix; neither reuses the other's activation.
        cheap_obs = _rollout_from_common_prefix(
            model, common_prefix, actions, cheap=True
        )
        full_obs = _rollout_from_common_prefix(
            model, common_prefix, actions, cheap=False
        )
        cheap_objective = objective_fn(cheap_obs, goal_z)
        full_objective = objective_fn(full_obs, goal_z)

    cheap_values = cheap_objective.detach().cpu().tolist()
    full_values = full_objective.detach().cpu().tolist()
    abs_error = [abs(float(a) - float(b)) for a, b in zip(cheap_values, full_values)]
    cheap_top30 = _stable_topk(cheap_objective, TOPK)
    full_top30 = _stable_topk(full_objective, TOPK)
    full_cutoff = float(full_values[full_top30[-1]])
    topk_actions = actions[full_top30]
    next_mu = topk_actions.mean(dim=0)
    next_sigma = topk_actions.std(dim=0)

    record = {
        "round_index": round_index,
        "candidate_population_seed": candidate_seed,
        "cem_noise_seed": noise_seed,
        "input_mu": mu.detach().cpu().tolist(),
        "input_sigma": sigma.detach().cpu().tolist(),
        "output_mu": next_mu.detach().cpu().tolist(),
        "output_sigma": next_sigma.detach().cpu().tolist(),
        "first_action": next_mu[0].detach().cpu().tolist(),
        "J_tilde": cheap_values,
        "J_full": full_values,
        "abs_error": abs_error,
        "cheap_top30": cheap_top30,
        "full_top30": full_top30,
        "full_cutoff": full_cutoff,
    }
    return record, next_mu, next_sigma


def _calibrate_observation(
    runtime: Mapping[str, Any],
    preprocessor: Any,
    objective_fn: Any,
    target: Mapping[str, Any],
    observation_index: int,
) -> Dict[str, Any]:
    import torch

    action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
    mu = torch.zeros(1, HORIZON, action_dim, device="cuda:0")
    sigma = torch.ones_like(mu)
    trans_obs_0 = preprocessor.transform_obs(target["obs_0"])
    trans_obs_g = preprocessor.transform_obs(target["obs_g"])
    trans_obs_0 = {key: value.to(mu.device) for key, value in trans_obs_0.items()}
    trans_obs_g = {key: value.to(mu.device) for key, value in trans_obs_g.items()}
    with torch.no_grad():
        common_prefix = _expand_prefix(
            runtime["model"].encode_obs(trans_obs_0), CANDIDATES
        )
        goal_z = _expand_prefix(runtime["model"].encode_obs(trans_obs_g), CANDIDATES)
    rounds: List[Dict[str, Any]] = []
    for round_index in range(OPT_STEPS):
        round_record, next_mu, next_sigma = _round_pair(
            runtime,
            objective_fn,
            common_prefix,
            goal_z,
            mu[0],
            sigma[0],
            observation_index,
            round_index,
        )
        rounds.append(round_record)
        mu = next_mu.unsqueeze(0)
        sigma = next_sigma.unsqueeze(0)
    return {
        "schema": "dino-wm-shared-prefix-cache-ebmf-cem-stage-a-v1",
        "observation_id": target["observation_id"],
        "observation_seed": target["observation_seed"],
        "candidate_count": CANDIDATES,
        "horizon": HORIZON,
        "topk": TOPK,
        "cem_opt_steps": OPT_STEPS,
        "cheap_evaluator": {
            "name": "cheap_L4",
            "layer_count": CHEAP_LAYERS,
            "total_predictor_layers": FULL_LAYERS,
            "includes_pos_embedding": True,
            "includes_dropout": True,
            "includes_final_layer_norm": True,
        },
        "reference_path": "factorized_full",
        "rounds": rounds,
    }


def _stable_upper_cutoff(values: Sequence[float], k: int) -> Tuple[float, List[int]]:
    ranked = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    top = ranked[:k]
    return float(values[top[-1]]), top


def _add_intervals(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    epsilon_by_round: List[float] = []
    for round_index in range(OPT_STEPS):
        epsilon_by_round.append(
            max(
                max(float(error) for error in record["rounds"][round_index]["abs_error"])
                for record in records
            )
        )

    ratio_values: List[float] = []
    coverage_values: List[bool] = []
    for record in records:
        for round_index, round_record in enumerate(record["rounds"]):
            epsilon = epsilon_by_round[round_index]
            cheap = [float(value) for value in round_record["J_tilde"]]
            full = [float(value) for value in round_record["J_full"]]
            lower = [value - epsilon for value in cheap]
            upper = [value + epsilon for value in cheap]
            tau, _ = _stable_upper_cutoff(upper, TOPK)
            ambiguous = [index for index, value in enumerate(lower) if value <= tau]
            full_top30 = [int(index) for index in round_record["full_top30"]]
            ambiguous_set = set(ambiguous)
            contains = all(index in ambiguous_set for index in full_top30)
            round_record.update(
                {
                    "epsilon_round": epsilon,
                    "epsilon_source": "calibration_max_abs_error",
                    "L": lower,
                    "U": upper,
                    "tau": tau,
                    "A": ambiguous,
                    "ambiguity_ratio": len(ambiguous) / CANDIDATES,
                    "full_top30_in_A": contains,
                    "recomputed_abs_error_max": max(
                        abs(cheap[index] - full[index])
                        for index in range(CANDIDATES)
                    ),
                }
            )
            ratio_values.append(len(ambiguous) / CANDIDATES)
            coverage_values.append(contains)

    sorted_ratios = sorted(ratio_values)
    median = (sorted_ratios[29] + sorted_ratios[30]) / 2.0
    p90 = sorted_ratios[math.ceil(0.90 * len(sorted_ratios)) - 1]
    return {
        "epsilon_by_round": epsilon_by_round,
        "ambiguity": {
            "record_count": len(ratio_values),
            "ratios": ratio_values,
            "median": median,
            "p90_nearest_rank": p90,
            "median_max": 0.25,
            "p90_max": 0.40,
            "gate_pass": median <= 0.25 and p90 <= 0.40,
        },
        "coverage": {
            "record_count": len(coverage_values),
            "all_calibration_records_contained": all(coverage_values),
            "positive_record_count": sum(coverage_values),
        },
    }


def _run(args: argparse.Namespace) -> int:
    _guard_compute_node()
    if args.n_observations != CALIBRATION_CASES:
        raise ValueError("frozen Stage A requires exactly six calibration observations")

    import torch
    from planning.objectives import create_objective_fn

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(root)
    preprocessor = _make_preprocessor(runtime)
    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    targets = _make_targets(runtime)
    records = [
        _calibrate_observation(
            runtime,
            preprocessor,
            objective_fn,
            target,
            observation_index,
        )
        for observation_index, target in enumerate(targets)
    ]
    interval_summary = _add_intervals(records)
    _write_jsonl(output / "calibration.jsonl", records)

    summary = {
        "schema": "dino-wm-shared-prefix-cache-ebmf-cem-stage-a-summary-v1",
        "schema_version": 1,
        "status": "complete",
        "stage": "stage_a_calibration",
        "claim_boundary": "calibration/readiness evidence only; no held-out, decision, latency, native-system, or closed-loop claim",
        "method": "Elite-Band Multi-Fidelity CEM (EBMF-CEM)",
        "reference_path": "factorized_full",
        "observations": list(CASE_IDS),
        "candidate_count": CANDIDATES,
        "horizon": HORIZON,
        "topk": TOPK,
        "cem_opt_steps": OPT_STEPS,
        "cheap_evaluator": {
            "name": "cheap_L4",
            "layer_count": CHEAP_LAYERS,
            "total_predictor_layers": FULL_LAYERS,
            "construction": "ViTPredictor pos embedding + dropout + first four Transformer blocks + existing final LayerNorm",
            "full_reevaluation_from_common_prefix": True,
            "cheap_autoregressive_activation_reused_by_full": False,
        },
        "runtime_identity": {
            "checkpoint": str(runtime["checkpoint"]),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "predictor_layers": runtime["predictor_layers"],
        },
        "calibration": interval_summary,
        "readiness": {
            "completeness_finite": "runner_complete",
            "cheap_identity": "cheap_L4",
            "epsilon_provenance": "calibration_max_abs_error",
            "stage_b_authorized": False,
            "stage_b_authorization_note": "Stage A output is not an independent authorization; verifier and the frozen contract must decide readiness.",
        },
        "files": {
            "calibration": "calibration.jsonl",
            "summary": "stage_a_summary.json",
        },
        "held_out": "NOT_RUN",
        "system_timing": "NOT_RUN",
        "closed_loop": "NOT_RUN",
    }
    _write_json(output / "stage_a_summary.json", summary)
    runtime["env"].close()
    return 0


def main() -> int:
    try:
        return _run(_parse_args())
    except Exception as exc:
        if "--output" in sys.argv:
            try:
                output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
                output.mkdir(parents=True, exist_ok=True)
                _write_json(
                    output / "stage_a_summary.json",
                    {
                        "schema": "dino-wm-shared-prefix-cache-ebmf-cem-stage-a-summary-v1",
                        "status": "failed",
                        "error": repr(exc),
                    },
                )
            except Exception:
                pass
        print(f"EBMF_CEM_STAGE_A_FAILED: {exc!r}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

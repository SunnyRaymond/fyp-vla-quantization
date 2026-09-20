"""Rank Student V2: candidate-specific context/action cross-attention screen.

This is an offline teacher/student experiment.  The full six-layer DINO-WM
teacher alone performs the chained CEM update.  V2 keeps the 7x7 spatial token
grid after fixed 2x2 pooling and scores each candidate with two lightweight
cross-attention blocks; it never performs an autoregressive student rollout.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


K = 300
HORIZON = 5
TOPK = 30
PAIR_NEGATIVE_COUNT = 60  # rank 31..90
CEM_ROUNDS = 10
FULL_LAYERS = 6
M_GRID = (60, 90, 120, 150, 180)
TRAIN_INDICES = tuple(range(20, 84))
CALIBRATION_INDICES = tuple(range(84, 92))
HELDOUT_INDICES = tuple(range(92, 100))
ALL_INDICES = TRAIN_INDICES + CALIBRATION_INDICES + HELDOUT_INDICES
TRAIN_SEED = 20260919
TRAIN_EPOCHS = 60

CONTEXT_ARCHITECTURE = {
    "architecture_id": "v2_context_aware_action_conditioned_cross_attention",
    "context_tokens": "candidate-independent z0 and zgoal visual/proprio context tokens",
    "action_tokens": "candidate action sequence tokens for each H=5 action sequence",
    "fusion": "action-conditioned cross-attention from action queries to context tokens",
    "cross_attention": True,
    "action_conditioned": True,
    "uses_observation_goal_context": True,
    "uses_autoregressive_rollout": False,
    "ranking_direction": "lower_is_better",
}
ACTION_ONLY_ARCHITECTURE = {
    "architecture_id": "v2_action_only_negative_control",
    "context_tokens": "none",
    "action_tokens": "candidate action sequence tokens",
    "fusion": "action_only_score_head",
    "cross_attention": False,
    "action_conditioned": True,
    "uses_observation_goal_context": False,
    "ranking_direction": "lower_is_better",
    "gate_required": False,
    "report_required": True,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _guard_compute_node() -> None:
    if not os.environ.get("PBS_JOBID", "").strip():
        raise RuntimeError("PBS_JOBID is empty; Rank Student V2 requires PBS")
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
        raise RuntimeError("GPU required; refusing CPU Rank Student V2")
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
    layers = getattr(getattr(model.predictor, "transformer", None), "layers", None)
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
    for dataset_index in ALL_INDICES:
        if dataset_index >= len(dset):
            raise RuntimeError(f"validation dataset has only {len(dset)} trajectories")
        _, _, _, env_info = dset[dataset_index]
        env.update_env(env_info)
        observation_seed = 100000 + dataset_index
        init_state, goal_state = env.sample_random_init_goal_states(observation_seed)
        obs_0, state_0 = env.prepare(observation_seed, init_state)
        obs_g, state_g = env.prepare(observation_seed, goal_state)
        targets.append(
            {
                "observation_id": f"wall_v2_case_{dataset_index:02d}",
                "rank_student_v2_label": f"wall_student_v2_case_{dataset_index:02d}",
                "dataset_index": dataset_index,
                "observation_seed": observation_seed,
                "obs_0": {key: _to_numpy(value)[None, None, ...] for key, value in obs_0.items()},
                "obs_g": {key: _to_numpy(value)[None, None, ...] for key, value in obs_g.items()},
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


def _encode_from_prefix(model: Any, prefix: Mapping[str, Any], act_0: Any, torch: Any) -> Any:
    act_emb = model.encode_act(act_0)
    visual = prefix["visual"]
    proprio = prefix["proprio"]
    if model.concat_dim == 0:
        return torch.cat([visual, proprio.unsqueeze(2), act_emb.unsqueeze(2)], dim=2)
    if model.concat_dim == 1:
        num_patches = visual.shape[2]
        proprio_tiled = proprio.unsqueeze(2).expand(-1, -1, num_patches, -1)
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, model.num_proprio_repeat)
        act_tiled = act_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
        act_repeated = act_tiled.repeat(1, 1, 1, model.num_action_repeat)
        return torch.cat([visual, proprio_repeated, act_repeated], dim=3)
    raise ValueError(f"unsupported model concat_dim={model.concat_dim!r}")


def _full_rollout(model: Any, encoded_prefix: Mapping[str, Any], actions: Any, torch: Any) -> Dict[str, Any]:
    visual = encoded_prefix["visual"]
    num_obs_init = visual.shape[1]
    if actions.shape[1] < num_obs_init:
        raise ValueError("candidate actions must include initial observation actions")
    z = _encode_from_prefix(model, encoded_prefix, actions[:, :num_obs_init], torch)
    suffix = actions[:, num_obs_init:]
    time_index = 0
    while time_index < suffix.shape[1]:
        z_pred = model.predict(z[:, -model.num_hist :])
        z_new = model.replace_actions_from_z(z_pred[:, -1:, ...], suffix[:, time_index : time_index + 1, :])
        z = torch.cat([z, z_new], dim=1)
        time_index += 1
    z_pred = model.predict(z[:, -model.num_hist :])
    z = torch.cat([z, z_pred[:, -1:, ...]], dim=1)
    z_obses, _ = model.separate_emb(z)
    return z_obses


def _pool_2x2(values: Any) -> Any:
    patches, dim = values.shape[-2:]
    side = math.isqrt(int(patches))
    if side * side != patches or side % 2:
        raise ValueError(f"expected an even square patch grid, got {patches}")
    values = values.reshape(side, side, dim).reshape(side // 2, 2, side // 2, 2, dim)
    return values.mean(dim=(1, 3)).reshape((side // 2) ** 2, dim)


def _prepare_context(runtime: Mapping[str, Any], preprocessor: Any, target: Dict[str, Any]) -> None:
    torch = runtime["torch"]
    device = torch.device("cuda:0")
    obs_0 = preprocessor.transform_obs(target["obs_0"])
    obs_g = preprocessor.transform_obs(target["obs_g"])
    obs_0 = {key: value.to(device) for key, value in obs_0.items()}
    obs_g = {key: value.to(device) for key, value in obs_g.items()}
    with torch.no_grad():
        target["prefix_one"] = runtime["model"].encode_obs(obs_0)
        target["goal_one"] = runtime["model"].encode_obs(obs_g)
        z0 = target["prefix_one"]["visual"][0, 0]
        zg = target["goal_one"]["visual"][0, 0]
        z0_pool = _pool_2x2(z0)
        zg_pool = _pool_2x2(zg)
        delta = zg_pool - z0_pool
        token_features = torch.cat([z0_pool, zg_pool, delta, delta.abs()], dim=-1)
        p0 = target["prefix_one"]["proprio"][0, 0]
        pg = target["goal_one"]["proprio"][0, 0]
        pd = pg - p0
        proprio_features = torch.cat([p0, pg, pd, pd.abs()], dim=-1)
    target["cached_context"] = {
        "z0_visual": z0.detach().cpu().half(),
        "zgoal_visual": zg.detach().cpu().half(),
        "z0_proprio": p0.detach().cpu().half(),
        "zgoal_proprio": pg.detach().cpu().half(),
        "token_features": token_features.detach().cpu().half(),
        "proprio_features": proprio_features.detach().cpu().half(),
    }


def _teacher_call(runtime: Mapping[str, Any], target: Dict[str, Any], round_index: int) -> Dict[str, Any]:
    torch = runtime["torch"]
    model = runtime["model"]
    action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
    candidate_seed = 1200000 + target["dataset_index"] * 100 + round_index
    _set_seed(torch, candidate_seed)
    input_mu = target["cem_mu"].detach().cpu().tolist()
    input_sigma = target["cem_sigma"].detach().cpu().tolist()
    with torch.no_grad():
        noise = torch.randn(K, HORIZON, action_dim)
        actions = noise.to("cuda:0") * target["cem_sigma"] + target["cem_mu"]
        actions[0] = target["cem_mu"]
        full_obs = _full_rollout(
            model, _expand_prefix(target["prefix_one"], K), actions, torch
        )
        full_values_tensor = runtime["objective_fn"](
            full_obs, _expand_prefix(target["goal_one"], K)
        )
        full_values = full_values_tensor.detach().cpu().float()
        full_order = sorted(
            range(K), key=lambda index: (float(full_values[index]), index)
        )
        elite = full_order[:TOPK]
        next_mu = actions[elite].mean(dim=0)
        next_sigma = actions[elite].std(dim=0)
    target["cem_mu"] = next_mu
    target["cem_sigma"] = next_sigma
    return {
        "observation_id": target["observation_id"],
        "rank_student_v2_label": target["rank_student_v2_label"],
        "dataset_index": target["dataset_index"],
        "observation_seed": target["observation_seed"],
        "round_index": round_index,
        "candidate_population_seed": candidate_seed,
        "cem_noise_seed": candidate_seed,
        "input_mu": input_mu,
        "input_sigma": input_sigma,
        "output_mu": next_mu.detach().cpu().tolist(),
        "output_sigma": next_sigma.detach().cpu().tolist(),
        "first_action": next_mu[0].detach().cpu().tolist(),
        "candidate_actions": actions.detach().cpu(),
        "J_full": full_values,
        "full_order": full_order,
        "full_top30": elite,
        "cached_context": target["cached_context"],
        "cem_update_source": "full_teacher",
        "student_used_for_cem_update": False,
    }


def _collect_split(runtime: Mapping[str, Any], preprocessor: Any, objective_fn: Any, targets: Mapping[int, Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    runtime["objective_fn"] = objective_fn
    records: List[Dict[str, Any]] = []
    action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
    for dataset_index in indices:
        target = targets[dataset_index]
        _prepare_context(runtime, preprocessor, target)
        # The candidate-independent encoded cache is retained; raw image arrays
        # are no longer needed once this observation has been prepared.
        target.pop("obs_0", None)
        target.pop("obs_g", None)
        target["cem_mu"] = runtime["torch"].zeros(HORIZON, action_dim, device="cuda:0")
        target["cem_sigma"] = runtime["torch"].ones_like(target["cem_mu"])
        for round_index in range(CEM_ROUNDS):
            records.append(_teacher_call(runtime, target, round_index))
        for key in ("prefix_one", "goal_one", "cem_mu", "cem_sigma"):
            target.pop(key, None)
    return records


def _build_models(torch: Any, token_input_dim: int, proprio_dim: int, action_dim: int) -> Tuple[Any, Any, Dict[str, int]]:
    nn = torch.nn

    class CrossAttentionStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_projection = nn.Sequential(
                nn.Linear(token_input_dim, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.proprio_projection = nn.Sequential(
                nn.Linear(proprio_dim, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.action_projection = nn.Sequential(
                nn.Linear(action_dim, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.action_temporal_projection = nn.Sequential(
                nn.Linear(HORIZON * 128, HORIZON * 128), nn.GELU()
            )
            self.action_position = nn.Parameter(torch.zeros(1, HORIZON, 128))
            self.cross_attention = nn.ModuleList(
                [nn.MultiheadAttention(128, 4, batch_first=True) for _ in range(2)]
            )
            self.query_norm = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
            self.ffn_norm = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
            self.ffn = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
                    for _ in range(2)
                ]
            )
            self.score_head = nn.Sequential(
                nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
            )

        def forward(self, token_features: Any, proprio_features: Any, actions: Any) -> Any:
            context = self.context_projection(token_features)
            proprio = self.proprio_projection(proprio_features).unsqueeze(1)
            context = context + proprio
            query = self.action_projection(actions)
            query = self.action_temporal_projection(query.flatten(1)).reshape(
                -1, HORIZON, 128
            ) + self.action_position
            for attention, query_norm, ffn_norm, ffn in zip(
                self.cross_attention, self.query_norm, self.ffn_norm, self.ffn
            ):
                attended, _ = attention(query, context, context, need_weights=False)
                query = query_norm(query + attended)
                query = ffn_norm(query + ffn(query))
            pooled_query = query.mean(dim=1)
            pooled_context = context.mean(dim=1)
            return self.score_head(torch.cat([pooled_query, pooled_context], dim=-1)).squeeze(-1)

    class ActionOnlyStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_mlp = nn.Sequential(
                nn.Linear(HORIZON * action_dim, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
            )

        def forward(self, actions: Any) -> Any:
            return self.action_mlp(actions.flatten(1)).squeeze(-1)

    context_model = CrossAttentionStudent().cuda()
    action_model = ActionOnlyStudent().cuda()
    counts = {
        "context_aware_student": sum(p.numel() for p in context_model.parameters()),
        "action_only_negative_control": sum(p.numel() for p in action_model.parameters()),
    }
    return context_model, action_model, counts


def _decision_loss(model: Any, action_model: Any, record: Mapping[str, Any], torch: Any) -> Tuple[Any, Any]:
    import torch.nn.functional as F

    actions = record["candidate_actions"].float().cuda()
    target = record["J_full"].float().cuda()
    target = (target - target.mean()) / target.std().clamp_min(1e-6)
    context = record["cached_context"]
    tokens = context["token_features"].float().cuda()
    proprio = context["proprio_features"].float().cuda()
    tokens = tokens.unsqueeze(0).expand(K, -1, -1)
    proprio = proprio.unsqueeze(0).expand(K, -1)
    elite = torch.tensor(record["full_order"][:TOPK], device="cuda", dtype=torch.long)
    negatives = torch.tensor(
        record["full_order"][TOPK : TOPK + PAIR_NEGATIVE_COUNT],
        device="cuda",
        dtype=torch.long,
    )

    def one_loss(scores: Any) -> Any:
        weights = torch.ones_like(target)
        weights[elite] = 4.0
        huber = F.smooth_l1_loss(scores, target, reduction="none")
        labels = torch.zeros_like(scores)
        labels[elite] = 1.0
        bce = F.binary_cross_entropy_with_logits(-scores, labels, weight=weights)
        # Explicit 30 x 60 comparisons: every elite is paired with ranks 31..90.
        pairwise = F.relu(
            0.10 + scores[elite].unsqueeze(1) - scores[negatives].unsqueeze(0)
        ).mean()
        return (huber * weights).mean() + bce + 0.5 * pairwise

    context_scores = model(tokens, proprio, actions)
    action_scores = action_model(actions)
    return one_loss(context_scores), one_loss(action_scores)


def _train_models(torch: Any, context_model: Any, action_model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _set_seed(torch, TRAIN_SEED)
    optimizer = torch.optim.AdamW(
        list(context_model.parameters()) + list(action_model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    order = list(range(len(records)))
    history: List[Dict[str, float]] = []
    for epoch in range(TRAIN_EPOCHS):
        random.Random(TRAIN_SEED + epoch).shuffle(order)
        context_model.train()
        action_model.train()
        context_total = action_total = 0.0
        for record_index in order:
            optimizer.zero_grad(set_to_none=True)
            context_loss, action_loss = _decision_loss(
                context_model, action_model, records[record_index], torch
            )
            (context_loss + action_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(context_model.parameters()) + list(action_model.parameters()), 1.0
            )
            optimizer.step()
            context_total += float(context_loss.detach().cpu())
            action_total += float(action_loss.detach().cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "context_loss": context_total / len(records),
                "action_only_loss": action_total / len(records),
            }
        )
    return {
        "seed": TRAIN_SEED,
        "epochs": TRAIN_EPOCHS,
        "record_count": len(records),
        "candidate_count_per_call": K,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss": "per-call standardized Huber + 4x elite-weighted top30 BCE/listwise classification + 30x60 rank31..90 pairwise hinge",
        "calibration_early_stopping": False,
        "history": history,
    }


def _score_arrays(model: Any, action_model: Any, record: Mapping[str, Any], torch: Any) -> Tuple[List[float], List[float]]:
    with torch.no_grad():
        actions = record["candidate_actions"].float().cuda()
        context = record["cached_context"]
        tokens = context["token_features"].float().cuda().unsqueeze(0).expand(K, -1, -1)
        proprio = context["proprio_features"].float().cuda().unsqueeze(0).expand(K, -1)
        context_scores = model(tokens, proprio, actions).detach().cpu().tolist()
        action_scores = action_model(actions).detach().cpu().tolist()
    return [float(x) for x in context_scores], [float(x) for x in action_scores]


def _stable_rank(values: Sequence[float]) -> List[int]:
    return sorted(range(len(values)), key=lambda index: (float(values[index]), index))


def _recall(rank: Sequence[int], full_top30: Sequence[int], m: int) -> float:
    return len(set(rank[:m]).intersection(full_top30)) / float(TOPK)


def _evaluate_split(records: Sequence[Dict[str, Any]], context_model: Any, action_model: Any, torch: Any, m_values: Sequence[int]) -> Dict[str, Any]:
    context_model.eval()
    action_model.eval()
    calls: List[Dict[str, Any]] = []
    for record in records:
        context_scores, action_scores = _score_arrays(context_model, action_model, record, torch)
        record["student_context_scores"] = context_scores
        record["student_action_only_scores"] = action_scores
        context_rank = _stable_rank(context_scores)
        action_rank = _stable_rank(action_scores)
        calls.append(
            {
                "observation_id": record["observation_id"],
                "rank_student_v2_label": record["rank_student_v2_label"],
                "dataset_index": record["dataset_index"],
                "round_index": record["round_index"],
                "recall_at_M": {
                    str(m): {
                        "context_aware_student": _recall(context_rank, record["full_top30"], m),
                        "action_only_negative_control": _recall(action_rank, record["full_top30"], m),
                    }
                    for m in m_values
                },
            }
        )
    return {"call_count": len(calls), "calls": calls}


def _aggregate(summary: Mapping[str, Any], m: int, key: str) -> Dict[str, Any]:
    values = [float(call["recall_at_M"][str(m)][key]) for call in summary["calls"]]
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {"call_count": len(values), "median_recall": median, "min_recall": min(values), "mean_recall": sum(values) / len(values), "recalls": values}


def _compact_metric(summary: Mapping[str, Any], m: int, key: str) -> Dict[str, Any]:
    metric = dict(_aggregate(summary, m, key))
    metric.pop("recalls", None)
    return metric


def _select_m(calibration_summary: Mapping[str, Any]) -> Dict[str, Any]:
    grid: Dict[str, Any] = {}
    feasible: List[int] = []
    for m in M_GRID:
        context = _compact_metric(calibration_summary, m, "context_aware_student")
        action = _compact_metric(calibration_summary, m, "action_only_negative_control")
        grid[str(m)] = {"context_aware": context, "action_only": action}
        if context["median_recall"] == 1.0 and context["min_recall"] >= 29 / 30:
            feasible.append(m)
    return {
        "schema": "dino-wm-shared-prefix-cache.rank-student-v2-selection",
        "schema_version": 2,
        "selected_M": min(feasible) if feasible else None,
        "candidate_M_grid": list(M_GRID),
        "selection_rule": "smallest_feasible_M_calibration_only",
        "selected_by": "calibration_only",
        "calibration_observation_ids": [f"wall_v2_case_{i:02d}" for i in CALIBRATION_INDICES],
        "heldout_observation_ids": [f"wall_v2_case_{i:02d}" for i in HELDOUT_INDICES],
        "heldout_used_for_selection": False,
        "calibration_metrics": grid,
    }


def _json_round(record: Mapping[str, Any], include_scores: bool) -> Dict[str, Any]:
    result = {
        "round_index": int(record["round_index"]),
        "candidate_population_seed": int(record["candidate_population_seed"]),
        "cem_noise_seed": int(record["cem_noise_seed"]),
        "candidate_actions": record["candidate_actions"].tolist(),
        "input_mu": record["input_mu"],
        "input_sigma": record["input_sigma"],
        "output_mu": record["output_mu"],
        "output_sigma": record["output_sigma"],
        "first_action": record["first_action"],
        "cem_update_source": "full_teacher",
        "student_used_for_cem_update": False,
        "J_full": record["J_full"].tolist(),
    }
    if include_scores:
        result["student_context_scores"] = record["student_context_scores"]
        result["student_action_only_scores"] = record["student_action_only_scores"]
    return result


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]], split: str, include_scores: bool) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["observation_id"]), []).append(record)
    with path.open("w", encoding="utf-8") as stream:
        for observation_id, group in grouped.items():
            group = sorted(group, key=lambda item: int(item["round_index"]))
            first = group[0]
            payload = {
                "schema": "dino-wm-shared-prefix-cache.rank-student-v2-teacher-record" if not include_scores else "dino-wm-shared-prefix-cache.rank-student-v2-evaluation-record",
                "schema_version": 2,
                "split": split,
                "observation_id": observation_id,
                "rank_student_v2_label": first["rank_student_v2_label"],
                "observation_seed": int(first["observation_seed"]),
                "candidate_count": K,
                "horizon": HORIZON,
                "topk": TOPK,
                "cem_opt_steps": CEM_ROUNDS,
                "rounds": [_json_round(record, include_scores) for record in group],
            }
            stream.write(json.dumps(payload, allow_nan=False) + "\n")


def _run(args: argparse.Namespace) -> int:
    _guard_compute_node()
    import torch
    from planning.objectives import create_objective_fn

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(root)
    preprocessor = _make_preprocessor(runtime)
    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    targets = {target["dataset_index"]: target for target in _make_targets(runtime)}

    records = {
        "train": _collect_split(runtime, preprocessor, objective_fn, targets, TRAIN_INDICES),
        "calibration": _collect_split(runtime, preprocessor, objective_fn, targets, CALIBRATION_INDICES),
        "heldout": _collect_split(runtime, preprocessor, objective_fn, targets, HELDOUT_INDICES),
    }
    _write_jsonl(output / "teacher_data.jsonl", records["train"], "train", False)
    torch.save(
        {
            "schema": "dino-wm-rank-student-v2-teacher-records",
            "settings": {"K": K, "H": HORIZON, "topk": TOPK, "cem_rounds": CEM_ROUNDS},
            "splits": records,
        },
        output / "teacher_records.pt",
    )

    first = records["train"][0]
    token_input_dim = int(first["cached_context"]["token_features"].shape[-1])
    proprio_dim = int(first["cached_context"]["proprio_features"].shape[-1])
    action_dim = int(first["candidate_actions"].shape[-1])
    context_model, action_model, parameter_counts = _build_models(
        torch, token_input_dim, proprio_dim, action_dim
    )
    train_metadata = _train_models(torch, context_model, action_model, records["train"])
    torch.save(
        {
            "schema": "dino-wm-rank-student-v2-checkpoint",
            "student": context_model.state_dict(),
            "action_only_negative_control": action_model.state_dict(),
            "architecture": {
                "token_input_dim": token_input_dim,
                "proprio_dim": proprio_dim,
                "action_dim": action_dim,
                "pooled_tokens": 49,
                "hidden_dim": 128,
                "action_query": "per-step MLP plus H=5 temporal projection",
                "cross_attention_blocks": 2,
                "autoregressive_rollout": False,
            },
            "parameter_counts": parameter_counts,
            "train_metadata": train_metadata,
        },
        output / "student_checkpoint.pt",
    )

    calibration_summary = _evaluate_split(records["calibration"], context_model, action_model, torch, M_GRID)
    selection = _select_m(calibration_summary)
    _write_json(output / "selection.json", selection)
    _write_jsonl(output / "calibration_results.jsonl", records["calibration"], "calibration", True)
    selected_m = selection["selected_M"]
    heldout_m = int(selected_m) if selected_m is not None else M_GRID[-1]
    heldout_summary = _evaluate_split(records["heldout"], context_model, action_model, torch, (heldout_m,))
    _write_jsonl(output / "heldout_results.jsonl", records["heldout"], "heldout", True)
    train_summary = _evaluate_split(records["train"], context_model, action_model, torch, (heldout_m,))

    training_summary = {
        "schema": "dino-wm-shared-prefix-cache.rank-student-v2-training-summary",
        "schema_version": 2,
        "training_status": "completed",
        "teacher_data_file": "teacher_data.jsonl",
        "train_observation_ids": [f"wall_v2_case_{i:02d}" for i in TRAIN_INDICES],
        "teacher_data_observation_count": len(TRAIN_INDICES),
        "teacher_data_round_count": len(TRAIN_INDICES) * CEM_ROUNDS,
        "uses_train_only": True,
        "calibration_used_for_training": False,
        "heldout_used_for_training": False,
        "calibration_and_heldout_in_training": False,
        "context_aware_student": {
            "status": "trained",
            "architecture": CONTEXT_ARCHITECTURE,
            "parameter_count": parameter_counts["context_aware_student"],
        },
        "action_only_control": {
            "reported": True,
            "status": "trained",
            "architecture": ACTION_ONLY_ARCHITECTURE,
            "parameter_count": parameter_counts["action_only_negative_control"],
        },
        "train_metadata": train_metadata,
    }
    _write_json(output / "student_training_summary.json", training_summary)

    summary = {
        "schema": "dino-wm-shared-prefix-cache.rank-student-v2-stage-a-summary",
        "schema_version": 2,
        "status": "complete",
        "stage": "rank_student_v2_stage_a",
        "claim_boundary": "offline ranking calibration and one held-out ranking evaluation only; no student-updated chained CEM, latency, memory, native-system, or closed-loop claim",
        "teacher": {"name": "full six-layer DINO-WM teacher", "full_only_cem_update": True, "student_autoregressive_rollout": False},
        "settings": {
            "candidate_count": K,
            "horizon": HORIZON,
            "topk": TOPK,
            "cem_opt_steps": CEM_ROUNDS,
            "M_grid": list(M_GRID),
            "train_indices": list(TRAIN_INDICES),
            "calibration_indices": list(CALIBRATION_INDICES),
            "heldout_indices": list(HELDOUT_INDICES),
            "observation_seed_rule": "100000 + dataset_index",
            "candidate_seed_rule": "1200000 + dataset_index*100 + round_index",
        },
        "model": {
            "architecture": "49 spatial context tokens with z0/zgoal/delta/absdelta -> 128, action H query MLP plus temporal projection, two residual cross-attention+FFN blocks, scalar fusion head",
            "parameter_counts": parameter_counts,
            "action_only_negative_control": True,
            "train_metadata": train_metadata,
        },
        "calibration_selection": selection,
        "selected_M": selected_m,
        "heldout_evaluation_M": heldout_m,
        "aggregate_recall": {
            "train": {"context_aware": _compact_metric(train_summary, heldout_m, "context_aware_student"), "action_only": _compact_metric(train_summary, heldout_m, "action_only_negative_control")},
            "calibration": {"context_aware": _compact_metric(calibration_summary, m, "context_aware_student") for m in M_GRID},
            "heldout": {"context_aware": _compact_metric(heldout_summary, heldout_m, "context_aware_student"), "action_only": _compact_metric(heldout_summary, heldout_m, "action_only_negative_control")},
        },
        "runtime_identity": {"checkpoint": str(runtime["checkpoint"]), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "predictor_layers": runtime["predictor_layers"]},
        "files": {"teacher_data": "teacher_data.jsonl", "teacher_records": "teacher_records.pt", "training_summary": "student_training_summary.json", "checkpoint": "student_checkpoint.pt", "calibration": "calibration_results.jsonl", "heldout": "heldout_results.jsonl", "selection": "selection.json", "summary": "stage_a_summary.json"},
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
                _write_json(output / "stage_a_summary.json", {"schema": "dino-wm-shared-prefix-cache.rank-student-v2-stage-a-summary", "status": "failed", "error": repr(exc)})
            except Exception:
                pass
        print(f"RANK_STUDENT_V2_STAGE_A_FAILED: {exc!r}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

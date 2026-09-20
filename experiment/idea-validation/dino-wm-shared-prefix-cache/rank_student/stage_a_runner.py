"""Train and calibrate a direct rank student for factorized DINO-WM CEM.

The teacher path is the existing full autoregressive world-model rollout.  The
student sees only a cached, candidate-independent ``z0``/``zgoal`` context and
an action sequence.  Training uses only the frozen train observations 00..11.  The
calibration split chooses one fixed M from the requested grid, and the
held-out split is evaluated once at that M.
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
TOTAL_OBSERVATIONS = 20
FULL_LAYERS = 6
M_GRID = (60, 90, 120, 150, 180)
TRAIN_INDICES = tuple(range(0, 12))
CALIBRATION_INDICES = tuple(range(12, 16))
HELDOUT_INDICES = tuple(range(16, 20))
TRAIN_SEED = 20260919
STUDENT_EPOCHS = 40


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _guard_compute_node() -> None:
    """Fail closed before model/data access unless running in PBS allocation."""

    if not os.environ.get("PBS_JOBID", "").strip():
        raise RuntimeError("PBS_JOBID is empty; rank-student experiment requires PBS")
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
    """Load the already prepared DINO-WM runtime on a compute node."""

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
        raise RuntimeError("GPU required; refusing CPU rank-student experiment")

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
    for dataset_index in range(TOTAL_OBSERVATIONS):
        if dataset_index >= len(dset):
            raise RuntimeError(f"validation dataset has only {len(dset)} trajectories")
        _, _, _, env_info = dset[dataset_index]
        env.update_env(env_info)
        observation_seed = 100000 + dataset_index
        init_state, goal_state = env.sample_random_init_goal_states(observation_seed)
        obs_0, state_0 = env.prepare(observation_seed, init_state)
        obs_g, state_g = env.prepare(observation_seed, goal_state)
        obs_0 = {key: _to_numpy(value)[None, None, ...] for key, value in obs_0.items()}
        obs_g = {key: _to_numpy(value)[None, None, ...] for key, value in obs_g.items()}
        targets.append(
            {
                # The frozen verifier uses wall_case_XX as the artifact ID.  The
                # rank-student experiment itself is identified by the schema and
                # split fields below.
                "observation_id": f"wall_case_{dataset_index:02d}",
                "dataset_index": dataset_index,
                "observation_seed": observation_seed,
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


def _rollout_from_common_prefix(
    model: Any, encoded_prefix: Mapping[str, Any], actions: Any, torch: Any
) -> Dict[str, Any]:
    """Run the same full predictor rollout as the existing Stage A runner."""

    visual = encoded_prefix["visual"]
    num_obs_init = visual.shape[1]
    if actions.shape[1] < num_obs_init:
        raise ValueError("candidate actions must include initial observation actions")
    act_0 = actions[:, :num_obs_init]
    action_suffix = actions[:, num_obs_init:]
    z = _encode_from_prefix(model, encoded_prefix, act_0, torch)
    time_index = 0
    while time_index < action_suffix.shape[1]:
        z_pred = model.predict(z[:, -model.num_hist :])
        z_new = z_pred[:, -1:, ...]
        z_new = model.replace_actions_from_z(
            z_new, action_suffix[:, time_index : time_index + 1, :]
        )
        z = torch.cat([z, z_new], dim=1)
        time_index += 1

    z_pred = model.predict(z[:, -model.num_hist :])
    z = torch.cat([z, z_pred[:, -1:, ...]], dim=1)
    z_obses, _ = model.separate_emb(z)
    return z_obses


def _stable_topk(values: Any, k: int) -> List[int]:
    flat = values.detach().cpu().tolist()
    return sorted(range(len(flat)), key=lambda index: (float(flat[index]), index))[:k]


def _spatial_pool_2x2(values: Any) -> Any:
    """Average each 2x2 patch block while retaining the pooled spatial layout."""

    patches, dim = values.shape[-2:]
    side = math.isqrt(int(patches))
    if side * side != patches or side % 2:
        raise ValueError(f"expected an even square patch grid, got {patches}")
    values = values.reshape(side, side, dim)
    values = values.reshape(side // 2, 2, side // 2, 2, dim)
    return values.mean(dim=(1, 3)).reshape((side // 2) ** 2, dim)


def _cached_context(prefix: Mapping[str, Any], goal: Mapping[str, Any]) -> Dict[str, Any]:
    """Create the candidate-independent z0/zgoal features used by the student."""

    import torch

    z0_visual = prefix["visual"][0, 0]
    zg_visual = goal["visual"][0, 0]
    z0_proprio = prefix["proprio"][0, 0]
    zg_proprio = goal["proprio"][0, 0]
    z0_pool = _spatial_pool_2x2(z0_visual)
    zg_pool = _spatial_pool_2x2(zg_visual)
    delta = zg_pool - z0_pool
    visual_features = torch.cat([z0_pool, zg_pool, delta, delta.abs()], dim=-1)
    proprio_delta = zg_proprio - z0_proprio
    proprio_features = torch.cat(
        [z0_proprio, zg_proprio, proprio_delta, proprio_delta.abs()], dim=-1
    )
    return {
        "z0_visual": z0_visual.detach().cpu().half(),
        "zgoal_visual": zg_visual.detach().cpu().half(),
        "z0_proprio": z0_proprio.detach().cpu().half(),
        "zgoal_proprio": zg_proprio.detach().cpu().half(),
        "visual_features": visual_features.detach().cpu().half(),
        "proprio_features": proprio_features.detach().cpu().half(),
    }


def _prepare_target_context(runtime: Mapping[str, Any], preprocessor: Any, target: Dict[str, Any]) -> None:
    """Encode each observation/goal once and reuse it across its 10 CEM calls."""

    torch = runtime["torch"]
    device = torch.device("cuda:0")
    trans_obs_0 = preprocessor.transform_obs(target["obs_0"])
    trans_obs_g = preprocessor.transform_obs(target["obs_g"])
    trans_obs_0 = {key: value.to(device) for key, value in trans_obs_0.items()}
    trans_obs_g = {key: value.to(device) for key, value in trans_obs_g.items()}
    with torch.no_grad():
        target["_prefix_one"] = runtime["model"].encode_obs(trans_obs_0)
        target["_goal_one"] = runtime["model"].encode_obs(trans_obs_g)
        target["_cached_context"] = _cached_context(target["_prefix_one"], target["_goal_one"])


def _evaluate_teacher_call(
    runtime: Mapping[str, Any], target: Mapping[str, Any], round_index: int
) -> Dict[str, Any]:
    """Generate one cached call, candidate pool, J_full, and full-only update."""

    torch = runtime["torch"]
    model = runtime["model"]
    action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
    common_prefix = _expand_prefix(target["_prefix_one"], CANDIDATES)
    goal_z = _expand_prefix(target["_goal_one"], CANDIDATES)
    candidate_seed = 1200000 + target["dataset_index"] * 100 + round_index
    _set_seed(torch, candidate_seed)
    input_mu = target["_cem_mu"].detach().cpu().tolist()
    input_sigma = target["_cem_sigma"].detach().cpu().tolist()
    with torch.no_grad():
        noise = torch.randn(CANDIDATES, HORIZON, action_dim)
        actions = noise.to("cuda:0") * target["_cem_sigma"] + target["_cem_mu"]
        actions[0] = target["_cem_mu"]
        full_obs = _rollout_from_common_prefix(model, common_prefix, actions, torch)
        full_objective = runtime["objective_fn"](full_obs, goal_z)
        full_values = full_objective.detach().cpu()
        full_order = _stable_topk(full_objective, CANDIDATES)
        elite_indices = full_order[:TOPK]
        next_mu = actions[elite_indices].mean(dim=0)
        next_sigma = actions[elite_indices].std(dim=0)

    target["_cem_mu"] = next_mu
    target["_cem_sigma"] = next_sigma
    return {
        "observation_id": target["observation_id"],
        "dataset_index": target["dataset_index"],
        "observation_seed": target["observation_seed"],
        "round_index": round_index,
        "candidate_population_seed": candidate_seed,
        "cem_noise_seed": candidate_seed,
        "input_mu": input_mu,
        "input_sigma": input_sigma,
        # Keep the exact teacher population in float32 so J_full and the
        # recorded action array describe the same candidates.
        "candidate_actions": actions.detach().cpu(),
        "J_full": full_values.float(),
        "full_order": full_order,
        "full_top30": elite_indices,
        "output_mu": next_mu.detach().cpu().tolist(),
        "output_sigma": next_sigma.detach().cpu().tolist(),
        "first_action": next_mu[0].detach().cpu().tolist(),
        "cem_update_source": "full_teacher",
        "student_used_for_cem_update": False,
        "cached_context": target["_cached_context"],
    }


def _collect_split_records(
    runtime: Mapping[str, Any],
    preprocessor: Any,
    objective_fn: Any,
    targets: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> List[Dict[str, Any]]:
    runtime["objective_fn"] = objective_fn
    records: List[Dict[str, Any]] = []
    for dataset_index in indices:
        target = dict(targets[dataset_index])
        _prepare_target_context(runtime, preprocessor, target)
        action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
        target["_cem_mu"] = runtime["torch"].zeros(HORIZON, action_dim, device="cuda:0")
        target["_cem_sigma"] = runtime["torch"].ones_like(target["_cem_mu"])
        for round_index in range(OPT_STEPS):
            records.append(_evaluate_teacher_call(runtime, target, round_index))
    return records


def _build_models(torch: Any, visual_dim: int, proprio_dim: int, action_dim: int, patch_count: int) -> Tuple[Any, Any, Dict[str, int]]:
    nn = torch.nn

    class ContextAwareStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_proj = nn.Sequential(
                nn.Linear(4 * visual_dim, 32), nn.LayerNorm(32), nn.GELU()
            )
            self.proprio_proj = nn.Sequential(
                nn.Linear(4 * proprio_dim, 32), nn.LayerNorm(32), nn.GELU()
            )
            self.context_proj = nn.Sequential(
                nn.Linear(32 * patch_count + 32, 128), nn.LayerNorm(128), nn.GELU()
            )
            self.action_proj = nn.Sequential(
                nn.Linear(HORIZON * action_dim, 64), nn.LayerNorm(64), nn.GELU()
            )
            self.score_head = nn.Sequential(
                nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1)
            )

        def forward(self, visual_features: Any, proprio_features: Any, actions: Any) -> Any:
            patches = self.patch_proj(visual_features).flatten(1)
            proprio = self.proprio_proj(proprio_features)
            context = self.context_proj(torch.cat([patches, proprio], dim=-1))
            action = self.action_proj(actions.flatten(1))
            return self.score_head(torch.cat([context, action], dim=-1)).squeeze(-1)

    class ActionOnlyStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_proj = nn.Sequential(
                nn.Linear(HORIZON * action_dim, 64), nn.LayerNorm(64), nn.GELU()
            )
            self.score_head = nn.Sequential(
                nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 1)
            )

        def forward(self, actions: Any) -> Any:
            return self.score_head(self.action_proj(actions.flatten(1))).squeeze(-1)

    context_model = ContextAwareStudent().cuda()
    action_model = ActionOnlyStudent().cuda()
    counts = {
        "context_aware_student": sum(parameter.numel() for parameter in context_model.parameters()),
        "action_only_negative_control": sum(parameter.numel() for parameter in action_model.parameters()),
    }
    return context_model, action_model, counts


def _call_loss(model: Any, action_model: Any, record: Mapping[str, Any], torch: Any) -> Tuple[Any, Any]:
    import torch.nn.functional as F

    actions = record["candidate_actions"].float().cuda()
    target = record["J_full"].float().cuda()
    target = (target - target.mean()) / target.std().clamp_min(1e-6)
    visual = record["cached_context"]["visual_features"].float().cuda()
    proprio = record["cached_context"]["proprio_features"].float().cuda()
    visual = visual.unsqueeze(0).expand(actions.shape[0], -1, -1)
    proprio = proprio.unsqueeze(0).expand(actions.shape[0], -1)
    elite = torch.tensor(record["full_order"][:TOPK], device="cuda", dtype=torch.long)
    negative = torch.tensor(record["full_order"][TOPK : 2 * TOPK], device="cuda", dtype=torch.long)

    pred = model(visual, proprio, actions)
    weights = torch.ones_like(target)
    weights[elite] = 4.0
    huber = F.smooth_l1_loss(pred, target, reduction="none")
    pairwise = F.relu(0.10 + pred[elite].mean() - pred[negative].mean())
    context_loss = (huber * weights).mean() + 0.5 * pairwise

    pred_action = action_model(actions)
    action_huber = F.smooth_l1_loss(pred_action, target, reduction="none")
    action_pairwise = F.relu(0.10 + pred_action[elite].mean() - pred_action[negative].mean())
    action_loss = (action_huber * weights).mean() + 0.5 * action_pairwise
    return context_loss, action_loss


def _train_models(torch: Any, context_model: Any, action_model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _set_seed(torch, TRAIN_SEED)
    optimizer = torch.optim.AdamW(
        list(context_model.parameters()) + list(action_model.parameters()),
        lr=2e-3,
        weight_decay=1e-4,
    )
    history: List[Dict[str, float]] = []
    order = list(range(len(records)))
    for epoch in range(STUDENT_EPOCHS):
        random.Random(TRAIN_SEED + epoch).shuffle(order)
        context_model.train()
        action_model.train()
        total_context = 0.0
        total_action = 0.0
        for record_index in order:
            optimizer.zero_grad(set_to_none=True)
            context_loss, action_loss = _call_loss(
                context_model, action_model, records[record_index], torch
            )
            loss = context_loss + action_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(context_model.parameters()) + list(action_model.parameters()), 1.0
            )
            optimizer.step()
            total_context += float(context_loss.detach().cpu())
            total_action += float(action_loss.detach().cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "context_loss": total_context / len(records),
                "action_only_loss": total_action / len(records),
            }
        )
    return {
        "seed": TRAIN_SEED,
        "epochs": STUDENT_EPOCHS,
        "record_count": len(records),
        "candidate_count_per_call": CANDIDATES,
        "optimizer": "AdamW",
        "learning_rate": 2e-3,
        "weight_decay": 1e-4,
        "loss": "per-call standardized J_full SmoothL1 with 4x full-elite weight plus pairwise elite-vs-next-30 hinge",
        "history": history,
    }


def _predict_rank(model: Any, record: Mapping[str, Any], torch: Any) -> List[int]:
    with torch.no_grad():
        actions = record["candidate_actions"].float().cuda()
        context = record["cached_context"]
        visual = context["visual_features"].float().cuda().unsqueeze(0).expand(CANDIDATES, -1, -1)
        proprio = context["proprio_features"].float().cuda().unsqueeze(0).expand(CANDIDATES, -1)
        predicted = model(visual, proprio, actions).detach().cpu().tolist()
    return sorted(range(CANDIDATES), key=lambda index: (float(predicted[index]), index))


def _predict_action_rank(model: Any, record: Mapping[str, Any], torch: Any) -> List[int]:
    with torch.no_grad():
        actions = record["candidate_actions"].float().cuda()
        predicted = model(actions).detach().cpu().tolist()
    return sorted(range(CANDIDATES), key=lambda index: (float(predicted[index]), index))


def _recall(rank: Sequence[int], full_top30: Sequence[int], m: int) -> float:
    return len(set(rank[:m]).intersection(full_top30)) / float(TOPK)


def _split_summary(
    records: Sequence[Mapping[str, Any]],
    context_model: Any,
    action_model: Any,
    torch: Any,
    m_values: Sequence[int],
) -> Dict[str, Any]:
    context_model.eval()
    action_model.eval()
    calls: List[Dict[str, Any]] = []
    for record in records:
        context_scores, action_scores = _score_arrays(
            context_model, action_model, record, torch
        )
        student_order = sorted(
            range(CANDIDATES), key=lambda index: (context_scores[index], index)
        )
        action_order = sorted(
            range(CANDIDATES), key=lambda index: (action_scores[index], index)
        )
        record["_student_context_scores"] = context_scores
        record["_student_action_only_scores"] = action_scores
        calls.append(
            {
                "observation_id": record["observation_id"],
                "dataset_index": record["dataset_index"],
                "observation_seed": record["observation_seed"],
                "round_index": record["round_index"],
                "candidate_population_seed": record["candidate_population_seed"],
                "full_score": [float(value) for value in record["J_full"].tolist()],
                "full_order": [int(value) for value in record["full_order"]],
                "full_top30": [int(value) for value in record["full_top30"]],
                "student_order": [int(value) for value in student_order],
                "action_only_order": [int(value) for value in action_order],
                "recall_at_M": {
                    str(m): {
                        "context_aware_student": _recall(student_order, record["full_top30"], m),
                        "action_only_negative_control": _recall(action_order, record["full_top30"], m),
                    }
                    for m in m_values
                },
            }
        )
    return {"call_count": len(calls), "calls": calls}


def _score_arrays(model: Any, action_model: Any, record: Mapping[str, Any], torch: Any) -> Tuple[List[float], List[float]]:
    """Return raw lower-is-better score arrays for the verifier-facing JSONL."""

    with torch.no_grad():
        actions = record["candidate_actions"].float().cuda()
        context = record["cached_context"]
        visual = context["visual_features"].float().cuda().unsqueeze(0).expand(CANDIDATES, -1, -1)
        proprio = context["proprio_features"].float().cuda().unsqueeze(0).expand(CANDIDATES, -1)
        context_scores = model(visual, proprio, actions).detach().cpu().tolist()
        action_scores = action_model(actions).detach().cpu().tolist()
    return [float(value) for value in context_scores], [float(value) for value in action_scores]


def _group_records(records: Sequence[Mapping[str, Any]]) -> List[List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["observation_id"]), []).append(record)
    return [sorted(items, key=lambda item: int(item["round_index"])) for items in grouped.values()]


def _json_round(record: Mapping[str, Any], context_scores: Sequence[float] | None = None, action_scores: Sequence[float] | None = None) -> Dict[str, Any]:
    """Convert one tensor-backed call to the frozen JSON/JSONL record shape."""

    result: Dict[str, Any] = {
        "round_index": int(record["round_index"]),
        "candidate_population_seed": int(record["candidate_population_seed"]),
        "cem_noise_seed": int(record["cem_noise_seed"]),
        "candidate_actions": record["candidate_actions"].float().tolist(),
        "input_mu": record["input_mu"],
        "input_sigma": record["input_sigma"],
        "output_mu": record["output_mu"],
        "output_sigma": record["output_sigma"],
        "first_action": record["first_action"],
        "cem_update_source": "full_teacher",
        "student_used_for_cem_update": False,
        "J_full": [float(value) for value in record["J_full"].tolist()],
    }
    if context_scores is not None and action_scores is not None:
        result["student_context_scores"] = [float(value) for value in context_scores]
        result["student_action_only_scores"] = [float(value) for value in action_scores]
    return result


def _write_record_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    split: str,
    context_model: Any | None = None,
    action_model: Any | None = None,
    torch: Any | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for group in _group_records(records):
            first = group[0]
            rounds: List[Dict[str, Any]] = []
            for record in group:
                context_scores = action_scores = None
                if context_model is not None and action_model is not None and torch is not None:
                    context_scores = record.get("_student_context_scores")
                    action_scores = record.get("_student_action_only_scores")
                    if context_scores is None or action_scores is None:
                        context_scores, action_scores = _score_arrays(
                            context_model, action_model, record, torch
                        )
                rounds.append(_json_round(record, context_scores, action_scores))
            payload = {
                "schema": (
                    "dino-wm-shared-prefix-cache.rank-student-teacher-record"
                    if context_model is None
                    else "dino-wm-shared-prefix-cache.rank-student-evaluation-record"
                ),
                "schema_version": 1,
                "split": split,
                "observation_id": first["observation_id"],
                "observation_seed": int(first["observation_seed"]),
                "candidate_count": CANDIDATES,
                "horizon": HORIZON,
                "topk": TOPK,
                "cem_opt_steps": OPT_STEPS,
                "rounds": rounds,
            }
            stream.write(json.dumps(payload, allow_nan=False) + "\n")


def _aggregate_recall(split_summary: Mapping[str, Any], m: int, model_key: str) -> Dict[str, float]:
    values = [float(call["recall_at_M"][str(m)][model_key]) for call in split_summary["calls"]]
    ordered = sorted(values)
    median = (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
    return {"median": median, "min": min(values), "mean": sum(values) / len(values)}


def _select_m(calibration_summary: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = []
    for m in M_GRID:
        context = _aggregate_recall(calibration_summary, m, "context_aware_student")
        action = _aggregate_recall(calibration_summary, m, "action_only_negative_control")
        candidates.append({"M": m, "context_aware": context, "action_only": action})
    feasible = [
        item["M"]
        for item in candidates
        if item["context_aware"]["median"] == 1.0
        and item["context_aware"]["min"] >= 29 / 30
    ]
    return {
        "selected_M": min(feasible) if feasible else None,
        "provenance": "smallest_feasible_M_calibration_only",
        "selection_rule": "median exactly 1.0 and minimum at least 29/30 on calibration context-aware recall",
        "calibration_gate": {
            "median_recall_required": 1.0,
            "min_recall_required": 29 / 30,
            "feasible_grid_values": feasible,
        },
        "grid": candidates,
    }


def _selection_artifact(calibration_summary: Mapping[str, Any], selected_m: int | None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for m in M_GRID:
        context_values = [
            float(call["recall_at_M"][str(m)]["context_aware_student"])
            for call in calibration_summary["calls"]
        ]
        action_values = [
            float(call["recall_at_M"][str(m)]["action_only_negative_control"])
            for call in calibration_summary["calls"]
        ]

        def summarize(values: Sequence[float]) -> Dict[str, Any]:
            ordered = sorted(values)
            middle = len(ordered) // 2
            median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
            return {
                "call_count": len(values),
                "median_recall": median,
                "min_recall": min(values),
                "recalls": list(values),
            }

        metrics[str(m)] = {
            "context_aware": summarize(context_values),
            "action_only": summarize(action_values),
        }
    return {
        "schema": "dino-wm-shared-prefix-cache.rank-student-selection",
        "schema_version": 1,
        "candidate_M_grid": list(M_GRID),
        "selection_rule": "smallest_feasible_M_calibration_only",
        "selected_by": "calibration_only",
        "calibration_observation_ids": [f"wall_case_{i:02d}" for i in CALIBRATION_INDICES],
        "heldout_observation_ids": [f"wall_case_{i:02d}" for i in HELDOUT_INDICES],
        "heldout_used_for_selection": False,
        "selected_M": selected_m,
        "calibration_metrics": metrics,
    }


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
    targets = _make_targets(runtime)

    records = {
        "train": _collect_split_records(runtime, preprocessor, objective_fn, targets, TRAIN_INDICES),
        "calibration": _collect_split_records(runtime, preprocessor, objective_fn, targets, CALIBRATION_INDICES),
        "heldout": _collect_split_records(runtime, preprocessor, objective_fn, targets, HELDOUT_INDICES),
    }
    _write_record_jsonl(output / "teacher_data.jsonl", records["train"], "train")
    # Candidate-level teacher material is kept in one inspectable artifact.
    torch.save(
        {
            "schema": "dino-wm-rank-student-teacher-records-v1",
            "splits": records,
            "settings": {"K": CANDIDATES, "H": HORIZON, "topk": TOPK, "cem_rounds": OPT_STEPS},
        },
        output / "teacher_records.pt",
    )

    first = records["train"][0]
    visual_dim = int(first["cached_context"]["visual_features"].shape[-1] // 4)
    proprio_dim = int(first["cached_context"]["proprio_features"].shape[-1] // 4)
    action_dim = int(first["candidate_actions"].shape[-1])
    patch_count = int(first["cached_context"]["visual_features"].shape[0])
    context_model, action_model, parameter_counts = _build_models(
        torch, visual_dim, proprio_dim, action_dim, patch_count
    )
    train_metadata = _train_models(torch, context_model, action_model, records["train"])

    training_summary = {
        "schema": "dino-wm-shared-prefix-cache.rank-student-training-summary",
        "schema_version": 1,
        "training_status": "completed",
        "teacher_data_file": "teacher_data.jsonl",
        "train_observation_ids": [f"wall_case_{i:02d}" for i in TRAIN_INDICES],
        "teacher_data_observation_count": len(TRAIN_INDICES),
        "teacher_data_round_count": len(TRAIN_INDICES) * OPT_STEPS,
        "uses_train_only": True,
        "calibration_used_for_training": False,
        "heldout_used_for_training": False,
        "context_aware_student": {
            "status": "trained",
            "architecture": "2x2-block pooled z0/zgoal/delta/absdelta context plus action-sequence MLP",
            "parameter_count": parameter_counts["context_aware_student"],
            "training": train_metadata,
        },
        "action_only_control": {
            "reported": True,
            "status": "trained",
            "parameter_count": parameter_counts["action_only_negative_control"],
        },
        "calibration_and_heldout_in_training": False,
    }
    _write_json(output / "student_training_summary.json", training_summary)

    checkpoint = {
        "schema": "dino-wm-rank-student-checkpoint-v1",
        "student": context_model.state_dict(),
        "action_only_negative_control": action_model.state_dict(),
        "architecture": {
            "visual_dim": visual_dim,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "patch_count": patch_count,
            "pooled_patch_grid": f"{int(math.sqrt(patch_count))}x{int(math.sqrt(patch_count))} after fixed 2x2-block averaging",
            "uses_autoregressive_rollout": False,
        },
        "parameter_counts": parameter_counts,
        "train_metadata": train_metadata,
    }
    torch.save(checkpoint, output / "student_checkpoint.pt")

    calibration_summary = _split_summary(
        records["calibration"], context_model, action_model, torch, M_GRID
    )
    selection = _select_m(calibration_summary)
    selected_m = selection["selected_M"]
    selected_m_for_eval = int(selected_m) if selected_m is not None else M_GRID[-1]
    selection_payload = _selection_artifact(calibration_summary, selected_m)
    _write_json(output / "selection.json", selection_payload)
    _write_record_jsonl(
        output / "calibration_results.jsonl",
        records["calibration"],
        "calibration",
        context_model,
        action_model,
        torch,
    )
    train_summary = _split_summary(
        records["train"], context_model, action_model, torch, (selected_m_for_eval,)
    )
    heldout_summary = _split_summary(
        records["heldout"], context_model, action_model, torch, (selected_m_for_eval,)
    )
    _write_record_jsonl(
        output / "heldout_results.jsonl",
        records["heldout"],
        "heldout",
        context_model,
        action_model,
        torch,
    )

    summary = {
        "schema": "dino-wm-rank-student-stage-a-summary-v1",
        "schema_version": 1,
        "status": "complete",
        "stage": "rank_student_stage_a",
        "claim_boundary": "student ranking calibration and one held-out ranking evaluation only; no latency, memory, chained student-CEM, native system, or closed-loop claim",
        "reference_path": "factorized_full",
        "teacher": {
            "name": "DINO-WM full predictor",
            "predictor_layers": FULL_LAYERS,
            "full_only_cem_update": True,
            "student_autoregressive_rollout": False,
        },
        "settings": {
            "candidate_count": CANDIDATES,
            "horizon": HORIZON,
            "topk": TOPK,
            "cem_opt_steps": OPT_STEPS,
            "M_grid": list(M_GRID),
            "dataset_split": {
                "train": [f"wall_case_{i:02d}" for i in TRAIN_INDICES],
                "calibration": [f"wall_case_{i:02d}" for i in CALIBRATION_INDICES],
                "heldout": [f"wall_case_{i:02d}" for i in HELDOUT_INDICES],
            },
            "requested_rank_student_labels": {
                "train": [f"wall_student_case_{i:02d}" for i in TRAIN_INDICES],
                "calibration": [f"wall_student_case_{i:02d}" for i in CALIBRATION_INDICES],
                "heldout": [f"wall_student_case_{i:02d}" for i in HELDOUT_INDICES],
            },
            "dataset_indices": {
                "train": list(TRAIN_INDICES),
                "calibration": list(CALIBRATION_INDICES),
                "heldout": list(HELDOUT_INDICES),
            },
            "observation_seed_rule": "100000 + dataset_index",
            "candidate_seed_rule": "1200000 + dataset_index*100 + round_index",
        },
        "runtime_identity": {
            "checkpoint": str(runtime["checkpoint"]),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "predictor_layers": runtime["predictor_layers"],
        },
        "student": {
            "architecture": "2x2-block pooled z0/zgoal/delta/absdelta patch projection + proprio projection + action-sequence MLP + fusion score head",
            "parameter_counts": parameter_counts,
            "train_metadata": train_metadata,
        },
        "splits": {
            "train": train_summary,
            "calibration": calibration_summary,
            "heldout": heldout_summary,
        },
        "calibration_selection": selection,
        "selected_M": selected_m,
        "heldout_evaluation_M": selected_m_for_eval,
        "heldout_selected_M_aggregate": {
            "context_aware_student": _aggregate_recall(heldout_summary, selected_m_for_eval, "context_aware_student"),
            "action_only_negative_control": _aggregate_recall(heldout_summary, selected_m_for_eval, "action_only_negative_control"),
        },
        "files": {
            "teacher_records": "teacher_records.pt",
            "student_checkpoint": "student_checkpoint.pt",
            "teacher_data": "teacher_data.jsonl",
            "training_summary": "student_training_summary.json",
            "calibration_results": "calibration_results.jsonl",
            "heldout_results": "heldout_results.jsonl",
            "selection": "selection.json",
            "summary": "stage_a_summary.json",
        },
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
                        "schema": "dino-wm-rank-student-stage-a-summary-v1",
                        "status": "failed",
                        "error": repr(exc),
                    },
                )
            except Exception:
                pass
        print(f"RANK_STUDENT_STAGE_A_FAILED: {exc!r}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen Stage-A mechanism smoke for the DINO-WM shared-prefix cache."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


N_CASES = 2
N_POPULATIONS = 2
N_CANDIDATES = 300
HORIZON = 5
TOPK = 30
MODES = ("baseline", "iteration_cache", "factorized_cache")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-cases", type=int, default=N_CASES)
    parser.add_argument("--cem-iters", type=int, default=N_POPULATIONS)
    return parser.parse_args()


def _guard_compute_node() -> None:
    if not os.environ.get("PBS_JOBID", "").strip():
        raise RuntimeError("PBS_JOBID is empty; Stage-A smoke requires a PBS allocation")
    host = socket.gethostname().lower()
    if any(token in host for token in ("login", "asp2a-login", "ntu-login")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, default=_json_default) + "\n")


def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _repeat_batch(value: Any, repeats: int) -> Any:
    return value[0:1].repeat((repeats,) + (1,) * (value.ndim - 1))


def _repeat_dict(values: Mapping[str, Any], repeats: int) -> Dict[str, Any]:
    return {key: _repeat_batch(value, repeats) for key, value in values.items()}


def _tensor_dict_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import torch

    return set(left) == set(right) and all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def _tensor_meta(value: Any, equal_to_baseline: bool) -> Dict[str, Any]:
    return {
        "equal_to_baseline": bool(equal_to_baseline),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


class CacheCoreRunner:
    """Use only the integrated cache_core helpers; no local fallback."""

    def __init__(self, model: Any, mode: str, cache_core: Any):
        self.model = model
        self.mode = mode
        self.cache_core = cache_core
        self.cached_prefix = None
        self.last_prefix = None

    def rollout(self, obs_batch: Mapping[str, Any], actions: Any) -> Mapping[str, Any]:
        if self.mode == "iteration_cache":
            if self.cached_prefix is None:
                self.cached_prefix = self.cache_core.encode_and_repeat(
                    self.model, obs_batch, mode=self.mode
                )
            prefix = self.cached_prefix
        elif self.mode == "factorized_cache":
            if self.cached_prefix is None:
                single = {key: value[:1] for key, value in obs_batch.items()}
                self.cached_prefix = self.cache_core.encode_and_repeat(
                    self.model, single, mode=self.mode, repeats=1
                )
            prefix = self.cache_core.repeat_encoded_prefix(
                self.cached_prefix, int(actions.shape[0])
            )
        else:
            raise ValueError(self.mode)
        self.last_prefix = prefix
        rollout, _ = self.cache_core.rollout_from_encoded_obs(
            self.model, prefix, actions
        )
        return rollout


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
        raise RuntimeError("GPU required; refusing accidental CPU execution")
    checkpoint_dir = root / "checkpoints" / "outputs" / "wall_single"
    config_path = checkpoint_dir / "hydra.yaml"
    checkpoint_path = checkpoint_dir / "checkpoints" / "model_latest.pth"
    for required in (root / "data", config_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(required)

    cache_core = importlib.import_module("cache_core")
    for name in ("encode_and_repeat", "repeat_encoded_prefix", "rollout_from_encoded_obs"):
        if not callable(getattr(cache_core, name, None)):
            raise AttributeError(f"cache_core.{name} is required")

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
    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    return {
        "torch": torch,
        "model": model,
        "model_cfg": model_cfg,
        "dset": trajectory_datasets["valid"],
        "env": env,
        "checkpoint": checkpoint_path,
        "cache_core": cache_core,
    }


def _make_targets(runtime: Mapping[str, Any]) -> List[Dict[str, Any]]:
    import numpy as np

    dset = runtime["dset"]
    env = runtime["env"]
    targets = []
    for case_id in range(N_CASES):
        if case_id >= len(dset):
            raise RuntimeError(f"validation dataset has only {len(dset)} trajectories")
        _, _, _, env_info = dset[case_id]
        env.update_env(env_info)
        seed = 100000 + case_id
        init_state, goal_state = env.sample_random_init_goal_states([seed])
        obs_0, state_0 = env.prepare([seed], init_state)
        obs_g, state_g = env.prepare([seed], goal_state)
        targets.append(
            {
                "case_id": case_id,
                "observation_id": f"wall_case_{case_id:02d}",
                "env_seed": seed,
                "obs_0": {
                    key: np.expand_dims(np.expand_dims(value, axis=0), axis=1)
                    for key, value in obs_0.items()
                },
                "obs_g": {
                    key: np.expand_dims(np.expand_dims(value, axis=0), axis=1)
                    for key, value in obs_g.items()
                },
                "state_0": np.asarray(state_0).tolist(),
                "state_g": np.asarray(state_g).tolist(),
            }
        )
    return targets


def _preprocessor(runtime: Mapping[str, Any]) -> Any:
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


def _candidate_pools(torch: Any, case_id: int, action_dim: int) -> List[Any]:
    pools = []
    for iteration in range(N_POPULATIONS):
        seed = 110000 + case_id * 100 + iteration
        generator = torch.Generator(device="cpu").manual_seed(seed)
        candidates = torch.randn(
            N_CANDIDATES, HORIZON, action_dim, generator=generator, dtype=torch.float32
        )
        candidates[0] = 0.0
        pools.append(candidates.cuda())
    return pools


def _objective_scores(
    model: Any,
    objective_fn: Any,
    trans_0: Mapping[str, Any],
    z_goal_single: Mapping[str, Any],
    candidates: Any,
    mode: str,
    cache: CacheCoreRunner | None,
) -> Tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    import torch

    obs_batch = _repeat_dict(trans_0, N_CANDIDATES)
    goal_batch = _repeat_dict(z_goal_single, N_CANDIDATES)
    with torch.no_grad():
        if mode == "baseline":
            rollout, _ = model.rollout(obs_0=obs_batch, act=candidates)
            prefix = model.encode_obs(obs_batch)
        else:
            if cache is None:
                raise RuntimeError(f"cache backend missing for {mode}")
            rollout = cache.rollout(obs_batch, candidates)
            prefix = cache.last_prefix
        scores = objective_fn(rollout, goal_batch).detach()
    return scores, prefix, rollout


def _rank_and_update(scores: Any, candidates: Any) -> Dict[str, Any]:
    import torch

    cpu_scores = scores.detach().cpu()
    order = torch.argsort(cpu_scores, stable=True)
    elite = order[:TOPK]
    elite_actions = candidates[elite]
    mu = elite_actions.mean(dim=0)
    sigma = elite_actions.std(dim=0)
    return {
        "objective": cpu_scores.tolist(),
        "full_order": order.tolist(),
        "topk30": elite.tolist(),
        "mu": mu.detach().cpu().tolist(),
        "sigma": sigma.detach().cpu().tolist(),
        "first_action": candidates[int(elite[0]), 0].detach().cpu().tolist(),
    }


def _downstream_mismatch(current_scores: Any, baseline_scores: Any) -> bool:
    import torch

    current_order = torch.argsort(current_scores.detach().cpu(), stable=True)
    baseline_order = torch.argsort(baseline_scores.detach().cpu(), stable=True)
    return bool(
        not torch.equal(current_scores, baseline_scores)
        or not torch.equal(current_order, baseline_order)
    )


def _control_record(current_scores: Any, baseline_scores: Any) -> Dict[str, Any]:
    mismatch = _downstream_mismatch(current_scores, baseline_scores)
    return {
        "fallback_disabled": True,
        "mismatch_detected": mismatch,
        "downstream_mismatch": mismatch,
    }


def _path_record(
    mode: str,
    scores: Any,
    prefix: Mapping[str, Any],
    rollout: Mapping[str, Any],
    baseline_scores: Any,
    baseline_prefix: Mapping[str, Any],
    baseline_rollout: Mapping[str, Any],
    candidates: Any,
) -> Dict[str, Any]:
    import torch

    prefix_equal = _tensor_dict_equal(prefix, baseline_prefix)
    rollout_equal = _tensor_dict_equal(rollout, baseline_rollout)
    score_equal = torch.equal(scores, baseline_scores)
    current = _rank_and_update(scores, candidates)
    baseline = _rank_and_update(baseline_scores, candidates)
    exact = bool(
        prefix_equal
        and rollout_equal
        and score_equal
        and current["full_order"] == baseline["full_order"]
        and current["topk30"] == baseline["topk30"]
        and current["mu"] == baseline["mu"]
        and current["sigma"] == baseline["sigma"]
        and current["first_action"] == baseline["first_action"]
    )
    result = {
        "cache_hit": mode != "baseline",
        "prefix": _tensor_meta(prefix["visual"], prefix_equal),
        "rollout": _tensor_meta(rollout["visual"], rollout_equal),
        "objective": current["objective"],
        "full_order": current["full_order"],
        "topk30": current["topk30"],
        "mu": current["mu"],
        "sigma": current["sigma"],
        "first_action": current["first_action"],
    }
    if mode != "baseline":
        result["canary"] = {
            "executed": True,
            "fallback_triggered": not exact,
        }
    return result


def _run(args: argparse.Namespace) -> int:
    import torch
    from planning.objectives import create_objective_fn
    from utils import move_to_device

    _guard_compute_node()
    if args.n_cases != N_CASES or args.cem_iters != N_POPULATIONS:
        raise ValueError("frozen Stage-A smoke requires exactly 2 cases and 2 populations")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(args.root.resolve())
    model = runtime["model"]
    preprocessor = _preprocessor(runtime)
    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    targets = _make_targets(runtime)
    action_dim = int(runtime["dset"].action_dim * runtime["model_cfg"].frameskip)
    prepared = []
    for target in targets:
        device = next(model.parameters()).device
        trans_0 = move_to_device(preprocessor.transform_obs(target["obs_0"]), device)
        trans_g = move_to_device(preprocessor.transform_obs(target["obs_g"]), device)
        with torch.no_grad():
            z_goal_single = model.encode_obs(trans_g)
        prepared.append((trans_0, z_goal_single))

    records: List[Dict[str, Any]] = []
    for case_index, target in enumerate(targets):
        trans_0, z_goal_single = prepared[case_index]
        stale_trans_0, _ = prepared[(case_index + 1) % len(prepared)]
        caches = {
            mode: CacheCoreRunner(model, mode, runtime["cache_core"])
            for mode in MODES[1:]
        }
        pools = _candidate_pools(runtime["torch"], case_index, action_dim)
        for iteration, candidates in enumerate(pools):
            _sync(torch)
            baseline_scores, baseline_prefix, baseline_rollout = _objective_scores(
                model, objective_fn, trans_0, z_goal_single, candidates, "baseline", None
            )
            _sync(torch)
            path_scores = {"baseline": baseline_scores}
            path_prefix = {"baseline": baseline_prefix}
            path_rollout = {"baseline": baseline_rollout}
            for mode in MODES[1:]:
                _sync(torch)
                scores, prefix, rollout = _objective_scores(
                    model, objective_fn, trans_0, z_goal_single, candidates, mode, caches[mode]
                )
                _sync(torch)
                path_scores[mode] = scores
                path_prefix[mode] = prefix
                path_rollout[mode] = rollout

            goal_batch = _repeat_dict(z_goal_single, N_CANDIDATES)
            wrong_prefix_rollout = {
                key: value[0:1].expand_as(value) for key, value in baseline_rollout.items()
            }
            wrong_scores = objective_fn(wrong_prefix_rollout, goal_batch).detach()
            stale_batch = _repeat_dict(stale_trans_0, N_CANDIDATES)
            with torch.no_grad():
                stale_rollout, _ = model.rollout(obs_0=stale_batch, act=candidates)
                stale_scores = objective_fn(stale_rollout, goal_batch).detach()

            record = {
                "population_id": f"wall_case_{case_index:02d}_cem_{iteration:02d}",
                "observation_id": target["observation_id"],
                "cem_iteration": iteration,
                "candidate_population_seed": 110000 + case_index * 100 + iteration,
                "candidate_count": N_CANDIDATES,
                "horizon": HORIZON,
                "topk": TOPK,
                "paths": {
                    mode: _path_record(
                        mode,
                        path_scores[mode],
                        path_prefix[mode],
                        path_rollout[mode],
                        baseline_scores,
                        baseline_prefix,
                        baseline_rollout,
                        candidates,
                    )
                    for mode in MODES
                },
                "negative_controls": {
                    "wrong_boundary": _control_record(wrong_scores, baseline_scores),
                    "stale_observation": _control_record(stale_scores, baseline_scores),
                },
            }
            records.append(record)

    _write_jsonl(out / "mechanism.jsonl", records)
    _write_json(
        out / "summary.json",
        {
            "status": "stage_a_mechanism_only",
            "schema": "dino-wm-shared-prefix-cache-stage-a-v1",
            "mechanism_file": "mechanism.jsonl",
            "n_cases": N_CASES,
            "populations": N_POPULATIONS,
            "candidate_count": N_CANDIDATES,
            "horizon": HORIZON,
            "topk": TOPK,
            "checkpoint": str(runtime["checkpoint"]),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "system_gate": "NOT_RUN",
            "closed_loop_gate": "NOT_AUTHORIZED",
        },
    )
    runtime["env"].close()
    return 0


def main() -> int:
    try:
        return _run(_args())
    except Exception as exc:
        if "--output" in sys.argv:
            try:
                output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
                output.mkdir(parents=True, exist_ok=True)
                _write_json(output / "summary.json", {"status": "failed", "error": repr(exc)})
            except Exception:
                pass
        print(f"STAGE_A_FAILED: {exc!r}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

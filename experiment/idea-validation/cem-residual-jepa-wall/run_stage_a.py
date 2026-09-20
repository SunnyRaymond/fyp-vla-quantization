#!/usr/bin/env python3
"""Frozen Wall Stage A: dense teacher residuals around each CEM mean rollout.

This runner reuses the existing Wall ``smoke_runner._runtime`` and
``_score_pool`` call path.  It never creates an environment or runs MPC.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import socket
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
TOPK = 30
CHUNK = 32
HORIZON = 5


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wall-root", type=Path, required=True, help="DINO-WM Wall checkout with source/checkpoints")
    p.add_argument("--workload", type=Path, required=True, help="frozen rankcal Wall workload.pkl")
    p.add_argument("--pools-root", type=Path, default=None, help="directory containing episodes/episode_NNN/pools.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    p.add_argument("--smoke-runner", type=Path, default=None)
    return p.parse_args()


def _require_compute_node() -> None:
    job = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    host = socket.gethostname().lower()
    if not job:
        raise RuntimeError("PBS_JOBID is required; refusing to run Stage A outside a PBS allocation")
    if not nodefile or not Path(nodefile).is_file() or not Path(nodefile).read_text(errors="ignore").strip():
        raise RuntimeError("PBS_NODEFILE is required and must name a non-empty allocation")
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/head/submit host: {host}")
    if shutil.which("nvidia-smi") is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU and nvidia-smi are required")


def _load_smoke(path: Path | None, wall_root: Path):
    candidates = [path] if path else []
    candidates += [wall_root / "smoke_runner.py", HERE.parent / "world-model-quantization" / "dino-wm-wall" / "smoke_runner.py"]
    for candidate in candidates:
        if candidate and candidate.is_file():
            spec = importlib.util.spec_from_file_location("wall_smoke_runner", candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {candidate}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("smoke_runner.py is required beside --wall-root or via --smoke-runner")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _episode_number(episode_id: str) -> int:
    return int(str(episode_id).rsplit(":", 1)[-1])


def _load_cases(workload_path: Path, pools_root: Path | None) -> list[dict[str, Any]]:
    with workload_path.open("rb") as stream:
        workload = pickle.load(stream)
    cases = list(workload.get("cases", []))
    if len(cases) != 26:
        raise ValueError(f"frozen workload must contain 26 pools, found {len(cases)}")
    root = (pools_root or workload_path.parent).resolve()
    per_episode: dict[str, int] = {}
    loaded: dict[int, Any] = {}
    result: list[dict[str, Any]] = []
    for case in cases:
        episode = str(case["episode_id"])
        number = _episode_number(episode)
        episode_path = root / "episodes" / f"episode_{number:03d}" / "pools.npz"
        if number not in loaded:
            loaded[number] = np.load(episode_path, allow_pickle=False)
        pool_index = per_episode.get(episode, 0)
        per_episode[episode] = pool_index + 1
        prefix = f"pool_{pool_index:02d}"
        arrays = loaded[number]
        needed = ["candidate_actions", "scores", "mu_before", "sigma_before"]
        if any(f"{prefix}_{key}" not in arrays for key in needed):
            raise KeyError(f"missing {prefix} arrays in {episode_path}")
        candidate = np.asarray(case["candidates"], dtype=np.float32)
        reference = np.asarray(case["reference_scores"], dtype=np.float32)
        mu = np.asarray(arrays[f"{prefix}_mu_before"], dtype=np.float32)
        sigma = np.asarray(arrays[f"{prefix}_sigma_before"], dtype=np.float32)
        if candidate.shape != (300, HORIZON, 10) or reference.shape != (300,):
            raise ValueError(f"bad candidate/reference shape for {case['case_id']}")
        if mu.shape != (HORIZON, 10) or sigma.shape != (HORIZON, 10) or not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError(f"bad mu/sigma for {case['case_id']}")
        if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
            raise ValueError(f"non-finite frozen pool {case['case_id']}")
        result.append({
            "case_id": str(case["case_id"]),
            "episode_id": episode,
            "episode_number": number,
            "obs_0": case["obs_0"],
            "obs_g": case["obs_g"],
            "candidates": candidate,
            "reference_scores": reference,
            "mu": mu,
            "sigma": sigma,
        })
    if sorted({p["episode_number"] for p in result}) != list(range(8)):
        raise ValueError("frozen workload must cover development episodes 000..007")
    return result


def _repeat(mapping: Mapping[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    return {key: value[:1].expand((n,) + tuple(value.shape[1:])) for key, value in mapping.items()}


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(v) for v in value.values())
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return True


class CausalResidualStudent(nn.Module):
    """Small causal predictor preserving dense visual-token residuals."""

    def __init__(self, visual_dim: int, proprio_dim: int, action_dim: int, hidden: int = 64, heads: int = 4):
        super().__init__()
        self.visual_dim, self.proprio_dim = visual_dim, proprio_dim
        self.visual_proj = nn.Linear(visual_dim, hidden)
        self.proprio_proj = nn.Linear(proprio_dim, hidden)
        self.eps_proj = nn.Linear(2 * action_dim, hidden)
        self.context_proj = nn.Linear(visual_dim + proprio_dim, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.visual_head = nn.Linear(hidden, visual_dim)
        self.proprio_head = nn.Linear(hidden, proprio_dim)

    def forward(self, base: Mapping[str, torch.Tensor], epsilon: torch.Tensor, sigma: torch.Tensor) -> dict[str, torch.Tensor]:
        visual = base["visual"]
        squeezed = visual.ndim == 3
        if squeezed:
            visual = visual.unsqueeze(2)
        if visual.ndim != 4:
            raise ValueError(f"expected visual latent [B,T,P,D], got {tuple(base['visual'].shape)}")
        b, t, patches, dim = visual.shape
        proprio = base["proprio"]
        context = torch.cat((visual[:, 0].mean(dim=1), proprio[:, 0]), dim=-1)
        x = self.visual_proj(visual) + self.proprio_proj(proprio).unsqueeze(2)
        x = x + self.eps_proj(torch.cat((epsilon, sigma), dim=-1)).unsqueeze(2)
        x = x + self.context_proj(context).view(b, 1, 1, -1)
        x = x.permute(0, 2, 1, 3).reshape(b * patches, t, -1)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        x = self.norm(self.temporal(x, mask=mask)).reshape(b, patches, t, -1).permute(0, 2, 1, 3)
        delta_visual = self.visual_head(x)
        delta_proprio = self.proprio_head(x.mean(dim=2))
        if squeezed:
            delta_visual = delta_visual.squeeze(2)
        return {"visual": delta_visual, "proprio": delta_proprio}


def _latent_loss(pred: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return 0.5 * (nn.functional.mse_loss(pred["visual"], target["visual"]) + nn.functional.mse_loss(pred["proprio"], target["proprio"]))


def _prepare(pool: Mapping[str, Any], preprocessor: Any, device: torch.device) -> dict[str, Any]:
    trans0 = preprocessor.transform_obs(pool["obs_0"])
    transg = preprocessor.transform_obs(pool["obs_g"])
    return {
        "trans0": {key: value.to(device) for key, value in trans0.items()},
        "transg": {key: value.to(device) for key, value in transg.items()},
    }


def _rollout(model: Any, trans0: Mapping[str, torch.Tensor], actions: torch.Tensor) -> dict[str, torch.Tensor]:
    rolled = model.rollout(obs_0=_repeat(trans0, int(actions.shape[0])), act=actions)[0]
    # ``VWorldModel.rollout`` returns the initial observation followed by H
    # predictions.  The residual contract covers only the H future latents.
    future = {key: value[:, 1 : HORIZON + 1] for key, value in rolled.items()}
    if any(value.shape[1] != HORIZON for value in future.values()):
        raise RuntimeError("teacher rollout did not return exactly H future latents")
    return future


def _base_and_goal(model: Any, prepared: Mapping[str, Any], mu: np.ndarray, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    actions = torch.as_tensor(mu[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        base = {key: value.detach() for key, value in _rollout(model, prepared["trans0"], actions).items()}
        goal = {key: value.detach() for key, value in model.encode_obs(prepared["transg"]).items()}
    return base, goal


def _train(model: Any, student: CausalResidualStudent, pools: Sequence[dict[str, Any]], prepared: Mapping[str, Any], bases: Mapping[str, dict[str, torch.Tensor]], freeze: Mapping[str, Any], device: torch.device) -> list[float]:
    cfg = freeze["training"]
    rng = random.Random(int(cfg["seed"]))
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg["learning_rate"]))
    losses: list[float] = []
    for _ in range(int(cfg["passes"])):
        order = list(pools); rng.shuffle(order)
        for pool in order:
            base_cpu = {key: value.cpu() for key, value in bases[pool["case_id"]].items()}
            actions_all = torch.as_tensor(pool["candidates"], dtype=torch.float32, device=device)
            mu = torch.as_tensor(pool["mu"], dtype=torch.float32, device=device)
            sigma = torch.as_tensor(pool["sigma"], dtype=torch.float32, device=device)
            for start in range(0, 300, int(cfg["chunk_size"])):
                end = min(start + int(cfg["chunk_size"]), 300)
                actions = actions_all[start:end]
                eps = (actions - mu) / sigma
                with torch.no_grad():
                    teacher = _rollout(model, prepared[pool["case_id"]]["trans0"], actions)
                base = {key: value.to(device).expand((end - start,) + tuple(value.shape[1:])) for key, value in base_cpu.items()}
                target = {key: teacher[key].detach() - base[key] for key in teacher}
                with torch.enable_grad():
                    pred = student(base, eps, sigma.expand_as(eps))
                    loss = _latent_loss(pred, target)
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))
    return losses


def _rank(values: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(values.detach().float().cpu(), stable=True), stable=True).float()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _rank(a), _rank(b); x, y = x - x.mean(), y - y.mean()
    den = torch.sqrt(x.square().sum() * y.square().sum())
    return float((x * y).sum() / den) if float(den) > 0 else 0.0


def _topk(a: torch.Tensor, b: torch.Tensor, k: int = TOPK) -> float:
    k = min(k, int(a.numel()), int(b.numel()))
    x = set(torch.argsort(a, stable=True)[:k].tolist()); y = set(torch.argsort(b, stable=True)[:k].tolist())
    return len(x & y) / max(k, 1)


def _evaluate(model: Any, student: CausalResidualStudent, pools: Sequence[dict[str, Any]], prepared: Mapping[str, Any], bases: Mapping[str, dict[str, torch.Tensor]], goals: Mapping[str, dict[str, torch.Tensor]], objective: Any, device: torch.device) -> list[dict[str, Any]]:
    student.eval(); records = []; shuffle_rng = torch.Generator(device="cpu").manual_seed(20260921)
    with torch.no_grad():
        for pool in pools:
            key = pool["case_id"]; base = {name: value.to(device) for name, value in bases[key].items()}; goal = goals[key]
            actions_all = torch.as_tensor(pool["candidates"], dtype=torch.float32, device=device)
            mu = torch.as_tensor(pool["mu"], dtype=torch.float32, device=device); sigma = torch.as_tensor(pool["sigma"], dtype=torch.float32, device=device)
            perm = torch.randperm(300, generator=shuffle_rng); perm = perm.roll(1) if bool(torch.equal(perm, torch.arange(300))) else perm
            all_eps = (actions_all - mu) / sigma
            shuffled_all_eps = all_eps.index_select(0, perm.to(device))
            truth, mean_cost, residual_cost, shuffled_cost = [], [], [], []
            residual_mse = []
            for start in range(0, 300, CHUNK):
                end = min(start + CHUNK, 300); actions = actions_all[start:end]; eps = all_eps[start:end]
                base_batch = {name: value.expand((end - start,) + tuple(value.shape[1:])) for name, value in base.items()}
                student_eps = eps; shuffled_eps = shuffled_all_eps[start:end]
                teacher = _rollout(model, prepared[key]["trans0"], actions)
                goal_batch = _repeat(goal, end - start)
                predicted_delta = student(base_batch, student_eps, sigma.expand_as(eps))
                shuffled_delta = student(base_batch, shuffled_eps, sigma.expand_as(eps))
                pred = {name: base_batch[name] + predicted_delta[name] for name in base_batch}
                shuffled = {name: base_batch[name] + shuffled_delta[name] for name in base_batch}
                truth.append(objective(teacher, goal_batch).cpu()); mean_cost.append(objective(base_batch, goal_batch).cpu())
                residual_cost.append(objective(pred, goal_batch).cpu()); shuffled_cost.append(objective(shuffled, goal_batch).cpu())
                residual_mse.append(float(_latent_loss(pred, teacher).cpu()))
            t, m, r, s = map(torch.cat, (truth, mean_cost, residual_cost, shuffled_cost))
            ref_err = float((t - torch.from_numpy(pool["reference_scores"])).abs().max())
            records.append({"case_id": key, "episode_id": pool["episode_id"], "candidate_count": 300, "finite": all(_finite(x) for x in (t, m, r, s)), "reference_max_abs_error": ref_err, "spearman": {"mean_only": _spearman(t, m), "residual": _spearman(t, r), "shuffled": _spearman(t, s)}, "top30": {"mean_only": _topk(t, m), "residual": _topk(t, r), "shuffled": _topk(t, s)}, "residual_latent_mse": float(statistics.mean(residual_mse))})
    return records


def _causality(student: CausalResidualStudent, pool: Mapping[str, Any], base: Mapping[str, torch.Tensor], device: torch.device, tolerance: float) -> dict[str, Any]:
    student.eval(); b = {key: value[:1].to(device) for key, value in base.items()}; sigma = torch.as_tensor(pool["sigma"][None], dtype=torch.float32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(20261920); a = torch.randn((1, HORIZON, 10), generator=gen); c = torch.randn((1, HORIZON, 10), generator=gen); checks = []
    with torch.no_grad():
        for cut in range(1, HORIZON):
            x = a.clone(); y = a.clone(); y[:, cut:] = c[:, cut:]
            px = student(b, x.to(device), sigma); py = student(b, y.to(device), sigma)
            delta = max(float((px[key][:, :cut] - py[key][:, :cut]).abs().max()) for key in px)
            checks.append({"cut": cut, "max_abs_delta": delta, "passed": delta <= tolerance})
    return {"tolerance": tolerance, "passed": all(x["passed"] for x in checks), "checks": checks}


def _latency(model: Any, student: CausalResidualStudent, pool: Mapping[str, Any], prepared: Mapping[str, Any], base: Mapping[str, torch.Tensor], freeze: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = freeze["evaluation"]["latency"]
    actions = torch.as_tensor(pool["candidates"], dtype=torch.float32, device=device); mu = torch.as_tensor(pool["mu"][None], dtype=torch.float32, device=device); sigma = torch.as_tensor(pool["sigma"][None], dtype=torch.float32, device=device); eps = (actions - mu) / sigma
    model.eval(); student.eval()
    def sync(): torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(int(cfg["warmup"])):
            _rollout(model, prepared["trans0"], actions); one = _rollout(model, prepared["trans0"], mu); student({key: value.expand((300,) + tuple(value.shape[1:])) for key, value in one.items()}, eps, sigma.expand_as(eps))
        teacher_ms, residual_ms = [], []
        for _ in range(int(cfg["repeats"])):
            sync(); start = time.perf_counter(); _rollout(model, prepared["trans0"], actions); sync(); teacher_ms.append((time.perf_counter() - start) * 1000)
            sync(); start = time.perf_counter(); one = _rollout(model, prepared["trans0"], mu); one = {key: value.expand((300,) + tuple(value.shape[1:])) for key, value in one.items()}; student(one, eps, sigma.expand_as(eps)); sync(); residual_ms.append((time.perf_counter() - start) * 1000)
    tm, rm = statistics.median(teacher_ms), statistics.median(residual_ms)
    return {"boundary": "300 teacher rollouts versus one base teacher rollout plus student batch", "teacher_300_ms": teacher_ms, "base_plus_student_ms": residual_ms, "teacher_median_ms": tm, "base_plus_student_median_ms": rm, "reduction": 1 - rm / max(tm, 1e-9)}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, (np.integer, np.floating)): return value.item()
    raise TypeError(type(value).__name__)


def main() -> int:
    args = _args(); args.output.mkdir(parents=True, exist_ok=True)
    try:
        _require_compute_node(); freeze = _load_json(args.freeze.resolve()); wall_root = args.wall_root.resolve(); smoke = _load_smoke(args.smoke_runner, wall_root)
        runtime = smoke._runtime(wall_root); model, device = runtime["model"], runtime["device"]
        for parameter in model.parameters(): parameter.requires_grad_(False)
        preprocessor = smoke._preprocessor(runtime); objective = smoke._objective()
        pools = _load_cases(args.workload.resolve(), args.pools_root.resolve() if args.pools_root else None)
        train = [p for p in pools if p["episode_number"] <= 4]; heldout = [p for p in pools if p["episode_number"] >= 5]
        prepared = {p["case_id"]: _prepare(p, preprocessor, device) for p in pools}; bases, goals = {}, {}
        for p in pools: bases[p["case_id"]], goals[p["case_id"]] = _base_and_goal(model, prepared[p["case_id"]], p["mu"], device)
        shape = bases[train[0]["case_id"]]["visual"].shape; visual_dim = int(shape[-1]); proprio_dim = int(bases[train[0]["case_id"]]["proprio"].shape[-1])
        student = CausalResidualStudent(visual_dim, proprio_dim, 10, int(freeze["student"]["hidden_dim"]), int(freeze["student"]["heads"])).to(device)
        losses = _train(model, student, train, prepared, bases, freeze, device); records = _evaluate(model, student, heldout, prepared, bases, goals, objective, device)
        causal = _causality(student, heldout[0], bases[heldout[0]["case_id"]], device, float(freeze["evaluation"]["causality_tolerance"]))
        latency = _latency(model, student, heldout[0], prepared[heldout[0]["case_id"]], bases[heldout[0]["case_id"]], freeze, device)
        d_s_base = [r["spearman"]["residual"] - r["spearman"]["mean_only"] for r in records]; d_s_shuffle = [r["spearman"]["residual"] - r["spearman"]["shuffled"] for r in records]; d_t_base = [r["top30"]["residual"] - r["top30"]["mean_only"] for r in records]; d_t_shuffle = [r["top30"]["residual"] - r["top30"]["shuffled"] for r in records]
        residual_spearman = [r["spearman"]["residual"] for r in records]
        residual_top30 = [r["top30"]["residual"] for r in records]
        thresholds = freeze["gates"]
        gates = {"finite": all(r["finite"] for r in records), "causal": causal["passed"], "residual_spearman_median": statistics.median(residual_spearman), "residual_top30_median": statistics.median(residual_top30), "median_delta_spearman_vs_mean": statistics.median(d_s_base), "median_delta_spearman_vs_shuffled": statistics.median(d_s_shuffle), "median_delta_top30_vs_mean": statistics.median(d_t_base), "median_delta_top30_vs_shuffled": statistics.median(d_t_shuffle), "latency_reduction": latency["reduction"]}
        gates["mechanism_pass"] = bool(gates["finite"] and gates["causal"] and gates["residual_spearman_median"] >= float(thresholds["residual_spearman_median_min"]) and gates["residual_top30_median"] >= float(thresholds["residual_top30_median_min"]) and gates["median_delta_spearman_vs_mean"] >= float(thresholds["paired_spearman_delta_residual_minus_base_min"]) and gates["median_delta_spearman_vs_shuffled"] >= float(thresholds["paired_spearman_delta_residual_minus_shuffled_min"]) and gates["median_delta_top30_vs_mean"] >= float(thresholds["paired_top30_delta_residual_minus_base_min"]) and gates["median_delta_top30_vs_shuffled"] >= float(thresholds["paired_top30_delta_residual_minus_shuffled_min"]) and gates["latency_reduction"] >= float(thresholds["latency_reduction_min"]))
        result = {"schema": "dino-wm-wall.cem-residual-jepa.stage-a-summary", "status": "complete", "freeze": str(args.freeze.resolve()), "workload": str(args.workload.resolve()), "train_pools": len(train), "heldout_pools": len(heldout), "training": {"updates": len(losses), "first_loss": losses[0], "last10_median": statistics.median(losses[-10:])}, "records": records, "causality": causal, "latency": latency, "gates": gates}
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in student.state_dict().items()}, "freeze_schema": freeze["schema"]}, args.output / "student_checkpoint.pt")
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
        return 0
    except Exception as exc:
        (args.output / "summary.json").write_text(json.dumps({"schema": "dino-wm-wall.cem-residual-jepa.stage-a-summary", "status": "failed", "error": repr(exc)}, indent=2), encoding="utf-8")
        print(f"CEM_RESIDUAL_WALL_STAGE_A_FAILED: {exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

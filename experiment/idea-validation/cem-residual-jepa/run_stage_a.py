#!/usr/bin/env python3
"""Frozen, predictor-only CEM-mean residual screen for DINO-WM PushT.

The official model and manifest helpers are imported lazily from the existing
``jepa-action-prefix-compiler`` experiment.  This file adds only the residual
student and paired block bookkeeping; it never runs a planner or environment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

H = 5
HERE = Path(__file__).resolve().parent
JEPA = HERE.parent / "jepa-action-prefix-compiler"
if str(JEPA) not in sys.path:
    sys.path.insert(0, str(JEPA))


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", type=Path, default=HERE / "FREEZE.json")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--checkpoint-config", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required")
    if not os.environ.get("PBS_NODEFILE") or not Path(os.environ["PBS_NODEFILE"]).is_file():
        raise RuntimeError("PBS_NODEFILE is required")
    host = os.uname().nodename.lower() if hasattr(os, "uname") else "unknown"
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/head/submit host: {host}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


class CausalResidualStudent(nn.Module):
    """Small causal residual predictor; it has no teacher or encoder reference."""

    def __init__(self, visual_patches: int, visual_dim: int, proprio_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.visual_patches = visual_patches
        self.visual_dim = visual_dim
        self.proprio_dim = proprio_dim
        self.base_proj = nn.Linear(visual_dim + proprio_dim, hidden)
        self.eps_proj = nn.Linear(2 * action_dim, hidden)
        self.context_proj = nn.Linear(visual_dim + proprio_dim, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.visual_head = nn.Linear(hidden, visual_patches * visual_dim)
        self.proprio_head = nn.Linear(hidden, proprio_dim)

    def forward(self, base: Mapping[str, torch.Tensor], epsilon: torch.Tensor, sigma: torch.Tensor, context: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base_step = torch.cat((base["visual"].mean(dim=2), base["proprio"]), dim=-1)
        context_step = torch.cat((context["visual"][:, :1].mean(dim=2), context["proprio"][:, :1]), dim=-1).expand(-1, H, -1)
        tokens = self.base_proj(base_step) + self.eps_proj(torch.cat((epsilon, sigma), dim=-1)) + self.context_proj(context_step)
        mask = torch.triu(torch.ones(H, H, dtype=torch.bool, device=tokens.device), diagonal=1)
        hidden = self.norm(self.temporal(tokens, mask=mask))
        return {
            "visual": self.visual_head(hidden).reshape(-1, H, self.visual_patches, self.visual_dim),
            "proprio": self.proprio_head(hidden),
        }


def _repeat(mapping: Mapping[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    return {k: v[:1].expand((count,) + tuple(v.shape[1:])) for k, v in mapping.items()}


def _teacher(model: Any, context: Mapping[str, torch.Tensor], actions: torch.Tensor) -> dict[str, torch.Tensor]:
    from run_dino_pusht_stage_a import _teacher_targets
    return _teacher_targets(model, context, actions)


def _sample(generator: torch.Generator, count: int, action_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn((count, H, action_dim), generator=generator, device="cpu").to(device)


def _mean_stats(model: Any, encoded: Mapping[str, Any], objective: Any, action_dim: int, seed: int, device: torch.device, M: int, K: int, floor: float) -> dict[str, torch.Tensor]:
    """Build fixed initial and one-step teacher-elite Normal distributions."""
    n = int(encoded["count"])
    mu = torch.zeros((n, H, action_dim), dtype=torch.float32)
    sigma = torch.ones_like(mu)
    refined_mu, refined_sigma = [], []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for i in range(n):
        context = {k: v[i:i + 1].to(device) for k, v in encoded["context"].items()}
        goal = {k: v[i:i + 1, -1:].to(device) for k, v in encoded["grounded"].items()}
        candidates = _sample(gen, M, action_dim, torch.device("cuda"))
        with torch.no_grad():
            target = _teacher(model, _repeat(context, M), candidates)
            cost = objective(target, _repeat(goal, M)).reshape(-1)
        elite = candidates.detach().cpu().index_select(0, torch.argsort(cost).detach().cpu()[:K])
        refined_mu.append(elite.mean(0))
        refined_sigma.append(elite.var(0, unbiased=False).clamp_min(floor).sqrt())
    return {
        "initial_mu": mu,
        "initial_sigma": sigma,
        "refined_mu": torch.stack(refined_mu),
        "refined_sigma": torch.stack(refined_sigma),
    }


def _base_rollouts(model: Any, encoded: Mapping[str, Any], distributions: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("initial", "refined"):
        mu = distributions[f"{name}_mu"]
        values = {"visual": [], "proprio": []}
        for i in range(int(encoded["count"])):
            context = {k: v[i:i + 1].to(device) for k, v in encoded["context"].items()}
            with torch.no_grad():
                base = _teacher(model, context, mu[i:i + 1].to(device))
            for key in values:
                values[key].append(base[key].detach().cpu())
        out[name] = {key: torch.cat(value, dim=0) for key, value in values.items()}
    return out


def _latent_loss(pred: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return 0.5 * (torch.nn.functional.mse_loss(pred["visual"], target["visual"]) + torch.nn.functional.mse_loss(pred["proprio"], target["proprio"]))


def _train(model: Any, student: CausalResidualStudent, encoded: Mapping[str, Any], distributions: Mapping[str, torch.Tensor], bases: Mapping[str, Mapping[str, torch.Tensor]], steps: int, batch: int, lr: float, seed: int, device: torch.device) -> list[float]:
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    rng = random.Random(seed)
    eps_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    n = int(encoded["count"])
    losses: list[float] = []
    student.train()
    for step in range(steps):
        name = "initial" if step % 2 == 0 else "refined"
        indices = [rng.randrange(n) for _ in range(batch)]
        idx = torch.tensor(indices, dtype=torch.long)
        context = {k: v.index_select(0, idx).to(device) for k, v in encoded["context"].items()}
        base = {k: v.index_select(0, idx).to(device) for k, v in bases[name].items()}
        mu = distributions[f"{name}_mu"].index_select(0, idx)
        sigma = distributions[f"{name}_sigma"].index_select(0, idx)
        epsilon = torch.randn((batch, H, mu.shape[-1]), generator=eps_rng)
        actions = (mu + sigma * epsilon).to(device)
        with torch.no_grad():
            target = _teacher(model, context, actions)
            residual_target = {k: target[k] - base[k] for k in target}
        prediction = student(base, epsilon.to(device), sigma.to(device), context)
        loss = _latent_loss(prediction, residual_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def _rank(values: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(values.detach().float().cpu(), stable=True), stable=True).float()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _rank(a), _rank(b)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-12)))


def _top30(a: torch.Tensor, b: torch.Tensor) -> float:
    k = min(30, a.numel(), b.numel())
    x = set(torch.argsort(a, stable=True)[:k].tolist())
    y = set(torch.argsort(b, stable=True)[:k].tolist())
    return len(x & y) / max(k, 1)


def _evaluate(model: Any, student: CausalResidualStudent, encoded: Mapping[str, Any], distributions: Mapping[str, torch.Tensor], bases: Mapping[str, Mapping[str, torch.Tensor]], objective: Any, action_dim: int, seeds: Sequence[int], device: torch.device) -> list[dict[str, Any]]:
    student.eval()
    records: list[dict[str, Any]] = []
    for seed in seeds:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        for i in range(8):
            context_one = {k: v[i:i + 1].to(device) for k, v in encoded["context"].items()}
            goal_one = {k: v[i:i + 1, -1:].to(device) for k, v in encoded["grounded"].items()}
            mu = distributions["refined_mu"][i:i + 1]
            sigma = distributions["refined_sigma"][i:i + 1]
            epsilon = torch.randn((300, H, action_dim), generator=gen)
            actions = (mu + sigma * epsilon).to(device)
            context = _repeat(context_one, 300)
            goal = _repeat(goal_one, 300)
            with torch.no_grad():
                teacher_target = _teacher(model, context, actions)
                base = {k: bases["refined"][k][i:i + 1].to(device).expand((300,) + tuple(bases["refined"][k].shape[1:])) for k in bases["refined"]}
                residual = {k: base[k] + student(base, epsilon.to(device), sigma.expand_as(epsilon).to(device), context)[k] for k in base}
                shuffled = epsilon.index_select(0, torch.randperm(300, generator=gen)).to(device)
                shuffled_pred = {k: base[k] + student(base, shuffled, sigma.expand_as(epsilon).to(device), context)[k] for k in base}
                truth_cost = objective(teacher_target, goal).reshape(-1)
                base_cost = objective(base, goal).reshape(-1)
                residual_cost = objective(residual, goal).reshape(-1)
                shuffled_cost = objective(shuffled_pred, goal).reshape(-1)
            records.append({
                "seed": int(seed), "context_index": i, "candidate_count": 300,
                "finite": all(_finite(x) for x in (teacher_target, base, residual, shuffled_pred, truth_cost, base_cost, residual_cost, shuffled_cost)),
                "spearman": {"base": _spearman(truth_cost, base_cost), "residual": _spearman(truth_cost, residual_cost), "shuffled": _spearman(truth_cost, shuffled_cost)},
                "top30": {"base": _top30(truth_cost, base_cost), "residual": _top30(truth_cost, residual_cost), "shuffled": _top30(truth_cost, shuffled_cost)},
                "relative_latent_mse": float(_latent_loss(residual, teacher_target).detach().cpu()),
            })
    return records


def _causality(student: CausalResidualStudent, encoded: Mapping[str, Any], bases: Mapping[str, Mapping[str, torch.Tensor]], distributions: Mapping[str, torch.Tensor], action_dim: int, device: torch.device, tolerance: float) -> dict[str, Any]:
    context = {k: v[:1].to(device) for k, v in encoded["context"].items()}
    base = {k: v[:1].to(device) for k, v in bases["refined"].items()}
    sigma = distributions["refined_sigma"][:1].to(device)
    gen = torch.Generator(device="cpu").manual_seed(20261920)
    a = torch.randn((1, H, action_dim), generator=gen)
    b = torch.randn((1, H, action_dim), generator=gen)
    cases = []
    with torch.no_grad():
        for cut in range(1, H):
            x = a.clone(); y = a.clone(); y[:, cut:] = b[:, cut:]
            p = student(base, x.to(device), sigma, context); q = student(base, y.to(device), sigma, context)
            delta = max((p[k][:, :cut] - q[k][:, :cut]).abs().max().item() for k in p)
            cases.append({"cut": cut, "max_abs_delta": delta, "passed": delta <= tolerance})
    return {"tolerance": tolerance, "passed": all(c["passed"] for c in cases), "cases": cases}


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _latency(model: Any, student: CausalResidualStudent, encoded: Mapping[str, Any], distributions: Mapping[str, torch.Tensor], bases: Mapping[str, Mapping[str, torch.Tensor]], action_dim: int, seed: int, device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    context_one = {k: v[:1].to(device) for k, v in encoded["context"].items()}
    base_one = {k: v[:1].to(device) for k, v in bases["refined"].items()}
    base_batch = {k: v.expand((300,) + tuple(v.shape[1:])) for k, v in base_one.items()}
    context_batch = _repeat(context_one, 300)
    sigma = distributions["refined_sigma"][:1]
    eps = torch.randn((300, H, action_dim), generator=gen)
    actions = (distributions["refined_mu"][:1] + sigma * eps).to(device)
    student.eval(); model.eval()
    def sync() -> None: torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3):
            _teacher(model, context_batch, actions); _teacher(model, context_one, distributions["refined_mu"][:1].to(device)); student(base_batch, eps.to(device), sigma.expand_as(eps).to(device), context_batch)
        teacher_ms, residual_ms = [], []
        for _ in range(10):
            sync(); start = time.perf_counter(); _teacher(model, context_batch, actions); sync(); teacher_ms.append((time.perf_counter() - start) * 1000)
            sync(); start = time.perf_counter(); base_now = _teacher(model, context_one, distributions["refined_mu"][:1].to(device)); base_now = {k: v.expand((300,) + tuple(v.shape[1:])) for k, v in base_now.items()}; student(base_now, eps.to(device), sigma.expand_as(eps).to(device), context_batch); sync(); residual_ms.append((time.perf_counter() - start) * 1000)
    return {"teacher_300_ms": teacher_ms, "base_plus_residual_ms": residual_ms, "teacher_median_ms": _median(teacher_ms), "base_plus_residual_median_ms": _median(residual_ms), "reduction": 1 - _median(residual_ms) / max(_median(teacher_ms), 1e-9), "boundary": "one base teacher rollout + residual batch versus 300 teacher rollouts"}


def _summary(records: Sequence[Mapping[str, Any]], causality: Mapping[str, Any], latency: Mapping[str, Any]) -> dict[str, Any]:
    finite = all(bool(r["finite"]) for r in records)
    residual = [float(r["spearman"]["residual"]) for r in records]
    base = [float(r["spearman"]["base"]) for r in records]
    shuffled = [float(r["spearman"]["shuffled"]) for r in records]
    residual_top = [float(r["top30"]["residual"]) for r in records]
    base_top = [float(r["top30"]["base"]) for r in records]
    shuffled_top = [float(r["top30"]["shuffled"]) for r in records]
    mechanism = {
        "finite": finite,
        "causal": bool(causality["passed"]),
        "median_spearman_delta_vs_base": _median([a - b for a, b in zip(residual, base)]),
        "median_spearman_delta_vs_shuffled": _median([a - b for a, b in zip(residual, shuffled)]),
        "median_top30_delta_vs_base": _median([a - b for a, b in zip(residual_top, base_top)]),
        "median_top30_delta_vs_shuffled": _median([a - b for a, b in zip(residual_top, shuffled_top)]),
    }
    replacement = {"spearman_median": _median(residual), "spearman_min": min(residual), "top30_median": _median(residual_top), "top30_min": min(residual_top)}
    gates = {
        "mechanism": mechanism,
        "replacement": replacement,
        "mechanism_pass": bool(finite and causality["passed"] and mechanism["median_spearman_delta_vs_base"] >= .05 and mechanism["median_spearman_delta_vs_shuffled"] >= .05 and mechanism["median_top30_delta_vs_base"] >= .10 and mechanism["median_top30_delta_vs_shuffled"] >= .10),
        "replacement_pass": bool(replacement["spearman_median"] > .99 and replacement["spearman_min"] > .95 and replacement["top30_median"] > .95 and replacement["top30_min"] > .80),
    }
    gates["decision"] = "GO" if gates["mechanism_pass"] and gates["replacement_pass"] else "NO-GO"
    return {"block_count": len(records), "unit": "context x seed block; candidates are not replicates", "per_block": list(records), "causality": dict(causality), "latency": dict(latency), "gates": gates}


def main() -> int:
    args = _args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    try:
        _require_compute_node()
        freeze = _load_json(args.freeze.resolve())
        if freeze.get("schema") != "dino-wm-pusht.cem-residual-jepa.stage-a-freeze": raise ValueError("unexpected freeze schema")
        asset_root = (args.asset_root or args.root).resolve()
        from run_dino_pusht_grounded_prefix import _RawEpisodeCache, _load_trajectory_datasets, _preencode_manifest
        from run_dino_pusht_query_coverage import _load_query_manifest
        from run_dino_pusht_stage_a import _load_official, _objective_cost, _require_assets, _set_seed
        _require_assets(asset_root, freeze); _set_seed(int(freeze["randomness"]["training_seed"]))
        manifest = _load_query_manifest(args.manifest, {"frame_skip": H})
        model, workspace, _, _, objective, action_dim, device = _load_official(args.root.resolve(), asset_root, freeze, output, 99, args.config, args.checkpoint, args.checkpoint_config, args.data_root)
        train_dset, heldout_dset = _load_trajectory_datasets(args.root.resolve(), asset_root, freeze, args.checkpoint, args.checkpoint_config)
        cache = _RawEpisodeCache(8)
        train = _preencode_manifest(train_dset, "train", manifest["splits"]["train"]["examples"], H, 8, cache, model, device)
        heldout = _preencode_manifest(heldout_dset, "heldout", manifest["splits"]["heldout"]["examples"], H, 8, cache, model, device)
        train_dist = _mean_stats(model, train, objective, action_dim, freeze["randomness"]["refined_cem_seed"], device, 64, 8, .05)
        heldout_dist = _mean_stats(model, heldout, objective, action_dim, freeze["randomness"]["refined_cem_seed"] + 77, device, 64, 8, .05)
        train_base = _base_rollouts(model, train, train_dist, device); heldout_base = _base_rollouts(model, heldout, heldout_dist, device)
        student = CausalResidualStudent(int(train_base["initial"]["visual"].shape[2]), int(train_base["initial"]["visual"].shape[3]), int(train_base["initial"]["proprio"].shape[-1]), action_dim, int(freeze["student"]["hidden_dim"])).to(device)
        losses = _train(model, student, train, train_dist, train_base, 500, 32, 3e-4, freeze["randomness"]["training_seed"], device)
        records = _evaluate(model, student, heldout, heldout_dist, heldout_base, objective, action_dim, freeze["randomness"]["heldout_seeds"], device)
        causality = _causality(student, heldout, heldout_base, heldout_dist, action_dim, device, 1e-6)
        latency = _latency(model, student, heldout, heldout_dist, heldout_base, action_dim, freeze["randomness"]["timing_seed"], device)
        result = _summary(records, causality, latency)
        result.update({"schema": "dino-wm-pusht.cem-residual-jepa.stage-a-summary", "freeze": str(args.freeze.resolve()), "manifest": str(args.manifest.resolve()), "training": {"steps": 500, "first_loss": losses[0], "last10_median": statistics.median(losses[-10:]), "last10_to_first": statistics.median(losses[-10:]) / max(losses[0], 1e-12)}, "seeds": freeze["randomness"]})
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()}, "freeze_schema": freeze["schema"]}, output / "student_checkpoint.pt")
        (output / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
        return 0
    except Exception as exc:
        (output / "summary.json").write_text(json.dumps({"schema": "dino-wm-pusht.cem-residual-jepa.stage-a-summary", "status": "failed", "error": repr(exc)}, indent=2), encoding="utf-8")
        print(f"CEM_RESIDUAL_STAGE_A_FAILED: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

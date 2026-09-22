#!/usr/bin/env python3
"""Predictor-level validation of the LeWM PushT compiled-world-model idea."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent / "02-horizon-weighted-recurrent-student" / "lewm-transfer"
ANCHOR_SCRIPT = TRANSFER / "anchor-aligned-bank" / "run_anchor_aligned_bank.py"
SCHEMA = "lewm.compiled-world-model.summary"
HORIZON = 5
ACTION_DIM = 10
LATENT_DIM = 192
HIDDEN_DIM = 256
RANKS = (32, 96, 192)
TRAIN_STEPS = 1500
BATCH_CONTEXTS = 4
NUM_CANDIDATES = 300
ELITE_COUNT = 30
DEV_SLICE = (552, 560)
TEST_SLICE = (592, 600)
DEV_SEEDS = (20300967, 20300968)
TEST_SEEDS = (20301201, 20301202)
SELECTION_SEED = 20300903
INIT_SEED = 20301203
SCHEDULE_SEED = 20301204
HORIZON_WEIGHTS = (1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-512", type=Path, required=True)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--temporal-freeze", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("status", "run"), default="status")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def load_anchor_module() -> Any:
    spec = importlib.util.spec_from_file_location("compiled_anchor_reference", ANCHOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(ANCHOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_freeze(value: Mapping[str, Any]) -> None:
    if value.get("schema") != "lewm.compiled-world-model.freeze" or int(value.get("schema_version", -1)) != 1:
        raise ValueError("unexpected freeze schema")
    if value.get("status") != "frozen_before_results":
        raise ValueError("freeze was not locked before results")
    if value["models"]["ranks"] != list(RANKS):
        raise ValueError("rank scan drifted")
    training = value["training"]
    for key, expected in {
        "updates": TRAIN_STEPS,
        "batch_contexts": BATCH_CONTEXTS,
        "initialization_seed": INIT_SEED,
        "schedule_seed": SCHEDULE_SEED,
    }.items():
        if int(training.get(key, -1)) != expected:
            raise ValueError(f"training contract drifted: {key}")
    data = value["data"]
    if data.get("development_slice") != "valid[552:560]" or data.get("final_test_slice") != "valid[592:600]":
        raise ValueError("evaluation slices drifted")
    scope = value["scope"]
    if scope.get("official_cem") != "NOT_RUN_BY_SCOPE" or scope.get("closed_loop") != "NOT_RUN_BY_SCOPE":
        raise ValueError("scope expanded beyond predictor-level")


def prefix_features(actions: Any) -> Any:
    """Return causal padded prefixes plus horizon one-hot: [B,C,H,55]."""
    import torch

    if actions.ndim != 4 or tuple(actions.shape[-2:]) != (HORIZON, ACTION_DIM):
        raise ValueError(f"actions must be [B,C,5,10], got {tuple(actions.shape)}")
    mask = torch.tril(torch.ones((HORIZON, HORIZON), device=actions.device, dtype=actions.dtype))
    expanded = actions.unsqueeze(2) * mask.reshape(1, 1, HORIZON, HORIZON, 1)
    onehot = torch.eye(HORIZON, device=actions.device, dtype=actions.dtype).reshape(1, 1, HORIZON, HORIZON)
    onehot = onehot.expand(actions.shape[0], actions.shape[1], -1, -1)
    return torch.cat((expanded.reshape(*actions.shape[:2], HORIZON, HORIZON * ACTION_DIM), onehot), dim=-1)


class DensePrefixModel:
    def __new__(cls) -> Any:
        import torch.nn as nn

        class Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(LATENT_DIM + HORIZON * ACTION_DIM + HORIZON),
                    nn.Linear(LATENT_DIM + HORIZON * ACTION_DIM + HORIZON, HIDDEN_DIM),
                    nn.GELU(),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                    nn.GELU(),
                    nn.Linear(HIDDEN_DIM, LATENT_DIM),
                )

            def predict_batch(self, contexts: Any, actions: Any) -> Any:
                if contexts.ndim != 3 or contexts.shape[-1] != LATENT_DIM:
                    raise ValueError("contexts must be [B,L,192]")
                features = prefix_features(actions)
                current = contexts[:, -1].reshape(contexts.shape[0], 1, 1, LATENT_DIM)
                current = current.expand(-1, actions.shape[1], HORIZON, -1)
                return self.net(__import__("torch").cat((current, features), dim=-1))

            def forward(self, contexts: Any, actions: Any) -> Any:
                if actions.ndim != 3:
                    raise ValueError("forward actions must be [C,5,10]")
                return self.predict_batch(contexts[:1], actions.unsqueeze(0))[0]

        return Impl()


class FactorizedModel:
    def __new__(cls, rank: int, context_basis: bool) -> Any:
        import torch
        import torch.nn as nn

        class Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rank = int(rank)
                self.context_basis = bool(context_basis)
                compiler_out = HORIZON * LATENT_DIM * (1 + self.rank) if self.context_basis else HORIZON * LATENT_DIM
                self.compiler = nn.Sequential(
                    nn.LayerNorm(LATENT_DIM),
                    nn.Linear(LATENT_DIM, HIDDEN_DIM),
                    nn.GELU(),
                    nn.Linear(HIDDEN_DIM, compiler_out),
                )
                if not self.context_basis:
                    self.global_basis = nn.Parameter(torch.empty(HORIZON, LATENT_DIM, self.rank))
                    nn.init.normal_(self.global_basis, std=1.0 / math.sqrt(LATENT_DIM))
                self.action_net = nn.Sequential(
                    nn.LayerNorm(HORIZON * ACTION_DIM + HORIZON),
                    nn.Linear(HORIZON * ACTION_DIM + HORIZON, HIDDEN_DIM),
                    nn.GELU(),
                    nn.Linear(HIDDEN_DIM, self.rank),
                )

            def prepare(self, contexts: Any) -> tuple[Any, Any]:
                current = contexts[:, -1]
                raw = self.compiler(current)
                if self.context_basis:
                    raw = raw.reshape(current.shape[0], HORIZON, LATENT_DIM, self.rank + 1)
                    return raw[..., 0], raw[..., 1:]
                bias = raw.reshape(current.shape[0], HORIZON, LATENT_DIM)
                basis = self.global_basis.unsqueeze(0).expand(current.shape[0], -1, -1, -1)
                return bias, basis

            def action_features(self, actions: Any) -> Any:
                return self.action_net(prefix_features(actions))

            def query(self, prepared: tuple[Any, Any], actions: Any) -> Any:
                bias, basis = prepared
                phi = self.action_features(actions)
                return bias[:, None] + torch.einsum("bhdr,bchr->bchd", basis, phi)

            def predict_batch(self, contexts: Any, actions: Any) -> Any:
                return self.query(self.prepare(contexts), actions)

            def forward(self, contexts: Any, actions: Any) -> Any:
                if actions.ndim != 3:
                    raise ValueError("forward actions must be [C,5,10]")
                return self.predict_batch(contexts[:1], actions.unsqueeze(0))[0]

        return Impl()


def make_model(label: str) -> Any:
    if label == "B1_dense":
        return DensePrefixModel()
    family, rank_text = label.split("_r")
    return FactorizedModel(int(rank_text), context_basis=family == "B3")


def synthetic_e0(freeze: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    torch.manual_seed(INIT_SEED)
    actions_a = torch.randn(2, 7, HORIZON, ACTION_DIM)
    actions_b = actions_a.clone()
    actions_b[:, :, 3:] = torch.randn_like(actions_b[:, :, 3:])
    features_a = prefix_features(actions_a)
    features_b = prefix_features(actions_b)
    causal = float((features_a[:, :, :3] - features_b[:, :, :3]).abs().max())
    model = FactorizedModel(32, True).eval()
    contexts = torch.randn(2, 1, LATENT_DIM)
    with torch.no_grad():
        prepared = model.prepare(contexts)
        whole = model.query(prepared, actions_a)
        chunked = torch.cat((model.query(prepared, actions_a[:, :3]), model.query(prepared, actions_a[:, 3:])), dim=1)
        swapped = model.predict_batch(contexts.flip(0), actions_a)
    chunk_error = float((whole - chunked).abs().max())
    context_effect = float((whole - swapped).abs().mean())
    dtype_results: dict[str, Any] = {}
    for dtype in (torch.float64, torch.float32):
        generator = torch.Generator().manual_seed(INIT_SEED + (0 if dtype == torch.float64 else 1))
        b = torch.randn(11, LATENT_DIM, dtype=dtype, generator=generator)
        basis = torch.randn(11, LATENT_DIM, 32, dtype=dtype, generator=generator)
        phi = torch.randn(11, 32, dtype=dtype, generator=generator)
        goal = torch.randn(11, LATENT_DIM, dtype=dtype, generator=generator)
        delta = b - goal
        explicit = (delta + torch.einsum("bdr,br->bd", basis, phi)).square().sum(-1)
        q = torch.einsum("bdr,bds->brs", basis, basis)
        p = torch.einsum("bdr,bd->br", basis, delta)
        k = delta.square().sum(-1)
        compiled = torch.einsum("br,brs,bs->b", phi, q, phi) + 2 * (phi * p).sum(-1) + k
        dtype_results[str(dtype)] = {
            "max_abs": float((explicit - compiled).abs().max()),
            "argmin_match": int(explicit.argmin()) == int(compiled.argmin()),
        }
    cfg = freeze["e0"]
    passed = (
        causal <= float(cfg["causal_prefix_tolerance"])
        and chunk_error <= float(cfg["batch_chunk_tolerance"])
        and context_effect > 0
        and dtype_results["torch.float64"]["max_abs"] <= float(cfg["fp64_compiled_cost_abs_tolerance"])
        and dtype_results["torch.float32"]["max_abs"] <= float(cfg["fp32_compiled_cost_abs_tolerance"])
        and all(item["argmin_match"] for item in dtype_results.values())
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "causal_prefix_max_abs": causal,
        "batch_chunk_max_abs": chunk_error,
        "context_swap_mean_abs_effect": context_effect,
        "synthetic_compiled_cost": dtype_results,
    }


def training_schedule(row_count: int) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(SCHEDULE_SEED)
    return torch.randint(0, row_count, (TRAIN_STEPS, BATCH_CONTEXTS), generator=generator)


def train_model(label: str, rows: Sequence[Mapping[str, Any]], schedule: Any, output: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.manual_seed(INIT_SEED)
    model = make_model(label).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = torch.tensor(HORIZON_WEIGHTS, dtype=torch.float32, device="cuda")
    trace = []
    started = time.perf_counter()
    for step in range(TRAIN_STEPS):
        indices = [int(x) for x in schedule[step]]
        contexts = torch.cat([rows[index]["latent_history"] for index in indices], dim=0).to("cuda", dtype=torch.float32)
        actions = torch.stack([rows[index]["future_actions"] for index in indices], dim=0).to("cuda", dtype=torch.float32)
        targets = torch.stack([rows[index]["teacher_targets"] for index in indices], dim=0).to("cuda", dtype=torch.float32)
        prediction = model.predict_batch(contexts, actions)
        per_horizon = (prediction - targets).square().mean(dim=(0, 1, 3))
        loss = (per_horizon * weights).mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite loss for {label} at {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, 499, 999, 1499):
            trace.append({"step": step + 1, "loss": float(loss.detach().cpu()), "per_horizon_mse": [float(x) for x in per_horizon.detach().cpu()]})
    torch.cuda.synchronize()
    checkpoint = output / f"{label}_step1500.pt"
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "label": label}, checkpoint)
    first = trace[0]["loss"]
    last = trace[-1]["loss"]
    return model.eval(), {
        "label": label,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "updates": TRAIN_STEPS,
        "trace": trace,
        "terminal_to_initial_loss_ratio": last / max(first, 1e-12),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }


def select_episode_ids(dataset: Path, manifest: Mapping[str, Any], selected_slice: tuple[int, int]) -> tuple[list[int], dict[str, Any]]:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(dataset, "r") as handle:
        lengths = [int(x) for x in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON * 5 + 1]
    random.Random(SELECTION_SEED).shuffle(valid)
    prefix = [int(row["episode_id"]) for row in manifest["splits"]["heldout"] + manifest["splits"]["train"]]
    if valid[: len(prefix)] != prefix:
        raise ValueError("manifest selection prefix cannot be reproduced")
    selected = valid[selected_slice[0] : selected_slice[1]]
    if len(selected) != 8 or set(selected) & set(valid[: selected_slice[0]]):
        raise ValueError("fresh selection overlaps excluded prefix")
    return selected, {
        "selection_seed": SELECTION_SEED,
        "selection_slice": f"valid[{selected_slice[0]}:{selected_slice[1]}]",
        "excluded_valid_prefix": selected_slice[0],
        "episode_ids": selected,
        "result_dependent_selection": False,
    }


def build_rows(anchor: Any, official: Any, dataset: Path, ids: Sequence[int], temporal: Mapping[str, Any], seeds: tuple[int, int]) -> tuple[list[Any], dict[str, Any]]:
    previous = tuple(anchor.FRESH_SEEDS)
    anchor.FRESH_SEEDS = seeds
    try:
        rows, metadata = anchor.build_fresh_rows(anchor.load_reference(), official, dataset, ids, temporal)
    finally:
        anchor.FRESH_SEEDS = previous
    if len(rows) != 24 or int(metadata.get("blocks", -1)) != 48:
        raise ValueError("fresh row contract drifted")
    return rows, metadata


def aggregate(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("empty metric group")
    output = {"blocks": len(blocks), "finite": all(bool(item["finite"]) for item in blocks)}
    for key in ("spearman", "top30_overlap", "relative_latent_mse"):
        values = [float(item[key]) for item in blocks]
        output[f"{key}_median"] = float(statistics.median(values))
        output[f"{key}_minimum"] = min(values)
        output[f"{key}_maximum"] = max(values)
    return output


def evaluate_model(reference: Any, official: Any, model: Any, rows: Sequence[Mapping[str, Any]], label: str, seeds: tuple[int, int]) -> dict[str, Any]:
    import torch

    blocks = []
    for row in rows:
        for block, seed in enumerate(seeds):
            actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
            target = reference.base._tensor(row["teacher_targets"][block], dtype=torch.float32).to("cuda")
            context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda").expand(actions.shape[0], -1, -1)
            with torch.no_grad():
                prediction = model(context, actions)
                cost = reference.base._official_objective(official, row, prediction)
            teacher = reference.base._tensor(row["teacher_objective"][block], dtype=torch.float32).to("cuda")
            item = {
                "label": label,
                "episode_id": int(row["episode_id"]),
                "context_id": row["context_id"],
                "stratum": row["stratum"],
                "action_prefix_seed": int(seed),
                "spearman": reference.base._spearman(teacher, cost),
                "top30_overlap": reference.base._topk_overlap(teacher, cost, ELITE_COUNT),
                "relative_latent_mse": float(((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).cpu()),
                "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(cost).all().item()),
            }
            blocks.append(item)
    grouped: dict[int, list[Any]] = defaultdict(list)
    for item in blocks:
        grouped[int(item["episode_id"])].append(item)
    return {
        "label": label,
        "overall": aggregate(blocks),
        "episodes": {str(key): aggregate(value) for key, value in sorted(grouped.items())},
        "strata": {name: aggregate([item for item in blocks if item["stratum"] == name]) for name in ("early", "middle", "late")},
        "per_block": blocks,
    }


def select_rank(dev: Mapping[str, Mapping[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    ranking = []
    for rank in RANKS:
        metrics = dev[f"B3_r{rank}"]["overall"]
        ranking.append({
            "rank": rank,
            "spearman_median": float(metrics["spearman_median"]),
            "top30_overlap_median": float(metrics["top30_overlap_median"]),
            "relative_latent_mse_median": float(metrics["relative_latent_mse_median"]),
        })
    chosen = max(ranking, key=lambda item: (item["spearman_median"], item["top30_overlap_median"], -item["relative_latent_mse_median"], -item["rank"]))
    return int(chosen["rank"]), ranking


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(x) for x in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def time_call(fn: Any) -> float:
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(); fn(); end.record(); torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def fit_official_quadratic(reference: Any, official: Any, model: Any, row: Mapping[str, Any], block: int, tolerance: float) -> dict[str, Any]:
    import torch

    actions = reference.base._tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
    context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
    goal = reference.base._tensor(row["goal_emb"], dtype=torch.float32).to("cuda").reshape(-1, LATENT_DIM)[-1]
    with torch.no_grad():
        prepared = model.prepare(context)
        prediction = model.query(prepared, actions.unsqueeze(0))[0]
        official_cost = reference.base._official_objective(official, row, prediction)
        raw = (prediction[:, -1] - goal).square().sum(-1)
        centered = raw - raw.mean()
        scale = ((centered * (official_cost - official_cost.mean())).sum() / centered.square().sum().clamp_min(1e-12))
        offset = official_cost.mean() - scale * raw.mean()
        residual = official_cost - (scale * raw + offset)
        bias, basis = prepared
        phi = model.action_features(actions.unsqueeze(0))[:, :, -1]
        delta = bias[:, -1] - goal.reshape(1, LATENT_DIM)
        terminal_basis = basis[:, -1]
        q = torch.einsum("bdr,bds->brs", terminal_basis, terminal_basis)
        p = torch.einsum("bdr,bd->br", terminal_basis, delta)
        k = delta.square().sum(-1)
        quadratic = torch.einsum("bcr,brs,bcs->bc", phi, q, phi) + 2 * torch.einsum("bcr,br->bc", phi, p) + k[:, None]
        compiled = scale * quadratic[0] + offset
        explicit_error = float((compiled - official_cost).abs().max().cpu())
        argmin_match = int(compiled.argmin()) == int(official_cost.argmin())
    return {
        "scale": float(scale.cpu()),
        "offset": float(offset.cpu()),
        "official_terminal_affine_max_abs": float(residual.abs().max().cpu()),
        "compiled_official_max_abs": explicit_error,
        "argmin_match": argmin_match,
        "status": "PASS" if float(residual.abs().max().cpu()) <= tolerance and explicit_error <= tolerance and argmin_match else "FAIL",
    }


def timing(reference: Any, official: Any, models: Mapping[str, Any], selected_rank: int, rows: Sequence[Mapping[str, Any]], repeats: int = 5) -> dict[str, Any]:
    import torch

    b1 = models["B1_dense"]
    b3 = models[f"B3_r{selected_rank}"]
    names = ("official_teacher_30", "B1_dense_30", "B3_explicit_30", "B4_compiled_30")
    samples = {name: [] for name in names}
    breakdown = {name: [] for name in ("B3_prepare", "B3_query300", "B4_compile", "B4_score300")}
    for row in rows[:8]:
        actions = reference.base._tensor(row["future_actions"][0], dtype=torch.float32).to("cuda")
        context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda")
        expanded = context.expand(actions.shape[0], -1, -1)
        goal = reference.base._tensor(row["goal_emb"], dtype=torch.float32).to("cuda").reshape(-1, LATENT_DIM)[-1]

        def teacher30() -> Any:
            value = None
            with torch.no_grad():
                for _ in range(30):
                    pred = reference.base.official_teacher_targets(official, expanded, actions)
                    value = reference.base._official_objective(official, row, pred)
            return value

        def dense30() -> Any:
            value = None
            with torch.no_grad():
                for _ in range(30):
                    pred = b1(expanded, actions)
                    value = reference.base._official_objective(official, row, pred)
            return value

        def explicit30() -> Any:
            value = None
            with torch.no_grad():
                prepared = b3.prepare(context)
                for _ in range(30):
                    pred = b3.query(prepared, actions.unsqueeze(0))[0]
                    value = reference.base._official_objective(official, row, pred)
            return value

        def compiled30() -> Any:
            value = None
            with torch.no_grad():
                bias, basis = b3.prepare(context)
                delta = bias[:, -1] - goal.reshape(1, LATENT_DIM)
                terminal_basis = basis[:, -1]
                q = torch.einsum("bdr,bds->brs", terminal_basis, terminal_basis)
                p = torch.einsum("bdr,bd->br", terminal_basis, delta)
                k = delta.square().sum(-1)
                for _ in range(30):
                    phi = b3.action_features(actions.unsqueeze(0))[:, :, -1]
                    value = torch.einsum("bcr,brs,bcs->bc", phi, q, phi) + 2 * torch.einsum("bcr,br->bc", phi, p) + k[:, None]
            return value

        functions = {"official_teacher_30": teacher30, "B1_dense_30": dense30, "B3_explicit_30": explicit30, "B4_compiled_30": compiled30}
        for _ in range(3):
            for function in functions.values():
                time_call(function)
        for _ in range(repeats):
            order = list(names)
            random.Random(20301205 + len(samples[names[0]])).shuffle(order)
            for name in order:
                samples[name].append(time_call(functions[name]))
        with torch.no_grad():
            prepared = b3.prepare(context)
            bias, basis = prepared
            delta = bias[:, -1] - goal.reshape(1, LATENT_DIM)
        breakdown["B3_prepare"].append(time_call(lambda: b3.prepare(context)))
        breakdown["B3_query300"].append(time_call(lambda: b3.query(prepared, actions.unsqueeze(0))))
        breakdown["B4_compile"].append(time_call(lambda: (
            torch.einsum("bdr,bds->brs", basis[:, -1], basis[:, -1]),
            torch.einsum("bdr,bd->br", basis[:, -1], delta),
            delta.square().sum(-1),
        )))
        q = torch.einsum("bdr,bds->brs", basis[:, -1], basis[:, -1])
        p = torch.einsum("bdr,bd->br", basis[:, -1], delta)
        k = delta.square().sum(-1)
        phi = b3.action_features(actions.unsqueeze(0))[:, :, -1]
        breakdown["B4_score300"].append(time_call(lambda: torch.einsum("bcr,brs,bcs->bc", phi, q, phi) + 2 * torch.einsum("bcr,br->bc", phi, p) + k[:, None]))
    summary = {name: {"p50_ms": float(statistics.median(values)), "p95_ms": percentile(values, 0.95), "samples": len(values)} for name, values in samples.items()}
    summary["breakdown"] = {name: {"p50_ms": float(statistics.median(values)), "p95_ms": percentile(values, 0.95)} for name, values in breakdown.items()}
    summary["speedups"] = {
        "B3_explicit_vs_B1_p50": summary["B1_dense_30"]["p50_ms"] / summary["B3_explicit_30"]["p50_ms"],
        "B4_compiled_vs_B3_explicit_p50": summary["B3_explicit_30"]["p50_ms"] / summary["B4_compiled_30"]["p50_ms"],
        "B4_compiled_vs_official_teacher_p50": summary["official_teacher_30"]["p50_ms"] / summary["B4_compiled_30"]["p50_ms"],
    }
    return summary


def final_gates(freeze: Mapping[str, Any], final: Mapping[str, Any], selected_rank: int, timing_result: Mapping[str, Any], e0_trained: Mapping[str, Any]) -> dict[str, Any]:
    b1 = final["B1_dense"]
    b2 = final[f"B2_r{selected_rank}"]
    b3 = final[f"B3_r{selected_rank}"]
    cfg = freeze["final_gates"]
    quality_cfg = cfg["absolute_quality"]
    quality = {
        "spearman_median_min": b3["overall"]["spearman_median"] >= float(quality_cfg["spearman_median_min"]),
        "top30_overlap_median_min": b3["overall"]["top30_overlap_median"] >= float(quality_cfg["top30_overlap_median_min"]),
        "relative_latent_mse_median_max": b3["overall"]["relative_latent_mse_median"] <= float(quality_cfg["relative_latent_mse_median_max"]),
        "finite_required": bool(b3["overall"]["finite"]),
    }
    paired = []
    for episode in sorted(b3["episodes"], key=int):
        b2e, b3e = b2["episodes"][episode], b3["episodes"][episode]
        ds = float(b3e["spearman_median"] - b2e["spearman_median"])
        dt = float(b3e["top30_overlap_median"] - b2e["top30_overlap_median"])
        paired.append({"episode_id": int(episode), "spearman_delta": ds, "top30_delta": dt, "joint_strict_improvement": (ds > 0 and dt >= 0) or (dt > 0 and ds >= 0)})
    mechanism_cfg = cfg["context_dependence_B3_vs_B2"]
    mechanism = {
        "episode_median_spearman_delta": float(statistics.median(item["spearman_delta"] for item in paired)),
        "episode_median_top30_delta": float(statistics.median(item["top30_delta"] for item in paired)),
        "joint_strict_improvement_episodes": sum(bool(item["joint_strict_improvement"]) for item in paired),
        "per_episode": paired,
    }
    mechanism["conditions"] = {
        "episode_median_spearman_delta_min": mechanism["episode_median_spearman_delta"] >= float(mechanism_cfg["episode_median_spearman_delta_min"]),
        "episode_median_top30_delta_min": mechanism["episode_median_top30_delta"] >= float(mechanism_cfg["episode_median_top30_delta_min"]),
        "joint_strict_improvement_episodes_min": mechanism["joint_strict_improvement_episodes"] >= int(mechanism_cfg["joint_strict_improvement_episodes_min"]),
    }
    dominate_cfg = cfg["not_dominated_by_B1"]
    not_dominated = {
        "spearman_within_tolerance": b3["overall"]["spearman_median"] >= b1["overall"]["spearman_median"] - float(dominate_cfg["spearman_median_tolerance"]),
        "top30_within_tolerance": b3["overall"]["top30_overlap_median"] >= b1["overall"]["top30_overlap_median"] - float(dominate_cfg["top30_median_tolerance"]),
        "thirty_query_latency_lower": timing_result["B3_explicit_30"]["p50_ms"] < timing_result["B1_dense_30"]["p50_ms"],
    }
    efficiency_cfg = cfg["efficiency"]
    speed = timing_result["speedups"]
    efficiency = {
        "B3_explicit_vs_B1": speed["B3_explicit_vs_B1_p50"] >= float(efficiency_cfg["B3_explicit_thirty_query_speedup_vs_B1_min"]),
        "B4_compiled_vs_B3": speed["B4_compiled_vs_B3_explicit_p50"] >= float(efficiency_cfg["B4_compiled_thirty_query_speedup_vs_B3_explicit_min"]),
        "B4_compiled_vs_teacher": speed["B4_compiled_vs_official_teacher_p50"] >= float(efficiency_cfg["B4_compiled_thirty_query_speedup_vs_official_teacher_min"]),
        "B3_p95_not_worse_than_B1": timing_result["B3_explicit_30"]["p95_ms"] <= timing_result["B1_dense_30"]["p95_ms"],
        "B4_p95_not_worse_than_B3": timing_result["B4_compiled_30"]["p95_ms"] <= timing_result["B3_explicit_30"]["p95_ms"],
    }
    groups = {
        "e0_trained_compilation": e0_trained.get("status") == "PASS",
        "absolute_quality": all(quality.values()),
        "context_dependence": all(mechanism["conditions"].values()),
        "not_dominated_by_B1": all(not_dominated.values()),
        "efficiency": all(efficiency.values()),
    }
    return {
        "status": "GO" if all(groups.values()) else "NO-GO",
        "groups": groups,
        "absolute_quality": {"conditions": quality},
        "context_dependence_B3_vs_B2": mechanism,
        "not_dominated_by_B1": {"conditions": not_dominated},
        "efficiency": {"conditions": efficiency, "speedups": speed},
    }


def run(args: argparse.Namespace, freeze: Mapping[str, Any]) -> dict[str, Any]:
    require_compute_node()
    import torch

    anchor = load_anchor_module()
    reference = anchor.load_reference()
    official = reference.base.load_official_checkpoint(args.stablewm_home.resolve())
    official.requires_grad_(False)
    training_rows = torch.load(args.training_rows.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(training_rows, list) or len(training_rows) != 512:
        raise ValueError("training rows must contain 512 contexts")
    for row in training_rows:
        if tuple(row["future_actions"].shape) != (64, HORIZON, ACTION_DIM) or tuple(row["teacher_targets"].shape) != (64, HORIZON, LATENT_DIM):
            raise ValueError("training row shape drifted")
    manifest = load_json(args.manifest_512.resolve())
    temporal = load_json(args.temporal_freeze.resolve())
    e0 = synthetic_e0(freeze)
    if e0["status"] != "PASS":
        raise RuntimeError(f"E0 failed before model training: {e0}")
    schedule = training_schedule(len(training_rows))
    labels = ["B1_dense"] + [f"{family}_r{rank}" for rank in RANKS for family in ("B2", "B3")]
    models: dict[str, Any] = {}
    training: dict[str, Any] = {}
    for label in labels:
        print(f"TRAIN_START {label}", flush=True)
        models[label], training[label] = train_model(label, training_rows, schedule, args.output.resolve())
        print(f"TRAIN_DONE {label}", flush=True)
    dev_ids, dev_selection = select_episode_ids(args.dataset.resolve(), manifest, DEV_SLICE)
    test_ids, test_selection = select_episode_ids(args.dataset.resolve(), manifest, TEST_SLICE)
    dev_rows, dev_meta = build_rows(anchor, official, args.dataset.resolve(), dev_ids, temporal, DEV_SEEDS)
    test_rows, test_meta = build_rows(anchor, official, args.dataset.resolve(), test_ids, temporal, TEST_SEEDS)
    dev = {label: evaluate_model(reference, official, model, dev_rows, label, DEV_SEEDS) for label, model in models.items()}
    selected_rank, rank_table = select_rank(dev)
    final_labels = ("B1_dense", f"B2_r{selected_rank}", f"B3_r{selected_rank}")
    final = {label: evaluate_model(reference, official, models[label], test_rows, label, TEST_SEEDS) for label in final_labels}
    trained_compile = fit_official_quadratic(reference, official, models[f"B3_r{selected_rank}"], test_rows[0], 0, float(freeze["e0"]["trained_b3_compiled_cost_abs_tolerance"]))
    timing_result = timing(reference, official, models, selected_rank, test_rows)
    gates = final_gates(freeze, final, selected_rank, timing_result, trained_compile)
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "verdict": gates["status"],
        "source": {
            "lewm_root": str(args.lewm_root.resolve()),
            "stablewm_home": str(args.stablewm_home.resolve()),
            "dataset": str(args.dataset.resolve()),
            "training_rows": str(args.training_rows.resolve()),
            "freeze": str(args.freeze.resolve()),
            "protocol": str(args.protocol.resolve()),
        },
        "scope": {"official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE", "joint_encoder_training": "NOT_RUN_BY_SCOPE"},
        "e0": {"synthetic": e0, "trained_selected_B3": trained_compile},
        "training": training,
        "development": {"selection": dev_selection, "metadata": dev_meta, "rank_rule": freeze["rank_selection"], "rank_table": rank_table, "selected_rank": selected_rank, "evaluations": dev},
        "final_test": {"selection": test_selection, "metadata": test_meta, "evaluations": final},
        "timing": timing_result,
        "gates": gates,
        "claim_boundary": "Predictor-level fixed-observation multi-query evidence only; no official CEM, full-planner, closed-loop, or encoder speedup claim.",
    }
    write_json(args.output.resolve() / "compiled_world_model_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = load_json(args.freeze.resolve())
    validate_freeze(freeze)
    if args.mode == "status":
        value = {"schema": SCHEMA, "status": "READY", "ranks": list(RANKS), "training_arms": 7, "development": "valid[552:560]", "final_test": "valid[592:600]", "official_cem": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "preflight_status.json", value)
        print(json.dumps(value))
        return 0
    for path in (args.lewm_root, args.stablewm_home, args.dataset, args.manifest_512, args.training_rows, args.temporal_freeze, args.protocol):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = run(args, freeze)
    print(json.dumps({"status": summary["status"], "verdict": summary["verdict"], "selected_rank": summary["development"]["selected_rank"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

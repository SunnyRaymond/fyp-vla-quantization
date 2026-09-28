#!/usr/bin/env python3
"""Frozen LeWM goal-query predictor pilot; all model/data work requires PBS."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = ("scalar", "scalar_aux", "projected_tail", "full_latent")
DIAGNOSTICS = ("projected_only", "exact_projected_only", "oracle_y_learned_tail")
LATENT_DIM = 192
HORIZON = 5
ACTION_DIM = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "transfer-root", "lewm-root", "stablewm-root", "stablewm-home",
        "dataset", "training-rows", "manifest", "temporal-freeze", "freeze", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> str:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_JOBID and an existing PBS_NODEFILE are required")
    host = socket.gethostname().split(".", 1)[0].lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit node: {host}")
    allocated = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in allocated:
        raise RuntimeError(f"current host {host} is not listed in PBS_NODEFILE")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not devices or devices == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not assigned by PBS")
    return host


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, value: Mapping[str, Any], atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + ".tmp") if atomic else path
    target.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if atomic:
        target.replace(path)


def append_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def file_identity(path: Path) -> dict[str, Any]:
    value = path.resolve()
    return {"path": str(value), "size_bytes": value.stat().st_size}


def source_commit(root: Path, expected: str) -> dict[str, Any]:
    root = root.resolve()
    if not (root / ".git").exists():
        return {
            "path": str(root), "declared_source_commit": expected,
            "current_head": None, "current_head_verified": False,
            "identity_basis": "source archive has no .git metadata; frozen commit is provenance only",
        }
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"cannot verify git HEAD at {root}: {exc}") from exc
    current = result.stdout.strip()
    if current.lower() != str(expected).lower():
        raise ValueError(f"source HEAD mismatch at {root}: expected {expected}, found {current}")
    return {
        "path": str(root), "declared_source_commit": expected,
        "current_head": current, "current_head_verified": True,
        "identity_basis": "git HEAD matches frozen commit",
    }


def validate_freeze(freeze: Mapping[str, Any]) -> None:
    if freeze.get("schema") != "lewm.goal-query.pilot.freeze" or freeze.get("status") != "frozen_before_execution":
        raise ValueError("unexpected or unfrozen goal-query pilot freeze")
    data, models, training, evaluation = (freeze[k] for k in ("data", "models", "training", "evaluation"))
    if data["train_contexts"] != 512 or data["train_candidates"] != 64 or data["evaluation_candidates"] != 300:
        raise ValueError("frozen context/candidate counts changed")
    if data["development_slice"] != [800, 808] or data["test_slice"] != [824, 832]:
        raise ValueError("frozen development/test slices changed")
    if data["development_action_seeds"] != [20302601, 20302602] or data["test_action_seeds"] != [20302603, 20302604]:
        raise ValueError("frozen action seeds changed")
    if models["arms"] != list(ARMS) or models["rank"] != 32 or models["latent_dim"] != LATENT_DIM:
        raise ValueError("frozen model arms or dimensions changed")
    if training["initialization_seeds"] != [20302611, 20302612] or training["schedule_seed"] != 20302613:
        raise ValueError("frozen training seeds changed")
    if training["updates"] != 1500 or training["batch_contexts"] != 4:
        raise ValueError("frozen training budget changed")
    if evaluation["timing"]["blocks"] != 6 or evaluation["timing"]["repeats"] != 10:
        raise ValueError("frozen timing design changed")


def source_identity(args: argparse.Namespace, freeze: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = args.stablewm_home / "pusht" / "lewm_object.ckpt"
    return {
        "lewm": source_commit(args.lewm_root, freeze["source"]["lewm_commit"]),
        "stablewm": source_commit(args.stablewm_root, freeze["source"]["stablewm_commit"]),
        "artifacts": {
            name: file_identity(path)
            for name, path in {
                "runner": Path(__file__), "freeze": args.freeze, "protocol": args.freeze.with_name("PROTOCOL.zh.md"),
                "teacher_checkpoint": checkpoint, "dataset": args.dataset,
                "training_rows": args.training_rows, "manifest": args.manifest,
                "temporal_freeze": args.temporal_freeze,
            }.items()
        },
        "hashes_computed": False,
    }


def load_reference(args: argparse.Namespace) -> Any:
    for path in (
        args.transfer_root / "state-action-prefix-gru",
        args.transfer_root,
        args.lewm_root,
        args.stablewm_root,
    ):
        if str(path.resolve()) not in sys.path:
            sys.path.insert(0, str(path.resolve()))
    return importlib.import_module("run_lewm_state_action_prefix_gru")


def select_episodes(dataset_path: Path, data_freeze: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[int], list[int], dict[str, Any]]:
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        lengths = [int(value) for value in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= 26]
    random.Random(int(data_freeze["selection_seed"])).shuffle(valid)
    train = [int(row["episode_id"]) for row in manifest["splits"]["train"]]
    old_heldout = [int(row["episode_id"]) for row in manifest["splits"]["heldout"]]
    prefix = old_heldout + train
    if valid[:len(prefix)] != prefix:
        raise ValueError("selection prefix does not reproduce the frozen 512-context manifest")
    dev_start, dev_end = (int(x) for x in data_freeze["development_slice"])
    test_start, test_end = (int(x) for x in data_freeze["test_slice"])
    if len(valid) < max(dev_end, test_end):
        raise RuntimeError(f"INCONCLUSIVE: only {len(valid)} eligible episodes; frozen slice ends at {max(dev_end, test_end)}")
    dev, test = valid[dev_start:dev_end], valid[test_start:test_end]
    excluded = set(train) | set(old_heldout)
    if len(dev) != 8 or len(test) != 8 or set(dev) & set(test) or (set(dev) | set(test)) & excluded:
        raise ValueError("frozen development/test parent-episode isolation failed")
    return dev, test, {
        "selection_seed": int(data_freeze["selection_seed"]), "eligible_episode_count": len(valid),
        "old_heldout_episode_ids": old_heldout, "train_episode_count": len(train),
        "development_episode_ids": dev, "test_episode_ids": test,
        "development_slice": f"valid[{dev_start}:{dev_end}]", "test_slice": f"valid[{test_start}:{test_end}]",
        "train_old_heldout_development_test_disjoint": True,
        "project_wide_prior_evaluation_exclusion_claim": False,
    }


def tensor(torch: Any, value: Any, dtype: Any = None) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(dtype=dtype) if dtype is not None else value.detach()
    return torch.as_tensor(value, dtype=dtype)


def flat_latent(torch: Any, value: Any) -> Any:
    result = tensor(torch, value, torch.float32)
    if result.numel() != LATENT_DIM:
        raise ValueError(f"expected one {LATENT_DIM}-D latent, got {tuple(result.shape)}")
    return result.reshape(LATENT_DIM)


def validate_training_rows(torch: Any, rows: Any, manifest: Mapping[str, Any], freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != int(freeze["data"]["train_contexts"]):
        raise ValueError("anchor-aligned training rows must contain exactly 512 contexts")
    manifest_rows = list(manifest.get("splits", {}).get("train", []))
    if len(manifest_rows) != len(rows):
        raise ValueError("manifest train split does not contain 512 contexts")
    valid = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("split") != "train":
            raise ValueError(f"training row {index} is missing its train identity")
        if int(row["episode_id"]) != int(manifest_rows[index]["episode_id"]):
            raise ValueError(f"training row order/episode does not match the frozen manifest at {index}")
        if tuple(tensor(torch, row["future_actions"]).shape) != (64, HORIZON, ACTION_DIM):
            raise ValueError(f"training action bank shape drifted at row {index}")
        if tuple(tensor(torch, row["teacher_targets"]).shape) != (64, HORIZON, LATENT_DIM):
            raise ValueError(f"training teacher target shape drifted at row {index}")
        flat_latent(torch, row["latent_history"])
        flat_latent(torch, row["goal_emb"])
        if row.get("stratum") not in ("early", "middle", "late"):
            raise ValueError(f"training row has unknown temporal stratum at {index}")
        if not all(bool(torch.isfinite(tensor(torch, row[key], torch.float32)).all()) for key in ("future_actions", "teacher_targets", "latent_history", "goal_emb", "teacher_objective")):
            raise ValueError(f"nonfinite training bank value at row {index}")
        valid.append(dict(row))
    return valid


def make_train_queries(torch: Any, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered_episodes = list(dict.fromkeys(int(row["episode_id"]) for row in rows))
    if len(ordered_episodes) < 4:
        raise ValueError("training split has fewer than four distinct parent episodes")
    first_by_episode: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        first_by_episode.setdefault(int(row["episode_id"]), row)
    if len(first_by_episode) != len(ordered_episodes):
        raise ValueError("training parent-episode indexing failed")
    episode_position = {episode: index for index, episode in enumerate(ordered_episodes)}
    contexts, actions, terminals, goal_rows, goal_sources = [], [], [], [], []
    for row in rows:
        episode = int(row["episode_id"])
        position = episode_position[episode]
        donors = [ordered_episodes[(position + offset) % len(ordered_episodes)] for offset in range(1, 4)]
        if episode in donors or len(set(donors)) != 3:
            raise ValueError("training goal donor rule did not produce three different parent episodes")
        goal_episodes = [episode] + donors
        contexts.append(flat_latent(torch, row["latent_history"]))
        actions.append(tensor(torch, row["future_actions"], torch.float32))
        terminals.append(tensor(torch, row["teacher_targets"], torch.float32)[:, -1])
        goal_rows.append(torch.stack([flat_latent(torch, first_by_episode[source]["goal_emb"]) for source in goal_episodes]))
        goal_sources.append(goal_episodes)
    cpu = {
        "contexts": torch.stack(contexts), "actions": torch.stack(actions), "terminals": torch.stack(terminals),
        "goals": torch.stack(goal_rows), "goal_sources": goal_sources,
        "episode_ids": [int(row["episode_id"]) for row in rows],
    }
    return cpu, {"parent_episode_count": len(ordered_episodes), "own_plus_three_cyclic_donors": True, "donor_goals_shared_across_contexts": True}


def build_model_class(torch: Any, nn: Any):
    class QueryPredictor(nn.Module):
        def __init__(self, arm: str, mean: Any, components: Any, cost_scale: Any, tail_scale: Any):
            super().__init__()
            self.arm = arm
            self.register_buffer("projection_mean", mean.detach().clone())
            self.register_buffer("components", components.detach().clone())
            self.register_buffer("cost_scale", cost_scale.detach().reshape(()).clone())
            self.register_buffer("tail_scale", tail_scale.detach().reshape(()).clone())
            self.trunk = nn.Sequential(
                nn.LayerNorm(LATENT_DIM + HORIZON * ACTION_DIM),
                nn.Linear(LATENT_DIM + HORIZON * ACTION_DIM, 256), nn.GELU(),
                nn.Linear(256, 256), nn.GELU(),
            )
            if arm in ("scalar", "scalar_aux"):
                self.goal_head = nn.Sequential(nn.Linear(448, 256), nn.GELU(), nn.Linear(256, 1))
            if arm in ("scalar_aux", "projected_tail"):
                self.y_head = nn.Linear(256, int(components.shape[1]))
                self.tail_head = nn.Sequential(nn.Linear(448, 256), nn.GELU(), nn.Linear(256, 1))
            if arm == "full_latent":
                self.full_head = nn.Linear(256, LATENT_DIM)

        @staticmethod
        def _goal_batch(goal: Any, count: int) -> Any:
            value = goal.reshape(-1, LATENT_DIM)
            if value.shape[0] == 1:
                value = value.expand(count, -1)
            if value.shape[0] != count:
                raise ValueError("goal batch must have one row or match the action batch")
            return value

        def _trunk(self, context: Any, actions: Any) -> tuple[Any, Any]:
            state = context.reshape(context.shape[0], -1, LATENT_DIM)[:, -1]
            if actions.shape[1:] != (HORIZON, ACTION_DIM):
                raise ValueError(f"native action input must be [N,5,10], got {tuple(actions.shape)}")
            return self.trunk(torch.cat((state, actions.reshape(actions.shape[0], -1)), dim=-1)), state

        def score(self, context: Any, actions: Any, goal: Any) -> Any:
            hidden, state = self._trunk(context, actions)
            goal_value = self._goal_batch(goal, hidden.shape[0])
            if self.arm in ("scalar", "scalar_aux"):
                return torch.nn.functional.softplus(self.goal_head(torch.cat((hidden, goal_value), dim=-1)).squeeze(-1)) * self.cost_scale
            if self.arm == "projected_tail":
                y = (state - self.projection_mean) @ self.components + self.y_head(hidden)
                tail = torch.nn.functional.softplus(self.tail_head(torch.cat((hidden, goal_value), dim=-1)).squeeze(-1)) * self.tail_scale
                goal_y = (goal_value - self.projection_mean) @ self.components
                return (y - goal_y).square().sum(dim=-1) + tail
            if self.arm == "full_latent":
                predicted = state + self.full_head(hidden)
                return (predicted - goal_value).square().sum(dim=-1)
            raise ValueError(f"unknown training arm: {self.arm}")

        def training_outputs(self, context: Any, actions: Any, goal: Any) -> dict[str, Any]:
            hidden, state = self._trunk(context, actions)
            goal_value = self._goal_batch(goal, hidden.shape[0])
            out: dict[str, Any] = {}
            if self.arm in ("scalar", "scalar_aux"):
                out["cost"] = torch.nn.functional.softplus(self.goal_head(torch.cat((hidden, goal_value), dim=-1)).squeeze(-1)) * self.cost_scale
            if self.arm in ("scalar_aux", "projected_tail"):
                out["y"] = (state - self.projection_mean) @ self.components + self.y_head(hidden)
                out["tail"] = torch.nn.functional.softplus(self.tail_head(torch.cat((hidden, goal_value), dim=-1)).squeeze(-1)) * self.tail_scale
            if self.arm == "projected_tail":
                goal_y = (goal_value - self.projection_mean) @ self.components
                out["cost"] = (out["y"] - goal_y).square().sum(dim=-1) + out["tail"]
            if self.arm == "full_latent":
                out["latent"] = state + self.full_head(hidden)
                out["cost"] = (out["latent"] - goal_value).square().sum(dim=-1)
            return out

        def projected_diagnostics(self, context: Any, actions: Any, goal: Any, teacher_terminal: Any) -> dict[str, Any]:
            if self.arm != "projected_tail":
                raise ValueError("projected diagnostics require the projected_tail training arm")
            hidden, state = self._trunk(context, actions)
            goal_value = self._goal_batch(goal, hidden.shape[0])
            goal_y = (goal_value - self.projection_mean) @ self.components
            yhat = (state - self.projection_mean) @ self.components + self.y_head(hidden)
            tail = torch.nn.functional.softplus(self.tail_head(torch.cat((hidden, goal_value), dim=-1)).squeeze(-1)) * self.tail_scale
            exact_y = (teacher_terminal - self.projection_mean) @ self.components
            return {
                "projected_only": (yhat - goal_y).square().sum(dim=-1),
                "exact_projected_only": (exact_y - goal_y).square().sum(dim=-1),
                "oracle_y_learned_tail": (exact_y - goal_y).square().sum(dim=-1) + tail,
            }

    return QueryPredictor


def parameter_count(model: Any, mode: str = "training") -> int:
    if mode == "training":
        names = {name for name, _ in model.named_parameters()}
    else:
        names = {name for name, _ in model.named_parameters() if name.startswith("trunk.")}
        active = {
            "scalar": ("goal_head.",), "scalar_aux": ("goal_head.",),
            "projected_tail": ("y_head.", "tail_head."), "full_latent": ("full_head.",),
        }[model.arm]
        names |= {name for name, _ in model.named_parameters() if name.startswith(active)}
    return int(sum(value.numel() for name, value in model.named_parameters() if name in names))


def validate_projection(torch: Any, components: Any, tolerance: float = 1e-5) -> float:
    if components.ndim != 2 or components.shape[0] != LATENT_DIM:
        raise ValueError("PCA components must be [192,r]")
    gram = components.T @ components
    identity = torch.eye(components.shape[1], dtype=components.dtype, device=components.device)
    error = float((gram - identity).abs().max().item()) if gram.numel() else 0.0
    if not math.isfinite(error) or error > tolerance:
        raise ValueError(f"PCA components are not orthonormal: max error {error}")
    return error


def decompose_cost(torch: Any, z: Any, goal: Any, mean: Any, components: Any) -> tuple[Any, Any, Any]:
    zc, gc = z - mean, goal - mean
    zy, gy = zc @ components, gc @ components
    zr, gr = zc - zy @ components.T, gc - gy @ components.T
    projected = (zy - gy).square().sum(dim=-1)
    tail = (zr - gr).square().sum(dim=-1)
    total = (z - goal).square().sum(dim=-1)
    return projected, tail, total


def self_checks(torch: Any, QueryPredictor: Any) -> dict[str, Any]:
    bad_projection_rejected = False
    try:
        validate_projection(torch, torch.zeros((LATENT_DIM, 2), dtype=torch.float64))
    except ValueError:
        bad_projection_rejected = True
    if not bad_projection_rejected:
        raise RuntimeError("self-check failed to reject a nonorthogonal projection")
    torch.manual_seed(20302615)
    q, _ = torch.linalg.qr(torch.randn((LATENT_DIM, 32), dtype=torch.float64))
    mean = torch.randn(LATENT_DIM, dtype=torch.float64)
    z, goal = torch.randn((12, LATENT_DIM), dtype=torch.float64), torch.randn((12, LATENT_DIM), dtype=torch.float64)
    projected, tail, direct = decompose_cost(torch, z, goal, mean, q)
    projected_vector = ((z - goal) @ q) @ q.T
    residual_vector = (z - goal) - projected_vector
    cross = projected_vector * residual_vector
    decomposition_error = float((projected + tail - direct).abs().max().item())
    cross_error = float(cross.sum(dim=-1).abs().max().item())
    p0 = torch.empty((LATENT_DIM, 0), dtype=torch.float64)
    p192 = torch.eye(LATENT_DIM, dtype=torch.float64)
    _, tail0, direct0 = decompose_cost(torch, z, goal, mean, p0)
    projected192, tail192, direct192 = decompose_cost(torch, z, goal, mean, p192)
    if decomposition_error > 1e-9 or cross_error > 1e-9 or not torch.allclose(tail0, direct0, atol=1e-10, rtol=1e-10) or not torch.allclose(projected192, direct192, atol=1e-10, rtol=1e-10) or not torch.allclose(tail192, torch.zeros_like(tail192), atol=1e-10, rtol=0):
        raise RuntimeError("projection decomposition self-check failed")
    components = torch.eye(LATENT_DIM, dtype=torch.float32, device="cuda")[:, :32]
    model = QueryPredictor("projected_tail", torch.zeros(LATENT_DIM, device="cuda"), components, torch.tensor(1.0, device="cuda"), torch.tensor(1.0, device="cuda")).to("cuda").eval()
    context = torch.randn((3, 1, LATENT_DIM), device="cuda")
    actions = torch.randn((3, HORIZON, ACTION_DIM), device="cuda")
    goals = torch.randn((3, 1, LATENT_DIM), device="cuda")
    with torch.no_grad():
        batched = model.score(context, actions, goals)
        singles = torch.stack([model.score(context[i:i + 1], actions[i:i + 1], goals[i:i + 1])[0] for i in range(3)])
    torch.cuda.synchronize()
    if not torch.allclose(batched, singles, atol=1e-6, rtol=1e-6):
        raise RuntimeError("native batch scoring differs from the same rows scored singly")
    del model
    torch.cuda.empty_cache()
    return {
        "status": "PASS", "nonorthogonal_projection_rejected": bad_projection_rejected,
        "rank0_and_rank192_boundaries": "PASS", "exact_decomposition_max_abs_fp64": decomposition_error,
        "orthogonal_cross_term_max_abs_fp64": cross_error, "native_batch_matches_single_rows": True,
    }


def fit_projection(torch: Any, terminal: Any, rank: int) -> tuple[Any, Any, dict[str, Any], tuple[Any, Any]]:
    targets = terminal.reshape(-1, LATENT_DIM).to(device="cuda", dtype=torch.float64)
    mean64 = targets.mean(dim=0)
    centered = targets - mean64
    covariance = centered.T @ centered / max(1, centered.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    indices = torch.argsort(eigenvalues, descending=True, stable=True)[:rank]
    components64 = eigenvectors[:, indices]
    orth_error64 = validate_projection(torch, components64, 1e-10)
    mean32, components32 = mean64.float(), components64.float()
    orth_error32 = validate_projection(torch, components32, 1e-5)
    variance = eigenvalues[indices].clamp_min(0).mean()
    return mean32, components32, {
        "fit_rows": int(targets.shape[0]), "rank": rank, "eigenvalues_descending": eigenvalues[indices].clamp_min(0).cpu().tolist(),
        "orthogonality_max_abs_fp64": orth_error64, "orthogonality_max_abs_fp32": orth_error32,
        "projected_target_mean_coordinate_variance": float(variance.item()),
    }, (mean64, components64)


def verified_official_costs(torch: Any, base: Any, official: Any, rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, Any]:
    rtol, atol = float(freeze["gates"]["cost_equality_rtol"]), float(freeze["gates"]["cost_equality_atol"])
    max_abs, checked = 0.0, 0
    for row in rows:
        targets = tensor(torch, row["teacher_targets"], torch.float32)
        objectives = tensor(torch, row["teacher_objective"], torch.float32)
        if targets.ndim == 3:
            target_blocks, objective_blocks = targets.unsqueeze(0), objectives.reshape(1, -1)
        elif targets.ndim == 4:
            target_blocks, objective_blocks = targets, objectives
        else:
            raise ValueError(f"unexpected teacher target shape for {row.get('context_id')}: {tuple(targets.shape)}")
        goal = flat_latent(torch, row["goal_emb"])
        for block in range(target_blocks.shape[0]):
            bank = target_blocks[block].to("cuda")
            goal_cuda = goal.to("cuda")
            manual = (bank[:, -1] - goal_cuda).square().sum(dim=-1)
            official_cost = base._official_objective(official, {"latent_history": tensor(torch, row["latent_history"], torch.float32).reshape(1, 1, LATENT_DIM).to("cuda"), "goal_emb": goal_cuda.reshape(1, 1, LATENT_DIM)}, bank)
            stored = objective_blocks[block].reshape(-1).to("cuda")
            error = float((manual - official_cost).abs().max().item())
            max_abs = max(max_abs, error)
            checked += int(manual.numel())
            if not torch.allclose(manual, official_cost, atol=atol, rtol=rtol) or not torch.allclose(stored, official_cost, atol=atol, rtol=rtol):
                raise RuntimeError(f"official terminal cost mismatch for row {row.get('context_id')} block {block}")
    return {"status": "PASS", "candidate_costs_checked": checked, "max_abs_error": max_abs, "criterion": "official terminal squared Euclidean cost; direct sum over latent dimension"}


def verify_fp64_decomposition(torch: Any, terminal: Any, goals: Any, mean32: Any, components32: Any, mean64: Any, components64: Any, atol: float, rtol: float) -> dict[str, Any]:
    comp64 = components64.double()
    maximum64, maximum32, maximum_relative32 = 0.0, 0.0, 0.0
    recovered32 = True
    orth64 = validate_projection(torch, comp64, 1e-10)
    orth32 = validate_projection(torch, components32, 1e-5)
    rows_per_chunk = 16
    for start in range(0, terminal.shape[0], rows_per_chunk):
        zs32 = terminal[start:start + rows_per_chunk]
        gs32 = goals[start:start + rows_per_chunk]
        zs, gs = zs32.double(), gs32.double()
        zy = (zs - mean64) @ comp64
        gy = (gs - mean64) @ comp64
        zr = (zs - mean64) - zy @ comp64.T
        gr = (gs - mean64) - gy @ comp64.T
        projected = (zy[:, :, None, :] - gy[:, None, :, :]).square().sum(dim=-1)
        tail = (zr[:, :, None, :] - gr[:, None, :, :]).square().sum(dim=-1)
        direct = (zs[:, :, None, :] - gs[:, None, :, :]).square().sum(dim=-1)
        maximum64 = max(maximum64, float((projected + tail - direct).abs().max().item()))
        zf, gf, mf, pf = zs32, gs32, mean32, components32
        zyf, gyf = (zf - mf) @ pf, (gf - mf) @ pf
        zrf = (zf - mf) - zyf @ pf.T
        grf = (gf - mf) - gyf @ pf.T
        projected32 = (zyf[:, :, None, :] - gyf[:, None, :, :]).square().sum(dim=-1)
        tail32 = (zrf[:, :, None, :] - grf[:, None, :, :]).square().sum(dim=-1)
        direct32 = (zf[:, :, None, :] - gf[:, None, :, :]).square().sum(dim=-1)
        error32 = (projected32 + tail32 - direct32).abs()
        maximum32 = max(maximum32, float(error32.max().item()))
        maximum_relative32 = max(maximum_relative32, float((error32 / direct32.abs().clamp_min(atol)).max().item()))
        recovered32 = recovered32 and torch.allclose(projected32 + tail32, direct32, atol=atol, rtol=rtol)
    if not math.isfinite(maximum64) or maximum64 > 1e-8 or not recovered32:
        raise RuntimeError(f"projected/tail decomposition failed: fp64={maximum64}, fp32={maximum32}, fp32_close={recovered32}")
    return {
        "status": "PASS", "fp64_max_abs_error": maximum64, "fp64_orthogonality_max_abs": orth64,
        "fp32_max_abs_cost_recovery_error": maximum32, "fp32_max_relative_cost_recovery_error": maximum_relative32,
        "fp32_orthogonality_max_abs": orth32, "fp32_cost_recovery_allclose": recovered32,
        "fp32_cost_recovery_tolerance": {"atol": atol, "rtol": rtol}, "rows_checked": int(terminal.shape[0]),
    }


def prepare_train_tensors(torch: Any, rows: Sequence[Mapping[str, Any]], train: Mapping[str, Any], mean: Any, components: Any, pca64: tuple[Any, Any], freeze: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contexts = train["contexts"].to("cuda")
    actions = train["actions"].to("cuda")
    terminals = train["terminals"].to("cuda")
    goals = train["goals"].to("cuda")
    mean_gpu, comp = mean.to("cuda"), components.to("cuda")
    pca_y = (terminals - mean_gpu) @ comp
    goal_y = (goals - mean_gpu) @ comp
    terminal_residual = (terminals - mean_gpu) - pca_y @ comp.T
    goal_residual = (goals - mean_gpu) - goal_y @ comp.T
    costs = (terminals[:, :, None, :] - goals[:, None, :, :]).square().sum(dim=-1)
    tail_costs = (terminal_residual[:, :, None, :] - goal_residual[:, None, :, :]).square().sum(dim=-1)
    if not all(bool(torch.isfinite(value).all()) for value in (contexts, actions, terminals, goals, costs, tail_costs, pca_y)):
        raise ValueError("nonfinite values in the detached training tuples")
    cost_bank_variance = costs.var(dim=1, unbiased=False)
    tail_bank_variance = tail_costs.var(dim=1, unbiased=False)
    global_cost_variance = float(costs.var(unbiased=False).item())
    global_tail_variance = float(tail_costs.var(unbiased=False).item())
    cost_floor = max(global_cost_variance * 1e-6, 1e-8)
    tail_floor = max(global_tail_variance * 1e-6, 1e-8)
    cost_rms = float(costs.square().mean().sqrt().item())
    tail_rms = float(tail_costs.square().mean().sqrt().item())
    projected_variance = float(pca_y.reshape(-1, pca_y.shape[-1]).var(dim=0, unbiased=False).mean().item())
    full_variance = float(terminals.reshape(-1, LATENT_DIM).var(dim=0, unbiased=False).mean().item())
    scales = {
        "cost_rms": max(cost_rms, 1e-8), "tail_rms": max(tail_rms, 1e-8),
        "cost_variance_floor": cost_floor, "tail_variance_floor": tail_floor,
        "projected_target_mean_coordinate_variance": max(projected_variance, 1e-8),
        "full_target_mean_coordinate_variance": max(full_variance, 1e-8),
        "cost_bank_variance_min": float(cost_bank_variance.min().item()),
        "tail_bank_variance_min": float(tail_bank_variance.min().item()),
        "cost_variance_floor_rule": "max(global training query-cost population variance * 1e-6, 1e-8)",
        "tail_variance_floor_rule": "max(global training tail-cost population variance * 1e-6, 1e-8)",
    }
    fp64 = verify_fp64_decomposition(torch, terminals, goals, mean_gpu, comp, pca64[0].to("cuda"), pca64[1].to("cuda"), float(freeze["gates"]["cost_equality_atol"]), float(freeze["gates"]["cost_equality_rtol"]))
    # Verify each stored own-goal training objective against the official criterion.
    own_cost = (terminals - goals[:, :1]).square().sum(dim=-1)
    own_stored_errors = []
    for index, row in enumerate(rows):
        manual = own_cost[index]
        stored = tensor(torch, row["teacher_objective"], torch.float32).reshape(-1).to("cuda")
        if stored.numel() != manual.numel() or not torch.allclose(stored, manual, atol=float(freeze["gates"]["cost_equality_atol"]), rtol=float(freeze["gates"]["cost_equality_rtol"])):
            raise RuntimeError(f"stored training objective differs from detached terminal target at row {index}")
        own_stored_errors.append(float((stored - manual).abs().max().item()))
    return {
        "contexts": contexts, "actions": actions, "terminals": terminals, "goals": goals,
        "costs": costs, "tail_costs": tail_costs, "projected_targets": pca_y,
        "scales": scales, "decomposition_check": fp64,
    }, {"status": "PASS", "own_goal_stored_objective_max_abs_error": max(own_stored_errors, default=0.0), "own_goal_rows_checked": len(rows), "fp64_decomposition": fp64}


def pack_training_batch(torch: Any, source: Mapping[str, Any], indices: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    context = source["contexts"].index_select(0, indices)
    actions = source["actions"].index_select(0, indices)
    goals = source["goals"].index_select(0, indices)
    count, candidates, goal_count = actions.shape[0], actions.shape[1], goals.shape[1]
    flat_context = context[:, None, None, :].expand(count, candidates, goal_count, LATENT_DIM).reshape(-1, LATENT_DIM)
    flat_actions = actions[:, :, None].expand(count, candidates, goal_count, HORIZON, ACTION_DIM).reshape(-1, HORIZON, ACTION_DIM)
    flat_goals = goals[:, None].expand(count, candidates, goal_count, LATENT_DIM).reshape(-1, LATENT_DIM)
    shape = (count, candidates, goal_count)
    labels = {
        "cost": source["costs"].index_select(0, indices),
        "tail": source["tail_costs"].index_select(0, indices),
        "y": source["projected_targets"].index_select(0, indices)[:, :, None].expand(count, candidates, goal_count, -1).reshape(-1, source["projected_targets"].shape[-1]),
        "latent": source["terminals"].index_select(0, indices)[:, :, None].expand(count, candidates, goal_count, LATENT_DIM).reshape(-1, LATENT_DIM),
        "shape": shape,
    }
    return flat_context, flat_actions, flat_goals, labels


def cost_loss(torch: Any, prediction: Any, target: Any, variance_floor: float, rms: float) -> Any:
    predicted_centered = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    target_variance = target_centered.square().mean(dim=1).clamp_min(variance_floor)
    centered = (predicted_centered - target_centered).square().mean(dim=1) / target_variance
    absolute = (prediction - target).square().mean()
    return centered.mean() + 0.1 * absolute / max(rms * rms, 1e-8)


def train_model(torch: Any, QueryPredictor: Any, arm: str, seed: int, schedule: Sequence[Any], source: Mapping[str, Any], output: Path, projection: tuple[Any, Any], freeze: Mapping[str, Any], trace_path: Path) -> dict[str, Any]:
    mean, components = projection
    scales = source["scales"]
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model = QueryPredictor(
        arm, mean.to("cuda"), components.to("cuda"),
        torch.tensor(scales["cost_rms"], dtype=torch.float32, device="cuda"),
        torch.tensor(scales["tail_rms"], dtype=torch.float32, device="cuda"),
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(freeze["training"]["learning_rate"]),
        weight_decay=float(freeze["training"]["weight_decay"]),
    )
    trace, last_components = [], {}
    updates = int(freeze["training"]["updates"])
    for step, ids in enumerate(schedule, 1):
        indices = torch.as_tensor(ids, dtype=torch.long, device="cuda")
        context, actions, goals, labels = pack_training_batch(torch, source, indices)
        outputs = model.training_outputs(context, actions, goals)
        count, candidates, goal_count = labels["shape"]
        outputs = {key: value.reshape(count, candidates, goal_count, *value.shape[1:]) for key, value in outputs.items()}
        labels["cost"] = labels["cost"]
        loss = cost_loss(torch, outputs["cost"], labels["cost"], scales["cost_variance_floor"], scales["cost_rms"])
        last_components = {"cost": float(loss.detach().item())}
        if arm in ("scalar_aux", "projected_tail"):
            y_target = labels["y"].reshape(count, candidates, goal_count, -1)
            y_error = (outputs["y"] - y_target).square().mean() / scales["projected_target_mean_coordinate_variance"]
            tail = cost_loss(torch, outputs["tail"], labels["tail"], scales["tail_variance_floor"], scales["tail_rms"])
            loss = loss + y_error + tail
            last_components.update({"projected_latent": float(y_error.detach().item()), "tail": float(tail.detach().item())})
        if arm == "full_latent":
            latent_target = labels["latent"].reshape(count, candidates, goal_count, LATENT_DIM)
            latent_loss = (outputs["latent"] - latent_target).square().mean() / scales["full_target_mean_coordinate_variance"]
            loss = loss + latent_loss
            last_components["full_latent"] = float(latent_loss.detach().item())
        if not bool(torch.isfinite(loss).item()) or not all(bool(torch.isfinite(value).all().item()) for value in outputs.values()):
            raise FloatingPointError(f"nonfinite training output/loss at {arm} seed {seed} step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == updates:
            entry = {"seed": seed, "arm": arm, "step": step, "total_loss": float(loss.detach().item()), "components": last_components, "finite": True}
            trace.append(entry)
            append_jsonl(trace_path, [entry])
    if len(trace) == 0 or trace[-1]["step"] != updates:
        raise RuntimeError("final frozen training update was not reached")
    checkpoint = output / "checkpoints" / f"seed_{seed}" / f"{arm}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": arm, "seed": seed, "step": updates, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, checkpoint)
    result = {
        "seed": seed, "arm": arm, "updates": updates, "final_step": updates,
        "last_recorded_total_loss": trace[-1]["total_loss"], "last_recorded_components": trace[-1]["components"],
        "finite": True, "checkpoint": file_identity(checkpoint),
        "training_parameter_count": parameter_count(model, "training"),
        "deployed_parameter_count": parameter_count(model, "deployed"),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def load_trained_model(torch: Any, QueryPredictor: Any, arm: str, seed: int, output: Path, projection: tuple[Any, Any], source: Mapping[str, Any]) -> Any:
    path = output / "checkpoints" / f"seed_{seed}" / f"{arm}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("arm") != arm or int(state.get("seed", -1)) != seed or int(state.get("step", -1)) != 1500:
        raise ValueError(f"unexpected final checkpoint identity: {path}")
    model = QueryPredictor(
        arm, projection[0].to("cuda"), projection[1].to("cuda"),
        torch.tensor(source["scales"]["cost_rms"], dtype=torch.float32, device="cuda"),
        torch.tensor(source["scales"]["tail_rms"], dtype=torch.float32, device="cuda"),
    ).to("cuda")
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    return model


def make_goal_queries(rows: Sequence[Mapping[str, Any]], episode_order: Sequence[int], same_stratum: bool) -> list[list[dict[str, Any]]]:
    by_key = {(int(row["episode_id"]), str(row["stratum"])): row for row in rows}
    first_by_episode: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        first_by_episode.setdefault(int(row["episode_id"]), row)
    order = list(dict.fromkeys(int(x) for x in episode_order))
    if len(order) < 4:
        raise ValueError("evaluation split requires at least four distinct parent episodes")
    position = {episode: index for index, episode in enumerate(order)}
    result = []
    for row in rows:
        episode = int(row["episode_id"])
        donors = [order[(position[episode] + offset) % len(order)] for offset in range(1, 4)]
        if episode in donors or len(set(donors)) != 3:
            raise ValueError("evaluation donor rule did not produce three distinct parent episodes")
        if same_stratum:
            donor_rows = [by_key[(donor, str(row["stratum"]))] for donor in donors]
        else:
            donor_rows = [first_by_episode[donor] for donor in donors]
        goals = [{"kind": "own", "source_episode_id": episode, "goal_emb": row["goal_emb"]}]
        goals.extend({"kind": "donor", "source_episode_id": donor, "goal_emb": donor_row["goal_emb"]} for donor, donor_row in zip(donors, donor_rows))
        result.append(goals)
    return result


def validate_fresh_rows(torch: Any, rows: Sequence[Mapping[str, Any]], episode_ids: Sequence[int], action_seeds: Sequence[int]) -> None:
    if len(rows) != len(episode_ids) * 3:
        raise ValueError(f"fresh slice has {len(rows)} rows; expected {len(episode_ids) * 3}")
    per_episode: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        episode = int(row["episode_id"])
        if episode not in set(int(x) for x in episode_ids):
            raise ValueError("fresh row episode is outside its frozen split")
        per_episode[episode].add(str(row["stratum"]))
        if tuple(tensor(torch, row["future_actions"]).shape) != (2, 300, HORIZON, ACTION_DIM):
            raise ValueError(f"fresh candidate shape drifted at {row.get('context_id')}")
        if tuple(tensor(torch, row["teacher_targets"]).shape) != (2, 300, HORIZON, LATENT_DIM):
            raise ValueError(f"fresh teacher target shape drifted at {row.get('context_id')}")
        if len(row.get("teacher_objective", [])) != 2:
            raise ValueError(f"fresh teacher objective shape drifted at {row.get('context_id')}")
    if set(per_episode) != set(int(x) for x in episode_ids) or any(strata != {"early", "middle", "late"} for strata in per_episode.values()):
        raise ValueError("fresh rows must contain early/middle/late anchors for all eight episodes")
    for row in rows:
        if [int(x) for x in row.get("candidate_seeds", action_seeds)] != list(action_seeds):
            # Existing builder records seeds in its metadata, not row values.
            if "candidate_seeds" in row:
                raise ValueError("fresh action seed identity drifted")


def build_fresh(torch: Any, reference: Any, official: Any, dataset: Path, episode_ids: Sequence[int], action_seeds: Sequence[int], anchors: Sequence[str], temporal_freeze: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ema = reference.ema
    previous_seeds, previous_anchors = ema.FRESH_SEEDS, ema.FRESH_ANCHORS
    ema.FRESH_SEEDS, ema.FRESH_ANCHORS = tuple(int(x) for x in action_seeds), tuple(anchors)
    try:
        rows, metadata = ema.build_fresh_rows(official, dataset, episode_ids, {"evaluation": temporal_freeze["evaluation"]})
    finally:
        ema.FRESH_SEEDS, ema.FRESH_ANCHORS = previous_seeds, previous_anchors
    validate_fresh_rows(torch, rows, episode_ids, action_seeds)
    return [dict(row) for row in rows], metadata


def query_metrics(torch: Any, base: Any, truth: Any, prediction: Any, actions: Any, episode_id: int, anchor: str, block_seed: int, goal_kind: str, goal_source: int, arm: str, training_seed: int) -> dict[str, Any]:
    if not all(bool(torch.isfinite(value).all().item()) for value in (truth, prediction, actions)):
        raise FloatingPointError(f"nonfinite evaluation values for {arm} seed {training_seed}")
    true_order = torch.argsort(truth)
    pred_order = torch.argsort(prediction)
    top = 30
    true_elite, pred_elite = true_order[:top], pred_order[:top]
    overlap = len(set(true_elite.detach().cpu().tolist()) & set(pred_elite.detach().cpu().tolist())) / top
    truth_center = truth - truth.mean()
    pred_center = prediction - prediction.mean()
    true_actions, pred_actions = actions[true_elite], actions[pred_elite]
    true_mean, pred_mean = true_actions.mean(dim=0), pred_actions.mean(dim=0)
    true_std, pred_std = true_actions.std(dim=0, unbiased=True), pred_actions.std(dim=0, unbiased=True)
    proposal_scale = true_std.square().mean().sqrt().clamp_min(1e-6)
    true_sorted = truth[true_order]
    return {
        "split": "", "training_seed": int(training_seed), "arm": arm,
        "episode_id": int(episode_id), "anchor": anchor, "action_seed": int(block_seed),
        "goal_kind": goal_kind, "goal_source_episode_id": int(goal_source), "candidate_count": int(truth.numel()),
        "bank_centered_cost_rmse": float((truth_center - pred_center).square().mean().sqrt().item()),
        "spearman": float(base._spearman(truth, prediction)), "top30_overlap": float(overlap),
        "argmin_match": bool(int(true_order[0].item()) == int(pred_order[0].item())),
        "teacher_top30_regret": float((truth[pred_elite].mean() - truth[true_elite].mean()).item()),
        "teacher_30_31_gap": float((true_sorted[30] - true_sorted[29]).item()),
        "teacher_30_31_gap_over_bank_std": float(((true_sorted[30] - true_sorted[29]) / truth.std(unbiased=False).clamp_min(1e-8)).item()),
        "elite_proposal_mean_drift_normalized": float(((pred_mean - true_mean).square().mean().sqrt() / proposal_scale).item()),
        "elite_proposal_std_drift_normalized": float(((pred_std - true_std).square().mean().sqrt() / proposal_scale).item()),
        "finite": True,
    }


def evaluate_split(torch: Any, base: Any, official: Any, QueryPredictor: Any, split: str, rows: Sequence[Mapping[str, Any]], goals_by_row: Sequence[Sequence[Mapping[str, Any]]], action_seeds: Sequence[int], training_seeds: Sequence[int], output: Path, projection: tuple[Any, Any], training: Mapping[str, Any], details_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str], list[dict[str, Any]]]]:
    per_arm: dict[tuple[int, str], list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for train_seed in training_seeds:
        for arm in ARMS:
            model = load_trained_model(torch, QueryPredictor, arm, int(train_seed), output, projection, training)
            detail_records: list[dict[str, Any]] = []
            for row, queries in zip(rows, goals_by_row):
                context = flat_latent(torch, row["latent_history"]).reshape(1, 1, LATENT_DIM).to("cuda")
                targets = tensor(torch, row["teacher_targets"], torch.float32).to("cuda")
                candidate_actions = tensor(torch, row["future_actions"], torch.float32).to("cuda")
                for block, action_seed in enumerate(action_seeds):
                    actions = candidate_actions[block]
                    terminal = targets[block, :, -1]
                    context_batch = context.expand(actions.shape[0], -1, -1)
                    for query in queries:
                        goal = flat_latent(torch, query["goal_emb"]).reshape(1, 1, LATENT_DIM).to("cuda")
                        truth = (terminal - goal.reshape(1, LATENT_DIM)).square().sum(dim=-1)
                        with torch.no_grad():
                            prediction = model.score(context_batch, actions, goal)
                            diagnostics = model.projected_diagnostics(context_batch, actions, goal, terminal) if arm == "projected_tail" else {}
                        item = query_metrics(torch, base, truth, prediction, actions, int(row["episode_id"]), str(row["stratum"]), int(action_seed), str(query["kind"]), int(query["source_episode_id"]), arm, int(train_seed))
                        item["split"] = split
                        item["context_id"] = str(row["context_id"])
                        detail_records.append(item)
                        for diagnostic_arm, diagnostic_score in diagnostics.items():
                            diag = query_metrics(torch, base, truth, diagnostic_score, actions, int(row["episode_id"]), str(row["stratum"]), int(action_seed), str(query["kind"]), int(query["source_episode_id"]), diagnostic_arm, int(train_seed))
                            diag["split"] = split
                            diag["context_id"] = str(row["context_id"])
                            detail_records.append(diag)
            if len(detail_records) == 0 or not all(item["finite"] for item in detail_records):
                raise RuntimeError(f"empty or nonfinite {split} evaluation for {arm} seed {train_seed}")
            append_jsonl(details_path, detail_records)
            for evaluated_arm in (arm, *(DIAGNOSTICS if arm == "projected_tail" else ())):
                selected_records = [item for item in detail_records if item["arm"] == evaluated_arm]
                if selected_records:
                    per_arm[(int(train_seed), evaluated_arm)] = selected_records
                    summaries.setdefault(str(train_seed), {})[evaluated_arm] = summarize_records(selected_records, split, evaluated_arm)
            del model
            torch.cuda.empty_cache()
    return summaries, per_arm


def aggregate_values(items: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(items)}
    for key in keys:
        values = [float(item[key]) for item in items if key in item and math.isfinite(float(item[key]))]
        if values:
            result[key + "_median"] = float(statistics.median(values))
            result[key + "_minimum"] = float(min(values))
            result[key + "_maximum"] = float(max(values))
    result["finite"] = bool(items) and all(bool(item.get("finite", False)) for item in items)
    return result


METRIC_KEYS = (
    "bank_centered_cost_rmse", "spearman", "top30_overlap", "argmin_match", "teacher_top30_regret",
    "teacher_30_31_gap", "teacher_30_31_gap_over_bank_std",
    "elite_proposal_mean_drift_normalized", "elite_proposal_std_drift_normalized",
)


def summarize_records(records: Sequence[Mapping[str, Any]], split: str, arm: str) -> dict[str, Any]:
    episode_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    goal_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    stratum_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        episode_groups[str(item["episode_id"])].append(item)
        goal_groups[str(item["goal_kind"])].append(item)
        stratum_groups[str(item["anchor"])].append(item)
    return {
        "split": split, "arm": arm, "overall_single_bank_descriptive": aggregate_values(records, METRIC_KEYS),
        "goal_type": {key: aggregate_values(value, METRIC_KEYS) for key, value in goal_groups.items()},
        "anchor": {key: aggregate_values(value, METRIC_KEYS) for key, value in stratum_groups.items()},
        "episodes": {key: aggregate_values(value, METRIC_KEYS) for key, value in episode_groups.items()},
        "candidate_and_goal_queries_are_nested_not_independent": True,
    }


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty timing sample")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def balanced_orders(arms: Sequence[str], blocks: int, seed: int) -> list[list[str]]:
    rng = random.Random(int(seed))
    base = list(arms)
    rng.shuffle(base)
    offsets = list(range(len(base))) * (blocks // len(base)) + list(range(blocks % len(base)))
    rng.shuffle(offsets)
    return [[base[(offset + index) % len(base)] for index in range(len(base))] for offset in offsets]


def time_native(torch: Any, base: Any, official: Any, models: Mapping[str, Any], timing_rows: Sequence[Mapping[str, Any]], action_seed: int, freeze: Mapping[str, Any], timing_path: Path) -> dict[str, Any]:
    config = freeze["evaluation"]["timing"]
    model_names = list(ARMS)
    timed_names = model_names + ["teacher"]
    if len(timing_rows) != int(config["blocks"]) or len({str(row["context_id"]) for row in timing_rows}) != len(timing_rows):
        raise ValueError("native timing requires six distinct context rows")
    banks = []
    for row in timing_rows:
        actions = tensor(torch, row["future_actions"], torch.float32)[0].to("cuda")
        context_single = flat_latent(torch, row["latent_history"]).reshape(1, 1, LATENT_DIM).to("cuda")
        goal = flat_latent(torch, row["goal_emb"]).reshape(1, 1, LATENT_DIM).to("cuda")
        context = context_single.expand(actions.shape[0], -1, -1)
        row_gpu = {"latent_history": context_single, "goal_emb": goal}
        banks.append({"row": row, "actions": actions, "context": context, "goal": goal, "row_gpu": row_gpu})

    def call(name: str, bank: Mapping[str, Any]) -> None:
        with torch.no_grad():
            if name == "teacher":
                target = base.official_teacher_targets(official, bank["context"], bank["actions"])
                base._official_objective(official, bank["row_gpu"], target)
            else:
                models[name].score(bank["context"], bank["actions"], bank["goal"])

    for name in timed_names:
        for _ in range(int(config["warmup"])):
            call(name, banks[0])
        torch.cuda.synchronize()
    wall_samples: dict[str, list[float]] = {name: [] for name in timed_names}
    event_samples: dict[str, list[float]] = {name: [] for name in timed_names}
    orders = balanced_orders(timed_names, int(config["blocks"]), int(config["balanced_random_arm_order_seed"]))
    orders_path = []
    for block, order in enumerate(orders):
        orders_path.append(order)
        bank = banks[block]
        for name in order:
            for repeat in range(int(config["repeats"])):
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                started = time.perf_counter()
                call(name, bank)
                end.record()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000.0
                wall_samples[name].append(elapsed)
                event_samples[name].append(float(start.elapsed_time(end)))
                append_jsonl(timing_path, [{"block": block, "repeat": repeat, "arm": name, "wall_milliseconds": elapsed, "cuda_event_milliseconds": event_samples[name][-1], "training_seed": int(timing_rows[0].get("timing_training_seed", -1)), "action_seed": int(action_seed), "context_id": str(bank["row"]["context_id"]), "episode_id": int(bank["row"]["episode_id"]), "anchor": str(bank["row"]["stratum"]), "goal_type": "own", "action_bank_index": 0}])
    teacher_p50 = quantile(wall_samples["teacher"], 0.5)
    return {
        "boundary": "six distinct cached H1-context/own-goal/action-bank-0 queries; teacher includes official five-step rollout plus criterion; predictor includes native final scoring",
        "active_contexts": 1, "warmup": int(config["warmup"]), "repeats_per_block": int(config["repeats"]),
        "blocks": int(config["blocks"]), "primary_clock": "CPU perf_counter with CUDA synchronize before and after each call", "balanced_arm_orders": orders_path,
        "timing_banks": [{"context_id": str(row["context_id"]), "episode_id": int(row["episode_id"]), "anchor": str(row["stratum"]), "action_seed": int(action_seed), "action_bank_index": 0, "goal_type": "own"} for row in timing_rows],
        "teacher_shadow_in_predictor_timing": False,
        "arms": {
            name: {"p50_ms": quantile(values, 0.5), "p95_ms": quantile(values, 0.95), "cuda_event_p50_ms_supplementary": quantile(event_samples[name], 0.5), "samples": len(values),
                  "teacher_p50_ms": teacher_p50, "scoring_p50_reduction": 1.0 - quantile(values, 0.5) / max(teacher_p50, 1e-12)}
            for name, values in wall_samples.items() if name != "teacher"
        },
        "teacher": {"p50_ms": teacher_p50, "p95_ms": quantile(wall_samples["teacher"], 0.95), "cuda_event_p50_ms_supplementary": quantile(event_samples["teacher"], 0.5), "samples": len(wall_samples["teacher"])},
    }


def time_all_models(torch: Any, base: Any, official: Any, QueryPredictor: Any, training_seeds: Sequence[int], timing_rows: Sequence[Mapping[str, Any]], action_seed: int, output: Path, projection: tuple[Any, Any], training: Mapping[str, Any], freeze: Mapping[str, Any], timing_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in training_seeds:
        models = {arm: load_trained_model(torch, QueryPredictor, arm, int(seed), output, projection, training) for arm in ARMS}
        rows = [dict(row, timing_training_seed=int(seed)) for row in timing_rows]
        result[str(seed)] = time_native(torch, base, official, models, rows, int(action_seed), freeze, timing_path)
        del models
        torch.cuda.empty_cache()
    return result


def predictor_quality(freeze: Mapping[str, Any], records: Sequence[Mapping[str, Any]], timing: Mapping[str, Any]) -> dict[str, Any]:
    gate = freeze["gates"]["predictor"]
    stats = aggregate_values(records, ("spearman", "top30_overlap"))
    timing_gate = float(timing["scoring_p50_reduction"]) >= float(gate["scoring_p50_reduction_min"])
    conditions = {
        "median_spearman_min": stats["spearman_median"] >= float(gate["median_spearman_min"]),
        "minimum_spearman_min": stats["spearman_minimum"] >= float(gate["minimum_spearman_min"]),
        "median_top30_min": stats["top30_overlap_median"] >= float(gate["median_top30_min"]),
        "minimum_top30_min": stats["top30_overlap_minimum"] >= float(gate["minimum_top30_min"]),
        "scoring_p50_reduction_min": timing_gate,
        "finite": stats["finite"],
    }
    return {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions, "metrics": stats, "scoring_timing": dict(timing), "unit": "all single-bank descriptive queries; candidate/goal/anchor/action-seed measurements are nested"}


def dev_selection(freeze: Mapping[str, Any], dev_records: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]], timing: Mapping[str, Any], params: Mapping[str, Any], selection_meta: Mapping[str, Any]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    passing = []
    for arm in ARMS:
        seeds = {}
        for seed in freeze["training"]["initialization_seeds"]:
            stats = predictor_quality(freeze, dev_records[(int(seed), arm)], timing[str(seed)]["arms"][arm])
            seeds[str(seed)] = stats
        both = all(item["status"] == "PASS" for item in seeds.values())
        by_arm[arm] = {"status": "PASS" if both else "FAIL", "both_training_seeds_pass": both, "per_seed": seeds, "parameter_counts": params[arm]}
        if both:
            metrics = [item["metrics"] for item in seeds.values()]
            p50s = [float(timing[str(seed)]["arms"][arm]["p50_ms"]) for seed in freeze["training"]["initialization_seeds"]]
            passing.append({"arm": arm,
                            "median_top30": statistics.median(float(value["top30_overlap_median"]) for value in metrics),
                            "median_spearman": statistics.median(float(value["spearman_median"]) for value in metrics),
                            "native_p50_ms": statistics.median(p50s)})
    ranking = sorted(passing, key=lambda item: (-item["median_top30"], -item["median_spearman"], item["native_p50_ms"], item["arm"]))
    return {
        "schema": "lewm.goal-query.pilot.selection", "status": "SELECTED" if ranking else "NO_DEV_ARM_QUALIFIED",
        "selected_arm": ranking[0]["arm"] if ranking else None, "ranking_lexicographic": ["median_top30_desc", "median_spearman_desc", "native_p50_ms_asc", "arm_name"],
        "eligible_arm_ranking": ranking, "development_quality": by_arm,
        "frozen_selection": dict(selection_meta), "test_rows_generated": False,
        "training_seed_pair_required": True,
    }


def episode_gate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        by_episode[str(item["episode_id"])].append(item)
    return {
        episode: {
            "top30_median": float(statistics.median(float(item["top30_overlap"]) for item in items)),
            "spearman_median": float(statistics.median(float(item["spearman"]) for item in items)),
        }
        for episode, items in by_episode.items()
    }


def paired_episode_gate(control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]], minimum_median: float, minimum_improved: int, spearman_drop_max: float | None = None) -> dict[str, Any]:
    c, t = episode_gate_records(control), episode_gate_records(treatment)
    if set(c) != set(t):
        raise ValueError("paired mechanism gate has mismatched episode sets")
    paired = []
    for episode in sorted(c, key=int):
        paired.append({"episode_id": int(episode), "top30_delta": t[episode]["top30_median"] - c[episode]["top30_median"], "spearman_delta": t[episode]["spearman_median"] - c[episode]["spearman_median"]})
    median_delta = float(statistics.median(item["top30_delta"] for item in paired))
    improved = sum(item["top30_delta"] > 0 for item in paired)
    conditions: dict[str, Any] = {"median_episode_top30_delta_min": median_delta >= minimum_median, "improved_parent_episodes_min": improved >= minimum_improved}
    median_spearman_delta = float(statistics.median(item["spearman_delta"] for item in paired))
    if spearman_drop_max is not None:
        conditions["median_episode_spearman_drop_max"] = median_spearman_delta >= -spearman_drop_max
    return {"status": "PASS" if all(conditions.values()) else "FAIL", "paired_parent_episodes": len(paired), "median_episode_top30_delta": median_delta, "median_episode_spearman_delta": median_spearman_delta, "improved_parent_episode_count": improved, "conditions": conditions, "per_episode": paired, "goal_donors_are_nested_and_not_independent": True}


def mechanism_gates(freeze: Mapping[str, Any], records: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]], timing: Mapping[str, Any]) -> dict[str, Any]:
    tail_by_seed, structure_by_seed = {}, {}
    for seed in freeze["training"]["initialization_seeds"]:
        seed = int(seed)
        projected = records[(seed, "exact_projected_only")]
        oracle = records[(seed, "oracle_y_learned_tail")]
        scalar = records[(seed, "scalar_aux")]
        structured = records[(seed, "projected_tail")]
        tail_by_seed[str(seed)] = paired_episode_gate(projected, oracle, 0.05, 6)
        structure = paired_episode_gate(scalar, structured, 0.05, 6, spearman_drop_max=0.02)
        ratio = float(timing[str(seed)]["arms"]["projected_tail"]["p50_ms"]) / max(float(timing[str(seed)]["arms"]["scalar_aux"]["p50_ms"]), 1e-12)
        structure["native_p50_ratio_projected_tail_over_scalar_aux"] = ratio
        structure["conditions"]["native_p50_ratio_max_1_10"] = ratio <= 1.10
        structure["status"] = "PASS" if all(structure["conditions"].values()) else "FAIL"
        structure_by_seed[str(seed)] = structure
    tail = {"status": "PASS" if all(value["status"] == "PASS" for value in tail_by_seed.values()) else "FAIL", "per_seed": tail_by_seed}
    structure = {"status": "PASS" if all(value["status"] == "PASS" for value in structure_by_seed.values()) else "FAIL", "per_seed": structure_by_seed}
    return {"tail_signal": tail, "structural_signal": structure}


def run(args: argparse.Namespace) -> dict[str, Any]:
    host = require_compute_node()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = load_json(args.freeze.resolve())
    validate_freeze(freeze)
    protocol_path = args.freeze.resolve().with_name("PROTOCOL.zh.md")
    for path in (args.transfer_root, args.lewm_root, args.stablewm_root, args.stablewm_home, args.dataset, args.training_rows, args.manifest, args.temporal_freeze, args.freeze, protocol_path):
        if not path.exists():
            raise FileNotFoundError(path)
    identity = source_identity(args, freeze)
    # Commit checks and PBS guard complete before importing the reference model helpers.
    reference = load_reference(args)
    base = reference.base
    import torch

    identity["reference_helpers"] = {
        name: file_identity(Path(path))
        for name, path in {
            "state_action_runner": reference.__file__,
            "recurrent_student_source": base.__file__,
            "fresh_row_builder": reference.ema.__file__,
        }.items()
    }
    identity["runtime"] = {"python": sys.executable, "python_version": sys.version.split()[0], "device": "CUDA"}

    if not torch.cuda.is_available():
        raise RuntimeError("PBS allocation has no usable CUDA device")
    torch.cuda.set_device(0)
    selftest = self_checks(torch, build_model_class(torch, __import__("torch.nn", fromlist=["nn"])))
    manifest = load_json(args.manifest.resolve())
    if manifest.get("schema") != "lewm-recurrent-student.context-manifest":
        raise ValueError("unexpected context manifest schema")
    train_rows_raw = torch.load(args.training_rows.resolve(), map_location="cpu", weights_only=False)
    train_rows = validate_training_rows(torch, train_rows_raw, manifest, freeze)
    temporal_freeze = load_json(args.temporal_freeze.resolve())
    if not isinstance(temporal_freeze.get("evaluation"), Mapping):
        raise ValueError("temporal freeze has no official evaluation action settings")
    dev_ids, test_ids, selection_meta = select_episodes(args.dataset.resolve(), freeze["data"], manifest)
    train_data, train_goal_meta = make_train_queries(torch, train_rows)
    train_goal_path = args.out / "training_goal_sources.jsonl"
    append_jsonl(train_goal_path, [
        {"split": "train", "context_id": str(row["context_id"]), "episode_id": int(row["episode_id"]), "goal_source_episode_ids": sources}
        for row, sources in zip(train_rows, train_data["goal_sources"])
    ])
    official = base.load_official_checkpoint(args.stablewm_home.resolve())
    official.requires_grad_(False)
    official.eval()
    identity["runtime_interfaces"] = {
        "official_model_class": f"{type(official).__module__}.{type(official).__qualname__}",
        "official_predict_callable": callable(getattr(official, "predict", None)),
        "official_criterion_callable": callable(getattr(official, "criterion", None)),
        "official_encoder_callable": callable(getattr(official, "encode", None)),
        "teacher_targets_helper": callable(getattr(base, "official_teacher_targets", None)),
        "official_objective_helper": callable(getattr(base, "_official_objective", None)),
        "fresh_row_builder": callable(getattr(reference.ema, "build_fresh_rows", None)),
    }
    training_cost_check = verified_official_costs(torch, base, official, train_rows, freeze)
    dev_rows, dev_meta = build_fresh(torch, reference, official, args.dataset.resolve(), dev_ids, freeze["data"]["development_action_seeds"], freeze["data"]["anchors"], temporal_freeze)
    dev_cost_check = verified_official_costs(torch, base, official, dev_rows, freeze)
    dev_goals = make_goal_queries(dev_rows, dev_ids, same_stratum=True)
    train_terminal = train_data["terminals"].to("cuda")
    mean, components, pca_meta, pca64 = fit_projection(torch, train_terminal, int(freeze["models"]["rank"]))
    if components.shape != (LATENT_DIM, int(freeze["models"]["rank"])):
        raise ValueError("training-only PCA rank/shape changed")
    validate_projection(torch, components.to("cuda"), 1e-5)
    train_tensors, train_checks = prepare_train_tensors(torch, train_rows, train_data, mean, components, pca64, freeze)
    QueryPredictor = build_model_class(torch, __import__("torch.nn", fromlist=["nn"]))
    schedule_gen = torch.Generator(device="cpu").manual_seed(int(freeze["training"]["schedule_seed"]))
    schedule = [torch.randperm(len(train_rows), generator=schedule_gen)[:int(freeze["training"]["batch_contexts"])] for _ in range(int(freeze["training"]["updates"]))]
    training_results: dict[str, Any] = {}
    trace_path = args.out / "training_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    for seed in freeze["training"]["initialization_seeds"]:
        training_results[str(seed)] = {}
        for arm in ARMS:
            training_results[str(seed)][arm] = train_model(torch, QueryPredictor, arm, int(seed), schedule, train_tensors, args.out, (mean, components), freeze, trace_path)
    parameter_counts = {arm: {
        "training": training_results[str(freeze["training"]["initialization_seeds"][0])][arm]["training_parameter_count"],
        "deployed": training_results[str(freeze["training"]["initialization_seeds"][0])][arm]["deployed_parameter_count"],
    } for arm in ARMS}
    dev_eval, dev_records = evaluate_split(torch, base, official, QueryPredictor, "development", dev_rows, dev_goals, freeze["data"]["development_action_seeds"], freeze["training"]["initialization_seeds"], args.out, (mean, components), train_tensors, args.out / "evaluation_details.jsonl")
    dev_timing_rows = dev_rows[:int(freeze["evaluation"]["timing"]["blocks"])]
    timing = time_all_models(torch, base, official, QueryPredictor, freeze["training"]["initialization_seeds"], dev_timing_rows, freeze["data"]["development_action_seeds"][0], args.out, (mean, components), train_tensors, freeze, args.out / "timing_samples.jsonl")
    selection = dev_selection(freeze, dev_records, timing, parameter_counts, selection_meta)
    selection["timing_rows"] = [{"context_id": row["context_id"], "episode_id": int(row["episode_id"]), "anchor": str(row["stratum"]), "action_seed": int(freeze["data"]["development_action_seeds"][0]), "action_bank_index": 0, "goal_type": "own"} for row in dev_timing_rows]
    write_json(args.out / "selection.json", selection, atomic=True)

    # Test HDF5 slices, encoder latents, teacher targets and scores are created only after selection.json exists.
    if not (args.out / "selection.json").is_file():
        raise RuntimeError("development selection was not durably written before test generation")
    test_rows, test_meta = build_fresh(torch, reference, official, args.dataset.resolve(), test_ids, freeze["data"]["test_action_seeds"], freeze["data"]["anchors"], temporal_freeze)
    test_cost_check = verified_official_costs(torch, base, official, test_rows, freeze)
    test_goals = make_goal_queries(test_rows, test_ids, same_stratum=True)
    test_eval, test_records = evaluate_split(torch, base, official, QueryPredictor, "test", test_rows, test_goals, freeze["data"]["test_action_seeds"], freeze["training"]["initialization_seeds"], args.out, (mean, components), train_tensors, args.out / "evaluation_details.jsonl")
    test_timing_rows = test_rows[:int(freeze["evaluation"]["timing"]["blocks"])]
    test_timing = time_all_models(torch, base, official, QueryPredictor, freeze["training"]["initialization_seeds"], test_timing_rows, freeze["data"]["test_action_seeds"][0], args.out, (mean, components), train_tensors, freeze, args.out / "timing_samples.jsonl")
    mechanisms = mechanism_gates(freeze, test_records, test_timing)
    selected = selection["selected_arm"]
    final_quality = {}
    for seed in freeze["training"]["initialization_seeds"]:
        final_quality[str(seed)] = predictor_quality(freeze, test_records[(int(seed), selected)], test_timing[str(seed)]["arms"][selected]) if selected else None
    final_pass = bool(selected) and all(item["status"] == "PASS" for item in final_quality.values())
    final_gate = {
        "schema": "lewm.goal-query.pilot.stage1-gate", "status": "GO_PREDICTOR_ONLY" if final_pass else "NO_GO_STOP",
        "selected_arm": selected, "development_selection_status": selection["status"],
        "final_test_quality_both_seeds": "PASS" if final_pass else "FAIL_OR_NO_DEV_SELECTION",
        "per_seed_final_quality": final_quality, "tail_signal": mechanisms["tail_signal"],
        "structural_signal": mechanisms["structural_signal"],
        "next_scope": "bounded adaptive CEM gate may be considered" if final_pass else "stop; no adaptive CEM or closed-loop escalation",
        "claim_boundary": "predictor-level only; no CEM, planner, or closed-loop claim",
    }
    write_json(args.out / "stage1_gate.json", final_gate)
    run_summary = {
        "schema": "lewm.goal-query.pilot.run-summary", "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE", "host": host, "pbs_job_id": os.environ["PBS_JOBID"],
        "source_identity": identity,
        "data_selection": selection_meta,
        "goal_donor_contract": {"training": train_goal_meta, "evaluation": "own goal plus next three distinct parent-episode goals in the same split and same anchor stratum", "project_wide_prior_evaluation_exclusion_claim": False},
        "integrity": {"self_checks": selftest, "training_official_cost": training_cost_check, "development_official_cost": dev_cost_check, "test_official_cost": test_cost_check, "training_projection": pca_meta, "training_labels_and_decomposition": train_checks},
        "training": {"updates": int(freeze["training"]["updates"]), "batch_contexts": int(freeze["training"]["batch_contexts"]), "same_context_schedule_all_arms_and_seeds": True, "same_context_action_goal_tuples": True, "optimizer": freeze["training"]["optimizer"], "scales_and_floors_training_only": train_tensors["scales"], "arms": training_results},
        "parameter_counts": parameter_counts,
        "development": {"fresh_rows": dev_meta, "arms": dev_eval, "timing_by_seed": timing, "selection_file": str((args.out / "selection.json").resolve())},
        "test": {"fresh_rows": test_meta, "arms": test_eval, "timing_by_seed": test_timing, "test_generated_after_selection": True},
        "stage1_gate": final_gate,
        "details": {"evaluation_jsonl": str((args.out / "evaluation_details.jsonl").resolve()), "training_goal_sources_jsonl": str(train_goal_path.resolve()), "training_trace_jsonl": str(trace_path.resolve()), "timing_samples_jsonl": str((args.out / "timing_samples.jsonl").resolve())},
        "scope": {"official_criterion_unchanged": True, "rank32_from_training_teacher_terminal_targets_only": True, "test_not_used_for_selection": True, "teacher_shadow_excluded_from_predictor_timing": True, "encoder_full_cem_and_closed_loop_speedups_not_claimed": True},
    }
    write_json(args.out / "run_summary.json", run_summary)
    return run_summary


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        # Keep engineering failures distinct from a scientific NO-GO and preserve a nonzero process result.
        args.out.mkdir(parents=True, exist_ok=True)
        failure = {"schema": "lewm.goal-query.pilot.error", "status": "INCONCLUSIVE_ENGINEERING_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "out": str(args.out.resolve())}
        write_json(args.out / "ERROR.json", failure)
        write_json(args.out / "stage1_gate.json", {"schema": "lewm.goal-query.pilot.stage1-gate", "status": "INCONCLUSIVE", "error": failure, "no_scientific_NO_GO_inference": True})
        write_json(args.out / "run_summary.json", {"schema": "lewm.goal-query.pilot.run-summary", "schema_version": 1, "status": "INCONCLUSIVE_ENGINEERING_FAILURE", "error": failure})
        raise
    print(json.dumps({"status": result["status"], "gate": result["stage1_gate"]["status"], "out": str(args.out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""LeWM PushT latent-delta Step 1/2 oracle diagnostic.

This runner deliberately keeps the experiment at a fixed-observation,
predictor/CEM diagnostic boundary.  It does not train a cheap predictor, claim
latency savings, or interact with the environment.  The official LeWM
predictor is run first; only its latent deltas are projected through frozen
calibration PCA bases for the oracle arms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


LATENT_DIM = 192
HORIZON = 5
ACTION_DIM = 10
OBSERVED_OFFSETS = (0, 5, 10, 15, 20, 25)
TOPK = 30
RANKS = (0, 4, 8, 16, 32, 64, 96, 128, 160, 192)
RANDOM_BASIS_SEED = 20300964
FULL_RANK_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing model work outside PBS")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES is required for the frozen GPU run")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def tensor(value: Any, *, dtype: Any = None, device: Any = None) -> Any:
    import torch

    result = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if device is not None:
        result = result.to(device=device)
    return result


def finite_tensor(value: Any) -> bool:
    import torch

    return bool(torch.is_tensor(value) and torch.isfinite(value).all().item())


def load_lewm_helpers() -> tuple[Any, Any]:
    """Import the already-frozen student-transfer data helpers."""
    helper_root = Path(__file__).resolve().parents[1] / "02-horizon-weighted-recurrent-student" / "lewm-transfer"
    if str(helper_root) not in sys.path:
        sys.path.insert(0, str(helper_root))
    from run_lewm_recurrent_student import HDF5EpisodeSliceReader, _normalise_pixels

    return HDF5EpisodeSliceReader, _normalise_pixels


def load_iteration_helpers(iteration_root: Path) -> dict[str, Any]:
    if str(iteration_root) not in sys.path:
        sys.path.insert(0, str(iteration_root))
    import run_lewm_pusht_iteration as iteration

    return {
        "prepare_policy_info": iteration.prepare_policy_info,
        "make_solver": iteration.make_solver,
        "TraceCallback": iteration.TraceCallback,
        "copy_info": iteration.copy_info,
        "set_seed": iteration.set_seed,
        "load_local_checkpoint": iteration.load_local_checkpoint,
    }


def load_model(stablewm_home: Path) -> Any:
    import torch

    path = stablewm_home / "pusht" / "lewm_object.ckpt"
    if not path.is_file():
        raise FileNotFoundError(path)
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(model, "encode") or not hasattr(model, "predict") or not hasattr(model, "criterion"):
        raise TypeError(f"checkpoint is not an official LeWM predictor/criterion object: {path}")
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def load_rows(path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError("prepared_rows.pt must contain a list")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "lewm-recurrent-student.context-manifest":
        raise ValueError("unexpected context manifest schema")
    train = [row for row in rows if row.get("split") == "train"]
    heldout = [row for row in rows if row.get("split") == "heldout"]
    if len(train) != 256 or len(heldout) != 8:
        raise ValueError(f"expected 256 train and 8 heldout rows, got {len(train)}/{len(heldout)}")
    if len(manifest.get("splits", {}).get("train", [])) != 256 or len(manifest.get("splits", {}).get("heldout", [])) != 8:
        raise ValueError("manifest does not contain the frozen 256/8 split")
    by_id = {str(row["context_id"]): row for row in rows}
    for split in ("train", "heldout"):
        for entry in manifest["splits"][split]:
            if str(entry["context_id"]) not in by_id:
                raise ValueError(f"prepared rows missing manifest context {entry['context_id']}")
    return rows, manifest


def check_row_shapes(rows: Sequence[Mapping[str, Any]]) -> None:
    import torch

    for row in rows:
        split = row.get("split")
        latent = tensor(row["latent_history"], dtype=torch.float32)
        if tuple(latent.shape) != (1, 1, LATENT_DIM):
            raise ValueError(f"{split} latent_history must be [1,1,192], got {tuple(latent.shape)}")
        actions = tensor(row["future_actions"], dtype=torch.float32)
        targets = tensor(row["teacher_targets"], dtype=torch.float32)
        expected = (4, HORIZON, ACTION_DIM) if split == "train" else (2, 300, HORIZON, ACTION_DIM)
        target_expected = (4, HORIZON, LATENT_DIM) if split == "train" else (2, 300, HORIZON, LATENT_DIM)
        if tuple(actions.shape) != expected or tuple(targets.shape) != target_expected:
            raise ValueError(f"{split} action/target shape mismatch: {tuple(actions.shape)} / {tuple(targets.shape)}")
        if not finite_tensor(latent) or not finite_tensor(actions) or not finite_tensor(targets):
            raise ValueError(f"non-finite prepared row: {row.get('context_id')}")


def teacher_deltas(row: Mapping[str, Any]) -> Any:
    """Convert detached z(t+1)..z(t+5) targets to five transition deltas."""
    import torch

    target = tensor(row["teacher_targets"], dtype=torch.float32)
    initial = tensor(row["latent_history"], dtype=torch.float32)[0, 0]
    if target.ndim == 3:
        previous = torch.cat((initial[None, None, :].expand(target.shape[0], 1, -1), target[:, :-1]), dim=1)
    elif target.ndim == 4:
        previous = torch.cat((initial[None, None, None, :].expand(target.shape[0], target.shape[1], 1, -1), target[:, :, :-1]), dim=2)
    else:
        raise ValueError(f"unexpected teacher target rank: {target.ndim}")
    return target - previous


def collect_observed_deltas(
    model: Any,
    dataset_path: Path,
    manifest: Mapping[str, Any],
    normalise_pixels: Any,
    reader_cls: Any,
) -> dict[str, Any]:
    """Encode frozen offsets 0,5,...,25 and return adjacent observed deltas."""
    import torch

    result: dict[str, Any] = {}
    with reader_cls(dataset_path) as reader:
        for split in ("train", "heldout"):
            for row in manifest["splits"][split]:
                context_id = str(row["context_id"])
                episode = reader.episode_slice(int(row["episode_id"]))
                base_step = int(row.get("history_steps", [0])[0])
                steps = [base_step + offset for offset in OBSERVED_OFFSETS]
                if min(steps) < 0 or max(steps) >= episode["length"]:
                    raise IndexError(f"observed offsets exceed episode for {context_id}")
                frames = normalise_pixels(episode["pixels"][steps]).unsqueeze(0).to("cuda")
                with torch.no_grad():
                    encoded = model.encode({"pixels": frames})["emb"]
                if tuple(encoded.shape) != (1, len(OBSERVED_OFFSETS), LATENT_DIM):
                    raise RuntimeError(f"observed encoder shape for {context_id}: {tuple(encoded.shape)}")
                result[context_id] = (encoded[:, 1:] - encoded[:, :-1]).squeeze(0).detach().cpu().float()
    return result


@dataclass
class PCABasis:
    mean: Any
    components: Any
    singular_values: Any
    total_centered_energy: float
    name: str

    @classmethod
    def fit(cls, vectors: Any, name: str) -> "PCABasis":
        import torch

        x = tensor(vectors, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != LATENT_DIM:
            raise ValueError(f"PCA input must be [N,192], got {tuple(x.shape)}")
        mean = x.mean(dim=0)
        centered = x - mean
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        total = float(centered.square().sum().item())
        return cls(mean=mean, components=vh, singular_values=singular, total_centered_energy=total, name=name)

    def project(self, values: Any, rank: int) -> Any:
        import torch

        if rank < 0 or rank > LATENT_DIM:
            raise ValueError(f"invalid rank {rank}")
        value = tensor(values, dtype=torch.float64)
        centered = value - self.mean.to(value.device)
        if rank == 0:
            return self.mean.to(value.device).expand_as(value)
        basis = self.components[:rank].to(value.device)
        return self.mean.to(value.device) + (centered @ basis.T) @ basis

    def frontier(self, values: Any, ranks: Sequence[int]) -> dict[str, Any]:
        import torch

        value = tensor(values, dtype=torch.float64)
        centered = value - self.mean
        denom_centered = centered.square().sum(dim=-1).clamp_min(1e-18)
        denom_raw = value.square().sum(dim=-1).clamp_min(1e-18)
        out: dict[str, Any] = {}
        for rank in ranks:
            approx = self.project(value, int(rank))
            residual = value - approx
            retained = 1.0 - residual.square().sum(dim=-1) / denom_centered
            out[str(rank)] = {
                "rank": int(rank),
                "centered_variance_retained_mean": float(retained.mean().item()),
                "centered_variance_retained_median": float(retained.median().item()),
                "centered_variance_retained_minimum": float(retained.min().item()),
                "relative_mse_mean": float((residual.square().sum(dim=-1) / denom_raw).mean().item()),
                "relative_mse_median": float((residual.square().sum(dim=-1) / denom_raw).median().item()),
                "relative_mse_maximum": float((residual.square().sum(dim=-1) / denom_raw).max().item()),
                "cosine_mean": float(torch.nn.functional.cosine_similarity(value, approx, dim=-1).mean().item()),
            }
        return out

    def metadata(self, ranks: Sequence[int]) -> dict[str, Any]:
        energy = self.singular_values.square()
        total = max(float(self.total_centered_energy), 1e-18)
        return {
            "name": self.name,
            "sample_count": int(self.components.shape[1] if self.components.ndim == 2 else 0),
            "latent_dim": LATENT_DIM,
            "singular_values": [float(x) for x in self.singular_values.tolist()],
            "cumulative_centered_variance": {
                str(rank): float(energy[:rank].sum().item() / total) if rank else 0.0 for rank in ranks
            },
        }


def delta_matrix(values: Any) -> Any:
    import torch

    value = tensor(values, dtype=torch.float64)
    return value.reshape(-1, LATENT_DIM)


def reconstruct_future(initial: Any, deltas: Any, basis: PCABasis, rank: int) -> Any:
    import torch

    raw = tensor(deltas, dtype=torch.float64)
    flat = raw.reshape(-1, LATENT_DIM)
    projected = basis.project(flat, rank).reshape(raw.shape)
    prefix_shape = projected.shape[:-2]
    state = tensor(initial, dtype=torch.float64).reshape(
        *([1] * len(prefix_shape)), LATENT_DIM
    )
    state = state.expand(*prefix_shape, LATENT_DIM).clone()
    futures = []
    for step in range(projected.shape[-2]):
        state = state + projected[..., step, :]
        futures.append(state.clone())
    return torch.stack(futures, dim=-2)


def _rank_values(values: Any) -> Any:
    import torch

    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64, device=values.device)
    return ranks


def spearman(first: Any, second: Any) -> float:
    import torch

    x = tensor(first, dtype=torch.float64).reshape(-1)
    y = tensor(second, dtype=torch.float64).reshape(-1)
    if x.numel() < 2:
        return 1.0
    xr = _rank_values(x)
    yr = _rank_values(y)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denominator = torch.linalg.vector_norm(xc) * torch.linalg.vector_norm(yc)
    if float(denominator) <= 1e-18:
        return 1.0 if torch.allclose(x, y) else 0.0
    return float((xc * yc).sum().item() / denominator.item())


def topk_overlap(first: Any, second: Any, k: int = TOPK) -> float:
    import torch

    k = min(k, int(tensor(first).numel()))
    a = set(torch.topk(tensor(first), k=k, largest=False).indices.tolist())
    b = set(torch.topk(tensor(second), k=k, largest=False).indices.tolist())
    return float(len(a & b) / max(k, 1))


def objective(model: Any, row: Mapping[str, Any], future: Any) -> Any:
    import torch

    prediction = tensor(future, dtype=torch.float32, device="cuda")
    initial = tensor(row["latent_history"], dtype=prediction.dtype, device=prediction.device)
    initial = initial[:, -1:, :].expand(prediction.shape[0], -1, -1)
    predicted_emb = torch.cat((initial, prediction), dim=1).unsqueeze(0)
    goal = tensor(row["goal_emb"], dtype=prediction.dtype, device=prediction.device)
    value = model.criterion({"predicted_emb": predicted_emb, "goal_emb": goal})
    if not torch.is_tensor(value):
        raise TypeError("official criterion did not return a tensor")
    return value.reshape(-1)


def first_action_metrics(actions: Any, teacher_index: int, approx_index: int) -> dict[str, float]:
    import torch

    ref = tensor(actions[teacher_index, 0, :2], dtype=torch.float64)
    got = tensor(actions[approx_index, 0, :2], dtype=torch.float64)
    difference = got - ref
    return {
        "normalized_l2": float(torch.linalg.vector_norm(difference).item() / max(float(torch.linalg.vector_norm(ref).item()), 1e-8)),
        "absolute_max": float(difference.abs().max().item()),
    }


def trajectory_svd_oracle(heldout_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    matrices = []
    metadata = []
    for row in heldout_rows:
        deltas = teacher_deltas(row)
        for bank in range(deltas.shape[0]):
            for candidate in range(deltas.shape[1]):
                matrices.append(deltas[bank, candidate])
                metadata.append({"context_id": str(row["context_id"]), "bank": bank, "candidate": candidate})
    matrix = torch.stack(matrices).double()
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    total = matrix.square().sum(dim=(1, 2)).clamp_min(1e-18)
    aggregate: dict[str, Any] = {}
    for rank in range(1, HORIZON + 1):
        # Batched rank-r reconstruction of each [5,192] trajectory matrix.
        reconstruction = (u[:, :, :rank] * singular[:, None, :rank]) @ vh[:, :rank, :]
        residual = matrix - reconstruction
        rel = residual.square().sum(dim=(1, 2)) / total
        retained = 1.0 - rel
        aggregate[str(rank)] = {
            "rank": rank,
            "trajectory_count": int(matrix.shape[0]),
            "relative_mse_median": float(rel.median().item()),
            "relative_mse_mean": float(rel.mean().item()),
            "relative_mse_maximum": float(rel.max().item()),
            "retained_energy_median": float(retained.median().item()),
            "retained_energy_minimum": float(retained.min().item()),
        }
    # Per-sample values are retained because this is an oracle lower-bound
    # diagnostic, not a population-level aggregate.
    per_sample = []
    all_rel = torch.zeros((matrix.shape[0], HORIZON), dtype=torch.float64)
    all_retained = torch.zeros_like(all_rel)
    for rank in range(1, HORIZON + 1):
        reconstruction = (u[:, :, :rank] * singular[:, None, :rank]) @ vh[:, :rank, :]
        residual = matrix - reconstruction
        rel = residual.square().sum(dim=(1, 2)) / total
        all_rel[:, rank - 1] = rel
        all_retained[:, rank - 1] = 1.0 - rel
    for index, meta in enumerate(metadata):
        per_sample.append({
            **meta,
            "relative_mse_by_rank": [float(x) for x in all_rel[index].tolist()],
            "retained_energy_by_rank": [float(x) for x in all_retained[index].tolist()],
        })
    return {"ranks": list(range(1, HORIZON + 1)), "aggregate": aggregate, "per_sample": per_sample}


def pca_frontier(
    observed_train: Any,
    model_train: Any,
    observed_heldout: Any,
    model_heldout: Any,
    ranks: Sequence[int],
) -> dict[str, Any]:
    observed_basis = PCABasis.fit(delta_matrix(observed_train), "observed")
    model_basis = PCABasis.fit(delta_matrix(model_train), "model")
    observed_flat = delta_matrix(observed_heldout)
    model_flat = delta_matrix(model_heldout)
    result = {
        "basis": {
            "observed": observed_basis.metadata(ranks),
            "model": model_basis.metadata(ranks),
        },
        "heldout": {
            "observed_on_observed_basis": observed_basis.frontier(observed_flat, ranks),
            "model_on_model_basis": model_basis.frontier(model_flat, ranks),
            "observed_on_model_basis": model_basis.frontier(observed_flat, ranks),
            "model_on_observed_basis": observed_basis.frontier(model_flat, ranks),
        },
        "heldout_by_horizon": {},
    }
    for horizon_index in range(HORIZON):
        observed_horizon = tensor(observed_heldout, dtype=None)[..., horizon_index, :].reshape(-1, LATENT_DIM)
        model_horizon = tensor(model_heldout, dtype=None)[..., horizon_index, :].reshape(-1, LATENT_DIM)
        result["heldout_by_horizon"][str(horizon_index + 1)] = {
            "observed_on_observed_basis": observed_basis.frontier(observed_horizon, ranks),
            "model_on_model_basis": model_basis.frontier(model_horizon, ranks),
            "observed_on_model_basis": model_basis.frontier(observed_horizon, ranks),
            "model_on_observed_basis": observed_basis.frontier(model_horizon, ranks),
        }
    result["basis"]["observed"]["fit_sample_count"] = int(delta_matrix(observed_train).shape[0])
    result["basis"]["model"]["fit_sample_count"] = int(delta_matrix(model_train).shape[0])
    return {"result": result, "observed_basis": observed_basis, "model_basis": model_basis}


def structure_gate_report(
    frontier: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
    """Read the frozen structure gate without changing its thresholds."""
    gate = freeze["step1"]["structure_gate"]
    max_rank = int(gate["maximum_rank"])
    min_retained = float(gate["minimum_heldout_centered_variance_retained"])
    max_relative_mse = float(gate["maximum_heldout_relative_mse"])
    report: dict[str, Any] = {
        "thresholds": {
            "maximum_rank": max_rank,
            "minimum_heldout_centered_variance_retained": min_retained,
            "maximum_heldout_relative_mse": max_relative_mse,
        },
        "primary_same_basis": {},
    }
    for name, frontier_name in (
        ("observed", "observed_on_observed_basis"),
        ("model", "model_on_model_basis"),
    ):
        candidates: list[int] = []
        diagnostics: dict[str, Any] = {}
        for rank_key, values in frontier[frontier_name].items():
            rank = int(rank_key)
            passed = bool(
                rank <= max_rank
                and float(values["centered_variance_retained_minimum"]) >= min_retained
                and float(values["relative_mse_maximum"]) <= max_relative_mse
            )
            diagnostics[rank_key] = {
                "centered_variance_retained_minimum": values["centered_variance_retained_minimum"],
                "relative_mse_maximum": values["relative_mse_maximum"],
                "passed": passed,
            }
            if passed:
                candidates.append(rank)
        report["primary_same_basis"][name] = {
            "passing_ranks": candidates,
            "rank_le_64_overall_pass": bool(candidates),
            "per_rank": diagnostics,
        }
    return report


def candidate_metrics(
    model: Any,
    row: Mapping[str, Any],
    deltas: Any,
    basis: PCABasis,
    rank: int,
    basis_name: str,
) -> dict[str, Any]:
    import torch

    actions = tensor(row["future_actions"], dtype=torch.float32)
    targets = tensor(row["teacher_targets"], dtype=torch.float32)
    teacher_objectives = tensor(row["teacher_objective"], dtype=torch.float32)
    initial = tensor(row["latent_history"], dtype=torch.float64)[0, 0]
    approx_future = reconstruct_future(initial, deltas, basis, rank).float()
    return candidate_metrics_from_future(
        model, row, actions, targets, teacher_objectives, approx_future, basis_name, rank
    )


def candidate_metrics_from_future(
    model: Any,
    row: Mapping[str, Any],
    actions: Any,
    targets: Any,
    teacher_objectives: Any,
    approx_future: Any,
    basis_name: str,
    rank: int,
) -> dict[str, Any]:
    import torch

    block_records = []
    with torch.no_grad():
        for bank in range(approx_future.shape[0]):
            teacher = teacher_objectives[bank]
            approximate = objective(model, row, approx_future[bank])
            target = targets[bank].to("cuda")
            delta_latent = approx_future[bank].to("cuda") - target
            teacher_index = int(torch.argmin(teacher).item())
            approximate_index = int(torch.argmin(approximate).item())
            action = first_action_metrics(actions[bank], teacher_index, approximate_index)
            objective_regret = float((teacher[approximate_index] - teacher.min()).item())
            mse = delta_latent.square().mean(dim=(0, 2))
            cosine = torch.nn.functional.cosine_similarity(approx_future[bank].to("cuda"), target, dim=2).mean(dim=0)
            block_records.append({
                "context_id": str(row["context_id"]),
                "bank": bank,
                "basis": basis_name,
                "rank": rank,
                "candidate_count": int(approximate.numel()),
                "spearman": spearman(teacher, approximate.cpu()),
                "top30_overlap": topk_overlap(teacher, approximate.cpu()),
                "argmin_match": bool(teacher_index == approximate_index),
                "teacher_argmin": teacher_index,
                "approximate_argmin": approximate_index,
                "first_action": action,
                "teacher_objective_regret": objective_regret,
                "objective_mse": float((approximate.cpu() - teacher).square().mean().item()),
                "latent_mse_by_horizon": [float(x.item()) for x in mse.cpu()],
                "latent_cosine_by_horizon": [float(x.item()) for x in cosine.cpu()],
                "latent_relative_mse": float(delta_latent.square().mean().item() / max(target.square().mean().item(), 1e-12)),
                "finite": bool(torch.isfinite(approximate).all().item() and torch.isfinite(approx_future[bank]).all().item()),
            })
    return {"basis": basis_name, "rank": rank, "per_block": block_records}


def trajectory_rank_reconstructions(initial: Any, deltas: Any, ranks: Sequence[int]) -> dict[int, Any]:
    """Per-trajectory oracle SVD, reusing one batched decomposition."""
    import torch

    raw = tensor(deltas, dtype=torch.float64)
    prefix = raw.shape[:-2]
    matrix = raw.reshape(-1, HORIZON, LATENT_DIM)
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    outputs: dict[int, Any] = {}
    for rank in ranks:
        projected = (u[:, :, :rank] * singular[:, None, :rank]) @ vh[:, :rank, :]
        projected = projected.reshape(*prefix, HORIZON, LATENT_DIM)
        state = tensor(initial, dtype=torch.float64).reshape(
            *([1] * len(prefix)), LATENT_DIM
        ).expand(*prefix, LATENT_DIM).clone()
        futures = []
        for step in range(HORIZON):
            state = state + projected[..., step, :]
            futures.append(state.clone())
        outputs[int(rank)] = torch.stack(futures, dim=-2)
    return outputs


def summarise_candidate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("empty candidate records")
    first_l2 = [float(item["first_action"]["normalized_l2"]) for item in records]
    first_abs = [float(item["first_action"]["absolute_max"]) for item in records]
    spears = [float(item["spearman"]) for item in records]
    top30 = [float(item["top30_overlap"]) for item in records]
    return {
        "basis": records[0]["basis"],
        "rank": int(records[0]["rank"]),
        "blocks": len(records),
        "finite": all(bool(item["finite"]) for item in records),
        "spearman_median": float(statistics.median(spears)),
        "spearman_minimum": min(spears),
        "top30_overlap_median": float(statistics.median(top30)),
        "top30_overlap_minimum": min(top30),
        "argmin_match_rate": float(sum(bool(item["argmin_match"]) for item in records) / len(records)),
        "first_action_normalized_l2_median": float(statistics.median(first_l2)),
        "first_action_normalized_l2_maximum": max(first_l2),
        "first_action_absolute_max_median": float(statistics.median(first_abs)),
        "first_action_absolute_maximum": max(first_abs),
        "teacher_objective_regret_median": float(statistics.median(float(item["teacher_objective_regret"]) for item in records)),
        "teacher_objective_regret_maximum": max(float(item["teacher_objective_regret"]) for item in records),
        "objective_mse_median": float(statistics.median(float(item["objective_mse"]) for item in records)),
        "latent_relative_mse_median": float(statistics.median(float(item["latent_relative_mse"]) for item in records)),
        "latent_mse_by_horizon_median": [float(statistics.median(float(item["latent_mse_by_horizon"][h]) for item in records)) for h in range(HORIZON)],
        "latent_cosine_by_horizon_median": [float(statistics.median(float(item["latent_cosine_by_horizon"][h]) for item in records)) for h in range(HORIZON)],
    }


def candidate_gate(summary: Mapping[str, Any], freeze: Mapping[str, Any]) -> bool:
    gate = freeze["step2"]["candidate_gate"]
    return bool(
        float(summary["spearman_median"]) >= float(gate["minimum_median_spearman"])
        and float(summary["spearman_minimum"]) >= float(gate["minimum_block_spearman"])
        and float(summary["top30_overlap_median"]) >= float(gate["minimum_median_top30_overlap"])
        and float(summary["top30_overlap_minimum"]) >= float(gate["minimum_block_top30_overlap"])
    )


def random_orthonormal_basis(rank: int, seed: int) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((LATENT_DIM, rank), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.T.contiguous()


class OracleProjectedModel:
    """CEM-facing wrapper: full rollout, delta projection, unchanged criterion."""

    def __init__(self, model: Any, basis: PCABasis, rank: int) -> None:
        self.model = model
        self.basis = basis
        self.rank = int(rank)

    def parameters(self) -> Any:
        return self.model.parameters()

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Any) -> Any:
        import torch

        with torch.no_grad():
            device = next(self.model.parameters()).device
            working = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in info_dict.items()
            }
            full = self.model.rollout(working, action_candidates)
            predicted = full["predicted_emb"]
            if predicted.ndim != 4 or predicted.shape[-1] != LATENT_DIM or predicted.shape[2] != HORIZON + 1:
                raise RuntimeError(f"unexpected official predicted_emb shape: {tuple(predicted.shape)}")
            full_delta = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
            flat = full_delta.reshape(-1, LATENT_DIM).double().cpu()
            projected = self.basis.project(flat, self.rank).to(device=predicted.device, dtype=predicted.dtype).reshape_as(full_delta)
            initial = predicted[:, :, :1, :]
            future = initial + torch.cumsum(projected, dim=2)
            projected_result = dict(full)
            projected_result["predicted_emb"] = torch.cat((initial, future), dim=2)
            if "goal_emb" not in projected_result:
                projected_result["goal_emb"] = self.model.encode({"pixels": working["goal"][:, 0]})["emb"]
            return self.model.criterion(projected_result)


def run_cem_call(
    model: Any,
    info: Mapping[str, Any],
    settings: Mapping[str, Any],
    seed: int,
    action_space: Any,
    config: Any,
    helpers: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    trace = helpers["TraceCallback"]()
    helpers["set_seed"](seed)
    solver = helpers["make_solver"](model, dict(settings), seed, trace, action_space, config)
    working = helpers["copy_info"](info)
    torch.cuda.synchronize()
    output = solver.solve(working)
    torch.cuda.synchronize()
    actions = output["actions"].detach().cpu()
    return {
        "seed": seed,
        "final_actions": actions.tolist(),
        "first_actions": actions[:, 0].tolist(),
        "costs": jsonable(output.get("costs", [])),
        "decision_trace": jsonable(trace.history),
        "finite": bool(torch.isfinite(output["actions"]).all().item()),
    }


def compare_cem(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    ref = tensor(reference["first_actions"], dtype=torch.float64)
    got = tensor(candidate["first_actions"], dtype=torch.float64)
    diff = got - ref
    norm = float(torch.linalg.vector_norm(diff).item() / max(float(torch.linalg.vector_norm(ref).item()), 1e-8))
    return {
        "first_action_normalized_l2": norm,
        "first_action_absolute_max": float(diff.abs().max().item()),
        "finite": bool(reference["finite"] and candidate["finite"]),
    }


def run_cem_diagnostic(
    model: Any,
    model_basis: PCABasis,
    selected_ranks: Sequence[int],
    stablewm_home: Path,
    freeze: Mapping[str, Any],
    helpers: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    cem_cfg = freeze["step2"]["cem"]
    observations, action_space = helpers["prepare_policy_info"](stablewm_home, int(cem_cfg["observations"]))
    from stable_worldmodel.policy import PlanConfig

    plan_config = PlanConfig(**cem_cfg["plan_config"])
    settings = dict(cem_cfg["solver"])
    seeds = [int(item) for item in cem_cfg["seeds"]]
    arms: list[tuple[str, Any]] = [("baseline", model), ("rank_192_full_control", OracleProjectedModel(model, model_basis, 192))]
    arms.extend((f"model_basis_rank_{rank}", OracleProjectedModel(model, model_basis, rank)) for rank in selected_ranks)
    records = []
    for observation_id, info in observations:
        for seed in seeds:
            results = {name: run_cem_call(arm, info, settings, seed, action_space, plan_config, helpers) for name, arm in arms}
            reference = results["baseline"]
            comparisons = {name: compare_cem(reference, result) for name, result in results.items() if name != "baseline"}
            records.append({"observation_id": observation_id, "seed": seed, "results": results, "comparisons": comparisons})
    gate = cem_cfg["maximum_first_action_normalized_l2"], cem_cfg["maximum_first_action_absolute_difference"]
    return {
        "observations": [item[0] for item in observations],
        "seeds": seeds,
        "arms": [name for name, _ in arms],
        "records": records,
        "approximate_ranks_skipped": not bool(selected_ranks),
        "planner_gate": {"maximum_first_action_normalized_l2": gate[0], "maximum_first_action_absolute_difference": gate[1]},
        "claim_boundary": "fixed-observation official CEM oracle diagnostic only; no speed claim and no closed-loop environment",
    }


def main() -> int:
    import torch

    args = parse_args()
    require_compute_node()
    freeze = load_json(args.freeze)
    if freeze.get("schema") != "lewm.delta-oracle.freeze":
        raise ValueError("unexpected delta-oracle freeze schema")
    rows, manifest = load_rows(args.prepared_rows, args.manifest)
    check_row_shapes(rows)
    train_rows = [row for row in rows if row.get("split") == "train"]
    heldout_rows = [row for row in rows if row.get("split") == "heldout"]
    args.output.mkdir(parents=True, exist_ok=True)

    reader_cls, normalise_pixels = load_lewm_helpers()
    iteration_root = args.freeze.parent.parent / "01-iteration-cache-cross-model-planners" / "cases" / "lewm-pusht-cem"
    helpers = load_iteration_helpers(iteration_root)
    model = load_model(args.stablewm_home.resolve())

    observed_by_context = collect_observed_deltas(model, args.dataset.resolve(), manifest, normalise_pixels, reader_cls)
    observed_train = torch_stack([observed_by_context[str(row["context_id"])] for row in train_rows])
    observed_heldout = torch_stack([observed_by_context[str(row["context_id"])] for row in heldout_rows])
    model_train = torch_stack([teacher_deltas(row) for row in train_rows])
    model_heldout = torch_stack([teacher_deltas(row) for row in heldout_rows])
    frontier = pca_frontier(observed_train, model_train, observed_heldout, model_heldout, RANKS)
    observed_basis = frontier["observed_basis"]
    model_basis = frontier["model_basis"]

    trajectory_oracle = trajectory_svd_oracle(heldout_rows)
    heldout_candidate_metrics: dict[str, Any] = {}
    candidate_summaries: dict[str, Any] = {}
    model_basis_candidates = []
    observed_basis_candidates = []
    trajectory_svd_candidates = []
    for row in heldout_rows:
        deltas = teacher_deltas(row)
        for rank in RANKS:
            item = candidate_metrics(model, row, deltas, model_basis, rank, "model")
            model_basis_candidates.extend(item["per_block"])
            item_observed = candidate_metrics(model, row, deltas, observed_basis, rank, "observed")
            observed_basis_candidates.extend(item_observed["per_block"])
        actions = tensor(row["future_actions"], dtype=torch.float32)
        targets = tensor(row["teacher_targets"], dtype=torch.float32)
        teacher_objectives = tensor(row["teacher_objective"], dtype=torch.float32)
        initial = tensor(row["latent_history"], dtype=torch.float64)[0, 0]
        trajectory_futures = trajectory_rank_reconstructions(initial, deltas, range(1, HORIZON + 1))
        for rank in range(1, HORIZON + 1):
            oracle_future = trajectory_futures[rank].float()
            trajectory_svd_candidates.extend(
                candidate_metrics_from_future(
                    model,
                    row,
                    actions,
                    targets,
                    teacher_objectives,
                    oracle_future,
                    "trajectory_svd",
                    rank,
                )["per_block"]
            )
    random_basis = PCABasis(
        mean=model_basis.mean,
        components=random_orthonormal_basis(64, RANDOM_BASIS_SEED),
        singular_values=model_basis.singular_values,
        total_centered_energy=model_basis.total_centered_energy,
        name="random_rank64",
    )
    random_candidates = []
    for row in heldout_rows:
        random_candidates.extend(candidate_metrics(model, row, teacher_deltas(row), random_basis, 64, "random_rank64")["per_block"])
    all_candidate_records = model_basis_candidates + observed_basis_candidates + trajectory_svd_candidates + random_candidates
    for item in all_candidate_records:
        key = f"{item['basis']}_rank_{item['rank']}"
        heldout_candidate_metrics.setdefault(key, []).append(item)
    for key, records in heldout_candidate_metrics.items():
        summary = summarise_candidate(records)
        summary["candidate_gate"] = candidate_gate(summary, freeze) if summary["basis"] != "random_rank64" else False
        candidate_summaries[key] = summary

    model_passing = [rank for rank in RANKS if rank != 192 and candidate_summaries.get(f"model_rank_{rank}", {}).get("candidate_gate")]
    selected_ranks = model_passing[:2]
    full_control = candidate_summaries["model_rank_192"]
    full_rank_exact = bool(
        full_control["latent_relative_mse_median"] <= FULL_RANK_TOLERANCE
        and full_control["spearman_minimum"] >= 1.0 - FULL_RANK_TOLERANCE
        and full_control["top30_overlap_minimum"] >= 1.0
    )
    # Always run baseline/full-rank controls.  Approximate-rank CEM is added
    # only after the full-rank projection passes the control check.
    cem = run_cem_diagnostic(
        model,
        model_basis,
        selected_ranks if full_rank_exact else (),
        args.stablewm_home.resolve(),
        freeze,
        helpers,
    )

    summary = {
        "schema": "lewm.delta-oracle.summary",
        "schema_version": 1,
        "status": "COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "source": {
            "prepared_rows": str(args.prepared_rows.resolve()),
            "context_manifest": str(args.manifest.resolve()),
            "dataset": str(args.dataset.resolve()),
            "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()),
            "lewm_commit": freeze["source"]["lewm_commit"],
            "stable_worldmodel_cem_commit": freeze["source"]["stable_worldmodel_cem_commit"],
        },
        "representation": freeze["representation_contract"],
        "split": freeze["split"],
        "step1": {
            "observed_offsets": list(OBSERVED_OFFSETS),
            "basis_fit": "centered PCA/SVD on train deltas only",
            "frontier": frontier["result"],
            "structure_gate": {
                "frozen": freeze["step1"]["structure_gate"],
                "evaluation": structure_gate_report(frontier["result"]["heldout"], freeze),
            },
        },
        "trajectory_svd_oracle": trajectory_oracle,
        "step2": {
            "heldout_blocks": 16,
            "candidate_metrics": candidate_summaries,
            "candidate_gate": freeze["step2"]["candidate_gate"],
            "random_control": {"basis": "deterministic random orthonormal rank-64", "seed": RANDOM_BASIS_SEED, "summary": candidate_summaries["random_rank64_rank_64"]},
            "selected_cem_ranks": selected_ranks,
            "full_rank_control": {"summary": full_control, "approximately_exact": full_rank_exact, "tolerance": FULL_RANK_TOLERANCE},
            "cem": cem,
            "interpretation_stop": not full_rank_exact,
        },
        "claim_boundary": "Representation and fixed-observation planner oracle diagnostic only. No cheap predictor, inference acceleration, or closed-loop PushT success is claimed.",
    }
    output_path = args.output.resolve() / "summary.json"
    output_path.write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(output_path), "full_rank_exact": full_rank_exact, "selected_cem_ranks": selected_ranks}))
    return 0


def torch_stack(values: Sequence[Any]) -> Any:
    import torch

    if not values:
        raise ValueError("cannot stack an empty sequence")
    return torch.stack([tensor(value, dtype=torch.float32) for value in values])


if __name__ == "__main__":
    raise SystemExit(main())

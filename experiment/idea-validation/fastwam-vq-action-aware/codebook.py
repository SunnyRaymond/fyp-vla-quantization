"""Numerical engine for row-scaled, two-book Fast-WAM weight VQ.

The encoded payload is two shared BF16 books, one BF16 scale per output row,
and two uint8 indices per padded four-weight block. Local assignment metrics
are block-diagonal quadratic approximations; a supplied loss closure is needed
for objectives with coupling across rows, blocks, tokens, or action dimensions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


FORMAT = "fastwam_additive_vq_row_scale_v1"
BLOCK_SIZE = 4
BOOKS = 2
BOOK_SIZE = 256
DEFAULT_SAMPLE_LIMIT = 8192
DEFAULT_ITERATIONS = 8
MAX_TUNING_STEPS = 100


def _check_weight(weight: Tensor) -> tuple[int, int]:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating [out_features, in_features] tensor")
    out_features, in_features = weight.shape
    if out_features < 1 or in_features < 1:
        raise ValueError("weight dimensions must be non-empty")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must contain only finite values")
    return out_features, in_features


class _MetricView:
    """Compact per-block metric view; shared input-block metrics stay shared."""

    def __init__(
        self,
        kind: str,
        values: Tensor,
        out_features: int,
        groups: int,
        row_scales: Tensor,
    ) -> None:
        self.kind = kind
        self.values = values
        self.out_features = out_features
        self.groups = groups
        self.row_scales = row_scales.float()
        self.shared_rows = values.shape[0] == 1

    def _select(self, ids: Tensor) -> tuple[str, Tensor]:
        rows = torch.div(ids, self.groups, rounding_mode="floor")
        groups = ids.remainder(self.groups)
        if self.shared_rows:
            selected = self.values[0, groups]
        else:
            selected = self.values[rows, groups]
        row_weight = self.row_scales[rows]

        if self.kind == "diag":
            return "diag", selected * row_weight.square()[:, None]
        if self.kind == "matrix":
            return "matrix", selected * row_weight.square()[:, None, None]
        # F is an action-projection factor and defines M = F^T F.
        factor = selected * row_weight[:, None, None]
        return "matrix", factor.transpose(-1, -2) @ factor

    def gather(self, ids: Tensor) -> tuple[str, Tensor]:
        return self._select(ids.to(device=self.values.device, dtype=torch.long))

    def chunk(self, start: int, end: int) -> tuple[str, Tensor]:
        ids = torch.arange(start, end, device=self.values.device)
        return self._select(ids)


def _metric_view(
    weight: Tensor,
    row_scales: Tensor,
    groups: int,
    second_moment: Tensor | None,
    block_metric: Tensor | None,
    projection_factor: Tensor | None,
) -> _MetricView:
    supplied = sum(value is not None for value in (second_moment, block_metric, projection_factor))
    if supplied > 1:
        raise ValueError("provide only one of second_moment, block_metric, projection_factor")

    out_features, in_features = weight.shape
    padded_width = groups * BLOCK_SIZE
    device = weight.device

    if second_moment is None and block_metric is None and projection_factor is None:
        values = torch.ones((1, groups, BLOCK_SIZE), dtype=torch.float32, device=device)
        values.reshape(-1)[in_features:] = 0
        kind = "diag"
    elif second_moment is not None:
        moment = second_moment.detach().to(device=device, dtype=torch.float32)
        if moment.shape == (in_features,):
            moment = moment.unsqueeze(0)
        elif moment.shape != (out_features, in_features):
            raise ValueError("second_moment must have shape [in_features] or [out_features, in_features]")
        if not bool(torch.isfinite(moment).all()) or bool((moment < 0).any()):
            raise ValueError("second_moment must be finite and non-negative")
        if moment.shape[0] == 1:
            moment = moment.clamp_min(1e-12)
        else:
            moment = moment.clamp_min(1e-12)
        if padded_width > in_features:
            moment = torch.nn.functional.pad(moment, (0, padded_width - in_features))
        values = moment.reshape(moment.shape[0], groups, BLOCK_SIZE)
        kind = "diag"
    elif block_metric is not None:
        metric = block_metric.detach().to(device=device, dtype=torch.float32)
        if metric.shape == (groups, BLOCK_SIZE, BLOCK_SIZE):
            metric = metric.unsqueeze(0)
        elif metric.shape != (out_features, groups, BLOCK_SIZE, BLOCK_SIZE):
            raise ValueError("block_metric must have shape [groups, 4, 4] or [out_features, groups, 4, 4]")
        if not bool(torch.isfinite(metric).all()):
            raise ValueError("block_metric must contain only finite values")
        asymmetry = (metric - metric.transpose(-1, -2)).abs().amax()
        magnitude = metric.abs().amax().clamp_min(1.0)
        if bool(asymmetry > 1e-5 * magnitude):
            raise ValueError("block_metric must be symmetric")
        metric = (metric + metric.transpose(-1, -2)) * 0.5
        flat = metric.reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
        invalid_psd = torch.zeros((), dtype=torch.bool, device=device)
        for start in range(0, flat.shape[0], 16384):
            chunk = flat[start : start + 16384]
            eigenvalues = torch.linalg.eigvalsh(chunk)
            tolerance = chunk.abs().amax(dim=(-2, -1)).clamp_min(1e-12) * 1e-6
            invalid_psd |= (eigenvalues[:, 0] < -tolerance).any()
        if bool(invalid_psd):
            raise ValueError("block_metric must be positive semidefinite")
        valid = torch.arange(groups * BLOCK_SIZE, device=device) < in_features
        mask = valid.reshape(groups, BLOCK_SIZE).float()
        metric = metric * mask[None, :, :, None] * mask[None, :, None, :]
        values = metric
        kind = "matrix"
    else:
        factor = projection_factor.detach().to(device=device, dtype=torch.float32)
        if factor.ndim == 3 and factor.shape[0] == groups and factor.shape[-1] == BLOCK_SIZE:
            factor = factor.unsqueeze(0)
        elif factor.ndim != 4 or factor.shape[0] != out_features or factor.shape[1] != groups or factor.shape[-1] != BLOCK_SIZE:
            raise ValueError("projection_factor must have shape [groups, rank, 4] or [out_features, groups, rank, 4]")
        if factor.shape[-2] < 1 or not bool(torch.isfinite(factor).all()):
            raise ValueError("projection_factor must have a non-empty rank and finite values")
        valid = torch.arange(groups * BLOCK_SIZE, device=device) < in_features
        factor = factor * valid.reshape(groups, BLOCK_SIZE)[None, :, None, :]
        values = factor
        kind = "factor"

    view = _MetricView(kind, values, out_features, groups, row_scales)
    trace_sum = torch.zeros((), dtype=torch.float32, device=device)
    for start in range(0, out_features * groups, 16384):
        metric_kind, chunk = view.chunk(start, min(start + 16384, out_features * groups))
        if metric_kind == "diag":
            trace_sum += chunk.sum()
        else:
            trace_sum += chunk.diagonal(dim1=-2, dim2=-1).sum()
    normalizer = float((trace_sum / max(out_features * in_features, 1)).item())
    if normalizer > 1e-12 and math.isfinite(normalizer):
        view.values = view.values / normalizer
    return view


def _metric_chunk(metric: _MetricView | tuple[str, Tensor] | Tensor, start: int, end: int) -> tuple[str, Tensor]:
    if isinstance(metric, _MetricView):
        return metric.chunk(start, end)
    if isinstance(metric, tuple):
        kind, values = metric
        return kind, values[start:end]
    if metric.ndim == 2:
        return "diag", metric[start:end]
    if metric.ndim == 3:
        return "matrix", metric[start:end]
    raise ValueError("metric must be a diagonal [N, D] or PSD [N, D, D] tensor")


def _assign(vectors: Tensor, codebook: Tensor, metric: _MetricView | tuple[str, Tensor] | Tensor) -> Tensor:
    assignments = torch.empty(vectors.shape[0], dtype=torch.long, device=vectors.device)
    centers = codebook.float()
    for start in range(0, vectors.shape[0], 4096):
        end = min(start + 4096, vectors.shape[0])
        values = vectors[start:end].float()
        kind, metric_values = _metric_chunk(metric, start, end)
        delta = values[:, None, :] - centers[None, :, :]
        if kind == "diag":
            distance = (delta.square() * metric_values[:, None, :]).sum(dim=-1)
        else:
            distance = torch.einsum("nkd,ndh,nkh->nk", delta, metric_values, delta)
        assignments[start:end] = distance.clamp_min(0).argmin(dim=1)
    return assignments


def _update_centroids(
    vectors: Tensor,
    assignments: Tensor,
    metric: _MetricView | tuple[str, Tensor] | Tensor,
    previous: Tensor,
) -> Tensor:
    centers = previous.shape[0]
    dim = vectors.shape[1]
    matrix_metric = None
    if isinstance(metric, _MetricView):
        matrix_metric = metric.kind != "diag"
    elif isinstance(metric, tuple):
        matrix_metric = metric[0] != "diag"
    else:
        matrix_metric = metric.ndim == 3

    count = torch.zeros(centers, dtype=torch.float32, device=vectors.device)
    rhs = torch.zeros((centers, dim), dtype=torch.float32, device=vectors.device)
    if matrix_metric:
        systems = torch.zeros((centers, dim, dim), dtype=torch.float32, device=vectors.device)
    else:
        systems = torch.zeros((centers, dim), dtype=torch.float32, device=vectors.device)

    for start in range(0, vectors.shape[0], 4096):
        end = min(start + 4096, vectors.shape[0])
        x = vectors[start:end].float()
        a = assignments[start:end]
        kind, m = _metric_chunk(metric, start, end)
        count.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
        if kind == "diag":
            systems.index_add_(0, a, m)
            rhs.index_add_(0, a, x * m)
        else:
            systems.index_add_(0, a, m)
            rhs.index_add_(0, a, torch.bmm(m, x.unsqueeze(-1)).squeeze(-1))

    if not matrix_metric:
        updated = rhs / systems.clamp_min(torch.finfo(torch.float32).tiny)
        updated = torch.where(systems > 0, updated, previous.float())
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(systems)
        tolerance = eigenvalues.abs().amax(dim=-1, keepdim=True).clamp_min(1.0) * 1e-7
        inverse = torch.where(
            eigenvalues > tolerance,
            eigenvalues.clamp_min(torch.finfo(torch.float32).tiny).reciprocal(),
            0.0,
        )
        projected = torch.matmul(eigenvectors.transpose(-1, -2), rhs.unsqueeze(-1)).squeeze(-1)
        updated = torch.matmul(eigenvectors, (inverse * projected).unsqueeze(-1)).squeeze(-1)
        updated = torch.where((count > 0)[:, None], updated, previous.float())
    return updated


def _kmeans(vectors: Tensor, metric: _MetricView | tuple[str, Tensor] | Tensor, generator: torch.Generator) -> Tensor:
    if vectors.shape[0] == 0:
        return torch.zeros((BOOK_SIZE, BLOCK_SIZE), dtype=torch.float32, device=vectors.device)
    chosen = torch.randperm(vectors.shape[0], generator=generator)[: min(vectors.shape[0], BOOK_SIZE)]
    if chosen.numel() < BOOK_SIZE:
        extra = torch.randint(vectors.shape[0], (BOOK_SIZE - chosen.numel(),), generator=generator)
        chosen = torch.cat((chosen, extra))
    chosen = chosen.to(vectors.device)
    centers = vectors[chosen].float().clone()
    for _ in range(5):
        assignment = _assign(vectors, centers, metric)
        centers = _update_centroids(vectors, assignment, metric, centers)
    return centers


def fit_vq(
    weight: Tensor,
    *,
    second_moment: Tensor | None = None,
    block_metric: Tensor | None = None,
    projection_factor: Tensor | None = None,
    seed: int = 20261009,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    iterations: int = DEFAULT_ITERATIONS,
) -> dict[str, Any]:
    """Fit fixed 2x256 additive VQ after BF16 per-output-row RMS scaling.

    ``second_moment`` is the old diagonal assignment weight, shaped ``[in]``
    or ``[out, in]``. ``block_metric`` supplies PSD metrics per input block,
    shared across rows as ``[groups, 4, 4]`` or row-specific as
    ``[out, groups, 4, 4]``. ``projection_factor`` supplies F with M=F^T F,
    shaped ``[groups, rank, 4]`` or ``[out, groups, rank, 4]``. Metrics affect
    only the independent block assignments and centroid updates; they are not
    a substitute for a coupled final-action loss.
    """
    out_features, in_features = _check_weight(weight)
    if sample_limit < 1 or iterations < 2:
        raise ValueError("sample_limit must be positive and iterations at least two")

    groups = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_width = groups * BLOCK_SIZE
    values = weight.detach().float()
    row_scale = values.square().mean(dim=1).sqrt().to(torch.bfloat16)
    scale_float = row_scale.float()
    safe_scale = torch.where(scale_float > 0, scale_float, torch.ones_like(scale_float))
    normalized = values / safe_scale[:, None]
    if padded_width > in_features:
        normalized = torch.nn.functional.pad(normalized, (0, padded_width - in_features))
    vectors = normalized.reshape(out_features, groups, BLOCK_SIZE).reshape(-1, BLOCK_SIZE).contiguous()
    metric = _metric_view(weight, row_scale, groups, second_moment, block_metric, projection_factor)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_count = min(vectors.shape[0], sample_limit)
    if vectors.shape[0] <= sample_limit:
        sample_ids = torch.arange(vectors.shape[0], device=weight.device)
    else:
        sample_ids = torch.randint(vectors.shape[0], (sample_count,), generator=generator).to(weight.device)
    sample = vectors[sample_ids]
    sample_metric = metric.gather(sample_ids)

    # Residual initialization, then alternating weighted codebook updates.
    book0 = _kmeans(sample, sample_metric, generator).to(torch.bfloat16).float()
    index0 = _assign(sample, book0, sample_metric)
    residual = sample - book0[index0]
    book1 = _kmeans(residual, sample_metric, generator).to(torch.bfloat16).float()
    index1 = _assign(residual, book1, sample_metric)
    for _ in range(iterations):
        target0 = sample - book1[index1]
        index0 = _assign(target0, book0, sample_metric)
        book0 = _update_centroids(target0, index0, sample_metric, book0).to(torch.bfloat16).float()
        residual = sample - book0[index0]
        index1 = _assign(residual, book1, sample_metric)
        book1 = _update_centroids(residual, index1, sample_metric, book1).to(torch.bfloat16).float()

    books = torch.stack((book0.to(torch.bfloat16), book1.to(torch.bfloat16)), dim=0)
    stored_books = books.float()
    all_index0 = _assign(vectors, stored_books[0], metric)
    all_index1 = _assign(vectors - stored_books[0][all_index0], stored_books[1], metric)
    for _ in range(2):
        all_index0 = _assign(vectors - stored_books[1][all_index1], stored_books[0], metric)
        all_index1 = _assign(vectors - stored_books[0][all_index0], stored_books[1], metric)
    indices = torch.stack((all_index0, all_index1), dim=-1).reshape(out_features, groups, BOOKS).to(torch.uint8)
    return {
        "format": FORMAT,
        "shape": (out_features, in_features),
        "block_size": BLOCK_SIZE,
        "books": books,
        "row_scales": row_scale,
        "indices": indices,
        "fit_stats": {
            "vectors": int(vectors.shape[0]),
            "sample_vectors": int(sample_count),
            "iterations": int(iterations),
            "assignment_metric": "block_metric" if block_metric is not None else "projection_factor" if projection_factor is not None else "second_moment" if second_moment is not None else "identity",
        },
    }


def _assignment_error_sum(vectors: Tensor, reconstruction: Tensor, metric: _MetricView) -> float:
    total = torch.zeros((), dtype=torch.float32, device=vectors.device)
    for start in range(0, vectors.shape[0], 4096):
        end = min(start + 4096, vectors.shape[0])
        delta = vectors[start:end] - reconstruction[start:end]
        kind, values = metric.chunk(start, end)
        if kind == "diag":
            error = (delta.square() * values).sum(dim=-1)
        else:
            error = torch.einsum("nd,ndh,nh->n", delta, values, delta)
        total += error.clamp_min(0).sum()
    return float(total.item())


def reassign_indices(
    encoding: dict[str, Any],
    weight_target: Tensor,
    *,
    second_moment: Tensor | None = None,
    block_metric: Tensor | None = None,
    projection_factor: Tensor | None = None,
    iterations: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reassign additive indices for a weight target with stored books/scales fixed.

    The target is normalized by the encoding's stored BF16 row scales. Assignment
    alternates book 0 and book 1 against that target. Returned objectives are
    block-metric weighted local weight errors, globally normalized by mean metric
    trace; they do not measure final action error.
    """
    out_features, in_features, groups = _check_encoding(encoding)
    target_out, target_in = _check_weight(weight_target)
    if (target_out, target_in) != (out_features, in_features):
        raise ValueError("weight_target shape must match encoded weight shape")
    if weight_target.device != encoding["books"].device:
        raise ValueError("weight_target and encoding tensors must be on the same device")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    scales = encoding["row_scales"].float()
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    target = weight_target.detach().float() / safe_scales[:, None]
    padded_width = groups * BLOCK_SIZE
    if padded_width > in_features:
        target = torch.nn.functional.pad(target, (0, padded_width - in_features))
    vectors = target.reshape(out_features, groups, BLOCK_SIZE).reshape(-1, BLOCK_SIZE).contiguous()
    metric = _metric_view(weight_target, encoding["row_scales"], groups, second_moment, block_metric, projection_factor)

    books = encoding["books"].float()
    indices = encoding["indices"].reshape(-1, BOOKS).long()
    index0, index1 = indices[:, 0], indices[:, 1]
    current_reconstruction = books[0][index0] + books[1][index1]
    objective_before = _assignment_error_sum(vectors, current_reconstruction, metric)
    for _ in range(iterations):
        index0 = _assign(vectors - books[1][index1], books[0], metric)
        index1 = _assign(vectors - books[0][index0], books[1], metric)
    reconstruction = books[0][index0] + books[1][index1]
    objective_after = _assignment_error_sum(vectors, reconstruction, metric)

    updated = dict(encoding)
    updated["indices"] = torch.stack((index0, index1), dim=-1).reshape(out_features, groups, BOOKS).to(torch.uint8)
    return updated, {
        "objective_before": objective_before,
        "objective_after": objective_after,
        "iterations": int(iterations),
        "assignment_metric": "block_metric" if block_metric is not None else "projection_factor" if projection_factor is not None else "second_moment" if second_moment is not None else "identity",
    }


def _check_encoding(encoding: dict[str, Any]) -> tuple[int, int, int]:
    if encoding.get("format") != FORMAT or encoding.get("block_size") != BLOCK_SIZE:
        raise ValueError("unsupported VQ encoding format")
    shape = encoding.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("encoding shape must be [out_features, in_features]")
    out_features, in_features = int(shape[0]), int(shape[1])
    if out_features < 1 or in_features < 1:
        raise ValueError("encoding dimensions must be non-empty")
    groups = (in_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    books, scales, indices = encoding.get("books"), encoding.get("row_scales"), encoding.get("indices")
    if not isinstance(books, Tensor) or books.shape != (BOOKS, BOOK_SIZE, BLOCK_SIZE) or books.dtype != torch.bfloat16:
        raise ValueError("books must be BF16 [2, 256, 4]")
    if not isinstance(scales, Tensor) or scales.shape != (out_features,) or scales.dtype != torch.bfloat16:
        raise ValueError("row_scales must be BF16 [out_features]")
    if not isinstance(indices, Tensor) or indices.shape != (out_features, groups, BOOKS) or indices.dtype != torch.uint8:
        raise ValueError("indices must be uint8 [out_features, ceil(in_features/4), 2]")
    if books.device != scales.device or books.device != indices.device:
        raise ValueError("books, row_scales, and indices must be on the same device")
    return out_features, in_features, groups


def decode_vq(encoding: dict[str, Any], *, output_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Decode the stored BF16 books/scales and uint8 indices, then trim tail padding."""
    out_features, in_features, _ = _check_encoding(encoding)
    books = encoding["books"].float()
    indices = encoding["indices"].long()
    vectors = books[0][indices[..., 0]] + books[1][indices[..., 1]]
    dense = vectors.reshape(out_features, -1)[:, :in_features] * encoding["row_scales"].float()[:, None]
    return dense.to(output_dtype)


def encoded_nbytes(encoding: dict[str, Any]) -> int:
    """Count tensor payload bytes, including padded-tail indices and row scales."""
    _check_encoding(encoding)
    return sum(encoding[key].numel() * encoding[key].element_size() for key in ("books", "row_scales", "indices"))


def weight_reconstruction_loss(weight_hat: Tensor, weight: Tensor, *, reduction: str = "mean") -> Tensor:
    """Squared weight error, useful for the weight-only initialization arm."""
    if weight_hat.shape != weight.shape:
        raise ValueError("weight_hat and weight must have the same shape")
    error = (weight_hat.float() - weight.float()).square()
    if reduction == "sum":
        return error.sum()
    if reduction == "mean":
        return error.mean()
    raise ValueError("reduction must be 'sum' or 'mean'")


def local_output_loss(
    weight_hat: Tensor,
    inputs_q: Tensor,
    teacher_outputs: Tensor,
    *,
    action_projection_factors: Tensor | None = None,
    reduction: str = "sum",
) -> Tensor:
    """Fit actual quantized inputs QZ to teacher rows ZB^T.

    ``inputs_q`` is QZ and ``teacher_outputs`` is the already computed ZB^T
    (or teacher output rows). Optional factors shaped ``[probe, tokenrow, out]``
    or ``[probe, tokenrow*out]`` define squared projected errors across tokens
    and output channels, preserving their coupling. This remains a local linear
    layer objective; it is not an end-to-end action objective by itself.
    """
    if weight_hat.ndim != 2 or inputs_q.ndim != 2 or teacher_outputs.ndim != 2:
        raise ValueError("weight_hat, inputs_q, and teacher_outputs must be matrices")
    out_features, in_features = weight_hat.shape
    if inputs_q.shape[1] != in_features or teacher_outputs.shape != (inputs_q.shape[0], out_features):
        raise ValueError("inputs_q and teacher_outputs do not match weight_hat")
    error = inputs_q.float() @ weight_hat.float().T - teacher_outputs.float()
    if action_projection_factors is not None:
        factors = action_projection_factors.to(device=error.device, dtype=torch.float32)
        if factors.ndim == 3 and factors.shape[1:] == error.shape:
            projected = (factors * error.unsqueeze(0)).sum(dim=(1, 2))
        elif factors.ndim == 2 and factors.shape[1] == error.numel():
            projected = factors @ error.reshape(-1)
        else:
            raise ValueError("action_projection_factors must be [probe, tokenrow, out] or [probe, tokenrow*out]")
        losses = projected.square()
    else:
        losses = error.square().reshape(-1)
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'sum' or 'mean'")


def _ste_bf16(values: Tensor) -> Tensor:
    rounded = values.to(torch.bfloat16).float()
    return values + (rounded - values).detach()


def _decode_ste(encoding: dict[str, Any], books: Tensor, row_scales: Tensor) -> Tensor:
    out_features, in_features, _ = _check_encoding(encoding)
    max_bf16 = torch.finfo(torch.bfloat16).max
    stored_books = _ste_bf16(books.clamp(-max_bf16, max_bf16))
    stored_scales = _ste_bf16(row_scales.clamp(0, max_bf16))
    indices = encoding["indices"].long()
    vectors = stored_books[0][indices[..., 0]] + stored_books[1][indices[..., 1]]
    dense = vectors.reshape(out_features, -1)[:, :in_features] * stored_scales[:, None]
    return _ste_bf16(dense)


def tune_fixed_indices(
    encoding: dict[str, Any],
    loss_fn: Callable[[Tensor], Tensor],
    *,
    max_steps: int = 32,
    patience: int = 8,
    learning_rate: float = 0.03,
    tune_scales: bool = True,
    min_delta: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune stored books (and optionally row scales) with fixed uint8 indices.

    The closure receives an FP32 tensor whose forward values equal the BF16
    rounded dense decode. Straight-through gradients pass through those BF16
    casts. The best evaluated encoded state is returned, with early stopping
    and a hard 100-update ceiling. This function never modifies model weights.
    """
    _check_encoding(encoding)
    if not 1 <= max_steps <= MAX_TUNING_STEPS:
        raise ValueError("max_steps must be between 1 and 100")
    if not 1 <= patience <= max_steps:
        raise ValueError("patience must be between 1 and max_steps")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative")

    books = torch.nn.Parameter(encoding["books"].float().detach().clone())
    stored_scales = encoding["row_scales"].float().detach()
    active_scales = (stored_scales > 0).float()
    log_scales = torch.nn.Parameter(stored_scales.clamp_min(torch.finfo(torch.float32).tiny).log().clone())
    params = [books, log_scales] if tune_scales else [books]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    loss_history: list[float] = []
    best_loss = math.inf
    best_step = 0
    best_books: Tensor | None = None
    best_scales: Tensor | None = None
    stale_steps = 0
    max_scale_log = math.log(torch.finfo(torch.bfloat16).max)

    for step in range(max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        candidate_scales = stored_scales if not tune_scales else log_scales.clamp(max=max_scale_log).exp() * active_scales
        decoded = _decode_ste(encoding, books, candidate_scales)
        loss = loss_fn(decoded)
        if not isinstance(loss, Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
            raise ValueError("loss_fn must return one finite scalar tensor")
        current = float(loss.detach().item())
        loss_history.append(current)
        if current < best_loss - min_delta:
            best_loss = current
            best_step = step
            best_books = books.detach().clamp(-torch.finfo(torch.bfloat16).max, torch.finfo(torch.bfloat16).max).to(torch.bfloat16).clone()
            best_scales = candidate_scales.detach().clamp(0, torch.finfo(torch.bfloat16).max).to(torch.bfloat16).clone()
            stale_steps = 0
        else:
            stale_steps += 1
        if step == max_steps or stale_steps >= patience:
            break
        if not loss.requires_grad:
            raise ValueError("loss_fn must be differentiable with respect to decoded weight")
        loss.backward()
        optimizer.step()

    assert best_books is not None and best_scales is not None
    best_encoding = dict(encoding)
    best_encoding["books"] = best_books
    best_encoding["row_scales"] = best_scales
    return best_encoding, {
        "loss_history": loss_history,
        "best_loss": best_loss,
        "best_step": best_step,
        "optimizer_steps": max(0, len(loss_history) - 1),
        "tuned_scales": bool(tune_scales),
    }

"""Small PyTorch core for Fast-WAM Smooth/Hadamard plus scalar/VQ weight quantization."""

from __future__ import annotations

import math
from typing import Any

import torch


def _segments(dim: int, block: int):
    if block < 1 or block & (block - 1):
        raise ValueError("block must be a positive power of two")
    start = 0
    while dim - start >= block:
        yield start, block
        start += block
    tail = dim - start
    while tail:
        width = 1 << (tail.bit_length() - 1)
        yield start, width
        start += width
        tail -= width


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over the last power-of-two dimension."""
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("Hadamard segment must have power-of-two width")
    y = x.float()
    h = 1
    while h < n:
        groups = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = groups.unbind(dim=-2)
        y = torch.stack((a + b, a - b), dim=-2).reshape_as(y)
        h *= 2
    return y * (1.0 / math.sqrt(n))


def _right_r(x: torch.Tensor, signs: torch.Tensor, hadamard: bool, block: int) -> torch.Tensor:
    pieces = []
    for start, width in _segments(x.shape[-1], block):
        part = x[..., start : start + width].float() * signs[start : start + width]
        if hadamard:
            part = _fwht(part)
        pieces.append(part)
    return torch.cat(pieces, dim=-1) if pieces else x.float()


def _right_r_inverse(x: torch.Tensor, signs: torch.Tensor, hadamard: bool, block: int) -> torch.Tensor:
    pieces = []
    for start, width in _segments(x.shape[-1], block):
        part = x[..., start : start + width].float()
        if hadamard:
            part = _fwht(part)
        part = part * signs[start : start + width]
        pieces.append(part)
    return torch.cat(pieces, dim=-1) if pieces else x.float()


def make_transform(
    act_max: torch.Tensor | None,
    weight: torch.Tensor,
    alpha: float | None = None,
    hadamard: bool = True,
    block: int = 128,
    seed: int = 20261006,
) -> dict[str, Any]:
    """Build S and signed block-Hadamard R; alpha=None gives S=I."""
    if weight.ndim != 2:
        raise ValueError("weight must be [out_features, in_features]")
    if block < 1 or block & (block - 1):
        raise ValueError("block must be a positive power of two")
    device = weight.device
    width = weight.shape[1]
    scale = torch.ones(width, dtype=torch.float32, device=device)
    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1] or None")
        if act_max is None or act_max.numel() != width:
            raise ValueError("act_max must have one value per input feature")
        tiny = torch.finfo(torch.float32).tiny
        amax = act_max.detach().to(device=device, dtype=torch.float32).reshape(-1).clamp_min(tiny)
        wmax = weight.detach().float().abs().amax(dim=0).clamp_min(tiny)
        log_scale = alpha * amax.log() - (1.0 - alpha) * wmax.log()
        # Normalize the geometric mean before applying the recipe's scale bounds.
        scale = (log_scale - log_scale.mean()).exp().clamp(1.0 / 16.0, 16.0)
    if hadamard:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bits = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int64)
        signs = (bits.mul(2).sub(1)).to(device=device, dtype=torch.float32)
    else:
        signs = torch.ones(width, dtype=torch.float32, device=device)
    return {"scale": scale, "signs": signs, "hadamard": bool(hadamard), "block": int(block), "input_dim": width}


def transform_input(x: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
    if x.shape[-1] != transform["input_dim"]:
        raise ValueError("input width does not match transform")
    x32 = x.float() / transform["scale"]
    return _right_r(x32, transform["signs"], transform["hadamard"], transform["block"])


def transform_weight(weight: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
    if weight.shape[-1] != transform["input_dim"]:
        raise ValueError("weight width does not match transform")
    scaled = weight.float() * transform["scale"]
    return _right_r(scaled, transform["signs"], transform["hadamard"], transform["block"])


def inverse_transform_input(z: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
    if z.shape[-1] != transform["input_dim"]:
        raise ValueError("input width does not match transform")
    y = _right_r_inverse(z.float(), transform["signs"], transform["hadamard"], transform["block"])
    return y * transform["scale"]


def quant_activation(z: torch.Tensor, bits: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-row absmax signed quantization using qmax 7 or 127."""
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    qmax = 7 if bits == 4 else 127
    z32 = z.float()
    row_max = z32.abs().amax(dim=-1, keepdim=True)
    scale = row_max / qmax
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    rounded = torch.round(z32 / safe_scale)
    clipped = rounded.clamp(-qmax, qmax)
    decoded = (clipped * scale).to(torch.bfloat16)
    stats = {
        "bits": bits,
        "qmax": qmax,
        "scale": scale,
        "zero_rows": int((row_max == 0).sum().item()),
        "saturation_count": int(((rounded < -qmax) | (rounded > qmax)).sum().item()),
        "elements": z32.numel(),
    }
    return decoded, stats


def decode_scalar_quantized(
    packed: torch.Tensor, scales: torch.Tensor, shape: tuple[int, int], group: int = 128
) -> torch.Tensor:
    out_features, in_features = shape
    if group < 1:
        raise ValueError("group must be positive")
    count = out_features * in_features
    nbytes = (count + 1) // 2
    if packed.numel() < nbytes:
        raise ValueError("packed tensor is too short")
    packed = packed.reshape(-1).to(torch.uint8)
    codes = torch.empty(nbytes * 2, dtype=torch.uint8, device=packed.device)
    codes[0::2] = packed[:nbytes] & 15
    codes[1::2] = packed[:nbytes] >> 4
    codes = codes[:count].to(torch.int16)
    signed = torch.where(codes >= 8, codes - 16, codes).float().reshape(shape)
    groups = (in_features + group - 1) // group
    padded_width = groups * group
    if padded_width != in_features:
        signed = torch.cat((signed, signed.new_zeros((out_features, padded_width - in_features))), dim=1)
    q = signed.reshape(out_features, groups, group)
    decoded = (q * scales.to(device=q.device, dtype=torch.float32).reshape(out_features, groups, 1))
    return decoded.reshape(out_features, padded_width)[:, :in_features].to(torch.bfloat16)


def scalar_quantize(weight: torch.Tensor, group: int = 128) -> dict[str, Any]:
    """Per-output-row grouped signed W4 with low nibble first and BF16 scales."""
    if weight.ndim != 2 or group < 1:
        raise ValueError("weight must be 2D and group must be positive")
    out_features, in_features = weight.shape
    groups = (in_features + group - 1) // group
    padded_width = groups * group
    values = weight.detach().float()
    if padded_width != in_features:
        values = torch.cat((values, values.new_zeros((out_features, padded_width - in_features))), dim=1)
    grouped = values.reshape(out_features, groups, group)
    scales = (grouped.abs().amax(dim=-1) / 7.0).to(torch.bfloat16)
    scales_f32 = scales.float()
    safe = torch.where(scales_f32 > 0, scales_f32, torch.ones_like(scales_f32))
    q = torch.round(grouped / safe.unsqueeze(-1)).clamp(-7, 7).to(torch.int16)
    q_real = q.reshape(out_features, padded_width)[:, :in_features].contiguous()
    codes = torch.remainder(q_real, 16).to(torch.uint8).reshape(-1)
    if codes.numel() & 1:
        codes = torch.cat((codes, codes.new_zeros(1)))
    packed = codes[0::2] | (codes[1::2] << 4)
    packed = packed.contiguous()
    decoded = decode_scalar_quantized(packed, scales, (out_features, in_features), group)
    storage_bytes = packed.numel() + scales.numel() * scales.element_size()
    return {
        "kind": "scalar",
        "packed": packed,
        "scales": scales,
        "shape": (out_features, in_features),
        "group": group,
        "decoded": decoded,
        "storage_bytes": storage_bytes,
        "effective_bpw": (8.0 * storage_bytes / weight.numel()) if weight.numel() else 0.0,
    }


def decode_vq_quantized(indices: torch.Tensor, codebooks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Decode stored [out, ceil(in/4), 2] uint8 indices with shared BF16 codebooks."""
    out_features, in_features = shape
    groups = (in_features + 3) // 4
    if indices.shape != (out_features, groups, 2):
        raise ValueError("indices shape does not match weight shape")
    if codebooks.shape != (2, 256, 4):
        raise ValueError("codebooks must have shape [2, 256, 4]")
    idx = indices.long()
    books = codebooks.float()
    vectors = books[0][idx[..., 0]] + books[1][idx[..., 1]]
    return vectors.reshape(out_features, groups * 4)[:, :in_features].to(torch.bfloat16)


def _assign(vectors: torch.Tensor, codebook: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    result = torch.empty(vectors.shape[0], dtype=torch.int64, device=vectors.device)
    codebook = codebook.float()
    for start in range(0, vectors.shape[0], 4096):
        x = vectors[start : start + 4096].float()
        m = metric[start : start + 4096].float()
        delta = x[:, None, :] - codebook[None, :, :]
        dist = (delta.square() * m[:, None, :]).sum(dim=-1)
        result[start : start + x.shape[0]] = dist.argmin(dim=1)
    return result


def _update_centroids(
    vectors: torch.Tensor, assignments: torch.Tensor, metric: torch.Tensor, previous: torch.Tensor
) -> torch.Tensor:
    sums = torch.zeros_like(previous, dtype=torch.float32)
    weights = torch.zeros_like(previous, dtype=torch.float32)
    sums.index_add_(0, assignments, vectors.float() * metric.float())
    weights.index_add_(0, assignments, metric.float())
    means = sums / weights.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where(weights > 0, means, previous.float())


def _kmeans(
    vectors: torch.Tensor, metric: torch.Tensor, generator: torch.Generator, steps: int = 5
) -> torch.Tensor:
    n = vectors.shape[0]
    if n == 0:
        return torch.zeros((256, 4), dtype=torch.float32, device=vectors.device)
    chosen = torch.randperm(n, generator=generator)[: min(n, 256)]
    if chosen.numel() < 256:
        extra = torch.randint(n, (256 - chosen.numel(),), generator=generator)
        chosen = torch.cat((chosen, extra))
    centers = vectors[chosen.to(vectors.device)].float().clone()
    for _ in range(steps):
        assignment = _assign(vectors, centers, metric)
        centers = _update_centroids(vectors, assignment, metric, centers)
    return centers


def fit_vq(
    weight: torch.Tensor,
    metric: torch.Tensor | None = None,
    seed: int = 20261006,
    sample_limit: int = 8192,
    iterations: int = 8,
) -> dict[str, Any]:
    """Fit two additive 256x4 codebooks; metric is diagonal activation second moment."""
    if weight.ndim != 2:
        raise ValueError("weight must be [out_features, in_features]")
    if sample_limit < 1 or iterations < 2:
        raise ValueError("sample_limit must be positive and iterations at least two")
    out_features, in_features = weight.shape
    groups = (in_features + 3) // 4
    padded_width = groups * 4
    values = weight.detach().float()
    if padded_width != in_features:
        values = torch.cat((values, values.new_zeros((out_features, padded_width - in_features))), dim=1)
    vectors = values.reshape(out_features, groups, 4).reshape(-1, 4).contiguous()

    if metric is None:
        coord_metric = torch.ones((out_features, groups, 4), dtype=torch.float32, device=weight.device)
        if padded_width != in_features:
            coord_metric.reshape(out_features, padded_width)[:, in_features:] = 0
    else:
        if metric.numel() != in_features:
            raise ValueError("metric must have one value per input feature")
        m = metric.detach().to(device=weight.device, dtype=torch.float32).reshape(-1).clamp_min(1e-12)
        m = m / m.mean()
        if padded_width != in_features:
            m = torch.cat((m, m.new_zeros(padded_width - in_features)))
        coord_metric = m.reshape(1, groups, 4).expand(out_features, -1, -1).contiguous()
    coord_metric = coord_metric.reshape(-1, 4)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_count = min(vectors.shape[0], sample_limit)
    if vectors.shape[0] <= sample_limit:
        sample_ids = torch.arange(vectors.shape[0], device=weight.device)
    else:
        # Sampling with replacement avoids a full-size CPU randperm for large linear weights.
        sample_ids = torch.randint(vectors.shape[0], (sample_count,), generator=generator).to(weight.device)
    sample = vectors[sample_ids]
    sample_metric = coord_metric[sample_ids]

    # Residual initialization: fit the first book, then fit its residual with the second.
    book0 = _kmeans(sample, sample_metric, generator).to(torch.bfloat16).float()
    init0 = _assign(sample, book0, sample_metric)
    residual = sample - book0[init0]
    book1 = _kmeans(residual, sample_metric, generator).to(torch.bfloat16).float()
    index1 = _assign(residual, book1, sample_metric)
    for _ in range(iterations):
        target0 = sample - book1[index1]
        index0 = _assign(target0, book0, sample_metric)
        book0 = _update_centroids(target0, index0, sample_metric, book0).to(torch.bfloat16).float()
        residual = sample - book0[index0]
        index1 = _assign(residual, book1, sample_metric)
        book1 = _update_centroids(residual, index1, sample_metric, book1).to(torch.bfloat16).float()

    # Indices are assigned against the stored BF16 codebooks, and decode uses those same values.
    book0_bf16, book1_bf16 = book0.to(torch.bfloat16), book1.to(torch.bfloat16)
    book0, book1 = book0_bf16.float(), book1_bf16.float()
    index0 = _assign(vectors, book0, coord_metric)
    index1 = _assign(vectors - book0[index0], book1, coord_metric)
    for _ in range(2):
        index0 = _assign(vectors - book1[index1], book0, coord_metric)
        index1 = _assign(vectors - book0[index0], book1, coord_metric)
    indices = torch.stack((index0, index1), dim=-1).reshape(out_features, groups, 2).to(torch.uint8)
    codebooks = torch.stack((book0_bf16, book1_bf16), dim=0)
    decoded_vectors = book0[index0] + book1[index1]
    decoded = decode_vq_quantized(indices, codebooks, (out_features, in_features))

    storage_bytes = indices.numel() + codebooks.numel() * codebooks.element_size()
    valid_metric = coord_metric > 0
    error = (vectors - decoded_vectors.to(torch.bfloat16).float()).square() * coord_metric
    mean_weighted_error = float(error.sum().item() / valid_metric.sum().clamp_min(1).item())
    return {
        "kind": "vq",
        "indices": indices,
        "codebooks": torch.stack((book0_bf16, book1_bf16), dim=0),
        "shape": (out_features, in_features),
        "decoded": decoded,
        "storage_bytes": storage_bytes,
        "effective_bpw": (8.0 * storage_bytes / weight.numel()) if weight.numel() else 0.0,
        "fit_stats": {
            "vectors": vectors.shape[0],
            "sample_vectors": sample_count,
            "iterations": iterations,
            "metric_weighted": metric is not None,
            "mean_weighted_squared_error": mean_weighted_error,
            "index_bytes": indices.numel(),
            "codebook_bytes": codebooks.numel() * codebooks.element_size(),
        },
    }

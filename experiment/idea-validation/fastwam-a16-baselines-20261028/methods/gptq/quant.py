"""GPTQ as in the official IST-DASLab/gptq defaults: per-row asymmetric min-max grid, no groups, plus --act-order.

Port of `gptq.py::GPTQ.fasterquant` and `quant.py::Quantizer` (https://github.com/IST-DASLab/gptq). Defaults
kept: blocksize 128, percdamp 0.01, groupsize -1, sym False, mse False. act-order is the official optional flag.
"""
from __future__ import annotations

import torch

BLOCK = 128
DAMP = 0.01


def find_params(w, bits):
    """quant.py::Quantizer.find_params(weight=True, perchannel=True, sym=False, mse=False)."""
    maxq = 2 ** bits - 1
    zero = torch.zeros(w.shape[0], device=w.device)
    xmin = torch.minimum(w.min(1).values, zero)
    xmax = torch.maximum(w.max(1).values, zero)
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp], xmax[tmp] = -1, +1
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    return scale.unsqueeze(1), zero.unsqueeze(1), maxq


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def quantize_layer(weight, hessian, bits, seed=0, act_order=True):
    """Returns the dequantized weight in original column order and the stored bits (codes + per-row scale/zero)."""
    del seed  # deterministic algorithm
    W = weight.float().clone()
    rows, columns = W.shape
    scale, zero, maxq = find_params(W, bits)
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    Q = torch.zeros_like(W)
    damp = DAMP * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=W.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    del H
    for i1 in range(0, columns, BLOCK):
        i2 = min(i1 + BLOCK, columns)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, d = W1[:, i], Hinv1[i, i]
            q = quantize(w.unsqueeze(1), scale, zero, maxq).flatten()
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    if act_order:
        Q = Q[:, torch.argsort(perm)]
    # Official packing: b-bit codes, b-bit zero points, one FP16 scale per row.
    bits_total = rows * columns * bits + rows * (bits + 16)
    return dict(dequant=Q, bits_total=float(bits_total),
                config=dict(method='gptq', bits=bits, groupsize=-1, sym=False, mse=False, act_order=act_order,
                            blocksize=BLOCK, percdamp=DAMP, upstream='IST-DASLab/gptq fasterquant'))

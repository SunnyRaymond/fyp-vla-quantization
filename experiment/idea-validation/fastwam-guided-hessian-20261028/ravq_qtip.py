"""QTIP one-shot quantization of one Linear for RAVQ: the A16 baselines' port (methods/qtip/quant.py), K in 1..4.

The port accepts K in 2..4 (the official CUDA kernels exist for those). The trellis code, Viterbi, packing and the CPU
decoder are generic in K (bitshift_codebook with L = 16, V = 2 shifts K * V bits per step), so this is the port's
quantize_layer with the K check widened and the packed codes returned (trellis, SU, SV, Wscale, LUT) for the calibrated
stage. Same padding, sign vectors from `seed`, BlockLDLQ + tail-biting Viterbi, sigma_reg, scale_override and decode
check; for K = 2 it equals the port bit for bit (test_ravq.py). No model imports.
"""
from __future__ import annotations

import time
import types

import torch


def quantize_layer(Q, weight, hessian, bits, seed=0):
    """Q: the port's module (methods/qtip/quant.py). weight [out, in], hessian [in, in] = E[x x^T].

    Returns dict(dequant [out, in] float32, bits_total, config, codes) with codes = the packed trellis (int16), SU, SV
    (Wscale folded in, as stored), Wscale, the [512, 2] LUT, kernel_layout and the padded shape."""
    K = int(bits)
    if K not in (1, 2, 3, 4):
        raise ValueError(f'K must be 1..4, got {bits}')
    t0 = time.monotonic()
    device = weight.device
    m, n = weight.shape
    if tuple(hessian.shape) != (n, n):
        raise ValueError(f'hessian {tuple(hessian.shape)} does not match in_features {n}')
    m_p, n_p = Q.padded_size(m), Q.padded_size(n)
    kernel_layout = m_p % 32 == 0 and n_p % 32 == 0
    buf_cols = Q.ldlq_buf_cols(n_p)
    SU, SV = Q.sign_vectors(n_p, m_p, seed)
    with torch.no_grad():
        H = hessian.detach().to(device=device, dtype=torch.float32)
        dm = torch.diag(H).mean()
        h_fallback = None
        if not bool(torch.isfinite(H).all()):
            raise ValueError('hessian has non-finite entries')
        if not float(dm) > 0:
            H = torch.eye(n, device=device)
            dm = torch.ones((), device=device)
            h_fallback = 'identity (mean diag(H) == 0)'
        W_p = torch.zeros(m_p, n_p, device=device, dtype=torch.float32)
        W_p[:m, :n] = weight.detach().to(device=device, dtype=torch.float32)
        attempts = [(torch.float32, Q.SIGMA_REG), (torch.float64, Q.SIGMA_REG),
                    (torch.float64, 10 * Q.SIGMA_REG), (torch.float64, 100 * Q.SIGMA_REG)]
        saved = extras = None
        tried = []
        for dtype_, sigma in attempts:
            if n_p == n:
                H_in, sigma_in = H.clone(), sigma
            else:
                H_in = torch.zeros(n_p, n_p, device=device, dtype=torch.float32)
                H_in[:n, :n] = Q.regularize_H(H.clone(), sigma)
                idx = torch.arange(n, n_p, device=device)
                H_in[idx, idx] = sigma * dm
                sigma_in = 0.0
            args = types.SimpleNamespace(sigma_reg=sigma_in, scale_override=Q.SCALE_OVERRIDE,
                                         td_x=Q.TD_X, td_y=Q.TD_Y, K=K, V=Q.V)
            cb = Q.codebook(K, device)
            try:
                saved, extras = Q.official_quantize_linear(
                    W_p.to(dtype_), H_in.to(dtype_), SU.to(device, dtype_), SV.to(device, dtype_), cb, args,
                    device, Q.MODEL_DTYPE, has_kernel=kernel_layout, for_kernel=True, buf_cols=buf_cols)
                tried.append(dict(dtype=str(dtype_), sigma_reg=sigma, ok=True))
                break
            except TypeError as exc:  # block_LDL returned None (Cholesky failed)
                if 'NoneType' not in str(exc):
                    raise
                tried.append(dict(dtype=str(dtype_), sigma_reg=sigma, ok=False))
                saved = extras = None
        if saved is None:
            raise RuntimeError(f'QTIP block_LDL failed for every attempt: {tried}')
        hatWr, cb = extras['hatWr'], extras['cb']
        del extras
        Wscale = saved['Wscale']
        decoded = Q.decode_packed(saved['trellis'], saved['tlut'], cb, K, m_p, n_p, kernel_layout)
        if not bool(torch.equal(decoded.to(device, hatWr.dtype) * Wscale, hatWr)):
            raise RuntimeError(f'QTIP K={K}: decoded trellis does not reproduce the LDLQ output')
        del decoded
        SU_b = saved['SU'].to(device, hatWr.dtype)
        SV_b = saved['SV'].to(device, hatWr.dtype)
        dequant = Q.inverse_incoherence(hatWr, SU_b, SV_b)[:m, :n].float().contiguous()
        del hatWr
    trellis_bits = saved['trellis'].numel() * 16
    bits_total = float(trellis_bits + n_p * Q.SU_BITS + m_p * Q.SV_BITS)
    if bits_total != Q.storage_bits(m, n, K):
        raise RuntimeError(f'QTIP K={K}: stored bits {bits_total} differ from storage_bits {Q.storage_bits(m, n, K)}')
    config = dict(method='QTIP', upstream=Q.UPSTREAM, upstream_commit=Q.UPSTREAM_COMMIT, code='HYB',
                  decode_mode=Q.DECODE_MODE, L=Q.L, K=K, V=Q.V, tlut_bits=Q.TLUT_BITS, td_x=Q.TD_X, td_y=Q.TD_Y,
                  kernel_bit_layout=kernel_layout, ldlq_buf_cols=buf_cols, sigma_reg=Q.SIGMA_REG,
                  scale_override=Q.SCALE_OVERRIDE, attempts=tried, hessian_fallback=h_fallback, fine_tuning='none',
                  seed=int(seed), shape=[m, n], padded_shape=[m_p, n_p], bpw=bits_total / (m * n),
                  Wscale=float(Wscale), proxy_err_official=float(saved['proxy_err']), decode_verified=True,
                  viterbi_update=getattr(cb.update, 'mode', 'official'), seconds=time.monotonic() - t0)
    codes = dict(trellis=saved['trellis'].cpu(), SU=saved['SU'].cpu(), SV=saved['SV'].cpu(), Wscale=float(Wscale),
                 tlut=saved['tlut'].cpu(), kernel_layout=kernel_layout, padded_shape=[m_p, n_p], K=K)
    return dict(dequant=dequant.to(device=weight.device, dtype=torch.float32), bits_total=bits_total, config=config,
                codes=codes)

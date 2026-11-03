"""CPU checks for the QTIP port: python -B test_qtip.py [--quick]. Self-contained (reads only this directory).

1. LUT: shipped k-means centroids match their hash; the scipy-free k-means equals scipy's bit for bit.
2. Hadamard: vendored had12/20/28/36 are Hadamard; pure-torch matmul_hadU == FWHT-based matmul_hadU_cuda.
3. Padding table for every Fast-WAM size.
4. fast_update (CUDA eager fallback) == official bitshift_codebook.update bit for bit, ties included.
5. Wrapper == vendored official routine (same SU/SV/H/W): dequant bitwise equal; the official stored format
   (unpack_quip -> BitshiftLinear.cache_hatW / forward through decode_compressed) reproduces the dequant.
6. Proxy error tr(dW H dW^T)/tr(W H W^T): QTIP << per-row RTN at the same bits, and 3-bit < 2-bit.
7. Odd shapes (in=7, in=8, out=7, 12k, 20k, 28k, 36k widths), degenerate Hessians, Cholesky-failure fp64
   retry, determinism, bit counts.
--quick runs 1-4, a tiny 5 and the bit counts (under a minute).
"""
from __future__ import annotations

import os
import sys
import time
import types

import numpy as np
import torch

import quant
from qtip_official import bitshift
from qtip_official.linear import quantize_linear as official_quantize_linear, unpack_quip
from qtip_official.matmul_had import (get_had12, get_had20, get_had28, get_had36, get_hadK, matmul_hadU,
                                      matmul_hadU_cuda, matmul_hadUt, matmul_hadUt_cuda)

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)


def proxy(w, q, h):
    d = q - w
    return float(((d @ h) * d).sum() / ((w @ h) * w).sum())


def correlated(samples, k, g):
    """Activation Hessian E[x x^T] with low-rank correlation and a few outlier channels."""
    base = torch.randn(samples, max(k // 4, 1), generator=g)
    x = base @ torch.randn(max(k // 4, 1), k, generator=g) + 0.1 * torch.randn(samples, k, generator=g)
    x[:, : min(3, k)] *= 20
    return x.t() @ x / samples


def rtn(w, bits):
    """Plain per-output-row asymmetric min-max round-to-nearest with 2^bits levels."""
    maxq = 2 ** bits - 1
    lo = torch.minimum(w.min(1, keepdim=True).values, torch.zeros(1))
    hi = torch.maximum(w.max(1, keepdim=True).values, torch.zeros(1))
    scale = ((hi - lo) / maxq).clamp_min(1e-12)
    zero = torch.round(-lo / scale)
    return scale * (torch.clamp(torch.round(w / scale) + zero, 0, maxq) - zero)


def test_lut():
    raw = bitshift.kmeans_tlut(9, 2)
    assert raw.shape == (512, 2) and raw.dtype == torch.float32
    cb = bitshift.bitshift_codebook(L=16, K=2, V=2, tlut_bits=9, decode_mode='quantlut_sym')
    assert abs(float(cb.tlut.std(unbiased=False)) - 0.9682458365518543) < 1e-6
    assert cb.lut.shape == (2, 1 << 16)
    assert torch.isin(cb.lut[1], cb.tlut[:, 1]).all() and torch.isin(cb.lut[0].abs(), cb.tlut[:, 0].abs()).all()
    try:
        import scipy.cluster.vq as svq
    except ImportError:
        svq = None
    if svq is not None:
        g = torch.Generator().manual_seed(1)
        guess, obs = torch.randn(64, 2, generator=g), torch.randn(1 << 15, 2, generator=g)
        mine = bitshift._scipy_kmeans_replica(obs.numpy(), guess.numpy())[0]
        ref = np.asarray(svq.kmeans(obs, guess)[0])
        assert np.array_equal(mine, ref), np.abs(mine - ref).max()
        log('k-means replica == scipy.cluster.vq.kmeans bit for bit (64 centroids, 32768 points)')
    if os.environ.get('QTIP_TEST_FULL_LUT') == '1':
        full = bitshift.kmeans_tlut(9, 2, use_cache=False)
        assert torch.equal(full, raw)
        log('full seed-0 k-means (2^20 points, 512 centroids) regenerated == shipped file')
    log(f'LUT ok: raw k-means file hash verified, normalized std {float(cb.tlut.std(unbiased=False)):.7f}')


def test_hadamard(sizes):
    for k, f in ((12, get_had12), (20, get_had20), (28, get_had28), (36, get_had36)):
        h = f()
        assert torch.equal(h @ h.T, k * torch.eye(k)), k
    g = torch.Generator().manual_seed(2)
    for n in sizes:
        x = torch.randn(5, n, generator=g)
        hadK, K = get_hadK(n)
        a, b = matmul_hadU(x), matmul_hadU_cuda(x, hadK, K)
        at, bt = matmul_hadUt(x), matmul_hadUt_cuda(x, hadK, K)
        assert torch.allclose(a, b, atol=1e-5) and torch.allclose(at, bt, atol=1e-5), n
        assert torch.allclose(matmul_hadU(matmul_hadUt(x)), x, atol=1e-5), n
        assert torch.allclose(a.norm(dim=1), x.norm(dim=1), rtol=1e-5), n
    log(f'Hadamard ok: had12/20/28/36 orthogonal; matmul_hadU == FWHT matmul_hadU_cuda for n in {sizes}')


def test_padding():
    table = {7: 16, 8: 16, 16: 16, 100: 112, 192: 192, 256: 256, 1024: 1024, 3072: 3072, 4096: 4096,
             6144: 6144, 14336: 14336, 18432: 18432}
    for n, p in table.items():
        assert quant.padded_size(n) == p, (n, quant.padded_size(n), p)
    hadk = {n: get_hadK(n)[1] for n in (192, 3072, 6144, 14336, 18432, 1024)}
    assert hadk == {192: 12, 3072: 12, 6144: 12, 14336: 28, 18432: 36, 1024: 1}, hadk
    log(f'padding ok: {table}; Hadamard K {hadk}')


def test_fast_update():
    g = torch.Generator().manual_seed(3)
    for K in (2, 3):
        cb = bitshift.bitshift_codebook(L=16, K=K, V=2, tlut_bits=9, decode_mode='quantlut_sym').to(torch.bfloat16)
        for B in (1, 5):
            for ties in (True, False):
                cost = (torch.randint(0, 9, (B, 1 << 16), generator=g).float() if ties
                        else 10 * torch.rand(B, 1 << 16, generator=g))
                thing = torch.randn(2, B, generator=g).to(torch.float16)
                p1, c1 = cb.update(cost, thing)
                p2, c2 = quant.fast_update(cb, cost, thing)
                assert torch.equal(p1, p2) and torch.equal(c1, c2), (K, B, ties)
    log('fast_update == official update bit for bit (K=2,3; with cost ties)')


def official_reference(W, H, K, seed, kernel_layout):
    m, n = W.shape
    SU, SV = quant.sign_vectors(n, m, seed)
    cb = bitshift.bitshift_codebook(L=16, K=K, V=2, tlut_bits=9, decode_mode='quantlut_sym')
    args = types.SimpleNamespace(sigma_reg=0.01, scale_override=0.9, td_x=16, td_y=16, K=K, V=2)
    return official_quantize_linear(W.clone(), H.clone(), SU, SV, cb, args, torch.device('cpu'), torch.bfloat16,
                                    has_kernel=kernel_layout, for_kernel=True, buf_cols=quant.ldlq_buf_cols(n))


def unsigned_op_missing(exc):
    return any(t in str(exc) for t in ('UInt16', 'UInt32', 'uint16', 'uint32'))


def test_official_equivalence(cases=((64, 128, 2), (48, 192, 3))):
    g = torch.Generator().manual_seed(4)
    for (m, n, K) in cases:
        W = 0.02 * torch.randn(m, n, generator=g)
        H = correlated(4 * n, n, g)
        kernel_layout = m % 32 == 0 and n % 32 == 0
        r = quant.quantize_layer(W, H, K, seed=5)
        saved, ex = official_reference(W, H, K, 5, kernel_layout)
        ref = quant.inverse_incoherence(ex['hatWr'], saved['SU'].float(), saved['SV'].float())
        assert torch.equal(r['dequant'], ref), 'wrapper differs from the vendored official routine'
        assert r['config']['proxy_err_official'] == saved['proxy_err']
        assert r['config']['kernel_bit_layout'] == kernel_layout and r['config']['decode_verified']
        msg = f'{m}x{n} K={K}: wrapper dequant == official routine (bitwise)'
        bufs = unpack_quip(saved)
        hl, kl = get_hadK(n)
        hr, kr = get_hadK(m)
        if kernel_layout:
            # production check path: unswizzle + pack_trellis decoding
            dec = quant.decode_packed(saved['trellis'], saved['tlut'], ex['cb'], K, m, n, True)
            assert torch.equal(dec * saved['Wscale'], ex['hatWr'])
            states = ex['cb'].unpack_trellis(quant.unswizzle_kernel_layout(saved['trellis'], m, n, K), 256)
            q = ex['Qidxs'].reshape(m // 16, 16, n // 16, 8).transpose(1, 2).reshape(-1, 128)
            assert torch.equal(states, q.to(states.dtype))
            msg += '; unswizzled trellis == Viterbi states'
            try:  # official inference path from the stored HF-format buffers
                dec_k = quant.decode_packed(saved['trellis'], saved['tlut'], ex['cb'], K, m, n, True,
                                            official_kernel_decoder=True)
                assert torch.equal(dec_k, dec)
                bl = bitshift.BitshiftLinear(16, 16, 16, K, 2, 9, 'quantlut_sym', dtype=torch.float32,
                                             tlut=bufs['tlut'], has_kernel=True)
                bl.cache_hatW(bufs['trellis'], hl, hr, kl, kr, m, n, 0, 8)
                w_off = (bufs['SV'] * bl.scale)[:, None] * bl.hatW * bufs['SU'].float()[None, :]
                rel = float((w_off - r['dequant']).abs().max() / r['dequant'].abs().max())
                assert rel < 1e-5, rel
                x = torch.randn(6, n, generator=g)
                y = bl.forward(x, bufs['trellis'], bufs['SU'], bufs['SV'], hl, hr, kl, kr, 0, 8, mode='eval')
                y_ref = x @ r['dequant'].T
                rel_y = float((y - y_ref).norm() / y_ref.norm())
                assert rel_y < 5e-3, rel_y  # the official forward rounds activations to fp16
                msg += (f'; official decode_compressed == unswizzled decode; official HF-format weight '
                        f'(cache_hatW, FWHT) max rel diff {rel:.1e}; official eval forward vs dense Linear '
                        f'rel {rel_y:.1e}')
            except (RuntimeError, NotImplementedError) as exc:
                if not unsigned_op_missing(exc):
                    raise
                msg += f'; SKIPPED official decode_compressed path (torch {torch.__version__}: {exc})'
        else:
            dec = quant.decode_packed(saved['trellis'], saved['tlut'], ex['cb'], K, m, n, False)
            assert torch.equal(dec * saved['Wscale'], ex['hatWr'])
            states = ex['cb'].unpack_trellis(saved['trellis'], 256)
            q = ex['Qidxs'].reshape(m // 16, 16, n // 16, 8).transpose(1, 2).reshape(-1, 128)
            assert torch.equal(states, q.to(states.dtype))
            msg += '; pack/unpack_trellis round trip and decode == LDLQ output (non-kernel bit layout)'
        log(msg)


def test_error_vs_rtn():
    g = torch.Generator().manual_seed(6)
    m, n = 128, 256
    W = 0.02 * torch.randn(m, n, generator=g)
    H = correlated(2048, n, g)
    errs = {}
    for K in (2, 3):
        t = time.time()
        r = quant.quantize_layer(W, H, K, seed=7)
        dt = time.time() - t
        e_q, e_r = proxy(W, r['dequant'], H), proxy(W, rtn(W, K), H)
        errs[K] = e_q
        assert e_q < 0.5 * e_r, (K, e_q, e_r)
        assert r['bits_total'] == quant.storage_bits(m, n, K)
        log(f'{m}x{n} K={K}: proxy QTIP {e_q:.5f} vs RTN {e_r:.5f}; bpw {r["bits_total"] / W.numel():.4f}; '
            f'CPU {dt:.1f}s ({torch.get_num_threads()} threads)')
    assert errs[3] < 0.5 * errs[2], errs


def test_shapes_and_hessians():
    g = torch.Generator().manual_seed(8)
    # in=7, in=8, out=7; in = 12*16 and 28*16 (kernel bit layout), 20*4 and 36*4 (pack_trellis layout); out = 12*16
    cases = [(32, 7, 2), (64, 8, 3), (7, 64, 2), (32, 192, 3), (32, 448, 2), (16, 80, 3), (16, 144, 2),
             (192, 64, 2)]
    for (m, n, K) in cases:
        W = torch.randn(m, n, generator=g)
        H = correlated(4 * n + 16, n, g)
        H[0, :] = 0
        H[:, 0] = 0  # dead input channel
        r = quant.quantize_layer(W, H, K, seed=9)
        dq, cfg = r['dequant'], r['config']
        assert dq.shape == (m, n) and dq.dtype == torch.float32 and torch.isfinite(dq).all()
        assert cfg['decode_verified'] and r['bits_total'] == quant.storage_bits(m, n, K)
        e = proxy(W, dq, H)
        assert e < 0.5, (m, n, K, e)
        log(f'shape {m}x{n} K={K}: padded {cfg["padded_shape"]}, Had K in/out {cfg["hadamard_K_in"]}/'
            f'{cfg["hadamard_K_out"]}, kernel layout {cfg["kernel_bit_layout"]}, proxy {e:.4f}, '
            f'bpw {r["bits_total"] / (m * n):.3f}')
    # rank-deficient, dead channels, huge dynamic range
    n = 128
    x = torch.randn(32, n, generator=g) * torch.logspace(-4, 2, n)
    x[:, 5:9] = 0
    H = x.t() @ x / 32
    r = quant.quantize_layer(torch.randn(64, n, generator=g), H, 2, seed=1)
    assert torch.isfinite(r['dequant']).all() and r['config']['decode_verified']
    log(f'rank-deficient ill-conditioned H (32 samples, n=128, 4 dead): ok, attempts {r["config"]["attempts"]}')
    r = quant.quantize_layer(torch.randn(64, n, generator=g), torch.zeros(n, n), 3, seed=1)
    assert torch.isfinite(r['dequant']).all() and r['config']['hessian_fallback']
    log(f'all-zero H: fallback "{r["config"]["hessian_fallback"]}"')


def test_cholesky_fallback():
    """Official block_LDL returning None crashes the official code; the wrapper retries in fp64."""
    import qtip_official.linear as lin
    g = torch.Generator().manual_seed(11)
    W, H = torch.randn(32, 64, generator=g), correlated(256, 64, g)
    orig = lin.block_LDL
    lin.block_LDL = lambda Hm, b, **kw: None if Hm.dtype == torch.float32 else orig(Hm, b, **kw)
    try:
        r = quant.quantize_layer(W, H, 2, seed=1)
    finally:
        lin.block_LDL = orig
    att = r['config']['attempts']
    assert [a['ok'] for a in att] == [False, True] and att[1]['dtype'] == 'torch.float64', att
    assert torch.isfinite(r['dequant']).all() and r['config']['decode_verified']
    e64, e32 = proxy(W, r['dequant'], H), proxy(W, quant.quantize_layer(W, H, 2, seed=1)['dequant'], H)
    assert abs(e64 - e32) < 0.2 * e32, (e64, e32)
    log(f'Cholesky failure -> fp64 retry ok (proxy fp64 {e64:.4f} vs fp32 {e32:.4f})')


def test_determinism():
    g = torch.Generator().manual_seed(10)
    W, H = torch.randn(32, 64, generator=g), correlated(256, 64, g)
    a = quant.quantize_layer(W, H, 2, seed=3)['dequant']
    b = quant.quantize_layer(W, H, 2, seed=3)['dequant']
    c = quant.quantize_layer(W, H, 2, seed=4)['dequant']
    assert torch.equal(a, b) and not torch.equal(a, c)
    log('deterministic given seed; a different seed changes the sign vectors')


def test_bit_counts():
    rows = []
    for (m, n) in ((3072, 3072), (1024, 1024), (18432, 3072), (3072, 14336), (1024, 7), (7, 1024), (4096, 8)):
        for K in (2, 3):
            b = quant.storage_bits(m, n, K)
            rows.append(f'{m}x{n} K={K}: {b} bits = {b / (m * n):.4f} bpw')
    assert quant.storage_bits(3072, 3072, 2) == 3072 * 3072 * 2 + 3072 * 16 + 3072 * 32
    assert quant.storage_bits(1024, 7, 2) == 1024 * 16 * 2 + 16 * 16 + 1024 * 32
    log('bit counts:\n    ' + '\n    '.join(rows))


def main():
    torch.set_grad_enabled(False)
    quick = '--quick' in sys.argv  # < 1 minute subset
    log(f'torch {torch.__version__}, numpy {np.__version__}, {torch.get_num_threads()} threads, '
        f'{"QUICK subset" if quick else "full suite"}')
    test_lut()
    test_hadamard((16, 48, 192, 448) if quick else (16, 48, 192, 320, 448, 576, 1024, 3072))
    test_padding()
    test_fast_update()
    if quick:
        test_official_equivalence(((32, 64, 2), (16, 48, 3)))
        test_bit_counts()
        log('QUICK QTIP SUBSET PASSED (full suite not run)')
        return
    test_official_equivalence()
    test_error_vs_rtn()
    test_shapes_and_hessians()
    test_cholesky_fallback()
    test_determinism()
    test_bit_counts()
    log('ALL QTIP TESTS PASSED')


if __name__ == '__main__':
    main()

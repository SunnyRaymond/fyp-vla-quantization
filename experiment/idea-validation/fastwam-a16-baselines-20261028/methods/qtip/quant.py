"""QTIP (Tseng et al., NeurIPS 2024) one-shot weight PTQ of one Linear, dequantized to a dense weight.

Official recipe (Cornell-RelaxML/qtip @ e90c6688, example.sh / quantize_finetune_llama.py): HYB code
(decode_mode quantlut_sym, L=16, Q=tlut_bits=9, V=2), K = bits, 16x16 trellis tiles (td_x=td_y=16), random
Hadamard incoherence on both sides, BlockLDLQ with tail-biting Viterbi, sigma_reg=1e-2, scale_override=0.9.
No fine-tuning of any kind (no LUT tuning, no layer or end-to-end fine-tuning).

The per-Linear body is the vendored official routine (qtip_official/linear.py). This wrapper only adds
padding to tile/Hadamard-compatible sizes, sign-vector generation from `seed`, the inverse incoherence
transform back to original coordinates, a decode check of the packed bitstream, and bit accounting.
Third-party code is GPL-3.0 (LICENSE); see UPSTREAM.md for every deviation.
"""
from __future__ import annotations

import math
import os
import sys
import time
import types
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qtip_official import bitshift, ldlq  # noqa: E402
from qtip_official.kernel_decompress import decode_compressed  # noqa: E402
from qtip_official.linear import quantize_linear as official_quantize_linear  # noqa: E402
from qtip_official.math_utils import regularize_H  # noqa: E402
from qtip_official.matmul_had import get_hadK, is_pow2, matmul_hadU  # noqa: E402

UPSTREAM = 'https://github.com/Cornell-RelaxML/qtip'
UPSTREAM_COMMIT = 'e90c6688c8dfae326a3a81b5eb032db7c6680ec0'

L = 16
V = 2
TLUT_BITS = 9
DECODE_MODE = 'quantlut_sym'
TD_X = TD_Y = 16
SIGMA_REG = 1e-2
SCALE_OVERRIDE = 0.9
MODEL_DTYPE = torch.bfloat16      # official orig_dtype = the model's parameter dtype (Fast-WAM is BF16)
SU_BITS = 16                      # QuantizedLinear.SU buffer: model dtype
SV_BITS = 32                      # QuantizedLinear.SV buffer: float32 (Wscale folded in)
SHARED_TLUT_BITS = (1 << TLUT_BITS) * V * 16  # one fp16 512x2 LUT, identical for every layer without LUT tuning
OFFICIAL_HAD_K = (172, 156, 140, 124, 116, 108, 60, 52, 36, 28, 20, 12)  # get_hadK test order
VENDORED_HAD_K = (36, 28, 20, 12)
# Viterbi step (bitshift_codebook.update):
#   'auto'     CUDA: torch.compile(official update) as in the official code (@torch.compile), falling back to
#              'fast' if compilation fails; CPU: 'official'.
#   'official' the vendored update, eager.   'compile' torch.compile(official update), no fallback.
#   'fast'     fast_update(): same outputs as the official update bit for bit (tested), less memory traffic.
UPDATE_MODE = os.environ.get('QTIP_UPDATE', 'auto')
# Decode the packed trellis and require it to reproduce the LDLQ output bit for bit.
VERIFY_DECODE = os.environ.get('QTIP_VERIFY_DECODE', '1') != '0'

_CODEBOOKS = {}


def had_ok(n):
    """True when the official get_hadK(n) succeeds with a vendored Hadamard factor."""
    for k in OFFICIAL_HAD_K:
        if n % k == 0:
            return k in VENDORED_HAD_K and is_pow2(n // k)
    return is_pow2(n)


def padded_size(n):
    """Smallest p >= n that is a multiple of the 16-wide trellis tile and an official Hadamard size."""
    p = max(TD_X, -(-n // TD_X) * TD_X)
    while not had_ok(p):
        p += TD_X
    return p


def storage_bits(out_features, in_features, bits):
    """Stored bits of one QuantizedLinear (official HF format, rcp=0) on the padded shape: trellis int16 buffer
    [(m/16)(n/16), ceil(256 K / 16)] = K bits per padded weight, SU [n] in the model dtype, SV [m] in float32
    (Wscale folded in). The 512x2 fp16 LUT is identical for every layer (no LUT tuning): not counted here."""
    m_p, n_p = padded_size(out_features), padded_size(in_features)
    trellis = (m_p // TD_X) * (n_p // TD_Y) * math.ceil(TD_X * TD_Y * bits / 16) * 16
    return trellis + n_p * SU_BITS + m_p * SV_BITS


def ldlq_buf_cols(n):
    """Official buf_cols is 128 and LDLQ asserts n % buf_cols == 0; use the largest of 128/64/32/16 that
    divides n (only the blocking of the same column-by-column recursion changes)."""
    for b in (128, 64, 32, 16):
        if n % b == 0:
            return b
    raise ValueError(n)


def fast_update(cb, cost, thing):
    """Returns exactly what bitshift_codebook.update returns (prev_state int64 [B, 2^(L-KV)], cost float32
    [B, 2^L]) for a bf16 LUT and fp16 input (float32 state error). Same float ops in the same order:
    state_err = (r0 - t0)^2 + (r1 - t1)^2; candidates of next-state group j are the previous states
    j + k * 2^(L-KV), k ascending, first minimum wins (torch.min(dim) as in the official gather + min)."""
    KV = cb.K * cb.V
    B = thing.shape[1]
    err = None
    for v in range(cb.V):
        d = (cb.recons_state[v] - thing[v].unsqueeze(-1)).square()
        err = d if err is None else err + d
    assert err.dtype == torch.float32 and cost.dtype == torch.float32
    best = torch.min(cost.view(B, 1 << KV, 1 << (cb.L - KV)), dim=1)
    new_cost = (err.view(B, 1 << (cb.L - KV), 1 << KV) + best.values.unsqueeze(-1)).view(B, 1 << cb.L)
    prev_state = (best.indices << (cb.L - KV)) + torch.arange(1 << (cb.L - KV), device=cost.device)
    return prev_state, new_cost


class _Update:
    """Instance-level replacement of cb.update according to UPDATE_MODE."""

    def __init__(self, cb, mode):
        self.official = cb.update
        self.fast = lambda cost, thing: fast_update(cb, cost, thing)
        self.mode = mode
        self.compiled = torch.compile(self.official) if mode in ('compile', 'compile+fast') else None

    def __call__(self, cost, thing):
        if self.mode in ('compile', 'compile+fast'):
            try:
                return self.compiled(cost, thing)
            except Exception as exc:
                if self.mode == 'compile':
                    raise
                warnings.warn(f'QTIP: torch.compile(update) failed ({type(exc).__name__}: {exc}); '
                              'using the bit-identical eager fast_update')
                self.mode = 'fast'
        if self.mode == 'fast':
            return self.fast(cost, thing)
        return self.official(cost, thing)


def codebook(K, device):
    """Official `bitshift.bitshift_codebook(L, K, V, tlut_bits, decode_mode)`; cached per (K, device).
    The official routine moves it to the device and to the model dtype in place (cb.to(device).to(dtype))."""
    key = (K, str(device))
    if key not in _CODEBOOKS:
        cb = bitshift.bitshift_codebook(L=L, K=K, V=V, tlut_bits=TLUT_BITS, decode_mode=DECODE_MODE)
        mode = UPDATE_MODE
        if mode == 'auto':
            mode = 'compile+fast' if torch.device(device).type == 'cuda' else 'official'
        if mode in ('compile', 'compile+fast'):
            try:  # official: torch._dynamo.config.cache_size_limit = 256 (lib/utils/matmul_had.py)
                torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
            except AttributeError:
                pass
        if mode != 'official':
            cb.update = _Update(cb, mode)
        _CODEBOOKS[key] = cb
    return _CODEBOOKS[key]


def sign_vectors(n, m, seed):
    """Official: SU = (torch.randn(n, device=cuda).sign() + 1e-5).sign(); SV likewise for m, after
    torch.manual_seed(layer_idx). Here a CPU generator seeded with `seed` draws them in the same order."""
    g = torch.Generator().manual_seed(int(seed))
    SU = (torch.randn(n, generator=g).sign() + 1e-5).sign()
    SV = (torch.randn(m, generator=g).sign() + 1e-5).sign()
    return SU, SV


def inverse_incoherence(hatWr, SU, SV):
    """diag(SV) Had_m hatWr Had_n^T diag(SU): the inverse of Wr = hadUt(hadUt(W.T * SV).T * SU), in the
    form of BitshiftLinear.cache_hatW (rcp == 0) with the official pure-torch matmul_hadU."""
    return matmul_hadU(matmul_hadU(hatWr).T).T * SV.unsqueeze(1) * SU.unsqueeze(0)


def unswizzle_kernel_layout(trellis, m, n, K):
    """Inverse of the CUDA-kernel bit swizzle applied after pack_trellis in quantize_finetune_decoder_layer
    (has_kernel branch); uint8/int16 views only."""
    x = trellis.contiguous().view(torch.uint8).reshape(m // 32, n // 32, 16 * 16 // 8, 2, 2, K)
    x = x.flip((-1, )).permute(0, 4, 1, 3, 2, 5).contiguous()
    return x.reshape(-1, 2).flip((-1, )).contiguous().view(torch.int16).reshape(trellis.shape)


def decode_packed(trellis, tlut, cb, K, m, n, kernel_layout, official_kernel_decoder=False):
    """LUT values (in units of Wscale) decoded from the packed int16 trellis on CPU (where the official code
    packs it), [m, n]. tlut: the stored [512, 2] LUT; cb supplies unpack_trellis (no tensors used).
    Kernel layout: unswizzle + pack_trellis decoding by default; official_kernel_decoder=True uses the official
    decode_compressed reference instead (needs uint16 cat / uint32 views, used by the tests)."""
    trellis = trellis.cpu()
    lut = bitshift.quantlut_sym(tlut.cpu().float().to(torch.float16), L, TLUT_BITS).T.contiguous()
    if kernel_layout and official_kernel_decoder:
        return decode_compressed(L, TLUT_BITS, K, int(math.log2(V)), m, n, trellis.reshape(-1),
                                 lut.T.contiguous()).float()
    if kernel_layout:
        trellis = unswizzle_kernel_layout(trellis, m, n, K)
    states = cb.unpack_trellis(trellis, TD_X * TD_Y)
    seq = lut[:, states.long()].permute(1, 2, 0).reshape(-1, TD_X * TD_Y)  # sequence order
    tiles = seq[:, ldlq._INV_PERMUTE.to(seq.device)]                       # row-major 16x16 tile
    return tiles.reshape(m // TD_X, n // TD_Y, TD_X, TD_Y).transpose(1, 2).reshape(m, n).float()


def quantize_layer(weight, hessian, bits, seed=0) -> dict:
    """weight [out, in] float32, hessian [in, in] float32 = E[x x^T]; bits = K in {2, 3, 4}.

    Returns dict(dequant=[out, in] float32 in original coordinates, bits_total=float, config=dict)."""
    K = int(bits)
    if K not in (2, 3, 4):
        raise ValueError(f'QTIP HYB kernels/configs exist for K in 2..4, got {bits}')
    t0 = time.monotonic()
    device = weight.device
    m, n = weight.shape
    if tuple(hessian.shape) != (n, n):
        raise ValueError(f'hessian {tuple(hessian.shape)} does not match in_features {n}')
    m_p, n_p = padded_size(m), padded_size(n)
    # The CUDA-kernel bit layout needs m, n multiples of 32; other padded shapes keep pack_trellis' layout.
    kernel_layout = m_p % 32 == 0 and n_p % 32 == 0
    buf_cols = ldlq_buf_cols(n_p)
    SU, SV = sign_vectors(n_p, m_p, seed)

    with torch.no_grad():
        H = hessian.detach().to(device=device, dtype=torch.float32)  # not modified below
        dm = torch.diag(H).mean()
        h_fallback = None
        if not bool(torch.isfinite(H).all()):
            raise ValueError('hessian has non-finite entries')
        if not float(dm) > 0:
            # Official regularize_H divides by mean(diag(H)) and would produce NaN; use H = I (plain MSE).
            H = torch.eye(n, device=device)
            dm = torch.ones((), device=device)
            h_fallback = 'identity (mean diag(H) == 0)'
        W_p = torch.zeros(m_p, n_p, device=device, dtype=torch.float32)
        W_p[:m, :n] = weight.detach().to(device=device, dtype=torch.float32)

        attempts = [(torch.float32, SIGMA_REG), (torch.float64, SIGMA_REG),
                    (torch.float64, 10 * SIGMA_REG), (torch.float64, 100 * SIGMA_REG)]
        saved = extras = None
        tried = []
        for dtype_, sigma in attempts:
            if n_p == n:
                H_in, sigma_in = H.clone(), sigma  # exactly the official call
            else:
                # Padded input channels behave like official dead channels: regularize_H on the true
                # n x n block (official diagmean), padded diagonal = sigma * diagmean, then sigma_reg = 0.
                H_in = torch.zeros(n_p, n_p, device=device, dtype=torch.float32)
                H_in[:n, :n] = regularize_H(H.clone(), sigma)
                idx = torch.arange(n, n_p, device=device)
                H_in[idx, idx] = sigma * dm
                sigma_in = 0.0
            args = types.SimpleNamespace(sigma_reg=sigma_in, scale_override=SCALE_OVERRIDE,
                                         td_x=TD_X, td_y=TD_Y, K=K, V=V)
            cb = codebook(K, device)
            try:
                saved, extras = official_quantize_linear(
                    W_p.to(dtype_), H_in.to(dtype_), SU.to(device, dtype_), SV.to(device, dtype_), cb, args,
                    device, MODEL_DTYPE, has_kernel=kernel_layout, for_kernel=True, buf_cols=buf_cols)
                tried.append(dict(dtype=str(dtype_), sigma_reg=sigma, ok=True))
                break
            except TypeError as exc:  # block_LDL returned None (Cholesky failed): official would crash here
                if 'NoneType' not in str(exc):
                    raise
                tried.append(dict(dtype=str(dtype_), sigma_reg=sigma, ok=False))
                saved = extras = None
        if saved is None:
            raise RuntimeError(f'QTIP block_LDL failed for every attempt: {tried}')

        hatWr, cb = extras['hatWr'], extras['cb']
        del extras
        Wscale = saved['Wscale']
        verified = None
        if VERIFY_DECODE:
            decoded = decode_packed(saved['trellis'], saved['tlut'], cb, K, m_p, n_p, kernel_layout)
            verified = bool(torch.equal(decoded.to(device, hatWr.dtype) * Wscale, hatWr))
            if not verified:
                raise RuntimeError('QTIP: decoded trellis does not reproduce the LDLQ output')
            del decoded
        SU_b = saved['SU'].to(device, hatWr.dtype)
        SV_b = saved['SV'].to(device, hatWr.dtype)
        dequant = inverse_incoherence(hatWr, SU_b, SV_b)[:m, :n].float().contiguous()
        del hatWr

    trellis_bits = saved['trellis'].numel() * 16
    su_bits = n_p * SU_BITS
    sv_bits = m_p * SV_BITS
    bits_total = float(trellis_bits + su_bits + sv_bits)
    assert bits_total == storage_bits(m, n, K)
    had_in, had_out = get_hadK(n_p)[1], get_hadK(m_p)[1]
    config = dict(
        method='QTIP', upstream=UPSTREAM, upstream_commit=UPSTREAM_COMMIT,
        code='HYB', decode_mode=DECODE_MODE, L=L, K=K, V=V, tlut_bits=TLUT_BITS, td_x=TD_X, td_y=TD_Y,
        lut='official quantlut_sym k-means LUT (torch seed 0), normalized to std 0.9682458365518543, '
            'rounded to the model dtype; global, not tuned',
        lut_dtype=str(MODEL_DTYPE), viterbi_input_dtype='torch.float16',
        incoherence='RHT both sides: Wr = Had_m^T diag(SV) W diag(SU) Had_n',
        hadamard_K_in=had_in, hadamard_K_out=had_out,
        rounding='BlockLDLQ (block 16) + tail-biting Viterbi', for_kernel_permute=True,
        kernel_bit_layout=kernel_layout, ldlq_buf_cols=buf_cols,
        sigma_reg=SIGMA_REG, scale_override=SCALE_OVERRIDE, attempts=tried, hessian_fallback=h_fallback,
        fine_tuning='none', seed=int(seed), sign_rng='torch.Generator(cpu).manual_seed(seed): SU then SV',
        shape=[m, n], padded_shape=[m_p, n_p],
        bits=dict(trellis=trellis_bits, SU=su_bits, SV_fp32_with_Wscale=sv_bits,
                  shared_tlut_not_counted=SHARED_TLUT_BITS),
        bpw=bits_total / (m * n), Wscale=float(Wscale),
        proxy_err_official=float(saved['proxy_err']), decode_verified=verified,
        viterbi_update=getattr(cb.update, 'mode', 'official'),
        seconds=time.monotonic() - t0,
    )
    return dict(dequant=dequant.to(device=weight.device, dtype=torch.float32), bits_total=bits_total,
                config=config)

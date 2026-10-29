"""QuIP# (Tseng et al., ICML 2024) one-shot weight quantizer for one Linear layer, wrapping the vendored official code.

Official code: https://github.com/Cornell-RelaxML/quip-sharp @ 1d8f873e9a2a8b86b12bb1064c312c5689b77d98 (GPL-3.0),
vendored in quipsharp_official.py / quipsharp_hadamard.py. Settings are the upstream defaults
(quantize_finetune_llama.py argparse + README advice): randomized Hadamard incoherence on both sides, block LDL with
block = codebook dimension 8, buffered LDLQ, 10 quip_tune_iters, sigma_reg 1e-2, no low-rank, no W/H rescaling.
2 bit = E8P12 with scale_override 0.9 (README / example.sh); 3 bit = E8P12RVQ3B with its default opt_scale 0.98 and
resid scale 2.04. No fine-tuning of any kind (one-shot PTQ comparison). See UPSTREAM.md for every deviation.

`dequant` is decoded from the stored representation exactly as the official QuantizedLinear does
(packed Qidxs -> codebook values -> inverse RHT with the stored FP16 SU and SV, SV carrying Wscale and the per-layer
RMS scale), then cropped to the original shape, so y = x @ dequant.T is the QuIP# layer output.
"""
from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import os
import sys
import time
from types import SimpleNamespace

import torch

UPSTREAM = "https://github.com/Cornell-RelaxML/quip-sharp"
UPSTREAM_COMMIT = "1d8f873e9a2a8b86b12bb1064c312c5689b77d98"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _sibling(name):
    """Load a vendored module next to this file under its own (unique) name, whatever the caller's sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


hadamard = _sibling("quipsharp_hadamard")
official = _sibling("quipsharp_official")

# Upstream quantize_finetune_llama.py argparse defaults (fine-tuning flags omitted: ft_epochs is not used here).
COMMON_ARGS = dict(
    sigma_reg=1e-2,
    sigma_reg2=1e-2,
    incoh_mode="had",
    lora_rank=0,
    quip_tune_iters=10,
    use_fp64=False,
    full_svd=False,
    no_use_buffered=False,
    rescale_WH=False,
    lowmem_ldlq=False,
)
# README: "We suggest using 0.9 for E8P12 and the default scales for 3 and 4 bit models."
BIT_ARGS = {
    2: dict(codebook="E8P12", scale_override=0.9, resid_scale_override=-1),
    3: dict(codebook="E8P12RVQ3B", scale_override=-1, resid_scale_override=-1),
}
# Official storage/kernel layout: Qidxs.view(m // 16, n // 64, 8, 4) (QuantizedE8P12Linear / RVQ3B), and both
# sides need a Hadamard size get_hadK accepts (K * 2^j).
OUT_MULTIPLE = 16
IN_MULTIPLE = 64
# The other persistent buffers of the official QuantizedLinear state dict (had_left_T/had_right are non-persistent
# constants): 'Wscale' (fp32 scalar, 1.0 once Wscale is fused into SV) and 'codebook_id' (int64 scalar).
WSCALE_BITS = 32
CODEBOOK_ID_BITS = 64
MAX_SIGMA_RETRIES = 3  # only used if the official fp32 -> fp64 fallback still raises (never seen for a PSD H)

_CODEBOOKS = {}


def _codebook(bits):
    if bits not in _CODEBOOKS:
        cls = official.E8P12_codebook if bits == 2 else official.E8P12RVQ3B_codebook
        _CODEBOOKS[bits] = cls()
    return _CODEBOOKS[bits]


def padded_dim(n, multiple):
    """Smallest N >= n with N % multiple == 0 and a vendored Hadamard size (K * 2^j)."""
    size = -(-max(n, 1) // multiple) * multiple
    while not hadamard.hadK_supported(size):
        size += multiple
    return size


def make_args(bits, in_padded):
    args = SimpleNamespace(**COMMON_ARGS, **{k: v for k, v in BIT_ARGS[bits].items() if k != "codebook"})
    args.buf_cols = 128 if in_padded % 128 == 0 else math.gcd(in_padded, 128)
    return args


@contextlib.contextmanager
def strict_fp32():
    """The official quantizer runs FP32 matmuls; keep TF32 off even if the caller enabled it globally."""
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@contextlib.contextmanager
def capture_official_log():
    logger = official.glog
    handler = _Collect()
    previous = (logger.level, logger.propagate)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield handler.lines
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous[0])
        logger.propagate = previous[1]


def symmetric_from_lower(H):
    """Upstream stores Hessians as their lower triangle (sym_to_flat) and rebuilds them with flat_to_sym."""
    out = torch.tril(H)
    out += torch.tril(H, -1).T
    return out


def run_official(W_pad, H_pad, bits, seed, args):
    """Vendored quip.py::quantize_linear body on padded inputs; SU/SV drawn after torch.manual_seed(seed) on the
    weight's device, with the caller's RNG state restored afterwards. Returns (hatW, attr, log lines)."""
    device = W_pad.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with strict_fp32(), torch.random.fork_rng(devices=devices), capture_official_log() as lines:
        torch.manual_seed(seed)
        hatW, attr = official.quantize_linear_inmem([W_pad], H_pad, _codebook(bits), args, device)
    return hatW, attr, lines


# --- decoding the official packed format (pure-torch replacement of quiptools_cuda.decompress_packed_e8p /
# --- decompress_e81b_packed): exact inverses of E8P12_codebook.maybe_pack_idxs and RVQ3B pack_e81b.
def unpack_e8p(packed, m, codes):
    """(m, codes/4) int64 -> (m, codes) E8P indices ((abs << 8) | signs)."""
    out = packed.view(m // 16, codes // 8, 8, 4).transpose(1, 2).reshape(m // 2, codes // 2)
    abs32 = out & 0xFFFFFFFF
    sign32 = (out >> 32) & 0xFFFFFFFF
    idxs = torch.empty(m // 2, codes // 2, 2, 2, dtype=torch.int64, device=packed.device)
    for i in range(4):
        signs = torch.zeros_like(abs32)
        for j in range(8):
            signs |= ((sign32 >> (4 * j + i)) & 1) << j
        idxs[:, :, i % 2, i // 2] = (((abs32 >> (8 * i)) & 0xFF) << 8) | signs
    return idxs.transpose(1, 2).reshape(m, codes)


def unpack_e81b(packed, m, codes):
    """(m, codes/8) int64 -> (m, codes) E8 1-bit residual indices."""
    idxs = torch.empty(m, codes, dtype=torch.int64, device=packed.device)
    for i in range(8):
        idxs[:, i::8] = (packed >> (8 * i)) & 0xFF
    return idxs


def unpack_qidxs(Qidxs, bits, m, n):
    """Packed Qidxs -> the unpacked LDLQ index matrix (m, n/8) exactly as LDLQ produced it."""
    codes = n // 8
    if bits == 2:
        return unpack_e8p(Qidxs, m, codes)
    init = unpack_e8p(Qidxs[:, :codes // 4].contiguous(), m, codes)
    resid = unpack_e81b(Qidxs[:, codes // 4:].contiguous(), m, codes)
    return (init << 16) + resid


def decode_values(idxs, bits, args):
    """Unpacked indices -> codebook values (m, n) in the incoherent, Wscale-normalized space (by_idxs semantics)."""
    cb = _codebook(bits)
    grid = cb.grid.to(idxs.device)
    if bits == 2:
        vals = grid[idxs]
    else:
        resid_scale = args.resid_scale_override if args.resid_scale_override > 0 else cb.opt_resid_scale
        vals = grid[idxs >> 16] + cb.e81b_grid.to(device=idxs.device, dtype=grid.dtype)[idxs & 0xFFFF] / resid_scale
    return vals.reshape(idxs.shape[0], -1)


def decode_stored(attr, bits, m_pad, n_pad, args):
    """Stored (Qidxs, SU, SV) -> W_hat in original (padded) coordinates, like QuantizedLinear's cached weight."""
    vals = decode_values(unpack_qidxs(attr["Qidxs"], bits, m_pad, n_pad), bits, args)
    return official.incoherence_process(vals.float(), attr["SU"].float(), attr["SV"].float(), None, args)


def format_bits(bits, m_pad, n_pad):
    """Closed form of stored_bits: bits*M*N code bits + FP16 SU (N) and SV (M) + Wscale + codebook_id."""
    codes = bits * m_pad * n_pad
    total = codes + 16 * n_pad + 16 * m_pad + WSCALE_BITS + CODEBOOK_ID_BITS
    return dict(codes=codes, SU=16 * n_pad, SV=16 * m_pad, Wscale=WSCALE_BITS, codebook_id=CODEBOOK_ID_BITS,
                total=total)


def stored_bits(attr):
    """Bits of every persistent buffer the official QuantizedLinear stores for this layer."""
    codes = attr["Qidxs"].numel() * attr["Qidxs"].element_size() * 8
    su = attr["SU"].numel() * attr["SU"].element_size() * 8
    sv = attr["SV"].numel() * attr["SV"].element_size() * 8
    assert attr["A"] is None and attr["B"] is None and attr["scaleWH"] is None
    total = codes + su + sv + WSCALE_BITS + CODEBOOK_ID_BITS
    return dict(codes=codes, SU=su, SV=sv, Wscale=WSCALE_BITS, codebook_id=CODEBOOK_ID_BITS, total=total)


def _quantize(weight, hessian, bits, seed=0):
    if bits not in BIT_ARGS:
        raise ValueError(f"QuIP# port supports bits 2 (E8P12) and 3 (E8P12RVQ3B), got {bits}")
    if weight.dim() != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError(f"shape mismatch: weight {tuple(weight.shape)}, hessian {tuple(hessian.shape)}")
    started = time.perf_counter()
    device = weight.device
    m, n = weight.shape
    W = weight.detach().to(torch.float32)
    H = symmetric_from_lower(hessian.detach().to(device=device, dtype=torch.float32))
    if not (torch.isfinite(W).all() and torch.isfinite(H).all()):
        raise ValueError("non-finite weight or Hessian")
    zero_hessian = not bool(torch.diag(H).sum() > 0)
    if zero_hessian:  # regularize_H would divide by 0; an all-zero H carries no weighting, so use the identity
        H = torch.eye(n, device=device)

    m_pad, n_pad = padded_dim(m, OUT_MULTIPLE), padded_dim(n, IN_MULTIPLE)
    if not bool(W.any()):
        # Upstream would divide by RMS(W) = 0. An all-zero layer is stored exactly by any codes with SV = 0.
        detail = format_bits(bits, m_pad, n_pad)
        config = dict(method="quipsharp", bits=bits, codebook=BIT_ARGS[bits]["codebook"], seed=seed, shape=[m, n],
                      padded_shape=[m_pad, n_pad], all_zero_weight=True, bits_detail=detail,
                      bpw=detail["total"] / (m * n), proxy_rel_err=0.0, upstream_commit=UPSTREAM_COMMIT,
                      fine_tuning="none (no within-block or end-to-end fine-tuning; one-shot PTQ)")
        return dict(dequant=torch.zeros(m, n, device=device), bits_total=float(detail["total"]), config=config)
    W_pad, H_pad = W, H
    if (m_pad, n_pad) != (m, n):
        W_pad = torch.zeros(m_pad, n_pad, device=device)
        W_pad[:m, :n] = W
        H_pad = torch.zeros(n_pad, n_pad, device=device)
        H_pad[:n, :n] = H  # padded inputs behave as dead channels (zero activations)
    args = make_args(bits, n_pad)

    retries = 0
    while True:
        try:
            hatW, attr, log = run_official(W_pad, H_pad.clone(), bits, seed, args)
            break
        except Exception as error:
            # Upstream signals "fp64 retry failed too" with a bare `raise Exception`; anything else (OOM, bugs)
            # propagates immediately.
            official_failure = type(error) is Exception or isinstance(error, torch.linalg.LinAlgError)
            if not official_failure or retries >= MAX_SIGMA_RETRIES:
                raise
            retries += 1
            args.sigma_reg *= 10
    # The official fp32 -> fp64 retry (LDLQ_buffered_lowmem in fp64) only announces itself through its log line.
    fp64_fallback = any("recomputing in fp64" in line for line in log)

    with strict_fp32():
        full = decode_stored(attr, bits, m_pad, n_pad, args)
        dequant = full[:m, :n].contiguous()
        delta = dequant - W
        proxy = float(((delta @ H) * delta).sum() / ((W @ H) * W).sum().clamp_min(1e-30))
    bits_detail = stored_bits(attr)
    assert bits_detail == format_bits(bits, m_pad, n_pad), bits_detail
    cb = _codebook(bits)
    divisor = args.scale_override if args.scale_override > 0 else cb.opt_scale
    rms = W_pad.square().mean().sqrt()
    # |SV| is one value per layer (sign * Wscale * RMS(W)); its FP16 storage scales dequant by this factor
    sv_factor = float(attr["SV"].float().abs().mean() / ((W_pad / rms).square().mean().sqrt() / divisor * rms))
    config = dict(
        method="quipsharp",
        upstream=UPSTREAM,
        upstream_commit=UPSTREAM_COMMIT,
        bits=bits,
        codebook=BIT_ARGS[bits]["codebook"],
        codesz=cb.codesz,
        ldl_block=cb.codesz,
        incoherence="randomized Hadamard both sides (incoh_mode=had)",
        args=dict(vars(args)),
        wscale_divisor=divisor,
        sv_fp16_factor=sv_factor,
        resid_scale=(args.resid_scale_override if args.resid_scale_override > 0 else cb.opt_resid_scale)
        if bits == 3 else None,
        fine_tuning="none (no within-block or end-to-end fine-tuning; one-shot PTQ)",
        seed=seed,
        shape=[m, n],
        padded_shape=[m_pad, n_pad],
        hadamard_K=dict(out=hadamard.get_hadK(m_pad)[1], inp=hadamard.get_hadK(n_pad)[1]),
        hessian="lower triangle mirrored, zero-padded, then official regularize_H (divide by mean diag, + sigma_reg)",
        zero_hessian_replaced_by_identity=zero_hessian,
        sigma_retries=retries,
        fp64_fallback=fp64_fallback,
        bits_detail=bits_detail,
        bpw=bits_detail["total"] / (m * n),
        proxy_rel_err=proxy,
        official_log=log,
        seconds=time.perf_counter() - started,
    )
    return dict(dequant=dequant, bits_total=float(bits_detail["total"]), config=config,
                hatW_official=hatW, attr=attr, W_pad=W_pad, args=args)


def quantize_layer(weight, hessian, bits, seed=0) -> dict:
    """weight [out, in] fp32, hessian [in, in] = mean x x^T (original coordinates), bits 2 or 3.

    Returns dict(dequant=float32 [out, in] in original coordinates, bits_total=float, config=dict)."""
    out = _quantize(weight, hessian, bits, seed)
    return dict(dequant=out["dequant"], bits_total=out["bits_total"], config=out["config"])

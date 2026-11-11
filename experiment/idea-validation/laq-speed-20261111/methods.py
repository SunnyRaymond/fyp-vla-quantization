"""The four published recipes on the block-level Linears of a WAM: fitting (prep) and the forward path (loop).

Adaptation boundary: AdaLN and per-dimension modulation block a global residual rotation, so every rotation is applied
locally at the input of each Linear and compensated in its own weight (x' = x R, W' = W R). Everything else follows the
official code: see README.zh.md.
"""
from __future__ import annotations

from collections import namedtuple
import re
import zlib

import torch
import torch.nn.functional as F

import qops as q

ROTATED = ("quarot", "spinquant")
SVD_SHIFT = 0.171875  # DeepCompressor's GELU lower bound
Role = namedtuple("Role", "expert block attn kind")
_FW = re.compile(r"^(video_expert|action_expert)\.blocks\.(\d+)\.(?:(self_attn|cross_attn)\.(q|k|v|o)|ffn\.(0|2))$")
_CO = re.compile(r"^net\.blocks\.(\d+)\._checkpoint_wrapped_module\.(?:(self_attn|cross_attn)\.(q_proj|k_proj|v_proj|"
                 r"output_proj)|mlp\.layer(1|2))$")
_KINDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "output_proj": "o"}


def parse(name):
    """Role of a block-level target Linear, or None for an interface or time-conditioning layer (kept in BF16)."""
    m = _FW.match(name)
    if m:
        expert, block, attn, kind, ffn = m.groups()
        expert = expert.split("_")[0]
        return Role(expert, int(block), attn.split("_")[0], kind) if attn else Role(expert, int(block), None,
                                                                                    "mlp0" if ffn == "0" else "mlp2")
    m = _CO.match(name)
    if m:
        block, attn, kind, mlp = m.groups()
        return Role("net", int(block), attn.split("_")[0], _KINDS[kind]) if attn else Role("net", int(block), None,
                                                                                         "mlp0" if mlp == "1" else "mlp2")
    return None


def stream_of(r):
    return f"{r.expert}.context" if r.attn == "cross" and r.kind in ("k", "v") else f"{r.expert}.hidden"


def attn_key(r):
    return f"{r.expert}.{r.block}.{r.attn}"


def seed_of(base, tag):
    return int(base) + zlib.crc32(tag.encode()) % (1 << 31)


class Had:
    """x -> x diag(signs) H (signs optional)."""

    def __init__(self, signs=None):
        self.s = signs

    def apply(self, x):
        return q.had(x if self.s is None else x * self.s)

    def inv(self, y):
        z = q.had(y, inverse=True)
        return z if self.s is None else z * self.s


class Dense:
    def __init__(self, r):
        self.r = r

    def apply(self, x):
        return x @ self.r

    def inv(self, y):
        return y @ self.r.t()


class HeadDense:
    """Per-head rotation R2 of the attention output (SpinQuant)."""

    def __init__(self, heads, head_dim, r):
        self.h, self.d, self.r = heads, head_dim, r

    def apply(self, x):
        return (x.reshape(*x.shape[:-1], self.h, self.d) @ self.r).reshape(x.shape)

    def inv(self, y):
        return (y.reshape(*y.shape[:-1], self.h, self.d) @ self.r.t()).reshape(y.shape)


class QuarotO:
    """QuaRot o_proj input: head-dim Hadamard, then Hadamard across heads."""

    def __init__(self, heads, head_dim):
        self.h, self.d = heads, head_dim

    def apply(self, x):
        y = q.had(x.reshape(*x.shape[:-1], self.h, self.d))
        return q.had(y.transpose(-1, -2)).transpose(-1, -2).reshape(x.shape)

    def inv(self, y):
        z = q.had(y.reshape(*y.shape[:-1], self.h, self.d).transpose(-1, -2), inverse=True).transpose(-1, -2)
        return q.had(z, inverse=True).reshape(y.shape)


def sandwich(xf, h):
    return xf.apply(xf.apply(h).t()).t()


def xf_for(arm, role, in_features, arch, seed, rotations, device):
    if role.kind == "mlp2":
        # The official R2/R4 is a plain Hadamard, made for SwiGLU outputs. A GELU output has a positive mean, which a plain
        # Hadamard concentrates in one coordinate; the randomized form (random signs, as R1 uses) removes it.
        return Had(q.random_signs(in_features, seed_of(seed, f"{role.expert}.{role.block}.mlp2"), device))
    if role.kind == "o":
        return QuarotO(arch["heads"], arch["head_dim"]) if arm == "quarot" else HeadDense(
            arch["heads"], arch["head_dim"], rotations[attn_key(role)])
    stream = stream_of(role)
    if arm == "quarot":
        return Had(q.random_signs(in_features, seed_of(seed, stream), device))
    return Dense(rotations[stream])


def actq_for(arm, role, arch):
    if arm == "quarot":
        return lambda v: q.a_token(v, sym=True, clip=0.9)
    if role.kind == "o":
        return lambda v: q.a_group(v, arch["head_dim"])
    return lambda v: q.a_token(v, sym=False)


# ------------------------------------------------------------------ fitting


def fit_rotated(arm, w, h, xf):
    """QuaRot / SpinQuant: per-row MSE-clipped GPTQ in rotated coordinates."""
    wc, hc = xf.apply(w), sandwich(xf, h)
    codes, scales = q.gptq_row(wc, hc)
    bank = q.encode_sym(codes, scales[:, None], w.shape[1])
    return bank, q.proxy_nmse(q.decode_sym(bank, torch.float32), wc, hc)


def fit_viditq(w, amax, bits, alpha, signs):
    """Channel balance (weight max^alpha / act max^(1-alpha)), then the two-stage affine weight quantization."""
    wmax = w.abs().amax(0).clamp_min(1e-8)
    mask = wmax.pow(alpha) / amax.to(w.device, torch.float32).clamp_min(1e-8).pow(1 - alpha)
    first = q.vid_dequant(*q.vid_quant(w / mask, bits))
    codes, delta, zero = q.vid_quant(q.had(first * signs), bits)
    return dict(q.encode_vid(codes, delta, zero, bits), mask=mask, signs=signs.to(torch.int8)), None


def _low_rank(m, rank, seed):
    """Truncated SVD of m; returns (b [r, rank] = U S, a [rank, c] = V^T) as in LowRankBranch."""
    retained = min(rank, *m.shape)
    with torch.random.fork_rng(devices=[m.device.index or 0] if m.device.type == "cuda" else []):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(m, q=min(retained + 8, *m.shape), niter=4)
    return u[:, :retained] * s[:retained], v[:, :retained].t()


def fit_svdquant(weights, x, unsigned, seed, cfg):
    """Joint fit of the Linears that share one input (q/k/v of a self-attention, k/v of a cross-attention, or one Linear).

    weights: list of [rows, in] FP32 on the GPU; x: calibration inputs [samples, in] FP32 (shifted already if unsigned).
    Per-layer grid search over the smoothing pair, then iterative calibration of the rank-32 branch against OutputsError.
    Returns (scale s, shared a [rank, in], list of (rows, codes, scales, b)).
    """
    rank, grids, iters = cfg["rank"], cfg["num_grids"], cfg["num_iters"]
    wcat = torch.cat(weights, 0)
    rows = [w.shape[0] for w in weights]
    xmax, wmax = x.abs().amax(0), wcat.abs().amax(0)
    choices = [i / grids for i in range(1, grids)]
    pairs = [(0.0, 0.0)] + [(a, 0.0) for a in choices] + [(a, 1 - a) for a in choices]
    ytrue = x @ wcat.t()

    def scale_for(alpha, beta):
        if alpha == 0 and beta == 0:
            return torch.ones_like(xmax)
        s = xmax.pow(alpha) if alpha > 0 else torch.ones_like(xmax)
        if beta > 0:
            s = s / wmax.pow(beta)
        s = torch.where(s == 0, torch.ones_like(s), s)
        if s.isnan().any() or s.isinf().any():
            s = torch.ones_like(xmax)
        return s

    def output_error(s, res_q, low):
        xt = x / s
        y = q.a_int4_group(xt, 64, unsigned) @ res_q.t() + (xt @ low.t() if low is not None else 0)
        return float((y - ytrue).square().sum())

    def quant(m):
        c, sc = q.rtn_int4(m, 64)
        return c, sc, c.float() * sc.repeat_interleave(64, 1)[:, :m.shape[1]]

    best, best_s = None, None
    for alpha, beta in pairs:
        s = scale_for(alpha, beta)
        t = wcat * s
        b, a = _low_rank(t, rank, seed)
        low = b @ a
        err = output_error(s, quant(t - low)[2], low)
        if best is None or err < best:
            best, best_s = err, s
    t = wcat * best_s
    # QuantLowRankCalibrator: branch = SVD(w - qw); qw = Q(w - branch); keep the best, stop early on no improvement.
    qw = torch.zeros_like(t)
    best_err, branch = None, None
    for it in range(iters):
        b, a = _low_rank(t - qw, rank, seed + it)
        low = b @ a
        _, _, res_q = quant(t - low)
        err = output_error(best_s, res_q, low)
        if best_err is None or err <= best_err:
            best_err, branch = err, (b, a)
        else:
            break
        qw = res_q
    b, a = branch
    bb, aa = b.to(torch.bfloat16), a.to(torch.bfloat16)
    low_bf = bb.float() @ aa.float()
    codes, scales, _ = quant(t - low_bf)
    out, start = [], 0
    for n in rows:
        out.append((n, codes[start:start + n], scales[start:start + n], bb[start:start + n]))
        start += n
    return best_s, aa, out, dict(best_alpha=None, objective=best_err)


# ------------------------------------------------------------------ forwards


def forward_for(arm, role, module, bank, xf, actq):
    """The fake-quantized forward of one Linear; module.weight already holds the decoded weight."""
    if arm in ROTATED:
        def run(x):
            return F.linear(actq(xf.apply(x.float())).to(x.dtype), module.weight, module.bias)
        return run
    if arm == "svdquant":
        s, up, down = bank["smoothing"], bank["up"], bank["down"]
        unsigned = role.kind == "mlp2"
        shift = SVD_SHIFT if unsigned else 0.0
        extra = bank["bias_shift"]

        def run(x):
            t = (x.float() + shift) / s
            y = F.linear(q.a_int4_group(t, 64, unsigned).to(x.dtype), module.weight, module.bias)
            y = y + F.linear(F.linear(t.to(x.dtype), down), up)
            return y if extra is None else y - extra.to(y.dtype)
        return run
    mask, signs = bank["mask"], bank["signs"].float()

    def run(x):
        v = q.had(x.float() * mask * signs)
        return F.linear(q.a_token(v, sym=True, clip=1.0, bits=8).to(x.dtype), module.weight, module.bias)
    return run


def install_arm(modules, index_of, bank_dir, arm, cfg, arch, device, rotations=None):
    """Decode every target Linear of `arm` and replace its forward. Returns the set of Linears a query reaches."""
    seen = set()
    for name, module in modules.items():
        role = parse(name)
        if role is None:
            continue
        bank = torch.load(bank_dir / f"{index_of[name]:04d}.pt", map_location=device, weights_only=True)
        if bank["module"] != name or tuple(bank["shape"]) != tuple(module.weight.shape):
            raise RuntimeError(f"Bank of {arm} does not match {name}")
        with torch.no_grad():
            module.weight.copy_(q.decode(bank).to(module.weight.dtype))
        xf = xf_for(arm, role, module.in_features, arch, cfg["rotation_seed"], rotations, device) if arm in ROTATED else None
        actq = actq_for(arm, role, arch) if arm in ROTATED else None
        run = forward_for(arm, role, module, bank, xf, actq)

        def counted(x, _run=run, _name=name):
            seen.add(_name)
            return _run(x)
        module.forward = counted
    return seen

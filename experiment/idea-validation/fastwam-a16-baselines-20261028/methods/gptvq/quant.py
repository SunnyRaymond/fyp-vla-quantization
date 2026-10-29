"""GPTVQ (van Baalen et al. 2024) per-layer weight quantizer for the Fast-WAM A16 baselines.

Wraps the official per-layer routine vendored in ./_upstream (qualcomm-ai-research/gptvq @ fd8a3ce):
GPTQ.fasterquant(use_vq=True) -> GPTQ.lut_m_step, with VQQuantizer configured as in the official experiment
scripts (test_bigtable.sh): --columns-per-group 256 --kmeans-iters 100 --kmeans-init-method mahalanobis
--hessian-weighted-lookups --include-m-step --codebook-bitwidth 8, llama.py defaults percdamp 0.01, no act-order,
fasterquant blocksize 128. The codebook group is twice the paper's 0.125-bpv group so that the side information
fits the 0.0625-bpv budget; codebooks are int8 with one FP32 scale per 256-column span (the code's default,
--quantize-per-codebook off). UPSTREAM.md lists every deviation.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import math
import re
import sys
import time
from pathlib import Path

import torch

UPSTREAM_REPO = "https://github.com/qualcomm-ai-research/gptvq"
UPSTREAM_COMMIT = "fd8a3ced81f767018a28a51a91af96777f885f9f"

# index bits per weight -> VQ dimension and codebook rows. A codebook has 2**(bits * vq_dim) centroids and is
# shared by a (rows_per_group x COLUMNS_PER_GROUP) block of the weight.
CONFIGS = {
    2: dict(vq_dim=4, rows_per_group=512),  # paper 2.125 bpv: 4D, 256x256 groups -> here 512x256 (131072 weights)
    3: dict(vq_dim=2, rows_per_group=64),   # paper 3.125 bpv: 2D, 32x256 groups -> here 64x256 (16384 weights)
}
COLUMNS_PER_GROUP = 256
KMEANS_ITERS = 100
KMEANS_INIT = "mahalanobis"
CODEBOOK_BITWIDTH = 8
QUANTIZE_PER_CODEBOOK = False   # False: one scale per column span (all codebooks of that span); True: per codebook
CODEBOOK_SCALE_BITS = 32        # the scale is the FP32 value quantize_centroids computes and dequantizes with
PERCDAMP = 0.01
PERCDAMP_FALLBACKS = (0.1, 1.0, 10.0)  # only used if the official damping leaves H not positive definite
BLOCKSIZE = 128
ASSIGNMENT_CHUNK_ELEMS = 1 << 27  # bound on G*chunk*K*D for the distance temporaries (512 MB each in FP32)

_PKG = "_gptvq_upstream_" + UPSTREAM_COMMIT[:7]


def _load_upstream():
    if _PKG not in sys.modules:
        root = Path(__file__).resolve().parent / "_upstream"
        spec = importlib.util.spec_from_file_location(_PKG, root / "__init__.py",
                                                      submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = package
        spec.loader.exec_module(package)
    return tuple(importlib.import_module(f"{_PKG}.{name}") for name in ("gptq", "vq_quant", "quant"))


gptq_mod, vq_mod, rtn_mod = _load_upstream()


def geometry(out_features, in_features, bits):
    """Group layout and the exact stored-bit count for one layer.

    Columns are zero-padded to a multiple of vq_dim (padded columns are dead inputs; their index bits are stored
    because they share vectors with real columns). Column spans are COLUMNS_PER_GROUP wide, the last one may be
    narrower. Rows are split into equal row groups of at most rows_per_group rows: an exact divisor split if one
    exists with at most twice the nominal group count, else ceil-split with < (#groups) zero rows appended
    (not stored). bits_total = out*in_pad*bits + n_codebooks*K*vq_dim*CODEBOOK_BITWIDTH + n_scales*CODEBOOK_SCALE_BITS.
    """
    cfg = CONFIGS[bits]
    d, rpg = cfg["vq_dim"], cfg["rows_per_group"]
    in_pad = -(-in_features // d) * d
    spans = -(-in_pad // COLUMNS_PER_GROUP)
    nominal = -(-out_features // rpg)
    rows = next((out_features // g for g in range(nominal, 2 * nominal + 1) if out_features % g == 0), None)
    if rows is None:
        rows = -(-out_features // nominal)
    groups = -(-out_features // rows)
    out_pad = groups * rows
    k = 2 ** (bits * d)
    codebooks = spans * groups
    scales = codebooks if QUANTIZE_PER_CODEBOOK else spans
    index_bits = out_features * in_pad * bits
    codebook_bits = codebooks * k * d * CODEBOOK_BITWIDTH
    scale_bits = scales * CODEBOOK_SCALE_BITS
    total = index_bits + codebook_bits + scale_bits
    return dict(vq_dim=d, n_centroids=k, rows_per_group=rows, columns_per_group=COLUMNS_PER_GROUP,
                groupsize=rows * COLUMNS_PER_GROUP, groups_per_column=groups, column_spans=spans,
                n_codebooks=codebooks, n_codebook_scales=scales, in_pad=in_pad, out_pad=out_pad,
                index_bits=index_bits, codebook_bits=codebook_bits, scale_bits=scale_bits, bits_total=total,
                bpw=total / (out_features * in_features),
                side_bpw=(codebook_bits + scale_bits) / (out_features * in_features),
                padding_bpw=(index_bits - out_features * in_features * bits) / (out_features * in_features))


def _mahalanobis_init_safe(X, n_centroids):
    """vq_quant.mahalanobis_init with Sigma^-1 replaced by pinv(Sigma) for the groups whose Sigma is singular."""
    vq_dim = X.shape[-1]
    mu = X.mean(1).unsqueeze(1)
    Xcentered = X - mu
    Sigma = torch.bmm(Xcentered.transpose(1, 2), Xcentered)  # G x D x D
    Lambda, info = torch.linalg.inv_ex(Sigma)
    bad = info != 0
    if bool(bad.any()):
        Lambda[bad] = torch.linalg.pinv(Sigma[bad])
    dists = (torch.bmm(Xcentered, Lambda) * Xcentered).sum(-1)  # G x N
    sorted_dists = torch.argsort(dists, dim=1)  # G x N
    idx = torch.round(torch.linspace(0, Xcentered.shape[1] - 1, n_centroids)).long()  # K
    idx = sorted_dists[:, idx].unsqueeze(-1).expand(-1, -1, vq_dim)
    return torch.gather(X, dim=1, index=idx)


class VQQuantizerPort(vq_mod.VQQuantizer):
    """Official VQQuantizer with two overrides that reduce to the official code on official-compatible inputs."""

    singular_inits = 0

    def get_groupsize(self, X, groupsize):
        if X.shape[1] % self.columns_per_group == 0:
            return super().get_groupsize(X, groupsize)
        # Official code asserts in_features % columns_per_group == 0; here the last column span is narrower.
        self.rows_per_group = groupsize // self.columns_per_group
        assert groupsize % self.columns_per_group == 0 and X.shape[0] % self.rows_per_group == 0
        self.groups_per_column = X.shape[0] // self.rows_per_group
        return self.columns_per_group

    def find_params(self, X, weight=True, H_inv_diag=None):
        try:
            return super().find_params(X, weight=weight, H_inv_diag=H_inv_diag)
        except torch.linalg.LinAlgError:  # torch.linalg.inv on a singular covariance in mahalanobis_init
            if self.kmeans_init_method != "mahalanobis":
                raise
        self.singular_inits += 1
        # Official find_params body; only the initialiser differs.
        X = X.reshape(self.groups_per_column, -1, self.vq_dim)
        if H_inv_diag is not None:
            H_inv_diag = H_inv_diag.reshape(1, -1, self.vq_dim)
            if self.rows_per_group > 1:
                H_inv_diag = H_inv_diag.tile(1, self.rows_per_group, 1)
        centroids = _mahalanobis_init_safe(X, self.n_centroids)
        extra_args = {}
        if self.quantize_during_kmeans and self.codebook_bitwidth is not None:
            extra_args = dict(codebook_bitwidth=self.codebook_bitwidth, per_codebook=self.quantize_per_codebook)
        vq_mod.kmeans_vq(X, centroids, iters=self.kmeans_iters, assignment_chunk_size=self.assignment_chunk_size,
                         H_inv_diag=H_inv_diag, **extra_args)
        if self.codebook_bitwidth is not None and not self.quantize_during_kmeans:
            vq_mod.quantize_centroids(centroids, self.codebook_bitwidth, per_codebook=self.quantize_per_codebook)
        self.all_centroids.append(centroids)


def make_quantizer(bits, assignment_chunk_size=None, port=True):
    """VQQuantizer built exactly like llama.py's QClass with the chosen settings."""
    cls = VQQuantizerPort if port else vq_mod.VQQuantizer
    quantizer = cls(
        vq_dim=CONFIGS[bits]["vq_dim"],
        columns_per_group=COLUMNS_PER_GROUP,
        vq_scaling_blocksize=-1,
        vq_scaling_norm="max",
        vq_scaling_n_bits=4,
        vq_scaling_domain="log",
        kmeans_init_method=KMEANS_INIT,
        assignment_chunk_size=assignment_chunk_size,
        kmeans_iters=KMEANS_ITERS,
        codebook_bitwidth=CODEBOOK_BITWIDTH,
        quantize_per_codebook=QUANTIZE_PER_CODEBOOK,
        quantize_during_kmeans=False,
        n_subsample=10000,
    )
    quantizer.configure(bits, perchannel=True, sym=False, mse=False)
    return quantizer


def fasterquant_kwargs(percdamp, groupsize, include_m_step):
    """llama.py -> GPTQ.fasterquant arguments for the official scripts' flags."""
    return dict(blocksize=BLOCKSIZE, percdamp=percdamp, groupsize=groupsize, actorder=False, static_groups=False,
                include_m_step=include_m_step, use_vq=True, svd_rank=None, hessian_weighted_lookups=True,
                only_init_kmeans=False)


def _choose_percdamp(H):
    """First damping (official 0.01 first) for which fasterquant's Cholesky chain succeeds on this H."""
    diag = torch.arange(H.shape[0], device=H.device)
    for percdamp in (PERCDAMP, *PERCDAMP_FALLBACKS):
        Hd = H.clone()
        dead = torch.diag(Hd) == 0
        Hd[dead, dead] = 1
        Hd[diag, diag] += percdamp * torch.mean(torch.diag(Hd))
        L, info = torch.linalg.cholesky_ex(Hd)
        del Hd
        if int(info) != 0:
            continue
        Hinv = torch.cholesky_inverse(L)
        del L
        U, info = torch.linalg.cholesky_ex(Hinv, upper=True)
        del Hinv
        if int(info) == 0 and bool(torch.isfinite(U).all()):
            return percdamp
    raise RuntimeError("Hessian is not positive definite even with percdamp=%g" % PERCDAMP_FALLBACKS[-1])


@contextlib.contextmanager
def _numerics(device, seed):
    """Official numerics (no TF32) and determinism for one call; global flags and RNG restored afterwards."""
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda" and not previous[2]:
        # gather-backward in lut_m_step uses atomics on CUDA; this selects the deterministic kernel.
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[:2]
        torch.use_deterministic_algorithms(previous[2], warn_only=previous[3])


def _parse_log(text):
    """The figures upstream prints: GPTQ-pass loss, M-step SNR (dB, tr(WGW^T)/tr(dW G dW^T)) and lr restarts."""
    error = re.findall(r"^error (\S+)$", text, re.M)
    gptq_time = re.findall(r"^time (\S+)$", text, re.M)
    snr = re.findall(r"^improvement: (\S+) -> (\S+)$", text, re.M)

    def num(s):  # JSON-safe (common.save uses allow_nan=False); non-finite only on zero-loss layers
        value = float(s)
        return value if math.isfinite(value) else None
    return dict(gptq_error=num(error[-1]) if error else None,
                gptq_pass_seconds=num(gptq_time[-1]) if gptq_time else None,
                m_step_snr_db=[num(snr[-1][0]), num(snr[-1][1])] if snr else None,
                m_step_restarts=len(re.findall(r"Restarting M-step", text)))


def quantize_layer(weight, hessian, bits, seed=0):
    """GPTVQ-quantize one Linear weight [out, in] given H = mean x x^T [in, in]; see module docstring."""
    if bits not in CONFIGS:
        raise ValueError(f"bits must be one of {sorted(CONFIGS)}, got {bits}")
    if weight.dim() != 2 or tuple(hessian.shape) != (weight.shape[1], weight.shape[1]):
        raise ValueError(f"weight {tuple(weight.shape)} and hessian {tuple(hessian.shape)} do not match")
    out_features, in_features = weight.shape
    device = weight.device
    geo = geometry(out_features, in_features, bits)
    start = time.perf_counter()
    log = io.StringIO()
    with torch.inference_mode(False), torch.no_grad(), _numerics(device, seed):
        W = torch.zeros(geo["out_pad"], geo["in_pad"], device=device, dtype=torch.float32)
        W[:out_features, :in_features] = weight.detach().to(torch.float32)
        H = torch.zeros(geo["in_pad"], geo["in_pad"], device=device, dtype=torch.float32)
        H[:in_features, :in_features] = hessian.detach().to(device=device, dtype=torch.float32)
        if not (bool(torch.isfinite(W).all()) and bool(torch.isfinite(H).all())):
            raise ValueError("weight and hessian must be finite")
        dead_inputs = int((torch.diag(H)[:in_features] == 0).sum())
        H.mul_(2)  # GPTQ.add_batch accumulates H = (2/n) sum x x^T; the caller passes (1/n) sum x x^T
        percdamp = _choose_percdamp(H)

        with torch.device("meta"):
            layer = torch.nn.Linear(geo["in_pad"], geo["out_pad"], bias=False)
        layer.weight = torch.nn.Parameter(W, requires_grad=False)
        gptq = gptq_mod.GPTQ(layer)
        gptq.H = H
        del H
        # Upstream get_assignments splits a 2-D H_inv_diag (the 1 x D weights vq_quantize passes) along D, which
        # breaks whenever chunk < N; vq_quantize has N = rows_per_group, so the chunk is never smaller than that.
        chunk = max(geo["rows_per_group"],
                    ASSIGNMENT_CHUNK_ELEMS // (geo["groups_per_column"] * geo["n_centroids"] * geo["vq_dim"]))
        quantizer = make_quantizer(bits, assignment_chunk_size=chunk)
        gptq.quantizer = quantizer
        with contextlib.redirect_stdout(log):
            gptq.fasterquant(**fasterquant_kwargs(percdamp, geo["groupsize"], include_m_step=True))
        gptq.free()
        dequant = layer.weight.data[:out_features, :in_features].contiguous()
        del layer, gptq, W
        if not bool(torch.isfinite(dequant).all()):
            raise RuntimeError("GPTVQ produced non-finite weights")
    seconds = time.perf_counter() - start

    config = dict(
        method="gptvq", upstream=UPSTREAM_REPO, upstream_commit=UPSTREAM_COMMIT,
        bits=bits, shape=[out_features, in_features], padded_shape=[geo["out_pad"], geo["in_pad"]],
        vq_dim=geo["vq_dim"], n_centroids=geo["n_centroids"], index_bits_per_vector=bits * geo["vq_dim"],
        columns_per_group=COLUMNS_PER_GROUP, rows_per_group=geo["rows_per_group"],
        rows_per_group_nominal=CONFIGS[bits]["rows_per_group"], groupsize=geo["groupsize"],
        groups_per_column=geo["groups_per_column"], column_spans=geo["column_spans"], n_codebooks=geo["n_codebooks"],
        codebook_bitwidth=CODEBOOK_BITWIDTH, quantize_per_codebook=QUANTIZE_PER_CODEBOOK,
        n_codebook_scales=geo["n_codebook_scales"], codebook_scale_bits=CODEBOOK_SCALE_BITS,
        kmeans_init_method=KMEANS_INIT, kmeans_iters=KMEANS_ITERS, hessian_weighted_lookups=True,
        include_m_step=True, quantize_during_kmeans=False, vq_scaling_blocksize=-1,
        svd_rank=None, act_order=False, blocksize=BLOCKSIZE, percdamp=percdamp, percdamp_official=PERCDAMP,
        hessian_scale="H_upstream = 2 * hessian (GPTQ.add_batch convention)",
        assignment_chunk_size=chunk, singular_init_fallbacks=quantizer.singular_inits, dead_inputs=dead_inputs,
        index_bits=geo["index_bits"], codebook_bits=geo["codebook_bits"], scale_bits=geo["scale_bits"],
        bpw=geo["bpw"], side_bpw=geo["side_bpw"], padding_bpw=geo["padding_bpw"],
        seed=seed, seed_note="Mahalanobis init is deterministic; the seed only fixes the RNG state",
        seconds=round(seconds, 3), device=str(device),
        **_parse_log(log.getvalue()),
    )
    return dict(dequant=dequant, bits_total=float(geo["bits_total"]), config=config)

"""CPU checks for the GPTVQ wrapper.

1. quantize_layer is bitwise equal to the vendored official routine driven exactly like llama.py
   (GPTQ.add_batch hooks -> VQQuantizer(QClass args) -> fasterquant(include_m_step=True, ...)).
2. Its Hessian-weighted error tr(dW H dW^T)/tr(W H W^T) is far below plain per-row RTN at the same bits.
3. Odd shapes (in=7, in=8, out=7, non-power-of-two dims, 1x1), dead and all-dead inputs, zero weights.
4. Determinism, chunked-assignment equivalence, inference_mode callers, restored global flags, bit accounting.

Run as `cd methods/gptvq && python -B test_gptvq.py` (CPU only; reads nothing outside this directory; the official
reference code is the vendored copy in ./_upstream). About 2.5 min on a busy 14-core laptop with torch 2.10.
"""
from __future__ import annotations

import contextlib
import io
import json
import time

import torch

import quant as Q

FASTWAM_SHAPES = [  # (out, in, count) from the task's Linear inventory
    (3072, 4096, 1), (3072, 3072, 242), (3072, 256, 1), (18432, 3072, 1), (14336, 3072, 30), (3072, 14336, 30),
    (192, 3072, 1), (1024, 4096, 31), (1024, 1024, 2), (1024, 256, 1), (6144, 1024, 1), (3072, 1024, 150),
    (1024, 3072, 60), (4096, 1024, 30), (1024, 7, 1), (7, 1024, 1), (4096, 8, 1),
]


def activations(n, k, g, dead=()):
    base = torch.randn(n, max(1, k // 4), generator=g)
    x = base @ torch.randn(max(1, k // 4), k, generator=g) + 0.1 * torch.randn(n, k, generator=g)
    x[:, :min(3, k)] *= 20  # a few outlier channels, as in real activations
    x[:, list(dead)] = 0
    return x


def weights(out, k, g):
    return torch.randn(out, k, generator=g) * 0.02 * torch.exp(0.5 * torch.randn(out, 1, generator=g))


def proxy(w, q, h):
    d = q - w
    return float(((d @ h) * d).sum() / ((w @ h) * w).sum())


def official(w, x, bits, groupsize):
    """Plain upstream GPTQ + VQQuantizer, called the way llama.py calls them."""
    layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    layer.weight.data = w.clone()
    gptq = Q.gptq_mod.GPTQ(layer)
    for chunk in x.split(512):
        # [tokens, 1, in]: add_batch counts inp.shape[0] as samples, so every token is one sample and
        # H_upstream = (2 / n_tokens) * sum x x^T. (llama.py feeds [1, 2048, in], i.e. 2048x larger H.)
        gptq.add_batch(chunk.unsqueeze(1), None)
    h_upstream = gptq.H.clone()
    gptq.quantizer = Q.make_quantizer(bits, port=False)
    with contextlib.redirect_stdout(io.StringIO()):
        gptq.fasterquant(**Q.fasterquant_kwargs(Q.PERCDAMP, groupsize, include_m_step=True))
    return layer.weight.data.clone(), h_upstream


def rtn(w, bits):
    """Per-row asymmetric min-max round-to-nearest (upstream quant.py Quantizer, llama.py --nearest)."""
    quantizer = Q.rtn_mod.Quantizer()
    quantizer.configure(bits, perchannel=True, sym=False, mse=False)
    quantizer.find_params(w, weight=True)
    return Q.rtn_mod.quantize(w, quantizer.scale, quantizer.zero, quantizer.maxq)


def check_storage(out, bits, w):
    """bits_total matches the formula, and every (row group x column span) block uses <= K distinct vectors."""
    q, cfg = out["dequant"], out["config"]
    rows, cols = w.shape
    geo = Q.geometry(rows, cols, bits)
    d, k = Q.CONFIGS[bits]["vq_dim"], 2 ** (bits * Q.CONFIGS[bits]["vq_dim"])
    in_pad, spans = -(-cols // d) * d, -(-(-(-cols // d) * d) // 256)
    scales = spans * (geo["groups_per_column"] if Q.QUANTIZE_PER_CODEBOOK else 1)
    expect = rows * in_pad * bits + spans * geo["groups_per_column"] * k * d * 8 + scales * 32
    assert out["bits_total"] == float(expect) == float(geo["bits_total"]), (out["bits_total"], expect)
    assert q.shape == w.shape and q.dtype == torch.float32 and q.device == w.device and torch.isfinite(q).all()
    json.dumps(cfg, allow_nan=False)  # the job saves configs with common.save (allow_nan=False)
    r, full = cfg["rows_per_group"], cols // d * d
    for g0 in range(0, rows, r):
        for c0 in range(0, in_pad, 256):
            block = q[g0:g0 + r, c0:min(c0 + 256, full)]  # vectors with no padded component
            if block.numel():
                assert torch.unique(block.reshape(-1, d), dim=0).shape[0] <= k
            if full < cols and c0 <= full < c0 + 256:  # the last vector's real prefix is a codeword prefix
                assert torch.unique(q[g0:g0 + r, full:cols], dim=0).shape[0] <= k


def main():
    start = time.perf_counter()
    g = torch.Generator().manual_seed(0)
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    print("== 1-2. bitwise equality with the official routine; error vs per-row RTN")
    for bits, rows, cols in ((2, 128, 256), (3, 128, 256), (2, 128, 512), (3, 256, 512)):
        w, x = weights(rows, cols, g), activations(4 * cols, cols, g, dead=(5,))
        geo = Q.geometry(rows, cols, bits)
        t = time.perf_counter()
        q_official, h_upstream = official(w, x, bits, geo["groupsize"])
        t_official = time.perf_counter() - t
        h_mean = x.t() @ x / x.shape[0]
        assert float((h_upstream / 2 - h_mean).norm() / h_mean.norm()) < 1e-5, "upstream H = 2 * mean x x^T"
        out = Q.quantize_layer(w, h_upstream / 2, bits)  # /2 is exact, so both routines see the same H
        assert torch.equal(out["dequant"], q_official), (bits, rows, cols, (out["dequant"] - q_official).abs().max())
        check_storage(out, bits, w)
        e_vq, e_rtn = proxy(w, out["dequant"], h_mean), proxy(w, rtn(w, bits), h_mean)
        assert e_vq < 0.25 * e_rtn, (bits, e_vq, e_rtn)
        c = out["config"]
        print(f"W{bits} {rows}x{cols}: bitwise_equal=True  rel_err gptvq={e_vq:.5f} rtn={e_rtn:.5f} "
              f"ratio={e_vq / e_rtn:.3f}  bpw={out['bits_total'] / w.numel():.5f}  groups={c['groups_per_column']}"
              f"x{c['column_spans']}  m_step_snr_db={c['m_step_snr_db']}  t_official={t_official:.1f}s "
              f"t_wrapper={c['seconds']:.1f}s")

    print("== 4. determinism, chunked assignment, inference_mode callers, flags")
    w, x = weights(32, 256, g), activations(1024, 256, g)
    h = x.t() @ x / x.shape[0]
    for bits in (2, 3):
        a, b = Q.quantize_layer(w, h, bits), Q.quantize_layer(w, h, bits, seed=0)
        assert torch.equal(a["dequant"], b["dequant"]), "same seed must give identical output"
        chunk = Q.ASSIGNMENT_CHUNK_ELEMS
        Q.ASSIGNMENT_CHUNK_ELEMS = 4096
        try:
            c = Q.quantize_layer(w, h, bits)
        finally:
            Q.ASSIGNMENT_CHUNK_ELEMS = chunk
        assert c["config"]["assignment_chunk_size"] * 4 <= 32 * 256 // Q.CONFIGS[bits]["vq_dim"]  # >= 4 chunks
        assert torch.equal(a["dequant"], c["dequant"]), "assignment chunking must not change the result"
        with torch.inference_mode():
            d = Q.quantize_layer(w.clone(), h.clone(), bits)
        assert torch.equal(a["dequant"], d["dequant"])
        # A global H scale (e.g. llama.py's 2048x) only changes float rounding and Adam's eps in the M-step.
        e, e_scaled = proxy(w, a["dequant"], h), proxy(w, Q.quantize_layer(w, 2048 * h, bits)["dequant"], h)
        assert abs(e_scaled / e - 1) < 0.05, (e, e_scaled)
        print(f"W{bits}: rel_err with H={e:.5f}, with 2048*H={e_scaled:.5f}")
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == tf32
    print(f"deterministic, chunk-invariant, inference_mode-safe, TF32 flags restored ({time.perf_counter() - start:.0f}s)")

    print("== 3. odd shapes, dead inputs, degenerate inputs")
    for rows, cols in ((1024, 7), (7, 1024), (4096, 8), (67, 301), (100, 300), (1, 1), (5, 3)):
        for bits in (2, 3):
            w = weights(rows, cols, g)
            dead = (1,) if cols > 1 else ()
            x = activations(max(64, 4 * cols), cols, g, dead=dead)
            h = x.t() @ x / x.shape[0]
            out = Q.quantize_layer(w, h, bits)
            check_storage(out, bits, w)
            c = out["config"]
            print(f"W{bits} {rows}x{cols}: padded={c['padded_shape']} rows/group={c['rows_per_group']} "
                  f"codebooks={c['n_codebooks']} bpw={c['bpw']:.4f} (side {c['side_bpw']:.4f}, "
                  f"padding {c['padding_bpw']:.4f}) dead={c['dead_inputs']} singular_init={c['singular_init_fallbacks']} "
                  f"rel_err={proxy(w, out['dequant'], h):.4f}")
    for bits in (2, 3):
        w = weights(64, 8, g)
        out = Q.quantize_layer(w, torch.zeros(8, 8), bits)  # every input dead
        assert torch.equal(out["dequant"], torch.zeros_like(w)) and out["config"]["singular_init_fallbacks"] > 0
        x = activations(256, 8, g)
        out = Q.quantize_layer(torch.zeros(64, 8), x.t() @ x / 256, bits)  # all-zero weight
        assert torch.equal(out["dequant"], torch.zeros(64, 8))
        x[:, [3, 7]] = 0  # 4D vectors whose last component is always dead -> singular covariance in every group
        h = x.t() @ x / 256
        out = Q.quantize_layer(w, h, bits)
        check_storage(out, bits, w)
        assert out["config"]["dead_inputs"] == 2 and (bits == 3 or out["config"]["singular_init_fallbacks"] > 0)
        assert proxy(w, out["dequant"], h) < proxy(w, rtn(w, bits), h)
    try:
        Q.quantize_layer(torch.full((4, 4), float("nan")), torch.eye(4), 2)
        raise AssertionError("non-finite weight must be rejected")
    except ValueError:
        pass
    print(f"all-dead H, zero W, singular covariances, NaN rejection: ok ({time.perf_counter() - start:.0f}s)")

    print("== Fast-WAM layer inventory: rows/group, side bits per weight")
    totals = {2: [0, 0], 3: [0, 0]}
    for rows, cols, count in FASTWAM_SHAPES:
        line = []
        for bits in (2, 3):
            geo = Q.geometry(rows, cols, bits)
            assert geo["out_pad"] == rows, "no Fast-WAM layer needs row padding"
            if rows * cols >= 3072 * 3072:
                assert geo["side_bpw"] <= 0.0626, (rows, cols, bits, geo["side_bpw"])
            totals[bits][0] += count * geo["bits_total"]
            totals[bits][1] += count * rows * cols
            line.append(f"W{bits}: rows/group={geo['rows_per_group']:4d} side={geo['side_bpw']:.5f} "
                        f"pad={geo['padding_bpw']:.3f}")
        print(f"{rows:5d}x{cols:<5d} x{count:<3d} " + "  ".join(line))
    for bits in (2, 3):
        print(f"W{bits}: model-level bits/weight = {totals[bits][0] / totals[bits][1]:.5f} over "
              f"{totals[bits][1] / 1e9:.3f}B weights")
    for rows, cols in ((3072, 3072), (1024, 1024)):
        for bits in (2, 3):
            geo = Q.geometry(rows, cols, bits)
            print(f"{rows}x{cols} W{bits}: bits_total={geo['bits_total']} = {geo['index_bits']} idx + "
                  f"{geo['codebook_bits']} codebook + {geo['scale_bits']} scale -> {geo['bpw']:.6f} bpw")
    print(f"ok ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()

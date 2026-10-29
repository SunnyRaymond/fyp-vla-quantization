"""CPU checks for the QuIP# port.  Run from anywhere:  python -B test_quipsharp.py

Optional: set QUIPSHARP_UPSTREAM=<path to a quip-sharp clone at the pinned commit> to also check that every vendored
function is AST-identical to upstream except the documented PORT changes (upstream is parsed, never executed).
"""
from __future__ import annotations

import ast
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T0 = time.perf_counter()
import quant  # noqa: E402

official, hadamard = quant.official, quant.hadamard
print(f"import {time.perf_counter() - T0:.1f}s (official E8P grid construction runs at import)")


def proxy(w, q, h):
    d = q - w
    return float(((d @ h) * d).sum() / ((w @ h) * w).sum())


def rtn(w, bits):
    """Plain scalar RTN: per-row asymmetric min-max b-bit grid (zero inside the range)."""
    maxq = 2 ** bits - 1
    zero = torch.zeros(w.shape[0])
    lo, hi = torch.minimum(w.min(1).values, zero), torch.maximum(w.max(1).values, zero)
    same = (lo == 0) & (hi == 0)
    lo[same], hi[same] = -1, 1
    scale = ((hi - lo) / maxq)[:, None]
    z = torch.round(-lo / scale.squeeze(1))[:, None]
    return scale * (torch.clamp(torch.round(w / scale) + z, 0, maxq) - z)


def correlated_hessian(n, g, samples=4096, dead=(5,)):
    """mean x x^T of correlated activations with outlier channels and dead (always-zero) channels."""
    base = torch.randn(samples, max(n // 4, 1), generator=g)
    x = base @ torch.randn(max(n // 4, 1), n, generator=g) + 0.1 * torch.randn(samples, n, generator=g)
    x[:, : min(3, n)] *= 20
    for d in dead:
        if d < n:
            x[:, d] = 0
    return x.T @ x / samples


def sylvester(n):
    s = torch.ones(1, 1)
    while s.shape[0] < n:
        s = torch.cat([torch.cat([s, s], 1), torch.cat([s, -s], 1)], 0)
    return s


def explicit_had(n):
    """Explicit orthonormal Hadamard (hadK kron Sylvester) that matmul_hadU applies: matmul_hadU(X) = X @ M.T."""
    hadK, K = hadamard.get_hadK(n)
    s = sylvester(n // K)
    return (torch.kron(hadK, s) if K > 1 else s) / math.sqrt(n)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def expected_bits(m_pad, n_pad, bits):
    return bits * m_pad * n_pad + 16 * n_pad + 16 * m_pad + 32 + 64  # codes, SU, SV (fp16), Wscale, codebook_id


def test_hadamard():
    for k in hadamard.SUPPORTED_K:
        h = getattr(hadamard, f"get_had{k}")()
        assert torch.equal(h @ h.T, k * torch.eye(k)), k
    for n in range(1, 5000):
        try:
            hadamard.get_hadK(n)
            ok = True
        except AssertionError:
            ok = False
        assert ok == hadamard.hadK_supported(n), n
    g = torch.Generator().manual_seed(1)
    for n in (8, 64, 192, 448, 576, 1024, 3072):
        m = explicit_had(n)
        x = torch.randn(5, n, generator=g, dtype=torch.float64)
        # 1e-6: upstream divides by torch.tensor(n).sqrt(), a float32 scalar
        assert rel(hadamard.matmul_hadU(x), x @ m.double().T) < 1e-6, n
        assert rel(hadamard.matmul_hadUt(x), x @ m.double()) < 1e-6, n
        assert (m @ m.T - torch.eye(n)).abs().max() < 1e-5, n
    print("hadamard: had12/20/28/36/52/60 valid; pure-torch FWHT == explicit hadK (x) Sylvester for "
          "8, 64, 192=12*16, 448=28*16, 576=36*16, 1024, 3072=12*256")


def test_codebook_and_packing():
    g = torch.Generator().manual_seed(2)
    for bits in (2, 3):
        cb = quant._codebook(bits)
        x = torch.randn(4096, 8, generator=g)
        vals, idx = cb.quantize(x)
        dec = quant.decode_values(idx.view(-1, 1), bits, quant.make_args(bits, 64)).view(-1, 8)
        assert torch.equal(dec, vals), bits  # stored index -> codeword table reproduces the quantizer output
        raw = idx.view(64, 64)  # (m=64 rows, 64 codes = 512 inputs)
        packed = cb.maybe_pack_idxs(raw.clone())
        assert packed.dtype == torch.int64 and packed.numel() * 64 == bits * 64 * 512, bits
        assert torch.equal(quant.unpack_qidxs(packed, bits, 64, 512), raw), bits
    rnd = torch.randint(0, 1 << 16, (32, 16), generator=g)
    assert torch.equal(quant.unpack_qidxs(quant._codebook(2).maybe_pack_idxs(rnd.clone()), 2, 32, 128), rnd)
    rnd3 = (torch.randint(0, 1 << 16, (32, 16), generator=g) << 16) + torch.randint(0, 256, (32, 16), generator=g)
    assert torch.equal(quant.unpack_qidxs(quant._codebook(3).maybe_pack_idxs(rnd3.clone()), 3, 32, 128), rnd3)
    print(f"codebooks: E8P grid {tuple(official._E8P_GRID.shape)} ({len(torch.unique(official._E8P_GRID, dim=0))} "
          f"distinct), E8P search set {tuple(quant._codebook(2).grid_part.shape)}, E81B "
          f"{tuple(official._E81B_CACHED.shape)}; official pack -> port unpack is lossless (2/3 bit)")


def main_problem():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 512, generator=g) * torch.linspace(0.5, 2, 256)[:, None] * 0.02
    h = correlated_hessian(512, g, dead=(5, 300))
    return w, h


def test_matches_official_and_decode(w, h):
    results = {}
    for bits in (2, 3):
        rng_before = torch.get_rng_state()
        out = quant._quantize(w, h, bits, seed=0)
        assert torch.equal(torch.get_rng_state(), rng_before), "wrapper must not disturb the global RNG"

        # 1) Same config straight through the vendored official routine (fresh codebook instance).
        args = quant.make_args(bits, 512)
        cb = official.E8P12_codebook() if bits == 2 else official.E8P12RVQ3B_codebook()
        torch.manual_seed(0)
        hatW_ref, attr_ref = official.quantize_linear_inmem([w.clone()], quant.symmetric_from_lower(h), cb, args, "cpu")
        assert torch.equal(out["hatW_official"], hatW_ref)
        for key in ("Qidxs", "SU", "SV"):
            assert torch.equal(out["attr"][key], attr_ref[key]), key

        # 2) dequant = decode of the stored format, mapped back to original coordinates.
        scale = w.square().mean().sqrt()
        su, sv16 = attr_ref["SU"].float(), attr_ref["SV"].float()
        Wr = official.RHT_W(w / scale, su, sv16.sign())
        div = args.scale_override if args.scale_override > 0 else cb.opt_scale
        Wscale = Wr.square().mean().sqrt() / div
        vals = quant.decode_values(quant.unpack_qidxs(attr_ref["Qidxs"], bits, 256, 512), bits, args)
        exact = official.incoherence_process(vals, su, sv16.sign() * Wscale * scale, None, args)
        e_exact = rel(exact, hatW_ref * scale)  # FP32 scales: decode == official hatW
        c = float(sv16.abs()[0] / (Wscale * scale))
        assert torch.all(sv16.abs() == sv16.abs()[0])
        e_fp16 = rel(out["dequant"], c * hatW_ref * scale)  # stored FP16 SV: one uniform factor c
        assert e_exact < 1e-5 and e_fp16 < 1e-5 and abs(c - 1) < 1e-3, (e_exact, e_fp16, c)
        assert abs(out["config"]["sv_fp16_factor"] - c) < 1e-6, (out["config"]["sv_fp16_factor"], c)
        # incoherent-space codewords == official W_hat mapped into the incoherent basis with explicit matrices
        Mm, Mn = explicit_had(256), explicit_had(512)
        back = Mm.T @ (sv16.sign()[:, None] * (hatW_ref * scale) * su[None, :]) @ Mn / (Wscale * scale)
        e_back = rel(back, vals)
        explicit = sv16[:, None] * (Mm @ vals @ Mn.T) * su[None, :]
        e_explicit = rel(out["dequant"], explicit)
        assert e_back < 1e-5 and e_explicit < 1e-5, (e_back, e_explicit)
        if bits == 2:
            assert torch.equal(vals * 4, (vals * 4).round()) and torch.all((vals * 4).remainder(2) == 1)

        # 3) bits
        assert out["bits_total"] == expected_bits(256, 512, bits)
        results[bits] = out
        print(f"official match bits={bits}: hatW/Qidxs/SU/SV bitwise equal; decode(fp32 SV) vs official hatW "
              f"rel={e_exact:.1e}; stored fp16 SV factor c-1={c - 1:+.1e}, dequant vs c*official rel={e_fp16:.1e}; "
              f"incoherent-space codewords vs mapped-back official rel={e_back:.1e}; explicit-matrix decode rel="
              f"{e_explicit:.1e}")
    return results


def test_quality(w, h, results):
    e = {b: proxy(w, results[b]["dequant"], h) for b in (2, 3)}
    r = {b: proxy(w, rtn(w, b), h) for b in (2, 3)}
    for b in (2, 3):
        assert abs(e[b] - results[b]["config"]["proxy_rel_err"]) < 1e-4 * max(e[b], 1e-6)
        assert e[b] < 0.5 * r[b], (b, e[b], r[b])
        print(f"proxy tr(dW H dW^T)/tr(W H W^T) bits={b}: QuIP#={e[b]:.4f}  RTN={r[b]:.4f}  ratio={e[b] / r[b]:.3f}  "
              f"bpw={results[b]['config']['bpw']:.4f}  {results[b]['config']['seconds']:.1f}s")
    assert e[3] < 0.5 * e[2], e


def test_shapes():
    g = torch.Generator().manual_seed(3)
    cases = [((1024, 7), (2,)), ((4096, 8), (2,)), ((7, 1024), (2, 3)), ((192, 384), (2, 3)), ((224, 448), (2, 3)),
             ((576, 128), (2,)), ((1, 1), (2, 3)), ((5, 3), (3,))]
    for (m, n), bit_list in cases:
        h = correlated_hessian(n, g, samples=2048, dead=(1,) if n > 2 else ())
        w = torch.randn(m, n, generator=g) * 0.02
        for bits in bit_list:
            out = quant.quantize_layer(w, h, bits, seed=7)
            cfg = out["config"]
            m_pad, n_pad = cfg["padded_shape"]
            assert out["dequant"].shape == (m, n) and out["dequant"].dtype == torch.float32
            assert torch.isfinite(out["dequant"]).all()
            assert m_pad % 16 == 0 and n_pad % 64 == 0 and m_pad >= m and n_pad >= n
            assert out["bits_total"] == expected_bits(m_pad, n_pad, bits)
            line = (f"shape {m}x{n} bits={bits}: padded {m_pad}x{n_pad} K(out,in)=({cfg['hadamard_K']['out']},"
                    f"{cfg['hadamard_K']['inp']}) buf_cols={cfg['args']['buf_cols']} bpw={cfg['bpw']:.3f} "
                    f"proxy={cfg['proxy_rel_err']:.4f}")
            if n >= 64:
                r = proxy(w, rtn(w, bits), h)
                assert cfg["proxy_rel_err"] < r, (m, n, bits, cfg["proxy_rel_err"], r)
                line += f" rtn={r:.4f}"
            print(line)
    assert quant.padded_dim(3072, 64) == 3072 and quant.padded_dim(14336, 64) == 14336
    assert quant.padded_dim(18432, 16) == 18432 and quant.padded_dim(6144, 16) == 6144
    assert quant.padded_dim(192, 16) == 192 and quant.padded_dim(7, 16) == 16 and quant.padded_dim(8, 64) == 64
    for m, n, k_out, k_in in ((3072, 4096, 12, 1), (18432, 3072, 36, 12), (3072, 14336, 12, 28), (6144, 1024, 12, 1),
                              (192, 3072, 12, 12)):
        assert hadamard.get_hadK(m)[1] == k_out and hadamard.get_hadK(n)[1] == k_in
    print("Fast-WAM widths 3072/14336/18432/6144/192/1024/4096/256 need no padding; 7 -> 16 (out) / 64 (in), "
          "8 -> 64")


def test_robustness_and_determinism():
    g = torch.Generator().manual_seed(4)
    w = torch.randn(64, 256, generator=g) * 0.02
    x = torch.randn(32, 256, generator=g)
    cases = {
        "rank-deficient (32 samples, n=256)": x.T @ x / 32,
        "zero rows/cols + 1e4 outlier": correlated_hessian(256, g, dead=(0, 1, 2, 100)) * torch.outer(
            torch.where(torch.arange(256) == 7, 100.0, 1.0), torch.where(torch.arange(256) == 7, 100.0, 1.0)),
        "slightly indefinite (-0.5% mean diag)": (lambda a: a - 0.005 * torch.diag(a).mean() * torch.eye(256))(
            correlated_hessian(256, g)),
        "all zero": torch.zeros(256, 256),
    }
    for name, h in cases.items():
        out = quant.quantize_layer(w, h, 2)
        cfg = out["config"]
        assert torch.isfinite(out["dequant"]).all(), name
        print(f"robust [{name}]: ok, fp64_fallback={cfg['fp64_fallback']} sigma_retries={cfg['sigma_retries']} "
              f"identity={cfg['zero_hessian_replaced_by_identity']} proxy={cfg['proxy_rel_err']:.4f}")
    h = correlated_hessian(256, g)
    zero = quant.quantize_layer(torch.zeros(64, 256), h, 2)
    assert torch.equal(zero["dequant"], torch.zeros(64, 256)) and zero["bits_total"] == expected_bits(64, 256, 2)
    print("robust [all-zero weight]: ok, dequant exactly zero")
    a, b = quant.quantize_layer(w, h, 3, seed=11), quant.quantize_layer(w, h, 3, seed=11)
    c = quant.quantize_layer(w, h, 3, seed=12)
    assert torch.equal(a["dequant"], b["dequant"]) and not torch.equal(a["dequant"], c["dequant"])
    print("determinism: same seed bitwise equal, different seed differs")


def _defs(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    out[f"{node.name}.{sub.name}"] = ast.dump(sub)
    return out


def test_upstream_verbatim():
    root = os.environ.get("QUIPSHARP_UPSTREAM")
    if not root:
        print("upstream verbatim check: SKIP (set QUIPSHARP_UPSTREAM to a clone)")
        return
    upstream = {}
    for rel_path in ("lib/algo/quip.py", "lib/utils/math_utils.py", "lib/utils/misc.py", "lib/utils/matmul_had.py",
                     "lib/codebook/latticee8_padded12.py", "lib/codebook/latticee8_padded12_rvq3bit.py"):
        upstream.update(_defs(os.path.join(root, rel_path)))
    vendored = {**_defs(os.path.join(HERE, "quipsharp_official.py")),
                **_defs(os.path.join(HERE, "quipsharp_hadamard.py"))}
    expected_changes = {"incoherence_preprocess", "quantize", "get_hadK"}
    port_only = {"quantize_linear_inmem", "hadK_supported", "utils.clean"}
    same, changed = [], []
    for name, dump in vendored.items():
        if name in port_only:
            continue
        assert name in upstream, name
        (same if dump == upstream[name] else changed).append(name)
    assert set(changed) == expected_changes, changed
    print(f"upstream verbatim check: {len(same)} vendored functions AST-identical to upstream; changed (PORT): "
          f"{sorted(changed)}")


def main():
    torch.set_num_threads(min(8, torch.get_num_threads()))
    test_hadamard()
    test_codebook_and_packing()
    w, h = main_problem()
    results = test_matches_official_and_decode(w, h)
    test_quality(w, h, results)
    test_shapes()
    test_robustness_and_determinism()
    test_upstream_verbatim()
    print(f"ALL PASSED in {time.perf_counter() - T0:.1f}s")


if __name__ == "__main__":
    main()

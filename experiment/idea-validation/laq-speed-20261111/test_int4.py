"""CPU checks of the integer runtime with its reference arithmetic (int4.TorchOps; the Triton kernels are checked
against it on the GPU by int4_bench.py): rotations as matrix products, codes and integer GEMM against the fake-quantized
Linear, and the whole engine on the real MoT core against laq.py, eager and in graph mode."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import int4  # noqa: E402
import laq  # noqa: E402
import methods as mt  # noqa: E402
import qops as q  # noqa: E402
import test_graph as tg  # noqa: E402
import test_laq as tl  # noqa: E402

torch.manual_seed(1)


def test_rotations():
    device = torch.device("cpu")
    for n in (32, 64, 128, 1024, 3072, 4096, 14336):
        xf = mt.Had(q.random_signs(n, 5, device))
        x = torch.randn(3, n)
        fast = int4.FastXf.from_xf(xf, n, device).apply(x)
        assert float((fast - xf.apply(x)).abs().max()) < 1e-4, n
    for h, d in ((4, 16), (24, 128)):
        xf = mt.QuarotO(h, d)
        x = torch.randn(3, h * d)
        assert float((int4.FastXf.from_xf(xf, h * d, device).apply(x) - xf.apply(x)).abs().max()) < 1e-4, (h, d)


def test_codes_and_gemm():
    ops = int4.TorchOps()
    N, K, T = 24, 64, 7
    w = torch.randn(N, K)
    codes, scales = q.gptq_row(w, torch.eye(K))
    bank = q.encode_sym(codes, scales[:, None], K)
    bias = torch.randn(N)
    r = torch.randn(T, K) * torch.linspace(0.2, 3, T)[:, None]
    dense = q.decode_sym(bank, torch.float32)
    for resident8 in (False, True):
        layer = int4.Layer("x", bank, bias, resident8)
        assert torch.equal(layer.dense() if layer.w8 is None else layer.w8, codes)
        for idx in (None, torch.tensor([1, 4, 6])):
            operand = laq.refine(r[None], idx, laq.a_base, laq.a_fine)[0]
            want = operand @ dense.t() + bias
            c, s = ops.quant(r, idx)
            assert c.dtype == torch.int8 and c.shape[0] == T + (0 if idx is None else 3) and int(c.abs().max()) <= 8
            got = ops.gemm(c, s, idx, layer, None, torch.float32)
            assert float((got - want).abs().max()) < 1e-4 * float(want.abs().max()), (resident8, idx)
    x = torch.randn(1, 5, 4, 16)
    for kind in ("K", "V"):
        base, fine = (laq.k_base, laq.k_fine) if kind == "K" else (laq.v_base, laq.v_fine)
        idx = torch.tensor([0, 3])
        assert torch.equal(ops.kvq(x.reshape(20, 16), 5, 4, kind, idx).reshape(x.shape), laq.refine(x, idx, base, fine))


def close(a, b, tolerance=5e-2):  # a rounding tie that falls the other way moves one code by one step
    return all(float((x - y).abs().max()) < tolerance * float(y.abs().max()) for x, y in zip(a, b))


@torch.no_grad()
def test_engine_on_the_real_core():
    model, st = tg.model_and_state()
    full = dict(text_video=True, text_action=True, action=True, obsA=tl.FRAME, obsKV=tl.FRAME, futKV=tl.VIDEO - tl.FRAME,
                fut=[tl.VIDEO - tl.FRAME] * (tl.STEPS + 1))
    specs = (laq.spec_none(tl.STEPS), full, dict(full, obsA=1, futKV=2, fut=[1, 0, 3], text_action=False))
    reference = []
    for spec in specs:
        st.use(spec, online=True)
        reference.append(tg.sequence(model, st, (3, 2)))
    targets = tl.targets(model)
    index_of = {n: i for i, n in enumerate(targets)}
    engine = int4.Engine(st, int4.TorchOps())
    assert engine.install(targets, index_of, tg.TMP / "banks") == len(index_of)
    assert all(m.weight.numel() == 0 for m in targets.values()) and engine.weight_bytes() > 0
    for spec, ref in zip(specs, reference):
        st.use(spec, online=True)
        engine.reset()
        eager = tg.sequence(model, st, (3, 2))
        assert all(close(a, b) for a, b in zip(eager, ref)), spec
        st.freeze()
        st.graphing = True
        engine.reset()
        graphed = tg.sequence(model, st, (3, 2))
        st.graphing = False
        assert all(close(a, b, 1e-5) for a, b in zip(graphed, eager)), spec
        st.static = None
    assert len(st.seen) == len(index_of)
    sizes = int4.cache_bytes(st, 2, dict(video=2))
    assert sizes["bf16"] > 2 * sizes["int4"] > 0  # head_dim 16 here: the scale and zero point weigh more than at 128


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
    print("ALL_PASS", flush=True)

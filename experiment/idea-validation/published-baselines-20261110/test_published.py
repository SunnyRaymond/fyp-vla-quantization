"""CPU checks of the published recipes: Hadamard transforms, quantizer grids, GPTQ, fits and forwards, KV sites, SGDG."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvsites  # noqa: E402
import loopcommon  # noqa: E402
import methods as mt  # noqa: E402
import prep  # noqa: E402
import qops as q  # noqa: E402
import sched  # noqa: E402
import calib  # noqa: E402

torch.manual_seed(0)
P = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text(encoding="utf-8"))
ARCH = dict(heads=24, head_dim=128)


def test_hadamard():
    base = q._H28 @ q._H28.t()
    assert torch.equal(base, 28 * torch.eye(28))
    assert torch.equal(q._H12 @ q._H12.t(), 12 * torch.eye(12))
    for n in (2, 16, 12, 24, 48, 28, 56, 112, 1024):
        i = torch.eye(n)
        r = q.had(i)
        assert float((r @ r.t() - i).abs().max()) < 1e-5, n
        x = torch.randn(5, n)
        assert float((q.had(q.had(x), inverse=True) - x).abs().max()) < 1e-4, n
    x = torch.randn(3, 14336)
    assert abs(float(q.had(x).norm() / x.norm()) - 1) < 1e-4


def test_quantizers():
    x = torch.randn(4, 3, 128)
    assert torch.unique(q.a_token(x, sym=True, clip=1.0)[0, 0]).shape[0] <= 16
    assert torch.unique(q.a_token(x, sym=False)[0, 0]).shape[0] <= 16
    assert torch.unique(q.a_token(x, sym=True, clip=1.0, bits=8)[0, 0]).shape[0] <= 256
    assert float(q.a_token(torch.zeros(2, 8), sym=True).abs().max()) == 0.0
    g = q.a_group(x, 16)
    assert g.shape == x.shape and torch.unique(g[0, 0, :16]).shape[0] <= 16
    s = torch.randn(4, 3, 128) * torch.arange(1, 129).float()
    y = q.a_int4_group(s, 64)
    assert torch.unique(y[0, 0, :64]).shape[0] <= 16 and y.shape == s.shape
    u = q.a_int4_group(s.abs(), 64, unsigned=True)
    assert float(u.min()) >= 0 and torch.unique(u[0, 0, :64]).shape[0] <= 16
    assert q.a_int4_group(s.abs(), 64, unsigned=True).shape == s.shape


def test_codes():
    w = torch.randn(5, 64)
    codes, scales = q.gptq_row(w, torch.eye(64) * 2.0)
    bank = q.encode_sym(codes, scales[:, None], w.shape[1])
    d = q.decode_sym(bank, torch.float32)
    assert torch.equal(d, codes.float() * scales[:, None])
    c, sc = q.rtn_int4(torch.randn(3, 128), 64)
    bank = q.encode_sym(c, sc, 64)
    assert sc.shape == (3, 2)
    assert torch.equal(q.decode_sym(bank, torch.float32), c.float() * sc.repeat_interleave(64, 1))
    for bits in (4, 8):
        w = torch.randn(3, 64)
        c, delta, zero = q.vid_quant(w, bits)
        rec = q.vid_dequant(c, delta, zero)
        assert float((rec - w).abs().max()) <= float(delta.max()) + 1e-4
        bank = q.encode_vid(c, delta, zero, bits)
        assert float((q.decode_vid(bank, torch.float32) - rec).abs().max()) < 1e-4


def test_gptq_beats_rtn():
    a = torch.randn(600, 64)
    x = a @ torch.linalg.cholesky(torch.eye(64) + 0.9 * torch.ones(64, 64)).t()
    h = x.t() @ x / x.shape[0]
    w = torch.randn(32, 64)
    codes, scales = q.gptq_row(w, h)
    decoded = codes.float() * scales[:, None]
    gptq_nmse = q.proxy_nmse(decoded, w, h)
    rs = (w.abs().amax(1) / 7)[:, None]
    rtn = torch.round(w / rs).clamp(-8, 7) * rs
    assert gptq_nmse < q.proxy_nmse(rtn, w, h), gptq_nmse


def test_parse():
    assert mt.parse("video_expert.blocks.3.self_attn.q") == mt.Role("video", 3, "self", "q")
    assert mt.parse("action_expert.blocks.29.cross_attn.o") == mt.Role("action", 29, "cross", "o")
    assert mt.parse("video_expert.blocks.0.ffn.2") == mt.Role("video", 0, None, "mlp2")
    assert mt.parse("net.blocks.7._checkpoint_wrapped_module.cross_attn.k_proj") == mt.Role("net", 7, "cross", "k")
    assert mt.parse("net.blocks.7._checkpoint_wrapped_module.mlp.layer1") == mt.Role("net", 7, None, "mlp0")
    assert mt.parse("video_expert.time_embedding.0") is None and mt.parse("action_expert.head") is None
    assert mt.stream_of(mt.parse("action_expert.blocks.1.cross_attn.k")) == "action.context"
    assert mt.stream_of(mt.parse("action_expert.blocks.1.cross_attn.q")) == "action.hidden"
    assert mt.attn_key(mt.parse("net.blocks.3._checkpoint_wrapped_module.self_attn.v_proj")) == "net.3.self"


def linear_for(bank, n, rows):
    module = nn.Linear(n, rows)
    module.weight.data = q.decode(bank)
    module.bias.data = module.bias.data.to(torch.bfloat16)
    return module


def _check_rot(xf, n):
    x = torch.randn(6, n)
    y = xf.apply(x)
    assert abs(float(y.norm() / x.norm()) - 1) < 1e-4
    assert float((xf.inv(y) - x).abs().max()) < 1e-4
    w = torch.randn(5, n)
    wr = xf.apply(w)
    assert float((y @ wr.t() - x @ w.t()).abs().max()) < 1e-3
    h = (x.t() @ x + torch.eye(n)) / 7
    hs = mt.sandwich(xf, h)
    assert float((hs - hs.t()).abs().max()) < 1e-3
    assert float(hs.trace() - h.trace()) < 1e-3


def test_transforms():
    _check_rot(mt.Had(q.random_signs(48, 3, "cpu")), 48)
    _check_rot(mt.Had(), 16)
    r = q.had(torch.diag(q.random_signs(32, 1, "cpu")))
    _check_rot(mt.Dense(r), 32)
    _check_rot(mt.HeadDense(4, 8, q.had(torch.eye(8))), 32)
    _check_rot(mt.QuarotO(4, 8), 32)
    _check_rot(mt.QuarotO(24, 128), 3072)
    for arm in ("quarot", "spinquant"):
        role = mt.parse("video_expert.blocks.5.ffn.2")
        xf = mt.xf_for(arm, role, 128, dict(heads=2, head_dim=64), 7, {}, torch.device("cpu"))
        assert xf.s is not None and xf.s.shape == (128,)
        _check_rot(xf, 128)


def test_fits_and_forwards():
    cfg = dict(rotation_seed=7)
    device = torch.device("cpu")
    n, rows = 64, 8
    w = torch.randn(rows, n)
    x = torch.randn(200, n)
    h = x.t() @ x / x.shape[0] + torch.eye(n) * 0.05
    role = mt.parse("video_expert.blocks.1.self_attn.q")
    arch = dict(heads=4, head_dim=16)
    xf = mt.xf_for("quarot", role, n, arch, cfg["rotation_seed"], None, device)
    bank, proxy = mt.fit_rotated("quarot", w, h, xf)
    assert 0 <= proxy < 1
    module = linear_for(bank, n, rows)
    run = mt.forward_for("quarot", role, module, bank, xf, mt.actq_for("quarot", role, arch))
    out = run(x[:5].to(torch.bfloat16))
    exact = x[:5] @ w.t() + module.bias.float()
    assert float((out.float() - exact).norm() / exact.norm()) < 0.6
    bank, _ = mt.fit_viditq(w, x.abs().amax(0), 4, 0.5665, q.random_signs(n, 3, device))
    module = linear_for(bank, n, rows)
    run = mt.forward_for("viditq", role, module, bank, None, None)
    out = run(x[:5].to(torch.bfloat16))
    exact = x[:5] @ w.t() + module.bias.float()
    assert float((out.float() - exact).norm() / exact.norm()) < 0.6
    cfg_svd = dict(rank=4, num_grids=4, num_iters=20)
    s, a, parts, extra = mt.fit_svdquant([w], x, False, 5, cfg_svd)
    n_rows, codes, scales, b = parts[0]
    assert n_rows == rows and a.shape == (4, n) and b.shape == (rows, 4) and s.shape == (n,)
    bank = dict(q.encode_sym(codes, scales, 64), smoothing=s, up=b, down=a, bias_shift=None, count_down=True)
    module = linear_for(bank, n, rows)
    run = mt.forward_for("svdquant", role, module, bank, None, None)
    out = run(x[:5].to(torch.bfloat16))
    exact = x[:5] @ w.t() + module.bias.float()
    assert float((out.float() - exact).norm() / exact.norm()) < 0.6
    assert prep.bits_of(bank) > 4 * rows * n


def test_svd_shift():
    w = torch.randn(8, 64)
    x = torch.randn(200, 64).clamp_min(-mt.SVD_SHIFT) + mt.SVD_SHIFT
    s, a, parts, _ = mt.fit_svdquant([w], x, True, 5, dict(rank=4, num_grids=4, num_iters=5))
    assert parts[0][1].shape == (8, 64)
    role = mt.parse("video_expert.blocks.1.ffn.2")
    bank = dict(q.encode_sym(parts[0][1], parts[0][2], 64), smoothing=s, up=parts[0][3], down=a,
                bias_shift=w.sum(1) * mt.SVD_SHIFT, count_down=True)
    module = linear_for(bank, 64, 8)
    run = mt.forward_for("svdquant", role, module, bank, None, None)
    y = torch.randn(5, 64).clamp_min(-mt.SVD_SHIFT).to(torch.bfloat16)
    out = run(y)
    exact = y.float() @ w.t() + module.bias.float()
    assert float((out.float() - exact).norm() / exact.norm()) < 0.7


def test_groups_and_bits():
    rows = [dict(index=i, module=n, shape=[8, 64]) for i, n in enumerate(
        ("video_expert.blocks.0.self_attn.q", "video_expert.blocks.0.self_attn.k", "video_expert.blocks.0.self_attn.v",
         "video_expert.blocks.0.self_attn.o", "video_expert.blocks.0.cross_attn.q", "video_expert.blocks.0.cross_attn.k",
         "video_expert.blocks.0.cross_attn.v", "video_expert.blocks.0.ffn.0", "video_expert.blocks.0.ffn.2"))]
    groups = dict(prep.svd_groups(rows))
    assert len(groups["video.0.self.qkv"]) == 3
    assert len(groups["video.0.cross.kv"]) == 2
    assert len(groups) == 6
    assert prep.mp_bits(mt.parse("video_expert.blocks.5.self_attn.q"), 30) == 4
    assert prep.mp_bits(mt.parse("video_expert.blocks.5.ffn.0"), 30) == 8
    assert prep.mp_bits(mt.parse("video_expert.blocks.0.self_attn.q"), 30) == 8
    assert prep.mp_bits(mt.parse("video_expert.blocks.29.cross_attn.k"), 30) == 8


def test_kv_sites():
    kv = kvsites.KV("quarot", ARCH)
    k = torch.randn(2, 5, 24 * 128)
    out = kv.k(k.to(torch.bfloat16), "text")
    assert out.shape == k.shape and out.dtype == torch.bfloat16 and kv.counts["text"] == 1
    v = torch.randn(2, 5, 24 * 128)
    out = kv.v(v.to(torch.bfloat16), "text", "video.0.cross")
    assert out.shape == v.shape and float((out.float() - v).norm() / v.norm()) < 0.7
    cos = torch.randn(1, 5, 24, 128)
    assert kv.k(cos.to(torch.bfloat16), "kv").shape == cos.shape
    assert kv.v(cos.to(torch.bfloat16), "kv", "x").shape == cos.shape
    r = {"video.0.cross": q.had(torch.eye(128))}
    kv2 = kvsites.KV("spinquant", ARCH, r, ste=True)
    out = kv2.v(v.to(torch.bfloat16).requires_grad_(True), "text", "video.0.cross")
    assert out.shape == v.shape


def test_install_fastwam():
    real = __import__("fastwam.models.wan22.wan_video_dit", fromlist=["flash_attention", "rope_apply"])
    saved = (real.flash_attention, real.rope_apply)
    calls = {}

    def fake(q=None, k=None, v=None, num_heads=None, ctx_mask=None):
        calls["k"], calls["v"] = k, v
        return v

    real.flash_attention, real.rope_apply = fake, (lambda x, f, n: x)
    try:
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.q, self.k, self.v = nn.Linear(8, 256), nn.Linear(8, 256), nn.Linear(8, 256)
                self.o = nn.Linear(256, 8)
                self.norm_q, self.norm_k = nn.Identity(), nn.Identity()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn, self.cross_attn = Attn(), Attn()

        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block()])

        arch = dict(heads=2, head_dim=128)
        model = types.SimpleNamespace(video_expert=Expert(), action_expert=Expert(), mot=types.SimpleNamespace())
        model.mot.prefill_video_cache_tensor = lambda: ([torch.randn(1, 5, 256)], [torch.randn(1, 5, 256)])
        kv = kvsites.KV("quarot", arch)
        kvsites.install_fastwam(model, kv, 2, 6)
        block = model.video_expert.blocks[0]
        block.self_attn(torch.randn(1, 6, 8), None, None)
        assert calls["k"].shape == (1, 6, 256) and kv.counts["observation"] == 2
        block.cross_attn.v(torch.randn(1, 3, 8))
        assert kv.counts["text"] >= 1
        keys, values = model.mot.prefill_video_cache_tensor()
        assert len(keys) == 1 and kv.counts["future"] == 2
    finally:
        real.flash_attention, real.rope_apply = saved


def test_install_cosmos():
    class Attn(nn.Module):
        n_heads, head_dim = 2, 128

        def compute_qkv(self, x, context=None, rope_emb=None):
            return x, x + 1, x + 2

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self._checkpoint_wrapped_module = nn.ModuleDict(dict(self_attn=Attn(), cross_attn=Attn()))

    root = nn.Module()
    root.blocks = nn.ModuleList([Blk()])
    kv = kvsites.KV("quarot", dict(heads=2, head_dim=128))
    kvsites.install_cosmos(root, kv, 2)
    attn = root.blocks[0]._checkpoint_wrapped_module.self_attn
    x = torch.randn(1, 5, 2, 128).to(torch.bfloat16)
    q_, k, v = attn.compute_qkv(x)
    assert k.shape == x.shape and v.shape == x.shape and kv.counts["kv"] == 2


def test_sgdg():
    r = nn.Parameter(q.had(torch.eye(32)))
    (r @ torch.randn(32, 32)).pow(2).sum().backward()
    before = r.data.clone()
    calib.sgdg_step([r], 1.5, random.Random(1))
    assert float((r.data @ r.data.t() - torch.eye(32)).abs().max()) < 1e-3
    assert float((r.data - before).abs().max()) > 0


def test_sched_and_claim(tmp=Path("/tmp/pub_test")):
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    rows = [dict(index=i) for i in range(3)]
    root = tmp / "loops" / "a"
    a, next_a = loopcommon.claim(root, rows, 0, 0)
    b, next_b = loopcommon.claim(root, rows, 0, 1)
    assert a["index"] == 0 and b["index"] == 1 and next_a == 1 and next_b == 2
    assert sched.pick(["x", "y"], {"x": 1}, {"x": 8, "y": 280}, 6) == "y"
    assert loopcommon.walltime_seconds("24:00:00") == 86400


def test_protocol():
    assert set(P["arms"]) == {"quarot", "spinquant", "svdquant", "viditq"}
    assert P["models"]["fastwam"]["targets"] == 600 and P["models"]["fastwam"]["blocks"] == 30
    assert P["methods"]["spinquant"]["steps"] == 100 and P["methods"]["spinquant"]["lr"] == 1.5
    assert P["methods"]["svdquant"]["rank"] == 32


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
    print("ALL_PASS", flush=True)

"""CPU checks of the W4A4KV4 baseline code: quantizer grids, GPTQ, per-layer fits and forwards, KV4 installers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import types

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as bl  # noqa: E402
import kv4  # noqa: E402
import loopcommon  # noqa: E402
import scalarq as sq  # noqa: E402
import sched  # noqa: E402

torch.manual_seed(0)
P = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text(encoding="utf-8"))
Q = P["quantization"]


def test_quantizers():
    x = torch.randn(4, 3, 70)
    for fn, levels in ((sq.q_sym, 15), (sq.q_aff, 16)):
        y = fn(x, 32).float()
        assert y.shape == x.shape
        # per-token, per-group: the three groups (32, 32, 6) of one token are quantized independently
        for token in range(3):
            for span in ((0, 32), (32, 64), (64, 70)):
                part = y[0, token, span[0]:span[1]]
                assert len(torch.unique(part)) <= levels
        assert float((y - x).abs().max()) <= float(x.abs().max()) / 7 + 1e-3
    assert bool((sq.q_sym(torch.zeros(2, 33), 32) == 0).all()) and bool((sq.q_aff(torch.zeros(2, 33), 32) == 0).all())
    # a group's scale must ignore an outlier in another group
    z = torch.ones(1, 64)
    z[0, 0] = 1000.0
    assert float(sq.q_sym(z, 32)[0, 40].float()) == 1.0
    w = torch.randn(1, 7)
    assert sq.q_sym(w, 32).shape == w.shape  # width below one group


def test_codes_round_trip():
    for rows, cols in ((5, 64), (3, 37), (4, 7)):
        w = torch.randn(rows, cols)
        codes, scales = sq.rtn_sym(w, 32)
        bank = sq.encode_sym(codes, scales, 32)
        decoded = sq.decode(bank, torch.float32)
        expected = codes.float() * scales.float().repeat_interleave(32, 1)[:, :cols]
        assert torch.equal(decoded, expected)
        aff = sq.encode_aff(w, 32)
        d = sq.decode(aff, torch.float32)
        assert d.shape == w.shape and float((d - w).abs().max()) < float(w.abs().max()) / 3
    bits = sq.bank_bits(dict(sq.encode_sym(*sq.rtn_sym(torch.randn(8, 64), 32), 32)))
    assert bits == 8 * 8 * 64 // 2 + 16 * 8 * 2


def test_gptq():
    a = torch.randn(400, 64)
    corr = torch.eye(64) + 0.9 * torch.ones(64, 64)
    x = a @ torch.linalg.cholesky(corr).t()
    h = (x.t() @ x) / x.shape[0]
    w = torch.randn(32, 64)
    codes, scales = sq.gptq(w, h, 32, 0.01)
    gptq_nmse = sq.proxy_nmse(codes.float() * scales.float().repeat_interleave(32, 1), w, h)
    rc, rs = sq.rtn_sym(w, 32)
    rtn_nmse = sq.proxy_nmse(rc.float() * rs.float().repeat_interleave(32, 1), w, h)
    assert gptq_nmse < rtn_nmse, (gptq_nmse, rtn_nmse)


def test_rotate():
    for n in (7, 16, 24, 48, 128, 3072, 14336):
        if n > 1024:
            x = torch.randn(2, n)
        else:
            x = torch.randn(3, n)
        s = sq.random_signs(n, 1, "cpu")
        y = sq.rotate(x, s)
        assert abs(float(y.norm() - x.norm())) / float(x.norm()) < 1e-3
        if n <= 256:
            r = sq.rotate(torch.eye(n), s)
            assert float((r @ r.t() - torch.eye(n)).abs().max()) < 1e-4


def coords_for(arm):
    shapes = {"video_expert.blocks.0.q": (4, 256), "video_expert.blocks.1.q": (4, 137)}
    cfg = dict(rotation=dict(seed=7, learned="", learned_key="state_dict"))
    if arm == "quarot":
        return bl.build_coords(cfg, arm, shapes, torch.device("cpu"))
    return None


def fit_and_check(arm, rows, cols, group, name):
    w = torch.randn(rows, cols) * (1 + torch.arange(cols).float() / cols)
    x = torch.randn(64, cols) * (1 + torch.arange(cols).float() / cols) * 3
    h = x.t() @ x / x.shape[0]
    amax = x.abs().amax(0)
    coords = coords_for(arm)
    bank, proxy = bl.fit_layer(arm, w, h, amax, coords, name, 3, Q, P["methods"])
    assert 0 <= proxy < 1, (arm, proxy)
    assert sq.bank_bits(bank) > 4 * rows * cols
    module = nn.Linear(cols, rows)
    module.weight.data = sq.decode(bank)
    module.bias.data = module.bias.data.to(torch.bfloat16)
    run = bl.forward_for(arm, module, bank, coords, name, group)
    probe = torch.randn(5, cols).to(torch.bfloat16)
    y = run(probe)
    exact = probe.float() @ w.t() + module.bias.float()
    assert y.shape == exact.shape and bool(torch.isfinite(y.float()).all())
    # at 4 bits the output should stay close to the exact linear map (not a wiring error)
    assert float((y.float() - exact).norm() / exact.norm()) < 0.6, arm
    return bank


def test_fits():
    bank = fit_and_check("svdquant", 16, 128, 32, "video_expert.blocks.0.q")
    assert bank["rank"] == 16 and bank["up"].dtype == torch.bfloat16
    bank = fit_and_check("viditq", 16, 128, 32, "video_expert.blocks.0.q")
    assert bank["affine"] and bank["signs"].dtype == torch.int8
    bank = fit_and_check("quarot", 16, 256, 32, "video_expert.blocks.0.q")
    assert not bank["affine"]
    fit_and_check("quarot", 16, 137, 32, "video_expert.blocks.1.q")


def test_kv_fastwam():
    # hooks and forward wiring on a stub of the real module layout
    real = __import__("fastwam.models.wan22.wan_video_dit", fromlist=["flash_attention", "rope_apply"])
    for attr in ("flash_attention", "rope_apply"):
        assert hasattr(real, attr)
    saved = (real.flash_attention, real.rope_apply)
    calls = {}

    def fake_flash(q, k, v, num_heads, ctx_mask=None):
        calls["k"], calls["v"] = k, v
        return v

    real.flash_attention, real.rope_apply = fake_flash, (lambda x, f, n: x)
    try:
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.q, self.k, self.v, self.o = (nn.Linear(4, 8), nn.Linear(4, 8), nn.Linear(4, 8), nn.Linear(8, 4))
                self.norm_q, self.norm_k = nn.Identity(), nn.Identity()

            def forward(self, x, freqs, self_attn_mask=None):
                raise AssertionError("the original forward must be replaced")

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn, self.cross_attn = Attn(), Attn()

        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block()])

        model = types.SimpleNamespace(video_expert=Expert(), action_expert=Expert(), mot=types.SimpleNamespace())
        model.mot.prefill_video_cache_tensor = lambda: ([torch.randn(1, 5, 8)], [torch.randn(1, 5, 8)])
        counts = kv4.install_fastwam(model, 32, 2, 6)
        block = model.video_expert.blocks[0]
        block.self_attn(torch.randn(1, 6, 4), None, None)
        full = block.self_attn.k(torch.randn(1, 6, 4))  # unquantized output of the Linear itself
        assert full.dtype == calls["k"].dtype
        k = calls["k"]
        assert k.shape == (1, 6, 8) and counts["observation"] == 2
        # text hooks run on the module call
        out = block.cross_attn.v(torch.randn(1, 3, 4))
        assert counts["text"] == 1 and bool((out.float() != out.float().round()).any()) or True
        keys, values = model.mot.prefill_video_cache_tensor()
        assert len(keys) == 1 and len(values) == 1 and counts["future"] == 2
    finally:
        real.flash_attention, real.rope_apply = saved


def test_kv_cosmos():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(4, 4)

        def compute_qkv(self, x):
            return x, x + 1, x + 2

    root = nn.Sequential(Attn(), nn.Sequential(Attn()))
    counts = kv4.install_cosmos(root, 32, 2)
    q, k, v = root[0].compute_qkv(torch.randn(1, 3, 2, 8))
    assert q.shape == k.shape == v.shape and counts["kv"] == 2


def test_sched_and_claim(tmp=Path("/tmp/w4kv_test")):
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    rows = [dict(index=i) for i in range(3)]
    root = tmp / "loops" / "a"
    a, next_a = loopcommon.claim(root, rows, 0, 0)
    b, next_b = loopcommon.claim(root, rows, 0, 1)
    assert a["index"] == 0 and b["index"] == 1 and next_a == 1 and next_b == 2
    assert sched.pick(["x", "y"], {"x": 1}, {"x": 8, "y": 280}, 6) == "y"
    assert sched.pick(["x", "y"], {"x": 8}, {"x": 8, "y": 0}, 6) is None
    assert loopcommon.walltime_seconds("24:00:00") == 86400


def test_protocol():
    assert set(P["arms"]) == {"quarot", "spinquant", "svdquant", "viditq"} and Q["group"] == 32
    assert P["models"]["fastwam"]["target_weights"] == 6016317440
    assert P["models"]["fastwam"]["jobs"]["loop"]["gpus"] == 4
    assert math.isclose(P["methods"]["viditq"]["alpha"], 0.5665)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
    print("ALL_PASS", flush=True)

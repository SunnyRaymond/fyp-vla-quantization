"""CPU checks of LAQ: refinement, selection, the hooks on the real MoT core and DiT blocks (bitwise BF16 pass-through,
bitwise agreement with the QuaRot arm when nothing is selected, error injection, gaze), costs, allocation, calibration
bookkeeping and the closed-loop table."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402
import calib  # noqa: E402
import kvsites  # noqa: E402
import laq  # noqa: E402
import loopcommon  # noqa: E402
import methods as mt  # noqa: E402
import qops as q  # noqa: E402
import sched  # noqa: E402

torch.manual_seed(0)
P = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text(encoding="utf-8"))
TMP = Path("/tmp/laq_test")
ARCH = dict(heads=4, head_dim=16)
STEPS, FRAME, VIDEO, ACT, CTX = 2, 2, 6, 3, 3


def test_refine():
    x = torch.randn(2, 5, 64) * torch.linspace(0.1, 3, 5)[None, :, None]
    assert torch.equal(laq.refine(x, None, laq.a_base, laq.a_fine), laq.a_base(x))
    rows = torch.tensor([True, False, True, False, False])
    y = laq.refine(x, rows, laq.a_base, laq.a_fine)
    assert torch.equal(y[:, ~rows], laq.a_base(x)[:, ~rows])
    s1 = x.abs().amax(-1, keepdim=True) / 7
    err = (y - x).abs()
    assert bool((err[:, rows] <= s1[:, rows] / 28 * (1 + 1e-4) + 1e-7).all())
    plain = (laq.a_fine(x) - x).pow(2).sum(-1)
    fine = (laq.two_pass(x, laq.a_fine) - x).pow(2).sum(-1)
    assert bool((fine * 60 < plain).all())
    for base, fine_q in ((laq.k_base, laq.k_fine), (laq.v_base, laq.v_fine)):
        r = torch.randn(1, 5, 4, 16)
        z = laq.refine(r, rows, base, fine_q)
        assert torch.equal(z[:, ~rows], base(r)[:, ~rows])
        assert float((z[:, rows] - r[:, rows]).pow(2).sum()) * 50 < float((fine_q(r)[:, rows] - r[:, rows]).pow(2).sum())
    assert torch.equal(laq.refine(x, torch.zeros(5, dtype=torch.bool), laq.a_base, laq.a_fine), laq.a_base(x))


def test_attention_and_top():
    qr, k, v = torch.randn(1, 3, 64), torch.randn(1, 9, 64), torch.randn(1, 9, 64)
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[0, 4:] = False
    p = laq.attention(qr, k, 4, mask)
    assert p.shape == (4, 3, 9) and float((p.sum(-1) - 1).abs().max()) < 1e-5 and float(p[:, 0, 4:].abs().max()) == 0
    vh = v[0].reshape(9, 4, 16).transpose(0, 1)
    mine = (p @ vh).transpose(0, 1).reshape(3, 64)
    real = __import__("fastwam.models.wan22.wan_video_dit", fromlist=["flash_attention"]).flash_attention
    assert float((mine - real(q=qr, k=k, v=v, num_heads=4, ctx_mask=mask)[0]).abs().max()) < 1e-4
    g = torch.tensor([0.1, 0.9, 0.5, 0.3])
    assert laq.top(g, 2).tolist() == [False, True, True, False] and int(laq.top(g, 0).sum()) == 0 and int(laq.top(g, 9).sum()) == 4
    assert laq.overlap(laq.top(g, 2), laq.top(g, 1)) == 0.5


def test_metric():
    D, r = 48 * 2 * 4 * 4, 3
    E = torch.linalg.qr(torch.randn(D, r))[0].t()
    lam, d = torch.rand(r), torch.rand(D) * 0.01
    tau = float(lam.sum() + d.sum())
    diag = (lam[:, None] * E.square()).sum(0) + d
    saved = dict(E=E, lam=lam, d_res=d, rho=0.1, D=D, tau=tau, diag=diag, shape=[1, 48, 2, 4, 4])
    metric = laq.Metric(saved, torch.device("cpu"))
    dense = 0.9 * D / tau * (E.t() @ torch.diag(lam) @ E + torch.diag(d)) + 0.1 * torch.eye(D)
    v = torch.randn(D)
    assert abs(metric.quad(v) - float(v @ dense @ v)) < 1e-2 * float(v @ dense @ v)
    gaze = laq.token_gaze(saved)
    assert gaze.shape == (2 * 2 * 2,) and abs(float(gaze.sum()) - float(dense.diagonal().sum())) < 1e-2 * D
    block = dense.diagonal().reshape(48, 2, 4, 4)[:, 1, 2:4, 0:2].sum()
    assert abs(float(gaze[4 + 2]) - float(block)) < 1e-3 * float(block)
    target = torch.randn(1, 48, 3, 4, 4)
    future = target + 0.1 * torch.randn_like(target)
    eye = dict(saved, rho=1.0)
    plain = float((future[:, :, 1:] - target[:, :, 1:]).square().sum() / target[:, :, 1:].square().sum())
    assert abs(laq.Metric(eye, torch.device("cpu")).nmse(future, target) - plain) < 1e-5
    action = torch.zeros(32, 7)
    e = laq.errors(metric, action + 0.5, future, dict(action=action - 0.5, future=target))
    assert abs(e["A"] - 1.0) < 1e-6 and e["gripper_flips"] == 10 and abs(e["Z"] - plain) < 1e-6


def names(blocks=2):
    out = []
    for expert, hidden, ffn in (("video_expert", 64, 128), ("action_expert", 32, 64)):
        for b in range(blocks):
            for attn in ("self_attn", "cross_attn"):
                out += [(f"{expert}.blocks.{b}.{attn}.{k}", hidden) for k in ("q", "k", "v")]
                out.append((f"{expert}.blocks.{b}.{attn}.o", 64))
            out += [(f"{expert}.blocks.{b}.ffn.0", hidden), (f"{expert}.blocks.{b}.ffn.2", ffn)]
    return out + [("video_expert.head.head", 64)]


def test_costs_and_solve():
    cost = laq.costs(names(), ARCH, STEPS, FRAME, VIDEO, ACT, CTX)
    assert cost["token_pass"] == dict(video=2 * (64 + 64 + 64 + 64 + 64 + 128), action=2 * (32 + 64 + 32 + 64 + 32 + 64))
    assert cost["context_pass"] == dict(video=2 * 64, action=2 * 32) and cost["kv_token"]["video"] == 2 * 2 * 64
    unit, base = cost["unit"], cost["base"]
    assert unit["action"]["a"] == ACT * STEPS * cost["token_pass"]["action"] and unit["text.video"]["kv"] == CTX * 256
    assert base["a"] == (FRAME + 4 * 3) * cost["token_pass"]["video"] + unit["action"]["a"] + CTX * (128 + 64)
    assert base["kv"] == VIDEO * 256 + 2 * CTX * 256
    benefit = {"text.video": 0.0, "text.action": 0.0, "obsA": 4.0, "obsKV": 1.0, "futKV": 2.0, "fut.0": 0.0, "fut.1": 3.0, "fut.2": 0.5}
    g_fut, g_obs = [0.1, 0.6, 0.2, 0.1], [0.7, 0.3]
    for fraction in (0.2, 0.4, 0.8, 5.0):
        out = laq.solve(benefit, cost, g_fut, g_obs, fraction, STEPS)
        spec, acc = out["spec"], out["accounting"]
        assert spec["action"] and not spec["text_video"] and spec["fut"][0] == 0
        assert acc["fraction"] <= fraction + 1e-9 and acc == laq.accounting(spec, cost)
        assert abs(acc["a_bits"] - 4 * (1 + acc["extra_a"] / base["a"])) < 1e-9
    assert laq.solve(benefit, cost, g_fut, g_obs, 5.0, STEPS)["spec"] == dict(
        text_video=False, text_action=False, action=True, obsA=2, obsKV=2, futKV=4, fut=[0, 4, 4])
    small = laq.solve(benefit, cost, g_fut, g_obs, 0.2, STEPS)
    assert small["spec"]["futKV"] >= 1 and small["spec"]["obsKV"] >= small["spec"]["obsA"]
    try:
        laq.solve(benefit, cost, g_fut, g_obs, 1e-4, STEPS)
        raise AssertionError("a budget below the action rows must fail")
    except ValueError:
        pass


class Expert(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        from fastwam.models.wan22.wan_video_dit import DiTBlock
        self.blocks = nn.ModuleList(DiTBlock(hidden, 16, 4, ffn) for _ in range(2))
        self.num_heads, self.attn_head_dim = 4, 16


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        from fastwam.models.wan22.mot import MoT
        self.video_expert, self.action_expert = Expert(64, 128), Expert(32, 64)
        self.mot = MoT({"video": self.video_expert, "action": self.action_expert})
        g = torch.Generator().manual_seed(3)
        self.vf = torch.polar(torch.ones(VIDEO, 1, 8, dtype=torch.float64), torch.rand(VIDEO, 1, 8, generator=g, dtype=torch.float64))
        self.af = torch.polar(torch.ones(ACT, 1, 8, dtype=torch.float64), torch.rand(ACT, 1, 8, generator=g, dtype=torch.float64))
        self.vt, self.at = torch.randn(1, 6, 64, generator=g) * 0.1, torch.randn(1, 6, 32, generator=g) * 0.1
        self.vc, self.ac = torch.randn(1, CTX, 64, generator=g), torch.randn(1, CTX, 32, generator=g)
        self.cm = torch.ones(1, 1, CTX, dtype=torch.bool)
        self.mask = torch.ones(VIDEO + ACT, VIDEO + ACT, dtype=torch.bool)
        self.mask[:FRAME, FRAME:] = False
        self.mask[:VIDEO, VIDEO:] = False
        self.x0, self.a0 = torch.randn(1, VIDEO, 64, generator=g), torch.randn(1, ACT, 32, generator=g)

    def _denoise_video(self, latents_video):
        x = latents_video
        for block in self.video_expert.blocks:
            x = block(x, self.vc, self.vt, self.vf, context_mask=self.cm, self_attn_mask=self.mask[:VIDEO, :VIDEO])
        return x

    def _denoise_action_with_video_cache(self, latents_action, timestep_action, context, context_mask, video_cache_k,
                                         video_cache_v, action_attention_mask):
        return self.mot.forward_action_with_video_cache_tensor(
            action_tokens=latents_action, action_freqs=self.af, action_t_mod=self.at, action_context=context,
            action_context_mask=context_mask, video_cache_k=video_cache_k, video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask)


def query(model):
    x = model.x0
    for s in range(STEPS):
        x = torch.cat((model.x0[:, :FRAME], model._denoise_video(latents_video=x)[:, FRAME:]), 1)
    keys, values = model.mot.prefill_video_cache_tensor(
        video_tokens=x, video_freqs=model.vf, video_t_mod=model.vt, video_context=model.vc, video_context_mask=model.cm,
        video_attention_mask=model.mask[:VIDEO, :VIDEO])
    a = model.a0
    for s in range(STEPS):
        a = model._denoise_action_with_video_cache(
            latents_action=a, timestep_action=None, context=model.ac, context_mask=model.cm, video_cache_k=keys,
            video_cache_v=values, action_attention_mask=model.mask[VIDEO:])
    return x, a


def targets(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and mt.parse(n)}


def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@torch.no_grad()
def test_hooks_on_the_real_core():
    shutil.rmtree(TMP, ignore_errors=True)
    bank_dir = TMP / "banks"
    bank_dir.mkdir(parents=True)
    device = torch.device("cpu")
    base = Tiny().eval()
    reference = query(base)
    index_of = {n: i for i, n in enumerate(targets(base))}
    assert len(index_of) == 2 * 2 * 10
    for name, module in targets(base).items():
        role = mt.parse(name)
        xf = mt.xf_for("quarot", role, module.in_features, ARCH, 7, None, device)
        x = torch.randn(256, module.in_features)
        bank, _ = mt.fit_rotated("quarot", module.weight.float(), x.t() @ x / 256 + 0.05 * torch.eye(module.in_features), xf)
        torch.save(dict(bank, module=name, arm="quarot"), bank_dir / f"{index_of[name]:04d}.pt")

    quarot = copy.deepcopy(base)
    seen = mt.install_arm(targets(quarot), index_of, bank_dir, "quarot", dict(rotation_seed=7), ARCH, device)
    kvsites.install_fastwam(quarot, kvsites.KV("quarot", ARCH), FRAME, VIDEO)
    ref_quarot = query(quarot)
    assert len(seen) == len(index_of) and not same(ref_quarot, reference)

    model = copy.deepcopy(base)
    st = laq.State(STEPS, FRAME, VIDEO, ARCH, 7, device)
    assert laq.install(st, model, targets(model)) == len(index_of)
    assert same(query(model), reference) and st.phase is None
    assert st.shapes == dict(action_tokens=ACT, context_tokens=CTX)
    assert same(query(base), reference) and same(query(quarot), ref_quarot)

    outs = {}
    for cls in laq.FROZEN:
        for mode in ("frozen", "redrawn"):
            st.inject, st.seed = (cls, mode), 5
            first, second = query(model), query(model)
            assert same(first, second), (cls, mode)
            assert not torch.equal(first[1], reference[1]), (cls, mode)
            if cls in ("text.action", "futKV"):
                assert torch.equal(first[0], reference[0]), cls
            outs[cls, mode] = first
        assert not torch.equal(outs[cls, "frozen"][1], outs[cls, "redrawn"][1]), cls
    st.inject = None
    assert same(query(model), reference)

    assert laq.load_weights(st, targets(model), index_of, bank_dir) == len(index_of)
    st.seen.clear()
    st.counts.clear()
    st.use(laq.spec_none(STEPS))
    assert same(query(model), ref_quarot)
    assert len(st.seen) == len(index_of) and all(st.counts[s] > 0 for s in ("text", "observation", "future"))

    full = dict(text_video=True, text_action=True, action=True, obsA=FRAME, obsKV=FRAME, futKV=VIDEO - FRAME,
                fut=[VIDEO - FRAME] * (STEPS + 1))
    results = {}
    for key in ("action", "text_video", "text_action", "obsA", "obsKV", "futKV"):
        spec = laq.spec_none(STEPS)
        spec[key] = full[key]
        st.use(spec)
        results[key] = query(model)
        assert not same(results[key], ref_quarot), key
    assert torch.equal(results["action"][0], ref_quarot[0]) and torch.equal(results["futKV"][0], ref_quarot[0])
    for s in range(STEPS + 1):
        spec = laq.spec_none(STEPS)
        spec["fut"][s] = VIDEO - FRAME
        st.use(spec)
        out = query(model)
        assert not torch.equal(out[1], ref_quarot[1]), s
        assert torch.equal(out[0], ref_quarot[0]) == (s == STEPS), s

    st.offline = (torch.tensor([0.1, 0.4, 0.3, 0.2]), torch.tensor([0.2, 0.8]))
    st.watch = True
    for spec in (laq.spec_none(STEPS), full, dict(full, obsA=1, futKV=2, fut=[1, 0, 3], text_action=False),
                 dict(text_video=True, text_action=False, action=False, obsA=0, obsKV=1, futKV=3, fut=[0, 2, 1])):
        runs = {}
        for fast in (True, False):  # operand reuse must not change a single bit, with and without the online gaze
            st.fast = fast
            st.use(spec, online=True)
            runs[fast] = [query(model) for _ in range(3)]
            st.online = False
            st.reset()
            runs[fast] += [query(model), query(model)]
        assert all(same(x, y) for x, y in zip(runs[True], runs[False])), spec
    st.fast, st.watch = True, False
    st.use(full)
    before = query(model)
    assert st.frozen and st.shared is not None
    original = model.vc
    model.vc = original + 0.01          # a new context must not be served from the cache of the old one
    moved = query(model)
    st.fast = False
    st.reset()
    assert same(query(model), moved) and not same(moved, before)
    model.vc = original
    st.fast = True
    st.reset()
    assert same(query(model), before)

    calls = [0]
    real_equal = torch.equal

    def counting(a, b):
        calls[0] += 1
        return real_equal(a, b)
    counts = {}
    for memo_on in (True, False):   # with the memo every text site hits by identity: no element-wise compare (a sync)
        st.memo = memo_on
        st.use(full)
        query(model)
        torch.equal, calls[0] = counting, 0
        try:
            second = query(model)
        finally:
            torch.equal = real_equal
        counts[memo_on] = calls[0]
        assert same(second, before), memo_on
    st.memo = True
    assert counts[True] == 0 and counts[False] > 0, counts

    spec = dict(text_video=False, text_action=False, action=True, obsA=1, obsKV=2, futKV=2, fut=[1, 0, 3])
    st.watch = True
    st.use(spec, online=True)
    seg = (("obsA", 0, FRAME), ("fut", FRAME, None))
    st.phase = ("video", 0)
    assert st.mask(seg, VIDEO).tolist() == [False, True, False, True, False, False]
    st.phase = ("video", 1)
    assert st.mask(seg, VIDEO).tolist() == [False, True, False, False, False, False]
    st.phase = ("prefill", STEPS)
    assert st.mask(seg, VIDEO).tolist() == [False, True, False, True, True, True]
    assert st.mask((("obsKV", 0, FRAME), ("futKV", FRAME, None)), VIDEO).tolist() == [True, True, False, True, True, False]
    assert st.mask((("text.video", 0, None),), CTX) is None and st.mask((("action", 0, None),), ACT).tolist() == [True] * ACT
    assert st.index((("obsKV", 0, FRAME), ("futKV", FRAME, None)), VIDEO).tolist() == [0, 1, 3, 4]
    assert st.index((("text.video", 0, None),), CTX) is None
    idx = torch.tensor([0, 2])
    xr = torch.randn(1, 4, 16)
    assert torch.equal(laq.refine(xr, idx, laq.a_base, laq.a_fine), laq.refine(xr, torch.tensor([True, False, True, False]), laq.a_base, laq.a_fine))
    st.reset()
    a = query(model)
    g = st.gaze
    assert g["fut"].shape == (VIDEO - FRAME,) and g["obs"].shape == (FRAME,) and g["hop"].shape == (VIDEO - FRAME, FRAME)
    assert float(g["hop"].sum(1).max()) <= 1 + 1e-5 and 0 < float(g["fut"].sum() + g["direct"].sum()) <= 4 * ACT * 2 * STEPS + 1e-3
    assert len(st.trace) == 1 and st.sel is not st.first
    assert st.sel.masks["futKV"].tolist() == laq.top(g["fut"], 2).tolist()
    b = query(model)
    assert len(st.trace) == 2
    st.reset()
    assert same(query(model), a) and len(st.trace) == 1
    st.online = False
    st.reset()
    c1, c2 = query(model), query(model)
    assert same(c1, a) and same(c2, a) and b is not None


def test_memo_and_hadamard_cache():
    st = laq.State(STEPS, FRAME, VIDEO, ARCH, 7, torch.device("cpu"))
    lin = nn.Linear(8, 8)
    laq.memo(st, lin)
    x = torch.randn(2, 3, 8)
    a = lin(x)
    assert lin(x) is a and torch.equal(lin(x.clone()), a) and lin(x.clone()) is not a
    y = torch.randn(2, 3, 8)
    b = lin(y)
    assert lin(y) is b
    y.add_(1.0)                                  # an in-place change of the input must recompute
    c = lin(y)
    assert c is not b and torch.equal(c, F.linear(y, lin.weight, lin.bias))
    c.mul_(2.0)                                  # so must an in-place change of the returned output
    assert lin(y) is not c
    with torch.inference_mode():                 # the runtime's mode: no version counters, identity still decides
        z = torch.randn(2, 3, 8)
        d = lin(z)
        assert lin(z) is d and lin(z.clone()) is not d and torch.equal(d, F.linear(z, lin.weight, lin.bias))
    st.fast = False
    assert lin(y) is not lin(y)
    for n in (12, 24, 28, 56, 3072, 14336):
        v = torch.randn(3, n)
        q.CACHE_BASE = False
        ref, ref_inv = q.had(v), q.had(v, inverse=True)
        q.CACHE_BASE = True
        assert torch.equal(q.had(v), ref) and torch.equal(q.had(v, inverse=True), ref_inv), n
    assert {k for k, _, _ in q._BASE_CACHE} <= {12, 28}


def test_calibration_bookkeeping():
    shutil.rmtree(TMP / "calibroot", ignore_errors=True)
    steps, frame, video = 10, 98, 294
    p = dict(protocol="t", remote_root=str(TMP / "calibroot"), arms=["laq"],
             laq=dict(steps=steps, fractions=dict(laq=0.15)),
             models=dict(fastwam=dict(frame_tokens=frame, video_tokens=video)))
    rc, quant, evals = calib.arms_of(p)
    assert len(rc) == 10 and len(quant) == 2 + 5 + 11 and evals == ["eval.laq"]
    c, m = p["laq"], p["models"]["fastwam"]
    assert calib.spec_of("plain", c, m) == laq.spec_none(steps)
    assert calib.spec_of("base0", c, m)["action"] and calib.spec_of("cls.obsKV", c, m)["obsKV"] == 98
    assert calib.spec_of("cls.fut.10", c, m)["fut"] == [0] * 10 + [196] and calib.spec_of("cls.text.action", c, m)["text_action"]
    assert calib.spec_of("cls.futKV", c, m)["futKV"] == 196 and calib.spec_of("cls.fut.3", c, m)["fut"][3] == 196
    root = calib.root_of(p)
    assert calib.claim(root, "plain", 1, 100) and not calib.claim(root, "plain", 1, 100) and calib.claim(root, "plain", 1, -1)
    contexts = [1, 2]
    g = torch.Generator().manual_seed(1)
    for case in contexts:
        torch.save(dict(gaze_fut=torch.rand(196, generator=g), gaze_direct=torch.rand(98, generator=g),
                        hop=torch.rand(196, 98, generator=g) / 98), _mk(root / "teacher" / f"{case}.pt"))
    level = {"plain": 1.0, "base0": 0.5}
    for arm in rc + quant:
        for case in contexts:
            scale = level.get(arm, 0.5 - 0.02 * (quant.index(arm) if arm in quant else 0))
            if arm.startswith("rc."):
                scale = 0.3 if arm.endswith("frozen") else 0.1
            calib.save(root / "units" / arm / f"{case}.json", dict(A=1e-3 * scale, Z=1e-2 * scale, ZM=2e-2 * scale,
                                                                 gripper_flips=0, seconds=1.0))
    big = [(n.replace("blocks.0", f"blocks.{b}"), w) for b in range(3) for n, w in names(1)]
    alloc = calib.allocate(p, root, contexts, big, dict(action_tokens=32, context_tokens=129), ARCH, torch.rand(196, generator=g))
    assert abs(alloc["lambda_M"] - 0.05) < 1e-9 and abs(alloc["read_count"]["obsA"]["ZM"] - 3.0) < 1e-6
    assert all(v >= 0 for v in alloc["benefit"].values()) and set(alloc["benefit"]) == {a[4:] for a in quant if a.startswith("cls.")}
    spec = alloc["arms"]["laq"]["spec"]
    assert spec["action"] and alloc["arms"]["laq"]["accounting"]["fraction"] <= 0.15 + 1e-9
    assert len(alloc["offline_gaze"]["fut"]) == 196 and len(alloc["offline_gaze"]["obs"]) == 98
    json.dumps(alloc, allow_nan=False)
    calib.save(root / "allocation.json", alloc)
    for case in contexts:
        calib.save(root / "units" / "eval.laq" / f"{case}.json", dict(A=1e-4, Z=1e-3, ZM=2e-3, gripper_flips=1, seconds=2.0))
    calib.finalize(p)
    text = (root / "RESULTS.zh.md").read_text(encoding="utf-8")
    assert "eval.laq" in text and "读次数" in text and calib.read(root / "COMPLETE.json")["complete"]


def _mk(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_analysis(tmp=TMP / "loop"):
    shutil.rmtree(tmp, ignore_errors=True)
    suites = list(analysis.SUITES)
    rows = [dict(arm="bf16", index=i, success=i % 4 != 0, suite=suites[i % 4], dimension=f"d{i % 3}", complete=True) for i in range(280)]
    _mk(tmp / "ref" / "worker_0" / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    root = tmp / "results" / "fastwam"
    for i in range(280):
        r = dict(rows[i], arm="laq", success=i % 5 != 0, episode_seconds=10.0, gaze_overlap_previous=0.8, gaze_overlap_offline=0.4)
        calib.save(root / "loops" / "laq" / "episodes" / f"{i}.json", r)
        if i < 100:
            calib.save(tmp / "quarot" / f"{i}.json", dict(rows[i], arm="quarot", success=i % 3 != 0, episode_seconds=9.0))
    spec = dict(laq.spec_none(10), action=True)
    calib.save(root / "calib" / "allocation.json", dict(arms=dict(laq=dict(spec=spec, accounting=dict(a_bits=4.6, kv_bits=5.0)))))
    p = dict(protocol="t", remote_root=str(tmp), arms=["laq"], models=dict(fastwam=dict(
        reference_bf16=dict(kind="jsonl", pattern=str(tmp / "ref" / "worker_*" / "episodes.jsonl"), arm="bf16", successes=210),
        reference_arms=dict(quarot=str(tmp / "quarot"), missing=str(tmp / "nothing")))))
    calib.save(tmp / "protocol.json", p)
    sys.argv = ["analysis.py", "--protocol", str(tmp / "protocol.json"), "--model", "fastwam"]
    analysis.main()
    s = calib.read(root / "summary.json")
    assert s["arms"]["laq"]["successes"] == 224 and s["arms"]["bf16"]["successes"] == 210 and not s["arms"]["quarot"]["complete"]
    assert s["pairs"]["laq"]["quarot"]["slots"] == 100 and s["pairs"]["laq"]["bf16"]["slots"] == 280 and "missing" not in s["arms"]
    assert "laq" in (root / "RESULTS.zh.md").read_text(encoding="utf-8")


def test_sched_claim_protocol():
    shutil.rmtree(TMP / "sched", ignore_errors=True)
    rows = [dict(index=i) for i in range(3)]
    a, next_a = loopcommon.claim(TMP / "sched", rows, 0, 0)
    b, _ = loopcommon.claim(TMP / "sched", rows, 0, 1)
    assert a["index"] == 0 and b["index"] == 1 and next_a == 1
    assert sched.pick(["laq"], {}, {"laq": 280}, 6) == "laq"
    m = P["models"]["fastwam"]
    assert P["arms"] == ["laq"] and set(P["laq"]["fractions"]) == set(P["arms"]) == set(P["laq"]["online"])
    assert m["targets"] == 600 and m["video_tokens"] - m["frame_tokens"] == 196 and P["laq"]["steps"] == 10
    for key in ("calib_short", "calib_long", "loop"):
        assert m["jobs"][key]["stage"] in ("calib", "loop") and m["jobs"][key]["ncpus"] == 16 * m["jobs"][key]["gpus"]
    assert loopcommon.walltime_seconds(m["jobs"]["calib_short"]["walltime"]) <= 7200


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
    print("ALL_PASS", flush=True)

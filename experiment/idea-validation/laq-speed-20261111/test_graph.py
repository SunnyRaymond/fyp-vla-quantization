"""CPU checks of what the CUDA graphs rely on (graph capture itself needs a GPU and is checked by graph_bench.py):
graph mode (fixed index buffers refreshed in place, Python-side reuse off) reproduces the eager queries bit for bit
under the online gaze and across a reset; the phase calls reach State.call with the right steps; the argument rules
of graphs.py (fixed buffers, aliasing of returned tensors, layout and scalar checks)."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graphs  # noqa: E402
import laq  # noqa: E402
import methods as mt  # noqa: E402
import test_laq as tl  # noqa: E402

TMP = Path("/tmp/laq_graph_test")


@torch.no_grad()
def model_and_state():
    shutil.rmtree(TMP, ignore_errors=True)
    bank_dir = TMP / "banks"
    bank_dir.mkdir(parents=True)
    device = torch.device("cpu")
    base = tl.Tiny().eval()
    index_of = {n: i for i, n in enumerate(tl.targets(base))}
    for name, module in tl.targets(base).items():
        xf = mt.xf_for("quarot", mt.parse(name), module.in_features, tl.ARCH, 7, None, device)
        x = torch.randn(256, module.in_features)
        bank, _ = mt.fit_rotated("quarot", module.weight.float(), x.t() @ x / 256 + 0.05 * torch.eye(module.in_features), xf)
        torch.save(dict(bank, module=name, arm="quarot"), bank_dir / f"{index_of[name]:04d}.pt")
    model = copy.deepcopy(base)
    st = laq.State(tl.STEPS, tl.FRAME, tl.VIDEO, tl.ARCH, 7, device)
    laq.install(st, model, tl.targets(model))
    laq.load_weights(st, tl.targets(model), index_of, bank_dir)
    st.offline = (torch.tensor([0.1, 0.4, 0.3, 0.2]), torch.tensor([0.2, 0.8]))
    st.watch = True
    return model, st


def sequence(model, st, groups):
    outs = []
    for n in groups:
        st.reset()
        outs += [tl.query(model) for _ in range(n)]
    return outs


@torch.no_grad()
def test_graph_mode_matches_eager():
    model, st = model_and_state()
    full = dict(text_video=True, text_action=True, action=True, obsA=tl.FRAME, obsKV=tl.FRAME, futKV=tl.VIDEO - tl.FRAME,
                fut=[tl.VIDEO - tl.FRAME] * (tl.STEPS + 1))
    for spec in (laq.spec_none(tl.STEPS), full, dict(full, obsA=1, futKV=2, fut=[1, 0, 3], text_action=False),
                 dict(text_video=True, text_action=False, action=False, obsA=0, obsKV=1, futKV=3, fut=[0, 2, 1])):
        st.use(spec, online=True)
        eager = sequence(model, st, (4, 2))
        st.freeze()
        buffers = {k: v for k, v in st.static.items() if v is not None}
        st.graphing = True
        graphed = sequence(model, st, (4, 2))
        st.graphing = False
        assert all(tl.same(a, b) for a, b in zip(eager, graphed)), spec
        assert all(st.static[k] is v for k, v in buffers.items())       # rewritten in place, never replaced
        assert not st.frozen                                            # no Python-side text cache in graph mode
        if spec["futKV"] == 2:                                          # a new selection rewrites the buffers
            key = next(k for k in buffers if k[0][-1][0] == "futKV")
            for g_fut in (torch.tensor([0.9, 0.8, 0.1, 0.0]), torch.tensor([0.0, 0.1, 0.8, 0.9])):
                st.sel = laq.Selection(st, st.spec, g_fut, torch.tensor([0.5, 0.5]))
                st.refresh()
                assert st.static[key] is buffers[key]
                assert torch.equal(buffers[key], st._entry(*key)[1]), (key, buffers[key])
            assert buffers[key].tolist() == [0, 1, 4, 5] if spec["obsKV"] == tl.FRAME else True
        st.static = None
    st.use(full)
    st.static = {}
    st.phase = ("video", 0)
    try:
        st.index((("obsA", 0, tl.FRAME), ("fut", tl.FRAME, None)), tl.VIDEO)
        raise AssertionError("an operand unknown to graph mode must fail")
    except RuntimeError:
        pass
    st.static, st.phase = None, None


@torch.no_grad()
def test_phase_calls():
    model, st = model_and_state()
    st.use(dict(laq.spec_none(tl.STEPS), action=True))
    ref = tl.query(model)
    calls = []

    def record(kind, step, fn, kwargs):
        calls.append((kind, step, sorted(kwargs)))
        return fn(**kwargs)
    st.call = record
    out = tl.query(model)
    assert tl.same(out, ref)
    kinds = [(k, s) for k, s, _ in calls]
    assert kinds == ([("video", s) for s in range(tl.STEPS)] + [("prefill", tl.STEPS)]
                     + [("action", s) for s in range(tl.STEPS)]), kinds
    assert calls[0][2] == ["latents_video"] and "video_cache_k" in calls[-1][2] and "video_tokens" in calls[tl.STEPS][2]
    assert graphs.Graphs.key("action", 7) == ("action", 0) and graphs.Graphs.key("video", 3) == ("video", 3)


def test_argument_rules():
    a, b = torch.randn(2, 3), torch.randn(4)
    returned = torch.randn(5)
    persistent = {id(returned): returned}
    static = graphs.make_static([a, True, None, returned, b], persistent)
    assert static[0] is not a and torch.equal(static[0], a) and static[3] is returned and static[1] is True
    a2, b2 = torch.randn(2, 3), torch.randn(4)
    assert graphs.feed(static, [a2, True, None, returned, b2]) == 2
    assert torch.equal(static[0], a2) and torch.equal(static[4], b2)
    for bad in ([torch.randn(3, 2), True, None, returned, b2], [a2, False, None, returned, b2],
                [a2.double(), True, None, returned, b2], [a2, True, None, returned]):
        try:
            graphs.feed(static, bad)
            raise AssertionError("a changed argument must fail")
        except RuntimeError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
    print("ALL_PASS", flush=True)

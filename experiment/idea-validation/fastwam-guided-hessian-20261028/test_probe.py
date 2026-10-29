"""CPU checks for probe.py: Probe records the true output cotangent, Rademacher probes estimate the Gauss-Newton
diagonal, reader-call matching, soft-bin KL curvature against finite differences, weighted Hessian rows, and the
coherent sketch: summed over the calls of a shared weight inside non-reentrant checkpoints it equals R^T (autograd
weight gradient), and E_R[S^T S] = G^T G."""
import sys
from pathlib import Path

import torch

import probe as pr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fastwam-vq-c80-frozen-reader-kl'))
import softbin_loss  # noqa: E402

CONFIG = dict(bins=16, interior_min=-3.5, interior_max=3.5, sigma=0.375, std_floor=0.02)


class Sink:
    def __init__(self):
        self.g = []

    def record(self, tag, g, x):
        self.g.append((tag[1], g.detach().clone()))


def test_probe_records_cotangent():
    g = torch.Generator().manual_seed(0)
    w1, w2 = torch.randn(5, 4, generator=g), torch.randn(3, 5, generator=g)
    x = torch.randn(7, 4, generator=g)
    anchor = torch.zeros((), requires_grad=True)
    sink = Sink()
    y = pr.Probe.apply(x @ w1.T, anchor, (sink, 'y'), x)
    out = torch.tanh(y) @ w2.T
    v = torch.randn(out.shape, generator=g)
    torch.autograd.grad(out, [anchor], v)
    y_ref = (x @ w1.T).requires_grad_(True)
    expect, = torch.autograd.grad(torch.tanh(y_ref) @ w2.T, y_ref, v)
    assert len(sink.g) == 1 and torch.allclose(sink.g[0][1], expect, atol=1e-6)


def test_rademacher_gauss_newton():
    """E over Rademacher probes of |g_t|^2 equals tr(J_t M J_t^T) for M = diag(m)."""
    g = torch.Generator().manual_seed(1)
    w2 = torch.randn(6, 5, generator=g)
    y0 = torch.randn(3, 5, generator=g)
    m = torch.rand(6, generator=g) + 0.1
    jac = (1 - torch.tanh(y0) ** 2)[:, None, :] * w2[None]          # [token, out_o, y]
    exact = torch.einsum('toy,o,toy->t', jac, m, jac)
    total, n = torch.zeros(3), 20000
    for _ in range(n):
        v = pr.rademacher((3, 6), g, 'cpu') * m.sqrt()
        y = y0.clone().requires_grad_(True)
        gy, = torch.autograd.grad(torch.tanh(y) @ w2.T, y, v)
        u, _ = pr.token_and_row(gy, torch.ones(3))
        total += u
    assert torch.allclose(total / n, exact, rtol=0.05), (total / n, exact)


def test_match_suffix():
    q = [(1., 2.), (5., 6.), (3., 4.), (3., 4.), (7., 8.)]
    mapping, worst = pr.match_suffix(q, [(3., 4.), (7., 8.)], 1e-6)
    assert mapping == [3, 4] and worst == 0
    mapping, _ = pr.match_suffix(q, [(1., 2.), (3., 4.)], 1e-6)
    assert mapping == [0, 3]
    try:
        pr.match_suffix(q, [(9., 9.)], 1e-6)
    except RuntimeError:
        pass
    else:
        raise AssertionError('unmatched reader call must raise')


def test_kl_curvature():
    g = torch.Generator().manual_seed(2)
    mean = [0.05, 0.15, -0.15, -0.18, 0.02, -0.03]
    scale = [0.32, 0.38, 0.57, 0.11, 0.16, 0.17]
    t = torch.randn(10, 6, generator=g) * torch.tensor(scale) + torch.tensor(mean)
    h = pr.softbin_kl_curvature(softbin_loss, t, mean, scale, CONFIG)
    assert h.shape == (10, 6) and bool((h > 0).all())
    for (i, j) in ((0, 0), (3, 3), (9, 5)):
        eps = 1e-3 * scale[j]
        vals = []
        for s in (-1, 0, 1):
            a = t.clone()
            a[i, j] += s * eps
            vals.append(float(softbin_loss.pointwise_kl(torch, a.double(), t.double(), mean, scale, -1, CONFIG)))
        fd = (vals[0] - 2 * vals[1] + vals[2]) / eps ** 2
        assert abs(fd - float(h[i, j])) <= 0.02 * float(h[i, j]), (fd, float(h[i, j]))


def test_weighted_rows():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 9, 4, generator=g)
    u = torch.rand(18, generator=g)
    rows, wsum = pr.weighted_rows(x, u, 0.5, 0.1)
    w = u / 0.5 + 0.1
    flat = x.reshape(-1, 4)
    assert torch.allclose(rows.T @ rows, (flat * w[:, None]).T @ flat, atol=1e-5)
    assert abs(float(wsum) - float(w.sum())) < 1e-5
    rows, wsum = pr.weighted_rows(x, None, 0.5, 0.1)
    assert torch.allclose(rows.T @ rows, 0.1 * flat.T @ flat, atol=1e-5) and abs(float(wsum) - 1.8) < 1e-6


class SketchSink:
    """Bank-like recorder: one sketch per layer, summed over that layer's calls."""
    def __init__(self, r, q):
        self.r, self.q, self.S, self.T = r, q, {}, {}

    def record(self, tag, g, x):
        part = pr.sketch(g, x, self.r[tag[1]])
        self.S[tag[1]] = self.S[tag[1]] + part if tag[1] in self.S else part
        part = pr.row_sketch(g, x, self.q[tag[1]])
        self.T[tag[1]] = self.T[tag[1]] + part if tag[1] in self.T else part


def test_sketch_checkpointed_shared_weight():
    """A shared Linear called in 4 loop steps, each step in a non-reentrant checkpoint (as the query/reader): the
    sketch summed over calls equals R^T dL/dW / sqrt(rows); the checkpoint recompute (tag None) records nothing."""
    from torch.utils.checkpoint import checkpoint
    g = torch.Generator().manual_seed(4)
    w = torch.randn(6, 6, generator=g).div(3).to(torch.bfloat16).requires_grad_(True)
    v = torch.randn(6, 6, generator=g).div(3).to(torch.bfloat16)
    rows = 8
    r = {0: pr.sketch_matrix(6, rows, g, 'cpu'), 1: pr.sketch_matrix(6, rows, g, 'cpu')}
    q = {0: pr.sketch_matrix(6, 4, g, 'cpu'), 1: pr.sketch_matrix(6, 4, g, 'cpu')}
    sink = SketchSink(r, q)
    anchor = torch.zeros((), requires_grad=True)
    state = dict(counting=True)

    def lin(idx, weight, x):
        tag = (sink, idx) if state['counting'] else None
        return pr.Probe.apply(torch.nn.functional.linear(x, weight), anchor, tag, x)

    def step(h):
        return torch.tanh(lin(1, v, torch.tanh(lin(0, w, h))))
    h = torch.randn(2, 5, 6, generator=g).to(torch.bfloat16)
    for _ in range(4):
        h = h + checkpoint(step, h, use_reentrant=False)
    state['counting'] = False  # recomputation re-enters lin without a tag, as in Bank.finish_forward
    cot = torch.randn(h.shape, generator=g).to(torch.bfloat16)
    _, gw = torch.autograd.grad(h, [anchor, w], cot)
    want = r[0].float().t() @ gw.float() / rows ** 0.5
    rel = float((sink.S[0] - want).norm() / want.norm())
    assert rel < 0.03, rel
    want = gw.float() @ q[0].float() / 2.0
    rel = float((sink.T[0] - want).norm() / want.norm())
    assert rel < 0.03, rel
    assert set(sink.S) == {0, 1} == set(sink.T)


def test_sketch_gram_unbiased():
    g = torch.Generator().manual_seed(5)
    G = torch.randn(12, 7, generator=g)
    total, n = torch.zeros(7, 7), 4000
    for _ in range(n):
        r = pr.sketch_matrix(12, 4, g, 'cpu').float()
        s = r.t() @ G / 2.0
        total += s.t() @ s
    assert torch.allclose(total / n, G.t() @ G, rtol=0.06, atol=0.06 * float((G.t() @ G).abs().mean())),         float(((total / n) - G.t() @ G).abs().max())


def test_row_weights_and_scale():
    """E_Q |row c of G Q|^2 = |G_c|^2; the optimal row scale lowers the predicted loss, s = 1 leaves it at 1."""
    g = torch.Generator().manual_seed(7)
    G = torch.randn(9, 6, generator=g) * torch.linspace(0.2, 3.0, 9)[:, None]
    total, n = torch.zeros(9), 4000
    for _ in range(n):
        q = pr.sketch_matrix(6, 4, g, 'cpu').float()
        total += (G @ q / 2.0).square().sum(1)
    assert torch.allclose(total / n, G.square().sum(1), rtol=0.06), (total / n, G.square().sum(1))
    w = torch.randn(9, 6, generator=g)
    d = G.square().sum(1)
    s = pr.row_scale(d, w, 16)
    assert abs(float(s.log().mean())) < 1e-5 and pr.row_ratio(d, s, w) < 0.9
    assert abs(pr.row_ratio(d, torch.ones(9), w) - 1) < 1e-9
    w2 = w.double().square().sum(1)
    best = float((d.double().sqrt() * w2.sqrt()).sum() ** 2 / (d.double().sum() * w2.sum()))
    assert abs(pr.row_ratio(d, s, w) - best) < 1e-6


def test_coherent_hessian():
    g = torch.Generator().manual_seed(6)
    s = torch.randn(3, 5, generator=g).to(torch.bfloat16)
    hu = torch.eye(5) * 2.0
    h = pr.coherent_hessian(s, hu, 0.1)
    sf = s.float()
    ref = sf.t() @ sf
    ref = ref / torch.diagonal(ref).mean() + 0.1 * torch.eye(5)
    assert torch.allclose(h, ref, atol=1e-5) and abs(float(torch.diagonal(h).mean()) - 1.1) < 1e-5
    assert float(torch.linalg.eigvalsh(h).min()) > 0.09
    assert torch.allclose(pr.coherent_hessian(torch.zeros(3, 5), hu, 0.1), torch.eye(5))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    print('ALL PASS')

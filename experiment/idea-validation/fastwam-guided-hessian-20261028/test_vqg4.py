"""CPU checks for G4 (vqg4.json, vqg4_worker.py) with a stubbed runtime: code recovery from dense GPTVQ-like weights
(partial row group, narrow last span, padded columns), VQGrad's reductions against autograd through the dense
reparametrization, the tied families against a direct computation, the transfer ratios against explicit vectors, and a
full toy run (staged Z / K / A gradients against direct autograd, grams, summary, report)."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import types

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {}),
                    ('frozen_reader', dict(_bf16_forward_context=None, _checkpoint_context_fn=None, BF16Reader=None)),
                    ('reader_loss', dict(reader_terms=lambda torch_, read, teacher, stats, config:
                                         dict(reader_kl=((read - teacher) ** 2).sum())))):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import vqg4_worker as vw  # noqa: E402
import worker as wk  # noqa: E402

torch.set_default_dtype(torch.float32)


def gptvq_weight(out, n, rows, g, dtype=torch.float32):
    """A dense weight with GPTVQ's layout: per (row group, 256-column span) a codebook of <= 256 4-vectors."""
    in_pad = -(-n // 4) * 4
    spans, groups = -(-in_pad // 256), -(-out // rows)
    scale = torch.rand(spans, generator=g) + 0.5
    w = torch.empty(out, in_pad)
    for gi in range(groups):
        for s in range(spans):
            r0, r1, c0, c1 = gi * rows, min((gi + 1) * rows, out), s * 256, min((s + 1) * 256, in_pad)
            k = int(torch.randint(5, 40, (1,), generator=g))
            book = torch.randint(-127, 128, (k, 4), generator=g).float() / 127 * scale[s]
            idx = torch.randint(0, k, (r1 - r0, (c1 - c0) // 4), generator=g)
            w[r0:r1, c0:c1] = book[idx].reshape(r1 - r0, c1 - c0)
    w = w.to(dtype)
    cfg = dict(vq_dim=4, columns_per_group=256, rows_per_group=rows, column_spans=spans, in_pad=in_pad)
    return w[:, :n].contiguous(), cfg


def make_layer(out, n, rows, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    w, cfg = gptvq_weight(out, n, rows, g, dtype)
    b = torch.randn(out, generator=g) * 0.1
    codes, entries, usage, worst = vw.recover(w, cfg)
    return vw.Layer(w, b.to(dtype), rows, cfg['column_spans'], cfg['in_pad'], codes, entries, usage), cfg, worst


def dense_weight(L, p):
    """The weight the families define: diag(e^row) (W + book offsets) diag(e^col)."""
    n = L.weight.shape[1]
    off = p['book'][L.entry_index()].reshape(L.codes.shape[0], -1)[:, :n]
    return p['row'].exp()[:, None] * (L.weight + off) * p['col'].exp()[None, :]


def zeros(L):
    out, n = L.weight.shape
    return dict(row=torch.zeros(out, requires_grad=True), col=torch.zeros(n, requires_grad=True),
                bias=torch.zeros(out, requires_grad=True), book=torch.zeros(L.books * 256, 4, requires_grad=True))


def test_recover():
    for out, n, rows in ((10, 520, 4), (7, 518, 3), (512, 256, 512)):
        L, cfg, worst = make_layer(out, n, rows, seed=out + n)
        assert worst <= 256 and L.books == -(-out // rows) * cfg['column_spans']
        rebuilt = L.entries.reshape(-1, 4)[L.entry_index()].reshape(out, -1)[:, :n]
        assert torch.equal(rebuilt, L.weight), (out, n, (rebuilt - L.weight).abs().max())
        assert int(L.usage.sum()) == out * cfg['in_pad'] // 4
    codes, _, _, worst = vw.recover(torch.randn(8, 256), dict(vq_dim=4, columns_per_group=256, rows_per_group=8,
                                                               column_spans=1))
    assert codes is None and worst == 512  # > 256 distinct vectors in a block: not coded as configured


def test_vqgrad():
    L, cfg, _ = make_layer(10, 518, 4, seed=3)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, 5, 518, generator=g, requires_grad=True)
    gy = torch.randn(2, 5, 10, generator=g)
    p = zeros(L)
    y = vw.VQGrad.apply(x, p['row'], p['col'], p['bias'], p['book'], L)
    got = torch.autograd.grad(y, [x] + [p[k] for k in ('row', 'col', 'bias', 'book')], gy)
    q = zeros(L)
    x2 = x.detach().clone().requires_grad_(True)
    y2 = F.linear(x2, dense_weight(L, q), L.bias + q['bias'])
    want = torch.autograd.grad(y2, [x2] + [q[k] for k in ('row', 'col', 'bias', 'book')], gy)
    assert torch.equal(y, F.linear(x.detach(), L.weight, L.bias))
    for a, b in zip(got, want):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (a - b).abs().max()
    assert L.tokens == 10 and float(L.absgw) > 0


def test_kmeans():
    g = torch.Generator().manual_seed(5)
    centers = torch.randn(6, 4, generator=g) * 10
    pts = torch.cat([c + 0.1 * torch.randn(50, 4, generator=g) for c in centers])
    a = vw.kmeans(pts, torch.ones(len(pts)), 6, 20)
    for k in range(6):
        assert len(set(a[k * 50:(k + 1) * 50].tolist())) == 1
    assert len(set(a.tolist())) == 6


def test_family_vectors():
    layers = [make_layer(10, 518, 4, seed=s)[0] for s in (6, 7)]
    g = torch.Generator().manual_seed(8)
    metas = dict(layer=[torch.randint(0, 256, (L.books * 256,), generator=g) for L in layers],
                 module=[torch.randint(0, 256, (L.books * 256,), generator=g) for L in layers])
    raw = {p: [dict(row=[torch.randn(10, generator=g) for _ in layers], col=[torch.randn(518, generator=g) for _ in layers],
                    bias=[torch.randn(10, generator=g) for _ in layers],
                    book=[torch.randn(L.books * 256, 4, generator=g) for L in layers]) for _ in range(3)]
           for p in ('Z', 'K')}
    out = {}
    for fam, t, x in vw.family_vectors(raw, layers, metas, dict(module_of=['video', 'video'], modules=['video'])):
        out[(fam, t)] = x
    order = [(p, s) for p in ('Z', 'K') for s in range(3)]
    for t, L in enumerate(layers):
        usage = L.usage.reshape(-1).double()
        a = L.scale.double().repeat_interleave(256)
        for i, (p, s) in enumerate(order):
            b = raw[p][s]['book'][t].double()
            want = torch.where(usage[:, None] > 0, b / usage.clamp_min(1).sqrt()[:, None], torch.zeros_like(b))
            assert torch.allclose(out[('book', t)][i], want.flatten())
            tied = torch.zeros(256, 4, dtype=torch.float64)
            metric = torch.zeros(256, dtype=torch.float64)
            for e in range(L.books * 256):
                m = int(metas['layer'][t][e])
                tied[m] += a[e] * b[e]
                metric[m] += a[e] ** 2 * usage[e]
            want = torch.where(metric[:, None] > 0, tied / metric.clamp_min(1e-300).sqrt()[:, None], torch.zeros_like(tied))
            assert torch.allclose(out[('layer', t)][i], want.flatten())
            assert torch.allclose(out[('row', t)][i], raw[p][s]['row'][t].double() / L.weight.norm(dim=1).double())
    # module: both layers summed into one tensor
    i = 1
    p, s = order[i]
    tied = torch.zeros(256, 4, dtype=torch.float64)
    metric = torch.zeros(256, dtype=torch.float64)
    for t, L in enumerate(layers):
        usage, a = L.usage.reshape(-1).double(), L.scale.double().repeat_interleave(256)
        b = raw[p][s]['book'][t].double()
        tied.index_add_(0, metas['module'][t], a[:, None] * b)
        metric.index_add_(0, metas['module'][t], a ** 2 * usage)
    want = torch.where(metric[:, None] > 0, tied / metric.clamp_min(1e-300).sqrt()[:, None], torch.zeros_like(tied))
    assert torch.allclose(out[('module', 0)][i], want.flatten())


def test_transfer():
    g = torch.Generator().manual_seed(9)
    cases = list(range(2000, 2008))
    samples = [(c, sd, 'suite%d' % (c % 2)) for sd in (0, 1) for c in cases]
    batches = [cases[:4], cases[4:]]
    S, dims = len(samples), (5, 7, 3)
    common = [torch.randn(d, generator=g) for d in dims]
    vecs = {p: [[common[t] * 0.5 + torch.randn(d, generator=g) for t, d in enumerate(dims)] for _ in range(S)]
            for p in vw.PATHS}
    gram = torch.stack([torch.stack([vecs[p][k][t] for p in vw.PATHS for k in range(S)]).double()
                        @ torch.stack([vecs[p][k][t] for p in vw.PATHS for k in range(S)]).double().t()
                        for t in range(len(dims))])
    refs = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    eta = 0.01
    res = vw.transfer(gram, refs, samples, batches, eta, {'F': ['Z', 'K'], 'A': ['A']})
    for src, members in (('F', ['Z', 'K']), ('A', ['A'])):
        num = {p: {'ctx_seed': 0.0} for p in ('Z', 'K', 'A')}
        den = {p: 0.0 for p in ('Z', 'K', 'A')}
        for b in batches:
            idx = [k for k, (c, sd, _) in enumerate(samples) if c in b and sd == 0]
            other = [k for k, (c, sd, _) in enumerate(samples) if c not in b and sd == 1]
            step = []
            for t in range(len(dims)):
                gb = sum(vecs[q][k][t] for q in members for k in idx).double() / len(idx)
                step.append(-eta * float(refs[t]) * gb / gb.norm())
            for p in ('Z', 'K', 'A'):
                dec = lambda k: -sum(float(vecs[p][k][t].double() @ step[t]) for t in range(len(dims)))
                den[p] += sum(dec(k) for k in idx) / len(idx)
                num[p]['ctx_seed'] += sum(dec(k) for k in other) / len(other)
        for p in ('Z', 'K', 'A'):
            got = res['tensor'][src][p]['ctx_seed']
            assert abs(got - num[p]['ctx_seed'] / den[p]) < 1e-6, (src, p, got, num[p]['ctx_seed'] / den[p])
        f = res['tensor'][src]['F']
        assert abs(f['in_decrease'] - (den['Z'] + den['K'])) < 1e-6 * abs(den['Z'] + den['K'])
    assert set(res) == {'tensor', 'global'}
    an = {fam: dict(transfer=res) for fam in vw.FAMILIES}
    rd = vw.reading(an, dict(go_min=0.2, weak_min=0.06))
    assert rd['verdict'] in ('go', 'weak', 'no_go') and rd['best'] in vw.BOOK_FAMILIES


def toy_work(root, beta=0.3, lam=0.5):
    """Imagine: future = tanh(layer0(x_c)); execute: action = layer1(future); a time layer that never runs. The frozen
    reader reads with layer1's master weight."""
    g = torch.Generator().manual_seed(11)
    names = ['video_expert.blocks.0.ffn.0', 'video_expert.time_embedding.0', 'action_expert.blocks.0.ffn.0']
    shapes = [(12, 520), (8, 260), (6, 12)]
    work = object.__new__(vw.G4Work)
    work.base = work.results = root / 'results'
    work.job, work.job_name = 'TEST', 'fwg4_0'
    work.enough = lambda: None
    work.progress = lambda *args, **kwargs: None
    work.tsave = lambda path, value: (Path(path).parent.mkdir(parents=True, exist_ok=True), torch.save(value, path))
    work.counters = dict(parameter_reverse_passes=0)
    work.count = lambda key: None
    arm = root / 'arm'
    modules = {}
    for i, (n, (o, k)) in enumerate(zip(names, shapes)):
        w, cfg = gptvq_weight(o, k, 4, g)
        b = torch.randn(o, generator=g) * 0.1
        mod = types.SimpleNamespace(out_features=o, in_features=k, weight=w, bias=b)
        mod.forward = (lambda x, mod=mod: F.linear(x, mod.weight, mod.bias))
        modules[n] = mod
        wk.save(arm / 'stats' / f'{i:04d}.json', dict(module=n, config={k: v for k, v in cfg.items() if k != 'in_pad'}))
    work.arm_root = lambda a: arm
    work.install = lambda a: None
    work.bank = types.SimpleNamespace(names=names, modules=modules, seen=set())
    m0, m1 = modules[names[0]], modules[names[2]]
    cases = list(range(2000, 2008))
    work.schedule = [cases[2 * k:2 * k + 2] for k in range(4)]
    xs = {c: torch.randn(3, 520, generator=g) for c in cases}
    teach = {(c, sd): dict(future=torch.randn(3, 12, generator=g), action=torch.randn(3, 6, generator=g))
             for c in cases for sd in (0, 1)}
    master1 = m1.weight + 0.05 * torch.randn(m1.weight.shape, generator=g)

    def query(case, seed=0, grad=False):
        with torch.set_grad_enabled(grad):
            future = torch.tanh(m0.forward(xs[case] * (1 + seed)))
            action = m1.forward(future)
        return action, future, None

    def reader(runtime, value, seed, future, checkpointed=True):
        with torch.set_grad_enabled(bool(checkpointed)):
            return m1.forward(future.float()), None

    @contextmanager
    def bf16_reader():
        yield lambda runtime, value, seed, future, checkpointed=True: (F.linear(future, master1, m1.bias), None)

    def loss(a, z, target):
        return ((a - target['action']) ** 2).mean() + lam * ((z - target['future']) ** 2).mean(), None, None

    work.query, work.reader, work.bf16_reader, work.loss = query, reader, bf16_reader, loss
    work.teacher = lambda case, seed=0: teach[(case, seed)]
    work.datum = lambda case: (None, 'task')
    work.rows = {c: dict(sampler_seeds=[0, 1], suite='suite%d' % (c % 2)) for c in cases}
    work.runtime = types.SimpleNamespace(description=None)
    work.cfg = dict(loss=dict(beta=beta))
    work.p = dict(reader_source=dict(action_stats=None, config=None))

    def direct(case, seed):
        """Per path, the gradients w.r.t. each attached layer's families through the dense reparametrization."""
        out = {}
        for path in ('Z', 'K', 'A'):
            ps = {n: zeros(L) for n, L in zip(work.names, work.layers)}
            L0, L1 = work.layers[0], work.layers[1]
            p0, p1 = ps[work.names[0]], ps[work.names[1]]
            t = teach[(case, seed)]
            future = torch.tanh(F.linear(xs[case] * (1 + seed), dense_weight(L0, p0), L0.bias + p0['bias']))
            if path == 'Z':
                val = lam * ((future - t['future']) ** 2).mean()
            elif path == 'K':
                val = beta * ((F.linear(future, master1, m1.bias) - t['action']) ** 2).sum()
            else:
                val = ((F.linear(t['future'], dense_weight(L1, p1), L1.bias + p1['bias']) - t['action']) ** 2).mean()
            keys = [(n, k) for n in work.names for k in ('row', 'col', 'bias', 'book')]
            gs = torch.autograd.grad(val, [ps[n][k] for n, k in keys], allow_unused=True)
            out[path] = {key: (v if v is not None else torch.zeros_like(ps[key[0]][key[1]])) for key, v in zip(keys, gs)}
        return out
    return work, direct


def test_toy_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work, direct = toy_work(root)
        work.attach()
        assert work.names == ['video_expert.blocks.0.ffn.0', 'action_expert.blocks.0.ffn.0']  # time layer excluded
        assert work.modules == ['action', 'video']
        grads, info = work.sample(2003, 1)
        assert info['replay_max_abs'] == 0.0 and info['invariant_rel'] < 1e-5
        want = direct(2003, 1)
        for path in vw.PATHS:
            for t, n in enumerate(work.names):
                for k in ('row', 'col', 'bias', 'book'):
                    a, b = grads[path][k][t].float(), want[path][(n, k)].float()
                    tol = 1e-2 * float(b.abs().max()) + 1e-6 if k == 'book' else 1e-4 * float(b.abs().max()) + 1e-6
                    assert torch.allclose(a, b, atol=tol), (path, n, k, (a - b).abs().max())
        # the future paths never reach the action layer, A never the video layer
        assert float(grads['Z']['book'][1].abs().max()) == 0 and float(grads['K']['row'][1].abs().max()) == 0
        assert float(grads['A']['book'][0].abs().max()) == 0
        work.run = types.MethodType(vw.G4Work.run, work)
        work.run()
        s = json.loads((root / 'results' / 'vqg4' / 'GPTVQ' / 'summary.json').read_text(encoding='utf-8'))
        assert len(s['samples']) == 16 and s['reading']['verdict'] in ('go', 'weak', 'no_go')
        text = (root / 'results' / 'vqg4' / 'GPTVQ' / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert 'G4' in text and '判读' in text
        saved = torch.load(root / 'results' / 'vqg4' / 'GPTVQ' / 'gram.pt', weights_only=True)
        assert saved['gram']['book'].shape == (2, 48, 48) and saved['gram']['module'].shape == (2, 48, 48)
        # resume: all samples on disk, summary removed -> same result without new gradients
        (root / 'results' / 'vqg4' / 'GPTVQ' / 'summary.json').unlink()
        work.sample = lambda *a: (_ for _ in ()).throw(AssertionError('resume must not recompute'))
        work.run()
        s2 = json.loads((root / 'results' / 'vqg4' / 'GPTVQ' / 'summary.json').read_text(encoding='utf-8'))
        assert s2['reading'] == s['reading']


def main():
    tests = [test_recover, test_vqgrad, test_kmeans, test_family_vectors, test_transfer, test_toy_run]
    for t in tests:
        t()
        print('PASS', t.__name__, flush=True)
    print('ALL PASSED')


if __name__ == '__main__':
    main()

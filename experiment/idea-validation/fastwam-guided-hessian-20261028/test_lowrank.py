"""CPU checks for the C2 v2 low-rank gate (lowrank.json, lowrank_worker.py) with a stubbed runtime: the LowRank branch
against autograd, the EoRA-style init against an eigendecomposition reference, the active forward, the staged
gradients of the joint and decoupled objectives against direct autograd on a toy imagine -> execute model, the step
rule and training loop (checkpoint, resume), init / bits / proxy / install, and the decision and report."""
from contextlib import contextmanager
import json
import math
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
import gate_worker as gw  # noqa: E402
import lowrank_worker as lw  # noqa: E402
import worker as wk  # noqa: E402

HOST = json.loads((HERE / 'host.json').read_text(encoding='utf-8'))
R = lw.LR['rank']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def make_work(root, shapes, seed=0, names=None):
    g = torch.Generator().manual_seed(seed)
    work = object.__new__(lw.LowRankWork)
    work.base = work.results = root / 'guided' / 'results'
    work.root = root / 'remote'
    work.baseline, work.host, work.job, work.job_name = root / 'baseline' / 'results', HOST, 'TEST', 'fwlr1_0'
    work.enough = lambda: None
    work.progress = lambda *args, **kwargs: None
    work.atomic_torch_save = lambda _torch, path, value: torch.save(value, path)
    work.counters = dict(parameter_reverse_passes=0)
    work.count = lambda key: None
    names = names or [f'video_expert.blocks.{i}.ffn.0' for i in range(len(shapes))]
    modules = {}
    for n, (o, i) in zip(names, shapes):
        w, b = torch.randn(o, i, generator=g) / i ** 0.5, torch.randn(o, generator=g) * 0.1
        modules[n] = types.SimpleNamespace(out_features=o, in_features=i, weight=w, bias=b,
                                           forward=(lambda x, w=w, b=b: F.linear(x, w, b)))
    work.bank = types.SimpleNamespace(names=names, modules=modules, seen=set())
    work.schedule = [[2000 + 10 * k + j for j in range(10)] for k in range(8)]
    work.attach()
    with torch.no_grad():
        for n in names:
            work.U[n].normal_(0, 0.1, generator=g)
            work.V[n].normal_(0, 0.1, generator=g)
    return work


def test_lowrank_function():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 5, 6, generator=g, requires_grad=True)
    u = torch.randn(4, R, generator=g, requires_grad=True)
    v = torch.randn(R, 6, generator=g, requires_grad=True)
    gy = torch.randn(2, 5, 4, generator=g)
    y = lw.LowRank.apply(x, u, v)
    got = torch.autograd.grad(y, (x, u, v), gy)
    x2, u2, v2 = (t.detach().clone().requires_grad_(True) for t in (x, u, v))
    y2 = F.linear(F.linear(x2, v2), u2)
    want = torch.autograd.grad(y2, (x2, u2, v2), gy)
    assert torch.allclose(y, y2, atol=1e-5)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
    assert lw.LowRank.apply(x.detach().to(torch.bfloat16), u.detach(), v.detach()).dtype == torch.bfloat16


def test_eora():
    g = torch.Generator().manual_seed(1)
    o, n = 7, 9
    dw = torch.randn(o, n, generator=g, dtype=torch.float64)
    x = torch.randn(40, n, generator=g, dtype=torch.float64) * torch.linspace(0.2, 3.0, n, dtype=torch.float64)
    h = x.t() @ x / 40
    u, v, info = lw.eora(dw, h, R, 0.01)
    hd = h + info['damping'] * torch.eye(n, dtype=torch.float64)
    lam, q = torch.linalg.eigh(hd)
    root = q @ torch.diag(lam.sqrt()) @ q.t()
    p, s, qt = torch.linalg.svd(dw @ root)
    ref = (p[:, :R] * s[:R]) @ qt[:R] @ torch.linalg.inv(root)
    assert torch.allclose(u @ v, ref, atol=1e-8), (u @ v - ref).abs().max()
    best = lw.energy(dw - u @ v, hd)
    for k in range(20):
        a = torch.randn(o, R, generator=g, dtype=torch.float64)
        b = torch.randn(R, n, generator=g, dtype=torch.float64)
        assert lw.energy(dw - (u + 1e-3 * a) @ (v + 1e-3 * b), hd) >= best - 1e-12
    assert best < lw.energy(dw, hd)


def test_branch_forward():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(4, 6)])
        name = work.bank.names[0]
        mod = work.bank.modules[name]
        x = torch.randn(3, 6)
        assert torch.equal(mod.forward(x), F.linear(x, mod.weight, mod.bias)) and not work.bank.seen
        work.active = True
        y = mod.forward(x)
        assert name in work.bank.seen
        assert torch.allclose(y, F.linear(x, mod.weight, mod.bias) + x @ work.V[name].t() @ work.U[name].t(), atol=1e-6)


def toy_model(work, beta=0.3, lam=0.5):
    """Imagine: future = layer0(x_c); execute: action = layer1(future); the frozen reader uses layer1's master weight
    only. Teacher action/future fixed per case."""
    g = torch.Generator().manual_seed(7)
    names = work.bank.names
    m0, m1 = (work.bank.modules[n] for n in names)
    xs = {c: torch.randn(5, m0.in_features, generator=g) for c in range(2000, 2080)}
    teach = {c: dict(future=torch.randn(5, m0.out_features, generator=g), action=torch.randn(5, m1.out_features, generator=g))
             for c in xs}
    master1 = m1.weight + 0.05 * torch.randn(m1.weight.shape, generator=g)

    def query(case, seed=0, grad=False):
        with torch.set_grad_enabled(grad):
            future = torch.tanh(m0.forward(xs[case]))
            action = m1.forward(future)
        return action, future, dict(video_steps=10, action_steps=10)

    def reader(runtime, value, seed, future, checkpointed=True):
        with torch.set_grad_enabled(bool(checkpointed)):
            return m1.forward(future.float()), None

    @contextmanager
    def bf16_reader():
        yield lambda runtime, value, seed, future, checkpointed=True: (F.linear(future, master1, m1.bias), None)

    def loss(a, z, target):
        return ((a - target['action']) ** 2).mean() + lam * ((z - target['future']) ** 2).mean(), None, None

    work.query, work.reader, work.bf16_reader, work.loss = query, reader, bf16_reader, loss
    work.teacher = lambda case, seed=0: teach[case]
    work.datum = lambda case: (None, 'task')
    work.rows = {c: dict(sampler_seeds=[0]) for c in xs}
    work.runtime = types.SimpleNamespace(description=None)
    work.cfg = dict(loss=dict(beta=beta))
    work.p = dict(reader_source=dict(action_stats=None, config=None))

    def direct(cases, decoupled):
        total = 0
        for c in cases:
            action, future, _ = query(c, grad=True)
            if decoupled:
                action = m1.forward(teach[c]['future'])
            a, _, _ = loss(action, future, teach[c])
            k = ((F.linear(future, master1, m1.bias) - teach[c]['action']) ** 2).sum()
            total = total + a + beta * k
        return total / len(cases)
    return direct


def test_staged_gradients():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(6, 5), (3, 6)])
        direct = toy_model(work)
        work.active = True
        cases = list(range(2000, 2004))
        for key, decoupled in (('J', False), ('D', True)):
            for p in work.parameters():
                p.grad = None
            out = work.batch_for(key)(cases, grad=True)
            got = [p.grad.clone() for p in work.parameters()]
            for p in work.parameters():
                p.grad = None
            ref = direct(cases, decoupled)
            want = torch.autograd.grad(ref, work.parameters(), allow_unused=True)
            assert abs(out['total'] - float(ref)) < 1e-5, (key, out['total'], float(ref))
            for a, b in zip(got, want):
                assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (key, (a - b).abs().max())
            assert out['replay_max_abs'] == 0.0
        # the decoupled action term reaches layer 0 (imagine) only through Z and K, never through the action
        for p in work.parameters():
            p.grad = None
        work.cfg['loss']['beta'] = 0.0
        lam_free = toy_model(work, beta=0.0, lam=0.0)
        work.batch_decoupled(cases, grad=True)
        name0 = work.bank.names[0]
        assert float(work.U[name0].grad.abs().max()) == 0.0 and float(work.V[name0].grad.abs().max()) == 0.0
        del lam_free


def toy_loss(work, target):
    def batch(cases, grad):
        with torch.set_grad_enabled(grad):
            total = sum(((work.U[n] @ work.V[n] - target[n]) ** 2).sum() for n in work.bank.names)
        if grad:
            total.backward()
        return dict(total=float(total), az=float(total), k=0.0)
    return batch


def test_step_and_train():
    shapes = [(4, 6), (3, 5)]
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), shapes)
        target = {n: torch.randn(o, i) * 0.05 for n, (o, i) in zip(work.bank.names, shapes)}
        work.batch = work.batch_decoupled = toy_loss(work, target)
        start = work.snapshot()
        out = work.step_check('J')
        assert all(torch.equal(work.U[n], start[n]['U']) and torch.equal(work.V[n], start[n]['V']) for n in work.bank.names)
        assert out['chosen'] == lw.tw.first_step_choice(out['rows'], lw.LR['training']['min_realized_fraction'])
        assert out['chosen'] is not None
        for p in work.parameters():
            p.grad = None
        base = work.batch(None, True)
        unit = sum(float(p.norm()) * float(p.grad.norm()) for p in work.parameters())
        for p in work.parameters():
            p.grad = None
        for row in out['rows']:
            assert abs(row['predicted'] - row['lr'] * unit) < 1e-6
            moved = {n: {k: v.clone() for k, v in f.items()} for n, f in start.items()}
            us, vs = [moved[n]['U'] for n in work.bank.names], [moved[n]['V'] for n in work.bank.names]
            grads = []
            for n in work.bank.names:
                ul, vl = start[n]['U'].clone().requires_grad_(True), start[n]['V'].clone().requires_grad_(True)
                ((ul @ vl - target[n]) ** 2).sum().backward()
                grads.append((ul.grad, vl.grad))
            lw.relative_step(us + vs, [a for a, _ in grads] + [b for _, b in grads], row['lr'])
            after = sum(float(((moved[n]['U'] @ moved[n]['V'] - target[n]) ** 2).sum()) for n in work.bank.names)
            assert abs(row['realized'] - (base['total'] - after)) < 1e-5
        state = work.train('D', out['chosen'])
        losses = [h['before']['total'] for h in state['history']]
        assert len(losses) == 8 and min(losses[1:]) < losses[0]
        snap = work.snapshot()
        again = make_work(Path(tmp), shapes, seed=3)
        again.batch = again.batch_decoupled = toy_loss(again, target)
        state2 = again.train('D', out['chosen'])
        assert len(state2['history']) == 8
        assert all(torch.equal(again.U[n], snap[n]['U']) for n in again.bank.names)


def build(work, shapes, arm=None):
    g = torch.Generator().manual_seed(4)
    prep, base = work.baseline / 'prep', work.arm_root(arm or lw.LR['baseline_arm'])
    for i, (name, (o, n)) in enumerate(zip(work.bank.names, shapes)):
        w = torch.randn(o, n, generator=g)
        x = torch.randn(64, n, generator=g)
        tsave(prep / 'masters' / f'{i:04d}.pt', dict(module=name, weight=w.to(torch.bfloat16)))
        tsave(prep / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=x.t() @ x / 64))
        q = (w + torch.randn(o, n, generator=g) * 0.3).to(torch.bfloat16)
        tsave(base / 'weights' / f'{i:04d}.pt', dict(module=name, weight=q))
        save(base / 'stats' / f'{i:04d}.json', dict(bits_total=2.0 * o * n, seconds=1.0))


def test_init_complete_install():
    shapes = [(8, 6), (5, 10)]
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), shapes)
        build(work, shapes)
        factors = work.init_factors()
        prep, base = work.baseline / 'prep', work.arm_root(lw.LR['baseline_arm'])
        for i, name in enumerate(work.bank.names):
            w = torch.load(prep / 'masters' / f'{i:04d}.pt', weights_only=True)['weight'].float()
            q = torch.load(base / 'weights' / f'{i:04d}.pt', weights_only=True)['weight'].float()
            h = torch.load(prep / 'hessians' / f'{i:04d}.pt', weights_only=True)['hessian']
            u, v, _ = lw.eora(w - q, h, R, lw.LR['init']['damping'])
            assert torch.allclose(factors[name]['U'] @ factors[name]['V'], u @ v, atol=1e-5)
        arm = lw.ARMS['E']
        tsave(work.lr_dir(arm) / 'factors.pt', {n: {k: v.to(torch.bfloat16) for k, v in f.items()} for n, f in factors.items()})
        done = work.complete(arm)
        weights = sum(o * n for o, n in shapes)
        assert abs(done['bpw'] - (2.0 * weights + 16 * R * sum(o + n for o, n in shapes)) / weights) < 1e-12
        assert done['proxy_nmse_weighted'] < done['baseline_proxy_U_weighted']
        installed = []
        original = gw.GateWork.install
        gw.GateWork.install = lambda self, a: installed.append(a)
        try:
            work.install(arm)
            assert work.active and installed == [lw.LR['baseline_arm']]
            n0 = work.bank.names[0]
            assert torch.equal(work.U[n0], factors[n0]['U'].to(torch.bfloat16).float())
            work.install('QTIP_G_W2')
            assert not work.active and installed[-1] == 'QTIP_G_W2'
        finally:
            gw.GateWork.install = original


def test_decide_and_report():
    def held(mixed, future, swap=0.5):
        return dict(mixed_loss=mixed, future_latent_nmse=future, first10motor_rmse=0.03, first10motor_mse=1.0,
                    first10motor_mse_swap=swap, first10gripper_rmse=0.08, first10motor_rmse_swap=0.02)
    fit = dict(mixed_loss=0.0025, future_latent_nmse=0.038, first10motor_rmse=0.029)
    step = dict(rows=[dict(lr=0.02, predicted=1.0, realized=0.7)], chosen=0.02, top_layers=[dict(module='m', share=0.3)])
    quant = dict(bpw=2.05, proxy_nmse_weighted=0.004)
    rows = dict(U=dict(tuned=False, fit=fit, heldout=held(0.00336, 0.0533)),
                E=dict(tuned=False, fit=fit, quant=quant, heldout=held(0.00330, 0.0510)),
                J=dict(tuned=True, eta=0.02, fit=fit, quant=quant, step=step, heldout=held(0.00310, 0.0500),
                       training=[dict(before=dict(total=0.003))]),
                D=dict(tuned=True, eta=0.02, fit=fit, quant=quant, step=step, heldout=held(0.00305, 0.0460),
                       training=[dict(before=dict(total=0.003))]))
    d = lw.decide(rows, lw.LR['decision'])
    assert d['go'] and d['decoupling_increment'] and not d['arms']['J']['passed'] and d['arms']['D']['passed']
    rows['D'] = dict(tuned=False, step=dict(step, chosen=None))
    d2 = lw.decide(rows, lw.LR['decision'])
    assert not d2['go'] and not d2['decoupling_increment']
    text = lw.report(dict(pbs_jobid='X', lowrank=lw.LR, rows=rows, decision=d2))
    assert '未训练' in text and 'D 未通过，不上闭环' in text


def test_diag_analysis():
    g = torch.Generator().manual_seed(11)
    names = ['video_expert.blocks.0.ffn.0.U', 'video_expert.time_projection.1.U', 'action_expert.blocks.0.ffn.0.V',
             'proprio_encoder.V']
    sizes = [7, 3, 5, 2]
    cases = list(range(2000, 2008))
    samples = [(c, sd, 'a' if c < 2004 else 'b') for sd in (0, 1) for c in cases]
    batches = [cases[:4], cases[4:]]
    shared = [torch.randn(n, generator=g, dtype=torch.float64) for n in sizes]
    fs = [[0.7 * shared[t] + torch.randn(n, generator=g, dtype=torch.float64) for t, n in enumerate(sizes)] for _ in samples]
    As = [[torch.randn(n, generator=g, dtype=torch.float64) * (0.0 if t == 1 else 1.0) for t, n in enumerate(sizes)]
          for _ in samples]
    gram = torch.stack([torch.stack([v[t] for v in fs + As]) @ torch.stack([v[t] for v in fs + As]).t()
                        for t in range(len(sizes))])
    P = [torch.randn(n, generator=g, dtype=torch.float64) for n in sizes]
    norms = torch.tensor([float(p.norm()) for p in P], dtype=torch.float64)
    eta = 0.01
    out = lw.diag_analysis(gram, norms, names, samples, batches, eta)
    index = {(c, sd): i for i, (c, sd, _) in enumerate(samples)}
    sums = {k: {p: 0.0 for p in ('F', 'A', 'total')} for k in ('in', 'seed', 'ctx', 'ctx_seed')}
    for bi, b in enumerate(batches):
        idx = [index[(c, 0)] for c in b]
        gbar = [sum(fs[i][t] + As[i][t] for i in idx) / len(idx) for t in range(len(sizes))]
        moved = [p.clone() for p in P]
        lw.relative_step(moved, gbar, eta)
        delta = [m - p for m, p in zip(moved, P)]
        row = out['batches'][bi]
        assert abs(row['unit'] - sum(float(p.norm() * x.norm()) for p, x in zip(P, gbar))) < 1e-9
        cats = {}
        for k, (c, sd, _) in enumerate(samples):
            cats.setdefault(('in' if sd == 0 else 'seed') if c in b else ('ctx' if sd == 0 else 'ctx_seed'), []).append(k)
        for key, ks in cats.items():
            dec_f = [-sum(float(fs[k][t] @ delta[t]) for t in range(len(sizes))) for k in ks]
            dec_a = [-sum(float(As[k][t] @ delta[t]) for t in range(len(sizes))) for k in ks]
            want = dict(F=sum(dec_f) / len(ks), A=sum(dec_a) / len(ks))
            want['total'] = want['F'] + want['A']
            for p, v in want.items():
                assert abs(row[key][p] - v) < 1e-9, (key, p, row[key][p], v)
                sums[key][p] += v
            assert abs(sum(row[key]['F_by_group'].values()) - want['F']) < 1e-9
        fb = torch.stack([torch.cat(fs[i]) for i in idx])
        assert abs(out['snr']['F'][bi] - float(fb.mean(0).square().sum() / fb.square().sum(1).mean())) < 1e-9
    for key in ('seed', 'ctx', 'ctx_seed'):
        for p in ('F', 'A', 'total'):
            assert abs(out['T'][key][p] - sums[key][p] / sums['in'][p]) < 1e-9
    whole = [torch.cat(v) for v in fs]
    cos = [float(whole[index[(c, 0)]] @ whole[index[(c, 1)]] / whole[index[(c, 0)]].norm() / whole[index[(c, 1)]].norm())
           for c in cases]
    assert abs(out['cosines']['all/F']['seed'] - sum(cos) / len(cos)) < 1e-9
    assert out['pair_counts'] == dict(seed=8, suite=2 * 2 * 6 + 2 * 4 * 3, other=2 * 4 * 4 * 2)
    assert 'video_time/A' not in out['cosines'] and abs(out['energy']['all/F'] - 1) < 1e-12
    assert lw.tensor_group('action_expert.time_projection.1.V') == 'action_time'
    # one direction for every sample: everything shares the step
    one = [[shared[t] for t in range(len(sizes))] for _ in samples]
    same = torch.stack([torch.stack([v[t] for v in one + one]) @ torch.stack([v[t] for v in one + one]).t()
                        for t in range(len(sizes))])
    out2 = lw.diag_analysis(same, norms, names, samples, batches, eta)
    assert all(abs(x - 1) < 1e-9 for x in out2['snr']['F'])
    assert all(abs(out2['T'][k][p] - 1) < 1e-9 for k in out2['T'] for p in ('F', 'A', 'total'))


def diag_toy(tmp):
    """A toy work with the class teacher (seed 0 from the stubbed cache, others from diag_teachers), suites, two
    sampler seeds per context, a stubbed host install, metrics, and E / J / D factors on disk."""
    work = make_work(tmp, [(6, 5), (3, 6)])
    toy_model(work)
    cache = {c: work.teacher(c) for c in range(2000, 2080)}
    del work.teacher
    work.active = False
    for c in lp_cases()[:lw.LR['diag']['teacher_check']['cases']]:
        action, future, _ = work.query(c)
        cache[c]['action'], cache[c]['future'] = action.detach(), future.detach()
    work.rows = {c: dict(sampler_seeds=[10 * c, 10 * c + 1], suite=f'suite{(c - 2000) // 20}') for c in range(2000, 2080)}

    def metrics(action, future, target):
        z = float((future - target['future']).square().sum() / target['future'].square().sum())
        return dict(mixed_loss=float((action - target['action']).square().mean() + 0.5 * (future - target['future']).square().mean()),
                    first10motor_rmse=float((action - target['action']).square().mean().sqrt()), future_latent_nmse=z)
    work.metrics = metrics
    g = torch.Generator().manual_seed(5)
    base = work.snapshot()
    for k, scale in (('E', 0.0), ('J', 0.05), ('D', 0.08)):
        f = {n: {kk: (v + scale * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for kk, v in x.items()}
             for n, x in base.items()}
        tsave(work.lr_dir(lw.ARMS[k]) / 'factors.pt', f)
    return work, cache


def lp_cases():
    return list(range(2000, 2080))


def test_sample_grads():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(6, 5), (3, 6)])
        direct = toy_model(work)
        work.active = True
        case, params = 2003, work.parameters()
        g, st, _ = work.sample_grads(case, 0)
        gf, ga = g['F'], g['A']
        only, _, future = work.sample_grads(case, 0, paths=('F',))
        assert set(only) == {'F'} and all(torch.equal(x, y) for x, y in zip(only['F'], gf))
        assert torch.equal(future, work.query(case)[1].detach())
        ref = direct([case], True)
        want = torch.autograd.grad(ref, params, allow_unused=True)
        for x, y, w in zip(gf, ga, want):
            assert torch.allclose(x + y, torch.zeros_like(x) if w is None else w, rtol=1e-4, atol=1e-6)
        t = work.teacher(case)
        _, future, _ = work.query(case, grad=True)
        with work.bf16_reader() as reader:
            read, _ = reader(None, None, 0, future)
        z = 0.5 * (future - t['future']).square().mean() + 0.3 * (read - t['action']).square().sum()
        for x, w in zip(gf, torch.autograd.grad(z, params, allow_unused=True)):
            assert torch.allclose(x, torch.zeros_like(x) if w is None else w, rtol=1e-4, atol=1e-6)
        n = len(work.bank.names)
        assert float(ga[0].abs().max()) == 0.0 and float(ga[n].abs().max()) == 0.0  # A never reaches the imagine layer
        assert st['replay_max_abs'] == 0.0 and abs(st['az'] + 0.3 * st['k'] - float(ref)) < 1e-5


def test_diag_teachers_and_flow():
    original = (getattr(lw.lp.Work, 'teacher', None), gw.GateWork.install)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work, cache = diag_toy(Path(tmp))
            lw.lp.Work.teacher = lambda self, case, seed=0: cache[case]
            gw.GateWork.install = lambda self, arm: None
            work.diag_teachers()
            assert wk.read(work.lr_dir('diag') / 'teacher_check.json')['worst'] == 0.0
            action, future, _ = work.query(2033, 1)
            t = work.teacher(2033, 1)
            assert torch.equal(t['future'], future.detach()) and t['sampler_seed'] == 20331
            assert work.teacher(2033, 0) is cache[2033]
            for k in ('E', 'J', 'D'):
                work.fit_arm(lw.ARMS[k])
            work.install(lw.ARMS['E'])
            work.step_check('D')
            work.diag()
            s = wk.read(work.lr_dir('diag') / 'summary.json')
            assert s['predicted_check']['passed'] and abs(s['predicted_check']['rel']) < 1e-5
            assert set(s['fits']) == {'E', 'J', 'D'} and set(s['fits']['D']) == {'0', '1'}
            g = torch.load(work.lr_dir('diag') / 'gram.pt', weights_only=True)
            samples = [tuple(x) for x in g['samples']]
            assert len(samples) == 80 and g['gram'].shape == (4, 160, 160)
            work.restore(torch.load(work.lr_dir(lw.ARMS['E']) / 'factors.pt', weights_only=True))
            work.active = True
            g41, _, _ = work.sample_grads(*samples[41][:2])
            g3, _, _ = work.sample_grads(*samples[3][:2])
            want = float(g41['F'][2].double().reshape(-1) @ g3['A'][2].double().reshape(-1))
            assert abs(float(g['gram'][2, 41, 80 + 3]) - want) <= 1e-5 * abs(want) + 1e-10
            text = (work.lr_dir('diag') / 'RESULTS.zh.md').read_text(encoding='utf-8')
            assert '判读（预先登记）' in text and '核对' in text
        with tempfile.TemporaryDirectory() as tmp:
            work, cache = diag_toy(Path(tmp))
            cache[2000]['future'] = cache[2000]['future'] + 1.0
            try:
                work.diag_teachers()
            except RuntimeError:
                assert wk.read(work.lr_dir('diag') / 'teacher_check.json')['worst'] > lw.LR['diag']['teacher_check']['future_nmse_max']
            else:
                raise AssertionError('a differing regenerated teacher passed')
    finally:
        if original[0] is None:
            del lw.lp.Work.teacher
        else:
            lw.lp.Work.teacher = original[0]
        gw.GateWork.install = original[1]


LAT = (1, 4, 3, 2, 2)  # toy future latent [1, C, T, H, W]
TF_NAMES = ['video_expert.blocks.0.self_attn.q', 'video_expert.blocks.0.ffn.2', 'proprio_encoder',
            'action_expert.blocks.0.ffn.0']


class FakeIDM:
    """infer_action shaped as FastWAMIDM's: noise from the seed, frame 0 from the image, the prompt context plus the
    proprio Linear, Euler video steps (x + delta v) with the frame-0 clamp after each, then the action stage (counted)."""

    def __init__(self, modules):
        self.m0, self.m1, self.proprio_encoder, _ = (modules[n] for n in TF_NAMES)
        self.infer_video_scheduler = types.SimpleNamespace(step=lambda model_output, delta, sample: sample + model_output * delta)
        self.reached_action = 0

    def _append_proprio_to_context(self, context, context_mask, proprio):
        return context + self.proprio_encoder.forward(proprio), context_mask

    def _denoise_video(self, *, latents_video, timestep_video, context, context_mask, video_self_attn_mask,
                       fuse_vae_embedding_in_latents):
        c = LAT[1]
        tok = latents_video.permute(0, 2, 3, 4, 1).reshape(-1, c)
        h = torch.tanh(self.m0.forward(tok) + context) * (1 + timestep_video / 1000)
        return self.m1.forward(h).reshape(1, *LAT[2:], c).permute(0, 4, 1, 2, 3)

    def schedule(self, n):
        return torch.linspace(1000, 100, n), torch.full((n,), -0.09)

    def infer_action(self, prompt, input_image, action_horizon, num_video_frames, proprio=None, num_inference_steps=10,
                     sigma_shift=None, seed=None, rand_device='cpu', tiled=False, compile_action_infer=False):
        x = torch.randn(LAT, generator=torch.Generator().manual_seed(seed))
        x[:, :, 0:1] = input_image
        context, mask = self._append_proprio_to_context(context=torch.full((1, 6), 0.1), context_mask=None, proprio=proprio)
        for t, d in zip(*self.schedule(num_inference_steps)):
            v = self._denoise_video(latents_video=x, timestep_video=t.reshape(1), context=context, context_mask=mask,
                                    video_self_attn_mask=None, fuse_vae_embedding_in_latents=True)
            x = self.infer_video_scheduler.step(v, d, x)
            x[:, :, 0:1] = input_image
        self.reached_action += 1


def nmse(s, t):
    return ((s.float()[:, :, 1:] - t.float()[:, :, 1:]) ** 2).sum() / (t.float()[:, :, 1:] ** 2).sum().clamp_min(1e-12)


def tf_toy(tmp, beta=0.3, lam=0.5):
    """A toy work for the teacher-forced objective: FakeIDM's video stage through the bank's Linears, a differentiable
    rollout query that matches it, a reader pooling frames 1: through the action Linear (the frozen BF16 reader uses a
    perturbed master), C80 teachers at seed 0 from the BF16 query, suites, two sampler seeds, E factors on disk."""
    work = make_work(tmp, [(6, 4), (4, 6), (6, 3), (5, 4)], names=TF_NAMES)
    model = FakeIDM(work.bank.modules)
    ma = work.bank.modules[TF_NAMES[3]]
    g = torch.Generator().manual_seed(9)
    master = ma.weight + 0.05 * torch.randn(ma.weight.shape, generator=g)
    data = {c: dict(image=torch.randn(1, LAT[1], 1, *LAT[3:], generator=g), proprio=torch.randn(1, 3, generator=g))
            for c in range(2000, 2080)}
    work.rows = {c: dict(sampler_seeds=[10 * c, 10 * c + 1], suite=f'suite{(c - 2000) // 20}') for c in data}
    work.runtime = types.SimpleNamespace(model=model, description=None, ev=types.SimpleNamespace(DEFAULT_PROMPT='{task}'))
    work.sampler = types.SimpleNamespace(_require_compute_allocation=lambda: None,
                                         _regular_inputs=lambda torch_, d: dict(image=d['image'], proprio=d['proprio']),
                                         _raw_idm_infer=lambda m: type(m).infer_action)
    work.datum = lambda case: (data[case], 'task')
    work.latent_reference = types.SimpleNamespace(future_nmse_loss=lambda torch_, s, t: nmse(s, t))
    work.cfg = dict(loss=dict(beta=beta, **{'lambda': lam}))
    work.p = dict(reader_source=dict(action_stats=None, config=None))

    def pool(future, weight):
        return F.linear(future.float()[:, :, 1:].mean(dim=(2, 3, 4)), weight, ma.bias)

    def query(case, seed=0, grad=False):
        d = data[case]
        with torch.set_grad_enabled(grad):
            x = torch.randn(LAT, generator=torch.Generator().manual_seed(work.rows[case]['sampler_seeds'][seed]))
            x[:, :, 0:1] = d['image']
            context, mask = model._append_proprio_to_context(context=torch.full((1, 6), 0.1), context_mask=None,
                                                             proprio=d['proprio'])
            for t, dl in zip(*model.schedule(10)):
                v = model._denoise_video(latents_video=x, timestep_video=t.reshape(1), context=context, context_mask=mask,
                                         video_self_attn_mask=None, fuse_vae_embedding_in_latents=True)
                x = torch.cat((d['image'], model.infer_video_scheduler.step(v, dl, x)[:, :, 1:]), dim=2)
            action = ma.forward(x[:, :, 1:].mean(dim=(2, 3, 4)))
        return action, x, dict(video_steps=10, action_steps=10)  # as the sampler: frame 0 clamped to the observation

    def observed(value, future):  # the frozen reader's check: the supplied frame 0 is the observation
        if not torch.equal(future[:, :, :1].to(value['image'].dtype), value['image']):
            raise RuntimeError('Future injection or observation-frame clamp verification failed')

    def reader(runtime, value, seed, future, checkpointed=True):
        observed(value, future)
        with torch.set_grad_enabled(bool(checkpointed)):
            return ma.forward(future.float()[:, :, 1:].mean(dim=(2, 3, 4))), None

    def frozen(runtime, value, seed, future, checkpointed=True):
        observed(value, future)
        return pool(future, master), None

    @contextmanager
    def bf16_reader():
        yield frozen

    def loss(a, z, target):
        a_term = ((a - target['action']) ** 2).mean()
        return a_term + lam * nmse(z, target['future']), a_term, nmse(z, target['future'])

    work.query, work.reader, work.bf16_reader, work.loss = query, reader, bf16_reader, loss
    work.active = False
    cache = {}
    for c in data:
        action, future, _ = query(c)
        cache[c] = dict(action=action.detach(), future=future.detach())
    gen = torch.Generator().manual_seed(5)
    f = {n: {k: (0.2 * torch.randn(v.shape, generator=gen)).to(torch.bfloat16) for k, v in x.items()}
         for n, x in work.snapshot().items()}
    tsave(work.lr_dir(lw.ARMS['E']) / 'factors.pt', f)

    def direct(case, seed, states):
        """lambda Z(x~) + beta K(x~) by plain autograd: the student's velocities at the teacher states."""
        d = data[case]
        context, mask = model._append_proprio_to_context(context=torch.full((1, 6), 0.1), context_mask=None,
                                                         proprio=d['proprio'])
        disp = 0
        for x, t, v, dl in zip(states['x'], states['t'], states['v'], states['delta']):
            vs = model._denoise_video(latents_video=x, timestep_video=t, context=context, context_mask=mask,
                                      video_self_attn_mask=None, fuse_vae_embedding_in_latents=True)
            disp = disp + dl * (vs - v)[:, :, 1:]
        base = states['future'].float()
        xt = torch.cat((base[:, :, :1], base[:, :, 1:] + disp), dim=2)
        t = work.teacher(case, seed)
        return lam * nmse(xt, t['future']) + beta * ((pool(xt, master) - t['action']) ** 2).sum(), xt
    return work, cache, direct


def test_video_stage_and_tf_grads():
    original = getattr(lw.lp.Work, 'teacher', None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work, cache, direct = tf_toy(Path(tmp))
            lw.lp.Work.teacher = lambda self, case, seed=0: cache[case]
            model, case = work.runtime.model, 2013
            loop = work.video_stage(case, 0)
            assert model.reached_action == 0 and len(loop['x']) == 10 and len(loop['delta']) == 10
            assert torch.equal(loop['future'], cache[case]['future'])
            assert all(torch.equal(loop['x'][i + 1][:, :, 1:], loop['out'][i][:, :, 1:]) for i in range(9))
            assert all(torch.equal(loop['x'][i + 1][:, :, :1], work.datum(case)[0]['image']) for i in range(9))
            states = dict(x=loop['x'], v=[v.detach() for v in loop['v']], t=loop['t'], delta=loop['delta'], future=loop['future'])
            again = work.video_stage(case, 0, states)  # teacher forcing with the masters returns the recorded velocities
            assert all(torch.equal(a, b) for a, b in zip(again['v'], states['v']))
            xt = lw.tf_future(states['future'], again['v'], states['v'], states['delta'])
            assert torch.equal(xt, states['future'].float())
            work.restore(torch.load(work.lr_dir(lw.ARMS['E']) / 'factors.pt', weights_only=True))
            work.active = True
            params = work.parameters()
            grads, values, parts = work.tf_grads(case, 0, states)
            ref, xt_ref = direct(case, 0, states)
            want = torch.autograd.grad(ref, params, allow_unused=True)
            assert abs(values['total'] - float(ref)) < 1e-6 * max(1.0, abs(float(ref))), (values['total'], float(ref))
            assert torch.allclose(parts['xt'], xt_ref.detach(), atol=1e-6)
            for a, b in zip(grads, want):
                assert torch.allclose(a, torch.zeros_like(a) if b is None else b, rtol=1e-4, atol=1e-7), (a - b).abs().max()
            assert any(float(a.abs().max()) > 0 for a in grads[:4]) and values['replay_max_abs'] == 0.0
            assert torch.allclose(sum(parts['steps']), (parts['xt'] - states['future'].float())[:, :, 1:], atol=1e-6)
            assert model.reached_action == 0
            check = work.tf_step_check(case, 0, states, grads, values)
            assert all(torch.equal(work.U[n], torch.load(work.lr_dir(lw.ARMS['E']) / 'factors.pt', weights_only=True)[n]['U'].float())
                       for n in work.bank.names)
            assert check['realized'] > 0 and abs(check['ratio'] - 1) < 0.2, check
    finally:
        if original is None:
            del lw.lp.Work.teacher
        else:
            lw.lp.Work.teacher = original


def test_tf_analysis():
    g = torch.Generator().manual_seed(12)
    names = ['video_expert.blocks.0.ffn.0.U', 'video_expert.time_projection.1.U', 'proprio_encoder.V']
    sizes = [6, 3, 2]
    cases = list(range(2000, 2008))
    samples = [(c, sd, 'a' if c < 2004 else 'b') for sd in (0, 1) for c in cases]
    batches = [cases[:4], cases[4:]]
    shared = [torch.randn(n, generator=g, dtype=torch.float64) for n in sizes]
    tf = [[0.8 * shared[t] + torch.randn(n, generator=g, dtype=torch.float64) for t, n in enumerate(sizes)] for _ in samples]
    ro = [[0.5 * tf[k][t] + torch.randn(n, generator=g, dtype=torch.float64) for t, n in enumerate(sizes)]
          for k in range(len(samples))]
    gram = torch.stack([torch.stack([v[t] for v in tf + ro]) @ torch.stack([v[t] for v in tf + ro]).t()
                        for t in range(len(sizes))])
    P = [torch.randn(n, generator=g, dtype=torch.float64) for n in sizes]
    norms = torch.tensor([float(p.norm()) for p in P], dtype=torch.float64)
    eta = 0.01
    out = lw.tf_analysis(gram, norms, names, samples, batches, eta)
    index = {(c, sd): i for i, (c, sd, _) in enumerate(samples)}
    paths = dict(tf=tf, ro=ro)
    sums = {src: {cat: {p: 0.0 for p in paths} for cat in ('in', 'seed', 'ctx', 'ctx_seed')} for src in paths}
    for bi, b in enumerate(batches):
        idx = [index[(c, 0)] for c in b]
        cats = {}
        for k, (c, sd, _) in enumerate(samples):
            cats.setdefault(('in' if sd == 0 else 'seed') if c in b else ('ctx' if sd == 0 else 'ctx_seed'), []).append(k)
        for src, vecs in paths.items():
            gbar = [sum(vecs[i][t] for i in idx) / len(idx) for t in range(len(sizes))]
            moved = [p.clone() for p in P]
            lw.relative_step(moved, gbar, eta)
            delta = [m - p for m, p in zip(moved, P)]
            row = out['batches'][bi][src]
            assert abs(row['unit'] - sum(float(p.norm() * x.norm()) for p, x in zip(P, gbar))) < 1e-9
            for key, ks in cats.items():
                for p, tv in paths.items():
                    want = sum(-sum(float(tv[k][t] @ delta[t]) for t in range(len(sizes))) for k in ks) / len(ks)
                    assert abs(row[key][p] - want) < 1e-9, (src, key, p)
                    sums[src][key][p] += want
    for src in paths:
        for key in ('seed', 'ctx', 'ctx_seed'):
            assert abs(out['T'][src][key] - sums[src][key][src] / sums[src]['in'][src]) < 1e-9
        for key in ('in', 'seed', 'ctx', 'ctx_seed'):
            assert abs(out['G'][src][key] - sums[src][key]['ro'] / sums['ro']['in']['ro']) < 1e-9
    assert abs(out['G']['ro']['in'] - 1) < 1e-12
    whole = {p: [torch.cat(v) for v in vecs] for p, vecs in paths.items()}
    self_cos = [float(whole['tf'][k] @ whole['ro'][k] / whole['tf'][k].norm() / whole['ro'][k].norm()) for k in range(len(samples))]
    assert abs(out['cross']['all']['self'] - sum(self_cos) / len(self_cos)) < 1e-9
    seed_cos = [float(whole['tf'][index[(c, 0)]] @ whole['tf'][index[(c, 1)]] / whole['tf'][index[(c, 0)]].norm()
                      / whole['tf'][index[(c, 1)]].norm()) for c in cases]
    assert abs(out['cosines']['all/tf']['seed'] - sum(seed_cos) / len(seed_cos)) < 1e-9
    fb = torch.stack([whole['ro'][index[(c, 0)]] for c in batches[1]])
    assert abs(out['snr']['ro'][1] - float(fb.mean(0).square().sum() / fb.square().sum(1).mean())) < 1e-9
    r = lw.tf_reading(out, lw.LR['tfdiag']['reading'])
    assert r['go'] == (r['shared'] and r['reaches']) and r['T_tf_ctx_seed'] == out['T']['tf']['ctx_seed']


def test_tfdiag_flow():
    original = (getattr(lw.lp.Work, 'teacher', None), gw.GateWork.install)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work, cache, direct = tf_toy(Path(tmp))
            lw.lp.Work.teacher = lambda self, case, seed=0: cache[case]
            gw.GateWork.install = lambda self, arm: None
            work.diag_teachers()
            work.diag_grads()  # diag's Gram at E: the ro block must reproduce its F block
            work.tfdiag()
            root = work.lr_dir('tfdiag')
            s = wk.read(root / 'summary.json')
            sc = s['state_check']
            assert (sc['future_nmse_worst'] == 0.0 and sc['chain_worst'] == 0.0 and sc['frame0_worst'] == 0.0
                    and sc['bf16_tf_max_abs'] == 0.0)
            assert s['ro_check']['passed'] and s['ro_check']['rel'] < 1e-9, s['ro_check']
            assert len(s['rows']) == 80 and set(s['reading']) >= {'go', 'shared', 'reaches'}
            assert s['step_check']['realized'] > 0
            g = torch.load(root / 'gram.pt', weights_only=True)
            samples = [tuple(x) for x in g['samples']]
            assert samples == work.tf_samples() and g['gram'].shape == (8, 160, 160)
            work.restore(torch.load(work.lr_dir(lw.ARMS['E']) / 'factors.pt', weights_only=True))
            work.active = True
            gt, _, _ = work.tf_grads(samples[41][0], samples[41][1], work.load_state(*samples[41][:2]))
            g3, _, _ = work.sample_grads(*samples[3][:2], paths=('F',))
            want = float(gt[5].double().reshape(-1) @ g3['F'][5].double().reshape(-1))
            assert abs(float(g['gram'][5, 41, 80 + 3]) - want) <= 1e-5 * abs(want) + 1e-10
            ref, _ = direct(samples[41][0], samples[41][1], work.load_state(*samples[41][:2]))
            assert abs(s['rows'][41]['total_tf'] - float(ref)) < 1e-6 * max(1.0, abs(float(ref)))
            text = (root / 'RESULTS.zh.md').read_text(encoding='utf-8')
            assert '判读（预先登记）' in text and '核对' in text
        with tempfile.TemporaryDirectory() as tmp:
            work, cache, _ = tf_toy(Path(tmp))
            lw.lp.Work.teacher = lambda self, case, seed=0: cache[case]
            work.diag_teachers()
            cache[2005]['future'] = cache[2005]['future'] + 0.5
            try:
                work.tf_states(work.tf_samples())
            except RuntimeError:
                assert wk.read(work.lr_dir('tfdiag') / 'state_check.json')['future_nmse_worst'] > 1e-6
            else:
                raise AssertionError('a differing teacher future passed the state check')
    finally:
        if original[0] is None:
            del lw.lp.Work.teacher
        else:
            lw.lp.Work.teacher = original[0]
        gw.GateWork.install = original[1]


TC_NAMES = ['video_expert.blocks.0.ffn.0', 'video_expert.time_embedding.0', 'video_expert.time_embedding.2',
            'video_expert.time_projection.1', 'action_expert.time_embedding.0', 'action_expert.time_embedding.2',
            'action_expert.time_projection.1', 'proprio_encoder']
TC_SHAPES = [(5, 4), (6, 3), (6, 6), (12, 6), (4, 3), (4, 4), (8, 4), (5, 2)]


def tc_toy(tmp, leak=False):
    """Masters, a stubbed host install, E factors and the A16 / E bookkeeping files; a query whose time-conditioning
    inputs are the timestep only (with leak, the video time embedding also sees the context)."""
    work = make_work(tmp, TC_SHAPES, names=TC_NAMES)
    mods = work.bank.modules
    g = torch.Generator().manual_seed(21)
    work.masters = {n: (m.weight + 0.2 * torch.randn(m.weight.shape, generator=g)).to(torch.bfloat16) for n, m in mods.items()}
    host = {n: m.weight.clone() for n, m in mods.items()}
    ctx = {c: torch.randn(2, 4, generator=g) for c in range(2000, 2080)}

    def query(case, seed=0, grad=False):
        out = []
        for t in (900.0, 500.0, 100.0):
            emb = torch.tensor([[math.sin(t), math.cos(t), t / 1000]]).repeat(3, 1)
            if leak:
                emb = emb + 1e-3 * ctx[case].mean()
            for head in ('video', 'action'):
                e = mods[f'{head}_expert.time_embedding.2'].forward(F.silu(mods[f'{head}_expert.time_embedding.0'].forward(emb)))
                out.append(mods[f'{head}_expert.time_projection.1'].forward(F.silu(e)).sum())
            out.append(mods['video_expert.blocks.0.ffn.0'].forward(ctx[case]).sum() + mods['proprio_encoder'].forward(ctx[case][:, :2]).sum())
        return torch.stack(out), None, dict(video_steps=10, action_steps=10)
    work.query = query
    base = work.arm_root(lw.LR['baseline_arm'])
    for i, (o, n) in enumerate(TC_SHAPES):
        save(base / 'stats' / f'{i:04d}.json', dict(bits_total=2.0 * o * n))
    save(base / 'COMPLETE.json', dict(bpw=2.0))
    weights = sum(o * n for o, n in TC_SHAPES)
    save(work.lr_dir(lw.ARMS['E']) / 'COMPLETE.json', dict(bpw=2.5, target_weights=weights))
    f = {n: {k: (0.1 * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for k, v in x.items()}
         for n, x in work.snapshot().items()}
    tsave(work.lr_dir(lw.ARMS['E']) / 'factors.pt', f)

    def host_install(self, arm):
        with torch.no_grad():
            for n, m in mods.items():
                m.weight.copy_(host[n])
    return work, host, f, host_install


def test_tcond():
    original = gw.GateWork.install
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work, host, f, host_install = tc_toy(Path(tmp))
            gw.GateWork.install = host_install
            mods, arms = work.bank.modules, lw.LR['tcond']['arms']
            video = ['video_expert.time_embedding.0', 'video_expert.time_embedding.2', 'video_expert.time_projection.1']
            assert work.tcond_modules(['video_time']) == video
            work.install(arms['U_TV']['arm'])
            assert not work.active
            for n, m in mods.items():
                want = work.masters[n].float() if n in video else host[n]
                assert torch.equal(m.weight, want), n
            work.install(arms['E_TVA']['arm'])
            timed = set(work.tcond_modules(['video_time', 'action_time']))
            assert work.active and len(timed) == 6
            for n in mods:
                if n in timed:
                    assert float(work.U[n].abs().max()) == 0 and float(work.V[n].abs().max()) == 0
                    assert torch.equal(mods[n].weight, work.masters[n].float())
                else:
                    assert torch.equal(work.U[n], f[n]['U'].float()) and torch.equal(mods[n].weight, host[n])
            work.active = False
            check = work.tcond_inputs()
            assert check['worst'] == 0.0
            assert check['modules']['video_expert.time_embedding.0']['distinct_rows'] == 3
            assert all(r['calls'] == 3 for r in check['modules'].values())
            ratios = {lw.LR['baseline_arm']: 1.0, lw.ARMS['E']: 0.9, arms['U_TV']['arm']: 0.85, arms['U_TA']['arm']: 1.0,
                      arms['U_TVA']['arm']: 0.84, arms['E_TVA']['arm']: 0.8}
            installed = []

            def evaluate(arm):
                installed.append(arm)
                work.install(arm)
                k = ratios[arm]
                save(work.base / 'heldout' / f'{arm}.json', dict(overall=dict(
                    future_latent_nmse=0.05 * k, mixed_loss=0.003 * k, first10motor_rmse=0.03 * k, first10gripper_rmse=0.08,
                    first10motor_rmse_swap=0.02, first10motor_mse=0.0009 * k, first10motor_mse_swap=0.0004)))
            for arm in (lw.LR['baseline_arm'], lw.ARMS['E']):
                evaluate(arm)
            installed.clear()
            work.evaluate = evaluate
            work.tcond()
            assert installed == [c['arm'] for c in arms.values()]
            s = wk.read(work.lr_dir('tcond') / 'summary.json')
            assert s['reading']['verdict'] == 'carries' and abs(s['reading']['r'] - 0.85) < 1e-12
            assert abs(s['ratios'][arms['E_TVA']['arm']]['future_latent_nmse'] - 0.8 / 0.9) < 1e-12
            total = sum(o * n for o, n in TC_SHAPES)
            idx = [TC_NAMES.index(n) for n in video]
            b = s['bits'][arms['U_TV']['arm']]
            freed = sum(2.0 * TC_SHAPES[i][0] * TC_SHAPES[i][1] for i in idx)
            assert abs(b['bpw_weights'] - (2.0 + (sum(16 * TC_SHAPES[i][0] * TC_SHAPES[i][1] for i in idx) - freed) / total)) < 1e-12
            assert abs(b['bpw_table'] - (2.0 + (16 * 3 * (6 + 12) - freed) / total)) < 1e-12
            e = s['bits'][arms['E_TVA']['arm']]
            all6 = [TC_NAMES.index(n) for n in timed]
            freed_e = sum(2.0 * TC_SHAPES[i][0] * TC_SHAPES[i][1] + 16 * R * sum(TC_SHAPES[i]) for i in all6)
            assert abs(e['bpw_table'] - (2.5 + (16 * 3 * (6 + 12 + 4 + 8) - freed_e) / total)) < 1e-12
            text = (work.lr_dir('tcond') / 'RESULTS.zh.md').read_text(encoding='utf-8')
            assert '判读（预先登记）' in text and '输入核对' in text
        with tempfile.TemporaryDirectory() as tmp:
            work, _, _, host_install = tc_toy(Path(tmp), leak=True)
            gw.GateWork.install = host_install
            try:
                work.tcond()
            except RuntimeError:
                assert wk.read(work.lr_dir('tcond') / 'input_check.json')['worst'] > 0
            else:
                raise AssertionError('context-dependent time inputs passed the check')
    finally:
        gw.GateWork.install = original


def test_stage_arms():
    import os
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(4, 6)])
        done = []
        work.load = work.capture_masters = work.attach = lambda: None
        work.diag = lambda: done.append('diag')
        work.tfdiag = lambda: done.append('tfdiag')
        work.tcond = lambda: done.append('tcond')
        work.stage('diag', 0, 1)
        work.stage('tfdiag', 0, 1)
        work.stage('tcond', 0, 1)
        assert done == ['diag', 'tfdiag', 'tcond']
        done.clear()
        work.run_arm = done.append
        os.environ.pop('FW_ARMS', None)
        work.stage('train', 0, 1)
        work.stage('train', 1, 2)
        os.environ['FW_ARMS'] = ''  # environment.sh passes an empty value unless the job sets it
        work.stage('train', 0, 2)
        os.environ['FW_ARMS'] = 'J'
        work.stage('train', 0, 1)
        assert done == ['J', 'D', 'D', 'J', 'J']
        for arms, shards in (('J', 2), ('J:E', 1)):
            os.environ['FW_ARMS'] = arms
            try:
                work.stage('train', 0, shards)
            except ValueError:
                continue
            raise AssertionError('bad arms accepted')
        os.environ.pop('FW_ARMS')
    grid = lw.LR['training']['step_grid']
    assert grid == sorted(grid) and {0.0025, 0.005, 0.01} <= set(grid)


def test_xfer_host_init_and_stage():
    """select_host: another host's arm names, baseline weights, init directory and method; init / bits / install on it;
    xinit / xtrain take host k on shard k and refuse a host list that is not one per GPU."""
    import os
    shapes = [(8, 6), (5, 10)]
    try:
        lw.select_host('GPTVQ')
        assert lw.ARMS == dict(E='GPTVQ_LRE_W2', J='GPTVQ_LRJ_W2', D='GPTVQ_LRD_W2'), lw.ARMS
        assert (lw.HOST, lw.BASELINE_ARM, lw.METHOD, lw.INIT_DIR) == ('GPTVQ', 'GPTVQ_W2', 'gptvq', 'init_GPTVQ')
        with tempfile.TemporaryDirectory() as tmp:
            work = make_work(Path(tmp), shapes)
            build(work, shapes, 'GPTVQ_W2')
            factors = work.init_factors()
            assert (work.lr_dir('init_GPTVQ') / 'factors.pt').exists() and not (work.lr_dir('init') / 'factors.pt').exists()
            prep, base = work.baseline / 'prep', work.arm_root('GPTVQ_W2')
            name = work.bank.names[0]
            w = torch.load(prep / 'masters' / '0000.pt', weights_only=True)['weight'].float()
            q = torch.load(base / 'weights' / '0000.pt', weights_only=True)['weight'].float()
            h = torch.load(prep / 'hessians' / '0000.pt', weights_only=True)['hessian']
            u, v, _ = lw.eora(w - q, h, R, lw.LR['init']['damping'])
            assert torch.allclose(factors[name]['U'] @ factors[name]['V'], u @ v, atol=1e-5)
            arm = lw.ARMS['E']
            tsave(work.lr_dir(arm) / 'factors.pt', {n: {k: x.to(torch.bfloat16) for k, x in f.items()} for n, f in factors.items()})
            done = work.complete(arm)
            assert done['method'] == 'gptvq' and done['arm'] == 'GPTVQ_LRE_W2' and done['proxy_nmse_weighted'] < done['baseline_proxy_U_weighted']
            installed = []
            original = gw.GateWork.install
            gw.GateWork.install = lambda self, a: installed.append(a)
            try:
                work.install(arm)
                assert work.active and installed == ['GPTVQ_W2']
                work.install('QTIP_LRE_W2')  # another host's arm is not a branch arm here
                assert not work.active and installed[-1] == 'QTIP_LRE_W2'
            finally:
                gw.GateWork.install = original
        with tempfile.TemporaryDirectory() as tmp:
            work = make_work(Path(tmp), [(4, 6)])
            seen = []
            work.load = work.capture_masters = work.attach = lambda: None
            work.init = lambda: seen.append(('E', lw.HOST, lw.ARMS['E']))
            work.run_arm = lambda key: seen.append((key, lw.HOST, lw.ARMS[key]))
            os.environ['FW_HOSTS'] = 'GPTVQ:QUIPSHARP'
            work.stage('xinit', 1, 2)
            work.stage('xtrain', 0, 2)
            assert seen == [('E', 'QUIPSHARP', 'QUIPSHARP_LRE_W2'), ('J', 'GPTVQ', 'GPTVQ_LRJ_W2')], seen
            for hosts, shards in (('GPTVQ:QUIPSHARP', 1), ('QTIP:GPTVQ', 2)):
                os.environ['FW_HOSTS'] = hosts
                try:
                    work.stage('xinit', 0, shards)
                except ValueError:
                    continue
                raise AssertionError(f'bad xfer hosts accepted: {hosts} on {shards}')
    finally:
        os.environ.pop('FW_HOSTS', None)
        lw.select_host(lw.LR['host'])
    assert (lw.ARMS['J'], lw.BASELINE_ARM, lw.INIT_DIR, lw.METHOD) == ('QTIP_LRJ_W2', 'QTIP_U_W2', 'init', 'qtip')


def test_xfinal():
    def held(mixed, future, motor, grip):
        return dict(mixed_loss=mixed, future_latent_nmse=future, first10motor_rmse=motor, first10gripper_rmse=grip,
                    first10motor_rmse_swap=0.02)
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(4, 6)])
        for host in ['QTIP'] + list(lw.LR['xfer']['hosts']):
            arms = lw.host_arms(host)
            base = lw.LR['baseline_arm'] if host == 'QTIP' else lw.LR['xfer']['hosts'][host]['baseline_arm']
            save(work.arm_root(base) / 'COMPLETE.json', dict(bpw=2.01))
            save(work.base / 'tune' / host / 'fit_before.json', dict(mixed_loss=0.004, rows=[1]))
            save(work.base / 'heldout' / f'{base}.json', dict(overall=held(0.004, 0.05, 0.03, 0.08)))
            for key in ('E', 'J'):
                arm = arms[key]
                save(work.lr_dir(arm) / 'arm.json', dict(tuned=True, eta=0.005) if key == 'J' else dict(tuned=False))
                save(work.lr_dir(arm) / 'COMPLETE.json', dict(bpw=2.05))
                save(work.lr_dir(arm) / 'fit.json', dict(mixed_loss=0.003, rows=[1]))
                save(work.base / 'heldout' / f'{arm}.json', dict(overall=held(0.0035 if key == 'E' else 0.0032, 0.05, 0.025, 0.03)))
            save(work.lr_dir(arms['J']) / 'step.json', dict(chosen=0.005, rows=[dict(lr=0.005, predicted=1.0, realized=0.6, loss={})]))
        s = work.xfinal()
        assert list(s['hosts']) == ['QTIP', 'GPTVQ', 'QUIPSHARP'], list(s['hosts'])
        j = s['hosts']['GPTVQ']['J']
        assert j['arm'] == 'GPTVQ_LRJ_W2' and abs(j['heldout_vs_U']['mixed_loss'] - 0.8) < 1e-12 and 'rows' not in j['fit']
        assert s['hosts']['GPTVQ']['U']['arm'] == 'GPTVQ_W2' and s['hosts']['QTIP']['J']['arm'] == 'QTIP_LRJ_W2'
        assert j['step']['chosen'] == 0.005 and 'loss' not in j['step']['rows'][0]
        text = (work.lr_dir('xfer') / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert 'GPTVQ W2' in text and 'QuIP# W2' in text and '0.800' in text, text
        assert lw.ARMS['J'] == 'QTIP_LRJ_W2'


if __name__ == '__main__':
    for test in (test_lowrank_function, test_eora, test_branch_forward, test_staged_gradients, test_step_and_train,
                 test_init_complete_install, test_decide_and_report, test_diag_analysis, test_sample_grads,
                 test_diag_teachers_and_flow, test_video_stage_and_tf_grads, test_tf_analysis, test_tfdiag_flow,
                 test_tcond, test_stage_arms, test_xfer_host_init_and_stage, test_xfinal):
        test()
        print('ok', test.__name__)
    print('ALL PASSED')

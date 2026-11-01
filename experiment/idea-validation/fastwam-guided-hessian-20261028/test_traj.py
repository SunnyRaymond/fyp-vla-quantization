"""CPU checks for the trajectory-coherent proxy test (traj.json, traj_worker.py) with a stubbed runtime: the accumulator
against direct sums (stepped / unstepped / check layers, count and shape guards), ranks and Spearman, aggregation and the
decision, arm weights (file / gate / low-rank) and per-layer proxies, and the whole run on a fake 10-step model in one
and in several memory passes (the H_all check, the saved H_traj, the report)."""
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
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {})):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import traj_worker as tw  # noqa: E402
import worker as wk  # noqa: E402

DELTAS = [-0.1] * 9 + [-0.05]


def direct_sums(calls, deltas, stepped):
    """calls: per query the list of call inputs [rows, k] of one layer -> raw H_S, H_J, rows."""
    k = calls[0][0].shape[1]
    hs, hj, rs, rj = torch.zeros(k, k), torch.zeros(k, k), 0, 0
    for q in calls:
        if not stepped:
            for x in q:
                hs += x.t() @ x
                rs += x.shape[0]
            continue
        a = 0
        for i, x in enumerate(q[:tw.STEPS]):
            hs += deltas[i] ** 2 * x.t() @ x
            a = a + deltas[i] * x
            rs += x.shape[0]
        hj += a.t() @ a
        rj += a.shape[0]
    return hs, (hj if stepped else hs), rs, rj


def test_accumulator():
    g = torch.Generator().manual_seed(0)
    counts = dict(a=11, b=1, c=11)
    acc = tw.Accumulator(DELTAS, counts, 'cpu')
    acc.begin(dict(a=4, b=3), check=dict(c=2))
    calls = dict(a=[], b=[], c=[])
    for _ in range(3):
        acc.start()
        qa, qb, qc = [], [], []
        base = torch.randn(5, 4, generator=g)
        for i in range(11):
            x = base + 0.3 * torch.randn(5, 4, generator=g)
            acc.observe('a', x)
            qa.append(x)
            y = torch.randn(2, 2, generator=g)
            acc.observe('c', y)
            qc.append(y)
        z = torch.randn(6, 3, generator=g)
        acc.observe('b', z)
        qb.append(z)
        acc.observe('other', torch.randn(2, 2))  # not watched
        acc.finish()
        calls['a'].append(qa)
        calls['b'].append(qb)
        calls['c'].append(qc)
    hs, hj, rs, rj = direct_sums(calls['a'], DELTAS, True)
    assert torch.allclose(acc.raw('a', 'S'), hs, atol=1e-5) and torch.allclose(acc.raw('a', 'J'), hj, atol=1e-5)
    assert acc.state['a']['rows_s'] == rs == 150 and acc.state['a']['rows_j'] == rj == 15
    assert torch.allclose(acc.mean('a', 'J'), hj / 15, atol=1e-6)
    hb, _, rb, _ = direct_sums(calls['b'], DELTAS, False)
    assert torch.allclose(acc.raw('b', 'S'), hb, atol=1e-5) and acc.raw('b', 'J') is acc.raw('b', 'S')
    assert torch.allclose(acc.mean('b', 'J'), hb / rb, atol=1e-6)
    hc = sum(x.t() @ x for q in calls['c'] for x in q)
    assert torch.allclose(acc.check['c']['h'], hc, atol=1e-5) and acc.check['c']['rows'] == 66
    # the cross-step terms: identical inputs at every step give (sum delta)^2 / sum delta^2 times H_S
    same = tw.Accumulator(DELTAS, dict(a=11), 'cpu')
    same.begin(dict(a=4))
    same.start()
    x = torch.randn(5, 4, generator=g)
    for _ in range(11):
        same.observe('a', x)
    same.finish()
    ratio = sum(DELTAS) ** 2 / sum(d * d for d in DELTAS)
    assert torch.allclose(same.raw('a', 'J'), ratio * same.raw('a', 'S'), rtol=1e-5)
    for bad in ('count', 'rows'):
        b = tw.Accumulator(DELTAS, dict(a=11), 'cpu')
        b.begin(dict(a=4))
        b.start()
        try:
            for i in range(10 if bad == 'count' else 11):
                b.observe('a', torch.randn(5 if bad == 'count' or i < 3 else 6, 4))
            b.finish()
        except RuntimeError:
            continue
        raise AssertionError(f'{bad} guard did not fire')
    try:
        tw.Accumulator(DELTAS[:9], counts, 'cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('wrong delta count accepted')


def test_spearman():
    assert tw.ranks([3.0, 1.0, 2.0, 2.0]) == [4.0, 1.0, 2.5, 2.5]
    assert abs(tw.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(tw.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    x, y = [0.3, 0.1, 0.5, 0.2, 0.4], [2.0, 1.0, 3.0, 5.0, 4.0]
    rx, ry = tw.ranks(x), tw.ranks(y)
    mx, my = sum(rx) / 5, sum(ry) / 5
    want = sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    assert abs(tw.spearman(x, y) - want) < 1e-12
    assert tw.spearman([1, 1, 1], [1, 2, 3]) is None
    assert not tw.decide(dict(future_latent_nmse=dict(U=None, S=0.1, J=0.9)), tw.TRAJ['reading'])['go']


def test_aggregate_ranking_decide():
    shapes = dict(l1=[2, 3], l2=[4, 5])
    arms = ['A', 'B', 'C']
    layers = {'l1': {a: dict(U=0.1 * k, S=0.2 * k, J=0.3 * (3 - k)) for k, a in enumerate(arms, 1)},
              'l2': {a: dict(U=0.5 * k, S=0.1, J=0.2 * (3 - k)) for k, a in enumerate(arms, 1)}}
    agg = tw.aggregate(layers, arms, shapes)
    assert abs(agg['B']['U']['weighted'] - (0.2 * 6 + 1.0 * 20) / 26) < 1e-12
    assert abs(agg['B']['U']['plain'] - (0.2 + 1.0) / 2) < 1e-12
    metrics = {a: dict(future_latent_nmse=0.05 - 0.01 * k, mixed_loss=0.003, first10motor_rmse=0.03 + k)
               for k, a in enumerate(arms, 1)}
    r = tw.ranking(agg, metrics, arms)
    assert abs(r['future_latent_nmse']['J'] - 1) < 1e-12 and abs(r['future_latent_nmse']['U'] + 1) < 1e-12
    rule = tw.TRAJ['reading']
    d = tw.decide(r, rule)
    assert d['go'] and d['rho']['J'] == 1
    r2 = dict(future_latent_nmse=dict(U=0.5, S=0.75, J=0.8))
    assert tw.decide(r2, rule)['go'] and not tw.decide(r2, rule)['coherence']
    assert not tw.decide(dict(future_latent_nmse=dict(U=0.5, S=0.0, J=0.65)), rule)['go']


NAMES = ['video_expert.time_embedding.0', 'video_expert.blocks.0.self_attn.q', 'video_expert.blocks.0.cross_attn.k',
         'video_expert.blocks.0.ffn.0', 'action_expert.blocks.0.ffn.0']
SHAPES = [(6, 3), (5, 4), (4, 4), (7, 4), (3, 4)]
CALLS = {NAMES[0]: 11, NAMES[1]: 11, NAMES[2]: 1, NAMES[3]: 11}


class FakeScheduler:
    def step(self, model_output, delta, sample):
        return sample + model_output * delta


def fake_world(root, seed=0):
    """A TrajWork without __init__: five Linears (four video, one action), a runtime whose infer runs 10 video steps
    and a prefill (the cross-attention key once per query), baseline prep files with H_U from the same calls, every arm's
    weights, the gate selection, low-rank factors and held-out metrics."""
    g = torch.Generator().manual_seed(seed)
    work = object.__new__(tw.TrajWork)
    work.base, work.baseline = root / 'new' / 'results', root / 'baseline' / 'results'
    work.out = work.base / 'traj'
    work.job, work.job_name = 'TEST', 'fwtr1_0'
    work.counting, work.dry_counts, work.acc, work.factors, work.selection = False, tw.Counter(), None, {}, None
    work.counters = dict(formal_infer_queries=0)
    work.progress = lambda *a, **k: None
    work.enough = lambda: None
    work.count = lambda key: None
    work.atomic_torch_save = lambda _torch, path, value: torch.save(value, path)
    masters = {}
    modules = {}
    for n, (o, k) in zip(NAMES, SHAPES):
        w = torch.randn(o, k, generator=g).to(torch.bfloat16)
        masters[n] = w
        modules[n] = types.SimpleNamespace(in_features=k, out_features=o, weight=w.float(), bias=None,
                                           forward=(lambda x, w=w: F.linear(x, w.float())))
    work.bank = types.SimpleNamespace(names=list(NAMES), modules=modules)
    work.rows = {c: dict(sampler_seeds=[10 * c]) for c in range(2000, 2080)}
    bases = {c: {n: torch.randn(3, k, generator=g) for n, (o, k) in zip(NAMES, SHAPES)} for c in work.rows}
    sched = FakeScheduler()

    def infer(value, seed_, measure=False):
        gen = torch.Generator().manual_seed(seed_)
        b = bases[value]
        sample = torch.zeros(1)
        for i in range(10):
            for n in NAMES[:4]:
                if n == NAMES[2] and i > 0:
                    continue
                work.bank.modules[n].forward(b[n] + 0.5 * torch.randn(b[n].shape, generator=gen))
            for _ in range(1):
                work.bank.modules[NAMES[4]].forward(b[NAMES[4]])
            sample = sched_holder.step(torch.ones(1), DELTAS[i], sample)
        for n in (NAMES[0], NAMES[1], NAMES[3]):  # prefill
            work.bank.modules[n].forward(b[n] * 2)
        return torch.zeros(32, 7), {}
    sched_holder = types.SimpleNamespace(step=sched.step)
    work.runtime = types.SimpleNamespace(model=types.SimpleNamespace(infer_video_scheduler=sched_holder), infer=infer,
                                         description=None)
    work.datum = lambda case: (case, 'task')
    work.memory_budget = lambda: 10 ** 12
    # H_U from the same calls (all of them), as the baselines' prep
    sums = {n: torch.zeros(k, k) for n, (o, k) in zip(NAMES, SHAPES)}
    rows = dict.fromkeys(NAMES, 0)
    plain = {n: m.forward for n, m in modules.items()}

    def collecting(n):
        def run(x):
            sums[n].addmm_(x.t(), x)
            rows[n] += x.shape[0]
            return plain[n](x)
        return run
    for n in NAMES:
        modules[n].forward = collecting(n)
    for c in work.rows:
        infer(c, work.rows[c]['sampler_seeds'][0])
    for n in NAMES:
        modules[n].forward = plain[n]
    prep = work.baseline / 'prep'
    for i, n in enumerate(NAMES):
        tsave(prep / 'masters' / f'{i:04d}.pt', dict(module=n, weight=masters[n]))
        tsave(prep / 'hessians' / f'{i:04d}.pt', dict(module=n, rows=rows[n], hessian=sums[n] / rows[n]))
    weights = {}
    for arm, spec in tw.TRAJ['arms'].items():
        if spec['kind'] != 'file':
            continue
        root_ = (work.baseline if spec['root'] == 'baseline' else work.base) / 'quantized' / spec['dir']
        scale = 0.05 + 0.02 * len(weights)
        weights[arm] = {}
        for i, n in enumerate(NAMES):
            w = (masters[n].float() + scale * torch.randn(masters[n].shape, generator=g)).to(torch.bfloat16)
            weights[arm][n] = w
            tsave(root_ / 'weights' / f'{i:04d}.pt', dict(module=n, weight=w))
    selection = dict(hosts={h: dict(layers=[1, 4]) for h in ('QTIP', 'GPTVQ', 'QUIPSHARP')})
    save(work.base / 'gate' / 'selection.json', selection)
    for h in ('QTIP', 'GPTVQ', 'QUIPSHARP'):
        for i in (1, 4):
            n = NAMES[i]
            tsave(work.base / 'gate' / h / 'weights' / f'{i:04d}.pt',
                  dict(module=n, weight=(masters[n].float() + 0.01 * torch.randn(masters[n].shape, generator=g)).to(torch.bfloat16)))
    for arm in ('QTIP_LRE_W2', 'QTIP_LRJ_W2', 'QTIP_LRD_W2'):
        tsave(work.base / 'lowrank' / arm / 'factors.pt',
              {n: dict(U=(0.1 * torch.randn(o, 2, generator=g)).to(torch.bfloat16),
                       V=(0.1 * torch.randn(2, k, generator=g)).to(torch.bfloat16)) for n, (o, k) in zip(NAMES, SHAPES)})
    for k, arm in enumerate(tw.TRAJ['arms']):
        save(work.base / 'heldout' / f'{arm}.json',
             dict(overall=dict(future_latent_nmse=0.05 + 0.001 * k, mixed_loss=0.003 + 1e-4 * (k % 5),
                               first10motor_rmse=0.03 - 0.001 * (k % 3))))
    work.load = lambda: None
    return work, masters, weights, bases


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_arm_weights():
    with tempfile.TemporaryDirectory() as tmp:
        work, masters, weights, _ = fake_world(Path(tmp))
        for i, n in enumerate(NAMES):
            assert torch.equal(work.arm_weight('QTIP_G_W2', i, n), weights['QTIP_G_W2'][n].float())
            gate = work.arm_weight('GATE_QTIP_G', i, n)
            want = weights['QTIP_G_W2' if i in (1, 4) else 'QTIP_U_W2'][n].float()
            assert torch.equal(gate, want)
            o = work.arm_weight('GATE_GPTVQ_O', i, n)
            if i in (1, 4):
                ref = torch.load(work.base / 'gate' / 'GPTVQ' / 'weights' / f'{i:04d}.pt', weights_only=True)['weight']
                assert torch.equal(o, ref.float())
            else:
                assert torch.equal(o, weights['GPTVQ_W2'][n].float())
            f = torch.load(work.base / 'lowrank' / 'QTIP_LRD_W2' / 'factors.pt', weights_only=True)[n]
            assert torch.allclose(work.arm_weight('QTIP_LRD_W2', i, n),
                                  weights['QTIP_U_W2'][n].float() + f['U'].float() @ f['V'].float(), atol=1e-6)
        bad = Path(tmp) / 'new' / 'results' / 'quantized' / 'QTIP_C_W2' / 'weights' / '0002.pt'
        torch.save(dict(module='wrong', weight=torch.zeros(1)), bad)
        try:
            work.arm_weight('QTIP_C_W2', 2, NAMES[2])
        except RuntimeError:
            pass
        else:
            raise AssertionError('a wrong module identity passed')


def run_flow(tmp, budget):
    work, masters, weights, bases = fake_world(Path(tmp))
    work.memory_budget = lambda: budget
    work.run()  # run() wraps the video Linears itself
    return work, masters, weights, bases


def test_flow():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as tmp2:
        work, masters, weights, bases = run_flow(tmp, 10 ** 12)
        s = wk.read(work.out / 'summary.json')
        assert s['deltas'] == DELTAS and s['h_all_check']['worst'] < 1e-6
        assert s['unstepped'] == [NAMES[2]] and set(s['calls']) == {'11', '1'}
        layers = wk.read(work.out / 'layers.json')
        assert set(layers) == set(NAMES[:4])
        # H_traj of one stepped layer directly from the fake inputs (base + noise per step, prefill left out)
        i, n = 1, NAMES[1]
        k = SHAPES[i][1]
        hj = torch.zeros(k, k)
        for c in work.rows:
            gen = torch.Generator().manual_seed(work.rows[c]['sampler_seeds'][0])
            a = 0
            for step in range(10):
                for m_ in NAMES[:4]:
                    if m_ == NAMES[2] and step > 0:
                        continue
                    x = bases[c][m_] + 0.5 * torch.randn(bases[c][m_].shape, generator=gen)
                    if m_ == n:
                        a = a + DELTAS[step] * x
            hj += a.t() @ a
        saved = torch.load(work.out / 'hessians' / f'{i:04d}.pt', weights_only=True)
        assert saved['module'] == n and saved['kind'] == 'traj' and saved['rows'] == 3 * 80
        assert torch.allclose(saved['hessian'], hj / (3 * 80), rtol=1e-5, atol=1e-7)
        m = masters[n].float()
        hu = torch.load(work.baseline / 'prep' / 'hessians' / f'{i:04d}.pt', weights_only=True)['hessian']
        e = weights['QTIP_G_W2'][n].float() - m
        want_u = float(((e @ hu) * e).sum() / ((m @ hu) * m).sum())
        want_j = float(((e @ saved['hessian']) * e).sum() / ((m @ saved['hessian']) * m).sum())
        assert abs(layers[n]['arms']['QTIP_G_W2']['U'] - want_u) < 1e-5 * max(1, want_u)
        assert abs(layers[n]['arms']['QTIP_G_W2']['J'] - want_j) < 1e-5 * max(1, want_j)
        assert layers[NAMES[2]]['arms']['QTIP_G_W2']['kappa'] == 1.0
        assert not (work.out / 'hessians' / '0002.pt').exists()
        assert layers[NAMES[0]]['arms']['GATE_QTIP_G'] == layers[NAMES[0]]['arms']['QTIP_U_W2']
        assert layers[n]['arms']['GATE_QTIP_G'] == layers[n]['arms']['QTIP_G_W2']
        text = (work.out / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert '判读（预先登记）' in text and 'QTIP_LRE_W2' in text
        assert set(s['decision']) == {'rho', 'go', 'coherence'}
        # several memory passes give the same numbers
        sizes = [k for _, k in SHAPES[:4]]
        work2, _, _, _ = run_flow(tmp2, 4 * 2 * max(sizes) ** 2 + sum(4 * k * k for k in sizes))
        layers2 = wk.read(work2.out / 'layers.json')
        for n_ in NAMES[:4]:
            for arm in tw.TRAJ['arms']:
                for v in tw.VARIANTS:
                    a, b = layers[n_]['arms'][arm][v], layers2[n_]['arms'][arm][v]
                    assert abs(a - b) <= 1e-6 * max(1.0, abs(a)), (n_, arm, v, a, b)
        assert wk.read(work2.out / 'summary.json')['ranking'] == s['ranking']


def test_dry_restores_scheduler():
    with tempfile.TemporaryDirectory() as tmp:
        work, _, _, _ = fake_world(Path(tmp))
        work.watch()
        before = work.runtime.model.infer_video_scheduler.step
        deltas, counts = work.dry()
        assert deltas == DELTAS and counts == CALLS
        assert work.runtime.model.infer_video_scheduler.step is before


if __name__ == '__main__':
    for test in (test_accumulator, test_spearman, test_aggregate_ranking_decide, test_arm_weights, test_dry_restores_scheduler,
                 test_flow):
        test()
        print('ok', test.__name__)
    print('ALL PASSED')

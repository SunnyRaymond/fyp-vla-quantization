"""CPU checks for RAVQ uncalibrated (ravq.json, ravq_worker.py, ravq_qtip.py) with a stubbed runtime: the role layout
on the real 614 names, the plan (budgets, greedy fill, choices), the widened QTIP port (K = 2 bit for bit equal to the
A16 port; K = 1 / 3 / 4 decode and error ordering), and a toy pipeline (614 tiny layers) through quant (stand-in
quantizer), assemble (symlinks, bits, COMPLETE), install and final (reading, report). 'cuda' tensors map to the CPU."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import torch

HERE = Path(__file__).resolve().parent
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {}),
                    ('frozen_reader', dict(_bf16_forward_context=None, _checkpoint_context_fn=None, BF16Reader=None))):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import ravq_qtip as rq  # noqa: E402
import ravq_worker as rw  # noqa: E402
import worker as wk  # noqa: E402

_to = torch.Tensor.to


def _cpu_to(self, *args, **kwargs):
    args = tuple('cpu' if a == 'cuda' else a for a in args)
    if kwargs.get('device') == 'cuda':
        kwargs['device'] = 'cpu'
    return _to(self, *args, **kwargs)


torch.Tensor.to = _cpu_to
spec = importlib.util.spec_from_file_location('fwrv_qtip_port', HERE / 'methods' / 'qtip' / 'quant.py')
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
BLOCK = ('self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o', 'cross_attn.q', 'cross_attn.k', 'cross_attn.v',
         'cross_attn.o', 'ffn.0', 'ffn.2')


def real_names():
    names = ['video_expert.text_embedding.0', 'video_expert.text_embedding.2', 'video_expert.time_embedding.0',
             'video_expert.time_embedding.2', 'video_expert.time_projection.1']
    names += [f'video_expert.blocks.{b}.{m}' for b in range(30) for m in BLOCK] + ['video_expert.head.head']
    names += ['action_expert.action_encoder', 'action_expert.text_embedding.0', 'action_expert.text_embedding.2',
              'action_expert.time_embedding.0', 'action_expert.time_embedding.2', 'action_expert.time_projection.1']
    names += [f'action_expert.blocks.{b}.{m}' for b in range(30) for m in BLOCK] + ['action_expert.head', 'proprio_encoder']
    assert len(names) == 614
    return names


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def jsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_roles():
    names = real_names()
    roles = [rw.role(n) for n in names]
    assert {r: roles.count(r) for r in rw.ROLES} == rw.ROLE_COUNTS
    assert rw.role('video_expert.head.head') == 'interface' and rw.role('proprio_encoder') == 'interface'
    assert rw.role('action_expert.time_projection.1') == 'time' and rw.role('video_expert.text_embedding.2') == 'text'
    try:
        rw.role('video_expert.something')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown name accepted')


def toy_layers(g):
    rows = []
    for i, n in enumerate(real_names()):
        # action blocks a third of the video blocks' size: their saved bits upgrade about a third of the video body
        m, k = (32, 48) if n.startswith('video_expert.blocks') else (16, 32) if 'blocks' in n else (16, 16)
        p2 = 0.01 + 0.04 * float(torch.rand((), generator=g))
        p3 = p2 * (0.2 + 0.2 * float(torch.rand((), generator=g)))
        rows.append(dict(index=i, module=n, role=rw.role(n), shape=[m, k],
                         bits={2: Q.storage_bits(m, k, 2), 3: Q.storage_bits(m, k, 3)}, proxy={2: p2, 3: p3}))
    return rows


def test_plan():
    layers = toy_layers(torch.Generator().manual_seed(0))
    table = rw.table_bits(layers, dict(video_expert=11, action_expert=10))
    assert sum(table.values()) == 16 * (11 * 32 + 10 * 32)  # 16x16 toy time Linears: outs 16 + 16 per expert
    arms = rw.planned_arms()
    plans = rw.plan(layers, arms, Q.storage_bits, table)
    for arm, c in arms.items():
        p = plans[arm]
        for L, k in zip(layers, p['choices']):
            r = L['role']
            want = ('bf16' if r == 'interface' else 'table' if r == 'time' else c['action_body'] if r == 'action_body'
                    else c['family'] if r == 'text' else None)
            if want is not None:
                assert k == want, (arm, L['module'], k)
            else:
                assert k in (c['video_body'], c.get('upgrade')), (arm, L['module'], k)
        assert abs(p['bits_total'] - sum(rw.layer_bits(L, k, Q.storage_bits, table) for L, k in zip(layers, p['choices']))) < 1e-6
        if 'budget_of' in c:
            ref = plans[c['budget_of']]['bits_total']
            assert p['bits_total'] <= ref and p['used'] <= p['budget']
            assert 50 < len(p['upgraded']) < 200, len(p['upgraded'])  # a partial fill, not all or nothing
            left = p['budget'] - p['used']
            for L in layers:  # greedy: every candidate left out costs more than what is left
                if L['role'] == 'video_body' and L['index'] not in p['upgraded']:
                    cost = rw.layer_bits(L, c['upgrade'], Q.storage_bits, table) - rw.layer_bits(L, c['video_body'], Q.storage_bits, table)
                    assert cost > left
            def density(L):
                cost = rw.layer_bits(L, c['upgrade'], Q.storage_bits, table) - rw.layer_bits(L, c['video_body'], Q.storage_bits, table)
                return L['shape'][0] * L['shape'][1] * (rw.proxy_of(L, c['video_body']) - rw.proxy_of(L, c['upgrade'])) / cost
            body = [L for L in layers if L['role'] == 'video_body']
            best = max(body, key=density)
            assert best['index'] in p['upgraded']  # the densest layer fits any budget that admits one upgrade
            assert min(density(layers[i]) for i in p['upgraded']) >= 0
    assert all(k in rw.NEW_BITS for _, k in rw.quant_list(plans))
    q = rw.quant_list(plans)
    assert sum(1 for _, k in q if k == 1) == 300 and any(k == 4 for _, k in q)
    assert {i for i, k in q if k == 4} == set(plans['RAVQU_W3']['upgraded'])
    assert rw.proxy_of(layers[10], 4) == layers[10]['proxy'][3] ** 2 / layers[10]['proxy'][2]


def test_qtip_port():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 128, generator=g) * 0.02
    x = torch.randn(512, 128, generator=g) @ torch.randn(128, 128, generator=g) * 0.1 + torch.randn(512, 128, generator=g)
    h = x.t() @ x / 512

    def proxy(q):
        d = q - w
        return float(((d @ h) * d).sum() / ((w @ h) * w).sum())
    ours = rq.quantize_layer(Q, w, h, 2, seed=7)
    port = Q.quantize_layer(w, h, 2, seed=7)
    assert torch.equal(ours['dequant'], port['dequant']) and ours['bits_total'] == port['bits_total']
    out = {k: rq.quantize_layer(Q, w, h, k, seed=7) for k in (1, 3, 4)}
    out[2] = ours
    p = {k: proxy(v['dequant']) for k, v in out.items()}
    assert p[1] > p[2] > p[3] > p[4] > 0, p
    for k, v in out.items():
        assert v['bits_total'] == Q.storage_bits(64, 128, k) and v['codes']['K'] == k
        assert v['codes']['trellis'].numel() * 16 + 128 * Q.SU_BITS + 64 * Q.SV_BITS == v['bits_total']
    try:
        Q.quantize_layer(w, h, 1, seed=7)
    except ValueError:
        pass
    else:
        raise AssertionError('the port accepted K = 1 (then ravq_qtip is not needed)')
    print('qtip proxies', p)


def toy_work(root):
    g = torch.Generator().manual_seed(1)
    layers = toy_layers(g)
    base = root / 'baseline' / 'results'
    for L in layers:
        i, (m, k), n = L['index'], L['shape'], L['module']
        master = torch.randn(m, k, generator=g)
        tsave(base / 'prep' / 'masters' / f'{i:04d}.pt', dict(module=n, weight=master.to(torch.bfloat16)))
        tsave(base / 'prep' / 'hessians' / f'{i:04d}.pt', dict(module=n, hessian=torch.eye(k)))
        for b, arm in ((2, 'QTIP_W2'), (3, 'QTIP_W3')):
            tsave(base / 'quantized' / arm / 'weights' / f'{i:04d}.pt',
                  dict(module=n, weight=(master + 0.1 / b * torch.randn(m, k, generator=g)).to(torch.bfloat16)))
            jsave(base / 'quantized' / arm / 'stats' / f'{i:04d}.json',
                  dict(module=n, shape=[m, k], bits_total=L['bits'][b], proxy_nmse=L['proxy'][b]))
    for b, arm in ((2, 'QTIP_W2'), (3, 'QTIP_W3')):
        jsave(base / 'quantized' / arm / 'COMPLETE.json', dict(complete=True, bpw=float(b), proxy_nmse_weighted=0.01 / b))
    work = object.__new__(rw.RavqWork)
    work.base = work.results = root / 'guided' / 'results'
    work.baseline, work.job, work.job_name = base, 'TEST', 'fwrv1_0'
    work.enough = lambda: None
    work.progress = lambda *a, **k: None
    work.atomic_torch_save = lambda _torch, path, value: torch.save(value, path)
    work.method = lambda name: Q
    work.source = types.ModuleType('calibration_source')  # the parent runtime's Work.__init__ sets this instance attribute
    tc ={f'{h}.time_embedding.0': dict(distinct_rows=s) for h, s in (('video_expert', 11), ('action_expert', 10))}
    jsave(work.base / 'lowrank' / 'tcond' / 'input_check.json', dict(worst=0.0, modules=tc))
    names = [L['module'] for L in layers]
    work.bank = types.SimpleNamespace(names=names, modules={
        n: types.SimpleNamespace(weight=torch.zeros(L['shape'], dtype=torch.bfloat16)) for n, L in zip(names, layers)})
    return work, layers


def fake_quantize(Q_, w, h, k, seed=0):
    m, n = w.shape
    return dict(dequant=w + 0.1 / (k + 1) * torch.randn(w.shape, generator=torch.Generator().manual_seed(seed)),
                bits_total=float(Q_.storage_bits(m, n, k)), config=dict(code='stand-in', K=k),
                codes=dict(trellis=torch.zeros(4, dtype=torch.int16), K=k))


def heldout(work, values):
    for arm, o in values.items():
        jsave(work.base / 'heldout' / f'{arm}.json', dict(overall=o))


def metrics(f, mixed=0.003, motor=0.03, swap=0.02, grip=0.08):
    return dict(mixed_loss=mixed, first10motor_rmse=motor, future_latent_nmse=f, first10gripper_rmse=grip,
                first10motor_rmse_swap=swap, first10motor_mse=motor ** 2, first10motor_mse_swap=swap ** 2,
                full32motor_rmse=motor, first10gripper_disagreement=0.0)


def test_pipeline():
    real = rq.quantize_layer
    rq.quantize_layer = fake_quantize
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work, layers = toy_work(Path(tmp))
            work.make_plan()
            todo = rw.quant_list(work.plans)
            owned = [set(wk.owned(s, 4, lambda j: 1.0, range(len(todo)))) for s in range(4)]
            assert set().union(*owned) == set(range(len(todo))) and sum(len(o) for o in owned) == len(todo)
            for s in range(4):
                work.job_name = f'fwrv1_{s}'
                work.quant(s, 4)
            for i, k in todo:
                assert (work.rv(f'qtip_K{k}', 'stats', f'{i:04d}.json')).exists()
                assert (work.rv(f'qtip_K{k}', 'codes', f'{i:04d}.pt')).exists()
            work.assemble()
            for arm, p in work.plans.items():
                root = work.arm_root(arm)
                done = wk.read(root / 'COMPLETE.json')
                assert done['complete'] and abs(done['bits_total'] - p['bits_total']) < 0.5
                assert done['bits'] == rw.RV['arms'][arm]['family']  # the A16 loop analysis reads it
                assert abs(done['bpw'] - done['bits_total'] / done['target_weights']) < 1e-12
                for i, k in enumerate(p['choices']):
                    target = Path(os.readlink(root / 'weights' / f'{i:04d}.pt'))
                    want = work.layer_source(i, k)[0]
                    assert target == want, (arm, i, target, want)
            p = work.plans['RAVQU_W2']
            a = next(i for i, L in enumerate(layers) if L['role'] == 'action_body')
            v = p['upgraded'][0]
            iface = next(i for i, L in enumerate(layers) if L['role'] == 'interface')
            root = work.arm_root('RAVQU_W2')
            assert 'qtip_K1' in os.readlink(root / 'weights' / f'{a:04d}.pt')
            assert 'QTIP_W3' in os.readlink(root / 'weights' / f'{v:04d}.pt')
            assert 'masters' in os.readlink(root / 'weights' / f'{iface:04d}.pt')
            assert wk.read(root / 'COMPLETE.json')['bits_total'] <= wk.read(work.arm_root('QTIP_W2I') / 'COMPLETE.json')['bits_total']
            wk.Work.install(work, 'RAVQU_W2')  # the held-out install path reads the symlinks
            got = work.bank.modules[layers[a]['module']].weight
            want = torch.load(work.rv('qtip_K1', 'weights', f'{a:04d}.pt'), weights_only=True)['weight']
            assert torch.equal(got, want)
            assert work.arm_root('QTIP_W3') == work.baseline / 'quantized' / 'QTIP_W3'
            work.assemble()  # idempotent
            heldout(work, dict(QTIP_U_W2=metrics(0.053, grip=0.086), QTIP_W3=metrics(0.02),
                               QTIP_W2I=metrics(0.05, grip=0.04), QTIP_A1W2I=metrics(0.05, swap=0.025),
                               RAVQU_W2=metrics(0.043, mixed=0.0029), QTIP_W3I=metrics(0.019),
                               QTIP_A2W3I=metrics(0.019, swap=0.021), RAVQU_W3=metrics(0.018)))
            work.final()
            d = wk.read(work.rv('DECISION.json'))
            assert d['RAVQU_W2']['go'] and not d['RAVQU_W3']['go'] and d['loops'] == ['QTIP_W2I', 'RAVQU_W2']
            text = work.rv('RESULTS.zh.md').read_text(encoding='utf-8')
            assert 'RAVQ 未校准，2-bit 预算' in text and '上闭环' in text
            s = wk.read(work.rv('summary.json'))
            assert s['reading']['checks']['budget_ok'] and s['reading']['checks']['future_identity_worst'] == 0
            for arm, root, wins in (('QTIP_W2', work.baseline, 200), ('QTIP_W2I', work.base, 205), ('RAVQU_W2', work.base, 210)):
                for e in range(280):
                    jsave(root / 'loops' / arm / 'episodes' / f'{e}.json', dict(success=e < wins))
                (root / 'loops' / arm / 'DONE').write_text('x')
            loop = work.loop_final()
            assert list(loop['arms']) == ['RAVQU_W2']  # loop_job.run: the RAVQU arms only (RAVQU_W3 is no-go here)
            r = loop['arms']['RAVQU_W2']
            assert r['successes'] == 210 and r['pairs']['QTIP_W2I'] == dict(reference=205, wins=5, losses=0, p=wk.mcnemar(5, 0))
            assert r['pairs']['QTIP_W2']['wins'] == 10
            run = rw.RV['loop_job'].pop('run')
            try:
                assert work.loop_final()['arms']['QTIP_W2I']['pairs']['QTIP_W2']['wins'] == 5
            finally:
                rw.RV['loop_job']['run'] = run
            (work.base / 'loops' / 'QTIP_W2I' / 'DONE').unlink()
            assert list(work.loop_final()['arms']['RAVQU_W2']['pairs']) == ['QTIP_W2']
            assert '闭环' in work.rv('loop', 'RESULTS.zh.md').read_text(encoding='utf-8')
            heldout(work, dict(QTIP_A1W2I=metrics(0.0506)))
            try:
                work.final()
            except RuntimeError:
                pass
            else:
                raise AssertionError('future moved by an action-only change was accepted')
    finally:
        rq.quantize_layer = real


def test_trace():
    """trace_worker: segment labels on synthetic gripper traces, and the discordant set / analysis on fake loops."""
    common = types.ModuleType('common')
    common.read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    common.save = jsave
    common.guard = lambda: None
    common.results = lambda p: Path(p['remote_root']) / 'results'
    sys.modules['common'] = common
    import trace_worker as tw

    def trace(success, cmd, w):
        return dict(success=success, command=cmd, width=[0.08] + w, raw=[0.9 if c > 0 else 0.1 for c in cmd],
                    steps=len(cmd), eef=[[0, 0, 0]] * (len(cmd) + 1))
    n = 40
    assert tw.label(trace(False, [-1.0] * n, [0.08] * n))['outcome'] == 'never_closed'
    assert tw.label(trace(False, [-1.0] * 10 + [1.0] * 30, [0.08] * 10 + [0.0] * 30))['outcome'] == 'empty_grasp'
    assert tw.label(trace(False, [-1.0] * 10 + [1.0] * 30, [0.08] * 10 + [0.03] * 30))['outcome'] == 'held_no_success'
    assert tw.label(trace(False, [-1.0] * 10 + [1.0] * 30, [0.08] * 10 + [0.03] * 20 + [0.0] * 10))['outcome'] == 'dropped'
    two = [-1.0] * 5 + [1.0] * 15 + [-1.0] * 5 + [1.0] * 15
    held_then = [0.08] * 5 + [0.03] * 15 + [0.08] * 5 + [0.03] * 10 + [0.0] * 5
    assert tw.label(trace(False, two, held_then))['outcome'] == 'held_then_dropped'
    flick = tw.label(trace(False, [-1.0] * 10 + [1.0, -1.0] * 3 + [-1.0] * 24, [0.08] * n))
    assert flick['outcome'] == 'never_closed' and flick['short_segments'] >= 5
    assert tw.label(trace(True, [-1.0] * n, [0.08] * n))['outcome'] == 'success'
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = dict(pair=['A', 'B'], loops={'A': str(root / 'la'), 'B': str(root / 'lb')})
        for arm, ok in (('A', lambda i: i < 200 or i == 250), ('B', lambda i: i < 201)):
            for i in range(280):
                jsave(Path(cfg['loops'][arm]) / 'episodes' / f'{i}.json', dict(success=ok(i)))
        p = dict(protocol='T', remote_root=str(root))
        assert tw.discordant(cfg, p) == [200, 250]
        for arm, ok in (('A', {200: False, 250: True}), ('B', {200: True, 250: False})):
            for i, s in ok.items():
                t = trace(s, [-1.0] * 10 + [1.0] * 30, [0.08] * 10 + [0.0] * 30)
                t.update(suite='libero_spatial', dimension='camera')
                jsave(root / 'results' / 'ravq' / 'trace' / arm / f'{i}.json', t)
        s = tw.analyze(cfg, p)
        assert s['reproduced'] == dict(A=2, B=2) and s['loser_outcomes'] == dict(A={'empty_grasp': 1}, B={'empty_grasp': 1})
        assert s['gripper_failures'] == dict(A=1, B=1)
        assert '空抓' in (root / 'results' / 'ravq' / 'trace' / 'RESULTS.zh.md').read_text(encoding='utf-8')


if __name__ == '__main__':
    for test in (test_roles, test_plan, test_pipeline, test_qtip_port, test_trace):
        test()
        print('PASS', test.__name__, flush=True)
    print('ALL PASS')

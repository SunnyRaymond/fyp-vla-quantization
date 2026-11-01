"""CPU checks for gate 1 (gate.json, gate_worker.py) with a stubbed runtime: damage ranking and coverage, head shares
with the phase-1 A_via_future definition, the pre-registered decision, the frozen-reader context swap, the GATE arm
install (baseline everywhere, selected layers from the phase-1 H~ arm or from H~_op), the requantization receipts, and
select / final on synthetic statistics. 'cuda' tensors are mapped to the CPU."""
import json
from pathlib import Path
import random
import sys
import tempfile
import types

import torch

HERE = Path(__file__).resolve().parent
SENTINEL = dict(original=object(), patched=object())
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {}),
                    ('frozen_reader', dict(_bf16_forward_context=SENTINEL['original'],
                                           _checkpoint_context_fn=SENTINEL['original'],
                                           BF16Reader=lambda runtime, weights: types.SimpleNamespace(
                                               runtime=runtime, original_weights=dict(weights))))):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import gate_worker as gw  # noqa: E402
import worker as wk  # noqa: E402

_to = torch.Tensor.to


def _cpu_to(self, *args, **kwargs):
    args = tuple('cpu' if a == 'cuda' else a for a in args)
    if kwargs.get('device') == 'cuda':
        kwargs['device'] = 'cpu'
    return _to(self, *args, **kwargs)


torch.Tensor.to = _cpu_to
HOST = json.loads((HERE / 'host.json').read_text(encoding='utf-8'))
FAKE = types.SimpleNamespace(quantize_layer=lambda w, h, bits, seed=0: dict(
    dequant=w.cpu() + torch.randn(w.shape, generator=torch.Generator().manual_seed(seed)) * 0.01,
    bits_total=float(bits * w.numel()), config=dict(code='stand-in')))
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def make_work(root):
    work = object.__new__(gw.GateWork)
    work.base = work.results = root / 'guided' / 'results'
    work.baseline, work.host, work.job, work.job_name = root / 'baseline' / 'results', HOST, 'TEST', 'fwgo_0'
    work.enough = lambda: None
    work.progress = lambda *args, **kwargs: None
    work.atomic_torch_save = lambda _torch, path, value: torch.save(value, path)
    work.method = lambda name: FAKE
    work.cfg = dict(probes=dict(diagnostic_cases_per_suite=2), suites=SUITES, evaluation=dict(case_ids=[3000]))
    work.rows = {c: dict(suite=SUITES[(c - 2000) // 20]) for c in range(2000, 2080)}
    return work


def test_pure():
    damage = gw.layer_damage([1.0, 2.0, 0.5, 4.0], [3.0, 1.0, 10.0, 0.0])
    assert damage == [3.0, 2.0, 5.0, 0.0]
    layers, coverage = gw.top_layers(damage, 2)
    assert layers == [0, 2] and abs(coverage - 0.8) < 1e-12
    shares = gw.head_shares([dict(A=3.0, Az=1.0, K=0.0, Z=4.0), dict(A=1.0, Az=1.0, K=1.0, Z=1.0)])
    assert abs(shares['A_via_future'] - (0.25 + 0.5) / 2) < 1e-12
    assert abs(shares['Z'] - (0.5 + 0.25) / 2) < 1e-12
    assert abs(gw.ess([1.0] * 10) - 10) < 1e-12 and abs(gw.ess([1.0] + [0.0] * 9) - 1) < 1e-12
    u = dict(mixed_loss=1.0, first10motor_mse=1.0, first10motor_mse_swap=0.5)
    better = dict(mixed_loss=0.9, first10motor_mse=0.9, first10motor_mse_swap=0.5)
    worse = dict(mixed_loss=1.1, first10motor_mse=1.0, first10motor_mse_swap=0.4)
    d = gw.decide(dict(a=dict(U=u, G=worse, O=better), b=dict(U=u, G=worse, O=better), c=dict(U=u, G=better, O=worse)), 2)
    assert d['go'] and d['passed'] == 2 and d['hosts']['c']['G']['c1_direction'] and not d['hosts']['a']['G']['mixed_down']
    assert not gw.decide(dict(a=dict(U=u, G=u, O=better), b=dict(U=u, G=u, O=worse)), 2)['go']


def test_bf16_reader_context():
    work = object.__new__(gw.GateWork)
    work.runtime, work.masters = object(), {'a': torch.ones(2, dtype=torch.bfloat16)}
    gw.fr._bf16_forward_context = gw.fr._checkpoint_context_fn = SENTINEL['patched']
    with work.bf16_reader() as reader:
        assert gw.fr._bf16_forward_context is SENTINEL['original'] and gw.fr._checkpoint_context_fn is SENTINEL['original']
        assert set(reader.original_weights) == {'a'}
    assert gw.fr._bf16_forward_context is SENTINEL['patched'] and gw.fr._checkpoint_context_fn is SENTINEL['patched']
    assert reader.original_weights == {}


def build_layers(work, shapes, selected):
    """Masters, H_U, phase-1 H~, H~_op, baseline and phase-1 arm weights, baseline seconds, modules.json, selection."""
    g = torch.Generator().manual_seed(0)
    prep, modules = work.baseline / 'prep', []
    for i, (o, n) in enumerate(shapes):
        name = f'video_expert.blocks.{i}.ffn.0'
        modules.append(dict(index=i, module=name, shape=[o, n]))
        w = torch.randn(o, n, generator=g)
        h = (lambda x: x.t() @ x / 64)(torch.randn(64, n, generator=g))
        tsave(prep / 'masters' / f'{i:04d}.pt', dict(module=name, weight=w.to(torch.bfloat16)))
        tsave(prep / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=h))
        tsave(work.base / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=h * 2))
        for host, c in gw.HOSTS.items():
            tsave(work.gate(host) / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=h * 3))
            for arm, level in ((c['baseline_arm'], 0.05), (c['bf16_point_arm'], 0.04)):
                q = (w + torch.randn(o, n, generator=g) * level).to(torch.bfloat16)
                tsave(work.arm_root(arm) / 'weights' / f'{i:04d}.pt', dict(module=name, weight=q))
            save(work.arm_root(c['baseline_arm']) / 'stats' / f'{i:04d}.json', dict(seconds=1.0 + i))
    save(prep / 'modules.json', modules)
    save(work.gate() / 'selection.json', dict(hosts={h: dict(layers=selected) for h in gw.HOSTS}))
    return modules


def test_install_routing():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        modules = build_layers(work, [(8, 4)] * 4, [1, 3])
        for i, m in enumerate(modules):
            tsave(work.gate('GPTVQ') / 'weights' / f'{i:04d}.pt', dict(module=m['module'], weight=torch.full((8, 4), 7.0).to(torch.bfloat16)))
        layers = {m['module']: types.SimpleNamespace(weight=torch.zeros(8, 4, dtype=torch.bfloat16)) for m in modules}
        work.bank = types.SimpleNamespace(names=[m['module'] for m in modules], modules=layers)

        def stored(arm, i):
            return torch.load(work.arm_root(arm) / 'weights' / f'{i:04d}.pt', weights_only=True)['weight']
        work.install(gw.gate_arm('GPTVQ', 'O'))
        for i, m in enumerate(modules):
            want = torch.full((8, 4), 7.0).to(torch.bfloat16) if i in (1, 3) else stored('GPTVQ_W2', i)
            assert torch.equal(layers[m['module']].weight, want), i
        work.install(gw.gate_arm('QTIP', 'G'))
        for i, m in enumerate(modules):
            assert torch.equal(layers[m['module']].weight, stored('QTIP_G_W2' if i in (1, 3) else 'QTIP_U_W2', i)), i


def test_quant():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        build_layers(work, [(16, 8), (8, 16), (16, 16)], [0, 2])
        work.quant(0, 2, device='cpu')
        work.quant(1, 2, device='cpu')
        for host in gw.HOSTS:
            assert sorted(p.name for p in (work.gate(host) / 'weights').glob('*.pt')) == ['0000.pt', '0002.pt']
            s = wk.read(work.gate(host) / 'wstats' / '0002.json')
            assert s['hessian'] == 'H_O' and s['bpw'] == 2.0 and set(s['proxies']) == {'U', 'G', 'O'}
            assert all(set(v) == {'H_U', 'H_G', 'H_O'} and all(x > 0 for x in v.values()) for v in s['proxies'].values())


def synthetic_stats(work, n):
    """C80 statistics per host (separate heads on the diagnostic contexts), sigma, the phase-1 diagnostic masses."""
    rnd = random.Random(1)
    diag = set(work.diagnostic_cases())
    groups = {h: dict(video_steps=[1.0] * 10, video_prefill=1.0, action_steps=[1.0] * 10, other=0.0) for h in ('A', 'V')}
    for case in range(2000, 2080):
        save(work.base / 'stats' / f'{case}.json', dict(case_id=case, separate=case in diag, groups=groups))
        if case in diag:
            masses = {h: {f'{i}|0': (2.0 if h == 'Az' else 1.0) for i in range(n)} for h in gw.HEADS}
            tsave(work.base / 'stats' / f'{case}.pt', dict(masses=masses))
    for host in gw.HOSTS:
        save(work.gate(host) / 'sigma.json', dict(sigma2=[1.0 + i / n for i in range(n)]))
        for case in range(2000, 2080):
            heads = gw.HEADS if case in diag else ('A', 'V')
            layer_mass = {h: [rnd.random() for _ in range(n)] for h in heads}
            if case in diag:
                layer_mass['Az'] = [3 * v for v in layer_mass['A']]
            save(work.gate(host) / 'stats' / f'{case}.json', dict(
                case_id=case, separate=case in diag, layer_mass=layer_mass, groups=groups,
                parity=dict(future_vs_kpass=0.0), calls=dict(signature_worst_relative=0.0),
                student_first10motor_mse=0.002, bf16_reader_first10motor_mse=0.001))


def test_select():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        n = 614
        save(work.baseline / 'prep' / 'modules.json', [dict(index=i, module=f'm{i}') for i in range(n)])
        synthetic_stats(work, n)
        work.select()
        sel = wk.read(work.gate() / 'selection.json')
        assert abs(sel['bf16_point']['head_share_unit_noise']['A_via_future'] - 2 / 3) < 1e-9
        assert abs(sel['bf16_point']['context_ess_unit_noise'] - 80) < 1e-9
        for host in gw.HOSTS:
            r = sel['hosts'][host]
            assert len(r['layers']) == gw.GATE['selection']['layers'] and r['modules'][0] == f"m{r['layers'][0]}"
            assert abs(r['head_share_unit_noise']['A_via_future'] - 0.75) < 1e-9
            assert abs(r['head_share_sigma_weighted']['A_via_future'] - 0.75) < 1e-9
            assert abs(r['head_share_sigma_weighted_bf16_point']['A_via_future'] - 2 / 3) < 1e-9
            assert 0 < r['coverage'] < 1 and 1 < r['context_ess'] <= 80
            rows = [wk.read(work.gate(host) / 'stats' / f'{c}.json') for c in range(2000, 2080)]
            sigma = wk.read(work.gate(host) / 'sigma.json')['sigma2']
            damage = [sigma[i] * sum(sum(x['layer_mass'][h][i] for h in x['layer_mass']) for x in rows) / 80 for i in range(n)]
            assert r['layers'] == sorted(sorted(range(n), key=lambda i: -damage[i])[:len(r['layers'])])


def test_final():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        build_layers(work, [(16, 8), (8, 16)], [0, 1])
        work.quant(0, 1, device='cpu')
        synthetic_stats(work, 614)
        save(work.baseline / 'prep' / 'modules.json', [dict(index=i, module=f'm{i}') for i in range(614)])
        work.select()
        sel = wk.read(work.gate() / 'selection.json')
        for h in gw.HOSTS:
            sel['hosts'][h]['layers'] = [0, 1]
        save(work.gate() / 'selection.json', sel)

        def held(arm, mixed, swap):
            save(work.base / 'heldout' / f'{arm}.json', dict(overall=dict(
                mixed_loss=mixed, first10motor_mse=1.0, first10motor_mse_swap=swap, first10motor_rmse=1.0,
                future_latent_nmse=0.05, first10gripper_rmse=0.08)))
        for h, c in gw.HOSTS.items():
            held(c['baseline_arm'], 1.0, 0.5)
            held(c['bf16_point_arm'], 1.1, 0.45)
            held(gw.gate_arm(h, 'G'), 1.01, 0.49)
            held(gw.gate_arm(h, 'O'), 0.98 if h != 'QUIPSHARP' else 1.02, 0.55)
            save(work.gate(h) / 'ucheck.json', dict(relative=dict(mixed_loss=0.0, first10motor_mse=1e-7)))
        s = work.final()
        assert s['decision']['go'] and s['decision']['passed'] == 2
        assert wk.read(work.gate() / 'DECISION.json')['go']
        text = (work.gate() / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert '2/3 个宿主满足，结论：推进全量' in text and 'QTIP W2' in text


def test_full():
    """full_quant reuses the gate's selected layers bitwise and quantizes the rest; full_complete / full_final write the
    receipts, paired counts against the baseline and the phase-1 arm, and the pooled line."""
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        build_layers(work, [(16, 8), (8, 16), (16, 16)], [1])
        work.quant(0, 1, device='cpu')
        work.full_quant(0, 2, device='cpu')
        work.full_quant(1, 2, device='cpu')
        for host in gw.HOSTS:
            arm = work.arm_root(gw.full_arm(host))
            assert arm == work.base / 'quantized' / gw.full_arm(host)
            assert sorted(p.name for p in (arm / 'weights').glob('*.pt')) == ['0000.pt', '0001.pt', '0002.pt']
            mine = torch.load(arm / 'weights' / '0001.pt', weights_only=True)['weight']
            assert torch.equal(mine, torch.load(work.gate(host) / 'weights' / '0001.pt', weights_only=True)['weight'])
            s = wk.read(arm / 'stats' / '0002.json')
            assert s['proxy_U'] == s['proxies']['O']['H_U'] and s['proxy_O'] == s['proxies']['O']['H_O']
        save(work.gate() / 'DECISION.json', dict(go=True, passed=2))
        rnd = random.Random(3)
        for h, c in gw.HOSTS.items():
            for role, root in (('U', work.baseline / 'loops' / c['baseline_loop']), ('G', work.base / 'loops' / c['bf16_point_arm']),
                               ('O', work.base / 'loops' / gw.full_arm(h))):
                for k in range(280):
                    save(root / 'episodes' / f'{k}.json', dict(success=rnd.random() < (0.8 if role == 'O' else 0.7)))
                (root / 'DONE').write_text('x')
            for arm in (c['baseline_arm'], c['bf16_point_arm'], gw.full_arm(h)):
                save(work.base / 'heldout' / f'{arm}.json', dict(overall=dict(
                    mixed_loss=1.0, first10motor_mse=1.0, first10motor_mse_swap=0.5, first10motor_rmse=1.0,
                    future_latent_nmse=0.05, first10gripper_rmse=0.08, first10motor_rmse_swap=0.7)))
        s = work.full_final()
        for h in gw.HOSTS:
            lp_ = s['hosts'][h]['loop']
            assert lp_['episodes'] == 280 and lp_['O'] > lp_['U'] - 40
            assert wk.read(work.arm_root(gw.full_arm(h)) / 'COMPLETE.json')['complete']
        assert s['pooled']['vs_U']['wins'] == sum(s['hosts'][h]['loop']['vs_U']['wins'] for h in gw.HOSTS)
        assert '三宿主合计' in (work.base / 'full' / 'RESULTS.zh.md').read_text(encoding='utf-8')


if __name__ == '__main__':
    for test in (test_pure, test_bf16_reader_context, test_install_routing, test_quant, test_select, test_final, test_full):
        test()
        print('ok', test.__name__)
    print('ALL PASSED')

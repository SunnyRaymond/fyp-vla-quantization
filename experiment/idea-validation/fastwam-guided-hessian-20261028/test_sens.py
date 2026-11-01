"""CPU checks for FLAQ module 1 (sens.json, sens_worker.py) with a stubbed runtime: the 64-group layout on the real
614 names, the rank / split / allocation helpers, the per-case terms, and an end-to-end toy (614 tiny Linears, W2 / W3
files, a query whose A / Z / K depend on the installed weights) through four shards and the final summary: configs,
exact restore, anchor, ranking recovery, selection within budget, report."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types

import torch

HERE = Path(__file__).resolve().parent
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {}),
                    ('frozen_reader', dict(_bf16_forward_context=None, _checkpoint_context_fn=None, BF16Reader=None)),
                    ('reader_loss', dict(reader_terms=lambda torch_, read, teacher, stats, config:
                                         dict(reader_kl=((read - teacher) ** 2).mean())))):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import sens_worker as sw  # noqa: E402

LAM, BETA = 0.04, 0.5
CASES = list(range(2000, 2080))
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


def test_groups():
    names = real_names()
    grp = sw.groups(names)
    assert len(grp) == 64 and sum(len(v) for v in grp.values()) == 614
    assert sorted(i for v in grp.values() for i in v) == list(range(614))
    assert [names[i] for i in grp['action.io']] == ['action_expert.action_encoder', 'action_expert.head', 'proprio_encoder']
    assert [names[i] for i in grp['video.head']] == ['video_expert.head.head']
    assert all(names[i].startswith('video_expert.blocks.7.') for i in grp['video.b07'])
    assert len(sw.config_list(grp)) == 66 and sw.config_list(grp)[:2] == ['none', 'all']
    assert sw.members('all', grp, 614) == list(range(614)) and sw.members('none', grp, 614) == []
    for bad in (names[:-1], names + ['video_expert.blocks.30.ffn.0']):
        try:
            sw.groups(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('layout check missed a change')
    try:
        sw.group_of('something_else')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown name accepted')


def test_helpers():
    assert sw.ranks([3.0, 1.0, 2.0, 2.0]) == [4.0, 1.0, 2.5, 2.5]
    assert abs(sw.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(sw.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert sw.spearman([1, 1, 1], [1, 2, 3]) == 0.0
    m, se = sw.mean_se([1.0, 2.0, 3.0])
    assert m == 2.0 and abs(se - (1 / 3) ** 0.5) < 1e-12
    suite = {c: f's{(c - 2000) // 20}' for c in CASES}
    first, second = sw.suite_split(CASES, suite, random.Random(0))
    assert len(first) == len(second) == 40 and not set(first) & set(second)
    assert all(sum(suite[c] == s for c in first) == 10 for s in set(suite.values()))
    dens = dict(a=5.0, b=4.0, c=3.0, d=2.0, e=1.0)
    means = dict(a=1.0, b=1.0, c=-1.0, d=1.0, e=1.0)
    bits = dict(a=10, b=30, c=1, d=5, e=5)
    chosen, used = sw.allocate(dens, means, bits, 20)
    assert chosen == ['a', 'd', 'e'] and used == 20  # b does not fit, c does not help
    rows = [dict(case_id=1, mixed_loss=0.3, future_latent_nmse=2.0, reader_kl=0.4)]
    t = sw.terms(rows, LAM, BETA)[1]
    assert abs(t['A'] - (0.3 - LAM * 2.0)) < 1e-12 and abs(t['A'] + t['Z'] + t['K'] - t['T']) < 1e-12
    assert sw.scalars(dict(a=1, b=torch.tensor(2.5), c=[1, 2], d='x')) == dict(a=1.0, b=2.5)
    ok = sw.anchor_check([dict(case_id=1, mixed_loss=1.0, future_latent_nmse=2.0)],
                         [dict(case_id=1, mixed_loss=1.0, future_latent_nmse=2.0)], 1e-6)
    bad = sw.anchor_check([dict(case_id=1, mixed_loss=1.0, future_latent_nmse=2.0)],
                          [dict(case_id=1, mixed_loss=1.0001, future_latent_nmse=2.0)], 1e-6)
    assert ok['passed'] and not bad['passed']


def test_takeover_split():
    """fwse1x takes over shards 1-3 of the 4-way split: its 4 processes get exactly those configs, once each, none of
    shard 0's, 12-13 each; without cover the split is the plain 4-way one."""
    n = 66
    plain = [sw.assigned(n, k, 4) for k in range(4)]
    assert sorted(i for p in plain for i in p) == list(range(n))
    taken = [sw.assigned(n, k, 4, [1, 2, 3], 4) for k in range(4)]
    flat = [i for p in taken for i in p]
    assert sorted(flat) == sorted(i for p in plain[1:] for i in p) and len(flat) == len(set(flat)) == 49
    assert not set(flat) & set(plain[0]) and all(len(p) in (12, 13) for p in taken)


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def jsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def toy(root):
    """614 Linears of 2x2; W2 and W3 files (W3 = W2 / 2); per Linear a fixed sensitivity. Z is driven by the video
    Linears, A by the action Linears plus a little of the video ones, K by Z (the reader reads the future)."""
    names = real_names()
    g = torch.Generator().manual_seed(0)
    work = object.__new__(sw.SensWork)
    work.base = root / 'guided' / 'results'
    work.baseline = root / 'baseline' / 'results'
    work.cfg = dict(loss={'lambda': LAM, 'beta': BETA})
    work.job, work.job_name = 'TEST', 'fwse1_0'
    work.enough = lambda: None
    work.progress = lambda *a, **k: None
    work.count = lambda key: None
    work.load = lambda: None
    work.capture_masters = lambda: None
    modules = {n: types.SimpleNamespace(weight=torch.zeros(2, 2)) for n in names}
    work.bank = types.SimpleNamespace(names=names, modules=modules)
    q = work.baseline / 'quantized'
    sens = torch.rand(614, generator=g) + 0.1
    for i, n in enumerate(names):
        w2 = torch.randn(2, 2, generator=g)
        tsave(q / 'QTIP_W2' / 'weights' / f'{i:04d}.pt', dict(module=n, weight=w2))
        tsave(q / 'QTIP_W3' / 'weights' / f'{i:04d}.pt', dict(module=n, weight=w2 / 2))
        jsave(q / 'QTIP_W2' / 'stats' / f'{i:04d}.json', dict(module=n, bits_total=8 + (i % 3)))
        jsave(q / 'QTIP_W3' / 'stats' / f'{i:04d}.json', dict(module=n, bits_total=12 + (i % 3) + (4 if 'ffn' in n else 0)))
    jsave(q / 'QTIP_W2' / 'COMPLETE.json', dict(complete=True, bpw=2.0, target_weights=614 * 4))
    jsave(q / 'QTIP_W3' / 'COMPLETE.json', dict(complete=True, bpw=3.0, target_weights=614 * 4))
    jsave(work.baseline / 'prep' / 'modules.json', [dict(module=n) for n in names])
    video = [i for i, n in enumerate(names) if n.startswith('video_expert.')]
    action = [i for i, n in enumerate(names) if not n.startswith('video_expert.')]

    def energy(idx):
        return sum(float(sens[i]) * float(modules[names[i]].weight.square().sum()) for i in idx)

    def query(case, seed=0, grad=False):
        f = 1 + 0.2 * ((case * 7) % 5) / 5
        z = f * energy(video)
        a = f * (energy(action) + 0.1 * energy(video))
        action_t = torch.zeros(32, 7)
        action_t[0, 0] = a
        return action_t, torch.tensor([z, 0.0, 0.0, 0.0]), dict(video_steps=10, action_steps=10)

    def metrics(action, future, target):
        a, z = float(action[0, 0]), float(future[0])
        return dict(mixed_loss=torch.tensor(a + LAM * z), future_latent_nmse=z, first10motor_rmse=a ** 0.5,
                    first10gripper_rmse=0.1, per_suite=[1, 2])

    @contextmanager
    def bf16_reader():
        yield lambda runtime, value, seed, future, checkpointed=True: (torch.full((32, 7), float(future[0]) ** 0.5), None)

    work.query, work.metrics, work.bf16_reader = query, metrics, bf16_reader
    work.teacher = lambda case, seed=0: dict(action=torch.zeros(32, 7), future=torch.zeros(4))
    work.datum = lambda case: (None, 'task')
    work.rows = {c: dict(sampler_seeds=[c], suite=f'libero_{(c - 2000) // 20}') for c in CASES}
    work.runtime = types.SimpleNamespace(description=None)
    work.p = dict(reader_source=dict(action_stats=None, config=None))
    return work, sens


def run_all(work, shards=4):
    for k in range(shards):
        work.job_name = f'fwse1_{k}'
        work.sens(k, shards)
    work.job_name = 'fwse1_0'


def test_put_identity():
    with tempfile.TemporaryDirectory() as tmp:
        work, _ = toy(Path(tmp))
        work.setup_host()
        work.bank.names = list(reversed(work.bank.names))
        try:
            work.put('base', [0])
        except RuntimeError:
            pass
        else:
            raise AssertionError('identity mismatch accepted')


def test_end_to_end():
    os.environ.pop('FW_HOSTS', None)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work, sens = toy(root)
        work.setup_host()
        sdir = work.sdir
        # one shard alone leaves configs missing: final waits
        work.job_name = 'fwse1g2_2'
        work.sens(2, 4)
        work.final()
        assert (sdir / 'waiting' / 'fwse1g2_2.json').exists() and not (sdir / 'summary.json').exists()
        run_all(work)
        configs = sorted(p.stem for p in (sdir / 'configs').glob('*.json'))
        assert len(configs) == 66
        # base weights back exactly
        for i, n in enumerate(work.bank.names):
            saved = torch.load(work.baseline / 'quantized' / 'QTIP_W2' / 'weights' / f'{i:04d}.pt', weights_only=True)
            assert torch.equal(work.bank.modules[n].weight, saved['weight'])
        shard = json.loads((sdir / 'shards' / 'fwse1_1.json').read_text())
        assert shard['restore_exact'] and len(shard['configs']) in (16, 17)
        none = json.loads((sdir / 'configs' / 'none.json').read_text())['rows']
        assert len(none) == 80 and none[0]['case_id'] == 2000 and 'per_suite' not in none[0]
        g = json.loads((sdir / 'configs' / 'video.b03.json').read_text())
        assert len(g['modules']) == 10 and all(m.startswith('video_expert.blocks.3.') for m in g['modules'])
        # final without the anchor file fails loudly; with it, the summary is complete
        try:
            work.final()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('missing anchor accepted')
        jsave(work.base / 'tune' / 'QTIP' / 'fit_before.json',
              dict(rows=[dict(case_id=r['case_id'], mixed_loss=r['mixed_loss'], future_latent_nmse=r['future_latent_nmse'])
                         for r in none]))
        work.final()
        s = json.loads((sdir / 'summary.json').read_text())
        assert s['anchor']['passed'] and s['restore']['exact'] and s['restore']['start_equals_none']
        assert s['restore']['shards'] == 5  # fwse1_0..3 and fwse1g2_2
        assert s['decision']['valid'] and s['decision']['stable'] and s['decision']['go']
        assert s['stability']['T']['even_odd']['overlap'] == 1.0 and s['stability']['T']['even_odd']['rho'] > 0.99
        # ranking recovers the planted per-group sensitivity: benefit per group = f * sens-weighted energy removed
        names = work.bank.names
        grp = sw.groups(names)
        w2 = {i: torch.load(work.baseline / 'quantized' / 'QTIP_W2' / 'weights' / f'{i:04d}.pt', weights_only=True)['weight']
              for i in range(614)}
        bits = {gname: sum((12 + (i % 3) + (4 if 'ffn' in names[i] else 0)) - (8 + (i % 3)) for i in idx)
                for gname, idx in grp.items()}
        planted = {}
        for gname, idx in grp.items():
            removed = sum(float(sens[i]) * 0.75 * float(w2[i].square().sum()) for i in idx)
            video = names[idx[0]].startswith('video_expert.')
            planted[gname] = (removed * (0.1 + LAM + BETA) if video else removed) / bits[gname]  # K = z here
        want = sorted(planted, key=lambda x: -planted[x])
        assert s['rank_total'][:10] == want[:10], (s['rank_total'][:10], want[:10])
        # A-only ranking ignores Z and K: video groups lose ground
        assert s['rank_action'] != s['rank_total']
        for gname, row in s['groups'].items():
            parts = row['A']['mean'] + row['Z']['mean'] + row['K']['mean']
            assert abs(parts - row['T']['mean']) <= 1e-9 * max(1.0, abs(row['T']['mean']))
            assert row['extra_bits'] == bits[gname]
        for name in ('total', 'action'):
            for b, x in s['selection'][name].items():
                assert x['extra_bits'] <= float(b) * 614 * 4 + 1e-9
                assert abs(x['bpw'] - (2.0 + x['extra_bits'] / (614 * 4))) < 1e-12
        add = s['additivity']
        assert abs(add['sum_single']['Z'] - add['all']['Z']) <= 1e-6 * abs(add['all']['Z'])  # Z is additive in the toy
        sel = json.loads((sdir / 'selection.json').read_text())
        assert len(sel['groups']) == 64 and sel['selection'] == s['selection']
        text = (sdir / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert '稳定性' in text and '进入步骤 2' in text and s['rank_total'][0] in text
        work.final()  # idempotent once the summary exists


def test_unstable_ranking():
    """Groups with no planted difference: the even/odd top-10 overlap falls under the bar and the decision is no-go."""
    gs = [f'video.b{i:02d}' for i in range(30)] + [f'action.b{i:02d}' for i in range(30)] + \
         ['video.cond', 'video.head', 'action.cond', 'action.io']
    rng = random.Random(1)
    rows = {}

    def row(c, base):
        return dict(case_id=c, suite=f's{(c - 2000) // 20}', mixed_loss=base, future_latent_nmse=1.0, reader_kl=0.1,
                    first10motor_rmse=0.1, reader_first10motor_mse=0.01)
    rows['none'] = [row(c, 1.0) for c in CASES]
    rows['all'] = [row(c, 0.5) for c in CASES]
    for g in gs:
        rows[g] = [row(c, 1.0 - 0.01 * (1 + rng.gauss(0, 1))) for c in CASES]
    out = sw.analyze(rows, {g: 1e6 for g in gs}, 4e9, 2.0, {c: f's{(c - 2000) // 20}' for c in CASES}, LAM, BETA)
    assert out['stability']['T']['even_odd']['overlap'] < 0.8 and not out['decision']['go']
    assert 0 <= out['stability']['T']['random_overlap_p10'] <= out['stability']['T']['random_overlap_mean'] <= 1


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print('ok', t.__name__, flush=True)
    print(f'all {len(tests)} tests passed')

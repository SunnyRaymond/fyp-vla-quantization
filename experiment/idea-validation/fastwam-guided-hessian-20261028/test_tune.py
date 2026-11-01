"""CPU checks for the C2 continuous-stage test (tune.json, tune_worker.py) with a stubbed runtime: ScaledLinear forward
and gradients against plain autograd, bitwise identity at s = t = 0, the LR rule, the Q0-recipe training loop on a toy
loss (checkpoint, resume, moment reset), folding with bits and proxies, and the final / closed-loop reports."""
import json
from pathlib import Path
import random
import sys
import tempfile
import types

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {}),
                    ('frozen_reader', dict(_bf16_forward_context=None, _checkpoint_context_fn=None, BF16Reader=None))):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import tune_worker as tw  # noqa: E402
import worker as wk  # noqa: E402

HOST = json.loads((HERE / 'host.json').read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    return value


def make_work(root, shapes):
    work = object.__new__(tw.TuneWork)
    work.base = work.results = root / 'guided' / 'results'
    work.root = root / 'remote'  # lp.Work's remote-root attribute, which must not shadow a method
    work.baseline, work.host, work.job, work.job_name = root / 'baseline' / 'results', HOST, 'TEST', 'fwgt_0'
    work.enough = lambda: None
    work.progress = lambda *args, **kwargs: None
    work.atomic_torch_save = lambda _torch, path, value: torch.save(value, path)
    work.to_cpu, work.optimizer_to_device = to_cpu, lambda *args: None
    names = [f'video_expert.blocks.{i}.ffn.0' for i in range(len(shapes))]
    modules = {n: types.SimpleNamespace(out_features=o, in_features=i, forward=lambda x: x) for n, (o, i) in zip(names, shapes)}
    work.bank = types.SimpleNamespace(names=names, modules=modules, seen=set())
    work.schedule = [[2000 + 10 * b + k for k in range(10)] for b in range(8)]
    work.attach()
    return work


def test_scaled_linear():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 5, 6, generator=g, requires_grad=True)
    w, b = torch.randn(4, 6, generator=g), torch.randn(4, generator=g)
    s = (torch.randn(4, generator=g) * 0.1).requires_grad_(True)
    t = (torch.randn(6, generator=g) * 0.1).requires_grad_(True)
    gy = torch.randn(2, 5, 4, generator=g)
    y = tw.ScaledLinear.apply(x, s, t, w, b)
    gx, gs, gt = torch.autograd.grad(y, (x, s, t), gy)
    x2, s2, t2 = x.detach().clone().requires_grad_(True), s.detach().clone().requires_grad_(True), t.detach().clone().requires_grad_(True)
    y2 = F.linear(x2, s2.exp()[:, None] * w * t2.exp()[None, :], b)
    rx, rs, rt = torch.autograd.grad(y2, (x2, s2, t2), gy)
    assert torch.allclose(y, y2, atol=1e-5)
    for a, r in ((gx, rx), (gs, rs), (gt, rt)):
        assert torch.allclose(a, r, rtol=1e-4, atol=1e-5), (a, r)
    wb = torch.randn(4, 6, generator=g).to(torch.bfloat16)
    assert torch.equal(tw.effective(wb, torch.zeros(4), torch.zeros(6)), wb)
    xb = torch.randn(3, 6, generator=g).to(torch.bfloat16)
    assert torch.equal(tw.ScaledLinear.apply(xb, torch.zeros(4), torch.zeros(6), wb, None), F.linear(xb, wb))


def test_active_forward_marks_seen():
    """The sampler checks bank.seen after every query: the tuned forward must register each Linear it runs."""
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(4, 6)])
        name = work.bank.names[0]
        mod = work.bank.modules[name]
        mod.weight, mod.bias = torch.randn(4, 6), None
        work.active = True
        y = mod.forward(torch.randn(2, 6))
        assert name in work.bank.seen and tuple(y.shape) == (2, 4)


def test_rules():
    rows = [dict(lr=5e-4, predicted=1.0, realized=0.9), dict(lr=2e-3, predicted=4.0, realized=2.5),
            dict(lr=8e-3, predicted=16.0, realized=3.0)]
    assert tw.first_step_choice(rows, 0.5) == 2e-3
    assert tw.first_step_choice([dict(lr=1e-3, predicted=1.0, realized=-0.1)], 0.5) is None
    assert tw.extra_bits([4, 6], True) == 0 and tw.extra_bits([4, 6], False) == 160
    u = dict(mixed_loss=1.0, future_latent_nmse=0.1)
    hosts = dict(a=dict(tuned=True, fit=dict(before=1.0, after=0.9), heldout=dict(U=u, T=dict(mixed_loss=0.96, future_latent_nmse=0.09))),
                 b=dict(tuned=True, fit=dict(before=1.0, after=0.9), heldout=dict(U=u, T=dict(mixed_loss=0.98, future_latent_nmse=0.09))),
                 c=dict(tuned=False))
    d = tw.decide(hosts, dict(min_hosts=2, mixed_ratio_max=0.97))
    assert d['passed'] == 1 and not d['go'] and d['hosts']['a']['passed'] and not d['hosts']['b']['passed']


def toy(work):
    """A + lambda Z + beta K stand-in: sum (s - 0.01)^2 + (t + 0.02)^2; with grad its gradient lands in .grad."""
    def batch(cases, grad):
        with torch.set_grad_enabled(grad):
            total = sum(((work.s[n] - 0.01) ** 2).sum() + ((work.t[n] + 0.02) ** 2).sum() for n in work.bank.names)
        if grad:
            total.backward()
        return dict(total=float(total), az=float(total), k=0.0)
    return batch


def test_step_and_train():
    """Toy gradients g_s = -0.02, g_t = 0.04 at 0: the step -eta g / max|g| moves t by eta and s by eta / 2; the
    predicted decrease is eta |g|^2 / max|g|. Training reaches the toy optimum and stays (zero gradient: no step)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(4, 6), (3, 5)])
        work.batch = toy(work)
        out = work.step_check('QTIP')
        want = tw.first_step_choice(out['rows'], tw.TUNE['training']['min_realized_fraction'])
        assert out['chosen'] == want and want is not None
        assert abs(out['grad_max'] - 0.04) < 1e-6 and out['scales'] == 4 + 6 + 3 + 5
        for row in out['rows']:
            eta = row['lr']
            assert abs(row['predicted'] - eta * (7 * 0.02 ** 2 + 11 * 0.04 ** 2) / 0.04) < 1e-6
            realized = 7 * (0.01 ** 2 - (eta / 2 - 0.01) ** 2) + 11 * (0.02 ** 2 - (0.02 - eta) ** 2)
            assert abs(row['realized'] - realized) < 1e-6
        assert all(float(p.abs().max()) == 0 for p in work.parameters())
        state = work.train('QTIP', want)
        assert state['updates'] == 8 and len(state['history']) == 8
        losses = [h['before']['total'] for h in state['history']]
        assert min(losses[1:]) < losses[0]  # fixed-size normalized steps may oscillate around the toy optimum
        snap = work.snapshot()
        again = make_work(Path(tmp), [(4, 6), (3, 5)])
        again.batch = toy(again)
        state2 = again.train('QTIP', want)
        assert len(state2['history']) == 8
        assert all(torch.equal(again.s[n], snap['s'][n]) for n in again.bank.names)
        assert wk.read(work.tune_dir('QTIP') / 'training.json')['eta'] == want


def build(work, host, shapes):
    g = torch.Generator().manual_seed(1)
    prep, c = work.baseline / 'prep', tw.HOSTS[host]
    for i, (name, (o, n)) in enumerate(zip(work.bank.names, shapes)):
        w = torch.randn(o, n, generator=g)
        x = torch.randn(64, n, generator=g)
        tsave(prep / 'masters' / f'{i:04d}.pt', dict(module=name, weight=w.to(torch.bfloat16)))
        tsave(prep / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=x.t() @ x / 64))
        q = (w + torch.randn(o, n, generator=g) * 0.05).to(torch.bfloat16)
        tsave(work.arm_root(c['baseline_arm']) / 'weights' / f'{i:04d}.pt', dict(module=name, weight=q))
        save(work.arm_root(c['baseline_arm']) / 'stats' / f'{i:04d}.json', dict(bits_total=2.0 * o * n, seconds=1.0))


def test_fold():
    shapes = [(4, 6), (3, 5)]
    with tempfile.TemporaryDirectory() as tmp:
        for host in ('QTIP', 'GPTVQ'):
            work = make_work(Path(tmp) / host, shapes)
            build(work, host, shapes)
            with torch.no_grad():
                for n in work.bank.names:
                    work.s[n].fill_(0.02)
                    work.t[n].fill_(-0.01)
            done = work.fold(host)
            arm = work.arm_root(tw.tuned_arm(host))
            assert arm == work.base / 'quantized' / tw.tuned_arm(host)
            q = torch.load(work.arm_root(tw.HOSTS[host]['baseline_arm']) / 'weights' / '0001.pt', weights_only=True)['weight']
            mine = torch.load(arm / 'weights' / '0001.pt', weights_only=True)['weight']
            assert torch.equal(mine, tw.effective(q, work.s[work.bank.names[1]].detach(), work.t[work.bank.names[1]].detach()))
            extra = 0 if tw.HOSTS[host]['free_scales'] else 16 * sum(o + n for o, n in shapes)
            weights = sum(o * n for o, n in shapes)
            assert abs(done['bpw'] - (2.0 * weights + extra) / weights) < 1e-12
            assert done['proxy_nmse_weighted'] > 0 and abs(done['s_abs_max'] - 0.02) < 1e-7


def test_reports():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(4, 6)])
        rnd = random.Random(2)
        for host, c in tw.HOSTS.items():
            r = work.tune_dir(host)
            tuned = host != 'GPTVQ'
            save(r / 'host.json', dict(tuned=tuned, eta=0.01))
            save(r / 'step.json', dict(rows=[dict(lr=0.01, predicted=1.0, realized=0.8, visible=5)], scales=10,
                                       chosen=0.01 if tuned else None, top_layers=[dict(module='m0', share=0.5)]))
            if not tuned:
                continue
            save(r / 'training.json', dict(history=[dict(before=dict(total=0.01 - 0.001 * k)) for k in range(8)]))
            for key, v in (('before', 0.0050), ('after', 0.0045)):
                save(r / f'fit_{key}.json', dict(mixed_loss=v, first10motor_rmse=0.03, future_latent_nmse=0.05))
            save(work.arm_root(tw.tuned_arm(host)) / 'COMPLETE.json', dict(
                bpw=2.016, proxy_nmse_weighted=0.007, baseline_proxy_U_weighted=0.006, s_abs_max=0.02, t_abs_max=0.01))
            for arm, mixed in ((c['baseline_arm'], 0.0040), (tw.tuned_arm(host), 0.0037)):
                save(work.base / 'heldout' / f'{arm}.json', dict(overall=dict(
                    mixed_loss=mixed, first10motor_rmse=0.03, future_latent_nmse=0.05 if mixed > 0.0038 else 0.045,
                    first10motor_mse=1.0, first10motor_mse_swap=0.5, first10gripper_rmse=0.08, first10motor_rmse_swap=0.02)))
            for root, p in ((work.baseline / 'loops' / c['baseline_loop'], 0.7), (work.base / 'loops' / tw.tuned_arm(host), 0.75)):
                for k in range(280):
                    save(root / 'episodes' / f'{k}.json', dict(success=rnd.random() < p))
                (root / 'DONE').write_text('x')
        s = work.final()
        assert s['decision']['go'] and s['decision']['passed'] == 2 and not s['decision']['hosts']['GPTVQ']['tuned']
        assert '2/3 个宿主满足，结论：跑闭环' in (work.tune_dir() / 'RESULTS.zh.md').read_text(encoding='utf-8')
        loop = work.loop_final()
        assert set(loop['hosts']) == {'QTIP', 'QUIPSHARP'} and loop['pooled']['wins'] == sum(r['wins'] for r in loop['hosts'].values())
        assert '三宿主合计' in (work.tune_dir() / 'loop' / 'RESULTS.zh.md').read_text(encoding='utf-8')


def test_split_hosts():
    import os
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp), [(4, 6)])
        done = []
        work.tune = done.append
        os.environ.pop('FW_HOSTS', None)
        work.stage('tune', 2, 3)
        os.environ['FW_HOSTS'] = 'QTIP'
        work.stage('tune', 0, 1)
        os.environ['FW_HOSTS'] = 'GPTVQ:QUIPSHARP'
        work.stage('tune', 1, 2)
        os.environ['FW_HOSTS'] = ''  # environment.sh passes an empty value for the unsplit job
        work.stage('tune', 0, 3)
        assert done == ['QUIPSHARP', 'QTIP', 'QUIPSHARP', 'QTIP']
        for hosts, shards in (('GPTVQ:QUIPSHARP', 3), ('QTIP:AQLM', 2), ('GPTVQ,QUIPSHARP', 1)):
            os.environ['FW_HOSTS'] = hosts
            try:
                work.stage('tune', 0, shards)
            except ValueError:
                continue
            raise AssertionError('bad split accepted')
        os.environ.pop('FW_HOSTS')


if __name__ == '__main__':
    for test in (test_scaled_linear, test_active_forward_marks_seen, test_rules, test_step_and_train, test_fold, test_reports,
                 test_split_hosts):
        test()
        print('ok', test.__name__)
    print('ALL PASSED')

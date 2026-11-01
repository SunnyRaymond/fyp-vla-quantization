"""CPU checks for lowrank_loop.py and lowrank_bench.py with stubbed loop_worker / common: the install (host weights,
factors, both anchor checks and their failures), the branched forward against lowrank_worker's LowRank on the same
factors, the fused and merged forms and their deviation check, the host-only install, the patched CountingLinears
forward, the argument split before loop_worker.main, and a bench of all four modes with its comparison."""
import json
from pathlib import Path
import sys
import tempfile
import types

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_lowrank  # noqa: E402  (its stubs let lowrank_worker import; LowRank is the reference branch)

SHAPES = [(6, 5), (4, 6), (5, 4)]  # a chain 5 -> 6 -> 4 -> 5
NAMES = [f'video_expert.blocks.{i}.ffn.0' for i in range(len(SHAPES))]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf-8')


class CountingLinears:
    """loop_worker.CountingLinears' structure: forwards from _forward(name, module), call counters."""
    def __init__(self, modules):
        from collections import Counter
        self.modules, self.calls, self.seen, self.track = modules, Counter(), set(), True
        for name, module in modules.items():
            module.forward = self._forward(name, module)

    def _forward(self, name, module):
        return lambda x: F.linear(x, module.weight, module.bias)


class Module:
    def __init__(self, w, b):
        self.weight, self.bias = w, b
        self.forward = lambda x: F.linear(x, self.weight, self.bias)

    def __call__(self, x):
        return self.forward(x)


def capture(p, runtime):
    h = runtime.x
    for n in NAMES:
        h = runtime.q.modules[n](h)
    return h.detach().float().cpu(), 2 * h.detach().float().cpu()


def build_runtime(p, arm, out):
    runtime = RUNTIME[0]
    loop_worker.install_weights(p, runtime, arm)
    loop_worker.CountingLinears(runtime.q.modules)
    return runtime


calls, RUNTIME = [], []
common = types.ModuleType('common')
common.__dict__.update(TARGETS=len(SHAPES), read=read, save=save, results=lambda p: Path(p['remote_root']) / 'results',
                       anchor_datum=lambda p, runtime: (None, 0))
loop_worker = types.ModuleType('loop_worker')
loop_worker.__dict__.update(CountingLinears=CountingLinears, capture=capture, install_weights=None,
                            build_runtime=build_runtime, main=lambda: calls.append(list(sys.argv)))
sys.modules['common'], sys.modules['loop_worker'] = common, loop_worker
import lowrank_loop as ll  # noqa: E402
import lowrank_bench as lb  # noqa: E402


def setup(tmp):
    """Masters, host W2 weights, BF16 factors; the BF16 anchor, the host's loop anchor and the arm's fit row computed
    here independently of lowrank_loop."""
    g = torch.Generator().manual_seed(0)
    masters = {n: (torch.randn(o, i, generator=g) / i ** 0.5).to(torch.bfloat16) for n, (o, i) in zip(NAMES, SHAPES)}
    biases = {n: (0.1 * torch.randn(o, generator=g)).to(torch.bfloat16) for n, (o, _) in zip(NAMES, SHAPES)}
    hosts = {n: (w.float() + 0.3 * torch.randn(w.shape, generator=g) / w.shape[1] ** 0.5).to(torch.bfloat16)
             for n, w in masters.items()}
    factors = {n: dict(U=(0.2 * torch.randn(o, 4, generator=g)).to(torch.bfloat16),
                       V=(0.2 * torch.randn(4, i, generator=g)).to(torch.bfloat16)) for n, (o, i) in zip(NAMES, SHAPES)}
    runtime = types.SimpleNamespace(q=types.SimpleNamespace(modules={n: Module(masters[n].clone(), biases[n]) for n in NAMES}),
                                    x=torch.randn(32, 5, generator=g).to(torch.bfloat16))
    p = dict(remote_root=str(tmp / 'remote'), panel=dict(anchor_case=2000))

    def run(weights, branch):
        h = runtime.x
        for n in NAMES:
            y = F.linear(h, weights[n], biases[n])
            if branch:
                y = y + test_lowrank.lw.LowRank.apply(h, factors[n]['U'].float(), factors[n]['V'].float())
            h = y
        return h.float()
    reference = run(masters, False)
    (Path(common.results(p)) / 'prep').mkdir(parents=True)
    torch.save(dict(action=reference), Path(common.results(p)) / 'prep' / 'bf16_anchor.pt')

    def rmse(a):
        return float((a - reference)[:10, :6].pow(2).mean().sqrt())
    host = tmp / 'host'
    for i, n in enumerate(NAMES):
        test_lowrank.tsave(host / 'weights' / f'{i:04d}.pt', dict(module=n, weight=hosts[n]))
    save(host / 'COMPLETE.json', dict(complete=True))
    test_lowrank.tsave(tmp / 'factors.pt', factors)
    save(tmp / 'startup.json', dict(anchor_action_vs_bf16=dict(first10_motor_rmse=rmse(run(hosts, False)))))
    save(tmp / 'fit.json', dict(rows=[dict(case_id=1999, first10motor_rmse=1.0),
                                      dict(case_id=2000, first10motor_rmse=rmse(run(hosts, True)))]))
    ll.CFG.update(lr_host=str(host), lr_factors=str(tmp / 'factors.pt'), lr_host_anchor=str(tmp / 'startup.json'),
                  lr_fit=str(tmp / 'fit.json'), lr_check=str(tmp / 'worker_0' / 'lowrank_check.json'), lr_mode='branch')
    return p, runtime, hosts, factors, biases


def exact(x, n, hosts, factors, biases):
    return F.linear(x, hosts[n], biases[n]) + test_lowrank.lw.LowRank.apply(x, factors[n]['U'].float(), factors[n]['V'].float())


def test_install_and_forward():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p, runtime, hosts, factors, biases = setup(tmp)
        ll.install_weights(p, runtime, 'QTIP_LRJ_W2')
        check = read(tmp / 'worker_0' / 'lowrank_check.json')
        assert check['passed'] and check['host_rel'] == 0.0 and check['arm_rel'] == 0.0, check
        for n in NAMES:
            m = runtime.q.modules[n]
            x = torch.randn(32, m.weight.shape[1]).to(torch.bfloat16)
            want = F.linear(x, hosts[n], biases[n]) + test_lowrank.lw.LowRank.apply(
                x, factors[n]['U'].float(), factors[n]['V'].float())
            assert torch.equal(m.weight, hosts[n]) and torch.equal(m(x), want)
        banks = loop_worker.CountingLinears.__new__(loop_worker.CountingLinears)
        loop_worker.CountingLinears._forward = ll._forward
        try:
            eager = capture(p, runtime)[0]
            banks.__init__(runtime.q.modules)
            assert torch.equal(capture(p, runtime)[0], eager) and banks.seen == set(NAMES)
            assert all(banks.calls[n] == 1 for n in NAMES)
        finally:
            loop_worker.CountingLinears._forward = CountingLinears._forward


def test_checks_fail():
    for key, broken in (('lr_fit', lambda d: d['rows'][1].update(first10motor_rmse=d['rows'][1]['first10motor_rmse'] * 1.01)),
                        ('lr_host_anchor', lambda d: d['anchor_action_vs_bf16'].update(first10_motor_rmse=1.0))):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p, runtime, *_ = setup(tmp)
            d = read(ll.CFG[key])
            broken(d)
            save(ll.CFG[key], d)
            try:
                ll.install_weights(p, runtime, 'QTIP_LRJ_W2')
            except RuntimeError:
                assert not read(tmp / 'worker_0' / 'lowrank_check.json')['passed']
            else:
                raise AssertionError(f'a broken {key} passed the install check')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p, runtime, _, factors, _ = setup(tmp)
        del factors[NAMES[1]]
        test_lowrank.tsave(tmp / 'factors.pt', factors)
        try:
            ll.install_weights(p, runtime, 'QTIP_LRJ_W2')
        except RuntimeError as error:
            assert 'exactly once' in str(error)
        else:
            raise AssertionError('missing factors passed')


def test_fast_modes():
    """fused: BF16 x V^T + addmm into the output; merged: W' = bf16(W + U V) behind a plain Linear. Both stay within
    BF16 rounding of the exact branch and pass the deviation check; with FAST_REL = 0 the check fails."""
    for mode in ('fused', 'merged'):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p, runtime, hosts, factors, biases = setup(tmp)
            ll.CFG['lr_mode'] = mode
            ll.install_weights(p, runtime, 'QTIP_LRJ_W2')
            check = read(tmp / 'worker_0' / 'lowrank_check.json')
            assert check['passed'] and check['mode'] == mode and check['arm_rel'] == 0.0, check
            assert 0 < check['fast_action_rel'] < 0.05 and 0 < check['fast_future_rel'] < 0.05, check
            for n in NAMES:
                m = runtime.q.modules[n]
                x = torch.randn(3, 7, m.weight.shape[1]).to(torch.bfloat16)
                want = exact(x, n, hosts, factors, biases)
                got = m(x)
                assert got.shape == want.shape and got.dtype == torch.bfloat16
                assert float((got.float() - want.float()).abs().max()) <= 2 ** -6 * float(want.float().abs().max())
                if mode == 'merged':
                    merged = (hosts[n].float() + factors[n]['U'].float() @ factors[n]['V'].float()).to(torch.bfloat16)
                    assert torch.equal(m.weight, merged) and torch.equal(got, F.linear(x, merged, biases[n]))
                else:
                    assert torch.equal(m.weight, hosts[n])
            limit = ll.FAST_REL
            ll.FAST_REL = 0.0
            try:
                ll.install_weights(p, runtime, 'QTIP_LRJ_W2')
            except RuntimeError:
                assert not read(tmp / 'worker_0' / 'lowrank_check.json')['passed']
            else:
                raise AssertionError(f'{mode} passed a zero deviation limit')
            finally:
                ll.FAST_REL = limit


def test_host_mode():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p, runtime, hosts, _, biases = setup(tmp)
        ll.CFG['lr_mode'] = 'host'
        ll.install_weights(p, runtime, 'QTIP_LRJ_W2')
        check = read(tmp / 'worker_0' / 'lowrank_check.json')
        assert check['passed'] and check['host_rel'] == 0.0 and 'arm_rel' not in check, check
        x = torch.randn(4, SHAPES[0][1]).to(torch.bfloat16)
        assert torch.equal(runtime.q.modules[NAMES[0]](x), F.linear(x, hosts[NAMES[0]], biases[NAMES[0]]))


def test_bench_and_compare():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p, runtime, *_ = setup(tmp)
        runtime.infer = lambda datum, seed, measure=False: capture(p, runtime)
        RUNTIME[:] = [runtime]
        save(tmp / 'protocol.json', p)
        lr = [f'--lr-host={ll.CFG["lr_host"]}', f'--lr-factors={ll.CFG["lr_factors"]}',
              f'--lr-host-anchor={ll.CFG["lr_host_anchor"]}', f'--lr-fit={ll.CFG["lr_fit"]}']
        try:
            for mode in ll.MODES:
                out = tmp / 'bench' / mode
                out.mkdir(parents=True)
                lb.main(['--protocol', str(tmp / 'protocol.json'), '--arm', 'QTIP_LRJ_W2', '--out', str(out),
                         '--calls', '3', '--warmup', '1', '--lr-mode', mode, *lr, f'--lr-check={out / "lowrank_check.json"}'])
                row = read(out / 'bench.json')
                assert row['mode'] == mode and len(row['calls']) == 3 and row['check']['passed'], row
            lb.main(['--compare', str(tmp / 'bench')])
            summary = read(tmp / 'bench' / 'summary.json')
            assert summary['adopt'] == 'merged' and summary['modes']['host']['vs_host'] == 1.0, summary
            assert all(summary['modes'][m]['passed'] for m in ('fused', 'merged')), summary
            check = read(tmp / 'bench' / 'merged' / 'lowrank_check.json')
            assert abs(summary['modes']['merged']['action_rel'] - check['fast_action_rel']) < 1e-12, (summary, check)
            # a fast mode that failed its install check (26068632) leaves only lowrank_check.json
            for name in ('bench.json', 'anchor.pt'):
                (tmp / 'bench' / 'merged' / name).unlink()
            save(tmp / 'bench' / 'merged' / 'lowrank_check.json', dict(check, passed=False))
            lb.main(['--compare', str(tmp / 'bench')])
            summary = read(tmp / 'bench' / 'summary.json')
            row = summary['modes']['merged']
            assert summary['adopt'] == 'fused' and not row['timed'] and not row['passed'], summary
            assert row['action_rel'] == check['fast_action_rel'], summary
        finally:
            loop_worker.CountingLinears._forward = CountingLinears._forward


def test_main_arguments():
    calls.clear()
    argv = sys.argv
    try:
        sys.argv = ['lowrank_loop.py']
        ll.main(['--protocol', 'p.json', '--arm', 'QTIP_LRJ_W2', '--worker', '3', '--lr-host', 'H', '--lr-factors', 'F',
                 '--lr-host-anchor', 'A', '--lr-fit', 'T', '--lr-check', 'C'])
        assert calls == [['lowrank_loop.py', '--protocol', 'p.json', '--arm', 'QTIP_LRJ_W2', '--worker', '3']]
        assert ll.CFG['lr_host'] == 'H' and ll.CFG['lr_check'] == 'C'
        assert loop_worker.install_weights is ll.install_weights and loop_worker.CountingLinears._forward is ll._forward
    finally:
        sys.argv = argv
        loop_worker.CountingLinears._forward = CountingLinears._forward


if __name__ == '__main__':
    for test in (test_install_and_forward, test_checks_fail, test_fast_modes, test_host_mode, test_bench_and_compare,
                 test_main_arguments):
        test()
        print('ok', test.__name__)
    print('ALL PASSED')

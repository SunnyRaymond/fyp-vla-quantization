"""CPU checks for phase 3 (host.json "hosts"): the three host quantizers load side by side through Work.method, arm routing
reaches the baseline and guided roots, and final_hosts writes the report from synthetic results (stubbed runtime)."""
import json
from pathlib import Path
import random
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {})):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import worker as wk  # noqa: E402

HOST = json.loads((HERE / 'host.json').read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_work(root):
    work = object.__new__(wk.Work)
    work.base = work.results = root / 'guided' / 'results'
    work.baseline, work.host, work.job = root / 'baseline' / 'results', HOST, 'TEST'
    return work


def test_methods_load():
    work = make_work(Path('.'))
    for name in ('qtip', 'gptvq', 'quipsharp'):
        assert callable(work.method(name).quantize_layer), name
    assert work.method('gptvq') is work.method('gptvq')


def test_arm_routing():
    work = make_work(Path('/r'))
    assert work.arm_root('GPTVQ_W2') == work.baseline / 'quantized' / 'GPTVQ_W2'
    assert work.arm_root('QUIPSHARP_G_W2') == work.base / 'quantized' / 'QUIPSHARP_G_W2'
    assert work.arm_root('QTIP_U_W2') == work.baseline / 'quantized' / 'QTIP_W2'


def test_final_hosts_report():
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        save(work.base / 'summary.json', dict(quant=dict(
            QTIP_U_W2=dict(proxy_U_weighted=0.0058, proxy_G_weighted=0.0077, bpw=2.016),
            QTIP_G_W2=dict(proxy_U_weighted=0.0079, proxy_G_weighted=0.0038, bpw=2.016))))
        for arm, c in HOST['hosts']['arms'].items():
            save(work.baseline / 'quantized' / c['baseline_arm'] / 'COMPLETE.json',
                 dict(complete=True, bpw=2.06, proxy_nmse_weighted=0.0096))
            for i in range(614):
                save(work.base / 'quantized' / arm / 'stats' / f'{i:04d}.json', dict(
                    shape=[8, 4], bits_total=66.0, seconds=1.0, config={}, proxy_U=0.01, proxy_G=0.005,
                    baseline_proxy_U=0.009, baseline_proxy_G=0.008))
        row = dict(first10motor_rmse=0.03, future_latent_nmse=0.05, late22motor_rmse=0.04, first10gripper_rmse=0.08,
                   first10motor_rmse_swap=0.02, mixed_loss=0.003)
        for arm in ['QTIP_U_W2', 'QTIP_G_W2'] + HOST['hosts']['eval_arms']:
            save(work.base / 'heldout' / f'{arm}.json', dict(overall=row))

        def episodes(path, rate):
            for i in range(280):
                save(path / 'episodes' / f'{i}.json', dict(success=random.random() < rate))
            (path / 'DONE').write_text('x')
        episodes(work.baseline / 'loops' / 'QTIP_W2', 0.73)
        episodes(work.base / 'loops' / 'QTIP_G_W2', 0.74)
        for arm, c in HOST['hosts']['arms'].items():
            episodes(work.baseline / 'loops' / c['baseline_arm'], 0.71)
            episodes(work.base / 'loops' / arm, 0.74)
        work.final_hosts()
        summary = json.loads((work.base / 'hosts' / 'summary.json').read_text())
        assert set(summary['hosts']) == {'qtip', 'gptvq', 'quipsharp'}
        for r in summary['hosts'].values():
            assert r['loop']['episodes'] == 280 and r['baseline']['heldout'] and r['guided']['heldout']
        for arm in HOST['hosts']['arms']:
            c = json.loads((work.base / 'quantized' / arm / 'COMPLETE.json').read_text())
            assert c['complete'] and abs(c['bpw'] - 66 / 32) < 1e-9 and abs(c['proxy_nmse_weighted'] - 0.01) < 1e-12
        text = (work.base / 'hosts' / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert 'GPTVQ' in text and 'QuIP#' in text and 'McNemar' in text
        print(text)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name, flush=True)
    print('ALL PASS')

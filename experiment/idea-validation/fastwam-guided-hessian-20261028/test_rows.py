"""CPU checks for phase 4 (rows.json, rows_worker.py) with a stubbed runtime and a stand-in quantizer whose rounding
noise is iid across the rows of the weight it is given (the model behind the row scale): the scale is applied before
and divided out after quantization, layers without probe signal keep s = 1, the row-weighted guided proxy drops against
the unscaled arm, and complete_rows / final_rows write the receipts and the report."""
import json
from pathlib import Path
import random
import sys
import tempfile
import types

import torch

HERE = Path(__file__).resolve().parent
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {})):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import rows_worker as rw  # noqa: E402
import worker as wk  # noqa: E402

HOST = json.loads((HERE / 'host.json').read_text(encoding='utf-8'))
SHAPES = [(64, 32), (48, 32), (16, 24)]


def noisy(w, seed, level=0.05):
    """Stand-in quantizer: iid noise with the scale of the given weight's RMS (as after QTIP's two-sided RHT)."""
    g = torch.Generator().manual_seed(seed)
    return w + torch.randn(w.shape, generator=g) * float(w.square().mean().sqrt()) * level


FAKE = types.SimpleNamespace(quantize_layer=lambda w, h, bits, seed=0: dict(
    dequant=noisy(w.cpu(), seed), bits_total=float(bits * w.numel()), config=dict(code='stand-in')))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def make_work(root):
    work = object.__new__(rw.RowWork)
    work.base = work.results = root / 'guided' / 'results'
    work.baseline, work.host, work.job = root / 'baseline' / 'results', HOST, 'TEST'
    work.enough = lambda: None
    work.progress = lambda *args, **kwargs: None
    work.atomic_torch_save = lambda _torch, path, value: torch.save(value, path)
    work.method = lambda name: FAKE
    return work


def build(work):
    """Three layers: two with strongly unequal row weights, one without probe signal (row_weight None)."""
    g = torch.Generator().manual_seed(0)
    prep, seed = work.baseline / 'prep', HOST['host']['seed_base']
    modules = []
    for i, (o, n) in enumerate(SHAPES):
        name = f'video_expert.blocks.{i}.ffn.0'
        modules.append(dict(index=i, module=name, shape=[o, n]))
        w = torch.randn(o, n, generator=g)
        x = torch.randn(256, n, generator=g)
        h = x.t() @ x / 256
        d = None if i == 2 else torch.rand(o, generator=g).pow(4) * 100 + 1e-3
        tsave(prep / 'masters' / f'{i:04d}.pt', dict(module=name, weight=w.to(torch.bfloat16)))
        tsave(prep / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=h))
        tsave(work.results / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=h, row_weight=d))
        base = noisy(w.to(torch.bfloat16).float(), seed + i)
        tsave(work.baseline / 'quantized' / 'QTIP_W2' / 'weights' / f'{i:04d}.pt', dict(module=name, weight=base.to(torch.bfloat16)))
        save(work.baseline / 'quantized' / 'QTIP_W2' / 'stats' / f'{i:04d}.json', dict(seconds=1.0 + i))
        tsave(work.base / 'quantized' / wk.GUIDED / 'weights' / f'{i:04d}.pt', dict(module=name, weight=base.to(torch.bfloat16)))
    save(prep / 'modules.json', modules)


def test_quantize_rows():
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        build(work)
        work.quantize_rows()
        out = work.arm_root(rw.ARM)
        assert out == work.base / 'quantized' / 'QTIP_GR_W2'
        for i, (o, n) in enumerate(SHAPES):
            s = json.loads((out / 'stats' / f'{i:04d}.json').read_text())
            w = torch.load(out / 'weights' / f'{i:04d}.pt', weights_only=True)['weight']
            assert tuple(w.shape) == (o, n) and s['bpw'] == 2.0
            if i == 2:
                assert not s['row_scale'] and s['row_scale_range'] == [1.0, 1.0] and s['row_ratio'] == 1.0
                continue
            assert s['row_scale'] and s['row_ratio'] < 0.8, s['row_ratio']
            assert 1 / 16 - 1e-6 <= s['row_scale_range'][0] < 1 < s['row_scale_range'][1] <= 16 + 1e-6
            p = s['proxies']
            # same seed, so the unscaled phase-1 stand-in and ours see the same noise draw
            assert p['QTIP_GR_W2']['G_rows'] < p['QTIP_G_W2']['G_rows'], p
            assert set(p) == {'QTIP_U_W2', 'QTIP_G_W2', 'QTIP_GR_W2'} and set(p['QTIP_GR_W2']) == {'U', 'G', 'G_rows'}
        work.quantize_rows()  # resumable: nothing left to do


def test_complete_and_final():
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        work = make_work(Path(tmp))
        build(work)
        work.quantize_rows()
        c = work.complete_rows()
        receipt = json.loads((work.arm_root(rw.ARM) / 'COMPLETE.json').read_text())
        assert receipt['complete'] and receipt['method'] == 'qtip' and receipt['bits'] == 2
        assert receipt['layers_scaled'] == 2 and receipt['layers'] == 3 and c['arm']['row_ratio_weighted'] < 1
        overall = dict(first10motor_rmse=0.03, future_latent_nmse=0.05, late22motor_rmse=0.04, first10gripper_rmse=0.07,
                       first10motor_rmse_swap=0.02, mixed_loss=0.003)
        for arm in rw.ARMS:
            save(work.results / 'heldout' / f'{arm}.json', dict(overall=overall))
        dims = {'camera_viewpoints': 18, 'light_conditions': 36}
        for root, rate in ((work.baseline / 'loops' / 'QTIP_W2', 0.73), (work.base / 'loops' / wk.GUIDED, 0.74),
                           (work.base / 'loops' / rw.ARM, 0.76)):
            for k in range(280):
                save(root / 'episodes' / f'{k}.json', dict(success=random.random() < rate))
            save(root / 'summary.json', dict(dimensions=dims))
            (root / 'DONE').write_text('x')
        work.final_rows()
        summary = json.loads((work.base / 'rows' / 'summary.json').read_text())
        mine = summary['loops'][rw.ARM]
        assert mine['episodes'] == 280 and set(mine) >= {'vs_QTIP_U_W2', f'vs_{wk.GUIDED}'}
        assert 0 <= mine['vs_QTIP_U_W2']['p'] <= 1
        report = (work.base / 'rows' / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert 'H̃ + 行 scale' in report and 'camera_viewpoints' in report and 'McNemar' in report


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    print('ALL PASS')

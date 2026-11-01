"""CPU checks for RAVQ calibrated (ravqc.json, ravqc_worker.py) with a stubbed runtime and the real QTIP port on small
layers: trellis -> pair codes -> decode reproduces QTIP's dequantized weight bit for bit (K = 1..4), recovery of the codes
from the stored weight, the fast Hadamard / incoherence against the port's, the decode adjoint, QTIPGrad's gradients
against autograd on the dense parametrization, the reader metric (rho = 1 gives the NMSE, the low-rank + diagonal form
against a dense M), the relative step's first-order prediction, v3's fold from the FP32 decode (a sub-spacing change is
realized in the mean), the family filter, the prequential C80 check with the final report, the transfer statistics on
synthetic gradients, and a toy pipeline codes -> attach -> per-context gradients (action / future parts summing to the
objective's, files, transfer report) -> step -> deploy -> install. 'cuda' tensors map to the CPU."""
import importlib.util
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
                    ('frozen_reader', dict(_bf16_forward_context=None, _checkpoint_context_fn=None, BF16Reader=None))):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import ravq_qtip as rq  # noqa: E402
import ravqc_worker as rc  # noqa: E402
import worker as wk  # noqa: E402

_to = torch.Tensor.to


def _cpu_to(self, *args, **kwargs):
    args = tuple('cpu' if a == 'cuda' else a for a in args)
    if kwargs.get('device') == 'cuda':
        kwargs['device'] = 'cpu'
    return _to(self, *args, **kwargs)


torch.Tensor.to = _cpu_to
spec = importlib.util.spec_from_file_location('fwrc_qtip_port', HERE / 'methods' / 'qtip' / 'quant.py')
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
PERM, INV = Q.ldlq._PERMUTE.clone(), Q.ldlq._INV_PERMUTE.clone()
TABLE = rc.lut_values(Q.codebook(2, 'cpu').tlut.to(torch.bfloat16))


def tsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def jsave(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def layer_data(m, n, seed):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(m, n, generator=g) * 0.02).to(torch.bfloat16).float()  # a BF16 master, as the model's
    x = torch.randn(4 * n, n, generator=g) @ torch.randn(n, n, generator=g) * 0.1 + torch.randn(4 * n, n, generator=g)
    return w, x.t() @ x / x.shape[0]


def rebuild_dequant(pairs, Wscale, SU, SV, m, n, dtype=torch.float32):
    D = rc.decode(pairs, TABLE, m, n, INV)
    hat = D.to(dtype) * torch.tensor(Wscale, dtype=dtype)
    return Q.inverse_incoherence(hat, SU.to(torch.bfloat16).to(dtype), SV.to(torch.bfloat16).to(dtype))[:m, :n].float()


def test_codes_roundtrip():
    w, h = layer_data(64, 128, 0)
    for k in (1, 2, 3, 4):
        out = rq.quantize_layer(Q, w, h, k, seed=11)
        pairs = rc.trellis_codes(Q, out['codes'])
        assert pairs.shape == (64 * 128 // 256, 128) and pairs.dtype == torch.int16
        dq = rebuild_dequant(pairs, out['codes']['Wscale'], out['codes']['SU'].float(), out['codes']['SV'].float(), 64, 128)
        assert torch.equal(dq, out['dequant']), k
        assert torch.equal(rc.lut_values(out['codes']['tlut']), TABLE)
        stored = out['dequant'].to(torch.bfloat16)
        SU, SV = Q.sign_vectors(128, 64, 11)
        assert torch.equal(SU, out['codes']['SU'].float()) and torch.equal(SV, out['codes']['SV'].float())
        D = rc.coherent(Q, stored.double(), SU.double(), SV.double()) / out['codes']['Wscale']
        rec, margin = rc.nearest_codes(rc.to_pairs(D, PERM).float(), TABLE)
        same = float((rec == pairs).float().mean())
        bitwise = torch.equal(rebuild_dequant(rec, out['codes']['Wscale'], SU, SV, 64, 128).to(torch.bfloat16), stored)
        print(f'K={k}: nearest-candidate recovery from BF16: pairs equal {same:.6f}, rebuild bitwise {bitwise}, '
              f'min margin {float(margin.min()):.4g} (diagnostic: the codes stage requantizes instead)')
        assert same > 0.99


def test_hadamard():
    g = torch.Generator().manual_seed(1)
    for n in (64, 128, 3072):
        X = torch.randn(5, n, generator=g)
        for t in (False, True):
            assert torch.allclose(rc.had(Q, X, t), Q.matmul_hadU(X, t), atol=1e-5, rtol=1e-5), (n, t)
    R = torch.randn(64, 128, generator=g)
    SU, SV = Q.sign_vectors(128, 64, 3)
    assert torch.allclose(rc.incoherent(Q, R, SU, SV), Q.inverse_incoherence(R, SU, SV), atol=1e-5)
    assert torch.allclose(rc.coherent(Q, rc.incoherent(Q, R, SU, SV), SU, SV), R, atol=1e-5)


def test_adjoint():
    g = torch.Generator().manual_seed(2)
    m, n = 32, 64
    pairs = (torch.randint(0, 512, (m * n // 256, 128), generator=g)
             | (torch.randint(0, 2, (m * n // 256, 128), generator=g) << 9)).to(torch.int16)
    t = torch.randn(512, 2, generator=g, dtype=torch.float64)
    dD = torch.randn(m, n, generator=g, dtype=torch.float64)
    lhs = float((rc.decode(pairs, t, m, n, INV) * dD).sum())
    rhs = float((t * rc.decode_adjoint(dD, pairs, PERM, torch.zeros(512, 2, dtype=torch.float64))).sum())
    assert abs(lhs - rhs) < 1e-9 * max(1, abs(lhs)), (lhs, rhs)
    s = rc.states_to_codes(torch.arange(1 << 16))
    lut = Q.bitshift.quantlut_sym(TABLE.to(torch.float16), 16, 9).float()
    val = torch.stack((TABLE[(s.long() & 511), 0] * (1 - 2 * ((s.long() >> 9) & 1)), TABLE[(s.long() & 511), 1]), -1)
    assert torch.equal(val, lut)


def toy_layer(m=64, n=128, seed=5, K=2, dtype=torch.float32):
    w, h = layer_data(m, n, seed)
    out = rq.quantize_layer(Q, w, h, K, seed=seed)
    mod = torch.nn.Linear(n, m, bias=True).to(dtype)
    with torch.no_grad():
        mod.weight.copy_(out['dequant'].to(dtype))
        mod.bias.copy_(torch.randn(m) * 0.1)
    for p in mod.parameters():
        p.requires_grad_(False)
    saved = dict(pairs=rc.trellis_codes(Q, out['codes']), Wscale=out['codes']['Wscale'],
                 SU=out['codes']['SU'], SV=out['codes']['SV'])
    return mod, saved


def test_qtipgrad():
    """QTIPGrad on a float32 module (no BF16 rounding) against autograd through the dense parametrization."""
    mod, saved = toy_layer()
    L = rc.QLayer(0, 'toy', mod, 'video_body@2', saved, Q, PERM, INV)
    g = torch.Generator().manual_seed(9)
    table = torch.zeros(512, 2, requires_grad=True)
    with torch.no_grad():
        L.s.copy_(0.01 * torch.randn(64, generator=g))
        L.t.copy_(0.01 * torch.randn(128, generator=g))
        L.db.copy_(0.01 * torch.randn(64, generator=g))
        table.copy_(0.05 * torch.randn(512, 2, generator=g))
    W0 = L.W0.clone()
    L.rebuild(table.detach(), TABLE)
    assert torch.equal(L.W0, W0)  # rebuild never writes into the stored start
    x = torch.randn(3, 7, 128, generator=g)
    R = torch.randn(3, 7, 64, generator=g)
    y = rc.QTIPGrad.apply(x, L.s, L.t, L.db, table, L)
    (y * R).sum().backward()
    got = dict(s=L.s.grad.clone(), t=L.t.grad.clone(), db=L.db.grad.clone(), table=table.grad.clone())
    s, t, db, tb = (v.detach().double().clone().requires_grad_(True) for v in (L.s, L.t, L.db, table))
    D = rc.decode(saved['pairs'], tb, 64, 128, INV)
    dW = rc.incoherent(Q, D * saved['Wscale'], saved['SU'].double(), saved['SV'].double())
    W = s.exp()[:, None] * (L.W0.double() + dW) * t.exp()[None, :]
    ref_y = F.linear(x.double(), W, L.b0.double() + db)
    assert torch.allclose(y.double(), ref_y, atol=1e-5, rtol=1e-4)
    (ref_y * R.double()).sum().backward()
    for k, v in (('s', s), ('t', t), ('db', db), ('table', tb)):
        err = float((got[k].double() - v.grad).norm() / v.grad.norm())
        assert err < 1e-4, (k, err)
    assert float(L.xsq) > 0 and L.tokens == 21


def test_rebuild_identity():
    mod, saved = toy_layer(dtype=torch.bfloat16)
    W0 = mod.weight.detach().clone()
    b0 = mod.bias.detach().clone()
    L = rc.QLayer(0, 'toy', mod, 'video_body@2', saved, Q, PERM, INV)
    L.rebuild(torch.zeros(512, 2), TABLE)
    assert torch.equal(mod.weight, W0) and torch.equal(mod.bias, b0)


def test_fold():
    """v3's fold (fwrc2r2 26074313): from the FP32 decode W_hi of the codes, whose BF16 rounding is W0, a change far
    below half a BF16 spacing is realized in the mean (a share of the elements moves one spacing); folding onto W0 itself
    rounded it away. Table offset and row scale, on a BF16 layer; the projection <W' - W0, dW> / <dW, dW> is the share
    of the intended change that reaches the weight."""
    mod, saved = toy_layer(m=128, n=256, seed=7, dtype=torch.bfloat16)
    W0 = mod.weight.detach().clone()
    L = rc.QLayer(0, 'toy', mod, 'video_body@2', saved, Q, PERM, INV)
    hi = L.decoded(TABLE)
    off = float((hi.to(torch.bfloat16) != W0).float().mean())
    assert off < 1e-3, off  # the fast-form decode rounds onto W0 (the rest start at W0)
    g = torch.Generator().manual_seed(8)
    d = torch.randn(512, 2, generator=g)
    dW = L.decoded(d)
    c = 2.0 ** -12 * float(W0.float().norm() / dW.norm())  # |dW| ~ 0.024% of |W|, about a tenth of half a spacing
    d, dW = d * c, dW * c

    def share(change):
        return float((change * dW_now).sum() / dW_now.square().sum())
    dW_now = dW
    L.rebuild(d, TABLE)
    new = share(mod.weight.float() - W0.float())
    old = share((W0.float() + dW).to(torch.bfloat16).float() - W0.float())
    print(f'table offset at 0.024% of |W|: realized share {new:.3f} from W_hi, {old:.3f} folded onto W0', flush=True)
    assert 0.85 < new < 1.15 and old < 0.5, (new, old)
    with torch.no_grad():
        L.s.fill_(2.0 ** -10)
    dW_now = W0.float() * (math.exp(2.0 ** -10) - 1)
    L.rebuild(torch.zeros(512, 2), TABLE)
    new = share(mod.weight.float() - W0.float())
    old = share((W0.float() * math.exp(2.0 ** -10)).to(torch.bfloat16).float() - W0.float())
    print(f'row scales +0.1%: realized share {new:.3f} from W_hi, {old:.3f} folded onto W0', flush=True)
    assert 0.85 < new < 1.15 and old == 0.0, (new, old)  # 0.1% is under every element's half spacing (0.2-0.39%)
    with torch.no_grad():
        L.s.zero_()
    L.rebuild(torch.zeros(512, 2), TABLE)
    assert torch.equal(mod.weight, W0)


def test_metric():
    g = torch.Generator().manual_seed(4)
    D, P, N = 60, 4, 10
    basis = torch.randn(5, D, generator=g)
    vectors = [(torch.randn(P, 5, generator=g) @ basis + 0.1 * torch.randn(P, D, generator=g)) * (1 + c)
               for c in range(N)]
    saved, stats, diag = rc.build_metric(vectors, list(range(N)), 8, 0.1)
    M = rc.Metric(saved, 'cpu')
    dense = M.c * (saved['E'].t() @ torch.diag(saved['lam']) @ saved['E'] + torch.diag(saved['d_res'])) + M.rho * torch.eye(D)
    assert abs(float(torch.trace(dense)) - D) < 1e-3 * D
    v = torch.randn(D, generator=g)
    assert abs(float(M.quad(v)) - float(v @ dense @ v)) < 1e-4 * float(v @ dense @ v)
    rows = [x / (P * x.square().sum(1).mean()).sqrt() for x in vectors]
    mbar = sum(r.t() @ r for r in rows) / N
    assert torch.allclose(diag.double(), mbar.diagonal().double(), atol=1e-6)
    assert stats['top_share'] > 0.8, stats
    one = rc.Metric(dict(saved, rho=1.0), 'cpu')
    fut = torch.randn(1, 3, 3, 2, 2, generator=g)
    tgt = torch.randn(1, 3, 3, 2, 2, generator=g)
    one.E, one.d = one.E[:, :24], one.d[:24]
    nm = float(((fut[:, :, 1:] - tgt[:, :, 1:]).square().sum()) / tgt[:, :, 1:].square().sum())
    assert abs(float(one.nmse(fut, tgt)) - nm) < 1e-6
    stab = rc.split_stability(vectors * 2, list(range(2 * N)), 4)
    assert -1 <= stab['diag_corr'] <= 1 and 0 <= stab['subspace_overlap'] <= 1.0001


def toy_work(root, layers):
    work = object.__new__(rc.RavqcWork)
    work.base = work.results = root / 'guided' / 'results'
    work.baseline = root / 'baseline' / 'results'
    work.job, work.job_name = 'TEST', 'fwrc1_0'
    work.enough = lambda: None
    work.progress = lambda *a, **k: None
    work.atomic_torch_save = lambda _torch, path, value: torch.save(value, path)
    work.method = lambda name: Q
    work.source = types.ModuleType('calibration_source')
    work.bank = types.SimpleNamespace(names=[n for n, _ in layers], modules=dict(layers), seen=set())
    return work


def test_pipeline():
    """codes (trellis for K = 1, recovery for K = 2 / 3) -> attach -> step -> deploy -> install, with every family (the
    v2 parameter set, kept for the ablations) and then the configured families (v3: the tables alone)."""
    configured = list(rc.RC['training']['families'])
    rc.RC['training']['families'] = ['row', 'col', 'bias', 'lut']
    try:
        pipeline(configured)
    finally:
        rc.RC['training']['families'] = configured


def pipeline(configured):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        names = ['video_expert.text_embedding.0', 'video_expert.blocks.0.self_attn.q', 'video_expert.blocks.0.ffn.0',
                 'action_expert.blocks.0.self_attn.q', 'video_expert.head.head']
        roles = ['text', 'video_body', 'video_body', 'action_body', 'interface']
        mods, rows_w2 = [], []
        work = toy_work(root, [])
        for i, (name, role) in enumerate(zip(names, roles)):
            m, n = (32, 64) if i % 2 else (64, 128)
            w, h = layer_data(m, n, 20 + i)
            k = {0: 2, 1: 2, 2: 3, 3: 1, 4: None}[i]
            jsave(work.baseline / 'quantized' / 'QTIP_W2' / 'stats' / f'{i:04d}.json', dict(module=name, shape=[m, n]))
            if k in (2, 3):
                out = Q.quantize_layer(w, h, k, seed=20261028 + i)
                src = work.baseline / 'quantized' / f'QTIP_W{k}' / 'weights' / f'{i:04d}.pt'
                tsave(src, dict(module=name, weight=out['dequant'].to(torch.bfloat16)))
                jsave(work.baseline / 'quantized' / f'QTIP_W{k}' / 'stats' / f'{i:04d}.json',
                      dict(module=name, shape=[m, n], config=out['config']))
            elif k == 1:
                out = rq.quantize_layer(Q, w, h, 1, seed=20261028 + i)
                src = work.rv('qtip_K1', 'weights', f'{i:04d}.pt')
                tsave(src, dict(module=name, weight=out['dequant'].to(torch.bfloat16)))
                tsave(work.rv('qtip_K1', 'codes', f'{i:04d}.pt'), dict(module=name, **out['codes']))
                jsave(work.rv('qtip_K1', 'stats', f'{i:04d}.json'), dict(module=name, config=out['config']))
            else:
                src = work.baseline / 'prep' / 'masters' / f'{i:04d}.pt'
                tsave(src, dict(module=name, weight=w.to(torch.bfloat16)))
            tsave(work.baseline / 'prep' / 'masters' / f'{i:04d}.pt', dict(module=name, weight=w.to(torch.bfloat16)))
            tsave(work.baseline / 'prep' / 'hessians' / f'{i:04d}.pt', dict(module=name, hessian=h))
            rows_w2.append(dict(index=i, module=name, role=role, choice=str(k) if k else 'bf16', source=str(src)))
            mod = torch.nn.Linear(n, m, bias=i != 1).to(torch.bfloat16)
            for p in mod.parameters():
                p.requires_grad_(False)
            mods.append((name, mod))
        work.bank = types.SimpleNamespace(names=names, modules=dict(mods), seen=set())
        for arm in ('RAVQU_W2', 'RAVQU_W3'):
            jsave(work.base / 'quantized' / arm / 'sources.json', rows_w2)
            for r in rows_w2:
                tsave(work.base / 'quantized' / arm / 'weights' / f"{r['index']:04d}.pt",
                      torch.load(r['source'], weights_only=True))
            jsave(work.base / 'quantized' / arm / 'COMPLETE.json', dict(complete=True, bits_total=1e6, target_weights=5e5,
                                                                        proxy_nmse_weighted=0.01))
        work.codes(0, 1)
        summary = work.codes_summary()
        assert summary['items'] == 4 and summary['requant_not_bitwise'] == 0, summary
        rep = wk.read(work.rc('codes', 'reports', 'fwrc1_0.json'))
        assert {r['index']: r['method'] for r in rep}[3] == 'trellis'
        assert all(r['method'] == 'requantized' and r['requant_bitwise'] for r in rep if r['index'] != 3)
        print('codes methods', {r['index']: r['method'] for r in rep})
        wk.Work.install(work, 'RAVQU_W2')
        work.capture_biases()
        work.install('RAVQU_W2')
        work.attach('RAVQC_W2')
        assert len(work.layers) == 4 and sorted(work.tables) == ['action_body@1', 'text@2', 'video_body@2', 'video_body@3']
        for L in work.layers:
            assert torch.equal(L.mod.weight, L.W0)
        gradtx_checks(work)
        g = torch.Generator().manual_seed(3)
        grads = dict(layers={L.name: dict(s=torch.randn(L.m, generator=g), t=torch.randn(L.n, generator=g),
                                          db=torch.randn(L.m, generator=g) if L.has_bias else torch.zeros(L.m))
                             for L in work.layers},
                     tables={k: torch.randn(512, 2, generator=g) for k in work.tables})
        for L in work.layers:
            L.xsq += 10.0
            L.tokens = 5
        start = work.snapshot()
        pred, parts = work.step(grads, 0.01)
        after = work.snapshot()
        first = 0.0
        for L in work.layers:
            for k in ('s', 't', 'db'):
                first += float((grads['layers'][L.name][k] * (after['layers'][L.name][k] - start['layers'][L.name][k])).sum())
        for k in work.tables:
            first += float((grads['tables'][k] * (after['tables'][k] - start['tables'][k])).sum())
        assert abs(first + pred) < 1e-4 * pred, (first, pred)
        # tune.json's rule: in relative units every element moves -eta g_u / max|g_u| (no per-tensor normalization, the
        # sign-like step that raised the loss in 26073904); the largest element moves exactly eta
        top, moved = max(float(r[4].abs().max()) for r in work.units(grads)), 0.0
        assert work.last_max['max_abs_g_u'] == top
        for fam, name, _, unit, gu in work.units(grads):
            if fam == 'lut':
                d = after['tables'][name] - start['tables'][name]
            else:
                key = {'row': 's', 'col': 't', 'bias': 'db'}[fam]
                d = after['layers'][name][key] - start['layers'][name][key]
            du = d / unit
            assert torch.allclose(du, -0.01 * gu / top, rtol=1e-5, atol=1e-9), (fam, name)
            moved = max(moved, float(du.abs().max()))
        assert abs(moved - 0.01) < 1e-7, moved
        assert abs(sum(parts.values()) - 0.01 * sum(float(r[4].square().sum()) for r in work.units(grads)) / top) < 1e-6 * pred
        changed = [L for L in work.layers if not torch.equal(L.mod.weight, L.W0)]
        assert len(changed) == 4
        work.lam_M = 1.0
        work.deploy('RAVQC_W2', dict(eta=0.01, updates=1))
        done = wk.read(work.arm_root('RAVQC_W2') / 'COMPLETE.json')
        assert done['extra_bits'] == 4 * 16 * 1024 and done['bits_total'] == 1e6 + done['extra_bits']
        want = {L.name: (L.mod.weight.clone(), L.mod.bias.clone() if L.has_bias else None) for L in work.layers}
        work.install('RAVQU_W2')
        assert all(torch.equal(work.bank.modules[n].weight, L.W0) for L in work.layers for n in [L.name])
        assert all(torch.equal(work.bank.modules[L.name].bias, L.b0) for L in work.layers if L.has_bias)
        work.install('RAVQC_W2')
        for n, (w, b) in want.items():
            assert torch.equal(work.bank.modules[n].weight, w)
            if b is not None:
                assert torch.equal(work.bank.modules[n].bias, b)
        work.restore(start)
        assert all(torch.equal(L.mod.weight, L.W0) for L in work.layers)
        # the configured families only (v3: the shared tables): no gradient or step on the scales and biases
        assert configured == ['lut'], configured
        rc.RC['training']['families'] = configured
        work.set_families()
        assert not any(L.s.requires_grad or L.t.requires_grad or L.db.requires_grad for L in work.layers)
        assert all(v.requires_grad for v in work.tables.values())
        assert {r[0] for r in work.units(grads)} == {'lut'}
        pred, parts = work.step(grads, 0.01)
        after = work.snapshot()
        assert parts['row'] == parts['col'] == parts['bias'] == 0.0 and parts['lut'] == pred > 0
        for L in work.layers:
            for k in ('s', 't', 'db'):
                assert torch.equal(after['layers'][L.name][k], start['layers'][L.name][k]), (L.name, k)
        top = max(float(r[4].abs().max()) for r in work.units(grads))
        moved = max(float(((after['tables'][k] - start['tables'][k]) / work.lut_rms).abs().max()) for k in work.tables)
        assert work.last_max['family'] == 'lut' and work.last_max['max_abs_g_u'] == top and abs(moved - 0.01) < 1e-7
        assert all(not torch.equal(L.mod.weight, L.W0) for L in work.layers)  # every group's table moved
        work.restore(start)
        assert all(torch.equal(L.mod.weight, L.W0) for L in work.layers)
        with work.original_biases():
            pass
        assert all(torch.equal(L.mod.bias, L.b0) for L in work.layers if L.has_bias)
        # an arm finished under an older update rule: its rule-dependent files keep the old job's suffix
        jsave(work.rc('RAVQC_W2') / 'arm.json', dict(arm='RAVQC_W2', tuned=False, pbs_jobid='26073904.pbs101'))
        jsave(work.rc('RAVQC_W2') / 'step.json', {})
        jsave(work.rc('RAVQC_W2') / 'fit_before.json', {})
        jsave(work.rc('endpoints', 'RAVQC_W2', '2000_0.json'), {})
        work.archive('RAVQC_W2')
        assert not (work.rc('RAVQC_W2') / 'arm.json').exists() and (work.rc('RAVQC_W2') / 'fit_before.json').exists()
        assert (work.rc('RAVQC_W2') / 'arm.r0_26073904.json').exists() and (work.rc('RAVQC_W2') / 'step.r0_26073904.json').exists()
        assert (work.rc('endpoints', 'RAVQC_W2.r0_26073904') / '2000_0.json').exists()
        assert (work.arm_root('RAVQC_W2').with_name('RAVQC_W2.r0_26073904') / 'COMPLETE.json').exists()
        assert not work.arm_root('RAVQC_W2').exists()


def stub_runtime(work):
    """A toy query through the attached bank (action and future read from every layer's output), cached teachers, and
    a frozen 'reader' K = |z / 2|^2: enough for staged / replay / objective / case_grads on the CPU."""
    from contextlib import contextmanager
    sys.modules.setdefault('reader_loss', types.ModuleType('reader_loss'))
    sys.modules['reader_loss'].reader_terms = lambda _torch, read, action, stats, cfg: dict(reader_kl=read.square().sum())

    def query(case, seed, grad=False):
        g = torch.Generator().manual_seed(case)
        with torch.set_grad_enabled(grad):
            y = torch.cat([work.bank.modules[n](torch.randn(4, work.bank.modules[n].in_features, generator=g)
                                                 .to(torch.bfloat16)).float().reshape(-1) for n in work.bank.names])
            return y[0:896:4].reshape(32, 7), y[1::41][:24].reshape(1, 2, 3, 2, 2), None

    def teacher(case, seed):
        g = torch.Generator().manual_seed(case + 1000)
        return dict(action=torch.randn(32, 7, generator=g), future=torch.randn(1, 2, 3, 2, 2, generator=g))

    @contextmanager
    def bf16_reader():
        yield lambda runtime, value, seed, leaf, checkpointed: (0.5 * leaf, None)

    work.query, work.teacher, work.bf16_reader = query, teacher, bf16_reader
    work.datum = lambda case: (None, 'toy')
    work.runtime = types.SimpleNamespace(description=None)
    work.rows = {c: dict(sampler_seeds=[0]) for c in range(2000, 2080)}
    work.p = dict(reader_source=dict(action_stats={}, config={}))
    work.count = lambda name: None
    work.counters = dict(parameter_reverse_passes=0)
    work.M = rc.Metric(dict(E=torch.zeros(1, 16), lam=torch.zeros(1), d_res=torch.zeros(16), rho=1.0, D=16, tau=1.0), 'cpu')
    work.lam_M, work.loss_checked = 0.7, True


def gradtx_checks(work):
    """case_grads: per context the action-part and future-part replays sum to its share of objective()'s gradient;
    gradtx_cases writes one file per context (and skips them on a rerun); gradtx_final reproduces the batch-1 step-check
    norm and writes its report."""
    stub_runtime(work)
    batch = [2000, 2001, 2002]
    work.zero()
    work.objective(batch, grad=True)
    joint = work.table_grads()
    assert all(float(v.abs().max()) > 0 for v in joint.values()), {g: float(v.abs().max()) for g, v in joint.items()}
    rows = list(work.case_grads(batch))
    assert [r['case'] for r, _, _ in rows] == batch and all(w == 0 for _, _, w in rows)
    # the layer outputs are BF16, so each replay's cotangent is rounded at every layer output (relative 2^-9): the two
    # parts are rounded apart and the joint replay at 1/n scale, so they agree to ~1e-3 of the norm, not bitwise
    rel = {}
    for g, v in joint.items():
        got = sum(p['action'][g] + p['future'][g] for _, p, _ in rows) / len(batch)
        rel[g] = float((got - v).norm() / v.norm())
    print('split vs joint table gradient, relative norm difference:', {g: f'{r:.1e}' for g, r in rel.items()}, flush=True)
    assert max(rel.values()) < 5e-3, rel
    for part in ('action', 'future'):
        assert sum(float(p[part][g].abs().sum()) for _, p, _ in rows for g in p[part]) > 0, part
    assert all(p.grad is None for p in work.params()), 'case_grads left gradients behind'
    cases = list(range(2000, 2012))
    work.gradtx_cases('RAVQC_W2', cases)
    files = sorted(work.gradtx_root('RAVQC_W2').glob('20*.pt'))
    assert len(files) == len(cases)
    stamp = [f.stat().st_mtime_ns for f in files]
    work.gradtx_cases('RAVQC_W2', cases, meta=False)
    assert [f.stat().st_mtime_ns for f in files] == stamp, 'finished contexts were recomputed'
    batches = [cases[0:4], cases[4:8], cases[8:12]]
    jsave(work.rc('RAVQC_W2', 'training.json'), dict(history=[dict(cases=b) for b in batches]))
    work.zero()
    work.objective(batches[0], grad=True)
    jsave(work.rc('RAVQC_W2', 'step.json'), dict(pbs_jobid='STEP', table_grad_norm={
        g: float(v.norm()) for g, v in work.table_grads().items()}))
    work.zero()
    s = work.gradtx_final('RAVQC_W2', cases)
    assert all(abs(c['ratio'] - 1) < 5e-3 for c in s['batch1_check'].values()), s['batch1_check']
    assert set(s['parts']) == {'action', 'future', 'total', 'total_normalized', 'total_trimmed'}
    assert s['parts']['total']['n'] == 12 and set(s['per_group']) == set(work.tables)
    text = (work.gradtx_root('RAVQC_W2') / 'RESULTS.zh.md').read_text(encoding='utf-8')
    assert '通用份额' in text and 'STEP' in text, text
    whole = s['parts']['total']['mean_energy']
    assert abs(sum(v['mean_energy'] for v in s['per_group'].values()) / whole - 1) < 1e-9  # groups are disjoint blocks
    for g, pp in s['per_group_parts'].items():
        assert set(pp) == {'action', 'future'} and pp['action']['n'] == 12, g
    # gradtx_resummary on a summary written before per_group_parts existed: reproduces the rest, adds the split
    import gradtx_resummary as gr
    path = work.gradtx_root('RAVQC_W2') / 'summary.json'
    old = json.loads(path.read_text(encoding='utf-8'))
    old.pop('per_group_parts')
    jsave(path, old)
    r = gr.resummarize(work.rc(), 'RAVQC_W2', 'RESUM', cases)
    assert r['pbs_jobid'] == old['pbs_jobid'] and r['resummary_pbs_jobid'] == 'RESUM' and 'per_group_parts' in r
    text = (work.gradtx_root('RAVQC_W2') / 'RESULTS.zh.md').read_text(encoding='utf-8')
    assert '平均梯度能量占比' in text and '| — | — |' not in text, text
    g0 = next(iter(old['per_group']))
    old['per_group'][g0]['generic'] += 1e-3
    jsave(path, old)
    try:
        gr.resummarize(work.rc(), 'RAVQC_W2', 'RESUM', cases)
        raise RuntimeError('resummary accepted a summary it does not reproduce')
    except AssertionError:
        pass


def test_transfer_stats():
    """g_c = mu + noise: the generic share is 1 / (1 + s^2), split-half T is 1 / (1 + s^2 / 40) (halves of 40), batch T
    1 / (1 + s^2 / 10); pure noise gives ~0; one context carrying the mean shows in top5."""
    g = torch.Generator().manual_seed(9)
    n, D = 80, 4000
    mu = torch.randn(D, generator=g, dtype=torch.float64)
    ids = list(range(2000, 2080))
    batches = [ids[10 * b:10 * b + 10] for b in range(8)]
    for s2 in (0.0, 9.0, 100.0):
        X = mu + s2 ** 0.5 * torch.randn(n, D, generator=g, dtype=torch.float64)
        s = rc.transfer_stats(X, ids, batches, splits=200)
        assert abs(s['generic'] - 1 / (1 + s2)) < 0.01 + 0.05 / (1 + s2), (s2, s['generic'])
        assert abs(s['split_T']['mean'] - 1 / (1 + s2 / 40)) < 0.04, (s2, s['split_T'])
        assert abs(s['batch_T']['mean'] - 1 / (1 + s2 / 10)) < 0.06, (s2, s['batch_T'])
    X = torch.randn(n, D, generator=g, dtype=torch.float64)
    s = rc.transfer_stats(X, ids, batches, splits=200)
    assert abs(s['generic']) < 0.01 and abs(s['split_T']['mean']) < 0.05 and abs(s['batch_T']['mean']) < 0.1, s
    X[7] += 60 * mu
    s = rc.transfer_stats(X, ids, batches, splits=200)
    assert s['top5'][0]['case'] == 2007 and s['top5'][0]['share'] > 0.9, s['top5']


def test_gradtx_arm():
    """The gradtx arm comes from the PBS job name (the container passes FW_JOB, not FW_ARM: 26075618 / 26075620)."""
    assert rc.gradtx_arm('fwrcg2') == 'RAVQC_W2' and rc.gradtx_arm('fwrcg3r1') == 'RAVQC_W3'
    for bad in ('fwrc2r3', 'fwrcg4', 'fwrcg2x'):
        try:
            rc.gradtx_arm(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_terms():
    work = object.__new__(rc.RavqcWork)
    saved = dict(E=torch.zeros(1, 16), lam=torch.zeros(1), d_res=torch.zeros(16), rho=1.0, D=16, tau=1.0)
    work.M = rc.Metric(saved, 'cpu')
    work.lam_M = 0.5
    lam = rc.RC['loss']['lambda']
    g = torch.Generator().manual_seed(6)
    a, ta = torch.randn(32, 7, generator=g), torch.randn(32, 7, generator=g)
    z, tz = torch.randn(1, 2, 3, 2, 2, generator=g), torch.randn(1, 2, 3, 2, 2, generator=g)
    target = dict(action=ta, future=tz)

    def loss(action, future, tgt):
        A = (action[:10, :6] - tgt['action'][:10, :6]).square().mean()
        Z = (future[:, :, 1:] - tgt['future'][:, :, 1:]).square().sum() / tgt['future'][:, :, 1:].square().sum()
        return A + lam * Z, A, Z
    work.loss = loss
    A, Z, ZM, total = work.terms(a, z, target)
    assert abs(float(ZM) - float(Z)) < 1e-6 and abs(float(total) - float(A + 0.5 * Z)) < 1e-6
    work.loss_checked = False
    work.loss = lambda *args: (loss(*args)[0] * 1.01, None, None)
    try:
        work.terms(a, z, target)
    except RuntimeError:
        pass
    else:
        raise AssertionError('a loss mismatch was accepted')


def test_unseen():
    """The prequential C80 check: batches 2.. before their own update over the same contexts at the start; batch 1
    (trained before anything else was) is left out."""
    rows = [dict(case_id=c, first10motor_mse=1.0 + c % 2, future_latent_nmse=0.1, future_metric_nmse=0.4)
            for c in range(2000, 2040)]
    hist = [dict(update=u + 1, cases=list(range(2000 + 10 * u, 2010 + 10 * u)),
                 before=dict(A=100.0, Z=100.0, ZM=100.0) if u == 0 else dict(A=0.75 * 1.5, Z=0.1, ZM=0.2))
            for u in range(4)]
    out = rc.unseen(hist, rows)
    assert abs(out['A'] - 0.75) < 1e-12 and abs(out['Z'] - 1.0) < 1e-12 and abs(out['ZM'] - 0.5) < 1e-12, out
    assert rc.unseen(hist[:1], rows) == dict(A=None, Z=None, ZM=None)
    # final()'s report on a summary of its shape (one trained arm, one not), so the last stage cannot fail on formatting
    o = dict(mixed_loss=0.001, first10motor_rmse=0.02, future_latent_nmse=0.03, first10gripper_rmse=0.04,
             first10motor_rmse_swap=0.01, future_metric_nmse=0.05)
    fit = dict(mixed_loss=0.001, future_latent_nmse=0.03, future_metric_nmse=0.05)
    arm = dict(arm=dict(eta=0.005), training=dict(lam_M=0.05, history=hist), fit_before=fit, fit_after=fit,
               complete=dict(bpw=2.98), unseen=out)
    s = dict(pbs_jobid='TEST', ravqc=rc.RC, heldout={a: o for a in ('RAVQU_W3', 'QTIP_W3I', 'QTIP_W3', 'RAVQC_W3')},
             zm={'RAVQU_W3': dict(future_metric_nmse=0.05)}, arms=dict(RAVQC_W3=arm, RAVQC_W2=dict(arm, unseen=dict(
                 A=None, Z=None, ZM=None))),
             ratios={'RAVQC_W3/RAVQU_W3': dict(o)},
             metric=dict(metric=dict(rank=32), stats=dict(top_share=0.5, effective_rank=41.5),
                         stability=dict(diag_corr=0.86, subspace_overlap=0.19)))
    text = rc.report(s)
    assert '0.750 / 1.000 / 0.500' in text and '— / — / —' in text and rc.RC['training']['update_rule_id'] in text
    assert '所有层共用的解码表' in text and '每行尺度' not in text, text


def test_reader_backward_passes():
    """fwrc1 26073411: the metric probes backward several times through one checkpointed reader graph, and the frozen
    reader's recompute context is a one-shot generator (the second pass raised AttributeError). With RavqcWork's reader the
    recompute context is re-entered once per pass, the BF16 routing (depth counter) is active during every recompute and
    each probe's gradient equals plain autograd; the patch is undone when the reader closes."""
    from contextlib import contextmanager, nullcontext
    from torch.utils.checkpoint import checkpoint
    gw = rc.gw
    entries = []

    @contextmanager
    def recompute(modules, original_weights, state):
        entries.append(1)
        state['recompute_depth'] += 1
        try:
            yield
        finally:
            state['recompute_depth'] -= 1

    gen = torch.Generator().manual_seed(11)
    w = torch.randn(6, 6, generator=gen, dtype=torch.float64)
    x = torch.randn(3, 6, generator=gen, dtype=torch.float64)
    seen = []

    def f(v, state):
        seen.append(state['recompute_depth'])
        return torch.tanh(v @ w) @ w

    def probes(context_fn, state):
        leaf = x.clone().requires_grad_(True)
        y = checkpoint(f, leaf, state, use_reentrant=False, context_fn=context_fn)
        return [torch.autograd.grad(y, [leaf], torch.full_like(y, p + 1.0), retain_graph=p < 2)[0] for p in range(3)]

    state = dict(recompute_depth=0)
    try:
        probes(lambda: (nullcontext(), recompute(None, None, state)), state)
    except AttributeError:
        plain_fails = True
    else:
        plain_fails = False
    print('one-shot recompute context fails on the second pass:', plain_fails, flush=True)
    assert plain_fails, 'the 26073411 failure did not reproduce on this torch'

    saved = (gw.fr._checkpoint_context_fn, getattr(gw.fr, '_recompute_bf16_context', None), gw.GateWork.bf16_reader)
    original = object()

    @contextmanager
    def parent(self):
        yield 'reader'

    gw.fr._checkpoint_context_fn, gw.fr._recompute_bf16_context, gw.GateWork.bf16_reader = original, recompute, parent
    try:
        work = object.__new__(rc.RavqcWork)
        with work.bf16_reader() as reader:
            assert reader == 'reader' and gw.fr._checkpoint_context_fn is rc.reusable_checkpoint_contexts
            entries.clear()
            seen.clear()
            state = dict(recompute_depth=0)
            got = probes(gw.fr._checkpoint_context_fn(None, None, state), state)
        assert gw.fr._checkpoint_context_fn is original
    finally:
        gw.fr._checkpoint_context_fn, gw.fr._recompute_bf16_context, gw.GateWork.bf16_reader = saved
    assert len(entries) == 3 and seen == [0, 1, 1, 1] and state['recompute_depth'] == 0, (entries, seen)
    leaf = x.clone().requires_grad_(True)
    y = f(leaf, dict(recompute_depth=0))
    for p, g in enumerate(got):
        ref, = torch.autograd.grad(y, [leaf], torch.full_like(y, p + 1.0), retain_graph=True)
        assert torch.allclose(g, ref, rtol=0, atol=1e-12), (p, (g - ref).abs().max())


if __name__ == '__main__':
    for test in (test_hadamard, test_adjoint, test_codes_roundtrip, test_qtipgrad, test_rebuild_identity, test_fold,
                 test_metric, test_terms, test_reader_backward_passes, test_gradtx_arm, test_transfer_stats, test_pipeline, test_unseen):
        test()
        print('PASS', test.__name__, flush=True)
    print('ALL PASS')

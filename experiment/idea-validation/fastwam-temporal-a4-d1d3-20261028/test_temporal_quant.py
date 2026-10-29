"""CPU checks for temporal_quant: grid equals the historical A4/A8 formulas bitwise, the cross-step error identity
holds, first-order feedback telescopes, DPCM tracks a drifting input, static inputs reuse their operand,
per-call bit selection follows the prefill/action rules, and the D1 recorder statistics are correct."""
import torch

import temporal_quant as tq


def historical_a4(z):  # rotation_bank.A4.forward
    scale = z.float().abs().amax(-1, keepdim=True) / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(z.float() / safe).clamp(-7, 7) * scale).to(torch.bfloat16)


def historical_a8(z):  # scalar_quant.a8
    z = z.float()
    scale = z.abs().amax(-1, keepdim=True) / 127
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(z / safe).clamp(-127, 127) * scale).to(torch.bfloat16)


def drifting(steps, shape, g, rho=0.9):
    """z_s = rho z_{s-1} + sqrt(1-rho^2) noise, with a few outlier channels."""
    z = [torch.randn(shape, generator=g)]
    for _ in range(steps - 1):
        z.append(rho * z[-1] + (1 - rho ** 2) ** .5 * torch.randn(shape, generator=g))
    for v in z:
        v[..., :3] *= 8
    return z


def test_grid():
    g = torch.Generator().manual_seed(0)
    for shape in ((1, 294, 3072), (1, 32, 1024), (1, 1, 7), (2, 5, 300)):
        z = torch.randn(shape, generator=g) * 3
        z[..., 0, :] = 0  # an all-zero token row
        assert torch.equal(tq.grid(z, 4).to(torch.bfloat16), historical_a4(z))
        assert torch.equal(tq.grid(z, 8).to(torch.bfloat16), historical_a8(z))
        # group >= width is per-token; group 32 bounds every channel group by its own absmax
        if shape[-1] <= 32:
            assert torch.equal(tq.grid(z, 4, 32), tq.grid(z, 4))
        else:
            q = tq.grid(z, 4, 32)
            assert q.shape == z.shape
            pad = (-shape[-1]) % 32
            zp = torch.nn.functional.pad(z, (0, pad)).reshape(*shape[:-1], -1, 32)
            qp = torch.nn.functional.pad(q, (0, pad)).reshape(*shape[:-1], -1, 32)
            err = (qp - zp).abs().amax(-1)
            assert bool((err <= zp.abs().amax(-1) / 7 / 2 + 1e-6).all())
            assert float((q - z).square().mean()) < float((tq.grid(z, 4) - z).square().mean())


def run_chain(zs, bits=4, innovation=False, feedback=0.0, group=None):
    st, outs, ws = tq.State(), [], []
    for z in zs:
        xhat, w = tq.step(z, bits, group, innovation, feedback, st)
        if innovation:
            st.xhat = xhat
        st.w = w if feedback else None
        outs.append(xhat)
        ws.append(w)
    return outs, ws


def test_error_identity_and_telescoping():
    g = torch.Generator().manual_seed(1)
    zs = drifting(10, (1, 64, 256), g)
    for innovation in (False, True):
        for h in (-1.0, -0.5):
            outs, ws = run_chain(zs, innovation=innovation, feedback=h)
            for s, (xhat, z) in enumerate(zip(outs, zs)):
                expect = ws[s] + (h * ws[s - 1] if s else 0)
                assert torch.allclose(xhat - z, expect, atol=1e-5), (innovation, h, s)
        outs, ws = run_chain(zs, innovation=innovation, feedback=-1.0)
        total = sum(x - z for x, z in zip(outs, zs))
        assert torch.allclose(total, ws[-1], atol=1e-4)
    # constant input: RTN error accumulates coherently, first-order feedback keeps the sum bounded
    const = [zs[0]] * 10
    rtn, _ = run_chain(const)
    sd, _ = run_chain(const, feedback=-1.0)
    acc_rtn = sum(x - z for x, z in zip(rtn, const)).square().mean()
    acc_sd = sum(x - z for x, z in zip(sd, const)).square().mean()
    assert float(acc_sd) * 20 < float(acc_rtn)


def test_innovation_tracks_drift():
    g = torch.Generator().manual_seed(2)
    zs = drifting(10, (1, 64, 512), g, rho=0.98)
    rtn, _ = run_chain(zs)
    inno, _ = run_chain(zs, innovation=True)
    e_rtn = sum(float((x - z).square().mean()) for x, z in zip(rtn[1:], zs[1:]))
    e_inno = sum(float((x - z).square().mean()) for x, z in zip(inno[1:], zs[1:]))
    assert e_inno * 4 < e_rtn


def test_quantizer_static_and_bits():
    stream = lambda name: name.split('.')[0]
    seen = []
    q = tq.CrossStepQuantizer(dict(bits=4, innovation=True, feedback=-1.0, prefill_bits=8, action_bits=8),
                              transform=lambda name, x: x.float(), stream=stream,
                              prefill=dict(stream='video', call_index=10),
                              recorder=lambda name, key, call, z: seen.append((name, call)))
    q.reset(0)
    g = torch.Generator().manual_seed(3)
    ctx = torch.randn(1, 129, 64, generator=g).to(torch.bfloat16)
    first = q.operand('video.ctx', ctx)
    for _ in range(10):
        assert q.operand('video.ctx', ctx.clone()) is first  # identical input: cached operand
    assert seen == [('video.ctx', 0)]
    bits = [q.bits_for('video.blk', s) for s in range(11)]
    assert bits == [4] * 10 + [8]
    assert q.bits_for('action.blk', 3) == 8 and q.bits_for('proprio.enc', 0) == 4
    zs = drifting(11, (1, 294, 128), g)
    for z in zs:
        q.operand('video.blk', z.to(torch.bfloat16))
    counts = q.call_counts()
    assert counts['video.ctx|129'] == dict(calls=11, static=10)
    assert counts['video.blk|294'] == dict(calls=11, static=0)
    q.reset(0)
    assert not q.states


def test_dither_reproducible_and_decorrelating():
    stream = lambda name: 'video'
    make = lambda: tq.CrossStepQuantizer(dict(bits=4, dither=True), transform=lambda n, x: x.float(), stream=stream)
    g = torch.Generator().manual_seed(4)
    zs = [torch.randn(1, 64, 256, generator=g).to(torch.bfloat16)]
    zs.append(zs[0] + 1e-2)  # nearly identical, not equal
    a, b = make(), make()
    a.reset(7)
    b.reset(7)
    oa = [a.operand('video.x', z) for z in zs]
    ob = [b.operand('video.x', z) for z in zs]
    assert all(torch.equal(x, y) for x, y in zip(oa, ob))
    e = [(o.float() - z.float()).flatten() for o, z in zip(oa, zs)]
    corr = float(e[0] @ e[1] / (e[0].norm() * e[1].norm()))
    r = [(tq.grid(z, 4) - z.float()).flatten() for z in zs]
    corr_rtn = float(r[0] @ r[1] / (r[0].norm() * r[1].norm()))
    # a 1e-2 shift against an A4 step of ~0.4 moves ~2% of elements across a rounding boundary: corr ~0.87
    assert corr_rtn > 0.8 and abs(corr) < 0.2, (corr_rtn, corr)


def test_recorder():
    def groups(name, tokens):
        return [('all', slice(None)), ('head', slice(0, 2))] if tokens == 6 else [('all', slice(None))]
    rec = tq.StepRecorder(groups, stream=lambda n: 'video')
    g = torch.Generator().manual_seed(5)
    z0 = torch.randn(1, 6, 128, generator=g)
    z1 = z0 + 0.1 * torch.randn(1, 6, 128, generator=g)
    rec('m', ('m', (1, 6, 128)), 0, z0)
    rec('m', ('m', (1, 6, 128)), 1, z1)
    rows = rec.flush()
    assert [r[1] for r in rows] == [0, 1] and set(rows[1][2]) == {'all', 'head'}
    a = rows[1][2]['all']
    assert abs(a['energy'] - float(z1.square().sum())) < 1e-2 * float(z1.square().sum())
    assert abs(a['innovation_energy'] - float((z1 - z0).square().sum())) < 1e-3
    expect = float(((z1 - z0).abs().amax(-1) / z1.abs().amax(-1)).mean())
    assert abs(a['absmax_ratio'] - expect) < 1e-5 and a['has_previous'] == 1 and rows[0][2]['all']['has_previous'] == 0
    r0, r1 = tq.grid(z0, 4) - z0, tq.grid(z1, 4) - z1
    assert abs(a['residual_dot'] - float((r0 * r1).sum())) < 1e-3
    assert not rec.rows and not rec.prev


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)

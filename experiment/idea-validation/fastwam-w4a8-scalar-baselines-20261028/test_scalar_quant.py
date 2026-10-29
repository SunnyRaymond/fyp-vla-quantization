"""CPU checks: GPTQ beats RTN on correlated inputs, equals MSE-clipped RTN when H = I, packing round-trips,
and the reused R128 rotations are orthogonal for every input width of the 614 targets."""
import sys
from pathlib import Path

import torch
from torch import nn

import scalar_quant as sq


def correlated(n, k, generator):
    base = torch.randn(n, k // 8 + 1, generator=generator)
    x = base.repeat_interleave(8, 1)[:, :k] + 0.3 * torch.randn(n, k, generator=generator)
    x[:, :3] *= 20  # outlier channels
    return x


def main():
    g = torch.Generator().manual_seed(0)
    for out, k in ((64, 256), (48, 300), (16, 7)):
        w = torch.randn(out, k, generator=g)
        x = correlated(4096, k, g) if k >= 8 else torch.randn(4096, k, generator=g)
        h = x.T @ x / x.shape[0]
        c_r, s_r = sq.rtn(w)
        c_g, s_g = sq.gptq(w, h)
        assert c_g.abs().max() <= 7 and c_r.abs().max() <= 7
        assert s_g.shape == (out, -(-k // 128)) and s_g.dtype == torch.bfloat16
        e_r = sq.proxy_nmse(w, sq.dequant(c_r, s_r), h)
        e_g = sq.proxy_nmse(w, sq.dequant(c_g, s_g), h)
        print(f'out={out} k={k}: RTN {e_r:.5f} GPTQ {e_g:.5f}')
        assert e_g < e_r
        assert torch.equal(sq.unpack(sq.pack(c_g), k), c_g)
    # H = I (no damping effect on ordering): GPTQ has nothing to propagate, so it is MSE-clipped RTN.
    w = torch.randn(32, 256, generator=g)
    c_g, s_g = sq.gptq(w, torch.eye(256))
    ref_s = torch.cat([sq.mse_scale(w[:, i:i + 128]) for i in (0, 128)], 1).to(torch.bfloat16)
    assert torch.equal(s_g, ref_s)
    ref_c = torch.cat([sq._codes(w[:, i:i + 128], ref_s[:, j:j + 1].float()) for j, i in enumerate((0, 128))], 1)
    assert torch.equal(c_g.float(), ref_c)
    x = torch.randn(5, 3072, dtype=torch.bfloat16)
    assert sq.a8(x).dtype == torch.bfloat16 and float((sq.a8(x).float() - x.float()).abs().max()) < 0.05
    print('bpw 3072x14336:', sq.bpw(3072, 14336))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fastwam-rotation-baselines'))
    import rotation
    widths = {'video': (3072, 14336, 256), 'action': (1024, 4096, 3072, 256, 7), 'proprio': (8,)}
    prefix = {'video': 'video_expert', 'action': 'action_expert', 'proprio': 'proprio_encoder'}
    modules = {f'{prefix[s]}.m{i}': nn.Linear(k, 1) for s, ks in widths.items() for i, k in enumerate(ks)}
    for mode in ('fixed', 'learned'):
        state = rotation.build_rotation_state(modules, mode=mode, seed=20261008)
        for name, m in modules.items():
            stream = state.stream_for(name)
            r = state.transform(torch.eye(m.in_features), stream).float()
            err = float((r.T @ r - torch.eye(m.in_features)).abs().max())
            assert err < 1e-4, (mode, name, err)
        print(mode, 'orthogonal for widths', sorted({m.in_features for m in modules.values()}))
    print('ALL PASS')


if __name__ == '__main__':
    main()

"""CPU checks for the GPTQ port: H = I reduces to per-row RTN; correlated H beats RTN; odd shapes work."""
from __future__ import annotations

import torch

from quant import find_params, quantize, quantize_layer


def proxy(w, q, h):
    d = q - w
    return float(((d @ h) * d).sum() / ((w @ h) * w).sum())


def correlated(n, k, generator):
    base = torch.randn(n, k // 4, generator=generator)
    x = base @ torch.randn(k // 4, k, generator=generator) + 0.1 * torch.randn(n, k, generator=generator)
    x[:, :3] *= 20  # a few outlier channels, as in real activations
    return x.t() @ x / n


def main():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 256, generator=g) * 0.02
    for bits in (2, 3):
        scale, zero, maxq = find_params(w, bits)
        rtn = quantize(w, scale, zero, maxq)
        same = quantize_layer(w, torch.eye(256), bits, act_order=False)['dequant']
        assert torch.allclose(same, rtn), 'H = I must reduce to per-row RTN'
        h = correlated(2048, 256, g)
        out = quantize_layer(w, h, bits)
        e_gptq, e_rtn = proxy(w, out['dequant'], h), proxy(w, rtn, h)
        assert e_gptq < 0.7 * e_rtn, (bits, e_gptq, e_rtn)
        levels = out['dequant'] / scale + zero
        assert (levels - levels.round()).abs().max() < 1e-3 and levels.min() > -1e-3 and levels.max() < maxq + 1e-3
        print(f'bits={bits} proxy gptq={e_gptq:.4f} rtn={e_rtn:.4f} bpw={out["bits_total"] / w.numel():.4f}')
    for shape in ((1024, 7), (7, 1024), (4096, 8), (192, 96)):
        h = correlated(512, shape[1], g) if shape[1] >= 8 else torch.eye(shape[1]) * 3
        h[0, :] = h[:, 0] = 0  # one dead input
        out = quantize_layer(torch.randn(*shape, generator=g), h, 2)
        assert out['dequant'].shape == shape and torch.isfinite(out['dequant']).all()
        assert (out['dequant'][:, 0] == 0).all(), 'dead input columns quantize from zero'
    print('ok')


if __name__ == '__main__':
    main()

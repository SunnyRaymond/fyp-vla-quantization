"""Small GPU checks run inside the experiment's PBS allocation."""
import json
import math
from pathlib import Path
import os

import torch
import sampler
import train_banks
from clip_banks import ClippedA4
import smooth_vq


def main():
    sampler._require_compute_allocation()
    torch.manual_seed(20261017)
    checked = 0
    for width in (32, 128, 256, 257):
        x = torch.randn(2, 3, width, device='cuda', dtype=torch.bfloat16)
        transform = smooth_vq.make_transform(
            weight=torch.randn(4, width, device='cuda'), act_max=x.float().reshape(-1, width).abs().amax(0),
            alpha=0.5, hadamard=True, block=128, seed=20261006)
        ratio = torch.tensor(1., device='cuda', requires_grad=True)
        expected = train_banks.A4Input.apply(x, transform)
        actual = ClippedA4.apply(x, ratio, transform)
        assert torch.equal(expected, actual), f'ratio=1 A4 bitwise mismatch: width={width}'
        a, b = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
        upstream = torch.randn_like(x)
        train_banks.A4Input.apply(a, transform).backward(upstream)
        ClippedA4.apply(b, ratio, transform).backward(upstream)
        assert torch.equal(a.grad, b.grad), f'ratio=1 inverse H128 gradient mismatch: width={width}'
        checked += 1

    transform = {'scale': torch.ones(4, device='cuda'), 'signs': torch.ones(4, device='cuda'),
                 'hadamard': False, 'block': 128, 'input_dim': 4}
    x = torch.tensor([[8., 1., -2., 0.]], device='cuda', requires_grad=True)
    ratio = torch.tensor(0.5, device='cuda', requires_grad=True)
    y = ClippedA4.apply(x, ratio, transform)
    assert float(y[0, 0]) == 4., y
    y[0, 0].backward()
    assert bool(torch.isfinite(ratio.grad)) and float(ratio.grad) > 0
    assert float(x.grad[0, 0]) == 0., 'clipped input gradient must be zero'
    expected_ratio_gradient = 8 / math.sqrt(4 * 7)
    assert math.isclose(float(ratio.grad), expected_ratio_gradient, rel_tol=1e-6)

    x = torch.zeros(2, 4, device='cuda', requires_grad=True)
    ratio = torch.tensor(1., device='cuda', requires_grad=True)
    y = ClippedA4.apply(x, ratio, transform)
    y.float().sum().backward()
    assert torch.equal(y, torch.zeros_like(y)) and float(ratio.grad) == 0.
    assert bool(torch.isfinite(x.grad).all())
    receipt = {'passed': True, 'ratio_one_bitwise_fixtures': checked,
               'clipped_scale_gradient_checked': True, 'zero_range_checked': True,
               'pbs_jobid': os.environ['PBS_JOBID']}
    (Path(os.environ['ARTIFACTS']) / 'clip_selfcheck.json').write_text(json.dumps(receipt, indent=2))
    print('CLIPPING_SELFCHECK_PASSED ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

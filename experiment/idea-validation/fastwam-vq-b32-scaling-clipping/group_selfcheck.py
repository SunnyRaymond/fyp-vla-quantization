"""Small GPU-only checks for grouped B32 A4 scaling inside a PBS allocation."""
import json
import os
from pathlib import Path


def main():
    import sampler

    sampler._require_compute_allocation()

    import torch
    import batched_transform
    import micro_codebook as micro
    import smooth_vq
    from clip_banks import ClippedA4
    from group_banks import GroupedA4, exponent_values

    torch.manual_seed(20261017)
    device = torch.device('cuda')

    # k=0 must preserve the established A4 forward and its input/ratio STEs.
    checked_widths = []
    for width in (32, 128, 257):
        x = torch.randn(2, 3, width, device=device, dtype=torch.bfloat16)
        transform = smooth_vq.make_transform(
            weight=torch.randn(4, width, device=device),
            act_max=x.float().reshape(-1, width).abs().amax(0),
            alpha=0.5, hadamard=True, block=128, seed=20261006)
        old_ratio = torch.tensor(1., device=device, requires_grad=True)
        new_ratio = torch.tensor(1., device=device, requires_grad=True)
        shadow = torch.zeros((width + 31) // 32, device=device, requires_grad=True)
        bounds = (torch.full_like(shadow, -1.), torch.full_like(shadow, 1.))
        old_y = ClippedA4.apply(x, old_ratio, transform)
        new_y = GroupedA4.apply(x, new_ratio, shadow, transform, *bounds, True)
        assert torch.equal(old_y, new_y), f'k=0 A4 output differs at width={width}'
        old_x = x.detach().clone().requires_grad_()
        new_x = x.detach().clone().requires_grad_()
        upstream = torch.randn_like(x)
        ClippedA4.apply(old_x, old_ratio, transform).backward(upstream)
        GroupedA4.apply(new_x, new_ratio, shadow, transform, *bounds, True).backward(upstream)
        assert torch.equal(old_x.grad, new_x.grad), f'k=0 input gradient differs at width={width}'
        assert torch.equal(old_ratio.grad, new_ratio.grad), f'k=0 ratio gradient differs at width={width}'
        checked_widths.append(width)

    # A legal random two-book d4 encoding with a partial B32 group and d4 tail.
    rows, width = 5, 37
    k = torch.tensor([1, -1], device=device, dtype=torch.int16)
    x = (torch.randn(3, width, device=device) * 0.25).to(torch.bfloat16)
    transform = smooth_vq.make_transform(
        weight=torch.randn(rows, width, device=device), act_max=x.float().abs().amax(0),
        alpha=0.5, hadamard=True, block=128, seed=20261006)
    encoding = {
        'format': micro.FORMAT, 'shape': (rows, width), 'block_size': 4, 'scale_block': 32,
        'books': (torch.randn(2, 256, 4, device=device) * 0.1).to(torch.bfloat16),
        'row_scales': torch.linspace(0.75, 1.25, rows, device=device).to(torch.bfloat16),
        'indices': torch.randint(256, (rows, (width + 3) // 4, 2), device=device).to(torch.uint8),
        'scale_exponents': torch.tensor([[123, 130], [124, 129], [125, 128], [126, 127], [127, 126]],
                                        device=device, dtype=torch.uint8),
    }
    weight = micro.decode_vq(encoding)
    folded_encoding = dict(encoding)
    folded_encoding['scale_exponents'] = (encoding['scale_exponents'].to(torch.int16) + k[None, :]).to(torch.uint8)
    folded_weight = micro.decode_vq(folded_encoding)
    d = torch.exp2(k.float()).repeat_interleave(32)[:width]
    expected_folded_weight = (weight.float() * d).to(torch.bfloat16)
    weight_error = folded_weight.float() - expected_folded_weight.float()
    assert torch.equal(folded_weight, expected_folded_weight), 'B32 exponent folding changed the decoded VQ content'

    ratio = torch.tensor(1.25, device=device)
    shadow = k.float()
    lower, upper = torch.full_like(shadow, -1.), torch.full_like(shadow, 1.)
    bias = (torch.randn(rows, device=device) * 0.05).to(torch.bfloat16)
    factored_input = GroupedA4.apply(x, ratio, shadow, transform, lower, upper, True)
    folded_input = GroupedA4.apply(x, ratio, shadow, transform, lower, upper, False)
    factored_y = torch.nn.functional.linear(factored_input, weight, bias)
    folded_y = torch.nn.functional.linear(folded_input, folded_weight, bias)
    folded_error = factored_y.float() - folded_y.float()
    folded_max = float(folded_error.abs().max())
    folded_rms = float(folded_error.square().mean().sqrt())
    assert torch.equal(factored_y, folded_y), \
        f'factorized/folded A4 Linear mismatch: max_abs={folded_max}'

    # With A4 disabled, per-group powers of two still commute through the BF16 Linear.
    raw_input = (torch.randn(3, width, device=device) * 0.25).to(torch.bfloat16)
    transformed_input = batched_transform.transform_input(raw_input, transform).to(torch.bfloat16)
    disabled_factored = torch.nn.functional.linear(transformed_input, weight, bias)
    disabled_folded_input = (transformed_input.float() / d).to(torch.bfloat16)
    disabled_folded = torch.nn.functional.linear(disabled_folded_input, folded_weight, bias)
    disabled_error = disabled_factored.float() - disabled_folded.float()
    disabled_max = float(disabled_error.abs().max())
    assert torch.equal(disabled_factored, disabled_folded), \
        f'A4-disabled BF16 Linear mismatch: max_abs={disabled_max}'

    # The custom shadow surrogate must stay connected and produce a finite signal,
    # including the short final B32 group.
    grad_width = 37
    grad_shadow = torch.zeros(2, device=device, requires_grad=True)
    grad_ratio = torch.tensor(0.8, device=device, requires_grad=True)
    grad_x = torch.randn(4, grad_width, device=device, dtype=torch.bfloat16)
    grad_upstream = torch.randn_like(grad_x)
    grad_y = GroupedA4.apply(grad_x, grad_ratio, grad_shadow, transform,
                             torch.full_like(grad_shadow, -1.), torch.full_like(grad_shadow, 1.), True)
    (grad_y.float() * grad_upstream.float()).sum().backward()
    assert grad_shadow.grad is not None, 'shadow gradient is disconnected'
    assert bool(torch.isfinite(grad_shadow.grad).all()), 'shadow gradient is nonfinite'
    assert bool((grad_shadow.grad != 0).any()), 'shadow gradient is all zero'

    zero_shadow = torch.zeros(2, device=device, requires_grad=True)
    zero_ratio = torch.tensor(1., device=device, requires_grad=True)
    zero_x = torch.zeros(2, grad_width, device=device, dtype=torch.bfloat16, requires_grad=True)
    zero_y = GroupedA4.apply(zero_x, zero_ratio, zero_shadow, transform,
                             torch.full_like(zero_shadow, -1.), torch.full_like(zero_shadow, 1.), True)
    zero_y.float().sum().backward()
    assert torch.equal(zero_y, torch.zeros_like(zero_y)), 'zero-range output is nonzero'
    assert float(zero_ratio.grad) == 0. and bool(torch.isfinite(zero_x.grad).all()), 'zero-range gradient invalid'
    assert zero_shadow.grad is not None and bool(torch.isfinite(zero_shadow.grad).all()), 'zero-range shadow gradient invalid'

    # Clamp against both legal exponent edges and retain the one-element B32 tail.
    original_exponents = torch.tensor([[1, 0], [254, 253]], device=device, dtype=torch.int16)
    edge_lower = torch.maximum(torch.full((2,), -1, device=device, dtype=torch.int16), -original_exponents.amin(0))
    edge_upper = torch.minimum(torch.full((2,), 1, device=device, dtype=torch.int16), 254 - original_exponents.amax(0))
    edge_shadow = torch.tensor([-100., 100.], device=device)
    edge_k = exponent_values(edge_shadow, edge_lower, edge_upper)
    adjusted = original_exponents + edge_k[None, :]
    assert torch.equal(edge_k, torch.tensor([-1, 1], device=device, dtype=torch.int16)), 'exponent edge clamp failed'
    assert bool(((adjusted >= 0) & (adjusted <= 254)).all()), 'folded exponent escaped legal range'
    tail_width = 257
    tail_k = exponent_values(torch.ones((tail_width + 31) // 32, device=device),
                             torch.full(((tail_width + 31) // 32,), -1., device=device),
                             torch.full(((tail_width + 31) // 32,), 1., device=device))
    tail_d = torch.exp2(tail_k.float()).repeat_interleave(32)[:tail_width]
    assert tail_k.numel() == 9 and tail_d.numel() == tail_width and float(tail_d[-1]) == 2., 'B32 tail scale failed'

    receipt = {
        'passed': True,
        'pbs_jobid': os.environ['PBS_JOBID'],
        'k0_bitwise': {'widths': checked_widths, 'output': True, 'input_gradient': True, 'ratio_gradient': True},
        'folded_vq_linear': {
            'encoding_shape': [rows, width], 'k': [1, -1], 'exact': bool(torch.equal(factored_y, folded_y)),
            'max_abs_difference': folded_max, 'rms_difference': folded_rms,
            'decoded_weight_max_abs_difference': float(weight_error.abs().max()),
        },
        'a4_disabled_linear': {'exact': bool(torch.equal(disabled_factored, disabled_folded)),
                               'max_abs_difference': disabled_max},
        'shadow_gradient': {'connected': True, 'finite': True,
                            'nonzero_groups': int((grad_shadow.grad != 0).sum()), 'groups': grad_shadow.numel()},
        'zero_range': {'output_zero': True, 'ratio_gradient_zero': True,
                       'input_gradient_finite': True, 'shadow_gradient_finite': True},
        'exponent_bounds_and_tail': {'edge_exponents': edge_k.tolist(), 'adjusted_min': int(adjusted.min()),
                                     'adjusted_max': int(adjusted.max()), 'tail_width': tail_width,
                                     'tail_groups': int(tail_k.numel()), 'tail_scale': float(tail_d[-1])},
    }
    artifact = Path(os.environ['ARTIFACTS']) / 'group_selfcheck.json'
    artifact.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('GROUP_SELFCHECK_PASSED ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

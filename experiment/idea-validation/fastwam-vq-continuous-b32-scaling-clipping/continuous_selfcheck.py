"""GPU-only checks for continuous B32 scales; run inside an approved PBS allocation."""
import json
import math
import os
from pathlib import Path


def _exact_or_bf16_close(actual, expected, torch, label):
    if torch.equal(actual, expected):
        return {'bitwise': True, 'max_abs_error': 0.0}
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2, msg=label)
    return {'bitwise': False, 'max_abs_error': float((actual.float() - expected.float()).abs().max()),
            'fallback': 'BF16 GEMM tolerance rtol=1e-2, atol=1e-2'}


def _identity_transform(width, device, torch):
    return {'scale': torch.ones(width, device=device), 'signs': torch.ones(width, device=device),
            'hadamard': False, 'block': 128, 'input_dim': width}


def _explicit_a4(x, ratio, d, transform, batched_transform, torch):
    z = batched_transform.transform_input(x, transform).float()
    v = z / d
    maximum = v.abs().amax(dim=-1, keepdim=True)
    scale = maximum * ratio / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (v.div(safe).round().clamp(-7, 7) * scale).to(torch.bfloat16)


def main():
    import sampler

    sampler._require_compute_allocation()

    import torch
    import batched_transform
    import smooth_vq
    from clip_banks import ClippedA4
    from continuous_banks import ContinuousBanks, ContinuousLinear, compensated_weight

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261017)
    device = torch.device('cuda')

    # D=1 must reproduce clipping-only Linear, including input and ratio gradients.
    parity = []
    for width in (32, 128, 257):
        source_x = torch.randn(4, width, device=device).to(torch.bfloat16)
        weight = torch.randn(5, width, device=device).to(torch.bfloat16)
        transform = smooth_vq.make_transform(weight=weight.float(),
            act_max=source_x.float().abs().amax(0),alpha=0.5,hadamard=True,block=128,seed=20261006)
        weight_before = weight.clone()
        bias = torch.randn(5, device=device).to(torch.bfloat16)
        for ratio_value in (1.0, 0.8):
            old_x = source_x.detach().clone().requires_grad_()
            new_x = source_x.detach().clone().requires_grad_()
            old_ratio = torch.tensor(ratio_value, device=device, requires_grad=True)
            new_ratio = torch.tensor(ratio_value, device=device, requires_grad=True)
            log2_scales = torch.zeros((width + 31) // 32, device=device, requires_grad=True)
            old_y = torch.nn.functional.linear(ClippedA4.apply(old_x, old_ratio, transform), weight, bias)
            new_y = ContinuousLinear.apply(new_x, new_ratio, log2_scales, weight, bias, transform)
            assert torch.equal(new_y, old_y), f'D=1 forward differs at width={width}, ratio={ratio_value}'
            upstream = torch.randn_like(old_y)
            old_y.backward(upstream)
            new_y.backward(upstream)
            input_check = _exact_or_bf16_close(new_x.grad, old_x.grad, torch,
                                                f'D=1 input gradient width={width}, ratio={ratio_value}')
            ratio_check = _exact_or_bf16_close(new_ratio.grad, old_ratio.grad, torch,
                                                f'D=1 ratio gradient width={width}, ratio={ratio_value}')
            assert log2_scales.grad is not None and bool(torch.isfinite(log2_scales.grad).all())
            if width == 32:
                assert torch.equal(log2_scales.grad, torch.zeros_like(log2_scales.grad)), \
                    'single-group centered scale gradient must be zero'
            parity.append({'width': width, 'ratio': ratio_value, 'transform':'Smooth0.5/H128', 'forward_exact': True,
                           'input_gradient': input_check, 'ratio_gradient': ratio_check,
                           'scale_gradient_connected_finite': True,
                           'single_group_centered_zero': width == 32})
            assert torch.equal(weight, weight_before), 'ContinuousLinear modified its source weight'

    # Force a deterministic signal in group zero so centering leaves nonzero multi-group gradients.
    width = 257
    transform = _identity_transform(width, device, torch)
    x = torch.randn(4, width, device=device).to(torch.bfloat16)
    x[:, 0] = 10.0
    ratio = torch.tensor(0.8, device=device, requires_grad=True)
    log2_scales = torch.zeros(9, device=device, requires_grad=True)
    weight = torch.zeros(3, width, device=device, dtype=torch.bfloat16)
    weight[:, 0] = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.bfloat16)
    source_weight = weight.clone()
    y = ContinuousLinear.apply(x, ratio, log2_scales, weight, None, transform)
    y.backward(torch.ones_like(y))
    assert log2_scales.grad is not None and bool(torch.isfinite(log2_scales.grad).all())
    nonzero_groups = int((log2_scales.grad != 0).sum())
    assert nonzero_groups > 1, 'multi-group continuous scale gradient has no nonzero signal'
    assert torch.equal(weight, source_weight), 'ContinuousLinear modified its source weight'

    # Fractional log2 scales are applied continuously, with no exponent rounding.
    width = 37
    transform = _identity_transform(width, device, torch)
    x = torch.full((2, width), 0.01, device=device, dtype=torch.bfloat16)
    x[:, 0] = 1.0
    x[:, -1] = 0.95
    ratio = torch.tensor(0.73, device=device)
    log2_scales = torch.tensor([-0.2, 0.3], device=device, requires_grad=True)
    weight = torch.zeros(1, width, device=device, dtype=torch.bfloat16)
    weight[0, -1] = 1.0
    weight_before = weight.clone()
    bias = torch.zeros(1, device=device, dtype=torch.bfloat16)
    d = torch.exp2(log2_scales.detach()).repeat_interleave(32)[:width]
    operand = _explicit_a4(x, ratio, d, transform, batched_transform, torch)
    manual_weight = (weight.float() * d).to(torch.bfloat16)
    expected = torch.nn.functional.linear(operand, manual_weight, bias)
    actual = ContinuousLinear.apply(x, ratio, log2_scales, weight, bias, transform)
    cached_weight = compensated_weight(weight, log2_scales.detach())
    cached_eval = torch.nn.functional.linear(operand, cached_weight, bias)
    assert torch.equal(actual, expected), 'fractional-log2 explicit forward is not exact'
    assert torch.equal(cached_weight, manual_weight), 'precomputed compensated weight differs'
    assert torch.equal(cached_eval, expected), 'fractional-log2 cached eval is not exact'
    rounded_d = torch.exp2(log2_scales.detach().round()).repeat_interleave(32)[:width]
    assert not torch.equal(d, rounded_d), 'fixture did not distinguish continuous scales from rounded exponents'
    unit_operand = _explicit_a4(x, ratio, torch.ones_like(d), transform, batched_transform, torch)
    unit_eval = torch.nn.functional.linear(unit_operand, weight, bias)
    assert not torch.equal(actual, unit_eval), 'continuous D fixture did not change the forward output'
    assert bool((log2_scales.detach() >= -0.25).all() and (log2_scales.detach() <= 0.5).all())
    assert torch.equal(weight, weight_before), 'compensation modified its source weight'

    # Check the declared stored-weight LSQ surrogate directly (not a hard-quant finite difference).
    rows, width = 4, 37
    transform = _identity_transform(width, device, torch)
    x = (torch.randn(rows, width, device=device) * 0.4).to(torch.bfloat16).requires_grad_()
    ratio = torch.tensor(0.73, device=device, requires_grad=True)
    log2_scales = torch.tensor([-0.2, 0.3], device=device, requires_grad=True)
    weight = torch.randn(5, width, device=device).to(torch.bfloat16)
    bias = torch.randn(5, device=device).to(torch.bfloat16)
    d = torch.exp2(log2_scales.detach()).repeat_interleave(32)[:width]
    scaled_weight = (weight.float() * d).to(torch.bfloat16)
    z = batched_transform.transform_input(x.detach(), transform).float()
    v = z / d
    maximum = v.abs().amax(dim=-1, keepdim=True)
    scale = maximum * ratio.detach() / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    u = v / safe
    q = u.round().clamp(-7, 7)
    inside = v.abs() <= maximum * ratio.detach()
    derivative = torch.where(inside, q - u, q).to(torch.bfloat16)
    base_scale = maximum / 7
    upstream = torch.randn(rows, 5, device=device).to(torch.bfloat16)
    g = torch.nn.functional.linear(upstream, scaled_weight.t()).float()
    term = g * derivative.float() * base_scale
    expected_ratio_grad = term.sum() / math.sqrt(x.numel() * 7)
    groups = (width + 31) // 32
    per_channel = term * ratio.detach() * math.log(2)
    padded = torch.nn.functional.pad(per_channel.reshape(rows, width), (0, groups * 32 - width))
    group_sum = padded.reshape(rows, groups, 32).sum((0, 2))
    counts = torch.tensor([32.0, 5.0], device=device)
    expected_log_grad = group_sum / (rows * counts * 7).sqrt()
    expected_log_grad = expected_log_grad - expected_log_grad.mean()
    expected_input_grad = (g * inside / d).to(x.dtype)
    y = ContinuousLinear.apply(x, ratio, log2_scales, weight, bias, transform)
    y.backward(upstream)
    torch.testing.assert_close(ratio.grad, expected_ratio_grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(log2_scales.grad, expected_log_grad, rtol=1e-6, atol=1e-7)
    assert torch.equal(x.grad, expected_input_grad), 'identity-transform input surrogate differs'
    assert bool(torch.isfinite(ratio.grad)) and bool(torch.isfinite(log2_scales.grad).all())

    # Zero input returns bias; its range-dependent parameter gradients are zero and input gradient finite.
    zero_x = torch.zeros(2, width, device=device, dtype=torch.bfloat16, requires_grad=True)
    zero_ratio = torch.tensor(0.8, device=device, requires_grad=True)
    zero_log2 = torch.tensor([-0.2, 0.3], device=device, requires_grad=True)
    zero_bias = torch.randn(5, device=device).to(torch.bfloat16)
    zero_y = ContinuousLinear.apply(zero_x, zero_ratio, zero_log2, weight, zero_bias, transform)
    assert torch.equal(zero_y, zero_bias.expand_as(zero_y)), 'zero-range output must equal bias'
    zero_y.backward(torch.ones_like(zero_y))
    assert float(zero_ratio.grad) == 0.0 and torch.equal(zero_log2.grad, torch.zeros_like(zero_log2.grad))
    assert bool(torch.isfinite(zero_x.grad).all()), 'zero-range input gradient is nonfinite'

    # Exercise the production projection method with a small uninitialized fixture.
    bounds = ContinuousBanks.__new__(ContinuousBanks)
    bounds.ratios = {'fixture': torch.nn.Parameter(torch.tensor(2.0, device=device))}
    bounds.minimum, bounds.maximum, bounds.train_weights = 0.5, 1.5, False
    bounds.shadows = {'fixture': torch.nn.Parameter(torch.tensor([-2.0, 0.0, 2.0], device=device))}
    bounds.lower, bounds.upper = -0.4, 0.6
    bounds.project()
    assert float(bounds.ratios['fixture']) == 1.5
    assert torch.equal(bounds.shadows['fixture'], torch.tensor([-0.4, 0.0, 0.6], device=device))
    assert (bounds.lower, bounds.upper) == (-0.4, 0.6), 'scale bounds changed during projection'

    receipt = {
        'passed': True,
        'pbs_jobid': os.environ['PBS_JOBID'],
        'tf32': False,
        'd1_clipping_parity': parity,
        'scale_gradient': {'single_group_centered_zero': True, 'multi_group_nonzero_groups': nonzero_groups,
                           'multi_group_count': 9, 'finite': True},
        'fractional_log2_forward': {'log2_scales': [-0.2, 0.3], 'no_rounding': True,
                                    'explicit_forward_exact': True, 'cached_eval_exact': True,
                                    'continuous_d_differs_from_one': True},
        'manual_stored_weight_lsq_surrogate': {'width': 37, 'groups': 2, 'tail_width': 5,
                                               'ratio_gradient_checked': True, 'log2_gradient_checked': True,
                                               'input_gradient_checked': True,
                                               'interpretation': 'stored-weight LSQ surrogate; absmax detached; not the exact derivative of hard quantization'},
        'zero_range': {'output_equals_bias': True, 'ratio_gradient_zero': True,
                       'log2_gradient_zero': True, 'input_gradient_finite': True},
        'bounds_and_source_weight': {'projection_clamped_to_configured_bounds': True,
                                     'bounds_unchanged': True, 'source_weights_unchanged': True},
    }
    artifact = Path(os.environ['ARTIFACTS']) / 'continuous_selfcheck.json'
    artifact.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('CONTINUOUS_SELFCHECK_PASSED ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

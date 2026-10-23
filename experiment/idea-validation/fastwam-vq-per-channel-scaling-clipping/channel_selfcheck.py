"""GPU-only per-channel checks against an independent live-amax STE and the completed B32 forward."""
import json
import math
import os
from pathlib import Path


def _assert_close(actual, expected, torch, label, rtol=2e-5, atol=2e-6):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=label)
    return float((actual.float() - expected.float()).abs().max())


def _identity_transform(width, device, torch):
    return {'scale': torch.ones(width, device=device), 'signs': torch.ones(width, device=device),
            'hadamard': False, 'block': 128, 'input_dim': width}


def _postprocess_group_gradient(raw, rows, width, torch):
    assert raw.shape == (width,)
    result = raw / math.sqrt(rows * 7)
    return result - result.mean()


def _reference_case(width, torch, coupled_surrogate, math_module, unique=False, nonunit=False):
    """Compare the production surrogate with an independent live-amax STE graph."""
    rows = 3
    device = torch.device('cuda')
    v_value = torch.full((rows, width), 0.5, device=device)
    v_value[:, 1] = 0.25
    v_value[:, 2] = 7.0
    v_value[:, 3] = 8.0
    if unique:
        tie_positions = (0,)
        v_value[:, 0] = 14.0
    elif width == 37:
        tie_positions = (0, 32)
        v_value[:, 0] = 14.0
        v_value[:, 32] = -14.0
    elif width == 128:
        tie_positions = (31, 32, 96)
        v_value[:, 31] = 14.0
        v_value[:, 32] = -14.0
        v_value[:, 96] = 14.0
    else:
        raise ValueError('reference fixture only covers the requested widths 37 and 128')

    columns = torch.arange(rows * width, device=device, dtype=torch.float32).reshape(rows, width)
    effective_g = 1.0 + (columns.remainder(7) / 8.0) + torch.arange(rows, device=device).reshape(rows, 1) / 8.0
    logs_value = ((torch.arange(width, device=device) % 3) - 1).float() if nonunit else torch.zeros(width, device=device)
    z_value = v_value * torch.exp2(logs_value)
    z = z_value.detach().clone().requires_grad_()
    groups = width
    log2_scales = logs_value.detach().clone().requires_grad_()
    ratio = torch.tensor(0.5, device=device, requires_grad=True)

    # Independent autograd reference: D remains live in v=z/D and in the stored-weight compensation.
    d = torch.exp2(log2_scales)
    v = z / d
    maximum = v.abs().amax(-1, keepdim=True)
    scale = maximum * ratio / 7.0
    u = v / torch.where(scale > 0, scale, torch.ones_like(scale))
    q0 = u.detach().round().clamp(-7, 7)
    inside_ref = (v.abs() <= maximum * ratio).detach()
    q_ste = q0 + inside_ref * (u - u.detach())
    dequant = q_ste * scale
    g0 = effective_g / d.detach()
    loss = (dequant * d * g0).sum()
    loss.backward()

    # This is the production local surrogate input, assembled independently from the reference graph.
    d_value = torch.exp2(logs_value)
    maximum_value = v_value.abs().amax(-1, keepdim=True)
    scale_value = maximum_value * 0.5 / 7.0
    u_value = v_value / scale_value
    q_value = u_value.round().clamp(-7, 7)
    inside = v_value.abs() <= maximum_value * 0.5
    raw_derivative = torch.where(inside, q_value - u_value, q_value)
    derivative = raw_derivative.to(torch.bfloat16)
    assert torch.equal(raw_derivative, derivative.float()), 'fixture derivative must be exactly BF16 representable'
    maxima = v_value.abs() == maximum_value
    max_sign = torch.where(maxima, v_value.sign(), torch.zeros_like(v_value)).to(torch.int8)
    tie_count = maxima.sum(-1, keepdim=True)

    z_grad, ratio_grad, log2_grad = coupled_surrogate(
        effective_g, inside, derivative, maximum_value / 7.0, d_value,
        torch.tensor(0.5, device=device), max_sign, tie_count)
    expected_log2 = _postprocess_group_gradient(log2_scales.grad, rows, width, torch)
    ratio_reference = ratio.grad / math_module.sqrt(effective_g.numel() * 7.0)
    z_error = _assert_close(z_grad, z.grad, torch, f'live-amax z gradient width={width}')
    ratio_error = _assert_close(ratio_grad, ratio_reference, torch,
                                f'live-amax ratio gradient width={width}')
    log_error = _assert_close(log2_grad, expected_log2, torch,
                              f'live-amax normalized log2-D gradient width={width}')

    # Explicitly reject the old detached-absmax formulas on this nonzero row-total fixture.
    old_z_grad = effective_g * inside / d_value
    term = effective_g * derivative.float() * (maximum_value / 7.0)
    old_per_channel = term * 0.5 * math_module.log(2.0)
    old_group_raw = old_per_channel.sum(0)
    old_log2_grad = _postprocess_group_gradient(old_group_raw, rows, width, torch)
    row_totals = (effective_g * derivative.float()).sum(-1)
    assert bool((row_totals.abs() > 1e-3).all()), 'old-formula rejection fixture needs nonzero row totals'
    old_z_differs = not torch.allclose(old_z_grad, z.grad, rtol=1e-5, atol=1e-6)
    old_log_differs = not torch.allclose(old_log2_grad, expected_log2, rtol=1e-5, atol=1e-6)
    assert old_z_differs, f'old input-gradient formula unexpectedly passed width={width}'
    assert old_log_differs, f'old log2-D formula unexpectedly passed width={width}'

    support = max_sign.float() / tie_count
    expected_support = torch.tensor([v_value[0, i].sign().item() / len(tie_positions)
                                     for i in tie_positions], device=device)
    actual_support = support[0, list(tie_positions)]
    assert torch.equal(actual_support, expected_support), f'amax tie share differs width={width}'
    assert bool((max_sign[0, list(tie_positions)] != 0).all())

    # The direct ratio estimator is intentionally unchanged from the continuous-B32 implementation.
    old_ratio = term.sum() / math_module.sqrt(effective_g.numel() * 7.0)
    ratio_carryover_error = _assert_close(ratio_grad, old_ratio, torch,
                                          f'direct ratio gradient carryover width={width}')
    return {'width': width, 'unique_max':unique, 'nonunit_D':nonunit, 'rows': rows, 'channels': groups, 'group_size': 1,
            'bf16_derivative_exact': True, 'live_amax_z_gradient_max_abs_error': z_error,
            'ratio_gradient_max_abs_error': ratio_error, 'log2_gradient_max_abs_error': log_error,
            'ratio_gradient_old_formula_max_abs_error': ratio_carryover_error,
            'tie_positions': list(tie_positions), 'ties_equal_share': True,
            'old_input_formula_rejected': old_z_differs, 'old_log2_formula_rejected': old_log_differs,
            'nonzero_row_totals': True}


def _forward_parity(torch, smooth_vq, ContinuousLinear, CoupledLinear, device):
    results = []
    for width in (32, 128, 257):
        groups = width
        x = torch.randn(3, width, device=device).to(torch.bfloat16)
        weight = torch.randn(5, width, device=device).to(torch.bfloat16)
        original_weight = weight.clone()
        bias = torch.randn(5, device=device).to(torch.bfloat16)
        transform = smooth_vq.make_transform(weight=weight.float(), act_max=x.float().abs().amax(0),
                                              alpha=0.5, hadamard=True, block=128, seed=20261006)
        for scale_kind in ('D1', 'fractional_D'):
            if scale_kind == 'D1':
                log2 = torch.zeros(groups, device=device)
            else:
                log2 = torch.linspace(-0.2, 0.3, groups, device=device)
            ratio = torch.tensor(0.8, device=device)
            old_groups = (width + 31) // 32
            old_log2 = torch.zeros(old_groups, device=device) if scale_kind == 'D1' else torch.linspace(-0.2, 0.3, old_groups, device=device)
            log2 = old_log2.repeat_interleave(32)[:width]
            old_y = ContinuousLinear.apply(x, ratio, old_log2, weight, bias, transform)
            new_y = CoupledLinear.apply(x, ratio, log2, weight, bias, transform)
            assert torch.equal(new_y, old_y), f'forward differs width={width}, scale={scale_kind}'
            assert torch.equal(weight, original_weight), f'forward modified source weight width={width}'
            results.append({'width': width, 'transform': 'Smooth0.5/H128', 'scale': scale_kind,
                            'forward_bitwise_equal': True, 'source_weight_unchanged': True})
    return results


def _cached_weight_check(torch, CoupledLinear, quantized_input,
                         compensated_weight, channel_scales, device):
    width = 37
    x = torch.randn(3, width, device=device).to(torch.bfloat16)
    weight = torch.randn(4, width, device=device).to(torch.bfloat16)
    original_weight = weight.clone()
    bias = torch.randn(4, device=device).to(torch.bfloat16)
    ratio = torch.tensor(0.73, device=device)
    log2 = torch.linspace(-0.2, 0.3, width, device=device)
    transform = _identity_transform(width, device, torch)
    operand = quantized_input(x, ratio, log2, transform)
    cached_weight = compensated_weight(weight, log2)
    explicit_weight = (weight.float() * channel_scales(log2, width)).to(torch.bfloat16)
    uncached = CoupledLinear.apply(x, ratio, log2, weight, bias, transform)
    cached = torch.nn.functional.linear(operand, cached_weight, bias)
    assert torch.equal(cached_weight, explicit_weight), 'cached compensated weight differs from explicit BF16 product'
    assert torch.equal(cached, uncached), 'cached compensated-weight forward differs'
    assert torch.equal(weight, original_weight), 'cached check modified its source weight'
    return {'width': width, 'tail_width': 5, 'cached_forward_bitwise_equal': True,
            'compensated_weight_explicitly_equal': True, 'source_weight_unchanged': True}


def _complete_backward_check(torch, smooth_vq, batched_transform, CoupledLinear,
                             channel_scales, compensated_weight, coupled_surrogate, device):
    width, rows = 128, 2
    x = torch.randn(rows, width, device=device).to(torch.bfloat16).requires_grad_()
    weight = torch.randn(4, width, device=device).to(torch.bfloat16)
    bias = torch.randn(4, device=device).to(torch.bfloat16)
    ratio = torch.tensor(0.73, device=device, requires_grad=True)
    log2 = torch.linspace(-0.2, 0.3, width, device=device, requires_grad=True)
    transform = smooth_vq.make_transform(weight=weight.float(), act_max=x.detach().float().abs().amax(0),
                                          alpha=0.5, hadamard=True, block=128, seed=20261006)
    y = CoupledLinear.apply(x, ratio, log2, weight, bias, transform)
    upstream = torch.randn_like(y)
    y.backward(upstream)

    z = batched_transform.transform_input(x.detach(), transform).float()
    d = channel_scales(log2.detach(), width)
    v = z / d
    maximum = v.abs().amax(-1, keepdim=True)
    scale = maximum * ratio.detach() / 7.0
    u = v / torch.where(scale > 0, scale, torch.ones_like(scale))
    q = u.round().clamp(-7, 7)
    inside = v.abs() <= maximum * ratio.detach()
    derivative = torch.where(inside, q - u, q).to(torch.bfloat16)
    max_sign = torch.where(v.abs() == maximum, v.sign(), torch.zeros_like(v)).to(torch.int8)
    tie_count = (v.abs() == maximum).sum(-1, keepdim=True)
    g = torch.nn.functional.linear(upstream, compensated_weight(weight, log2.detach()).t()).float()
    z_grad, ratio_grad, log2_grad = coupled_surrogate(
        g, inside, derivative, maximum / 7.0, d, ratio.detach(), max_sign, tie_count)
    expected_x = batched_transform.inverse_right_r(z_grad, transform['signs'],
        transform['hadamard'], transform['block']) / transform['scale']
    x_error = _assert_close(x.grad, expected_x.to(x.dtype), torch, 'complete backward input gradient',
                            rtol=1e-3, atol=1e-3)
    ratio_error = _assert_close(ratio.grad, ratio_grad, torch, 'complete backward ratio gradient')
    log_error = _assert_close(log2.grad, log2_grad, torch, 'complete backward log2-D gradient')
    return {'width': width, 'rows': rows, 'transform': 'Smooth0.5/H128',
            'input_gradient_max_abs_error': x_error, 'ratio_gradient_max_abs_error': ratio_error,
            'log2_gradient_max_abs_error': log_error}


def _zero_and_single_group_checks(torch, CoupledLinear, coupled_surrogate, device):
    width = 32
    transform = _identity_transform(width, device, torch)
    x = torch.zeros(2, width, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(3, width, device=device).to(torch.bfloat16)
    bias = torch.randn(3, device=device).to(torch.bfloat16)
    ratio = torch.tensor(0.8, device=device, requires_grad=True)
    log2 = torch.zeros(width, device=device, requires_grad=True)
    y = CoupledLinear.apply(x, ratio, log2, weight, bias, transform)
    assert torch.equal(y, bias.expand_as(y)), 'zero input output must equal bias'
    y.backward(torch.ones_like(y))
    assert bool(torch.isfinite(x.grad).all()), 'zero input gradient is nonfinite'
    assert bool(torch.isfinite(ratio.grad)) and float(ratio.grad) == 0.0, 'zero input ratio gradient must be finite zero'
    assert bool(torch.isfinite(log2.grad).all()) and torch.equal(log2.grad, torch.zeros_like(log2.grad)), \
        'zero input log2-D gradient must be finite zero'

    g = torch.ones(2, width, device=device)
    inside = torch.ones_like(g, dtype=torch.bool)
    derivative = torch.full_like(g, 0.5).to(torch.bfloat16)
    max_sign = torch.zeros_like(g, dtype=torch.int8)
    tie_count = torch.full((2, 1), width, device=device, dtype=torch.int64)
    _, _, single_log_grad = coupled_surrogate(g, inside, derivative, torch.ones(2, 1, device=device),
        torch.ones(width, device=device), torch.tensor(0.8, device=device), max_sign, tie_count)
    assert torch.equal(single_log_grad, torch.zeros_like(single_log_grad)), \
        'uniform channel-gradient component must be centered to zero'
    return {'zero_input': {'output_equals_bias': True, 'input_gradient_finite': True,
                           'ratio_gradient_finite_zero': True, 'log2_gradient_finite_zero': True},
            'common_scale_component_gradient_zero': True}


def main():
    import sampler

    sampler._require_compute_allocation()

    import torch
    import batched_transform
    import smooth_vq
    from coupled_banks import CoupledLinear as ContinuousLinear
    from channel_banks import (ChannelLinear as CoupledLinear, coupled_surrogate,
                               channel_scales, compensated_weight, quantized_input)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261018)
    device = torch.device('cuda')
    assert torch.cuda.is_available(), 'approved PBS allocation does not expose CUDA'

    reference = [_reference_case(width, torch, coupled_surrogate, math) for width in (37, 128)]
    reference.append(_reference_case(37, torch, coupled_surrogate, math, unique=True))
    reference.append(_reference_case(37, torch, coupled_surrogate, math, nonunit=True))
    reference.append(_reference_case(128, torch, coupled_surrogate, math, unique=True, nonunit=True))
    forward = _forward_parity(torch, smooth_vq, ContinuousLinear, CoupledLinear, device)
    cached = _cached_weight_check(torch, CoupledLinear, quantized_input,
                                  compensated_weight, channel_scales, device)
    complete_backward = _complete_backward_check(torch, smooth_vq, batched_transform, CoupledLinear,
        channel_scales, compensated_weight, coupled_surrogate, device)
    edge_cases = _zero_and_single_group_checks(torch, CoupledLinear, coupled_surrogate, device)

    receipt = {
        'passed': True,
        'pbs_jobid': os.environ['PBS_JOBID'],
        'independent_reference_passed': True,
        'old_surrogate_rejected': True,
        'tf32': False,
        'independent_live_amax_reference': reference,
        'forward_parity': forward,
        'cached_compensated_weight': cached,
        'complete_backward_vs_surrogate_and_inverseR': complete_backward,
        'edge_cases': edge_cases,
        'interpretation': 'stored-weight LSQ/BF16-coefficient surrogate with live dynamic-absmax chain; not the exact derivative of hard quantization',
    }
    artifact = Path(os.environ['ARTIFACTS']) / 'channel_selfcheck.json'
    artifact.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('CHANNEL_SELFCHECK_PASSED ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

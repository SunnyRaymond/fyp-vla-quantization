"""Allocation-only validation of the new weight adjoint and preserved activation surrogate."""
import json
import os
from pathlib import Path
import socket


def identity(width, torch):
    return {'scale': torch.ones(width, device='cuda'), 'signs': torch.ones(width, device='cuda'),
            'hadamard': False, 'block': 128, 'input_dim': width}


def close(torch, actual, expected, label, bf16=False):
    torch.testing.assert_close(actual, expected, rtol=0.008 if bf16 else 2e-5,
                               atol=0.002 if bf16 else 2e-6, msg=label)
    return float((actual.float() - expected.float()).abs().max())


def linear_case(torch, smooth_vq, old, joint, width, batch_shape, nonunit, hadamard):
    x_value = torch.randn(*batch_shape, width, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(5, width, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(5, device='cuda', dtype=torch.bfloat16)
    transform = (smooth_vq.make_transform(weight=weight.float(),
                 act_max=x_value.float().reshape(-1, width).abs().amax(0),
                 alpha=0.5, hadamard=True, block=128, seed=20261006)
                 if hadamard else identity(width, torch))
    logs_value = (torch.linspace(-0.6, 0.6, width, device='cuda') if nonunit
                  else torch.zeros(width, device='cuda'))
    upstream = torch.randn(*batch_shape, 5, device='cuda', dtype=torch.bfloat16)
    x_old = x_value.clone().requires_grad_()
    r_old = torch.tensor(0.73, device='cuda', requires_grad=True)
    d_old = logs_value.clone().requires_grad_()
    y_old = old.ChannelLinear.apply(x_old, r_old, d_old, weight, bias, transform)
    y_old.backward(upstream)
    x_new = x_value.clone().requires_grad_()
    r_new = r_old.detach().clone().requires_grad_()
    d_new = logs_value.clone().requires_grad_()
    w_new = weight.clone().requires_grad_()
    y_new = joint.JointChannelLinear.apply(x_new, r_new, d_new, w_new, bias, transform)
    assert torch.equal(y_new, y_old), 'Joint must preserve the completed per-channel forward'
    y_new.backward(upstream)
    assert torch.equal(x_new.grad, x_old.grad), 'Input surrogate changed'
    assert torch.equal(r_new.grad, r_old.grad), 'Ratio surrogate changed'
    assert torch.equal(d_new.grad, d_old.grad), 'D surrogate changed'

    # Independent native autograd differentiates the actual BF16 compensated F.linear path.
    qx = old.quantized_input(x_value, r_old.detach(), logs_value, transform).detach()
    w_ref = weight.clone().requires_grad_()
    y_ref = torch.nn.functional.linear(qx, (w_ref.float() * torch.exp2(logs_value)).to(torch.bfloat16), bias)
    y_ref.backward(upstream)
    weight_error = close(torch, w_new.grad, w_ref.grad, 'Compensated weight adjoint', bf16=True)
    assert bool(torch.isfinite(w_new.grad).all()) and bool((w_new.grad != 0).any())
    return {'width': width, 'batch_shape': list(batch_shape), 'nonunit_D': nonunit,
            'hadamard': hadamard, 'forward_bitwise': True, 'activation_gradients_bitwise': True,
            'weight_gradient_max_abs': weight_error}


def master_case(torch, train_banks, old, joint, nonunit):
    width, outputs = 32, 5
    books_value = torch.randn(2, 256, 4, device='cuda') * 0.1
    logs_value = torch.linspace(-0.2, 0.2, outputs, device='cuda')
    indices = torch.randint(0, 256, (outputs, width // 4, 2), device='cuda', dtype=torch.uint8)
    exponents = torch.full((outputs, 1), 127, device='cuda', dtype=torch.uint8)
    active = torch.ones(outputs, device='cuda')
    stored = books_value.to(torch.bfloat16).float()
    vectors = stored[0][indices[..., 0].long()] + stored[1][indices[..., 1].long()]
    dense = (vectors * logs_value.exp().to(torch.bfloat16).float()[:, None, None]).reshape(outputs, width).to(torch.bfloat16)
    x = torch.randn(2, 3, width, device='cuda', dtype=torch.bfloat16)
    ratio = torch.tensor(0.81, device='cuda')
    log2d = (torch.linspace(-0.6, 0.6, width, device='cuda') if nonunit
             else torch.zeros(width, device='cuda'))
    bias = torch.randn(outputs, device='cuda', dtype=torch.bfloat16)
    upstream = torch.randn(2, 3, outputs, device='cuda', dtype=torch.bfloat16)
    transform = identity(width, torch)
    books = books_value.clone().requires_grad_()
    logs = logs_value.clone().requires_grad_()
    decoded = train_banks.DecodedWeight.apply(books, logs, active, indices, exponents, dense)
    # First three arguments are frozen: this catches a missing weight-only save/backward branch.
    result = joint.JointChannelLinear.apply(x, ratio, log2d, decoded, bias, transform)
    result.backward(upstream)
    books_ref = books_value.clone().requires_grad_()
    logs_ref = logs_value.clone().requires_grad_()
    decoded_ref = train_banks.DecodedWeight.apply(books_ref, logs_ref, active, indices, exponents, dense)
    operand = old.quantized_input(x, ratio, log2d, transform).detach()
    reference = torch.nn.functional.linear(operand,
        (decoded_ref.float() * torch.exp2(log2d)).to(torch.bfloat16), bias)
    assert torch.equal(result, reference)
    reference.backward(upstream)
    assert all(v.grad is not None and bool(torch.isfinite(v.grad).all())
               and bool((v.grad != 0).any()) for v in (books, logs))
    return {'nonunit_D': nonunit,
            'books_gradient_max_abs': close(torch, books.grad, books_ref.grad, 'VQ book upstream adjoint'),
            'log_row_gain_gradient_max_abs': close(torch, logs.grad, logs_ref.grad, 'VQ row-gain upstream adjoint'),
            'masters_connected_finite_nonzero': True, 'weight_only_gradient_path': True}


def zero_case(torch, joint):
    width = 37
    x = torch.zeros(2, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(5, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    r = torch.tensor(0.8, device='cuda', requires_grad=True)
    d = torch.zeros(width, device='cuda', requires_grad=True)
    y = joint.JointChannelLinear.apply(x, r, d, w, None, identity(width, torch))
    y.sum().backward()
    assert torch.equal(y, torch.zeros_like(y))
    assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in (x, w, r, d))
    assert torch.equal(w.grad, torch.zeros_like(w.grad))
    assert float(r.grad) == 0 and torch.equal(d.grad, torch.zeros_like(d.grad))
    return {'zero_input_finite': True, 'zero_weight_ratio_D_gradients': True}


def main():
    if not os.environ.get('PBS_JOBID') or not os.environ.get('PBS_NODEFILE') or 'login' in socket.gethostname():
        raise RuntimeError('A real compute allocation is required')
    import torch
    import sampler
    import smooth_vq
    import channel_banks as old
    import joint_banks as joint
    import train_banks
    sampler._require_compute_allocation()
    original = json.loads((Path(os.environ['ARTIFACTS']) / 'channel_selfcheck.json').read_text())
    if not original.get('passed') or original.get('pbs_jobid') != os.environ['PBS_JOBID']:
        raise RuntimeError('Completed per-channel independent live-amax selfcheck is required')
    torch.manual_seed(20261018)
    torch.backends.cuda.matmul.allow_tf32 = False
    linear = [linear_case(torch, smooth_vq, old, joint, width, shape, nonunit, hadamard)
              for width, shape, hadamard in ((37, (3,), False), (128, (2, 3), True), (257, (2,), True))
              for nonunit in (False, True)]
    masters = [master_case(torch, train_banks, old, joint, nonunit) for nonunit in (False, True)]
    result = {'passed': True, 'pbs_jobid': os.environ['PBS_JOBID'],
              'weight_gradient_reference_passed': True, 'frozen_parameter_gradients_preserved': True,
              'master_gradient_connected': True, 'independent_reference_passed': True,
              'linear_cases': linear, 'master_cases': masters, 'zero_case': zero_case(torch, joint),
              'scope': 'New BF16 compensated-weight adjoint and unchanged declared per-channel STE; not an exact hard-round derivative.'}
    (Path(os.environ['ARTIFACTS']) / 'joint_selfcheck.json').write_text(json.dumps(result, indent=2) + '\n')
    print('JOINT_CHANNEL_SELFCHECK_COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

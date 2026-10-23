"""Necessary loss and staged-chain-rule checks, only inside a CPU/GPU allocation."""
import json
import os
from pathlib import Path
from anchor_selfcheck import require_allocation


def main():
    require_allocation()
    import torch
    from reader_loss import reader_terms, gate
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    config = {'bins': 16, 'interior_min': -3.5, 'interior_max': 3.5,
              'sigma': .375, 'std_floor': .02}
    stats = {'action': {'mean': [0.] * 6, 'scale': [1.] * 6}}
    target = torch.randn(32, 7, requires_grad=True)
    same = reader_terms(torch, target.detach().clone(), target, stats, config)
    assert all(abs(float(value)) < 1e-7 for value in same.values())
    action = (target.detach() + .1).requires_grad_(True)
    value = reader_terms(torch, action, target, stats, config)['reader_kl']
    value.backward()
    assert value > 0 and target.grad is None and torch.isfinite(action.grad).all()
    assert action.grad[:10, :6].abs().sum() > 0
    assert action.grad[10:].abs().sum() == 0 and action.grad[:, 6].abs().sum() == 0
    # Compare a joint graph with output cotangents followed by one student VJP.
    theta = torch.randn(7, requires_grad=True)
    matrix_a, matrix_z, matrix_r = torch.randn(5, 7), torch.randn(9, 7), torch.randn(5, 9)
    a, z = matrix_a @ theta, matrix_z @ theta
    direct = a.square().mean() + .2 * z.square().mean() + .3 * (matrix_r @ z).square().mean()
    expected, = torch.autograd.grad(direct, theta)
    a_leaf = a.detach().requires_grad_(True)
    z_leaf = z.detach().requires_grad_(True)
    loss = a_leaf.square().mean() + .2 * z_leaf.square().mean() + .3 * (matrix_r @ z_leaf).square().mean()
    ga, gz = torch.autograd.grad(loss, (a_leaf, z_leaf))
    staged, = torch.autograd.grad((matrix_a @ theta, matrix_z @ theta), theta, (ga, gz))
    torch.testing.assert_close(staged, expected)
    good = {'first10motor_rmse': {'delta': -.001, 'median_paired_delta': -.001, 'wins': 21},
            'future_latent_nmse': {'delta': 0.}}
    assert gate(good)
    good['first10motor_rmse']['wins'] = 20
    assert not gate(good)
    # The endpoint decomposition must preserve cancellation as well as error norms.
    effect = torch.tensor([1.0, -2.0, 3.0])
    gap = torch.tensor([-0.75, 1.5, -2.5])
    direct_error = (effect + gap).square().mean()
    decomposed_error = effect.square().mean() + gap.square().mean() + 2 * (effect * gap).mean()
    torch.testing.assert_close(direct_error, decomposed_error)
    assert (effect * gap).mean() < 0
    from reuse_reference import queries
    protocol = json.loads((Path(os.environ['ARTIFACTS']) / 'protocol.json').read_text())
    assert len(queries(protocol, 'calibration')) == 80
    assert len(queries(protocol, 'heldout')) == 80
    assert len(set(protocol['calibration']['update_order'])) == 80
    assert protocol['training']['updates'] == 80
    result = {'passed': True, 'pbs_jobid': os.environ['PBS_JOBID'],
              'checks': ['identity', 'teacher_detached', 'first10motor_only',
                         'finite_input_gradient', 'staged_vjp_equals_joint', 'full_gate',
                         'reader_error_cancellation', 'full_C80_and_heldout_rosters'],
              'model_queries': 0, 'training_updates': 0}
    (Path(os.environ['ARTIFACTS']) / 'selfcheck.json').write_text(json.dumps(result, indent=2) + '\n')
    print('READER_SELFCHECK_PASSED', result, flush=True)


if __name__ == '__main__':
    main()

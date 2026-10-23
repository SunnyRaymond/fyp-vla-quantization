"""Small mathematical checks, executed only in a genuine CPU/GPU allocation."""
import json
import os
from pathlib import Path
import socket


def require_allocation():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if (not os.environ.get('PBS_JOBID') or 'login' in host or not nodefile
            or host not in {n.split('.')[0] for n in Path(nodefile).read_text().split()}):
        raise RuntimeError('A genuine PBS compute allocation is required')


def main():
    require_allocation()
    import torch
    from anchor_loss import latent_tokens, neighbor_mask, teacher_anchor_kl, objectives
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    checks = []
    config = {'pool_size': 2, 'view_widths': [4, 4], 'exclude_observed_frames': 1, 'std_floor': .02}
    stats = {'latent': {'mean': [0.] * 5, 'scale': [1.] * 5}}
    teacher = torch.randn(2, 5, 4, 4, 8, requires_grad=True)
    student = teacher.detach().clone().requires_grad_(True)
    same = teacher_anchor_kl(torch, student, teacher, stats, config, 1.)
    assert float(same) < 1e-7
    checks.append('identity_zero_and_batch_channel_time_generic')
    altered = (teacher.detach() + .2 * torch.randn_like(teacher)).requires_grad_(True)
    loss = teacher_anchor_kl(torch, altered, teacher, stats, config, 1.)
    loss.backward()
    assert float(loss) > 0 and teacher.grad is None
    assert torch.isfinite(altered.grad).all() and altered.grad[:, :, 1:].abs().sum() > 0
    assert altered.grad[:, :, :1].abs().sum() == 0
    checks.append('finite_student_future_gradient_teacher_detached_frame0_zero')
    tokens, ids = latent_tokens(torch, teacher.detach(), stats, config)
    mask = neighbor_mask(torch, ids)
    assert not bool(mask.diagonal().any())
    assert not bool(mask[ids[:, 1, None] != ids[None, :, 1]].any())
    assert not bool(mask[(ids[:, 0, None] - ids[None, :, 0]).abs() > 1].any())
    checks.append('no_self_cross_view_or_nonadjacent_time_edges')
    bias = teacher.detach().clone()
    bias[:, :, 1:] += torch.arange(1., 6.).reshape(1, 5, 1, 1, 1) * .2
    assert float(teacher_anchor_kl(torch, bias, teacher, stats, config, 1.)) > 1e-6
    frame0 = teacher.detach().clone(); frame0[:, :, :1] += 99
    assert float(teacher_anchor_kl(torch, frame0, teacher, stats, config, 1.)) < 1e-7
    checks.append('fixed_anchors_detect_common_shift_and_ignore_frame0')
    other = teacher.detach().clone(); other[..., 4:] += 10
    first_view_mask = ids[:, 1] == 0
    from anchor_loss import distances, _log_probs
    other_tokens, _ = latent_tokens(torch, other, stats, config)
    logp = _log_probs(torch, distances(tokens, tokens), mask, 1.)
    logq = _log_probs(torch, distances(other_tokens, tokens), mask, 1.)
    assert torch.equal(logp[:, first_view_mask], logq[:, first_view_mask])
    checks.append('independent_camera_views')
    action = torch.randn(32, 7, requires_grad=True)
    target = {'action': action.detach().clone(), 'future': teacher.detach()}
    action_student = action + .1
    from anchor_loss import loss_terms
    values = loss_terms(torch, action_student, altered, target, stats, config, 1.)
    p = {'training': {'rho': 1.5, 'normalization_c': .028303017887797623}}
    base, added = objectives(values, p, .01)
    torch.testing.assert_close(added - base, .01 * values['latent_anchor_kl'])
    assert float(values['action_mse']) > 0
    checks.append('original_mse_nmse_retained_additive_kl_only')
    result = {'passed': True, 'pbs_jobid': os.environ['PBS_JOBID'], 'checks': checks,
              'model_forwards': 0, 'training_updates': 0}
    (Path(os.environ['ARTIFACTS']) / 'anchor_selfcheck.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ANCHOR_SELFCHECK_PASSED ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

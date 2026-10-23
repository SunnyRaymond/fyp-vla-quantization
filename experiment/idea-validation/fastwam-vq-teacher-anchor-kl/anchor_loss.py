"""Conditional neighbor KL with fixed teacher anchors and aligned latent tokens."""
import math


def latent_tokens(torch, future, stats, config):
    if future.ndim != 5 or future.shape[2] <= config['exclude_observed_frames']:
        raise ValueError('Expected [batch,channel,time,height,width] with future frames')
    widths = config['view_widths']
    pool = config['pool_size']
    if (not widths or any(type(w) is not int or w < pool for w in widths)
            or sum(widths) != future.shape[-1] or pool < 1 or future.shape[-2] < pool):
        raise ValueError('Explicit view widths and pooling must fit the latent grid')
    channels = future.shape[1]
    values = future.float()[:, :, config['exclude_observed_frames']:]
    mean = torch.as_tensor(stats['latent']['mean'], device=values.device, dtype=torch.float32).detach()
    scale = torch.as_tensor(stats['latent']['scale'], device=values.device, dtype=torch.float32).detach()
    if (mean.numel() != channels or scale.numel() != channels
            or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all())
            or not bool((scale > 0).all())):
        raise ValueError('Frozen teacher channel statistics differ from latent channels')
    values = (values - mean.reshape(1, -1, 1, 1, 1)) / scale.clamp_min(config['std_floor']).reshape(1, -1, 1, 1, 1)
    blocks, identities = [], []
    for time in range(values.shape[2]):
        offset = 0
        for view, width in enumerate(widths):
            block = torch.nn.functional.avg_pool2d(values[:, :, time, :, offset:offset + width], pool, pool)
            height, cols = block.shape[-2:]
            blocks.append(block.flatten(2).transpose(1, 2))
            identities.extend((time, view, row, col) for row in range(height) for col in range(cols))
            offset += width
    return torch.cat(blocks, dim=1), torch.tensor(identities, device=values.device)


def neighbor_mask(torch, identities):
    time, view, row, col = identities.T
    dt = (time[:, None] - time[None, :]).abs()
    dx = (col[:, None] - col[None, :]).abs()
    dy = (row[:, None] - row[None, :]).abs()
    same_view = view[:, None] == view[None, :]
    spatial = (dt == 0) & (torch.maximum(dx, dy) == 1)
    temporal = (dt == 1) & (dx == 0) & (dy == 0)
    mask = same_view & (spatial | temporal)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError('Each latent token needs at least one eligible neighbor')
    return mask


def distances(left, right):
    return (left[:, :, None, :] - right[:, None, :, :]).square().mean(dim=-1)


def _log_probs(torch, distance, mask, temperature):
    logits = (-distance / temperature).masked_fill(~mask, torch.finfo(distance.dtype).min)
    return torch.log_softmax(logits, dim=-1)


def teacher_anchor_kl(torch, student, teacher, stats, config, temperature, diagnostics=False):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Frozen calibration-only temperature must be positive')
    if student.shape != teacher.shape or student.device != teacher.device:
        raise ValueError('Teacher/student latent shape and device must match')
    query, identities = latent_tokens(torch, student, stats, config)
    with torch.no_grad():
        anchors, target_ids = latent_tokens(torch, teacher.detach(), stats, config)
        if not torch.equal(identities, target_ids):
            raise ValueError('Teacher/student token identities differ')
        mask = neighbor_mask(torch, identities)
        target_logp = _log_probs(torch, distances(anchors, anchors), mask, temperature)
        target_prob = target_logp.exp()
    student_logp = _log_probs(torch, distances(query, anchors), mask, temperature)
    per_token = (target_prob * (target_logp - student_logp)).sum(dim=-1)
    if not bool(torch.isfinite(per_token.detach()).all()) or bool((per_token.detach() < -1e-5).any()):
        raise FloatingPointError('Nonfinite or materially negative teacher-anchor KL')
    loss = per_token.clamp_min(0).mean()
    if not diagnostics:
        return loss
    with torch.no_grad():
        entropy = -(target_prob * target_logp).sum(dim=-1)
        normalizer = mask.sum(-1).float().log().clamp_min(1e-12)
        details = {'teacher_entropy': float(entropy.mean()),
                   'normalized_teacher_entropy': float((entropy / normalizer).mean()),
                   'teacher_max_probability': float(target_prob.max(-1).values.mean()),
                   'tokens_per_sample': query.shape[1], 'channels': query.shape[2],
                   'minimum_neighbors': int(mask.sum(-1).min()),
                   'maximum_neighbors': int(mask.sum(-1).max())}
    return loss, details


def loss_terms(torch, action, future, target, stats, config, temperature):
    target_action = target['action'].detach().to(action.device)
    target_future = target['future'].detach().to(future.device)
    if tuple(action.shape) != (32, 7) or target_action.shape != action.shape:
        raise ValueError('Fast-WAM adapter expects [32,7]; first10 motor6 only')
    student_z = future.float()[:, :, config['exclude_observed_frames']:]
    teacher_z = target_future.float()[:, :, config['exclude_observed_frames']:]
    return {'action_mse': (action[:10, :6].float() - target_action[:10, :6].float()).square().mean(),
            'latent_nmse': (student_z - teacher_z).square().sum() / teacher_z.square().sum().clamp_min(1e-12),
            'latent_anchor_kl': teacher_anchor_kl(torch, future, target_future, stats, config, temperature)}


def objectives(values, protocol, beta):
    base = values['action_mse'] + protocol['training']['rho'] * protocol['training']['normalization_c'] * values['latent_nmse']
    return base, base + beta * values['latent_anchor_kl']

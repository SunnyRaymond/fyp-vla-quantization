"""One frozen-reader auxiliary target; the original action/NMSE terms stay exact."""
from softbin_loss import pointwise_kl


def base_terms(torch, action, future, target):
    a = target['action'].detach().to(action.device)
    z = target['future'].detach().to(future.device)
    return {
        'action_mse': (action[:10, :6].float() - a[:10, :6].float()).square().mean(),
        'latent_nmse': (future[:, :, 1:].float() - z[:, :, 1:].float()).square().sum()
                       / z[:, :, 1:].float().square().sum().clamp_min(1e-12),
    }


def reader_terms(torch, action, target_action, stats, config):
    if tuple(action.shape) != (32, 7) or tuple(target_action.shape) != (32, 7):
        raise ValueError('Reader targets must be normalized full32x7 actions')
    target = target_action.detach().to(action.device)
    scale = torch.as_tensor(stats['action']['scale'], device=action.device).clamp_min(config['std_floor'])
    return {
        'reader_kl': pointwise_kl(torch, action[:10, :6], target[:10, :6],
                                stats['action']['mean'], stats['action']['scale'], -1, config),
        'reader_mse': ((action[:10, :6].float() - target[:10, :6].float()) / scale).square().mean(),
    }


def base_objective(values, protocol):
    return values['action_mse'] + protocol['training']['rho'] * protocol['training']['normalization_c'] * values['latent_nmse']


def gate(metrics):
    a, z = metrics['first10motor_rmse'], metrics['future_latent_nmse']
    return bool(a['delta'] < 0 and a['median_paired_delta'] < 0
                and a['wins'] >= 21 and z['delta'] <= 0)

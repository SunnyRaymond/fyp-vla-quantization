"""CPU construction of probes with matched *actual BF16* norms in every module."""
import math
import torch


def build_modules(directions, previous):
    generator = torch.Generator(device='cpu').manual_seed(previous['sample_seed'] + 7919)
    modules = {}
    for name, d in directions['modules'].items():
        valid = d['valid_coordinates']
        g = d['gradient'].float() * valid
        idx = d['indices'].long()
        scales = d['row_scales'].float()[d['row_ids']]
        books = d['books'].to(torch.bfloat16).float()
        baseline = (((books[0][idx[:, 0]] + books[1][idx[:, 1]]) * scales[:, None]) * valid).to(torch.bfloat16)
        sums, divisor = torch.zeros((256, 4)), torch.zeros((256, 4))
        sums.index_add_(0, idx[:, 0], scales[:, None] * g)
        divisor.index_add_(0, idx[:, 0], scales.square()[:, None] * valid)
        modules[name] = dict(indices=idx, scales=scales, books=books, valid=valid,
                             baseline=baseline, g=g, dc=-sums / divisor.clamp_min(1e-30),
                             untied=-g, random=torch.randn(g.shape, generator=generator) * valid)
    return modules


def candidate(d, kind, alpha):
    if kind == 'shared':
        book = (d['books'][0] + alpha * d['dc']).to(torch.bfloat16).float()
        value = (((book[d['indices'][:, 0]] + d['books'][1][d['indices'][:, 1]])
                  * d['scales'][:, None]) * d['valid']).to(torch.bfloat16)
    else:
        value = (d['baseline'].float() + alpha * d[kind]).to(torch.bfloat16)
    norm = float((value.float() - d['baseline'].float()).square().sum()) ** 0.5
    return value, norm


def fit_module(d, kind, target):
    direction = (d['scales'][:, None] * d['dc'][d['indices'][:, 0]] * d['valid']
                 if kind == 'shared' else d[kind])
    nominal = float(direction.square().sum()) ** 0.5
    if target <= 0 or nominal <= 0:
        raise RuntimeError('Every selected module must have a positive target and direction')
    lo, hi = 0., target / nominal
    for _ in range(35):
        if candidate(d, kind, hi)[1] >= target:
            break
        hi *= 2
    else:
        raise RuntimeError('Cannot bracket a module BF16 update norm')
    for _ in range(30):
        mid = (lo + hi) / 2
        if candidate(d, kind, mid)[1] < target:
            lo = mid
        else:
            hi = mid
    options = [(*candidate(d, kind, alpha), alpha) for alpha in (lo, hi)]
    return min(options, key=lambda value: abs(value[1] - target))


def make_updates(directions, old_updates, old_norms, previous, p):
    modules = build_modules(directions, previous)
    if set(modules) != set(old_updates['baseline']):
        raise RuntimeError('Selected support changed')
    for name, d in modules.items():
        if not torch.equal(d['baseline'], old_updates['baseline'][name]):
            raise RuntimeError('CPU decoder differs from previous actual GPU-checked baseline')
    values, norms, reused = {}, {}, {}
    for profile in p['allocation_profiles']:
        anchor = {'sbudget': 'shared', 'ubudget': 'untied'}[profile]
        for level, eps in enumerate(p['intervention_step_norms']):
            source_label = f'{anchor}_{level}'
            requested = {name: old_norms[source_label]['per_module'][name]['delta_sq'] ** 0.5 for name in modules}
            shared, targets = {}, {}
            for name, d in modules.items():
                if anchor == 'shared':
                    alpha = old_norms[source_label]['alpha']
                    value, actual = candidate(d, 'shared', alpha)
                    if not torch.equal(value, old_updates['values'][source_label][name]):
                        raise RuntimeError('Original shared update cannot be recovered bitwise')
                    shared[name] = (value, actual, alpha)
                else:
                    shared[name] = fit_module(d, 'shared', requested[name])
                targets[name] = shared[name][1]
                if abs(targets[name] / requested[name] - 1) > p['maximum_budget_snap_relative_change']:
                    raise RuntimeError('Attainable shared norm would substantially change the allocation profile')
            for kind in ('shared', 'untied', 'random'):
                label = f'{profile}_{kind}_{level}'
                per_module, updates = {}, {}
                for name, d in modules.items():
                    target = targets[name]
                    if kind == 'shared':
                        value, actual, alpha = shared[name]
                    elif kind == anchor and targets[name] == requested[name]:
                        alpha = old_norms[source_label]['alpha']
                        value, actual = candidate(d, kind, alpha)
                        if not torch.equal(value, old_updates['values'][source_label][name]):
                            raise RuntimeError('Own-profile update cannot be reused bitwise')
                        value = old_updates['values'][source_label][name]
                    else:
                        value, actual, alpha = fit_module(d, kind, target)
                    error = abs(actual - target) / target
                    if not math.isfinite(actual) or error > p['module_norm_relative_tolerance']:
                        raise RuntimeError(f'Module norm cannot meet frozen tolerance: {label}/{name}: {error}')
                    updates[name] = value
                    per_module[name] = dict(requested_anchor_norm=requested[name], target_norm=target,
                        budget_snap_relative_change=target / requested[name] - 1,
                        actual_norm=actual, relative_error=error,
                        delta_sq=actual ** 2, alpha=alpha,
                        predicted_mean_cal_latent_change=float((d['g'] * (value.float() - d['baseline'].float())).sum()))
                values[label] = updates
                norms[label] = dict(profile=profile, kind=kind, epsilon=eps, anchor=source_label,
                    reused_source_label=None, per_module=per_module,
                    total_delta_norm=sum(v['delta_sq'] for v in per_module.values()) ** 0.5,
                    predicted_mean_cal_latent_change=sum(v['predicted_mean_cal_latent_change'] for v in per_module.values()))
                possible = f'{kind}_{level}'
                if all(torch.equal(updates[n], old_updates['values'][possible][n]) for n in modules):
                    reused[label] = possible
                    norms[label]['reused_source_label'] = possible
            for name in modules:
                magnitudes = [norms[f'{profile}_{kind}_{level}']['per_module'][name]['actual_norm']
                              for kind in ('shared', 'untied', 'random')]
                spread = max(magnitudes) / min(magnitudes) - 1
                if spread > p['pairwise_module_norm_relative_tolerance']:
                    raise RuntimeError(f'Pairwise norm spread too large: {profile}/{level}/{name}: {spread}')
    return dict(values=values, norms=norms, reused_labels=reused,
                flat_ids=old_updates['flat_ids'], baseline=old_updates['baseline'])

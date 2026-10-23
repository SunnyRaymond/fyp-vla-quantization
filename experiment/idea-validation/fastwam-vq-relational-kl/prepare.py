"""Validate structural distributions and freeze teacher-only C80 coordinates."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

from relational_selfcheck import require_allocation


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    import torch
    from relational_loss import relational_losses, teacher_distances
    from softbin_loss import losses
    torch.set_num_threads(4)
    p = json.loads(Path(args.protocol).read_text())
    root = Path(p['remote_root'])
    out = root / 'results/prepare'
    out.mkdir(parents=True, exist_ok=True)
    check = json.loads((Path(os.environ['ARTIFACTS']) / 'relational_selfcheck.json').read_text())
    if not check['passed'] or check['pbs_jobid'] != os.environ['PBS_JOBID']:
        raise RuntimeError('Synthetic distribution checks required in this allocation')
    stats = json.loads((Path(p['source_dual_root']) / 'results/fit_stats/summary.json').read_text())
    if not stats['complete'] or stats['heldout_used'] or stats['calibration_case_ids'] != p['calibration']['case_ids']:
        raise RuntimeError('Original frozen C80 teacher coordinates differ')
    plan = json.loads(Path(p['calibration']['plan_file']).read_text())
    if plan['calibration_cohorts']['c80'] != p['calibration']['case_ids']:
        raise RuntimeError('C80 plan differs')
    plan_rows = {v['case_id']: v for v in plan['inputs']}
    teacher_dir = Path(p['control_root']) / 'results/teacher/calibration'
    receipt = json.loads((teacher_dir / 'receipt.json').read_text())
    if not receipt['complete'] or receipt['queries'] != 80 or receipt['case_ids'] != p['calibration']['case_ids'] or receipt['seed_indices'] != [0]:
        raise RuntimeError('Complete original shared C80 teacher required')
    targets = []
    distances = {'action': [], 'latent': []}
    for case in p['calibration']['case_ids']:
        target = torch.load(teacher_dir / f'{case}_0.pt', map_location='cpu', weights_only=True)
        if target['plan_row'] != plan_rows[case] or target['sampler_seed'] != plan_rows[case]['sampler_seeds'][0]:
            raise RuntimeError('Teacher binding differs')
        target['action'] = target['action'].float()
        target['future'] = target['future'].float()
        targets.append((case, target))
        d = teacher_distances(torch, target['action'], target['future'], stats)
        for key in distances:
            distances[key].append(d[key])
    config = {key + '_temperature': max(float(torch.cat(value).median()), 1e-6) for key, value in distances.items()}
    a_scale = torch.tensor(stats['action']['scale']).reshape(1, 6)
    z_scale = torch.tensor(stats['latent']['scale']).reshape(1, 48, 1, 1, 1)
    generator = torch.Generator().manual_seed(p['relational']['corruption_seed'])
    rows = []
    for case, target in targets:
        a, z = target['action'], target['future']
        noisy_a, noisy_z = a.clone(), z.clone()
        noisy_a[:10, :6] += p['relational']['corruption_std'] * a_scale * torch.randn((10, 6), generator=generator)
        noisy_z[:, :, 1:] += p['relational']['corruption_std'] * z_scale * torch.randn(z[:, :, 1:].shape, generator=generator)
        noise_anchor = losses(torch, noisy_a, noisy_z, a, z, stats, p['softbin'])
        noise_rel = relational_losses(torch, noisy_a, noisy_z, a, z, stats, config)
        reverse = a.clone()
        reverse[:10, :6] = a[:10, :6].flip(0)
        bias = a.clone()
        # Per-axis matching preserves both total and standardized MSE.
        bias[:10, :6] += (reverse[:10, :6] - a[:10, :6]).square().mean(0).sqrt()
        spatial = z.clone()
        spatial[:, :, 1:] = torch.cat((torch.roll(z[:, :, 1:, :, :14], 2, -2), torch.roll(z[:, :, 1:, :, 14:], 2, -2)), -1)
        z_bias = z.clone()
        z_bias[:, :, 1:] += (spatial[:, :, 1:] - z[:, :, 1:]).square().mean((0, 2, 3, 4), keepdim=True).sqrt()
        swapped = z.clone()
        swapped[:, :, 1:] = z[:, :, 1:].flip(2)
        conditions = {'same': (a, z), 'action_reverse': (reverse, z), 'action_matched_bias': (bias, z),
                      'latent_spatial_roll': (a, spatial), 'latent_matched_bias': (a, z_bias), 'latent_time_swap': (a, swapped)}
        results = {}
        for name, (sa, sz) in conditions.items():
            rel = relational_losses(torch, sa, sz, a, z, stats, config)
            results[name] = {k: float(v) for k, v in rel.items()}
            if name != 'same':
                anchor = losses(torch, sa, sz, a, z, stats, p['softbin'])
                results[name].update(action_kl=float(anchor['action_kl']), latent_kl=float(anchor['latent_kl']))
            results[name].update(action_mse=float((sa[:10, :6] - a[:10, :6]).square().mean()),
                                 latent_mse=float((sz[:, :, 1:] - z[:, :, 1:]).square().mean()))
        torch.testing.assert_close(torch.tensor(results['action_reverse']['action_mse']), torch.tensor(results['action_matched_bias']['action_mse']), rtol=2e-5, atol=1e-9)
        torch.testing.assert_close(torch.tensor(results['latent_spatial_roll']['latent_mse']), torch.tensor(results['latent_matched_bias']['latent_mse']), rtol=2e-5, atol=1e-9)
        if max(results['same']['action_temporal_kl'], results['same']['latent_spatiotemporal_kl']) > 1e-7:
            raise RuntimeError('Teacher self KL differs from zero')
        rows.append({'case_id': case, 'seed_index': 0, 'noise': {**{k: float(v) for k, v in noise_rel.items()},
                     'action_kl': float(noise_anchor['action_kl']), 'latent_kl': float(noise_anchor['latent_kl'])}, 'conditions': results})
        save(out / 'progress.json', {'pbs_jobid': os.environ['PBS_JOBID'], 'phase': 'definition_validation', 'queries': len(rows), 'model_forwards': 0})
        print('RELATIONAL_DEFINITION_CASE', case, flush=True)
    means = {key: statistics.mean(r['noise'][key] for r in rows) for key in rows[0]['noise']}
    if not all(math.isfinite(v) and v > 0 for v in means.values()):
        raise RuntimeError('Nondegenerate corruption terms required')
    fraction = p['relational']['relative_corruption_weight']
    coefficients = {'action_beta': fraction * means['action_kl'] / means['action_temporal_kl'],
                    'latent_beta': fraction * means['latent_kl'] / means['latent_spatiotemporal_kl']}
    aggregate = {}
    for name in rows[0]['conditions']:
        aggregate[name] = {key: {'mean': statistics.mean(r['conditions'][name][key] for r in rows),
                                 'median': statistics.median(r['conditions'][name][key] for r in rows),
                                 'positive_cases_gt_1e-7': sum(r['conditions'][name][key] > 1e-7 for r in rows)}
                           for key in rows[0]['conditions'][name]}
    # Generic order/space errors must be visible across real calibration states.
    if aggregate['action_reverse']['action_temporal_kl']['median'] <= 1e-7 or aggregate['latent_spatial_roll']['latent_spatiotemporal_kl']['median'] <= 1e-7:
        raise RuntimeError('Relational definition did not detect real teacher order/space corruption')
    validation = {'complete': True, 'passed': True, 'pbs_jobid': os.environ['PBS_JOBID'], 'heldout_used': False,
                  'calibration_case_ids': p['calibration']['case_ids'], 'aggregate': aggregate, 'rows': rows,
                  'boundary': 'fixed-graph affinity KL; rigid translation/sign/direction symmetries remain; softbin anchors required; not joint policy density'}
    save(out / 'definition_validation.json', validation)
    summary = {'protocol': p['protocol'], 'complete': True, 'passed': True, 'pbs_jobid': os.environ['PBS_JOBID'],
               'calibration_case_ids': p['calibration']['case_ids'], 'heldout_used': False, 'new_teacher_queries': 0,
               'model_forwards': 0, 'relational_config': config, 'coefficients': coefficients, 'corruption_means': means,
               'synthetic_checks': check['checks'], 'real_teacher_corruption_summary': aggregate,
               'coefficient_rule': p['relational']['coefficient_rule']}
    save(out / 'summary.json', summary)
    print('RELATIONAL_PREPARE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

"""Freeze C80 teacher coordinates/temperature without new model queries."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
from anchor_selfcheck import require_allocation


def read(path):
    return json.loads(Path(path).read_text())


def main():
    require_allocation()
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True)
    p = read(parser.parse_args().protocol)
    import torch
    from anchor_loss import teacher_anchor_kl
    torch.set_num_threads(4)
    stats = read(Path(p['source_dual_root']) / 'results/fit_stats/summary.json')
    previous = read(Path(p['source_relational_root']) / 'results/prepare/summary.json')
    check = read(Path(os.environ['ARTIFACTS']) / 'anchor_selfcheck.json')
    if not check['passed'] or check['pbs_jobid'] != os.environ['PBS_JOBID']:
        raise RuntimeError('Same-allocation mathematical checks required')
    for value in (stats, previous):
        if (not value['complete'] or value['heldout_used']
                or value['calibration_case_ids'] != p['calibration']['case_ids']):
            raise RuntimeError('Only complete original-C80 teacher preparation may be reused')
    temperature = previous['relational_config']['latent_temperature']
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError('Invalid teacher-only temperature')
    plan = read(p['calibration']['plan_file'])
    if plan['calibration_cohorts']['c80'] != p['calibration']['case_ids']:
        raise RuntimeError('Original C80 differs')
    plan_rows = {r['case_id']: r for r in plan['inputs']}
    teacher_root = Path(p['control_root']) / 'results/teacher/calibration'
    receipt = read(teacher_root / 'receipt.json')
    source_receipt = read(Path(p['source_per_channel_root']) / 'results/train_eval/training_receipt.json')
    if (not receipt['complete'] or receipt['queries'] != 80
            or receipt['case_ids'] != p['calibration']['case_ids'] or receipt['seed_indices'] != [0]
            or not source_receipt['complete'] or source_receipt['updates'] != 80
            or source_receipt['pbs_jobid'] != p['source_per_channel_job']
            or source_receipt['update_order'] != p['calibration']['update_order']):
        raise RuntimeError('Original teacher/control training receipt differs')
    rows = []
    for case in p['calibration']['case_ids']:
        target = torch.load(teacher_root / f'{case}_0.pt', map_location='cpu', weights_only=True)
        if target['plan_row'] != plan_rows[case] or target['sampler_seed'] != plan_rows[case]['sampler_seeds'][0]:
            raise RuntimeError('Teacher input/seed binding differs')
        z = target['future'].float()
        same, diagnostics = teacher_anchor_kl(torch, z, z, stats, p['anchor'], temperature, diagnostics=True)
        shifted = z.clone()
        start = 0
        for width in p['anchor']['view_widths']:
            shifted[:, :, 1:, :, start:start + width] = torch.roll(z[:, :, 1:, :, start:start + width], 2, -2)
            start += width
        bias = z.clone()
        bias[:, :, 1:] += (shifted[:, :, 1:] - z[:, :, 1:]).square().mean((0, 2, 3, 4), keepdim=True).sqrt()
        swapped = z.clone(); swapped[:, :, 1:] = z[:, :, 1:].flip(2)
        errors = {name: float(teacher_anchor_kl(torch, value, z, stats, p['anchor'], temperature))
                  for name, value in (('same', z), ('spatial_roll', shifted), ('matched_channel_bias', bias), ('time_swap', swapped))}
        if float(same) > 1e-7:
            raise RuntimeError('Teacher self KL differs from zero')
        rows.append({'case_id': case, 'seed_index': 0, 'errors': errors, 'diagnostics': diagnostics})
        print('ANCHOR_TEACHER_CASE', case, flush=True)
    means = {key: statistics.mean(r['errors'][key] for r in rows) for key in rows[0]['errors']}
    if means['spatial_roll'] <= 1e-7 or means['time_swap'] <= 1e-7:
        raise RuntimeError('Real C80 structural corruptions are invisible')
    out = Path(p['remote_root']) / 'results/prepare'; out.mkdir(parents=True, exist_ok=True)
    result = {'protocol': p['protocol'], 'complete': True, 'passed': True,
              'pbs_jobid': os.environ['PBS_JOBID'], 'heldout_used': False,
              'calibration_case_ids': p['calibration']['case_ids'], 'temperature': temperature,
              'stats_source': str(Path(p['source_dual_root']) / 'results/fit_stats/summary.json'),
              'model_forwards': 0, 'new_teacher_queries': 0, 'corruption_means': means,
              'mean_normalized_teacher_entropy': statistics.mean(r['diagnostics']['normalized_teacher_entropy'] for r in rows),
              'rows': rows, 'coefficient_frozen': False,
              'next': 'GPU anchor_kl freezes beta from complete C80 initialization before update1'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('ANCHOR_PREPARE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

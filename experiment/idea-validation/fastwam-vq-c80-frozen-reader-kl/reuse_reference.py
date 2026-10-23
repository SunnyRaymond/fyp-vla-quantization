"""Bindings and existing C8/C80 teachers; all tensor access is allocation-only."""
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def queries(p, split):
    spec = p['calibration' if split == 'calibration' else 'evaluation']
    return [(case, seed) for case in spec['case_ids']
            for seed in ([0] if split == 'calibration' else spec['seed_indices'])]


def validate_teacher(target, row, seed):
    if (target['plan_row'] != row or target['sampler_seed'] != row['sampler_seeds'][seed]
            or target.get('seed_index', seed) != seed):
        raise ValueError('Existing teacher input/seed binding differs')


def load_reference(torch, p):
    import latent_reference
    import train_c8_reference as original
    if read(Path(p['baseline_root']) / 'protocol.json') != p['baseline_protocol']:
        raise ValueError('Original C8 initialization protocol differs')
    cache, saved_rows, c = original._load_reference(
        torch, Path(p['baseline_root']) / 'results/reference',
        original._binding(p['baseline_protocol']), p['c8_case_ids'], latent_reference)
    plan = read(p['calibration']['plan_file'])
    rows = {row['case_id']: row for row in plan['inputs']}
    if (plan['status'] != 'complete' or plan['calibration_cohorts']['c80'] != p['calibration']['case_ids']
            or plan['calibration_cohorts']['c8'] != p['c8_case_ids']
            or any(rows[case] != row['plan_row'] for case, row in saved_rows.items())
            or not math.isclose(c, p['training']['normalization_c'], rel_tol=1e-12)):
        raise ValueError('Frozen calibration/normalization differs')
    directory = Path(p['teacher_calibration_directory'])
    receipt = read(directory / 'receipt.json')
    if not receipt['complete'] or receipt['queries'] != 80:
        raise ValueError('Complete existing C80 teacher receipt required')
    for case in p['calibration']['case_ids']:
        target = torch.load(directory / f'{case}_0.pt', map_location='cpu', weights_only=True)
        validate_teacher(target, rows[case], 0)
        if str(case) in cache['teacher']:
            for key in ('action', 'future'):
                if not torch.equal(target[key], cache['teacher'][str(case)][key]):
                    raise ValueError('Existing C80 and original C8 teacher differ')
        cache['teacher'][str(case)] = target
    return cache, saved_rows, rows

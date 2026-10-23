"""CPU combination and finite-step reports for complete frozen C80 gradients."""
import argparse
import importlib.util
import math
import os
from pathlib import Path

from control_common import guard, read, save


def tensor_save(torch, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def merge_shards(torch, parts, expected_ids):
    first = parts[0]
    keys, originals = first['parameter_keys'], first['originals']
    if len(set(keys)) != len(keys) or len(keys) != len(originals):
        raise ValueError('Parameter key/original coverage differs')
    ids = [case_id for part in parts for case_id in part['case_ids']]
    if sorted(ids) != sorted(expected_ids) or len(set(ids)) != len(ids):
        raise ValueError('Disjoint full cohort required')
    for part in parts:
        if (part['parameter_keys'] != keys or len(part['originals']) != len(keys)
                or len(part['gradient_sum']) != len(keys)
                or not all(torch.equal(a, b) for a, b in zip(originals, part['originals']))
                or [row['case_id'] for row in part['case_metrics']] != part['case_ids']
                or not math.isclose(math.fsum(row['loss'] for row in part['case_metrics']),
                                    part['loss_sum'], rel_tol=1e-6, abs_tol=1e-8)):
            raise ValueError('Gradient shard parameters or baseline loss differ')
    gradients = []
    for index, original in enumerate(originals):
        if original.dtype != torch.float32 or not bool(torch.isfinite(original).all()):
            raise ValueError('Finite FP32 original parameter required')
        live = [part['gradient_sum'][index] for part in parts
                if part['gradient_sum'][index] is not None]
        if any(value.shape != original.shape or value.dtype != torch.float32
               or not bool(torch.isfinite(value).all()) for value in live):
            raise ValueError('Invalid gradient tensor')
        gradient = sum(live) / len(expected_ids) if live else None
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise ValueError('Full-cohort mean gradient is non-finite')
        gradients.append(gradient)
    baseline = math.fsum(part['loss_sum'] for part in parts) / len(expected_ids)
    return keys, originals, gradients, baseline


def project_candidate(torch, keys, originals, delta, fraction, bounds):
    result, actual = {'ratios': {}, 'log2_scales': {}}, []
    for key, original, step in zip(keys, originals, delta):
        kind, name = key.split('::', 1)
        if kind not in bounds:
            raise ValueError('Unrecognized parameter kind')
        value = (original + fraction * step).clamp(*bounds[kind])
        result['ratios' if kind == 'ratio' else 'log2_scales'][name] = value
        actual.append(value - original)
    if len(actual) != len(keys) or len(delta) != len(keys):
        raise ValueError('Candidate delta coverage differs')
    return result, actual


def direction(torch, p, config, root, unit):
    from aggregate import completed
    kind = unit.removeprefix('c80_direction_')
    objective = config['objective_keys'][kind]
    parts, source_jobs = [], {}
    for shard in (0, 1):
        source_unit = f'c80_grad_{kind}_{shard}'
        summary = completed(root, source_unit)
        if (summary.get('gate_protocol') != config['protocol']
                or summary.get('objective') != objective
                or summary.get('completed_cases') != 40
                or not summary.get('native_parity_all_zero')
                or not summary.get('original_parameters_restored')
                or not summary.get('frozen_reader_and_model_unchanged')):
            raise ValueError('Complete frozen 40-case gradient receipt required')
        part = torch.load(root / 'results' / source_unit / 'gradient.pt',
                          map_location='cpu', weights_only=True)
        expected = {'protocol': config['protocol'], 'parent_protocol': p['protocol'],
                    'unit': source_unit, 'objective': objective, 'shard': shard,
                    'pbs_jobid': summary['pbs_jobid'], 'config': config, 'complete': True}
        if any(part.get(key) != value for key, value in expected.items()):
            raise ValueError('Gradient tensor/receipt binding differs')
        parts.append(part)
        source_jobs[source_unit] = summary['pbs_jobid']
    keys, originals, gradients, baseline = merge_shards(torch, parts, config['case_ids'])
    ratio_count = sum(key.startswith('ratio::') for key in keys)
    if len(keys) != 1228 or ratio_count != 614 or any(
            not key.startswith('channel::') for key in keys[614:]):
        raise ValueError('Original614 ratio +614 channel parameter order required')
    if any(not any(g is not None and bool(g.ne(0).any()) for g in group)
           for group in (gradients[:614], gradients[614:])):
        raise ValueError('Both D/r groups require a nonzero finite full-C80 gradient')
    norm = sum(g.square().sum() for g in gradients if g is not None).sqrt()
    factor = min(1.0, config['gradient_norm_clip'] / (float(norm) + 1e-6))
    groups = [{'params': originals[:614], 'lr': config['ratio_learning_rate']},
              {'params': originals[614:], 'lr': config['channel_learning_rate']}]
    if [group['lr'] for group in groups] != [
            p['source_closed_protocol']['training']['clip_learning_rate'],
            p['source_closed_protocol']['scaling']['learning_rate']]:
        raise ValueError('Original Adam learning rates differ')
    spec = importlib.util.spec_from_file_location('c80_direction_helpers', config['direction_helper'])
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    delta = helpers.adam_delta(torch, [None if g is None else g * factor for g in gradients], groups)
    clip, scale = p['source_closed_protocol']['clipping'], p['source_closed_protocol']['scaling']
    bounds = {'ratio': (clip['minimum_ratio'], clip['maximum_ratio']),
              'channel': (scale['minimum_log2'], scale['maximum_log2'])}
    out = root / 'results' / unit
    candidates = []
    for fraction in config['fractions']:
        values, actual = project_candidate(torch, keys, originals, delta, fraction, bounds)
        path = out / ('candidate_' + str(fraction).replace('.', 'p') + '.pt')
        tensor_save(torch, path, {
            'protocol': config['protocol'], 'parent_protocol': p['protocol'],
            'direction_unit': unit, 'direction_pbs_jobid': os.environ['PBS_JOBID'],
            'objective': objective, 'fraction': fraction, 'case_ids': config['case_ids'],
            **values})
        candidates.append({
            'fraction': fraction, 'path': str(path),
            'surrogate_g_dot_delta': math.fsum(float((g * step).sum())
                                              for g, step in zip(gradients, actual) if g is not None),
            'projected_step_l2': float(sum(step.square().sum() for step in actual).sqrt())})
    tensor_save(torch, out / 'full_gradient.pt', {
        'protocol': config['protocol'], 'parent_protocol': p['protocol'],
        'objective': objective, 'case_ids': config['case_ids'],
        'parameter_keys': keys, 'originals': originals, 'mean_gradient': gradients,
        'source_gradient_jobs': source_jobs})
    return {'objective': objective, 'case_ids': config['case_ids'], 'baseline_loss': baseline,
            'parameter_keys': keys, 'source_gradient_jobs': source_jobs, 'candidates': candidates,
            'surrogate_gradient_norm': float(norm), 'gradient_clip_factor': factor,
            'candidate_parameter_proposals': len(candidates), 'training_updates': 0,
            'saved_trained_checkpoints': 0, 'heldout_inputs_used': False,
            'sensor_evaluation_inputs_used': False}


def aggregate(p, config, root):
    from aggregate import completed
    result, jobs = {}, {}
    for kind, objective in config['objective_keys'].items():
        unit = f'c80_direction_{kind}'
        source = completed(root, unit)
        if source.get('gate_protocol') != config['protocol'] or source.get('case_ids') != config['case_ids']:
            raise ValueError('Full-C80 direction binding differs')
        jobs[unit] = source['pbs_jobid']
        evaluations = []
        for shard in (0, 1):
            eval_unit = f'c80_eval_{kind}_{shard}'
            receipt = completed(root, eval_unit)
            if (receipt.get('gate_protocol') != config['protocol']
                    or receipt.get('objective') != objective or receipt.get('completed_cases') != 40
                    or receipt.get('direction_unit') != unit
                    or receipt.get('direction_pbs_jobid') != source['pbs_jobid']
                    or not receipt.get('native_and_reader_initial_noise_parity_valid')
                    or not receipt.get('original_parameters_restored')):
                raise ValueError('Complete40-case finite evaluation required')
            evaluations.append(receipt)
            jobs[eval_unit] = receipt['pbs_jobid']
        values = []
        for proposal in source['candidates']:
            fraction = proposal['fraction']
            rows = [next(row for row in receipt['candidates'] if row['fraction'] == fraction)
                    for receipt in evaluations]
            ids = [case_id for row in rows for case_id in row['case_ids']]
            if sorted(ids) != config['case_ids'] or len(set(ids)) != 80 or any(row['cases'] != 40 for row in rows):
                raise ValueError('Each candidate must cover disjoint all80 cases')
            value = math.fsum(row['loss_sum'] for row in rows) / 80
            if not math.isfinite(value):
                raise ValueError('Non-finite finite-step objective')
            change = value - source['baseline_loss']
            values.append({**proposal, 'cases': 80, 'actual_loss': value,
                           'actual_loss_delta': change, 'actual_descent': change < 0})
        result[objective] = {'baseline_loss': source['baseline_loss'], 'cases': 80,
                             'candidates': values,
                             'any_actual_descent': any(row['actual_descent'] for row in values)}
    return {'objectives': result, 'source_jobs': jobs, 'full_c80_finite_step_evaluated': True,
            'training_updates': 0, 'heldout_inputs_used': False,
            'sensor_evaluation_inputs_used': False,
            'closedloop_improvement_proven': False,
            'scope': 'isolated calibration objectives at one fixed checkpoint; full80 gradient and same80 finite evaluation; no training or heldout tuning'}


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True,
                        choices=('c80_direction_response', 'c80_direction_latent',
                                 'aggregate_c80_parameter_gate'))
    args = parser.parse_args()
    import torch
    p = read(Path(__file__).with_name('protocol.json'))
    config = read(Path(__file__).with_name('c80_gate_protocol.json'))
    if config['parent_protocol'] != p['protocol'] or config['case_ids'] != list(range(2000, 2080)):
        raise ValueError('Frozen all80 protocol differs')
    root = Path(p['remote_root'])
    state = {'protocol': p['protocol'], 'gate_protocol': config['protocol'],
             'unit': args.unit, 'pbs_jobid': os.environ['PBS_JOBID'], 'complete': False}
    path = root / 'results' / args.unit / 'summary.json'
    save(path, state)
    try:
        result = (direction(torch, p, config, root, args.unit)
                  if args.unit.startswith('c80_direction_') else aggregate(p, config, root))
        save(path, {**state, **result, 'complete': True})
    except Exception as exc:
        save(path, {**state, 'failure': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()

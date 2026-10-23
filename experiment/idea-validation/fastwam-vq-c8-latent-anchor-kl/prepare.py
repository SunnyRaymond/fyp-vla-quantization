"""C8-only KL preparation and reuse of the trained rho1.5 control records."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
from anchor_selfcheck import require_allocation


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def queries(p, split):
    return [(c, s) for c in p['calibration' if split == 'calibration' else 'evaluation']['case_ids']
            for s in ([0] if split == 'calibration' else p['evaluation']['seed_indices'])]


def load_reference(torch, p):
    import latent_reference
    import train_c8_reference as original
    if read(Path(p['baseline_root']) / 'protocol.json') != p['baseline_protocol']:
        raise RuntimeError('Original C8 protocol changed')
    cache, saved_rows, c = original._load_reference(
        torch, Path(p['baseline_root']) / 'results/reference',
        original._binding(p['baseline_protocol']), p['calibration']['case_ids'], latent_reference)
    plan = read(p['calibration']['plan_file'])
    rows = {r['case_id']: r for r in plan['inputs']}
    if (plan['status'] != 'complete' or plan['calibration_cohorts']['c8'] != p['calibration']['case_ids']
            or any(rows[case] != row['plan_row'] for case, row in saved_rows.items())
            or not math.isclose(c, p['training']['normalization_c'], rel_tol=1e-12)):
        raise RuntimeError('Frozen C8 input/normalization binding differs')
    return cache, saved_rows, rows


def reuse_control(p, rows):
    source_root = Path(p['baseline_cache_root'])
    source = read(source_root / 'protocol.json')
    parent = source['source_clipping_protocol']
    if (source != p['baseline_cache_protocol'] or source['baseline_root'] != p['baseline_root']
            or parent['parent_root'] != p['baseline_root'] or parent['parent_arm'] != 'rho1.5'
            or parent['parent_updates'] != 8):
        raise RuntimeError('Control cache must come from the original trained C8 rho1.5 update8')
    trained = read(Path(p['baseline_root']) / 'results/training/rho1.5/training_receipt.json')
    if not trained['complete'] or trained['updates'] != 8 or trained['normalization_c'] != p['training']['normalization_c']:
        raise RuntimeError('Original eight-update control receipt differs')
    count = 0
    for split in ('calibration', 'heldout'):
        for case, seed in queries(p, split):
            path = source_root / 'results' / split / 'baseline/queries' / f'{case}_{seed}.json'
            record = read(path)
            expected = {'protocol': source['protocol'], 'arm': 'baseline', 'split': split, 'updates': 0,
                        'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                        'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed]}
            if any(record.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Reused control input/checkpoint binding differs: ' + str(path))
            if any(not math.isfinite(record[m]) or record[m] < 0 for m in p['evaluation']['metrics']):
                raise RuntimeError('Invalid control metrics')
            save(Path(p['remote_root']) / 'results/baseline' / split / 'queries' / path.name,
                 {**record, 'new_protocol': p['protocol'], 'source_path': str(path)})
            count += 1
    return count


def main():
    require_allocation()
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True)
    p = read(parser.parse_args().protocol)
    import torch
    from anchor_loss import latent_tokens, neighbor_mask, distances, teacher_anchor_kl, loss_terms
    torch.set_num_threads(4)
    check = read(Path(os.environ['ARTIFACTS']) / 'anchor_selfcheck.json')
    if not check['passed'] or check['pbs_jobid'] != os.environ['PBS_JOBID']:
        raise RuntimeError('Same-allocation KL selfcheck required')
    cache, _, rows = load_reference(torch, p)
    ids = p['calibration']['case_ids']
    future = torch.cat([cache['teacher'][str(c)]['future'].float()[:, :, 1:] for c in ids])
    stats = {'latent': {'mean': future.mean((0, 2, 3, 4)).tolist(),
                        'scale': future.std((0, 2, 3, 4), correction=0).clamp_min(p['anchor']['std_floor']).tolist()}}
    positive = []
    for case in ids:
        tokens, identity = latent_tokens(torch, cache['teacher'][str(case)]['future'], stats, p['anchor'])
        distance = distances(tokens, tokens)[:, neighbor_mask(torch, identity)]
        positive.append(distance[distance > 0])
    temperature = float(torch.cat(positive).median())
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError('C8 teacher neighbor distances must be nondegenerate')
    initial, diagnostics, corruptions = [], [], []
    for case in ids:
        target, parent = cache['teacher'][str(case)], cache['parent'][str(case)]
        values = loss_terms(torch, parent['action'], parent['future'], target, stats, p['anchor'], temperature)
        initial.append({'case_id': case, **{k: float(v) for k, v in values.items()}})
        z = target['future'].float()
        same, info = teacher_anchor_kl(torch, z, z, stats, p['anchor'], temperature, diagnostics=True)
        shifted = z.clone(); offset = 0
        for width in p['anchor']['view_widths']:
            shifted[:, :, 1:, :, offset:offset + width] = torch.roll(z[:, :, 1:, :, offset:offset + width], 2, -2)
            offset += width
        swapped = z.clone(); swapped[:, :, 1:] = z[:, :, 1:].flip(2)
        errors = {name: float(teacher_anchor_kl(torch, value, z, stats, p['anchor'], temperature))
                  for name, value in [('same', z), ('spatial_roll', shifted), ('time_swap', swapped)]}
        if float(same) > 1e-7:
            raise RuntimeError('Teacher identity KL must be zero')
        diagnostics.append(info); corruptions.append({'case_id': case, **errors})
    means = {key: statistics.mean(r[key] for r in initial) for key in ('action_mse', 'latent_nmse', 'latent_anchor_kl')}
    if any(not math.isfinite(v) or v <= 0 for v in means.values()):
        raise RuntimeError('C8 initial calibration losses must be nondegenerate')
    corruption_means = {key: statistics.mean(r[key] for r in corruptions) for key in ('same', 'spatial_roll', 'time_swap')}
    if min(corruption_means['spatial_roll'], corruption_means['time_swap']) <= 1e-7:
        raise RuntimeError('Real latent structural corruptions must be detectable')
    beta = p['anchor']['coefficient_fraction'] * p['training']['rho'] * p['training']['normalization_c'] * means['latent_nmse'] / means['latent_anchor_kl']
    count = reuse_control(p, rows)
    teacher_receipt = read(Path(p['teacher_heldout_directory']) / 'receipt.json')
    if (not teacher_receipt['complete'] or teacher_receipt['queries'] != 80
            or teacher_receipt['case_ids'] != p['evaluation']['case_ids'] or teacher_receipt['seed_indices'] != [0, 1]):
        raise RuntimeError('Reusable heldout teacher coverage differs')
    result = {'protocol': p['protocol'], 'complete': True, 'passed': True, 'pbs_jobid': os.environ['PBS_JOBID'],
              'heldout_used': False, 'calibration_case_ids': ids, 'temperature': temperature, 'beta': beta,
              'stats': stats, 'normalization_c': p['training']['normalization_c'], 'initial_means': means,
              'initial_rows': initial, 'corruption_means': corruption_means,
              'mean_normalized_teacher_entropy': statistics.mean(r['normalized_teacher_entropy'] for r in diagnostics),
              'baseline_records': count, 'baseline_checkpoint_updates': 8,
              'baseline_cache_additional_updates': 0, 'baseline_new_model_queries': 0,
              'model_forwards': 0, 'new_teacher_queries': 0, 'new_closedloop_episodes': 0,
              'reference_binding': cache['binding'], 'coefficient_frozen': True}
    save(Path(p['remote_root']) / 'results/prepare/summary.json', result)
    print('C8_ANCHOR_PREPARE_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

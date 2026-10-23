"""CPU-only analysis of existing query records and small VQ master checkpoints."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import statistics


def read(path):
    return json.loads(path.read_text())


def paired(records, left, right):
    result = {}
    for metric in ('first10motor_rmse', 'future_latent_nmse', 'late22motor_rmse'):
        rows = []
        for case in range(3000, 3040):
            a = statistics.mean(records[left][case, seed][metric] for seed in (0, 1))
            b = statistics.mean(records[right][case, seed][metric] for seed in (0, 1))
            rows.append({'case': case, 'suite': records[left][case, 0]['suite'],
                         'difference': a - b})
        differences = [r['difference'] for r in rows]
        result[metric] = {'lower_cases': sum(x < 0 for x in differences),
                          'higher_cases': sum(x > 0 for x in differences),
                          'equal_cases': sum(x == 0 for x in differences),
                          'mean_difference': statistics.mean(differences),
                          'median_difference': statistics.median(differences),
                          'largest_increases': sorted(rows, key=lambda r: r['difference'], reverse=True)[:5],
                          'largest_decreases': sorted(rows, key=lambda r: r['difference'])[:5]}
    result['both_primary_lower_cases'] = sum(all(
        statistics.mean(records[left][case, seed][metric] for seed in (0, 1)) <
        statistics.mean(records[right][case, seed][metric] for seed in (0, 1))
        for metric in ('first10motor_rmse', 'future_latent_nmse')) for case in range(3000, 3040))
    return result


def main():
    host = socket.gethostname().split('.')[0]
    assert os.environ.get('PBS_JOBID') and 'login' not in host
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().splitlines()
    assert host in {n.split('.')[0] for n in nodes}
    import torch
    torch.set_num_threads(2)
    root = Path(os.environ['NEW_ROOT'])
    protocol = read(root / 'protocol.json')
    comparison = read(root / 'results/comparison.json')
    records = {}
    for arm in protocol['arms']:
        rows = [read(root / 'results' / arm / 'queries' / f'{case}_{seed}.json')
                for case in protocol['evaluation']['case_ids'] for seed in (0, 1)]
        assert len(rows) == 80 and all(r['arm'] == arm for r in rows)
        records[arm] = {(r['case_id'], r['seed_index']): r for r in rows}
        assert len(records[arm]) == 80
        for case, seed in records[arm]:
            assert records[arm][case, seed]['sampler_seed'] == records['baseline'][case, seed]['sampler_seed']
    contrasts = {f'{a}_minus_{b}': paired(records, a, b) for a, b in (
        ('joint', 'baseline'), ('joint', 'weight_only'), ('clip_only', 'baseline'), ('weight_only', 'baseline'))}
    suite_changes = {}
    for suite in comparison['arms']['baseline']['by_suite']:
        baseline = comparison['arms']['baseline']['by_suite'][suite]
        suite_changes[suite] = {arm: {metric: 100 * (value / baseline[metric] - 1)
                                     for metric, value in comparison['arms'][arm]['by_suite'][suite].items()}
                                for arm in ('weight_only', 'clip_only', 'joint')}
    parent = torch.load(Path(protocol['parent_root']) / 'results/training/rho1.5/deployment.pt',
                        weights_only=True, map_location='cpu')['module_masters']
    assert len(parent) == 614
    payloads = {arm: torch.load(root / 'results' / arm / 'deployment.pt',
                               weights_only=True, map_location='cpu')
                for arm in ('weight_only', 'clip_only', 'joint')}
    stats = {}
    for arm, payload in payloads.items():
        masters = payload['module_masters']
        assert set(masters) == set(parent) == set(payload['ratios'])
        groups = {}
        for name, original in parent.items():
            group = name.split('.')[0]
            row = groups.setdefault(group, {'modules': 0, 'book_numel': 0, 'book_base_sq': 0.,
                'book_delta_sq': 0., 'book_bf16_base_sq': 0., 'book_bf16_delta_sq': 0.,
                'book_bf16_changed': 0, 'gain_numel': 0, 'log_gain_delta_sq': 0.,
                'gain_bf16_changed': 0, 'book_update_dot_weight_only': 0.,
                'weight_only_book_delta_sq': 0., 'ratios': []})
            a, b = masters[name]['books'].float(), original['books'].float()
            d = a - b
            dw = payloads['weight_only']['module_masters'][name]['books'].float() - b
            ab, bb = a.bfloat16().float(), b.bfloat16().float()
            ag, bg = masters[name]['log_scales'].float(), original['log_scales'].float()
            row['modules'] += 1
            row['book_numel'] += a.numel()
            row['book_base_sq'] += float(b.square().sum())
            row['book_delta_sq'] += float(d.square().sum())
            row['book_bf16_base_sq'] += float(bb.square().sum())
            row['book_bf16_delta_sq'] += float((ab-bb).square().sum())
            row['book_bf16_changed'] += int((ab != bb).sum())
            row['gain_numel'] += ag.numel()
            row['log_gain_delta_sq'] += float((ag-bg).square().sum())
            row['gain_bf16_changed'] += int((ag.exp().bfloat16() != bg.exp().bfloat16()).sum())
            row['book_update_dot_weight_only'] += float((d*dw).sum())
            row['weight_only_book_delta_sq'] += float(dw.square().sum())
            row['ratios'].append(float(payload['ratios'][name]))
        stats[arm] = {}
        for group, row in groups.items():
            ratios = row.pop('ratios')
            stats[arm][group] = {
                'modules':row['modules'], 'book_parameters':row['book_numel'],
                'row_gain_parameters':row['gain_numel'],
                'book_master_relative_l2_change':math.sqrt(row['book_delta_sq']/row['book_base_sq']),
                'book_bf16_relative_l2_change':math.sqrt(row['book_bf16_delta_sq']/row['book_bf16_base_sq']),
                'book_bf16_changed_fraction':row['book_bf16_changed']/row['book_numel'],
                'log_gain_change_rms':math.sqrt(row['log_gain_delta_sq']/row['gain_numel']),
                'gain_bf16_changed_fraction':row['gain_bf16_changed']/row['gain_numel'],
                'book_update_cosine_weight_only':row['book_update_dot_weight_only']/max(
                    math.sqrt(row['book_delta_sq']*row['weight_only_book_delta_sq']),1e-30),
                'clip_min':min(ratios), 'clip_mean':statistics.mean(ratios),
                'clip_below_one':sum(x < 1 for x in ratios),
                'clip_below_point98':sum(x < .98 for x in ratios)}
    receipts = {arm:read(root/'results'/arm/'training_receipt.json') for arm in payloads}
    norms = {arm:{'min':min(r['gradient_norm'] for r in t['history']),
                  'max':max(r['gradient_norm'] for r in t['history'])} for arm,t in receipts.items()}
    top_clips = {arm:sorted(({'module':name, 'ratio':float(r)} for name,r in payload['ratios'].items()),
                           key=lambda r:r['ratio'])[:12] for arm,payload in payloads.items() if arm != 'weight_only'}
    result = {'complete':True, 'pbs_jobid':os.environ['PBS_JOBID'], 'new_model_forwards':0,
              'cases_per_arm':40, 'seeds_per_case':2, 'paired_descriptives':contrasts,
              'suite_percent_change_vs_baseline':suite_changes, 'parameter_changes':stats,
              'module_prefix_counts':dict(Counter(n.split('.')[0] for n in parent)),
              'module_examples':list(parent)[:12], 'most_clipped_modules':top_clips,
              'gradient_norms':norms,
              'limits':['case-level descriptives after averaging paired seeds; no hypothesis test',
                        'parameter drift is not decoded weight or activation/output error',
                        'training histories are pre-update on different contexts, not endpoint C8 evaluations']}
    target = root/'results/diagnostic.json'
    assert not target.exists(), 'Preserve an existing diagnostic result'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],'new_model_forwards':0,
                      'prefixes':result['module_prefix_counts']},indent=2),flush=True)


if __name__ == '__main__':
    main()

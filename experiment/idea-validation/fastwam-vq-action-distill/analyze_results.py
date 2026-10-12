"""Descriptive paired recovery analysis; numeric work runs only in PBS."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def compare(before, after):
    differences = [b - a for a, b in zip(before, after)]
    return {'case_count': len(before), 'before_mean': statistics.mean(before),
            'after_mean': statistics.mean(after), 'mean_difference': statistics.mean(differences),
            'relative_mean_change': statistics.mean(after) / statistics.mean(before) - 1,
            'before_median': statistics.median(before), 'after_median': statistics.median(after),
            'median_paired_difference': statistics.median(differences),
            'improved_cases': sum(d < 0 for d in differences),
            'worsened_cases': sum(d > 0 for d in differences), 'unchanged_cases': sum(d == 0 for d in differences),
            'paired_difference_sd': statistics.stdev(differences) if len(before) > 1 else None}


def rmse(action, teacher, first, last, channels):
    values = [(action[t][c] - teacher[t][c]) ** 2 for t in range(first, last) for c in channels]
    return math.sqrt(statistics.mean(values))


def run(args):
    from sampler import _require_compute_allocation
    _require_compute_allocation()
    import torch

    root = args.protocol.parent
    p = json.loads(args.protocol.read_text())
    result = root / 'results' / ('full-' + args.job)
    s = json.loads((result / 'summary.json').read_text())
    assert s['status'] == 'complete' and s['job_id'] == args.job
    assert s['updates'] == 16 and s['completed_epochs'] == 2 and s['test']['context_count'] == 44
    args.out.mkdir()
    payload = torch.load(result / 'checkpoint.pt', map_location='cpu', weights_only=True)
    state = payload['run_state']
    train = state['training_history']
    evaluations = state['evaluation_rows']
    totals = {'book_scalars': sum(value['books'].numel() for value in payload['best_state'].values()),
              'row_scales': sum(value['log_scales'].numel() for value in payload['best_state'].values())}
    del payload
    gc.collect()
    cached, teachers = {}, {}
    source = Path(p['source_root'])
    for line in (source / 'results/full/actions.jsonl').open():
        row = json.loads(line)
        if row['arm'] == p['initial_arm']:
            cached[row['case_id'], row['seed_index']] = row
        elif row['arm'] == 'bf16':
            teachers[row['case_id'], row['seed_index']] = row
    metadata = {row['case_id']: row for row in json.loads((source / 'plan.json').read_text())['inputs']}
    metrics = ['motor_rmse_first10_vs_bf16', 'gripper_rmse_first10_vs_bf16', 'motor_rmse_32_vs_bf16']
    summary_keys = ['mean_motor_rmse_first10', 'mean_gripper_rmse_first10', 'mean_motor_rmse_32']

    def records(split, epoch):
        return [row for row in evaluations if row['split'] == split and row['epoch'] == epoch]

    def per_case(rows, metric):
        cases = sorted({row['case_id'] for row in rows})
        return {case: statistics.mean(row[metric] for row in rows if row['case_id'] == case) for case in cases}

    test_rows = records('test', s['selected_epoch'])
    assert len(test_rows) == 44 and len({(r['case_id'], r['seed_index']) for r in test_rows}) == 44
    initial_rows = [cached[row['case_id'], row['seed_index']] for row in test_rows]
    for before, after in zip(initial_rows, test_rows):
        teacher = teachers[after['case_id'], after['seed_index']]
        assert before['sampler_seed'] == teacher['sampler_seed']
        for row in (before, after):
            assert abs(rmse(row['action'], teacher['action'], 0, 10, range(6)) - row[metrics[0]]) < 1e-6
            row['motor_rmse_late22_vs_bf16'] = rmse(row['action'], teacher['action'], 10, 32, range(6))
    before_cases, after_cases = {}, {}
    comparisons = {}
    for metric in metrics + ['motor_rmse_late22_vs_bf16']:
        before_cases[metric], after_cases[metric] = per_case(initial_rows, metric), per_case(test_rows, metric)
        cases = sorted(before_cases[metric])
        comparisons[metric] = compare([before_cases[metric][c] for c in cases], [after_cases[metric][c] for c in cases])
    for metric, key in zip(metrics, summary_keys):
        assert abs(comparisons[metric]['before_mean'] - s['cached_initial_test'][key]) < 1e-8
        assert abs(comparisons[metric]['after_mean'] - s['test'][key]) < 1e-8

    calibration = records('calibration', s['selected_epoch'])
    calibration_initial = [cached[row['case_id'], row['seed_index']] for row in calibration]
    calibration_comparison = {metric: compare([row[metric] for row in calibration_initial],
                                             [row[metric] for row in calibration]) for metric in metrics}
    case_rows = []
    for case in p['test_cases']:
        meta = metadata[case]
        row = {key: meta.get(key) for key in ('domain', 'suite', 'dimension', 'variant', 'original_task', 'state_id')}
        row['case_id'] = case
        for metric in metrics + ['motor_rmse_late22_vs_bf16']:
            before, after = before_cases[metric][case], after_cases[metric][case]
            row.update({metric + '_before': before, metric + '_after': after,
                        metric + '_delta': after - before, metric + '_relative': after / before - 1})
        case_rows.append(row)
    with (args.out / 'paired_cases.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    split_composition = {}
    for split, cases in (('selection', p['selection_cases']), ('test', p['test_cases'])):
        split_composition[split] = {}
        for case in cases:
            m = metadata[case]
            key = str(m['domain']) + '/' + str(m['suite'])
            split_composition[split][key] = split_composition[split].get(key, 0) + 1
    subgroups = {}
    for domain in sorted({row['domain'] for row in case_rows}):
        group = [row for row in case_rows if row['domain'] == domain]
        subgroups[domain] = compare([r[metrics[0] + '_before'] for r in group],
                                   [r[metrics[0] + '_after'] for r in group])
    ranked = sorted(case_rows, key=lambda row: row[metrics[0] + '_delta'])
    training = {'history_count': len(train), 'connected_min': min(row['connected'] for row in train),
                'finite_min': min(row['finite'] for row in train), 'nonzero_min': min(row['nonzero'] for row in train),
                'gradient_norm_max': max(row['gradient_norm_before_clip'] for row in train),
                'gradient_clip': p['gradient_norm_clip'],
                'epoch_preupdate_mse': {str(epoch): statistics.mean(row['actual_preupdate_motor_mse'] for row in train if row['epoch'] == epoch)
                                        for epoch in (1, 2)},
                'per_update': train,
                'interpretation': 'Online pre-update losses; these are not common-checkpoint calibration evaluations.'}
    report = {'status': 'complete', 'analysis_job': os.environ['PBS_JOBID'], 'experiment_job': args.job,
              'cohort': {'cases': 22, 'seeds_per_case': 2, 'contexts': 44},
              'estimator': 'Mean within each case over two seed RMSEs, then equal-weight mean across cases; positive delta is worse.',
              'statistical_scope': 'Descriptive paired recovery on a previously seen cohort; no p-value, independent generalization or closed-loop claim.',
              'test': comparisons, 'calibration': calibration_comparison, 'selection_history': s['selection_history'],
              'split_composition': split_composition, 'exploratory_domain_subgroups': subgroups,
              'worst_degradations': list(reversed(ranked[-5:])), 'largest_improvements': ranked[:5],
              'case_rows': case_rows, 'training': training, 'parameter_totals': totals,
              'changed_stored_book_scalars': s['changed_stored_book_scalars'],
              'changed_stored_row_scales': s['changed_stored_row_scales'],
              'teacher_inference_queries': s['teacher_inference_queries'],
              'optimizer_reset_after_legacy_step': s['optimizer_reset_after_legacy_step'],
              'selected_epoch': s['selected_epoch'], 'test_queries_before_freeze': s['test_queries_before_freeze']}
    save(args.out / 'analysis.json', report)
    save(args.out / 'summary.json', {key: value for key, value in report.items() if key not in ('training', 'case_rows', 'selection_history')})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        a = [r[metrics[0] + '_before'] for r in case_rows]
        b = [r[metrics[0] + '_after'] for r in case_rows]
        axes[0].scatter(a, b, s=24, color='#276fbf')
        limit = max(a + b) * 1.08
        axes[0].plot([0, limit], [0, limit], color='#888888', linestyle='--')
        axes[0].set(xlabel='Initial VQ motor RMSE (first 10)', ylabel='Distilled motor RMSE (first 10)',
                    title='Paired test cases; below line is better', xlim=(0, limit), ylim=(0, limit))
        for r in list(reversed(ranked[-2:])):
            axes[0].annotate(str(r['case_id']), (r[metrics[0] + '_before'], r[metrics[0] + '_after']), fontsize=8)
        deltas = [r[metrics[0] + '_delta'] for r in case_rows]
        axes[1].bar(range(22), deltas, color=['#d95f59' if v > 0 else '#279d85' for v in deltas])
        axes[1].axhline(0, color='#888888', linewidth=0.7)
        axes[1].set_xticks(range(22), [str(r['case_id']) for r in case_rows], rotation=90, fontsize=7)
        axes[1].set(title='Paired change; positive is worse', ylabel='Change in first-10 motor RMSE')
        history = s['selection_history']
        axes[2].plot([r['epoch'] for r in history], [r['mean_motor_rmse_first10'] for r in history], marker='o', color='#276fbf')
        axes[2].set(title='Selection-only checkpoint choice', xlabel='Epoch (0 = initialization)', ylabel='Selection motor RMSE', xticks=[0, 1, 2])
        fig.savefig(args.out / 'paired_recovery.png', dpi=130)
        plt.close(fig)
        print('PLOT_SAVED', flush=True)
    except ImportError:
        print('MATPLOTLIB_UNAVAILABLE; JSON and CSV analysis still complete', flush=True)
    (args.out / 'ANALYSIS_COMPLETE').touch()
    print('ANALYSIS_COMPLETE ' + json.dumps(report['test']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())

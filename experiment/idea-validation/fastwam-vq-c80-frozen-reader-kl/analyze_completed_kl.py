"""CPU-only cached endpoint analysis after the user cancelled the matched arm."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics as st
from anchor_selfcheck import require_allocation
from reuse_reference import read, save, queries
from aggregate import compare, METRICS

BASE_METRICS = METRICS[:5]


def per_case(records, metrics):
    groups = {}
    for row in records:
        groups.setdefault(row['case_id'], []).append(row)
    return {case: {'suite': rows[0]['suite'],
                   **{metric: st.mean(row[metric] for row in rows) for metric in metrics}}
            for case, rows in groups.items()}


def timing(rows, key):
    values = [float(row[key]) for row in rows]
    steady = values[1:] or values
    return {'records': len(values), 'per_update_seconds': values,
            'median_all_seconds': st.median(values) if values else None,
            'median_excluding_first_seconds': st.median(steady) if steady else None,
            'scope': 'descriptive recorded update durations; different timing boundaries and cases'}


def main(p):
    require_allocation()
    root = Path(p['remote_root'])
    worker = read(root / 'results/reader_kl/summary.json')
    if (not worker['complete'] or worker['protocol'] != p['protocol']
            or worker['updates'] != 80 or worker['evaluation_queries'] != 160
            or worker['reader_evaluation_queries'] != 160):
        raise ValueError('Complete final80 KL student and reader endpoints required')
    source = Path(p['baseline_cache_root'])
    if read(source / 'protocol.json') != p['baseline_cache_protocol']:
        raise ValueError('Historical C8 incumbent binding differs')
    splits, paired, final_readers, groups = {}, {}, {}, {}
    for split in ('calibration', 'heldout'):
        baseline_rows, candidate_rows = [], []
        for case, seed in queries(p, split):
            baseline = read(source / 'results' / split / 'baseline/queries' / f'{case}_{seed}.json')
            candidate = read(root / 'results/reader_kl' / split / 'queries' / f'{case}_{seed}.json')
            if (baseline['protocol'] != p['baseline_cache_protocol']['protocol']
                    or baseline['arm'] != 'baseline' or baseline['updates'] != 0
                    or candidate['protocol'] != p['protocol'] or candidate['arm'] != 'reader_kl'
                    or candidate['updates'] != 80 or baseline['plan_row'] != candidate['plan_row']
                    or baseline['sampler_seed'] != candidate['sampler_seed']
                    or baseline['case_id'] != case or candidate['case_id'] != case
                    or baseline['seed_index'] != seed or candidate['seed_index'] != seed):
                raise ValueError('Historical/candidate paired binding differs')
            for record, metrics in ((baseline, BASE_METRICS), (candidate, METRICS)):
                if any(not math.isfinite(record[m]) or record[m] < 0 for m in metrics):
                    raise ValueError('Nonfinite endpoint metric')
            baseline_rows.append(baseline)
            candidate_rows.append(candidate)
        b, c = per_case(baseline_rows, BASE_METRICS), per_case(candidate_rows, METRICS)
        ids = sorted(b)
        suites = sorted({row['suite'] for row in b.values()})
        expected_per_suite = 20 if split == 'calibration' else 10
        if len(suites) != 4 or any(sum(row['suite'] == suite for row in b.values()) != expected_per_suite for suite in suites):
            raise ValueError('Complete fixed four-suite cohort required')
        splits[split] = {
            'cases': len(ids), 'queries_per_arm': len(candidate_rows),
            'overall': {m: compare(b, c, ids, m) for m in BASE_METRICS},
            'by_suite': {suite: {m: compare(b, c, [case for case in ids if b[case]['suite'] == suite], m)
                                  for m in BASE_METRICS} for suite in suites},
            'by_seed': {str(seed): {m: {
                'historical_c8_az': st.mean(row[m] for row in baseline_rows if row['seed_index'] == seed),
                'c80_reader_kl': st.mean(row[m] for row in candidate_rows if row['seed_index'] == seed)}
                for m in BASE_METRICS} for seed in sorted({row['seed_index'] for row in candidate_rows})}}
        paired[split] = [{'case_id': case, 'suite': b[case]['suite'], 'baseline': b[case], 'candidate': c[case],
                          'delta': {m: c[case][m] - b[case][m] for m in BASE_METRICS}} for case in ids]
        final_readers[split] = {
            m: {'mean': st.mean(c[case][m] for case in ids),
                'median_case': st.median(c[case][m] for case in ids)} for m in METRICS[5:]}
        decomp = {'student_motor_mse': st.mean(row['first10motor_mse'] for row in candidate_rows),
                  'future_reader_effect_mse': st.mean(row['reader_first10motor_rmse'] ** 2 for row in candidate_rows),
                  'quantized_reader_gap_mse': st.mean(row['reader_gap_first10motor_rmse'] ** 2 for row in candidate_rows),
                  'cross_term': st.mean(row['reader_error_cross_term'] for row in candidate_rows)}
        if not math.isclose(decomp['student_motor_mse'], decomp['future_reader_effect_mse']
                            + decomp['quantized_reader_gap_mse'] + decomp['cross_term'], rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError('Cached reader error decomposition differs')
        final_readers[split]['decomposition'] = decomp
        groups[split] = {
            'nmse_better_motor_better': sum(r['delta']['future_latent_nmse'] < 0 and r['delta']['first10motor_rmse'] < 0 for r in paired[split]),
            'nmse_better_motor_worse': sum(r['delta']['future_latent_nmse'] < 0 and r['delta']['first10motor_rmse'] > 0 for r in paired[split])}
    c8_root = root.parent / 'fastwam-vq-c8-frozen-reader-kl-20261022'
    c8_heldout = [read(c8_root / 'results/reader_kl/heldout/queries' / f'{case}_{seed}.json')
                  for case, seed in queries(p, 'heldout')]
    candidate_lookup = {(row['case_id'], row['seed_index']): row for row in candidate_rows}
    for row in c8_heldout:
        other = candidate_lookup[(row['case_id'], row['seed_index'])]
        if row['plan_row'] != other['plan_row'] or row['sampler_seed'] != other['sampler_seed']:
            raise ValueError('Historical C8 reader-KL input/seed binding differs')
    c8_cases = per_case(c8_heldout, BASE_METRICS)
    c80_cases = {row['case_id']: row['candidate'] for row in paired['heldout']}
    c8_kl_comparison = {m: compare(c8_cases, c80_cases, sorted(c8_cases), m) for m in BASE_METRICS}
    initial = worker['preparation']['initial_means']
    reader_fit = {m: {'initial_raw_c80': initial[m], 'final_c80': final_readers['calibration'][m]['mean'],
                      'relative_change': final_readers['calibration'][m]['mean'] / initial[m] - 1}
                  for m in ('reader_kl', 'reader_mse')}
    import torch
    torch.set_num_threads(4)
    old_receipt = read(Path(p['baseline_root']) / 'results/training/rho1.5/training_receipt.json')
    old = torch.load(old_receipt['checkpoint'], map_location='cpu', weights_only=True)
    old_history = old['run_state']['training_history']
    del old
    az_path = root / 'results/az/checkpoint.pt'
    az = torch.load(az_path, map_location='cpu', weights_only=True) if az_path.is_file() else None
    timings = {'original_c8_az': timing(old_history, 'gradient_seconds'),
               'c80_reader_kl': timing(worker['training_receipt']['history'], 'seconds'),
               'cancelled_c80_az': timing(az['history'], 'seconds') if az else None,
               'cancelled_c80_az_committed_updates': az['updates'] if az else 0,
               'implementation': {'original_c8_az': 'one differentiable student forward/backward per update',
                                  'c80_az': 'one no-grad student forward plus differentiable student replay/VJP',
                                  'c80_kl': 'same staged student path plus frozen-reader backward and dense-weight parking'}}
    del az
    summary = {'protocol': p['protocol'], 'unit': 'cached_analysis', 'complete': True,
               'pbs_jobid': os.environ['PBS_JOBID'], 'baseline_records': 160, 'candidate_records': 160,
               'comparison_reference': 'historical C8 A+Z rho1.5 update8; NOT a matched C80 control',
               'matched_c80_control_completed': False, 'scope_changed_by_user': True,
               'splits': splits, 'final_reader_metrics': final_readers, 'reader_calibration_fit': reader_fit,
               'c80_vs_historical_c8_reader_kl': c8_kl_comparison, 'mechanism_counts': groups,
               'timings': timings, 'new_training_updates': 0, 'new_model_queries': 0,
               'new_teacher_queries': 0, 'new_closedloop_episodes': 0,
               'kl_training_queries': {'student_full': worker['new_model_queries'],
                                       'reader_only': worker['new_bf16_reader_queries']},
               'beta': worker['training_receipt']['beta'],
               'decision': 'stop_after_user_requested_cached_analysis_no_further_training',
               'limits': ['No matched C80 A+Z endpoint: cannot identify incremental KL benefit',
                          'C8/C80 changes coverage, updates and reader standardization together',
                          'Historical heldout is a development regression cohort, not independent generalization',
                          'No original C8 frozen-reader endpoints: heldout reader metrics are absolute only']}
    save(root / 'results/cached_analysis/summary.json', summary)
    save(root / 'results/cached_analysis/paired_cases.json', paired)
    save(root / 'results/comparison.json', summary)
    lines = ['# C80 frozen-reader KL：已有缓存分析', '',
             '用户取消同预算 C80 A+Z；以下比较参照是历史 C8 A+Z，不证明 KL 的独立收益。',
             '缓存分析新增训练、模型/teacher forward、闭环均为0。', '']
    for split, result in splits.items():
        lines += ['## ' + split, '', '| Metric | C8 A+Z | C80 KL | Change | W/L |', '|---|---:|---:|---:|---:|']
        for m, value in result['overall'].items():
            lines.append(f"| {m} | {value['baseline']:.9g} | {value['candidate']:.9g} | {100*value['relative_change']:+.3f}% | {value['wins']}/{value['losses']} |")
        lines.append('')
    lines += ['## Frozen-reader endpoints', '', '```json', json.dumps(final_readers, indent=2), '```',
              '', '## Initial to final calibration reader fit', '', '```json', json.dumps(reader_fit, indent=2), '```',
              '', '## Recorded update durations', '', '```json', json.dumps(timings, indent=2), '```',
              '', 'KL causal attribution remains unresolved because the matched C80 A+Z arm was cancelled.', '']
    (root / 'results/ANALYSIS.zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print('CACHED_C80_KL_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    main(read(parser.parse_args().protocol))

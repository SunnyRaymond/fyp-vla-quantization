"""Complete paired heldout aggregation and the single predeclared ablation gate."""
import argparse
import math
import os
from pathlib import Path
import statistics as st
from anchor_selfcheck import require_allocation
from reuse_reference import read, save, queries
from reader_loss import gate

METRICS = ('first10motor_rmse', 'full32motor_rmse', 'late22motor_rmse',
           'first10gripper_rmse', 'future_latent_nmse')


def compare(base, candidate, case_ids, metric):
    left = [base[case][metric] for case in case_ids]
    right = [candidate[case][metric] for case in case_ids]
    differences = [b - a for a, b in zip(left, right)]
    return {'baseline': st.mean(left), 'candidate': st.mean(right),
            'delta': st.mean(differences),
            'relative_change': st.mean(right) / st.mean(left) - 1 if st.mean(left) else None,
            'median_paired_delta': st.median(differences),
            'wins': sum(value < 0 for value in differences),
            'losses': sum(value > 0 for value in differences),
            'ties': sum(value == 0 for value in differences), 'cases': len(case_ids),
            'worst_case_delta': max(differences), 'best_case_delta': min(differences)}


def run(p, arm):
    require_allocation()
    root = Path(p['remote_root'])
    results = root / 'results'
    worker = read(results / arm / 'summary.json')
    receipt = worker['training_receipt']
    if (not worker['complete'] or worker['protocol'] != p['protocol'] or worker['updates'] != 8
            or worker['evaluation_queries'] != 88 or worker['new_teacher_queries'] != 0
            or worker['baseline_new_model_queries'] != 0 or worker['new_closedloop_episodes'] != 0
            or receipt['activation_clipping'] or receipt['per_channel_scaling']
            or receipt['update_order'] != p['calibration']['case_ids']):
        raise ValueError('The complete original recipe/evaluation contract is required')
    splits = {}
    paired_rows = {}
    for split in ('calibration', 'heldout'):
        records = {'baseline': {}, arm: {}}
        baseline_queries, candidate_queries = [], []
        for case, seed in queries(p, split):
            baseline = read(results / 'baseline' / split / 'queries' / f'{case}_{seed}.json')
            candidate = read(results / arm / split / 'queries' / f'{case}_{seed}.json')
            if (baseline['new_protocol'] != p['protocol'] or candidate['protocol'] != p['protocol']
                    or candidate['arm'] != arm or candidate['updates'] != 8
                    or baseline['plan_row'] != candidate['plan_row']
                    or baseline['sampler_seed'] != candidate['sampler_seed']
                    or baseline['seed_index'] != seed or candidate['seed_index'] != seed
                    or baseline['case_id'] != case or candidate['case_id'] != case
                    or baseline['suite'] != candidate['suite']):
                raise ValueError('Paired case/input/seed/checkpoint binding differs')
            for name, record in (('baseline', baseline), (arm, candidate)):
                if any(not math.isfinite(record[metric]) or record[metric] < 0 for metric in METRICS):
                    raise ValueError('Nonfinite or negative evaluation metric')
                records[name].setdefault(case, []).append(record)
            baseline_queries.append(baseline)
            candidate_queries.append(candidate)
        cases = {name: {case: {'suite': rows[0]['suite'],
                              **{metric: st.mean(row[metric] for row in rows) for metric in METRICS}}
                        for case, rows in values.items()} for name, values in records.items()}
        case_ids = sorted(cases['baseline'])
        suites = sorted({row['suite'] for row in cases['baseline'].values()})
        expected_count = 2 if split == 'calibration' else 10
        if len(suites) != 4 or any(sum(row['suite'] == suite for row in cases['baseline'].values()) != expected_count for suite in suites):
            raise ValueError('Full frozen four-suite cohort required')
        overall = {metric: compare(cases['baseline'], cases[arm], case_ids, metric) for metric in METRICS}
        by_suite = {suite: {metric: compare(cases['baseline'], cases[arm],
                             [case for case in case_ids if cases['baseline'][case]['suite'] == suite], metric)
                           for metric in METRICS} for suite in suites}
        seed_metrics = {}
        for seed in sorted({row['seed_index'] for row in candidate_queries}):
            seed_metrics[str(seed)] = {metric: {
                'baseline': st.mean(row[metric] for row in baseline_queries if row['seed_index'] == seed),
                'candidate': st.mean(row[metric] for row in candidate_queries if row['seed_index'] == seed)}
                for metric in METRICS}
        splits[split] = {'cases': len(case_ids), 'queries_per_arm': len(candidate_queries),
                         'overall': overall, 'by_suite': by_suite, 'by_seed': seed_metrics,
                         'aggregation': 'mean seeds within case, cases within suite, four equal-size suites equally'}
        paired_rows[split] = [{'case_id': case, 'suite': cases['baseline'][case]['suite'],
                               'baseline': cases['baseline'][case], 'candidate': cases[arm][case],
                               'delta': {metric: cases[arm][case][metric] - cases['baseline'][case][metric] for metric in METRICS}}
                              for case in case_ids]
    unit = 'aggregate' if arm == 'reader_kl' else 'aggregate_mse'
    advance = gate(splits['heldout']['overall'])
    summary = {'protocol': p['protocol'], 'unit': unit, 'arm': arm, 'complete': True,
               'pbs_jobid': os.environ['PBS_JOBID'], 'baseline_records': 88, 'candidate_records': 88,
               'splits': splits, 'advance_to_reader_mse_ablation': arm == 'reader_kl' and advance,
               'fixed_gate_passed': advance,
               'decision': ('run_one_matched_reader_mse_ablation' if advance else 'stop_this_recipe_no_parameter_sweep')
                           if arm == 'reader_kl' else 'matched_reader_mse_ablation_complete',
               'beta': receipt['beta'], 'initial_parity': receipt['initial_parity'],
               'actual_infer_parity': receipt['actual_infer_parity'],
               'new_model_queries': worker['new_model_queries'],
               'new_bf16_reader_queries': worker['new_bf16_reader_queries'],
               'parameter_reverse_passes': worker['parameter_reverse_passes'],
               'baseline_new_model_queries': 0, 'new_teacher_queries': 0, 'new_closedloop_episodes': 0,
               'scope': 'fixed-input decoded BF16 VQ/dynamic A4/BF16 KV; historical heldout is a development regression cohort; no independent generalization or closedloop claim'}
    if arm == 'reader_mse':
        kl = read(results / 'aggregate/summary.json')
        if not kl['advance_to_reader_mse_ablation']:
            raise ValueError('MSE ablation is forbidden after a failed KL gate')
        summary['kl_comparison'] = {metric: {'reader_kl': kl['splits']['heldout']['overall'][metric]['candidate'],
                                            'reader_mse': splits['heldout']['overall'][metric]['candidate']}
                                    for metric in METRICS}
    save(results / unit / 'summary.json', summary)
    save(results / unit / 'paired_cases.json', paired_rows)
    save(results / 'comparison.json', summary)
    lines = ['# C8 frozen-reader ' + arm + '：完整配对结果', '',
             '原 C8 A+Z、rho1.5、8 updates；原 baseline 88 queries 和 teacher 全部复用。',
             '只训练原 codebooks/log row gains，无 activation clip/per-channel；未运行闭环。', '',
             '主评测为实际量化 student，非 frozen-reader 自身得分。', '']
    for split, result in splits.items():
        lines += ['## ' + split, '', '| Metric | Baseline | Candidate | Delta | Paired median | W/L/T |',
                  '|---|---:|---:|---:|---:|---:|']
        for metric, value in result['overall'].items():
            lines.append(f"| {metric} | {value['baseline']:.9g} | {value['candidate']:.9g} | {value['delta']:+.9g} | {value['median_paired_delta']:+.9g} | {value['wins']}/{value['losses']}/{value['ties']} |")
        lines.append('')
    lines += ['冻结 gate：motor 均值下降、case 配对中位数下降、至少21/40改善，且 latent NMSE 均值不退步。',
              'Gate passed: ' + str(advance) + '；决定：' + summary['decision'], '',
              'beta=' + str(receipt['beta']) + '；仅用 C8 初始输入标定一次。',
              '不使用 heldout 选择 checkpoint、系数或训练顺序；两个 sampler seeds 不是独立训练重复。',
              '这是固定输入保真验证；不代表闭环、native INT4 或独立泛化收益。', '']
    destination = results / ('ANALYSIS.zh.md' if arm == 'reader_kl' else 'ABLATION.zh.md')
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print('FROZEN_READER_AGGREGATE_COMPLETE', summary['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--arm', choices=('reader_kl', 'reader_mse'), default='reader_kl')
    args = parser.parse_args()
    run(read(args.protocol), args.arm)

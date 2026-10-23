"""Matched C80 student and frozen-reader endpoint aggregation; no automatic followup."""
import argparse
import math
import os
from pathlib import Path
import statistics as st
from anchor_selfcheck import require_allocation
from reuse_reference import read, save, queries
from reader_loss import gate

METRICS = ('first10motor_rmse', 'full32motor_rmse', 'late22motor_rmse',
           'first10gripper_rmse', 'future_latent_nmse', 'reader_kl', 'reader_mse',
           'reader_first10motor_rmse', 'reader_gap_first10motor_rmse')


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


def run(p):
    require_allocation()
    arm = 'reader_kl'
    root = Path(p['remote_root'])
    results = root / 'results'
    workers = {name: read(results / name / 'summary.json') for name in ('az', 'reader_kl')}
    for name, worker in workers.items():
        receipt = worker['training_receipt']
        if (not worker['complete'] or worker['protocol'] != p['protocol'] or worker['updates'] != 80
                or worker['evaluation_queries'] != 160 or worker['reader_evaluation_queries'] != 160
                or worker['new_teacher_queries'] != 0 or worker['new_closedloop_episodes'] != 0
                or receipt['activation_clipping'] or receipt['per_channel_scaling']
                or receipt['update_order'] != p['calibration']['update_order']
                or any(value['max_abs'] != 0 for value in receipt['initial_parity'].values())
                or any(row['coverage']['connected'] != 614 or row['coverage']['finite'] != 614
                       for row in receipt['history'])
                or len(receipt['history']) != 80):
            raise ValueError('Both complete matched C80 recipes and all reader endpoints are required')
    worker = workers[arm]
    receipt = worker['training_receipt']
    splits = {}
    decompositions = {}
    paired_rows = {}
    for split in ('calibration', 'heldout'):
        records = {'baseline': {}, arm: {}}
        baseline_queries, candidate_queries = [], []
        for case, seed in queries(p, split):
            baseline = read(results / 'az' / split / 'queries' / f'{case}_{seed}.json')
            candidate = read(results / arm / split / 'queries' / f'{case}_{seed}.json')
            if (baseline['protocol'] != p['protocol'] or baseline['arm'] != 'az' or baseline['updates'] != 80 or candidate['protocol'] != p['protocol']
                    or candidate['arm'] != arm or candidate['updates'] != 80
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
        expected_count = 20 if split == 'calibration' else 10
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
        decompositions[split] = {}
        for name, values in (('az', baseline_queries), (arm, candidate_queries)):
            decomp = {
                'student_motor_mse': st.mean(row['first10motor_mse'] for row in values),
                'future_reader_effect_mse': st.mean(row['reader_first10motor_rmse'] ** 2 for row in values),
                'quantized_reader_gap_mse': st.mean(row['reader_gap_first10motor_rmse'] ** 2 for row in values),
                'cross_term': st.mean(row['reader_error_cross_term'] for row in values)}
            if not math.isclose(decomp['student_motor_mse'], decomp['future_reader_effect_mse']
                                + decomp['quantized_reader_gap_mse'] + decomp['cross_term'],
                                rel_tol=1e-5, abs_tol=1e-8):
                raise ValueError('Aggregate reader decomposition differs')
            decompositions[split][name] = decomp
        splits[split] = {'cases': len(case_ids), 'queries_per_arm': len(candidate_queries),
                         'overall': overall, 'by_suite': by_suite, 'by_seed': seed_metrics,
                         'aggregation': 'mean seeds within case, cases within suite, four equal-size suites equally'}
        paired_rows[split] = [{'case_id': case, 'suite': cases['baseline'][case]['suite'],
                               'baseline': cases['baseline'][case], 'candidate': cases[arm][case],
                               'delta': {metric: cases[arm][case][metric] - cases['baseline'][case][metric] for metric in METRICS}}
                              for case in case_ids]
    unit = 'aggregate'
    advance = gate(splits['heldout']['overall'])
    summary = {'protocol': p['protocol'], 'unit': unit, 'arm': arm, 'complete': True,
               'pbs_jobid': os.environ['PBS_JOBID'], 'baseline_records': 160, 'candidate_records': 160,
               'splits': splits, 'reader_decomposition': decompositions, 'fixed_gate_passed': advance,
               'decision': 'candidate_improves_no_automatic_followup' if advance else 'stop_this_recipe_no_parameter_sweep',
               'beta': receipt['beta'], 'initial_parity': receipt['initial_parity'],
               'actual_infer_parity': receipt['actual_infer_parity'],
               'new_model_queries': {name: value['new_model_queries'] for name, value in workers.items()},
               'new_bf16_reader_queries': {name: value['new_bf16_reader_queries'] for name, value in workers.items()},
               'parameter_reverse_passes': {name: value['parameter_reverse_passes'] for name, value in workers.items()},
               'baseline_new_model_queries': 0, 'new_teacher_queries': 0, 'new_closedloop_episodes': 0,
               'scope': 'fixed-input decoded BF16 VQ/dynamic A4/BF16 KV; historical heldout is a development regression cohort; no independent generalization or closedloop claim'}
    summary['mechanism_counts'] = {}
    for split, pairs in paired_rows.items():
        summary['mechanism_counts'][split] = {
            'reader_better_student_better': sum(row['delta']['reader_first10motor_rmse'] < 0 and row['delta']['first10motor_rmse'] < 0 for row in pairs),
            'reader_better_student_worse': sum(row['delta']['reader_first10motor_rmse'] < 0 and row['delta']['first10motor_rmse'] > 0 for row in pairs),
            'reader_worse_student_better': sum(row['delta']['reader_first10motor_rmse'] > 0 and row['delta']['first10motor_rmse'] < 0 for row in pairs),
            'reader_worse_student_worse': sum(row['delta']['reader_first10motor_rmse'] > 0 and row['delta']['first10motor_rmse'] > 0 for row in pairs)}
    save(results / unit / 'summary.json', summary)
    save(results / unit / 'paired_cases.json', paired_rows)
    save(results / 'comparison.json', summary)
    lines = ['# C80 A+Z / frozen-reader KL：完整配对结果', '',
             '两组同一原始 C8 banks 初始化、C80 80 updates；teacher 与 C80 action 统计全部复用。',
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
              'beta=' + str(receipt['beta']) + '；仅用 C80 初始输入标定一次。',
              '不使用 heldout 选择 checkpoint、系数或训练顺序；两个 sampler seeds 不是独立训练重复。',
              '这是固定输入保真验证；不代表闭环、native INT4 或独立泛化收益。', '']
    destination = results / 'ANALYSIS.zh.md'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print('FROZEN_READER_AGGREGATE_COMPLETE', summary['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    run(read(args.protocol))

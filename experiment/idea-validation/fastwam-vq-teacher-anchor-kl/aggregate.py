"""Complete-cohort paired fixed-input report; no checkpoint or coefficient selection."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
from anchor_selfcheck import require_allocation

METRICS = ('first10motor_rmse', 'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse',
           'future_latent_nmse', 'action_mse', 'latent_nmse', 'latent_anchor_kl',
           'baseline_objective', 'composite_objective', 'gripper_command_errors')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    require_allocation()
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True)
    p = read(parser.parse_args().protocol); root = Path(p['remote_root'])
    coefficient = read(root / 'results/anchor_kl/loss_scaling.json')
    beta = coefficient['beta']; records, summaries = {}, {}
    for arm in p['arms']:
        completed = read(root / 'results' / arm / 'summary.json')
        if (not completed['complete'] or completed['updates'] != 80 or completed['evaluation_queries'] != 160
                or completed['new_teacher_queries'] != 0):
            raise RuntimeError('Both complete final80 methods are required')
        for split in ('calibration', 'heldout'):
            ids = p['calibration' if split == 'calibration' else 'evaluation']['case_ids']
            expected = [(case, seed) for case in ids for seed in ([0] if split == 'calibration' else [0, 1])]
            rows = [read(root / 'results' / split / arm / 'queries' / f'{case}_{seed}.json') for case, seed in expected]
            if len(rows) != 80 or [(r['case_id'], r['seed_index']) for r in rows] != expected:
                raise RuntimeError('Complete ordered 80-query cohort required')
            for r in rows:
                if r['protocol'] != p['protocol'] or r['arm'] != arm or r['split'] != split or r['updates'] != 80:
                    raise RuntimeError('Endpoint identity differs')
                r['composite_objective'] = r['baseline_objective'] + beta * r['latent_anchor_kl']
                if any(not math.isfinite(r[m]) or r[m] < 0 for m in METRICS):
                    raise RuntimeError('Invalid endpoint metric')
            suites = sorted({r['suite'] for r in rows}); by_suite = {}
            if len(suites) != 4:
                raise RuntimeError('Four suites required')
            for suite in suites:
                sr = [r for r in rows if r['suite'] == suite]; cases = sorted({r['case_id'] for r in sr})
                if len(sr) != 20 or len(cases) != (20 if split == 'calibration' else 10):
                    raise RuntimeError('Suite case coverage differs')
                by_suite[suite] = {m: statistics.mean(statistics.mean(r[m] for r in sr if r['case_id'] == case) for case in cases) for m in METRICS}
            records[split, arm] = rows
            summaries[split, arm] = {'queries': 80, 'cases': len(ids), 'by_suite': by_suite,
                                     'overall': {m: statistics.mean(v[m] for v in by_suite.values()) for m in METRICS}}
    paired = {}
    for split in ('calibration', 'heldout'):
        base = {(r['case_id'], r['seed_index']): r for r in records[split, 'mse_nmse']}
        candidate = records[split, 'anchor_kl']; cases = []
        for case in sorted({r['case_id'] for r in candidate}):
            rows = [r for r in candidate if r['case_id'] == case]
            for row in rows:
                reference = base[case, row['seed_index']]
                if row['plan_row'] != reference['plan_row'] or row['sampler_seed'] != reference['sampler_seed']:
                    raise RuntimeError('Paired input/seed differs')
            cases.append({'case_id': case, 'suite': rows[0]['suite'],
                          **{m: statistics.mean(r[m] - base[case, r['seed_index']][m] for r in rows) for m in METRICS}})
        paired[split] = {'cases': cases, 'summary': {m: {'mean_delta': statistics.mean(r[m] for r in cases),
                         'median_delta': statistics.median(r[m] for r in cases),
                         'improved': sum(r[m] < 0 for r in cases), 'worsened': sum(r[m] > 0 for r in cases),
                         'tied': sum(r[m] == 0 for r in cases)} for m in METRICS}}
    result = {'protocol': p['protocol'], 'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
              'calibration_records': 160, 'heldout_records': 160, 'beta': beta, 'loss_scaling': coefficient,
              'splits': {split: {arm: summaries[split, arm] for arm in p['arms']} for split in ('calibration', 'heldout')},
              'paired': paired, 'heldout_selection': False, 'new_teacher_queries': 0,
              'closedloop_required': True, 'independent_generalization_claim': False}
    save(root / 'results/aggregate/summary.json', result)
    save(root / 'results/comparison.json', result)
    report = ['# Additive latent teacher-anchor KL：第一轮固定输入结果', '',
              '同一C80/teacher/初始化/80updates；原MSE+NMSE训练复用且完整端点精确重放；新方法只增加latent KL。', '',
              f'冻结beta={beta:.12g}。按case内平均seeds、suite内平均case、四suite等权；完整heldout40cases/80queries。', '',
              '| Split | Arm | Motor RMSE | Latent NMSE | Anchor KL | Common composite | Gripper wrong /10 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for split in ('calibration', 'heldout'):
        for arm in p['arms']:
            row = summaries[split, arm]['overall']
            report.append('| ' + split + ' | ' + arm + ' | ' + ' | '.join(f'{row[m]:.9g}' for m in ('first10motor_rmse', 'future_latent_nmse', 'latent_anchor_kl', 'composite_objective', 'gripper_command_errors')) + ' |')
    report += ['', '这些cohorts已用于前期工程分析；不据此声明独立泛化、统计显著性或native低比特速度。完整280 paired闭环仍须另行完成。', '',
               '逐suite、full32/late22/gripper与case配对保留于comparison.json。未按heldout选择checkpoint、温度或beta。']
    (root / 'results/ANALYSIS.zh.md').write_text('\n'.join(report) + '\n')
    print('ANCHOR_AGGREGATE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

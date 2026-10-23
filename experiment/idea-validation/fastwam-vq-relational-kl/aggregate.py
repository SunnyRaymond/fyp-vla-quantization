"""Summarize the paired structural-loss comparison without checkpoint selection."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

from relational_selfcheck import require_allocation

METRICS = ('action_kl', 'latent_kl', 'action_temporal_kl', 'latent_spatiotemporal_kl',
           'softbin_objective', 'relational_objective', 'first10motor_rmse', 'future_latent_nmse')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    root = Path(p['remote_root'])
    prep = json.loads((root / 'results/prepare/summary.json').read_text())
    if not prep['complete'] or not prep['passed'] or prep['heldout_used']:
        raise RuntimeError('Successful teacher-only definition validation required')
    summaries, records = {}, {}
    for arm in p['arms']:
        training = json.loads((root / 'results' / arm / 'summary.json').read_text())
        if not training['complete'] or training['updates'] != 80 or training['evaluation_queries'] != 160 or training['new_teacher_queries'] != 0:
            raise RuntimeError('Both final80 endpoints must be complete')
        for split in ('calibration', 'heldout'):
            expected = [(case, seed) for case in p['calibration' if split == 'calibration' else 'evaluation']['case_ids']
                        for seed in ([0] if split == 'calibration' else [0, 1])]
            rows = [json.loads((root / 'results' / split / arm / 'queries' / f'{case}_{seed}.json').read_text()) for case, seed in expected]
            if len(rows) != 80 or len({(r['case_id'], r['seed_index']) for r in rows}) != 80:
                raise RuntimeError('Exact eighty-query coverage required')
            if any(r['arm'] != arm or r['split'] != split or r['updates'] != 80 or r['protocol'] != p['protocol'] for r in rows):
                raise RuntimeError('Endpoint identity differs')
            if any(not math.isfinite(r[m]) for r in rows for m in METRICS):
                raise RuntimeError('Nonfinite endpoint metric')
            records[split, arm] = rows
            by_suite = {}
            for suite in sorted({r['suite'] for r in rows}):
                sr = [r for r in rows if r['suite'] == suite]
                cases = sorted({r['case_id'] for r in sr})
                if len(cases) != (20 if split == 'calibration' else 10):
                    raise RuntimeError('Four-suite case coverage differs')
                by_suite[suite] = {m: statistics.mean(statistics.mean(r[m] for r in sr if r['case_id'] == case) for case in cases) for m in METRICS}
            if len(by_suite) != 4:
                raise RuntimeError('Four suites required')
            summaries[split, arm] = {'queries': 80, 'cases': 80 if split == 'calibration' else 40,
                                     'overall': {m: statistics.mean(s[m] for s in by_suite.values()) for m in METRICS}, 'by_suite': by_suite}
    paired, changes = {}, {}
    for split in ('calibration', 'heldout'):
        base = {(r['case_id'], r['seed_index']): r for r in records[split, 'softbin']}
        candidate = records[split, 'relational']
        details = []
        for case in sorted({r['case_id'] for r in candidate}):
            rr = [r for r in candidate if r['case_id'] == case]
            for r in rr:
                previous = base[r['case_id'], r['seed_index']]
                if r['plan_row'] != previous['plan_row'] or r['sampler_seed'] != previous['sampler_seed']:
                    raise RuntimeError('Cross-arm paired input binding differs')
            details.append({'case_id': case, 'suite': rr[0]['suite'],
                            **{m: statistics.mean(r[m] - base[r['case_id'], r['seed_index']][m] for r in rr) for m in METRICS}})
        paired[split] = {'cases': details, 'summary': {m: {'median_delta': statistics.median(r[m] for r in details),
                          'improved_cases': sum(r[m] < 0 for r in details), 'worsened_cases': sum(r[m] > 0 for r in details),
                          'tied_cases': sum(r[m] == 0 for r in details)} for m in METRICS}}
        changes[split] = {m: 100 * (summaries[split, 'relational']['overall'][m] / summaries[split, 'softbin']['overall'][m] - 1) for m in METRICS}
    comparison = {'protocol': p['protocol'], 'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
                  'splits': {split: {arm: summaries[split, arm] for arm in p['arms']} for split in ('calibration', 'heldout')},
                  'percent_change_relational_vs_softbin': changes, 'frozen_temperatures': prep['relational_config'],
                  'frozen_coefficients': prep['coefficients'], 'heldout_selection': False}
    scale = json.loads((Path(p['source_dual_root']) / 'results/prepare/loss_scaling.json').read_text())
    alpha_a, alpha_z = scale['action_kl_coefficient'], scale['latent_kl_coefficient']
    latent_weight = p['training']['rho'] * p['training']['normalization_c']
    beta_a, beta_z = prep['coefficients']['action_beta'], prep['coefficients']['latent_beta']
    contributions = {}
    for split in ('calibration', 'heldout'):
        contributions[split] = {}
        for arm in p['arms']:
            mean = summaries[split, arm]['overall']
            values = {'action_anchor': alpha_a * mean['action_kl'],
                      'latent_anchor': latent_weight * alpha_z * mean['latent_kl'],
                      'action_relation': alpha_a * beta_a * mean['action_temporal_kl'],
                      'latent_relation': latent_weight * alpha_z * beta_z * mean['latent_spatiotemporal_kl']}
            contributions[split][arm] = {'weighted_terms': values,
                'fractions_of_common_composite': {key: value / sum(values.values()) for key, value in values.items()}}
    comparison['component_contributions'] = contributions
    heldout = changes['heldout']
    decision = {'complete': True, 'heldout_used_for_training_or_selection': False,
                'both_structural_terms_lower': all(heldout[m] < 0 for m in ('action_temporal_kl', 'latent_spatiotemporal_kl')),
                'common_composite_lower': heldout['relational_objective'] < 0,
                'common_softbin_lower': heldout['softbin_objective'] < 0,
                'rmse_nmse_diagnostic': {m: heldout[m] for m in ('first10motor_rmse', 'future_latent_nmse')},
                'closed_loop_tested': False, 'automatic_next_training': False,
                'scope': 'one trajectory per arm; paired fixed-input evidence, no claim of full joint policy density or task success'}
    save(root / 'results/comparison.json', comparison)
    save(root / 'results/paired_details.json', paired)
    save(root / 'results/decision.json', decision)
    report = ['# 第一轮时空关系 KL 对照', '',
              '两组使用相同 C80、共享 teacher、原 held-out40cases×2seeds、初始化、80updates 和可训练参数。soft-bin 复用原 dual_kl final80；关系组仅增加 temporal/spatial KL。', '',
              '温度和关系项系数仅用完整 C80 teacher 及预设 standardized noise=0.1、相对权重0.25 冻结。未使用新的 development set 或 held-out 选参数。', '',
              '| Split | Arm | Action KL | Latent KL | Temporal KL | Spatiotemporal KL | Soft-bin objective | Composite objective | Motor RMSE | Latent NMSE |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('calibration', 'heldout'):
        for arm in p['arms']:
            overall = summaries[split, arm]['overall']
            report.append('| ' + split + ' | ' + arm + ' | ' + ' | '.join(f'{overall[m]:.8g}' for m in METRICS) + ' |')
    report += ['', '## Held-out 相对变化', '', '| Metric | Relational vs soft-bin | Improved cases / 40 | Median paired delta |', '|---|---:|---:|---:|']
    for m in METRICS:
        pair = paired['heldout']['summary'][m]
        report.append(f"| {m} | {heldout[m]:+.3f}% | {pair['improved_cases']}/40 | {pair['median_delta']:+.6g} |")
    report += ['', '## 统一 composite 各项占比', '', '| Split | Arm | Action anchor | Latent anchor | Action relation | Latent relation |', '|---|---|---:|---:|---:|---:|']
    for split in ('calibration', 'heldout'):
        for arm in p['arms']:
            fractions = contributions[split][arm]['fractions_of_common_composite']
            report.append('| ' + split + ' | ' + arm + ' | ' + ' | '.join(f'{100 * value:.2f}%' for value in fractions.values()) + ' |')
    report += ['', '## 分布定义验证与边界', '',
               '受控错误使用现有完整 C80 teacher；不产生新的 teacher/student forward。时间错序、空间错位与匹配 MSE 的整体偏置分别报告，不能把所有等幅误差视为相同。', '',
               '关系距离存在平移、整体反号和部分时间方向对称性；独立自检明确包含线性匀速动作反转、uniform temporal latent shift 的反例。soft-bin 绝对值约束保留以覆盖这些盲区。', '',
               '此轮衡量固定位置邻接的关系分布，未估计完整 action–latent 生成联合密度，也未执行环境闭环。结构 loss 下降和固定输入 RMSE/NMSE 变化均不能替代闭环成功率。', '',
               '每组仅一条训练轨迹；40个held-out case先平均各自两个seed，不能将80queries视为80个独立case。', '']
    (root / 'results/ANALYSIS.zh.md').write_text('\n'.join(report), encoding='utf-8')
    save(root / 'results/aggregate/summary.json', {'protocol': p['protocol'], 'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
         'calibration_records': 160, 'heldout_records': 160, 'updates_per_arm': 80, 'new_teacher_queries': 0, 'decision': decision})
    print('RELATIONAL_AGGREGATE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

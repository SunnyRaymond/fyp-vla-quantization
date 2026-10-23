"""Reports every fixed arm; heldout results never select a checkpoint or recipe."""
import argparse
import json
from pathlib import Path

from cohort_metrics import aggregate_records, _comparison, METRICS, LABELS, SUITES

PROTOCOL = 'fastwam-r128-improvements-v1'
C8 = (2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def lane_aggregates(out, p, lane, rows):
    result = {}
    splits = [('calibration', C8, [0], 2), ('heldout', range(3000, 3040), [0, 1], 10)]
    if lane.startswith('coverage_'):
        splits.append(('c80', range(2000, 2080), [0], 20))
    for split, cases, seeds, per_suite in splits:
        result[split] = {}
        for spec in p['lanes'][lane]['arms']:
            arm = spec['name']
            records = []
            for case in cases:
                for seed in seeds:
                    record = read(Path(out) / split / arm / 'queries' / f'{case}_{seed}.json')
                    identity = dict(protocol=PROTOCOL, lane=lane, arm=arm, split=split,
                        case_id=case, seed_index=seed, suite=rows[case]['suite'],
                        sampler_seed=rows[case]['sampler_seeds'][seed])
                    if any(record.get(k) != v for k, v in identity.items()):
                        raise ValueError('Query identity differs: ' + str(identity))
                    records.append(record)
            result[split][arm] = aggregate_records(records, p['training']['latent_weight'],
                case_ids=tuple(cases), seed_indices=tuple(seeds), cases_per_suite=per_suite,
                suite_by_case={c: rows[c]['suite'] for c in cases})
    return result


def comparisons(lane, heldout):
    pairs = [('H_W16A4', name) for name in heldout if name != 'H_W16A4']
    if lane == 'optimization':
        pairs += [('R8_W16A4', 'grouped_lr1e4'),
                  ('R8_W16A4', 'interleaved_lr1e3'),
                  ('grouped_lr1e4', 'interleaved_lr1e4'),
                  ('interleaved_lr1e3', 'interleaved_lr1e4')]
    elif lane == 'coverage_c80':
        pairs.append(('R8_W16A4', 'C80_once_lr1e4'))
    return {candidate + '_vs_' + baseline: _comparison(baseline, candidate, heldout)
            for baseline, candidate in pairs}


def context_diagnostics(calibration, heldout, rows):
    """Descriptive decompositions, retaining all cases in the primary means."""
    result = {}
    seen = {(rows[c]['suite'], rows[c]['task_id']) for c in C8}
    base = calibration['H_W16A4']['by_case']
    for name, aggregate in calibration.items():
        if name == 'H_W16A4':
            continue
        values = aggregate['by_case']
        gains = {c: base[c]['mixed_loss'] - values[c]['mixed_loss'] for c in base}
        total = sum(gains.values())
        result[name] = dict(c8_case2019_mixed_gain=gains['2019'], c8_sum_mixed_gain=total,
            c8_case2019_share_of_net_gain=None if total == 0 else gains['2019'] / total,
            c8_mixed_improved_cases=sum(v > 0 for v in gains.values()), heldout_task_groups={})
        if name not in heldout:
            continue
        for label, flag in [('C8_seen_tasks_new_demo', True), ('C8_unseen_tasks', False)]:
            ids = [str(c) for c in range(3000, 3040)
                   if ((rows[c]['suite'], rows[c]['task_id']) in seen) == flag]
            group = dict(cases=len(ids), by_arm={})
            for arm in ('H_W16A4', name):
                subset = heldout[arm]['by_case']
                suite_means = {s: {m: sum(subset[c][m] for c in ids if rows[int(c)]['suite'] == s)
                    / sum(rows[int(c)]['suite'] == s for c in ids) for m in METRICS} for s in SUITES}
                group['by_arm'][arm] = {m: sum(suite_means[s][m] for s in SUITES) / 4 for m in METRICS}
            h, z = group['by_arm']['H_W16A4'], group['by_arm'][name]
            group['relative_change_percent'] = {m: None if h[m] == 0 else 100 * (z[m] / h[m] - 1) for m in METRICS}
            result[name]['heldout_task_groups'][label] = group
    return result


def markdown(report):
    lines = ['# R128 improvements: fixed-input results', '',
             '所有误差越低越好。完整固定 heldout 为40 contexts ×2 seeds，先seed→context→suite，再四suite等权。',
             '所有预先指定的arm均报告；没有按heldout挑选checkpoint或最佳配置。', '']
    for split in ('calibration', 'heldout', 'c80'):
        if split not in report:
            continue
        lines += [f'## {split}', '',
                  '| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |',
                  '|---|---:|---:|---:|---:|']
        h = report[split].get('H_W16A4')
        for arm, value in report[split].items():
            x = value['overall']
            delta = 'n/a' if h is None else f"{100 * (x['first10motor_rmse'] / h['overall']['first10motor_rmse'] - 1):+.3f}%"
            lines.append(f"| {arm} | {x['first10motor_rmse']:.9f} | {x['future_latent_nmse']:.9f} | {x['mixed_loss']:.9f} | {delta} |")
        lines.append('')
    lines += ['## Comparisons', '']
    for name, value in report['comparisons'].items():
        delta = value['candidate_relative_change_percent']
        win = value['paired_case_win_loss']['first10motor_rmse']
        lines.append(f"- {name}: motor {delta['first10motor_rmse']:+.3f}%, latent {delta['future_latent_nmse']:+.3f}%, mixed {delta['mixed_loss']:+.3f}%; motor improved/worsened {win['improved']}/{win['worsened']} of40 contexts.")
    lines += ['', '## Evidence boundary', '',
        'BF16原始weights、A4 activations、BF16 KV；固定Smooth/signs，D=I，clipping ratio=1，无VQ。',
        '固定输入数值证据，不代表closed-loop、native kernel、延迟或显著性检验。',
        '用户要求仅训练C80并复用原C8-R8与H对照。C80与原C8-R8的样本处理量、LR和梯度累积方式不同，是整体recipe对照，不能单独归因于calibration数量。', '']
    return '\n'.join(lines)


def write_lane_report(out, p, lane):
    if p.get('protocol') != PROTOCOL or lane not in p['lanes']:
        raise ValueError('Unexpected lane protocol')
    plan = read(p['calibration']['plan_file'])
    if plan.get('status') != 'complete' or plan['calibration_cohorts']['c8'] != list(C8):
        raise ValueError('Completed frozen plan differs')
    rows = {r['case_id']: r for r in plan['inputs']}
    source = read(Path(p['diagnostic_root']) / 'results/comparison.json')
    if (source.get('protocol') != p['diagnostic_protocol'] or not source.get('complete')
            or source.get('calibration_case_ids') != list(C8)
            or source.get('heldout_queries_per_arm') != 80):
        raise ValueError('Completed H/R source comparison differs')
    values = lane_aggregates(out, p, lane, rows)
    budgets = {}
    if lane != 'path':
        training = read(Path(out) / 'training.json')
        if training.get('protocol') != PROTOCOL or training.get('lane') != lane:
            raise ValueError('Training history protocol differs')
        for spec in p['lanes'][lane]['arms']:
            name = spec['name']
            arm = training['arms'][name]
            history = arm['history']
            cases = [case for update in history for case in update['cases']]
            expected = p['lanes'][lane]['training_exposures']
            if (len(history) != 8 or [r['update'] for r in history] != list(range(1, 9))
                    or len(cases) != expected):
                raise ValueError('Committed training history does not match the update/exposure budget')
            budgets[name] = dict(updates=len(history), training_exposures=len(cases),
                unique_contexts=len(set(cases)), per_suite_exposures={s:
                    sum(rows[c]['suite'] == s for c in cases) for s in SUITES})
    for split in ('calibration', 'heldout'):
        for name in ('H_W16A4', 'R8_W16A4'):
            if name not in values[split]:
                values[split][name] = source[split][name]
    report = dict(protocol=PROTOCOL, lane=lane, complete=True, **values,
        comparisons=comparisons(lane, values['heldout']),
        context_diagnostics=context_diagnostics(values['calibration'], values['heldout'], rows),
        training_budget=budgets,
        source_job=p['diagnostic_job'], source_protocol=p['diagnostic_protocol'],
        frozen_design=p['design'], target_linears=614)
    save(Path(out) / 'comparison.json', report)
    (Path(out) / 'RESULTS.zh.md').write_text(markdown(report), encoding='utf-8')
    return report


def combine(reports):
    by_lane = {r['lane']: r for r in reports}
    if set(by_lane) != {'path', 'optimization', 'coverage_c80'}:
        raise ValueError('Three completed lanes required')
    if any(r.get('protocol') != PROTOCOL or not r.get('complete') for r in reports):
        raise ValueError('Incomplete or unexpected lane report')
    merged = {split: {} for split in ('calibration', 'heldout', 'c80')}
    for report in reports:
        for split in merged:
            for arm, value in report.get(split, {}).items():
                if arm in merged[split] and merged[split][arm] != value:
                    raise ValueError('Duplicated anchor arm differs')
                merged[split][arm] = value
    pairs = {}
    for report in reports:
        pairs.update(report['comparisons'])
    a, b = 'R8_W16A4', 'C80_once_lr1e4'
    pairs[b + '_vs_' + a] = _comparison(a, b, merged['heldout'])
    report = dict(protocol=PROTOCOL, complete=True, **merged, comparisons=pairs,
        lanes=list(by_lane), frozen_design=reports[0]['frozen_design'],
        target_linears=614, coverage_comparison=_comparison(a, b, merged['heldout']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = combine([read(p) for p in args.reports])
    save(out / 'comparison.json', report)
    (out / 'RESULTS.zh.md').write_text(markdown(report), encoding='utf-8')

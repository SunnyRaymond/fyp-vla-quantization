"""Metadata-only calibration coverage; does not collect or select training data."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import statistics

from control_common import guard, read, save

FAMILIES = ('motion_blur', 'gaussian_blur', 'zoom_blur', 'fog', 'glass_blur')


def inventory(p, plan, sensor_plan):
    source = p['source_relational_protocol']
    ids = source['calibration']['case_ids']
    held_ids = p['frozen_evaluation']['heldout_case_ids']
    rows = {r['case_id']: r for r in plan['inputs']}
    if (plan.get('status') != 'complete' or plan['calibration_cohorts']['c80'] != ids
            or len(rows) != len(plan['inputs']) or set(ids) & set(held_ids)):
        raise ValueError('Original complete calibration plan and frozen case separation required')
    train, held = [rows[i] for i in ids], [rows[i] for i in held_ids]
    if len(train) != 80 or len(held) != 40 or any(r['domain'] != 'original' for r in train+held):
        raise ValueError('Expected original C80 and original heldout40')
    task = lambda r: (r['suite'], r['original_task'])
    train_tasks = defaultdict(list)
    held_tasks = defaultdict(list)
    for r in train:
        train_tasks[task(r)].append(r)
    for r in held:
        held_tasks[task(r)].append(r)
    if set(train_tasks) != set(held_tasks) or len(train_tasks) != 40:
        raise ValueError('Frozen training and heldout task inventory differs')
    for key, examples in train_tasks.items():
        if (len(examples) != 2 or len(held_tasks[key]) != 1
                or len({r['demo_id'] for r in examples}) != 2
                or {r['demo_id'] for r in examples} & {r['demo_id'] for r in held_tasks[key]}):
            raise ValueError('Training requires two demos and distinct heldout demo per task')
    suites = {}
    for suite in plan['suite_order']:
        selected = [r for r in train if r['suite'] == suite]
        suites[suite] = {'contexts': len(selected),
            'tasks': len({task(r) for r in selected}),
            'demonstrations': len({(*task(r), r['demo_id']) for r in selected}),
            'phase_counts': dict(Counter(r['phase'] for r in selected)),
            'frame0_contexts': sum(r['frame_index'] == 0 for r in selected),
            'mean_demo_fraction': statistics.fmean(r['frame_index'] / max(1, r['demo_length']-1) for r in selected)}
    noise = Counter()
    levels = defaultdict(list)
    missing_tasks = []
    for case in sensor_plan['cases']:
        row = case['row']
        match = re.search(r'_noise_(\d+)$', row['task_name'])
        if not match or not 1 <= int(match[1]) <= 50:
            raise ValueError('Frozen sensor noise level is invalid')
        level = int(match[1])
        family = FAMILIES[(level-1)//10]
        noise[family] += 1
        levels[family].append((level-1)%10+1)
        if task(row) not in train_tasks:
            missing_tasks.append(case['index'])
    return {'calibration_contexts': 80, 'heldout_contexts': 40, 'calibration_tasks': 40,
        'by_suite': suites, 'demonstration_overlap': 0,
        'observations_per_calibration_demonstration': 1,
        'declared_training_domains': dict(Counter(r['domain'] for r in train)),
        'declared_training_dimensions': dict(Counter(r['dimension'] for r in train)),
        'declared_training_sources': ['original demonstration simulator state snapshots'],
        'declared_training_sensor_corruption_contexts': sum(r['dimension'] == 'sensor_noise' for r in train),
        'declared_training_student_collected_contexts': 0,
        'sensor_noise_cases': len(sensor_plan['cases']),
        'sensor_task_indices_absent_from_c80': missing_tasks,
        'sensor_family_counts': dict(noise), 'sensor_severity_values': dict(levels),
        'evidence_scope': 'metadata and source provenance only; no numerical input/feature support or response-fidelity claim',
        'training_pool_modified': False, 'new_model_queries': 0, 'new_training_updates': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        rows, train_ids, held_ids, cases = [], [], [], []
        for suite in range(4):
            for number in range(10):
                base = {'suite': str(suite), 'original_task': str(number), 'domain': 'original',
                        'dimension': 'original', 'phase': 'middle', 'frame_index': 50, 'demo_length': 101}
                for demo in (1, 2, 3):
                    case_id = len(rows)
                    rows.append({**base, 'case_id': case_id, 'demo_id': demo})
                    (held_ids if demo == 3 else train_ids).append(case_id)
                cases.append({'index': len(cases), 'row': {**base, 'task_name': 'task_noise_15'}})
        p = {'source_relational_protocol': {'calibration': {'case_ids': train_ids}},
             'frozen_evaluation': {'heldout_case_ids': held_ids}}
        plan = {'status': 'complete', 'inputs': rows, 'suite_order': [str(x) for x in range(4)],
                'calibration_cohorts': {'c80': train_ids}}
        result = inventory(p, plan, {'cases': cases})
        assert result['demonstration_overlap'] == 0 and result['sensor_family_counts'] == {'gaussian_blur': 40}
        rows[-1]['demo_id'] = 1
        try:
            inventory(p, plan, {'cases': cases})
        except ValueError as exc:
            assert 'distinct heldout demo' in str(exc)
        else:
            raise AssertionError('Leaking a calibration demonstration into heldout was accepted')
        print('Coverage checks passed: separate demonstrations; deliberate heldout leakage rejected')
        return
    guard()
    p = read(Path(__file__).with_name('protocol.json'))
    root = Path(p['remote_root'])
    plan = read(Path(p['source_relational_protocol']['calibration']['plan_file']))
    result = inventory(p, plan, read(root/'data/sensor_plan.json'))
    summary = {'protocol': p['protocol'], 'unit': 'coverage_inventory',
               'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'], **result}
    save(root/'results/coverage_inventory/summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()

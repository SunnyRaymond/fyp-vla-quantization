"""Summarize existing results on a CPU allocation; no model loading or refitting."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics


def paired_ratios(scalar, vq):
    assert set(scalar) == set(vq)
    ratios = [vq[k] / scalar[k] for k in scalar if scalar[k] > 0]
    return {'count': len(scalar), 'vq_lower_count': sum(vq[k] < scalar[k] for k in scalar),
            'vq_higher_count': sum(vq[k] > scalar[k] for k in scalar),
            'equal_count': sum(vq[k] == scalar[k] for k in scalar),
            'zero_scalar_count': sum(v == 0 for v in scalar.values()),
            'median_vq_over_scalar_ratio': statistics.median(ratios) if ratios else None}


def run(root):
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    assert os.environ.get('PBS_JOBID') and nodefile and 'login' not in host.lower()
    assert host in {n.split('.')[0] for n in Path(nodefile).read_text().split()}
    toy = paired_ratios({'a': 2, 'b': 2}, {'a': 1, 'b': 4})
    assert toy['vq_lower_count'] == toy['vq_higher_count'] == 1 and toy['median_vq_over_scalar_ratio'] == 1.25
    data = root / 'results/full'
    s = json.loads((data / 'summary.json').read_text())
    p = json.loads((data / 'protocol.json').read_text())
    frozen = json.loads((data / 'frozen_winners.json').read_text())
    queries = [json.loads(line) for line in (data / 'queries.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (data / 'module_bank_stats.jsonl').read_text().splitlines()]
    assert s['status'] == 'complete' and s['phase'] == 'full'
    assert len(queries) == s['query_count'] == 444
    assert s['calibrated_module_count'] == s['target_module_count'] == 614
    assert all(q['episodes'] == q['post_query_env_steps'] == 0 for q in queries)
    grouped = defaultdict(list)
    for q in queries:
        assert math.isfinite(q['motor_rmse_first10_vs_bf16'])
        grouped[q['split'], q['arm']].append(q)
    expected = {}
    for group in ('selection', 'test'):
        cases = s['split'][group + '_cases']
        contexts = {(c, seed) for c in cases for seed in p['evaluation_seed_indices']}
        results = {**s[group], **s['numerical_controls'][group]}
        expected[group, 'bf16'] = contexts
        for arm, result in results.items():
            expected[group, arm] = contexts
            raw = grouped[group, arm]
            per_case = {str(c): sum(q['motor_rmse_first10_vs_bf16'] for q in raw if q['case_id'] == c) / 2 for c in cases}
            assert abs(sum(per_case.values()) / len(cases) - result['mean_motor_rmse_first10']) < 1e-10
            assert result['per_case_motor_rmse_first10'] == per_case
    contexts = {(c, 0) for c in p['calibration_cases']}
    expected['calibration', 'bf16_calibration'] = contexts
    for config in p['phase1_configs']:
        expected['calibration', config['name'] + '_bf16w_a4'] = contexts
    assert set(grouped) == set(expected)
    for key, context_set in expected.items():
        actual = [(q['case_id'], q['seed_index']) for q in grouped[key]]
        assert len(actual) == len(set(actual)) and set(actual) == context_set
    first_test = next(i for i, q in enumerate(queries) if q['split'] == 'test')
    assert first_test == 224 and all(q['split'] == 'test' for q in queries[first_test:])
    for kind, scores in frozen['selection_scores'].items():
        assert min(scores, key=scores.get) == frozen['winners'][kind] == s['winners'][kind]
    assert not frozen['preflight_only'] and frozen['test_queries_before_freeze'] == 0
    banks = defaultdict(dict)
    for row in rows:
        key = row['config'], row['kind']
        assert row['module'] not in banks[key]
        assert math.isfinite(row['weight_rmse']) and row['weight_rmse'] >= 0
        banks[key][row['module']] = row
    assert set(banks) == {(c, k) for c in p['phase2_config_names'] for k in ('scalar', 'vq')}
    weights, storage = {}, {}
    for config in p['phase2_config_names']:
        left, right = banks[config, 'scalar'], banks[config, 'vq']
        assert len(left) == len(right) == 614 and set(left) == set(right)
        weights[config] = {}
        for stream in ('all', 'video_expert', 'action_expert', 'proprio_encoder'):
            names = [n for n in left if stream == 'all' or n.startswith(stream + '.')]
            if names:
                weights[config][stream] = paired_ratios({n: left[n]['weight_rmse'] for n in names}, {n: right[n]['weight_rmse'] for n in names})
        for kind in ('scalar', 'vq'):
            receipt = s['selection'][config + '_' + kind + '_a4']['storage']
            assert sum(r['storage_bytes'] for r in banks[config, kind].values()) == receipt['encoded_weight_payload_bytes']
            storage[config + '_' + kind] = receipt
    case_pairs = {}
    for config in ('identity', s['winners']['scalar'], s['winners']['vq']):
        l = s['test'][config + '_scalar_a4']
        r = s['test'][config + '_vq_a4']
        case_pairs[config] = {**paired_ratios(l['per_case_motor_rmse_first10'], r['per_case_motor_rmse_first10']),
                              'mean_vq_over_scalar_ratio': r['mean_motor_rmse_first10'] / l['mean_motor_rmse_first10']}
    gpu = []
    log = (root / 'artifacts' / s['pbs_jobid'] / 'job.log').read_text()
    for line in log.splitlines():
        match = re.search(r', (\d+) %, (\d+) MiB, (\d+) MiB', line) if line.startswith('GPU_SAMPLE') else None
        if match:
            gpu.append(tuple(map(int, match.groups())))
    assert gpu
    result = {'status': 'complete', 'pbs_jobid': os.environ['PBS_JOBID'], 'source_full_job': s['pbs_jobid'],
              'query_coverage': {'queries': len(queries), 'calibration': 56, 'selection': 168, 'test': 220,
                                 'distinct_test_cases': 10, 'test_seeds_per_case': 2, 'queries_before_first_test': first_test,
                                 'all_declared_contexts_present_once': True, 'primary_means_reproduced_from_raw_queries': True},
              'weight_rmse_comparison': weights,
              'weight_rmse_comparison_unit': 'equal-weighted modules; same transformed coordinates; raw weight RMSE, not action or global parameter-weighted error',
              'paired_case_action_comparison': case_pairs, 'storage': storage,
              'gpu_monitor': {'sample_count': len(gpu), 'peak_used_mib': max(v[1] for v in gpu),
                              'total_mib': max(v[2] for v in gpu), 'mean_utilization_percent_diagnostic_only': sum(v[0] for v in gpu) / len(gpu)}}
    out = root / 'results/analysis'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Refusing to overwrite completed analysis')
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'full_log_tail.txt').write_text('\n'.join(log.splitlines()[-64:]) + '\n')
    print(json.dumps({'complete': True, 'queries': len(queries), 'module_bank_rows': len(rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)

"""Analyze a frozen completed-shard snapshot in a CPU PBS allocation."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import socket
import statistics

import aggregate


def guard():
    host = socket.gethostname().split('.')[0]
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().split()
    if (not os.environ.get('PBS_JOBID') or 'login' in host
            or host not in {node.split('.')[0] for node in nodes}):
        raise RuntimeError('Approved PBS compute allocation required')


def describe(values):
    values = list(values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Expected finite nonempty data')
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values))


def first_chunk(record):
    chunks = record.get('first_action_chunks')
    if not chunks or len(chunks[0]) != 32 or any(len(step) != 7 for step in chunks[0]):
        raise ValueError('Expected recorded first normalized [32,7] action chunk')
    return chunks[0]


def action_difference(a, b, horizon):
    a, b = first_chunk(a)[:horizon], first_chunk(b)[:horizon]
    motor_diff = [(x[j] - y[j]) ** 2 for x, y in zip(a, b) for j in range(6)]
    grip_diff = [(x[6] - y[6]) ** 2 for x, y in zip(a, b)]
    per_dim = [math.sqrt(statistics.mean((x[j] - y[j]) ** 2 for x, y in zip(a, b)))
               for j in range(7)]
    return math.sqrt(statistics.mean(motor_diff)), math.sqrt(statistics.mean(grip_diff)), per_dim


def summarize(records, label, shard_count):
    grouped = defaultdict(dict)
    for record in records:
        grouped[record['variant_id']][record['arm']] = record
    for variant, paired in grouped.items():
        if set(paired) != set(aggregate.ARMS):
            raise ValueError(f'Unpaired variant {variant}')
        for key in ('index', 'suite', 'dimension', 'task_id', 'env_seed', 'state_id',
                    'initial_state_path', 'description'):
            if len({r[key] for r in paired.values()}) != 1:
                raise ValueError(f'Pairing mismatch: {variant}/{key}')
    result = dict(label=label, shard_count=shard_count, variants=len(grouped),
                  episode_slots=len(records), arms={}, cells=[], tasks=[], paired={}, first_action={})
    for arm in aggregate.ARMS:
        arm_records = [r for r in records if r['arm'] == arm]
        success = sum(r['success'] for r in arm_records)
        result['arms'][arm] = dict(episodes=len(arm_records), successes=success,
            success_rate=success / len(arm_records),
            termination=dict(Counter(r['termination'] for r in arm_records)),
            steps=describe(r['steps'] for r in arm_records),
            replans=describe(r['replans'] for r in arm_records),
            episode_seconds=describe(r['episode_seconds'] for r in arm_records),
            successful_variants=[r['variant_id'] for r in arm_records if r['success']])
    for fields, target in [(('suite', 'dimension'), 'cells'), (('suite', 'original_task'), 'tasks')]:
        groups = defaultdict(list)
        for paired in grouped.values():
            reference = paired['bf16']
            groups[tuple(str(reference.get(f)) for f in fields)].append(paired)
        for key, pairs in sorted(groups.items()):
            entry = dict(zip(fields, key), n=len(pairs), arms={})
            for arm in aggregate.ARMS:
                entry['arms'][arm] = sum(p[arm]['success'] for p in pairs)
            result[target].append(entry)
    comparisons = [('w4a8', 'bf16'), ('w4a4', 'bf16'), ('w4a4kv4', 'bf16'),
                   ('w4a4', 'w4a8'), ('w4a4kv4', 'w4a4')]
    for arm, reference in comparisons:
        key = f'{arm}_vs_{reference}'
        counts = Counter((p[reference]['success'], p[arm]['success']) for p in grouped.values())
        result['paired'][key] = dict(n=len(grouped), both_success=counts[True, True],
            reference_only_success=counts[True, False], arm_only_success=counts[False, True],
            both_fail=counts[False, False],
            success_rate_difference=(counts[False, True] - counts[True, False]) / len(grouped))
        by_horizon = {}
        for horizon in (10, 32):
            differences = [action_difference(p[arm], p[reference], horizon) for p in grouped.values()]
            by_horizon[str(horizon)] = dict(motor_rmse=describe(d[0] for d in differences),
                gripper_rmse=describe(d[1] for d in differences),
                per_dimension_rmse_median=[statistics.median(d[2][j] for d in differences) for j in range(7)])
        result['first_action'][key] = by_horizon
    return result


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = aggregate.read_json(args.sources)
    entries = aggregate.source_list(sources)
    if len(entries) < 60 or any(s['state'] not in ('F', 'X') or s['exit_status'] != 0 for s in entries):
        raise ValueError('Snapshot requires at least 60 confirmed successful-terminal shards')
    _, rows, by_id, by_index = aggregate.manifest_rows(args.manifest)
    episodes, provenance, caveats, missing = aggregate.load_shards(
        args.sources.resolve(), sources, rows, by_id, by_index, allow_missing=True)
    expected = sum((s['stop'] - s['start']) * 4 for s in entries)
    if caveats or len(episodes) != expected:
        raise ValueError('Completed snapshot has missing slots or failed source caveats')
    primary_indices = {i for s in entries[:60] for i in range(s['start'], s['stop'])}
    primary = [r for r in episodes.values() if r['index'] in primary_indices]
    result = dict(protocol=aggregate.PROTOCOL, final_benchmark_complete=False,
        selection='First 60 completed shards in submission-index order; supplementary all completed at the frozen scheduler snapshot',
        snapshot_utc=sources['generated_at_utc'], pbs_jobid=os.environ['PBS_JOBID'], hostname=socket.gethostname(),
        primary=summarize(primary, 'first_60_completed', 60),
        supplementary=summarize(list(episodes.values()), 'all_completed_snapshot', len(entries)),
        source_provenance=provenance,
        interpretation_limits=[
            'Descriptive paired analysis of completed submitted tasks, not the complete 1400-variant/5600-slot benchmark.',
            'Variants share base tasks; they are not independent replications across all LIBERO-Plus suites.',
            'First action is paired at reset observation and sampler seed; later chunks follow divergent closed-loop observations and are excluded from fixed-observation error comparisons.',
            'Policy control_step_timeout is a valid failure; successful job exit and real-quant evidence certify execution, not preserved model accuracy.',
            'Final fixed-context Lat./Spd./Peak/physical Storage remain pending the separate performance job.'])
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate.write_json(args.output / 'analysis.json', result)
    with (args.output / 'episodes_compact.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['index', 'variant_id', 'suite', 'dimension', 'original_task', 'arm', 'success', 'termination', 'steps', 'replans', 'episode_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(primary, key=lambda r: (r['index'], aggregate.ARMS.index(r['arm']))))
    lines = ['# Completed-shard paired analysis', '', f"Frozen snapshot: {result['snapshot_utc']}",
        f"Primary: 60 shards, {result['primary']['variants']} paired variants, {len(primary)} episodes.",
        f"Supplementary: {len(entries)} shards, {len(episodes)} episodes.", '',
        '| Arm | Success | Rate |', '|---|---:|---:|']
    for arm, value in result['primary']['arms'].items():
        lines.append(f"| {arm} | {value['successes']}/{value['episodes']} | {value['success_rate']:.2%} |")
    lines += ['', '| Suite | Dimension | n | BF16 | W4A8 | W4A4 | W4A4KV4 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for cell in result['primary']['cells']:
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(cell['suite'], cell['dimension'], cell['n'], *(cell['arms'][a] for a in aggregate.ARMS)))
    lines += ['', '| First 10 predicted motor actions | Median normalized RMSE |', '|---|---:|']
    for comparison, value in result['primary']['first_action'].items():
        lines.append(f"| {comparison} | {value['10']['motor_rmse']['median']:.6f} |")
    lines += ['', '## Interpretation limits', ''] + ['- ' + x for x in result['interpretation_limits']]
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('PARTIAL_ANALYSIS_COMPLETE ' + json.dumps({arm: (v['successes'], v['episodes']) for arm, v in result['primary']['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()

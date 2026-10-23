"""Read completed endpoint and episode logs inside a CPU PBS allocation only."""
import collections
import json
import math
import os
from pathlib import Path
import socket
import statistics as st

BASE = Path('/scratch/users/ntu/yguo017')
ROOT = BASE / 'fastwam-vq-c80-joint-rmse-20261017'
OWNER = BASE / 'fastwam-vq-c80-clip-joint-plus-20261017'
CONTROL = BASE / 'fastwam-vq-c80-activation-clipping-20261017'
ROTATION = BASE / 'fastwam-rotation-baselines-20261008/results'


def read(path):
    return json.loads(path.read_text())


def describe(values):
    values = sorted(values)
    return {'n': len(values), 'mean': st.mean(values), 'median': st.median(values),
            'p90_nearest_rank': values[math.ceil(.9 * len(values)) - 1], 'max': max(values)}


def rmse(a, b):
    values = [(a[t][c] - b[t][c]) ** 2 for t in range(10) for c in range(6)]
    return math.sqrt(st.mean(values))


def main():
    assert os.environ.get('PBS_JOBID') and 'login' not in socket.gethostname()
    assert rmse([[0] * 7] * 32, [[1] * 7] * 32) == 1
    episodes = collections.defaultdict(dict)
    paths = [OWNER / 'results/evaluations' / arm / f'worker_{w}/episodes.jsonl'
             for arm in ('clip_only', 'joint') for w in range(4)]
    paths += [ROTATION / f'worker_{w}/episodes.jsonl' for w in range(4)]
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['arm'] not in ('clip_only', 'joint', 'bf16'):
                    continue
                assert row['index'] not in episodes[row['arm']]
                episodes[row['arm']][row['index']] = row
    assert {k: len(v) for k, v in episodes.items()} == {'clip_only': 280, 'joint': 280, 'bf16': 280}
    initial, discordant, tasks = [], [], collections.defaultdict(set)
    for i in range(280):
        b, c, j = [episodes[arm][i] for arm in ('bf16', 'clip_only', 'joint')]
        assert len({(r['variant_id'], r['env_seed'], r['sampler_seed'], r['state_id'])
                    for r in (b, c, j)}) == 1
        tasks[j['suite']].add(j.get('original_task'))
        item = {k: j.get(k) for k in ('index', 'variant_id', 'suite', 'dimension', 'original_task', 'subtype')}
        item.update(bf16_success=b['success'], clip_success=c['success'], joint_success=j['success'],
                    clip_steps=c['steps'], joint_steps=j['steps'],
                    clip_initial_motor_rmse=rmse(c['first_action_chunks'][0], b['first_action_chunks'][0]),
                    joint_initial_motor_rmse=rmse(j['first_action_chunks'][0], b['first_action_chunks'][0]))
        initial.append(item)
        if c['success'] != j['success']:
            discordant.append(item)
    grouped = {}
    for dim in ['all'] + sorted({x['dimension'] for x in initial}):
        rows = [x for x in initial if dim == 'all' or x['dimension'] == dim]
        grouped[dim] = {arm: describe([x[f'{arm}_initial_motor_rmse'] for x in rows])
                        for arm in ('clip', 'joint')}
        grouped[dim]['joint_lower_error'] = sum(x['joint_initial_motor_rmse'] < x['clip_initial_motor_rmse'] for x in rows)
    endpoints, endpoint_rows = {}, {}
    for arm in ('baseline', 'clip_only', 'joint_c80'):
        origin = ROOT if arm == 'joint_c80' else CONTROL
        rows = [read(origin / f'results/heldout/{arm}/queries/{case}_{seed}.json')
                for case in range(3000, 3040) for seed in (0, 1)]
        by_case = {case: {metric: st.mean(r[metric] for r in rows if r['case_id'] == case)
                         for metric in ('first10motor_rmse', 'future_latent_nmse')}
                   for case in range(3000, 3040)}
        endpoint_rows[arm] = by_case
        endpoints[arm] = {metric: describe([r[metric] for r in by_case.values()])
                          for metric in ('first10motor_rmse', 'future_latent_nmse')}
    deltas = [{'case_id': case,
               'motor_delta_joint_minus_clip': endpoint_rows['joint_c80'][case]['first10motor_rmse'] - endpoint_rows['clip_only'][case]['first10motor_rmse'],
               'latent_delta_joint_minus_clip': endpoint_rows['joint_c80'][case]['future_latent_nmse'] - endpoint_rows['clip_only'][case]['future_latent_nmse']}
              for case in range(3000, 3040)]
    result = {'pbs_jobid': os.environ['PBS_JOBID'], 'new_model_forwards': 0,
              'tasks_per_suite': {k: sorted(v) for k, v in tasks.items()},
              'initial_query_error_boundary': 'normalized first10 x6 motor RMSE on paired initial-state/seed protocol; later queries are different states and are not compared',
              'initial_query_errors': grouped, 'discordant_episodes': discordant,
              'camera_all_episodes': [r for r in initial if r['dimension'] == 'camera_viewpoints'],
              'original_heldout_case_descriptives': endpoints,
              'worst_motor_regressions': sorted(deltas, key=lambda r: r['motor_delta_joint_minus_clip'], reverse=True)[:8]}
    out = ROOT / 'results/mechanism_diagnostic.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(out), 'discordant': len(discordant), 'complete': True}))


if __name__ == '__main__':
    main()

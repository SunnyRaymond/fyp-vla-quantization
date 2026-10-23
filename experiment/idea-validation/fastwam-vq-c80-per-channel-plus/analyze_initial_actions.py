"""Zero-forward analysis of completed episodes; run only in a CPU PBS allocation."""
import collections
import json
import math
import os
from pathlib import Path
import socket
import statistics

BASE = Path('/scratch/users/ntu/yguo017')
ROOT = BASE / 'fastwam-vq-c80-per-channel-plus-20261018'


def describe(values):
    values = sorted(values)
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p90': values[math.ceil(.9 * len(values)) - 1], 'max': max(values)}


def errors(action, teacher):
    if any(len(x) != 32 or any(len(y) != 7 for y in x) for x in (action, teacher)):
        raise ValueError('Saved actions must be [32,7]')
    delta = [[action[t][c] - teacher[t][c] for c in range(7)] for t in range(10)]
    if not all(math.isfinite(v) for row in delta for v in row):
        raise ValueError('Nonfinite saved action')
    return {'motor_rmse': math.sqrt(sum(v*v for row in delta for v in row[:6]) / 60),
            'gripper_rmse': math.sqrt(sum(row[6]**2 for row in delta) / 10),
            'axis_rmse': [math.sqrt(sum(row[c]**2 for row in delta) / 10) for c in range(6)],
            'axis_chunk_bias': [sum(row[c] for row in delta) / 10 for c in range(6)]}


def main():
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('PBS_JOBID') or 'login' in host:
        raise RuntimeError('Real PBS compute allocation required')
    if host not in {x.split('.')[0] for x in Path(os.environ['PBS_NODEFILE']).read_text().split()}:
        raise RuntimeError('Host outside allocation')
    assert errors([[0]*7]*32, [[1]*7]*32)['motor_rmse'] == 1
    episodes = collections.defaultdict(dict)
    paths = [ROOT / f'results/evaluations/per_channel_clip/worker_{w}/episodes.jsonl' for w in range(4)]
    paths += [BASE / f'fastwam-vq-c80-clip-joint-plus-20261017/results/evaluations/{a}/worker_{w}/episodes.jsonl'
              for a in ('clip_only', 'joint') for w in range(4)]
    paths += [BASE / f'fastwam-rotation-baselines-20261008/results/worker_{w}/episodes.jsonl' for w in range(4)]
    wanted = {'bf16', 'clip_only', 'joint', 'per_channel_clip'}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['arm'] not in wanted:
                    continue
                if row['index'] in episodes[row['arm']]:
                    raise ValueError('Duplicate episode')
                episodes[row['arm']][row['index']] = row
    assert set(episodes) == wanted and all(set(v) == set(range(280)) for v in episodes.values())
    cases = []
    for index in range(280):
        rows = {a: episodes[a][index] for a in wanted}
        identity = ('variant_id', 'suite', 'dimension', 'task_id', 'state_id', 'env_seed', 'sampler_seed')
        assert len({tuple(str(r[k]) for k in identity) for r in rows.values()}) == 1
        b = rows['bf16']
        item = {k: b.get(k) for k in ('index', 'variant_id', 'suite', 'dimension', 'task_id', 'original_task')}
        item['arms'] = {a: {'success': r['success'], 'steps': r['steps'], 'termination': r['termination'],
                              **errors(r['first_action_chunks'][0], b['first_action_chunks'][0])}
                        for a, r in rows.items() if a != 'bf16'}
        item['bf16_success'] = b['success']
        cases.append(item)
    def summarize(rows):
        result = {}
        for arm in ('clip_only', 'joint', 'per_channel_clip'):
            result[arm] = {m: describe([r['arms'][arm][m] for r in rows])
                           for m in ('motor_rmse', 'gripper_rmse')}
            result[arm]['axis_rmse_mean'] = [statistics.mean(r['arms'][arm]['axis_rmse'][c] for r in rows) for c in range(6)]
            result[arm]['absolute_chunk_bias_mean'] = [statistics.mean(abs(r['arms'][arm]['axis_chunk_bias'][c]) for r in rows) for c in range(6)]
        result['per_channel_motor_better_than_clip_cases'] = sum(r['arms']['per_channel_clip']['motor_rmse'] < r['arms']['clip_only']['motor_rmse'] for r in rows)
        return result
    lost = [r for r in cases if r['arms']['clip_only']['success'] and not r['arms']['per_channel_clip']['success']]
    gained = [r for r in cases if not r['arms']['clip_only']['success'] and r['arms']['per_channel_clip']['success']]
    assert len(lost) == 15 and len(gained) == 11
    result = {'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'], 'new_model_forwards': 0,
              'new_episodes': 0, 'episodes_per_arm': 280, 'paired_initial_states': 280,
              'scope': 'normalized first10 actions on paired initial-state protocol, not later rollout states; raw pixels were not stored for a new pixel-parity test; no command-sign inference from normalized gripper',
              'all': summarize(cases),
              'by_suite': {s: summarize([r for r in cases if r['suite'] == s]) for s in sorted({r['suite'] for r in cases})},
              'by_dimension': {d: summarize([r for r in cases if r['dimension'] == d]) for d in sorted({r['dimension'] for r in cases})},
              'lost_to_clip': {'episodes': lost, 'summary': summarize(lost)},
              'gained_over_clip': {'episodes': gained, 'summary': summarize(gained)}}
    out = ROOT / 'results/initial_action_diagnostic.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'output': str(out), 'new_model_forwards': 0,
                      'lost': len(lost), 'gained': len(gained)}))


if __name__ == '__main__':
    main()

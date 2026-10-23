"""Export paired task/variant outcomes in an approved CPU PBS allocation."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import socket

ARMS = ('bf16', 'spinquant_adapted_w4a4', 'micro_scale_vq_a4')
PAIRS = ((ARMS[2], ARMS[1]), (ARMS[2], ARMS[0]), (ARMS[1], ARMS[0]))
ROOT = Path('/scratch/users/ntu/yguo017/fastwam-vq-plus-20261012')
REFERENCE = Path('/scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008/results')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if len(payload) > 256 * 1024:
        raise ValueError('Analysis export exceeds the bounded small-file limit')
    path.write_bytes(payload)


def run(out):
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if (not os.environ.get('PBS_JOBID') or 'login' in host or not nodefile
            or host not in {v.split('.')[0] for v in Path(nodefile).read_text().split()}):
        raise RuntimeError('Approved compute-node PBS allocation required')
    manifest = read(REFERENCE / 'manifest.json')
    assert manifest == read(ROOT / 'results/manifest.json')
    rows = manifest['variants']
    assert len(rows) == 280 and read(ROOT / 'results/summary.json')['complete'] is True
    found = {}
    paths = [REFERENCE / f'worker_{i}/episodes.jsonl' for i in range(4)]
    paths += [ROOT / f'results/worker_{i}/episodes.jsonl' for i in (2, 3)]
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if record['arm'] not in ARMS:
                    continue
                index, arm = record['index'], record['arm']
                row = rows[index]
                assert record['complete'] is True and type(record['success']) is bool
                assert record['variant_id'] == row['variant_id']
                assert record['suite'] == row['suite'] and record['dimension'] == row['dimension']
                assert record['env_seed'] == 800000000 + index * 1000
                assert record['sampler_seed'] == 900000000 + index * 1000
                assert (arm, index) not in found
                found[arm, index] = record
    assert set(found) == {(arm, index) for arm in ARMS for index in range(280)}

    def bucket(indices):
        n = len(indices)
        successes = {arm: sum(found[arm, i]['success'] for i in indices) for arm in ARMS}
        pairs = {}
        for candidate, reference in PAIRS:
            wins = [i for i in indices if found[candidate, i]['success'] and not found[reference, i]['success']]
            losses = [i for i in indices if found[reference, i]['success'] and not found[candidate, i]['success']]
            pairs[candidate + '_vs_' + reference] = {
                'wins': len(wins), 'losses': len(losses), 'ties': n - len(wins) - len(losses),
                'net_successes': len(wins) - len(losses), 'win_indices': wins, 'loss_indices': losses}
        patterns = collections.Counter(''.join(str(int(found[a, i]['success'])) for a in ARMS) for i in indices)
        return {'episodes': n, 'successes': successes, 'paired': pairs, 'patterns': dict(sorted(patterns.items()))}

    groups = {'overall': {'all': list(range(280))}}
    for label, key in (('suites', lambda r: r['suite']), ('dimensions', lambda r: r['dimension']),
                       ('cells', lambda r: r['suite'] + '::' + r['dimension']),
                       ('tasks', lambda r: r['suite'] + '::' + r['original_task'])):
        values = collections.defaultdict(list)
        for i, row in enumerate(rows):
            values[key(row)].append(i)
        groups[label] = dict(sorted(values.items()))
    summary = {label: {key: bucket(indices) for key, indices in values.items()} for label, values in groups.items()}
    existing = read(ROOT / 'results/summary.json')
    for label in ('suites', 'dimensions', 'cells'):
        for key, value in summary[label].items():
            for arm in ARMS:
                assert value['successes'][arm] == existing[label][key]['arms'][arm]['successes']
    assert summary['overall']['all']['successes'] == {'bf16': 216, 'spinquant_adapted_w4a4': 203, 'micro_scale_vq_a4': 204}
    compact = []
    for i, row in enumerate(rows):
        compact.append({**{key: row.get(key) for key in ('index', 'variant_id', 'suite', 'original_task', 'task_name', 'dimension', 'subtype', 'difficulty')},
                        'pattern': ''.join(str(int(found[a, i]['success'])) for a in ARMS),
                        'outcomes': {a: {'success': found[a, i]['success'], 'steps': found[a, i]['steps'],
                                        'replans': found[a, i]['replans']} for a in ARMS}})
    discordant = [row for row in compact if row['pattern'] not in ('000', '111')]
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'triplet_analysis.json', {'complete': True, 'job_id': os.environ['PBS_JOBID'],
         'arms_in_pattern_order': ARMS, 'paired_outcome_slots': len(found), 'groups': summary,
         'discordant_variants': len(discordant), 'original_tasks': len(summary['tasks']),
         'scope': 'Descriptive paired frozen-cohort comparison; exploratory task/perturbation breakdown, no hypothesis tests'})
    save(out / 'triplet_cases.json', discordant)
    print(f'TRIPLET_COMPLETE paired_slots=840 variants=280 tasks={len(summary["tasks"])} discordant_variants={len(discordant)}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out)

"""Frozen bindings shared by collection, training and endpoint evaluation."""
import copy
import json
import os
from pathlib import Path
import socket

PROTOCOL = 'fastwam-vq-rollout-c80-v1'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE', '')
    if not os.environ.get('PBS_JOBID') or 'login' in host or not Path(nodefile).is_file():
        raise RuntimeError('Real PBS compute allocation required')
    if host not in {s.split('.')[0] for s in Path(nodefile).read_text().split()}:
        raise RuntimeError('Host is outside the allocated PBS nodes')


def validate(p):
    if p['protocol'] != PROTOCOL:
        raise ValueError('Unexpected rollout C80 protocol')
    original = p['source_c80_protocol']
    for key in ('training', 'clipping', 'evaluation'):
        if p[key] != original[key]:
            raise ValueError('Original C80 contract changed: ' + key)
    if p['calibration']['case_ids'] != list(range(2000, 2080)):
        raise ValueError('Exactly 80 training contexts required')
    if p['calibration']['update_order'] != original['calibration']['update_order']:
        raise ValueError('Training update order changed')
    if p['rollout']['anchor_ids'] != list(range(2000, 2080, 2)):
        raise ValueError('Anchor selection changed')
    if p['rollout']['visited_ids'] != list(range(2001, 2080, 2)):
        raise ValueError('Rollout selection changed')
    if p['rollout']['max_executed_actions'] != 10 or p['channel_scaling'] is not False:
        raise ValueError('Rollout execution/scaling contract changed')
    return p


def expected_rows(p):
    original = read(p['calibration']['original_plan'])
    if original['status'] != 'complete':
        raise RuntimeError('Original input plan is incomplete')
    rows = {r['case_id']: copy.deepcopy(r) for r in original['inputs']}
    for case in p['rollout']['visited_ids']:
        rows[case]['rollout'] = {
            'collector_checkpoint': p['baseline_checkpoint'],
            'source_case_id': case, 'max_executed_actions': 10,
            'collector_sampler_seed': rows[case]['sampler_seeds'][0],
            'stop_at_terminal': True, 'selection': 'none; fixed one state per source context',
        }
    if len(rows) != 120 or len({(rows[c]['suite'], rows[c]['task_id']) for c in p['rollout']['visited_ids']}) != 40:
        raise RuntimeError('Expected 40 tasks and original 120 input identities')
    return rows


def load_observation(np, p, row):
    import calibration_source
    root = Path(p['remote_root']) / 'data' if 'rollout' in row else Path(p['calibration']['original_plan']).parent
    obs, meta = calibration_source.load_observation(np, root, row)
    if 'rollout' in row and meta.get('rollout') != row['rollout']:
        raise RuntimeError('Rollout observation provenance changed')
    return obs, meta


def teacher_path(p, row, seed=0):
    if row['case_id'] >= 3000:
        return Path(p['heldout_teacher']) / f"{row['case_id']}_{seed}.pt"
    root = Path(p['remote_root']) / 'results/teacher' if 'rollout' in row else Path(p['original_calibration_teacher'])
    return root / f"{row['case_id']}_{seed}.pt"

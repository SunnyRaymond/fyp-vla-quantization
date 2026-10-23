"""Small shared helpers; all tensor work requires an actual compute allocation."""
import json
import os
from pathlib import Path
import socket


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or 'login' in host.lower():
        raise RuntimeError('Real PBS compute allocation required')
    if host not in {n.split('.')[0] for n in Path(nodefile).read_text().split()}:
        raise RuntimeError('Host is outside the assigned allocation')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def seeds(row):
    original = list(row['sampler_seeds'])
    if len(original) != 2:
        raise RuntimeError('Original paired sampler seeds required')
    values = original + [1900000000 + int(row['case_id']) * 32 + k for k in range(2, 16)]
    if len(set(values)) != 16:
        raise RuntimeError('Seeds must be distinct within a state')
    return values


def load_protocol():
    return json.loads((Path(os.environ['NEW_ROOT']) / 'protocol.json').read_text())


def packet_valid(d, p, row, k):
    return (d.get('protocol') == p['protocol'] and d.get('case_id') == row['case_id']
            and d.get('seed_index') == k and d.get('sampler_seed') == seeds(row)[k]
            and d.get('plan_row') == row and d.get('suite') == row['suite'])

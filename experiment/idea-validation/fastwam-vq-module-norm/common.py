"""Frozen diagnostic scope and compute-allocation guard."""
import json
import os
from pathlib import Path
import socket

PROTOCOL = 'fastwam-vq-module-norm-v1'
ARMS = ('parent', 'action_only', 'action_latent')
DONORS = ('bf16', 'parent', 'action_only', 'action_latent')


def weight_vectors(torch, weight):
    """Match the decoder's row-wise 4D padding, including narrow Linears."""
    padding = (-weight.shape[1]) % 4
    return (torch.nn.functional.pad(weight, (0, padding)) if padding else weight).reshape(-1, 4)


def vector_mask(torch, shape, ids):
    groups = (shape[1] + 3) // 4
    return (ids[:, None] % groups) * 4 + torch.arange(4, device=ids.device)[None] < shape[1]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or 'login' in host.lower():
        raise RuntimeError('A genuine PBS compute allocation is required')
    if host not in {n.split('.')[0] for n in Path(nodefile).read_text().split()}:
        raise RuntimeError('Host is outside the approved PBS allocation')


def units(p, shard):
    if p['protocol'] != PROTOCOL or p['calibration_cases'] != [0, 2, 4, 6, 10, 12, 14, 16]:
        raise RuntimeError('Frozen diagnostic cohort changed')
    return [(arm, case) for arm in ARMS for i, case in enumerate(p['calibration_cases']) if i % 4 == shard]


def stem(arm, case):
    return f'{arm}_case_{case}_seed_0'


def contexts(p):
    rows, ordinal = [], 0
    for split, cases, seeds in (('calibration',p['calibration_cases'],[0]),
                                ('selection',p['selection_cases'],[0,1]),('test',p['test_cases'],[0,1])):
        for case in cases:
            for seed in seeds:
                rows.append(dict(split=split,case_id=case,seed_index=seed,shard=ordinal%4))
            ordinal += 1
    if len(rows)!=60 or len({(r['case_id'],r['seed_index']) for r in rows})!=60:
        raise RuntimeError('Full frozen cohort required')
    return rows

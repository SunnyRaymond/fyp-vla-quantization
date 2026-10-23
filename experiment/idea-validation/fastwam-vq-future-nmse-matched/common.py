"""Frozen scope and crash-safe small receipts; no numerical imports on login nodes."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket

PROTOCOL = 'fastwam-vq-future-nmse-matched-v1'
DONORS = ('bf16', 'parent', 'action_only', 'action_latent')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or 'login' in host.lower():
        raise RuntimeError('A genuine PBS compute allocation is required')
    if host not in {n.split('.')[0] for n in Path(nodefile).read_text().split()}:
        raise RuntimeError('Host is outside the approved PBS allocation')


def contexts(p):
    rows, ordinal = [], 0
    for split, cases, seeds, count in (
        ('calibration', p['calibration_cases'], [0], 8),
        ('selection', p['selection_cases'], [0, 1], 4),
        ('test', p['test_cases'], [0, 1], 22),
    ):
        if len(cases) != count:
            raise RuntimeError('Frozen case count changed')
        for case in cases:
            for seed in seeds:
                rows.append(dict(split=split, case_id=case, seed_index=seed, shard=ordinal % 2))
            ordinal += 1
    keys = [(r['case_id'], r['seed_index']) for r in rows]
    if (p['protocol'] != PROTOCOL or len(rows) != 60 or len(set(keys)) != 60
            or [sum(r['shard'] == s for r in rows) for s in (0, 1)] != [30, 30]
            or len({r['case_id'] for r in rows}) != 34):
        raise RuntimeError('Frozen cohort identity/coverage changed')
    return rows


def stem(c):
    return f"case_{c['case_id']}_seed_{c['seed_index']}"


def aggregate(rows, metric):
    cases = {}
    seen = set()
    for r in rows:
        key = r['case_id'], r['seed_index']
        if key in seen:
            raise RuntimeError('Duplicate paired context')
        seen.add(key)
        cases.setdefault(r['case_id'], []).append(float(r[metric]))
    if not cases:
        raise RuntimeError('Empty cohort')
    means = {k: sum(v) / len(v) for k, v in cases.items()}
    return dict(mean=sum(means.values()) / len(means), per_case=means,
                case_count=len(means), context_count=len(rows))


def paired(left, right, metric):
    a, b = aggregate(left, metric), aggregate(right, metric)
    if set(a['per_case']) != set(b['per_case']):
        raise RuntimeError('Paired case sets differ')
    delta = {k: a['per_case'][k] - b['per_case'][k] for k in a['per_case']}
    return dict(mean_delta=a['mean'] - b['mean'], improved=sum(v < 0 for v in delta.values()),
                worsened=sum(v > 0 for v in delta.values()), tied=sum(v == 0 for v in delta.values()),
                per_case_delta=delta)


def protocol():
    p=json.loads((Path(os.environ['ARTIFACTS'])/'protocol.json').read_text())
    contexts(p)
    return p


def receipt(phase,**values):
    r=dict(protocol=PROTOCOL,phase=phase,job_id=os.environ['PBS_JOBID'],**values)
    save(Path(os.environ['ARTIFACTS'])/'summary.json',r)
    return r

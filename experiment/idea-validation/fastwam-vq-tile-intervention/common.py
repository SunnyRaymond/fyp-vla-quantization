"""Frozen cohort, small receipts and an allocation guard."""
import json
import os
from pathlib import Path
import socket

PROTOCOL = 'fastwam-vq-tile-intervention-v1'


def guard():
    host = socket.gethostname().split('.')[0]
    nodes = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodes or 'login' in host.lower():
        raise RuntimeError('Approved PBS compute allocation required')
    if host not in {s.split('.')[0] for s in Path(nodes).read_text().split()}:
        raise RuntimeError('Host is outside the approved allocation')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def protocol():
    p = json.loads((Path(os.environ['ARTIFACTS']) / 'protocol.json').read_text())
    if p['protocol'] != PROTOCOL or p['calibration_cases'] != [0,2,4,6,10,12,14,16]:
        raise RuntimeError('Frozen protocol changed')
    return p


def contract(path,p):
    path=Path(path)
    if path.is_file() and json.loads(path.read_text())!=p:
        raise RuntimeError('Resume protocol changed')
    save(path,p)


def contexts(p):
    return [dict(split='calibration',case_id=c,seed_index=0,shard=i) for i,c in enumerate(p['calibration_cases'])]


def stem(c):
    return f"case_{c['case_id']}_seed_{c['seed_index']}"


def receipt(phase, **values):
    s = dict(protocol=PROTOCOL, phase=phase, job_id=os.environ['PBS_JOBID'], **values)
    save(Path(os.environ['ARTIFACTS'])/'summary.json',s)
    return s


def paired_summary(rows, metric):
    from statistics import mean
    groups = {}
    for r in rows: groups.setdefault(r['case_id'],[]).append(r[metric])
    cases = {str(k):mean(v) for k,v in groups.items()}
    return dict(mean=mean(cases.values()),per_case=cases,case_count=len(cases),context_count=len(rows))

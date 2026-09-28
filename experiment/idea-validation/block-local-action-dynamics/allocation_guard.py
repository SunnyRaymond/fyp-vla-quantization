"""Fail closed before numerical work outside an approved PBS allocation."""
import os
import socket
from pathlib import Path


def ensure_allocation(require_gpu=False):
    job = os.environ.get('PBS_JOBID', '')
    nodefile = Path(os.environ.get('PBS_NODEFILE', '/nonexistent'))
    host = socket.gethostname().lower().split('.')[0]
    if not job or not nodefile.is_file() or any(x in host for x in ('login', 'head', 'submit')):
        raise RuntimeError('Numerical work requires an approved PBS compute allocation')
    nodes = {x.lower().split('.')[0] for x in nodefile.read_text().split()}
    if host not in nodes:
        raise RuntimeError('Current host is not in this PBS allocation')
    if require_gpu and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise RuntimeError('This job requires an allocated GPU')
    return {'job_id': job, 'hostname': host, 'nodefile': str(nodefile)}

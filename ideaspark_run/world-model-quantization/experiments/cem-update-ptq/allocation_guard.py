"""Fail closed unless this process is on its currently running PBS allocation."""
import os
from pathlib import Path
import re
import socket
import subprocess


def require_allocation():
    job = os.environ.get('PBS_JOBID', '')
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE', '')
    if not job or not re.fullmatch(r'[0-9]+[.][A-Za-z0-9_.-]+', job):
        raise RuntimeError('A real PBS_JOBID is required; never fabricate environment variables')
    if 'login' in host.lower() or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError('Login nodes and missing PBS nodefiles are prohibited')
    nodes = {line.strip().split('.')[0] for line in Path(nodefile).read_text().splitlines()}
    if host not in nodes:
        raise RuntimeError('Actual hostname is absent from the PBS node allocation')
    evidence = subprocess.check_output(['qstat', '-f', job], universal_newlines=True, timeout=30)
    if not re.search(r'job_state\s*=\s*R\b', evidence):
        raise RuntimeError('PBS does not confirm a running allocation')
    execution = re.search(r'exec_host\s*=\s*(\S+)', evidence)
    if not execution or host not in execution.group(1):
        raise RuntimeError('PBS exec_host does not match actual hostname')
    return {'job_id': job, 'hostname': host, 'verified': True}


if __name__ == '__main__':
    import json
    print(json.dumps(require_allocation()))

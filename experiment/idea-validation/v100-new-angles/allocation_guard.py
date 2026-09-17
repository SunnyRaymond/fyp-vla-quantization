"""Validate actual CCDS compute-node ownership, normalizing DNS name case."""
import getpass
import os
import re
import socket
import subprocess


def _run(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=20).stdout.strip()


def _host(value):
    return value.strip().split('.', 1)[0].casefold()


def require_allocation():
    job = os.environ.get('SLURM_JOB_ID', '')
    actual = _host(socket.gethostname())
    env_nodes = os.environ.get('SLURM_JOB_NODELIST', '')
    if not re.fullmatch(r'[0-9]+', job) or not re.fullmatch(r'tc1n[0-9]{2}', actual) or not env_nodes:
        raise RuntimeError('Real CCDS compute-node SLURM allocation required')
    record = _run('scontrol', 'show', 'job', '-o', job)
    fields = dict(item.split('=', 1) for item in record.split() if '=' in item)
    if fields.get('JobId') != job or fields.get('JobState') != 'RUNNING':
        raise RuntimeError('Scheduler does not confirm this running job')
    if fields.get('UserId', '').split('(', 1)[0] != getpass.getuser():
        raise RuntimeError('Job owner differs from current user')
    if fields.get('Partition') != 'UGGPU-TC1':
        raise RuntimeError('Unexpected partition')
    recorded = fields.get('NodeList', '')
    hosts = {_host(x) for x in _run('scontrol', 'show', 'hostnames', recorded).splitlines()}
    env_hosts = {_host(x) for x in _run('scontrol', 'show', 'hostnames', env_nodes).splitlines()}
    if actual not in hosts or hosts != env_hosts:
        raise RuntimeError('Actual hostname or environment differs from scheduler allocation')
    return {'scheduler': 'slurm', 'job_id': job, 'hostname': actual,
            'nodelist': recorded, 'partition': fields['Partition'],
            'user': getpass.getuser(), 'verified': True,
            'hostname_comparison': 'DNS case normalized; actual hostname is read from socket'}


if __name__ == '__main__':
    import json
    print(json.dumps(require_allocation()))

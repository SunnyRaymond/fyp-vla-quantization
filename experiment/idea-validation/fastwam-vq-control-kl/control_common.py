"""Small shared contracts; numerical work requires a real compute allocation."""
import json
import os
from pathlib import Path
import socket
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = Path(os.environ.get('PBS_NODEFILE', '/nonexistent'))
    if not os.environ.get('PBS_JOBID') or 'login' in host.lower() or not nodefile.is_file():
        raise RuntimeError('A real PBS compute allocation is required')
    if host not in {line.split('.')[0] for line in nodefile.read_text().split()}:
        raise RuntimeError('Host is outside the allocated PBS nodes')


class Deadline:
    def __init__(self, seconds=6600):
        self.started = int(os.environ['FW_JOB_START_EPOCH'])
        self.seconds = seconds

    def near(self, reserve=180):
        return time.time() - self.started >= self.seconds - reserve

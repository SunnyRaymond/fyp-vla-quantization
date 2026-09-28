"""Small control-file transfer / PBS queries; never run workloads on login."""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'nscc-access'))
from aspire2a_shell import connect
import paramiko

p = argparse.ArgumentParser()
p.add_argument('operation', choices=['command', 'put', 'get'])
p.add_argument('args', nargs='+')
a = p.parse_args()
j, t = connect()
try:
    if a.operation == 'command':
        if len(a.args) != 1:
            raise ValueError('Pass one reviewed lightweight shell command')
        c = t.open_session(timeout=20)
        c.exec_command(a.args[0])
        print(c.makefile().read().decode('utf-8', errors='replace'), end='')
        print(c.makefile_stderr().read().decode('utf-8', errors='replace'), end='', file=sys.stderr)
        sys.exit(c.recv_exit_status())
    with paramiko.SFTPClient.from_transport(t) as s:
        source, destination = a.args
        size = Path(source).stat().st_size if a.operation == 'put' else s.stat(source).st_size
        # Score JSON is about 300 KiB; still a small result/control file.
        # Raw tensor artifacts remain on compute storage.
        limit = 524288 if str(source).endswith('.json') else 262144
        if size > limit:
            raise ValueError('Small-file limit exceeded; large transfer requires compute allocation')
        if a.operation == 'put':
            s.put(source, destination)
        else:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            s.get(source, destination)
        print(f'{a.operation}: {size} bytes')
finally:
    t.close()
    j.close()

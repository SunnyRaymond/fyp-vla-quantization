"""Use the existing verified ASPIRE2A transport; never copy credentials."""
import argparse
import sys
from pathlib import Path
import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nscc-access'))
from aspire2a_shell import connect

p = argparse.ArgumentParser()
p.add_argument('action', choices=['exec', 'put', 'get'])
p.add_argument('source')
p.add_argument('destination', nargs='?')
a = p.parse_args()
jump, remote = connect()
try:
    if a.action == 'exec':
        channel = remote.open_session(timeout=30)
        channel.exec_command(a.source)
        import time
        while True:
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536)); sys.stdout.buffer.flush()
            if channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(65536)); sys.stderr.buffer.flush()
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            time.sleep(.1)
        sys.exit(channel.recv_exit_status())
    else:
        with paramiko.SFTPClient.from_transport(remote) as sftp:
            if a.action == 'put':
                sftp.put(a.source, a.destination)
            else:
                sftp.get(a.source, a.destination)
finally:
    remote.close()
    jump.close()

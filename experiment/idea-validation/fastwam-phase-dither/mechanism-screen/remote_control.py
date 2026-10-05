"""Small control-file transport. Never load model/data on the login node."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'nscc-access'))
# This run is nested under experiment/idea-validation/fastwam-phase-dither.
from aspire2a_shell import connect

def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['command', 'put', 'get'])
    p.add_argument('source')
    p.add_argument('destination', nargs='?')
    a = p.parse_args()
    jump, remote = connect()
    try:
        if a.operation == 'command':
            c = remote.open_session(timeout=20)
            c.exec_command(a.source)
            out = c.makefile('rb').read().decode('utf-8', 'replace')
            err = c.makefile_stderr('rb').read().decode('utf-8', 'replace')
            print(out, end='')
            if err:
                print(err, end='', file=sys.stderr)
            return c.recv_exit_status()
        import paramiko
        with paramiko.SFTPClient.from_transport(remote) as s:
            if a.operation == 'put':
                if Path(a.source).stat().st_size > 262144:
                    raise ValueError('Only small control files may be uploaded.')
                s.put(a.source, a.destination)
            else:
                if s.stat(a.source).st_size > 524288:
                    raise ValueError('Only compact result files may be downloaded.')
                s.get(a.source, a.destination)
        return 0
    finally:
        remote.close()
        jump.close()

if __name__ == '__main__':
    raise SystemExit(main())

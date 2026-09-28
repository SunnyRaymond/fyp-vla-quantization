"""Small source staging and lightweight PBS control only; no remote computation."""
import argparse
import json
import shlex
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACCESS = ROOT.parents[2] / 'nscc-access'
sys.path.insert(0, str(ACCESS))
import aspire2a_shell as access
import paramiko

REMOTE = '/scratch/users/ntu/yguo017/block-local-action-dynamics-round1'
FILES = ('FREEZE.json', 'PROTOCOL.zh.md', 'allocation_guard.py', 'controlled_system.py',
         'models.py', 'test_mechanism.py', 'run_experiment.py', 'write_report.py', 'run_round1.pbs')


def connect():
    try:
        return access.connect()
    except (socket.timeout, TimeoutError, OSError) as exc:
        print(f'Jump route unavailable ({type(exc).__name__}); checking pinned direct route', file=sys.stderr)
        credentials = access.read_credentials()
        transport = paramiko.Transport(socket.create_connection((access.ASPIRE2A_HOST, 22), timeout=20))
        transport.start_client(timeout=20)
        access.verify_host_key(transport, access.ASPIRE2A_HOST, access.NSCC_KNOWN_HOSTS)
        transport.auth_password(credentials['NSCC_USERNAME'], credentials['NSCC_PASSWORD'])
        if not transport.is_authenticated():
            transport.close()
            raise RuntimeError('ASPIRE2A authentication failed')
        return None, transport


def command(transport, value, compact=False):
    session = transport.open_session(timeout=20)
    session.settimeout(60)
    session.exec_command(value)
    stdout = session.makefile('rb').read().decode('utf-8', errors='replace')
    stderr = session.makefile_stderr('rb').read().decode('utf-8', errors='replace')
    status = session.recv_exit_status()
    display = stdout
    if compact:
        wanted = {'job_state', 'queue', 'comment', 'Exit_status', 'stime',
                  'resources_used.walltime', 'Resource_List.select', 'Resource_List.walltime'}
        fields = {}
        current = None
        for line in stdout.splitlines():
            key, separator, value = line.strip().partition(' = ')
            if separator:
                current = key if key in wanted else None
                if current:
                    fields[current] = value
            elif current:
                fields[current] += line.strip()
        display = '\n'.join([stdout.splitlines()[0]] + [f'{key} = {value}' for key, value in fields.items()]) + '\n'
    print(display, end='')
    if stderr:
        print(stderr, end='', file=sys.stderr)
    if status:
        raise RuntimeError(f'Remote control command exited {status}')
    return stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('probe', 'stage', 'submit', 'status', 'progress', 'fetch-summary'))
    parser.add_argument('--job')
    parser.add_argument('--variant', choices=('primary', 'global16'), default='primary')
    args = parser.parse_args()
    base = REMOTE if args.variant == 'primary' else REMOTE + '/global16'
    tracker = ROOT / ('JOB.json' if args.variant == 'primary' else 'JOB_global16.json')
    pbs_name = 'run_round1.pbs' if args.variant == 'primary' else 'run_global16.pbs'
    jump, transport = connect()
    try:
        remote = shlex.quote(base)
        if args.action == 'probe':
            command(transport, 'hostname; qstat -Qf normal; qstat -Qf gdev; test -x /scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python && echo PYTHON_READY; qstat -u yguo017')
        elif args.action == 'stage':
            # Source files are bounded small control files; datasets never pass through login.
            names = FILES if args.variant == 'primary' else (
                'GLOBAL16_FREEZE.json', 'allocation_guard.py', 'controlled_system.py',
                'models.py', 'test_mechanism.py', 'run_experiment.py', 'write_global16_report.py', pbs_name)
            paths = [ROOT / name for name in names]
            if sum(path.stat().st_size for path in paths) > 200_000:
                raise RuntimeError('Source exceeds small control-file staging limit')
            command(transport, f'mkdir -p {remote}/source {remote}/runs')
            sftp = paramiko.SFTPClient.from_transport(transport)
            try:
                for path in paths:
                    target = f'{base}/source/' + ('FREEZE.json' if path.name == 'GLOBAL16_FREEZE.json' else path.name)
                    with sftp.open(target + '.uploading', 'wb') as stream:
                        stream.write(path.read_bytes().replace(b'\r\n', b'\n'))
                    sftp.posix_rename(target + '.uploading', target)
            finally:
                sftp.close()
            print(f'Staged {len(paths)} small control files')
        elif args.action == 'submit':
            if tracker.exists():
                raise RuntimeError('A tracked job already exists; inspect it instead of duplicate submission')
            job = command(transport, f'cd {remote}/source && qsub {pbs_name}').strip()
            tracker.write_text(json.dumps({'job_id':job,'remote_root':base}, indent=2)+'\n')
        else:
            job = args.job or json.loads(tracker.read_text())['job_id']
            if not all(c.isalnum() or c in '._-' for c in job):
                raise ValueError('Invalid PBS job id')
            output = f'{base}/runs/{job}'
            if args.action == 'status':
                command(transport, f'qstat -xf {shlex.quote(job)}', compact=True)
            elif args.action == 'progress':
                command(transport, f'if test -f {shlex.quote(output)}/mechanism_tests.json; then head -n 6 {shlex.quote(output)}/mechanism_tests.json; fi; if test -f {shlex.quote(output)}/progress.json; then cat {shlex.quote(output)}/progress.json; fi; if test -f {shlex.quote(output)}/job_status.txt; then cat {shlex.quote(output)}/job_status.txt; fi; if test -f {shlex.quote(output)}/job.log; then tail -n 16 {shlex.quote(output)}/job.log; fi')
            else:
                sftp = paramiko.SFTPClient.from_transport(transport)
                local = ROOT / 'results' / job
                local.mkdir(parents=True, exist_ok=True)
                try:
                    names = ('summary.json', 'DONE.json', 'mechanism_tests.json', 'job_status.txt', 'execution_identity.json', 'gpu_info.txt', 'FREEZE.json', 'REPORT.zh.md', 'DECISION.json', 'job.log')
                    if args.variant == 'global16':
                        names += ('global16_comparison.json',)
                    for name in names:
                        path = f'{output}/{name}'
                        if sftp.stat(path).st_size >= 200 * 1024:
                            raise RuntimeError('Result exceeds small control-file retrieval limit')
                        sftp.get(path, str(local / name))
                    try:
                        if sftp.stat(f'{output}/h10_comparison.svg').st_size < 200 * 1024:
                            sftp.get(f'{output}/h10_comparison.svg', str(local / 'h10_comparison.svg'))
                    except FileNotFoundError:
                        pass
                    print(f'Retrieved small result summaries to {local}')
                finally:
                    sftp.close()
    finally:
        transport.close()
        if jump is not None:
            jump.close()


if __name__ == '__main__':
    main()

"""Bounded small-file control; recover durable submission intents before retrying."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'fastwam-a4-phases12'
sys.path.insert(0, str(HERE.parent / 'fastwam-libero-plus-pilot'))
import remote
spec = importlib.util.spec_from_file_location('previous_phase_control', OLD / 'control.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
ROOT = '/scratch/users/ntu/yguo017/fastwam-smooth-vq-phases12-20261006'
FILES = ('protocol.json', 'environment.sh', 'run.pbs', 'smooth_vq.py', 'test_smooth_vq.py', 'run_validation.py', 'analysis.py', 'analysis.pbs')
HANDLE = re.compile(r'^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def scheduler(transport, job):
    if not HANDLE.fullmatch(job):
        raise ValueError('Invalid job handle')
    code, output = remote.command(transport, f'qstat -xf {shlex.quote(job)}', timeout=35)
    fields = {}
    for key in ('job_state', 'Exit_status', 'Job_Name', 'exec_host', 'comment', 'resources_used.walltime'):
        match = re.search(r'^\s*' + re.escape(key) + r'\s*=\s*(.*)$', output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return {'job_id': job, 'query_rc': code, **fields}


def verified_success(transport, phase):
    receipt = json.loads((HERE / f'{phase}_handle.json').read_text(encoding='utf-8'))
    job = receipt['job_id']
    observed = scheduler(transport, job)
    if observed['query_rc'] or observed['job_state'] not in ('F', 'X') or observed['Exit_status'] != '0':
        raise RuntimeError(f'{phase} is not verified successful: {observed}')
    artifact = f'{ROOT}/artifacts/{job}'
    code, _ = remote.command(transport, f'test -f {shlex.quote(artifact)}/PIPELINE_COMPLETE && '
                             f'test "$(cat {shlex.quote(artifact)}/exit_code.txt)" = 0 && '
                             f'test -s {ROOT}/results/{phase}/summary.json', timeout=25)
    if code:
        raise RuntimeError(f'{phase} completion artifacts are missing')
    return observed


def submit(transport, phase, attempt):
    if phase == 'full':
        verified_success(transport, 'preflight')
    elif phase == 'analysis':
        verified_success(transport, 'full')
    path = HERE / f'{phase}_intent_{attempt}.json'
    receipt = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if receipt.get('job_id'):
        save(HERE / f'{phase}_handle.json', receipt)
        print('CONFIRMED_HANDLE', receipt['job_id'])
        return
    name = f'fwsv1006{phase[0]}{attempt}'
    code, output = remote.command(transport, f'qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1', timeout=35)
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError('PBS history unavailable; submission not attempted')
    matches = output.split()
    if matches:
        if len(matches) != 1 or not HANDLE.fullmatch(matches[0]):
            raise RuntimeError('Ambiguous submission history; submission not attempted')
        receipt.update(job_id=matches[0], status='recovered', phase=phase, job_name=name)
    else:
        if receipt:
            raise RuntimeError('Unconfirmed durable intent exists; recover it before another attempt')
        receipt = {'phase': phase, 'attempt': attempt, 'job_name': name,
                   'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'intent_before_qsub'}
        save(path, receipt)
        hours = {'preflight': '02:00:00', 'full': '08:00:00', 'analysis': '00:10:00'}[phase]
        script = 'analysis.pbs' if phase == 'analysis' else 'run.pbs'
        command = f'qsub -q normal -N {shlex.quote(name)} -l walltime={hours} -v FW_PHASE={phase} {ROOT}/{script}'
        code, output = remote.command(transport, command, timeout=40)
        if code or not HANDLE.fullmatch(output.strip()):
            receipt.update(status='unknown_or_rejected', message=output[:2000])
            save(path, receipt)
            raise RuntimeError('qsub outcome unknown/rejected; do not repeat this attempt blindly')
        receipt.update(job_id=output.strip(), status='confirmed')
    save(path, receipt)
    save(HERE / f'{phase}_handle.json', receipt)
    print('CONFIRMED_HANDLE', receipt['job_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'pull', 'verify'))
    parser.add_argument('--phase', choices=('preflight', 'full', 'analysis'))
    parser.add_argument('--attempt', default='a')
    parser.add_argument('--job')
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9]{0,5}', args.attempt):
        parser.error('Use a short lowercase attempt')
    if args.action in ('submit', 'verify') and not args.phase:
        parser.error('--phase required')
    jump = transport = None
    try:
        if args.native:
            if args.action not in ('submit', 'status', 'verify'):
                parser.error('--native only supports lightweight command actions')
            remote.command = previous.native_command
        else:
            jump, transport = remote.connect()
        if args.action == 'upload':
            code, _ = remote.command(transport, f'mkdir -p {ROOT}', timeout=25)
            if code:
                raise RuntimeError('Unable to create control directory')
            with remote.sftp_client(transport) as sftp:
                for name in args.files or FILES:
                    if name not in FILES:
                        raise ValueError('Unknown control file')
                    data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise ValueError('Control file exceeds small-transfer limit')
                    with sftp.file(f'{ROOT}/{name}', 'wb') as stream:
                        stream.write(data)
                    print('UPLOADED', name, len(data))
        elif args.action == 'submit':
            submit(transport, args.phase, args.attempt)
        elif args.action == 'verify':
            value = verified_success(transport, args.phase)
            save(HERE / f'{args.phase}_terminal_evidence.json', value)
            print(json.dumps(value))
        elif args.action == 'status':
            job = args.job or json.loads((HERE / f'{args.phase}_handle.json').read_text())['job_id']
            observed = scheduler(transport, job)
            save(HERE / 'last_scheduler_snapshot.json', observed)
            print(json.dumps(observed))
            if observed['job_state'] in ('R', 'F', 'X'):
                code, output = remote.command(transport, f'tail -n 30 {ROOT}/artifacts/{shlex.quote(job)}/job.log', timeout=25)
                print('LOG_TAIL_RC', code)
                print(output)
        else:
            if not args.files:
                parser.error('--files required')
            with remote.sftp_client(transport) as sftp:
                for name in args.files:
                    relative = Path(name)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Invalid artifact path')
                    source = f'{ROOT}/{relative.as_posix()}'
                    if sftp.stat(source).st_size > remote.LIMIT:
                        raise ValueError('Only small summary/control files may be pulled')
                    with sftp.file(source, 'rb') as stream:
                        data = stream.read()
                    target = HERE / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    print('PULLED', relative.as_posix(), len(data))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

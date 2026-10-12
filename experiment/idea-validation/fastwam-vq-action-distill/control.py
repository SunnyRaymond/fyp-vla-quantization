"""Reuse trusted bounded SSH control for the two-hour distillation job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
ROOT = '/scratch/users/ntu/yguo017/fastwam-vq-action-distill-20261011'
spec = importlib.util.spec_from_file_location('distill_trusted_control', HERE.parent / 'fastwam-vq-proxy-diagnostic/control.py')
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
trusted.HERE, trusted.ROOT = HERE, ROOT
trusted.FILES = ('protocol.json', 'control.py', 'environment.sh', 'gpu.pbs', 'sampler.py', 'train_banks.py',
                 'checkpointing.py', 'test_checkpointing.py', 'test_driver.py', 'run_distill.py')
remote, save, scheduler, read_remote_json = trusted.remote, trusted.save, trusted.scheduler, trusted.read_remote_json


def submit(transport, attempt, resume_job=None):
    trusted.protocol()
    if not re.fullmatch(r'[a-z][a-z0-9]{0,5}', attempt):
        raise ValueError('Invalid attempt label')
    if resume_job is not None and not trusted.HANDLE.fullmatch(resume_job):
        raise ValueError('Invalid resume job handle')
    if resume_job is not None:
        previous = scheduler(transport, resume_job)
        if previous['query_rc'] or previous['job_state'] not in ('F', 'X'):
            raise RuntimeError(f'Resume source is not a confirmed terminal job: {previous}')
    name = f'fwvkd1011{attempt}'
    intent, handle = HERE / f'intent_{attempt}.json', HERE / 'full_handle.json'
    receipt = json.loads(intent.read_text()) if intent.exists() else {}
    if receipt and receipt.get('resume_job') != resume_job:
        raise RuntimeError('Attempt intent belongs to a different resume job')
    if receipt.get('job_id'):
        if not trusted.HANDLE.fullmatch(receipt['job_id']):
            raise RuntimeError('Attempt intent contains an invalid PBS handle')
        save(handle, receipt)
        print('CONFIRMED_HANDLE', receipt['job_id'])
        return
    rc, output = remote.command(transport, f'qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1', timeout=35)
    if rc not in (0, 1) or (rc == 1 and output.strip()):
        raise RuntimeError('Scheduler submission history is unavailable')
    matches = output.split()
    if matches:
        if len(matches) != 1 or not trusted.HANDLE.fullmatch(matches[0]):
            raise RuntimeError('Ambiguous existing submission')
        receipt.update(job_id=matches[0], status='recovered', job_name=name, resume_job=resume_job)
    else:
        if receipt:
            raise RuntimeError('Unconfirmed intent exists; recover before resubmitting')
        receipt = {'status': 'intent_before_qsub', 'job_name': name, 'attempt': attempt,
                   'created_utc': datetime.now(timezone.utc).isoformat(),
                   'walltime': '02:00:00', 'queue': 'normal', 'project': 'personal-yguo017',
                   'select': 'select=1:ncpus=16:mem=110gb:ngpus=1', 'remote_root': ROOT,
                   'resume_job': resume_job}
        save(intent, receipt)
        resume_arg = f'-v {shlex.quote("FW_RESUME_JOB=" + resume_job)} ' if resume_job else ''
        command = f'qsub {resume_arg}-N {shlex.quote(name)} {shlex.quote(ROOT + "/gpu.pbs")}'
        rc, output = remote.command(transport, command, timeout=40)
        if rc or not trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status='unknown_or_rejected', message=output[:2000])
            save(intent, receipt)
            raise RuntimeError('qsub outcome unknown/rejected; inspect intent before retry')
        receipt.update(job_id=output.strip(), status='confirmed')
    save(intent, receipt)
    save(handle, receipt)
    print('CONFIRMED_HANDLE', receipt['job_id'])


def _checkpoint_evidence(transport, job, *, require_resume_ready=False, require_checkpoint_ready=True):
    artifact = f'{ROOT}/artifacts/{job}'
    result = f'{ROOT}/results/full-{job}'
    checks = [f'test -s {shlex.quote(result + "/checkpoint.pt")}',
              f'test -s {shlex.quote(result + "/checkpoint_receipt.json")}']
    if require_checkpoint_ready:
        checks.append(f'test -f {shlex.quote(result + "/CHECKPOINT_READY")}')
    if require_resume_ready:
        checks.extend((
            f'test -f {shlex.quote(artifact + "/RESUME_READY")}',
            f'test "$(cat {shlex.quote(artifact + "/exit_code.txt")})" = 0',
        ))
    rc, _ = remote.command(transport, ' && '.join(checks), timeout=25)
    if rc:
        raise RuntimeError('Required checkpoint files or continuation markers are missing')
    receipt = read_remote_json(transport, result + '/checkpoint_receipt.json')
    if (receipt.get('schema_version') != 2 or receipt.get('job_id') != job
            or receipt.get('target_linears') != 614 or not isinstance(receipt.get('phase'), str)
            or not receipt['phase'] or not isinstance(receipt.get('updates_completed'), int)
            or not isinstance(receipt.get('epochs_completed'), int)
            or not 0 <= receipt['updates_completed'] <= 16 or not 0 <= receipt['epochs_completed'] <= 2):
        raise RuntimeError('Checkpoint receipt identity or progress is invalid')
    return receipt


def _walltime_seconds(value):
    if not value:
        return None
    try:
        parts = [int(part) for part in value.split(':')]
    except ValueError:
        return None
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 4:
        days, hours, minutes, seconds = parts
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    return None


def _next_continue_attempt(resume_job):
    existing = []
    for path in HERE.glob('intent_c*.json'):
        if path.stat().st_size > trusted.LIMIT:
            raise RuntimeError(f'Attempt intent exceeds control-file limit: {path.name}')
        value = json.loads(path.read_text())
        if value.get('resume_job') == resume_job:
            existing.append(path.stem.removeprefix('intent_'))
    if len(existing) > 1:
        raise RuntimeError('Multiple continue intents already refer to this job')
    if existing:
        return existing[0]
    index = 1
    while (HERE / f'intent_c{index}.json').exists():
        index += 1
    return f'c{index}'


def continue_job(transport):
    path = HERE / 'full_handle.json'
    if not path.is_file() or path.stat().st_size > trusted.LIMIT:
        raise RuntimeError('No bounded current full-job handle is available')
    job = json.loads(path.read_text()).get('job_id')
    if not isinstance(job, str) or not trusted.HANDLE.fullmatch(job):
        raise RuntimeError('Current full-job handle is invalid')
    observed = scheduler(transport, job)
    if observed['query_rc'] or observed['job_state'] not in ('F', 'X'):
        raise RuntimeError(f'Current full job is not terminal: {observed}')

    hard_limit = (observed['Exit_status'] == '271'
                  or (_walltime_seconds(observed.get('resources_used.walltime')) or 0) >= 7200)
    if observed['Exit_status'] == '0':
        artifact = f'{ROOT}/artifacts/{job}'
        rc, _ = remote.command(
            transport,
            f'test -f {shlex.quote(artifact + "/RESUME_READY")} && '
            f'test "$(cat {shlex.quote(artifact + "/exit_code.txt")})" = 0', timeout=25,
        )
        if rc and not hard_limit:
            raise RuntimeError('Clean continuation requires RESUME_READY and a zero exit receipt')
        if rc == 0:
            summary = read_remote_json(transport, artifact + '/summary.json')
            receipt = _checkpoint_evidence(transport, job, require_resume_ready=True)
            if (summary.get('status') != 'resumable' or summary.get('job_id') != job
                    or summary.get('target_linears') != 614):
                raise RuntimeError('Clean continuation requires a matching resumable summary')
        else:
            receipt = _checkpoint_evidence(transport, job, require_checkpoint_ready=False)
            summary_path = f'{ROOT}/artifacts/{job}/summary.json'
            summary_rc, _ = remote.command(transport, f'test -s {shlex.quote(summary_path)}', timeout=25)
            if summary_rc == 0 and read_remote_json(transport, summary_path).get('status') == 'complete':
                raise RuntimeError('A complete summary already exists; refusing to submit duplicate work')
    else:
        if not hard_limit:
            raise RuntimeError(f'Automatic continuation is blocked for this failure: {observed}')
        receipt = _checkpoint_evidence(transport, job, require_checkpoint_ready=False)
        summary_path = f'{ROOT}/artifacts/{job}/summary.json'
        rc, _ = remote.command(transport, f'test -s {shlex.quote(summary_path)}', timeout=25)
        if rc == 0 and read_remote_json(transport, summary_path).get('status') == 'complete':
            raise RuntimeError('A complete summary already exists; refusing to submit duplicate work')

    attempt = _next_continue_attempt(job)
    print('CONTINUE_FROM', job, 'ATTEMPT', attempt, 'CHECKPOINT_UPDATES', receipt['updates_completed'])
    submit(transport, attempt, resume_job=job)


def verify(transport, job):
    observed = scheduler(transport, job)
    if observed['query_rc'] or observed['job_state'] not in ('F', 'X') or observed['Exit_status'] != '0':
        raise RuntimeError(f'Not a successful terminal job: {observed}')
    artifact = f'{ROOT}/artifacts/{job}'
    summary = read_remote_json(transport, artifact + '/summary.json')
    if summary.get('job_id') != job or summary.get('target_linears') != 614:
        raise RuntimeError('Summary identity or target coverage does not match the terminal job')
    if summary.get('status') == 'resumable':
        checkpoint = _checkpoint_evidence(transport, job, require_resume_ready=True)
        evidence = {**observed, 'status': 'resumable', 'summary': summary,
                    'checkpoint_receipt': checkpoint, 'resume_ready': True}
        save(HERE / 'full_terminal_evidence.json', evidence)
        print(json.dumps(evidence))
        return evidence
    if summary.get('status') != 'complete':
        raise RuntimeError(f'Unexpected terminal summary status: {summary.get("status")}')
    rc, _ = remote.command(transport, f'test -f {shlex.quote(artifact + "/PIPELINE_COMPLETE")} && '
                           f'test "$(cat {shlex.quote(artifact + "/exit_code.txt")})" = 0', timeout=25)
    if rc:
        raise RuntimeError('Missing complete marker or zero exit receipt')
    if (summary.get('completed_epochs') != 2 or summary.get('updates') != 16
            or summary.get('test', {}).get('case_count') != 22
            or summary.get('test', {}).get('context_count') != 44
            or summary.get('test_queries_before_freeze') != 0):
        raise RuntimeError('Complete summary does not meet the 2-epoch/16-update/22-case/44-context gates')
    evidence = {**observed, 'status': 'complete', 'summary': summary, 'success_marker': True}
    save(HERE / 'full_terminal_evidence.json', evidence)
    print(json.dumps(evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('upload', 'submit', 'continue', 'status', 'verify', 'pull'))
    parser.add_argument('--attempt', default='a')
    parser.add_argument('--resume-job')
    parser.add_argument('--files', nargs='+')
    args = parser.parse_args()
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        if args.action == 'upload':
            trusted.upload(transport, args.files)
        elif args.action == 'submit':
            submit(transport, args.attempt, resume_job=args.resume_job)
        elif args.action == 'continue':
            continue_job(transport)
        else:
            job = json.loads((HERE / 'full_handle.json').read_text())['job_id']
            if args.action == 'status':
                observed = scheduler(transport, job)
                save(HERE / 'last_scheduler_snapshot.json', observed)
                print(json.dumps(observed))
                if observed['job_state'] in ('R', 'F', 'X'):
                    path = f'{ROOT}/artifacts/{job}/job.log'
                    rc, output = remote.command(transport, f'test ! -f {shlex.quote(path)} || tail -c 7000 {shlex.quote(path)}', timeout=25)
                    print('LOG_TAIL_RC', rc, output)
            elif args.action == 'verify':
                verify(transport, job)
            else:
                if not args.files:
                    raise ValueError('--files required for pull')
                trusted.pull(transport, args.files)
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

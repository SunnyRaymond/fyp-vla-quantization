"""Bounded control for the ten-minute CPU numerical preflight."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

import control as ctl

HERE, ROOT = ctl.HERE, ctl.ROOT


def upload(transport):
    ctl.protocol()
    ctl.trusted.FILES = ('cpu.pbs', 'run_cpu_preflight.py')
    ctl.trusted.upload(transport, None)
    # The only changed helper lines add an optional CPU device to existing assertions.
    local = (HERE / 'micro_codebook.py').read_bytes()
    target = ROOT + '/micro_codebook.py'
    with ctl.remote.sftp_client(transport) as sftp:
        if sftp.stat(target).st_size > ctl.trusted.LIMIT:
            raise RuntimeError('Oversized existing helper')
        with sftp.file(target, 'rb') as stream:
            prior = stream.read(ctl.trusted.LIMIT + 1)
        if prior != local:
            old = "def self_check():\n    device = torch.device('cuda')"
            new = "def self_check(device='cuda'):\n    device = torch.device(device)"
            if prior.decode('utf-8').replace(old, new).encode('utf-8') != local:
                raise RuntimeError('Refusing a change beyond the optional self-check device')
            backup = HERE / 'micro_codebook.uploaded-v1.py'
            if not backup.exists():
                backup.write_bytes(prior)
            temporary = ROOT + '/.micro_codebook.cpu-support.tmp'
            with sftp.file(temporary, 'wb') as stream:
                stream.write(local)
            sftp.posix_rename(temporary, target)
    print('CPU_PREFLIGHT_FILES_READY')


def submit(transport):
    intent = HERE / 'cpu_intent_a.json'
    name = 'fwvmcp1011a'
    if intent.exists():
        receipt = json.loads(intent.read_text())
        if receipt.get('job_id'):
            ctl.save(HERE / 'cpu_handle.json', receipt)
            print('CONFIRMED_CPU_HANDLE', receipt['job_id'])
            return
        matches = ctl.query_named_submission(transport, name)
        if len(matches) != 1:
            raise RuntimeError('Unconfirmed CPU intent; inspect before retry')
        receipt.update(job_id=matches[0], status='recovered')
    else:
        matches = ctl.query_named_submission(transport, name)
        if len(matches) > 1:
            raise RuntimeError('Ambiguous CPU submission history')
        receipt = {'job_name': name, 'walltime': '00:10:00', 'phase': 'cpu_preflight',
                   'created_utc': datetime.now(timezone.utc).isoformat(), 'remote_root': ROOT}
        if matches:
            receipt.update(job_id=matches[0], status='recovered')
        else:
            receipt['status'] = 'intent_before_qsub'
            ctl.save(intent, receipt)
            rc, output = ctl.remote.command(transport, 'qsub -N ' + shlex.quote(name) + ' ' + shlex.quote(ROOT + '/cpu.pbs'), timeout=40)
            if rc or not ctl.trusted.HANDLE.fullmatch(output.strip()):
                receipt['status'] = 'unknown_or_rejected'
                ctl.save(intent, receipt)
                raise RuntimeError('CPU qsub outcome unknown/rejected; no automatic retry')
            receipt.update(job_id=output.strip(), status='confirmed')
    ctl.save(intent, receipt)
    ctl.save(HERE / 'cpu_handle.json', receipt)
    print('CONFIRMED_CPU_HANDLE', receipt['job_id'])


def inspect(transport, verify=False):
    job = json.loads((HERE / 'cpu_handle.json').read_text())['job_id']
    observed = ctl.scheduler(transport, job)
    ctl.save(HERE / 'cpu_last_scheduler_snapshot.json', observed)
    artifact = ROOT + '/artifacts/' + job
    rc, _ = ctl.remote.command(transport, 'test -f ' + shlex.quote(artifact + '/summary.json'), timeout=25)
    summary = ctl.read_remote_json(transport, artifact + '/summary.json') if rc == 0 else None
    if verify:
        if observed['query_rc'] or observed['job_state'] not in ('F', 'X') or observed['Exit_status'] != '0':
            raise RuntimeError('CPU job is not successfully terminal')
        rc, _ = ctl.remote.command(transport, 'test -f ' + shlex.quote(artifact + '/PIPELINE_COMPLETE') + ' && test "$(cat ' + shlex.quote(artifact + '/exit_code.txt') + ')" = 0', timeout=25)
        if rc or not summary or summary.get('status') != 'complete' or summary.get('phase') != 'cpu_preflight' or summary.get('job_id') != job or len(summary.get('checks', [])) != 4:
            raise RuntimeError('CPU preflight completion evidence is incomplete')
        ctl.save(HERE / 'cpu_terminal_evidence.json', {**observed, 'summary': summary, 'success_marker': True})
    print(json.dumps({'scheduler': observed, 'summary': summary}))
    if not verify:
        rc, output = ctl.remote.command(transport, 'if test -f ' + shlex.quote(artifact + '/job.log') + '; then tail -c 4000 ' + shlex.quote(artifact + '/job.log') + '; fi', timeout=25)
        print('CPU_LOG_TAIL_RC', rc, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'verify'))
    args = parser.parse_args()
    jump = transport = None
    try:
        jump, transport = ctl.remote.direct_connect()
        if args.action == 'upload':
            upload(transport)
        elif args.action == 'submit':
            submit(transport)
        else:
            inspect(transport, verify=args.action == 'verify')
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()

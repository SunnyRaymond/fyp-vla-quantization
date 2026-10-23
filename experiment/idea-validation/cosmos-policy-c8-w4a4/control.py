"""Bounded control files and durable PBS handles; workloads run only on compute nodes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cosmos_remote', HERE.parent/'fastwam-libero-plus-pilot'/'remote.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)
PROTOCOL = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
ROOT = PROTOCOL['remote_root']
JOBS = {'preparation': ('cosp1015prep', 'prepare.pbs'), 'smoke': ('cosp1015smoke', 'smoke.pbs'),
        'checks': ('cosp1015check', 'checks.pbs'),
        'spinquant': ('cosp1015spin', 'spinquant.pbs'), 'vq': ('cosp1015vq', 'vq.pbs'),
        'evaluation': ('cosp1015eval', 'evaluate.pbs')}
HANDLE = re.compile(r'^\d+\.[A-Za-z0-9_.-]+$')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def status(transport, job):
    if not HANDLE.fullmatch(job):
        raise ValueError('Invalid PBS handle')
    rc, output = remote.command(transport, 'qstat -xf '+shlex.quote(job), timeout=35)
    fields = {'job_id': job, 'query_rc': rc, 'observed_utc': datetime.now(timezone.utc).isoformat()}
    for key in ('Job_Name', 'job_state', 'queue', 'exec_host', 'Exit_status', 'comment', 'resources_used.walltime'):
        match = re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(.*)$', output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return fields


def upload(transport):
    rc, output = remote.command(transport, 'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'))
    if rc:
        raise RuntimeError(output)
    names = []
    with remote.sftp_client(transport) as sftp:
        for path in HERE.iterdir():
            if path.suffix not in ('.py', '.pbs', '.sh', '.md') and path.name != 'protocol.json':
                continue
            data = path.read_bytes().replace(b'\r\n', b'\n')
            if len(data) > remote.LIMIT:
                raise RuntimeError('Not a small control file: '+path.name)
            target = ROOT+'/'+path.name
            with sftp.file(target+'.tmp', 'wb') as stream:
                stream.write(data)
            sftp.posix_rename(target+'.tmp', target)
            names.append(path.name)
    return {'uploaded': names, 'remote_root': ROOT}


def require_predecessors(transport, stage):
    required = {'smoke': ('preparation',), 'spinquant': ('preparation', 'checks'),
                'vq': ('preparation', 'checks'), 'evaluation': ('checks', 'smoke', 'spinquant', 'vq')}.get(stage, ())
    summaries = {}
    with remote.sftp_client(transport) as sftp:
        for predecessor in required:
            handle = json.loads((HERE/(predecessor+'_handle.json')).read_text())
            observed = status(transport, handle['job_id'])
            if (observed['query_rc'] != 0 or observed['job_state'] != 'F'
                    or observed['Exit_status'] != '0'):
                raise RuntimeError(predecessor+' has not completed successfully')
            if predecessor == 'preparation':
                path, marker = 'environment_ready.json', 'PREPARATION_COMPLETE'
            else:
                path = 'results/'+predecessor+'/summary.json'
                marker = 'CHECKS_COMPLETE' if predecessor == 'checks' else 'PIPELINE_COMPLETE'
            with sftp.file(ROOT+'/'+path, 'rb') as stream:
                data = stream.read(remote.LIMIT+1)
            if len(data) > remote.LIMIT:
                raise RuntimeError('Predecessor summary is unexpectedly large')
            summary = json.loads(data)
            expected_status = 'ready_for_gpu_smoke' if predecessor == 'preparation' else 'complete'
            if summary.get('status') != expected_status:
                raise RuntimeError(predecessor+' evidence is incomplete')
            sftp.stat(ROOT+'/artifacts/'+handle['job_id']+'/'+marker)
            summaries[predecessor] = summary
    if stage == 'evaluation':
        for calibration in ('spinquant', 'vq'):
            summary = summaries[calibration]
            if (summary.get('completed_updates') != summary.get('expected_updates')
                    or summary.get('bpw_match_within_5_percent') is not True
                    or summary.get('parity', {}).get('status') != 'passed'):
                raise RuntimeError('Calibration updates, BPW match, or endpoint parity are incomplete')


def submit(transport, stage, dependency=None, resume=False, repair_note=None):
    name, script = JOBS[stage]
    handle_file, intent_file = HERE/(stage+'_handle.json'), HERE/(stage+'_intent.json')
    current_handle_file = handle_file
    previous = None
    if handle_file.exists():
        previous = json.loads(handle_file.read_text())
        observed = status(transport, previous['job_id'])
        if not resume and not repair_note:
            return observed
        if observed['query_rc'] != 0 or observed['job_state'] not in ('F', 'X'):
            raise RuntimeError('The exact previous job is not verified terminal; refusing another submission')
        if observed['Job_Name'] != previous['job_name']:
            raise RuntimeError('Previous PBS job identity differs')
        if resume:
            if stage not in ('spinquant', 'vq', 'checks') or observed['Exit_status'] != '0':
                raise RuntimeError('Only completed checks or successfully checkpointed calibration may repeat')
            with remote.sftp_client(transport) as sftp:
                with sftp.file(ROOT+'/results/'+stage+'/summary.json', 'rb') as stream:
                    data = stream.read(remote.LIMIT+1)
                expected = 'complete' if stage == 'checks' else 'checkpointed'
                if len(data) > remote.LIMIT or json.loads(data).get('status') != expected:
                    raise RuntimeError('Previous stage is not a validated '+expected+' result')
                marker = 'CHECKS_COMPLETE' if stage == 'checks' else 'PIPELINE_CHECKPOINTED'
                sftp.stat(ROOT+'/artifacts/'+previous['job_id']+'/'+marker)
        elif observed['Exit_status'] in (None, '0'):
            raise RuntimeError('A repair retry requires a verified nonzero previous exit')
        sequence = int(previous.get('sequence', 0))+1
        save(HERE/f'{stage}_r{sequence-1}_handle.json', previous)
        handle_file = HERE/f'{stage}_r{sequence}_handle.json'
        intent_file = HERE/f'{stage}_r{sequence}_intent.json'
        name += 'r'+str(sequence)
    else:
        if resume or repair_note:
            raise RuntimeError('No original stage handle exists')
        sequence = 0
    if intent_file.exists():
        raise RuntimeError('Unresolved submission intent; inspect the exact existing job before retrying')
    require_predecessors(transport, stage)
    rc, output = remote.command(transport, 'qselect -u yguo017 -N '+shlex.quote(name), timeout=35)
    if output.strip():
        raise RuntimeError('A matching PBS job already exists: '+output.strip())
    if rc not in (0, 153):
        raise RuntimeError('Cannot establish existing job state')
    resources = PROTOCOL['resources'][stage]
    select = f"1:ncpus={resources['ncpus']}:mem={resources['memory_gb']}gb"
    if resources['gpus']:
        select += f":ngpus={resources['gpus']}"
    arguments = ['qsub', '-P', 'personal-yguo017', '-q', resources['queue'], '-N', name,
                 '-l', 'select='+select, '-l', 'walltime='+resources['walltime'], '-j', 'oe',
                 '-o', ROOT+'/artifacts/'+stage+'.pbs.out']
    if dependency:
        if not HANDLE.fullmatch(dependency):
            raise ValueError('Invalid dependency')
        arguments += ['-W', 'depend=afterok:'+dependency]
    arguments += [ROOT+'/'+script]
    intent = {'stage': stage, 'job_name': name, 'remote_root': ROOT, 'resources': resources,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'dependency': dependency,
              'sequence': sequence, 'previous_job_id': previous['job_id'] if previous else None,
              'repair_note': repair_note,
              'state': 'unresolved_until_qsub_receipt'}
    save(intent_file, intent)
    with remote.sftp_client(transport) as sftp:
        with sftp.file(ROOT+'/control/'+intent_file.name, 'wb') as stream:
            stream.write((json.dumps(intent, indent=2)+'\n').encode())
    rc, output = remote.command(transport, ' '.join(shlex.quote(a) for a in arguments), timeout=40)
    lines = output.strip().splitlines()
    if rc or not lines or not HANDLE.fullmatch(lines[-1].strip()):
        raise RuntimeError('Submission outcome needs reconciliation: '+output[-1000:])
    result = {**intent, 'job_id': lines[-1].strip(), 'state': 'submitted'}
    save(handle_file, result)
    save(current_handle_file, result)
    with remote.sftp_client(transport) as sftp:
        with sftp.file(ROOT+'/control/'+handle_file.name, 'wb') as stream:
            stream.write((json.dumps(result, indent=2)+'\n').encode())
        if current_handle_file.name != handle_file.name:
            with sftp.file(ROOT+'/control/'+current_handle_file.name, 'wb') as stream:
                stream.write((json.dumps(result, indent=2)+'\n').encode())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'read', 'pull'))
    parser.add_argument('--stage', choices=tuple(JOBS), default='preparation')
    parser.add_argument('--dependency')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--repair-note')
    parser.add_argument('--files', nargs='*', default=[])
    args = parser.parse_args()
    jump, transport = remote.connect()
    try:
        if args.action == 'upload':
            result = upload(transport)
        elif args.action == 'submit':
            result = submit(transport, args.stage, args.dependency, args.resume, args.repair_note)
        elif args.action == 'status':
            handle = json.loads((HERE/(args.stage+'_handle.json')).read_text())
            result = status(transport, handle['job_id'])
            save(HERE/(args.stage+'_status.json'), result)
        else:
            result = {}
            with remote.sftp_client(transport) as sftp:
                for name in args.files:
                    parts = Path(name).parts
                    if Path(name).is_absolute() or '..' in parts:
                        raise ValueError('Only task-relative small files are allowed')
                    with sftp.file(ROOT+'/'+name.replace('\\', '/'), 'rb') as stream:
                        data = stream.read(remote.LIMIT+1)
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('File exceeds small control limit: '+name)
                    if args.action == 'pull':
                        path = HERE/'retrieved'/name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    result[name] = data.decode('utf-8', errors='replace')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

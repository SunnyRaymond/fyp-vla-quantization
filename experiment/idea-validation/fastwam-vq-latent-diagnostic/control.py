"""Small-file PBS submission, resumable shard monitoring, and bounded evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
ROOT = '/scratch/users/ntu/yguo017/fastwam-vq-latent-diagnostic-20261013'
spec = importlib.util.spec_from_file_location('latent_trusted_control', HERE.parent / 'fastwam-vq-proxy-diagnostic/control.py')
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
trusted.HERE, trusted.ROOT = HERE, ROOT
trusted.FILES = ('protocol.json', 'control.py', 'environment.sh', 'gpu.pbs', 'run_future.py',
                 'future_adapter.py', 'metrics.py', 'checkpointing.py', 'cpu.pbs', 'analyze_results.py',
                 'cache_preflight.py', 'cache_preflight.pbs', 'test_analysis_scope.py')
remote, save, scheduler, read_remote_json = trusted.remote, trusted.save, trusted.scheduler, trusted.read_remote_json


def handle(shard):
    value = json.loads((HERE / f'shard_{shard}_handle.json').read_text())
    if not trusted.HANDLE.fullmatch(value['job_id']) or value['shard'] != shard:
        raise RuntimeError('Invalid current shard handle')
    return value


def inspect(transport, shard, tail=False):
    value = handle(shard)
    job = value['job_id']
    observed = scheduler(transport, job)
    save(HERE / f'shard_{shard}_scheduler.json', observed)
    if observed['query_rc']:
        return observed
    artifact = f'{ROOT}/artifacts/{job}'
    if observed['job_state'] in ('F', 'X'):
        if observed['Exit_status'] != '0':
            observed['markers_valid'] = False
            save(HERE / f'shard_{shard}_terminal.json', observed)
            if tail:
                rc, output = remote.command(transport, f'test ! -f {shlex.quote(artifact + "/job.log")} || '
                                            f'tail -c 6000 {shlex.quote(artifact + "/job.log")}', timeout=25)
                print('LOG_TAIL', shard, rc, output)
            return observed
        observed['summary'] = read_remote_json(transport, artifact + '/summary.json')
        summary = observed['summary']
        if (summary['job_id'] != job or summary['shard'] != shard
                or summary['protocol'] != 'fastwam-vq-latent-diagnostic-v1'):
            raise RuntimeError('Terminal summary identity mismatch')
        marker = 'PIPELINE_COMPLETE' if summary['status'] == 'complete' else 'RESUME_READY'
        rc, _ = remote.command(transport, f'test -f {shlex.quote(artifact + "/" + marker)} && '
                               f'test "$(cat {shlex.quote(artifact + "/exit_code.txt")})" = 0', timeout=25)
        observed['markers_valid'] = rc == 0
        save(HERE / f'shard_{shard}_terminal.json', observed)
    if tail and observed['job_state'] in ('R', 'F', 'X'):
        rc, output = remote.command(transport, f'test ! -f {shlex.quote(artifact + "/job.log")} || '
                                    f'tail -c 6000 {shlex.quote(artifact + "/job.log")}', timeout=25)
        print('LOG_TAIL', shard, rc, output)
    return observed


def submit(transport, shard, attempt, resume=None):
    trusted.protocol()
    if not attempt.isalnum() or not attempt.islower() or len(attempt) > 6:
        raise ValueError('Use a short lowercase attempt label')
    intent = HERE / f'shard_{shard}_intent_{attempt}.json'
    name = f'fwz1013s{shard}{attempt}'
    receipt = json.loads(intent.read_text()) if intent.exists() else {}
    if receipt and receipt.get('resume_job') != resume:
        raise RuntimeError('Submission intent belongs to another resume source')
    if receipt.get('job_id'):
        save(HERE / f'shard_{shard}_handle.json', receipt)
        return receipt
    rc, output = remote.command(transport, f'qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1', timeout=35)
    if rc not in (0, 1) or (rc == 1 and output.strip()):
        raise RuntimeError('Submission history unavailable')
    matches = output.split()
    if matches:
        if len(matches) != 1 or not trusted.HANDLE.fullmatch(matches[0]):
            raise RuntimeError('Ambiguous matching PBS job')
        receipt.update(job_id=matches[0], status='recovered', shard=shard, job_name=name, resume_job=resume)
    else:
        if receipt:
            raise RuntimeError('Unknown submission outcome; recover the existing intent before retry')
        if resume is None and (HERE / f'shard_{shard}_handle.json').exists():
            raise RuntimeError('A shard handle already exists; use continue after terminal validation')
        receipt = {'status': 'intent_before_qsub', 'shard': shard, 'job_name': name, 'attempt': attempt,
                   'created_utc': datetime.now(timezone.utc).isoformat(), 'remote_root': ROOT,
                   'resume_job': resume, 'walltime': '02:00:00', 'queue': 'normal',
                   'project': 'personal-yguo017', 'select': 'select=1:ncpus=16:mem=110gb:ngpus=1'}
        save(intent, receipt)
        variables = f'FW_SHARD={shard}' + (f',FW_RESUME_JOB={resume}' if resume else '')
        rc, output = remote.command(transport, f'qsub -v {shlex.quote(variables)} -N {shlex.quote(name)} '
                                    f'{shlex.quote(ROOT + "/gpu.pbs")}', timeout=40)
        if rc or not trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status='unknown_or_rejected', message=output[:1200])
            save(intent, receipt)
            raise RuntimeError('Submission outcome unknown/rejected; no duplicate retry')
        receipt.update(job_id=output.strip(), status='confirmed')
    save(intent, receipt)
    save(HERE / f'shard_{shard}_handle.json', receipt)
    return receipt


def continue_job(transport, shard):
    observed = inspect(transport, shard)
    if (observed['query_rc'] or observed['job_state'] not in ('F', 'X')
            or observed['Exit_status'] != '0' or not observed['markers_valid']
            or observed['summary']['status'] != 'resumable'):
        raise RuntimeError('Continuation requires a matching successful resumable terminal job')
    index = 1
    while (HERE / f'shard_{shard}_intent_c{index}.json').exists():
        value = json.loads((HERE / f'shard_{shard}_intent_c{index}.json').read_text())
        if value['resume_job'] == observed['job_id']:
            return submit(transport, shard, f'c{index}', observed['job_id'])
        index += 1
    return submit(transport, shard, f'c{index}', observed['job_id'])


def verify(transport, shard):
    observed = inspect(transport, shard)
    summary = observed.get('summary', {})
    if (observed['query_rc'] or observed['job_state'] not in ('F', 'X') or observed['Exit_status'] != '0'
            or not observed.get('markers_valid') or summary.get('status') != 'complete'
            or summary.get('context_count') != 30 or summary.get('native_context_queries') != 90
            or summary.get('teacher_future_queries') != 60 or summary.get('selected_epoch') != 2
            or summary.get('target_linears') != 614
            or set(summary.get('self_future_parity', {})) != {'bf16', 'parent', 'distilled'}):
        raise RuntimeError('Full shard completion gates are not satisfied')
    return observed


def small_cpu_submit(transport, prefix, name, script, variables=None):
    path = HERE / f'{prefix}_handle.json'
    if path.exists():
        return json.loads(path.read_text())
    intent = HERE / f'{prefix}_intent.json'
    receipt = json.loads(intent.read_text()) if intent.exists() else {}
    if receipt and receipt.get('variables') != variables:
        raise RuntimeError('CPU submission intent belongs to another analysis scope')
    rc, output = remote.command(transport, 'qselect -x -u yguo017 -N ' + shlex.quote(name) + ' 2>&1', timeout=35)
    if rc not in (0, 1) or (rc == 1 and output.strip()):
        raise RuntimeError('Analysis submission history unavailable')
    matches = output.split()
    if matches:
        if len(matches) != 1 or not trusted.HANDLE.fullmatch(matches[0]):
            raise RuntimeError('Ambiguous analysis job')
        receipt.update(job_id=matches[0], status='recovered')
    else:
        if receipt:
            raise RuntimeError('Unconfirmed analysis intent; recover it without duplicate submission')
        receipt = {'status': 'intent_before_qsub', 'job_name': name, 'remote_root': ROOT,
                   'variables': variables}
        save(intent, receipt)
        command = 'qsub -N ' + shlex.quote(name)
        if variables:
            command += ' -v ' + shlex.quote(variables)
        rc, output = remote.command(transport, command + ' ' + shlex.quote(ROOT + '/' + script), timeout=40)
        if rc or not trusted.HANDLE.fullmatch(output.strip()):
            raise RuntimeError('Analysis submission outcome unknown/rejected')
        receipt.update(job_id=output.strip(), status='confirmed')
    save(intent, receipt)
    save(path, receipt)
    return receipt


def analysis_prefix(shard=None):
    return 'analysis' if shard is None else f'analysis_shard_{shard}'


def analysis_upload(transport, shard=None):
    for selected in (0, 1) if shard is None else (shard,):
        verify(transport, selected)
    names = ('control.py', 'analyze_results.py', 'cpu.pbs', 'test_analysis_scope.py')
    with remote.sftp_client(transport) as sftp:
        for name in names:
            data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
            if len(data) > trusted.LIMIT:
                raise RuntimeError('Analysis control file exceeds 256 KiB')
            with sftp.file(f'{ROOT}/{name}', 'wb') as stream:
                stream.write(data)
            print('ANALYSIS_UPLOADED', name, len(data))


def analysis_submit(transport, shard=None):
    for selected in (0, 1) if shard is None else (shard,):
        verify(transport, selected)
    name = 'fwz1013cpu' if shard is None else f'fwz1013p{shard}'
    variables = None if shard is None else f'FW_ANALYSIS_SHARD={shard}'
    return small_cpu_submit(transport, analysis_prefix(shard), name, 'cpu.pbs', variables)


def cpu_status(transport, prefix, require_complete=False):
    job = json.loads((HERE / f'{prefix}_handle.json').read_text())['job_id']
    observed = scheduler(transport, job)
    save(HERE / f'{prefix}_scheduler.json', observed)
    if require_complete:
        if observed['query_rc'] or observed['job_state'] not in ('F', 'X') or observed['Exit_status'] != '0':
            raise RuntimeError('Analysis has not completed successfully')
        artifact = f'{ROOT}/artifacts/{job}'
        rc, _ = remote.command(transport, f'test -f {shlex.quote(artifact + "/PIPELINE_COMPLETE")} && '
                               f'test "$(cat {shlex.quote(artifact + "/exit_code.txt")})" = 0', timeout=25)
        if rc:
            raise RuntimeError('Analysis completion receipts are missing')
        observed['summary'] = read_remote_json(transport, artifact + '/summary.json')
        if prefix == 'cache':
            summary = observed['summary']
            if (summary.get('job_id') != job or summary.get('status') != 'complete'
                    or summary.get('context_count') != 60 or summary.get('selected_epoch') != 2
                    or summary.get('matched_references') != {'bf16': 60, 'parent': 60, 'distilled': 60}
                    or summary.get('tensor_metrics_self_check') is not True):
                raise RuntimeError('CPU cached-reference preflight coverage is incomplete')
        elif prefix == 'analysis' or prefix.startswith('analysis_shard_'):
            summary = observed['summary']
            selected = [0, 1] if prefix == 'analysis' else [int(prefix.rsplit('_', 1)[1])]
            gates = summary.get('gates', {})
            if (summary.get('job_id') != job or summary.get('status') != 'complete'
                    or summary.get('protocol') != 'fastwam-vq-latent-diagnostic-v1'
                    or summary.get('selected_epoch') != 2 or summary.get('selected_shards') != selected
                    or summary.get('full_cohort_complete') is not (len(selected) == 2)
                    or gates.get('context_count') != 30 * len(selected)
                    or gates.get('native_context_queries') != 90 * len(selected)
                    or gates.get('teacher_future_queries') != 60 * len(selected)
                    or gates.get('test_case_count') != 11 * len(selected)):
                raise RuntimeError('CPU analysis identity or declared coverage is incomplete')
        save(HERE / f'{prefix}_terminal.json', observed)
    return observed


def repair_upload(transport):
    receipts = []
    for shard in (0, 1):
        observed = inspect(transport, shard)
        if (observed['query_rc'] or observed['job_state'] not in ('F', 'X')
                or observed['Exit_status'] == '0'):
            raise RuntimeError('Repair upload requires both old shard jobs confirmed failed/terminal')
        failure = read_remote_json(transport, f'{ROOT}/results/shard-{shard}/failure.json')
        if (failure.get('job_id') != observed['job_id'] or failure.get('stage') != 'source_validation'
                or 'Missing historical distilled action' not in failure.get('message', '')):
            raise RuntimeError('This repair does not match the confirmed cache-loading failure')
        receipts.append({'scheduler': observed, 'failure': failure})
    with remote.sftp_client(transport) as sftp:
        for name in ('control.py', 'run_future.py', 'cache_preflight.py', 'cache_preflight.pbs'):
            data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
            if len(data) > trusted.LIMIT:
                raise RuntimeError('Repair control file exceeds 256 KiB')
            with sftp.file(f'{ROOT}/{name}', 'wb') as stream:
                stream.write(data)
            print('REPAIR_UPLOADED', name, len(data))
    save(HERE / 'cache_repair_receipt.json', {'reason': 'use resumable evaluation_rows across all allocations',
                                          'old_jobs': receipts, 'full_scope_unchanged': True})


def resubmit_repaired(transport, shard, attempt):
    observed = inspect(transport, shard)
    if (observed['query_rc'] or observed['job_state'] not in ('F', 'X') or observed['Exit_status'] == '0'):
        raise RuntimeError('Repaired resubmission requires a confirmed failed terminal source')
    if not (HERE / 'cache_repair_receipt.json').is_file():
        raise RuntimeError('No applied cache repair receipt')
    cpu_status(transport, 'cache', True)
    return submit(transport, shard, attempt, observed['job_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'continue', 'verify', 'pull', 'source',
                                         'analysis-upload', 'analysis-submit', 'analysis-status', 'analysis-verify', 'repair-upload',
                                         'cache-submit', 'cache-status', 'cache-verify', 'resubmit'))
    parser.add_argument('--shard', type=int, choices=(0, 1))
    parser.add_argument('--attempt', default='a')
    parser.add_argument('--tail', action='store_true')
    parser.add_argument('--files', nargs='+')
    args = parser.parse_args()
    if args.action in ('submit', 'status', 'continue', 'verify', 'resubmit') and args.shard is None:
        parser.error('--shard is required')
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        if args.action == 'upload':
            trusted.upload(transport, args.files)
        elif args.action == 'submit':
            print(json.dumps(submit(transport, args.shard, args.attempt)))
        elif args.action == 'status':
            print(json.dumps(inspect(transport, args.shard, args.tail)))
        elif args.action == 'continue':
            print(json.dumps(continue_job(transport, args.shard)))
        elif args.action == 'verify':
            print(json.dumps(verify(transport, args.shard)))
        elif args.action == 'analysis-submit':
            print(json.dumps(analysis_submit(transport, args.shard)))
        elif args.action == 'analysis-upload':
            analysis_upload(transport, args.shard)
        elif args.action in ('analysis-status', 'analysis-verify'):
            print(json.dumps(cpu_status(transport, analysis_prefix(args.shard), args.action == 'analysis-verify')))
        elif args.action == 'repair-upload':
            repair_upload(transport)
        elif args.action == 'cache-submit':
            print(json.dumps(small_cpu_submit(transport, 'cache', 'fwz1013cache', 'cache_preflight.pbs')))
        elif args.action in ('cache-status', 'cache-verify'):
            print(json.dumps(cpu_status(transport, 'cache', args.action == 'cache-verify')))
        elif args.action == 'resubmit':
            print(json.dumps(resubmit_repaired(transport, args.shard, args.attempt)))
        elif args.action == 'source':
            p = trusted.protocol()
            for root, job in ((p['source_root'], p['source_full_job']),
                              (p['distill_result'].split('/results/')[0], p['distill_job'])):
                artifact = f'{root}/artifacts/{job}'
                rc, _ = remote.command(transport, f'test -f {shlex.quote(artifact + "/PIPELINE_COMPLETE")} && '
                                       f'test "$(cat {shlex.quote(artifact + "/exit_code.txt")})" = 0', timeout=25)
                if rc:
                    raise RuntimeError('The completed source artifact receipts are unavailable')
                result = read_remote_json(transport, artifact + '/summary.json')
                print(json.dumps({'source_job': job, 'status': result['status'],
                                  'selected_epoch': result.get('selected_epoch')}))
        else:
            trusted.pull(transport, args.files)
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

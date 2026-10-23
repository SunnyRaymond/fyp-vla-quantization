"""Small-file PBS control; computations, model reads, and aggregation stay on compute nodes."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('anchor_source_control', HERE.parent / 'fastwam-vq-relational-kl/control.py')
source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
remote = source.remote
P = json.loads((HERE / 'protocol.json').read_text())
ROOT = P['remote_root']
JOBS = {'prepare': 'fwakprep', 'mse_nmse': 'fwakbase', 'anchor_kl': 'fwaktrain',
        'aggregate': 'fwakagg', 'closedloop_prepare': 'fwakclprep', 'closedloop': 'fwakcl',
        'closedloop_aggregate': 'fwakclagg'}
DEPENDENCIES = {'prepare': [], 'mse_nmse': ['prepare'], 'anchor_kl': ['prepare'],
                'aggregate': ['mse_nmse', 'anchor_kl'], 'closedloop_prepare': ['prepare'],
                'closedloop': ['anchor_kl', 'closedloop_prepare'], 'closedloop_aggregate': ['closedloop']}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_json(sftp, path):
    return source.c.source.read_json(sftp, path)


def write_json(sftp, path, value):
    return source.c.source.write_json(sftp, path, value)


def observe(transport, sftp, unit, handle):
    q = source.c.source.previous.scheduler(transport, handle['job_id'])
    summary = read_json(sftp, ROOT + '/results/' + unit + '/summary.json') or {}
    progress = read_json(sftp, ROOT + '/results/' + unit + '/progress.json') or {}
    if progress.get('pbs_jobid') != handle['job_id']:
        progress = {}
    prefix = ROOT + '/artifacts/' + handle['job_id']
    code, marker = None, False
    try:
        with sftp.file(prefix + '/exit_code.txt') as stream:
            code = int(stream.read(32))
    except OSError as error:
        if error.errno != 2:
            raise
    try:
        sftp.stat(prefix + '/PHASE_COMPLETE'); marker = True
    except OSError as error:
        if error.errno != 2:
            raise
    coverage = summary.get('complete') is True and summary.get('pbs_jobid') == handle['job_id']
    if unit == 'prepare':
        coverage &= summary.get('passed') is True and summary.get('heldout_used') is False and summary.get('calibration_case_ids') == P['calibration']['case_ids']
    elif unit in P['arms']:
        coverage &= summary.get('updates') == 80 and summary.get('evaluation_queries') == 160 and summary.get('new_teacher_queries') == 0
    elif unit == 'aggregate':
        coverage &= summary.get('calibration_records') == 160 and summary.get('heldout_records') == 160
    elif unit == 'closedloop_prepare':
        coverage &= summary.get('reference_episodes') == 280 and summary.get('reference_successes') == 203
    elif unit in ('closedloop', 'closedloop_aggregate'):
        coverage &= summary.get('episodes') == 280
    passed = bool(q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
                  and code == 0 and marker and coverage)
    return {'handle': handle, 'scheduler': q, 'summary': summary, 'progress': progress,
            'artifact_exit_code': code, 'phase_complete_marker': marker, 'passed': passed}


def resources(unit):
    if unit in ('prepare', 'aggregate', 'closedloop_prepare', 'closedloop_aggregate'):
        return 'cpu.pbs', P['cpu_allocation']['walltime'], 0
    if unit == 'closedloop':
        return 'closedloop.pbs', P['closedloop_allocation']['walltime'], 2
    return 'gpu.pbs', P['allocation']['walltime'], 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'retry', 'status', 'logs', 'pull'))
    parser.add_argument('--unit', choices=tuple(JOBS))
    args = parser.parse_args()
    jump, transport = remote.direct_connect()
    try:
        with remote.sftp_client(transport) as sftp:
            if args.action == 'upload':
                rc, out = remote.command(transport, 'mkdir -p ' + shlex.quote(ROOT + '/control') + ' ' + shlex.quote(ROOT + '/artifacts'), timeout=20)
                if rc:
                    raise RuntimeError(out)
                old = read_json(sftp, ROOT + '/protocol.json')
                if old and old != P:
                    raise RuntimeError('Remote protocol is frozen')
                names = ['protocol.json', 'allocation.sh', 'cpu.pbs', 'gpu.pbs', 'closedloop.pbs',
                         'anchor_loss.py', 'anchor_selfcheck.py', 'prepare.py', 'worker.py',
                         'aggregate.py', 'closedloop.py', 'closedloop_prepare.py', 'closedloop_aggregate.py']
                uploaded = []
                for name in names:
                    path = HERE / name
                    if not path.is_file():
                        continue
                    data = path.read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Only bounded small code/control uploads are allowed')
                    with sftp.file(ROOT + '/' + name + '.tmp', 'wb') as stream:
                        stream.write(data)
                    sftp.posix_rename(ROOT + '/' + name + '.tmp', ROOT + '/' + name)
                    uploaded.append(name)
                for name in ('allocation.sh', 'cpu.pbs', 'gpu.pbs', 'closedloop.pbs'):
                    rc, out = remote.command(transport, 'bash -n ' + shlex.quote(ROOT + '/' + name), timeout=15)
                    if rc:
                        raise RuntimeError(out)
                result = {'uploaded': uploaded, 'shell_syntax_passed': True}
            else:
                handles = read_json(sftp, ROOT + '/control/handles.json') or {}
                if args.action in ('submit', 'retry'):
                    unit = args.unit
                    if not unit:
                        parser.error('--unit is required')
                    if read_json(sftp, ROOT + '/protocol.json') != P:
                        raise RuntimeError('Local and remote frozen protocols differ')
                    previous = handles.get(unit)
                    if previous and args.action == 'submit':
                        result = {'already_submitted': True, **previous}
                    else:
                        if args.action == 'retry':
                            if previous is None:
                                raise RuntimeError('Retry requires an existing authoritative handle')
                            observed = observe(transport, sftp, unit, previous)
                            q = observed['scheduler']
                            if (q['query_rc'] != 0 or q['job_state'] not in ('F', 'X')
                                    or q['Exit_status'] in (None, '0') or observed['passed']):
                                raise RuntimeError('Retry requires a verified unsuccessful terminal job; Q/R/query failure never authorizes retry')
                            save(HERE / ('previous_' + unit + '_' + previous['job_id'] + '.json'), observed)
                        dependencies = []
                        for dependency in DEPENDENCIES[unit]:
                            if dependency not in handles:
                                raise RuntimeError('Missing prerequisite handle: ' + dependency)
                            observed = observe(transport, sftp, dependency, handles[dependency])
                            q = observed['scheduler']
                            if dependency == 'prepare' and not observed['passed']:
                                raise RuntimeError('Complete CPU definition validation required before GPU submission')
                            if q['query_rc'] != 0 or (q['job_state'] in ('F', 'X') and not observed['passed']):
                                raise RuntimeError('Prerequisite is failed, incomplete, or unknown: ' + dependency)
                            dependencies.append(handles[dependency]['job_id'])
                        script, walltime, gpus = resources(unit)
                        attempt = (previous.get('attempt', 1) + 1) if previous else 1
                        intent_path = ROOT + '/control/' + unit + '_attempt' + str(attempt) + '_intent.json'
                        if read_json(sftp, intent_path):
                            raise RuntimeError('Reconcile existing submission intent before retrying qsub')
                        handle = {'unit': unit, 'attempt': attempt, 'job_name': JOBS[unit], 'status': 'intent_before_qsub',
                                  'gpus': gpus, 'walltime': walltime, 'created_utc': datetime.now(timezone.utc).isoformat(),
                                  'afterok_job_ids': dependencies}
                        if previous:
                            handle['previous_attempt'] = previous
                        write_json(sftp, intent_path, handle)
                        save(HERE / (unit + '_attempt' + str(attempt) + '_intent.json'), handle)
                        afterok = ' -W ' + shlex.quote('depend=afterok:' + ':'.join(dependencies)) if dependencies else ''
                        command = ('qsub -N ' + JOBS[unit] + afterok + ' -v ' + shlex.quote('FW_AK_UNIT=' + unit)
                                   + ' -o ' + shlex.quote(ROOT + '/artifacts/' + unit + '_attempt' + str(attempt) + '.pbs.out')
                                   + ' ' + shlex.quote(ROOT + '/' + script))
                        rc, out = remote.command(transport, command, timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', out.strip()):
                            raise RuntimeError('Unknown qsub outcome; reconcile saved intent: ' + out[:400])
                        handle.update(status='confirmed', job_id=out.strip()); handles[unit] = handle
                        write_json(sftp, intent_path, handle); write_json(sftp, ROOT + '/control/handles.json', handles)
                        save(HERE / 'handles.json', handles); result = handle
                else:
                    observed = {unit: observe(transport, sftp, unit, handle) for unit, handle in handles.items()
                                if not args.unit or unit == args.unit}
                    snapshot = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': observed}
                    save(HERE / 'last_status.json', snapshot)
                    if args.action == 'logs':
                        if not args.unit:
                            parser.error('--unit is required')
                        rc, out = remote.command(transport, 'tail -c 9000 ' + shlex.quote(ROOT + '/artifacts/' + handles[args.unit]['job_id'] + '/job.log'), timeout=20)
                        result = {'query_rc': rc, 'tail': out}
                    else:
                        if args.action == 'pull':
                            for unit, row in observed.items():
                                if row['summary']:
                                    save(HERE / 'results' / unit / 'summary.json', row['summary'])
                            for name in ('anchor_kl/loss_scaling', 'comparison'):
                                value = read_json(sftp, ROOT + '/results/' + name + '.json')
                                if value:
                                    save(HERE / 'results' / (name + '.json'), value)
                            for name in ('ANALYSIS.zh.md', 'CLOSEDLOOP.zh.md'):
                                try:
                                    with sftp.file(ROOT + '/results/' + name, 'rb') as stream:
                                        data = stream.read(remote.LIMIT + 1)
                                except OSError as error:
                                    if error.errno != 2:
                                        raise
                                    continue
                                if len(data) > remote.LIMIT:
                                    raise RuntimeError('Report exceeds bounded retrieval')
                                path = HERE / 'results' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                        result = {'checked_utc': snapshot['checked_utc'], 'units': {unit: {
                            'job_id': row['handle']['job_id'], 'scheduler': row['scheduler'], 'passed': row['passed'],
                            'phase': row['progress'].get('phase'), 'updates': row['progress'].get('updates'),
                            'queries': row['progress'].get('queries'), 'episodes': row['progress'].get('episodes'),
                            'error': row['progress'].get('error')} for unit, row in observed.items()}}
            print(json.dumps(result, indent=2))
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

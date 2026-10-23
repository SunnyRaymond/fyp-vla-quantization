"""Small-file PBS submission and read-only status; never runs workloads on login."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('clipping_rotation_control',
    HERE.parent / 'fastwam-rotation-baselines/control.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
remote = previous.remote
P = json.loads((HERE / 'protocol.json').read_text())
ROOT = P['remote_root']
FILES = ('protocol.json', 'experiment.py', 'clip_banks.py', 'selfcheck.py', 'run.pbs', 'aggregate.pbs')
JOBS = {'prepare': 'fwcl17p', 'weight_only': 'fwcl17w', 'clip_only': 'fwcl17a',
        'joint': 'fwcl17j', 'aggregate': 'fwcl17agg', 'prepare_validation': 'fwcl17pv'}


def read_json(sftp, path):
    try:
        with sftp.file(path, 'rb') as stream:
            data = stream.read(remote.LIMIT + 1)
    except OSError as exc:
        if exc.errno == 2:
            return None
        raise
    if len(data) > remote.LIMIT:
        raise RuntimeError('Control file exceeds 256 KiB')
    return json.loads(data)


def write_json(sftp, path, value):
    data = (json.dumps(value, indent=2) + '\n').encode()
    if len(data) > remote.LIMIT:
        raise RuntimeError('Control file exceeds 256 KiB')
    with sftp.file(path + '.tmp', 'wb') as stream:
        stream.write(data)
    sftp.posix_rename(path + '.tmp', path)


def upload(transport):
    rc, output = remote.command(transport, 'mkdir -p ' + shlex.quote(ROOT + '/control') +
                                ' ' + shlex.quote(ROOT + '/artifacts'))
    if rc:
        raise RuntimeError(output)
    with remote.sftp_client(transport) as sftp:
        for name in FILES:
            data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
            if len(data) > remote.LIMIT:
                raise RuntimeError('Only small code/control files may be uploaded')
            with sftp.file(ROOT + '/' + name + '.tmp', 'wb') as stream:
                stream.write(data)
            sftp.posix_rename(ROOT + '/' + name + '.tmp', ROOT + '/' + name)
    return {'uploaded': list(FILES)}


def all_handles(sftp):
    return read_json(sftp, ROOT + '/control/handles.json') or {}


def observe(transport, key, handle, sftp):
    result = {'handle': handle, 'scheduler': previous.scheduler(transport, handle['job_id'])}
    arm = 'baseline' if key == 'prepare' else key
    for name in ('progress', 'summary'):
        result[name] = read_json(sftp, ROOT + '/results/' + arm + '/' + name + '.json')
    if key == 'aggregate':
        result['comparison'] = read_json(sftp, ROOT + '/results/comparison.json')
    if key == 'prepare_validation':
        result['summary'] = read_json(sftp, ROOT + '/results/baseline/validation.json')
    if key == 'prepare':
        result['teacher_receipt'] = read_json(sftp, ROOT + '/results/teacher/receipt.json')
    result['phase_complete_marker'] = False
    try:
        sftp.stat(ROOT + '/artifacts/' + handle['job_id'] + '/PHASE_COMPLETE')
        result['phase_complete_marker'] = True
    except OSError as exc:
        if exc.errno != 2:
            raise
    result['artifact_exit_code'] = None
    try:
        with sftp.file(ROOT + '/artifacts/' + handle['job_id'] + '/exit_code.txt', 'rb') as stream:
            result['artifact_exit_code'] = int(stream.read(32).strip())
    except OSError as exc:
        if exc.errno != 2:
            raise
    return result


def passed(observation):
    scheduler = observation['scheduler']
    summary = observation.get('summary') or observation.get('comparison') or {}
    teacher = observation.get('teacher_receipt')
    teacher_ok = (teacher is None and observation['handle']['unit'] != 'prepare') or (
        teacher is not None and teacher.get('complete') is True and teacher.get('queries') == 80
        and teacher.get('case_ids') == P['evaluation']['case_ids']
        and teacher.get('seed_indices') == P['evaluation']['seed_indices'])
    return (teacher_ok and observation.get('phase_complete_marker') is True
            and scheduler['query_rc'] == 0 and scheduler['job_state'] in ('F', 'X')
            and scheduler['Exit_status'] == '0' and observation['artifact_exit_code'] == 0
            and summary.get('complete') is True
            and summary.get('pbs_jobid') == observation['handle']['job_id']
            and (summary.get('queries') == 80 or summary.get('arms') is not None))


def status(transport, key=None):
    with remote.sftp_client(transport) as sftp:
        handles = all_handles(sftp)
        results = {k: observe(transport, k, h, sftp) for k, h in handles.items()
                   if key is None or k == key}
    return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': results}


def prepare_passed(transport, handles, sftp):
    if 'prepare' not in handles:
        return False
    original = observe(transport, 'prepare', handles['prepare'], sftp)
    if passed(original):
        return True
    if 'prepare_validation' not in handles:
        return False
    validation = observe(transport, 'prepare_validation', handles['prepare_validation'], sftp)
    receipt = validation.get('summary') or {}
    return (passed(validation) and original['scheduler']['job_state'] in ('F', 'X')
            and original['scheduler']['Exit_status'] == '1'
            and original['artifact_exit_code'] == 1
            and (original.get('summary') or {}).get('complete') is True
            and receipt.get('source_job_id') == handles['prepare']['job_id']
            and receipt.get('new_model_forwards') == 0
            and receipt.get('recomputed_from_query_records') is True)


def submit(transport, key, resume=False):
    with remote.sftp_client(transport) as sftp:
        if read_json(sftp, ROOT + '/protocol.json') != P:
            raise RuntimeError('Upload the frozen protocol before submission')
        handles = all_handles(sftp)
        old = handles.get(key)
        if old and not resume:
            return {'existing': True, 'handle': old}
        if resume:
            if not old:
                raise RuntimeError('No previous handle to resume')
            state = observe(transport, key, old, sftp)
            if (state['scheduler']['query_rc'] != 0
                    or state['scheduler']['job_state'] not in ('F', 'X')
                    or state['scheduler']['Exit_status'] != '75'
                    or (state.get('progress') or {}).get('phase') != 'checkpointed'):
                raise RuntimeError('Only an authoritative soft-deadline checkpoint may resume')
        if key in ('weight_only', 'clip_only', 'joint'):
            if not prepare_passed(transport, handles, sftp):
                raise RuntimeError('Shared teacher and baseline must finish before training')
        if key == 'prepare_validation':
            source = observe(transport, 'prepare', handles['prepare'], sftp)
            if (source['scheduler']['query_rc'] != 0 or source['scheduler']['job_state'] not in ('F', 'X')
                    or source['scheduler']['Exit_status'] != '1' or source['artifact_exit_code'] != 1
                    or not (source.get('summary') or {}).get('complete')
                    or (source.get('summary') or {}).get('queries') != 80):
                raise RuntimeError('CPU repair requires terminal failed GPU job with complete numerical results')
        if key == 'aggregate':
            if (not prepare_passed(transport, handles, sftp)
                    or any(arm not in handles or not passed(observe(transport, arm, handles[arm], sftp))
                           for arm in ('weight_only', 'clip_only', 'joint'))):
                raise RuntimeError('All four arms must finish before CPU aggregation')
        attempt = old.get('attempt', 0) + 1 if old else 0
        job_name = JOBS[key] + (f'r{attempt}' if attempt else '')
        intent_path = ROOT + '/control/' + job_name + '.json'
        intent = read_json(sftp, intent_path)
        rc, output = remote.command(transport, 'qselect -x -u yguo017 -N ' + shlex.quote(job_name), timeout=35)
        if rc:
            raise RuntimeError('PBS history query failed; submission not attempted')
        matches = output.split()
        if len(matches) > 1 or any(not previous.HANDLE.fullmatch(job) for job in matches):
            raise RuntimeError('Ambiguous PBS history; submission not attempted')
        if matches:
            handle = {**(intent or {}), 'job_id': matches[0], 'unit': key,
                      'job_name': job_name, 'attempt': attempt, 'status': 'confirmed'}
        else:
            if intent:
                raise RuntimeError('Unresolved submission intent exists; do not submit twice')
            env = {'FW_CLIP_PHASE': 'prepare' if key == 'prepare' else 'arm',
                   'FW_CLIP_ARM': key}
            cpu = key in ('aggregate', 'prepare_validation')
            if key == 'prepare_validation':
                env.update(FW_CLIP_PHASE='validate_prepare', FW_VALIDATE_SOURCE_JOB=handles['prepare']['job_id'])
            handle = {'unit': key, 'job_name': job_name, 'attempt': attempt,
                      'created_utc': datetime.now(timezone.utc).isoformat(),
                      'remote_root': ROOT, 'status': 'intent_before_qsub',
                      'walltime': '00:10:00' if cpu else '01:59:00',
                      'gpus': 0 if cpu else 1, 'env': env}
            write_json(sftp, intent_path, handle)
            previous.save(HERE / (job_name + '_intent.json'), handle)
            script = 'aggregate.pbs' if cpu else 'run.pbs'
            command = ('qsub -N ' + shlex.quote(job_name) + ' -o ' +
                       shlex.quote(ROOT + '/artifacts/' + job_name + '.pbs.out') +
                       ' -v ' + shlex.quote(','.join(k + '=' + v for k, v in env.items())) +
                       ' ' + shlex.quote(ROOT + '/' + script))
            rc, output = remote.command(transport, command, timeout=40)
            job_id = output.strip()
            if rc or not previous.HANDLE.fullmatch(job_id):
                handle.update(status='unknown_or_rejected', response=output[:1000], return_code=rc)
                write_json(sftp, intent_path, handle)
                raise RuntimeError('qsub outcome unknown/rejected; do not repeat blindly')
            handle.update(job_id=job_id, status='confirmed')
        write_json(sftp, intent_path, handle)
        if old:
            write_json(sftp, ROOT + '/control/' + key + '_previous_' + str(old['attempt']) + '.json', old)
        handles[key] = handle
        write_json(sftp, ROOT + '/control/handles.json', handles)
        previous.save(HERE / 'handles.json', handles)
        return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('probe', 'upload', 'submit', 'resume', 'status', 'logs', 'pull'))
    parser.add_argument('--unit', choices=tuple(JOBS))
    args = parser.parse_args()
    jump, transport = remote.direct_connect()
    try:
        if args.action == 'probe':
            with remote.sftp_client(transport) as sftp:
                parent = P['parent_root'] + '/results/training/rho1.5/training_receipt.json'
                receipt = read_json(sftp, parent)
                result = {'parent_complete': (receipt or {}).get('complete'),
                          'parent_updates': (receipt or {}).get('updates'),
                          'parent_source_c8': (receipt or {}).get('source_c8')}
            rc, output = remote.command(transport, 'qstat -u yguo017', timeout=35)
            result.update(queue_query_rc=rc, jobs=output)
        elif args.action == 'upload':
            result = upload(transport)
        elif args.action in ('submit', 'resume'):
            if not args.unit:
                parser.error('--unit is required')
            result = submit(transport, args.unit, args.action == 'resume')
        elif args.action == 'status':
            result = status(transport, args.unit)
            previous.save(HERE / 'last_status.json', result)
        elif args.action == 'pull':
            result = status(transport)
            with remote.sftp_client(transport) as sftp:
                comparison = read_json(sftp, ROOT + '/results/comparison.json')
                if comparison:
                    previous.save(HERE / 'results/comparison.json', comparison)
                try:
                    with sftp.file(ROOT + '/results/RESULTS.zh.md', 'rb') as stream:
                        report = stream.read(remote.LIMIT + 1)
                    if len(report) > remote.LIMIT:
                        raise RuntimeError('Result report exceeds small-file limit')
                    target = HERE / 'results/RESULTS.zh.md'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(report)
                except OSError as exc:
                    if exc.errno != 2:
                        raise
                for key, observation in result['units'].items():
                    if observation.get('summary'):
                        arm = 'baseline' if key in ('prepare', 'prepare_validation') else key
                        filename = 'validation.json' if key == 'prepare_validation' else 'summary.json'
                        previous.save(HERE / 'results' / arm / filename, observation['summary'])
                    if key in ('weight_only', 'clip_only', 'joint'):
                        receipt = read_json(sftp, ROOT + '/results/' + key + '/training_receipt.json')
                        if receipt:
                            previous.save(HERE / 'results' / key / 'training_receipt.json', receipt)
                    handle = observation['handle']
                    if handle.get('gpus') == 1 and observation.get('artifact_exit_code') is not None:
                        prefix = ROOT + '/artifacts/' + handle['job_id']
                        check = read_json(sftp, prefix + '/clip_selfcheck.json')
                        if check:
                            previous.save(HERE / 'results/evidence' / handle['job_id'] / 'clip_selfcheck.json', check)
                        with sftp.file(prefix + '/job.log', 'rb') as stream:
                            head = stream.read(4096)
                            stream.seek(max(0, sftp.stat(prefix + '/job.log').st_size - 8192))
                            tail = stream.read(8192)
                        samples = list(dict.fromkeys(line for line in (head + b'\n' + tail).decode(
                            'utf-8', errors='replace').splitlines() if line.startswith('GPU_SAMPLE timestamp=')))
                        previous.save(HERE / 'results/evidence' / handle['job_id'] / 'gpu_telemetry.json',
                                      {'job_id': handle['job_id'], 'source': prefix + '/job.log',
                                       'read_scope': 'first4096 and last8192 bytes only', 'samples': samples})
            previous.save(HERE / 'last_status.json', result)
        else:
            if not args.unit:
                parser.error('--unit is required')
            with remote.sftp_client(transport) as sftp:
                handle = all_handles(sftp)[args.unit]
            rc, output = remote.command(transport, 'tail -n 32 ' + shlex.quote(
                ROOT + '/artifacts/' + handle['job_id'] + '/job.log') + ' 2>/dev/null; tail -n 12 ' +
                shlex.quote(ROOT + '/artifacts/' + handle['job_name'] + '.pbs.out') + ' 2>/dev/null', timeout=35)
            result = {'job_id': handle['job_id'], 'query_rc': rc, 'tail': output[-14000:]}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

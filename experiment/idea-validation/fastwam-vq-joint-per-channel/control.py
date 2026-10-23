"""Bounded PBS control for the joint per-channel experiment."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
ROOT = P['remote_root']
JOBS = {'train_eval': 'fwjpc80', 'aggregate': 'fwjpcag'}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = _load('joint_per_channel_remote_control', HERE.parent / 'fastwam-vq-activation-clipping' / 'control.py').remote
c80_control = _load('joint_per_channel_c80_source_control', HERE.parent / 'fastwam-vq-c80-activation-clipping' / 'control.py')
per_control = _load('joint_per_channel_source_control', HERE.parent / 'fastwam-vq-per-channel-scaling-clipping' / 'control.py')
joint_control = _load('joint_per_channel_c80_control', HERE.parent / 'fastwam-vq-c80-joint-rmse' / 'control.py')


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def _original_c80_status(transport):
    original = c80_control.status(transport)  # current remote artifacts and summaries
    evidence_path = HERE.parent / 'fastwam-vq-c80-activation-clipping' / 'completion_evidence.json'
    history = json.loads(evidence_path.read_text(encoding='utf-8'))
    protocol = P['source_control_protocol']['protocol']
    if history.get('complete') is not True or history.get('protocol') != protocol:
        raise RuntimeError('Original C80 completion evidence does not match the frozen protocol')
    fallback_units = []
    for unit in ('prepare', 'train_eval', 'aggregate'):
        current = original['units'].get(unit)
        if not current:
            raise RuntimeError('Original C80 live artifact observation is missing for ' + unit)
        if c80_control.passed(current):
            current['scheduler_provenance'] = 'live_qstat'
            continue
        live_scheduler = current.get('scheduler') or {}
        old = history.get('units', {}).get(unit) or {}
        old_handle = old.get('handle') or {}
        handle = current.get('handle') or {}
        binding = ('unit', 'attempt', 'job_name', 'remote_root', 'job_id')
        scheduler = old.get('scheduler') or {}
        if (live_scheduler.get('query_rc') == 0
                or any(handle.get(key) != old_handle.get(key) for key in binding)
                or scheduler.get('query_rc') != 0 or scheduler.get('job_state') not in ('F', 'X')
                or str(scheduler.get('Exit_status')) != '0'
                or scheduler.get('job_id') != handle.get('job_id')
                or scheduler.get('Job_Name') != handle.get('job_name')):
            raise RuntimeError('Original C80 scheduler is unresolved and has no matching historical F0 evidence: ' + unit)
        if not c80_control.passed({**current, 'scheduler': scheduler}):
            raise RuntimeError('Original C80 live artifact/summary completion failed for ' + unit)
        current['scheduler_live'] = live_scheduler
        current['scheduler_provenance'] = 'completion_evidence.json historical F0; live artifact and summary passed'
        current['scheduler_completion_evidence'] = scheduler
        fallback_units.append(unit)
    original['completion_evidence_fallback_units'] = fallback_units
    return original


def _verify_sources(transport):
    original = _original_c80_status(transport)
    per = per_control.status(transport)
    if (not all(unit in per['units'] for unit in ('train_eval', 'aggregate'))
            or not all(per_control.passed(per['units'][unit]) for unit in ('train_eval', 'aggregate'))):
        raise RuntimeError('Original per-channel train_eval and aggregate F0 evidence are required')
    joint = joint_control.status(transport)
    upstream = joint['upstream']
    if not upstream.get('ready') and (upstream.get('scheduler') or {}).get('query_rc') != 0:
        historical = json.loads((HERE.parent / 'fastwam-vq-c80-joint-rmse' /
                                 'completion_evidence.json').read_text(encoding='utf-8'))['upstream']
        scheduler = historical.get('scheduler') or {}
        job = upstream.get('final_job_id')
        if (historical.get('ready') is not True or scheduler.get('query_rc') != 0
                or scheduler.get('job_state') not in ('F', 'X') or scheduler.get('Exit_status') != '0'
                or scheduler.get('job_id') != job
                or any(upstream.get(k) != historical.get(k) for k in
                       ('root', 'protocol', 'receipt', 'progress', 'final_job_id'))):
            raise RuntimeError('Joint-C80 source has no matching historical F0 evidence')
        prefix = upstream['root'] + '/artifacts/' + job
        with remote.sftp_client(transport) as sftp:
            summary = per_control.source.read_json(sftp, prefix + '/calibration_summary.json')
            with sftp.file(prefix + '/exit_code.txt', 'rb') as stream:
                exit_code = int(stream.read(32).strip())
            sftp.stat(prefix + '/TRAINING_COMPLETE')
            for key in ('checkpoint', 'reference_action'):
                sftp.stat(P['source_joint_protocol']['joint_source'][key])
        if (exit_code != 0 or summary != historical.get('summary') or not summary.get('complete')
                or summary.get('pbs_jobid') != job or summary.get('updates') != 80):
            raise RuntimeError('Joint-C80 current source artifacts differ from completed evidence')
        upstream.update(ready=True, reason='historical F0 plus matching live receipt, progress and artifacts',
                        scheduler_completion_evidence=scheduler, scheduler_provenance='historical F0',
                        summary=summary, artifact_exit_code=exit_code, training_complete_marker=True)
    if not (joint['upstream'].get('ready') is True
            and all(unit in joint['units'] for unit in ('evaluate', 'aggregate'))
            and all(joint_control.passed(joint['units'][unit]) for unit in ('evaluate', 'aggregate'))):
        raise RuntimeError('Completed joint-C80 source and consumer are required')
    return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'original_c80': original,
            'per_channel': per, 'joint_c80_consumer': joint}


def _compact_source_evidence(evidence):
    result = {'checked_utc': evidence['checked_utc']}
    for name in ('original_c80', 'per_channel', 'joint_c80_consumer'):
        source = evidence[name]
        rows = {}
        for unit, observed in source.get('units', {}).items():
            rows[unit] = {k: observed[k] for k in ('handle', 'scheduler', 'scheduler_live',
                'scheduler_completion_evidence', 'scheduler_provenance', 'artifact_exit_code',
                'phase_complete_marker') if k in observed}
            rows[unit]['summary'] = {k: (observed.get('summary') or {}).get(k) for k in
                ('protocol', 'complete', 'pbs_jobid', 'updates', 'training_updates',
                 'evaluation_queries', 'training_updates_this_task', 'all_parameters_frozen')}
        result[name] = {'units': rows}
        if 'upstream' in source:
            result[name]['upstream'] = {k: source['upstream'].get(k) for k in
                ('ready', 'reason', 'root', 'final_job_id', 'scheduler', 'scheduler_completion_evidence',
                 'scheduler_provenance', 'artifact_exit_code', 'training_complete_marker')}
    return result


def _observe(transport, sftp, unit, handle):
    artifact = ROOT + '/artifacts/' + handle['job_id']
    result = {'handle': handle, 'scheduler': per_control.source.previous.scheduler(transport, handle['job_id']),
              'phase_complete_marker': False, 'artifact_exit_code': None}
    filename = 'results/comparison.json' if unit == 'aggregate' else 'results/train_eval/summary.json'
    result['summary'] = per_control.source.read_json(sftp, ROOT + '/' + filename)
    progress_path = 'results/aggregate/progress.json' if unit == 'aggregate' else 'results/train_eval/progress.json'
    result['progress'] = per_control.source.read_json(sftp, ROOT + '/' + progress_path)
    if unit == 'train_eval':
        result['receipt'] = per_control.source.read_json(sftp, ROOT + '/results/train_eval/training_receipt.json')
    try:
        sftp.stat(artifact + '/PHASE_COMPLETE')
        result['phase_complete_marker'] = True
    except OSError as exc:
        if exc.errno != 2:
            raise
    try:
        with sftp.file(artifact + '/exit_code.txt', 'rb') as stream:
            result['artifact_exit_code'] = int(stream.read(32).strip())
    except OSError as exc:
        if exc.errno != 2:
            raise
    return result


def _passed(observation):
    q = observation['scheduler']; summary = observation.get('summary') or {}
    unit = observation['handle']['unit']
    if unit == 'train_eval':
        receipt = observation.get('receipt') or {}
        checks = receipt.get('gradient_selfchecks') or {}
        required_checks = {'clip_selfcheck.json': ('passed',),
            'channel_selfcheck.json': ('passed', 'independent_reference_passed', 'old_surrogate_rejected'),
            'joint_selfcheck.json': ('passed', 'weight_gradient_reference_passed',
                                     'frozen_parameter_gradients_preserved', 'master_gradient_connected',
                                     'independent_reference_passed')}
        job_id = observation['handle']['job_id']
        checks_pass = all(isinstance(checks.get(name), dict)
            and checks[name].get('pbs_jobid') == job_id
            and all(checks[name].get(key) is True for key in keys)
            for name, keys in required_checks.items())
        coverage = (summary.get('evaluation_queries') == 160 and summary.get('updates') == 80
            and all(summary.get(k) is True for k in ('indices_unchanged', 'original_exponents_unchanged',
                'continuous_scale_metadata_separate', 'module_masters_fp32', 'codebooks_optimized',
                'row_gains_optimized'))
            and receipt.get('complete') is True and receipt.get('updates') == 80
            and receipt.get('pbs_jobid') == job_id and receipt.get('initial_parity_max_abs') == 0
            and len(receipt.get('history', [])) == 80
            and [row.get('update') for row in receipt.get('history', [])] == list(range(1, 81))
            and receipt.get('update_order') == P['calibration']['update_order']
            and [row.get('case_id') for row in receipt.get('history', [])] == P['calibration']['update_order']
            and receipt.get('codebook_changed_modules', 0) > 0 and receipt.get('row_gain_changed_modules', 0) > 0
            and all(receipt.get('cached_query_parity', {}).get(key) == 0
                    for key in ('action_max_abs', 'future_max_abs'))
            and checks_pass)
    else:
        coverage = (summary.get('evaluation_queries') == 800 and summary.get('training_updates') == 80
            and summary.get('new_endpoint_records') == 160 and summary.get('reused_endpoint_records') == 640
            and summary.get('total_endpoint_records') == 800)
    return (q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
        and observation['artifact_exit_code'] == 0 and observation['phase_complete_marker'] is True
        and summary.get('complete') is True and summary.get('pbs_jobid') == observation['handle']['job_id'] and coverage)


def _status(transport):
    with remote.sftp_client(transport) as sftp:
        handles = per_control.source.read_json(sftp, ROOT + '/control/handles.json') or {}
        units = {unit: _observe(transport, sftp, unit, handle) for unit, handle in handles.items() if unit in JOBS}
    return {'checked_utc': datetime.now(timezone.utc).isoformat(),
            'units': {unit: {'handle': value['handle'], 'scheduler': value['scheduler'],
                'passed': _passed(value), 'phase': (value.get('progress') or {}).get('phase'),
                'updates': (value.get('progress') or {}).get('updates'),
                'queries': (value.get('progress') or {}).get('queries'),
                'split': (value.get('progress') or {}).get('split')} for unit, value in units.items()}}


def _submit(transport, unit):
    with remote.sftp_client(transport) as sftp:
        if per_control.source.read_json(sftp, ROOT + '/protocol.json') != P:
            raise RuntimeError('Remote protocol differs from this frozen local contract')
        handles = per_control.source.read_json(sftp, ROOT + '/control/handles.json') or {}
        if handles.get(unit):
            raise RuntimeError('An existing handle blocks duplicate submission; inspect status first')
        intent = ROOT + '/control/' + unit + '_intent_0.json'
        if any(name.startswith(unit + '_intent_') and name.endswith('.json')
               for name in sftp.listdir(ROOT + '/control')):
            raise RuntimeError('An existing intent requires reconciliation; do not submit again')
        if unit == 'aggregate':
            source_handle = handles.get('train_eval')
            if not source_handle or not _passed(_observe(transport, sftp, 'train_eval', source_handle)):
                raise RuntimeError('Verified train_eval completion is required before aggregate')
        else:
            evidence = _verify_sources(transport)
            _save(HERE / 'source_evidence.json', evidence)
            per_control.source.write_json(sftp, ROOT + '/control/source_evidence.json', _compact_source_evidence(evidence))
        attempt = 0
        job = {'unit': unit, 'attempt': attempt, 'job_name': JOBS[unit],
               'created_utc': datetime.now(timezone.utc).isoformat(), 'remote_root': ROOT,
               'status': 'intent_before_qsub', 'gpus': 1 if unit == 'train_eval' else 0,
               'walltime': P['allocation']['walltime'] if unit == 'train_eval' else '00:10:00'}
        per_control.source.write_json(sftp, intent, job)
        _save(HERE / (unit + '_intent_0.json'), job)
        script = '/run.pbs' if unit == 'train_eval' else '/aggregate.pbs'
        command = ('qsub -N ' + shlex.quote(JOBS[unit]) + ' -o '
            + shlex.quote(ROOT + '/artifacts/' + JOBS[unit] + '_0.pbs.out') + ' -v '
            + shlex.quote('FW_JPC_UNIT=' + unit) + ' ' + shlex.quote(ROOT + script))
        rc, output = remote.command(transport, command, timeout=40)
        job_id = output.strip()
        if rc or not per_control.source.previous.HANDLE.fullmatch(job_id):
            job.update(status='unknown_or_rejected', return_code=rc, response=output[:1000])
            per_control.source.write_json(sftp, intent, job); _save(HERE / (unit + '_intent_0.json'), job)
            raise RuntimeError('qsub result is uncertain or rejected; inspect existing intent and PBS before retrying')
        job.update(job_id=job_id, status='confirmed')
        per_control.source.write_json(sftp, intent, job)
        handles[unit] = job
        per_control.source.write_json(sftp, ROOT + '/control/handles.json', handles)
        _save(HERE / 'handles.json', handles)
        return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('probe', 'upload', 'submit', 'status', 'pull', 'logs'))
    parser.add_argument('--unit', choices=tuple(JOBS))
    args = parser.parse_args()
    jump, transport = remote.direct_connect()
    try:
        if args.action == 'probe':
            result = _verify_sources(transport)
            _save(HERE / 'source_evidence.json', result)
            result = {'source_completion_verified': True, 'evidence': result}
        elif args.action == 'upload':
            evidence = _verify_sources(transport)
            rc, output = remote.command(transport, 'mkdir -p ' + shlex.quote(ROOT + '/control') + ' '
                                        + shlex.quote(ROOT + '/artifacts'), timeout=20)
            if rc:
                raise RuntimeError(output)
            with remote.sftp_client(transport) as sftp:
                old = per_control.source.read_json(sftp, ROOT + '/protocol.json')
                if old is not None and old != P:
                    raise RuntimeError('Remote frozen protocol differs; refusing overwrite')
                for name in ('worker.py', 'joint_banks.py', 'joint_selfcheck.py', 'protocol.json', 'run.pbs', 'aggregate.pbs'):
                    data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Only small protocol and control files may be uploaded')
                    with sftp.file(ROOT + '/' + name + '.tmp', 'wb') as stream:
                        stream.write(data)
                    sftp.posix_rename(ROOT + '/' + name + '.tmp', ROOT + '/' + name)
                per_control.source.write_json(sftp, ROOT + '/control/source_evidence.json', _compact_source_evidence(evidence))
            rc, output = remote.command(transport, 'bash -n ' + shlex.quote(ROOT + '/run.pbs')
                                        + ' && bash -n ' + shlex.quote(ROOT + '/aggregate.pbs'), timeout=20)
            if rc:
                raise RuntimeError(output)
            _save(HERE / 'source_evidence.json', evidence)
            result = {'uploaded': True, 'shell_syntax_passed': True, 'source_completion_verified': True}
        elif args.action == 'submit':
            if not args.unit:
                parser.error('--unit is required')
            result = _submit(transport, args.unit)
        elif args.action == 'pull':
            with remote.sftp_client(transport) as sftp:
                handles = per_control.source.read_json(sftp, ROOT + '/control/handles.json') or {}
                observations = {unit: _observe(transport, sftp, unit, handle)
                                for unit, handle in handles.items() if unit in JOBS}
                _save(HERE / 'completion_observation.json', {
                    'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': observations,
                    'verified': {unit: _passed(value) for unit, value in observations.items()}})
                for unit, observation in observations.items():
                    if observation.get('summary'):
                        _save(HERE / 'results' / unit / 'worker_summary.json', observation['summary'])
                for name in ('comparison.json', 'train_eval/training_receipt.json',
                             'calibration/joint_per_channel/summary.json',
                             'heldout/joint_per_channel/summary.json'):
                    value = per_control.source.read_json(sftp, ROOT + '/results/' + name)
                    if value is not None:
                        _save(HERE / 'results' / name, value)
                report_path = ROOT + '/results/RESULTS.zh.md'
                try:
                    with sftp.file(report_path, 'rb') as stream:
                        data = stream.read(remote.LIMIT + 1)
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Report exceeds bounded read limit')
                    (HERE / 'results/RESULTS.zh.md').parent.mkdir(parents=True, exist_ok=True)
                    (HERE / 'results/RESULTS.zh.md').write_bytes(data)
                except OSError as exc:
                    if exc.errno != 2:
                        raise
                for unit, observation in observations.items():
                    if unit != 'train_eval' or observation.get('artifact_exit_code') is None:
                        continue
                    job_id = observation['handle']['job_id']
                    artifact = ROOT + '/artifacts/' + job_id
                    evidence = HERE / 'results/evidence' / job_id
                    for filename in ('clip_selfcheck.json', 'channel_selfcheck.json', 'joint_selfcheck.json'):
                        value = per_control.source.read_json(sftp, artifact + '/' + filename)
                        if value is not None:
                            _save(evidence / filename, value)
                    try:
                        with sftp.file(artifact + '/job.log', 'rb') as stream:
                            head = stream.read(4096)
                            stream.seek(max(0, sftp.stat(artifact + '/job.log').st_size - 12288))
                            tail = stream.read(12288)
                        samples = list(dict.fromkeys(line for line in (head + b'\n' + tail).decode(
                            'utf-8', errors='replace').splitlines() if line.startswith('GPU_SAMPLE timestamp=')))
                        _save(evidence / 'gpu_telemetry.json', {'source': artifact + '/job.log',
                            'read_scope': 'first4096/last12288 bytes', 'samples': samples})
                    except OSError as exc:
                        if exc.errno != 2:
                            raise
            result = {'pulled_units': sorted(observations), 'files': ['comparison.json',
                'train_eval/training_receipt.json', 'RESULTS.zh.md', 'selfchecks', 'bounded GPU telemetry']}
        else:
            result = _status(transport)
            if args.action == 'logs':
                if not args.unit:
                    parser.error('--unit is required')
                item = result['units'].get(args.unit)
                if not item:
                    raise RuntimeError('No exact recorded handle exists for this unit')
                handle = item['handle']
                rc, output = remote.command(transport, 'tail -n 24 '
                    + shlex.quote(ROOT + '/artifacts/' + handle['job_id'] + '/job.log')
                    + ' 2>/dev/null; tail -n 10 '
                    + shlex.quote(ROOT + '/artifacts/' + handle['job_name'] + '_' + str(handle['attempt']) + '.pbs.out')
                    + ' 2>/dev/null', timeout=25)
                result = {'job_id': handle['job_id'], 'query_rc': rc, 'tail': output[-11000:]}
        print(json.dumps(result, indent=2))
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

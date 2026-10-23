"""Bounded small-file PBS operations; no numerical workload on login nodes."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('control_kl_previous', HERE.parent/'fastwam-vq-relational-kl/control.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
remote, source = previous.remote, previous.c.source
P = json.loads((HERE/'protocol.json').read_text())
ROOT = P['remote_root']


def prerequisites(unit):
    if unit == 'prepare':
        return ()
    if unit == 'sensor_0':
        return ('prepare',)
    if unit == 'coverage_inventory':
        return ('prepare',)
    if unit == 'calibration_pairs':
        return ('coverage_inventory',)
    if unit == 'training_pairs':
        return ('coverage_inventory',)
    if unit == 'training_teacher_plan':
        return ('training_pairs', 'calibration_0', 'calibration_1')
    if unit == 'training_teacher':
        return ('training_teacher_plan', 'aggregate_calibration')
    if unit == 'sensor_1':
        return ('sensor_0',)
    if unit in ('capture_mse_0', 'capture_mse_1', 'capture_rel_0', 'capture_rel_1'):
        return ('sensor_0',)
    if unit in ('reader_0', 'reader_1'):
        return ('sensor_'+unit[-1],)
    if unit in ('calibration_0', 'calibration_1'):
        return ('aggregate_sensor', 'calibration_pairs')
    if unit == 'parameter_gate':
        return ('aggregate_calibration', 'reader_0')
    if unit == 'gate_selfcheck':
        return ('prepare',)
    if unit == 'training_loss_selfcheck':
        return ('prepare',)
    if unit == 'training_runtime_selfcheck':
        return ('prepare', 'training_loss_selfcheck')
    if unit == 'training_evaluation_selfcheck':
        return ('prepare',)
    if unit == 'training_closedloop_selfcheck':
        return ('prepare', 'training_evaluation_selfcheck')
    if unit == 'final_c80_selfcheck':
        return ('prepare', 'training_evaluation_selfcheck')
    match = re.fullmatch(r'final_c80_([a-z][a-z0-9_]{0,25})_([01])', unit)
    if match:
        needed = ('training_teacher', 'training_evaluation_selfcheck', 'final_c80_selfcheck',
                  'aggregate_c80_parameter_gate', 'training_loss_selfcheck', 'training_runtime_selfcheck')
        return needed if match[1] == 'mse_reference' else ('train_' + match[1], *needed)
    if unit == 'aggregate_final_c80':
        config = json.loads((HERE/'training_protocol.json').read_text(encoding='utf-8'))
        return tuple('final_c80_' + arm + '_' + str(shard)
                     for arm in ('mse_reference', *config['arms']) for shard in (0, 1))
    match = re.fullmatch(r'closedloop_([a-z][a-z0-9_]{0,25})_([01])', unit)
    if match:
        return ('train_' + match[1], 'training_evaluation_selfcheck',
                'training_closedloop_selfcheck')
    match = re.fullmatch(r'aggregate_closedloop_([a-z][a-z0-9_]{0,25})', unit)
    if match:
        return tuple('closedloop_' + match[1] + '_' + str(shard) for shard in (0, 1))
    match = re.fullmatch(r'heldout_([a-z][a-z0-9_]{0,25})_([01])', unit)
    if match:
        return ('train_' + match[1], 'training_evaluation_selfcheck')
    match = re.fullmatch(r'aggregate_heldout_([a-z][a-z0-9_]{0,25})', unit)
    if match:
        return tuple('heldout_' + match[1] + '_' + str(shard) for shard in (0, 1))
    if re.fullmatch(r'train_[a-z][a-z0-9_]{0,25}', unit):
        return ('aggregate_c80_parameter_gate', 'training_teacher',
                'training_loss_selfcheck', 'training_runtime_selfcheck')
    if unit == 'trajectory_trace':
        return ('aggregate_trajectory',)
    match = re.fullmatch(r'c80_grad_(response|latent)_([01])', unit)
    if match:
        return ('parameter_gate', 'gate_selfcheck', 'calibration_' + match[2])
    match = re.fullmatch(r'c80_direction_(response|latent)', unit)
    if match:
        return tuple('c80_grad_' + match[1] + '_' + str(shard) for shard in (0, 1))
    match = re.fullmatch(r'c80_eval_(response|latent)_([01])', unit)
    if match:
        return ('c80_direction_' + match[1], 'calibration_' + match[2])
    if unit == 'aggregate_c80_parameter_gate':
        return tuple('c80_eval_' + kind + '_' + str(shard)
                     for kind in ('response', 'latent') for shard in (0, 1))
    if unit in ('trajectory_0', 'trajectory_1'):
        return ('capture_mse_'+unit[-1], 'capture_rel_'+unit[-1], 'reader_0')
    if unit in ('aggregate_sensor', 'aggregate_reader', 'aggregate_trajectory', 'aggregate_calibration'):
        prefix = unit.removeprefix('aggregate_')
        return (prefix+'_0', prefix+'_1')
    if unit == 'aggregate_coverage':
        return ('calibration_0', 'calibration_1', 'sensor_0', 'sensor_1', 'trajectory_0', 'trajectory_1')
    raise ValueError('Unimplemented unit: '+unit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def observe(t, f, unit, handle):
    prefix = ROOT+'/artifacts/'+handle['job_id']
    row = {'handle': handle, 'scheduler': source.previous.scheduler(t, handle['job_id']),
           'summary': source.read_json(f, ROOT+'/results/'+unit+'/summary.json'),
           'progress': source.read_json(f, ROOT+'/results/'+unit+'/progress.json'),
           'exit_code': None, 'marker': False}
    try:
        with f.file(prefix+'/exit_code.txt') as stream:
            row['exit_code'] = int(stream.read(32))
        f.stat(prefix+'/PHASE_COMPLETE')
        row['marker'] = True
    except OSError as exc:
        if exc.errno != 2:
            raise
    q, summary = row['scheduler'], row['summary'] or {}
    row['passed'] = bool(q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
                         and row['exit_code'] == 0 and row['marker'] and summary.get('complete') is True
                         and summary.get('unit') == unit
                         and summary.get('pbs_jobid') == handle['job_id'])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'repair', 'adopt', 'status', 'pull', 'logs'))
    parser.add_argument('--unit')
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.unit and not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', args.unit):
        parser.error('Invalid unit')
    jump, transport = remote.direct_connect()
    try:
        with remote.sftp_client(transport) as f:
            if args.action == 'upload':
                rc, text = remote.command(transport, 'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'), timeout=20)
                if rc:
                    raise RuntimeError(text)
                old = source.read_json(f, ROOT+'/protocol.json')
                if old and old != P:
                    raise RuntimeError('Remote protocol is frozen')
                names = []
                for path in HERE.iterdir():
                    if path.suffix not in ('.py', '.pbs') and path.name not in (
                            'protocol.json', 'parameter_gate_protocol.json', 'c80_gate_protocol.json', 'training_data_protocol.json', 'training_protocol.json'):
                        continue
                    if path.name in ('parameter_gate_protocol.json', 'c80_gate_protocol.json', 'training_data_protocol.json', 'training_protocol.json'):
                        prior_gate = source.read_json(f, ROOT+'/'+path.name)
                        if prior_gate and prior_gate != json.loads(path.read_text(encoding='utf-8')):
                            raise RuntimeError('Child parameter-gate protocol is frozen: ' + path.name)
                    data = path.read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Only small source/control uploads are permitted')
                    with f.file(ROOT+'/'+path.name+'.tmp', 'wb') as stream:
                        stream.write(data)
                    f.posix_rename(ROOT+'/'+path.name+'.tmp', ROOT+'/'+path.name)
                    names.append(path.name)
                    if path.suffix == '.pbs':
                        rc, output = remote.command(transport, 'bash -n '+shlex.quote(ROOT+'/'+path.name), timeout=15)
                        if rc:
                            raise RuntimeError(output)
                result = {'uploaded': names, 'shell_syntax_passed': True}
            else:
                handles = source.read_json(f, ROOT+'/control/handles.json') or {}
                if args.action == 'adopt':
                    if args.unit != 'sensor_1' or 'sensor_0' not in handles:
                        raise RuntimeError('Only a completed sensor_1 result in the sensor_0 allocation may be adopted')
                    if args.unit in handles:
                        result = handles[args.unit]
                    else:
                        parent = handles['sensor_0']
                        if not observe(transport, f, 'sensor_0', parent)['passed']:
                            raise RuntimeError('Parent allocation is not successfully complete')
                        candidate = {**parent, 'unit': 'sensor_1', 'shared_allocation_with': 'sensor_0',
                                     'submitted_new_job': False, 'status': 'completed_shared_allocation'}
                        if not observe(transport, f, 'sensor_1', candidate)['passed']:
                            raise RuntimeError('No successfully completed sensor_1 result in the parent allocation')
                        handles['sensor_1'] = candidate
                        source.write_json(f, ROOT+'/control/handles.json', handles)
                        save(HERE/'handles.json', handles)
                        result = candidate
                elif args.action in ('submit', 'repair'):
                    unit = args.unit
                    if not unit:
                        parser.error('--unit required')
                    needed = prerequisites(unit)
                    if source.read_json(f, ROOT+'/protocol.json') != P:
                        raise RuntimeError('Remote protocol differs')
                    repair = None
                    if args.action == 'repair':
                        if unit not in handles or not args.reason:
                            raise RuntimeError('Known unit and explicit repair reason required')
                        prior = observe(transport, f, unit, handles[unit])
                        q = prior['scheduler']
                        if q['query_rc'] != 0 or q['job_state'] not in ('F', 'X') or prior['passed']:
                            raise RuntimeError('A verified failed terminal job is required; live/unknown jobs cannot restart')
                        repair = handles[unit]
                        save(HERE/('failed_'+unit+'_'+repair['job_id']+'.json'), prior)
                    if unit in handles and repair is None:
                        result = handles[unit]
                    else:
                        preparation = unit in ('prepare', 'coverage_inventory')
                        cpu = (preparation or unit in ('calibration_pairs', 'training_pairs', 'training_teacher_plan', 'gate_selfcheck', 'training_loss_selfcheck', 'training_runtime_selfcheck', 'training_evaluation_selfcheck', 'training_closedloop_selfcheck', 'final_c80_selfcheck', 'trajectory_trace')
                               or unit.startswith(('aggregate', 'c80_direction_')))
                        script = 'prepare.pbs' if preparation else ('cpu.pbs' if cpu else 'gpu.pbs')
                        f.stat(ROOT+'/'+script)
                        for dependency in needed:
                            if dependency not in handles or not observe(transport, f, dependency, handles[dependency])['passed']:
                                raise RuntimeError('Verified complete prerequisite is required: '+dependency)
                        training_arm = unit[6:] if unit.startswith('train_') else None
                        evaluation_match = re.fullmatch(r'(heldout|closedloop|final_c80)_([a-z][a-z0-9_]{0,25})_([01])', unit)
                        aggregate_match = re.fullmatch(r'aggregate_(heldout|closedloop)_([a-z][a-z0-9_]{0,25})', unit)
                        if evaluation_match or aggregate_match:
                            training_arm = (evaluation_match or aggregate_match)[2]
                        if training_arm is not None or unit == 'aggregate_final_c80':
                            config_path = HERE/'training_protocol.json'
                            if not config_path.is_file():
                                raise RuntimeError('A frozen training_protocol.json is required before submission')
                            training_config = json.loads(config_path.read_text(encoding='utf-8'))
                            reference_eval = (evaluation_match and evaluation_match[1] == 'final_c80'
                                              and training_arm == 'mse_reference')
                            if (source.read_json(f, ROOT+'/training_protocol.json') != training_config
                                    or 'mse_reference' in training_config.get('arms', {})
                                    or (training_arm is not None and training_arm not in training_config.get('arms', {})
                                        and not reference_eval)):
                                raise RuntimeError('Local/remote training config or requested arm differs')
                        intent_name = unit+('_repair_of_'+repair['job_id'] if repair else '')+'_intent.json'
                        intent = ROOT+'/control/'+intent_name
                        if source.read_json(f, intent):
                            raise RuntimeError('Reconcile existing submission intent; never duplicate qsub')
                        h = {'unit': unit, 'created_utc': datetime.now(timezone.utc).isoformat(),
                             'gpus': 0 if cpu else 1, 'walltime': '00:20:00' if cpu else '01:59:00',
                             'status': 'intent_before_qsub'}
                        if repair:
                            h.update(repair_of=repair['job_id'], repair_reason=args.reason, previous_attempt=repair)
                        source.write_json(f, intent, h)
                        save(HERE/intent_name, h)
                        job_environment = 'FW_CONTROL_UNIT='+unit
                        if repair:
                            job_environment += ',FW_CONTROL_REPAIR_OF='+repair['job_id']
                        command = ('qsub -N fwctrl'+str(len(handles)+1)+' -v '+shlex.quote(job_environment)
                                   +' -o '+shlex.quote(ROOT+'/artifacts/'+unit+'.pbs.out')+' '+shlex.quote(ROOT+'/'+script))
                        rc, output = remote.command(transport, command, timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', output.strip()):
                            raise RuntimeError('Unknown/rejected qsub outcome; reconcile intent: '+output[:400])
                        h.update(status='confirmed', job_id=output.strip())
                        handles[unit] = h
                        source.write_json(f, intent, h)
                        source.write_json(f, ROOT+'/control/handles.json', handles)
                        save(HERE/'handles.json', handles)
                        result = h
                else:
                    units = {u: observe(transport, f, u, h) for u, h in handles.items() if not args.unit or u == args.unit}
                    snapshot = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': units}
                    save(HERE/'last_status.json', snapshot)
                    if args.action == 'logs':
                        if not args.unit or args.unit not in handles:
                            parser.error('Known --unit required')
                        rc, output = remote.command(transport, 'tail -c 10000 '+shlex.quote(ROOT+'/artifacts/'+handles[args.unit]['job_id']+'/job.log'), timeout=20)
                        result = {'rc': rc, 'tail': output}
                    else:
                        if args.action == 'pull':
                            for unit, row in units.items():
                                if row['summary']:
                                    save(HERE/'results'/unit/'summary.json', row['summary'])
                            for name in ('sensor_plan',):
                                value = source.read_json(f, ROOT+'/data/'+name+'.json')
                                if value:
                                    save(HERE/'data'/(name+'.json'), value)
                            value = source.read_json(f, ROOT+'/results/prepare/first_chunk_metrics.json')
                            if value:
                                save(HERE/'results/prepare/first_chunk_metrics.json', value)
                        result = {'checked_utc': snapshot['checked_utc'], 'units': {u: {
                            'job_id': r['handle']['job_id'], 'scheduler': r['scheduler'], 'passed': r['passed'],
                            'summary': r['summary'], 'progress': r['progress']} for u, r in units.items()}}
            print(json.dumps(result, indent=2))
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()

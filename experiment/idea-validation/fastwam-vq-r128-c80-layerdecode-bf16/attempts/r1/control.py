"""Bounded, durable PBS control for the two mutually exclusive profiling lanes."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('layerdecode_control_base', HERE.parent / 'fastwam-vq-r128-c80-fused-tuning' / 'control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
_original_save_state = control._save_state
MODES = ('bf16', 'eager', 'input_fused', 'layer_decode_fp32', 'layer_decode_bf16_u', 'layer_decode_bf16_all')
SOURCES = ('protocol.json', 'run.pbs', 'environment.sh', 'reference_profile.py', 'export_runtime.py',
           'fused_input.py', 'fused_vq.py', 'kernel_runtime.py', 'layer_runtime.py', 'profile_worker.py', 'finalize.py')


def configure(lane):
    p = json.loads((HERE / lane / 'protocol.json').read_text())
    control.HERE = HERE / lane
    control.PROTOCOL_ID = p['protocol']
    control.ROOT = p['remote_root']
    control.JOB_NAME = p['job_name']
    control.JOB_NAME_RE = re.compile('^' + re.escape(control.JOB_NAME) + '$')
    control.EXECUTION_QUEUE = p['allocation']['execution_queue']
    control.SELECT = '1:ncpus=16:mem=110gb:ngpus=1'
    control.WALLTIME = p['allocation']['walltime']
    control.ALLOCATION = {k:p['allocation'][k] for k in ('queue', 'execution_queue', 'gpus', 'ncpus', 'memory_gb', 'walltime')}
    control.FILES = SOURCES
    control.ARTIFACTS = control.ROOT + '/artifacts'
    control.RESULTS = control.ROOT + '/results'
    control.STATE_PATH = control.ROOT + '/control/state.json'
    return p


def local_bytes(name):
    path = control.HERE / name if name == 'protocol.json' else HERE / name
    if not path.is_file() or path.stat().st_size >= control.LIMIT:
        raise RuntimeError('Missing or oversized source: ' + str(path))
    data = path.read_bytes().replace(b'\r\n', b'\n')
    data.decode('utf-8')
    return data


def save_state(transport, value):
    if value.get('handle'):
        value['handle'].update(ncpus=16, memory_gb=110)
    _original_save_state(transport, value)


def complete_gate(value):
    h, s = value.get('handle') or {}, value.get('scheduler') or {}
    r, a = value.get('summary') or {}, value.get('artifact') or {}
    return (h.get('remote_root') == control.ROOT and h.get('protocol') == control.PROTOCOL_ID
            and h.get('job_name') == control.JOB_NAME and s.get('query_rc') == 0
            and s.get('job_state') == 'F' and s.get('Exit_status') == '0'
            and s.get('Job_Name') == control.JOB_NAME and s.get('queue') == control.EXECUTION_QUEUE
            and s.get('Resource_List.walltime') == control.WALLTIME
            and a.get('exit_code') == '0' and a.get('pipeline_complete') is True
            and r.get('protocol') == control.PROTOCOL_ID and r.get('pbs_jobid') == h.get('job_id')
            and r.get('complete') is True and r.get('stage1_gate', {}).get('passed') is True
            and r.get('stage2_complete') is True and r.get('new_training_updates') == 0
            and r.get('closed_loop_episodes') == 0 and r.get('full_queries_per_mode') == 80
            and set(r.get('modes', {})) == set(MODES))


def pull(transport):
    value = control.status(transport)
    if not value.get('handle'):
        raise RuntimeError('No submitted job')
    target = control.HERE / 'pulled' / value['handle']['job_id']
    files, skipped = [], {}
    names = ['summary.json', 'progress.json', 'storage.json', 'stage1_gate.json', 'kernel_validation.json', 'RESULTS.zh.md']
    for mode in MODES:
        names.extend(mode + '/' + n for n in ('summary.json', 'samples.json', 'validation.json', 'operators.json'))
    with control.remote.sftp_client(transport) as sftp:
        for name in names:
            data, reason = control._pull_bytes(sftp, control.RESULTS + '/' + name)
            if reason:
                skipped[name] = reason
            elif data is not None:
                dest = 'RESULTS.generated.md' if name == 'RESULTS.zh.md' else name
                control._save_bytes(target / dest, data)
                files.append(dest)
    control._save_bytes(target / 'status.json', (json.dumps(value, indent=2) + '\n').encode())
    return {**control._compact(value), 'pulled_files': files, 'skipped': skipped}


def race(transport):
    states = {}
    for lane in ('long', 'short'):
        configure(lane)
        states[lane] = control.status(transport)
    p = configure('long')
    with control.remote.sftp_client(transport) as sftp:
        winner = control._json(sftp, p['race_root'] + '/control/winner.json')
    winner_lane = winner.get('lane') if winner else None
    if winner_lane is None:
        running = [lane for lane,v in states.items() if (v.get('scheduler') or {}).get('query_rc') == 0
                   and (v.get('scheduler') or {}).get('job_state') == 'R']
        if len(running) == 1:
            winner_lane = running[0]
    canceled = []
    if winner_lane:
        expected = (states[winner_lane].get('handle') or {}).get('job_id')
        if winner and winner.get('job_id') != expected:
            raise RuntimeError('Shared winner does not match the submitted handle')
        other = 'short' if winner_lane == 'long' else 'long'
        value = states[other]
        scheduler = value.get('scheduler') or {}
        handle = value.get('handle') or {}
        if scheduler.get('query_rc') == 0 and scheduler.get('job_state') in ('Q', 'R', 'H'):
            configure(other)
            if scheduler.get('Job_Name') != control.JOB_NAME or not control.HANDLE.fullmatch(handle.get('job_id', '')):
                raise RuntimeError('Refusing cancellation without exact job identity')
            rc, output = control.remote.command(transport, 'qdel ' + shlex.quote(handle['job_id']), timeout=35)
            checked = control._scheduler(transport, handle['job_id'])
            canceled.append(dict(lane=other, job_id=handle['job_id'], qdel_rc=rc,
                                 response=output[-1000:], scheduler_after=checked))
    result = dict(checked_utc=control.utc_now(), winner=winner, winner_lane=winner_lane,
                  lanes={lane:control._compact(v) for lane,v in states.items()}, cancellations=canceled)
    control._save_bytes(HERE / 'race_status.json', (json.dumps(result, indent=2) + '\n').encode())
    return result


control._local_bytes = local_bytes
control._save_state = save_state
control.complete_gate = complete_gate
control.pull = pull


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'verify', 'logs', 'pull', 'race'))
    parser.add_argument('--lane', choices=('long', 'short'), default='long')
    args = parser.parse_args()
    configure(args.lane)
    jump, transport = control.direct_connect()
    try:
        if args.action == 'race': result = race(transport)
        elif args.action == 'upload': result = {'uploaded':control.upload(transport)}
        elif args.action == 'submit': result = control.submit(transport)
        elif args.action == 'status': result = control._compact(control.status(transport))
        elif args.action == 'logs': result = control.logs(transport)
        elif args.action == 'pull': result = pull(transport)
        else:
            value = control.status(transport)
            result = {**control._compact(value), 'verified_complete':complete_gate(value)}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if args.action == 'verify' and not result['verified_complete'] else 0
    finally:
        transport.close()
        if jump: jump.close()


if __name__ == '__main__':
    raise SystemExit(main())

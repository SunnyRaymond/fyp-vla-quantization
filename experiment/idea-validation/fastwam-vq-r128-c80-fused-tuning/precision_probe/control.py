"""Reuse the bounded durable PBS controller for one numerical precision probe."""
from pathlib import Path
import importlib.util
import json
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fused_precision_control_base', HERE.parent / 'control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
control.HERE = HERE
control.PROTOCOL_ID = 'fastwam-vq-r128-c80-fused-precision-v1'
control.ROOT = '/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-fused-precision-20261024-r2'
control.EXECUTION_QUEUE = 'gdev'
control.JOB_NAME = 'vqfp0002'
control.JOB_NAME_RE = re.compile(r'^vqfp0002$')
control.SELECT = '1:ncpus=16:mem=110gb:ngpus=1'
control.WALLTIME = '00:30:00'
control.ALLOCATION = dict(queue='normal', execution_queue='gdev', gpus=1,
                          ncpus=16, memory_gb=110, walltime=control.WALLTIME)
control.FILES = ('protocol.json', 'run.pbs', 'environment.sh', 'precision_worker.py')
control.ARTIFACTS = control.ROOT + '/artifacts'
control.RESULTS = control.ROOT + '/results'
control.STATE_PATH = control.ROOT + '/control/state.json'
_save_state = control._save_state


def save_state(transport, value):
    if value.get('handle'):
        value['handle'].update(ncpus=16, memory_gb=110)
    _save_state(transport, value)


def complete_gate(value):
    h, s = value.get('handle') or {}, value.get('scheduler') or {}
    r, a = value.get('summary') or {}, value.get('artifact') or {}
    return (h.get('remote_root') == control.ROOT and h.get('job_name') == control.JOB_NAME
            and h.get('protocol') == control.PROTOCOL_ID
            and s.get('query_rc') == 0 and s.get('job_state') == 'F'
            and s.get('Exit_status') == '0' and s.get('Job_Name') == control.JOB_NAME
            and s.get('queue') == 'gdev' and s.get('Resource_List.walltime') == control.WALLTIME
            and a.get('exit_code') == '0' and a.get('pipeline_complete') is True
            and r.get('protocol') == control.PROTOCOL_ID and r.get('pbs_jobid') == h.get('job_id')
            and r.get('complete') is True and r.get('completed_shapes') == 6
            and r.get('all_finite') is True)


def pull(transport):
    value = control.status(transport)
    job = value['handle']['job_id']
    target = HERE / 'pulled' / job
    files = []
    with control.remote.sftp_client(transport) as sftp:
        for name in ('progress.json', 'summary.json', 'precision.json'):
            data, reason = control._pull_bytes(sftp, control.RESULTS + '/' + name)
            if reason:
                raise RuntimeError(name + ': ' + reason)
            if data is not None:
                control._save_bytes(target / name, data)
                files.append(name)
    control._save_bytes(target / 'status.json', (json.dumps(control._compact(value), indent=2) + '\n').encode())
    return {**control._compact(value), 'pulled_files': files}


control.complete_gate = complete_gate
control.pull = pull
control._save_state = save_state

if __name__ == '__main__':
    raise SystemExit(control.main())

"""Bounded SSH/PBS control. Login nodes only see small files and scheduler calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('future_reader_trusted', HERE.parent / 'fastwam-vq-proxy-diagnostic/control.py')
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
ROOT = '/scratch/users/ntu/yguo017/fastwam-vq-residual-capacity-20261016'
trusted.HERE, trusted.ROOT = HERE, ROOT
trusted.FILES = ('protocol.json','common.py','allocation.py','capacity.py','score.py','evaluate.py','summarize.py','prepare.py','test_capacity.py','environment.sh','run.pbs','README.zh.md','control.py')
remote, save, scheduler, read_json = trusted.remote, trusted.save, trusted.scheduler, trusted.read_remote_json
PHASES = ('prepare','score','allocate','evaluate','analyze')
GPU = ('score','evaluate')


def lane(phase, shard):
    if phase not in PHASES or shard not in range(8) or (phase == 'score' and shard >= 4):
        raise ValueError('Invalid phase/shard')
    return f'{phase}_{shard}' if phase in GPU else phase


def inspect(t, phase, shard=0, tail=False, require=False):
    key = lane(phase, shard)
    h = json.loads((HERE / f'{key}_handle.json').read_text())
    observed = scheduler(t, h['job_id'])
    save(HERE / f'{key}_scheduler.json', observed)
    if observed['query_rc']:
        if require:
            raise RuntimeError('Scheduler query is inconclusive')
        return observed
    artifact = f"{ROOT}/artifacts/{h['job_id']}"
    if observed['job_state'] in ('F', 'X') and observed['Exit_status'] == '0':
        summary = read_json(t, artifact + '/summary.json')
        if summary['job_id'] != h['job_id'] or summary['protocol'] != 'fastwam-vq-residual-capacity-v1':
            raise RuntimeError('Terminal artifact identity mismatch')
        marker = 'PIPELINE_COMPLETE' if summary['status'] == 'complete' else 'RESUME_READY'
        rc, _ = remote.command(t, f'test -f {shlex.quote(artifact + "/" + marker)} && '
                              f'test "$(cat {shlex.quote(artifact + "/exit_code.txt")})" = 0', timeout=25)
        observed.update(summary=summary, markers_valid=rc == 0)
        save(HERE / f'{key}_terminal.json', observed)
    if tail and observed['job_state'] in ('R', 'F', 'X'):
        rc, output = remote.command(t, f'test ! -f {shlex.quote(artifact + "/job.log")} || tail -c 5000 '
                                   f'{shlex.quote(artifact + "/job.log")}', timeout=25)
        observed['log_tail'] = output
    if require:
        s = observed.get('summary', {})
        if (observed['job_state'] not in ('F', 'X') or observed['Exit_status'] != '0'
                or not observed.get('markers_valid') or s.get('status') != 'complete'):
            raise RuntimeError(f'Lane is not complete: {key}')
        if phase in GPU:
            config=json.loads((HERE/'protocol.json').read_text())
            expected = 2 if phase == 'score' else [9,9,7,7,7,7,7,7][shard]*len(config['allocation_arms'])
            if (s['phase'] != phase or s['shard'] != shard or (s['context_count'] if phase == 'score' else s['row_count']) != expected):
                raise RuntimeError('Incomplete reader coverage')
        elif phase == 'prepare':
            if s['target_linears'] != 614 or s['context_count'] != 60 or s['model_forward_count'] != 0:
                raise RuntimeError('Incomplete prepared references')
        elif phase == 'allocate' and s['score_context_count'] != 8:
            raise RuntimeError('Incomplete score coverage')
        elif phase == 'analyze' and (s['total_contexts'] != 60 or s['test_case_count'] != 22 or s['row_count'] != 60*len(json.loads((HERE/'protocol.json').read_text())['allocation_arms'])):
            raise RuntimeError('Incomplete analysis coverage')
    return observed


def prerequisites(t, phase):
    if phase == 'prepare': return
    inspect(t, 'prepare', require=True)
    if phase == 'allocate':
        for shard in range(4): inspect(t, 'score', shard, require=True)
    if phase in ('evaluate','analyze'): inspect(t, 'allocate', require=True)
    if phase == 'analyze':
        for shard in range(8): inspect(t, 'evaluate', shard, require=True)


def submit(t, phase, shard=0, attempt='a', resume_job=None):
    key = lane(phase, shard)
    if not re.fullmatch('[a-z][a-z0-9]{0,5}', attempt):
        raise ValueError('Invalid attempt label')
    prerequisites(t, phase)
    handle = HERE / f'{key}_handle.json'
    intent = HERE / f'{key}_intent_{attempt}.json'
    if handle.exists() and resume_job is None:
        return json.loads(handle.read_text())
    short = {'prepare':'p','score':'s','allocate':'a','evaluate':'e','analyze':'f'}[phase]
    name = f'fwrc16{short}{shard}{attempt}'
    receipt = json.loads(intent.read_text()) if intent.exists() else {}
    if receipt and receipt.get('resume_job') != resume_job:
        raise RuntimeError('Attempt identity changed')
    if receipt.get('job_id'):
        save(handle, receipt)
        return receipt
    rc, output = remote.command(t, f'qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1', timeout=35)
    if rc not in (0, 1) or (rc == 1 and output.strip()):
        raise RuntimeError('Cannot establish submission history')
    jobs = output.split()
    if jobs:
        if len(jobs) != 1 or not trusted.HANDLE.fullmatch(jobs[0]):
            raise RuntimeError('Ambiguous matching submission')
        receipt.update(job_id=jobs[0], status='recovered', phase=phase, shard=shard, resume_job=resume_job)
    else:
        if receipt:
            raise RuntimeError('Prior submission outcome unknown; do not resubmit')
        gpu = phase in GPU
        resources = 'select=1:ncpus=16:mem=110gb:ngpus=1' if gpu else 'select=1:ncpus=8:mem=48gb'
        walltime = '02:00:00' if gpu else '00:20:00'
        receipt = dict(status='intent_before_qsub', created_utc=datetime.now(timezone.utc).isoformat(),
                       phase=phase, shard=shard, job_name=name, resume_job=resume_job, remote_root=ROOT,
                       resources=resources, walltime=walltime)
        save(intent, receipt)
        variables = f'FW_PHASE={phase},FW_SHARD={shard}'
        cmd = (f'qsub -q normal -P personal-yguo017 -N {shlex.quote(name)} -l {shlex.quote(resources)} '
               f'-l walltime={walltime} -v {shlex.quote(variables)} {shlex.quote(ROOT + "/run.pbs")}')
        rc, output = remote.command(t, cmd, timeout=40)
        if rc or not trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status='unknown_or_rejected', message=output[:1500])
            save(intent, receipt)
            raise RuntimeError('qsub outcome unknown/rejected; no automatic duplicate')
        receipt.update(job_id=output.strip(), status='confirmed')
    save(intent, receipt)
    save(handle, receipt)
    return receipt


def continue_job(t, phase, shard):
    before = inspect(t, phase, shard)
    if (before['query_rc'] or before['job_state'] not in ('F', 'X') or before['Exit_status'] != '0'
            or not before.get('markers_valid') or before['summary']['status'] != 'resumable'):
        raise RuntimeError('Continuation needs verified successful resumable terminal evidence')
    key, index = lane(phase, shard), 1
    while (HERE / f'{key}_intent_c{index}.json').exists():
        old = json.loads((HERE / f'{key}_intent_c{index}.json').read_text())
        if old['resume_job'] == before['job_id']:
            return submit(t, phase, shard, f'c{index}', before['job_id'])
        index += 1
    return submit(t, phase, shard, f'c{index}', before['job_id'])


def main():
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=('upload', 'submit', 'status', 'verify', 'continue', 'pull'))
    a.add_argument('--phase', choices=PHASES, default='prepare')
    a.add_argument('--shard', type=int, choices=range(8), default=0)
    a.add_argument('--tail', action='store_true')
    a.add_argument('--files', nargs='+')
    args = a.parse_args()
    j, t = remote.direct_connect()
    try:
        if args.action == 'upload':
            trusted.upload(t, args.files)
        elif args.action == 'submit':
            print(json.dumps(submit(t, args.phase, args.shard)))
        elif args.action == 'continue':
            print(json.dumps(continue_job(t, args.phase, args.shard)))
        elif args.action in ('status', 'verify'):
            print(json.dumps(inspect(t, args.phase, args.shard, args.tail, args.action == 'verify')))
        else:
            trusted.pull(t, args.files)
    finally:
        t.close()
        if j:
            j.close()


if __name__ == '__main__':
    main()

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
ROOT = '/scratch/users/ntu/yguo017/fastwam-vq-future-nmse-matched-20261017'
trusted.HERE, trusted.ROOT = HERE, ROOT
trusted.FILES = ('protocol.json','common.py','matching.py','prepare.py','run_reader.py','analyze.py','test_design.py','environment.sh','run.pbs','README.zh.md','control.py')
remote, save, scheduler, read_json = trusted.remote, trusted.save, trusted.scheduler, trusted.read_remote_json
PHASES = ('prepare','matched','analyze')


def lane(phase,shard):
    if phase not in PHASES or shard not in (0,1) or (phase!='matched' and shard!=0):raise ValueError('Invalid phase/shard')
    return f'matched_{shard}' if phase=='matched' else phase


def inspect(t,phase,shard=0,tail=False,require=False):
    key=lane(phase,shard);h=json.loads((HERE/f'{key}_handle.json').read_text())
    o=scheduler(t,h['job_id']);save(HERE/f'{key}_scheduler.json',o)
    if o['query_rc']:
        if require:raise RuntimeError('Scheduler query is inconclusive')
        return o
    artifact=f"{ROOT}/artifacts/{h['job_id']}"
    if o['job_state'] in ('F','X') and o['Exit_status']=='0':
        summary=read_json(t,artifact+'/summary.json')
        if summary['job_id']!=h['job_id'] or summary['protocol']!='fastwam-vq-future-nmse-matched-v1' or summary['phase']!=phase:raise RuntimeError('Terminal artifact identity mismatch')
        marker='PIPELINE_COMPLETE' if summary['status']=='complete' else 'RESUME_READY'
        rc,_=remote.command(t,'test -f '+shlex.quote(artifact+'/'+marker)+' && test "$(cat '+shlex.quote(artifact+'/exit_code.txt')+')" = 0',timeout=25)
        o.update(summary=summary,markers_valid=rc==0);save(HERE/f'{key}_terminal.json',o)
    if tail and o['job_state'] in ('R','F','X'):
        rc,log=remote.command(t,'test ! -f '+shlex.quote(artifact+'/job.log')+' || tail -c 5000 '+shlex.quote(artifact+'/job.log'),timeout=25);o['log_tail']=log
    if require:
        ss=o.get('summary',{})
        if o['job_state'] not in ('F','X') or o['Exit_status']!='0' or not o.get('markers_valid') or ss.get('status')!='complete':raise RuntimeError('Lane is not complete: '+key)
        if phase=='prepare' and (ss['context_count']!=60 or ss['matched_count']!=60 or ss['cached_reader_rows']!=120 or ss['model_forward_count']!=0):raise RuntimeError('Incomplete matched controls')
        if phase=='matched' and (ss['shard']!=shard or ss['context_count']!=30 or ss['row_count']!=30 or ss['replay_count_this_segment']!=4 or ss['replay_max_abs']!=0):raise RuntimeError('Incomplete reader shard')
        if phase=='analyze' and (ss['context_count']!=60 or ss['row_count']!=180 or ss['new_matched_rows']!=60 or ss['cached_reference_rows']!=120 or ss['test_case_count']!=22):raise RuntimeError('Incomplete paired analysis')
    return o


def prerequisites(t,phase):
    if phase=='prepare':return
    inspect(t,'prepare',require=True)
    if phase=='analyze':
        for i in (0,1):inspect(t,'matched',i,require=True)


def submit(t, phase, shard=0, attempt='a', resume_job=None):
    key = lane(phase, shard)
    if not re.fullmatch('[a-z][a-z0-9]{0,5}', attempt):
        raise ValueError('Invalid attempt label')
    prerequisites(t, phase)
    handle = HERE / f'{key}_handle.json'
    intent = HERE / f'{key}_intent_{attempt}.json'
    if handle.exists() and resume_job is None:
        return json.loads(handle.read_text())
    short = {'prepare':'p','matched':'m','analyze':'f'}[phase]
    name = f'fwnm17{short}{shard}{attempt}'
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
        gpu = phase=='matched'
        resources = 'select=1:ncpus=16:mem=110gb:ngpus=1' if gpu else 'select=1:ncpus=8:mem=32gb'
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
    a.add_argument('--shard', type=int, choices=(0, 1), default=0)
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

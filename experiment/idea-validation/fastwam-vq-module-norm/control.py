"""Small-file SSH/PBS control only; no tensors or numerical work on login."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
ROOT = '/scratch/users/ntu/yguo017/fastwam-vq-module-norm-20261017'
spec = importlib.util.spec_from_file_location('conflict_trusted', HERE.parent / 'fastwam-vq-proxy-diagnostic/control.py')
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
trusted.HERE, trusted.ROOT = HERE, ROOT
trusted.FILES = ('protocol.json','common.py','prepare.py','prepare_norms.py','test_norms.py','storage.py','environment.sh','run.pbs','control.py','README.zh.md','analyze.py','evaluate.py','fast_a4.py','batched_transform.py','transform_checks.py')
remote, save, scheduler, read_json = trusted.remote, trusted.save, trusted.scheduler, trusted.read_remote_json
PHASES = ('prepare','evaluate','analyze')
GPU = ('evaluate',)


def lane(phase, shard=0):
    if phase not in PHASES or shard not in range(4):
        raise ValueError('Invalid phase/shard')
    return f'{phase}_{shard}' if phase in GPU else phase


def inspect(t, phase, shard=0, tail=False, require=False):
    key = lane(phase,shard)
    h = json.loads((HERE / f'{key}_handle.json').read_text())
    r = scheduler(t,h['job_id'])
    save(HERE / f'{key}_scheduler.json',r)
    artifact = f"{ROOT}/artifacts/{h['job_id']}"
    if not r['query_rc'] and r['job_state'] in ('F','X') and r['Exit_status']=='0':
        s = read_json(t,artifact+'/summary.json')
        if s['protocol']!='fastwam-vq-module-norm-v1' or s['job_id']!=h['job_id'] or s['phase']!=phase:
            raise RuntimeError('Terminal receipt identity mismatch')
        marker='PIPELINE_COMPLETE' if s['status']=='complete' else 'RESUME_READY'
        rc,_=remote.command(t,f'test -f {shlex.quote(artifact+"/"+marker)} && test "$(cat {shlex.quote(artifact+"/exit_code.txt")})" = 0',timeout=25)
        r.update(summary=s,markers_valid=rc==0)
        if phase in GPU and s.get('runtime_transform_implementation')=='batched_h128_v2':
            check=read_json(t,artifact+'/transform_check.json')
            if not check.get('passed') or not check.get('bitwise_equal') or check.get('fixtures')!=218 or check.get('max_abs')!=0 or not check.get('forward_inverse_checked'):
                raise RuntimeError('Batched transform GPU parity check failed or missing')
            r['transform_check']=check
        save(HERE/f'{key}_terminal.json',r)
    if tail and not r['query_rc'] and r['job_state'] in ('R','F','X'):
        _,r['log_tail']=remote.command(t,f'test ! -f {shlex.quote(artifact+"/job.log")} || tail -c 4500 {shlex.quote(artifact+"/job.log")}',timeout=25)
    if require:
        s=r.get('summary',{})
        if r['query_rc'] or r['job_state'] not in ('F','X') or r['Exit_status']!='0' or not r.get('markers_valid') or s.get('status')!='complete':
            raise RuntimeError(f'Lane not verified complete: {key}')
        if phase=='evaluate' and (s['shard']!=shard or s['row_count']!=[192,192,168,168][shard]):
            raise RuntimeError('Incomplete intervention coverage')
    return r


def prerequisites(t,phase):
    if phase=='evaluate': inspect(t,'prepare',require=True)
    if phase=='analyze':
        for shard in range(4): inspect(t,'evaluate',shard,require=True)


def submit(t,phase,shard=0,attempt='a',resume_job=None):
    key=lane(phase,shard)
    if not re.fullmatch('[a-z][a-z0-9]{0,5}',attempt):
        raise ValueError('Invalid attempt')
    handle=HERE/f'{key}_handle.json'
    if handle.exists() and resume_job is None:
        return json.loads(handle.read_text())
    prerequisites(t,phase)
    tag={'prepare':'p','evaluate':'e','analyze':'a'}[phase]
    name=f'fwmn17{tag}{shard}{attempt}'
    intent=HERE/f'{key}_intent_{attempt}.json'
    receipt=json.loads(intent.read_text()) if intent.exists() else {}
    if receipt and receipt.get('resume_job')!=resume_job:
        raise RuntimeError('Attempt resume identity changed')
    if receipt.get('job_id'):
        save(handle,receipt)
        return receipt
    rc,output=remote.command(t,f'qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1',timeout=35)
    if rc not in (0,1) or (rc==1 and output.strip()):
        raise RuntimeError('Submission history unavailable; no qsub')
    jobs=output.split()
    if jobs:
        if len(jobs)!=1 or not trusted.HANDLE.fullmatch(jobs[0]):
            raise RuntimeError('Ambiguous prior submission')
        receipt.update(job_id=jobs[0],status='recovered',phase=phase,shard=shard,resume_job=resume_job)
    else:
        if receipt:
            raise RuntimeError('Previous submission outcome unknown; do not duplicate')
        resources='select=1:ncpus=16:mem=110gb:ngpus=1' if phase in GPU else 'select=1:ncpus=8:mem=32gb'
        walltime='02:00:00' if phase in GPU else '00:30:00'
        receipt=dict(status='intent_before_qsub',created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,shard=shard,
                     job_name=name,resume_job=resume_job,remote_root=ROOT,resources=resources,walltime=walltime)
        save(intent,receipt)
        cmd=f'qsub -q normal -P personal-yguo017 -N {name} -l {resources} -l walltime={walltime} -v FW_PHASE={phase},FW_SHARD={shard} {ROOT}/run.pbs'
        rc,output=remote.command(t,cmd,timeout=40)
        if rc or not trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status='unknown_or_rejected',message=output[:1500])
            save(intent,receipt)
            raise RuntimeError('qsub rejected/unknown; no automatic duplicate')
        receipt.update(job_id=output.strip(),status='confirmed')
    save(intent,receipt); save(handle,receipt)
    return receipt


def continue_job(t,phase,shard):
    r=inspect(t,phase,shard)
    if r['query_rc'] or r['job_state'] not in ('F','X') or r['Exit_status']!='0' or not r.get('markers_valid') or r['summary']['status']!='resumable':
        raise RuntimeError('Continuation requires verified successful RESUME_READY terminal')
    i=1
    while (HERE/f'{lane(phase,shard)}_intent_c{i}.json').exists():
        old=json.loads((HERE/f'{lane(phase,shard)}_intent_c{i}.json').read_text())
        if old['resume_job']==r['job_id']:
            return submit(t,phase,shard,f'c{i}',r['job_id'])
        i+=1
    return submit(t,phase,shard,f'c{i}',r['job_id'])


def main():
    a=argparse.ArgumentParser()
    a.add_argument('action',choices=('upload','submit','status','verify','continue','pull'))
    a.add_argument('--phase',choices=PHASES,default='prepare')
    a.add_argument('--shard',type=int,choices=range(4),default=0)
    a.add_argument('--tail',action='store_true')
    a.add_argument('--files',nargs='+')
    args=a.parse_args()
    j,t=remote.direct_connect()
    try:
        if args.action=='upload': trusted.upload(t,args.files)
        elif args.action=='pull': trusted.pull(t,args.files)
        elif args.action=='submit': print(json.dumps(submit(t,args.phase,args.shard)))
        elif args.action=='continue': print(json.dumps(continue_job(t,args.phase,args.shard)))
        else: print(json.dumps(inspect(t,args.phase,args.shard,args.tail,args.action=='verify')))
    finally:
        t.close()
        if j: j.close()


if __name__=='__main__': main()

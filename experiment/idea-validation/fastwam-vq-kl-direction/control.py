"""Bounded PBS control for KL parity/direction jobs; compute stays in allocations."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('kl_source_control',HERE.parent/'fastwam-vq-dual-kl/control.py')
source=importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
remote=source.remote; P=json.loads((HERE/'protocol.json').read_text()); ROOT=P['remote_root']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8'); temp.replace(path)


def observe(t,f,unit,h):
    prefix=ROOT+'/artifacts/'+h['job_id']; summary=source.source.read_json(f,ROOT+'/results/'+unit+'/summary.json')
    row={'handle':h,'scheduler':source.source.previous.scheduler(t,h['job_id']),
         'progress':source.source.read_json(f,ROOT+'/results/'+unit+'/progress.json'),
         'summary':summary,'exit_code':None,'marker':False,'selfcheck':None}
    try:
        with f.file(prefix+'/exit_code.txt') as stream: row['exit_code']=int(stream.read(32))
        f.stat(prefix+'/PHASE_COMPLETE'); row['marker']=True
    except OSError as exc:
        if exc.errno!=2: raise
    row['selfcheck']=source.source.read_json(f,prefix+'/direction_selfcheck.json')
    if unit=='aggregate':
        row['passed']=(row['scheduler']['query_rc']==0 and row['scheduler']['job_state'] in ('F','X')
            and row['scheduler']['Exit_status']=='0' and row['exit_code']==0 and row['marker']
            and bool(summary and summary.get('complete') and summary['pbs_jobid']==h['job_id']
                     and summary['parity_cases']==320 and summary['gradient_queries']==320
                     and summary['direction_conditions']==24 and summary['training_updates']==0))
        return row
    row['passed']=(row['scheduler']['query_rc']==0 and row['scheduler']['job_state'] in ('F','X')
        and row['scheduler']['Exit_status']=='0' and row['exit_code']==0 and row['marker']
        and bool(row['selfcheck'] and row['selfcheck'].get('status')=='passed'
                 and row['selfcheck'].get('cuda')=='passed' and row['selfcheck'].get('pbs_jobid')==h['job_id'])
        and bool(summary and summary.get('complete') and summary['pbs_jobid']==h['job_id']
                 and summary['parity_cases']==80 and summary['gradient_query_count']==80
                 and summary['parity_action_max_abs']==0 and summary['parity_future_max_abs']==0
                 and len(summary['direction_conditions'])==6 and summary['training_updates']==0
                 and summary['original_parameters_restored']))
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','submit','status','pull','logs'))
    parser.add_argument('--unit',choices=(*P['units'],'aggregate'));parser.add_argument('--resume',action='store_true')
    parser.add_argument('--rerun-aggregate',action='store_true');args=parser.parse_args()
    jump,t=remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action=='upload':
                rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
                if rc:raise RuntimeError(out)
                existing=source.source.read_json(f,ROOT+'/protocol.json')
                if existing and existing!=P:raise RuntimeError('Frozen diagnostic protocol differs')
                names=('protocol.json','worker.py','gpu.pbs','direction_utils.py','direction_selfcheck.py','summarize.py','cpu.pbs')
                for name in names:
                    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n');assert len(data)<remote.LIMIT
                    with f.file(ROOT+'/'+name+'.tmp','wb') as stream:stream.write(data)
                    f.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
                for script in ('gpu.pbs','cpu.pbs'):
                    rc,out=remote.command(t,'bash -n '+shlex.quote(ROOT+'/'+script),timeout=20)
                    if rc:raise RuntimeError(out)
                result={'uploaded':list(names)}
            else:
                handles=source.source.read_json(f,ROOT+'/control/handles.json') or {}
                if args.action=='submit':
                    if not args.unit:parser.error('--unit required')
                    unit=args.unit; previous=handles.get(unit); should_submit=previous is None
                    if args.rerun_aggregate:
                        if unit!='aggregate' or args.resume or not previous or not observe(t,f,unit,previous)['passed']:
                            raise RuntimeError('Report rerun requires a completed passed CPU aggregate')
                        should_submit=True
                    if args.resume:
                        if not previous:raise RuntimeError('No known job to resume')
                        observed=observe(t,f,unit,previous)
                        if not (observed['scheduler']['query_rc']==0 and observed['scheduler']['job_state'] in ('F','X')
                                and observed['scheduler']['Exit_status']=='75' and observed['exit_code']==75
                                and (observed['progress'] or {}).get('phase')=='checkpointed'):
                            raise RuntimeError('Resume needs verified terminal checkpointed exit75')
                        should_submit=True
                    if not should_submit:result=previous
                    else:
                        if source.source.read_json(f,ROOT+'/protocol.json')!=P:raise RuntimeError('Protocol differs')
                        if unit=='aggregate':
                            for dependency in P['units']:
                                if dependency not in handles or not observe(t,f,dependency,handles[dependency])['passed']:
                                    raise RuntimeError('Aggregate requires complete passed direction jobs: '+dependency)
                        attempt=len((previous or {}).get('job_ids',[]))+1
                        intent=ROOT+'/control/'+unit+'_attempt'+str(attempt)+'.json'
                        if source.source.read_json(f,intent):raise RuntimeError('Reconcile existing intent; never duplicate')
                        cpu=unit=='aggregate'
                        short_resume=args.resume and not cpu
                        walltime='00:10:00' if cpu else ('00:45:00' if short_resume else '01:59:00')
                        h={'unit':unit,'status':'intent_before_qsub','gpus':0 if cpu else 1,'walltime':walltime,
                           'job_ids':(previous or {}).get('job_ids',[]),'created_utc':datetime.now(timezone.utc).isoformat()}
                        source.source.write_json(f,intent,h);save(HERE/(unit+'_intent.json'),h)
                        variables='FW_DIRECTION_UNIT='+unit+(',FW_DIRECTION_SOFT_DEADLINE_SECONDS=2520' if short_resume else '')
                        rc,out=remote.command(t,'qsub -N fwkdir'+(' -l walltime='+walltime if short_resume else '')+' -v '+shlex.quote(variables)
                            +' -o '+shlex.quote(ROOT+'/artifacts/'+unit+'.pbs.out')+' '+shlex.quote(ROOT+('/cpu.pbs' if cpu else '/gpu.pbs')),timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()):raise RuntimeError('Uncertain qsub: '+out[:400])
                        h.update(status='confirmed',job_id=out.strip());h['job_ids'].append(h['job_id']);handles[unit]=h
                        source.source.write_json(f,intent,h);source.source.write_json(f,ROOT+'/control/handles.json',handles)
                        save(HERE/'handles.json',handles);result=h
                else:
                    rows={u:observe(t,f,u,h) for u,h in handles.items() if not args.unit or u==args.unit}
                    save(HERE/'last_status.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),'units':rows})
                    if args.action=='logs':
                        if not args.unit:parser.error('--unit required')
                        rc,tail=remote.command(t,'tail -c 9000 '+shlex.quote(ROOT+'/artifacts/'+handles[args.unit]['job_id']+'/job.log'),timeout=20)
                        result={'query_rc':rc,'tail':tail}
                    else:
                        if args.action=='pull':
                            for u,row in rows.items():
                                if row['summary']:save(HERE/'results'/u/'summary.json',row['summary'])
                                if row['selfcheck']:save(HERE/'results'/u/'direction_selfcheck.json',row['selfcheck'])
                            comparison=source.source.read_json(f,ROOT+'/results/comparison.json')
                            if comparison:save(HERE/'results/comparison.json',comparison)
                            for name in ('ANALYSIS.zh.md','real_loss_steps.png'):
                                try:
                                    with f.file(ROOT+'/results/'+name,'rb') as stream:data=stream.read(remote.LIMIT+1)
                                except OSError as exc:
                                    if exc.errno==2:continue
                                    raise
                                if len(data)>remote.LIMIT:raise RuntimeError('Report exceeds bounded retrieval: '+name)
                                (HERE/'results'/name).write_bytes(data)
                        result={'units':{u:{'job_id':r['handle']['job_id'],'scheduler':r['scheduler'],'passed':r['passed'],
                            'phase':(r['progress'] or {}).get('phase'),'cases':(r['progress'] or {}).get('cases'),
                            'direction':(r['progress'] or {}).get('direction'),'fraction':(r['progress'] or {}).get('fraction')} for u,r in rows.items()}}
            print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':main()

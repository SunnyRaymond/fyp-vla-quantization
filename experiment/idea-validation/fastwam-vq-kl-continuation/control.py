"""Small PBS controls for the matched continuation experiment."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('continuation_source',HERE.parent/'fastwam-vq-dual-kl/control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
remote=c.remote
P=json.loads((HERE/'protocol.json').read_text());ROOT=P['remote_root']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def observe(t,f,unit,h):
    prefix=ROOT+'/artifacts/'+h['job_id']
    row={'handle':h,'scheduler':c.source.previous.scheduler(t,h['job_id']),
         'summary':c.source.read_json(f,ROOT+'/results/'+unit+'/summary.json'),
         'progress':c.source.read_json(f,ROOT+'/results/'+unit+'/progress.json'),'artifact_exit':None,'marker':False}
    try:
        with f.file(prefix+'/exit_code.txt') as stream:row['artifact_exit']=int(stream.read(32))
        f.stat(prefix+'/PHASE_COMPLETE');row['marker']=True
    except OSError as exc:
        if exc.errno!=2:raise
    q,s=row['scheduler'],row['summary'] or {}
    if unit=='aggregate':
        coverage=s.get('heldout_records')==320 and s.get('calibration_records')==320 and s.get('additional_updates_per_arm')==80
    else:
        coverage=(s.get('updates')==80 and s.get('total_updates')==160 and s.get('evaluation_queries')==160
            and s.get('new_teacher_queries')==0 and s.get('source_optimizer_discarded') is True
            and s.get('snapshots')=={'0':80,'20':80,'40':80,'60':80,'80':80})
    row['passed']=(q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='0'
        and row['artifact_exit']==0 and row['marker'] and s.get('complete') is True and s.get('pbs_jobid')==h['job_id'] and coverage)
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','submit','status','logs','pull'))
    parser.add_argument('--unit',choices=(*P['arms'],'aggregate'));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    jump,t=remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action=='upload':
                rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
                if rc:raise RuntimeError(out)
                existing=c.source.read_json(f,ROOT+'/protocol.json')
                if existing and existing!=P:raise RuntimeError('Continuation protocol frozen')
                decision=json.loads((HERE/'decision.json').read_text())
                if not decision.get('complete') or decision.get('heldout_used') is not False or decision.get('sampling_scope')!='C80' or decision.get('learning_rate_factor')!=P['learning_rate_factor']:
                    raise RuntimeError('Complete C80-only learning-rate decision required')
                names=('protocol.json','decision.json','worker.py','gpu.pbs','analysis.py','cpu.pbs')
                for name in names:
                    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n');assert len(data)<remote.LIMIT
                    with f.file(ROOT+'/'+name+'.tmp','wb') as stream:stream.write(data)
                    f.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
                for name in ('gpu.pbs','cpu.pbs'):
                    rc,out=remote.command(t,'bash -n '+shlex.quote(ROOT+'/'+name),timeout=20)
                    if rc:raise RuntimeError(out)
                result={'uploaded':list(names)}
            else:
                handles=c.source.read_json(f,ROOT+'/control/handles.json') or {}
                if args.action=='submit':
                    if not args.unit:parser.error('--unit required')
                    unit=args.unit;previous=handles.get(unit);submit=previous is None
                    if args.resume:
                        if not previous:raise RuntimeError('No known resumable job')
                        r=observe(t,f,unit,previous)
                        if not (r['scheduler']['query_rc']==0 and r['scheduler']['job_state'] in ('F','X') and r['scheduler']['Exit_status']=='75'
                                and r['artifact_exit']==75 and (r['progress'] or {}).get('phase')=='checkpointed'):
                            raise RuntimeError('Resume requires verified terminal checkpointed exit75')
                        submit=True
                    if not submit:result=previous
                    else:
                        if c.source.read_json(f,ROOT+'/protocol.json')!=P:raise RuntimeError('Protocol differs')
                        decision=c.source.read_json(f,P['decision_file'])
                        if not decision or not decision['complete'] or decision['heldout_used'] or decision['learning_rate_factor']!=P['learning_rate_factor']:
                            raise RuntimeError('C80-only decision differs')
                        if unit=='aggregate':
                            for dependency in P['arms']:
                                if dependency not in handles or not observe(t,f,dependency,handles[dependency])['passed']:
                                    raise RuntimeError('Incomplete training prerequisite: '+dependency)
                        intent=ROOT+'/control/'+unit+'_attempt'+str(len((previous or {}).get('job_ids',[]))+1)+'.json'
                        if c.source.read_json(f,intent):raise RuntimeError('Reconcile existing submission intent')
                        cpu=unit=='aggregate';h={'unit':unit,'status':'intent_before_qsub','gpus':0 if cpu else 1,
                            'walltime':'00:15:00' if cpu else '01:59:00','created_utc':datetime.now(timezone.utc).isoformat(),
                            'job_ids':(previous or {}).get('job_ids',[])}
                        c.source.write_json(f,intent,h);save(HERE/(unit+'_intent.json'),h)
                        rc,out=remote.command(t,'qsub -N fwklcont -v '+shlex.quote('FW_KL_UNIT='+unit)
                            +' -o '+shlex.quote(ROOT+'/artifacts/'+unit+'.pbs.out')+' '+shlex.quote(ROOT+('/cpu.pbs' if cpu else '/gpu.pbs')),timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()):raise RuntimeError('Uncertain submission: '+out[:400])
                        h.update(status='confirmed',job_id=out.strip());h['job_ids'].append(h['job_id']);handles[unit]=h
                        c.source.write_json(f,intent,h);c.source.write_json(f,ROOT+'/control/handles.json',handles);save(HERE/'handles.json',handles);result=h
                else:
                    rows={u:observe(t,f,u,h) for u,h in handles.items() if not args.unit or u==args.unit}
                    save(HERE/'last_status.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),'units':rows})
                    if args.action=='logs':
                        if not args.unit:parser.error('--unit required')
                        rc,tail=remote.command(t,'tail -c 9000 '+shlex.quote(ROOT+'/artifacts/'+handles[args.unit]['job_id']+'/job.log'),timeout=20)
                        result={'query_rc':rc,'tail':tail}
                    else:
                        if args.action=='pull':
                            for unit,row in rows.items():
                                if row['summary']:save(HERE/'results'/unit/'summary.json',row['summary'])
                            for name in ('comparison','paired_details','decision'):
                                value=c.source.read_json(f,ROOT+'/results/'+name+'.json')
                                if value:save(HERE/'results'/(name+'.json'),value)
                            for name in ('ANALYSIS.zh.md','c80_training_curves.png'):
                                try:
                                    with f.file(ROOT+'/results/'+name,'rb') as stream:data=stream.read(remote.LIMIT+1)
                                except OSError as exc:
                                    if exc.errno==2:continue
                                    raise
                                if len(data)>remote.LIMIT:raise RuntimeError('Report exceeds retrieval bound')
                                (HERE/'results'/name).parent.mkdir(exist_ok=True);(HERE/'results'/name).write_bytes(data)
                        result={'units':{u:{'job_id':r['handle']['job_id'],'scheduler':r['scheduler'],'passed':r['passed'],
                            'phase':(r['progress'] or {}).get('phase'),'updates':(r['progress'] or {}).get('updates'),
                            'snapshot_step':(r['progress'] or {}).get('snapshot_step'),'snapshot_queries':(r['progress'] or {}).get('snapshot_queries'),
                            'queries':(r['progress'] or {}).get('queries')} for u,r in rows.items()}}
            print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':main()

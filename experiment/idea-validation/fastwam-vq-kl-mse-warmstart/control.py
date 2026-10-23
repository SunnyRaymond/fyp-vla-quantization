"""PBS controls for matched one-batch refinement; login work stays bounded."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('refine_source',HERE.parent/'fastwam-vq-dual-kl/control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
remote=c.remote
P=json.loads((HERE/'protocol.json').read_text());ROOT=P['remote_root']
ARMS=[P['arm']]
REPAIRABLE_ERRORS={
    "TypeError: object of type 'generator' has no len()": 'mean accepts generator',
    "AttributeError: 'numpy.ndarray' object has no attribute 'cpu'": 'runtime.command returns NumPy; preserve its values via tolist',
}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def decision_ok(value):
    return bool(value and value.get('complete') and value.get('heldout_used') is False
                and value.get('sampling_scope')=='C80' and value.get('method')=='frozen_C80_KL_candidate_from_MSE')


def observe(t,f,unit,h):
    prefix=ROOT+'/artifacts/'+h['job_id'];s=c.source.read_json(f,ROOT+'/results/'+unit+'/summary.json') or {}
    row={'handle':h,'scheduler':c.source.previous.scheduler(t,h['job_id']),
         'summary':s,'progress':c.source.read_json(f,ROOT+'/results/'+unit+'/progress.json'),
         'artifact_exit':None,'marker':False,'selfcheck':None}
    if row['progress'] and row['progress'].get('pbs_jobid')!=h['job_id']:
        row['previous_progress']=row['progress'];row['progress']=None
    try:
        with f.file(prefix+'/exit_code.txt') as stream:row['artifact_exit']=int(stream.read(32))
        f.stat(prefix+'/PHASE_COMPLETE');row['marker']=True
    except OSError as exc:
        if exc.errno!=2:raise
    if unit=='aggregate':
        coverage=(s.get('calibration_records')==80 and s.get('heldout_records')==80
                  and s.get('new_teacher_queries')==0 and s.get('arm')==P['arm']
                  and s.get('checkpoint_arm')=='mse_nmse'
                  and s.get('new_gradient_queries')==0 and s.get('reused_gradient_queries')==80)
    else:
        row['selfcheck']=c.source.read_json(f,prefix+'/direction_selfcheck.json') or {}
        check=row['selfcheck'];selection=s.get('selection') or {}
        coverage=(s.get('source_updates')==80 and s.get('refinement_updates') in (0,1)
                  and s.get('total_updates')==80+s.get('refinement_updates',-1)
                  and s.get('checkpoint_arm')=='mse_nmse'
                  and s.get('gradient_queries_new')==0 and s.get('gradient_queries_reused')==80
                  and s.get('calibration_queries')==s.get('heldout_queries')==80
                  and s.get('evaluation_queries')==160 and s.get('new_teacher_queries')==0
                  and s.get('source_parity_passed') is True and s.get('deployment_parity_passed') is True
                  and s.get('frozen_verified') is True
                  and selection.get('complete') is True and selection.get('heldout_used') is False
                  and check.get('status')=='passed' and check.get('cuda')=='passed' and check.get('pbs_jobid')==h['job_id'])
    q=row['scheduler']
    row['passed']=(q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='0'
                   and row['artifact_exit']==0 and row['marker'] and s.get('complete') is True
                   and s.get('pbs_jobid')==h['job_id'] and coverage)
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','submit','status','logs','pull'))
    parser.add_argument('--unit',choices=(*ARMS,'aggregate'));parser.add_argument('--resume',action='store_true')
    parser.add_argument('--repair',action='store_true');args=parser.parse_args()
    if args.resume and args.repair:parser.error('Choose either checkpointed resume or verified bug repair')
    jump,t=remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action=='upload':
                rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
                if rc:raise RuntimeError(out)
                existing=c.source.read_json(f,ROOT+'/protocol.json')
                if existing and existing!=P:raise RuntimeError('Refinement protocol is frozen')
                if not decision_ok(json.loads((HERE/'decision.json').read_text())):raise RuntimeError('Complete C80-only decision required')
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
                    if args.resume or args.repair:
                        if not previous or (args.resume and unit=='aggregate'):raise RuntimeError('Resume requires a known compatible job')
                        r=observe(t,f,unit,previous)
                        terminal=r['scheduler']['query_rc']==0 and r['scheduler']['job_state'] in ('F','X')
                        repair_description=None
                        if args.repair and unit=='aggregate':
                            rc,tail=remote.command(t,'tail -c 10000 '+shlex.quote(ROOT+'/artifacts/'+previous['job_id']+'/job.log'),timeout=20)
                            expected='RuntimeError: Required artifact is missing: '+P['source_mse_root']+'/results/train_eval/worker_summary.json'
                            verified=(terminal and r['scheduler']['Exit_status']=='1' and r['artifact_exit']==1 and rc==0 and expected in tail)
                            repair_description='Use original remote summary.json and receipt PBS binding; worker_summary.json is a local retrieval alias'
                        elif args.repair:
                            verified=(terminal and r['scheduler']['Exit_status']=='1' and r['artifact_exit']==1
                                      and (r['progress'] or {}).get('phase')=='failed'
                                      and (r['progress'] or {}).get('error') in REPAIRABLE_ERRORS)
                        else:
                            verified=(terminal and r['scheduler']['Exit_status']=='75' and r['artifact_exit']==75
                                      and (r['progress'] or {}).get('phase')=='checkpointed')
                        if not verified:raise RuntimeError('Resume/repair requires its verified terminal failure condition')
                        submit=True
                    if not submit:result=previous
                    else:
                        if c.source.read_json(f,ROOT+'/protocol.json')!=P:raise RuntimeError('Protocol differs')
                        if not decision_ok(c.source.read_json(f,P['decision_file'])):raise RuntimeError('C80-only decision differs')
                        if unit=='aggregate':
                            for dependency in ARMS:
                                if dependency not in handles or not observe(t,f,dependency,handles[dependency])['passed']:
                                    raise RuntimeError('Incomplete refinement prerequisite: '+dependency)
                        attempt=len((previous or {}).get('job_ids',[]))+1
                        intent=ROOT+'/control/'+unit+'_attempt'+str(attempt)+'.json'
                        if c.source.read_json(f,intent):raise RuntimeError('Reconcile existing submission intent; never duplicate')
                        cpu=unit=='aggregate';short=unit=='dual_kl';wall='00:10:00' if cpu else ('00:45:00' if short else '01:59:00')
                        h={'unit':unit,'status':'intent_before_qsub','gpus':0 if cpu else 1,'walltime':wall,
                           'created_utc':datetime.now(timezone.utc).isoformat(),'job_ids':(previous or {}).get('job_ids',[])}
                        if args.repair:h.update(repair_of=previous['job_id'],repair=repair_description or REPAIRABLE_ERRORS[r['progress']['error']])
                        c.source.write_json(f,intent,h);save(HERE/(unit+'_intent.json'),h)
                        variables='FW_REFINE_UNIT='+unit+(',FW_REFINE_SOFT_DEADLINE_SECONDS=2520' if short else '')
                        rc,out=remote.command(t,'qsub -N fwklwarm'+(' -l walltime='+wall if short else '')+' -v '+shlex.quote(variables)
                            +' -o '+shlex.quote(ROOT+'/artifacts/'+unit+'.pbs.out')+' '+shlex.quote(ROOT+('/cpu.pbs' if cpu else '/gpu.pbs')),timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()):raise RuntimeError('Uncertain submission: '+out[:400])
                        h.update(status='confirmed',job_id=out.strip());h['job_ids'].append(h['job_id']);handles[unit]=h
                        c.source.write_json(f,intent,h);c.source.write_json(f,ROOT+'/control/handles.json',handles)
                        save(HERE/'handles.json',handles);result=h
                else:
                    rows={u:observe(t,f,u,h) for u,h in handles.items() if not args.unit or u==args.unit}
                    save(HERE/'last_status.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),'units':rows})
                    if args.action=='logs':
                        if not args.unit:parser.error('--unit required')
                        rc,tail=remote.command(t,'tail -c 10000 '+shlex.quote(ROOT+'/artifacts/'+handles[args.unit]['job_id']+'/job.log'),timeout=20)
                        result={'query_rc':rc,'tail':tail}
                    else:
                        if args.action=='pull':
                            (HERE/'results').mkdir(exist_ok=True)
                            for unit,row in rows.items():
                                if row['summary']:save(HERE/'results'/unit/'summary.json',row['summary'])
                                if unit!='aggregate':
                                    selection=c.source.read_json(f,ROOT+'/results/'+unit+'/selection.json')
                                    if selection:save(HERE/'results'/unit/'selection.json',selection)
                            for name in ('comparison','paired_details','decision'):
                                value=c.source.read_json(f,ROOT+'/results/'+name+'.json')
                                if value:save(HERE/'results'/(name+'.json'),value)
                            try:
                                with f.file(ROOT+'/results/ANALYSIS.zh.md','rb') as stream:data=stream.read(remote.LIMIT+1)
                            except OSError as exc:
                                if exc.errno!=2:raise
                            else:
                                if len(data)>remote.LIMIT:raise RuntimeError('Report exceeds bounded retrieval')
                                (HERE/'results/ANALYSIS.zh.md').write_bytes(data)
                        result={'units':{u:{'job_id':r['handle']['job_id'],'scheduler':r['scheduler'],'passed':r['passed'],
                            'phase':(r['progress'] or {}).get('phase'),'cases':(r['progress'] or {}).get('cases'),
                            'direction':(r['progress'] or {}).get('direction'),'fraction':(r['progress'] or {}).get('fraction'),
                            'split':(r['progress'] or {}).get('split'),'queries':(r['progress'] or {}).get('queries')} for u,r in rows.items()}}
            print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':main()

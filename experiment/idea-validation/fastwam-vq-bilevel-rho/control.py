"""Small control files and PBS queries only; workloads require compute allocations."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bilevel_source_control',HERE.parent/'fastwam-vq-c80-activation-clipping/control.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
remote=source.remote
P=json.loads((HERE/'protocol.json').read_text())
ROOT=P['remote_root']
UNITS=('data_prepare','preflight','search','train_low','train_current','train_selected','outer_evaluation','evaluate_low','evaluate_current','evaluate_selected','aggregate','stage1_adaptive','stage2_rho1','stage2_rho225','stage2_compare')
CPU={'data_prepare','aggregate','stage2_compare'}
FILES=('protocol.json','stage2_protocol.json','completed_prerequisites.json','control.py','prepare_data.py','worker.py','core.py','stage1_adaptive.py','stage2.py','run.pbs','cpu.pbs','README.zh.md','NUMERICS_REPAIR.zh.md')

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)

def read(f,path):
    return source.source.read_json(f,path)

def write(f,path,value):
    return source.source.write_json(f,path,value)

def observe(t,f,unit,h):
    row={'handle':h,'scheduler':source.source.previous.scheduler(t,h['job_id'])}
    for name in ('summary','progress'):
        row[name]=read(f,ROOT+'/results/'+unit+'/'+name+'.json')
    artifacts=ROOT+'/artifacts/'+h['job_id']
    try:
        with f.file(artifacts+'/exit_code.txt','rb') as s:row['artifact_exit_code']=int(s.read(32))
    except OSError as e:
        if e.errno!=2:raise
        row['artifact_exit_code']=None
    try:f.stat(artifacts+'/PHASE_COMPLETE');row['marker']=True
    except OSError as e:
        if e.errno!=2:raise
        row['marker']=False
    return row

def passed(row):
    q=row['scheduler'];s=row.get('summary') or {}
    unit=row['handle']['unit']
    coverage=(s.get('counts')=={'development':16,'outer':8,'confirmation':8} if unit=='data_prepare'
              else len(s.get('states',[]))==2 if unit=='preflight'
              else s.get('rounds')==8 and s.get('updates')==16 if unit=='search'
              else s.get('updates')==16 if unit.startswith('train_')
              else s.get('queries')==24 and s.get('comparator_frozen') is True if unit=='outer_evaluation'
              else s.get('queries')==96 and s.get('split_queries')=={'confirmation':16,'historical':80} if unit.startswith('evaluate_')
              else s.get('queries')==24 and s.get('split_queries')=={'outer':8,'confirmation':16}
                   and s.get('comparison_records')==72 and s.get('cached_baseline_queries')==48
                   and s.get('new_training_updates')==0 and s.get('new_teacher_queries')==0
                   and s.get('master_parameters_unchanged') is True if unit=='stage1_adaptive'
              else s.get('updates')==16 and s.get('queries')==24 and s.get('horizon_queries')=={'1':8,'8':8,'16':8}
                   and s.get('initial_state_exact') is True and s.get('evaluation_restore_exact') is True
                   and s.get('new_teacher_queries')==0 if unit in ('stage2_rho1','stage2_rho225')
              else s.get('candidate_training_updates')==32 and s.get('candidate_queries')==48
                   and s.get('cached_reference_queries')==8 and s.get('selection_frozen_before_new_confirmation') is True if unit=='stage2_compare'
              else s.get('endpoint_records')==312 if unit=='aggregate' else False)
    return (q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='0'
            and row.get('artifact_exit_code')==0 and row.get('marker') is True
            and s.get('complete') is True and s.get('protocol')==P['protocol']
            and s.get('pbs_jobid')==row['handle']['job_id'] and coverage)

def dependency_passed(row,archive):
    if passed(row):return True
    # PBS may purge old terminal history; only a previously captured terminal
    # receipt plus fresh artifacts can qualify an old dependency, never a live unit.
    if row['scheduler']['query_rc']!=153:return False
    prior=(archive or {}).get('units',{}).get(row['handle']['unit'],{})
    if not ((archive or {}).get('complete') and (archive or {}).get('checked_utc')
            and prior.get('job_id')==row['handle']['job_id'] and prior.get('passed') is True
            and prior.get('artifact_exit_code')==0 and prior.get('marker') is True
            and prior.get('matching_complete_summary') is True):return False
    historical={'query_rc':0,'job_state':prior.get('pbs_state'),'Exit_status':prior.get('exit_status')}
    return passed({**row,'scheduler':historical})

def require(t,f,handles,unit):
    archive=read(f,ROOT+'/completed_prerequisites.json')
    if unit not in handles or not dependency_passed(observe(t,f,unit,handles[unit]),archive):
        raise RuntimeError('Verified complete prerequisite required: '+unit)

def status(t,unit=None):
    with remote.sftp_client(t) as f:
        handles=read(f,ROOT+'/control/handles.json') or {}
        rows={u:observe(t,f,u,h) for u,h in handles.items() if unit is None or u==unit}
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),'remote_root':ROOT,'units':rows}

def wall_seconds(value):
    parts=value.split(':')
    if len(parts)!=3 or any(not p.isdigit() for p in parts):raise ValueError('Invalid PBS walltime')
    hours,minutes,seconds=map(int,parts)
    if minutes>=60 or seconds>=60:raise ValueError('Invalid PBS walltime')
    return hours*3600+minutes*60+seconds

def gpu_cost(t,unit=None):
    jobs={}
    with remote.sftp_client(t) as f:
        for name in f.listdir(ROOT+'/control'):
            if not re.fullmatch(r'[a-z0-9_]+_intent_\d+\.json',name):continue
            h=read(f,ROOT+'/control/'+name)
            if h and h.get('job_id') and h.get('gpus',0)>0 and (unit is None or h['unit']==unit):jobs[h['job_id']]=h
    rows=[];by_unit={};complete=bool(jobs)
    for job,h in sorted(jobs.items()):
        q=source.source.previous.scheduler(t,job)
        value=q.get('resources_used.walltime')
        seconds=wall_seconds(value) if value is not None else None
        terminal=q['query_rc']==0 and q['job_state'] in ('F','X')
        complete &= terminal and seconds is not None
        gpu_hours=seconds*h['gpus']/3600 if seconds is not None else None
        row={'unit':h['unit'],'job_id':job,'gpus':h['gpus'],'job_state':q['job_state'],
             'exit_status':q['Exit_status'],'walltime':value,'gpu_hours':gpu_hours,'query_rc':q['query_rc']}
        rows.append(row)
        if gpu_hours is not None:by_unit[h['unit']]=by_unit.get(h['unit'],0)+gpu_hours
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),'complete':bool(complete),'gpu_jobs':rows,
            'by_unit_gpu_hours':by_unit,'total_new_gpu_hours':sum(by_unit.values()),
            'scope':'PBS used walltime x GPUs, including loading/checkpointing and all confirmed attempts; historical source jobs excluded; complete only when all queried GPU jobs are terminal with usage available'}

def retryable_preflight(row):
    q=row['scheduler'];progress=row.get('progress') or {};summary=row.get('summary') or {}
    return (q['query_rc']==0 and q['job_state'] in ('F','X')
            and q['Exit_status'] in ('0','1') and row.get('artifact_exit_code')==int(q['Exit_status'])
            and ((q['Exit_status']=='0' and passed(row) and summary.get('gate_passed') is False)
                 or (q['Exit_status']=='1' and progress.get('phase')=='failed')))


def submit(t,unit,resume=False,retry_reason=None):
    if unit not in UNITS:raise ValueError(unit)
    with remote.sftp_client(t) as f:
        if read(f,ROOT+'/protocol.json')!=P:raise RuntimeError('Remote protocol differs')
        handles=read(f,ROOT+'/control/handles.json') or {};old=handles.get(unit)
        if old and not resume and not retry_reason:return old
        if retry_reason:
            if unit!='preflight' or not old or resume:raise RuntimeError('Repair retry is only for an existing preflight')
            if set(handles)-{'data_prepare','preflight'}:raise RuntimeError('Cannot replace a gate after downstream work exists')
            row=observe(t,f,unit,old)
            if not retryable_preflight(row):raise RuntimeError('Preflight repair retry requires a verified failed terminal attempt')
            archive=ROOT+'/results/preflight/attempt_'+str(old['attempt'])
            rc,out=remote.command(t,'mkdir -p '+shlex.quote(archive),timeout=20)
            if rc:raise RuntimeError(out)
            for name in ('summary','progress','evidence'):
                value=read(f,ROOT+'/results/preflight/'+name+'.json')
                if value is not None:write(f,archive+'/'+name+'.json',value)
            write(f,archive+'/retry_reason.json',{'reason':retry_reason,'previous_handle':old,'verified_terminal':row['scheduler']})
        if resume:
            if not old:raise RuntimeError('No handle to resume')
            row=observe(t,f,unit,old);q=row['scheduler']
            if not(q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='75'
                   and row.get('artifact_exit_code')==75 and (row.get('progress') or {}).get('phase')=='checkpointed'):
                raise RuntimeError('Only verified exit75/checkpointed units may resume')
        if unit=='preflight':require(t,f,handles,'data_prepare')
        if unit not in ('data_prepare','preflight'):
            require(t,f,handles,'preflight')
            if not (read(f,ROOT+'/results/preflight/summary.json') or {}).get('gate_passed'):
                raise RuntimeError('Finite difference/state restoration gate did not pass')
        if unit in ('train_selected','outer_evaluation','evaluate_low','evaluate_current','evaluate_selected','aggregate','stage1_adaptive'):
            require(t,f,handles,'search')
        if unit=='stage1_adaptive':
            for u in ('outer_evaluation','evaluate_current','evaluate_selected'):require(t,f,handles,u)
        if unit.startswith('stage2_'):
            stage2=json.loads((HERE/'stage2_protocol.json').read_text())
            if read(f,ROOT+'/stage2_protocol.json')!=stage2:raise RuntimeError('Frozen stage2 protocol differs')
            for u in ('stage1_adaptive','train_current','outer_evaluation'):require(t,f,handles,u)
            if unit=='stage2_compare':
                for u in ('stage2_rho1','stage2_rho225'):require(t,f,handles,u)
        if unit=='train_selected':
            require(t,f,handles,'train_low');require(t,f,handles,'train_current')
        if unit=='outer_evaluation':
            require(t,f,handles,'train_low');require(t,f,handles,'train_current');require(t,f,handles,'train_selected')
        if unit.startswith('evaluate_'):require(t,f,handles,'outer_evaluation')
        if unit in ('evaluate_current','evaluate_selected'):require(t,f,handles,'evaluate_low')
        if unit=='aggregate':
            for u in ('evaluate_low','evaluate_current','evaluate_selected'):require(t,f,handles,u)
        attempt=old['attempt']+1 if old else 0
        intent_path=ROOT+'/control/'+unit+'_intent_'+str(attempt)+'.json'
        if read(f,intent_path):raise RuntimeError('Submission intent exists; reconcile before retry')
        name='fwbi'+{'data_prepare':'data','preflight':'pre','search':'search','train_low':'low','train_current':'cur','train_selected':'sel','outer_evaluation':'outer','evaluate_low':'el','evaluate_current':'ec','evaluate_selected':'es','aggregate':'agg','stage1_adaptive':'adap','stage2_rho1':'s2lo','stage2_rho225':'s2hi','stage2_compare':'s2cmp'}[unit]+str(attempt)
        cpu=unit in CPU
        if unit=='train_selected':
            selected=read(f,ROOT+'/results/selected_rho.json')
            cpu=any(math.isclose(selected['rho'],rho,rel_tol=1e-12,abs_tol=1e-15) for rho in P['training']['fixed_rhos'].values())
        h={'unit':unit,'attempt':attempt,'job_name':name,'remote_root':ROOT,
           'created_utc':datetime.now(timezone.utc).isoformat(),'status':'intent_before_qsub',
           'gpus':0 if cpu else 1,'walltime':'01:59:00'}
        if retry_reason:h['repair_reason']=retry_reason
        write(f,intent_path,h);save(HERE/(unit+'_intent_'+str(attempt)+'.json'),h)
        cmd='qsub -N '+shlex.quote(name)+' -o '+shlex.quote(ROOT+'/artifacts/'+name+'.pbs.out')+' -v '+shlex.quote('FW_BILEVEL_UNIT='+unit)+' '+shlex.quote(ROOT+('/cpu.pbs' if cpu else '/run.pbs'))
        rc,out=remote.command(t,cmd,timeout=40);job=out.strip()
        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',job):
            h.update(status='unknown_or_rejected',return_code=rc,response=out[:1000]);write(f,intent_path,h);save(HERE/(unit+'_intent_'+str(attempt)+'.json'),h)
            raise RuntimeError('qsub outcome unknown/rejected; never retry blindly')
        h.update(job_id=job,status='confirmed');write(f,intent_path,h)
        if old:write(f,ROOT+'/control/'+unit+'_previous_'+str(old['attempt'])+'.json',old)
        handles[unit]=h;write(f,ROOT+'/control/handles.json',handles);save(HERE/'handles.json',handles)
        return h

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('upload','submit','resume','retry','status','pull','logs','cost'));a.add_argument('--unit',choices=UNITS);a.add_argument('--reason');args=a.parse_args()
    j,t=remote.direct_connect()
    try:
        if args.action=='upload':
            rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
            if rc:raise RuntimeError(out)
            with remote.sftp_client(t) as f:
                current=read(f,ROOT+'/protocol.json')
                if current is not None and current!=P:raise RuntimeError('Do not overwrite frozen protocol')
                stage2=read(f,ROOT+'/stage2_protocol.json')
                if stage2 is not None and stage2!=json.loads((HERE/'stage2_protocol.json').read_text()):raise RuntimeError('Do not overwrite frozen stage2 protocol')
                for name in FILES:
                    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    if len(data)>remote.LIMIT:raise ValueError('Small control files only')
                    with f.file(ROOT+'/'+name+'.tmp','wb') as s:s.write(data)
                    f.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
            result={'uploaded':list(FILES)}
        elif args.action in ('submit','resume','retry'):
            if not args.unit:a.error('--unit required')
            if args.action=='retry' and not args.reason:a.error('--reason required for a repair retry')
            result=submit(t,args.unit,args.action=='resume',args.reason if args.action=='retry' else None)
        elif args.action=='cost':
            result=gpu_cost(t,args.unit);save(HERE/'results'/args.unit/'GPU_COST.json' if args.unit else HERE/'results/GPU_COST.json',result)
        else:
            result=status(t,args.unit);save(HERE/'last_status.json',result)
            if args.action=='pull':
                with remote.sftp_client(t) as f:
                    for u,r in result['units'].items():
                        for n in ('summary','progress'):
                            if r.get(n) is not None:save(HERE/'results'/u/(n+'.json'),r[n])
                    for name in ('reuse.json','selected_rho.json','fixed_comparator.json','comparison.json','preflight/evidence.json'):
                        x=read(f,ROOT+'/results/'+name)
                        if x is not None:save(HERE/'results'/name,x)
                    if args.unit=='stage1_adaptive':
                        x=read(f,ROOT+'/results/stage1_adaptive/comparison.json')
                        if x is not None:save(HERE/'results/stage1_adaptive/comparison.json',x)
                    if args.unit=='stage2_compare':
                        x=read(f,ROOT+'/results/stage2_compare/comparison.json')
                        if x is not None:save(HERE/'results/stage2_compare/comparison.json',x)
                    plan=read(f,ROOT+'/data/plan.json')
                    if plan is not None and plan.get('status')=='complete':save(HERE/'data/plan.json',plan)
            if args.action=='logs':
                if not args.unit:a.error('--unit required')
                h=result['units'][args.unit]['handle']
                rc,out=remote.command(t,'tail -c 9000 '+shlex.quote(ROOT+'/artifacts/'+h['job_id']+'/job.log'),timeout=25);result={'rc':rc,'tail':out}
        print(json.dumps(result,indent=2,ensure_ascii=False))
    finally:
        t.close()
        if j:j.close()

if __name__=='__main__':main()

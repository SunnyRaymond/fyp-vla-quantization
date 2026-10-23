"""Bounded PBS control files only; no workload executes on the login node."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint_rmse_c80_control',HERE.parent/'fastwam-vq-c80-activation-clipping/control.py')
parent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
source=parent.source
remote=source.remote
P=json.loads((HERE/'protocol.json').read_text())
ROOT=P['remote_root']
JOBS={'evaluate':'fw80jrm','aggregate':'fw80jrma'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def observe(transport,sftp,unit,handle):
    prefix=ROOT+'/artifacts/'+handle['job_id']
    result={'handle':handle,'scheduler':source.previous.scheduler(transport,handle['job_id']),
            'phase_complete_marker':False,'artifact_exit_code':None}
    for name in ('summary','progress'):
        filename='results/comparison.json' if unit=='aggregate' and name=='summary' else 'results/'+unit+'/'+name+'.json'
        result[name]=source.read_json(sftp,ROOT+'/'+filename)
    try:
        sftp.stat(prefix+'/PHASE_COMPLETE')
        result['phase_complete_marker']=True
    except OSError as exc:
        if exc.errno!=2:raise
    try:
        with sftp.file(prefix+'/exit_code.txt','rb') as stream:
            result['artifact_exit_code']=int(stream.read(32).strip())
    except OSError as exc:
        if exc.errno!=2:raise
    return result


def source_status(transport):
    target=P['joint_source'];owner=target['root']
    with remote.sftp_client(transport) as sftp:
        p=source.read_json(sftp,owner+'/protocol.json')
        receipt=source.read_json(sftp,target['receipt'])
        progress=source.read_json(sftp,target['progress'])
        result={'ready':False,'reason':'training not complete','root':owner,'protocol':p,'receipt':receipt,'progress':progress}
        if not p:return {**result,'reason':'owner protocol not published yet'}
        if (p.get('protocol')!=target['protocol'] or p.get('source_c80_protocol')!=P['source_c80_protocol']
            or any(p.get(k)!=P[k] for k in ('calibration','training','clipping'))):
            raise RuntimeError('Owner C80 joint contract differs from frozen consumer inputs')
        job=(receipt or {}).get('pbs_jobid') or (progress or {}).get('pbs_jobid')
        if not job:return result
        if not source.previous.HANDLE.fullmatch(job):raise RuntimeError('Invalid source job ID')
        result.update(final_job_id=job,scheduler=source.previous.scheduler(transport,job))
        q=result['scheduler']
        if q['query_rc']!=0:return {**result,'reason':'source scheduler query inconclusive'}
        if not receipt or not receipt.get('complete'):return result
        if (receipt.get('protocol')!=p['protocol'] or receipt.get('arm')!='joint' or receipt.get('updates')!=80
            or receipt.get('target_linears')!=614 or receipt.get('calibration_ids')!=P['calibration']['case_ids']
            or receipt.get('update_order')!=P['calibration']['update_order']
            or [h.get('case_id') for h in receipt.get('history',[])]!=P['calibration']['update_order']):
            raise RuntimeError('Final source receipt does not describe complete C80 joint80')
        prefix=owner+'/artifacts/'+job
        result.update(artifact_exit_code=None,training_complete_marker=False,summary=source.read_json(sftp,prefix+'/calibration_summary.json'))
        try:
            with sftp.file(prefix+'/exit_code.txt','rb') as stream:result['artifact_exit_code']=int(stream.read(32).strip())
        except OSError as exc:
            if exc.errno!=2:raise
        try:
            sftp.stat(prefix+'/TRAINING_COMPLETE');result['training_complete_marker']=True
        except OSError as exc:
            if exc.errno!=2:raise
        summary=result['summary'] or {}
        complete=(q['job_state'] in ('F','X') and q['Exit_status']=='0' and result['artifact_exit_code']==0
            and result['training_complete_marker'] and summary.get('complete') is True
            and summary.get('pbs_jobid')==job and summary.get('updates')==80
            and (progress or {}).get('complete') is True and progress.get('pbs_jobid')==job)
        if complete:
            sftp.stat(target['checkpoint']);sftp.stat(target['reference_action'])
            result.update(ready=True,reason='verified final joint80 deployment')
        return result


def passed(observation):
    q=observation['scheduler'];s=observation.get('summary') or {}
    unit=observation['handle']['unit']
    coverage=(s.get('evaluation_queries')==480 and s.get('new_training_updates')==0 if unit=='aggregate'
              else s.get('evaluation_queries')==160 and s.get('training_updates_this_task')==0
              and s.get('all_parameters_frozen') is True)
    return (q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='0'
            and observation['artifact_exit_code']==0 and observation['phase_complete_marker'] is True
            and s.get('complete') is True and s.get('pbs_jobid')==observation['handle']['job_id'] and coverage)


def status(transport,unit=None):
    with remote.sftp_client(transport) as sftp:
        handles=source.read_json(sftp,ROOT+'/control/handles.json') or {}
        rows={key:observe(transport,sftp,key,h) for key,h in handles.items() if unit is None or key==unit}
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),'upstream':source_status(transport),'units':rows}


def submit(transport,unit,resume=False):
    with remote.sftp_client(transport) as sftp:
        if source.read_json(sftp,ROOT+'/protocol.json')!=P:
            raise RuntimeError('Frozen ablation protocol differs')
        handles=source.read_json(sftp,ROOT+'/control/handles.json') or {}
        old=handles.get(unit)
        if old and not resume:return old
        if resume:
            if not old:raise RuntimeError('No confirmed handle to resume')
            state=observe(transport,sftp,unit,old)
            if (state['scheduler']['query_rc']!=0 or state['scheduler']['job_state'] not in ('F','X')
                    or state['scheduler']['Exit_status']!='75' or state['artifact_exit_code']!=75
                    or (state.get('progress') or {}).get('phase')!='checkpointed'):
                raise RuntimeError('Only a verified soft-deadline evaluation may resume')
        dependency={'aggregate':'evaluate'}.get(unit)
        if dependency and (dependency not in handles or not passed(observe(transport,sftp,dependency,handles[dependency]))):
            raise RuntimeError('Completed prerequisite required: '+dependency)
        if unit=='evaluate':
            upstream=source_status(transport)
            if not upstream['ready']:raise RuntimeError('Final C80 joint checkpoint is not ready: '+upstream['reason'])
            verified=parent.status(transport)
            if not all(parent.passed(verified['units'][key]) for key in ('prepare','train_eval','aggregate')):
                raise RuntimeError('Completed baseline/clip_only controls required')
            evidence={'joint_source':upstream,'controls':verified,'checked_utc':datetime.now(timezone.utc).isoformat()}
            save(HERE/'source_evidence.json',evidence)
            source.write_json(sftp,ROOT+'/control/source_evidence.json',evidence)
        attempt=(old['attempt']+1) if old else 0
        intent=ROOT+'/control/'+unit+'_intent_'+str(attempt)+'.json'
        pending=source.read_json(sftp,intent)
        if pending:
            raise RuntimeError('An existing submission intent requires explicit reconciliation; do not duplicate')
        cpu=unit=='aggregate'
        handle={'unit':unit,'attempt':attempt,'job_name':JOBS[unit],
                'created_utc':datetime.now(timezone.utc).isoformat(),'remote_root':ROOT,
                'status':'intent_before_qsub','gpus':0 if cpu else 1,
                'walltime':'00:10:00' if unit=='aggregate' else '01:59:00'}
        source.write_json(sftp,intent,handle)
        save(HERE/(unit+'_intent_'+str(attempt)+'.json'),handle)
        cmd=('qsub -N '+shlex.quote(JOBS[unit])+' -o '+shlex.quote(ROOT+'/artifacts/'+JOBS[unit]+'_'+str(attempt)+'.pbs.out')
             +' -v '+shlex.quote('FW_C80_UNIT='+unit)+' '+shlex.quote(ROOT+('/aggregate.pbs' if unit=='aggregate' else '/run.pbs')))
        rc,out=remote.command(transport,cmd,timeout=40)
        job=out.strip()
        if rc or not source.previous.HANDLE.fullmatch(job):
            handle.update(status='unknown_or_rejected',return_code=rc,response=out[:1000])
            source.write_json(sftp,intent,handle)
            save(HERE/(unit+'_intent_'+str(attempt)+'.json'),handle)
            raise RuntimeError('qsub outcome unknown/rejected; inspect intent and PBS before retrying')
        handle.update(job_id=job,status='confirmed')
        source.write_json(sftp,intent,handle)
        if old:source.write_json(sftp,ROOT+'/control/'+unit+'_previous_'+str(old['attempt'])+'.json',old)
        handles[unit]=handle
        source.write_json(sftp,ROOT+'/control/handles.json',handles)
        save(HERE/'handles.json',handles)
        return handle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('probe','upload','submit','resume','status','pull','logs'))
    parser.add_argument('--unit',choices=tuple(JOBS))
    args=parser.parse_args()
    jump,t=remote.direct_connect()
    try:
        if args.action=='probe':
            result=source_status(t)
            save(HERE/'upstream_status.json',result)
        elif args.action=='upload':
            rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
            if rc:raise RuntimeError(out)
            with remote.sftp_client(t) as sftp:
                existing=source.read_json(sftp,ROOT+'/protocol.json')
                if existing is not None and existing!=P:raise RuntimeError('Frozen C80 joint RMSE protocol differs')
                for name in ('worker.py','protocol.json','run.pbs','aggregate.pbs'):
                    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    if len(data)>remote.LIMIT:raise RuntimeError('Small control/code files only')
                    with sftp.file(ROOT+'/'+name+'.tmp','wb') as f:f.write(data)
                    sftp.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
            rc,out=remote.command(t,'bash -n '+shlex.quote(ROOT+'/run.pbs')+' && bash -n '+shlex.quote(ROOT+'/aggregate.pbs'),timeout=20)
            if rc:raise RuntimeError(out)
            result={'uploaded':True,'shell_syntax_passed':True}
        elif args.action in ('submit','resume'):
            if not args.unit:parser.error('--unit required')
            result=submit(t,args.unit,args.action=='resume')
        else:
            result=status(t,args.unit)
            save(HERE/'last_status.json',result)
            if args.action=='pull':
                with remote.sftp_client(t) as sftp:
                    files=['comparison.json']+[
                        split+'/'+P['arm']+'/summary.json' for split in ('calibration','heldout')]
                    for filename in files:
                        value=source.read_json(sftp,ROOT+'/results/'+filename)
                        if value:save(HERE/'results'/filename,value)
                    try:
                        with sftp.file(ROOT+'/results/RESULTS.zh.md','rb') as f:data=f.read(remote.LIMIT+1)
                        if len(data)>remote.LIMIT:raise RuntimeError('Report exceeds bounded limit')
                        (HERE/'results/RESULTS.zh.md').write_bytes(data)
                    except OSError as exc:
                        if exc.errno!=2:raise
                    for key,o in result['units'].items():
                        if o.get('summary'):save(HERE/'results'/key/'worker_summary.json',o['summary'])
                        if o['handle']['gpus']==1 and o.get('artifact_exit_code') is not None:
                            prefix=ROOT+'/artifacts/'+o['handle']['job_id']
                            for filename in ('clip_selfcheck.json',):
                                check=source.read_json(sftp,prefix+'/'+filename)
                                save(HERE/'results/evidence'/o['handle']['job_id']/filename,check)
                            with sftp.file(prefix+'/job.log','rb') as f:
                                head=f.read(4096);f.seek(max(0,sftp.stat(prefix+'/job.log').st_size-12288));tail=f.read(12288)
                            samples=list(dict.fromkeys(line for line in (head+b'\n'+tail).decode('utf-8',errors='replace').splitlines() if line.startswith('GPU_SAMPLE timestamp=')))
                            save(HERE/'results/evidence'/o['handle']['job_id']/'gpu_telemetry.json',{'source':prefix+'/job.log','read_scope':'first4096/last12288 bytes','samples':samples})
            elif args.action=='logs':
                if not args.unit:parser.error('--unit required')
                h=result['units'][args.unit]['handle']
                rc,out=remote.command(t,'tail -n 24 '+shlex.quote(ROOT+'/artifacts/'+h['job_id']+'/job.log')+' 2>/dev/null; tail -n 10 '+shlex.quote(ROOT+'/artifacts/'+h['job_name']+'_'+str(h['attempt'])+'.pbs.out')+' 2>/dev/null',timeout=25)
                result={'job_id':h['job_id'],'query_rc':rc,'tail':out[-11000:]}
        if 'units' in result:
            upstream=result.get('upstream') or {}
            shown={**result,'upstream':{k:upstream.get(k) for k in ('ready','reason','final_job_id','scheduler')},
                'units':{key:{**row,'summary':{k:v for k,v in (row.get('summary') or {}).items()
                    if k not in ('training','arms','history')},'progress':row.get('progress')} for key,row in result['units'].items()}}
        elif 'ready' in result:
            shown={k:result.get(k) for k in ('ready','reason','root','final_job_id','scheduler')}
        else:shown=result
        print(json.dumps(shown,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':
    main()

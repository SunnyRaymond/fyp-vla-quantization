"""Bounded PBS control files only; no workload executes on the login node."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ablation_clipping_control',HERE.parent/'fastwam-vq-activation-clipping/control.py')
source=importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
remote=source.remote
P=json.loads((HERE/'protocol.json').read_text())
ROOT=P['remote_root']
JOBS={'data_prepare':'fw80data','prepare':'fw80prep','train_eval':'fw80clip','aggregate':'fw80agg'}


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


def passed(observation):
    q=observation['scheduler'];s=observation.get('summary') or {}
    unit=observation['handle']['unit']
    coverage=(s.get('evaluation_queries')==320 and s.get('training_updates')==80 if unit=='aggregate'
              else s.get('evaluation_queries')==160 and s.get('updates')==(80 if unit=='train_eval' else 0)
              if unit in ('prepare','train_eval') else s.get('development_contexts')==40)
    return (q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='0'
            and observation['artifact_exit_code']==0 and observation['phase_complete_marker'] is True
            and s.get('complete') is True and s.get('pbs_jobid')==observation['handle']['job_id'] and coverage)


def status(transport,unit=None):
    with remote.sftp_client(transport) as sftp:
        handles=source.read_json(sftp,ROOT+'/control/handles.json') or {}
        rows={key:observe(transport,sftp,key,h) for key,h in handles.items() if unit is None or key==unit}
    return {'checked_utc':datetime.now(timezone.utc).isoformat(),'units':rows}


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
        dependency={'train_eval':'prepare','aggregate':'train_eval'}.get(unit)
        if dependency and (dependency not in handles or not passed(observe(transport,sftp,dependency,handles[dependency]))):
            raise RuntimeError('Completed prerequisite required: '+dependency)
        attempt=(old['attempt']+1) if old else 0
        intent=ROOT+'/control/'+unit+'_intent_'+str(attempt)+'.json'
        pending=source.read_json(sftp,intent)
        if pending:
            raise RuntimeError('An existing submission intent requires explicit reconciliation; do not duplicate')
        cpu=unit in ('data_prepare','aggregate')
        handle={'unit':unit,'attempt':attempt,'job_name':JOBS[unit],
                'created_utc':datetime.now(timezone.utc).isoformat(),'remote_root':ROOT,
                'status':'intent_before_qsub','gpus':0 if cpu else 1,
                'walltime':'00:10:00' if unit=='aggregate' else '01:59:00'}
        source.write_json(sftp,intent,handle)
        save(HERE/(unit+'_intent_'+str(attempt)+'.json'),handle)
        cmd=('qsub -N '+shlex.quote(JOBS[unit])+' -o '+shlex.quote(ROOT+'/artifacts/'+JOBS[unit]+'_'+str(attempt)+'.pbs.out')
             +' -v '+shlex.quote('FW_C80_UNIT='+unit)+' '+shlex.quote(ROOT+('/aggregate.pbs' if unit=='aggregate' else '/cpu.pbs' if unit=='data_prepare' else '/run.pbs')))
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
            result=source.status(t)
            if not source.passed(result['units']['aggregate']):raise RuntimeError('Source comparison incomplete')
            save(HERE/'source_evidence.json',result)
            result={'source_complete':True}
        elif args.action=='upload':
            rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
            if rc:raise RuntimeError(out)
            with remote.sftp_client(t) as sftp:
                existing=source.read_json(sftp,ROOT+'/protocol.json')
                if existing is not None and existing!=P:
                    cancelled=json.loads((HERE/'cancelled_development_protocol.json').read_text())
                    handles=source.read_json(sftp,ROOT+'/control/handles.json') or {}
                    if existing!=cancelled or any(h['gpus'] for h in handles.values()):
                        raise RuntimeError('Do not overwrite a launched GPU protocol')
                    if 'data_prepare' in handles:
                        old=observe(t,sftp,'data_prepare',handles['data_prepare'])
                        if old['scheduler']['query_rc']!=0 or old['scheduler']['job_state'] not in ('F','X'):
                            raise RuntimeError('Cancelled CPU preparation is not yet terminal')
                    source.write_json(sftp,ROOT+'/control/cancelled_development_protocol.json',existing)
                for name in ('worker.py','prepare_data.py','protocol.json','run.pbs','cpu.pbs','aggregate.pbs'):
                    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    if len(data)>remote.LIMIT:raise RuntimeError('Small control/code files only')
                    with sftp.file(ROOT+'/'+name+'.tmp','wb') as f:f.write(data)
                    sftp.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
            rc,out=remote.command(t,'bash -n '+shlex.quote(ROOT+'/run.pbs')+' && bash -n '+shlex.quote(ROOT+'/aggregate.pbs')+' && bash -n '+shlex.quote(ROOT+'/cpu.pbs'),timeout=20)
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
                    files=['comparison.json','train_eval/training_receipt.json']+[
                        split+'/'+arm+'/summary.json' for split in ('calibration','heldout') for arm in P['arms']]
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
                            check=source.read_json(sftp,prefix+'/clip_selfcheck.json')
                            save(HERE/'results/evidence'/o['handle']['job_id']/'clip_selfcheck.json',check)
                            with sftp.file(prefix+'/job.log','rb') as f:
                                head=f.read(4096);f.seek(max(0,sftp.stat(prefix+'/job.log').st_size-12288));tail=f.read(12288)
                            samples=list(dict.fromkeys(line for line in (head+b'\n'+tail).decode('utf-8',errors='replace').splitlines() if line.startswith('GPU_SAMPLE timestamp=')))
                            save(HERE/'results/evidence'/o['handle']['job_id']/'gpu_telemetry.json',{'source':prefix+'/job.log','read_scope':'first4096/last12288 bytes','samples':samples})
            elif args.action=='logs':
                if not args.unit:parser.error('--unit required')
                h=result['units'][args.unit]['handle']
                rc,out=remote.command(t,'tail -n 24 '+shlex.quote(ROOT+'/artifacts/'+h['job_id']+'/job.log')+' 2>/dev/null; tail -n 10 '+shlex.quote(ROOT+'/artifacts/'+h['job_name']+'_'+str(h['attempt'])+'.pbs.out')+' 2>/dev/null',timeout=25)
                result={'job_id':h['job_id'],'query_rc':rc,'tail':out[-11000:]}
        print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':
    main()

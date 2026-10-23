"""Only small control files, qsub/qstat and bounded log reads on login nodes."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('visited_parent',HERE.parent/'fastwam-vq-rollout-c80/control.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
remote,source=parent.remote,parent.source
P=json.loads((HERE/'protocol.json').read_text())
ROOT=P['remote_root']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def observe(t,f,handle):
    result={'handle':handle,'scheduler':source.previous.scheduler(t,handle['job_id']),
            'progress':source.read_json(f,ROOT+'/results/progress.json'),
            'summary':source.read_json(f,ROOT+'/results/summary.json'),
            'artifact_exit_code':None,'phase_complete_marker':False}
    prefix=ROOT+'/artifacts/'+handle['job_id']
    try:
        with f.file(prefix+'/exit_code.txt') as stream:result['artifact_exit_code']=int(stream.read(32))
    except OSError as exc:
        if exc.errno!=2:raise
    try:f.stat(prefix+'/PHASE_COMPLETE');result['phase_complete_marker']=True
    except OSError as exc:
        if exc.errno!=2:raise
    return result


def passed(row):
    q,s=row['scheduler'],row.get('summary') or {}
    return (q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='0'
            and row['artifact_exit_code']==0 and row['phase_complete_marker']
            and s.get('complete') is True and s.get('records_completed')==600
            and s.get('new_model_queries')==566 and s.get('training_updates')==0
            and s.get('pbs_jobid')==row['handle']['job_id'])


def validate_sources(t,f):
    cp=source.read_json(f,P['c80_root']+'/protocol.json')
    rp=source.read_json(f,P['visited_root']+'/protocol.json')
    prep=source.read_json(f,P['visited_root']+'/results/prepare/summary.json')
    if (cp['protocol']!=P['c80_protocol'] or rp['protocol']!=P['visited_protocol']
            or cp['source_c80_protocol']!=rp['source_c80_protocol']
            or rp['rollout']['visited_ids']!=P['case_ids']
            or not prep.get('complete') or prep.get('visited_contexts')!=40
            or prep.get('pbs_jobid')!=P['visited_prepare_job']):
        raise RuntimeError('Frozen C80 or existing visited source differs')
    jobs={'visited_prepare':(P['visited_prepare_job'],P['visited_root'])}
    receipts={}
    for arm in ('clip_only','joint'):
        receipt=source.read_json(f,P['c80_root']+'/results/training/'+arm+'/training_receipt.json')
        if not receipt.get('complete') or receipt.get('updates')!=80 or receipt.get('arm')!=arm:
            raise RuntimeError('Source final80 receipt missing')
        jobs[arm]=(receipt['pbs_jobid'],P['c80_root'])
        receipts[arm]={k:receipt[k] for k in ('arm','updates','complete','pbs_jobid','weights_unchanged')}
    evidence={'checked_utc':datetime.now(timezone.utc).isoformat(),'jobs':{},'receipts':receipts}
    for name,(job,root) in jobs.items():
        q=source.previous.scheduler(t,job)
        with f.file(root+'/artifacts/'+job+'/exit_code.txt') as stream:code=int(stream.read(32))
        scheduler_evidence=q
        if q['query_rc']!=0 and name in ('clip_only','joint'):
            # Source jobs can age out of PBS history; retain their captured terminal evidence.
            archived=json.loads((HERE.parent/'fastwam-vq-c80-clip-joint-plus/training_monitor_state.json').read_text())['units'][name]
            observation=archived['last_observation']
            summary=source.read_json(f,root+'/artifacts/'+job+'/calibration_summary.json')
            f.stat(root+'/artifacts/'+job+'/TRAINING_COMPLETE')
            if (archived['job_id']==job and observation.get('query_rc')==0
                    and observation.get('scheduler_state') in ('F','X')
                    and observation.get('exit_status')=='0' and observation.get('training_complete') is True
                    and summary.get('complete') is True and summary.get('pbs_jobid')==job):
                scheduler_evidence={'query_rc':0,'job_state':observation['scheduler_state'],
                                    'Exit_status':observation['exit_status'],'source':'archived_terminal_observation',
                                    'observed_utc':observation['observed_utc']}
        if scheduler_evidence['query_rc']!=0 or scheduler_evidence['job_state'] not in ('F','X') or scheduler_evidence['Exit_status']!='0' or code!=0:
            raise RuntimeError('Complete source job required: '+name)
        evidence['jobs'][name]={'current_scheduler_query':q,'scheduler':scheduler_evidence,'artifact_exit_code':code}
    source.write_json(f,ROOT+'/control/source_evidence.json',evidence)
    save(HERE/'source_evidence.json',evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','submit','status','pull','logs'))
    args=parser.parse_args()
    jump,t=remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action=='upload':
                rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
                if rc:raise RuntimeError(out)
                existing=source.read_json(f,ROOT+'/protocol.json')
                if existing and existing!=P:raise RuntimeError('Existing protocol is immutable')
                for name in ('protocol.json','worker.py','metrics.py','test_metrics.py','test_contract.py','run.pbs'):
                    content=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    if len(content)>remote.LIMIT:raise RuntimeError('Control file is too large')
                    with f.file(ROOT+'/'+name+'.tmp','wb') as stream:stream.write(content)
                    f.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
                result={'uploaded':True}
            else:
                handle=source.read_json(f,ROOT+'/control/handle.json')
                if args.action=='submit':
                    if source.read_json(f,ROOT+'/protocol.json')!=P:raise RuntimeError('Remote protocol differs')
                    if handle:result=handle
                    else:
                        validate_sources(t,f)
                        intent=ROOT+'/control/submission_intent.json'
                        if source.read_json(f,intent):raise RuntimeError('Prior qsub intent exists; reconcile before retry')
                        handle={'created_utc':datetime.now(timezone.utc).isoformat(),'job_name':'fwvdiag','walltime':'01:59:00','gpus':1,'status':'intent_before_qsub'}
                        source.write_json(f,intent,handle)
                        rc,out=remote.command(t,'qsub -o '+shlex.quote(ROOT+'/artifacts/submit.pbs.out')+' '+shlex.quote(ROOT+'/run.pbs'),timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()):raise RuntimeError('Unknown/rejected qsub outcome; do not duplicate: '+out[:400])
                        handle.update(job_id=out.strip(),status='confirmed')
                        source.write_json(f,ROOT+'/control/handle.json',handle)
                        source.write_json(f,intent,handle)
                        save(HERE/'handle.json',handle);result=handle
                else:
                    if not handle:raise RuntimeError('No submitted handle')
                    result={'checked_utc':datetime.now(timezone.utc).isoformat(),**observe(t,f,handle)}
                    save(HERE/'last_status.json',result)
                    if args.action=='pull':
                        for name in ('comparison.json','summary.json','sensor_selfcheck.json','input_perturbations.json',
                                     'diagnostic_details.json','postprocess_summary.json'):
                            remote_name=name.replace('.json','.compact.json') if name in ('comparison.json','input_perturbations.json') else name
                            value=source.read_json(f,ROOT+'/results/'+remote_name)
                            if value:save(HERE/'results'/name,value)
                        try:
                            with f.file(ROOT+'/results/RESULTS.zh.md') as stream:data=stream.read(remote.LIMIT+1)
                            if len(data)>remote.LIMIT:raise RuntimeError('Report exceeds small-file limit')
                            (HERE/'results/RESULTS.zh.md').write_bytes(data)
                        except OSError as exc:
                            if exc.errno!=2:raise
                    if args.action=='logs':
                        rc,out=remote.command(t,'tail -c 16384 '+shlex.quote(ROOT+'/artifacts/'+handle['job_id']+'/job.log'),timeout=25)
                        result={'job_id':handle['job_id'],'query_rc':rc,'tail':out[-14000:]}
            print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':main()

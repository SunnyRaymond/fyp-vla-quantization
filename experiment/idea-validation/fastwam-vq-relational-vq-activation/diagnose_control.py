"""Bounded SSH controls for a read-only gradient diagnostic, not a training job."""
from datetime import datetime, timezone
import json
import re
import shlex
import sys

import control as c


def main():
    action=sys.argv[1]
    if action not in ('submit','status','pull','logs'):raise ValueError(action)
    handle_path=c.HERE/'vq_diagnosis_handle.json'
    local=json.loads(handle_path.read_text()) if handle_path.exists() else None
    jump,t=c.remote.direct_connect()
    try:
        with c.remote.sftp_client(t) as f:
            remote_handle=c.ROOT+'/control/vq_diagnosis_handle.json'
            handle=c.source.read_json(f,remote_handle) or local
            if action=='submit':
                if handle:
                    if not handle.get('job_id'):raise RuntimeError('Reconcile prior intent; do not resubmit')
                    c.save(handle_path,handle);print(json.dumps(handle));return
                for name in ('diagnose_vq.py','diagnose_vq.pbs'):
                    data=(c.HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    assert len(data)<c.remote.LIMIT
                    with f.file(c.ROOT+'/'+name+'.tmp','wb') as stream:stream.write(data)
                    f.posix_rename(c.ROOT+'/'+name+'.tmp',c.ROOT+'/'+name)
                rc,text=c.remote.command(t,'bash -n '+shlex.quote(c.ROOT+'/diagnose_vq.pbs'),timeout=20)
                if rc:raise RuntimeError(text)
                handle={'status':'intent_before_qsub','created_utc':datetime.now(timezone.utc).isoformat(),
                        'gpus':1,'walltime':'01:59:00','purpose':'read-only C8 checkpoint/gradient diagnosis',
                        'training_updates':0,'optimizer_steps':0,'heldout_used':False}
                c.source.write_json(f,remote_handle,handle);c.save(handle_path,handle)
                command='qsub -N fwvqkldiag -o '+shlex.quote(c.ROOT+'/artifacts/vq_diagnosis.pbs.out')+' '+shlex.quote(c.ROOT+'/diagnose_vq.pbs')
                rc,text=c.remote.command(t,command,timeout=40)
                if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',text.strip()):
                    raise RuntimeError('Unknown qsub outcome; reconcile exact intent: '+text[:200])
                handle.update(status='confirmed',job_id=text.strip())
                c.source.write_json(f,remote_handle,handle);c.save(handle_path,handle)
                print(json.dumps(handle));return
            if not handle or not handle.get('job_id'):raise RuntimeError('No confirmed diagnostic handle')
            job=handle['job_id'];prefix=c.ROOT+'/artifacts/'+job
            if action=='logs':
                rc,text=c.remote.command(t,'if [ -f '+shlex.quote(prefix+'/job.log')+' ]; then tail -c 5000 '+shlex.quote(prefix+'/job.log')+'; fi',timeout=20)
                print(text);return
            scheduler=c.source.previous.scheduler(t,job)
            summary=c.source.read_json(f,c.ROOT+'/results/vq_diagnosis/summary.json')
            partial=None if summary else c.source.read_json(f,c.ROOT+'/results/vq_diagnosis/partial.json')
            exit_code,marker=None,False
            try:
                with f.file(prefix+'/exit_code.txt') as stream:exit_code=int(stream.read(32))
                f.stat(prefix+'/PHASE_COMPLETE');marker=True
            except OSError as error:
                if error.errno!=2:raise
            passed=bool(scheduler.get('query_rc')==0 and scheduler.get('job_state') in ('F','X')
                        and scheduler.get('Exit_status')=='0' and exit_code==0 and marker
                        and summary and summary.get('complete') and summary['pbs_jobid']==job
                        and summary['new_model_queries']==24 and summary['new_training_updates']==0
                        and summary['new_teacher_queries']==0 and summary['optimizer_step_calls']==0)
            state={'checked_utc':datetime.now(timezone.utc).isoformat(),'job_id':job,'scheduler':scheduler,
                   'exit_code':exit_code,'marker':marker,'passed':passed,
                   'progress':{k:len(v['cases']) for k,v in (summary or partial or {}).get('checkpoints',{}).items()}}
            c.save(c.HERE/'vq_diagnosis_status.json',state)
            if action=='pull':
                if summary:c.save(c.HERE/'results/vq_diagnosis/summary.json',summary)
                elif partial:c.save(c.HERE/'results/vq_diagnosis/partial.json',partial)
            print(json.dumps(state,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':main()

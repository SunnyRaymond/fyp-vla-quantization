"""Small-file control for the read-only CPU diagnosis; no training submission."""
import json
import re
import shlex
import sys
from datetime import datetime, timezone

import control as c


def main():
    action = sys.argv[1]
    if action not in ('submit','status','pull'):
        raise ValueError(action)
    path = c.HERE/'diagnosis_handle.json'
    handle = json.loads(path.read_text()) if path.exists() else None
    jump,t = c.remote.direct_connect()
    try:
        with c.remote.sftp_client(t) as f:
            if action == 'submit':
                if handle:
                    print(json.dumps(handle)); return
                for name in ('diagnose_c80.py','diagnose_c80.pbs'):
                    data = (c.HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    assert len(data) < c.remote.LIMIT
                    with f.file(c.ROOT+'/'+name,'wb') as stream: stream.write(data)
                rc,out = c.remote.command(t,'bash -n '+shlex.quote(c.ROOT+'/diagnose_c80.pbs'),timeout=20)
                if rc: raise RuntimeError(out)
                handle = {'status':'intent_before_qsub','created_utc':datetime.now(timezone.utc).isoformat(),'gpus':0,'walltime':'00:10:00'}
                c.save(path,handle)
                rc,out = c.remote.command(t,'qsub -N fwkldiag -o '+shlex.quote(c.ROOT+'/artifacts/c80_diagnosis.pbs.out')+' '+shlex.quote(c.ROOT+'/diagnose_c80.pbs'),timeout=40)
                if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()): raise RuntimeError('Reconcile uncertain submission: '+out[:400])
                handle.update(status='confirmed',job_id=out.strip()); c.save(path,handle)
                print(json.dumps(handle)); return
            job = handle['job_id']; scheduler = c.source.previous.scheduler(t,job)
            summary = c.source.read_json(f,c.ROOT+'/results/c80_diagnosis/summary.json')
            prefix = c.ROOT+'/artifacts/'+job
            exit_code,marker = None,False
            try:
                with f.file(prefix+'/exit_code.txt') as stream: exit_code = int(stream.read(32))
                f.stat(prefix+'/PHASE_COMPLETE'); marker = True
            except OSError as exc:
                if exc.errno != 2: raise
            passed = scheduler['job_state'] in ('F','X') and scheduler['Exit_status']=='0' and exit_code==0 and marker and bool(summary and summary.get('complete') and summary['pbs_jobid']==job)
            state = {'scheduler':scheduler,'exit_code':exit_code,'marker':marker,'passed':passed}
            c.save(c.HERE/'diagnosis_status.json',state)
            if action=='pull' and passed:
                c.save(c.HERE/'results/c80_diagnosis/summary.json',summary)
            if scheduler['job_state'] in ('F','X') and not passed:
                rc,tail = c.remote.command(t,'tail -c 5000 '+shlex.quote(prefix+'/job.log'),timeout=20)
                state['tail']=tail
            print(json.dumps(state,indent=2))
    finally:
        t.close()
        if jump: jump.close()


if __name__=='__main__': main()

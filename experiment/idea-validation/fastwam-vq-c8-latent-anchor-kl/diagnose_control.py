"""Bounded small-file control for a single CPU-only posthoc diagnosis."""
import argparse
from datetime import datetime,timezone
import json
import re
import shlex
import control as c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('submit','status','pull'))
    parser.add_argument('--revision',choices=('1','2'),default='1')
    args=parser.parse_args();action=args.action
    unit='diagnosis' if args.revision=='1' else 'diagnosis_v2'
    script='diagnose.pbs' if args.revision=='1' else 'diagnose_v2.pbs'
    jump,transport=c.remote.direct_connect()
    try:
        with c.remote.sftp_client(transport) as sftp:
            hp=c.ROOT+'/control/'+unit+'_handle.json'
            handle=c.read_json(sftp,hp)
            if action=='submit':
                if handle: print(json.dumps(handle));return
                intent=c.ROOT+'/control/'+unit+'_intent.json'
                if c.read_json(sftp,intent): raise RuntimeError('Reconcile previous intent before any qsub')
                for name in ('diagnose.py',script):
                    data=(c.HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    assert len(data)<c.remote.LIMIT
                    with sftp.file(c.ROOT+'/'+name,'wb') as f:f.write(data)
                rc,out=c.remote.command(transport,'bash -n '+shlex.quote(c.ROOT+'/'+script),timeout=15)
                if rc:raise RuntimeError(out)
                handle={'unit':unit,'job_name':'fw8akdiag','gpus':0,'new_model_forwards':0,'created_utc':datetime.now(timezone.utc).isoformat()}
                c.write_json(sftp,intent,handle)
                rc,out=c.remote.command(transport,'qsub -N fw8akdiag -o '+shlex.quote(c.ROOT+'/artifacts/'+unit+'.pbs.out')+' '+shlex.quote(c.ROOT+'/'+script),timeout=40)
                if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()):raise RuntimeError('Unknown qsub outcome: '+out[:300])
                handle['job_id']=out.strip();c.write_json(sftp,hp,handle);c.save(c.HERE/(unit+'_handle.json'),handle)
                print(json.dumps(handle));return
            if not handle:raise RuntimeError('Missing diagnosis handle')
            row=c.observe(transport,sftp,unit,handle)
            passed=row['passed'] and row['summary'].get('new_model_forwards')==0
            row['passed']=passed
            c.save(c.HERE/(unit+'_status.json'),row)
            if action=='pull' and row['summary']:
                c.save(c.HERE/'results'/unit/'summary.json',row['summary'])
            print(json.dumps({'scheduler':row['scheduler'],'passed':passed,'artifact_exit_code':row['artifact_exit_code'],'phase_complete_marker':row['phase_complete_marker'],'complete':row['summary'].get('complete')},indent=2))
            if row['scheduler']['job_state'] in ('F','X') and not passed:
                rc,out=c.remote.command(transport,'tail -c 5000 '+shlex.quote(c.ROOT+'/artifacts/'+handle['job_id']+'/job.log'),timeout=15)
                print(out)
    finally:
        transport.close()
        if jump:jump.close()


if __name__=='__main__':main()

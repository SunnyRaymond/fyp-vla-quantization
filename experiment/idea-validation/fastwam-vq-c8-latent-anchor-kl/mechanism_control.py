"""Bounded SSH/PBS control; original experiment handles and results stay intact."""
import argparse
from datetime import datetime,timezone
import json
import re
import shlex
import control as c

UNITS={'mechanism_preflight':('mechanism_preflight.pbs','fw8akmcheck',0),
       'mechanism':('mechanism.pbs','fw8akmech',1),
       'mechanism_initial_opt':('mechanism_initial_opt.pbs','fw8akmopt',0),
       'mechanism_analysis':('mechanism_analysis.pbs','fw8akmagg',0)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('upload','submit','retry','status','pull','logs'))
    parser.add_argument('--unit',choices=tuple(UNITS))
    args=parser.parse_args()
    config=json.loads((c.HERE/'mechanism_protocol.json').read_text())
    jump,transport=c.remote.direct_connect()
    try:
        with c.remote.sftp_client(transport) as sftp:
            handles=c.read_json(sftp,c.ROOT+'/control/mechanism_handles.json') or {}
            if args.action=='upload':
                old=c.read_json(sftp,c.ROOT+'/mechanism_protocol.json')
                if old and old!=config:raise ValueError('Mechanism protocol is frozen')
                if c.read_json(sftp,c.ROOT+'/protocol.json')!=c.P:raise ValueError('Parent protocol differs')
                analysis_only=args.unit=='mechanism_analysis'
                if analysis_only and handles.get('mechanism_analysis'):
                    analysis_q=c.source.c.source.previous.scheduler(transport,handles['mechanism_analysis']['job_id'])
                    gpu_q=c.source.c.source.previous.scheduler(transport,handles['mechanism']['job_id'])
                    if (analysis_q['query_rc']!=0 or gpu_q['query_rc']!=0
                            or not (analysis_q['job_state'] in ('F','X')
                                    or analysis_q['job_state']=='H' and gpu_q['job_state'] in ('Q','R'))):
                        raise ValueError('Analysis upload requires terminal analysis or a verified dependency hold while GPU is live')
                if not analysis_only:
                    for handle in handles.values():
                        q=c.source.c.source.previous.scheduler(transport,handle['job_id'])
                        if q['query_rc']!=0 or q['job_state'] not in ('F','X'):raise ValueError('Do not overwrite scripts of a live or unknown job')
                if analysis_only and handles.get('mechanism_initial_opt'):
                    opt_q=c.source.c.source.previous.scheduler(transport,handles['mechanism_initial_opt']['job_id'])
                    if opt_q['query_rc']!=0 or opt_q['job_state'] not in ('F','X'):
                        raise ValueError('Do not replace source of a live initial optimizer analysis')
                names=['mechanism_analysis.py','mechanism_analysis.pbs','mechanism_initial_opt.pbs'] if analysis_only else [
                    'mechanism_protocol.json','mechanism.pbs','mechanism_preflight.pbs','mechanism_math.py','mechanism_worker.py','mechanism_preflight.py']
                paths={name:c.HERE/name for name in names}
                if not analysis_only:paths['mechanism_frozen_reader.py']=c.HERE.parent/'fastwam-vq-control-kl/frozen_reader.py'
                for name,path in paths.items():
                    data=path.read_bytes().replace(b'\r\n',b'\n')
                    if len(data)>c.remote.LIMIT:raise ValueError('Small control uploads only')
                    with sftp.file(c.ROOT+'/'+name+'.tmp','wb') as f:f.write(data)
                    sftp.posix_rename(c.ROOT+'/'+name+'.tmp',c.ROOT+'/'+name)
                for script in [name for name in names if name.endswith('.pbs')]:
                    rc,out=c.remote.command(transport,'bash -n '+shlex.quote(c.ROOT+'/'+script),timeout=15)
                    if rc:raise RuntimeError(out)
                print(json.dumps({'uploaded':list(paths),'protocol':config['protocol']}));return
            if args.action in ('submit','retry'):
                if not args.unit:parser.error('--unit required')
                unit=args.unit;previous=handles.get(unit)
                if previous:
                    row=c.observe(transport,sftp,unit,previous)
                    if args.action=='submit':print(json.dumps({'existing':previous,'scheduler':row['scheduler']}));return
                    q=row['scheduler']
                    if q['query_rc']!=0 or q['job_state'] not in ('F','X') or row['passed']:
                        raise ValueError('Retry requires verified failed terminal state')
                    c.save(c.HERE/('mechanism_previous_'+previous['job_id']+'.json'),row)
                if c.read_json(sftp,c.ROOT+'/mechanism_protocol.json')!=config:raise ValueError('Upload frozen protocol first')
                dependencies=[]
                if unit=='mechanism':
                    parent=handles.get('mechanism_preflight')
                    if not parent:raise ValueError('CPU preflight required')
                    row=c.observe(transport,sftp,'mechanism_preflight',parent)
                    if not row['passed']:raise ValueError('CPU preflight must finish successfully before GPU submission')
                    dependencies=[parent['job_id']]
                elif unit=='mechanism_analysis':
                    parent=handles.get('mechanism')
                    if not parent:raise ValueError('Mechanism handle required')
                    row=c.observe(transport,sftp,'mechanism',parent);q=row['scheduler']
                    if q['query_rc']!=0 or (q['job_state'] in ('F','X') and not row['passed']):
                        raise ValueError('GPU dependency is failed or unknown')
                    dependencies=[parent['job_id']]
                elif unit=='mechanism_initial_opt':
                    panels=c.read_json(sftp,c.ROOT+'/results/mechanism/initial/panels.json') or []
                    if len(panels)!=8 or any(sorted(r['case_id'] for r in panel['cases'])!=sorted(config['case_ids']) for panel in panels):
                        raise ValueError('Initial preview requires all declared initial C8 panels')
                attempt=previous.get('attempt',1)+1 if previous else 1
                intent=c.ROOT+'/control/'+unit+'_attempt'+str(attempt)+'_intent.json'
                if c.read_json(sftp,intent):raise ValueError('Reconcile existing intent before qsub')
                script,name,gpus=UNITS[unit]
                handle={'unit':unit,'attempt':attempt,'gpus':gpus,'created_utc':datetime.now(timezone.utc).isoformat(),
                        'afterok_job_ids':dependencies,'protocol':config['protocol']}
                c.write_json(sftp,intent,handle)
                after=' -W '+shlex.quote('depend=afterok:'+':'.join(dependencies)) if dependencies else ''
                rc,out=c.remote.command(transport,'qsub -N '+name+after+' -o '+shlex.quote(c.ROOT+'/artifacts/'+unit+'_attempt'+str(attempt)+'.pbs.out')+' '+shlex.quote(c.ROOT+'/'+script),timeout=40)
                if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()):raise RuntimeError('Unknown qsub outcome; reconcile intent: '+out[:300])
                handle['job_id']=out.strip();handles[unit]=handle
                c.write_json(sftp,intent,handle);c.write_json(sftp,c.ROOT+'/control/mechanism_handles.json',handles)
                c.save(c.HERE/'mechanism_handles.json',handles);print(json.dumps(handle));return
            observed={unit:c.observe(transport,sftp,unit,handle) for unit,handle in handles.items() if not args.unit or unit==args.unit}
            snapshot={'checked_utc':datetime.now(timezone.utc).isoformat(),'units':observed}
            c.save(c.HERE/'mechanism_status.json',snapshot)
            if args.action=='logs':
                if not args.unit:parser.error('--unit required')
                rc,out=c.remote.command(transport,'tail -c 9000 '+shlex.quote(c.ROOT+'/artifacts/'+handles[args.unit]['job_id']+'/job.log'),timeout=20)
                print(json.dumps({'query_rc':rc,'tail':out}));return
            if args.action=='pull':
                for unit,row in observed.items():
                    if row['summary']:c.save(c.HERE/'results'/unit/'summary.json',row['summary'])
                if observed.get('mechanism'):
                    for point in config['points']:
                        for name in ('gradient_geometry.json','panels.json'):
                            value=c.read_json(sftp,c.ROOT+'/results/mechanism/'+point+'/'+name)
                            if value:c.save(c.HERE/'results/mechanism'/point/name,value)
                if observed.get('mechanism_analysis',{}).get('passed'):
                    for point in config['points']:
                        value=c.read_json(sftp,c.ROOT+'/results/mechanism_analysis/'+point+'_optimizer_geometry.json')
                        if value:c.save(c.HERE/'results/mechanism_analysis'/(point+'_optimizer_geometry.json'),value)
                    with sftp.file(c.ROOT+'/results/MECHANISM_ANALYSIS.zh.md','rb') as f:data=f.read(c.remote.LIMIT+1)
                    if len(data)>c.remote.LIMIT:raise ValueError('Bounded report retrieval exceeded')
                    (c.HERE/'results/MECHANISM_ANALYSIS.zh.md').write_bytes(data)
            print(json.dumps({'checked_utc':snapshot['checked_utc'],'units':{
                unit:{'scheduler':row['scheduler'],'passed':row['passed'],'artifact_exit_code':row['artifact_exit_code'],
                      'phase_complete_marker':row['phase_complete_marker'],'progress':row['progress'],
                      'summary_complete':row['summary'].get('complete')} for unit,row in observed.items()}},indent=2))
    finally:
        transport.close()
        if jump:jump.close()


if __name__=='__main__':main()

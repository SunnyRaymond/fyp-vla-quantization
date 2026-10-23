"""Only bounded control/JSON operations on login nodes; no workload there."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pc_visited_source',HERE.parent/'fastwam-vq-visited-diagnostic/control.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
remote,source=parent.remote,parent.source
P=json.loads((HERE/'protocol.json').read_text());ROOT=P['remote_root']
JOBS={'run':'fwpcvisit','aggregate':'fwpcvagg'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def observe(t,f,unit,h):
    prefix=ROOT+'/artifacts/'+h['job_id']
    result={'handle':h,'scheduler':source.previous.scheduler(t,h['job_id']),
        'summary':source.read_json(f,ROOT+'/results/'+('summary.json' if unit=='run' else 'aggregate_summary.json')),
        'progress':source.read_json(f,ROOT+'/results/progress.json') if unit=='run' else None,
        'artifact_exit_code':None,'phase_complete_marker':False}
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
    coverage=(s.get('records_completed')==200 and s.get('new_model_queries')==205
        and s.get('reused_endpoint_records')==600 and s.get('matched_input_metadata')==200
        and s.get('all_parameters_frozen') is True and all(v==0 for v in s.get('parities',{'missing':1}).values())
        if row['handle']['unit']=='run' else s.get('records')==800 and s.get('sensitivity_rows')==240)
    return(q['query_rc']==0 and q['job_state'] in ('F','X') and q['Exit_status']=='0'
        and row['artifact_exit_code']==0 and row['phase_complete_marker'] and s.get('complete') is True
        and s.get('pbs_jobid')==row['handle']['job_id'] and coverage)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','submit','status','pull','logs'))
    parser.add_argument('--unit',choices=tuple(JOBS));args=parser.parse_args()
    jump,t=remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action=='upload':
                rc,out=remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
                if rc:raise RuntimeError(out)
                old=source.read_json(f,ROOT+'/protocol.json')
                if old and old!=P:raise RuntimeError('Remote protocol is immutable')
                for name in ('worker.py','protocol.json','source_channel_protocol.json','run.pbs','aggregate.pbs','test_extra.py'):
                    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    if len(data)>remote.LIMIT:raise RuntimeError('Small code/control files only')
                    with f.file(ROOT+'/'+name+'.tmp','wb') as stream:stream.write(data)
                    f.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
                rc,out=remote.command(t,'bash -n '+shlex.quote(ROOT+'/run.pbs')+' && bash -n '+shlex.quote(ROOT+'/aggregate.pbs'),timeout=20)
                if rc:raise RuntimeError(out)
                result={'uploaded':True,'shell_syntax_passed':True}
            else:
                handles=source.read_json(f,ROOT+'/control/handles.json') or {}
                if args.action=='submit':
                    unit=args.unit
                    if not unit:parser.error('--unit required')
                    if source.read_json(f,ROOT+'/protocol.json')!=P:raise RuntimeError('Frozen protocol differs')
                    if unit in handles:result=handles[unit]
                    else:
                        if unit=='aggregate':
                            if 'run' not in handles or not passed(observe(t,f,'run',handles['run'])):raise RuntimeError('Complete GPU prerequisite required')
                        else:
                            old=source.read_json(f,P['source_diagnostic_root']+'/results/summary.json')
                            train=source.read_json(f,P['channel']['results']+'/training_receipt.json')
                            if not old['complete'] or old['records_completed']!=600 or old['pbs_jobid']!=P['source_diagnostic_job']:
                                raise RuntimeError('Completed source diagnostic required')
                            if not train['complete'] or train['updates']!=80 or train['pbs_jobid']!=P['channel']['training_job']:
                                raise RuntimeError('Completed per-channel final80 required')
                            save(HERE/'source_evidence.json',{'source_summary':old,'channel_training_receipt':train})
                        intent=ROOT+'/control/'+unit+'_intent.json'
                        if source.read_json(f,intent):raise RuntimeError('Existing submission intent: reconcile, never duplicate')
                        h={'unit':unit,'job_name':JOBS[unit],'gpus':1 if unit=='run' else 0,
                           'walltime':'01:59:00' if unit=='run' else '00:10:00',
                           'created_utc':datetime.now(timezone.utc).isoformat(),'status':'intent_before_qsub'}
                        source.write_json(f,intent,h);save(HERE/(unit+'_intent.json'),h)
                        script='/run.pbs' if unit=='run' else '/aggregate.pbs'
                        rc,out=remote.command(t,'qsub -o '+shlex.quote(ROOT+'/artifacts/'+unit+'.pbs.out')+' '+shlex.quote(ROOT+script),timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()):raise RuntimeError('Unknown/rejected qsub; reconcile intent: '+out[:400])
                        h.update(status='confirmed',job_id=out.strip());handles[unit]=h
                        source.write_json(f,intent,h);source.write_json(f,ROOT+'/control/handles.json',handles)
                        save(HERE/'handles.json',handles);result=h
                else:
                    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'units':{u:observe(t,f,u,h) for u,h in handles.items() if not args.unit or u==args.unit}}
                    save(HERE/'last_status.json',result)
                    if args.action=='pull':
                        if 'aggregate' in result['units'] and passed(result['units']['aggregate']):
                            def piece(name):
                                value=source.read_json(f,ROOT+'/results/'+name+'.compact.json')
                                if value is None:raise RuntimeError('Missing completed report piece: '+name)
                                return value
                            comparison=piece('comparison.index')
                            suites=comparison.pop('suite_names')
                            if suites!=['libero_spatial','libero_object','libero_goal','libero_10']:
                                raise RuntimeError('Report suite coverage differs')
                            comparison['overall']=piece('comparison.overall')
                            comparison['by_suite']={suite:piece('comparison.'+suite) for suite in suites}
                            details=piece('diagnostic_details.index')
                            details['base_cases']=[row for suite in suites for row in piece('diagnostic_details.'+suite)]
                            save(HERE/'results/comparison.json',comparison)
                            save(HERE/'results/diagnostic_details.json',details)
                        for name in ('summary','aggregate_summary','source_parity','input_reuse','sensor_selfcheck'):
                            value=source.read_json(f,ROOT+'/results/'+name+'.json')
                            if value:save(HERE/'results'/(name+'.json'),value)
                    if args.action=='logs':
                        if not args.unit:parser.error('--unit required')
                        h=handles[args.unit]
                        rc,out=remote.command(t,'tail -c 14000 '+shlex.quote(ROOT+'/artifacts/'+h['job_id']+'/job.log'),timeout=25)
                        result={'job_id':h['job_id'],'query_rc':rc,'tail':out}
                    else:
                        result={'checked_utc':result['checked_utc'],'units':{u:{'job_id':v['handle']['job_id'],'scheduler':v['scheduler'],'passed':passed(v),'phase':(v.get('progress') or {}).get('phase'),'new_endpoints':(v.get('progress') or {}).get('records_completed'),'queries':(v.get('progress') or {}).get('new_model_queries')} for u,v in result['units'].items()}}
            print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':main()

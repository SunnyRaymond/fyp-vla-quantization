"""Only bounded control/JSON operations on login nodes; no workload there."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pc_visited_source',HERE.parent/'fastwam-vq-per-channel-scaling-clipping/control.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
remote,source=parent.remote,parent.source
P=json.loads((HERE/'protocol.json').read_text());ROOT=P['remote_root']
JOBS={'train_eval':'fwpcgrip','aggregate':'fwpcgrag'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def observe(t,f,unit,h):
    prefix=ROOT+'/artifacts/'+h['job_id']
    result={'handle':h,'scheduler':source.previous.scheduler(t,h['job_id']),
        'summary':source.read_json(f,ROOT+'/results/'+('train_eval/summary.json' if unit=='train_eval' else 'aggregate_summary.json')),
        'progress':source.read_json(f,ROOT+'/results/train_eval/progress.json') if unit=='train_eval' else None,
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
    if row['handle']['unit']=='train_eval':
        coverage=(s.get('updates')==80 and s.get('evaluation_queries')==160 and s.get('visited_queries')==200
            and s.get('total_model_queries')==445 and s.get('new_parity_queries_this_job')==6
            and s.get('lambda_g')==1/6 and s.get('matched_visited_input_metadata')==200
            and all(s.get(k) is True for k in ('codebooks_unchanged','indices_unchanged','row_gains_unchanged','original_exponents_unchanged','continuous_scale_metadata_separate','gradient_coupling_complete','input_absmax_chain'))
            and all(s.get(k)==0 for k in ('initial_parity_max_abs','actual_infer_parity_max_abs','bf16_visited_parity_max_abs','visited_repeat_max_abs'))
            and all(v==0 for v in s.get('cached_query_parity',{'missing':1}).values()))
    else:coverage=(s.get('records')==1000 and s.get('sensitivity_rows')==320 and s.get('original_records')==640 and s.get('training_updates')==80)
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
                for name in ('worker.py','analysis.py','protocol.json','source_per_channel_protocol.json','run.pbs','aggregate.pbs','channel_banks.py','channel_selfcheck.py','gripper_loss.py','gripper_selfcheck.py'):
                    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    if len(data)>remote.LIMIT:raise RuntimeError('Small code/control files only')
                    with f.file(ROOT+'/'+name+'.tmp','wb') as stream:stream.write(data)
                    f.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
                rc,out=remote.command(t,'bash -n '+shlex.quote(ROOT+'/run.pbs')+' && bash -n '+shlex.quote(ROOT+'/aggregate.pbs'),timeout=20)
                if rc:raise RuntimeError(out)
                result={'uploaded':True,'shell_syntax_passed':True,'gpu_budget':'one GPU job, walltime01:59, no automatic resubmit'}
            else:
                handles=source.read_json(f,ROOT+'/control/handles.json') or {}
                if args.action=='submit':
                    unit=args.unit
                    if not unit:parser.error('--unit required')
                    if source.read_json(f,ROOT+'/protocol.json')!=P:raise RuntimeError('Frozen protocol differs')
                    if unit in handles:result=handles[unit]
                    else:
                        if unit=='aggregate':
                            if 'train_eval' not in handles or not passed(observe(t,f,'train_eval',handles['train_eval'])):raise RuntimeError('Complete GPU prerequisite required')
                        else:
                            verified=parent.status(t)
                            if not all(parent.passed(verified['units'][key]) for key in ('train_eval','aggregate')):
                                raise RuntimeError('Completed per-channel training/comparison required')
                            if verified['units']['train_eval']['handle']['job_id']!=P['source_per_channel_job']:
                                raise RuntimeError('Unexpected per-channel source job')
                            vr=P['visited_source_root']
                            previous=source.read_json(f,vr+'/results/summary.json')
                            aggregated=source.read_json(f,vr+'/results/aggregate_summary.json')
                            if not previous['complete'] or previous['records_completed']!=200 or previous['pbs_jobid']!=P['visited_source_gpu_job'] or not aggregated['complete'] or aggregated['records']!=800 or aggregated['pbs_jobid']!=P['visited_source_aggregate_job']:
                                raise RuntimeError('Complete existing visited panel required')
                            save(HERE/'source_evidence.json',{'per_channel':verified,'visited_gpu':previous,'visited_aggregate':aggregated})
                        intent=ROOT+'/control/'+unit+'_intent.json'
                        if source.read_json(f,intent):raise RuntimeError('Existing submission intent: reconcile, never duplicate')
                        h={'unit':unit,'job_name':JOBS[unit],'gpus':1 if unit=='train_eval' else 0,
                           'walltime':'01:59:00' if unit=='train_eval' else '00:10:00',
                           'created_utc':datetime.now(timezone.utc).isoformat(),'status':'intent_before_qsub'}
                        source.write_json(f,intent,h);save(HERE/(unit+'_intent.json'),h)
                        script='/run.pbs' if unit=='train_eval' else '/aggregate.pbs'
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
                            for split in ('calibration','heldout'):
                                save(HERE/'results'/(split+'_details.json'),[row for suite in suites for row in piece(split+'_details.'+suite)])
                        for name in ('train_eval/summary','train_eval/training_receipt','aggregate_summary','input_reuse'):
                            value=source.read_json(f,ROOT+'/results/'+name+'.json')
                            if value:save(HERE/'results'/(name+'.json'),value)
                    if args.action=='logs':
                        if not args.unit:parser.error('--unit required')
                        h=handles[args.unit]
                        rc,out=remote.command(t,'tail -c 14000 '+shlex.quote(ROOT+'/artifacts/'+h['job_id']+'/job.log'),timeout=25)
                        result={'job_id':h['job_id'],'query_rc':rc,'tail':out}
                    else:
                        result={'checked_utc':result['checked_utc'],'units':{u:{'job_id':v['handle']['job_id'],'scheduler':v['scheduler'],'passed':passed(v),'phase':(v.get('progress') or {}).get('phase'),'updates':(v.get('progress') or {}).get('updates'),'queries':(v.get('progress') or {}).get('queries'),'visited_completed':(v.get('progress') or {}).get('visited_completed'),'total_model_queries':(v.get('progress') or {}).get('total_model_queries')} for u,v in result['units'].items()}}
            print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump:jump.close()


if __name__=='__main__':main()

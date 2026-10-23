"""Bounded PBS control only; all data and numerical operations run in allocations."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dual_kl_source_control',HERE.parent/'fastwam-vq-activation-clipping/control.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
remote = source.remote
P = json.loads((HERE/'protocol.json').read_text())
ROOT = P['remote_root']
JOBS = {'fit_stats':'fwklstat','prepare':'fwklprep','action_kl':'fwklact','latent_kl':'fwklz','dual_kl':'fwklboth','aggregate':'fwklagg'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def observe(t,f,unit,h):
    prefix = ROOT+'/artifacts/'+h['job_id']
    row = {'handle':h,'scheduler':source.previous.scheduler(t,h['job_id']),
           'summary':source.read_json(f,ROOT+'/results/'+unit+'/summary.json'),
           'progress':source.read_json(f,ROOT+'/results/'+unit+'/progress.json'),
           'artifact_exit_code':None,'phase_complete_marker':False}
    try:
        with f.file(prefix+'/exit_code.txt') as s: row['artifact_exit_code'] = int(s.read(32))
    except OSError as exc:
        if exc.errno != 2: raise
    try: f.stat(prefix+'/PHASE_COMPLETE'); row['phase_complete_marker'] = True
    except OSError as exc:
        if exc.errno != 2: raise
    return row


def passed(row):
    q,s,u = row['scheduler'],row.get('summary') or {},row['handle']['unit']
    coverage = True
    if u in P['new_training_arms']:
        coverage = s.get('updates') == 80 and s.get('evaluation_queries') == 160 and s.get('new_teacher_queries') == 0
    elif u == 'prepare':
        coverage = s.get('initial_calibration_queries') == 80 and s.get('control_evaluation_queries') == 160
    elif u == 'fit_stats':
        coverage = s.get('calibration_case_ids') == P['calibration']['case_ids'] and s.get('heldout_used') is False
    elif u == 'aggregate':
        coverage = s.get('heldout_records') == 320 and s.get('calibration_records') == 320 and s.get('training_updates_per_arm') == 80
    return (q['query_rc'] == 0 and q['job_state'] in ('F','X') and q['Exit_status'] == '0'
            and row['artifact_exit_code'] == 0 and row['phase_complete_marker']
            and s.get('complete') is True and s.get('pbs_jobid') == row['handle']['job_id'] and coverage)


def training_coefficients_ready(f,handles):
    h=handles.get('prepare')
    value=source.read_json(f,ROOT+'/results/prepare/loss_scaling.json')
    if not h or not value:return False
    rows=value.get('initial_rows',[])
    means=value.get('initial_means',{})
    return (value.get('protocol')==P['protocol'] and value.get('complete') is True
        and value.get('pbs_jobid')==h['job_id'] and value.get('heldout_used') is False
        and value.get('case_ids')==P['calibration']['case_ids'] and value.get('seed_indices')==[0]
        and [r.get('case_id') for r in rows]==P['calibration']['case_ids']
        and all(r.get('seed_index')==0 for r in rows)
        and all(isinstance(value.get(k),(int,float)) and math.isfinite(value[k]) and value[k]>0
                for k in ('action_kl_coefficient','latent_kl_coefficient'))
        and all(isinstance(means.get(k),(int,float)) and math.isfinite(means[k]) and means[k]>0
                for k in ('action_mse','latent_nmse','action_kl','latent_kl')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('upload','submit','status','logs','pull'))
    parser.add_argument('--unit',choices=tuple(JOBS))
    args = parser.parse_args()
    jump,t = remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action == 'upload':
                rc,out = remote.command(t,'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'),timeout=20)
                if rc: raise RuntimeError(out)
                old = source.read_json(f,ROOT+'/protocol.json')
                if old and old != P: raise RuntimeError('Remote protocol frozen')
                names = ['protocol.json','source_protocol.json','fit_stats.py','cpu.pbs','gpu.pbs','worker.py','analysis.py','softbin_loss.py','softbin_selfcheck.py','channel_banks.py','channel_selfcheck.py']
                uploaded = []
                for name in names:
                    if not (HERE/name).is_file(): continue
                    data = (HERE/name).read_bytes().replace(b'\r\n',b'\n')
                    if len(data) > remote.LIMIT: raise RuntimeError('Small code/control files only')
                    with f.file(ROOT+'/'+name+'.tmp','wb') as s: s.write(data)
                    f.posix_rename(ROOT+'/'+name+'.tmp',ROOT+'/'+name)
                    uploaded.append(name)
                rc,out = remote.command(t,'bash -n '+shlex.quote(ROOT+'/cpu.pbs')+' && bash -n '+shlex.quote(ROOT+'/gpu.pbs'),timeout=20)
                if rc: raise RuntimeError(out)
                result = {'uploaded':uploaded,'shell_syntax_passed':True}
            else:
                handles = source.read_json(f,ROOT+'/control/handles.json') or {}
                if args.action == 'submit':
                    unit = args.unit
                    if not unit: parser.error('--unit required')
                    if source.read_json(f,ROOT+'/protocol.json') != P: raise RuntimeError('Protocol differs')
                    if unit in handles:
                        result = handles[unit]
                    else:
                        deps = ([] if unit in ('fit_stats','aggregate') else ['fit_stats'])
                        for dep in deps:
                            if dep not in handles or not passed(observe(t,f,dep,handles[dep])):
                                raise RuntimeError('Completed prerequisite required: '+dep)
                        if unit in P['new_training_arms']:
                            if not training_coefficients_ready(f,handles):
                                raise RuntimeError('Frozen complete C80 loss coefficients required; control replay may continue concurrently')
                            prep=observe(t,f,'prepare',handles['prepare'])
                            q=prep['scheduler'];progress=prep.get('progress') or {}
                            if q['query_rc']!=0 or q['job_state'] not in ('R','F','X') or (q['job_state'] in ('F','X') and q['Exit_status']!='0') or progress.get('phase') in ('failed','checkpointed'):
                                raise RuntimeError('Preparation must be live or successfully complete')
                        afterok=[]
                        if unit=='aggregate':
                            for dep in ('prepare',*P['new_training_arms']):
                                if dep not in handles or handles[dep].get('status')!='confirmed':
                                    raise RuntimeError('Confirmed GPU prerequisite handle required: '+dep)
                                afterok.append(handles[dep]['job_id'])
                        if unit == 'fit_stats':
                            old = source.read_json(f,P['source_per_channel_root']+'/results/train_eval/summary.json')
                            if not old or not old['complete'] or old['updates'] != 80 or old['evaluation_queries'] != 160:
                                raise RuntimeError('Completed original MSE/NMSE control required')
                        intent = ROOT+'/control/'+unit+'_intent.json'
                        if source.read_json(f,intent): raise RuntimeError('Reconcile existing submission intent; never duplicate')
                        cpu = unit in ('fit_stats','aggregate')
                        h = {'unit':unit,'job_name':JOBS[unit],'status':'intent_before_qsub','gpus':0 if cpu else 1,
                             'walltime':'00:20:00' if cpu else '01:59:00','created_utc':datetime.now(timezone.utc).isoformat()}
                        if afterok:h['afterok_job_ids']=afterok
                        source.write_json(f,intent,h); save(HERE/(unit+'_intent.json'),h)
                        script = '/cpu.pbs' if cpu else '/gpu.pbs'
                        dependency=' -W '+shlex.quote('depend=afterok:'+':'.join(afterok)) if afterok else ''
                        rc,out = remote.command(t,'qsub -N '+JOBS[unit]+dependency+' -v '+shlex.quote('FW_KL_UNIT='+unit)+' -o '+shlex.quote(ROOT+'/artifacts/'+unit+'.pbs.out')+' '+shlex.quote(ROOT+script),timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',out.strip()): raise RuntimeError('Unknown qsub outcome; reconcile intent: '+out[:400])
                        h.update(status='confirmed',job_id=out.strip()); handles[unit] = h
                        source.write_json(f,intent,h); source.write_json(f,ROOT+'/control/handles.json',handles)
                        save(HERE/'handles.json',handles); result = h
                else:
                    rows = {u:observe(t,f,u,h) for u,h in handles.items() if not args.unit or u == args.unit}
                    snapshot = {'checked_utc':datetime.now(timezone.utc).isoformat(),'units':rows}
                    save(HERE/'last_status.json',snapshot)
                    if args.action == 'logs':
                        if not args.unit: parser.error('--unit required')
                        h = handles[args.unit]
                        rc,out = remote.command(t,'tail -c 12000 '+shlex.quote(ROOT+'/artifacts/'+h['job_id']+'/job.log'),timeout=25)
                        result = {'job_id':h['job_id'],'query_rc':rc,'tail':out}
                    else:
                        if args.action == 'pull':
                            for u,row in rows.items():
                                if row['summary']: save(HERE/'results'/u/'summary.json',row['summary'])
                                if row['handle']['gpus']:
                                    for name in ('softbin_selfcheck','channel_selfcheck'):
                                        value=source.read_json(f,ROOT+'/artifacts/'+row['handle']['job_id']+'/'+name+'.json')
                                        if value:save(HERE/'results'/u/(name+'.json'),value)
                            for name in ('fit_stats/summary','prepare/loss_scaling','comparison','paired_details','decision'):
                                value = source.read_json(f,ROOT+'/results/'+name+'.json')
                                if value: save(HERE/'results'/(name+'.json'),value)
                            for split in ('calibration','heldout'):
                                for arm in P['arms']:
                                    value=source.read_json(f,ROOT+'/results/'+split+'/'+arm+'/summary.json')
                                    if value:save(HERE/'results'/split/arm/'summary.json',value)
                            for name in ('ANALYSIS.zh.md',):
                                try:
                                    with f.file(ROOT+'/results/'+name,'rb') as s: data=s.read(remote.LIMIT+1)
                                except OSError as exc:
                                    if exc.errno == 2: continue
                                    raise
                                if len(data)>remote.LIMIT: raise RuntimeError('Report exceeds bounded retrieval')
                                path=HERE/'results'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                        result = {'checked_utc':snapshot['checked_utc'],'training_coefficients_ready':training_coefficients_ready(f,handles),'units':{u:{'job_id':r['handle']['job_id'],'scheduler':r['scheduler'],'passed':passed(r),
                            'phase':(r.get('progress') or {}).get('phase'),'updates':(r.get('progress') or {}).get('updates'),
                            'queries':(r.get('progress') or {}).get('queries'),'split':(r.get('progress') or {}).get('split')} for u,r in rows.items()}}
            print(json.dumps(result,indent=2))
    finally:
        t.close()
        if jump: jump.close()


if __name__ == '__main__': main()

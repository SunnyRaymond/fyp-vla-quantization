"""Constant-rho trajectories and finite-horizon outer comparisons on PBS."""
import argparse
import json
import math
import os
from pathlib import Path

from core import METRICS,aggregate
from worker import Engine,NUMERICS,guard,same,save,training_info

def validate_config(p,s):
    if not (s['parent_protocol']==p['protocol'] and s['numerics']==NUMERICS
            and s['unit_rhos']=={'stage2_rho1':1.0,'stage2_rho225':2.25}
            and s['updates']==16 and s['horizons']==[1,8,16]
            and s['outer_case_ids']==p['development']['outer_case_ids'] and s['outer_seed_indices']==[0]
            and s['source_initial_checkpoint']==p['remote_root']+'/results/initial_state.pt'
            and s['reference_arm']=='current' and s['reference_rho']==1.5
            and s['stage3_gate']=={'minimum_primary_improvement_percent':1.0,'maximum_secondary_regression_percent':5.0}):
        raise RuntimeError('Stage2 differs from the authorized paired protocol')

def validate_training(state,p,s,unit):
    if (state['stage2_protocol']!=s['protocol'] or state['unit']!=unit
            or state['rho']!=s['unit_rhos'][unit] or not state['initial_state_exact']
            or state['source_initial_checkpoint']!=s['source_initial_checkpoint']):
        raise RuntimeError('Stage2 checkpoint binding differs')
    h=state['history'];n=state['updates']
    if (not 0<=n<=16 or len(h)!=n or [r['update'] for r in h]!=list(range(1,n+1))
            or [r['case_id'] for r in h]!=(p['calibration']['case_ids']*2)[:n]
            or any(r['rho']!=state['rho'] or r['coverage']['connected']!=614 or r['coverage']['finite']!=614 for r in h)
            or [r['update'] for r in h if r['optimizer_reset_this_update']]!=([1] if n else [])):
        raise RuntimeError('Stage2 committed training history differs')

def evaluate_horizon(e,s,state,horizon):
    checkpoint=e.out/f'update_{horizon}.pt';before=e.snapshot(horizon)
    e.checkpoint(checkpoint,{'snapshot':before,'rho':state['rho'],'stage2_protocol':s['protocol'],
                             'source_initial_checkpoint':s['source_initial_checkpoint']})
    records=[]
    try:
        for case in s['outer_case_ids']:
            plan=e.rows[case];path=e.out/'queries'/str(horizon)/f'{case}_0.json'
            binding={'protocol':e.p['protocol'],'stage2_protocol':s['protocol'],'unit':e.unit,'rho':state['rho'],
                     'updates':horizon,'case_id':case,'seed_index':0,'sampler_seed':plan['sampler_seeds'][0],
                     'suite':plan['suite'],'plan_row':plan,'checkpoint':str(checkpoint),
                     'source_initial_checkpoint':s['source_initial_checkpoint']}
            if path.exists():
                row=json.loads(path.read_text())
                if any(row.get(k)!=v for k,v in binding.items()):raise RuntimeError('Horizon query binding differs')
            else:
                action,future=e.query(case,0)
                row={**binding,**e.metrics(action,future,e.teacher(case,0)),'pbs_jobid':os.environ['PBS_JOBID']}
                save(path,row)
            records.append(row)
    finally:
        e.restore(before)
        if not same(e.torch,e.snapshot(horizon),before):raise RuntimeError('Horizon evaluation failed to restore training state')
    result={'protocol':e.p['protocol'],'stage2_protocol':s['protocol'],'unit':e.unit,'updates':horizon,
            'rho':state['rho'],'checkpoint':str(checkpoint),'evaluation_restore_exact':True,
            **aggregate(records,s['outer_case_ids'],[0])}
    if horizon==1:
        reference=json.loads((e.root/'results/preflight/evidence.json').read_text())['initial']
        side='minus' if state['rho']==1.0 else 'plus'
        observed=aggregate([r for r in records if r['case_id'] in reference['batch']],reference['batch'],[0])['overall']['first10motor_rmse']
        result['initial_preflight_replay_difference']=abs(observed-reference[side+'_rmse'])
        if result['initial_preflight_replay_difference']>s['h1_preflight_replay_tolerance']:
            raise RuntimeError('First update differs from the existing deterministic preflight')
    save(e.out/f'outer_update_{horizon}.json',result)
    return result

def train(e,s):
    p=e.p;unit=e.unit;checkpoint=e.out/'checkpoint.pt'
    e.install()
    if checkpoint.exists():
        payload=e.torch.load(checkpoint,map_location='cpu',weights_only=True)
        state=payload['training_state'];validate_training(state,p,s,unit);e.restore(payload['snapshot'])
    else:
        initial=e.torch.load(s['source_initial_checkpoint'],map_location='cpu',weights_only=True)
        if initial['updates']!=0:raise RuntimeError('Common initial state must be update0')
        e.restore(initial);exact=same(e.torch,e.snapshot(0),initial)
        if not exact:raise RuntimeError('Initial parameters/Adam/RNG did not restore exactly')
        state={'stage2_protocol':s['protocol'],'unit':unit,'rho':s['unit_rhos'][unit],'updates':0,
               'source_initial_checkpoint':s['source_initial_checkpoint'],'initial_state_exact':exact,
               'history':[],'evaluations':{}}
    def persist():
        validate_training(state,p,s,unit)
        e.checkpoint(checkpoint,{'snapshot':e.snapshot(state['updates']),'training_state':state})
        save(e.out/'training_state.json',state)
    persist()
    # On resume, finish a pending milestone before committing any further update.
    while True:
        n=state['updates']
        if n in s['horizons'] and str(n) not in state['evaluations']:
            state['evaluations'][str(n)]=evaluate_horizon(e,s,state,n);persist()
            e.progress(phase='horizon_complete',updates=n,rho=state['rho'],horizons_complete=sorted(state['evaluations'],key=int))
        if n==16:break
        case=p['calibration']['case_ids'][n%8]
        state['history'].append(e.train_step(case,state['rho'],n,'stage2_committed'))
        state['updates']+=1;persist()
    if e.teacher_new:raise RuntimeError('Stage2 may only reuse teachers')
    e.complete(stage2_protocol=s['protocol'],updates=16,rho=state['rho'],new_training_updates=16,
               queries=24,horizon_queries={str(h):state['evaluations'][str(h)]['queries'] for h in s['horizons']},
               source_initial_checkpoint=s['source_initial_checkpoint'],initial_state_exact=state['initial_state_exact'],
               evaluation_restore_exact=all(v['evaluation_restore_exact'] for v in state['evaluations'].values()),
               new_teacher_queries=0,checkpoint=str(e.out/'update_16.pt'),source_format='bilevel',
               evaluations=state['evaluations'])

def select_candidates(final,baseline,s):
    decisions={}
    for unit,metrics in final.items():
        changes={m:100*(metrics[m]/baseline[m]-1) if baseline[m]>0 else (0 if metrics[m]==0 else None) for m in METRICS}
        gate=s['stage3_gate'];eligible=(changes['first10motor_rmse'] is not None
            and changes['first10motor_rmse']<=-gate['minimum_primary_improvement_percent']
            and all(changes[m] is not None and changes[m]<=gate['maximum_secondary_regression_percent'] for m in METRICS if m!='first10motor_rmse'))
        decisions[unit]={'rho':s['unit_rhos'][unit],'percent_change_vs_rho1_5':changes,'eligible':eligible}
    eligible=[u for u in final if decisions[u]['eligible']]
    chosen=min(eligible,key=lambda u:(final[u]['first10motor_rmse'],s['unit_rhos'][u])) if eligible else None
    return decisions,chosen

def aggregate_stage2(p,s):
    root=Path(p['remote_root']);out=root/'results/stage2_compare';out.mkdir(parents=True,exist_ok=True)
    plan={r['case_id']:r for r in json.loads((root/'data/plan.json').read_text())['inputs']}
    summaries={};receipts={};per_case={}
    for unit,rho in s['unit_rhos'].items():
        folder=root/'results'/unit;receipt=json.loads((folder/'summary.json').read_text())
        state=json.loads((folder/'training_state.json').read_text());validate_training(state,p,s,unit)
        if not (receipt.get('complete') and receipt.get('stage2_protocol')==s['protocol'] and receipt.get('updates')==16
                and receipt.get('initial_state_exact') and receipt.get('evaluation_restore_exact') and receipt.get('new_teacher_queries')==0):
            raise RuntimeError('Candidate training or numerical gates incomplete')
        receipts[unit]={k:receipt[k] for k in ('pbs_jobid','rho','updates','checkpoint','source_initial_checkpoint','initial_state_exact','evaluation_restore_exact')}
        summaries[unit]={};per_case[unit]={}
        for h in s['horizons']:
            rows=[]
            for case in s['outer_case_ids']:
                r=json.loads((folder/'queries'/str(h)/f'{case}_0.json').read_text())
                if (r['unit']!=unit or r['rho']!=rho or r['updates']!=h or r['plan_row']!=plan[case]
                        or r['checkpoint']!=str(folder/f'update_{h}.pt') or r['stage2_protocol']!=s['protocol']
                        or r['sampler_seed']!=plan[case]['sampler_seeds'][0]):raise RuntimeError('Candidate endpoint binding differs')
                rows.append(r)
            summaries[unit][str(h)]=aggregate(rows,s['outer_case_ids'],[0])
            per_case[unit][str(h)]=[{k:r[k] for k in ('case_id','suite','seed_index',*METRICS)} for r in rows]
    info=training_info(root,'current');baseline_rows=[]
    for case in s['outer_case_ids']:
        r=json.loads((root/'results/outer_evaluation/queries/current/outer'/f'{case}_0.json').read_text())
        if (r['rho']!=1.5 or r['updates']!=16 or r['checkpoint']!=info['checkpoint']
                or r['plan_row']!=plan[case] or r['seed_index']!=0):raise RuntimeError('Cached rho1.5 endpoint differs')
        baseline_rows.append(r)
    baseline=aggregate(baseline_rows,s['outer_case_ids'],[0]);directions=[]
    for h in s['horizons']:
        low=summaries['stage2_rho1'][str(h)]['overall']['first10motor_rmse'];high=summaries['stage2_rho225'][str(h)]['overall']['first10motor_rmse']
        contrast=abs(high-low)/max((high+low)/2,1e-12)
        directions.append({'updates':h,'rho1_rmse':low,'rho225_rmse':high,
                           'gradient_log_rho':(high-low)/(math.log(2.25)-math.log(1.0)),
                           'relative_contrast':contrast,'preferred_rho':1.0 if low<high else 2.25 if high<low else None,
                           'direction_above_deadband':contrast>s['relative_direction_deadband']})
    final={u:summaries[u]['16']['overall'] for u in s['unit_rhos']};decisions,chosen=select_candidates(final,baseline['overall'],s)
    result={'protocol':p['protocol'],'stage2_protocol':s['protocol'],'unit':'stage2_compare','pbs_jobid':os.environ['PBS_JOBID'],
            'numerics':NUMERICS,'complete':True,'candidate_training_updates':32,'candidate_queries':48,'cached_reference_queries':8,
            'horizons':[1,8,16],'summaries':summaries,'per_case':per_case,'reference_final16':baseline,'training_receipts':receipts,
            'outer_directions':directions,'eligibility':decisions,'selected_unit':chosen,
            'selected_rho':s['unit_rhos'][chosen] if chosen else 1.5,'stage3_eligible':chosen is not None,
            'selection_frozen_before_new_confirmation':True,'new_teacher_queries':0,
            'reference_scope':'rho1.5 cached final16 only; no reference values inferred at1/8',
            'evidence':'single C8 training trajectory; fixed-input decoded BF16/fake A4; no new confirmation inspected in stage2'}
    save(out/'comparison.json',result);save(out/'summary.json',result)
    lines=['# Stage2 完整训练长度的 rho 对比','',
           '共同初始状态；两个候选各16步，在1/8/16步评估完整outer8×seed0。固定rho1.5仅复用final16对照。','',
           '| Updates | rho1.0 motor RMSE | rho2.25 motor RMSE | log rho割线方向 | Preferred rho |',
           '|---|---:|---:|---:|---:|']
    for r in directions:lines.append(f"| {r['updates']} | {r['rho1_rmse']:.9f} | {r['rho225_rmse']:.9f} | {r['gradient_log_rho']:+.9f} | {r['preferred_rho']} |")
    lines+=['',f"固定rho1.5 final16 motor RMSE={baseline['overall']['first10motor_rmse']:.9f}。",'',
            '| Candidate | first10 motor变化 | full32 motor变化 | late22 motor变化 | gripper变化 | future NMSE变化 | Stage3 eligible |',
            '|---|---:|---:|---:|---:|---:|---|']
    for unit,r in decisions.items():lines.append('| '+str(r['rho'])+' | '+' | '.join(f"{r['percent_change_vs_rho1_5'][m]:+.2f}%" for m in METRICS)+' | '+str(r['eligible'])+' |')
    lines+=['',f"冻结选择rho={result['selected_rho']}；stage3_eligible={result['stage3_eligible']}。",'',
            '只有final16主指标至少改善1%且其他四项退化均不超过5%才可进入stage3；排序变化本身不替代该门槛。逐case、各horizon五项指标见comparison.json。']
    (out/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
    save(out/'progress.json',{'protocol':p['protocol'],'unit':'stage2_compare','pbs_jobid':os.environ['PBS_JOBID'],'complete':True,'phase':'complete'})
    print('STAGE2_COMPLETE '+json.dumps(result),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--unit',required=True);a.add_argument('--protocol',type=Path,required=True);a.add_argument('--stage2-protocol',type=Path,required=True);args=a.parse_args()
    guard();p=json.loads(args.protocol.read_text());s=json.loads(args.stage2_protocol.read_text());validate_config(p,s);e=None
    try:
        if args.unit=='stage2_compare':aggregate_stage2(p,s);return 0
        if args.unit not in s['unit_rhos']:raise RuntimeError('Unknown stage2 unit')
        gate=json.loads((Path(p['remote_root'])/'results/preflight/summary.json').read_text())
        if not gate.get('gate_passed') or gate.get('numerics')!=NUMERICS:raise RuntimeError('Matching deterministic preflight required')
        e=Engine(p,args.unit);train(e,s);return 0
    except TimeoutError as error:
        if e:e.progress(phase='checkpointed',reason=str(error))
        return 75
    except Exception as error:
        if e:e.progress(phase='failed',error=type(error).__name__+': '+str(error))
        raise

if __name__=='__main__':raise SystemExit(main())

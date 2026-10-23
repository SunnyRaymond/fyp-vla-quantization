"""Evaluate the existing adaptive endpoint; no training or teacher generation."""
import argparse
import json
import math
import os
from pathlib import Path

from core import METRICS,aggregate
from worker import Engine,NUMERICS,eval_rows,guard,same,save,training_info

UNIT='stage1_adaptive'

def validate_search(saved,p,summary,selected):
    state=saved['search_state'];snap=saved['snapshot']
    if not (summary.get('complete') and summary.get('protocol')==p['protocol']
            and summary.get('numerics')==NUMERICS and summary.get('rounds')==8 and summary.get('updates')==16
            and selected.get('complete') and selected.get('protocol')==p['protocol']
            and selected.get('numerics')==NUMERICS and selected.get('pbs_jobid')==summary.get('pbs_jobid')):
        raise RuntimeError('Completed deterministic search receipts required')
    if not (state.get('protocol')==p['protocol'] and state.get('numerics')==NUMERICS
            and state.get('round')==8 and state.get('updates')==16 and snap.get('updates')==16
            and state.get('commits_in_round')==0 and state.get('round_update') is None and not state.get('probes')
            and snap.get('optimizer_reset_after_update1') is True and snap['optimizer_state']['state']):
        raise RuntimeError('Adaptive snapshot is not the completed search endpoint')
    history=state['history'];training=state['training_history']
    if history!=selected['history'] or [r['round'] for r in history]!=list(range(1,9)):
        raise RuntimeError('Adaptive rho trajectory differs from selection receipt')
    if ([r['case_id'] for r in training]!=p['calibration']['case_ids']*2
            or [r['update'] for r in training]!=list(range(1,17))
            or [r['rho'] for r in training]!=[h['rho_after'] for h in history for _ in range(2)]
            or not math.isclose(state['rho'],selected['rho'],rel_tol=1e-12)):
        raise RuntimeError('Adaptive committed training history differs')
    return state,snap

def baseline_rows(e,arm,split,cases,seeds):
    info=training_info(e.root,arm)
    unit='outer_evaluation' if split=='outer' else 'evaluate_'+arm
    rows=[]
    for case in cases:
        for seed in seeds:
            row=e.rows[case]
            value=json.loads((e.root/'results'/unit/'queries'/arm/split/f'{case}_{seed}.json').read_text())
            binding={'protocol':e.p['protocol'],'arm':arm,'split':split,'case_id':case,'seed_index':seed,
                     'sampler_seed':row['sampler_seeds'][seed],'checkpoint':info['checkpoint'],
                     'rho':info['rho'],'updates':16,'suite':row['suite'],'plan_row':row}
            if any(value.get(k)!=v for k,v in binding.items()):raise RuntimeError('Cached fixed-rho endpoint binding differs')
            rows.append(value)
    return rows

def compare(records,p):
    summaries={};paired={};changes={}
    for split,cases,seeds in (('outer',p['development']['outer_case_ids'],[0]),
                             ('confirmation',p['development']['confirmation_case_ids'],[0,1])):
        summaries[split]={arm:aggregate(records[split][arm],cases,seeds) for arm in ('adaptive','current','selected')}
        paired[split]=[];changes[split]={}
        for arm in ('current','selected'):
            changes[split][arm]={m:percent(summaries[split]['adaptive']['overall'][m],summaries[split][arm]['overall'][m]) for m in METRICS}
        for case in cases:
            means={arm:{m:sum(r[m] for r in records[split][arm] if r['case_id']==case)/len(seeds) for m in METRICS}
                   for arm in ('adaptive','current','selected')}
            row=next(r for r in records[split]['adaptive'] if r['case_id']==case)
            paired[split].append({'case_id':case,'suite':row['suite'],'phase':row['plan_row']['phase'],
                'means':means,'adaptive_percent_vs':{arm:{m:percent(means['adaptive'][m],means[arm][m]) for m in METRICS} for arm in ('current','selected')},
                'seed_records':[{k:r[k] for k in ('arm','seed_index',*METRICS)} for arm in ('adaptive','current','selected') for r in records[split][arm] if r['case_id']==case]})
    return {'summaries':summaries,'adaptive_percent_vs_fixed':changes,'paired_cases':paired}

def percent(new,base):
    return 100*(new/base-1) if base>0 else (0.0 if new==0 else None)

def report(result,path):
    lines=['# Adaptive 搜索终点与固定 rho 的阶段1对比','',
           '仅评估已有8轮16步adaptive checkpoint；复用teacher及固定rho1.5/约2的已有结果。没有训练、更新rho或生成teacher。旧confirmation已被观察，本次为路径差异诊断。','',
           '| Split | Arm | first10 motor | full32 motor | late22 motor | gripper | future NMSE |',
           '|---|---|---:|---:|---:|---:|---:|']
    for split in ('outer','confirmation'):
        for arm in ('current','selected','adaptive'):
            m=result['summaries'][split][arm]['overall'];lines.append('| '+split+' | '+arm+' | '+' | '.join(f'{m[k]:.9f}' for k in METRICS)+' |')
    for split in ('outer','confirmation'):
        lines+=['',f'## {split} 逐case first10 motor RMSE','',
                '| Case | Suite | Phase | 固定rho1.5 | 固定rho约2 | Adaptive | vs rho1.5 | vs rho约2 |',
                '|---|---|---|---:|---:|---:|---:|---:|']
        for r in result['paired_cases'][split]:
            m=r['means'];v=r['adaptive_percent_vs'];lines.append(f"| {r['case_id']} | {r['suite']} | {r['phase']} | {m['current']['first10motor_rmse']:.9f} | {m['selected']['first10motor_rmse']:.9f} | {m['adaptive']['first10motor_rmse']:.9f} | {v['current']['first10motor_rmse']:+.2f}% | {v['selected']['first10motor_rmse']:+.2f}% |")
    lines+=['','全部五项逐case指标及两个seed的原始值见comparison.json。每case先平均seeds，再按suite平均，四suite等权。',
            '', '证据边界：fixed-input teacher fidelity，decoded BF16/fake A4、单一C8训练轨迹；不代表closed-loop成功率、native kernel性能或训练seed复现。阶段2和3未提交。']
    path.write_text('\n'.join(lines)+'\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True);args=a.parse_args()
    guard();p=json.loads(args.protocol.read_text());e=None
    try:
        root=Path(p['remote_root']);gate=json.loads((root/'results/preflight/summary.json').read_text())
        if not gate.get('gate_passed') or gate.get('numerics')!=NUMERICS:raise RuntimeError('Deterministic preflight gate required')
        source=json.loads((root/'results/search/summary.json').read_text());selected=json.loads((root/'results/selected_rho.json').read_text())
        e=Engine(p,UNIT);checkpoint=root/'results/search/checkpoint.pt'
        saved=e.torch.load(checkpoint,map_location='cpu',weights_only=True)
        state,snap=validate_search(saved,p,source,selected)
        records={};splits=(('outer',p['development']['outer_case_ids'],[0]),('confirmation',p['development']['confirmation_case_ids'],[0,1]))
        # All source tensors remain read-only; a missing teacher must fail, never regenerate.
        for split,cases,seeds in splits:
            records[split]={arm:baseline_rows(e,arm,split,cases,seeds) for arm in ('current','selected')}
            for case in cases:
                for seed in seeds:e.teacher(case,seed)
        e.install();e.restore(snap)
        info={'checkpoint':str(checkpoint),'rho':state['rho']}
        for split,cases,seeds in splits:
            records[split]['adaptive']=eval_rows(e,'adaptive',cases,seeds,split,info)
        unchanged=same(e.torch,e.banks.snapshot(),snap['module_masters'])
        if e.g or e.teacher_new or not unchanged:raise RuntimeError('Stage1 must not train or mutate the checkpoint')
        result={'protocol':p['protocol'],'unit':UNIT,'pbs_jobid':os.environ['PBS_JOBID'],'numerics':NUMERICS,'complete':True,
                'queries':24,'split_queries':{'outer':8,'confirmation':16},'cached_baseline_queries':48,'comparison_records':72,
                'new_training_updates':0,'new_teacher_queries':0,'new_gradient_queries_this_job':e.g,'new_inference_queries_this_job':e.f,
                'master_parameters_unchanged':unchanged,'source_checkpoint':str(checkpoint),'source_search_jobid':source['pbs_jobid'],
                'source_committed_updates':16,'source_rounds':8,'final_rho':state['rho'],'rho_schedule':[r['rho'] for r in state['training_history']],
                'stage2_submitted':False,'stage3_submitted':False,**compare(records,p)}
        save(e.out/'comparison.json',result);report(result,e.out/'RESULTS.zh.md')
        e.complete(**{k:v for k,v in result.items() if k not in ('protocol','unit','pbs_jobid','numerics','complete')})
        print('STAGE1_COMPLETE '+json.dumps(result),flush=True);return 0
    except TimeoutError as error:
        if e:e.progress(phase='checkpointed',reason=str(error),resume_scope='reuse completed endpoint JSON; reload the same read-only adaptive checkpoint')
        return 75
    except Exception as error:
        if e:e.progress(phase='failed',error=type(error).__name__+': '+str(error))
        raise

if __name__=='__main__':raise SystemExit(main())

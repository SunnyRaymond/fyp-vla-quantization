"""Calibration-only allocation, then paired full-cohort analysis, on CPU nodes."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from common import guard,protocol,contexts,stem,save,receipt,paired_summary


def allocate(p):
    import torch
    from allocation import ranked_levels,normalized_positive,module_matched
    from checkpointing import atomic_torch_save
    from test_capacity import self_check
    self_check()
    root=Path(p['remote_root'])
    layout=json.loads((root/'layout.json').read_text())
    captured=[]
    for i,case in enumerate(p['calibration_cases']):
        data=torch.load(root/f'results/score-{i%4}/case_{case}.pt',map_location='cpu',weights_only=True)
        if data['protocol']!=p['protocol'] or data['case_id']!=case or data['seed_index']!=0: raise RuntimeError('Calibration score identity differs')
        captured.append(data)
    scores={key:torch.stack([d['scores'][key] for d in captured]).mean(0) for key in captured[0]['scores']}
    scores['future_reader_damage']=0.5*(normalized_positive(torch,scores['latent_damage'])+normalized_positive(torch,scores['reader_damage']))
    no_positive_signal=not bool(scores['future_reader_damage'].abs().sum()>0)
    n=layout['total_tiles']; levels={}
    for arm in p['allocation_arms']:
        if arm.startswith('matched_module_'): continue
        if arm=='uniform' or (arm=='future_reader_damage' and no_positive_signal): levels[arm]=torch.ones(n,dtype=torch.uint8)
        elif arm.startswith('random'):
            g=torch.Generator().manual_seed(p['allocation_seed']+int(arm[-1]))
            levels[arm]=ranked_levels(torch,torch.rand(n,generator=g))
        else: levels[arm]=ranked_levels(torch,scores[arm])
        if len(levels[arm])!=n or int(levels[arm].long().sum())!=n: raise RuntimeError('Unequal allocation byte budget')
    for arm in p['allocation_arms']:
        if not arm.startswith('matched_module_'): continue
        if arm=='matched_module_random':
            s=torch.rand(n,generator=torch.Generator().manual_seed(p['allocation_seed']+100))
        else: s=scores[arm.removeprefix('matched_module_')]
        levels[arm]=module_matched(torch,levels['future_reader_damage'],s,layout['modules'])
    canonical={}
    for arm in p['allocation_arms']:
        canonical[arm]=next((previous for previous in canonical if torch.equal(levels[previous],levels[arm])),arm)
    overlap={}
    for arm,a in levels.items():
        overlap[arm]={}
        for other,b in levels.items():
            overlap[arm][other]=float(torch.minimum(a,b).long().sum()/torch.maximum(a,b).long().sum())
    stability={}
    for arm in ('latent_damage','reader_damage','future_reader_damage','joint_fisher'):
        values=[]
        for excluded in range(8):
            avg={key:torch.stack([d['scores'][key] for i,d in enumerate(captured) if i!=excluded]).mean(0) for key in ('latent_damage','reader_damage','joint_fisher')}
            avg['future_reader_damage']=0.5*(normalized_positive(torch,avg['latent_damage'])+normalized_positive(torch,avg['reader_damage']))
            candidate=ranked_levels(torch,avg[arm]); full=levels[arm]
            values.append(float(torch.minimum(candidate,full).long().sum()/torch.maximum(candidate,full).long().sum()))
        stability[arm]=dict(leave_one_context_out_jaccard=values,mean=statistics.mean(values),minimum=min(values))
    top_modules={}
    for arm,l in levels.items():
        values=[]
        for m in layout['modules']:
            capacity=int(l[m['offset']:m['offset']+m['tiles']].long().sum())*p['tile_vectors']//2
            values.append(dict(module=m['name'],residual_index_bytes=capacity))
        top_modules[arm]=sorted(values,key=lambda x:x['residual_index_bytes'],reverse=True)[:20]
    payload=dict(protocol=p['protocol'],levels=levels,canonical_arm=canonical,budget=layout['budget'],scores=scores,calibration_cases=p['calibration_cases'])
    atomic_torch_save(torch,root/'allocations.pt',payload)
    result=receipt('allocate',status='complete',score_context_count=8,total_tiles=n,allocation_arms=p['allocation_arms'],
                   budget=layout['budget'],no_positive_combined_damage_signal=no_positive_signal,
                   canonical_arm=canonical,per_module_matched_controls=[a for a in levels if a.startswith('matched_module_')],
                   weighted_allocation_jaccard=overlap,calibration_stability=stability,top_modules=top_modules,model_forward_count=0)
    save(root/'allocation_summary.json',result)


def analyze(p):
    root=Path(p['remote_root']); arms=p['allocation_arms']; rows=[]
    allocation=json.loads((root/'allocation_summary.json').read_text())
    for c in contexts(p):
        for arm in arms:
            path=root/f"results/evaluate-{c['shard']}"/arm/f'{stem(c)}.json'
            r=json.loads(path.read_text())
            if (r['protocol']!=p['protocol'] or r['arm']!=arm or r['case_id']!=c['case_id'] or r['seed_index']!=c['seed_index'] or r['split']!=c['split']
                    or r['budget_bytes']!=allocation['budget']['total_bytes'] or not path.with_suffix('.pt').is_file()):
                raise RuntimeError('Final row identity/budget mismatch')
            rows.append(r)
    if len(rows)!=60*len(arms): raise RuntimeError('Incomplete full-cohort allocation comparison')
    fields=['reader_first10motor_rmse','reader_full32motor_rmse','reader_late22motor_rmse','reader_first10gripper_rmse',
            'native_first10motor_rmse','native_full32motor_rmse','native_late22motor_rmse','native_first10gripper_rmse','future_nmse','added_future_nmse','reader_drift_from_full_rmse']
    tables={}; comparisons={}
    for split,count in [('calibration',8),('selection',4),('test',22)]:
        for arm in arms:
            selected=[r for r in rows if r['split']==split and r['arm']==arm]
            tables[f'{split}/{arm}']={m:paired_summary(selected,m) for m in fields}
            if any(v['case_count']!=count for v in tables[f'{split}/{arm}'].values()): raise RuntimeError('Case denominator differs')
    for arm in ('future_reader_damage','latent_damage','reader_damage'):
        comparisons[arm]={}
        for baseline in (arms if arm=='future_reader_damage' else ['uniform']):
            metrics={}
            for m in fields:
                a=tables[f'test/{arm}'][m]; b=tables[f'test/{baseline}'][m]
                ds={c:a['per_case'][c]-b['per_case'][c] for c in a['per_case']}
                loo={c:statistics.mean(v for k,v in ds.items() if k!=c) for c in ds}
                metrics[m]=dict(mean_delta=a['mean']-b['mean'],median_delta=statistics.median(ds.values()),
                    improved=sum(v<0 for v in ds.values()),worsened=sum(v>0 for v in ds.values()),tied=sum(v==0 for v in ds.values()),per_case_delta=ds,
                    leave_one_case_out_mean_delta_min=min(loo.values()),leave_one_case_out_mean_delta_max=max(loo.values()))
            comparisons[arm][baseline]=metrics
    prior=Path(p['future_reader_root'])
    teacher=json.loads((prior/'results/analysis_teacher/analysis.json').read_text())
    cross=json.loads((prior/'results/analysis_cross/analysis.json').read_text())
    full_reference=dict(reader=teacher['tables']['test/bf16/action_only'],native=cross['tables']['test/action_only/action_only'],scope='Cached full two-book action-only epoch2; outside sparse matched budget')
    result=dict(protocol=p['protocol'],phase='analyze',job_id=os.environ['PBS_JOBID'],status='complete',row_count=len(rows),total_contexts=60,test_case_count=22,test_context_count=44,model_forward_count=0,
        tables=tables,comparisons=comparisons,full_reference=full_reference,budget=allocation['budget'],claim_scope=p['claim_scope'])
    out=root/'results/analysis';out.mkdir(parents=True,exist_ok=True)
    save(out/'analysis.json',result)
    compact=dict(protocol=p['protocol'],status='complete',row_count=len(rows),total_contexts=60,test_case_count=22,
        tables={name:{m:{k:v for k,v in value.items() if k!='per_case'} for m,value in metrics.items()} for name,metrics in tables.items()},
        comparisons={arm:{base:{m:{k:v for k,v in value.items() if k!='per_case_delta'} for m,value in metrics.items()} for base,metrics in contrasts.items()} for arm,contrasts in comparisons.items()},
        test_per_case={arm:{m:tables[f'test/{arm}'][m]['per_case'] for m in ('reader_first10motor_rmse','native_first10motor_rmse','future_nmse')} for arm in arms},
        full_reference=full_reference,budget=allocation['budget'],claim_scope=p['claim_scope'])
    (out/'primary_summary.json').write_text(json.dumps(compact,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    if (out/'primary_summary.json').stat().st_size>262144: raise RuntimeError('Review summary exceeds bounded-transfer limit')
    header=['split','case_id','seed_index','sampler_seed','arm','domain','dimension','budget_bytes']+fields
    with (out/'paired_contexts.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=header,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with (out/'paired_primary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=header[:8]+['reader_first10motor_rmse','native_first10motor_rmse','future_nmse'],extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    lines=['# Residual VQ capacity diagnostic','',f'All{len(arms)} matched-budget arms, full60 contexts; test22 cases/44 contexts. Paired seeds averaged within each case.','',
           '| Allocation | Fixed-reader first10 | Native first10 | Future NMSE | First10 versus uniform | Improved cases | Median delta |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        t=tables[f'test/{arm}'];u=tables['test/uniform']['reader_first10motor_rmse'];a=t['reader_first10motor_rmse']
        ds=[a['per_case'][c]-u['per_case'][c] for c in a['per_case']]
        d=dict(mean_delta=a['mean']-u['mean'],improved=sum(v<0 for v in ds),median_delta=statistics.median(ds))
        lines.append(f"| {arm} | {t[fields[0]]['mean']:.9f} | {t['native_first10motor_rmse']['mean']:.9f} | {t['future_nmse']['mean']:.9f} | {d['mean_delta']:.9f} | {d['improved']}/22 | {d['median_delta']:.9f} |")
    lines+=['',f"Matched bytes: {allocation['budget']}",'',p['claim_scope'],'',
            'Ranking overlap and calibration leave-one-context-out stability: allocation_summary.json. Full horizons/gripper and proposed-versus-all-baseline paired contrasts: primary_summary.json; detailed per-case secondary metrics: analysis.json.',
            'Joint Fisher here uses rollout action+future distillation losses at the frozen student, not QuantWAMs pretraining/co-training objective. It is an information-matched Fisher control, not a reproduction claim.']
    (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt('analyze',status='complete',row_count=len(rows),total_contexts=60,test_case_count=22,test_context_count=44,model_forward_count=0,
            bounded_result='results/analysis/primary_summary.json')


if __name__=='__main__':
    guard()
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=('allocate','analyze'),required=True)
    args=a.parse_args();(allocate if args.phase=='allocate' else analyze)(protocol())

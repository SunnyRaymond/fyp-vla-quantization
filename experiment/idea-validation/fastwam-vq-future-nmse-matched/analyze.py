"""Analyze all60 contexts and decide the frozen engineering progression gate."""
import csv
import json
from pathlib import Path
from statistics import mean,median
from common import guard,protocol,contexts,stem,aggregate,paired,save,receipt
from matching import gate,nmse

METRICS=('first10motor_rmse','full32motor_rmse','late22motor_rmse','first10gripper_rmse')


def main():
    guard()
    import torch
    import metrics
    p=protocol();root=Path(p['remote_root']);manifest=json.loads((root/'input_manifest.json').read_text());entries={stem(c):c for c in manifest['contexts']}
    rows=[];replays=[]
    for shard in (0,1):
        out=root/f'results/matched-{shard}';s=json.loads((out/'summary.json').read_text())
        if s['protocol']!=p['protocol'] or s['status']!='complete' or s['shard']!=shard or s['row_count']!=30 or s['context_count']!=30 or s['replay_max_abs']!=0 or s['replay_count_this_segment']!=4:raise RuntimeError('Incomplete matched GPU shard')
        for path in (out/'replays').glob('*.json'):
            r=json.loads(path.read_text())
            if r['protocol']!=p['protocol'] or r['max_abs']!=0:raise RuntimeError('Replay differs')
            replays.append(r)
    for c in contexts(p):
        d=torch.load(entries[stem(c)]['prepared_path'],map_location='cpu',weights_only=True)
        if d['protocol']!=p['protocol'] or d['context']!=c:raise RuntimeError('Prepared identity differs')
        for donor in ('action_only','action_latent','matched'):
            if donor=='matched':
                path=root/f"results/matched-{c['shard']}"/f'{stem(c)}_matched.json';r=json.loads(path.read_text());payload=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
                if payload['row']!=r or r['context']!=c or r['protocol']!=p['protocol'] or not r['future_injection_verified']:raise RuntimeError('New reader result identity differs')
                action=payload['action']
            else:action=d['reader_actions'][donor];r=d['reader_rows'][donor]
            mm=metrics.action_metrics(torch,action,d['teacher_action'])
            if any(abs(mm[k]-r[k])>1e-12 for k in METRICS):raise RuntimeError('Cached action metric differs from saved outputs')
            future=nmse(torch,d['futures'][donor],d['teacher_future'])
            mse=float((action[:10,:6].double()-d['teacher_action'][:10,:6].double()).square().mean())
            row=dict(**c,sampler_seed=d['sampler_seed'],reader='bf16',donor=donor,domain=entries[stem(c)]['domain'],dimension=entries[stem(c)]['dimension'],
                **mm,first10motor_mse=mse,future_nmse=future,alpha=d['matching']['alpha'],absolute_nmse_gap=d['matching']['absolute_nmse_gap'],relative_nmse_gap=d['matching']['relative_nmse_gap'],direction_cosine=d['matching']['direction_cosine'])
            rows.append(row)
        q=d['matching']
        if abs(q['c_nmse']-q['b_nmse'])>q['matching_tolerance'] or (q['direction_cosine'] is not None and q['direction_cosine']<p['direction_cosine_minimum']):raise RuntimeError('NMSE or direction match failed')
    if len(rows)!=180:raise RuntimeError('Full A/B/C coverage required')
    tables={};comparisons={}
    for split in ('calibration','selection','test'):
        for donor in ('action_only','action_latent','matched'):
            rr=[r for r in rows if r['split']==split and r['donor']==donor]
            tables[f'{split}/{donor}']={k:aggregate(rr,k) for k in METRICS+('first10motor_mse','future_nmse')}
        for left,right in (('action_latent','matched'),('matched','action_only'),('action_latent','action_only')):
            a=[r for r in rows if r['split']==split and r['donor']==left];b=[r for r in rows if r['split']==split and r['donor']==right]
            cc={k:paired(a,b,k) for k in METRICS+('first10motor_mse','future_nmse')}
            for v in cc.values():
                values=list(v['per_case_delta'].values());v['median_delta']=median(values)
                v['leave_one_case_out_mean_min']=min(mean(values[:i]+values[i+1:]) for i in range(len(values)))
                v['leave_one_case_out_mean_max']=max(mean(values[:i]+values[i+1:]) for i in range(len(values)))
            comparisons[f'{split}/{left}-minus-{right}']=cc
    progression=gate(comparisons['test/action_latent-minus-matched']['first10motor_rmse']['per_case_delta'])
    result=receipt('analyze',status='complete',context_count=60,test_case_count=22,test_context_count=44,row_count=180,new_matched_rows=60,cached_reference_rows=120,
        model_forward_count=0,new_future_generation=0,progression_gate_passed=progression['passed'],progression=progression,progression_definition=p['progression_gate'],
        tables=tables,comparisons=comparisons,replay_count=len(replays),replay_max_abs=max(r['max_abs'] for r in replays),
        max_relative_nmse_gap=max(e['relative_nmse_gap'] for e in entries.values()),max_absolute_nmse_gap=max(e['absolute_nmse_gap'] for e in entries.values()),
        min_direction_cosine=min((e['direction_cosine'] for e in entries.values() if e['direction_cosine'] is not None),default=None),claim_scope=p['claim_scope'])
    out=root/'results/analysis';out.mkdir(parents=True,exist_ok=True);save(out/'analysis.json',result)
    with (out/'paired_contexts.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Equal-future-NMSE control','',p['claim_scope'],'','| Split / donor | first10 motor | full32 motor | late22 motor | gripper | future NMSE |','|---|---:|---:|---:|---:|---:|']
    for name,v in tables.items():lines.append('| '+name+' | '+' | '.join(f"{v[k]['mean']:.9g}" for k in METRICS+('future_nmse',))+' |')
    lines+=['','Frozen progression gate: '+json.dumps(progression),'','Matching and replay: '+json.dumps({k:result[k] for k in ('max_relative_nmse_gap','max_absolute_nmse_gap','min_direction_cosine','replay_count','replay_max_abs')})]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()

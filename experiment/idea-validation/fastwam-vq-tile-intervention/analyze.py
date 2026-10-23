"""Case-blocked prediction checks and exact loss/output interaction decomposition."""
import csv
import json
import math
from pathlib import Path
import statistics as st
from common import guard,protocol,save,receipt
from design import interaction

TARGETS={'native_mse':'action_damage','future_nmse':'latent_damage','reader_mse':'reader_damage'}


def rank(x):
    order=sorted(range(len(x)),key=lambda i:x[i]);r=[0.0]*len(x);j=0
    while j<len(order):
        k=j+1
        while k<len(order) and x[order[k]]==x[order[j]]:k+=1
        for i in order[j:k]:r[i]=(j+k-1)/2
        j=k
    return r


def corr(a,b):
    if len(a)<2 or len(set(a))<2 or len(set(b))<2:return None
    return st.correlation(a,b)


def sign(v,floor):return 1 if v>floor else (-1 if v<-floor else 0)


def predictive(rows):
    a=[r['prediction'] for r in rows];b=[r['actual_delta'] for r in rows]
    informative=[r for r in rows if sign(r['prediction'],r['noise_floor']) or sign(r['actual_delta'],r['noise_floor'])]
    correct=sum(sign(r['prediction'],r['noise_floor'])==sign(r['actual_delta'],r['noise_floor']) for r in informative)
    return dict(n=len(rows),pearson=corr(a,b),spearman=corr(rank(a),rank(b)),mean_prediction=st.mean(a),mean_actual_delta=st.mean(b),
        mean_absolute_error=st.mean(abs(x-y) for x,y in zip(a,b)),median_absolute_error=st.median(abs(x-y) for x,y in zip(a,b)),
        informative_sign_n=len(informative),sign_correct=correct,sign_accuracy=correct/len(informative) if informative else None,
        actual_positive=sum(sign(r['actual_delta'],r['noise_floor'])>0 for r in rows),actual_negative=sum(sign(r['actual_delta'],r['noise_floor'])<0 for r in rows),
        mean_quadratic_loss_term=st.mean(r.get('quadratic_loss_term',0) for r in rows),
        mean_absolute_finite_response_residual=st.mean(abs(r.get('finite_response_residual',0)) for r in rows))


def interaction_stats(rows):
    return dict(n=len(rows),median_absolute_interaction=st.median(abs(r['interaction']) for r in rows),
        median_interaction_ratio=st.median(r['interaction_ratio'] for r in rows),max_interaction_ratio=max(r['interaction_ratio'] for r in rows),
        median_output_nonadditivity_ratio=st.median(r['output_nonadditivity_ratio'] for r in rows),
        mean_absolute_quadratic_cross_term=st.mean(abs(r['quadratic_loss_cross_term']) for r in rows),
        mean_absolute_response_nonadditivity_loss=st.mean(abs(r['response_nonadditivity_loss']) for r in rows),
        sign_flip_count=sum(r['additive_sign_flip'] for r in rows),
        nonzero_additive_count=sum(abs(r['sum_single_delta'])>r['noise_floor'] for r in rows))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    guard()
    import torch
    p=protocol();root=Path(p['remote_root']);plan=json.loads((root/'plan.json').read_text())
    tiles={t['id']:t for t in plan['tiles']};conditions={c['id']:c for c in plan['conditions']}
    single=[];multi=[];baselines=[];fisher=[];negative_controls=[];row_count=0
    manifest=json.loads((Path(p['future_reader_root'])/'input_manifest.json').read_text())
    latent_weight=json.loads((Path(p['parent_root'])/'prepare_summary.json').read_text())['latent_weight']
    for shard,case in enumerate(p['calibration_cases']):
        out=root/f'results/evaluate-{shard}';e=next(c for c in manifest['contexts'] if c['case_id']==case and c['seed_index']==0)
        old=torch.load(e['cache_path'],map_location='cpu',weights_only=True)['captures']['bf16']
        teacher={'native_mse':old['action'][:10,:6].double(),'reader_mse':old['action'][:10,:6].double(),'future_nmse':old['video_latents'][:,:,1:].double()}
        def loss(y,target):
            err=(y-teacher[target]).square()
            return float(err.sum()/teacher[target].square().sum().clamp_min(1e-12)) if target=='future_nmse' else float(err.mean())
        vectors={};observed={}
        for cid,cond in conditions.items():
            r=json.loads((out/'rows'/f'{cid}.json').read_text())
            n=torch.load(out/'native'/f'{cid}.pt',map_location='cpu',weights_only=True)
            b=torch.load(out/'rows'/f'{cid}.pt',map_location='cpu',weights_only=True)
            if r['protocol']!=p['protocol'] or r['case_id']!=case or r['condition']!=cond or n['condition']!=cond or n['sampler_seed']!=e['sampler_seed'] or b['row']!=r:raise RuntimeError('Result identity differs')
            vectors[cid]={'native_mse':n['action'][:10,:6].double(),'reader_mse':b['reader_action'][:10,:6].double(),'future_nmse':n['future'][:,:,1:].double()}
            observed[cid]={target:loss(y,target) for target,y in vectors[cid].items()}
            if any(abs(observed[cid][k]-r[k])>1e-12 for k in TARGETS):raise RuntimeError('Saved metric differs from outputs')
            if cid=='baseline_start':
                if any(v['max_abs']!=0 for v in r['native_parity'].values()) or r['reader_parity']['max_abs']!=0:raise RuntimeError('Baseline source parity not exact')
            row_count+=1
        base=observed['baseline_start'];floors={k:max(2*abs(observed['baseline_repeat'][k]-base[k]),1e-12) for k in TARGETS}
        baselines.append(dict(case_id=case,baseline=base,repeat=observed['baseline_repeat'],noise_floor=floors,
            repeat_output_max_abs={k:float((vectors['baseline_repeat'][k]-vectors['baseline_start'][k]).abs().max()) for k in TARGETS}))
        for tid,t in tiles.items():
            cid='single_'+tid
            for target,score_key in TARGETS.items():
                prediction=t['predictions'][str(case)][score_key]
                dy=vectors[cid][target]-vectors['baseline_start'][target]
                e0=vectors['baseline_start'][target]-teacher[target]
                denom=teacher[target].square().sum().clamp_min(1e-12) if target=='future_nmse' else dy.numel()
                linear=float(2*(e0*dy).sum()/denom);quadratic=float(dy.square().sum()/denom)
                actual=observed[cid][target]-base[target]
                if abs(actual-linear-quadratic)>1e-12:raise RuntimeError('Single MSE decomposition differs')
                single.append(dict(case_id=case,tile_id=tid,module=t['module'],label=t['label'],target=target,
                    prediction=prediction,actual_delta=actual,noise_floor=floors[target],
                    finite_output_linear_term=linear,quadratic_loss_term=quadratic,finite_response_residual=linear-prediction))
            da=observed[cid]['native_mse']-base['native_mse'];dz=observed[cid]['future_nmse']-base['future_nmse']
            pred=t['predictions'][str(case)]
            fisher.append(dict(case_id=case,tile_id=tid,module=t['module'],action_fisher=pred['action_fisher'],joint_fisher=pred['joint_fisher'],
                native_delta=da,joint_delta=da+latent_weight*dz))
            if t['module'].startswith('action_'):
                negative_controls.append(dict(case_id=case,tile_id=tid,future_max_abs=float((vectors[cid]['future_nmse']-vectors['baseline_start']['future_nmse']).abs().max()),
                    reader_max_abs=float((vectors[cid]['reader_mse']-vectors['baseline_start']['reader_mse']).abs().max())))
        for cid,cond in conditions.items():
            if cond['size']<2:continue
            for target in TARGETS:
                y0=vectors['baseline_start'][target];individual=[vectors['single_'+tid][target]-y0 for tid in cond['tile_ids']]
                yadd=y0+sum(individual);dy=vectors[cid][target]-y0
                decomposition=interaction(observed[cid][target],base[target],[observed['single_'+tid][target] for tid in cond['tile_ids']],loss(yadd,target))
                if abs(decomposition['interaction']-decomposition['quadratic_loss_cross_term']-decomposition['response_nonadditivity_loss'])>1e-12:raise RuntimeError('Interaction decomposition differs')
                norm_den=sum(float(v.square().sum().sqrt()) for v in individual)
                loss_den=sum(abs(observed['single_'+tid][target]-base[target]) for tid in cond['tile_ids'])
                branches={tiles[t]['module'].split('_')[0] for t in cond['tile_ids']}
                branch=next(iter(branches)) if len(branches)==1 else 'mixed'
                multi.append(dict(case_id=case,condition_id=cid,kind=cond['kind'],branch=branch,size=cond['size'],target=target,**decomposition,
                    interaction_ratio=abs(decomposition['interaction'])/max(loss_den,floors[target]),
                    output_nonadditivity_norm=float((dy-sum(individual)).square().sum().sqrt()),
                    output_nonadditivity_ratio=float((dy-sum(individual)).square().sum().sqrt())/max(norm_den,1e-12),noise_floor=floors[target],
                    additive_sign_flip=sign(decomposition['joint_delta'],floors[target])!=sign(decomposition['sum_single_delta'],floors[target]) and bool(sign(decomposition['joint_delta'],floors[target])) and bool(sign(decomposition['sum_single_delta'],floors[target]))))
        del vectors,old,teacher
    if row_count!=432 or len(single)!=672 or len(multi)!=576 or len(fisher)!=224 or len(negative_controls)!=112:raise RuntimeError('Full diagnostic coverage differs')
    by_module={};by_case={};mean_tile=[]
    for target in TARGETS:
        for module in p['selected_modules']:
            chosen=[r for r in single if r['target']==target and r['module']==module]
            by_module[target+'/'+module]=predictive(chosen)
            for tid in {r['tile_id'] for r in chosen}:
                rr=[r for r in chosen if r['tile_id']==tid]
                mean_tile.append(dict(case_id='mean8',tile_id=tid,module=module,label=rr[0]['label'],target=target,prediction=st.mean(r['prediction'] for r in rr),actual_delta=st.mean(r['actual_delta'] for r in rr),noise_floor=st.mean(r['noise_floor'] for r in rr),finite_output_linear_term=st.mean(r['finite_output_linear_term'] for r in rr),quadratic_loss_term=st.mean(r['quadratic_loss_term'] for r in rr),finite_response_residual=st.mean(r['finite_response_residual'] for r in rr)))
        for case in p['calibration_cases']:
            for module in p['selected_modules']:
                by_case[f'{target}/{case}/{module}']=predictive([r for r in single if r['target']==target and r['case_id']==case and r['module']==module])
    mean_stats={target+'/'+module:predictive([r for r in mean_tile if r['target']==target and r['module']==module]) for target in TARGETS for module in p['selected_modules']}
    fisher_stats={}
    for module in p['selected_modules']:
        rr=[r for r in fisher if r['module']==module]
        for score,target in [('action_fisher','native_delta'),('joint_fisher','joint_delta')]:
            fisher_stats[module+'/'+score]=dict(n=len(rr),spearman_signed=corr(rank([r[score] for r in rr]),rank([r[target] for r in rr])),spearman_absolute=corr(rank([r[score] for r in rr]),rank([abs(r[target]) for r in rr])))
    interaction_summary={}
    branch_interaction={}
    for target in TARGETS:
        for size in (2,4,8):interaction_summary[f'{target}/size{size}']=interaction_stats([r for r in multi if r['target']==target and r['size']==size])
        for branch in ('video','action','mixed'):
            rr=[r for r in multi if r['target']==target and r['branch']==branch]
            branch_interaction[target+'/'+branch]=interaction_stats(rr)
    summary=dict(protocol=p['protocol'],status='complete',row_count=row_count,case_count=8,tile_count=28,single_intervention_rows=224,multi_intervention_rows=192,
        baseline_rows=16,by_module=by_module,mean_tile_predictive=mean_stats,fisher_rank=fisher_stats,interaction_by_size=interaction_summary,interaction_by_branch=branch_interaction,baseline_noise=baselines,
        action_module_negative_controls=dict(n=len(negative_controls),max_future_abs=max(r['future_max_abs'] for r in negative_controls),max_reader_abs=max(r['reader_max_abs'] for r in negative_controls)),
        scope=p['claim_scope'],interpretation='Correlations are descriptive within4predefined modules; contexts/tiles are repeated measurements, not independent population replicates. Signed scores predict MSE/NMSE change; Fisher only ranked on its corresponding native/joint objective. Loss nonadditivity is decomposed because quadratic cross terms exist even for additive outputs.')
    out=root/'results/analysis';out.mkdir(parents=True,exist_ok=True)
    save(out/'primary_summary.json',summary);save(out/'per_case_predictive.json',by_case)
    write_csv(out/'single_tile.csv',single);write_csv(out/'mean_tile.csv',mean_tile);write_csv(out/'interactions.csv',multi);write_csv(out/'fisher.csv',fisher)
    lines=['# Tile intervention results','',summary['interpretation'],'','| Target / module | N | Spearman | Sign correct / informative | Mean prediction | Mean true delta |','|---|---:|---:|---:|---:|---:|']
    for key,v in by_module.items():lines.append(f"| {key} | {v['n']} | {v['spearman']} | {v['sign_correct']}/{v['informative_sign_n']} | {v['mean_prediction']:.10g} | {v['mean_actual_delta']:.10g} |")
    lines+=['','| Target / simultaneous size | N | Median abs interaction / sum abs singles | Median output nonadditivity ratio | Sign flips |','|---|---:|---:|---:|---:|']
    for key,v in interaction_summary.items():lines.append(f"| {key} | {v['n']} | {v['median_interaction_ratio']:.6g} | {v['median_output_nonadditivity_ratio']:.6g} | {v['sign_flip_count']}/{v['nonzero_additive_count']} |")
    lines+=['','Action-module future/reader negative controls: '+json.dumps(summary['action_module_negative_controls']),'','Baseline repeated noise: '+json.dumps(baselines)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt('analyze',status='complete',row_count=432,case_count=8,tile_count=28,single_intervention_rows=224,multi_intervention_rows=192,baseline_rows=16,model_forward_count=0)


if __name__=='__main__':main()

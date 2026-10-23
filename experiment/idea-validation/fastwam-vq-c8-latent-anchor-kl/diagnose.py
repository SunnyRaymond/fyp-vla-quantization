"""Post-hoc CPU diagnostics on saved tensors; no model or checkpoint is loaded."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import statistics as st
from anchor_selfcheck import require_allocation
from prepare import read, save, queries, load_reference


def summary(x):
    x = sorted(float(v) for v in x)
    def quantile(q):
        index = (len(x) - 1) * q; lower = int(index)
        return x[lower] + (x[min(lower + 1, len(x) - 1)] - x[lower]) * (index - lower)
    return {'n': len(x), 'mean': st.mean(x), 'median': st.median(x),
            'min': x[0], 'max': x[-1], 'p10': quantile(.1), 'p90': quantile(.9)}


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i]); out = [0.] * len(x); i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        for k in range(i, j): out[order[k]] = (i + j - 1) / 2
        i = j
    return out


def correlation(x, y):
    x = [v - st.mean(x) for v in x]; y = [v - st.mean(y) for v in y]
    denominator = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
    return sum(a*b for a, b in zip(x, y)) / denominator if denominator else None


def equal_suite(rows, key):
    suites = sorted({r['suite'] for r in rows})
    return st.mean(st.mean(r[key] for r in rows if r['suite'] == s) for s in suites)


def paired_summary(p, root, split):
    calibration_tasks = set()
    for c in p['calibration']['case_ids']:
        r=read(root/'results/baseline/calibration/queries'/f'{c}_0.json')['plan_row']
        calibration_tasks.add((r['suite'],r['task_id']))
    per_case = {}; records = []
    for case, seed in queries(p, split):
        baseline = read(root / 'results/baseline' / split / 'queries' / f'{case}_{seed}.json')
        candidate = read(root / 'results/anchor_kl' / split / 'queries' / f'{case}_{seed}.json')
        assert baseline['plan_row'] == candidate['plan_row']
        records.append((baseline, candidate))
        per_case.setdefault(case, []).append((baseline, candidate))
    rows = []
    for case, values in per_case.items():
        row = {'case_id': case, 'suite': values[0][0]['suite'], 'plan_row': values[0][0]['plan_row']}
        for metric in p['evaluation']['metrics']:
            row['base_' + metric] = st.mean(b[metric] for b, c in values)
            row['candidate_' + metric] = st.mean(c[metric] for b, c in values)
            row['delta_' + metric] = row['candidate_' + metric] - row['base_' + metric]
        rows.append(row)
    result = {'cases': len(rows), 'records': len(records), 'metrics': {}}
    for metric in p['evaluation']['metrics']:
        key = 'delta_' + metric
        values = [r[key] for r in rows]
        top = sorted(rows, key=lambda r: r[key])
        result['metrics'][metric] = {
            'baseline_case_distribution': summary([r['base_' + metric] for r in rows]),
            'candidate_case_distribution': summary([r['candidate_' + metric] for r in rows]),
            'paired_delta': summary(values),
            'leave_one_case_out_equal_suite_delta': summary([equal_suite([v for v in rows if v is not r], key) for r in rows]),
            'largest_improvements': [{k: r[k] for k in ('case_id', 'suite', 'base_' + metric, 'candidate_' + metric, key)} for r in top[:3]],
            'largest_regressions': [{k: r[k] for k in ('case_id', 'suite', 'base_' + metric, 'candidate_' + metric, key)} for r in top[-3:]],
            'exclude_two_largest_improvements_delta': equal_suite(top[2:], key),
        }
    motor = 'first10motor_rmse'; latent = 'future_latent_nmse'
    result['motor_latent_association'] = {
        'baseline_spearman': correlation(ranks([r['base_' + motor] for r in rows]), ranks([r['base_' + latent] for r in rows])),
        'candidate_spearman': correlation(ranks([r['candidate_' + motor] for r in rows]), ranks([r['candidate_' + latent] for r in rows])),
        'paired_delta_spearman': correlation(ranks([r['delta_' + motor] for r in rows]), ranks([r['delta_' + latent] for r in rows])),
        'quadrants': dict(Counter(('motor_better' if r['delta_' + motor] < 0 else 'motor_worse') + '/' +
                                 ('latent_better' if r['delta_' + latent] < 0 else 'latent_worse') for r in rows)),
        'scope': 'descriptive association on this fixed cohort; no causal or closed-loop interpretation',
    }
    result['action_aggregation'] = {}
    for arm, index in [('baseline', 0), ('candidate', 1)]:
        mse = st.mean(pair[index]['first10motor_mse'] for pair in records)
        result['action_aggregation'][arm] = {'mean_query_mse': mse, 'root_mean_query_mse': math.sqrt(mse),
                                            'mean_query_rmse': st.mean(pair[index][motor] for pair in records)}
    result['by_seed'] = {str(seed): {metric: st.mean(c[metric]-b[metric] for b,c in records if b['seed_index']==seed)
                                   for metric in p['evaluation']['metrics']} for seed in sorted({b['seed_index'] for b,c in records})}
    result['case_metadata'] = [{'case_id': r['case_id'], **r['plan_row']} for r in rows]
    result['calibration_task_groups'] = {}
    for name,seen in [('tasks_in_c8',True),('tasks_not_in_c8',False)]:
        subset=[r for r in rows if ((r['suite'],r['plan_row']['task_id']) in calibration_tasks)==seen]
        if not subset: continue
        result['calibration_task_groups'][name]={'cases':len(subset),'cases_by_suite':dict(Counter(r['suite'] for r in subset)),
            'metrics':{m:{'baseline':equal_suite(subset,'base_'+m),'candidate':equal_suite(subset,'candidate_'+m),
                'paired_delta':summary([r['delta_'+m] for r in subset]),'improved_cases':sum(r['delta_'+m]<0 for r in subset)}
                for m in p['evaluation']['metrics']}}
    return result


def latent_diagnostics(torch, z, target, prep, p):
    from anchor_loss import teacher_anchor_kl, latent_tokens, distances, neighbor_mask
    z = z.detach().float().requires_grad_(True); target = target.detach().float()
    lam = p['training']['rho'] * p['training']['normalization_c']; beta = prep['beta']
    mse = (z[:, :, 1:] - target[:, :, 1:]).square().sum() / target[:, :, 1:].square().sum()
    kl, info = teacher_anchor_kl(torch, z, target, prep['stats'], p['anchor'], prep['temperature'], diagnostics=True)
    gm = torch.autograd.grad(lam * mse, z, retain_graph=True)[0].flatten().double()
    gk = torch.autograd.grad(beta * kl, z)[0].flatten().double()
    norm_m, norm_k = float(gm.norm()), float(gk.norm())
    result = {'latent_nmse': float(mse.detach()), 'latent_anchor_kl': float(kl.detach()),
              'weighted_kl_to_nmse_loss': float(beta*kl.detach()/(lam*mse.detach())),
              'weighted_kl_to_nmse_future_gradient_norm': norm_k/norm_m,
              'future_gradient_cosine': float(torch.dot(gm,gk))/(norm_m*norm_k), **info}
    with torch.no_grad():
        u, ids = latent_tokens(torch, z, prep['stats'], p['anchor'])
        anchors, _ = latent_tokens(torch, target, prep['stats'], p['anchor'])
        mask = neighbor_mask(torch, ids); delta = u[0] - anchors[0]
        invisible, total, local_ranks = 0., float(delta.square().sum()), []
        visible_queries=anchors.clone()
        for i in range(len(ids)):
            selected = anchors[0, mask[i]].double()
            differences = selected[1:] - selected[0]
            _, singular, vh = torch.linalg.svd(differences, full_matrices=False)
            rank = int((singular > singular.max()*1e-7).sum()); local_ranks.append(rank)
            error = delta[i].double()
            projected = (error @ vh[:rank].T) @ vh[:rank]
            visible_queries[0,i]+=projected.float()
            invisible += float((error - projected).square().sum())
        result['pooled_standardized_error_in_exact_kl_nullspace_fraction'] = invisible/total
        result['local_anchor_rank'] = summary(local_ranks)
        standardized = (z[:, :, 1:] - target[:, :, 1:]) / torch.tensor(prep['stats']['latent']['scale']).view(1,-1,1,1,1)
        result['standardized_residual_energy_removed_by_pooling_fraction'] = 1 - 4 * total / float(standardized.square().sum())
        result['standardized_residual_energy_in_full_kl_nullspace_fraction'] = 1 - 4 * (total-invisible) / float(standardized.square().sum())
        from anchor_loss import _log_probs
        original_probs=_log_probs(torch,distances(u,anchors),mask,prep['temperature']).exp()
        projected_probs=_log_probs(torch,distances(visible_queries,anchors),mask,prep['temperature']).exp()
        invariant_error=float((original_probs-projected_probs).abs().max())
        if invariant_error>3e-6: raise RuntimeError('Analytic nullspace projection changed the probability distribution')
        result['nullspace_projection_probability_max_abs']=invariant_error
        d = distances(anchors, anchors)[0]
        spatial = mask & (ids[:, 0, None] == ids[None, :, 0])
        temporal = mask & ~spatial
        result['teacher_spatial_distance_median'] = float(d[spatial].median())
        result['teacher_temporal_distance_median'] = float(d[temporal].median())
        result['future_nmse_per_time'] = [float((z[:,:,t]-target[:,:,t]).square().sum()/target[:,:,t].square().sum()) for t in (1,2)]
        err_delta = (z[:,:,2]-z[:,:,1]) - (target[:,:,2]-target[:,:,1])
        result['future_temporal_difference_nmse'] = float(err_delta.square().sum()/(target[:,:,2]-target[:,:,1]).square().sum().clamp_min(1e-12))
    return result


def main():
    require_allocation()
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--output-name',default='diagnosis')
    args=parser.parse_args();p=read(args.protocol);root=Path(p['remote_root'])
    import torch
    torch.set_num_threads(4)
    prep=read(root/'results/prepare/summary.json')
    cache, _, _=load_reference(torch,p)
    result={'protocol':p['protocol'],'pbs_jobid':os.environ['PBS_JOBID'],'complete':False,
            'posthoc':True,'new_model_forwards':0,'parameter_gradients_computed':False,
            'gradient_scope':'partial derivatives w.r.t. saved future latent; not full parameter gradients',
            'paired':{split:paired_summary(p,root,split) for split in ('calibration','heldout')}}
    result['latent_diagnostics']={'initial':[],'final':[]}
    for case in p['calibration']['case_ids']:
        teacher=cache['teacher'][str(case)]['future']
        final=torch.load(root/'results/anchor_kl/tensors/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
        for phase, z in [('initial',cache['parent'][str(case)]['future']),('final',final['future'])]:
            record={'case_id':case,**latent_diagnostics(torch,z,teacher,prep,p)}
            result['latent_diagnostics'][phase].append(record)
        print('DIAGNOSED_C8',case,flush=True)
    result['latent_diagnostic_means']={phase:{key:summary([row[key] for row in rows]) for key in rows[0]
                                           if key!='case_id' and isinstance(rows[0][key],(int,float))}
                                       for phase,rows in result['latent_diagnostics'].items()}
    train=read(root/'results/anchor_kl/summary.json')['training_receipt']
    result['training']={'gradient_norm':summary([h['gradient_norm'] for h in train['history']]),
                        'clip_activated_steps':sum(h['gradient_norm']>1 for h in train['history']),
                        'history':train['history'],
                        'warning':'Each step uses a different case; raw step losses are not a learning curve.'}
    result['scaling']={'lambda_nmse':p['training']['rho']*p['training']['normalization_c'],
                       'beta':prep['beta'],'temperature':prep['temperature'],
                       'teacher_scale_range': [min(prep['stats']['latent']['scale']),max(prep['stats']['latent']['scale'])],
                       'std_floor_bound_channels':sum(s<=p['anchor']['std_floor'] for s in prep['stats']['latent']['scale'])}
    result['complete']=True
    save(root/'results'/args.output_name/'summary.json',result)
    print('DIAGNOSIS_COMPLETE new_model_forwards=0',flush=True)


if __name__=='__main__': main()

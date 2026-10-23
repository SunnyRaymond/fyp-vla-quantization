"""Offline C80 loss diagnostics from completed records; no model forward/training."""
import json
import math
import os
from pathlib import Path
import statistics

from analysis import guard


ARMS = ('mse_nmse', 'action_kl', 'latent_kl', 'dual_kl')


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return sum(values) / len(values)


def describe(values):
    x = sorted(float(v) for v in values)
    assert x and all(math.isfinite(v) for v in x)
    return {'n': len(x), 'min': x[0], 'p10': x[round(.1*(len(x)-1))],
            'median': statistics.median(x), 'mean': mean(x),
            'p90': x[round(.9*(len(x)-1))], 'max': x[-1]}


def correlation(x, y):
    dx, dy = [v-mean(x) for v in x], [v-mean(y) for v in y]
    denominator = math.sqrt(sum(v*v for v in dx)*sum(v*v for v in dy))
    return sum(a*b for a,b in zip(dx,dy))/denominator if denominator else None


def terms(record):
    return {'a': record.get('action_mse', record.get('first10motor_mse')),
            'z': record.get('latent_nmse', record.get('future_latent_nmse')),
            'ka': record.get('action_kl'), 'kz': record.get('latent_kl')}


def objective(v, arm, scale):
    a = scale['action_kl_coefficient']*v['ka'] if arm in ('action_kl','dual_kl') else v['a']
    z = scale['latent_kl_coefficient']*v['kz'] if arm in ('latent_kl','dual_kl') else v['z']
    return a+scale['original_latent_weight']*z


def optimizer_diagnostics(torch, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['updates'] == 80
    opt = saved['optimizer']
    result = {}
    for name, group in zip(('ratio','channel'), opt['param_groups']):
        rms, updates, steps = [], [], set()
        for identifier in group['params']:
            state = opt['state'][identifier]
            step = int(state['step']); steps.add(step)
            m = state['exp_avg'].float()/(1-group['betas'][0]**step)
            v = (state['exp_avg_sq'].float()/(1-group['betas'][1]**step)).sqrt()
            rms.append(v.reshape(-1))
            updates.append((group['lr']*m/(v+group['eps'])).reshape(-1))
        rms, delta = torch.cat(rms), torch.cat(updates)
        result[name] = {'lr': group['lr'], 'epsilon': group['eps'], 'state_steps': sorted(steps),
            'coordinates': rms.numel(), 'fraction_rms_below_epsilon': float((rms<group['eps']).float().mean()),
            'fraction_rms_below_10epsilon': float((rms<10*group['eps']).float().mean()),
            'rms_quantiles_p10_p50_p90': torch.quantile(rms, torch.tensor([.1,.5,.9])).tolist(),
            'last_unprojected_step_l2': float(delta.norm()),
            'last_unprojected_step_abs_quantiles_p50_p90_p99': torch.quantile(delta.abs(), torch.tensor([.5,.9,.99])).tolist()}
    scales = torch.cat([v.reshape(-1).float() for v in saved['shadow_log2_scales'].values()])
    ratios = torch.tensor([float(v) for v in saved['ratios'].values()])
    result['bounds'] = {'channel_at_lower': int((scales == -1).sum()), 'channel_at_upper': int((scales == 1).sum()),
                        'ratio_at_lower': int((ratios == .5).sum()), 'ratio_at_upper': int((ratios == 1).sum())}
    return result


def main():
    job = guard()
    import torch
    from softbin_loss import softbin_log_probs
    torch.set_num_threads(4)
    root = Path(os.environ['NEW_ROOT'])
    p = read(root/'protocol.json'); source = Path(p['source_per_channel_root'])
    scale = read(root/'results/prepare/loss_scaling.json')
    stats = read(root/'results/fit_stats/summary.json')
    initial = {v['case_id']: terms(v) for v in scale['initial_rows']}
    order = p['calibration']['update_order']
    assert len(order) == len(set(order)) == 80
    receipts = {'mse_nmse': read(source/'results/train_eval/training_receipt.json')}
    receipts.update({a: read(root/'results'/a/'training_receipt.json') for a in ARMS[1:]})
    records = {a: {c: read(root/'results/calibration'/a/'queries'/f'{c}_0.json') for c in order} for a in ARMS}
    for arm in ARMS:
        assert [h['case_id'] for h in receipts[arm]['history']] == order
        assert all(records[arm][c]['case_id'] == c and records[arm][c]['seed_index'] == 0 for c in order)
    result = {'complete': False, 'pbs_jobid': job, 'new_training_updates': 0, 'new_model_forwards': 0,
        'new_teacher_forwards': 0, 'uses_heldout': False, 'sampling_unit': 'original C80 case, one frozen seed; descriptive post-hoc diagnosis',
        'objective_cross_matrix': {}, 'arms': {}, 'case_rows': {}}
    for arm in ARMS:
        end = {c: terms(records[arm][c]) for c in order}
        result['objective_cross_matrix'][arm] = {loss: mean([objective(end[c],loss,scale) for c in order]) for loss in ARMS}
        history = receipts[arm]['history']; cases = []
        for index, h in enumerate(history):
            case = h['case_id']; final = objective(end[case],arm,scale)
            control = objective(terms(records['mse_nmse'][case]),arm,scale)
            before = objective(terms(h),arm,scale)
            assert math.isclose(before, h['loss'], rel_tol=2e-6, abs_tol=1e-9)
            cases.append({'case_id':case,'update':index+1,'suite':records[arm][case]['suite'],
                'initial':objective(initial[case],arm,scale),'pre_visit':before,'final':final,
                'control_final_same_objective':control,'final_minus_pre_visit':final-before,
                'final_minus_control':final-control,'gradient_norm':h['gradient_norm']})
        blocks = []
        for start in range(0,80,20):
            rows = cases[start:start+20]
            blocks.append({'updates':[start+1,start+20], **{name:mean([r[name] for r in rows]) for name in ('initial','pre_visit','final','control_final_same_objective','final_minus_pre_visit','final_minus_control')},
                           'final_worse_than_pre_visit':sum(r['final_minus_pre_visit']>0 for r in rows)})
        delta = [r['final_minus_control'] for r in cases]
        result['arms'][arm] = {'gradient_norm':describe([h['gradient_norm'] for h in history]),
            'gradient_clip_trigger_count':sum(h['gradient_norm']>p['training']['gradient_norm_clip'] for h in history),
            'final_vs_initial_pct':100*(mean([r['final'] for r in cases])/mean([r['initial'] for r in cases])-1),
            'final_vs_control_same_objective_pct':100*(mean([r['final'] for r in cases])/mean([r['control_final_same_objective'] for r in cases])-1),
            'final_vs_control_cases_improved':sum(d<0 for d in delta), 'final_vs_control_case_deltas':describe(delta),
            'final_vs_previsit_cases_worse':sum(r['final_minus_pre_visit']>0 for r in cases), 'blocks':blocks,
            'suite_final_minus_control':{s:mean([r['final_minus_control'] for r in cases if r['suite']==s]) for s in sorted({r['suite'] for r in cases})},
            'last_update_logged_train_vs_final_eval':{**cases[-1],'relative_change_pct':100*(cases[-1]['final']/cases[-1]['pre_visit']-1),
                'same_parameter_checkpointed_vs_no_grad_parity_verified':False},
            'largest_final_minus_control':sorted(cases,key=lambda r:r['final_minus_control'],reverse=True)[:5],
            'optimizer':optimizer_diagnostics(torch, source/'results/train_eval/checkpoint.pt' if arm=='mse_nmse' else root/'results'/arm/'checkpoint.pt')}
        result['case_rows'][arm] = cases
        action_errors = torch.tensor([records[arm][c]['action'] for c in order])[:,:10,:6]-torch.tensor([records[arm][c]['teacher_action'] for c in order])[:,:10,:6]
        result['arms'][arm]['action_axis_mse'] = action_errors.square().mean((0,1)).tolist()
        print('DIAG_ARM', arm, flush=True)
    energy, standardized_energy, latent_outside, action_outside, latent_far, counts = [], [], 0, 0, 0, [0,0]
    teacher_root = Path(p['control_root'])/'results/teacher/calibration'
    for c in order:
        target = torch.load(teacher_root/f'{c}_0.pt',map_location='cpu',weights_only=True)
        assert target['plan_row'] == records['mse_nmse'][c]['plan_row']
        z = target['future'][:,:,1:].double()
        a = target['action'][:10,:6].double()
        zs = (z-torch.tensor(stats['latent']['mean'],dtype=torch.float64).view(1,-1,1,1,1))/torch.tensor(stats['latent']['scale'],dtype=torch.float64).view(1,-1,1,1,1)
        az = (a-torch.tensor(stats['action']['mean'],dtype=torch.float64))/torch.tensor(stats['action']['scale'],dtype=torch.float64)
        energy.append(float(z.square().mean())); standardized_energy.append(float(zs.square().mean()))
        latent_outside += int((zs.abs()>3.5).sum()); latent_far += int((zs.abs()>4.625).sum()); action_outside += int((az.abs()>3.5).sum())
        counts[0] += zs.numel(); counts[1] += az.numel()
    result['normalization'] = {'teacher_energy_per_query':describe(energy),
        'teacher_energy_max_over_min':max(energy)/min(energy),
        'latent_teacher_fraction_outside_bins':latent_outside/counts[0],
        'latent_teacher_fraction_beyond_3sigma_past_edge':latent_far/counts[0],
        'action_teacher_fraction_outside_bins':action_outside/counts[1],
        'channel_inverse_variance_max_over_min':{name:(max(stats[name]['scale'])/min(stats[name]['scale']))**2 for name in ('action','latent')},
        'action_inverse_variance': [1/s**2 for s in stats['action']['scale']],
        'initial_latent_kl_over_nmse':describe([initial[c]['kz']/initial[c]['z'] for c in order]),
        'initial_energy_vs_latent_kl_over_nmse_log_pearson':correlation([math.log(v) for v in energy],[math.log(initial[c]['kz']/initial[c]['z']) for c in order]),
        'teacher_energy_by_case':dict(zip(order,energy))}
    x = torch.tensor([0,.25,2.5,3,3.5,4,4.5,5,6],requires_grad=True)
    logp = softbin_log_probs(torch,x,torch.linspace(-3.5,3.5,15),.375)
    score = torch.stack([torch.autograd.grad(logp[:,k].sum(),x,retain_graph=True)[0] for k in range(16)],-1)
    fisher = (logp.detach().exp()*score.square()).sum(-1)
    result['softbin_fisher_information'] = [{'standardized_value':float(v),'fisher':float(i),'fraction_of_unbinned_gaussian':float(i*.375**2)} for v,i in zip(x.detach(),fisher.detach())]
    result['limits'] = ['One trajectory per arm; descriptive diagnostics only.',
        'Pre-visit losses use different intermediate checkpoints; their mean is not a full-C80 training curve.',
        'Update80 compares its logged pre-update checkpointed training loss with final no-grad evaluation on the same case/seed. Same-parameter training-vs-evaluation parity at final80 was not checked; do not assign the entire gap to the update without that check.',
        'Final-minus-previsit otherwise includes own and all later updates; these differences do not establish catastrophic forgetting or full-C80 divergence.',
        'Final Adam moments do not identify per-component gradient conflicts or per-step stability.',
        'No model or teacher forward, no training, and no held-out fitting.']
    result['complete'] = True
    out = root/'results/c80_diagnosis'; out.mkdir(exist_ok=True)
    payload = json.dumps(result,allow_nan=False,separators=(',',':'))
    assert len(payload.encode()) < 240*1024
    (out/'summary.json').write_text(payload+'\n')
    print('C80_DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

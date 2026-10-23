"""Read-only C8 checkpoint/loss-gradient diagnosis; never performs an optimizer step."""
import json
import math
import os
from pathlib import Path
import time

from relational_selfcheck import require_allocation


KEYS = ('action_kl', 'action_temporal_kl', 'latent_kl',
        'latent_spatiotemporal_kl', 'action_mse', 'latent_nmse')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n'
    if path.name=='summary.json':assert len(payload.encode())<240*1024
    path.write_text(payload)


def matrix(torch, vectors):
    gram = torch.stack([torch.stack([torch.dot(a.double(), b.double()) for b in vectors])
                        for a in vectors])
    norm = gram.diag().clamp_min(0).sqrt()
    cosine = gram / (norm[:, None] * norm[None, :]).clamp_min(1e-30)
    return {'norm': norm.tolist(), 'gram': gram.tolist(), 'cosine': cosine.tolist()}


def quantiles(torch, x):
    return torch.quantile(x.float().reshape(-1), torch.tensor([.1, .5, .9, .99])).tolist()


def flat(torch, gradients, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).float().reshape(-1)
                      for g, p in zip(gradients, parameters)]).cpu()


def norm_summary(torch, vectors, book_mask):
    kl = sum(vectors[:4]); mse = sum(vectors[4:])
    ks, ms = float(kl.norm()), float(mse.norm())
    result = {'keys': list(KEYS), 'weighted_components': matrix(torch, vectors),
              'books': matrix(torch, [g[book_mask] for g in vectors]),
              'log_gains': matrix(torch, [g[~book_mask] for g in vectors]),
              'kl_total_norm': ks, 'mse_total_norm': ms,
              'kl_vs_mse_cosine': float(torch.dot(kl.double(), mse.double()) / max(ks*ms, 1e-30)),
              'kl_cancellation_ratio': ks / max(sum(float(g.norm()) for g in vectors[:4]), 1e-30)}
    result['component_dot_kl_total'] = [float(torch.dot(g.double(), kl.double())) for g in vectors]
    return result


def optimizer_state(torch, checkpoint):
    opt = checkpoint.get('optimizer', checkpoint.get('optimizer_state'))
    group = opt['param_groups'][0]
    result = {'lr': group['lr'], 'eps': group['eps'], 'betas': group['betas'], 'groups': {}}
    direction = []
    for parity, name in enumerate(('books', 'log_gains')):
        rms, delta, steps = [], [], set()
        for identifier in group['params'][parity::2]:
            state = opt['state'][identifier]; step = int(state['step']); steps.add(step)
            m = state['exp_avg'].float() / (1-group['betas'][0]**step)
            v = (state['exp_avg_sq'].float() / (1-group['betas'][1]**step)).sqrt()
            rms.append(v.reshape(-1)); delta.append((-group['lr']*m/(v+group['eps'])).reshape(-1))
        rms, delta = torch.cat(rms), torch.cat(delta)
        result['groups'][name] = {'state_steps': sorted(steps), 'coordinates': rms.numel(),
            'rms_quantiles_p10_p50_p90_p99': quantiles(torch, rms),
            'fraction_rms_below_epsilon': float((rms<group['eps']).float().mean()),
            'fraction_rms_below_10epsilon': float((rms<10*group['eps']).float().mean()),
            'last_adam_direction_abs_quantiles': quantiles(torch, delta.abs()),
            'last_adam_direction_l2': float(delta.norm())}
    # Parameter order is interleaved books/log-gains, matching TrainableBanks.parameters.
    for identifier in group['params']:
        state = opt['state'][identifier]; step = int(state['step'])
        m = state['exp_avg'].float()/(1-group['betas'][0]**step)
        v = (state['exp_avg_sq'].float()/(1-group['betas'][1]**step)).sqrt()
        direction.append((-group['lr']*m/(v+group['eps'])).reshape(-1))
    return result, torch.cat(direction)


def main():
    job, host = require_allocation()
    import numpy as np
    import torch
    import runner
    import sampler
    import calibration_source
    import clipping_reference as ref
    import relational_reference as prior
    from train_banks import TrainableBanks
    from softbin_loss import losses
    from relational_loss import relational_losses

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    fixture=[torch.tensor([1.,2.],requires_grad=True),torch.tensor([3.],requires_grad=True)]
    f1=fixture[0].square().sum();f2=fixture[0].sum()+fixture[1].sum()
    g1=flat(torch,torch.autograd.grad(f1,fixture,allow_unused=True,retain_graph=True),fixture)
    g2=flat(torch,torch.autograd.grad(f2,fixture,retain_graph=True),fixture)
    gt=flat(torch,torch.autograd.grad(f1+f2,fixture),fixture)
    assert torch.equal(g1+g2,gt)
    assert matrix(torch,[torch.tensor([1.,0.]),torch.tensor([0.,2.])])['cosine']==[[1.,0.],[0.,1.]]
    root = Path(os.environ['NEW_ROOT']); p = read(root/'protocol.json')
    out = root/'results/vq_diagnosis'; out.mkdir(parents=True, exist_ok=True)
    cases = p['vq_training']['case_ids']
    plan_path = Path(p['calibration']['plan_file']); plan = read(plan_path)
    rows = {x['case_id']:x for x in plan['inputs']}
    assert cases == plan['calibration_cohorts']['c8'] and len(cases)==8
    stats = read(Path(p['source_dual_root'])/'results/fit_stats/summary.json')
    scale = read(Path(p['source_dual_root'])/'results/prepare/loss_scaling.json')
    prep = read(Path(p['source_relational_root'])/'results/prepare/summary.json')
    config, coef = prep['relational_config'], prep['coefficients']
    weight = p['training']['rho']*p['training']['normalization_c']
    weights = [scale['action_kl_coefficient'], scale['action_kl_coefficient']*coef['action_beta'],
               weight*scale['latent_kl_coefficient'], weight*scale['latent_kl_coefficient']*coef['latent_beta'], 1., weight]
    raw = torch.load(Path(p['baseline_root'])/'results/reference/reference_cache.pt',map_location='cpu',weights_only=True)
    teachers = {}
    for case in cases:
        target = torch.load(Path(p['control_root'])/'results/teacher/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
        assert target['plan_row']==rows[case] and target['sampler_seed']==rows[case]['sampler_seeds'][0]
        assert torch.equal(target['action'],raw['teacher'][str(case)]['action'])
        assert torch.equal(target['future'],raw['teacher'][str(case)]['future'])
        teachers[case] = target

    def terms(action,future,target):
        return prior._terms(torch,losses,relational_losses,action,future,target,stats,p,config)

    def values(v):
        soft,total = prior._objectives(v,p,scale,coef)
        return {**{k:float(x.detach()) for k,x in v.items()},'softbin_objective':float(soft.detach()),
                'relational_objective':float(total.detach()),'mse_objective':float((v['action_mse']+weight*v['latent_nmse']).detach()),
                'first10motor_rmse':math.sqrt(float(v['action_mse'].detach()))}

    result = {'protocol':p['protocol'],'pbs_jobid':job,'hostname':host,'complete':False,
        'new_training_updates':0,'optimizer_step_calls':0,'new_teacher_queries':0,'uses_heldout':False,
        'case_ids':cases,'seed_indices':[0],'weights':dict(zip(KEYS,weights)),
        'gradient_scope':'exact six weighted component VJPs through deployed sampler STE on all original C8 cases',
        'checkpoints':{},'original_training_history':read(root/'results/vq_kl/training_receipt.json')['vq_history']}
    deadline = ref.Deadline()
    runtime = runner.Runtime(out); runtime.model.eval(); runtime.arm('bf16')
    banks = TrainableBanks(runtime,p['calibration']['bank_directory'])
    initial = banks.snapshot(); parameters = banks.parameters()
    assert not any(v.requires_grad for v in runtime.model.parameters())
    book_mask = torch.cat([torch.full((q.numel(),),i%2==0,dtype=torch.bool) for i,q in enumerate(parameters)])
    identities = {n:{k:(e[k].data_ptr(),e[k]._version) for k in ('indices','scale_exponents')} for n,e in banks.encodings.items()}
    data = {}
    def datum(case):
        if case not in data:
            obs,meta = calibration_source.load_observation(np,plan_path.parent,rows[case])
            data[case] = runtime.datum(obs),meta
        return data[case]

    mse_path = Path(p['baseline_root'])/'results/training/rho1.5'
    sources = {'raw':None,'mse8':mse_path/'deployment.pt','kl8':root/'results/vq_kl/deployment.pt'}
    for label,path in sources.items():
        deadline.check(300)
        saved = torch.load(path,map_location='cpu',weights_only=True) if path else None
        if saved: assert saved['updates']==8
        banks.restore(initial if saved is None else saved['module_masters'])
        point = {'cases':[], 'new_model_queries':0, 'source':str(path) if path else p['calibration']['bank_directory']}
        if label!='raw':
            cp_path = mse_path/'checkpoint.pt' if label=='mse8' else root/'results/vq_kl/checkpoint.pt'
            checkpoint = torch.load(cp_path,map_location='cpu',weights_only=True)
            if label=='kl8':
                assert checkpoint['updates']==saved['updates']==8 and checkpoint['unit']=='vq_kl'
                assert checkpoint['protocol']==saved['protocol']==p
            else:
                assert checkpoint['run_state']['updates_completed']==saved['updates']==8
                assert checkpoint['contract']['binding']['rho']==saved['rho']==1.5
                assert checkpoint['contract']['case_ids']==saved['case_ids']==cases
            assert set(checkpoint['module_masters'])==set(saved['module_masters'])
            assert all(torch.equal(checkpoint['module_masters'][n][k],saved['module_masters'][n][k])
                       for n in saved['module_masters'] for k in ('books','log_scales'))
            point['checkpoint_deployment_masters_equal']=True
            point['optimizer'],last_direction = optimizer_state(torch,checkpoint)
            if label=='mse8':point['original_training_history']=checkpoint['run_state']['training_history']
            del checkpoint
        else: last_direction=None
        sums = [torch.zeros(int(book_mask.numel())) for _ in KEYS]
        kl_case_grads, mse_case_grads = [], []
        move = {}
        for key in ('books','log_scales'):
            changes=[]; deployed=[]; fraction=[]
            for name in banks.modules:
                x = (initial if saved is None else saved['module_masters'])[name][key].float()
                base=initial[name][key].float();changes.append((x-base).reshape(-1))
                b = x if key=='books' else x.exp(); b0 = base if key=='books' else base.exp()
                deployed.append((b.to(torch.bfloat16)!=b0.to(torch.bfloat16)).reshape(-1))
            delta=torch.cat(changes)
            move[key]={'master_delta_l2':float(delta.norm()),'master_delta_abs_quantiles':quantiles(torch,delta.abs()),
                       'stored_bf16_coordinates_changed_fraction':float(torch.cat(deployed).float().mean())}
        point['parameter_change_from_raw']=move
        for case in cases:
            deadline.check(300); started=time.monotonic()
            value,meta=datum(case);runtime.description=meta['description'];banks.seen.clear()
            for q in parameters:q.grad=None
            action,counts,future=sampler.differentiable_action(runtime,value,rows[case]['sampler_seeds'][0],checkpointed=True,return_future=True)
            assert len(banks.seen)==614
            v=terms(action,future,teachers[case]);record={'case_id':case,'suite':rows[case]['suite'],'metrics':values(v),**counts}
            if label in ('raw','kl8'):
                witness = raw['parent'][str(case)] if label=='raw' else torch.load(root/'results/vq_kl/tensors/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
                record['cached_output_parity']={k:float((a.detach().float().cpu()-witness[k].float()).abs().max()) for k,a in [('action',action),('future',future)]}
                assert max(record['cached_output_parity'].values())==0
            else:
                witness=read(Path(p['control_root'])/'results/calibration/baseline/queries'/f'{case}_0.json')
                record['cached_metric_parity']={k:abs(record['metrics'][k]-witness[old])
                    for k,old in [('first10motor_rmse','first10motor_rmse'),('latent_nmse','future_latent_nmse')]}
                assert all(math.isclose(record['metrics'][k],witness[old],rel_tol=2e-6,abs_tol=1e-8)
                    for k,old in [('first10motor_rmse','first10motor_rmse'),('latent_nmse','future_latent_nmse')])
            # Output-space norms use independent detached leaves: action and future are
            # causally coupled in the sampler, so differentiating both original outputs
            # and then treating them as independent would double-count the action path.
            leaves=[action.detach().float().requires_grad_(),future.detach().float().requires_grad_()]
            local=terms(*leaves,teachers[case]);output_vectors=[]
            for key,w in zip(KEYS,weights):
                gs=torch.autograd.grad(w*local[key],leaves,allow_unused=True,retain_graph=True)
                output_vectors.append(flat(torch,gs,leaves))
            record['output_space']=matrix(torch,output_vectors)
            del local,leaves,output_vectors
            gradients=[];coverage=[]
            for index,(key,w) in enumerate(zip(KEYS,weights)):
                gs=torch.autograd.grad(w*v[key],parameters,allow_unused=True,retain_graph=index<len(KEYS)-1)
                assert all(bool(torch.isfinite(g).all()) for g in gs if g is not None)
                g=flat(torch,gs,parameters);gradients.append(g);sums[index].add_(g)
                coverage.append({'connected_parameter_tensors':sum(g is not None for g in gs),
                                 'nonzero_coordinates':int(torch.count_nonzero(g)), 'coordinates':g.numel()})
                del gs
            record['gradient_coverage']=coverage
            record['parameter_space']=norm_summary(torch,gradients,book_mask)
            kl_case_grads.append(sum(gradients[:4]));mse_case_grads.append(sum(gradients[4:]))
            if last_direction is not None:
                record['last_adam_direction_dot_current_kl_gradient']=float(torch.dot(last_direction.double(),kl_case_grads[-1].double()))
            torch.cuda.synchronize();record['seconds']=time.monotonic()-started
            point['cases'].append(record);point['new_model_queries']+=1
            result['checkpoints'][label]=point
            save(out/'partial.json',result)
            print('VQ_DIAG_PROGRESS '+json.dumps({'checkpoint':label,'case_id':case,'completed_cases':len(point['cases']),'seconds':record['seconds']}),flush=True)
            del action,future,v,gradients,witness
        means=[g/len(cases) for g in sums]
        point['mean_case_gradient']=norm_summary(torch,means,book_mask)
        point['case_kl_gradient_cosines']=matrix(torch,kl_case_grads)
        point['case_mse_gradient_cosines']=matrix(torch,mse_case_grads)
        point['c8_mean_metrics']={key:sum(c['metrics'][key] for c in point['cases'])/len(cases) for key in point['cases'][0]['metrics']}
        if last_direction is not None:
            point['last_adam_direction_dot_c8_mean_kl_gradient']=float(torch.dot(last_direction.double(),sum(means[:4]).double()))
        assert all((banks.encodings[n][k].data_ptr(),banks.encodings[n][k]._version)==identity
                   for n,keys in identities.items() for k,identity in keys.items())
        result['checkpoints'][label]=point
        save(out/'partial.json',result)
        del saved,sums,means,kl_case_grads,mse_case_grads,last_direction
    result.update(complete=True,new_model_queries=sum(x['new_model_queries'] for x in result['checkpoints'].values()),
                  new_parameter_space_component_vjps=24*len(KEYS),new_output_space_component_vjps=24*len(KEYS),
                  all_original_checkpoints_read_only=True,
                  fixed_indices_exponents_unchanged=True,
                  limitations=['Only initial/final checkpoints exist; eight pre-visit losses are not a convergence curve.',
                               'STE gradients are surrogate derivatives of a BF16/fake-A4 path.',
                               'A nonzero final gradient alone does not establish that additional updates improve loss.',
                               'Adam direction dot products are local diagnostics, not executed updates or line searches.'])
    save(out/'summary.json',result)
    print('VQ_DIAGNOSIS_COMPLETE',flush=True)


if __name__=='__main__':main()

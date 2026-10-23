"""Full-C80 parity and counterfactual Adam directions; never save trained weights."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False, separators=(',',':'))+'\n')
    tmp.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().split()
    if not os.environ.get('PBS_JOBID') or 'login' in host or host not in {n.split('.')[0] for n in nodes}:
        raise RuntimeError('Real PBS compute allocation required')


def selected_terms(values, loss, scale):
    action = values['action_kl']*scale['action_kl_coefficient'] if loss in ('action_kl','dual_kl') else values['action_mse']
    latent = values['latent_kl']*scale['latent_kl_coefficient'] if loss in ('latent_kl','dual_kl') else values['latent_nmse']
    return action, latent*scale['original_latent_weight']


def main():
    guard()
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol',required=True); parser.add_argument('--unit',required=True)
    args = parser.parse_args(); p = read(args.protocol); unit = p['units'][args.unit]
    import torch
    import numpy as np
    import runner
    import sampler
    import calibration_source
    import clipping_reference as ref
    from channel_banks import ChannelBanks
    from checkpointing import atomic_torch_save
    from softbin_loss import losses
    from direction_utils import adam_delta, gradient_summary

    torch.set_num_threads(8); torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = Path(p['source_root']); root = Path(p['remote_root']); out = root/'results'/args.unit
    out.mkdir(parents=True,exist_ok=True)
    old = read(source/'protocol.json'); scale = read(source/'results/prepare/loss_scaling.json'); stats = read(source/'results/fit_stats/summary.json')
    assert old['calibration']['case_ids'] == p['case_ids'] and not scale['heldout_used'] and not stats['heldout_used']
    source_dir = Path(old['source_per_channel_root'])/'results/train_eval' if unit['checkpoint_arm']=='mse_nmse' else source/'results'/unit['checkpoint_arm']
    saved = torch.load(source_dir/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert saved['updates']==80 and [h['case_id'] for h in saved['history']]==old['calibration']['update_order']
    state = {'protocol':p['protocol'],'unit':args.unit,'pbs_jobid':os.environ['PBS_JOBID'],
             'phase':'loading','complete':False,'new_model_queries':0,'new_teacher_queries':0,'training_updates':0}
    def progress(**kw):
        state.update(kw); save(out/'progress.json',state)
        print('DIRECTION_PROGRESS '+json.dumps(state),flush=True)
    progress(); started = time.monotonic(); deadline = ref.Deadline(); runtime = None
    try:
        data_root, rows, cache, deployment, reference_action = ref.load_inputs(old['source_clipping_protocol'],torch)
        runtime = runner.Runtime(out); runtime.model.eval(); runtime.arm('bf16')
        banks = ChannelBanks(runtime,old['calibration']['bank_directory'],deployment['module_masters'],old['clipping'],old['scaling'])
        banks.restore_clips(saved['ratios']); banks.restore_scales(saved['shadow_log2_scales'])
        groups = banks.optimizer_groups(old['training'],old['scaling'])
        for group,name in zip(groups,('ratio','channel')): group['name']=name
        parameters = [v for group in groups for v in group['params']]
        originals = [v.detach().clone() for v in parameters]
        data = {}
        def datum(case):
            if case not in data:
                obs,meta = calibration_source.load_observation(np,data_root,rows[case]); data[case] = runtime.datum(obs),meta
            return data[case]
        def target(case):
            value = torch.load(Path(old['control_root'])/'results/teacher/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
            assert value['plan_row']==rows[case] and value['sampler_seed']==rows[case]['sampler_seeds'][0]
            return value
        def terms(action,future,teacher):
            return losses(torch,action,future,teacher['action'].to(action.device),teacher['future'].to(future.device),stats,old['softbin'])
        def query(case):
            deadline.check(); value,meta=datum(case)
            result=ref.query(runtime,sampler,torch,value,meta['description'],rows[case]['sampler_seeds'][0],banks)
            state['new_model_queries']+=1
            return result[0],result[1]
        def grad_forward(case):
            deadline.check(); value,meta=datum(case); runtime.description=meta['description']; banks.seen.clear()
            action,counts,future=sampler.differentiable_action(runtime,value,rows[case]['sampler_seeds'][0],checkpointed=True,return_future=True)
            state['new_model_queries']+=1
            assert len(banks.seen)==614 and counts=={'video_steps':10,'action_steps':10}
            return action,future
        def row_values(case,action,future):
            values=terms(action,future,target(case)); a,z=selected_terms(values,unit['loss'],scale)
            return {'case_id':case,'seed_index':0,'suite':rows[case]['suite'],'sampler_seed':rows[case]['sampler_seeds'][0],
                    **{k:float(v.detach()) for k,v in values.items()},'objective':float((a+z).detach())}
        def restore():
            with torch.no_grad():
                for parameter,value in zip(parameters,originals): parameter.copy_(value)
        def set_delta(delta,fraction):
            actual=[]; index=0
            with torch.no_grad():
                for group in groups:
                    lower,upper = (.5,1.) if group['name']=='ratio' else (-1.,1.)
                    for parameter in group['params']:
                        parameter.copy_((originals[index]+fraction*delta[index]).clamp(lower,upper))
                        actual.append(parameter-originals[index]); index+=1
            return actual

        grad_file=out/'gradient_state.pt'; base=[]; parity=[]
        full=[torch.zeros_like(v) for v in parameters]
        if grad_file.is_file():
            prior=torch.load(grad_file,map_location='cpu',weights_only=True)
            assert prior['protocol']==p and prior['unit']==args.unit
            base,parity=prior['baseline'],prior['parity']
            assert [r['case_id'] for r in base]==p['case_ids'][:len(base)]
            full=[v.to('cuda') for v in prior['full_gradient']]
            state['new_model_queries']=prior['logical_queries']
        progress(phase='parity_gradient',cases=len(base))
        for case in p['case_ids'][len(base):]:
            deadline.check(120)
            action,future=query(case); baseline=row_values(case,action,future)
            reference=read(source/'results/calibration'/unit['checkpoint_arm']/'queries'/f'{case}_0.json')
            for a,b in (('action_mse','first10motor_mse'),('latent_nmse','future_latent_nmse'),('action_kl','action_kl'),('latent_kl','latent_kl')):
                if not math.isclose(baseline[a],reference[b],rel_tol=1e-6,abs_tol=1e-9):
                    raise RuntimeError('Source endpoint replay differs: '+str(case)+'/'+a)
            for parameter in parameters: parameter.grad=None
            ga,gz=grad_forward(case)
            error={'case_id':case,'action_max_abs':float((ga.detach().float().cpu()-action.float()).abs().max()),
                   'future_max_abs':float((gz.detach().float().cpu()-future.float()).abs().max())}
            save(out/'parity'/f'{case}.json',error)
            if error['action_max_abs']>p['parity_max_abs'] or error['future_max_abs']>p['parity_max_abs']:
                save(out/'parity_failure.json',error); raise RuntimeError('Same-state checkpointed/no-grad parity failed: '+str(error))
            values=terms(ga,gz,target(case)); a,z=selected_terms(values,unit['loss'],scale); (a+z).backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,float('inf'),error_if_nonfinite=True)
            assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in parameters)
            for total,parameter in zip(full,parameters): total.add_(parameter.grad.detach(),alpha=1/80)
            baseline['per_query_surrogate_gradient_norm']=float(norm)
            base.append(baseline); parity.append(error)
            del ga,gz,values,a,z,action,future
            for parameter in parameters: parameter.grad=None
            if len(base)%5==0 or len(base)==80:
                atomic_torch_save(torch,grad_file,{'protocol':p,'unit':args.unit,'baseline':base,'parity':parity,
                    'full_gradient':[v.cpu() for v in full],'logical_queries':state['new_model_queries']})
            progress(phase='parity_gradient',cases=len(base),last_case=case)

        component_file=out/'component_summary.json'
        if component_file.is_file():
            components=read(component_file)
        else:
            components=[]
            for case in p['component_cases']:
                deadline.check(180); ga,gz=grad_forward(case); values=terms(ga,gz,target(case))
                a,z=selected_terms(values,unit['loss'],scale)
                a_grad=torch.autograd.grad(a,parameters,retain_graph=True,allow_unused=True)
                z_grad=torch.autograd.grad(z,parameters,allow_unused=True)
                components.append({'case_id':case,'summary':gradient_summary(torch,a_grad,z_grad,groups)})
                del ga,gz,values,a,z,a_grad,z_grad
                progress(phase='component_gradients',cases=len(components),last_case=case)
            save(component_file,components)

        comparison=[]
        for direction in p['directions']:
            delta=adam_delta(torch,full,groups,saved['optimizer'] if direction=='saved_adam' else None)
            for fraction in p['fractions']:
                label=direction+'_f'+str(fraction).replace('.','p')
                actual=set_delta(delta,fraction)
                predicted=float(sum((g*d).double().sum() for g,d in zip(full,actual)))
                step_l2=float(sum(d.double().square().sum() for d in actual).sqrt())
                evaluated=[]
                try:
                    for case in p['case_ids']:
                        deadline.check(90); path=out/'directions'/label/f'{case}_0.json'
                        if path.is_file():
                            record=read(path)
                            assert record['protocol']==p['protocol'] and record['unit']==args.unit and record['case_id']==case
                        else:
                            action,future=query(case); record={'protocol':p['protocol'],'unit':args.unit,'direction':direction,'fraction':fraction,
                                'pbs_jobid':os.environ['PBS_JOBID'],**row_values(case,action,future)}
                            save(path,record); del action,future
                        evaluated.append(record)
                        progress(phase='direction_evaluation',direction=direction,fraction=fraction,cases=len(evaluated),last_case=case)
                finally:
                    restore()
                differences=[v['objective']-b['objective'] for v,b in zip(evaluated,base)]
                comparison.append({'direction':direction,'fraction':fraction,'queries':80,'surrogate_linear_prediction':predicted,'projected_step_l2':step_l2,
                    'baseline_objective':sum(v['objective'] for v in base)/80,'objective':sum(v['objective'] for v in evaluated)/80,
                    'mean_delta':sum(differences)/80,'improved_cases':sum(v<0 for v in differences),'worsened_cases':sum(v>0 for v in differences),
                    'by_suite':{suite:sum(d for d,b in zip(differences,base) if b['suite']==suite)/20 for suite in sorted({b['suite'] for b in base})},
                    'metrics':{key:sum(v[key] for v in evaluated)/80 for key in ('action_mse','latent_nmse','action_kl','latent_kl')}})
                save(out/'direction_summary.json',comparison)
            del delta
        restore()
        assert all(torch.equal(v.detach(),original) for v,original in zip(parameters,originals))
        summary={**state,'complete':True,'phase':'complete','checkpoint_arm':unit['checkpoint_arm'],'loss':unit['loss'],
            'calibration_queries':80,'parity_cases':80,'parity_action_max_abs':max(v['action_max_abs'] for v in parity),
            'parity_future_max_abs':max(v['future_max_abs'] for v in parity),'gradient_query_count':80,
            'full_gradient_norm':float(sum(v.double().square().sum() for v in full).sqrt()),
            'full_gradient_construction':'mean of 80 unclipped per-query surrogate gradients, accumulated in FP32 after each unscaled backward',
            'component_gradient_case_count':len(components),'component_gradients':components,
            'direction_conditions':comparison,'baseline_metrics':{k:sum(v[k] for v in base)/80 for k in ('action_mse','latent_nmse','action_kl','latent_kl','objective')},
            'source_optimizer_steps':sorted({int(v['step']) for v in saved['optimizer']['state'].values()}),
            'original_parameters_restored':True,'saved_trained_checkpoints':0,'training_updates':0,
            'heldout_used':False,'seconds_inside_worker':time.monotonic()-started}
        save(out/'summary.json',summary); progress(phase='complete',complete=True)
        print('KL_DIRECTION_COMPLETE',flush=True)
    except TimeoutError as exc:
        progress(phase='checkpointed',reason=str(exc)); return 75
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc)); raise
    finally:
        if runtime is not None and runtime.env is not None: runtime.env.close()
    return 0


if __name__=='__main__': raise SystemExit(main())

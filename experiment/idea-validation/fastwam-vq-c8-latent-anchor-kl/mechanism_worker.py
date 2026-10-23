"""C8 parameter-gradient, reversible-step and frozen-reader mechanism probes."""
import copy
import gc
import os
from pathlib import Path
import statistics as st
import time

from anchor_selfcheck import require_allocation
from anchor_loss import loss_terms,teacher_anchor_kl
from prepare import read,save,load_reference
from worker import Deadline
from mechanism_math import (TERMS,alignment,group_alignment,dot,mix,adam_delta,
                            case_geometry,visible_projection,matched_repairs,selfcheck)


def run():
    require_allocation()
    import numpy as np
    import torch
    import sampler
    import runner
    import calibration_source
    from train_banks import TrainableBanks
    from checkpointing import atomic_torch_save

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    p=read(Path(__file__).with_name('protocol.json'))
    config=read(Path(__file__).with_name('mechanism_protocol.json'))
    root=Path(p['remote_root']);out=root/'results/mechanism';out.mkdir(parents=True,exist_ok=True)
    prep=read(root/'results/prepare/summary.json')
    if (config['parent_protocol']!=p['protocol'] or config['case_ids']!=p['calibration']['case_ids']
            or config['heldout_used'] or config['training_updates']!=0):
        raise ValueError('Mechanism scope differs from the original C8 contract')
    contract=out/'contract.json'
    if contract.is_file() and read(contract)!=config:raise ValueError('Resume mechanism protocol differs')
    save(contract,config)
    save(out/'selfcheck.json',selfcheck(torch,prep,p['anchor']))
    cache,saved_rows,rows=load_reference(torch,p)
    checkpoint=torch.load(root/'results/anchor_kl/checkpoint.pt',map_location='cpu',weights_only=True)
    if checkpoint['protocol']!=p or checkpoint['updates']!=8:raise ValueError('KL checkpoint binding differs')
    endpoint_receipt=read(root/'results/anchor_kl/summary.json')
    if not endpoint_receipt['complete'] or endpoint_receipt['updates']!=8:raise ValueError('KL parent is incomplete')
    state=read(out/'progress.json') if (out/'progress.json').is_file() else {}
    state.update(protocol=config['protocol'],unit='mechanism',pbs_jobid=os.environ['PBS_JOBID'],complete=False,
        training_updates=0,heldout_used=False,new_baseline_model_queries=0,new_teacher_full_queries=0,
        new_closedloop_episodes=0)
    for name in ('new_student_full_queries','new_reader_queries','new_parameter_reverse_passes','new_reader_reverse_passes'):
        state.setdefault(name,0)
    deadline=Deadline();runtime=None;data={};summaries={}

    def progress(stage,**values):
        state.update(stage=stage,**values);save(out/'progress.json',state)
        print('MECHANISM_PROGRESS',stage,values,flush=True)

    def count(name,stage):
        state[name]+=1;progress(stage,query_counter=state[name])

    def datum(case):
        if case not in data:
            obs,meta=calibration_source.load_observation(np,Path(p['calibration']['plan_file']).parent,rows[case])
            if meta!=saved_rows[case]['metadata']:raise ValueError('Original C8 input metadata differs')
            data[case]=runtime.datum(obs),meta
        value,meta=data[case];runtime.description=meta['description']
        return value

    def target(case):return cache['teacher'][str(case)]

    def endpoint(point,case):
        if point=='initial':return cache['parent'][str(case)]
        value=torch.load(root/'results/anchor_kl/tensors/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
        if value['plan_row']!=rows[case] or value['updates']!=8:raise ValueError('Saved endpoint differs')
        return value

    def weighted(values):
        return {'action':values['action_mse'],
                'weighted_nmse':values['latent_nmse']*p['training']['rho']*p['training']['normalization_c'],
                'weighted_kl':values['latent_anchor_kl']*prep['beta']}

    def metrics(action,future,case):
        t=target(case)
        values=loss_terms(torch,action,future,t,prep['stats'],p['anchor'],prep['temperature'])
        v={k:float(v.detach()) for k,v in values.items()}
        v['base_objective']=v['action_mse']+p['training']['rho']*p['training']['normalization_c']*v['latent_nmse']
        v['original_total_objective']=v['base_objective']+prep['beta']*v['latent_anchor_kl']
        error=action.float()-t['action'].float()
        v.update(first10motor_rmse=float(error[:10,:6].square().mean().sqrt()),
                 full32motor_rmse=float(error[:,:6].square().mean().sqrt()),
                 late22motor_rmse=float(error[10:,:6].square().mean().sqrt()),
                 first10gripper_rmse=float(error[:10,6].square().mean().sqrt()))
        return v

    def binding(point,case,label):
        return {'protocol':config['protocol'],'point':point,'case_id':case,'label':label,
                'sampler_seed':rows[case]['sampler_seeds'][0]}

    def load_bound(path,expected):
        value=torch.load(path,map_location='cpu',weights_only=True)
        if value['binding']!=expected:raise ValueError('Committed diagnostic binding differs: '+str(path))
        return value

    try:
        progress('loading')
        runtime=runner.Runtime(out);runtime.model.eval();runtime.arm('bf16')
        # The reader phase reuses this immutable original copy; never serialize credentials or reload a model.
        original_weights={name:m.weight.detach().cpu().clone() for name,m in runtime.q.modules.items()}
        banks=TrainableBanks(runtime,p['calibration']['bank_directory']);parameters=banks.parameters()
        if len(parameters)!=1228:raise ValueError('Expected only original books and log-row-gain parameters')
        names=list(banks.modules);initial=banks.snapshot()
        for point in config['points']:
            masters=initial if point=='initial' else checkpoint['module_masters']
            banks.restore(masters)
            gradients=[]
            for case in config['case_ids']:
                path=out/'gradients'/point/f'{case}.pt';expected=binding(point,case,'gradient')
                if path.is_file():
                    gradients.append(load_bound(path,expected));continue
                deadline.check(1200);started=time.monotonic();progress('parameter_gradient',point=point,case_id=case)
                value=datum(case);banks.seen.clear()
                count('new_student_full_queries','parameter_forward')
                action,counts,future=sampler.differentiable_action(runtime,value,expected['sampler_seed'],checkpointed=True,return_future=True)
                reference=endpoint(point,case)
                parity={key:float((actual.detach().float().cpu()-reference[key].float()).abs().max())
                        for key,actual in [('action',action),('future',future)]}
                if max(parity.values())>1e-6 or len(banks.seen)!=614:raise ValueError('Same-point original endpoint replay failed')
                t={key:target(case)[key].to(action.device) for key in ('action','future')}
                values=loss_terms(torch,action,future,t,prep['stats'],p['anchor'],prep['temperature'])
                terms=weighted(values);grads={};unused={}
                for i,term in enumerate(TERMS):
                    count('new_parameter_reverse_passes','parameter_backward_'+term)
                    g=torch.autograd.grad(terms[term],parameters,retain_graph=i<len(TERMS)-1,allow_unused=True)
                    unused[term]=sum(v is None for v in g)
                    grads[term]=[torch.zeros_like(parameter,device='cpu') if v is None else v.detach().float().cpu()
                                 for parameter,v in zip(parameters,g)]
                    if not all(bool(torch.isfinite(v).all()) for v in grads[term]):raise ValueError('Nonfinite component gradient')
                    del g
                record={'binding':expected,'point':point,'case_id':case,'parameter_names':names,'gradients':grads,
                        'unused_parameter_tensors':unused,'metrics':metrics(action.detach().cpu(),future.detach().cpu(),case),
                        'parity':parity,'counts':counts,'seconds':time.monotonic()-started,
                        'action':action.detach().cpu(),'future':future.detach().cpu(),'pbs_jobid':os.environ['PBS_JOBID']}
                path.parent.mkdir(parents=True,exist_ok=True);atomic_torch_save(torch,path,record);gradients.append(record)
                del action,future,values,terms,grads,t,reference
                gc.collect();torch.cuda.empty_cache()
                progress('parameter_gradient_committed',point=point,case_id=case,seconds=record['seconds'])
            average,geometry=case_geometry(gradients)
            save(out/point/'gradient_geometry.json',geometry)
            originals=[masters[n][key] for n in names for key in ('books','log_scales')]
            base_metrics={m:st.mean(r['metrics'][m] for r in gradients) for m in gradients[0]['metrics']}
            proposals=[]
            modes=['fresh']+(['saved'] if point=='final_kl' else [])
            for mode in modes:
                for fraction in config['beta_fractions']:
                    delta=adam_delta(torch,originals,mix(average,fraction),config['learning_rate'],
                                     checkpoint['optimizer'] if mode=='saved' else None)
                    proposals.append((mode+'_beta'+str(fraction),delta))
            base_delta=adam_delta(torch,originals,mix(average,0),config['learning_rate'])
            kl_delta=adam_delta(torch,originals,average['weighted_kl'],config['learning_rate'])
            factor=(dot(base_delta,base_delta)/dot(kl_delta,kl_delta))**.5
            proposals.append(('fresh_kl_only_norm_matched',[v*factor for v in kl_delta]))
            panels=[]
            for label,delta in proposals:
                for fraction in config['step_fractions']:
                    panel=label+'_step'+str(fraction);actual_delta=[v*fraction for v in delta]
                    records=[];deadline.check(300)
                    try:
                        with torch.no_grad():
                            for parameter,original,step in zip(parameters,originals,actual_delta):parameter.copy_(original.to(parameter.device)+step.to(parameter.device))
                        banks.refresh()
                        changed={group:{'changed':0,'coordinates':0} for group in ('books','row_gains')}
                        for i,(parameter,original) in enumerate(zip(parameters,originals)):
                            current=parameter.detach().cpu();before=original
                            if i%2:current,before=current.exp(),before.exp()
                            key='row_gains' if i%2 else 'books'
                            changed[key]['changed']+=int((current.to(torch.bfloat16)!=before.to(torch.bfloat16)).sum())
                            changed[key]['coordinates']+=current.numel()
                        for case in config['case_ids']:
                            path=out/'probes'/point/panel/f'{case}.pt';expected=binding(point,case,panel)
                            if path.is_file():records.append(load_bound(path,expected));continue
                            deadline.check(120);value=datum(case);count('new_student_full_queries','parameter_probe')
                            action,counts,future=sampler.differentiable_action(runtime,value,expected['sampler_seed'],checkpointed=False,return_future=True)
                            record={'binding':expected,'case_id':case,'metrics':metrics(action.cpu(),future.cpu(),case),
                                    'action':action.cpu(),'future':future.cpu(),'counts':counts,'pbs_jobid':os.environ['PBS_JOBID']}
                            path.parent.mkdir(parents=True,exist_ok=True);atomic_torch_save(torch,path,record);records.append(record)
                            del action,future
                        observed={m:st.mean(r['metrics'][m] for r in records) for m in base_metrics}
                        panel_summary={'label':panel,'case_count':len(records),'metrics':observed,
                            'beta_fraction':None if 'kl_only' in label else float(label.split('beta')[-1]),
                            'optimizer_state':'saved' if label.startswith('saved') else 'fresh','step_fraction':fraction,
                            'delta':{m:observed[m]-base_metrics[m] for m in base_metrics},
                            'predicted_weighted_component_delta':{t:dot(average[t],actual_delta) for t in TERMS},
                            'parameter_delta_norm':dot(actual_delta,actual_delta)**.5,'stored_coordinates_changed':changed,
                            'cases':[{'case_id':r['case_id'],**r['metrics']} for r in records]}
                        panels.append(panel_summary);save(out/point/'panels.json',panels)
                        progress('parameter_probe_committed',point=point,panel=panel,delta=panel_summary['delta'])
                    finally:
                        banks.restore(masters)
                        if not all(torch.equal(parameter.detach().cpu(),original) for parameter,original in zip(parameters,originals)):
                            raise ValueError('Reversible probe failed exact master rollback')
            summaries[point]={'gradient_records':len(gradients),'baseline_metrics':base_metrics,'geometry':geometry,'panels':panels}
            del gradients,average,proposals,base_delta,kl_delta
            gc.collect()
        # Restore the original BF16 Linear weights and forwards before any reader intervention.
        for name,module in banks.modules.items():
            module.forward=banks.forwards[name];module.weight.data=original_weights[name].to(module.weight.device)
        for parameter in runtime.model.parameters():parameter.requires_grad_(False);parameter.grad=None
        del banks,parameters,initial,masters,originals,checkpoint,delta,actual_delta
        gc.collect();torch.cuda.empty_cache()
        from mechanism_frozen_reader import BF16Reader
        reader=BF16Reader(runtime,{name:m.weight.detach() for name,m in runtime.q.modules.items()})
        reader_records=[]
        for case in config['case_ids']:
            path=out/'reader'/f'{case}.pt';expected=binding('final_kl',case,'reader')
            if path.is_file():reader_records.append(load_bound(path,expected));continue
            deadline.check(600);value=datum(case);t=target(case);z0=endpoint('final_kl',case)['future'].float()
            repairs,repair_check=matched_repairs(torch,z0,t['future'],prep,p['anchor'],config['repair_nmse_relative_tolerance'])
            outputs={};reader_counts={};derivatives=None;noise=None
            for label,z in [('teacher',t['future']),('original',z0),('repair_visible',repairs['visible']),('repair_null',repairs['null'])]:
                use_grad=label=='original'
                leaf=z.detach().to('cuda',dtype=torch.float32).requires_grad_(use_grad)
                count('new_reader_queries','reader_'+label)
                action,counts=reader(runtime,value,expected['sampler_seed'],leaf,checkpointed=use_grad)
                if counts['video_dit_calls']!=0 or not counts['teacher_future_injection_verified']:raise ValueError('Reader intervention contract failed')
                if noise is None:noise=counts['action_initial_noise']
                elif not torch.equal(noise,counts['action_initial_noise']):raise ValueError('Reader action noise changed')
                if label=='teacher':
                    error=float((action.detach().float().cpu()-t['action'].float()).abs().max())
                    if error>1e-6:raise ValueError('Original BF16 teacher reader parity failed: '+str(error))
                outputs[label]={'action':action.detach().cpu(),'metrics':metrics(action.detach().cpu(),z,case)}
                reader_counts[label]={key:v for key,v in counts.items() if not torch.is_tensor(v)}
                if use_grad:
                    target_action=t['action'].to(action.device)
                    reader_loss=(action[:10,:6].float()-target_action[:10,:6].float()).square().mean()
                    count('new_reader_reverse_passes','reader_latent_backward')
                    gr=torch.autograd.grad(reader_loss,leaf)[0].detach().float().cpu()
                    if float(gr[:,:,:1].abs().max())!=0:raise ValueError('Reader gradient reached clamped frame0')
                    local=z0.detach().clone().requires_grad_(True)
                    values=loss_terms(torch,outputs[label]['action'],local,t,prep['stats'],p['anchor'],prep['temperature'])
                    gz=torch.autograd.grad(weighted(values)['weighted_nmse'],local,retain_graph=True)[0]
                    gk=torch.autograd.grad(weighted(values)['weighted_kl'],local)[0]
                    visible,_=visible_projection(torch,gr[:,:,1:],t['future'],prep,p['anchor']);null=gr[:,:,1:]-visible
                    derivatives={'reader_vs_nmse':alignment([gr],[gz]),'reader_vs_kl':alignment([gr],[gk]),
                        'reader_gradient_raw_null_energy_fraction':float(null.double().square().sum()/gr.double().square().sum())}
                    del gr,gz,gk,reader_loss,values,local,visible,null
                del leaf,action,counts
                gc.collect();torch.cuda.empty_cache()
            record={'binding':expected,'case_id':case,'matching':repair_check,'outputs':outputs,
                    'reader_gradient':derivatives,'counts':reader_counts,'pbs_jobid':os.environ['PBS_JOBID']}
            path.parent.mkdir(parents=True,exist_ok=True);atomic_torch_save(torch,path,record);reader_records.append(record)
            progress('reader_committed',case_id=case)
        result={**state,'complete':True,'stage':'complete','config':config,'points':summaries,
                'reader_records':[{'case_id':r['case_id'],'matching':r['matching'],'reader_gradient':r['reader_gradient'],
                    'outputs':{k:v['metrics'] for k,v in r['outputs'].items()}} for r in reader_records],
                'gradient_records':sum(s['gradient_records'] for s in summaries.values()),
                'parameter_probe_panels':sum(len(s['panels']) for s in summaries.values()),
                'parameter_probe_queries':sum(len(s['panels'])*8 for s in summaries.values()),
                'reader_record_count':len(reader_records),'completed_pbs_jobid':os.environ['PBS_JOBID']}
        if (result['gradient_records']!=16 or result['parameter_probe_panels']!=22
                or result['parameter_probe_queries']!=176 or result['reader_record_count']!=8):
            raise ValueError('Declared complete diagnostic coverage not met')
        save(out/'details.json',result)
        compact=copy.deepcopy(result)
        for point,value in compact['points'].items():
            value['geometry'].pop('per_case')
            for panel in value['panels']:panel.pop('cases')
        save(out/'summary.json',compact);progress('complete',complete=True)
        print('C8_MECHANISM_COMPLETE',flush=True)
        return 0
    except TimeoutError as error:
        progress('checkpointed',reason=str(error));return 75
    except Exception as error:
        progress('failed',error=type(error).__name__+': '+str(error));raise
    finally:
        if runtime is not None and runtime.env is not None:runtime.env.close()


if __name__=='__main__':raise SystemExit(run())

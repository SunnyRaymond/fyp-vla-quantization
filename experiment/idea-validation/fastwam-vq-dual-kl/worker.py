"""Original C80 two-factor loss comparison; model operations are PBS-only."""
import argparse
import json
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80


def objective(values,unit,p,scale):
    a = scale['action_kl_coefficient']*values['action_kl'] if unit in ('action_kl','dual_kl') else values['action_mse']
    z = scale['latent_kl_coefficient']*values['latent_kl'] if unit in ('latent_kl','dual_kl') else values['latent_nmse']
    return a+p['training']['rho']*p['training']['normalization_c']*z


def run(p,unit):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from channel_banks import ChannelBanks
    from softbin_loss import losses
    from metrics_reference import metrics
    from checkpointing import atomic_torch_save,capture_rng,restore_rng,optimizer_to_device,to_cpu

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261017)
    root=Path(p['remote_root']);out=root/'results'/unit
    out.mkdir(parents=True,exist_ok=True)
    artifact=Path(os.environ['ARTIFACTS'])
    source_protocol=json.loads((artifact/'source_protocol.json').read_text())
    if json.loads((Path(p['source_per_channel_root'])/'protocol.json').read_text())!=source_protocol:
        raise RuntimeError('Original per-channel protocol differs')
    for key in ('training','calibration','evaluation','clipping','scaling'):
        if key=='evaluation':
            if any(p[key][k]!=source_protocol[key][k] for k in ('case_ids','seed_indices','aggregation')):
                raise RuntimeError('Original heldout differs')
        elif p[key]!=source_protocol[key]:
            raise RuntimeError('Loss-only invariant changed: '+key)
    gradient_check=json.loads((artifact/'channel_selfcheck.json').read_text())
    kl_check=json.loads((artifact/'softbin_selfcheck.json').read_text())
    if not all(gradient_check.get(k) is True for k in ('passed','independent_reference_passed','old_surrogate_rejected')) or not kl_check['passed']:
        raise RuntimeError('Independent gradient and soft-bin checks required')
    if any(v['pbs_jobid']!=os.environ['PBS_JOBID'] for v in (gradient_check,kl_check)):
        raise RuntimeError('Selfcheck belongs to another allocation')
    stats=json.loads((root/'results/fit_stats/summary.json').read_text())
    if not stats['complete'] or stats['heldout_used'] or stats['calibration_case_ids']!=p['calibration']['case_ids']:
        raise RuntimeError('Frozen calibration-only statistics required')
    data_root,rows,cache,deployment,reference_action=ref.load_inputs(p['source_clipping_protocol'],torch)
    teacher_dirs={'calibration':Path(p['control_root'])/'results/teacher/calibration',
                  'heldout':Path(p['clipping_root'])/'results/teacher'}
    for split,directory in teacher_dirs.items():
        receipt=json.loads((directory/'receipt.json').read_text())
        expected=p['calibration' if split=='calibration' else 'evaluation']['case_ids']
        if not receipt['complete'] or receipt['queries']!=80 or receipt['case_ids']!=expected or receipt['seed_indices']!=([0] if split=='calibration' else [0,1]):
            raise RuntimeError('Shared teacher coverage differs')
    state={'protocol':p['protocol'],'unit':unit,'pbs_jobid':os.environ['PBS_JOBID'],'complete':False,
           'phase':'loading','updates':0,'new_teacher_queries':0,'new_model_queries':0}

    def progress(**values):
        state.update(values);ref.save_json(out/'progress.json',state)
        print('DUAL_KL_PROGRESS '+json.dumps(state),flush=True)

    runtime=None;deadline=ref.Deadline();progress()
    try:
        runtime=runner.Runtime(out);runtime.model.eval();runtime.arm('bf16');data={}
        banks=ChannelBanks(runtime,p['calibration']['bank_directory'],deployment['module_masters'],p['clipping'],p['scaling'])
        if any(bool((v!=0).any()) for v in banks.shadows.values()) or any(float(v)!=1 for v in banks.ratios.values()):
            raise RuntimeError('Fresh D=1/r=1 initialization required')
        groups=banks.optimizer_groups(p['training'],p['scaling'])
        parameters=[v for group in groups for v in group['params']]
        if {id(v) for v in parameters}!={id(v) for v in list(banks.ratios.values())+list(banks.shadows.values())}:
            raise RuntimeError('Only original ratios and channel scales may train')
        if any(v.requires_grad for v in runtime.model.parameters()) or any(v.requires_grad for v in list(banks.books.values())+list(banks.logs.values())):
            raise RuntimeError('Weights, codebooks and gains must remain frozen')

        def datum(case):
            if case not in data:
                obs,meta=calibration_source.load_observation(np,data_root,rows[case]);data[case]=runtime.datum(obs),meta
            return data[case]

        def query(case,seed):
            deadline.check();value,meta=datum(case)
            result=ref.query(runtime,sampler,torch,value,meta['description'],rows[case]['sampler_seeds'][seed],banks)
            state['new_model_queries']+=1
            return result

        def teacher(split,case,seed):
            saved=torch.load(teacher_dirs[split]/f'{case}_{seed}.pt',map_location='cpu',weights_only=True)
            if saved['plan_row']!=rows[case] or saved['sampler_seed']!=rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Shared teacher binding differs')
            return saved

        def terms(action,future,target):
            return losses(torch,action,future,target['action'].to(action.device),target['future'].to(future.device),stats,p['softbin'])

        first=p['calibration']['case_ids'][0]
        initial=query(first,0)
        parity=float((initial[0].float()-torch.tensor(reference_action['action'],dtype=torch.float32)).abs().max())
        if parity!=0:raise RuntimeError('Original baseline forward parity failed')
        progress(initial_parity_max_abs=parity)

        if unit=='prepare':
            initial_rows=[]
            for case in p['calibration']['case_ids']:
                action,future,counts=initial if case==first else query(case,0)
                target=teacher('calibration',case,0)
                values=terms(action,future,target)
                old=json.loads((Path(p['control_root'])/'results/calibration/baseline/queries'/f'{case}_0.json').read_text())
                measured=metrics(torch,latent_reference,action,future,target,p['training'])
                if any(measured[m]!=old[m] for m in c80.METRICS):
                    raise RuntimeError('Original calibration baseline replay differs')
                initial_rows.append({'case_id':case,'seed_index':0,**{k:float(v) for k,v in values.items()}})
                progress(phase='initial_calibration',queries=len(initial_rows),last_case=case)
            means={k:sum(r[k] for r in initial_rows)/80 for k in ('action_mse','latent_nmse','action_kl','latent_kl')}
            if any(v<=0 for v in means.values()):raise RuntimeError('Initial loss scaling requires positive finite means')
            scale={'protocol':p['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],'case_ids':p['calibration']['case_ids'],
                'seed_indices':[0],'heldout_used':False,'initial_means':means,'initial_rows':initial_rows,
                'action_kl_coefficient':means['action_mse']/means['action_kl'],
                'latent_kl_coefficient':means['latent_nmse']/means['latent_kl'],
                'original_latent_weight':p['training']['rho']*p['training']['normalization_c'],'rule':p['loss_scaling']['rule']}
            ref.save_json(out/'loss_scaling.json',scale)
            saved=torch.load(Path(p['source_per_channel_root'])/'results/train_eval/deployment.pt',map_location='cpu',weights_only=True)
            old_receipt=json.loads((Path(p['source_per_channel_root'])/'results/train_eval/training_receipt.json').read_text())
            if saved['updates']!=80 or old_receipt['update_order']!=p['calibration']['update_order'] or not old_receipt['complete']:
                raise RuntimeError('Completed exact80 control required')
            banks.restore_clips(saved['ratios']);banks.restore_scales(saved['log2_scales'])
            banks.cache_compensated_weights()
            arm='mse_nmse'
            training={'complete':True,'updates':80,'update_order':old_receipt['update_order'],
                      'reused_training':True,'source_job_id':p['source_per_channel_job'],'source_receipt':str(Path(p['source_per_channel_root'])/'results/train_eval/training_receipt.json')}
        else:
            del initial
            scale=json.loads((root/'results/prepare/loss_scaling.json').read_text())
            if not scale['complete'] or scale['heldout_used'] or scale['case_ids']!=p['calibration']['case_ids']:
                raise RuntimeError('Calibration-only frozen scaling required')
            optimizer=torch.optim.Adam(groups);checkpoint_path=out/'checkpoint.pt';updates=0;history=[]
            if checkpoint_path.is_file():
                saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
                if saved['protocol']!=p or saved['unit']!=unit or saved['loss_scaling']!=scale:
                    raise RuntimeError('Resume contract differs')
                updates,history=saved['updates'],saved['history']
                if [h['case_id'] for h in history]!=p['calibration']['update_order'][:updates]:raise RuntimeError('Update order changed')
                banks.restore_clips(saved['ratios']);banks.restore_scales(saved['shadow_log2_scales'])
                optimizer.load_state_dict(saved['optimizer']);optimizer_to_device(torch,optimizer,'cuda');restore_rng(torch,saved['rng'])
            while updates<80:
                deadline.check(max([h['seconds'] for h in history[-3:]],default=300)*1.15)
                case=p['calibration']['update_order'][updates];value,meta=datum(case)
                runtime.description=meta['description'];banks.seen.clear();target=teacher('calibration',case,0)
                optimizer.zero_grad(set_to_none=True);started=time.monotonic()
                action,counts,future=sampler.differentiable_action(runtime,value,rows[case]['sampler_seeds'][0],checkpointed=True,return_future=True)
                state['new_model_queries']+=1
                values=terms(action,future,target);loss=objective(values,unit,p,scale)
                if len(banks.seen)!=614 or not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError('Training loss or module coverage failed')
                loss.backward();coverage=banks.coverage()
                if any(v['connected']!=614 or v['finite']!=614 or v['nonzero']<1 for v in coverage.values()):
                    raise RuntimeError('Ratio/channel gradients disconnected or nonfinite')
                norm=torch.nn.utils.clip_grad_norm_(parameters,p['training']['gradient_norm_clip'],error_if_nonfinite=True)
                optimizer.step()
                if updates==0 and p['training']['optimizer_reset_after_update1']:optimizer.state.clear()
                banks.project();torch.cuda.synchronize();updates+=1
                history.append({'update':updates,'case_id':case,'seed_index':0,**{k:float(v.detach()) for k,v in values.items()},
                    'loss':float(loss.detach()),'gradient_norm':float(norm),'coverage':coverage,'seconds':time.monotonic()-started,
                    'pbs_jobid':os.environ['PBS_JOBID'],**counts})
                del action,future,values,loss
                atomic_torch_save(torch,checkpoint_path,{'protocol':p,'unit':unit,'loss_scaling':scale,'updates':updates,'history':history,
                    'ratios':banks.clip_snapshot(),'shadow_log2_scales':banks.scale_snapshot(),
                    'optimizer':to_cpu(optimizer.state_dict()),'rng':capture_rng(torch)})
                progress(phase='training',updates=updates,last_update=history[-1])
            progress(phase='deployment_parity',updates=80)
            uncached=query(first,0);rounding=banks.cache_compensated_weights();cached=query(first,0)
            folded={'action_max_abs':float((uncached[0].float()-cached[0].float()).abs().max()),
                    'future_max_abs':float((uncached[1].float()-cached[1].float()).abs().max())}
            if any(v!=0 for v in folded.values()):raise RuntimeError('Cached deployment differs from training forward')
            runtime.current_arm=unit;runtime.description=datum(first)[1]['description']
            actual,_=runtime.infer(datum(first)[0],rows[first]['sampler_seeds'][0],measure=False)
            state['new_model_queries']+=1
            actual_parity=float((actual.float().cpu()-cached[0].float()).abs().max())
            if actual_parity!=0:raise RuntimeError('Actual inference action differs')
            del actual,uncached,cached
            frozen=banks.verify_frozen(deployment['module_masters'])
            atomic_torch_save(torch,out/'deployment.pt',{'protocol':p,'unit':unit,'updates':80,
                'baseline_checkpoint':str(Path(p['baseline_root'])/'results/training/rho1.5/deployment.pt'),
                'ratios':banks.clip_snapshot(),'log2_scales':banks.scale_snapshot(),'scale_metadata_dtype':'float32_log2'})
            training={'protocol':p['protocol'],'complete':True,'unit':unit,'updates':80,'history':history,'update_order':p['calibration']['update_order'],
                'loss_scaling':scale,'initial_parity_max_abs':parity,'cached_query_parity':folded,'actual_infer_parity_max_abs':actual_parity,
                'clipping':banks.clip_summary(),'scaling':banks.scaling_summary(),'gradient_coupling_complete':True,'input_absmax_chain':True,
                'softbin_selfcheck':kl_check,'gradient_selfcheck':gradient_check,'pbs_jobid':os.environ['PBS_JOBID'],**frozen}
            ref.save_json(out/'training_receipt.json',training)
            ref.save_json(out/'weight_rounding.json',rounding)
            arm=unit

        endpoint_count=0
        for split in ('calibration','heldout'):
            records=[]
            for case,seed in c80.queries(p,split):
                path=root/'results'/split/arm/'queries'/f'{case}_{seed}.json'
                if path.is_file():
                    record=json.loads(path.read_text())
                    expected={'protocol':p['protocol'],'arm':arm,'split':split,'updates':80,'plan_row':rows[case],
                              'case_id':case,'seed_index':seed,'sampler_seed':rows[case]['sampler_seeds'][seed]}
                    if any(record.get(k)!=v for k,v in expected.items()):raise RuntimeError('Saved endpoint binding differs')
                else:
                    target=teacher(split,case,seed);action,future,counts=query(case,seed)
                    values=terms(action,future,target)
                    measured=metrics(torch,latent_reference,action,future,target,p['training'])
                    if unit=='prepare':
                        previous=json.loads((Path(p['source_per_channel_root'])/'results'/split/'per_channel_clip/queries'/f'{case}_{seed}.json').read_text())
                        if previous['plan_row']!=rows[case] or any(previous[m]!=measured[m] for m in c80.METRICS):
                            raise RuntimeError('Reused MSE/NMSE checkpoint endpoint replay differs')
                    record={'protocol':p['protocol'],'arm':arm,'split':split,'updates':80,'case_id':case,'suite':rows[case]['suite'],
                        'plan_row':rows[case],'seed_index':seed,'sampler_seed':rows[case]['sampler_seeds'][seed],
                        'pbs_jobid':os.environ['PBS_JOBID'],'counts':counts,**measured,
                        'action_kl':float(values['action_kl']),'latent_kl':float(values['latent_kl']),
                        'objective_loss':float(objective(values,arm,p,scale)),
                        'action':action.float().tolist(),'commands':runtime.command(action.clone()).tolist(),
                        'teacher_action':target['action'].float().tolist(),'teacher_commands':runtime.command(target['action'].clone()).tolist()}
                    ref.save_json(path,record)
                records.append(record);endpoint_count+=1
                progress(phase='evaluation',split=split,queries=len(records),last_case=case,last_seed=seed)
            summary={'protocol':p['protocol'],'complete':True,'arm':arm,'split':split,'updates':80,
                     'pbs_jobid':os.environ['PBS_JOBID'],**c80.aggregate_records(records,p,split)}
            for metric in ('action_kl','latent_kl','objective_loss'):
                summary['overall'][metric]=sum(r[metric] for r in records)/len(records)
                for suite in ref.SUITES:
                    rr=[r for r in records if r['suite']==suite]
                    summary['by_suite'][suite][metric]=sum(r[metric] for r in rr)/len(rr)
            ref.save_json(root/'results'/split/arm/'summary.json',summary)
        summary={**state,'complete':True,'phase':'complete','updates':80,'evaluation_queries':endpoint_count,
                 'training_receipt':training,'new_teacher_queries':0}
        if unit=='prepare':summary.update(initial_calibration_queries=80,control_evaluation_queries=160,control_replay_max_abs=0)
        ref.save_json(out/'summary.json',summary)
        progress(phase='complete',complete=True,updates=80)
        print('DUAL_KL_WORKER_COMPLETE',flush=True)
        return 0
    except TimeoutError as exc:
        progress(phase='checkpointed',reason=str(exc));return 75
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if runtime is not None and runtime.env is not None:runtime.env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--unit',choices=('prepare','action_kl','latent_kl','dual_kl'),required=True)
    args=parser.parse_args();c80.guard();p=json.loads(Path(args.protocol).read_text())
    return run(p,args.unit)


if __name__=='__main__':raise SystemExit(main())

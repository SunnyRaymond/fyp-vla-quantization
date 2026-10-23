"""Single-factor gripper-loss extension of the frozen C80 per-channel recipe."""
import argparse
import json
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80
import discrete_reference as discrete
import uncoupled_reference as uncoupled
import coupled_reference as coupled
from gripper_loss import action_losses


def validate_record(record,p,row,split,seed):
    expected={'protocol':p['protocol'],'arm':p['arm'],'split':split,'updates':80,
        'case_id':row['case_id'],'suite':row['suite'],'plan_row':row,'seed_index':seed,
        'sampler_seed':row['sampler_seeds'][seed],'weight_path':'continuous_per_channel_scaled_BF16_weights'}
    if any(record.get(k)!=v for k,v in expected.items()):raise RuntimeError('Saved endpoint binding differs')


def run(p):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from metrics_reference import metrics
    from channel_banks import ChannelBanks
    from checkpointing import atomic_torch_save,capture_rng,restore_rng,optimizer_to_device,to_cpu

    sampler._require_compute_allocation()
    gradient_check=json.loads((Path(os.environ['ARTIFACTS'])/'channel_selfcheck.json').read_text())
    if gradient_check['pbs_jobid']!=os.environ['PBS_JOBID'] or not all(gradient_check.get(k) is True for k in ('passed','independent_reference_passed','old_surrogate_rejected')):
        raise RuntimeError('Independent absmax coupling selfcheck required before model loading')
    grip_check=json.loads((Path(os.environ['ARTIFACTS'])/'gripper_selfcheck.json').read_text())
    if not grip_check['passed'] or grip_check['pbs_jobid']!=os.environ['PBS_JOBID']:
        raise RuntimeError('Gripper loss analytic-gradient check required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261017)
    root=Path(p['remote_root']);source=Path(p['control_root']);sp=p['source_clipping_protocol']
    if json.loads((source/'protocol.json').read_text())!=p['source_control_protocol']:
        raise RuntimeError('Completed C80 control protocol changed')
    previous=json.loads((source/'results/comparison.json').read_text())
    if not previous['complete'] or previous['evaluation_queries']!=320:raise RuntimeError('Complete control endpoints required')
    if json.loads((Path(p['discrete_root'])/'protocol.json').read_text())!=p['source_discrete_protocol']:
        raise RuntimeError('Completed discrete B32 protocol changed')
    if json.loads((Path(p['uncoupled_root'])/'protocol.json').read_text())!=p['source_uncoupled_protocol']:
        raise RuntimeError('Completed uncoupled B32 protocol changed')
    if json.loads((Path(p['coupled_root'])/'protocol.json').read_text())!=p['source_coupled_protocol']:
        raise RuntimeError('Completed coupled B32 protocol changed')
    data_root,rows,cache,deployment,reference_action=ref.load_inputs(sp,torch)
    plan=json.loads(Path(p['calibration']['plan_file']).read_text())
    if plan['calibration_cohorts']['c80']!=p['calibration']['case_ids']:raise RuntimeError('C80 changed')
    for split,receipt_path in (
        ('calibration',source/'results/teacher/calibration/receipt.json'),
        ('heldout',Path(p['clipping_root'])/'results/teacher/receipt.json')):
        receipt=json.loads(receipt_path.read_text())
        expected_ids=p['calibration' if split=='calibration' else 'evaluation']['case_ids']
        if not receipt['complete'] or receipt['queries']!=80 or receipt['case_ids']!=expected_ids or receipt['seed_indices']!=([0] if split=='calibration' else [0,1]):
            raise RuntimeError('Shared teacher coverage differs')
    out=root/'results/train_eval';out.mkdir(parents=True,exist_ok=True)
    state={'protocol':p['protocol'],'unit':'train_eval','pbs_jobid':os.environ['PBS_JOBID'],
        'phase':'loading','complete':False,'updates':0,'new_teacher_queries':0,
        'new_evaluation_queries_this_job':0,'new_parity_queries_this_job':0}

    def progress(**values):
        state.update(values);ref.save_json(out/'progress.json',state)
        ref.save_json(out/('executions_'+os.environ['PBS_JOBID']+'.json'),state)
        print('CHANNEL_PROGRESS '+json.dumps(state),flush=True)

    runtime=None;deadline=ref.Deadline();progress()
    try:
        runtime=runner.Runtime(out);runtime.model.eval();runtime.arm('bf16');data={}
        visited_data,visited_info,visited_rows=prepare_visited(runtime,torch,np,p,deadline,state,progress)
        banks=ChannelBanks(runtime,p['calibration']['bank_directory'],deployment['module_masters'],p['clipping'],p['scaling'])
        if any(bool((v!=0).any()) for v in banks.shadows.values()) or any(float(v)!=1 for v in banks.ratios.values()):
            raise RuntimeError('Fresh original-baseline D=1/r=1 required')
        groups=banks.optimizer_groups(p['training'],p['scaling']);parameters=[v for g in groups for v in g['params']]
        if {id(v) for v in parameters}!={id(v) for v in list(banks.ratios.values())+list(banks.shadows.values())}:
            raise RuntimeError('Only ratios and per-channel shadows may be optimized')
        if any(v.requires_grad for v in runtime.model.parameters()) or any(v.requires_grad for v in list(banks.books.values())+list(banks.logs.values())):
            raise RuntimeError('Model weights, codebooks and gains must be frozen')

        def datum(case):
            if case not in data:
                obs,meta=calibration_source.load_observation(np,data_root,rows[case]);data[case]=runtime.datum(obs),meta
            return data[case]

        def query(case,seed):
            deadline.check();value,meta=datum(case)
            return ref.query(runtime,sampler,torch,value,meta['description'],rows[case]['sampler_seeds'][seed],banks)

        def teacher(split,case,seed):
            directory=source/'results/teacher/calibration' if split=='calibration' else Path(p['clipping_root'])/'results/teacher'
            saved=torch.load(directory/f'{case}_{seed}.pt',map_location='cpu',weights_only=True)
            if saved['plan_row']!=rows[case] or saved['sampler_seed']!=rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Shared teacher binding differs')
            return saved

        first=p['calibration']['case_ids'][0]
        initial=query(first,0);state['new_parity_queries_this_job']+=1
        parity=float((initial[0].float()-torch.tensor(reference_action['action'],dtype=torch.float32)).abs().max())
        if parity!=0:raise RuntimeError('D=I,r=1 must reproduce the original baseline exactly')
        del initial
        progress(initial_parity_max_abs=parity,scaling=banks.scaling_summary())
        optimizer=torch.optim.Adam(groups);checkpoint_path=out/'checkpoint.pt';updates=0;history=[]
        if checkpoint_path.is_file():
            saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            if saved['protocol']!=p:raise RuntimeError('Resume protocol changed')
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
            action_mse,gripper_mse=action_losses(action,target['action'].to('cuda'),p['training']['lambda_g'])
            latent_nmse=latent_reference.future_nmse_loss(torch,future,target['future'].to('cuda'))
            loss=action_mse+p['training']['lambda_g']*gripper_mse+p['training']['rho']*p['training']['normalization_c']*latent_nmse
            if len(banks.seen)!=614 or not loss.requires_grad or not bool(torch.isfinite(loss)):raise RuntimeError('Training loss/module coverage failed')
            loss.backward();coverage=banks.coverage()
            if any(v['connected']!=614 or v['finite']!=614 or v['nonzero']<1 for v in coverage.values()):raise RuntimeError('Ratio/channel gradients disconnected or nonfinite')
            norm=torch.nn.utils.clip_grad_norm_(parameters,p['training']['gradient_norm_clip'],error_if_nonfinite=True)
            optimizer.step()
            if updates==0 and p['training']['optimizer_reset_after_update1']:optimizer.state.clear()
            banks.project();torch.cuda.synchronize();updates+=1
            history.append({'update':updates,'case_id':case,'seed_index':0,'action_mse':float(action_mse),
                'gripper_mse':float(gripper_mse),'lambda_g':p['training']['lambda_g'],'future_latent_nmse':float(latent_nmse),'loss':float(loss),'gradient_norm':float(norm),
                'coverage':coverage,'seconds':time.monotonic()-started,'pbs_jobid':os.environ['PBS_JOBID'],**counts})
            del action,future,action_mse,gripper_mse,latent_nmse,loss
            atomic_torch_save(torch,checkpoint_path,{'protocol':p,'updates':updates,'history':history,
                'ratios':banks.clip_snapshot(),'shadow_log2_scales':banks.scale_snapshot(),
                'optimizer':to_cpu(optimizer.state_dict()),'rng':capture_rng(torch)})
            progress(phase='training',updates=updates,last_update=history[-1],clipping=banks.clip_summary(),scaling=banks.scaling_summary())

        progress(phase='folding_parity',updates=80)
        uncached=query(first,0);state['new_parity_queries_this_job']+=1
        rounding=banks.cache_compensated_weights();folded=query(first,0)
        ref.save_json(out/'weight_rounding.json',rounding)
        rounding_summary={k:v for k,v in rounding.items() if k!='per_module'};state['new_parity_queries_this_job']+=1
        folded_parity={'action_max_abs':float((uncached[0].float()-folded[0].float()).abs().max()),
            'future_max_abs':float((uncached[1].float()-folded[1].float()).abs().max())}
        if any(v!=0 for v in folded_parity.values()):raise RuntimeError('Uncached and cached continuous BF16 compensated query outputs differ')
        del uncached
        runtime.current_arm=p['arm'];runtime.description=datum(first)[1]['description']
        actual,_=runtime.infer(datum(first)[0],rows[first]['sampler_seeds'][0],measure=False)
        state['new_parity_queries_this_job']+=1
        actual_parity=float((actual.float().cpu()-folded[0].float()).abs().max());del actual
        if actual_parity!=0:raise RuntimeError('Deployment infer parity differs')
        frozen=banks.verify_frozen(deployment['module_masters'])
        atomic_torch_save(torch,out/'deployment.pt',{'protocol':p,'updates':80,
            'baseline_checkpoint':str(Path(p['baseline_root'])/'results/training/rho1.5/deployment.pt'),
            'ratios':banks.clip_snapshot(),'log2_scales':banks.scale_snapshot(),'scale_metadata_dtype':'float32_log2'})
        training={'protocol':p['protocol'],'complete':True,'updates':80,'history':history,'update_order':p['calibration']['update_order'],
            'lambda_g':p['training']['lambda_g'],'loss_only_change':True,'initial_parity_max_abs':parity,'actual_infer_parity_max_abs':actual_parity,'cached_query_parity':folded_parity,'clipping':banks.clip_summary(),
            'scaling':banks.scaling_summary(),'weight_compensation_rounding':rounding_summary,'gradient_selfcheck':gradient_check,'gripper_selfcheck':grip_check,
            'gradient_coupling_complete':True,'input_absmax_chain':True,'pbs_jobid':os.environ['PBS_JOBID'],**frozen}
        ref.save_json(out/'training_receipt.json',training)
        for split in ('calibration','heldout'):
            records=[]
            for case,seed in c80.queries(p,split):
                path=root/'results'/split/p['arm']/'queries'/f'{case}_{seed}.json'
                if path.is_file():
                    record=json.loads(path.read_text());validate_record(record,p,rows[case],split,seed)
                else:
                    target=teacher(split,case,seed)
                    if split=='calibration' and case==first:action,future,counts=folded
                    else:
                        action,future,counts=query(case,seed);state['new_evaluation_queries_this_job']+=1
                    record={'protocol':p['protocol'],'arm':p['arm'],'split':split,'updates':80,
                        'case_id':case,'suite':rows[case]['suite'],'plan_row':rows[case],'seed_index':seed,
                        'sampler_seed':rows[case]['sampler_seeds'][seed],'weight_path':'continuous_per_channel_scaled_BF16_weights',
                        'pbs_jobid':os.environ['PBS_JOBID'],'counts':counts,
                        **metrics(torch,latent_reference,action,future,target,p['training'])}
                    record.update(action_payload(runtime,torch,np,action))
                    record['teacher']=action_payload(runtime,torch,np,target['action'])
                    record['gripper_augmented_loss']=record['mixed_loss']+p['training']['lambda_g']*record['first10gripper_rmse']**2
                    ref.save_json(path,record)
                records.append(record);progress(phase='evaluation',split=split,queries=len(records),last_case=case,last_seed=seed)
            ref.save_json(root/'results'/split/p['arm']/'summary.json',{'protocol':p['protocol'],'complete':True,
                'arm':p['arm'],'split':split,'updates':80,'pbs_jobid':os.environ['PBS_JOBID'],
                'clipping':banks.clip_summary(),'scaling':banks.scaling_summary(),**c80.aggregate_records(records,p,split)})
        evaluate_visited(runtime,banks,torch,np,p,visited_data,visited_info,visited_rows,deadline,state,progress)
        state['training_queries']=80;state['total_model_queries']=80+state['new_evaluation_queries_this_job']+state['new_parity_queries_this_job']+state['visited_queries']
        if state['total_model_queries']!=445 or state['new_parity_queries_this_job']!=6:raise RuntimeError('Full query budget/coverage differs')
        ref.save_json(out/'summary.json',{**state,'complete':True,'phase':'complete','updates':80,
            'evaluation_queries':160,'lambda_g':p['training']['lambda_g'],'cached_query_parity':folded_parity,'clipping':banks.clip_summary(),
            'scaling':banks.scaling_summary(),'actual_infer_parity_max_abs':actual_parity,'gradient_coupling_complete':True,'input_absmax_chain':True,**frozen})
        progress(phase='complete',complete=True,updates=80);print('CHANNEL_WORKER_COMPLETE',flush=True);return 0
    except TimeoutError as exc:
        progress(phase='checkpointed',reason=str(exc));return 75
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if runtime is not None and runtime.env is not None:runtime.env.close()


def validate_protocol(p):
    original=json.loads((Path(os.environ['ARTIFACTS'])/'source_per_channel_protocol.json').read_text())
    if json.loads((Path(p['source_per_channel_root'])/'protocol.json').read_text())!=original:
        raise RuntimeError('Completed source per-channel protocol changed')
    allowed={'protocol','remote_root','arm','arms','scope','training'}
    if any(p[k]!=v for k,v in original.items() if k not in allowed):
        raise RuntimeError('Only the declared gripper objective may change')
    training={**original['training'],'lambda_g':1/6,
        'loss':'first10 x6 motor MSE + (1/6) * first10 gripper MSE + rho * frozen normalization_c * future global NMSE excluding frame0'}
    if p['training']!=training or p['arm']!='per_channel_gripper' or p['budget']['expected_total_queries']!=445:
        raise RuntimeError('Frozen gripper-only recipe differs')
    return p


def action_payload(runtime,torch,np,action):
    action=action.detach().float().cpu()
    command=runtime.command(action.clone())
    continuous=runtime.ev._denormalize_action(action.clone(),runtime.processor)[0].copy()
    continuous[:,6]=-(2*continuous[:,6]-1)
    if not np.array_equal(np.sign(continuous[:,6]),command[:,6]):
        raise RuntimeError('Actual gripper conversion differs')
    return {'action':action.tolist(),'commands':command.tolist(),'continuous_commands':continuous.tolist()}


def prepare_visited(runtime,torch,np,p,deadline,state,progress):
    import visited_common as vc
    import visited_reference as vref
    import calibration_source
    vp=p['visited_source_protocol'];vroot=Path(p['visited_source_root'])
    if json.loads((vroot/'protocol.json').read_text())!=vp:
        raise RuntimeError('Frozen visited protocol changed')
    old=json.loads((vroot/'results/summary.json').read_text())
    agg=json.loads((vroot/'results/aggregate_summary.json').read_text())
    if not old['complete'] or old['records_completed']!=200 or old['pbs_jobid']!=p['visited_source_gpu_job'] or not agg['complete'] or agg['records']!=800:
        raise RuntimeError('Completed shared visited panel required')
    rp=vc.validate(json.loads((Path(vp['visited_root'])/'protocol.json').read_text()))
    rows=vc.expected_rows(rp)
    if {r['case_id']:r for r in json.loads((Path(vp['visited_root'])/'data/plan.json').read_text())['inputs']}!=rows:
        raise RuntimeError('Visited observation plan differs')
    datums={};info={};teachers={}
    for case in vp['case_ids']:
        obs,meta=calibration_source.load_observation(np,Path(vp['visited_root'])/'data',rows[case])
        for variant in vref.VARIANTS:
            perturbed,details=vref.source.perturb_observation(np,obs,variant,vp)
            datums[case,variant]=runtime.datum(perturbed)
            details['description']=meta['description'];info[case,variant]=details
        for variant in vref.VARIANTS:
            for field in ('image','proprio'):
                delta=datums[case,variant][field].float()-datums[case,'base'][field].float()
                info[case,variant]['model_'+field+'_delta_rms']=float(delta.square().mean().sqrt())
                info[case,variant]['model_'+field+'_delta_max_abs']=float(delta.abs().max())
            info[case,variant]['normalized_proprio_bf16']=datums[case,variant]['proprio'].float().cpu().tolist()
            teacher=vref.read(Path(vp['source_diagnostic_root'])/'results/records/bf16'/f'{case}_{variant}.json')
            vref.source.validate_record(teacher,vp['source_diagnostic_protocol'],rows[case],'bf16',variant)
            if info[case,variant]!=teacher['input_perturbation']:
                raise RuntimeError('Frozen visited input metadata differs')
            teachers[case,variant]=teacher
    first=vp['case_ids'][0];runtime.description=info[first,'base']['description'];deadline.check()
    fresh,_=runtime.infer(datums[first,'base'],rows[first]['sampler_seeds'][0],measure=False)
    state['new_parity_queries_this_job']+=1
    replay=float((fresh.float().cpu()-torch.tensor(teachers[first,'base']['action'])).abs().max());del fresh
    if replay!=0:raise RuntimeError('Fresh BF16 visited replay differs')
    state['bf16_visited_parity_max_abs']=replay;state['matched_visited_input_metadata']=200
    ref.save_json(Path(p['remote_root'])/'results/input_reuse.json',{'complete':True,'matched_inputs':200,
        'bf16_replay_max_abs':replay,'image_scope':'frozen raw observations/preprocessing and metadata; old full image tensors not persisted'})
    progress(phase='visited_inputs_verified')
    return datums,info,rows


def evaluate_visited(runtime,banks,torch,np,p,datums,info,rows,deadline,state,progress):
    import visited_reference as vref
    vp=p['visited_source_protocol'];root=Path(p['remote_root']);first=vp['case_ids'][0]
    first_action=None;state['visited_queries']=0
    for case in vp['case_ids']:
        for variant in vref.VARIANTS:
            deadline.check();runtime.description=info[case,variant]['description'];banks.seen.clear()
            action,_=runtime.infer(datums[case,variant],rows[case]['sampler_seeds'][0],measure=False)
            state['visited_queries']+=1
            if len(banks.seen)!=614:raise RuntimeError('All614 per-channel Linears required')
            payload=action_payload(runtime,torch,np,action)
            if first_action is None:first_action=payload['action']
            record={'protocol':p['protocol'],'case_id':case,'suite':rows[case]['suite'],'arm':p['arm'],'variant':variant,
                'seed_index':0,'sampler_seed':rows[case]['sampler_seeds'][0],'plan_row':rows[case],
                'source_checkpoint':str(root/'results/train_eval/deployment.pt'),'pbs_jobid':os.environ['PBS_JOBID'],
                'training_updates':80,'lambda_g':1/6,'input_perturbation':info[case,variant],**payload}
            ref.save_json(root/'results/records'/p['arm']/f'{case}_{variant}.json',record)
            progress(phase='visited_evaluation',visited_completed=state['visited_queries'],visited_case_id=case,variant=variant)
    deadline.check();runtime.description=info[first,'base']['description']
    repeated,_=runtime.infer(datums[first,'base'],rows[first]['sampler_seeds'][0],measure=False)
    state['new_parity_queries_this_job']+=1
    repeat=float((repeated.float().cpu()-torch.tensor(first_action)).abs().max());del repeated
    if repeat!=0:raise RuntimeError('Visited repeat parity differs')
    if runtime.q._native_int4_gemm_calls or runtime.q._kv_active:raise RuntimeError('Unexpected INT4/KV4 path')
    state['visited_repeat_max_abs']=repeat


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--unit',choices=('train_eval','aggregate'),required=True);args=parser.parse_args()
    c80.guard();p=validate_protocol(json.loads(Path(args.protocol).read_text()))
    if args.unit=='aggregate':
        from analysis import aggregate
        aggregate(p);return 0
    return run(p)


if __name__=='__main__':raise SystemExit(main())

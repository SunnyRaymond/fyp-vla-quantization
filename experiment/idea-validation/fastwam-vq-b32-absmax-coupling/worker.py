"""C80 B32 grouped scaling and clipping; every model operation is PBS-only."""
import argparse
import json
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80
import discrete_reference as discrete
import uncoupled_reference as uncoupled


def validate_record(record,p,row,split,seed):
    expected={'protocol':p['protocol'],'arm':p['arm'],'split':split,'updates':80,
        'case_id':row['case_id'],'suite':row['suite'],'plan_row':row,'seed_index':seed,
        'sampler_seed':row['sampler_seeds'][seed],'weight_path':'continuous_B32_scaled_BF16_weights'}
    if any(record.get(k)!=v for k,v in expected.items()):raise RuntimeError('Saved endpoint binding differs')


def run(p):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from metrics_reference import metrics
    from coupled_banks import CoupledBanks
    from checkpointing import atomic_torch_save,capture_rng,restore_rng,optimizer_to_device,to_cpu

    sampler._require_compute_allocation()
    gradient_check=json.loads((Path(os.environ['ARTIFACTS'])/'coupled_selfcheck.json').read_text())
    if gradient_check['pbs_jobid']!=os.environ['PBS_JOBID'] or not all(gradient_check.get(k) is True for k in ('passed','independent_reference_passed','old_surrogate_rejected')):
        raise RuntimeError('Independent absmax coupling selfcheck required before model loading')
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
        print('B32_PROGRESS '+json.dumps(state),flush=True)

    runtime=None;deadline=ref.Deadline();progress()
    try:
        runtime=runner.Runtime(out);runtime.model.eval();runtime.arm('bf16');data={}
        banks=CoupledBanks(runtime,p['calibration']['bank_directory'],deployment['module_masters'],p['clipping'],p['scaling'])
        groups=banks.optimizer_groups(p['training'],p['scaling']);parameters=[v for g in groups for v in g['params']]
        if {id(v) for v in parameters}!={id(v) for v in list(banks.ratios.values())+list(banks.shadows.values())}:
            raise RuntimeError('Only ratios and grouped shadows may be optimized')
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
            action_mse=(action[:10,:6].float()-target['action'][:10,:6].to('cuda').float()).square().mean()
            latent_nmse=latent_reference.future_nmse_loss(torch,future,target['future'].to('cuda'))
            loss=action_mse+p['training']['rho']*p['training']['normalization_c']*latent_nmse
            if len(banks.seen)!=614 or not loss.requires_grad or not bool(torch.isfinite(loss)):raise RuntimeError('Training loss/module coverage failed')
            loss.backward();coverage=banks.coverage()
            if any(v['connected']!=614 or v['finite']!=614 or v['nonzero']<1 for v in coverage.values()):raise RuntimeError('Ratio/group gradients disconnected or nonfinite')
            norm=torch.nn.utils.clip_grad_norm_(parameters,p['training']['gradient_norm_clip'],error_if_nonfinite=True)
            optimizer.step()
            if updates==0 and p['training']['optimizer_reset_after_update1']:optimizer.state.clear()
            banks.project();torch.cuda.synchronize();updates+=1
            history.append({'update':updates,'case_id':case,'seed_index':0,'action_mse':float(action_mse),
                'future_latent_nmse':float(latent_nmse),'loss':float(loss),'gradient_norm':float(norm),
                'coverage':coverage,'seconds':time.monotonic()-started,'pbs_jobid':os.environ['PBS_JOBID'],**counts})
            del action,future,action_mse,latent_nmse,loss
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
        frozen=banks.verify_frozen(deployment['module_masters'])
        atomic_torch_save(torch,out/'deployment.pt',{'protocol':p,'updates':80,
            'baseline_checkpoint':str(Path(p['baseline_root'])/'results/training/rho1.5/deployment.pt'),
            'ratios':banks.clip_snapshot(),'log2_scales':banks.scale_snapshot(),'scale_metadata_dtype':'float32_log2'})
        training={'protocol':p['protocol'],'complete':True,'updates':80,'history':history,'update_order':p['calibration']['update_order'],
            'initial_parity_max_abs':parity,'cached_query_parity':folded_parity,'clipping':banks.clip_summary(),
            'scaling':banks.scaling_summary(),'weight_compensation_rounding':rounding_summary,'gradient_selfcheck':gradient_check,
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
                        'sampler_seed':rows[case]['sampler_seeds'][seed],'weight_path':'continuous_B32_scaled_BF16_weights',
                        'pbs_jobid':os.environ['PBS_JOBID'],'counts':counts,
                        **metrics(torch,latent_reference,action,future,target,p['training'])}
                    ref.save_json(path,record)
                records.append(record);progress(phase='evaluation',split=split,queries=len(records),last_case=case,last_seed=seed)
            ref.save_json(root/'results'/split/p['arm']/'summary.json',{'protocol':p['protocol'],'complete':True,
                'arm':p['arm'],'split':split,'updates':80,'pbs_jobid':os.environ['PBS_JOBID'],
                'clipping':banks.clip_summary(),'scaling':banks.scaling_summary(),**c80.aggregate_records(records,p,split)})
        ref.save_json(out/'summary.json',{**state,'complete':True,'phase':'complete','updates':80,
            'evaluation_queries':160,'cached_query_parity':folded_parity,'clipping':banks.clip_summary(),
            'scaling':banks.scaling_summary(),'gradient_coupling_complete':True,'input_absmax_chain':True,**frozen})
        progress(phase='complete',complete=True,updates=80);print('B32_WORKER_COMPLETE',flush=True);return 0
    except TimeoutError as exc:
        progress(phase='checkpointed',reason=str(exc));return 75
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if runtime is not None and runtime.env is not None:runtime.env.close()


def aggregate(p):
    root=Path(p['remote_root']);source=Path(p['control_root'])
    old=json.loads((source/'results/comparison.json').read_text());sp=p['source_control_protocol']
    if not old['complete'] or old['protocol']!=sp['protocol']:raise RuntimeError('Completed controls required')
    plan=json.loads(Path(p['calibration']['plan_file']).read_text());rows={r['case_id']:r for r in plan['inputs']}
    arms={};changes={};paired={}
    for split in ('calibration','heldout'):
        arms[split]={};records_by_arm={}
        for arm in p['arms']:
            origin=root if arm==p['arm'] else (Path(p['discrete_root']) if arm=='b32_scale_clip' else (Path(p['uncoupled_root']) if arm=='continuous_b32_clip' else source))
            records=[json.loads((origin/'results'/split/arm/'queries'/f'{case}_{seed}.json').read_text()) for case,seed in c80.queries(p,split)]
            for record in records:
                if arm==p['arm']:validate_record(record,p,rows[record['case_id']],split,record['seed_index'])
                elif arm=='continuous_b32_clip':uncoupled.validate_record(record,p['source_uncoupled_protocol'],rows[record['case_id']],split,record['seed_index'])
                elif arm=='b32_scale_clip':discrete.validate_record(record,p['source_discrete_protocol'],rows[record['case_id']],split,record['seed_index'])
                else:c80.validate_binding(record,sp,rows[record['case_id']],split,arm,record['seed_index'])
            stats=c80.aggregate_records(records,p,split)
            saved=json.loads((origin/'results'/split/arm/'summary.json').read_text())
            if not saved['complete'] or any(saved[k]!=v for k,v in stats.items()):raise RuntimeError('Complete query records and summary differ')
            arms[split][arm]=saved;records_by_arm[arm]=records
        changes[split]={};paired[split]={}
        for control in ('baseline','clip_only','b32_scale_clip','continuous_b32_clip'):
            changes[split][control]={m:100*(arms[split][p['arm']]['overall'][m]/arms[split][control]['overall'][m]-1) for m in c80.METRICS}
            paired[split][control]={}
            for metric in ('first10motor_rmse','future_latent_nmse'):
                means={a:{case:sum(r[metric] for r in rr if r['case_id']==case)/sum(r['case_id']==case for r in rr)
                    for case in sorted({r['case_id'] for r in rr})} for a,rr in records_by_arm.items()}
                delta=[means[p['arm']][case]-v for case,v in means[control].items()]
                paired[split][control][metric]={'cases':len(delta),'improved':sum(v<0 for v in delta),'worsened':sum(v>0 for v in delta),'tied':sum(v==0 for v in delta)}
    receipt=json.loads((root/'results/train_eval/training_receipt.json').read_text())
    keys=('codebooks_unchanged','indices_unchanged','row_gains_unchanged','original_exponents_unchanged','continuous_scale_metadata_separate','gradient_coupling_complete','input_absmax_chain')
    if receipt['updates']!=80 or not all(receipt[k] for k in keys) or [h['case_id'] for h in receipt['history']]!=p['calibration']['update_order']:
        raise RuntimeError('Exact80 frozen-content training required')
    if any(v!=0 for v in receipt['cached_query_parity'].values()):raise RuntimeError('Actual cached continuous weight parity required')
    advance=all(changes['heldout'][control][m]<0 for control in ('baseline','clip_only') for m in ('first10motor_rmse','future_latent_nmse'))
    result={'protocol':p['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],'training_updates':80,
        'evaluation_queries':800,'new_teacher_queries':0,'reused_control_endpoint_records':640,
        'arms':arms,'percent_change_vs_controls':changes,'paired_case_counts_after_seed_average':paired,
        'both_primary_better_than_both_controls':advance,
        'coupling_both_primary_better_than_uncoupled':all(changes['heldout']['continuous_b32_clip'][m]<0 for m in ('first10motor_rmse','future_latent_nmse')),'training':receipt,'scope':p['scope']}
    ref.save_json(root/'results/comparison.json',result)
    lines=['# Continuous B32 + clipping80 with dynamic absmax gradient coupling','','原baseline初始化，C80/原loss/seeds/teacher固定；没有新增teacher或development set。',
        '每组原held-out为40 contexts ×2 seeds=80 queries，四suites等权；另完成每组C80的80-query端点评估。','',
        '| Held-out配置 | First10 motor RMSE | Future latent NMSE |','|---|---:|---:|']
    for arm,saved in arms['heldout'].items():
        m=saved['overall'];lines.append(f"| {arm} | {m['first10motor_rmse']:.9f} | {m['future_latent_nmse']:.9f} |")
    lines+=['','新方法相对对照，负值为改善：','']
    for control,d in changes['heldout'].items():lines.append(f"- vs {control}: action {d['first10motor_rmse']:+.3f}%，latent {d['future_latent_nmse']:+.3f}%。")
    lines+=['',f"相对两组对照两项均改善：{advance}。",'Late22/full32/gripper仅诊断，不进入训练loss或推进门槛。',
        '评估使用实际continuous scale补偿后的BF16 weights；codebooks/indices/row gains/original exponents冻结；uncached/cached完整query输出一致；单独保存FP32 log2 scale metadata。',
        '仅fixed-input decoded-BF16 VQ+A4数值证据，无native kernel或closed-loop收益声明。']
    (root/'results/RESULTS.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'results/PIPELINE_COMPLETE').write_text('complete\n');print('B32_AGGREGATE_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--unit',choices=('train_eval','aggregate'),required=True)
    args=parser.parse_args();c80.guard();p=json.loads(Path(args.protocol).read_text())
    if args.unit=='aggregate':aggregate(p);return 0
    return run(p)


if __name__=='__main__':raise SystemExit(main())

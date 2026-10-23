"""C80 clipping-only training and the original paired held-out evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time

import clipping_reference as ref

METRICS=(*ref.METRICS,'first10motor_mse','mixed_loss')


def guard():
    host=socket.gethostname().split('.')[0]
    if not os.environ.get('PBS_JOBID') or 'login' in host:
        raise RuntimeError('A real PBS compute allocation is required')
    if host not in {n.split('.')[0] for n in Path(os.environ['PBS_NODEFILE']).read_text().split()}:
        raise RuntimeError('Host is outside the PBS allocation')


def queries(p,split):
    return [(c,s) for c in p['calibration' if split=='calibration' else 'evaluation']['case_ids']
            for s in ([0] if split=='calibration' else [0,1])]


def aggregate_records(records,p,split):
    keys=[(r['case_id'],r['seed_index']) for r in records]
    if len(keys)!=80 or len(set(keys))!=80 or set(keys)!=set(queries(p,split)):
        raise RuntimeError('Exact80 paired queries required for '+split)
    per_suite=20 if split=='calibration' else 10
    by_suite={}
    for suite in ref.SUITES:
        rows=[r for r in records if r['suite']==suite];cases=sorted({r['case_id'] for r in rows})
        if len(cases)!=per_suite or len(rows)!=20:raise RuntimeError('Four-suite coverage differs')
        by_suite[suite]={m:sum(sum(r[m] for r in rows if r['case_id']==c)
            /sum(r['case_id']==c for r in rows) for c in cases)/per_suite for m in METRICS}
    overall={m:sum(v[m] for v in by_suite.values())/4 for m in METRICS}
    if any(not math.isfinite(v) or v<0 for v in overall.values()):raise RuntimeError('Invalid metrics')
    return {'queries':80,'cases':80 if split=='calibration' else 40,'overall':overall,'by_suite':by_suite}


def validate_binding(record,p,row,split,arm,seed):
    expected={'protocol':p['protocol'],'split':split,'arm':arm,'updates':80 if arm=='clip_only' else 0,
        'plan_row':row,'case_id':row['case_id'],'suite':row['suite'],'seed_index':seed,
        'sampler_seed':row['sampler_seeds'][seed]}
    if any(record.get(k)!=v for k,v in expected.items()):raise RuntimeError('Saved query binding differs')


def run(p,unit):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from metrics_reference import metrics
    from clip_banks import ClipBanks
    from checkpointing import atomic_torch_save,capture_rng,restore_rng,optimizer_to_device,to_cpu

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261017)
    root=Path(p['remote_root']);sp=p['source_clipping_protocol']
    if json.loads((Path(p['clipping_root'])/'protocol.json').read_text())!=sp:
        raise RuntimeError('Source clipping protocol differs')
    data_root,rows,cache,deployment,reference_action=ref.load_inputs(sp,torch)
    source_plan=json.loads(Path(p['calibration']['plan_file']).read_text())
    if source_plan['calibration_cohorts']['c80']!=p['calibration']['case_ids']:
        raise RuntimeError('Frozen C80 cohort differs')
    old_teacher=Path(p['clipping_root'])/'results/teacher'
    receipt=json.loads((old_teacher/'receipt.json').read_text())
    if not receipt['complete'] or receipt['queries']!=80 or receipt['case_ids']!=p['evaluation']['case_ids'] or receipt['seed_indices']!=[0,1]:
        raise RuntimeError('Original shared held-out teacher coverage differs')
    out=root/'results'/unit;out.mkdir(parents=True,exist_ok=True)
    state={'protocol':p['protocol'],'unit':unit,'pbs_jobid':os.environ['PBS_JOBID'],'phase':'loading',
        'complete':False,'updates':0,'new_teacher_queries_this_job':0,'new_evaluation_queries_this_job':0,
        'new_parity_queries_this_job':0}

    def progress(**values):
        state.update(values);ref.save_json(out/'progress.json',state)
        ref.save_json(out/('executions_'+os.environ['PBS_JOBID']+'.json'),state)
        print('C80_PROGRESS '+json.dumps(state),flush=True)

    deadline=ref.Deadline();runtime=None;progress()
    try:
        runtime=runner.Runtime(out);runtime.model.eval();runtime.arm('bf16');data={}

        def datum(case):
            if case not in data:
                obs,meta=calibration_source.load_observation(np,data_root,rows[case])
                data[case]=runtime.datum(obs),meta
            return data[case]

        def teacher_path(split,case,seed):
            return (root/'results/teacher/calibration' if split=='calibration' else old_teacher)/f'{case}_{seed}.pt'

        def load_teacher(split,case,seed):
            saved=torch.load(teacher_path(split,case,seed),map_location='cpu',weights_only=True)
            if saved['plan_row']!=rows[case] or saved['sampler_seed']!=rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Shared teacher binding differs')
            return saved

        def query(case,seed,banks=None):
            deadline.check();value,meta=datum(case)
            return ref.query(runtime,sampler,torch,value,meta['description'],rows[case]['sampler_seeds'][seed],banks)

        if unit=='prepare':
            reused=0
            for index,case in enumerate(p['calibration']['case_ids']):
                path=teacher_path('calibration',case,0)
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.is_file():saved=load_teacher('calibration',case,0)
                else:
                    if str(case) in cache['teacher']:
                        old=cache['teacher'][str(case)]
                        if old['sampler_seed']!=rows[case]['sampler_seeds'][0]:raise RuntimeError('C8 teacher seed differs')
                        saved={**old,'plan_row':rows[case],'reused_from':str(Path(p['baseline_root'])/'results/reference/reference_cache.pt')}
                    else:
                        action,future,counts=query(case,0);state['new_teacher_queries_this_job']+=1
                        saved={'action':action,'future':future,'counts':counts,'pbs_jobid':os.environ['PBS_JOBID'],
                            'plan_row':rows[case],'sampler_seed':rows[case]['sampler_seeds'][0]}
                    atomic_torch_save(torch,path,saved)
                reused+=int('reused_from' in saved)
                progress(phase='teacher',queries=index+1,last_case=case)
            ref.save_json(root/'results/teacher/calibration/receipt.json',{'complete':True,'queries':80,
                'case_ids':p['calibration']['case_ids'],'seed_indices':[0],'reused_c8_objects':reused,
                'pbs_jobid':os.environ['PBS_JOBID']})
        else:
            prep=json.loads((root/'results/prepare/summary.json').read_text())
            if not prep['complete'] or prep['evaluation_queries']!=160:raise RuntimeError('Baseline/teacher preparation incomplete')
        banks=ClipBanks(runtime,p['calibration']['bank_directory'],deployment['module_masters'],'clip_only',p['clipping'])
        if any(v.requires_grad for v in runtime.model.parameters()) or any(
            v.requires_grad for v in list(banks.books.values())+list(banks.logs.values())):
            raise RuntimeError('All model weights, VQ codebooks and gains must be frozen')
        groups=banks.optimizer_groups(p['training']);parameters=[v for g in groups for v in g['params']]
        if len(parameters)!=614 or {id(v) for v in parameters}!={id(v) for v in banks.ratios.values()}:
            raise RuntimeError('Only614 ratios may be optimized')
        first=p['calibration']['case_ids'][0];parity_result=query(first,0,banks)
        state['new_parity_queries_this_job']+=1
        parity=float((parity_result[0].float()-torch.tensor(reference_action['action'],dtype=torch.float32)).abs().max())
        if parity>1e-6:raise RuntimeError('Baseline reference parity failed')
        progress(initial_parity_max_abs=parity,weights_frozen=True,trainable_ratios=614)
        updates=0;history=[]
        if unit=='train_eval':
            optimizer=torch.optim.Adam(groups);checkpoint_path=out/'checkpoint.pt'
            if checkpoint_path.is_file():
                saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
                if saved['protocol']!=p:raise RuntimeError('Resume protocol differs')
                updates,history=saved['updates'],saved['history']
                if [h['case_id'] for h in history]!=p['calibration']['update_order'][:updates]:raise RuntimeError('Resume order differs')
                banks.restore_clips(saved['ratios']);optimizer.load_state_dict(saved['optimizer'])
                optimizer_to_device(torch,optimizer,'cuda');restore_rng(torch,saved['rng'])
            while updates<80:
                deadline.check(max([h['seconds'] for h in history[-3:]],default=300)*1.15)
                case=p['calibration']['update_order'][updates];value,meta=datum(case)
                runtime.description=meta['description'];banks.seen.clear();teacher=load_teacher('calibration',case,0)
                optimizer.zero_grad(set_to_none=True);started=time.monotonic()
                action,counts,future=sampler.differentiable_action(runtime,value,rows[case]['sampler_seeds'][0],checkpointed=True,return_future=True)
                action_mse=(action[:10,:6].float()-teacher['action'][:10,:6].to('cuda').float()).square().mean()
                latent_nmse=latent_reference.future_nmse_loss(torch,future,teacher['future'].to('cuda'))
                loss=action_mse+p['training']['rho']*p['training']['normalization_c']*latent_nmse
                if len(banks.seen)!=614 or not loss.requires_grad or not bool(torch.isfinite(loss)):raise RuntimeError('Training coverage/loss failed')
                loss.backward();coverage=banks.coverage()
                if any(v['connected']!=614 or v['finite']!=614 or v['nonzero']<1 for v in coverage.values()):raise RuntimeError('Gradient coverage failed')
                norm=torch.nn.utils.clip_grad_norm_(parameters,p['training']['gradient_norm_clip'],error_if_nonfinite=True)
                optimizer.step()
                if updates==0 and p['training']['optimizer_reset_after_update1']:optimizer.state.clear()
                banks.project();torch.cuda.synchronize();updates+=1
                history.append({'update':updates,'case_id':case,'seed_index':0,'action_mse':float(action_mse),
                    'future_latent_nmse':float(latent_nmse),'loss':float(loss),'gradient_norm':float(norm),
                    'coverage':coverage,'seconds':time.monotonic()-started,'pbs_jobid':os.environ['PBS_JOBID'],**counts})
                del action,future,action_mse,latent_nmse,loss
                atomic_torch_save(torch,checkpoint_path,{'protocol':p,'updates':updates,'history':history,
                    'ratios':banks.clip_snapshot(),'optimizer':to_cpu(optimizer.state_dict()),'rng':capture_rng(torch)})
                progress(phase='training',updates=updates,last_update=history[-1],clipping=banks.clip_summary())
            for name,master in deployment['module_masters'].items():
                if not torch.equal(banks.books[name].detach().cpu(),master['books']) or not torch.equal(banks.logs[name].detach().cpu(),master['log_scales']):raise RuntimeError('Frozen VQ master changed')
            atomic_torch_save(torch,out/'deployment.pt',{'protocol':p,'updates':80,
                'baseline_checkpoint':str(Path(p['baseline_root'])/'results/training/rho1.5/deployment.pt'),'ratios':banks.clip_snapshot()})
            ratio_values=sorted(float(v) for v in banks.clip_snapshot().values())
            ref.save_json(out/'training_receipt.json',{'protocol':p['protocol'],'complete':True,'updates':80,
                'history':history,'calibration_ids':p['calibration']['case_ids'],'update_order':p['calibration']['update_order'],
                'weights_unchanged':True,'trainable_ratios':614,'initial_parity_max_abs':parity,'clipping':banks.clip_summary(),
                'clipping_distribution':{'median':(ratio_values[306]+ratio_values[307])/2,
                    'below_0.9':sum(v<0.9 for v in ratio_values),'at_minimum':sum(v<=0.5000001 for v in ratio_values)},
                'pbs_jobid':os.environ['PBS_JOBID']})
        else:
            for v in banks.ratios.values():v.requires_grad_(False)
        arm='baseline' if unit=='prepare' else 'clip_only'
        for split in ('calibration','heldout'):
            records=[]
            for case,seed in queries(p,split):
                path=root/'results'/split/arm/'queries'/f'{case}_{seed}.json'
                if path.is_file():
                    record=json.loads(path.read_text());validate_binding(record,p,rows[case],split,arm,seed)
                else:
                    reused=None
                    if arm=='baseline' and split=='heldout':
                        source=Path(p['clipping_root'])/'results/baseline/queries'/f'{case}_{seed}.json'
                        prior=json.loads(source.read_text())
                        if prior['arm']!='baseline' or prior['updates']!=0 or prior['case_id']!=case or prior['sampler_seed']!=rows[case]['sampler_seeds'][seed] or prior['suite']!=rows[case]['suite']:raise RuntimeError('Original baseline binding differs')
                        measured={k:prior[k] for k in ref.METRICS};measured['first10motor_mse']=measured['first10motor_rmse']**2
                        measured['mixed_loss']=measured['first10motor_mse']+p['training']['rho']*p['training']['normalization_c']*measured['future_latent_nmse']
                        counts=prior['counts'];reused=str(source)
                    elif arm=='baseline' and split=='calibration' and case in p['calibration']['c8_reuse_ids'] and case!=first:
                        source=Path(p['ablation_root'])/'results/c8/baseline/queries'/f'{case}_0.json';prior=json.loads(source.read_text())
                        if prior['plan_row']!=rows[case] or prior['sampler_seed']!=rows[case]['sampler_seeds'][seed] or prior['weights']!='baseline' or prior['ratios']!='one':raise RuntimeError('Reusable C8 baseline differs')
                        measured={k:prior[k] for k in METRICS};counts=prior['counts'];reused=str(source)
                    else:
                        teacher=load_teacher(split,case,seed)
                        if arm=='baseline' and split=='calibration' and case==first:action,future,counts=parity_result
                        else:
                            action,future,counts=query(case,seed,banks);state['new_evaluation_queries_this_job']+=1
                        measured=metrics(torch,latent_reference,action,future,teacher,p['training'])
                    record={'protocol':p['protocol'],'arm':arm,'split':split,'updates':updates,'case_id':case,
                        'suite':rows[case]['suite'],'plan_row':rows[case],'seed_index':seed,'sampler_seed':rows[case]['sampler_seeds'][seed],
                        'counts':counts,'pbs_jobid':os.environ['PBS_JOBID'],**measured}
                    if reused:record['reused_from']=reused
                    ref.save_json(path,record)
                records.append(record);progress(phase='evaluation',split=split,arm=arm,queries=len(records),last_case=case,last_seed=seed)
            stats=aggregate_records(records,p,split)
            if arm=='baseline' and split=='heldout':
                original=json.loads((Path(p['clipping_root'])/'results/baseline/summary.json').read_text())
                if any(stats['overall'][m]!=original['overall'][m] for m in ref.METRICS):raise RuntimeError('Original held-out baseline metrics changed')
            ref.save_json(root/'results'/split/arm/'summary.json',{'protocol':p['protocol'],'complete':True,
                'arm':arm,'split':split,'updates':updates,'pbs_jobid':os.environ['PBS_JOBID'],
                'clipping':banks.clip_summary(),'initial_parity_max_abs':parity,**stats})
        ref.save_json(out/'summary.json',{**state,'complete':True,'phase':'complete','updates':updates,
            'evaluation_queries':160,'weights_frozen':True,'clipping':banks.clip_summary()})
        progress(phase='complete',complete=True,updates=updates);print('C80_WORKER_COMPLETE',flush=True);return 0
    except TimeoutError as exc:
        progress(phase='checkpointed',reason=str(exc));return 75
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if runtime is not None and runtime.env is not None:runtime.env.close()


def aggregate(p):
    root=Path(p['remote_root']);plan=json.loads(Path(p['calibration']['plan_file']).read_text());rows={r['case_id']:r for r in plan['inputs']}
    arms={};changes={};paired_cases={}
    for split in ('calibration','heldout'):
        arms[split]={};endpoint_rows={}
        for arm in p['arms']:
            records=[json.loads((root/'results'/split/arm/'queries'/f'{case}_{seed}.json').read_text()) for case,seed in queries(p,split)]
            for r in records:validate_binding(r,p,rows[r['case_id']],split,arm,r['seed_index'])
            stats=aggregate_records(records,p,split);saved=json.loads((root/'results'/split/arm/'summary.json').read_text())
            if not saved['complete'] or any(saved[k]!=v for k,v in stats.items()):raise RuntimeError('Summary differs from complete records')
            arms[split][arm]=saved
            endpoint_rows[arm]=records
        changes[split]={m:100*(arms[split]['clip_only']['overall'][m]/arms[split]['baseline']['overall'][m]-1) for m in METRICS}
        paired_cases[split]={}
        for metric in ('first10motor_rmse','future_latent_nmse'):
            means={arm:{case:sum(r[metric] for r in rr if r['case_id']==case)/sum(r['case_id']==case for r in rr)
                for case in sorted({r['case_id'] for r in rr})} for arm,rr in endpoint_rows.items()}
            delta=[means['clip_only'][c]-v for c,v in means['baseline'].items()]
            paired_cases[split][metric]={'cases':len(delta),'improved':sum(v<0 for v in delta),
                'worsened':sum(v>0 for v in delta),'tied':sum(v==0 for v in delta)}
    receipt=json.loads((root/'results/train_eval/training_receipt.json').read_text())
    if receipt['updates']!=80 or not receipt['weights_unchanged'] or [h['case_id'] for h in receipt['history']]!=p['calibration']['update_order']:raise RuntimeError('Exact80 frozen-weight updates required')
    result={'protocol':p['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],'training_updates':80,
        'evaluation_queries':320,'arms':arms,'percent_change_vs_baseline':changes,
        'paired_case_counts_after_seed_average':paired_cases,
        'heldout_both_primary_improved':all(changes['heldout'][m]<0 for m in ('first10motor_rmse','future_latent_nmse')),
        'training':receipt,'scope':p['scope']}
    ref.save_json(root/'results/comparison.json',result)
    lines=['# C80 clip_only 与原 held-out 结果','','最终 update80，weights 冻结；四 suites 等权。',
        'Calibration 为80 contexts × 1 seed；原 held-out 为40 contexts × 2 seeds。复用原 teacher、seeds 和 baseline。',
        '未使用新的 development set。','',
        '| Split / arm | First10 motor RMSE | Future latent NMSE | Full32 motor RMSE | Mixed loss |','|---|---:|---:|---:|---:|']
    for split in arms:
        for arm,s in arms[split].items():
            o=s['overall'];lines.append(f"| {split} / {arm} | {o['first10motor_rmse']:.9f} | {o['future_latent_nmse']:.9f} | {o['full32motor_rmse']:.9f} | {o['mixed_loss']:.9f} |")
    lines+=['','相对 baseline 变化，负值为改善：','']
    for split,c in changes.items():lines.append(f"- {split}: action {c['first10motor_rmse']:+.3f}%，latent {c['future_latent_nmse']:+.3f}%，mixed loss {c['mixed_loss']:+.3f}%。")
    lines+=['','同一 C80 warm start 增加了数据与更新次数；不能单独归因于数据量。',
        '原 held-out baseline 的附带 action MSE 由保存的逐查询 RMSE 平方还原，mixed loss 包含相应舍入误差；主指标直接复用，完全一致。',
        '原 held-out 已参与多次诊断，本轮只能作为固定 cohort 的工程比较。',
        '仅 fixed-input decoded-BF16 VQ + A4 证据，无 closed-loop 或 native INT4 性能结论。']
    (root/'results/RESULTS.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'results/PIPELINE_COMPLETE').write_text('complete\n');print('C80_AGGREGATE_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--unit',choices=('prepare','train_eval','aggregate'),required=True)
    args=parser.parse_args();guard();p=json.loads(Path(args.protocol).read_text())
    if args.unit=='aggregate':aggregate(p);return 0
    return run(p,args.unit)


if __name__=='__main__':raise SystemExit(main())

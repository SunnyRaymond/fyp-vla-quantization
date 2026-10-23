"""Evaluation-only checkpoint swaps and complete frozen C8 endpoint evaluation."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import time

import clipping_reference as ref


def guard():
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('PBS_JOBID') or 'login' in host:
        raise RuntimeError('A real PBS compute allocation is required')
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().splitlines()
    if host not in {n.split('.')[0] for n in nodes}:
        raise RuntimeError('Host is outside the PBS allocation')


def binding(protocol, unit, split, arm):
    if split == 'heldout':
        sources = protocol['units'][unit]['heldout']
    else:
        sources = {'weights': arm, 'ratios': arm if arm in ('clip_only', 'joint') else 'one'}
    return {'protocol': protocol['protocol'], 'unit': unit, 'split': split,
            'arm': arm, **sources, 'training_updates': 0}


def aggregate_c8(records, protocol):
    expected = {(case, 0) for case in protocol['c8_evaluation']['case_ids']}
    keys = [(r['case_id'], r['seed_index']) for r in records]
    if len(keys) != 8 or len(set(keys)) != 8 or set(keys) != expected:
        raise ValueError('C8 evaluation requires the exact eight calibration contexts')
    if Counter(r['suite'] for r in records) != Counter({s: 2 for s in ref.SUITES}):
        raise ValueError('C8 must retain two contexts per suite')
    metrics = protocol['c8_evaluation']['metrics']
    by_suite = {suite: {m: sum(r[m] for r in records if r['suite'] == suite)/2
                        for m in metrics} for suite in ref.SUITES}
    overall = {m: sum(s[m] for s in by_suite.values())/4 for m in metrics}
    if not all(math.isfinite(v) and v >= 0 for v in overall.values()):
        raise ValueError('Nonfinite or negative C8 metric')
    return {'queries':8, 'cases':8, 'overall':overall, 'by_suite':by_suite}


def metrics(torch, latent_reference, action, future, teacher, training):
    delta = action.float()-teacher['action'].float()
    rmse = lambda x: float(x.square().mean().sqrt())
    mse = float(delta[:10,:6].square().mean())
    latent = float(latent_reference.future_nmse_loss(torch, future, teacher['future']))
    return {'first10motor_mse':mse, 'first10motor_rmse':rmse(delta[:10,:6]),
            'full32motor_rmse':rmse(delta[:,:6]), 'late22motor_rmse':rmse(delta[10:,:6]),
            'first10gripper_rmse':rmse(delta[:10,6]), 'future_latent_nmse':latent,
            'mixed_loss':mse+training['rho']*training['normalization_c']*latent}


def run(protocol, unit):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from clip_banks import ClipBanks

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    root, source = Path(protocol['remote_root']), Path(protocol['source_root'])
    sp = json.loads((source/'protocol.json').read_text())
    if sp != protocol['source_protocol']:
        raise RuntimeError('Frozen clipping protocol changed')
    data_root, by_id, calibration_cache, deployment, reference_action = ref.load_inputs(sp, torch)
    receipt = json.loads((source/'results/teacher/receipt.json').read_text())
    if (not receipt['complete'] or receipt['queries'] != 80 or
            receipt['case_ids'] != protocol['evaluation']['case_ids'] or receipt['seed_indices'] != [0,1]):
        raise RuntimeError('Shared held-out teacher coverage changed')
    out = root/'results'/unit
    out.mkdir(parents=True,exist_ok=True)
    state = {'protocol':protocol['protocol'], 'unit':unit, 'pbs_jobid':os.environ['PBS_JOBID'],
             'complete':False, 'training_updates':0, 'new_teacher_queries':0,
             'new_query_executions_this_job':0, 'phase':'loading'}

    def progress(**values):
        state.update(values)
        ref.save_json(out/'progress.json',state)
        ref.save_json(out/('executions_'+os.environ['PBS_JOBID']+'.json'),state)
        print('ABLATION_PROGRESS '+json.dumps(state),flush=True)

    deadline = ref.Deadline()
    runtime = None
    progress()
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        payloads = {'baseline':deployment}
        required = set(protocol['units'][unit]['c8']) | {'joint'}
        for arm in required-{'baseline'}:
            payload = torch.load(source/'results'/arm/'deployment.pt',map_location='cpu',weights_only=True)
            training_receipt = json.loads((source/'results'/arm/'training_receipt.json').read_text())
            if (payload['protocol'] != sp or payload['arm'] != arm or payload['updates'] != 8 or
                    not training_receipt['complete'] or training_receipt['updates'] != 8 or
                    training_receipt['pbs_jobid'] != protocol['source_jobs'][arm] or
                    set(payload['module_masters']) != set(deployment['module_masters'])):
                raise RuntimeError('Source checkpoint binding changed: '+arm)
            payloads[arm] = payload
        banks = ClipBanks(runtime,sp['calibration']['bank_directory'],deployment['module_masters'],
                          'clip_only',sp['clipping'])
        for value in banks.ratios.values():
            value.requires_grad_(False)
        if any(p.requires_grad for p in runtime.model.parameters()) or any(
                p.requires_grad for p in list(banks.books.values())+list(banks.logs.values())+list(banks.ratios.values())):
            raise RuntimeError('Evaluation must freeze every model/bank parameter')
        ones = {name:torch.tensor(1.) for name in banks.modules}
        data = {}

        def datum(case):
            if case not in data:
                obs, meta = calibration_source.load_observation(np,data_root,by_id[case])
                data[case] = runtime.datum(obs),meta
            return data[case]

        def query(case, seed_index):
            deadline.check()
            value,meta = datum(case)
            started = time.monotonic()
            result = ref.query(runtime,sampler,torch,value,meta['description'],
                               by_id[case]['sampler_seeds'][seed_index],banks)
            state['new_query_executions_this_job'] += 1
            progress(last_query_seconds=time.monotonic()-started)
            return result

        progress(phase='initial_parity')
        first = sp['calibration']['case_ids'][0]
        parity_result = query(first,0)
        expected = torch.tensor(reference_action['action'],dtype=torch.float32)
        parity = float((parity_result[0].float()-expected).abs().max())
        if parity > 1e-6:
            raise RuntimeError('Initial C8 reference action parity failed')
        progress(initial_parity_max_abs=parity,all_parameters_frozen=True)
        completed = []
        assignments = [('c8',arm) for arm in protocol['units'][unit]['c8']]+[('heldout',unit)]
        for split,arm in assignments:
            b = binding(protocol,unit,split,arm)
            banks.restore(payloads[b['weights']]['module_masters'])
            banks.restore_clips(ones if b['ratios']=='one' else payloads[b['ratios']]['ratios'])
            clip_summary = banks.clip_summary()
            if b['ratios']=='one' and (clip_summary['min'] != 1 or clip_summary['max'] != 1):
                raise RuntimeError('Ratio=1 intervention is not exact')
            target = root/'results'/split/arm
            target.mkdir(parents=True,exist_ok=True)
            cases = protocol['evaluation']['case_ids'] if split=='heldout' else sp['calibration']['case_ids']
            seeds = [0,1] if split=='heldout' else [0]
            records = []
            progress(phase='evaluating',split=split,arm=arm,queries=0)
            for case in cases:
                for seed in seeds:
                    path = target/'queries'/f'{case}_{seed}.json'
                    if path.is_file():
                        record = json.loads(path.read_text())
                        if (any(record.get(k)!=v for k,v in b.items()) or
                                record['plan_row'] != by_id[case] or
                                record['sampler_seed'] != by_id[case]['sampler_seeds'][seed]):
                            raise RuntimeError('Saved evaluation query binding differs')
                    else:
                        if split=='heldout':
                            teacher = torch.load(source/'results/teacher'/f'{case}_{seed}.pt',
                                                 map_location='cpu',weights_only=True)
                            if teacher['plan_row'] != by_id[case] or teacher['sampler_seed'] != by_id[case]['sampler_seeds'][seed]:
                                raise RuntimeError('Shared teacher input/seed differs')
                        else:
                            teacher = calibration_cache['teacher'][str(case)]
                        if split=='c8' and arm=='baseline' and case==first:
                            action,future,counts = parity_result
                        else:
                            action,future,counts = query(case,seed)
                        record = {**b,'case_id':case,'seed_index':seed,'suite':by_id[case]['suite'],
                                  'plan_row':by_id[case], 'sampler_seed':by_id[case]['sampler_seeds'][seed],
                                  'pbs_jobid':os.environ['PBS_JOBID'],'counts':counts,
                                  **metrics(torch,latent_reference,action,future,teacher,sp['training'])}
                        ref.save_json(path,record)
                    records.append(record)
                    progress(queries=len(records),last_case=case,last_seed=seed)
            stats = ref.aggregate(records,sp) if split=='heldout' else aggregate_c8(records,protocol)
            summary = {**b,**stats,'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
                       'source_weight_job':protocol['source_jobs'][b['weights']],
                       'source_ratio_job':None if b['ratios']=='one' else protocol['source_jobs'][b['ratios']],
                       'clipping':clip_summary,'initial_parity_max_abs':parity,
                       'source_query_jobs':sorted({r['pbs_jobid'] for r in records})}
            ref.save_json(target/'summary.json',summary)
            completed.append({'split':split,'arm':arm,'queries':len(records)})
            progress(completed=completed)
        summary = {**state,'complete':True,'phase':'complete','completed':completed,
                   'evaluation_queries':sum(r['queries'] for r in completed),
                   'all_parameters_frozen':True,'source_protocol':sp['protocol']}
        ref.save_json(out/'summary.json',summary)
        progress(phase='complete',complete=True)
        return 0
    except TimeoutError as exc:
        progress(phase='checkpointed',reason=str(exc))
        return 75
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


def aggregate(protocol):
    root,source = Path(protocol['remote_root']),Path(protocol['source_root'])
    old = json.loads((source/'results/comparison.json').read_text())
    if not old['complete']:
        raise RuntimeError('Original four-arm results are incomplete')
    heldout,c8 = dict(old['arms']),{}
    for unit,spec in protocol['units'].items():
        worker = json.loads((root/'results'/unit/'summary.json').read_text())
        if not worker['complete'] or worker['training_updates'] != 0 or worker['evaluation_queries'] != 96:
            raise RuntimeError('Each evaluation worker must complete its exact 96 queries')
        for split,arm in [('heldout',unit)]+[('c8',arm) for arm in spec['c8']]:
            b = binding(protocol,unit,split,arm)
            cases = protocol['evaluation']['case_ids'] if split=='heldout' else protocol['calibration']['case_ids']
            seeds = [0,1] if split=='heldout' else [0]
            rows = [json.loads((root/'results'/split/arm/'queries'/f'{case}_{seed}.json').read_text())
                    for case in cases for seed in seeds]
            for row in rows:
                if any(row.get(k)!=v for k,v in b.items()):
                    raise RuntimeError('Aggregate source binding differs')
            calculated = ref.aggregate(rows,protocol['source_protocol']) if split=='heldout' else aggregate_c8(rows,protocol)
            summary = json.loads((root/'results'/split/arm/'summary.json').read_text())
            if any(summary[k]!=v for k,v in calculated.items()) or not summary['complete']:
                raise RuntimeError('Summary does not match complete query records')
            (heldout if split=='heldout' else c8)[arm] = summary
    if set(c8)!=set(protocol['c8_evaluation']['arms']) or len(heldout)!=6:
        raise RuntimeError('All requested endpoint and hybrid arms are required')
    changes = {}
    for arm,summary in heldout.items():
        changes[arm] = {m:100*(summary['overall'][m]/heldout['baseline']['overall'][m]-1)
                        for m in ref.METRICS}
    c8_changes = {arm:{m:100*(s['overall'][m]/c8['baseline']['overall'][m]-1)
                       for m in protocol['c8_evaluation']['metrics']} for arm,s in c8.items()}
    result = {'protocol':protocol['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
              'training_updates':0,'new_teacher_queries':0,'evaluation_queries':192,
              'heldout':heldout,'c8':c8,'heldout_percent_change_vs_baseline':changes,
              'c8_percent_change_vs_baseline':c8_changes,'scope':protocol['scope']}
    ref.save_json(root/'results/comparison.json',result)
    lines = ['# Activation clipping checkpoint 组合与 C8 端点评估','','无重新训练；复用原 shared teacher 和配对 seeds。',
             '', '| Held-out arm | First10 motor RMSE | Future latent NMSE | Late22 motor RMSE |',
             '|---|---:|---:|---:|']
    for arm,s in heldout.items():
        o=s['overall'];lines.append(f"| {arm} | {o['first10motor_rmse']:.9f} | {o['future_latent_nmse']:.9f} | {o['late22motor_rmse']:.9f} |")
    lines += ['', '每组 held-out 均为40 contexts × 2 seeds=80 queries，四suite等权。',
              '', '| C8 final checkpoint | Action MSE | Future latent NMSE | Mixed loss |', '|---|---:|---:|---:|']
    for arm,s in c8.items():
        o=s['overall'];lines.append(f"| {arm} | {o['first10motor_mse']:.9f} | {o['future_latent_nmse']:.9f} | {o['mixed_loss']:.9f} |")
    lines += ['', 'C8每组完整8 contexts、原seed_index=0，每suite两例。Mixed loss沿用rho1.5与冻结normalization_c。',
              '两组新held-out组合160 queries，加四组C8端点32 queries，共192；启动parity query单独记录。',
              '仅fixed-input decoded-BF16 VQ+A4证据；不构成统计显著性、closed-loop或native INT4结论。']
    (root/'results/RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
    (root/'results/PIPELINE_COMPLETE').write_text('complete\n')
    print('ABLATION_AGGREGATE_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--phase',choices=('evaluate','aggregate'),required=True)
    parser.add_argument('--unit')
    args=parser.parse_args()
    guard()
    protocol=json.loads(Path(args.protocol).read_text())
    if args.phase=='aggregate':
        aggregate(protocol)
        return 0
    if args.unit not in protocol['units']:
        parser.error('Unknown evaluation unit')
    return run(protocol,args.unit)


if __name__=='__main__':
    raise SystemExit(main())

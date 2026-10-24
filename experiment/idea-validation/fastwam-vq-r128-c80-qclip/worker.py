"""Frozen final Q0 -> activation-only Qclip, cached-teacher fixed-input evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = 'fastwam-vq-r128-c80-qclip-v1'
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'mixed_az_loss',
           'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse', 'first10gripper_disagreement')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def validate(p):
    t, c, e = p['training'], p['clipping'], p['expected']
    if (p['protocol'] != PROTOCOL or t['updates'] != 80 or t['contexts_per_update'] != 1
            or t['exposures'] != 80 or t['learning_rate'] != .01
            or t['latent_weight'] != .04245452683169643 or t['beta'] != .006083668576461056
            or t['optimizer'] != 'Adam' or t['gradient_norm_clip'] != 1
            or not t['optimizer_reset_after_update1']
            or (c['initial_ratio'], c['minimum_ratio'], c['maximum_ratio']) != (1, .5, 1)
            or p['calibration']['case_ids'] != list(range(2000,2080))
            or Counter(p['calibration']['update_order']) != Counter(range(2000,2080))
            or p['evaluation']['case_ids'] != list(range(3000,3040))
            or p['evaluation']['seed_indices'] != [0,1]
            or e != dict(heldout_queries_per_arm=80,calibration_queries_per_arm=80,
                         target_linears=614,new_teacher_queries=0,closedloop_episodes=0)):
        raise ValueError('Frozen Qclip experiment contract differs')
    return p


def aggregate(records, cases, seeds, suites):
    keys = [(r['case_id'], r['seed_index']) for r in records]
    if Counter(keys) != Counter((c,s) for c in cases for s in seeds):
        raise ValueError('Complete paired endpoint coverage required')
    result = {}
    for suite in suites:
        rows = [r for r in records if r['suite'] == suite]
        ids = sorted({r['case_id'] for r in rows})
        if len(ids) != len(cases)//4:
            raise ValueError('Equal four-suite context coverage required')
        result[suite] = {m:sum(sum(r[m] for r in rows if r['case_id']==c)/len(seeds)
                              for c in ids)/len(ids) for m in METRICS}
        result[suite].update(contexts=len(ids),queries=len(rows))
    return dict(queries=len(records), contexts=len(cases), by_suite=result,
                overall={m:sum(result[s][m] for s in suites)/4 for m in METRICS})


def run(cfg):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import q0_worker as parent
    from clip_bank import ClipBank, selfcheck
    from reader_runtime import reader_session
    from reader_loss import reader_terms

    class Work(parent.Work):
        def progress(self, phase, **extra):
            # Parent initialization uses this hook before performing any workload.
            self.root = Path(cfg['remote_root'])
            self.out = self.unit = self.root/'results'
            self.unit.mkdir(parents=True, exist_ok=True)
            value = dict(protocol=PROTOCOL, pbs_jobid=self.job, phase=phase,
                         counters=self.counters, **extra)
            save(self.out/'progress.json', value)
            print('QCLIP_PROGRESS '+json.dumps(value, allow_nan=False), flush=True)

        def enough(self):
            if self.stopped or time.time()+300 >= self.deadline:
                raise TimeoutError('Committed clipping updates and endpoint records retained')

    parent_p = read(Path(__file__).with_name('parent_protocol.json'))
    work = Work(parent_p, 'reader_kl')
    work.deadline = float(os.environ['FW_JOB_START_EPOCH'])+float(os.environ['FW_SOFT_DEADLINE_SECONDS'])
    counters_path = work.out/'query_counters.json'
    work.counters = dict(student_queries=0,formal_infer_queries=0,reader_queries=0,
                         parameter_reverse_passes=0,new_teacher_queries=0)
    if counters_path.exists():
        prior = read(counters_path)
        if prior.get('protocol') != PROTOCOL:
            raise ValueError('Query counter binding differs')
        work.counters.update(prior['counters'])
    # Parent count writes the parent's protocol; override for this experiment.
    def count(key):
        work.counters[key] += 1
        save(counters_path, dict(protocol=PROTOCOL,counters=work.counters))
    work.count = count
    receipt = read(work.heldout_teacher_root/'receipt.json')
    if not receipt.get('complete') or receipt['case_ids'] != cfg['evaluation']['case_ids'] or receipt['seed_indices'] != [0,1]:
        raise ValueError('Cached heldout teachers differ')
    source_summary = read(cfg['source_summary'])
    if (not source_summary.get('complete') or source_summary.get('pbs_jobid') != cfg['source_job']
            or source_summary.get('arm') != 'reader_kl' or source_summary.get('updates') != 8):
        raise ValueError('Final Q0 checkpoint is not complete')
    prep = read(Path(cfg['parent_root'])/'results/reader_preparation/summary.json')
    if (not prep.get('complete') or prep.get('heldout_used') is not False
            or prep.get('frozen_protocol') != parent_p or prep.get('pbs_jobid') != cfg['source_job']
            or prep.get('calibration_case_ids') != cfg['calibration']['case_ids']
            or prep.get('beta') != cfg['training']['beta']):
        raise ValueError('Frozen reader statistics/preparation binding differs')
    reader_stats = prep['stats']
    work.progress('loading_final_q0')
    work.load_model()
    source = torch.load(cfg['source_checkpoint'], map_location='cpu', weights_only=True)
    if (source['protocol'] != parent_p or source['arm'] != 'reader_kl' or source['updates'] != 8
            or source['target_linears'] != 614 or source['beta'] != cfg['training']['beta']
            or any(not torch.equal(source['rotations'][s],v) for s,v in work.initial_rotation.items())):
        raise ValueError('Q0 deployment binding differs')
    work.bank.restore(source['module_masters'])
    for v in work.bank.parameters():
        v.requires_grad_(False)
        v.grad = None
    check = selfcheck()
    clips = ClipBank(work.bank)
    params = clips.parameters()
    if len(params) != 614 or any(v.requires_grad for v in work.runtime.model.parameters()):
        raise ValueError('Only 614 ratio parameters may train')
    anchor = torch.load(cfg['source_anchor'], map_location='cpu', weights_only=True)
    a,z,_ = work.query(2000)
    parity = max(float((a.float().cpu()-anchor['action'].float()).abs().max()),
                 float((z.float().cpu()-anchor['future'].float()).abs().max()))
    if parity > 1e-6:
        raise ValueError('r=1 final Q0 source action/future parity failed')
    del a,z
    save(work.out/'preflight.json', dict(protocol=PROTOCOL,pbs_jobid=work.job,passed=True,
         initial_parity_max_abs=parity,clipping_selfcheck=check,source_checkpoint=cfg['source_checkpoint']))

    def metrics(action, future, target):
        values = work.metrics(action,future,target)
        values['mixed_az_loss'] = values.pop('mixed_loss')
        actual_commands = work.runtime.command(action.detach().clone())
        teacher_commands = work.runtime.command(target['action'].detach().clone())
        values['first10gripper_disagreement'] = float((actual_commands[:10,6]
            != teacher_commands[:10,6]).mean())
        return values

    def evaluate(arm):
        splits = {}
        for split,cases,seeds in (('calibration',cfg['calibration']['case_ids'],[0]),
                                  ('heldout',cfg['evaluation']['case_ids'],[0,1])):
            records = []
            for case in cases:
                for seed in seeds:
                    work.enough()
                    path = work.out/'endpoints'/arm/split/f'{case}_{seed}.json'
                    binding = dict(protocol=PROTOCOL,arm=arm,split=split,case_id=case,seed_index=seed,
                        sampler_seed=work.rows[case]['sampler_seeds'][seed],plan_row=work.rows[case],
                        source_checkpoint=cfg['source_checkpoint'],suite=work.rows[case]['suite'])
                    if path.exists():
                        row = read(path)
                        if any(row.get(k) != v for k,v in binding.items()):
                            raise ValueError('Reusable endpoint identity differs')
                    else:
                        reused = arm=='q0' and case==2000 and seed==0
                        if reused:
                            action,future = anchor['action'].to('cuda'),anchor['future'].to('cuda')
                        else:
                            action,future,_ = work.query(case,seed)
                        row = dict(binding,pbs_jobid=work.job,reused_source_anchor=reused,
                                   **metrics(action,future,work.teacher(case,seed)))
                        if any(not math.isfinite(row[k]) or row[k]<0 for k in METRICS):
                            raise ValueError('Nonfinite endpoint metric')
                        save(path,row)
                        del action,future
                    records.append(row)
                    work.progress('evaluation',arm=arm,split=split,queries=len(records),case_id=case,seed_index=seed)
            splits[split] = aggregate(records,cases,seeds,cfg['suites'])
        save(work.out/f'{arm}_summary.json',dict(protocol=PROTOCOL,arm=arm,complete=True,**splits))
        return splits

    q0 = evaluate('q0')
    optimizer = torch.optim.Adam(params,lr=cfg['training']['learning_rate'])
    checkpoint = work.out/'checkpoint.pt'
    state = dict(updates=0,exposures=0,history=[])
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['protocol'] != cfg or saved['source_checkpoint'] != cfg['source_checkpoint']:
            raise ValueError('Resume experiment binding differs')
        state = saved['state']
        if [h['case_id'] for h in state['history']] != cfg['calibration']['update_order'][:state['updates']]:
            raise ValueError('Resume update schedule differs')
        clips.restore(saved['ratios'])
        optimizer.load_state_dict(saved['optimizer'])
        work.optimizer_to_device(torch,optimizer,'cuda')
        work.restore_rng(torch,saved['rng'])
    beta = cfg['training']['beta']
    while state['updates'] < 80:
        work.enough()
        case = cfg['calibration']['update_order'][state['updates']]
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        action,future,_ = work.query(case)
        target = work.teacher(case)
        before = metrics(action,future,target)
        a = action.detach().float().requires_grad_(True)
        z = future.detach().float().requires_grad_(True)
        total,_,_ = work.loss(a,z,target)
        ga,gz = torch.autograd.grad(total,(a,z))
        saved_a,saved_z = action.detach().cpu(),future.detach().cpu()
        ga,gz = ga.detach(),gz.detach()
        del action,future,a,z,total
        with reader_session(torch,work.runtime,work.bank,work.masters) as reader:
            leaf = saved_z.to('cuda').float().requires_grad_(True)
            reader_action,_ = work.reader_action(reader,case,leaf,True)
            terms = reader_terms(torch,reader_action,target['action'],
                reader_stats,parent_p['reader_source']['config'])
            kl = float(terms['reader_kl'].detach())
            rg, = torch.autograd.grad(beta*terms['reader_kl'],leaf)
            if not torch.isfinite(rg).all() or bool((rg[:,:,:1]!=0).any()):
                raise ValueError('Reader future gradient invalid')
            gz = gz+rg.detach()
            del leaf,reader_action,terms,rg
        action,future,_ = work.query(case,grad=True)
        replay = max(float((action.detach().float().cpu()-saved_a.float()).abs().max()),
                     float((future.detach().float().cpu()-saved_z.float()).abs().max()))
        if replay != 0:
            raise ValueError('Staged clipping replay differs')
        torch.autograd.backward((action,future),(ga.to(action),gz.to(future)))
        work.count('parameter_reverse_passes')
        connected = sum(v.grad is not None for v in params)
        nonzero = sum(v.grad is not None and bool(v.grad.abs()>0) for v in params)
        if connected != 614 or nonzero == 0:
            raise ValueError('Clipping gradient coverage invalid')
        norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step()
        if state['updates']==0:
            optimizer.state.clear()
        clips.project()
        state['updates'] += 1
        state['exposures'] += 1
        state['history'].append(dict(update=state['updates'],case_id=case,seed_index=0,before=before,
            reader_kl=kl,total_loss=before['mixed_az_loss']+beta*kl,gradient_norm=float(norm),
            gradient_connected=connected,gradient_nonzero=nonzero,replay_max_abs=replay,
            seconds=time.monotonic()-start,pbs_jobid=work.job,clipping=clips.summary()))
        work.atomic_torch_save(torch,checkpoint,dict(protocol=cfg,source_checkpoint=cfg['source_checkpoint'],
            state=state,ratios=clips.snapshot(),optimizer=work.to_cpu(optimizer.state_dict()),rng=work.capture_rng(torch)))
        save(work.out/'training.json',dict(protocol=PROTOCOL,**state))
        work.progress('training',updates=state['updates'],exposures=state['exposures'],clipping=clips.summary())
        del action,future,ga,gz,saved_a,saved_z,target
        gc.collect()
        torch.cuda.empty_cache()
    for v in params:
        v.requires_grad_(False)
        v.grad = None
    optimizer.state.clear()
    weights_frozen = all(torch.equal(work.bank.books[n].detach().cpu(),s['books'])
        and torch.equal(work.bank.logs[n].detach().cpu(),s['log_scales'])
        for n,s in source['module_masters'].items())
    rotation_frozen = all(torch.equal(v,work.bank.rotations.raw[n].detach().cpu())
                          for n,v in work.initial_rotation.items())
    if not weights_frozen or not rotation_frozen:
        raise ValueError('Frozen VQ or rotation changed')
    work.atomic_torch_save(torch,work.out/'deployment.pt',dict(protocol=cfg,ratios=clips.snapshot(),
        source_checkpoint=cfg['source_checkpoint'],updates=80,exposures=80,weights_frozen=True,rotation_frozen=True))
    qclip = evaluate('qclip')
    action,future,_ = work.query(2000)
    value,description = work.datum(2000)
    work.runtime.description = description
    work.count('formal_infer_queries')
    actual,_ = work.runtime.infer(value,work.rows[2000]['sampler_seeds'][0],measure=False)
    formal = float((actual.float().cpu()-action.float().cpu()).abs().max())
    if formal > 1e-6:
        raise ValueError('Clipping formal infer parity failed')
    changes = {s:{m:100*(qclip[s]['overall'][m]/q0[s]['overall'][m]-1)
                 if q0[s]['overall'][m] else None for m in METRICS} for s in q0}
    summary = dict(protocol=PROTOCOL,pbs_jobid=work.job,status='complete',complete=True,
        updates=80,exposures=80,target_linears=614,weights_frozen=True,rotation_frozen=True,
        heldout_queries_per_arm=80,calibration_queries_per_arm=80,new_teacher_queries=0,closedloop_episodes=0,
        initial_parity_max_abs=parity,formal_parity_max_abs=formal,beta=beta,clipping=clips.summary(),
        arms=dict(q0=q0,qclip=qclip),change_percent=changes,counters=work.counters,
        reused_teacher_packets=160,reused_q0_source_anchors=1,scope=cfg['scope'])
    save(work.out/'summary.json',summary)
    lines=['# Q-Clip vs Q0: fixed-input results','','Fixed final80; same frozen weights/R; no closed-loop.',
           '', '| Split | Arm | Motor RMSE | Latent NMSE | Full32 motor RMSE | Gripper RMSE |',
           '|---|---|---:|---:|---:|---:|']
    for split in q0:
        for arm,data in (('Q0',q0),('Qclip',qclip)):
            m=data[split]['overall']
            lines.append(f"| {split} | {arm} | {m['first10motor_rmse']:.9f} | {m['future_latent_nmse']:.9f} | {m['full32motor_rmse']:.9f} | {m['first10gripper_rmse']:.9f} |")
    (work.out/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    work.progress('complete',updates=80,exposures=80)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args()
    cfg=validate(read(args.protocol))
    try:
        run(cfg)
    except BaseException as error:
        save(Path(cfg['remote_root'])/'results/failure.json',dict(protocol=PROTOCOL,
            pbs_jobid=os.environ.get('PBS_JOBID'),error=type(error).__name__,message=str(error),
            checkpointed=isinstance(error,TimeoutError)))
        if isinstance(error,TimeoutError):
            raise SystemExit(75)
        raise


if __name__=='__main__':
    main()

"""R8 -> new C8 moments/MX-VQ -> A+Z VQ8 -> frozen paired 80-query evaluation."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import time

PROTOCOL = 'fastwam-vq-learned-r128-c8-v1'
METRICS = ('first10motor_rmse', 'full32motor_rmse', 'late22motor_rmse',
           'first10gripper_rmse', 'future_latent_nmse')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_protocol(p):
    if (p['protocol'] != PROTOCOL or p['target_linears'] != 614
            or p['calibration']['case_ids'] != [2000,2019,2021,2025,2051,2057,2066,2070]
            or p['evaluation']['case_ids'] != list(range(3000,3040))
            or p['evaluation']['seed_indices'] != [0,1]
            or p['comparison_arms'] != ['H+VQ8','R8+VQ8']):
        raise RuntimeError('Frozen scope differs')
    t, q = p['training'], p['quantization']
    if (t['rho'] != 1.5 or t['normalization_c'] != .028303017887797623
            or t['rotation_updates'] != 8 or t['vq_updates'] != 8
            or not math.isclose(t['latent_weight'],t['rho']*t['normalization_c'],rel_tol=1e-14)
            or q['D'] != 1 or q['clipping_ratio'] != 1 or q['native_kernel']):
        raise RuntimeError('Frozen loss/precision/budget differs')


def run(p):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import numpy as np
    import runner
    import calibration_source as source
    import latent_reference
    import micro_codebook as micro
    from c8_train_banks import TrainableBanks
    from rotation_bank import RotationBanks, VQBanks
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, to_cpu, optimizer_to_device
    from aggregate import write_report
    from selfcheck import run as numerical_selfcheck

    validate_protocol(p)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(p['training']['random_seed'])
    root, parent = Path(p['remote_root']), Path(p['parent_root'])
    out = root / 'results'
    out.mkdir(parents=True, exist_ok=True)
    job_id = os.environ['PBS_JOBID']
    start = time.monotonic()
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['allocation']['soft_deadline_seconds']
    stop = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum))
    if (out/'summary.json').exists() and read(out/'summary.json').get('complete'):
        raise RuntimeError('Experiment is already complete; do not resubmit')
    state_path = out/'state.pt'
    payload = torch.load(state_path, map_location='cpu', weights_only=True) if state_path.exists() else None
    if payload and payload['protocol'] != p:
        raise RuntimeError('Resume protocol differs')
    state = payload['state'] if payload else {'stage':'rotation', 'rotation_updates':0, 'vq_updates':0,
                 'rotation_history':[], 'vq_history':[], 'timing':{}, 'new_teacher_queries':0,
                 'lineage':[], 'created_job_id':job_id}
    state['lineage'].append(job_id)

    def progress(**extra):
        state.update(extra)
        value = {**state, 'protocol':PROTOCOL, 'pbs_jobid':job_id,
                 'wall_seconds_this_job':time.monotonic()-start}
        save(out/'progress.json', value)
        print('R128_PROGRESS '+json.dumps(value),flush=True)

    def enough():
        if stop or time.time()+300 >= deadline:
            progress(status='checkpointed', reason='allocation soft deadline or signal')
            raise TimeoutError('Committed units retained; resume the same experiment')

    progress(status='running', stage=state['stage'])
    checks = numerical_selfcheck()
    save(Path(os.environ['ARTIFACTS'])/'selfcheck.json', {**checks,'pbs_jobid':job_id})
    state['selfcheck'] = checks
    plan = read(p['calibration']['plan_file'])
    if plan.get('status') != 'complete' or plan['calibration_cohorts']['c8'] != p['calibration']['case_ids']:
        raise RuntimeError('Frozen C8 plan changed')
    rows = {r['case_id']:r for r in plan['inputs']}
    cases = p['calibration']['case_ids']
    if Counter(rows[c]['suite'] for c in cases) != Counter({s:2 for s in p['suites']}):
        raise RuntimeError('C8 suite balance differs')
    if Counter(rows[c]['suite'] for c in p['evaluation']['case_ids']) != Counter({s:10 for s in p['suites']}):
        raise RuntimeError('Heldout suite balance differs')
    refroot = parent/'results/reference'
    cache = torch.load(refroot/'reference_cache.pt',map_location='cpu',weights_only=True)
    receipt = read(refroot/'reference_receipt.json')
    if (not receipt.get('complete') or not receipt.get('passed')
            or set(cache['teacher']) != {str(c) for c in cases}
            or cache['normalization']['normalization_c'] != p['training']['normalization_c']
            or cache.get('source_c8',{}).get('source_job_id') != p['calibration']['source_job_id']
            or any(saved['plan_row'] != rows[saved['case_id']] for saved in cache['rows'])):
        raise RuntimeError('Shared C8 reference binding differs')
    source_receipt = read(parent/'results/training/rho1.5/training_receipt.json')
    baseline_protocol = read(Path(p['baseline_root'])/'protocol.json')
    baseline_validation = read(Path(p['baseline_root'])/'results/baseline/validation.json')
    if (not source_receipt.get('complete') or source_receipt.get('updates') != 8
            or source_receipt.get('normalization_c') != p['training']['normalization_c']
            or baseline_protocol.get('parent_root') != str(parent)
            or baseline_protocol.get('parent_arm') != 'rho1.5' or baseline_protocol.get('parent_updates') != 8):
        raise RuntimeError('H+VQ8 is not the exact completed parent')
    if (baseline_validation.get('complete') is not True
            or baseline_validation.get('recomputed_from_query_records') is not True
            or baseline_validation.get('new_model_forwards') != 0):
        raise RuntimeError('H baseline has no completed independent post-cleanup validation')
    teacher_root = Path(p['baseline_root'])/'results/teacher'
    heldout_receipt = read(teacher_root/'receipt.json')
    if (not heldout_receipt.get('complete') or heldout_receipt.get('case_ids') != p['evaluation']['case_ids']
            or heldout_receipt.get('seed_indices') != [0,1]):
        raise RuntimeError('Shared heldout teacher coverage differs')
    baseline = []
    for case in p['evaluation']['case_ids']:
        for seed in (0,1):
            record = read(Path(p['baseline_root'])/'results/baseline/queries'/f'{case}_{seed}.json')
            if (record['case_id'] != case or record['seed_index'] != seed or record['arm'] != 'baseline'
                    or record.get('protocol') != baseline_protocol['protocol'] or record.get('updates') != 0
                    or record['sampler_seed'] != rows[case]['sampler_seeds'][seed]
                    or record['suite'] != rows[case]['suite']):
                raise RuntimeError('H baseline query binding differs')
            baseline.append(record)
    runtime = runner.Runtime(out)
    runtime.model.eval()
    runtime.arm('bf16')
    raw_weights = None
    data = {}

    def datum(case):
        if case not in data:
            observation, metadata = source.load_observation(np,Path(p['calibration']['plan_file']).parent,rows[case])
            data[case] = (runtime.datum(observation),metadata['description'])
        return data[case]

    def query(case, seed, bank, grad=False):
        value, description = datum(case)
        runtime.description, runtime.current_arm = description,'R8+VQ8'
        bank.seen.clear()
        with torch.set_grad_enabled(grad):
            if hasattr(bank,'rotations'):
                # Cayley solves are outside both sampler and Linear checkpoint regions.
                for stream in bank.rotations.raw:
                    bank.rotations.matrix(stream)
            action, counts, future = sampler.differentiable_action(runtime,value,rows[case]['sampler_seeds'][seed],
                                        checkpointed=grad,return_future=True)
        if len(bank.seen) != 614:
            raise RuntimeError('Full sampler did not reach 614 Linears')
        return action, future, counts

    def teacher(case, seed):
        if case in cases:
            return cache['teacher'][str(case)]
        saved = torch.load(teacher_root/f'{case}_{seed}.pt',map_location='cpu',weights_only=True)
        if (saved.get('case_id') != case or saved.get('seed_index') != seed
                or saved.get('plan_row') != rows[case]
                or saved.get('protocol') != baseline_protocol['protocol']
                or saved.get('sampler_seed') != rows[case]['sampler_seeds'][seed]):
            raise RuntimeError('Heldout teacher tensor identity differs')
        if tuple(saved['action'].shape)!=(32,7) or tuple(saved['future'].shape)!=(1,48,3,14,28):
            raise RuntimeError('Teacher tensor shapes differ')
        return saved

    def loss(action, future, target):
        action_mse = (action[:10,:6].float()-target['action'][:10,:6].to('cuda').float()).square().mean()
        latent_nmse = latent_reference.future_nmse_loss(torch,future,target['future'].to('cuda'))
        return action_mse+p['training']['latent_weight']*latent_nmse,action_mse,latent_nmse

    def metrics(action,future,target):
        total,a,z = loss(action,future,target)
        d=action.float()-target['action'].to('cuda').float()
        return {'first10motor_rmse':float(a.sqrt()), 'first10motor_mse':float(a),
                'full32motor_rmse':float(d[:,:6].square().mean().sqrt()),
                'late22motor_rmse':float(d[10:,:6].square().mean().sqrt()),
                'first10gripper_rmse':float(d[:10,6].square().mean().sqrt()),
                'future_latent_nmse':float(z),'mixed_loss':float(total)}

    # A fresh exact parent replay checks checkpoint/decoder/infer identity before any learning.
    if not state.get('parent_replay'):
        enough()
        raw_weights = {n:m.weight.detach().cpu().clone() for n,m in runtime.q.modules.items()}
        deployed = torch.load(parent/'results/training/rho1.5/deployment.pt',map_location='cpu',weights_only=True)
        if (deployed.get('arm')!='rho1.5' or deployed.get('updates')!=8
                or deployed.get('case_ids')!=cases or len(deployed.get('module_masters',{}))!=614):
            raise RuntimeError('Parent deployment identity/coverage differs')
        old = TrainableBanks(runtime,p['calibration']['bank_directory'])
        old.restore(deployed['module_masters'])
        action,future,counts=query(cases[0],0,old)
        saved=read(parent/'results/training/rho1.5/reference_action.json')
        if (saved.get('schema_version')!=1 or saved.get('protocol')!='fastwam-vq-c8-action-latent-plus-v1'
                or saved.get('arm')!='rho1.5' or saved.get('case_id')!=cases[0]
                or saved.get('seed_index')!=0 or saved.get('plan_row')!=rows[cases[0]]
                or saved.get('recipe',{}).get('normalization_c')!=p['training']['normalization_c']
                or saved.get('recipe',{}).get('updates')!=8
                or saved.get('sampler_seed')!=rows[cases[0]]['sampler_seeds'][0]):
            raise RuntimeError('Parent saved action binding differs')
        expected=torch.tensor(saved['action'],device='cuda')
        replay_error=float((action.float()-expected).abs().max())
        value,_=datum(cases[0])
        actual,_=runtime.infer(value,rows[cases[0]]['sampler_seeds'][0],measure=False)
        infer_error=float((actual.float().cpu()-action.float().cpu()).abs().max())
        if replay_error>1e-6 or infer_error>1e-6:
            raise RuntimeError(f'Parent source/infer parity failed {replay_error}/{infer_error}')
        state['parent_replay']={'saved_action_max_abs':replay_error,'formal_infer_max_abs':infer_error}
        for name,mod in runtime.q.modules.items():
            mod.forward=old.forwards[name]
            mod.weight.data=raw_weights[name].to('cuda')
        del old,deployed,action,future,actual,raw_weights
        torch.cuda.empty_cache()

    rb=RotationBanks(runtime,p['calibration']['bank_directory'])
    if payload:
        rb.rotations.restore(payload['rotations'])
    bank=rb
    optimizer=None

    def persist():
        snapshot={'protocol':p,'state':state,'rotations':rb.rotations.snapshot(),
                  'optimizer':to_cpu(optimizer.state_dict()) if optimizer is not None else None,
                  'rng':capture_rng(torch)}
        if isinstance(bank,VQBanks):
            snapshot['module_masters']=bank.snapshot()
        atomic_torch_save(torch,state_path,snapshot)

    if state['stage']=='rotation':
        optimizer=torch.optim.Adam(rb.rotations.parameters(),lr=p['training']['rotation_learning_rate'])
        if payload and payload['optimizer'] is not None:
            optimizer.load_state_dict(payload['optimizer']);optimizer_to_device(torch,optimizer,'cuda')
            restore_rng(torch,payload['rng'])
        while state['rotation_updates']<8:
            enough()
            index=state['rotation_updates'];case=cases[index];tick=time.monotonic()
            rb.rotations.refresh();optimizer.zero_grad(set_to_none=True)
            action,future,counts=query(case,0,rb,True)
            total,a,z=loss(action,future,teacher(case,0))
            if not future.requires_grad or not torch.isfinite(total):
                raise RuntimeError('R loss/future disconnected or nonfinite')
            total.backward()
            gradients={s:bool(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0)
                       for s,v in rb.rotations.raw.items()}
            if not gradients or not all(gradients.values()):
                raise RuntimeError('Rotation gradient failed: '+str(gradients))
            norm=torch.nn.utils.clip_grad_norm_(rb.rotations.parameters(),1,error_if_nonfinite=True)
            optimizer.step();rb.rotations.refresh();torch.cuda.synchronize()
            state['rotation_updates']+=1
            row={'update':state['rotation_updates'],'case_id':case,'action_mse':float(a.detach()),
                 'future_latent_nmse':float(z.detach()),'mixed_loss':float(total.detach()),
                 'gradient_norm':float(norm),'seconds':time.monotonic()-tick,'gradients':gradients,**counts}
            state['rotation_history'].append(row)
            del action,future,total,a,z
            persist();progress(last_update=row)
        state['timing']['rotation_updates_seconds']=sum(r['seconds'] for r in state['rotation_history'])
        state['stage']='moments';optimizer=None
        rb.rotations.freeze();persist()
    else:
        rb.rotations.freeze()
    if max(rb.rotations.orthogonality().values())>5e-5:
        raise RuntimeError('Final R orthogonality failed')
    atomic_torch_save(torch,out/'rotation.pt',{'protocol':p,'updates':8,'raw':rb.rotations.snapshot(),
        'residual_matrices':{s:rb.rotations.matrix(s).detach().cpu() for s in rb.rotations.raw},
        'orthogonality':rb.rotations.orthogonality()})

    momentdir=out/'moments';momentdir.mkdir(exist_ok=True)
    if state['stage']=='moments':
        rb.collect=True
        for case in cases:
            path=momentdir/f'{case}.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved['protocol']!=p or saved['case_id']!=case or len(saved['moments'])!=614:
                    raise RuntimeError('Moment resume binding differs')
                continue
            enough();tick=time.monotonic();rb.moments.clear();rb.moment_rows.clear()
            action,future,counts=query(case,0,rb)
            if len(rb.moments)!=614 or any(v<1 for v in rb.moment_rows.values()):
                raise RuntimeError('Moment layer/row coverage failed')
            moments={n:(v/rb.moment_rows[n]).cpu() for n,v in rb.moments.items()}
            atomic_torch_save(torch,path,{'protocol':p,'case_id':case,'moments':moments,
                'metrics':metrics(action,future,teacher(case,0)),'seconds':time.monotonic()-tick})
            del action,future
            progress(last_moment_case=case)
        rb.collect=False;rb.moments.clear();rb.moment_rows.clear()
        state['stage']='fit';persist()
    bankdir=out/'banks';bankdir.mkdir(exist_ok=True)
    if state['stage']=='fit':
        percontext=[torch.load(momentdir/f'{case}.pt',map_location='cpu',weights_only=True) for case in cases]
        for index,(name,mod) in enumerate(rb.modules.items()):
            path=bankdir/f'{index:04d}.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved['protocol']!=PROTOCOL or saved['module']!=name or saved['module_index']!=index:
                    raise RuntimeError('Fitted bank resume binding differs')
                continue
            enough();tick=time.monotonic()
            with torch.no_grad():
                weight=rb.weight(name)
                moment=torch.stack([v['moments'][name] for v in percontext]).mean(0).to('cuda')
                encoded=micro.fit_vq(weight,second_moment=moment,seed=p['fitting']['seed'],
                        sample_limit=p['fitting']['samples'],iterations=p['fitting']['iterations'])
                micro.check_encoding(encoded)
                decoded=micro.decode_vq(encoded).float()
                relative_nmse=float((decoded-weight).square().sum()/weight.square().sum().clamp_min(1e-12))
            atomic_torch_save(torch,path,{'protocol':PROTOCOL,'module':name,'module_index':index,
                'rotation_updates':8,'encoded':to_cpu(encoded),'weight_nmse':relative_nmse,
                'encoded_bytes':micro.encoded_nbytes(encoded),'seconds':time.monotonic()-tick})
            del weight,moment,encoded,decoded
            progress(fitted_linears=index+1,last_fit_module=name)
        state['stage']='vq';persist()
        del percontext
    rb.close()
    bank=VQBanks(runtime,bankdir,rb.rotations,rb.transforms,PROTOCOL)
    if state['stage']=='vq':
        if payload and 'module_masters' in payload:
            bank.restore(payload['module_masters'])
        optimizer=torch.optim.Adam(bank.parameters(),lr=p['training']['vq_learning_rate'])
        if payload and 'module_masters' in payload and payload['optimizer'] is not None:
            optimizer.load_state_dict(payload['optimizer']);optimizer_to_device(torch,optimizer,'cuda')
            restore_rng(torch,payload['rng'])
        while state['vq_updates']<8:
            enough();index=state['vq_updates'];case=cases[index];tick=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            action,future,counts=query(case,0,bank,True)
            total,a,z=loss(action,future,teacher(case,0))
            if not future.requires_grad or not torch.isfinite(total):
                raise RuntimeError('VQ loss/future disconnected or nonfinite')
            total.backward();coverage=bank.gradient_report()
            if coverage['connected']!=614 or coverage['finite']!=614 or coverage['nonzero']<1:
                raise RuntimeError('VQ gradients failed '+str(coverage))
            norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            if index==0:
                optimizer.state.clear()
            bank.refresh();torch.cuda.synchronize();state['vq_updates']+=1
            row={'update':state['vq_updates'],'case_id':case,'action_mse':float(a.detach()),
                 'future_latent_nmse':float(z.detach()),'mixed_loss':float(total.detach()),
                 'gradient_norm':float(norm),'coverage':coverage,'seconds':time.monotonic()-tick,**counts}
            state['vq_history'].append(row)
            del action,future,total,a,z
            persist();progress(last_update=row)
        state['timing']['vq_updates_seconds']=sum(r['seconds'] for r in state['vq_history'])
        state['stage']='evaluation';optimizer=None;persist()
    elif payload and 'module_masters' in payload:
        bank.restore(payload['module_masters'])
    else:
        raise RuntimeError('Evaluation checkpoint lacks trained VQ masters')
    for value in bank.parameters():
        value.requires_grad_(False)
    atomic_torch_save(torch,out/'deployment.pt',{'protocol':p,'rotation_updates':8,'vq_updates':8,
        'rotations':rb.rotations.snapshot(),'module_masters':bank.snapshot(),'bank_directory':str(bankdir),
        'target_linears':614,'D':1,'r':1})
    if not state.get('deployment_parity'):
        enough();action,future,counts=query(cases[0],0,bank)
        value,_=datum(cases[0]);actual,_=runtime.infer(value,rows[cases[0]]['sampler_seeds'][0],measure=False)
        error=float((action.float().cpu()-actual.float().cpu()).abs().max())
        if error>1e-6:
            raise RuntimeError('Final formal-infer parity failed '+str(error))
        state['deployment_parity']={'formal_infer_max_abs':error}
        del action,future,actual
        persist()
    records=[]
    for split,selected,seeds in (('calibration',cases,[0]),('heldout',p['evaluation']['case_ids'],[0,1])):
        for case in selected:
            for seed in seeds:
                path=out/split/'queries'/f'{case}_{seed}.json'
                binding={'protocol':PROTOCOL,'arm':'R8+VQ8','rotation_updates':8,'vq_updates':8,
                    'case_id':case,'suite':rows[case]['suite'],'seed_index':seed,
                    'sampler_seed':rows[case]['sampler_seeds'][seed],'split':split,'plan_row':rows[case]}
                if path.exists():
                    record=read(path)
                    if any(record.get(k)!=v for k,v in binding.items()):
                        raise RuntimeError('Saved endpoint binding differs')
                else:
                    enough();tick=time.monotonic();action,future,counts=query(case,seed,bank)
                    record={**binding,**metrics(action,future,teacher(case,seed)),
                            'counts':counts,'pbs_jobid':job_id,'seconds':time.monotonic()-tick}
                    if any(not math.isfinite(record[m]) or record[m]<0 for m in METRICS):
                        raise RuntimeError('Invalid endpoint metric')
                    save(path,record);del action,future
                if split=='heldout':records.append(record)
                progress(stage='evaluation',split=split,last_case=case,last_seed=seed,
                         heldout_completed=len(records))
    provenance={'baseline_root':p['baseline_root'],'parent_receipt':str(parent/'results/training/rho1.5/training_receipt.json'),
        'baseline_validation':baseline_validation,
        'shared_teacher':str(teacher_root),'parent_replay':state['parent_replay'],
        'deployment_parity':state['deployment_parity'],'new_teacher_queries':0,'timing':state['timing'],
        'extra_rotation_training':True,'equal_compute':False,'execution':p['rotation']['execution']}
    write_report(out,p,baseline,records,provenance)
    state['stage']='complete';persist()
    summary={'protocol':PROTOCOL,'status':'complete','complete':True,'pbs_jobid':job_id,
        'rotation_updates':8,'vq_updates':8,'target_linears':614,'heldout_queries':80,'calibration_queries':8,
        'all_parameters_frozen':all(not v.requires_grad for v in list(bank.parameters())+list(rb.rotations.parameters())
                                    +list(runtime.model.parameters())),
        'D':1,'r':1,'rho':1.5,'normalization_c':p['training']['normalization_c'],
        'parent_replay':state['parent_replay'],'deployment_parity':state['deployment_parity'],
        'orthogonality':rb.rotations.orthogonality(),'new_teacher_queries':0,'scope':p['scope'],
        'comparison_file':str(out/'comparison.json'),'lineage':state['lineage'],'timing':state['timing']}
    save(out/'summary.json',summary);progress(status='complete',complete=True)
    print('R128_COMPLETE '+json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();p=read(args.protocol)
    try:
        run(p)
    except TimeoutError:
        raise SystemExit(75)
    except Exception as exc:
        path=Path(p['remote_root'])/'results/failure.json'
        save(path,{'protocol':PROTOCOL,'pbs_jobid':os.environ.get('PBS_JOBID'),
                   'error':type(exc).__name__+': '+str(exc),'time':time.time()})
        raise


if __name__=='__main__':
    main()

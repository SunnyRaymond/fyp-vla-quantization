"""C8 finite-difference bilevel experiment. All data/model work requires PBS."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import time

from core import SUITES,METRICS,aggregate,rho_step

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def guard():
    host=socket.gethostname().split('.')[0];node=os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not node or 'login' in host.lower():raise RuntimeError('PBS compute allocation required')
    if host not in {n.split('.')[0] for n in Path(node).read_text().split()}:raise RuntimeError('Host outside PBS allocation')

class Deadline:
    def __init__(self):
        self.end=float(os.environ['FW_JOB_START_EPOCH'])+float(os.environ.get('FW_SOFT_DEADLINE_SECONDS',6300));self.stopped=False
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
    def stop(self,*_):self.stopped=True
    def check(self,seconds=90):
        if self.stopped or time.time()+seconds+180>=self.end:raise TimeoutError('Soft deadline; reuse committed operations')

def same(torch,a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(torch,a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(torch,x,y) for x,y in zip(a,b))
    return a==b

class Engine:
    def __init__(self,p,unit):
        import torch
        import numpy as np
        import sampler,runner,calibration_source,latent_reference
        from checkpointing import atomic_torch_save,capture_rng,restore_rng,optimizer_to_device,to_cpu
        sampler._require_compute_allocation()
        self.torch,self.np,self.sampler,self.source,self.latent=torch,np,sampler,calibration_source,latent_reference
        self.atomic,self.rng,self.set_rng,self.opt_device,self.cpu=atomic_torch_save,capture_rng,restore_rng,optimizer_to_device,to_cpu
        self.p,self.unit,self.root=p,unit,Path(p['remote_root']);self.out=self.root/'results'/unit;self.out.mkdir(parents=True,exist_ok=True)
        self.deadline=Deadline();self.g=self.f=self.teacher_new=0;self.banks=None;self.opt=None;self.data={}
        self.plan=json.loads((self.root/'data/plan.json').read_text())
        if self.plan.get('status')!='complete' or self.plan.get('protocol')!=p['protocol']:raise RuntimeError('Development plan incomplete')
        self.rows={r['case_id']:r for r in self.plan['inputs']}
        reference_receipt=json.loads((Path(p['parent_root'])/'results/reference/reference_receipt.json').read_text())
        teacher_receipt=json.loads((Path(p['heldout_teacher_root'])/'receipt.json').read_text())
        if not reference_receipt.get('complete') or not reference_receipt.get('passed'):raise RuntimeError('C8 reference receipt is incomplete')
        if not teacher_receipt.get('complete') or teacher_receipt.get('queries')!=80 or teacher_receipt.get('case_ids')!=p['evaluation']['historical_case_ids'] or teacher_receipt.get('seed_indices')!=[0,1]:
            raise RuntimeError('Historical teacher receipt differs from frozen40 x2')
        self.cache=torch.load(Path(p['parent_root'])/'results/reference/reference_cache.pt',map_location='cpu',weights_only=True)
        if not math.isclose(self.cache['normalization']['normalization_c'],p['training']['normalization_c'],rel_tol=1e-12):raise RuntimeError('Normalization changed')
        if [r['case_id'] for r in self.cache['rows']]!=p['calibration']['case_ids']:raise RuntimeError('Frozen C8 IDs differ')
        for r in self.cache['rows']:
            if r['plan_row']!=self.rows[r['case_id']]:raise RuntimeError('C8 teacher input identity differs')
        torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(p['training']['seed'])
        self.progress(phase='loading')
        self.runtime=runner.Runtime(self.out);self.runtime.model.eval();self.runtime.arm('bf16')

    def progress(self,**values):
        row={'protocol':self.p['protocol'],'unit':self.unit,'pbs_jobid':os.environ['PBS_JOBID'],
             'new_gradient_queries_this_job':self.g,'new_inference_queries_this_job':self.f,'new_teacher_queries_this_job':self.teacher_new,**values}
        save(self.out/'progress.json',row)
        save(self.out/('execution_'+os.environ['PBS_JOBID']+'.json'),row)
        print('BILEVEL_PROGRESS '+json.dumps(row),flush=True)

    def operation(self,kind,started,**extra):
        row={'kind':kind,'seconds':time.monotonic()-started,'pbs_jobid':os.environ['PBS_JOBID'],**extra}
        with (self.out/('operations_'+os.environ['PBS_JOBID']+'.jsonl')).open('a') as stream:stream.write(json.dumps(row)+'\n')
        self.progress(phase=kind,last_operation=row)

    def datum(self,case):
        if case not in self.data:
            base=self.root/'data' if case>=4000 else Path(self.p['calibration_root'])/'data'
            obs,meta=self.source.load_observation(self.np,base,self.rows[case])
            self.data[case]=self.runtime.datum(obs),meta
        return self.data[case]

    def query(self,case,seed=0):
        self.deadline.check(90);tick=time.monotonic();value,meta=self.datum(case);self.runtime.description=meta['description']
        if self.banks:self.banks.seen.clear()
        action,counts,future=self.sampler.differentiable_action(self.runtime,value,self.rows[case]['sampler_seeds'][seed],checkpointed=False,return_future=True)
        if self.banks and len(self.banks.seen)!=614:raise RuntimeError('Query target coverage differs')
        self.f+=1
        result=action.detach().cpu(),future.detach().cpu()
        del action,future
        self.operation('inference',tick,case_id=case,seed_index=seed,**counts)
        return result

    def teacher_path(self,case,seed):
        if case>=4000:return self.root/'results/teacher'/f'{case}_{seed}.pt'
        return Path(self.p['heldout_teacher_root'])/f'{case}_{seed}.pt'

    def teacher(self,case,seed=0):
        if case in self.p['calibration']['case_ids'] and seed==0:return self.cache['teacher'][str(case)]
        row=self.rows[case];x=self.torch.load(self.teacher_path(case,seed),map_location='cpu',weights_only=True)
        if x['plan_row']!=row or x['sampler_seed']!=row['sampler_seeds'][seed]:raise RuntimeError('Teacher identity mismatch')
        return x

    def make_teachers(self,cases,seeds):
        if self.banks is not None:raise RuntimeError('Teacher generation must precede VQ installation')
        for c in cases:
            for s in seeds:
                path=self.teacher_path(c,s)
                if path.exists():self.teacher(c,s);continue
                if self.unit not in ('preflight','evaluate_low'):raise RuntimeError('This unit cannot write teacher cache')
                action,future=self.query(c,s);path.parent.mkdir(parents=True,exist_ok=True)
                self.atomic(self.torch,path,{'protocol':self.p['protocol'],'plan_row':self.rows[c],
                    'sampler_seed':self.rows[c]['sampler_seeds'][s],'action':action,'future':future,'pbs_jobid':os.environ['PBS_JOBID']})
                self.teacher_new+=1

    def install(self):
        from train_banks import TrainableBanks
        self.banks=TrainableBanks(self.runtime,self.p['calibration']['bank_directory'])
        self.opt=self.torch.optim.Adam(self.banks.parameters(),lr=self.p['training']['learning_rate'])

    def snapshot(self,updates):
        return {'module_masters':self.banks.snapshot(),'optimizer_state':self.cpu(self.opt.state_dict()),
                'rng_state':self.rng(self.torch),'updates':updates,'optimizer_reset_after_update1':updates>=1}

    def restore(self,saved):
        self.banks.restore(saved['module_masters'])
        # Adam keeps CPU step tensors by reference; clone before loading a reusable snapshot.
        self.opt.load_state_dict(self.cpu(saved['optimizer_state']));self.opt_device(self.torch,self.opt,'cuda')
        self.opt.zero_grad(set_to_none=True);self.set_rng(self.torch,saved['rng_state'])

    def train_step(self,case,rho,index,kind):
        self.deadline.check(600);tick=time.monotonic();torch=self.torch;value,meta=self.datum(case)
        self.runtime.description=meta['description'];self.banks.seen.clear();self.opt.zero_grad(set_to_none=True)
        action,counts,future=self.sampler.differentiable_action(self.runtime,value,self.rows[case]['sampler_seeds'][0],checkpointed=True,return_future=True)
        target=self.teacher(case,0)
        action_mse=(action[:10,:6].float()-target['action'][:10,:6].to('cuda').float()).square().mean()
        latent=self.latent.future_nmse_loss(torch,future,target['future'].to('cuda'))
        loss=action_mse+rho*self.p['training']['normalization_c']*latent
        if len(self.banks.seen)!=614 or not bool(torch.isfinite(loss)):raise RuntimeError('Loss or target coverage failed')
        loss.backward();coverage=self.banks.gradient_report()
        if coverage['connected']!=614 or coverage['finite']!=614 or coverage['nonzero']==0:raise RuntimeError('Gradient coverage failed')
        norm=torch.nn.utils.clip_grad_norm_(self.banks.parameters(),self.p['training']['gradient_norm_clip'],error_if_nonfinite=True)
        self.opt.step()
        if index==0:self.opt.state.clear()
        self.banks.refresh();torch.cuda.synchronize();self.g+=1
        row={'case_id':case,'update':index+1,'rho':rho,'action_mse':float(action_mse.detach()),'future_latent_nmse':float(latent.detach()),
             'gradient_norm':float(norm),'coverage':coverage,'optimizer_reset_this_update':index==0,'seconds':time.monotonic()-tick}
        del action,future,action_mse,latent,loss
        self.operation(kind,tick,**row,**counts)
        return row

    def metrics(self,action,future,target):
        delta=action.float()-target['action'].float()
        rmse=lambda x:float(x.square().mean().sqrt())
        return {'first10motor_rmse':rmse(delta[:10,:6]),'full32motor_rmse':rmse(delta[:,:6]),
                'late22motor_rmse':rmse(delta[10:,:6]),'first10gripper_rmse':rmse(delta[:10,6]),
                'future_latent_nmse':float(self.latent.future_nmse_loss(self.torch,future,target['future']))}

    def dev(self,cases):
        rows=[]
        for c in cases:
            action,future=self.query(c);rows.append({'case_id':c,'seed_index':0,'suite':self.rows[c]['suite'],
                **self.metrics(action,future,self.teacher(c))})
        return aggregate(rows,cases,[0]),rows

    def checkpoint(self,path,payload):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);self.atomic(self.torch,path,payload)

    def complete(self,**values):
        result={'protocol':self.p['protocol'],'unit':self.unit,'pbs_jobid':os.environ['PBS_JOBID'],'complete':True,**values}
        save(self.out/'summary.json',result);self.progress(phase='complete',complete=True,**values)
        return result

def legacy_snapshot(e,arm):
    p=e.p;root=Path(p['parent_root'])/'results/training'/arm
    receipt=json.loads((root/'training_receipt.json').read_text())
    payload=e.torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
    deployment=e.torch.load(root/'deployment.pt',map_location='cpu',weights_only=True)
    updates=16 if arm=='rho0.1' else 8;rho=.1 if arm=='rho0.1' else 1.5
    binding=receipt['binding'];state=payload['run_state']
    if not receipt.get('complete') or receipt['updates']!=updates or receipt['case_ids']!=p['calibration']['case_ids']:raise RuntimeError('Source checkpoint coverage differs')
    if binding['learning_rate']!=p['training']['learning_rate'] or binding['gradient_norm_clip']!=p['training']['gradient_norm_clip']:raise RuntimeError('Source optimizer differs')
    if binding['rho']!=rho or binding['normalization_c']!=p['training']['normalization_c']:raise RuntimeError('Source rho/c differs')
    if any(binding['source_c8'].get(k)!=p['calibration'][k] for k in ('case_ids','bank_directory','source_job_id')):raise RuntimeError('Source C8 bank lineage differs')
    if payload['contract']['binding']!=binding or state['updates_completed']!=updates or state['cursor']!=[updates//8,0]:raise RuntimeError('Source cursor/contract differs')
    if not state['optimizer_reset_after_update1'] or not payload['optimizer_state']['state']:raise RuntimeError('Source Adam/reset missing')
    hist=state['training_history']
    if [r['case_id'] for r in hist]!=p['calibration']['case_ids']*(updates//8):raise RuntimeError('Source training order differs')
    if not same(e.torch,payload['module_masters'],deployment['module_masters']):raise RuntimeError('Source deployment/checkpoint differ')
    snap={'module_masters':payload['module_masters'],'optimizer_state':payload['optimizer_state'],'rng_state':payload['rng_state'],
          'updates':updates,'optimizer_reset_after_update1':True}
    return snap,{'arm':arm,'rho':rho,'updates':updates,'checkpoint':str(root/'checkpoint.pt'),
                 'deployment':str(root/'deployment.pt'),'validated':True,'source_job_id':receipt['last_job_id']}

def preflight(e):
    p=e.p;cases=p['development']['outer_case_ids'];e.make_teachers(cases,[0]);e.install()
    initial=e.snapshot(0)
    if not (e.root/'results/initial_state.pt').exists():e.checkpoint(e.root/'results/initial_state.pt',initial)
    low,low_info=legacy_snapshot(e,'rho0.1');warm,warm_info=legacy_snapshot(e,'rho1.5')
    save(e.root/'results/reuse.json',{'protocol':p['protocol'],'complete':True,'low':low_info,'current':warm_info,
        'calibration_teacher_reused':8,'historical_teacher_reused':80,'normalization_c':p['training']['normalization_c']})
    save(e.root/'results/train_low/summary.json',{'protocol':p['protocol'],'complete':True,'updates':16,'rho':.1,
        'checkpoint':low_info['checkpoint'],'source_format':'legacy','new_gradient_queries':0,'reused_updates':16})
    del low
    path=e.out/'evidence.json';evidence=json.loads(path.read_text()) if path.exists() else {}
    for label,snap,batch,repeat in (('initial',initial,p['development']['outer_batches'][0],False),
                                  ('rho1.5_update8',warm,p['development']['outer_batches'][1],True)):
        if label in evidence:continue
        e.deadline.check(2400)
        e.restore(snap);case=p['calibration']['case_ids'][snap['updates']%8]
        before=e.query(case);e.restore(snap)
        probe=rho_step(1.5,1,1,p['bilevel']);values={};restore_checks=[];steps={}
        for side in ('minus','plus','repeat_plus') if repeat else ('minus','plus'):
            e.restore(snap)
            rho=probe['rho_minus' if side=='minus' else 'rho_plus']
            steps[side]=e.train_step(case,rho,snap['updates'],'preflight_virtual_'+side)
            result,_=e.dev(batch);values[side]=result['overall']['first10motor_rmse']
            e.restore(snap);restore_checks.append(same(e.torch,e.snapshot(snap['updates']),snap))
        after=e.query(case)
        parity=max(float((a.float()-b.float()).abs().max()) for a,b in zip(before,after))
        e.restore(snap);restore_checks.append(same(e.torch,e.snapshot(snap['updates']),snap))
        noise=abs(values['repeat_plus']-values['plus']) if repeat else 0.0
        direction=rho_step(1.5,values['minus'],values['plus'],p['bilevel'])
        evidence[label]={'state_updates':snap['updates'],'batch':batch,'steps':steps,'objectives':values,**direction,
            'state_restore_exact':all(restore_checks),'restored_output_max_abs':parity,'branch_replay_difference':noise,
            'signal_exceeds_replay_noise':abs(values['plus']-values['minus'])>3*noise}
        save(path,evidence)
    noise=evidence['rho1.5_update8']['branch_replay_difference']
    restored=all(x['state_restore_exact'] and x['restored_output_max_abs']<=p['preflight']['maximum_restore_output_difference'] for x in evidence.values())
    replay=noise<=p['preflight']['maximum_replay_rmse_difference']
    signal=any(x['signal_above_deadband'] and abs(x['plus_rmse']-x['minus_rmse'])>3*noise for x in evidence.values())
    e.complete(gate_passed=restored and replay and signal,restoration_passed=restored,replay_passed=replay,
               finite_difference_signal_passed=signal,states=list(evidence),evidence_path=str(path))

def search(e):
    p=e.p;e.install();path=e.out/'checkpoint.pt'
    if path.exists():
        saved=e.torch.load(path,map_location='cpu',weights_only=True);state=saved['search_state'];e.restore(saved['snapshot'])
    else:
        initial=e.torch.load(e.root/'results/initial_state.pt',map_location='cpu',weights_only=True);e.restore(initial)
        state={'protocol':p['protocol'],'round':0,'updates':0,'rho':p['bilevel']['initial_rho'],
               'probes':{},'commits_in_round':0,'round_update':None,'history':[],'training_history':[]}
    if state['protocol']!=p['protocol']:raise RuntimeError('Search checkpoint protocol differs')
    def persist():e.checkpoint(path,{'search_state':state,'snapshot':e.snapshot(state['updates'])})
    persist()
    while state['round']<p['bilevel']['rounds']:
        round_index=state['round'];batch=p['development']['outer_batches'][round_index%2]
        if state['round_update'] is None:
            start=e.snapshot(state['updates']);probe=rho_step(state['rho'],1,1,p['bilevel'])
            case=p['calibration']['case_ids'][state['updates']%8]
            for side in ('minus','plus'):
                if side in state['probes']:continue
                # Admission covers the entire virtual update, four queries and restore.
                e.deadline.check(900);e.restore(start)
                step=e.train_step(case,probe['rho_'+side],state['updates'],'search_virtual_'+side)
                result,_=e.dev(batch)
                state['probes'][side]={'rmse':result['overall']['first10motor_rmse'],'step':step,'batch':batch}
                e.restore(start)
                if not same(e.torch,e.snapshot(state['updates']),start):raise RuntimeError('Search virtual state did not restore exactly')
                persist()
            update=rho_step(state['rho'],state['probes']['minus']['rmse'],state['probes']['plus']['rmse'],p['bilevel'])
            state['round_update']={'round':round_index+1,'batch':batch,**update};state['rho']=update['rho_after'];persist()
        while state['commits_in_round']<p['bilevel']['committed_updates_per_round']:
            case=p['calibration']['case_ids'][state['updates']%8]
            step=e.train_step(case,state['rho'],state['updates'],'search_committed')
            state['training_history'].append(step);state['updates']+=1;state['commits_in_round']+=1;persist()
            e.progress(phase='search_committed',round=round_index+1,updates=state['updates'],rho=state['rho'])
        state['history'].append(state['round_update']);state['round']+=1
        state['round_update']=None;state['probes']={};state['commits_in_round']=0;persist()
        save(e.out/'history.json',state['history'])
    if state['updates']!=16 or len(state['history'])!=8:raise RuntimeError('Incomplete bilevel trajectory')
    selected={'protocol':p['protocol'],'complete':True,'rho':state['rho'],'rounds':8,'updates':16,
              'selection':'final round8 rho; independent fixed-rho retraining required','history':state['history'],
              'pbs_jobid':os.environ['PBS_JOBID']}
    save(e.root/'results/selected_rho.json',selected)
    e.complete(rounds=8,updates=16,rho=state['rho'],trajectory_checkpoint=str(path))

def training_info(root,arm):
    row=json.loads((root/'results'/('train_'+arm)/'summary.json').read_text())
    if not row.get('complete') or row.get('updates')!=16:raise RuntimeError('Training endpoint is incomplete: '+arm)
    return row

def read_training(e,info):
    x=e.torch.load(info['checkpoint'],map_location='cpu',weights_only=True)
    return x if info['source_format']=='legacy' else x['snapshot']

def fixed_training(e,arm):
    p=e.p
    rho=p['training']['fixed_rhos']['current'] if arm=='current' else json.loads((e.root/'results/selected_rho.json').read_text())['rho']
    if arm=='selected':
        for other,other_rho in p['training']['fixed_rhos'].items():
            if math.isclose(rho,other_rho,rel_tol=1e-12,abs_tol=1e-15):
                info=training_info(e.root,other)
                e.complete(updates=16,rho=other_rho,checkpoint=info['checkpoint'],source_format=info['source_format'],
                           reused_updates=16,new_committed_updates=0,reused_from=other)
                return
    e.install();path=e.out/'checkpoint.pt';state_path=e.out/'training_state.json'
    if path.exists():
        x=e.torch.load(path,map_location='cpu',weights_only=True);snap=x['snapshot'];state=x['training_state'];e.restore(snap)
        if state['rho']!=rho or state['arm']!=arm:raise RuntimeError('Fixed training binding changed')
    else:
        if arm=='current':snap,reuse=legacy_snapshot(e,'rho1.5');reused=8
        else:
            snap=e.torch.load(e.root/'results/initial_state.pt',map_location='cpu',weights_only=True);reused=0
        e.restore(snap);state={'protocol':p['protocol'],'arm':arm,'rho':rho,'updates':snap['updates'],'reused_updates':reused,'history':[]}
    def persist():
        e.checkpoint(path,{'snapshot':e.snapshot(state['updates']),'training_state':state});save(state_path,state)
    persist()
    while state['updates']<16:
        case=p['calibration']['case_ids'][state['updates']%8]
        step=e.train_step(case,rho,state['updates'],'fixed_committed');state['history'].append(step);state['updates']+=1;persist()
    e.complete(updates=16,rho=rho,checkpoint=str(path),source_format='bilevel',reused_updates=state['reused_updates'],
               new_committed_updates=16-state['reused_updates'])

def load_endpoint(e,arm):
    info=training_info(e.root,arm);x=read_training(e,info)
    masters=x['module_masters']
    if e.banks is None:e.install()
    e.banks.restore(masters);e.opt.zero_grad(set_to_none=True)
    return info

def eval_rows(e,arm,cases,seeds,split,info):
    rows=[];out=e.out/'queries'/arm/split
    for case in cases:
        for seed in seeds:
            path=out/f'{case}_{seed}.json';row=e.rows[case]
            binding={'protocol':e.p['protocol'],'arm':arm,'split':split,'case_id':case,'seed_index':seed,
                     'sampler_seed':row['sampler_seeds'][seed],'checkpoint':info['checkpoint'],'rho':info['rho'],
                     'updates':16,'suite':row['suite'],'plan_row':row}
            if path.exists():
                value=json.loads(path.read_text())
                if any(value.get(k)!=v for k,v in binding.items()):raise RuntimeError('Cached endpoint binding differs')
            else:
                action,future=e.query(case,seed);value={**binding,**e.metrics(action,future,e.teacher(case,seed)),
                    'pbs_jobid':os.environ['PBS_JOBID']};save(path,value)
            rows.append(value)
    return rows

def outer_evaluation(e):
    summaries={}
    for arm in e.p['evaluation']['arms']:
        info=load_endpoint(e,arm);cases=e.p['development']['outer_case_ids']
        rows=eval_rows(e,arm,cases,[0],'outer',info)
        summaries[arm]={'rho':info['rho'],**aggregate(rows,cases,[0])}
    comparator=min(('low','current'),key=lambda a:(summaries[a]['overall']['first10motor_rmse'],summaries[a]['rho']))
    frozen={'protocol':e.p['protocol'],'complete':True,'fixed_comparator':comparator,
            'selection_metric':'outer-development first10 motor RMSE; ties prefer lower rho',
            'summaries':summaries,'queries':24,'pbs_jobid':os.environ['PBS_JOBID']}
    save(e.root/'results/fixed_comparator.json',frozen)
    e.complete(queries=24,comparator_frozen=True,fixed_comparator=comparator,summaries=summaries)

def evaluate(e,arm):
    frozen=json.loads((e.root/'results/fixed_comparator.json').read_text())
    if not frozen.get('complete') or frozen.get('queries')!=24:raise RuntimeError('Comparator must be frozen before confirmation/test')
    cases=e.p['development']['confirmation_case_ids']
    if arm=='low':
        e.make_teachers(cases,[0,1])
        save(e.root/'results/teacher/confirmation_receipt.json',{'protocol':e.p['protocol'],'complete':True,
            'case_ids':cases,'seed_indices':[0,1],'queries':16,'pbs_jobid':os.environ['PBS_JOBID']})
    else:
        receipt=json.loads((e.root/'results/teacher/confirmation_receipt.json').read_text())
        if not receipt.get('complete') or receipt['case_ids']!=cases or receipt['queries']!=16:raise RuntimeError('Confirmation teachers unavailable')
    info=load_endpoint(e,arm);summaries={}
    for split,ids,seeds in (('confirmation',cases,[0,1]),('historical',e.p['evaluation']['historical_case_ids'],[0,1])):
        rows=eval_rows(e,arm,ids,seeds,split,info);summaries[split]=aggregate(rows,ids,seeds)
        save(e.out/(split+'_summary.json'),summaries[split])
    e.complete(arm=arm,rho=info['rho'],updates=16,queries=96,split_queries={'confirmation':16,'historical':80},summaries=summaries)

def aggregate_final(p):
    root=Path(p['remote_root']);out=root/'results/aggregate';out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((root/'results/selected_rho.json').read_text())
    comparator=json.loads((root/'results/fixed_comparator.json').read_text())
    summaries={};records={};endpoint_count=0
    for arm in p['evaluation']['arms']:
        info=training_info(root,arm);summaries[arm]={};records[arm]={}
        for split,unit,ids,seeds in (
            ('outer','outer_evaluation',p['development']['outer_case_ids'],[0]),
            ('confirmation','evaluate_'+arm,p['development']['confirmation_case_ids'],[0,1]),
            ('historical','evaluate_'+arm,p['evaluation']['historical_case_ids'],[0,1])):
            rows=[]
            for c in ids:
                for s in seeds:
                    r=json.loads((root/'results'/unit/'queries'/arm/split/f'{c}_{s}.json').read_text())
                    if r['checkpoint']!=info['checkpoint'] or r['rho']!=info['rho'] or r['updates']!=16:raise RuntimeError('Final endpoint binding differs')
                    rows.append(r)
            summaries[arm][split]=aggregate(rows,ids,seeds);records[arm][split]=rows;endpoint_count+=len(rows)
    fixed=comparator['fixed_comparator'];changes={};wins={};advance=True
    for split in ('confirmation','historical'):
        base=summaries[fixed][split]['overall'];new=summaries['selected'][split]['overall']
        changes[split]={m:100*(new[m]/base[m]-1) if base[m]>0 else (0 if new[m]==0 else None) for m in METRICS}
        advance &= changes[split]['first10motor_rmse'] is not None and changes[split]['first10motor_rmse']<=-1
        advance &= all(changes[split][m] is not None and changes[split][m]<=5 for m in METRICS if m!='first10motor_rmse')
        ids=p['development']['confirmation_case_ids'] if split=='confirmation' else p['evaluation']['historical_case_ids']
        deltas=[]
        for c in ids:
            a=sum(r['first10motor_rmse'] for r in records['selected'][split] if r['case_id']==c)/2
            b=sum(r['first10motor_rmse'] for r in records[fixed][split] if r['case_id']==c)/2
            deltas.append({'case_id':c,'delta':a-b})
        wins[split]={'improved':sum(r['delta']<0 for r in deltas),'worsened':sum(r['delta']>0 for r in deltas),
                     'tied':sum(r['delta']==0 for r in deltas),'case_deltas':deltas}
    operations=[]
    units=('preflight','search','train_current','train_selected','outer_evaluation','evaluate_low','evaluate_current','evaluate_selected')
    for unit in units:
        folder=root/'results'/unit
        if folder.exists():
            for path in folder.glob('operations_*.jsonl'):
                operations.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    gradients=[x for x in operations if x['kind']!='inference'];inferences=[x for x in operations if x['kind']=='inference']
    cost={'new_gradient_queries_including_preflight_and_virtual':len(gradients),'new_inference_queries_including_teacher':len(inferences),
          'gradient_operation_seconds':sum(x['seconds'] for x in gradients),'inference_operation_seconds':sum(x['seconds'] for x in inferences),
          'reused_fixed_training_updates':sum(training_info(root,a).get('reused_updates',0) for a in p['evaluation']['arms']),
          'reused_C8_teacher_queries':8,'reused_historical_teacher_queries':80,
          'scope':'new operation time excludes model loading, bank initialization and checkpoint I/O; virtual/replayed operations included'}
    result={'protocol':p['protocol'],'unit':'aggregate','complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
            'endpoint_records':endpoint_count,'rho_hat':selected['rho'],'fixed_comparator':fixed,'summaries':summaries,
            'percent_change_vs_fixed':changes,'paired_case_wins':wins,'advance_to_C80_or_clipping':bool(advance),
            'gpu_operation_cost':cost,'evidence_boundary':p['evaluation']['evidence']}
    save(root/'results/comparison.json',result);save(out/'summary.json',result)
    lines=['# C8 bilevel rho pilot 结果','',f"rho_hat={selected['rho']:.9g}; development选出的固定对照={fixed}。",'',
        '| Arm | Split | first10 motor RMSE | full32 motor RMSE | late22 motor RMSE | gripper RMSE | future NMSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in p['evaluation']['arms']:
        for split in ('outer','confirmation','historical'):
            m=summaries[arm][split]['overall'];lines.append('| '+arm+' | '+split+' | '+' | '.join(f'{m[k]:.9f}' for k in METRICS)+' |')
    lines+=['',f'工程推进gate：{bool(advance)}。', '',p['evaluation']['evidence']]
    (root/'results/RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
    save(out/'progress.json',{'protocol':p['protocol'],'phase':'complete','complete':True,'pbs_jobid':os.environ['PBS_JOBID']})
    print('BILEVEL_COMPLETE '+json.dumps(result),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--unit',required=True);a.add_argument('--protocol',type=Path,required=True);args=a.parse_args()
    guard();p=json.loads(args.protocol.read_text());root=Path(p['remote_root']);e=None
    try:
        if args.unit=='aggregate':aggregate_final(p);return 0
        if args.unit!='preflight':
            gate=json.loads((root/'results/preflight/summary.json').read_text())
            if gate.get('protocol')!=p['protocol'] or not gate.get('gate_passed'):raise RuntimeError('Preflight gate is mandatory')
        e=Engine(p,args.unit)
        if args.unit=='preflight':preflight(e)
        elif args.unit=='search':search(e)
        elif args.unit in ('train_current','train_selected'):fixed_training(e,args.unit.removeprefix('train_'))
        elif args.unit=='outer_evaluation':outer_evaluation(e)
        elif args.unit.startswith('evaluate_'):evaluate(e,args.unit.removeprefix('evaluate_'))
        else:raise NotImplementedError(args.unit)
        return 0
    except TimeoutError as exc:
        if e:e.progress(phase='checkpointed',reason=str(exc))
        return 75
    except Exception as exc:
        row={'protocol':p['protocol'],'unit':args.unit,'pbs_jobid':os.environ['PBS_JOBID'],'phase':'failed','error':type(exc).__name__+': '+str(exc)}
        save(root/'results'/args.unit/'progress.json',row);raise
    finally:
        if e and e.runtime.env is not None:e.runtime.env.close()

if __name__=='__main__':raise SystemExit(main())

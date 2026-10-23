"""Full-cohort finite capacity intervention, with atomic native/reader checkpoints."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
from common import guard,protocol,contexts,stem,save,receipt,contract


def main(shard):
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import future_adapter
    import latent_reference
    import metrics
    from capacity import Banks
    from checkpointing import atomic_torch_save
    p=protocol(); root=Path(p['remote_root']); prior=Path(p['future_reader_root'])
    cases=[c for c in contexts(p) if c['shard']==shard]; arms=p['allocation_arms']
    allocations=torch.load(root/'allocations.pt',map_location='cpu',weights_only=True)
    if allocations['protocol']!=p['protocol'] or list(allocations['levels'])!=arms: raise RuntimeError('Allocation contract differs')
    canonical=allocations['canonical_arm']
    manifest=json.loads((prior/'input_manifest.json').read_text())
    entries={(c['case_id'],c['seed_index']):c for c in manifest['contexts']}
    old_rows={r['case_id']:r for r in json.loads((Path(p['roots']['old_inputs'])/'plan.json').read_text())['inputs']}
    new_rows={r['case_id']:r for r in json.loads((Path(p['source_root'])/'plan.json').read_text())['inputs']}
    out=root/'results'/f'evaluate-{shard}'; out.mkdir(parents=True,exist_ok=True)
    contract(out/'protocol.json',p)
    stopped=[]; signal.signal(signal.SIGTERM,lambda *_:stopped.append(True))
    deadline=float(os.environ['FW_JOB_START_EPOCH'])+p['walltime_seconds']; durations=[]; queries=0
    def finish(status):
        done=[(arm,c) for arm in arms for c in cases if (out/arm/f'{stem(c)}.json').is_file() and (out/arm/f'{stem(c)}.pt').is_file()]
        native=sum((out/'native'/arm/f'{stem(c)}.pt').is_file() for arm in arms for c in cases)
        s=receipt('evaluate',status=status,shard=shard,row_count=len(done),expected_rows=len(cases)*len(arms),
                  context_count=len({(c['case_id'],c['seed_index']) for _,c in done}),native_capture_count=native,new_queries=queries,episodes=0)
        save(out/'summary.json',s)
    def budget():
        return bool(stopped) or time.time()+1.2*max([120]+durations[-5:])+p['save_reserve_seconds']>=deadline
    def datum(c):
        loc=Path(p['roots']['old_inputs']) if c['case_id']<1000 else Path(p['source_root'])
        row=(old_rows if c['case_id']<1000 else new_rows)[c['case_id']]
        obs,meta=source.load_case(loc,row); runtime.description=meta['description']
        return runtime.datum(obs),row
    if all((out/arm/f'{stem(c)}.json').is_file() for arm in arms for c in cases): finish('complete'); return
    runtime=pilot.Runtime(out); runtime.arm('bf16'); banks=Banks(runtime,p)
    try:
        for arm in arms:
            if canonical[arm]!=arm: continue
            missing=[c for c in cases if not (out/'native'/arm/f'{stem(c)}.pt').is_file()]
            if not missing: continue
            banks.activate(arm,allocations['levels'][arm])
            for c in missing:
                if budget(): finish('resumable'); return
                tick=time.monotonic(); d,row=datum(c); e=entries[c['case_id'],c['seed_index']]
                old=torch.load(e['cache_path'],map_location='cpu',weights_only=True)
                print('CAPACITY_NATIVE',arm,stem(c),flush=True)
                output=future_adapter.run_capture(runtime,d,e['sampler_seed'])
                if not torch.equal(output['_diagnostic']['action_initial_noise'],old['captures']['bf16']['_diagnostic']['action_initial_noise']): raise RuntimeError('Paired noise changed')
                if len(banks.seen)!=614: raise RuntimeError('Incomplete native target coverage')
                native_path=out/'native'/arm/f'{stem(c)}.pt'
                native_path.parent.mkdir(parents=True,exist_ok=True)
                atomic_torch_save(torch,native_path,dict(protocol=p['protocol'],context=c,arm=arm,action=output['action'].cpu(),future=output['video_latents'].cpu(),sampler_seed=e['sampler_seed']))
                queries+=1; durations.append(time.monotonic()-tick); finish('running')
                del output,d,old
        banks.activate('bf16')
        for c in cases:
            d,row=datum(c); e=entries[c['case_id'],c['seed_index']]
            old=torch.load(e['cache_path'],map_location='cpu',weights_only=True); teacher=old['captures']['bf16']
            full_reader_path=next(path for path in (prior/'results/teacher-0'/f'{stem(c)}_action_only.pt',prior/'results/teacher-1'/f'{stem(c)}_action_only.pt') if path.is_file())
            full_reader=torch.load(full_reader_path,map_location='cpu',weights_only=True)['action']
            for arm in arms:
                path=out/arm/f'{stem(c)}.json'
                if path.is_file(): continue
                if budget(): finish('resumable'); return
                if canonical[arm]!=arm:
                    cached=torch.load(out/canonical[arm]/f'{stem(c)}.pt',map_location='cpu',weights_only=True)
                    result=dict(cached['row'],arm=arm,job_id=os.environ['PBS_JOBID'],reused_from_identical_allocation=canonical[arm])
                    path.parent.mkdir(parents=True,exist_ok=True)
                    atomic_torch_save(torch,path.with_suffix('.pt'),dict(row=result,reader_action=cached['reader_action']))
                    save(path,result);finish('running')
                    continue
                tick=time.monotonic()
                native=torch.load(out/'native'/arm/f'{stem(c)}.pt',map_location='cpu',weights_only=True)
                if native['protocol']!=p['protocol'] or native['context']!=c or native['arm']!=arm or native['sampler_seed']!=e['sampler_seed']: raise RuntimeError('Native resume identity differs')
                print('CAPACITY_FIXED_READER',arm,stem(c),flush=True)
                output=future_adapter.run_teacher_future(runtime,d,e['sampler_seed'],native['future'])
                if not output['_diagnostic']['teacher_future_injection_verified'] or not torch.equal(output['_diagnostic']['action_initial_noise'],teacher['_diagnostic']['action_initial_noise']): raise RuntimeError('Future/noise intervention differs')
                result=dict(protocol=p['protocol'],**c,arm=arm,job_id=os.environ['PBS_JOBID'],sampler_seed=e['sampler_seed'],domain=row['domain'],dimension=row['dimension'],
                    budget_bytes=allocations['budget']['total_bytes'],future_nmse=float(latent_reference.future_nmse_loss(torch,native['future'].float(),teacher['video_latents'].float())),
                    added_future_nmse=float((native['future'].float()[:,:,1:]-old['captures']['distilled_self']['video_latents'].float()[:,:,1:]).square().sum()/teacher['video_latents'].float()[:,:,1:].square().sum().clamp_min(1e-12)),
                    reader_drift_from_full_rmse=metrics.action_metrics(torch,output['action'].cpu(),full_reader)['first10motor_rmse'],
                    **{'native_'+k:v for k,v in metrics.action_metrics(torch,native['action'],teacher['action']).items()},
                    **{'reader_'+k:v for k,v in metrics.action_metrics(torch,output['action'].cpu(),teacher['action']).items()})
                path.parent.mkdir(parents=True,exist_ok=True)
                atomic_torch_save(torch,path.with_suffix('.pt'),dict(row=result,reader_action=output['action'].float().cpu()))
                save(path,result); queries+=1; durations.append(time.monotonic()-tick); finish('running')
                del output,native
            del old,teacher,d
            gc.collect()
        finish('complete')
    finally:
        banks.close()
        if runtime.env: runtime.env.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--shard',type=int,choices=range(8),required=True)
    main(a.parse_args().shard)

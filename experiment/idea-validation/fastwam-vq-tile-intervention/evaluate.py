"""Full paired single/multi-tile interventions; atomically resume each query."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from common import guard,protocol,save,receipt,contract,stem,contexts


def mse(torch,a,b):return float((a[:10,:6].double().cpu()-b[:10,:6].double().cpu()).square().mean())


def future_nmse(torch,a,b):
    a=a.double().cpu()[:,:,1:];b=b.double().cpu()[:,:,1:]
    return float((a-b).square().sum()/b.square().sum().clamp_min(1e-12))


def parity(torch,a,b,label,tolerance=0):
    d=a.detach().float().cpu()-b.float().cpu()
    r=dict(max_abs=float(d.abs().max()),rmse=float(d.square().mean().sqrt()))
    if not bool(torch.isfinite(d).all()) or r['max_abs']>tolerance:raise RuntimeError(f'{label} parity differs: {r}')
    return r


def main(shard):
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import future_adapter
    import metrics
    from checkpointing import atomic_torch_save
    from banks import Banks
    p=protocol();root=Path(p['remote_root']);prior=Path(p['future_reader_root'])
    c=contexts(p)[shard];plan=json.loads((root/'plan.json').read_text());cs=plan['conditions']
    out=root/f'results/evaluate-{shard}';out.mkdir(parents=True,exist_ok=True);contract(out/'protocol.json',p)
    stop=[];signal.signal(signal.SIGTERM,lambda *_:stop.append(True));durations=[];queries=0
    def finish(status):
        native=sum((out/'native'/f"{s['id']}.pt").is_file() for s in cs)
        rows=sum((out/'rows'/f"{s['id']}.json").is_file() and (out/'rows'/f"{s['id']}.pt").is_file() for s in cs)
        s=receipt('evaluate',status=status,shard=shard,case_id=c['case_id'],seed_index=0,native_capture_count=native,row_count=rows,expected_rows=54,new_queries=queries,episodes=0)
        save(out/'summary.json',s)
    def budget():return bool(stop) or time.time()+1.2*max([120]+durations[-5:])+p['save_reserve_seconds']>=float(os.environ['FW_JOB_START_EPOCH'])+p['walltime_seconds']
    if all((out/'rows'/f"{s['id']}.json").is_file() and (out/'rows'/f"{s['id']}.pt").is_file() for s in cs):finish('complete');return
    manifest=json.loads((prior/'input_manifest.json').read_text());e=next(x for x in manifest['contexts'] if x['case_id']==c['case_id'] and x['seed_index']==0)
    old=torch.load(e['cache_path'],map_location='cpu',weights_only=True);teacher=old['captures']['bf16'];full=old['captures']['distilled_self']
    rows=json.loads((Path(p['roots']['old_inputs'])/'plan.json').read_text())['inputs'];row=next(r for r in rows if r['case_id']==c['case_id'])
    runtime=pilot.Runtime(out);runtime.arm('bf16');banks=Banks(runtime,p,plan)
    obs,meta=source.load_case(Path(p['roots']['old_inputs']),row);runtime.description=meta['description'];datum=runtime.datum(obs)
    try:
        for cond in cs:
            path=out/'native'/f"{cond['id']}.pt"
            if path.is_file():continue
            if budget():finish('resumable');return
            tick=time.monotonic();banks.patch(cond['tile_ids'])
            print('TILE_NATIVE',c['case_id'],cond['id'],cond['tile_ids'],flush=True)
            output=future_adapter.run_capture(runtime,datum,e['sampler_seed'])
            if len(banks.seen)!=614:raise RuntimeError('Incomplete native module coverage')
            if not torch.equal(output['_diagnostic']['action_initial_noise'],teacher['_diagnostic']['action_initial_noise']):raise RuntimeError('Paired action noise differs')
            if not torch.equal(output['video_latents'][:,:,:1].cpu(),full['video_latents'][:,:,:1]):raise RuntimeError('Observed frame changed')
            checks={}
            if cond['id']=='baseline_start':
                checks=dict(action=parity(torch,output['action'],full['action'],'source action'),future=parity(torch,output['video_latents'],full['video_latents'],'source future'))
            elif cond['id']=='baseline_repeat':
                first=torch.load(out/'native/baseline_start.pt',map_location='cpu',weights_only=True)
                checks=dict(action=parity(torch,output['action'],first['action'],'repeat action',p['parity_max_abs_tolerance']),future=parity(torch,output['video_latents'],first['future'],'repeat future',p['parity_max_abs_tolerance']))
            path.parent.mkdir(exist_ok=True)
            atomic_torch_save(torch,path,dict(protocol=p['protocol'],case_id=c['case_id'],seed_index=0,sampler_seed=e['sampler_seed'],condition=cond,
                action=output['action'].detach().cpu(),future=output['video_latents'].detach().cpu(),parity=checks))
            queries+=1;durations.append(time.monotonic()-tick);finish('running');del output
        banks.activate('bf16')
        expected_path=next(x for x in (prior/'results/teacher-0'/f'{stem(c)}_action_only.pt',prior/'results/teacher-1'/f'{stem(c)}_action_only.pt') if x.is_file())
        expected=torch.load(expected_path,map_location='cpu',weights_only=True)['action']
        for cond in cs:
            path=out/'rows'/f"{cond['id']}.json"
            if path.is_file():continue
            if budget():finish('resumable');return
            tick=time.monotonic();native=torch.load(out/'native'/f"{cond['id']}.pt",map_location='cpu',weights_only=True)
            if native['protocol']!=p['protocol'] or native['case_id']!=c['case_id'] or native['condition']!=cond or native['sampler_seed']!=e['sampler_seed']:raise RuntimeError('Native resume contract differs')
            print('TILE_FIXED_READER',c['case_id'],cond['id'],flush=True)
            output=future_adapter.run_teacher_future(runtime,datum,e['sampler_seed'],native['future'])
            if not output['_diagnostic']['teacher_future_injection_verified'] or not torch.equal(output['_diagnostic']['action_initial_noise'],teacher['_diagnostic']['action_initial_noise']):raise RuntimeError('Reader future/noise intervention differs')
            checks={}
            if cond['id']=='baseline_start':checks=parity(torch,output['action'],expected,'source fixed reader')
            result=dict(protocol=p['protocol'],job_id=os.environ['PBS_JOBID'],**c,sampler_seed=e['sampler_seed'],condition=cond,
                native_mse=mse(torch,native['action'],teacher['action']),reader_mse=mse(torch,output['action'],teacher['action']),future_nmse=future_nmse(torch,native['future'],teacher['video_latents']),
                reader_parity=checks,native_parity=native['parity'],
                **{'native_'+k:v for k,v in metrics.action_metrics(torch,native['action'],teacher['action']).items()},
                **{'reader_'+k:v for k,v in metrics.action_metrics(torch,output['action'].cpu(),teacher['action']).items()})
            path.parent.mkdir(exist_ok=True)
            atomic_torch_save(torch,path.with_suffix('.pt'),dict(row=result,reader_action=output['action'].detach().cpu()))
            save(path,result);queries+=1;durations.append(time.monotonic()-tick);finish('running');del native,output
        finish('complete')
    finally:
        banks.close()
        if runtime.env:runtime.env.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,choices=range(8),required=True);main(a.parse_args().shard)

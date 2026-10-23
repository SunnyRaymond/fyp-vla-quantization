"""Query only matched C on a fixed original BF16 reader, with endpoint replays."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from common import guard,protocol,contexts,stem,save,receipt
from matching import nmse


class Budget(Exception):pass


def main(shard):
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import future_adapter
    import metrics
    from checkpointing import atomic_torch_save
    p=protocol();root=Path(p['remote_root']);expected=[c for c in contexts(p) if c['shard']==shard]
    manifest=json.loads((root/'input_manifest.json').read_text());entries={stem(e):e for e in manifest['contexts']}
    if manifest['protocol']!=p['protocol'] or len(entries)!=60:raise RuntimeError('Incomplete prepared matched cohort')
    out=root/f'results/matched-{shard}';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').is_file() and json.loads((out/'protocol.json').read_text())!=p:raise RuntimeError('Resume protocol changed')
    save(out/'protocol.json',p)
    stop=[];signal.signal(signal.SIGTERM,lambda *_:stop.append(True));durations=[];queries=0;replays=[];runtime=None
    def rows():
        result=[]
        for c in expected:
            path=out/f'{stem(c)}_matched.json'
            if not path.is_file():continue
            r=json.loads(path.read_text())
            if r['protocol']!=p['protocol'] or r['context']!=c or r['sampler_seed']!=entries[stem(c)]['sampler_seed'] or not path.with_suffix('.pt').is_file():raise RuntimeError('Resume row differs')
            result.append(r)
        return result
    def finish(status):
        rr=rows();s=receipt('matched',status=status,shard=shard,reader='bf16',row_count=len(rr),context_count=len(rr),expected_rows=30,
            new_queries_this_segment=queries,replay_count_this_segment=len(replays),replay_max_abs=max([0.]+[r['max_abs'] for r in replays]),
            new_future_generation=0,episodes=0)
        save(out/'summary.json',s)
    def budget():
        if stop or time.time()+1.2*max([90.]+durations[-5:])+p['save_reserve_seconds']>=float(os.environ['FW_JOB_START_EPOCH'])+p['walltime_seconds']:raise Budget()
    def load(c):
        d=torch.load(entries[stem(c)]['prepared_path'],map_location='cpu',weights_only=True)
        if d['protocol']!=p['protocol'] or d['context']!=c or d['sampler_seed']!=entries[stem(c)]['sampler_seed']:raise RuntimeError('Prepared input differs')
        return d
    def datum(c):
        row=old_rows[c['case_id']] if c['case_id']<1000 else new_rows[c['case_id']]
        location=Path(p['roots']['old_inputs']) if c['case_id']<1000 else Path(p['source_root'])
        if row['sampler_seeds'][c['seed_index']]!=entries[stem(c)]['sampler_seed']:raise RuntimeError('Frozen input seed differs')
        obs,meta=source.load_case(location,row);runtime.description=meta['description']
        return runtime.datum(obs)
    def query(c,d,donor):
        nonlocal queries
        budget();tick=time.monotonic();print('MATCH_READER',stem(c),donor,flush=True)
        output=future_adapter.run_teacher_future(runtime,datum(c),d['sampler_seed'],d['futures'][donor])
        diag=output['_diagnostic']
        if not diag['teacher_future_injection_verified'] or diag['video_dit_calls']!=0 or diag['action_dit_calls']!=10 or not torch.equal(diag['action_initial_noise'],d['action_initial_noise']):raise RuntimeError('Future/noise/reader intervention differs')
        if not torch.equal(output['video_latents'][:,:,1:].cpu(),d['futures'][donor][:,:,1:]):raise RuntimeError('Actual injected BF16 future differs')
        queries+=1;durations.append(time.monotonic()-tick)
        return output
    def replay(tag):
        c=expected[0];d=load(c)
        for donor in ('action_only','action_latent'):
            output=query(c,d,donor);delta=output['action'].float().cpu()-d['reader_actions'][donor].float().cpu()
            check=dict(protocol=p['protocol'],job_id=os.environ['PBS_JOBID'],context=c,donor=donor,tag=tag,max_abs=float(delta.abs().max()),rmse=float(delta.square().mean().sqrt()))
            if check['max_abs']!=0 or not bool(torch.isfinite(delta).all()):raise RuntimeError('Source reader replay is not exact: '+str(check))
            save(out/'replays'/f"{os.environ['PBS_JOBID']}_{tag}_{donor}.json",check);replays.append(check)
            del output
    try:
        budget();torch.backends.cuda.matmul.allow_tf32=False
        old_rows={r['case_id']:r for r in json.loads((Path(p['roots']['old_inputs'])/'plan.json').read_text())['inputs']}
        new_rows={r['case_id']:r for r in json.loads((Path(p['source_root'])/'plan.json').read_text())['inputs']}
        runtime=pilot.Runtime(out);runtime.arm('bf16')
        if runtime.model.torch_dtype!=torch.bfloat16:raise RuntimeError('Actual reader latent dtype changed')
        replay('start')
        for c in expected:
            path=out/f'{stem(c)}_matched.json'
            if path.is_file():continue
            d=load(c);output=query(c,d,'matched');actual=nmse(torch,output['video_latents'].cpu(),d['teacher_future'])
            check=d['matching']
            if abs(actual-check['b_nmse'])>check['matching_tolerance']:raise RuntimeError('Reader-injected future NMSE did not match B')
            r=dict(protocol=p['protocol'],job_id=os.environ['PBS_JOBID'],context=c,**c,sampler_seed=d['sampler_seed'],reader='bf16',donor='matched',
                domain=entries[stem(c)]['domain'],dimension=entries[stem(c)]['dimension'],future_injection_verified=True,video_dit_calls=0,action_dit_calls=10,
                injected_future_nmse=actual,matching=check,**metrics.action_metrics(torch,output['action'],d['teacher_action']))
            atomic_torch_save(torch,path.with_suffix('.pt'),dict(row=r,action=output['action'].float().cpu()))
            save(path,r);finish('running');del d,output
        replay('end')
        if len(rows())!=30 or len(replays)!=4:raise RuntimeError('Incomplete matched reader shard')
        finish('complete')
    except Budget:finish('resumable')
    finally:
        if runtime and runtime.env:runtime.env.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,choices=(0,1),required=True);main(a.parse_args().shard)

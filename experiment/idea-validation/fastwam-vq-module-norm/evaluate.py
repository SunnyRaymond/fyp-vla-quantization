"""Finite, equal-energy probes of shared versus untied codeword updates."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
from common import PROTOCOL, guard, save, contexts, weight_vectors


def variants(p):
    return [(f'{profile}_{kind}_{level}',kind,eps)
            for profile in p['allocation_profiles']
            for level,eps in enumerate(p['intervention_step_norms'])
            for kind in ('shared','untied','random')]


def key(c,label):
    return f"case_{c['case_id']}_seed_{c['seed_index']}_{label}"



def run(shard):
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import future_adapter
    import metrics
    from reader_support import ReaderWeights
    from checkpointing import atomic_torch_save,to_cpu
    import fast_a4
    import transform_checks
    transform_checks.run(Path(os.environ['ARTIFACTS']) / 'transform_check.json')
    root=Path(os.environ['ROOT']); p=json.loads((root/'protocol.json').read_text())
    expected=[c for c in contexts(p) if c['shard']==shard]
    out=root/'results'/f'evaluate-{shard}'; out.mkdir(parents=True,exist_ok=True)
    contract=out/'protocol.json'
    if contract.exists() and json.loads(contract.read_text())!=p: raise RuntimeError('Resume intervention protocol differs')
    save(contract,p)
    labels=[v[0] for v in variants(p)]
    job=os.environ['PBS_JOBID']; deadline=float(os.environ['FW_JOB_START_EPOCH'])+p['walltime_seconds']
    durations=[]; queries=0; interrupted=[]
    signal.signal(signal.SIGTERM,lambda *_:interrupted.append(True))

    def receipt(status):
        complete=[(c,label) for c in expected for label in labels if (out/f'{key(c,label)}.json').exists()]
        generated=sum((out/f'{key(c,label)}.pt').exists() or (out/f'{key(c,label)}.json').exists() for c in expected for label in labels)
        s=dict(protocol=PROTOCOL,phase='evaluate',shard=shard,job_id=job,status=status,
               row_count=len(complete),expected_rows=len(expected)*12,context_count=len(expected),
               generated=generated,new_queries=queries,reused_rows=sum(bool(json.loads((out/f'{key(c,label)}.json').read_text()).get('reused')) for c,label in complete),teacher_reader_unquantized=True,
               runtime_transform_implementation=fast_a4.IMPLEMENTATION)
        save(out/'summary.json',s); save(Path(os.environ['ARTIFACTS'])/'summary.json',s)

    def budget():
        if interrupted or time.time()+1.2*max([90]+durations[-5:])+p['save_reserve_seconds']>=deadline:
            receipt('resumable'); return True
        return False

    if all((out/f'{key(c,l)}.json').exists() for c in expected for l in labels):
        receipt('complete'); return
    manifest=json.loads((Path(p['future_reader_root'])/'input_manifest.json').read_text())
    entries={(c['case_id'],c['seed_index']):c for c in manifest['contexts']}
    old_root=Path(p['roots']['old_inputs'])
    oldrows={r['case_id']:r for r in json.loads((old_root/'plan.json').read_text())['inputs']}
    newrows={r['case_id']:r for r in json.loads((Path(p['source_root'])/'plan.json').read_text())['inputs']}
    runtime=pilot.Runtime(out); runtime.arm('bf16'); runtime.q.disable()
    weights=None

    def datum(c):
        where=old_root if c['case_id']<1000 else Path(p['source_root'])
        row=(oldrows if c['case_id']<1000 else newrows)[c['case_id']]
        observation,meta=source.load_case(where,row)
        runtime.description=meta['description']
        return runtime.datum(observation)

    try:
        need_generation=any(not (out/f'{key(c,l)}.json').exists() and not (out/f'{key(c,l)}.pt').exists() for c in expected for l in labels)
        if need_generation:
            weights=ReaderWeights(runtime,p,'action_only'); fast_a4.install_reader(weights); weights.activate('action_only')
            probes=torch.load(root/'results/prepare/updates.pt',weights_only=True,map_location='cpu')
            updates,norms=probes['values'],probes['norms']
            for name,ids in probes['flat_ids'].items():
                if not torch.equal(weight_vectors(torch,weights.weights['action_only'][name])[ids],probes['baseline'][name]):
                    raise RuntimeError('Global probe decoder baseline differs from GPU decoded weights')
            normpath=out/'step_norms.json'
            if normpath.exists() and json.loads(normpath.read_text())!=norms:
                raise RuntimeError('Resumed perturbation construction changed')
            save(normpath,norms)
            for label,kind,eps in variants(p):
                for name,ids in probes['flat_ids'].items():
                    weights.runtime.q.modules[name].weight.data=weights.weights['action_only'][name].to('cuda')
                    parameter=weights.runtime.q.modules[name].weight.data
                    view=weight_vectors(torch,parameter)
                    view.index_copy_(0,ids.to('cuda'),updates[label][name].to('cuda'))
                    parameter.copy_(view.reshape(parameter.shape[0],-1)[:,:parameter.shape[1]])
                for c in expected:
                    target=out/f'{key(c,label)}.pt'
                    if target.exists() or (out/f'{key(c,label)}.json').exists(): continue
                    if budget(): return
                    d=datum(c); entry=entries[c['case_id'],c['seed_index']]
                    old=torch.load(entry['cache_path'],weights_only=True,map_location='cpu')
                    tick=time.monotonic(); print('GENERATE_PROBE',label,c['case_id'],c['seed_index'],flush=True)
                    output=to_cpu(future_adapter.run_capture(runtime,d,entry['sampler_seed']))
                    if not torch.equal(output['_diagnostic']['action_initial_noise'],old['captures']['bf16']['_diagnostic']['action_initial_noise']):
                        raise RuntimeError('Finite intervention changed paired action noise')
                    baseline=old['captures']['distilled_self']; teacher=old['captures']['bf16']
                    row=dict(protocol=PROTOCOL,context=c,**c,label=label,kind=kind,epsilon=eps,allocation_profile=label.split('_')[0],reused=False,sampler_seed=entry['sampler_seed'],job_id=job,transform_implementation=fast_a4.IMPLEMENTATION,
                        future_nmse=metrics.latent_metrics(torch,output['video_latents'],teacher['video_latents'])['future_latent_nmse'],
                        baseline_future_nmse=metrics.latent_metrics(torch,baseline['video_latents'],teacher['video_latents'])['future_latent_nmse'],
                        native=metrics.action_metrics(torch,output['action'],teacher['action']),
                        baseline_native=metrics.action_metrics(torch,baseline['action'],teacher['action']))
                    if not torch.equal(output['video_latents'][:,:,0],teacher['video_latents'][:,:,0]):
                        raise RuntimeError('Observation frame0 clamp changed')
                    atomic_torch_save(torch,target,dict(protocol=PROTOCOL,context=c,label=label,row=row,output=output))
                    durations.append(time.monotonic()-tick); queries+=1; receipt('running')
                    del output,old,d
            del updates,probes
            weights.activate('bf16')
        for c in expected:
            d=datum(c); entry=entries[c['case_id'],c['seed_index']]
            old=torch.load(entry['cache_path'],weights_only=True,map_location='cpu')
            basepath=Path(p['future_reader_root'])/'results'/f"teacher-{entry['shard']}"/f"case_{c['case_id']}_seed_{c['seed_index']}_action_only.json"
            baseline_teacher=json.loads(basepath.read_text())
            for label in labels:
                final=out/f'{key(c,label)}.json'
                if final.exists(): continue
                if budget(): return
                payload=torch.load(out/f'{key(c,label)}.pt',weights_only=True,map_location='cpu')
                tick=time.monotonic(); print('TEACHER_PROBE',label,c['case_id'],c['seed_index'],flush=True)
                output=to_cpu(future_adapter.run_teacher_future(runtime,d,entry['sampler_seed'],payload['output']['video_latents']))
                diag=output['_diagnostic']
                if not diag['teacher_future_injection_verified'] or diag['video_dit_calls']!=0 or diag['action_dit_calls']!=10:
                    raise RuntimeError('Teacher reader intervention failed')
                if not torch.equal(diag['action_initial_noise'],old['captures']['bf16']['_diagnostic']['action_initial_noise']):
                    raise RuntimeError('Teacher reader noise changed')
                row=payload['row']
                row.update(teacher=metrics.action_metrics(torch,output['action'],old['captures']['bf16']['action']),
                           baseline_teacher={k:baseline_teacher[k] for k in ('first10motor_rmse','full32motor_rmse','late22motor_rmse','first10gripper_rmse')},
                           teacher_reader_unquantized=True,future_injection_verified=True)
                save(final,row)
                durations.append(time.monotonic()-tick); queries+=1; receipt('running')
                del payload,output
            del old,d
            gc.collect()
        receipt('complete')
    finally:
        if weights: weights.close()
        if runtime.env: runtime.env.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--shard',type=int,choices=range(4),required=True)
    run(a.parse_args().shard)

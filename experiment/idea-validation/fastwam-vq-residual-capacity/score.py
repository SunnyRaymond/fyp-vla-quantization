"""Calibration-only residual damage through native, future and fixed-reader paths."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
from common import guard,protocol,stem,save,receipt,contract


def main(shard):
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import sampler
    import latent_reference
    from checkpointing import atomic_torch_save
    from capacity import Banks,differentiable_reader,parity
    p=protocol(); root=Path(p['remote_root']); prior=Path(p['future_reader_root'])
    cases=p['calibration_cases'][shard::4]
    if len(cases)!=2: raise RuntimeError('Score shard must contain2 calibration contexts')
    out=root/'results'/f'score-{shard}'; out.mkdir(parents=True,exist_ok=True)
    contract(out/'protocol.json',p)
    prep=json.loads((root/'prepare_summary.json').read_text())
    manifest=json.loads((prior/'input_manifest.json').read_text())
    entries={(c['case_id'],c['seed_index']):c for c in manifest['contexts']}
    rows={r['case_id']:r for r in json.loads((Path(p['roots']['old_inputs'])/'plan.json').read_text())['inputs']}
    stopped=[]; signal.signal(signal.SIGTERM,lambda *_:stopped.append(True))
    deadline=float(os.environ['FW_JOB_START_EPOCH'])+p['walltime_seconds']
    durations=[]; queries=0
    def finish(status):
        done=[c for c in cases if (out/f'case_{c}.json').is_file() and (out/f'case_{c}.pt').is_file()]
        s=receipt('score',status=status,shard=shard,context_count=len(done),expected_contexts=2,
                  case_ids=done,new_native_queries=queries,model_scope='614 Linears; full sampler adjoints')
        save(out/'summary.json',s)
    if all((out/f'case_{c}.json').is_file() for c in cases): finish('complete'); return
    runtime=pilot.Runtime(out); runtime.arm('bf16'); banks=Banks(runtime,p,scoring=True)
    layout=json.loads((root/'layout.json').read_text())
    if banks.total_tiles!=layout['total_tiles'] or list(banks.layout)!=[m['name'] for m in layout['modules']]: raise RuntimeError('Byte layout differs')
    try:
        for case in cases:
            if (out/f'case_{case}.json').is_file(): continue
            if stopped or time.time()+1.2*max([1200]+durations)+p['save_reserve_seconds']>=deadline:
                finish('resumable'); return
            tick=time.monotonic(); e=entries[case,0]
            old=torch.load(e['cache_path'],map_location='cpu',weights_only=True)
            full=old['captures']['distilled_self']; teacher=old['captures']['bf16']
            obs,meta=source.load_case(Path(p['roots']['old_inputs']),rows[case]); runtime.description=meta['description']; datum=runtime.datum(obs)
            # Finish the frozen BF16 reader adjoint before changing any model weights.
            banks.activate('bf16')
            z=full['video_latents'].to(device='cuda',dtype=runtime.model.torch_dtype).clone().requires_grad_(True)
            print('READER_ADJOINT',case,flush=True)
            action,counts,injected=differentiable_reader(runtime,datum,e['sampler_seed'],z)
            # Historical manifests use their own two-shard assignment.
            teacher_path=next(path for path in (prior/'results/teacher-0'/f'{stem(e)}_action_only.pt',prior/'results/teacher-1'/f'{stem(e)}_action_only.pt') if path.is_file())
            expected=torch.load(teacher_path,map_location='cpu',weights_only=True)['action']
            check_reader=parity(torch,action,expected,p,'BF16 reader')
            lr=(action[:10,:6].float()-teacher['action'][:10,:6].to('cuda').float()).square().mean()
            qz=torch.autograd.grad(lr,z)[0].detach().float().cpu()
            if not bool(torch.isfinite(qz).all()) or not bool(qz.abs().sum()>0): raise RuntimeError('Reader adjoint is disconnected/nonfinite')
            if not bool((qz[:,:,:1]==0).all()): raise RuntimeError('Observation frame was not clamped')
            del action,injected,z,lr
            gc.collect(); torch.cuda.empty_cache()
            banks.activate('full',grad=True); banks.stats={}; banks.collect=True
            print('NATIVE_ADJOINT',case,flush=True)
            action,counts,future=sampler.differentiable_action(runtime,datum,e['sampler_seed'],checkpointed=True,return_future=True)
            banks.collect=False
            if len(banks.seen)!=614: raise RuntimeError('Native sampler did not visit all614 targets')
            checks=dict(reader=check_reader,action=parity(torch,action,full['action'],p,'full student action'),future=parity(torch,future,full['video_latents'],p,'full student future'))
            la=(action[:10,:6].float()-teacher['action'][:10,:6].to('cuda').float()).square().mean()
            lz=latent_reference.future_nmse_loss(torch,future,teacher['video_latents'].to('cuda'))
            chain=(future.float()*qz.to('cuda')).sum()
            ga=banks.backward(la,'action',retain=True)
            print('BACKWARD_COMPLETE action',case,len(ga),flush=True)
            gz=banks.backward(lz,'latent',retain=True)
            print('BACKWARD_COMPLETE latent',case,len(gz),flush=True)
            connected=dict(action=len(ga),latent=len(gz))
            scores=banks.scores(ga,gz,prep['latent_weight'])
            del ga,gz
            gc.collect()
            gr=banks.backward(chain,'reader_future')
            print('BACKWARD_COMPLETE reader_future',case,len(gr),flush=True)
            connected['reader_future']=len(gr)
            scores['reader_damage']=banks.reader_scores(gr)
            del gr
            gc.collect()
            scores.update(banks.static)
            scores.update(banks.outliers())
            if any(not bool(torch.isfinite(v).all()) or len(v)!=banks.total_tiles for v in scores.values()): raise RuntimeError('Score shape/nonfinite error')
            result=dict(protocol=p['protocol'],case_id=case,seed_index=0,sampler_seed=e['sampler_seed'],job_id=os.environ['PBS_JOBID'],
                        parity=checks,action_mse=float(la.detach()),future_nmse=float(lz.detach()),counts=counts,
                        connected=connected,seconds=time.monotonic()-tick)
            atomic_torch_save(torch,out/f'case_{case}.pt',dict(**result,scores=scores))
            save(out/f'case_{case}.json',result)
            durations.append(result['seconds']); queries+=1; finish('running')
            print('SCORE_CONTEXT_COMPLETE',case,result['seconds'],flush=True)
            del scores,old,full,teacher,datum,action,future,la,lz,chain,qz
            gc.collect(); torch.cuda.empty_cache()
        finish('complete')
    finally:
        banks.close()
        if runtime.env: runtime.env.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--shard',type=int,choices=range(4),required=True)
    main(a.parse_args().shard)

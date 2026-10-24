"""GPU validation and complete paired kernel deployment panel, inside PBS only."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time

from reference_profile import guard, save, freeze_numerics, memory, statistics, max_abs_difference


def progress(root, mode, phase, **fields):
    row = dict(protocol='fastwam-vq-r128-c80-fused-kernel-v1', pbs_jobid=os.environ['PBS_JOBID'],
               mode=mode, phase=phase, **fields)
    save(root/'progress.json', row)
    print('KERNEL_PROGRESS '+json.dumps(row), flush=True)


def cuda_time(call, repeat=10):
    import torch
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(repeat):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/repeat


def selftest(p, root):
    import torch
    import fused_input
    import fused_vq
    import micro_codebook as micro
    import export_runtime as export
    import batched_transform
    progress(root,'selftest','synthetic_kernel_validation')
    input_result = fused_input.self_test()
    save(root/'input_preflight.json',input_result)
    print('INPUT_PREFLIGHT_PASSED '+json.dumps(input_result),flush=True)
    vq_result = fused_vq.self_test()
    save(root/'synthetic_preflight.json',dict(input=input_result,vq=vq_result))
    bundle = export.load_bundle(p['export_reuse']['bundle_path'])
    rows, benches = [], []
    # Every actual encoding is checked, including all tails; no refit or teacher query.
    shapes = {(294,3072,3072),(32,3072,1024),(129,3072,3072),(32,1024,3072),
              (129,3072,1024),(294,14336,3072),(294,3072,14336),
              (32,1024,4096),(32,4096,1024),(1,4096,8),(32,7,1024),(32,1024,7)}
    completed = set()
    for index,(name,encoded) in enumerate(bundle['quantized'].items()):
        enc = {k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in encoded.items()}
        expected = micro.decode_vq(enc)
        decoded = fused_vq.decode_weight(enc)
        same = torch.equal(decoded.view(torch.int16),expected.view(torch.int16))
        if not same:
            raise ValueError('Actual decoder differs: '+name)
        n,k = map(int,enc['shape'])
        rows.append(dict(module=name,N=n,K=k,decoder_bitwise=True))
        for m,nn,kk in sorted(shapes):
            key=(m,nn,kk)
            if (n,k)!=(nn,kk) or key in completed:
                continue
            x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
            bias=torch.zeros(n,device='cuda',dtype=torch.bfloat16)
            prepared=fused_vq.prepare(enc)
            actual=fused_vq.prepared_linear(x,prepared,bias)
            ref=torch.nn.functional.linear(x,expected,bias)
            diff=actual.float()-ref.float()
            nrmse=float(diff.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12))
            if not bool(torch.isfinite(actual).all()) or nrmse>p['kernel_validation']['gemm_local_nrmse_sanity_max']:
                raise ValueError(f'GEMM local sanity failed {key}: {nrmse}')
            benches.append(dict(module=name,M=m,N=n,K=k,nrmse=nrmse,max_abs=float(diff.abs().max()),
                dense_gpu_ms=cuda_time(lambda:torch.nn.functional.linear(x,expected,bias)),
                packed_gpu_ms=cuda_time(lambda:fused_vq.prepared_linear(x,prepared,bias))))
            completed.add(key)
            del x,bias,actual,ref,diff,prepared
        del decoded,expected,enc
        if index%32==0:
            progress(root,'selftest','all_actual_decoders',modules=index+1,expected=614)
    if len(rows)!=614 or completed!=shapes:
        raise ValueError(f'Incomplete decoder/shape validation: {len(rows)}, missing={shapes-completed}')
    save(root/'microbench.json',dict(measurement='CUDA event microbench, 3warmup10repeat',rows=benches,
        native_integer_gemm=False,all_shapes=len(shapes)))
    # Probe allocation-fill sensitivity without changing any main-panel flag.
    name=next(n for n,e in bundle['quantized'].items() if tuple(e['shape'])==(3072,3072))
    transform={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in bundle['transforms'][name].items()}
    x=torch.randn((294,3072),device='cuda',dtype=torch.bfloat16)
    fill=[]
    for enabled in [True,False]:
        torch.utils.deterministic.fill_uninitialized_memory=enabled
        ref=batched_transform.transform_input(x,transform)
        actual=fused_input.smooth_h(x,transform)
        if not torch.equal(ref,actual):
            raise ValueError('Fill sensitivity probe parity failed')
        fill.append(dict(fill_uninitialized_memory=enabled,
            eager_smooth_h_gpu_ms=cuda_time(lambda:batched_transform.transform_input(x,transform)),
            fused_smooth_h_gpu_ms=cuda_time(lambda:fused_input.smooth_h(x,transform))))
    torch.utils.deterministic.fill_uninitialized_memory=True
    save(root/'memory_fill.json',dict(scope='one captured model shape, random input, microbench only',rows=fill))
    save(root/'kernel_validation.json',dict(protocol=p['protocol'],pbs_jobid=os.environ['PBS_JOBID'],
        complete=True,synthetic_input=input_result,synthetic_vq=vq_result,
        decoder_modules=len(rows),decoder_all_bitwise=True,all_shapes_validated=True,
        new_training_updates=0))
    for offset in range(0,len(rows),64):
        save(root/'per_layer'/f'decoder_{offset//64:02d}.json',rows[offset:offset+64])
    progress(root,'selftest','complete',modules=len(rows))


def profile(p,root,mode):
    import numpy as np
    import torch
    import calibration_source
    import export_runtime as export
    import kernel_runtime
    import reference_profile
    out=root/mode
    out.mkdir(parents=True,exist_ok=True)
    progress(root,mode,'loading')
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if mode=='bf16':
        import runner
        runtime=runner.Runtime(out)
        runtime.arm('bf16')
        for param in runtime.model.parameters(): param.requires_grad_(False)
    else:
        bundle=export.load_bundle(p['export_reuse']['bundle_path'])
        runtime=kernel_runtime.make_runtime(out,p,bundle,direct=mode=='packed_direct')
        kernel_runtime.install(runtime,bundle,packed=mode.startswith('packed_'),
                               fused=mode!='eager',direct=mode=='packed_direct')
        del bundle
    torch.cuda.synchronize()
    load_seconds=time.perf_counter()-started
    load_memory=memory(torch)
    plan_path=Path(p['sources']['calibration_plan'])
    plan=json.loads(plan_path.read_text())
    rows={r['case_id']:r for r in plan['inputs']}
    settings=p['profiling']
    if plan.get('test_case_ids',settings['case_ids'])!=settings['case_ids']:
        raise ValueError('Frozen panel changed')
    def datum(case):
        torch.cuda.synchronize()
        t=time.perf_counter()
        obs,meta=calibration_source.load_observation(np,plan_path.parent,rows[case])
        runtime.description=meta['description']
        value=runtime.datum(obs)
        torch.cuda.synchronize()
        return value,(time.perf_counter()-t)*1000
    progress(root,mode,'anchor_and_warmup')
    value,_=datum(2000)
    action,_=runtime.infer(value,rows[2000]['sampler_seeds'][0],measure=False)
    anchor_error=future_error=None
    if mode!='bf16':
        anchor=torch.load(p['source']['anchor_tensor'],map_location='cpu',weights_only=True)
        anchor_error=max_abs_difference(action,anchor['action'])
        import sampler
        with torch.no_grad():
            replay,counts,future=sampler.differentiable_action(runtime,value,
                rows[2000]['sampler_seeds'][0],checkpointed=False,return_future=True)
        if counts!={'video_steps':10,'action_steps':10}:
            raise ValueError('Sampler step counts differ')
        future_error=max_abs_difference(future,anchor['future'])
        future_nrmse=float((future.detach().float().cpu()-anchor['future'].float()).square().mean().sqrt()/
                           anchor['future'].float().square().mean().sqrt().clamp_min(1e-12))
        if mode in ['eager','input_fused'] and max(anchor_error,future_error)>settings['parity_atol']:
            raise ValueError(f'Exact anchor failed {mode}: {anchor_error}, {future_error}')
        save(out/'validation.json',dict(anchor_action_max_abs=anchor_error,anchor_future_max_abs=future_error,
            anchor_future_nrmse=future_nrmse,changed_gemm_order=mode.startswith('packed_'),
            input_modules_checked=len(runtime.banks.validation),input_checks=runtime.banks.validation))
        if mode!='eager' and len(runtime.banks.validation)!=614:
            raise ValueError('Input operand validation did not cover614 modules')
        runtime.banks.validate_input=False
        del anchor,replay,future
    del value,action
    value,_=datum(3000)
    for _ in range(settings['warmup_queries']):
        runtime.infer(value,rows[3000]['sampler_seeds'][0],measure=False)
    del value
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident=memory(torch)
    if mode!='bf16': runtime.banks.calls.clear()
    samples,actions=[],{}
    progress(root,mode,'measuring',expected=80)
    for case in settings['case_ids']:
        value,prep=datum(case)
        for seed_index in settings['seed_indices']:
            action,metric=runtime.infer(value,rows[case]['sampler_seeds'][seed_index],measure=True)
            actions[f'{case}_{seed_index}']=action.detach().cpu()
            samples.append(dict(case_id=case,seed_index=seed_index,preprocessing_wall_ms=prep,**metric))
            del action
        del value
        progress(root,mode,'measuring',queries=len(samples),expected=80)
    save(out/'samples.json',samples)
    torch.save(actions,out/'actions.pt')
    metadata=getattr(runtime,'install_metadata',{})
    summary=dict(protocol=p['protocol'],pbs_jobid=os.environ['PBS_JOBID'],mode=mode,complete=True,
        queries=len(samples),source_job=p['source']['source_job'],new_training_updates=0,
        metrics={k:statistics(samples,k) for k in ['full_query_wall_ms','denoising_latency_ms',
            'peak_allocated_gpu_GB','peak_reserved_gpu_GB','baseline_allocated_gpu_GB']},
        load_seconds=load_seconds,load_transient_memory=load_memory,resident_after_warmup=resident,
        anchor_action_max_abs=anchor_error,anchor_future_max_abs=future_error,installation=metadata,
        gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        scope=settings['primary_scope'],instrumentation_used_for_primary_latency=False,
        compiled=False,cuda_graph=False)
    if mode!='bf16':
        summary['coverage']=dict(modules=len(runtime.banks.seen),timed_linear_calls=sum(runtime.banks.calls.values()),
            dense_target_weights=sum(mod.weight is not None for mod in runtime.q.modules.values()),
            packed_modules=len(runtime.banks.encodings))
    save(out/'summary.json',summary)
    # One trace after all primary measurements; no nested event instrumentation.
    if mode in settings['diagnostic_profile_modes']:
        progress(root,mode,'diagnostic_profiler')
        value,_=datum(3000)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=True,with_stack=False) as prof:
            runtime.infer(value,rows[3000]['sampler_seeds'][0],measure=False)
        prof.export_chrome_trace(str(out/'trace.json'))
        operators=[dict(name=e.key[:240],calls=e.count,self_device_us=getattr(e,'self_device_time_total',0),
                        self_cpu_us=e.self_cpu_time_total) for e in prof.key_averages()]
        operators.sort(key=lambda r:-r['self_device_us'])
        save(out/'operators.json',dict(intrusive=True,rows=operators[:160]))
    progress(root,mode,'complete',queries=len(samples))


def main():
    guard()
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['selftest','bf16','eager','input_fused','packed_fused','packed_direct'])
    args=parser.parse_args()
    p=json.loads((Path(os.environ['ARTIFACTS'])/'protocol.json').read_text())
    root=Path(p['remote_root'])/'results'
    import torch
    freeze_numerics(torch)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.manual_seed(20261023)
    if args.mode=='selftest': selftest(p,root)
    else: profile(p,root,args.mode)


if __name__=='__main__':
    main()

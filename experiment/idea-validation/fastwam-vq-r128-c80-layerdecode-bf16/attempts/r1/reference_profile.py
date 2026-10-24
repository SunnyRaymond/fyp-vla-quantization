"""Frozen-checkpoint export and inference profiling inside a PBS GPU allocation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import socket
import time


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE', '')
    if (not os.environ.get('PBS_JOBID') or 'login' in host.lower() or not nodefile
            or host not in {x.split('.')[0] for x in Path(nodefile).read_text().split()}):
        raise RuntimeError('Approved PBS compute allocation required')


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('Empty measurements')
    pos = (len(ordered)-1) * fraction
    lower, upper = math.floor(pos), math.ceil(pos)
    return ordered[lower] + (ordered[upper]-ordered[lower]) * (pos-lower)


def statistics(rows, field):
    values = [r[field] for r in rows]
    return dict(n=len(values), mean=sum(values)/len(values), p50=percentile(values,.5),
                p95=percentile(values,.95), minimum=min(values), maximum=max(values))


def memory(torch):
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def max_abs_difference(left, right):
    return float((left.detach().float().cpu()-right.detach().float().cpu()).abs().max())


def freeze_numerics(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def inference_runtime(out, p, bundle):
    """Instantiate from exported configuration; no calibration/BF16 checkpoint is loaded."""
    import torch
    import runner
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from quantization import Quantization
    from libero_state_loader import ensure_numpy_compat
    import collections
    ensure_numpy_compat()
    from experiments.libero import eval_libero_single as ev
    from libero.libero import benchmark
    config = bundle['assets']['resolved_config']
    if config is None:
        raise ValueError('Inference bundle must contain its resolved model/processor configuration')
    runtime = runner.Runtime.__new__(runner.Runtime)
    runtime.torch, runtime.ev, runtime.benchmark, runtime.out = torch, ev, benchmark, out
    runtime.cfg = OmegaConf.create(config)
    runtime.model = instantiate(runtime.cfg.model, model_dtype=torch.bfloat16, device='cuda')
    runtime.model.to('cuda').eval()
    runtime.processor = instantiate(runtime.cfg.data.train.processor).eval()
    stats = json.loads(bundle['assets']['dataset_stats_json'])
    if stats is None:
        raise ValueError('Inference bundle lacks its normalizer statistics')
    # Use the original parser on an exported copy, rather than the training asset.
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    stats_path = out / 'exported_dataset_stats.json'
    save(stats_path, stats)
    runtime.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    if bundle.get('processor_state'):
        runtime.processor.load_state_dict(bundle['processor_state'])
    runtime.q = Quantization(runtime.model, group_size=128, kv4=False)
    runtime.meter = runner.QueryMeter(runtime.model, torch, steps=10)
    runtime.height, runtime.width = map(int, runtime.cfg.data.train.video_size)
    runtime.suites, runtime.env = {}, None
    runtime.weight_converted, runtime.calls = False, 0
    runtime.arm_calls = collections.Counter()
    runtime.current_arm = 'reader_kl'
    return runtime


class Detail:
    """Intrusive attribution only; these times never enter primary latency samples."""
    def __init__(self, runtime, torch):
        self.runtime, self.torch = runtime, torch
        self.events, self.patches, self.stack = [], [], []
        self.cache_bytes = 0

    def patch(self, obj, name, replacement):
        existed = name in getattr(obj, '__dict__', {})
        previous = getattr(obj, name)
        self.patches.append((obj,name,existed,previous))
        setattr(obj,name,replacement)

    def measure(self, category, label, call, shape=None):
        start, end = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        with self.torch.profiler.record_function(category + '/' + label):
            result = call()
        end.record()
        self.events.append((category,label,shape,start,end))
        return result

    def __enter__(self):
        import torch.nn.functional as F
        original = F.linear
        def linear(input,weight,bias=None):
            x,w,b = input,weight,bias
            label = self.stack[-1] if self.stack else 'other'
            shape = dict(M=x.numel()//x.shape[-1], N=w.shape[0], K=w.shape[1],
                         input_shape=list(x.shape), weight_dtype=str(w.dtype))
            return self.measure('gemm',label,lambda:original(x,w,b),shape)
        self.patch(F,'linear',linear)
        for name,mod in self.runtime.q.modules.items():
            old = mod.forward
            def forward(x,*args,_old=old,_name=name,**kwargs):
                self.stack.append(_name)
                try:
                    return self.measure('target_linear',_name,lambda:_old(x,*args,**kwargs))
                finally:
                    self.stack.pop()
            self.patch(mod,'forward',forward)
        import batched_transform
        old = batched_transform.transform_input
        self.patch(batched_transform,'transform_input',
            lambda *a,**k:self.measure('component','smooth_signed_h',lambda:old(*a,**k)))
        if hasattr(self.runtime,'banks'):
            rotations = self.runtime.banks.rotations
            old_r = rotations.apply_after_h
            self.patch(rotations,'apply_after_h',
                lambda *a,**k:self.measure('component','learned_r',lambda:old_r(*a,**k)))
        # A4 and decode names are explicit in the two runtime implementations.
        import rotation_bank
        old_a4 = rotation_bank.A4.forward
        self.patch(rotation_bank.A4,'forward',staticmethod(
            lambda *a,**k:self.measure('component','a4_quant_dequant',lambda:old_a4(*a,**k))))
        import export_runtime
        if hasattr(export_runtime,'a4'):
            old_a = export_runtime.a4
            self.patch(export_runtime,'a4',
                lambda *a,**k:self.measure('component','a4_quant_dequant',lambda:old_a(*a,**k)))
        for name in ('_denoise_video','_denoise_action_with_video_cache'):
            if hasattr(self.runtime.model,name):
                old_stage = getattr(self.runtime.model,name)
                def stage(*args,_old=old_stage,_name=name,**kwargs):
                    return self.measure('stage',_name,lambda:_old(*args,**kwargs))
                self.patch(self.runtime.model,name,stage)
        mot = self.runtime.model.mot
        old_prefill = mot.prefill_video_cache_tensor
        def prefill(*a,**k):
            result = self.measure('stage','video_cache_prefill',lambda:old_prefill(*a,**k))
            seen = set()
            def visit(value):
                if self.torch.is_tensor(value):
                    storage = value.untyped_storage()
                    key = (str(value.device),storage.data_ptr())
                    if key not in seen:
                        seen.add(key)
                        return storage.nbytes()
                elif isinstance(value,(tuple,list)):
                    return sum(visit(v) for v in value)
                elif isinstance(value,dict):
                    return sum(visit(v) for v in value.values())
                return 0
            self.cache_bytes = max(self.cache_bytes,visit(result))
            return result
        self.patch(mot,'prefill_video_cache_tensor',prefill)
        return self

    def __exit__(self,*exc):
        for obj,name,existed,previous in reversed(self.patches):
            if existed:
                setattr(obj,name,previous)
            else:
                delattr(obj,name)

    def summarize(self):
        self.torch.cuda.synchronize()
        grouped = {}
        for category,label,shape,start,end in self.events:
            key = (category,label,json.dumps(shape,sort_keys=True))
            row = grouped.setdefault(key,dict(category=category,label=label,shape=shape,calls=0,gpu_ms=0))
            row['calls'] += 1
            row['gpu_ms'] += start.elapsed_time(end)
        return dict(intrusive=True, scope='single fixed-input query; nested categories must not be summed',
                    cache_payload_bytes=self.cache_bytes, rows=sorted(grouped.values(),key=lambda r:-r['gpu_ms']))


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['legacy','bf16','inference_dense','stream_decode'],required=True)
    args = parser.parse_args()
    p = json.loads((Path(os.environ['ARTIFACTS'])/'protocol.json').read_text())
    root = Path(p['remote_root'])/'results'
    out = root/args.mode
    out.mkdir(parents=True,exist_ok=True)
    mode = args.mode
    job = os.environ['PBS_JOBID']
    def progress(phase,**kw):
        value = dict(protocol=p['protocol'],pbs_jobid=job,mode=mode,phase=phase,**kw)
        save(root/'progress.json',value)
        save(out/'progress.json',value)
        print('PROFILE_PROGRESS '+json.dumps(value),flush=True)
    progress('loading')
    import numpy as np
    import torch
    import runner
    import calibration_source
    import export_runtime
    freeze_numerics(torch)
    # Formal infer returns CPU actions, while sampler replay returns CUDA tensors.
    assert max_abs_difference(torch.tensor([1.25],device='cuda'),torch.tensor([1.25])) == 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if mode == 'legacy':
        from source_runtime import R128Runtime
        old = json.loads(Path(p['source_evaluation_protocol']).read_text())
        runtime = R128Runtime(out,old,2)
        receipt = runtime.prepare(root)
        save(out/'source_anchor_parity.json',receipt)
        if p.get('export_reuse'):
            exported = json.loads((root/'storage.json').read_text())
            if (exported['source_job']!=p['source']['source_job'] or exported['arm']!='reader_kl'
                    or exported['target_modules']!=614
                    or exported['bundle_path']!=p['export_reuse']['bundle_path']
                    or Path(exported['bundle_path']).stat().st_size!=exported['torch_file_bytes']):
                raise ValueError('Existing export source/size binding differs')
        else:
            exported = export_runtime.export_bundle(runtime,root/'export',p)
            details = exported.pop('per_module')
            exported.pop('excluded_dense_state_keys',None)
            for offset in range(0,len(details),64):
                save(root/'per_layer'/f'storage_{offset//64:02d}.json',details[offset:offset+64])
            exported['whole_model_tensor_compression_ratio'] = (
                exported['original_dense_model_state']['unique_storage_bytes'] /
                exported['whole_bundle']['unique_storage_bytes'])
            save(root/'storage.json',exported)
    elif mode == 'bf16':
        runtime = runner.Runtime(out)
        runtime.arm('bf16')
        for param in runtime.model.parameters():
            param.requires_grad_(False)
    else:
        stored = json.loads((root/'storage.json').read_text())
        bundle = export_runtime.load_bundle(stored['bundle_path'])
        runtime = inference_runtime(out,p,bundle)
        export_runtime.install_bundle(runtime,bundle,mode='dense' if mode=='inference_dense' else 'stream_decode')
        del bundle
    load_seconds = time.perf_counter()-started
    torch.cuda.synchronize()
    load_memory = memory(torch)
    runtime.model.eval()
    if any(v.requires_grad for v in runtime.model.parameters()):
        raise ValueError('Inference runtime contains trainable parameters')
    plan_path = Path(p['sources']['calibration_plan'])
    plan = json.loads(plan_path.read_text())
    rows = {r['case_id']:r for r in plan['inputs']}
    expected = p['profiling']
    if 'test_case_ids' in plan and plan['test_case_ids'] != expected['case_ids']:
        raise ValueError('Frozen heldout cases differ')
    def datum(case):
        torch.cuda.synchronize()
        start = time.perf_counter()
        obs,meta = calibration_source.load_observation(np,plan_path.parent,rows[case])
        runtime.description = meta['description']
        value = runtime.datum(obs)
        torch.cuda.synchronize()
        return value,(time.perf_counter()-start)*1000
    # Formal inference replay of the exact current checkpoint anchor.
    value,_ = datum(2000)
    action,_ = runtime.infer(value,rows[2000]['sampler_seeds'][0],measure=False)
    anchor_error = None
    future_error = None
    if mode != 'bf16':
        anchor = torch.load(p['source']['anchor_tensor'],map_location='cpu',weights_only=True)
        anchor_error = max_abs_difference(action,anchor['action'])
        if mode != 'legacy':
            import sampler
            with torch.no_grad():
                replay,counts,future = sampler.differentiable_action(runtime,value,
                    rows[2000]['sampler_seeds'][0],checkpointed=False,return_future=True)
            future_error = max_abs_difference(future,anchor['future'])
            replay_error = max_abs_difference(replay,action)
            if counts != {'video_steps':10,'action_steps':10} or max(future_error,replay_error)>expected['parity_atol']:
                raise ValueError('Exported full action/future sampler parity failed')
            del replay,future
        else:
            future_error = receipt['source_future_max_abs']
        if anchor_error > expected['parity_atol']:
            raise ValueError(f'Inference export anchor differs: {anchor_error}')
        del anchor
    del value,action
    value,_ = datum(3000)
    progress('warmup',load_seconds=load_seconds,anchor_max_abs=anchor_error)
    for _ in range(expected['warmup_queries']):
        runtime.infer(value,rows[3000]['sampler_seeds'][0],measure=False)
    del value
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = memory(torch)
    samples,actions = [],{}
    cases = expected['case_ids'] if mode!='stream_decode' else [2000,3000]
    seeds = expected['seed_indices'] if mode!='stream_decode' else [0]
    for case in cases:
        value,prep_ms = datum(case)
        for seed_index in seeds:
            action,metric = runtime.infer(value,rows[case]['sampler_seeds'][seed_index],measure=True)
            samples.append(dict(case_id=case,seed_index=seed_index,preprocessing_wall_ms=prep_ms,**metric))
            actions[f'{case}_{seed_index}'] = action.detach().cpu()
            del action
            progress('measuring',queries=len(samples),expected=len(cases)*len(seeds))
        del value
    torch.save(actions,out/'actions.pt')
    save(out/'samples.json',samples)
    metrics = {k:statistics(samples,k) for k in ('full_query_wall_ms','denoising_latency_ms',
        'peak_allocated_gpu_GB','peak_reserved_gpu_GB','baseline_allocated_gpu_GB')}
    profile_file = None
    detail = None
    if mode != 'stream_decode':
        value,_ = datum(expected['detail_case_id'])
        progress('intrusive_profile',queries=len(samples))
        activities = [torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        # Linear wrappers record shapes; avoid profiler tensor retention on the 40GB GPU.
        with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=True,
                                    with_stack=False) as prof:
            with Detail(runtime,torch) as detail:
                runtime.infer(value,rows[3000]['sampler_seeds'][0],measure=False)
        profile_file = str(out/'trace.json')
        prof.export_chrome_trace(profile_file)
        save(out/'detail.json',detail.summarize())
        operators = []
        for event in prof.key_averages(group_by_input_shape=True):
            operators.append(dict(name=event.key,count=event.count,input_shapes=event.input_shapes,
                self_cpu_us=event.self_cpu_time_total,
                self_gpu_us=getattr(event,'self_device_time_total',getattr(event,'self_cuda_time_total',0)),
                gpu_us=getattr(event,'device_time_total',getattr(event,'cuda_time_total',0)),
                self_gpu_memory_bytes=getattr(event,'self_device_memory_usage',getattr(event,'self_cuda_memory_usage',0))))
        save(out/'operators.json',sorted(operators,key=lambda r:-r['self_gpu_us']))
        del prof,value,detail
    props = torch.cuda.get_device_properties(0)
    summary = dict(protocol=p['protocol'],pbs_jobid=job,mode=mode,complete=True,
        queries=len(samples),primary_measurements=mode!='stream_decode',metrics=metrics,
        source_job=p['source']['source_job'],source_arm='reader_kl',new_training_updates=0,
        anchor_max_abs=anchor_error,anchor_future_max_abs=future_error,
        anchor_passed=mode=='bf16' or max(anchor_error,future_error)<=expected['parity_atol'],
        load_seconds=load_seconds,load_transient_memory=load_memory,resident_after_warmup=resident,
        trace=profile_file,torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=props.name,gpu_uuid=str(props.uuid),gpu_total_bytes=props.total_memory,
        native_low_bit_kernel=False,compiled=False,cuda_graph=False,scope=expected['primary_scope'],
        instrumentation_used_for_primary_latency=False,
        preprocessing_scope='load cached observation and convert to GPU datum; reported separately')
    save(out/'summary.json',summary)
    progress('complete',queries=len(samples),expected=len(cases)*len(seeds))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if os.environ.get('ROOT'):
            save(Path(os.environ.get('NEW_ROOT',os.environ['ROOT']))/'results/failure.json',
                 dict(pbs_jobid=os.environ.get('PBS_JOBID'),error=repr(exc)))
        raise

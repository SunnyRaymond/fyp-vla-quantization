"""Aggregate complete paired measurements; report GEMM numerical changes explicitly."""
from __future__ import annotations
import json
import os
from pathlib import Path
from reference_profile import guard, save


def errors(torch, actual, reference, keys):
    diffs=[actual[k].float()-reference[k].float() for k in sorted(keys)]
    return dict(action_max_abs=max(float(x.abs().max()) for x in diffs),
        full32_action_rmse=float(torch.stack([x.square().mean() for x in diffs]).mean().sqrt()),
        first10_motor_rmse=float(torch.stack([x[:10,:6].square().mean() for x in diffs]).mean().sqrt()),
        full32_motor_rmse=float(torch.stack([x[:,:6].square().mean() for x in diffs]).mean().sqrt()),
        first10_gripper_threshold_disagreement=sum(float(((actual[k][:10,6]>.5)!=(reference[k][:10,6]>.5)).float().mean()) for k in keys)/len(keys))


def main():
    guard()
    import torch
    p=json.loads((Path(os.environ['ARTIFACTS'])/'protocol.json').read_text())
    root=Path(p['remote_root'])/'results'
    modes=p['profiling']['primary_modes']
    summaries={m:json.loads((root/m/'summary.json').read_text()) for m in modes}
    outputs={m:torch.load(root/m/'actions.pt',map_location='cpu',weights_only=True) for m in modes}
    keys={f'{c}_{s}' for c in p['profiling']['case_ids'] for s in p['profiling']['seed_indices']}
    validation=json.loads((root/'kernel_validation.json').read_text())
    if not validation['complete'] or not validation['decoder_all_bitwise'] or validation['decoder_modules']!=614:
        raise ValueError('Kernel validation incomplete')
    for m,row in summaries.items():
        if not row['complete'] or row['queries']!=80 or set(outputs[m])!=keys:
            raise ValueError('Incomplete full80 panel: '+m)
        if row['pbs_jobid']!=os.environ['PBS_JOBID'] or row['new_training_updates']!=0:
            raise ValueError('Unbound measurement: '+m)
        if not row['deterministic_algorithms'] or not row['fill_uninitialized_memory']:
            raise ValueError('Numerical/performance flags differ: '+m)
        if m!='bf16' and row['coverage']['modules']!=614:
            raise ValueError('Target coverage incomplete: '+m)
        if m.startswith('packed_') and (row['coverage']['packed_modules']!=614 or row['coverage']['dense_target_weights']!=0):
            raise ValueError('Packed weights not fully resident: '+m)
    if len({r['gpu_uuid'] for r in summaries.values()})!=1:
        raise ValueError('Different comparison GPUs')
    comparison={m:errors(torch,outputs[m],outputs['eager'],keys) for m in modes}
    if comparison['input_fused']['action_max_abs']>p['profiling']['parity_atol']:
        raise ValueError('Input fusion changed full80 action outputs')
    # Both packed runtimes use the same arithmetic. A loader change must preserve it exactly.
    loader_error=errors(torch,outputs['packed_direct'],outputs['packed_fused'],keys)
    if loader_error['action_max_abs']>p['profiling']['parity_atol']:
        raise ValueError('Direct loader changed packed inference')
    baseline=summaries['bf16']['metrics']['full_query_wall_ms']['p50']
    for m,row in summaries.items():
        row['p50_speedup_vs_bf16']=baseline/row['metrics']['full_query_wall_ms']['p50']
        row['error_vs_eager']=comparison[m]
        row['error_vs_bf16']=errors(torch,outputs[m],outputs['bf16'],keys)
    parity=dict(input_fusion_full80=comparison['input_fused'],direct_loader_full80=loader_error,
        packed_full80=comparison['packed_fused'],packed_outputs_bitwise=comparison['packed_fused']['action_max_abs']==0,
        decoder_all614_bitwise=True,closed_loop_validated=False)
    save(root/'parity.json',parity)
    summary=dict(protocol=p['protocol'],pbs_jobid=os.environ['PBS_JOBID'],complete=True,
        export_complete=True,parity_passed=True,profile_complete=True,kernel_validation_complete=True,
        native_vq_kernel=True,native_integer_gemm=False,new_training_updates=0,closed_loop_episodes=0,
        source_job=p['source']['source_job'],modes=summaries,parity=parity,
        storage=json.loads((root/'storage.json').read_text()),
        main_scope=p['profiling']['primary_scope'],
        accuracy_claim='Input transform and decoder equivalence verified; changed GEMM action errors reported; no new closed-loop accuracy claim')
    save(root/'summary.json',summary)
    report=['# Frozen C80 R128 A+Z+KL fused VQ kernel results','',
        'All modes: same GPU, full40 contexts x2 seeds, fresh process, no training. GPU GB=1e9, storage GiB=2^30.','',
        '| Mode | n | Mean ms | p50 ms | p95 ms | Speedup vs BF16 | Warm allocated GB | Query peak allocated GB | Query peak reserved GB | Load peak allocated GB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,r in summaries.items():
        met=r['metrics'];lat=met['full_query_wall_ms']
        report.append(f"| {m} | {r['queries']} | {lat['mean']:.3f} | {lat['p50']:.3f} | {lat['p95']:.3f} | {r['p50_speedup_vs_bf16']:.4f} | {r['resident_after_warmup']['allocated_bytes']/1e9:.6f} | {met['peak_allocated_gpu_GB']['maximum']:.6f} | {met['peak_reserved_gpu_GB']['maximum']:.6f} | {r['load_transient_memory']['peak_allocated_bytes']/1e9:.6f} |")
    report += ['', '## Full80 error relative to the frozen eager VQ checkpoint','',
        '| Mode | Action max abs | Full32 action RMSE | First10 motor RMSE | First10 gripper threshold disagreement |',
        '|---|---:|---:|---:|---:|']
    for m,r in comparison.items():
        report.append(f"| {m} | {r['action_max_abs']:.9g} | {r['full32_action_rmse']:.9g} | {r['first10_motor_rmse']:.9g} | {r['first10_gripper_threshold_disagreement']:.9g} |")
    report += ['', 'BF16 is the unquantized reference, so its error relative to eager VQ is expected.',
        'Packed weights are decoded by tile inside GEMM. BF16 operands, FP32 transform/decoder/accumulator. No native INT4/INT8 arithmetic.',
        'Decoded weights are bitwise equal; GEMM accumulation order may change. Full-output errors above do not establish closed-loop equivalence.',
        'Raw operators.json is an intrusive diagnostic recorded after primary timing.',
        'Storage is unchanged from the frozen inference bundle; external cached nonweight assets are excluded.','']
    (root/'RESULTS.zh.md').write_text('\n'.join(report))
    print('FUSED_PIPELINE_COMPLETE '+json.dumps(dict(queries_per_mode=80,modes=modes,source_job=p['source']['source_job'])),flush=True)


if __name__=='__main__': main()

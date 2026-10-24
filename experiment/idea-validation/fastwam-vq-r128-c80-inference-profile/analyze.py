"""Aggregate the frozen inference profiles and choose a VQ kernel implementation plan."""
from __future__ import annotations
from collections import defaultdict
import json
import os
from pathlib import Path

from profile_worker import guard, save


def main():
    guard()
    import torch
    p = json.loads((Path(os.environ['ARTIFACTS'])/'protocol.json').read_text())
    root = Path(p['remote_root'])/'results'
    job = os.environ['PBS_JOBID']
    modes = p['profiling']['primary_modes'] + ['stream_decode']
    summaries = {m:json.loads((root/m/'summary.json').read_text()) for m in modes}
    expected_keys = {f'{case}_{seed}' for case in p['profiling']['case_ids'] for seed in [0,1]}
    outputs = {m:torch.load(root/m/'actions.pt',map_location='cpu',weights_only=True) for m in modes}
    for mode,s in summaries.items():
        if (not s['complete'] or s['pbs_jobid']!=job or s['source_job']!=p['source']['source_job']
                or s['new_training_updates']!=0 or not s['anchor_passed']):
            raise ValueError('Incomplete/differently bound profile: '+mode)
        if mode!='stream_decode' and (s['queries']!=80 or set(outputs[mode])!=expected_keys):
            raise ValueError('Full frozen 40x2 query panel is incomplete: '+mode)
    if len({s['gpu_uuid'] for s in summaries.values()})!=1:
        raise ValueError('Performance modes were not measured on the same GPU')
    max_abs = max(float((outputs['legacy'][k].float()-outputs['inference_dense'][k].float()).abs().max())
                  for k in expected_keys)
    if max_abs>p['profiling']['parity_atol']:
        raise ValueError(f'Export differs over full paired panel: {max_abs}')
    anchor = torch.load(p['source']['anchor_tensor'],map_location='cpu',weights_only=True)
    stream_error = max(float((v.float()-(anchor['action'] if k=='2000_0' else outputs['legacy'][k]).float()).abs().max())
                       for k,v in outputs['stream_decode'].items())
    if stream_error>p['profiling']['parity_atol'] or set(outputs['stream_decode'])!={'2000_0','3000_0'}:
        raise ValueError('Compressed residency diagnostic parity failed')
    parity = dict(protocol=p['protocol'],pbs_jobid=job,passed=True,full_panel_queries=80,
        legacy_vs_inference_action_max_abs=max_abs,stream_action_max_abs=stream_error,
        anchor={m:dict(action=s['anchor_max_abs'],future=s['anchor_future_max_abs'])
                for m,s in summaries.items() if m!='bf16'})
    save(root/'parity.json',parity)
    storage = json.loads((root/'storage.json').read_text())
    details = {m:json.loads((root/m/'detail.json').read_text()) for m in p['profiling']['primary_modes']}
    attribution = {}
    for mode,detail in details.items():
        categories = defaultdict(float)
        for row in detail['rows']:
            categories[row['category']] += row['gpu_ms']
        linear = [r for r in detail['rows'] if r['category']=='target_linear']
        gemms = [r for r in detail['rows'] if r['category']=='gemm' and r['shape']]
        by_shape = {}
        for row in gemms:
            sh = row['shape']
            stream = ('video' if row['label'].startswith('video_expert.') else
                      'action' if row['label'].startswith('action_expert.') else
                      'proprio' if row['label']=='proprio_encoder' or row['label'].startswith('proprio_encoder.') else 'other')
            key = (stream,sh['M'],sh['N'],sh['K'])
            record = by_shape.setdefault(key,dict(stream=stream,M=sh['M'],N=sh['N'],K=sh['K'],calls=0,gpu_ms=0))
            record['calls'] += row['calls']
            record['gpu_ms'] += row['gpu_ms']
        attribution[mode] = dict(intrusive=True,categories_gpu_ms=dict(categories),
            component_gpu_ms={r['label']:r['gpu_ms'] for r in detail['rows'] if r['category']=='component'},
            stages=[r for r in detail['rows'] if r['category']=='stage'],
            top_target_linears=linear[:20],top_gemm_shapes=sorted(by_shape.values(),key=lambda r:-r['gpu_ms'])[:24],
            cache_payload_bytes=detail['cache_payload_bytes'],
            warning='Nested stage/linear/component/GEMM times overlap; do not add categories')
        for offset in range(0,len(linear),64):
            save(root/'per_layer'/f'{mode}_{offset//64:02d}.json',linear[offset:offset+64])
    baseline = summaries['bf16']['metrics']['full_query_wall_ms']['p50']
    for mode,s in summaries.items():
        s['p50_speedup_vs_bf16'] = baseline/s['metrics']['full_query_wall_ms']['p50']
        errors = []
        if mode!='stream_decode':
            for key in sorted(expected_keys):
                a,b = outputs[mode][key].float(),outputs['bf16'][key].float()
                errors.append(dict(motor_mse=float((a[:10,:6]-b[:10,:6]).square().mean()),
                    gripper_disagreement=float(((a[:10,6]>.5)!=(b[:10,6]>.5)).float().mean())))
            s['first10_motor_rmse_vs_bf16'] = (sum(r['motor_mse'] for r in errors)/len(errors))**.5
            s['first10_gripper_threshold_disagreement'] = sum(r['gripper_disagreement'] for r in errors)/len(errors)
    all_hot = attribution['inference_dense']['top_gemm_shapes']
    hot = [r for r in all_hot if r['stream'] in ('video','action','proprio')]
    component = attribution['inference_dense']['component_gpu_ms']
    kernel_plan = dict(status='implementation_plan_from_measured_profile',native_kernel_implemented=False,
        source_job=p['source']['source_job'],preserve_checkpoint=True,new_training_updates=0,
        priority_shapes=hot[:12],non_target_gemm_bottlenecks=[r for r in all_hot if r['stream']=='other'],
        measured_components_gpu_ms=component,
        phase1='Keep the verified FP32 Smooth/signed-H128/U and A4 rounding. Replace dense F.linear with a packed VQ weight consumer.',
        weight_decoder='Load two uint8 indices per d4 vector; FP32 codeword addition; apply original ldexp(BF16 row gain, B32 exponent-127) in FP32; cast each decoded weight to BF16 before MMA.',
        kernel='Tile-decode directly in registers/shared memory, BF16 Tensor Core MMA with FP32 accumulation and bias epilogue. Never materialize a full decoded weight in global memory.',
        layout='Offline reorder indices for measured N/K tiles; K tile multiple of32 and d4 alignment; 4096-byte BF16 codebooks reusable per layer. Padding must preserve masked tails.',
        phase2='Use measured Smooth/H/U/A4 cost to decide input-transform fusion; preserve operation order and separately validate changed floating-point accumulation.',
        phase3='Consider INT8/INT4/FP4 only as a separately calibrated decoder; current arbitrary BF16 VQ values cannot be sent directly to integer MMA.',
        acceptance='Full80 paired action replay plus saved full action/future anchor, actual compressed residency, profiler evidence of no full-weight decode, kernel and full-policy latency on the same GPU.',
        limitation='No speedup is predicted from bit count. Large-M video GEMMs may be compute-bound; small-M action GEMMs and large-M video GEMMs need separate dispatch decisions.')
    save(root/'kernel_plan.json',kernel_plan)
    summary = dict(protocol=p['protocol'],pbs_jobid=job,complete=True,export_complete=True,
        parity_passed=True,profile_complete=True,new_training_updates=0,
        source_job=p['source']['source_job'],source_arm='reader_kl',queries_per_primary_mode=80,
        modes=summaries,storage=storage,parity=parity,attribution=attribution,
        kernel_plan=kernel_plan,closed_loop_claim=False,native_low_bit_kernel=False,
        scope=p['profiling']['primary_scope'],bf16_comparator='same original checkpoint, same input/noise/10+10 steps, fresh process',
        dependencies='FastWAM code and cached base VAE/text/tokenizer assets remain required; export inference does not load calibration deployments, source banks or original training checkpoint.')
    save(root/'summary.json',summary)
    lines=['# C80 R128 A+Z+reader KL inference export 与 profiling','',
        f'PBS job `{job}`；source `{p["source"]["source_job"]}` / reader_kl；new training updates = 0。',
        '',f'导出：`{storage["bundle_path"]}`。Architecture 与 cached VAE/text/tokenizer assets 仍为运行依赖。',
        '',f'完整40 contexts ×2 seeds action parity max abs = {max_abs:.8g}；另通过保存的完整 action/future anchor。',
        '', '| Mode | n | policy p50 ms | p95 ms | speedup vs BF16 | peak allocated GB | peak reserved GB | resident GB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,s in summaries.items():
        metrics=s['metrics']
        lines.append(f'| {mode} | {s["queries"]} | {metrics["full_query_wall_ms"]["p50"]:.3f} | '
            f'{metrics["full_query_wall_ms"]["p95"]:.3f} | {s["p50_speedup_vs_bf16"]:.3f}x | '
            f'{metrics["peak_allocated_gpu_GB"]["maximum"]:.4f} | {metrics["peak_reserved_gpu_GB"]["maximum"]:.4f} | '
            f'{s["resident_after_warmup"]["allocated_bytes"]/1e9:.4f} |')
    lines += ['', 'Latency 从已准备的 GPU datum 到 action output；观察预处理另报，排除环境和机器人动作执行。',
        'stream_decode 只有2个诊断 queries，逐Linear临时解码完整weight，不作主speedup结果。所有模式使用BF16 GEMM；未实现native VQ low-bit arithmetic。',
        '', f'Packed encoding payload：{storage["encoding"]["unique_storage_bytes"]/2**30:.6f} GiB；'
        f'{storage["effective_encoding_bits_per_target_weight"]:.6f} bits/target weight。',
        f'加入 transforms/U：{storage["target_with_transforms"]["unique_storage_bytes"]/2**30:.6f} GiB；'
        f'{storage["effective_deployment_bits_per_target_weight"]:.6f} bits/target weight。',
        f'Whole inference bundle tensor payload：{storage["whole_bundle"]["unique_storage_bytes"]/2**30:.6f} GiB；'
        f'实际 archive：{storage["torch_file_bytes"]/2**30:.6f} GiB。',
        '', '详细 trace 为单独 intrusive query；nested timing 不能相加，不能替代上表80次无hook测量。',
        '', '| Highest-cost GEMM stream | M | N | K | calls | intrusive GPU ms |',
        '|---|---:|---:|---:|---:|---:|']
    for r in hot[:12]:
        lines.append(f'| {r["stream"]} | {r["M"]} | {r["N"]} | {r["K"]} | {r["calls"]} | {r["gpu_ms"]:.3f} |')
    lines += ['', '实现方案：首版保持当前数值，采用 indices tile-load → 双codeword FP32相加 → B32 scale → BF16取整 → BF16 MMA/FP32累加 → bias；移除global memory里的完整decoded weight。',
        '按上表实际shapes选择dispatch；再根据已测Smooth/H/U/A4时间决定融合输入变换。当前checkpoint不直接改成INT8/INT4/FP4。',
        '', '内存的load transient peak、warm resident、query peak分别保存；不得把packed archive压缩比解释为整模型显存压缩比。',
        '该报告提供固定输入的部署与性能证据，未新增闭环episodes。']
    (root/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
    save(root/'progress.json',dict(protocol=p['protocol'],pbs_jobid=job,phase='complete'))
    print('PROFILE_COMPLETE '+json.dumps(dict(pbs_jobid=job,parity_max_abs=max_abs)),flush=True)


if __name__=='__main__':
    main()

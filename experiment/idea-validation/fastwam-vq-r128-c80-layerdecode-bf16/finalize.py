"""Compare the frozen full panel and enforce the first-round gate before BF16 runs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

FIRST = ('bf16', 'eager', 'input_fused', 'layer_decode_fp32')
SECOND = ('layer_decode_bf16_u', 'layer_decode_bf16_all')


def compare(torch, actual, reference, keys):
    differences = [actual[k].float() - reference[k].float() for k in sorted(keys)]
    if not all(bool(torch.isfinite(d).all()) for d in differences):
        raise ValueError('Non-finite output comparison')
    return dict(action_max_abs=max(float(d.abs().max()) for d in differences),
        full32_action_rmse=float(torch.stack([d.square().mean() for d in differences]).mean().sqrt()),
        first10_motor_rmse=float(torch.stack([d[:10,:6].square().mean() for d in differences]).mean().sqrt()),
        full32_motor_rmse=float(torch.stack([d[:,:6].square().mean() for d in differences]).mean().sqrt()),
        first10_gripper_disagreements=sum(int(((actual[k][:10,6]>.5)!=(reference[k][:10,6]>.5)).sum()) for k in keys),
        first10_gripper_denominator=len(keys)*10)


def first_gate(p, summaries, comparisons, validation):
    reference, candidate = summaries['eager'], summaries['layer_decode_fp32']
    checks = {}
    for mode in FIRST:
        row = summaries[mode]
        checks[mode + '_full80'] = row.get('complete') is True and row.get('queries') == 80
        checks[mode + '_source'] = row.get('source_job') == p['source']['source_job'] and row.get('new_training_updates') == 0
        checks[mode + '_flags'] = (row.get('deterministic_algorithms') is True
            and row.get('fill_uninitialized_memory') is True and row.get('allow_tf32') is False
            and row.get('allow_bf16_reduced_precision_reduction') is True
            and row.get('compiled') is False and row.get('cuda_graph') is False)
        if mode != 'bf16':
            coverage = row.get('coverage', {})
            checks[mode + '_coverage'] = coverage.get('modules') == 614 and coverage.get('timed_linear_calls') == 80*6436
            checks[mode + '_anchor'] = all(isinstance(row.get(k), (int,float)) and math.isfinite(row[k]) and row[k] <= 1e-6
                for k in ('anchor_action_max_abs', 'anchor_future_max_abs'))
    coverage = candidate.get('coverage', {})
    checks['compressed_residency'] = (coverage.get('encoded_target_modules') == 614
        and coverage.get('dense_target_weights') == 0 and coverage.get('decoded_calls') == 80*6436
        and coverage.get('scratch_bytes', 0) > 0)
    checks['same_gpu_family'] = all('A100' in summaries[m].get('gpu', '') for m in FIRST)
    checks['decoder_all614'] = (validation.get('complete') is True and validation.get('decoder_modules') == 614
        and validation.get('decoder_all_bitwise') is True and validation.get('linear_all_exact') is True
        and validation.get('linear_shape_count') == 12)
    for mode in ('input_fused', 'layer_decode_fp32'):
        checks[mode + '_full80_output_alignment'] = comparisons[mode]['action_max_abs'] <= 1e-6
    ratios = {}
    for metric in ('p50', 'mean'):
        ratio = candidate['metrics']['full_query_wall_ms'][metric] / reference['metrics']['full_query_wall_ms'][metric]
        ratios['latency_' + metric] = ratio
        checks['lower_latency_' + metric] = math.isfinite(ratio) and ratio < 1
    ratio = candidate['metrics']['peak_allocated_gpu_GB']['maximum'] / reference['metrics']['peak_allocated_gpu_GB']['maximum']
    ratios['query_peak_allocated'] = ratio
    checks['lower_query_peak'] = math.isfinite(ratio) and ratio < 1
    ratio = candidate['load_transient_memory']['peak_allocated_bytes'] / reference['load_transient_memory']['peak_allocated_bytes']
    ratios['load_peak_allocated'] = ratio
    checks['lower_load_peak'] = math.isfinite(ratio) and ratio < 1
    return dict(passed=all(checks.values()), reference='eager', candidate='layer_decode_fp32',
                checks=checks, failed_checks=[k for k,v in checks.items() if not v], ratios=ratios,
                comparison_scope='full40 contexts x2 seeds on A100; existing reference panels reused by user instruction; action alignment plus source anchor future')


def load_rows(p, root, modes, torch):
    summaries, outputs = {}, {}
    keys = {f'{c}_{s}' for c in p['profiling']['case_ids'] for s in p['profiling']['seed_indices']}
    for mode in modes:
        summary = json.loads((root / mode / 'summary.json').read_text())
        reused = mode in p.get('reference_reuse', {}).get('modes', [])
        expected_protocol = p['reference_reuse']['profiling_protocol'] if reused else p['protocol']
        expected_job = p['reference_reuse']['profiling_job'] if reused else os.environ['PBS_JOBID']
        if (summary['protocol'] != expected_protocol or summary['pbs_jobid'] != expected_job
                or summary['mode'] != mode or not summary['complete'] or summary['queries'] != 80):
            raise ValueError('Incomplete or unbound measurement: ' + mode)
        if reused:
            # The old receipt omitted this flag; its frozen worker calls this recorded source.
            source = Path(p['reference_reuse']['artifact_root']) / 'reference_profile.py'
            if 'torch.backends.cuda.matmul.allow_tf32 = False' not in source.read_text():
                raise ValueError('Missing frozen TF32 flag evidence for reused references')
            summary['allow_tf32'] = False
            summary['reused_reference'] = dict(profiling_job=expected_job,
                profiling_protocol=expected_protocol, flag_evidence=str(source),
                basis=p['reference_reuse']['basis'])
        actions = torch.load(root / mode / 'actions.pt', map_location='cpu', weights_only=True)
        if set(actions) != keys or any(tuple(a.shape) != (32,7) or not bool(torch.isfinite(a).all()) for a in actions.values()):
            raise ValueError('Wrong full-panel action coverage or shape: ' + mode)
        samples = json.loads((root / mode / 'samples.json').read_text())
        if len(samples) != 80 or {f"{r['case_id']}_{r['seed_index']}" for r in samples} != keys:
            raise ValueError('Wrong latency cohort: ' + mode)
        fields = ('full_query_wall_ms', 'denoising_latency_ms', 'peak_allocated_gpu_GB', 'peak_reserved_gpu_GB')
        if any(not math.isfinite(r[k]) or r[k] < 0 for r in samples for k in fields):
            raise ValueError('Non-finite performance sample: ' + mode)
        summaries[mode], outputs[mode] = summary, actions
    new_rows = [r for r in summaries.values() if not r.get('reused_reference')]
    if len({r['gpu_uuid'] for r in new_rows}) != 1:
        raise ValueError('New candidate modes used different GPUs')
    return summaries, outputs, keys


def main():
    from reference_profile import guard, save
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('first','final'), required=True)
    args = parser.parse_args()
    p = json.loads((Path(os.environ['ARTIFACTS']) / 'protocol.json').read_text())
    root = Path(p['remote_root']) / 'results'
    import torch
    summaries, outputs, keys = load_rows(p, root, FIRST, torch)
    comparisons = {m:compare(torch, outputs[m], outputs['eager'], keys) for m in FIRST}
    validation = json.loads((root/'kernel_validation.json').read_text())
    if validation['pbs_jobid'] != os.environ['PBS_JOBID'] or validation['protocol'] != p['protocol']:
        raise ValueError('Decoder validation job binding differs')
    gate = first_gate(p, summaries, comparisons, validation)
    gate.update(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], full80_errors_vs_eager=comparisons)
    save(root/'stage1_gate.json', gate)
    print('FIRST_ROUND_GATE ' + json.dumps(gate), flush=True)
    if args.phase == 'first':
        return 0 if gate['passed'] else 3
    modes = list(FIRST)
    if gate['passed']:
        summaries, outputs, keys = load_rows(p, root, FIRST+SECOND, torch)
        modes += list(SECOND)
    for mode in modes:
        row = summaries[mode]
        if mode in SECOND:
            coverage = row.get('coverage', {})
            if (coverage.get('modules') != 614 or coverage.get('timed_linear_calls') != 80*6436
                    or coverage.get('decoded_calls') != 80*6436 or coverage.get('encoded_target_modules') != 614
                    or coverage.get('dense_target_weights') != 0):
                raise ValueError('Incomplete BF16 target coverage: ' + mode)
            for flag in ('deterministic_algorithms','fill_uninitialized_memory','allow_bf16_reduced_precision_reduction'):
                if row.get(flag) is not True: raise ValueError('Different BF16 measurement flag: ' + flag)
            if row.get('allow_tf32') is not False or row.get('new_training_updates') != 0:
                raise ValueError('BF16 precision flags or training scope differ')
        row['errors_vs_eager'] = compare(torch, outputs[mode], outputs['eager'], keys)
        row['errors_vs_layer_decode_fp32'] = compare(torch, outputs[mode], outputs['layer_decode_fp32'], keys)
        row['errors_vs_bf16'] = compare(torch, outputs[mode], outputs['bf16'], keys)
        latency = row['metrics']['full_query_wall_ms']['p50']
        row['p50_speedup_vs_eager'] = summaries['eager']['metrics']['full_query_wall_ms']['p50']/latency
        row['p50_speedup_vs_bf16'] = summaries['bf16']['metrics']['full_query_wall_ms']['p50']/latency
    summary = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], complete=gate['passed'],
        stage1_gate=gate, stage2_complete=gate['passed'],
        stage2_status='complete' if gate['passed'] else 'not_run_first_gate_failed',
        source_job=p['source']['source_job'], modes=summaries, full_queries_per_mode=80,
        new_training_updates=0, closed_loop_episodes=0, native_integer_gemm=False,
        storage=json.loads((root/'storage.json').read_text()),
        main_scope=p['profiling']['primary_scope'],
        precision_boundary='BF16 U uses native BF16 GEMM with FP32 accumulation; broader input variant retains FP32 helper operations; weight decoding unchanged')
    save(root/'summary.json', summary)
    report = ['# C80 R128 A+Z+KL：逐层解码与 BF16 对照','',
        '固定40 contexts x2 seeds，各80 queries；已有对照复用25943391结果，新候选在同一张A100完成。零训练更新，零新闭环。',
        'GPU memory使用decimal GB，weight storage使用GiB。主计时无profiler。','',
        '| Mode | Mean s | p50 s | p95 s | Speedup vs eager | Speedup vs BF16 | Query peak GB | Load peak GB | First10 motor RMSE vs first round | Gripper disagreement /800 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in modes:
        r=summaries[mode];lat=r['metrics']['full_query_wall_ms'];err=r['errors_vs_layer_decode_fp32']
        report.append(f"| {mode} | {lat['mean']/1000:.6f} | {lat['p50']/1000:.6f} | {lat['p95']/1000:.6f} | {r['p50_speedup_vs_eager']:.4f} | {r['p50_speedup_vs_bf16']:.4f} | {r['metrics']['peak_allocated_gpu_GB']['maximum']:.6f} | {r['load_transient_memory']['peak_allocated_bytes']/1e9:.6f} | {err['first10_motor_rmse']:.9g} | {err['first10_gripper_disagreements']} |")
    report += ['', 'First-round gate: '+str(gate['passed']), 'Failed checks: '+str(gate['failed_checks']),
        'BF16 intermediate arithmetic can change outputs; fixed-input errors do not establish closed-loop equivalence.',
        'Load durations depend on cache/order and are not a fair cold-start speedup comparison.',
        'Intrusive operators.json records are localization evidence and are not an additive E2E decomposition.','']
    (root/'RESULTS.zh.md').write_text('\n'.join(report))
    print('LAYERDECODE_PIPELINE_FINISHED ' + json.dumps(dict(first_gate=gate['passed'], modes=modes)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

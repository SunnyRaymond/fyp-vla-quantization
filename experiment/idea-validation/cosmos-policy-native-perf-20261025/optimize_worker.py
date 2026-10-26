"""Validate an inference-only engine candidate against all 80 saved reference queries."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.old_root))
    import common
    common.guard()
    from engine_profile import _install_arm
    import engine_optimized
    import validation
    import benchmark
    protocol = common.read_json(args.old_root/'protocol.json')
    out = args.root/'results/optimized_candidate_v1'
    out.mkdir(parents=True, exist_ok=True)
    runtime = common.prepare_runtime(protocol, args.old_root, out)
    torch = common.torch_module()
    device = next(runtime.model.parameters()).device
    props = torch.cuda.get_device_properties(device)
    device_key = 'NVIDIA_A100-SXM4-40GB_sm80'
    reference_dir = args.old_root/'results/ours/engine_validation'/device_key
    manifest = common.read_json(args.old_root/'results/ours/deployment/manifest.json')
    expected = validation.identity(protocol, manifest,
                                   {'name': props.name, 'capability': [props.major, props.minor]})
    if not validation.valid_receipt(common.read_json(reference_dir/'summary.json'), expected):
        raise RuntimeError('Frozen 80-query reference receipt does not match this artifact/device')
    _install_arm(runtime, protocol, args.old_root, out, 'ours_vqwam', {})
    diagnostic_keys = [(3000, 0), (3009, 1), (3020, 0), (3039, 1)]
    reference_timings = []
    for case, seed in diagnostic_keys[:3]:
        benchmark.run_one(common, runtime, common.get_context(runtime, case, seed),
                          torch=torch, device=device)
    for case, seed in diagnostic_keys:
        context = common.get_context(runtime, case, seed)
        for repetition in range(3):
            torch.cuda.reset_peak_memory_stats(device)
            row = benchmark.run_one(common, runtime, context, torch=torch, device=device)
            row.update(case_id=case, seed_index=seed, repetition=repetition,
                       peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1e9)
            reference_timings.append(row)
    common.save_json(out/'current_engine_diagnostic.json', {
        'pbs_job_id': os.environ['PBS_JOBID'], 'timings': reference_timings,
        'scope': 'same-allocation current-engine diagnostic; not the formal 720-sample panel'})
    engine_optimized.install(runtime, protocol, args.old_root, out)
    checks, timings = [], []
    for case, seed in validation.panel_keys(protocol):
        reference = common.load_tensors(reference_dir/'reference'/f'{case}_{seed}.pt')
        if reference.get('identity') != expected or reference.get('key') != [case, seed]:
            raise RuntimeError('Stored reference query binding differs')
        context = common.get_context(runtime, case, seed)
        result = common.endpoint(runtime, context)
        row = {'case_id': case, 'seed_index': seed, **validation._compare(result, reference)}
        checks.append(row)
        common.save_json(out/'gate_progress.json', {'completed': len(checks), 'expected': 80, 'latest': row})
        del result, reference
        if not row['passed']:
            common.save_json(out/'gate_failed.json', row)
            raise RuntimeError('Optimized engine failed the unchanged full80 action/future tolerance')
        print(f'OPTIMIZED_PARITY {len(checks)}/80 case={case} seed={seed}', flush=True)
    for case, seed in diagnostic_keys:
        context = common.get_context(runtime, case, seed)
        for repetition in range(3):
            torch.cuda.reset_peak_memory_stats(device)
            row = benchmark.run_one(common, runtime, context, torch=torch, device=device)
            row.update(case_id=case, seed_index=seed, repetition=repetition,
                       peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1e9)
            timings.append(row)
    reference_mean = statistics.fmean(row['denoising_seconds'] for row in reference_timings)
    candidate_mean = statistics.fmean(row['denoising_seconds'] for row in timings)
    common.save_json(out/'summary.json', {'status': 'complete', 'pbs_job_id': os.environ['PBS_JOBID'],
        'source_job': manifest['source_job'], 'completed_updates': manifest['completed_updates'],
        'variant': 'all_nfe_graph_recompute_cross_kv_v1', 'new_training_updates': 0,
        'new_closed_loop_episodes': 0, 'full80_gate': checks, 'gate_queries': 80,
        'diagnostic_timings_only': timings, 'current_engine_diagnostic_timings': reference_timings,
        'same_allocation_diagnostic_comparison': {
            'current_mean_seconds': reference_mean, 'candidate_mean_seconds': candidate_mean,
            'speedup_vs_current': reference_mean/candidate_mean,
            'candidate_mean_improved': candidate_mean < reference_mean,
            'scope': 'four representative queries x three repeats each; selection diagnostic, not formal six-arm performance'},
        'engine': runtime.install_metadata,
        'graphs': runtime.optimized_vq_graphs.metadata()})


if __name__ == '__main__':
    main()

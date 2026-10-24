"""Fixed-panel profiling for reusable layer-wise VQ decode paths; run inside PBS only."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time


MODES = (
    'bf16', 'eager', 'input_fused', 'layer_decode_fp32',
    'layer_decode_bf16_u', 'layer_decode_bf16_all',
)
PREFLIGHT_SHAPES = (
    (294, 3072, 3072), (32, 3072, 1024), (129, 3072, 3072),
    (32, 1024, 3072), (129, 3072, 1024), (294, 14336, 3072),
    (294, 3072, 14336), (32, 1024, 4096), (32, 4096, 1024),
    (1, 4096, 8), (32, 7, 1024), (32, 1024, 7),
)
EXPECTED_MODULES = 614
EXPECTED_LINEAR_CALLS = 514880


def validate_panel(settings):
    """Reject a reduced or shifted panel before loading the model."""
    expected_cases = list(range(3000, 3040))
    if settings.get('case_ids') != expected_cases:
        raise ValueError('Frozen profile panel must be cases 3000-3039')
    if settings.get('seed_indices') != [0, 1]:
        raise ValueError('Frozen profile panel must use both paired seeds')
    if settings.get('warmup_queries') != 3 or settings.get('queries_per_mode') != 80:
        raise ValueError('Frozen profile requires three warmups and 80 measured queries')
    return expected_cases


def validate_coverage(module_count, linear_calls):
    if module_count != EXPECTED_MODULES or linear_calls != EXPECTED_LINEAR_CALLS:
        raise ValueError(
            f'Incomplete formal coverage: modules={module_count}, calls={linear_calls}; '
            f'expected {EXPECTED_MODULES}/{EXPECTED_LINEAR_CALLS}'
        )


def progress(root, p, mode, phase, **fields):
    from reference_profile import save

    row = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'],
               mode=mode, phase=phase, **fields)
    save(root / 'progress.json', row)
    print('PROFILE_PROGRESS ' + json.dumps(row), flush=True)


def _reset_counter(owner, name):
    if owner is None or not hasattr(owner, name):
        return
    value = getattr(owner, name)
    if hasattr(value, 'clear'):
        value.clear()
    elif isinstance(value, (int, float)):
        setattr(owner, name, type(value)(0))


def _counter_total(counter):
    if counter is None:
        return None
    if isinstance(counter, (int, float)):
        return int(counter)
    return int(sum(counter.values()))


def _check_gpu_family(torch, p):
    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    allocation = p.get('allocation', {})
    expected = (p.get('hardware', {}).get('gpu_family')
                or allocation.get('gpu_family')
                or p.get('expected_gpu_family'))
    if expected is None and allocation.get('execution_queue') == 'g1':
        expected = 'A100'
    if expected and str(expected).lower().replace(' family', '') not in name.lower():
        raise RuntimeError(f'Expected {expected} GPU family, got {name}')
    return props, name


def _install_mode(out, p, bundle, mode):
    if mode in ('eager', 'input_fused'):
        import kernel_runtime

        runtime = kernel_runtime.make_runtime(out, p, bundle, direct=False)
        kernel_runtime.install(runtime, bundle, packed=False,
                               fused=(mode != 'eager'), direct=False)
    else:
        import layer_runtime

        runtime = layer_runtime.make_runtime(out, p, bundle, direct=True)
        layer_runtime.install(runtime, bundle, mode=mode)
    return runtime


def _validation_rows(runtime):
    banks = getattr(runtime, 'banks', None)
    return getattr(banks, 'validation', {}) if banks is not None else {}


def _check_first_round(mode, runtime):
    rows = _validation_rows(runtime)
    if mode == 'input_fused' and len(rows) != EXPECTED_MODULES:
        raise ValueError(f'Input operand checks covered {len(rows)}/614 modules')
    if mode.startswith('layer_decode_'):
        if len(rows) != EXPECTED_MODULES:
            raise ValueError(f'Layer decoder diagnostics covered {len(rows)}/614 modules')
        pending = [name for name, row in rows.items()
                   if isinstance(row, dict) and row.get('status') == 'pending']
        if pending:
            raise ValueError(f'First-round layer diagnostics still pending for {len(pending)} modules')
        if mode == 'layer_decode_fp32':
            for name, row in rows.items():
                if row.get('smooth_h_max_abs') != 0 or row.get('a4_bitwise') is not True:
                    raise ValueError(f'FP32 operand gate failed at {name}: {row}')
    banks = getattr(runtime, 'banks', None)
    if banks is not None and hasattr(banks, 'validate_input'):
        banks.validate_input = False


def profile(p, root, mode):
    import numpy as np
    import torch
    import calibration_source
    import export_runtime as export
    from reference_profile import max_abs_difference, memory, save, statistics

    if mode not in MODES:
        raise ValueError(f'Unsupported mode: {mode}')
    settings = p['profiling']
    validate_panel(settings)
    plan_path = Path(p['sources']['calibration_plan'])
    plan = json.loads(plan_path.read_text())
    if plan.get('test_case_ids', settings['case_ids']) != settings['case_ids']:
        raise ValueError('Frozen calibration cases changed')
    rows = {row['case_id']: row for row in plan['inputs']}
    if not set(settings['case_ids'] + [2000]).issubset(rows):
        raise ValueError('Frozen plan is missing the anchor or a measured case')

    out = root / mode
    out.mkdir(parents=True, exist_ok=True)
    progress(root, p, mode, 'loading')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if mode == 'bf16':
        import runner

        runtime = runner.Runtime(out)
        runtime.arm('bf16')
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)
        bundle = None
    else:
        bundle = export.load_bundle(p['export_reuse']['bundle_path'])
        runtime = _install_mode(out, p, bundle, mode)
        del bundle
        bundle = None
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    load_memory = memory(torch)

    def datum(case_id):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        observation, metadata = calibration_source.load_observation(
            np, plan_path.parent, rows[case_id])
        runtime.description = metadata['description']
        value = runtime.datum(observation)
        torch.cuda.synchronize()
        return value, (time.perf_counter() - begin) * 1000

    progress(root, p, mode, 'anchor_and_warmup')
    value, _ = datum(2000)
    anchor_seed = rows[2000]['sampler_seeds'][0]
    action, _ = runtime.infer(value, anchor_seed, measure=False)
    anchor = torch.load(p['source']['anchor_tensor'], map_location='cpu', weights_only=True)
    anchor_action_error = max_abs_difference(action, anchor['action'])
    if not bool(torch.isfinite(action).all()):
        raise ValueError('Non-finite source-anchor action')
    import sampler

    with torch.no_grad():
        replay, counts, future = sampler.differentiable_action(
            runtime, value, anchor_seed, checkpointed=False, return_future=True)
    if counts != {'video_steps': 10, 'action_steps': 10}:
        raise ValueError(f'Sampler step counts changed: {counts}')
    anchor_future_error = max_abs_difference(future, anchor['future'])
    if not bool(torch.isfinite(future).all()):
        raise ValueError('Non-finite source-anchor future')
    anchor_future_nrmse = float(
        (future.detach().float().cpu() - anchor['future'].float()).square().mean().sqrt()
        / anchor['future'].float().square().mean().sqrt().clamp_min(1e-12)
    )
    strict_anchor = mode in ('eager', 'input_fused', 'layer_decode_fp32')
    if strict_anchor and max(anchor_action_error, anchor_future_error) > settings['parity_atol']:
        raise ValueError(
            f'Anchor action/future gate failed for {mode}: '
            f'{anchor_action_error}, {anchor_future_error}'
        )
    _check_first_round(mode, runtime)
    validation = _validation_rows(runtime)
    save(out / 'validation.json', dict(
        mode=mode, anchor_action_max_abs=anchor_action_error,
        anchor_future_max_abs=anchor_future_error,
        anchor_future_nrmse=anchor_future_nrmse,
        strict_anchor_gate=strict_anchor,
        operand_modules_checked=len(validation), operand_checks=validation,
    ))
    del value, action, anchor, replay, future

    value, _ = datum(3000)
    for _ in range(settings['warmup_queries']):
        runtime.infer(value, rows[3000]['sampler_seeds'][0], measure=False)
    del value
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = memory(torch)
    banks = getattr(runtime, 'banks', None)
    if banks is not None:
        banks.calls.clear()
        banks.seen.clear()
        _reset_counter(banks, 'decoded_calls')
    _reset_counter(runtime, 'decoded_calls')
    progress(root, p, mode, 'measuring', expected=80)

    samples, actions = [], {}
    for case_id in settings['case_ids']:
        value, preparation_ms = datum(case_id)
        for seed_index in settings['seed_indices']:
            seed = rows[case_id]['sampler_seeds'][seed_index]
            action, metrics = runtime.infer(value, seed, measure=True)
            if not bool(torch.isfinite(action).all()):
                raise ValueError(f'Non-finite action at case {case_id}, seed {seed_index}')
            actions[f'{case_id}_{seed_index}'] = action.detach().cpu()
            samples.append(dict(case_id=case_id, seed_index=seed_index,
                                preprocessing_wall_ms=preparation_ms, **metrics))
            del action
        del value
        progress(root, p, mode, 'measuring', queries=len(samples), expected=80)
    if len(samples) != 80:
        raise ValueError(f'Expected 80 measured queries, got {len(samples)}')
    save(out / 'samples.json', samples)
    torch.save(actions, out / 'actions.pt')

    props, gpu_name = _check_gpu_family(torch, p)
    metadata = getattr(runtime, 'install_metadata', {})
    summary = dict(
        protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode,
        complete=True, queries=len(samples), source_job=p['source']['source_job'],
        new_training_updates=0,
        metrics={key: statistics(samples, key) for key in (
            'full_query_wall_ms', 'denoising_latency_ms',
            'peak_allocated_gpu_GB', 'peak_reserved_gpu_GB',
            'baseline_allocated_gpu_GB')},
        load_seconds=load_seconds, load_transient_memory=load_memory,
        resident_after_warmup=resident,
        anchor_action_max_abs=anchor_action_error,
        anchor_future_max_abs=anchor_future_error,
        anchor_future_nrmse=anchor_future_nrmse,
        strict_anchor_gate=strict_anchor, installation=metadata,
        gpu=gpu_name, gpu_uuid=str(props.uuid),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        allow_bf16_reduced_precision_reduction=(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        scope=settings['primary_scope'],
        instrumentation_used_for_primary_latency=False,
        compiled=False, cuda_graph=False, native_vq_kernel=False,
        native_integer_gemm=False,
    )
    if banks is not None:
        modules = len(banks.seen)
        calls = sum(banks.calls.values())
        validate_coverage(modules, calls)
        modules_by_name = getattr(runtime, 'q', None)
        target_modules = getattr(modules_by_name, 'modules', {})
        dense_count = sum(module.weight is not None for module in target_modules.values())
        encodings = getattr(banks, 'encodings', {})
        is_layer_decode = mode.startswith('layer_decode_')
        if is_layer_decode and (len(encodings) != EXPECTED_MODULES or dense_count != 0):
            raise ValueError(
                f'Layer decode must retain 614 encodings and zero dense target weights; '
                f'got {len(encodings)}/{dense_count}'
            )
        decoded_calls = getattr(banks, 'decoded_calls', getattr(runtime, 'decoded_calls', None))
        decoded_call_count = _counter_total(decoded_calls)
        if is_layer_decode and decoded_call_count != EXPECTED_LINEAR_CALLS:
            raise ValueError(
                f'Layer decoder coverage mismatch: {decoded_call_count} decoded calls, '
                f'expected {EXPECTED_LINEAR_CALLS}'
            )
        summary['coverage'] = dict(
            modules=modules, timed_linear_calls=calls,
            packed_modules=len(encodings), encoded_target_modules=len(encodings),
            dense_target_weights=dense_count,
            decoded_calls=decoded_call_count,
            scratch_bytes=getattr(banks, 'scratch_bytes', 0),
            native_vq_kernel=False, native_integer_gemm=False,
        )
    save(out / 'summary.json', summary)

    if mode in settings.get('diagnostic_profile_modes', []):
        progress(root, p, mode, 'diagnostic_profiler')
        value, _ = datum(3000)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=True, with_stack=False) as profiler:
            runtime.infer(value, rows[3000]['sampler_seeds'][0], measure=False)
        profiler.export_chrome_trace(str(out / 'trace.json'))
        operators = [dict(name=event.key[:240], calls=event.count,
                          self_device_us=getattr(event, 'self_device_time_total', 0),
                          self_cpu_us=event.self_cpu_time_total)
                     for event in profiler.key_averages()]
        operators.sort(key=lambda row: -row['self_device_us'])
        save(out / 'operators.json', dict(intrusive=True, rows=operators[:160]))
    progress(root, p, mode, 'complete', queries=len(samples))


def preflight(p, root):
    """Validate every encoded decoder and representative F.linear shapes inside PBS."""
    import torch
    import export_runtime as export
    import micro_codebook as micro
    import layer_runtime
    from reference_profile import save

    out = root / 'preflight'
    out.mkdir(parents=True, exist_ok=True)
    progress(root, p, 'preflight', 'loading')
    props, gpu_name = _check_gpu_family(torch, p)
    bundle = export.load_bundle(p['export_reuse']['bundle_path'])
    runtime = layer_runtime.make_runtime(out, p, bundle, direct=True)
    layer_runtime.install(runtime, bundle, mode='layer_decode_fp32')
    banks = runtime.banks
    encodings = banks.encodings
    prepared = banks.prepared
    scratch = banks.scratch
    decoder_rows = []
    by_shape = {}
    if len(encodings) != EXPECTED_MODULES:
        raise ValueError(f'Preflight expected 614 encodings, got {len(encodings)}')
    for index, (name, encoding) in enumerate(encodings.items()):
        expected = micro.decode_vq(encoding)
        decoded = layer_runtime.decode_prepared(prepared[name], scratch)
        if decoded.shape != expected.shape or not torch.equal(
                decoded.view(torch.int16), expected.view(torch.int16)):
            raise ValueError(f'Scratch decoder raw BF16 mismatch at {name}')
        shape = tuple(map(int, encoding['shape']))
        by_shape.setdefault(shape, name)
        decoder_rows.append(dict(module=name, shape=list(shape), raw_bf16_equal=True))
        del expected, decoded
        if index % 64 == 0:
            progress(root, p, 'preflight', 'decoder_checks', modules=index + 1,
                     expected=EXPECTED_MODULES)

    linear_rows = []
    torch.manual_seed(20261023)
    for m, n, k in PREFLIGHT_SHAPES:
        name = by_shape.get((n, k))
        if name is None:
            raise ValueError(f'No encoded target module for preflight shape {(m, n, k)}')
        expected = micro.decode_vq(encodings[name])
        decoded = layer_runtime.decode_prepared(prepared[name], scratch)
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        bias = torch.zeros(n, device='cuda', dtype=torch.bfloat16)
        reference = torch.nn.functional.linear(x, expected, bias)
        actual = torch.nn.functional.linear(x, decoded, bias)
        exact = torch.equal(actual.view(torch.int16), reference.view(torch.int16))
        if not exact:
            raise ValueError(f'F.linear parity failed for {(m, n, k)}')
        linear_rows.append(dict(M=m, N=n, K=k, module=name, exact=True))
        del expected, decoded, x, bias, reference, actual
    if len(decoder_rows) != EXPECTED_MODULES or len(linear_rows) != len(PREFLIGHT_SHAPES):
        raise ValueError('Incomplete scratch decoder preflight')
    save(root / 'kernel_validation.json', dict(
        protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], complete=True,
        decoder_modules=len(decoder_rows), decoder_all_bitwise=True,
        decoder_rows=decoder_rows, linear_shape_count=len(linear_rows),
        linear_all_exact=True, linear_rows=linear_rows,
        gpu=gpu_name, gpu_uuid=str(props.uuid),
        native_vq_kernel=False, native_integer_gemm=False, new_training_updates=0,
    ))
    progress(root, p, 'preflight', 'complete', modules=len(decoder_rows),
             linear_shapes=len(linear_rows))


def main():
    # Import the guard before Torch; this script must never load a model outside PBS.
    from reference_profile import guard

    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['preflight', *MODES])
    args = parser.parse_args()
    p = json.loads((Path(os.environ['ARTIFACTS']) / 'protocol.json').read_text())
    root = Path(p['remote_root']) / 'results'
    import torch
    from reference_profile import freeze_numerics

    freeze_numerics(torch)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(20261023)
    _check_gpu_family(torch, p)
    if args.mode == 'preflight':
        preflight(p, root)
    else:
        profile(p, root, args.mode)


if __name__ == '__main__':
    main()

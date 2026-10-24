"""Fixed-panel runtime profiling; invoke only inside the approved PBS allocation."""
from __future__ import annotations

import argparse
import faulthandler
from collections.abc import Mapping
import gc
import json
import os
from pathlib import Path
import signal
import time


MODES = ('c_decoder', 'c_buffer')
EXPECTED_CASES = list(range(3000, 3040))
EXPECTED_SEEDS = [0, 1]
EXPECTED_MODULES = 614
EXPECTED_STEPS = {'steps_video': 10, 'steps_action': 10, 'batch_size': 1,
                  'action_horizon': 32, 'video_frames': 9}
_FAILURE_ROOT = None
_FAILURE_DIR = None
_PHASE = 'startup'
_MODE = None


def validate_panel(settings):
    """Reject a reduced or shifted panel and any changed query geometry."""
    if settings.get('case_ids') != EXPECTED_CASES:
        raise ValueError('Frozen profile panel must be cases 3000-3039')
    if settings.get('seed_indices') != EXPECTED_SEEDS:
        raise ValueError('Frozen profile panel must use both paired seeds')
    if settings.get('warmup_queries') != 3 or settings.get('queries_per_mode') != 80:
        raise ValueError('Frozen profile requires three warmups and 80 measured queries')
    for key, expected in EXPECTED_STEPS.items():
        if settings.get(key) != expected:
            raise ValueError(f'Frozen inference geometry changed: {key}={settings.get(key)!r}')
    return list(EXPECTED_CASES)


def validate_module_coverage(summary):
    """Require all target modules to have been exercised during anchor/warmup."""
    modules = summary.get('module_count', summary.get('modules'))
    seen = summary.get('seen_count', summary.get('seen_modules'))
    if modules != EXPECTED_MODULES or seen != EXPECTED_MODULES:
        raise ValueError(f'Expected all 614 target modules hit; got modules={modules}, seen={seen}')


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _coverage_summary(banks):
    value = getattr(banks, 'coverage_summary', {})
    return _jsonable(value() if callable(value) else value)


def _bank_actual_counters(banks):
    calls = getattr(banks, 'calls', {})
    decoded = getattr(banks, 'decoded_calls', {})
    return dict(
        python_seen_modules=len(getattr(banks, 'seen', ())),
        python_target_linear_calls=sum(calls.values()) if hasattr(calls, 'values') else None,
        python_decoded_calls=sum(decoded.values()) if hasattr(decoded, 'values') else None,
    )


def _decoder_coverage(banks):
    decoded = getattr(banks, 'decoded_calls', {})
    hit_modules = {name for name, count in decoded.items() if count} if hasattr(decoded, 'items') else set()
    return dict(
        expected_modules=EXPECTED_MODULES,
        target_modules=len(getattr(banks, 'modules', ())),
        decoded_module_hits=len(hit_modules),
        decoded_calls=sum(decoded.values()) if hasattr(decoded, 'values') else None,
        all_614_modules_hit=len(hit_modules) == EXPECTED_MODULES,
    )


def _optimization_report(runtime):
    optimization = getattr(runtime, 'optimization', None)
    if optimization is None or not callable(getattr(optimization, 'report', None)):
        raise RuntimeError('Runtime must expose optimization.report()')
    return _jsonable(optimization.report())


def validate_optimization_coverage(report):
    logical_modules = report.get('logical_coverage', {}).get('modules')
    if logical_modules != EXPECTED_MODULES:
        raise ValueError(f'Expected logical coverage of 614 modules, got {logical_modules}')
    validation = report.get('operand_validation', {})
    if validation:
        checked = validation.get('checked_modules')
        passed = validation.get('bitwise_passed')
        if checked != EXPECTED_MODULES or passed != EXPECTED_MODULES or validation.get('failed'):
            raise ValueError(f'Operand validation did not pass all 614 modules: {validation}')


def _check_gpu_family(torch, p):
    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    allocation = p.get('allocation', {})
    expected = (p.get('hardware', {}).get('gpu_family')
                or allocation.get('gpu_family')
                or p.get('expected_gpu_family') or 'A100')
    if expected and str(expected).lower().replace(' family', '') not in name.lower():
        raise RuntimeError(f'Expected {expected} GPU family, got {name}')
    return props, name


def _freeze_numerics(torch):
    from reference_profile import freeze_numerics

    freeze_numerics(torch)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.manual_seed(20261023)


def _capture_anchor(runtime, value, seed, torch):
    """Capture only the native action path's video latents for the anchor query."""
    model = runtime.model
    original = model.infer_action
    captured = []

    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        if not isinstance(output, dict) or 'video_latents' not in output:
            raise ValueError('Native infer_action did not return video_latents for anchor parity')
        future = output['video_latents']
        if not torch.is_tensor(future) or not bool(torch.isfinite(future).all()):
            raise ValueError('Native anchor future is missing or non-finite')
        captured.append(future.detach().to(device='cpu'))
        return output

    model.infer_action = capture
    try:
        action, metrics = runtime.infer(value, seed, measure=False)
    finally:
        model.infer_action = original
    if not captured:
        raise ValueError('Native anchor inference did not call model.infer_action')
    return action, metrics, captured[-1], len(captured)


def _action_error(torch, candidate, reference):
    left = candidate.detach().float().cpu()
    right = reference.detach().float().cpu()
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f'Action shape differs: candidate={tuple(left.shape)}, reference={tuple(right.shape)}')
    if not bool(torch.isfinite(left).all()):
        raise ValueError('Non-finite candidate action')
    difference = (left - right).abs()
    return (float(difference.max()), float(difference.square().mean().sqrt()),
            int(left.numel()), bool(torch.equal(candidate.detach().cpu(), reference.detach().cpu())))


def _metric_delta(before, after):
    """Report numeric counter deltas only when both snapshots expose the same scalar."""
    result = {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        return result
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(old, (int, float)):
            result[key] = value - old
    return result


def progress(root, out, p, mode, phase, **fields):
    global _PHASE
    from reference_profile import save

    _PHASE = phase
    row = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'],
               mode=mode, phase=phase, **fields)
    save(root / 'progress.json', row)
    save(out / 'progress.json', row)
    print('PROFILE_PROGRESS ' + json.dumps(row), flush=True)


def profile(p, root, mode):
    import numpy as np
    import torch
    import calibration_source
    import export_runtime
    import abc_runtime
    import decoder_runtime
    import inventory
    from reference_profile import max_abs_difference, memory, save, statistics

    if mode not in MODES:
        raise ValueError(f'Unsupported mode: {mode}')
    settings = p['profiling']
    cases = validate_panel(settings)
    plan_path = Path(p['sources']['calibration_plan'])
    plan = json.loads(plan_path.read_text())
    if plan.get('test_case_ids', settings['case_ids']) != settings['case_ids']:
        raise ValueError('Frozen calibration cases changed')
    rows = {row['case_id']: row for row in plan['inputs']}
    if not set(cases + [2000]).issubset(rows):
        raise ValueError('Frozen plan is missing the anchor or a measured case')

    out = root / mode
    out.mkdir(parents=True, exist_ok=True)
    progress(root, out, p, mode, 'loading')
    props, gpu_name = _check_gpu_family(torch, p)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    bundle = export_runtime.load_bundle(p['export_reuse']['bundle_path'])
    runtime = abc_runtime.make_runtime(out, p, bundle, direct=True)
    banks = abc_runtime.install(runtime, bundle, mode)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    load_memory = memory(torch)
    torch.cuda.reset_peak_memory_stats()
    setup_started = time.perf_counter()
    progress(root, out, p, mode, 'decoder_preflight')
    decoder = decoder_runtime.prepare(runtime, bundle, p, out,
        reuse_path=(root / 'c_decoder' / 'tuning.json') if mode == 'c_buffer' else None)
    banks.decoder = decoder
    del bundle
    gc.collect()
    torch.cuda.empty_cache()
    probe = inventory.install_runtime_probe(runtime)
    runtime.model.eval()
    if any(parameter.requires_grad for parameter in runtime.model.parameters()):
        raise ValueError('Inference runtime contains trainable parameters')

    def datum(case_id):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        observation, metadata = calibration_source.load_observation(
            np, plan_path.parent, rows[case_id])
        runtime.description = metadata['description']
        value = runtime.datum(observation)
        torch.cuda.synchronize()
        return value, (time.perf_counter() - begin) * 1000

    progress(root, out, p, mode, 'anchor')
    value, _ = datum(2000)
    anchor_seed = rows[2000]['sampler_seeds'][0]
    with torch.no_grad():
        action, _, future, anchor_infer_calls = _capture_anchor(runtime, value, anchor_seed, torch)
    if mode == 'runtime_compile':
        faulthandler.cancel_dump_traceback_later()
        signal.alarm(0)
    anchor = torch.load(p['source']['anchor_tensor'], map_location='cpu', weights_only=True)
    expected_action_shape = tuple(settings['action_shape'])
    if tuple(action.shape) != expected_action_shape or tuple(anchor['action'].shape) != expected_action_shape:
        raise ValueError(f'Anchor action shape differs from frozen shape {expected_action_shape}')
    if tuple(future.shape) != tuple(anchor['future'].shape):
        raise ValueError('Native anchor future shape differs from the original anchor')
    anchor_action_shape = tuple(action.shape)
    anchor_reference_action_shape = tuple(anchor['action'].shape)
    anchor_future_shape = tuple(future.shape)
    anchor_reference_future_shape = tuple(anchor['future'].shape)
    anchor_action_error, _, _, anchor_action_exact = _action_error(torch, action, anchor['action'])
    anchor_future_error, _, _, anchor_future_exact = _action_error(torch, future, anchor['future'])
    anchor_future_nrmse = float(
        (future.float() - anchor['future'].float()).square().mean().sqrt()
        / anchor['future'].float().square().mean().sqrt().clamp_min(1e-12)
    )
    if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all()):
        raise ValueError('Non-finite source-anchor output')
    if max(anchor_action_error, anchor_future_error) != 0:
        raise ValueError('Candidate must exactly preserve the original checkpoint anchor')
    optimization = runtime.optimization
    diagnostics_before_anchor_end = getattr(optimization, 'diagnostics', None)
    if hasattr(optimization, 'diagnostics'):
        optimization.diagnostics = False
    del action, future, anchor, value

    progress(root, out, p, mode, 'warmup', expected=settings['warmup_queries'])
    value, _ = datum(3000)
    for _ in range(settings['warmup_queries']):
        runtime.infer(value, rows[3000]['sampler_seeds'][0], measure=False)
    torch.cuda.synchronize()
    coverage_after_warmup = _coverage_summary(banks)
    validate_module_coverage(coverage_after_warmup)
    optimization_after_warmup = _optimization_report(runtime)
    validate_optimization_coverage(optimization_after_warmup)
    actual_counters_after_warmup = _bank_actual_counters(banks)
    decoder_coverage_after_warmup = _decoder_coverage(banks)
    if not decoder_coverage_after_warmup['all_614_modules_hit']:
        raise ValueError(f"Workspace decoder did not hit all 614 modules: {decoder_coverage_after_warmup}")
    inventory_report = inventory.finalize_runtime_probe(probe)
    inventory_report['coverage_scope'] = 'anchor and 3 warmups including graph capture; hooks removed before primary timing'
    inventory_report = inventory.write_runtime_probe(out, inventory_report)
    del value
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = memory(torch)
    setup_validation_warmup_seconds = time.perf_counter() - setup_started

    reset_metrics = getattr(optimization, 'reset_metrics', None)
    if not callable(reset_metrics):
        raise RuntimeError('Runtime must expose optimization.reset_metrics()')
    reset_metrics()
    coverage_before_primary = _coverage_summary(banks)
    actual_counters_before_primary = _bank_actual_counters(banks)
    optimization_before_primary = _optimization_report(runtime)
    torch.cuda.reset_peak_memory_stats()
    progress(root, out, p, mode, 'measuring', expected=80)

    baseline_path = root / 'layer_decode_fp32' / 'actions.pt'
    baseline_actions = torch.load(baseline_path, map_location='cpu', weights_only=True)
    expected_keys = {f'{case_id}_{seed_index}' for case_id in cases
                     for seed_index in settings['seed_indices']}
    if set(baseline_actions) != expected_keys:
        raise ValueError(f'Baseline action panel differs: {len(baseline_actions)}/{len(expected_keys)} keys')
    baseline_action_shapes = {tuple(action.shape) for action in baseline_actions.values()}
    if baseline_action_shapes != {expected_action_shape}:
        raise ValueError(f'Baseline action shape differs from frozen shape: {baseline_action_shapes}')

    samples, actions, action_errors, action_exact = [], {}, [], []
    action_squared_error_sum, action_element_count = 0.0, 0
    full80_action_shapes = set()
    for case_id in cases:
        value, preparation_ms = datum(case_id)
        for seed_index in settings['seed_indices']:
            key = f'{case_id}_{seed_index}'
            seed = rows[case_id]['sampler_seeds'][seed_index]
            action, metrics = runtime.infer(value, seed, measure=True)
            if not bool(torch.isfinite(action).all()):
                raise ValueError(f'Non-finite action at case {case_id}, seed {seed_index}')
            if tuple(action.shape) != expected_action_shape:
                raise ValueError(f'Action shape changed at {key}: {tuple(action.shape)}')
            error_max, error_rmse, count, exact = _action_error(torch, action, baseline_actions[key])
            action_errors.append(error_max)
            action_exact.append(exact)
            full80_action_shapes.add(tuple(action.shape))
            action_squared_error_sum += error_rmse * error_rmse * count
            action_element_count += count
            actions[key] = action.detach().cpu()
            samples.append(dict(case_id=case_id, seed_index=seed_index,
                                preprocessing_wall_ms=preparation_ms,
                                action_max_abs_vs_layer_decode_fp32=error_max,
                                action_rmse_vs_layer_decode_fp32=error_rmse,
                                **metrics))
            del action
        del value
        progress(root, out, p, mode, 'measuring', queries=len(samples), expected=80)
    if len(samples) != 80:
        raise ValueError(f'Expected 80 measured queries, got {len(samples)}')
    save(out / 'samples.json', samples)
    torch.save(actions, out / 'actions.pt')

    coverage_after_primary = _coverage_summary(banks)
    actual_counters_after_primary = _bank_actual_counters(banks)
    optimization_after_primary = _optimization_report(runtime)
    module_calls = optimization.graph.module_call_counts()
    if sum(module_calls.values()) != 5258 * 80:
        raise ValueError('Graph and outside-graph module frequencies do not match the frozen C path')
    decoder.update_frequencies(module_calls, query_count=len(samples))
    decoder_report = decoder.report()
    save(out / 'tuning.json', decoder_report)
    coverage_delta = _metric_delta(coverage_before_primary, coverage_after_primary)
    logical_before = optimization_before_primary.get('logical_coverage', {})
    logical_after = optimization_after_primary.get('logical_coverage', {})
    logical_delta = _metric_delta(logical_before, logical_after)
    logical_target_calls = logical_delta.get('target_linear_calls')
    python_target_calls = actual_counters_after_primary['python_target_linear_calls']
    action_rmse = (action_squared_error_sum / action_element_count) ** 0.5
    full80_parity = dict(
        comparison='full80_actions_vs_cached_layer_decode_fp32',
        max_abs=max(action_errors), rmse=action_rmse,
        exact=all(action_exact), exact_queries=sum(action_exact), queries=len(samples),
        action_elements=action_element_count, atol=settings['parity_atol'],
        passed=max(action_errors) <= settings['parity_atol'],
    )
    parity = dict(
        anchor_action_max_abs=anchor_action_error,
        anchor_future_max_abs=anchor_future_error,
        anchor_future_nrmse=anchor_future_nrmse,
        anchor_infer_action_calls=anchor_infer_calls,
        full80_action_max_abs_vs_layer_decode_fp32=max(action_errors),
        full80_action_rmse_vs_layer_decode_fp32=action_rmse,
        full80_queries=len(samples), full80_action_elements=action_element_count,
        full80_per_query_max_abs=action_errors,
    )
    workspace_validation = optimization_after_warmup.get('operand_validation', {})
    validation = dict(
        protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode,
        complete=True,
        anchor_action_max_abs=anchor_action_error,
        anchor_future_max_abs=anchor_future_error,
        anchor_future_nrmse=anchor_future_nrmse,
        anchor_action_exact=anchor_action_exact,
        anchor_future_exact=anchor_future_exact,
        strict_anchor_gate=max(anchor_action_error, anchor_future_error) <= settings['parity_atol'],
        full80_action_max_abs_vs_layer_decode_fp32=max(action_errors),
        full80_action_rmse_vs_layer_decode_fp32=action_rmse,
        full80_action_parity_passed=full80_parity['passed'],
        full80_action_exact_queries=full80_parity['exact_queries'],
        finite_outputs_verified=True,
        operand_modules_checked=workspace_validation.get('checked_modules'),
        source_anchor_parity=dict(
            action_max_abs=anchor_action_error, future_max_abs=anchor_future_error,
            future_nrmse=anchor_future_nrmse,
            action_exact=anchor_action_exact, future_exact=anchor_future_exact,
            action_shape=list(anchor_action_shape), future_shape=list(anchor_future_shape),
            passed=max(anchor_action_error, anchor_future_error) <= settings['parity_atol'],
            atol=settings['parity_atol'], infer_action_calls=anchor_infer_calls,
        ),
        full80_action_parity_vs_layer_decode_fp32=full80_parity,
        output_shapes=dict(
            expected_action=list(expected_action_shape),
            anchor_action=list(anchor_action_shape),
            anchor_reference_action=list(anchor_reference_action_shape),
            anchor_future=list(anchor_future_shape),
            anchor_reference_future=list(anchor_reference_future_shape),
            baseline_action_shapes=[list(shape) for shape in sorted(baseline_action_shapes)],
            full80_action_shapes=[list(shape) for shape in sorted(full80_action_shapes)],
        ),
        cohort=dict(case_ids=cases, seed_indices=list(settings['seed_indices']),
                    warmup_queries=settings['warmup_queries'], queries=len(samples),
                    steps_video=settings['steps_video'], steps_action=settings['steps_action'],
                    batch_size=settings['batch_size'], action_horizon=settings['action_horizon'],
                    video_frames=settings['video_frames']),
        workspace_decoder_coverage=dict(
            **decoder_coverage_after_warmup,
            operand_validation=workspace_validation,
        ),
        decoder_optimization=decoder_report,
    )
    save(out / 'validation.json', validation)
    save(out / 'parity.json', parity)

    profiler_modes = settings.get('diagnostic_profile_modes', [])
    operators = None
    profiler_receipt = dict(attempted=False, completed=False)
    if mode in profiler_modes:
        progress(root, out, p, mode, 'diagnostic_profiler')
        profiler = None
        try:
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
            save(out / 'operators.json', dict(
                intrusive=True, activities=['CPU', 'CUDA'], rows=operators[:160]))
            profiler_receipt = dict(attempted=True, completed=True,
                                    trace_path=str(out / 'trace.json'), operator_rows=min(160, len(operators)))
        except Exception as exc:
            import traceback
            profiler_receipt = dict(attempted=True, completed=False, error=repr(exc),
                                    traceback_tail=traceback.format_exc().splitlines()[-8:])
            try:
                save(out / 'profiler_failure.json', profiler_receipt)
            except Exception:
                pass
        finally:
            try:
                if profiler is not None:
                    del profiler
                if 'value' in locals():
                    del value
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as exc:
                import traceback
                profiler_receipt.update(attempted=True, completed=False,
                                        cleanup_error=repr(exc),
                                        cleanup_traceback=traceback.format_exc())
                try:
                    save(out / 'profiler_failure.json', profiler_receipt)
                except Exception:
                    pass

    metrics = {key: statistics(samples, key) for key in (
        'full_query_wall_ms', 'denoising_latency_ms',
        'peak_allocated_gpu_GB', 'peak_reserved_gpu_GB', 'baseline_allocated_gpu_GB')}
    metadata = _jsonable(getattr(runtime, 'install_metadata', {}))
    anchor_gate = max(anchor_action_error, anchor_future_error) <= settings['parity_atol']
    summary = dict(
        protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode,
        complete=True, queries=len(samples), source_job=p['source']['source_job'],
        source_arm='reader_kl', new_training_updates=0, closed_loop_episodes=0,
        cohort=dict(case_ids=cases, seed_indices=list(settings['seed_indices']),
                    warmup_queries=settings['warmup_queries'], queries=len(samples),
                    steps_video=settings['steps_video'], steps_action=settings['steps_action'],
                    batch_size=settings['batch_size'], action_horizon=settings['action_horizon'],
                    video_frames=settings['video_frames']),
        metrics=metrics, load_seconds=load_seconds, load_transient_memory=load_memory,
        resident_after_warmup=resident,
        setup_validation_warmup_seconds=setup_validation_warmup_seconds,
        setup_peak_allocated_bytes=resident['peak_allocated_bytes'],
        setup_peak_reserved_bytes=resident['peak_reserved_bytes'],
        performance_accuracy=dict(
            full80_action_max_abs_vs_layer_decode_fp32=max(action_errors),
            full80_action_rmse_vs_layer_decode_fp32=action_rmse,
            anchor_action_max_abs=anchor_action_error,
            anchor_future_max_abs=anchor_future_error,
            anchor_future_nrmse=anchor_future_nrmse,
            anchor_passed=anchor_gate, anchor_infer_action_calls=anchor_infer_calls,
        ),
        optimization=dict(
            diagnostics_before_anchor_end=diagnostics_before_anchor_end,
            diagnostics_after_anchor=getattr(optimization, 'diagnostics', None),
            after_warmup=optimization_after_warmup,
            before_primary=optimization_before_primary,
            after_primary=optimization_after_primary,
            logical_coverage_delta=logical_delta,
            logical_target_linear_calls_per_query=(
                logical_target_calls / len(samples) if logical_target_calls is not None else None),
            python_target_linear_calls_per_query=(
                python_target_calls / len(samples) if python_target_calls is not None else None),
        ),
        coverage=dict(
            interpretation='banks coverage counters are actual Python-observed counters; '
                           'optimization.logical_coverage includes graph capture/replay coverage',
            after_warmup=coverage_after_warmup,
            before_primary=coverage_before_primary,
            after_primary=coverage_after_primary,
            actual_python_counters=dict(
                after_warmup=actual_counters_after_warmup,
                before_primary=actual_counters_before_primary,
                after_primary=actual_counters_after_primary,
            ),
            actual_counter_delta=coverage_delta,
            logical_coverage_before=logical_before,
            logical_coverage_after=logical_after,
        ),
        installation=metadata, gpu=gpu_name, gpu_uuid=str(props.uuid),
        decoder_optimization=decoder_report,
        parameter_execution=inventory_report,
        gpu_total_bytes=props.total_memory,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        allow_cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        allow_bf16_reduced_precision_reduction=(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        scope=settings['primary_scope'],
        instrumentation_used_for_primary_latency=False,
        compiled=bool(metadata.get('compiled', mode == 'runtime_compile')),
        cuda_graph=bool(metadata.get('cuda_graph', False)),
        native_vq_kernel=False, native_integer_gemm=False,
        diagnostic_profiler_used=operators is not None,
        diagnostic_profiler=profiler_receipt,
    )
    save(out / 'summary.json', summary)
    progress(root, out, p, mode, 'complete', queries=len(samples), expected=80)


def main():
    # Guard before importing Torch, model modules, or tensor-bearing bundles.
    from reference_profile import guard

    guard()
    global _FAILURE_ROOT, _FAILURE_DIR, _MODE
    _FAILURE_ROOT = Path(os.environ.get('NEW_ROOT', os.environ.get('ROOT', '.'))) / 'results'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=MODES)
    args = parser.parse_args()
    _MODE = args.mode
    _FAILURE_DIR = _FAILURE_ROOT / _MODE
    p = json.loads((Path(os.environ['ARTIFACTS']) / 'protocol.json').read_text())
    _FAILURE_ROOT = Path(p['remote_root']) / 'results'
    _FAILURE_DIR = _FAILURE_ROOT / _MODE
    if args.mode == 'runtime_compile':
        # Avoid shared-cache locks and async compiler worker startup on the cluster.
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(os.environ['ARTIFACTS']) / 'inductor_cache')
        os.environ['TRITON_CACHE_DIR'] = str(Path(os.environ['ARTIFACTS']) / 'triton_cache')
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(180, repeat=True)

        def compile_timeout(signum, frame):
            faulthandler.dump_traceback(all_threads=True)
            raise TimeoutError('Compile setup exceeded 1200 seconds; completed ABC receipts are preserved')

        signal.signal(signal.SIGALRM, compile_timeout)
        signal.alarm(1200)
    _freeze_numerics(__import__('torch'))
    _check_gpu_family(__import__('torch'), p)
    profile(p, _FAILURE_ROOT, args.mode)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if _FAILURE_DIR is not None or _FAILURE_ROOT is not None:
            from reference_profile import save

            failure_path = (_FAILURE_DIR / 'failure.json' if _FAILURE_DIR is not None
                            else _FAILURE_ROOT / 'failure.json')
            save(failure_path, dict(
                pbs_jobid=os.environ.get('PBS_JOBID'), mode=_MODE,
                phase=_PHASE, error=repr(exc), traceback=traceback.format_exc()))
        raise

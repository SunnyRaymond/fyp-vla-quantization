"""Resumable two-worker LIBERO-Plus evaluation for the frozen anchor-KL deployment."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import socket
import time

PROTOCOL = 'fastwam-vq-teacher-anchor-kl-v1'
COHORT_PROTOCOL = 'fastwam-rotation-baselines-v1'
ARM = 'anchor_kl'
IMPLEMENTATION = 'batched_h128_v2'
TIMING_FIELDS = ('denoising_latency_ms', 'full_query_wall_ms', 'peak_allocated_gpu_GB',
                 'baseline_allocated_gpu_GB', 'peak_reserved_gpu_GB')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def worker_indices(worker_id):
    if type(worker_id) is not int or worker_id not in (0, 1):
        raise ValueError('worker_id must be 0 or 1')
    return list(range(worker_id, 280, 2))


def environment_seed(index):
    return 800_000_000 + index * 1000


def sampler_seed(index, replan=0):
    return 900_000_000 + index * 1000 + replan


def baseline_checkpoint(protocol):
    return protocol['baseline_root'].rstrip('/') + '/results/training/rho1.5/deployment.pt'


def validate_protocol(protocol, source_protocol=None):
    if not isinstance(protocol, dict) or protocol.get('protocol') != PROTOCOL:
        raise ValueError('Teacher-anchor protocol identity differs')
    closed = protocol.get('closedloop', {})
    root = closed.get('source_root', '').rstrip('/')
    if (not root or closed.get('manifest') != root + '/results/manifest.json'
            or closed.get('reference_arm') != 'per_channel_clip'
            or closed.get('expected_reference_success') != 203
            or closed.get('variants') != 280 or closed.get('workers') != 2
            or closed.get('indices_rule') != 'worker_id + 2*k'
            or closed.get('execute_steps') != 10 or closed.get('settling_steps') != 30
            or closed.get('maximum_steps') != {'libero_10': 700, 'other_suites': 400}
            or closed.get('environment_seed') != '800000000+index*1000'
            or closed.get('sampler_seed') != '900000000+index*1000+replan'
            or closed.get('checkpoint') != 'results/anchor_kl/deployment.pt'
            or closed.get('reference_endpoint') != 'results/anchor_kl/reference_endpoint.pt'):
        raise ValueError('Closed-loop source, cohort, seed or execution contract changed')
    if (protocol.get('training', {}).get('additional_updates') != 80
            or protocol.get('expected', {}).get('closedloop_new_episodes') != 280
            or protocol.get('closedloop_allocation', {}).get('workers') != 2):
        raise ValueError('Final80 or two-worker closed-loop contract changed')
    if source_protocol is not None:
        if (source_protocol.get('remote_root') != root
                or source_protocol.get('protocol') != 'fastwam-vq-softbin-relational-plus-v1'
                or source_protocol.get('cohort', {}).get('manifest_protocol') != COHORT_PROTOCOL
                or source_protocol.get('references', {}).get('per_channel_protocol')
                != 'fastwam-vq-c80-per-channel-plus-v1'
                or source_protocol.get('sources', {}).get('baseline_deployment')
                != baseline_checkpoint(protocol)):
            raise ValueError('Closed-loop source_root is not the frozen local paired-evaluation protocol')
    return protocol


def load_manifest(path):
    doc = read_json(path)
    rows = doc.get('variants')
    if doc.get('protocol') != COHORT_PROTOCOL or not isinstance(rows, list) or len(rows) != 280:
        raise ValueError('Expected the frozen original 280-variant manifest')
    identities, cells = set(), collections.Counter()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get('index')) is not int
                or row['index'] != index or row.get('state_id', 0) != 0
                or 'task_id' not in row):
            raise ValueError(f'Invalid frozen manifest row {index}')
        identity = str(row.get('variant_id', ''))
        if not identity or identity in identities:
            raise ValueError(f'Duplicate/missing variant identity at {index}')
        identities.add(identity)
        cells[row.get('suite'), row.get('dimension')] += 1
    if (len(cells) != 28 or set(cells.values()) != {10}
            or len({suite for suite, _ in cells}) != 4
            or len({dimension for _, dimension in cells}) != 7):
        raise ValueError('Expected the original 4-suite x 7-dimension x 10 cohort')
    return doc, rows


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def validate_training_artifacts(protocol, deployment, receipt, source_protocol=None):
    expected_order = protocol['calibration']['update_order']
    history = receipt.get('history')
    if (receipt.get('complete') is not True or receipt.get('updates') != 80
            or receipt.get('update_order') != expected_order or not isinstance(history, list)
            or len(history) != 80
            or [entry.get('update') for entry in history] != list(range(1, 81))
            or [entry.get('case_id') for entry in history] != expected_order):
        raise ValueError('Anchor-KL training receipt is not the complete frozen update80 history')
    expected_source = protocol.get('source_expected', {})
    if source_protocol is not None:
        expected_source = source_protocol.get('source_expected', expected_source)
    expected_parity = expected_source.get('cached_query_parity',
                                          {'action_max_abs': 0.0, 'future_max_abs': 0.0})
    clipping, scaling = receipt.get('clipping'), receipt.get('scaling')
    if (not isinstance(clipping, dict) or clipping.get('count') != 614
            or not isinstance(scaling, dict) or scaling.get('modules') != 614
            or scaling.get('group_size') != 1
            or type(scaling.get('channels')) is not int or scaling['channels'] <= 0
            or scaling.get('raw_scale_metadata_bytes') != scaling['channels'] * 4
            or (expected_source.get('scale_channels') is not None
                and scaling.get('channels') != expected_source['scale_channels'])
            or (expected_source.get('scale_metadata_bytes') is not None
                and scaling.get('raw_scale_metadata_bytes') != expected_source['scale_metadata_bytes'])
            or scaling.get('scale_metadata_dtype') != 'float32_log2'
            or scaling.get('cached_compensated_weight_evaluation') is not True
            or receipt.get('cached_query_parity') != expected_parity):
        raise ValueError('Training receipt clip/scale coverage or cached-query parity differs')
    source = baseline_checkpoint(protocol)
    if (deployment.get('protocol') != protocol or deployment.get('unit') != ARM
            or deployment.get('updates') != 80 or deployment.get('baseline_checkpoint') != source
            or deployment.get('scale_metadata_dtype') != 'float32_log2'):
        raise ValueError('Anchor-KL deployment identity or baseline differs')
    if (not isinstance(deployment.get('ratios'), dict)
            or not isinstance(deployment.get('log2_scales'), dict)
            or len(deployment['ratios']) != 614
            or set(deployment['ratios']) != set(deployment['log2_scales'])):
        raise ValueError('Anchor-KL deployment must contain all 614 clipping and scale banks')
    if receipt.get('pbs_jobid') is not None and not isinstance(receipt['pbs_jobid'], str):
        raise ValueError('Invalid training PBS identity')
    return receipt


def validate_endpoint(endpoint, rows, torch):
    row = rows.get(2000)
    if row is None:
        raise ValueError('Original C80 calibration row 2000 is missing')
    expected = {'protocol': PROTOCOL, 'unit': ARM, 'updates': 80,
                'case_id': 2000, 'seed_index': 0, 'plan_row': row,
                'sampler_seed': row['sampler_seeds'][0],
                'counts': {'video_steps': 10, 'action_steps': 10}}
    if any(endpoint.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved final80 reference endpoint is differently bound')
    if not isinstance(endpoint.get('pbsjobid'), str) or not endpoint['pbsjobid']:
        raise ValueError('Final80 reference endpoint is missing its PBS identity')
    for key, shape in (('action', (32, 7)), ('future', (1, 48, 3, 14, 28))):
        value = endpoint.get(key)
        if (not torch.is_tensor(value) or value.dtype != torch.float32 or value.device.type != 'cpu'
                or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid CPU float32 source endpoint {key}')
    return row


def _validate_saved_episode(record, row, worker_id, protocol):
    index = record.get('index')
    if type(index) is not int or index not in worker_indices(worker_id):
        raise ValueError('Record is outside its frozen worker shard')
    expected = {
        'protocol': PROTOCOL, 'unit': f'closedloop_{worker_id}', 'arm': ARM, 'worker_id': worker_id,
        'complete': True, 'suite': row['suite'], 'dimension': row['dimension'],
        'task_id': int(row['task_id']), 'state_id': 0,
        'env_seed': environment_seed(index), 'sampler_seed': sampler_seed(index),
    }
    if (any(record.get(key) != value for key, value in expected.items())
            or str(record.get('variant_id')) != str(row['variant_id'])
            or type(record.get('success')) is not bool
            or record.get('transform_implementation') != IMPLEMENTATION):
        raise ValueError(f'Frozen candidate episode identity differs at {index}')
    if not isinstance(record.get('pbs_job_id'), str) or not record['pbs_job_id']:
        raise ValueError('Episode is missing its PBS job identity')
    steps, replans, metrics = record.get('steps'), record.get('replans'), record.get('metrics')
    if (type(steps) is not int or steps < 1 or type(replans) is not int or replans < 1
            or not isinstance(metrics, list) or len(metrics) != replans):
        raise ValueError('Episode step or replan evidence is incomplete')
    for replan, metric in enumerate(metrics):
        if (metric.get('replan') != replan or metric.get('sampler_seed') != sampler_seed(index, replan)
                or metric.get('transform_implementation') != IMPLEMENTATION
                or metric.get('pbs_job_id') != record['pbs_job_id']
                or type(metric.get('first_query_in_run')) is not bool):
            raise ValueError('Invalid per-query seed or run evidence')
        if any(not _finite(metric.get(key)) or metric[key] < 0 for key in TIMING_FIELDS):
            raise ValueError('Invalid query timing evidence')
    evidence = record.get('vq_evidence')
    if (not isinstance(evidence, dict) or evidence.get('observed_module_count') != 614
            or evidence.get('native_int4_gemm_calls') != 0
            or evidence.get('kv_packed_prefills') != 0 or evidence.get('kv_layer_reads') != 0
            or type(evidence.get('quantized_linear_calls')) is not int
            or evidence['quantized_linear_calls'] <= 0):
        raise ValueError('Decoded-BF16 VQ/A4 execution evidence differs')
    gripper = record.get('gripper_command_values')
    if (not isinstance(gripper, list) or len(gripper) != steps
            or any(not _finite(value) or value not in (-1.0, 0.0, 1.0) for value in gripper)):
        raise ValueError('Executed gripper commands must retain exact signed command values')
    return index


def recover_episodes(out, rows, worker_id, protocol):
    directory = Path(out) / 'episodes'
    found = {}
    if not directory.exists():
        return found
    for path in sorted(directory.glob('*.json')):
        record = read_json(path)
        index = record.get('index')
        if path.name != f'{index:03d}.json' or index in found:
            raise ValueError(f'Unexpected or duplicate atomic episode file: {path.name}')
        _validate_saved_episode(record, rows[index], worker_id, protocol)
        found[index] = record
    expected = worker_indices(worker_id)
    if list(found) != expected[:len(found)]:
        raise ValueError('Saved episodes are not a complete frozen worker prefix')
    return found


def _metadata(protocol, deployment, receipt, endpoint):
    source = protocol['source_clipping_protocol']
    return {
        'protocol': PROTOCOL, 'arm': ARM, 'source_training_protocol': PROTOCOL,
        'source_training_unit': ARM, 'source_training_job': receipt.get('pbs_jobid'),
        'source_endpoint_job': endpoint.get('pbs_jobid'),
        'source_deployment': protocol['remote_root'] + '/results/anchor_kl/deployment.pt',
        'baseline_deployment': baseline_checkpoint(protocol),
        'module_count': 614, 'execution': 'decoded_bf16_vq', 'activation_bits': 4,
        'nominal_weight_bits': 4, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
        'native_int4_gemm_calls': 0, 'transform_implementation': IMPLEMENTATION,
        'smooth_alpha': .5, 'hadamard_block': 128, 'scale_block': 32,
        'channel_scale_group_size': 1, 'scale_metadata_dtype': 'float32_log2',
        'fixed_indices': True, 'fixed_scale_exponents': True,
        'train_weights': False, 'train_clips': False, 'train_channel_scales': False,
        'updates': 80, 'training_updates': 80, 'new_training_updates': 0,
        'calibration_ids': protocol['calibration']['case_ids'],
        'cached_compensated_weight_evaluation': True,
        'ratios': receipt.get('clipping'), 'scaling': receipt.get('scaling'),
        'source_cached_query_parity': receipt.get('cached_query_parity'),
        'deployment_unit': deployment['unit'],
    }


def _compute_setup(protocol, source_protocol, results_root, worker_out, worker_id):
    import numpy as np
    import types
    import torch
    import sampler
    import runner
    import clipping_reference as ref
    import calibration_source
    import batched_transform
    from channel_banks import ChannelBanks

    sampler._require_compute_allocation()
    runner.guard()
    if batched_transform.IMPLEMENTATION != IMPLEMENTATION:
        raise RuntimeError('Expected the frozen batched H128 implementation')
    p = protocol
    data_root, source_rows, _, base_deployment, _ = ref.load_inputs(p['source_clipping_protocol'], torch)
    out = Path(results_root) / 'anchor_kl'
    deployment = torch.load(out / 'deployment.pt', map_location='cpu', weights_only=True)
    receipt = read_json(out / 'training_receipt.json')
    validate_training_artifacts(p, deployment, receipt, source_protocol)
    endpoint = torch.load(out / 'reference_endpoint.pt', map_location='cpu', weights_only=True)
    source_row = validate_endpoint(endpoint, source_rows, torch)

    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = runner.Runtime(worker_out)
    if any(not callable(getattr(runtime, name, None))
           for name in ('episode', 'task', 'command', 'infer', 'datum')):
        raise RuntimeError('Copied runner.Runtime must include the frozen task/episode/infer route')
    import previous_vq_adapter
    episode_runtime = previous_vq_adapter._runtime_type()
    if not issubclass(episode_runtime, runner.Runtime) or not callable(
            getattr(episode_runtime, 'episode', None)):
        raise RuntimeError('The frozen decoded-BF16 VQ episode adapter does not match runner.Runtime')
    # The pilot runner's episode gate requires packed integer GEMM. Reuse the
    # prior decoded-VQ loop while retaining this deployment's own runner methods.
    runtime.episode = types.MethodType(episode_runtime.episode, runtime)
    source_command = runtime.command

    def command_from_original_cpu_precision(action):
        if not torch.is_tensor(action):
            raise TypeError('Runtime.command requires the original action tensor')
        return source_command(action.detach().cpu())

    runtime.command = command_from_original_cpu_precision
    runtime.model.eval()
    runtime.arm('bf16')
    runtime.protocol, runtime.worker_id = p, worker_id
    runtime.current_arm = ARM
    runtime.transform_implementation = batched_transform.IMPLEMENTATION
    runtime.run_queries = runtime.linear_calls = 0
    runtime.arm_calls[ARM] = 0
    banks = ChannelBanks(runtime, p['calibration']['bank_directory'],
                         base_deployment['module_masters'], p['clipping'], p['scaling'])
    runtime.banks = banks
    if (set(deployment['ratios']) != set(runtime.q.modules)
            or set(deployment['log2_scales']) != set(runtime.q.modules)
            or len(runtime.q.modules) != 614):
        raise RuntimeError('Anchor-KL deployment does not cover all 614 canonical Linears')
    banks.restore_clips(deployment['ratios'])
    banks.restore_scales(deployment['log2_scales'])
    for value in list(runtime.model.parameters()) + [
            item for group in (banks.books, banks.logs, banks.ratios, banks.shadows)
            for item in group.values()]:
        value.requires_grad_(False)
    for module in runtime.q.modules.values():
        forward = module.forward

        def counted(value, _forward=forward):
            runtime.linear_calls += 1
            return _forward(value)

        module.forward = counted

    # Match the source's uncached, cached and formal-infer startup replay before episodes.
    observation, meta = calibration_source.load_observation(np, data_root, source_row)
    runtime.description = meta['description']
    datum, seed = runtime.datum(observation), endpoint['sampler_seed']

    def query():
        banks.seen.clear()
        with torch.no_grad():
            action, counts, future = sampler.differentiable_action(
                runtime, datum, seed, checkpointed=False, return_future=True)
        if (len(banks.seen) != 614 or counts != endpoint['counts']
                or tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
                or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())
                or runtime.q._native_int4_gemm_calls != 0 or runtime.q._kv_active):
            raise RuntimeError('Final80 source replay coverage or decoded-BF16 execution gate failed')
        return action.detach().float().cpu(), future.detach().float().cpu()

    try:
        uncached_action, uncached_future = query()
        errors = (float((uncached_action - endpoint['action']).abs().max()),
                  float((uncached_future - endpoint['future']).abs().max()))
        banks.cache_compensated_weights()
        cached_action, cached_future = query()
        cache_errors = (float((cached_action - uncached_action).abs().max()),
                        float((cached_future - uncached_future).abs().max()))
        actual, _ = runtime.infer(datum, seed, measure=False)
        infer_error = float((actual.detach().float().cpu() - endpoint['action']).abs().max())
        if (not all(math.isfinite(value) for value in errors + cache_errors + (infer_error,))
                or errors != (0.0, 0.0) or cache_errors != (0.0, 0.0) or infer_error != 0.0
                or len(banks.seen) != 614 or not banks.cached
                or any(value.requires_grad for value in runtime.model.parameters())):
            raise RuntimeError(f'Anchor-KL source/cache/formal-infer parity failed: '
                               f'source={errors}, cache={cache_errors}, infer={infer_error}')
        metadata = _metadata(p, deployment, receipt, endpoint)
        runtime.install_metadata = metadata
        setup = {
            'protocol': PROTOCOL, 'unit': f'closedloop_{worker_id}',
            'worker_id': worker_id, 'arm': ARM, 'passed': True,
            'pbs_job_id': os.environ['PBS_JOBID'], 'observed_modules': len(banks.seen),
            'case_id': 2000, 'seed_index': 0, 'sampler_seed': seed,
            'source_action_max_abs': errors[0], 'source_future_max_abs': errors[1],
            'uncached_cached_action_max_abs': cache_errors[0],
            'uncached_cached_future_max_abs': cache_errors[1],
            'actual_infer_action_max_abs': infer_error, 'startup_queries': 3,
            'training_updates': 80, 'new_training_updates': 0,
            'cached_compensated_weights': True, 'install_metadata': metadata,
        }
        return runtime, banks, setup
    except Exception:
        runtime.close()
        raise


def _validate_setup(setup, worker_id, protocol):
    if (not isinstance(setup, dict) or setup.get('protocol') != PROTOCOL
            or setup.get('unit') != f'closedloop_{worker_id}'
            or setup.get('worker_id') != worker_id or setup.get('arm') != ARM
            or setup.get('passed') is not True or setup.get('observed_modules') != 614
            or setup.get('case_id') != 2000 or setup.get('seed_index') != 0
            or setup.get('sampler_seed') != 910_004_000
            or setup.get('training_updates') != 80 or setup.get('new_training_updates') != 0
            or setup.get('cached_compensated_weights') is not True
            or not isinstance(setup.get('pbs_job_id'), str) or not setup['pbs_job_id']):
        raise ValueError('Saved worker setup receipt differs from the frozen source gate')
    for key, maximum in (('source_action_max_abs', 0.0), ('source_future_max_abs', 0.0),
                         ('actual_infer_action_max_abs', 0.0),
                         ('uncached_cached_action_max_abs', 0.0),
                         ('uncached_cached_future_max_abs', 0.0)):
        value = setup.get(key)
        if not _finite(value) or value < 0 or value > maximum:
            raise ValueError(f'Saved startup parity failed: {key}')
    if setup.get('install_metadata', {}).get('protocol') != PROTOCOL:
        raise ValueError('Saved setup is not bound to the anchor-KL deployment')
    if (not isinstance(setup['install_metadata'].get('source_endpoint_job'), str)
            or not setup['install_metadata']['source_endpoint_job']):
        raise ValueError('Saved setup is not bound to its final80 reference endpoint job')


def _record_episode(runtime, banks, row, index, worker_id, protocol, query_total, run_first):
    job_id = os.environ['PBS_JOBID']
    original_infer = runtime.infer
    original_task = runtime.task
    gripper = []

    def tracked_infer(datum, seed, measure=True):
        nonlocal query_total, run_first
        action, metric = original_infer(datum, seed, measure=measure)
        if measure:
            metric.update({
                'replan': seed - sampler_seed(index),
                'query_index_in_arm': query_total,
                'first_query_in_arm': query_total == 0,
                'first_query_in_run': run_first,
                'sampler_seed': seed, 'pbs_job_id': job_id,
                'transform_implementation': IMPLEMENTATION,
            })
            query_total += 1
            run_first = False
        return action, metric

    def traced_task(task_row, task_index):
        observation = original_task(task_row, task_index)
        raw_step = runtime.env.step

        def traced_step(action, *args, **kwargs):
            gripper.append(float(action[-1]))
            return raw_step(action, *args, **kwargs)

        runtime.env.step = traced_step
        return observation

    runtime.infer, runtime.task = tracked_infer, traced_task
    runtime.linear_calls = 0
    banks.seen.clear()
    before = runtime.q.summary()
    try:
        record = runtime.episode(row, index)
    finally:
        runtime.infer, runtime.task = original_infer, original_task
    after = runtime.q.summary()
    evidence = {
        **runtime.install_metadata,
        'observed_module_count': len(banks.seen),
        'native_int4_gemm_calls': after['native_int4_gemm_calls'] - before['native_int4_gemm_calls'],
        'kv_packed_prefills': after['kv_packed_prefills'] - before['kv_packed_prefills'],
        'kv_layer_reads': after['kv_layer_reads'] - before['kv_layer_reads'],
        'quantized_linear_calls': runtime.linear_calls,
    }
    if len(banks.seen) != 614 or runtime.linear_calls <= 0 or any(
            evidence[key] != 0 for key in ('native_int4_gemm_calls', 'kv_packed_prefills', 'kv_layer_reads')):
        raise RuntimeError('Episode did not execute all decoded-BF16 VQ/A4 layers')
    record.update({
        'protocol': PROTOCOL, 'unit': f'closedloop_{worker_id}',
        'arm': ARM, 'worker_id': worker_id,
        'index': index, 'variant_id': str(row['variant_id']), 'suite': row['suite'],
        'dimension': row['dimension'], 'task_id': int(row['task_id']), 'state_id': 0,
        'env_seed': environment_seed(index), 'sampler_seed': sampler_seed(index),
        'pbs_job_id': job_id, 'complete': True,
        'transform_implementation': IMPLEMENTATION, 'vq_evidence': evidence,
        'gripper_command_values': gripper,
        'gripper_command_counts': {str(value): gripper.count(value) for value in (-1.0, 0.0, 1.0)},
    })
    _validate_saved_episode(record, row, worker_id, protocol)
    return record, query_total, run_first


def run_worker(args):
    protocol = read_json(args.protocol)
    source_protocol = read_json(Path(protocol['closedloop']['source_root']) / 'protocol.json')
    validate_protocol(protocol, source_protocol)
    manifest_path = args.manifest or Path(protocol['closedloop']['manifest'])
    manifest, rows = load_manifest(manifest_path)
    if Path(manifest_path).as_posix() != protocol['closedloop']['manifest']:
        # Permit equivalent local path spellings only when the caller is on Windows.
        if os.name != 'nt' or Path(manifest_path).resolve() != Path(protocol['closedloop']['manifest']).resolve():
            raise ValueError('Manifest argument differs from the frozen source manifest path')
    if manifest.get('protocol') != source_protocol['cohort']['manifest_protocol']:
        raise ValueError('Manifest protocol differs from the paired reference source')
    worker_id = args.worker_id
    indices = worker_indices(worker_id)
    results_root = Path(args.results or (Path(protocol['remote_root']) / 'results'))
    out = results_root / f'closedloop_{worker_id}'
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / 'protocol.json'
    if frozen_path.exists() and read_json(frozen_path) != protocol:
        raise ValueError('Cannot resume with a changed teacher-anchor protocol')
    save(frozen_path, protocol)

    completed = recover_episodes(out, rows, worker_id, protocol)
    setup_path = out / 'setup_receipt.json'
    setup = read_json(setup_path) if setup_path.is_file() else None
    if completed and setup is None:
        raise ValueError('Saved episodes exist without their source-parity setup receipt')
    if setup is not None:
        _validate_setup(setup, worker_id, protocol)
    raw_start = os.environ.get('FW_JOB_START_EPOCH')
    try:
        start_epoch = float(raw_start)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('PBS runner must export a valid FW_JOB_START_EPOCH') from exc
    if not math.isfinite(start_epoch) or start_epoch <= 0:
        raise RuntimeError('FW_JOB_START_EPOCH must be positive and finite')
    soft_deadline = protocol['closedloop_allocation']['soft_deadline_seconds']
    stop = False
    runtime = banks = None

    def receipt(status, **extra):
        return {
            'protocol': PROTOCOL, 'arm': ARM, 'unit': f'closedloop_{worker_id}',
            'worker_id': worker_id, 'pbs_jobid': os.environ['PBS_JOBID'],
            'status': status, 'complete': status == 'complete', 'episodes': len(completed),
            'expected': 140, 'indices': indices, 'completed_indices': list(completed),
            'soft_deadline_reached': stop, 'setup_receipt_file': 'setup_receipt.json',
            **extra,
        }

    try:
        if len(completed) < len(indices):
            save(out / 'progress.json', receipt('running', stage='model_loading'))
            runtime, banks, fresh_setup = _compute_setup(
                protocol, source_protocol, results_root, out, worker_id)
            if setup is not None and fresh_setup['install_metadata'] != setup['install_metadata']:
                raise ValueError('Resumed worker loaded different final80 deployment metadata')
            setup = fresh_setup
            save(setup_path, setup)
            query_total = sum(len(item['metrics']) for item in completed.values())
            runtime.arm_calls[ARM] = query_total
            run_first = True
            for index in indices:
                if index in completed:
                    continue
                if time.time() - start_epoch >= soft_deadline:
                    stop = True
                    break
                save(out / 'progress.json', receipt('running', stage='episode', index=index))
                record, query_total, run_first = _record_episode(
                    runtime, banks, rows[index], index, worker_id, protocol, query_total, run_first)
                save(out / 'episodes' / f'{index:03d}.json', record)
                completed[index] = record
                save(out / 'progress.json', receipt('running', stage='episode_saved', index=index))
                print(f'SLOT_COMPLETE worker={worker_id} index={index} success={record["success"]}', flush=True)
        elif setup is None:
            raise ValueError('Completed worker is missing its setup receipt')

        status = 'checkpointed' if stop else 'complete'
        if not stop and set(completed) != set(indices):
            raise RuntimeError('Worker completed coverage differs from its frozen 140-slot shard')
        result = receipt(status)
        save(out / 'summary.json', result)
        save(out / 'progress.json', result)
        print(f'WORKER_{status.upper()} worker={worker_id} episodes={len(completed)}', flush=True)
        return result
    except Exception as exc:
        save(out / 'failure.json', {
            'protocol': PROTOCOL, 'worker_id': worker_id,
            'type': type(exc).__name__, 'message': str(exc), 'completed': len(completed),
        })
        raise
    finally:
        if runtime is not None:
            if getattr(runtime, 'env', None) is not None:
                runtime.env.close()
            close = getattr(runtime, 'close', None)
            if close is None:
                close = getattr(runtime.q, 'close', None)
            if close is not None:
                close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        help='defaults to the protocol-bound frozen LIBERO-Plus manifest')
    parser.add_argument('--results', type=Path)
    parser.add_argument('--worker-id', type=int, choices=(0, 1), required=True)
    result = run_worker(parser.parse_args())
    if result['complete'] is not True:
        raise SystemExit(75)


if __name__ == '__main__':
    main()

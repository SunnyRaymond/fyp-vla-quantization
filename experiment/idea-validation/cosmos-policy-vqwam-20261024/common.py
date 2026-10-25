"""Shared Cosmos assets and prepared endpoints; all tensor work requires PBS."""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import types

HERE = Path(__file__).resolve().parent
_VERIFIED_ALLOCATION = None
_TORCH = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def guard():
    global _VERIFIED_ALLOCATION
    identity = (os.getpid(), os.environ.get('PBS_JOBID'), os.environ.get('PBS_NODEFILE'))
    if _VERIFIED_ALLOCATION == identity:
        return
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or 'login' in host.lower():
        raise RuntimeError('A genuine PBS compute allocation is required')
    nodes = {n.split('.')[0] for n in Path(nodefile).read_text().split()}
    if host not in nodes:
        raise RuntimeError('Host is outside the approved allocation')
    _VERIFIED_ALLOCATION = identity


def torch_module():
    global _TORCH
    guard()
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def save_tensors(path, value):
    torch = torch_module()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(value, temp)
    temp.replace(path)


def load_tensors(path, device='cpu'):
    # Trusted task captures include NumPy RNG state and pinned conditioning objects.
    return torch_module().load(path, map_location=device, weights_only=False)


def cpu_tree(value):
    torch = torch_module()
    if torch.is_tensor(value):
        with torch.inference_mode(False):
            return value.detach().to(device='cpu', memory_format=torch.preserve_format).clone(
                memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def ensure_imports(p):
    snapshot = HERE.parent if HERE.name == 'control_sources' else HERE
    candidates = [snapshot / 'legacy_sources', snapshot / 'source_helpers',
                  HERE / 'legacy', HERE / 'helpers', snapshot / 'engine_sources',
                  Path(p['asset_root']), Path(p['asset_root']) / 'vendor/cosmos-policy']
    if os.name == 'nt':
        candidates.append(HERE.parent / 'cosmos-policy-c8-w4a4')
    for path in reversed(candidates):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(1, str(path))


def _raw_observation(self, case_id):
    """Use existing C80/heldout captures without the legacy C8-only membership gate."""
    p = self.protocol
    allowed = set(p['calibration']['case_ids']) | set(p['performance']['case_ids'])
    case_id = int(case_id)
    if case_id not in allowed:
        raise ValueError('Case is outside the frozen C80/heldout panel')
    source = Path(p['calibration']['source_root'])
    plan = read_json(p['calibration']['plan_file'])
    if plan.get('status') != 'complete' or plan.get('protocol') != 'fastwam-vq-calibration-suites-v1':
        raise RuntimeError('Original observation source is incomplete or different')
    row = next((dict(x) for x in plan['inputs'] if int(x['case_id']) == case_id), None)
    if row is None:
        raise KeyError('Frozen source does not contain case ' + str(case_id))
    data_root = (source / 'data').resolve()
    for key in ('observation_file', 'metadata_file'):
        rel = Path(row[key])
        if rel.is_absolute():
            raise ValueError('Observation file must remain source-relative')
        full = (data_root / rel).resolve()
        full.relative_to(data_root)
        if not full.is_file():
            raise FileNotFoundError(full)
        row[key] = str(full)
    import numpy as np
    with np.load(row['observation_file'], allow_pickle=False) as data:
        observation = {key: data[key].copy() for key in data.files}
    metadata = read_json(row['metadata_file'])
    if (metadata.get('domain') != 'original' or int(metadata.get('case_id', -1)) != case_id
            or metadata.get('suite') != row['suite']
            or int(metadata.get('task_id', -1)) != int(row['task_id'])):
        raise RuntimeError('Frozen observation metadata differs')
    required = {'agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
                'robot0_eef_quat', 'robot0_gripper_qpos'}
    if not required.issubset(observation):
        raise RuntimeError('Missing observation fields')
    return observation, metadata, row


def prepare_runtime(p, root, out):
    guard()
    if isinstance(p, (str, Path)):
        p = read_json(p)
    ensure_imports(p)
    from cosmos_runtime import CosmosRuntime
    assets = Path(p['asset_root'])
    cache_path = Path(root) / 'results/language_cache/embeddings.pkl'
    cache_summary = read_json(cache_path.with_name('summary.json'))
    if (cache_summary.get('protocol') != p['protocol']
            or cache_summary.get('status') != 'complete' or not cache_path.is_file()):
        raise RuntimeError('A fully covered writable language cache is required before model loading')
    # The legacy constructor validates its read-only asset root; new work keeps its own ROOT.
    task_root_environment = os.environ.get('ROOT')
    os.environ['ROOT'] = str(assets)
    try:
        runtime = CosmosRuntime(p, assets, out, arm='bf16',
                                text_embeddings_path=str(cache_path))
    finally:
        if task_root_environment is None:
            os.environ.pop('ROOT', None)
        else:
            os.environ['ROOT'] = task_root_environment
    runtime.task_root = Path(root)
    runtime.c8_raw_observation = types.MethodType(_raw_observation, runtime)
    runtime.query_count_by_arm = {arm: 0 for arm in p['arms']}
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.controller._weight_requires_grad = {name: False for name in runtime.target_linears}
    original_query = runtime.query

    def query(*args, **kwargs):
        begin_query(runtime)
        try:
            return original_query(*args, **kwargs)
        finally:
            end_query(runtime)

    runtime.query = query
    return runtime


def begin_query(runtime):
    fn = getattr(runtime, 'invalidate_query_cache', None)
    if callable(fn):
        fn()
    fn = getattr(runtime, 'begin_query', None)
    if callable(fn):
        fn()


def end_query(runtime):
    fn = getattr(runtime, 'end_query', None)
    if callable(fn):
        fn()


def get_context(runtime, case_id, seed_index=0):
    torch = torch_module()
    case_id, seed_index = int(case_id), int(seed_index)
    path = runtime.task_root / 'results/teacher/contexts' / f'{case_id}_{seed_index}.pt'
    fallback = path.with_name(f'{case_id}.pt')
    if path.is_file() or (seed_index == 0 and fallback.is_file()):
        return load_tensors(path if path.is_file() else fallback)
    original_seed_index = runtime.protocol['calibration']['seed_index']
    runtime.protocol['calibration']['seed_index'] = seed_index
    try:
        batch, metadata, row = runtime.c8_data_batch(case_id)
    finally:
        runtime.protocol['calibration']['seed_index'] = original_seed_index
    seeds = row.get('sampler_seeds', [])
    if seed_index >= len(seeds):
        raise RuntimeError('Frozen source lacks requested sampler seed')
    import calibrate
    context = {'case_id': case_id, 'seed_index': seed_index, 'seed': int(seeds[seed_index]),
               'data_batch': cpu_tree(batch), 'metadata': metadata, 'plan_row': row,
               'latent_indices': cpu_tree(batch.get('latent_indices')),
               'pre_forward_rng_state': calibrate._rng_state(torch), 'endpoint_version': 4}
    save_tensors(path, context)
    if seed_index == 0 and not fallback.exists():
        save_tensors(fallback, context)
    return context


def extract_endpoint(runtime, generated, original, batch, indices, calls, packets):
    import endpoint as legacy
    torch = torch_module()
    size = generated.shape[0]
    device = generated.device
    action_idx = legacy._tensor_index(batch, 'action_latent_idx', size, device, torch, indices)
    values = legacy._gather_slot(generated, action_idx, size, torch).reshape(size, -1)
    width = int(runtime.protocol['inference']['chunk_size']) * 7
    chunks = values.shape[1] // width
    if chunks < 1:
        raise RuntimeError('Official action slot is too short')
    action = values[:, :chunks * width].reshape(size, chunks, width // 7, 7).mean(1)
    lo = torch.as_tensor(runtime.dataset_stats['actions_min'], device=device, dtype=torch.float32)
    hi = torch.as_tensor(runtime.dataset_stats['actions_max'], device=device, dtype=torch.float32)
    if runtime.protocol['inference']['unnormalize_actions']:
        action = .5 * (action.float() + 1) * (hi - lo) + lo
    future = {}
    for key, field in [('future_wrist', 'future_wrist_image_latent_idx'),
                       ('future_image', 'future_image_latent_idx')]:
        idx = legacy._tensor_index(batch, field, size, device, torch, indices)
        future[key] = legacy._gather_slot(generated, idx, size, torch)
    return legacy.EndpointResult(generated, original, action, future,
                                 int(runtime.protocol['inference']['denoising_steps']), calls, packets)


def endpoint(runtime, context, grad_enabled=False, collect_packets=False,
             checkpoint_denoiser=False, **kwargs):
    """Inference retains native sampler/denoise hooks; training removes only no_grad."""
    torch = torch_module()
    import calibrate
    import endpoint as legacy
    calibrate._restore_rng(context['pre_forward_rng_state'], torch)
    begin_query(runtime)
    try:
        if grad_enabled:
            return legacy.run_endpoint(runtime.model, context['data_batch'], int(context['seed']),
                runtime.protocol, grad_enabled=True, collect_packets=collect_packets,
                checkpoint_denoiser=checkpoint_denoiser, dataset_stats=runtime.dataset_stats,
                latent_indices=context.get('latent_indices'), **kwargs)
        device = next(runtime.model.parameters()).device
        with torch.inference_mode(False):
            batch = legacy._tree_copy_to_device(context['data_batch'], device, torch)
            legacy._preserve_video_layout(batch, torch)
        original_denoise = runtime.model.denoise
        state = {'calls': 0, 'packets': []}

        def counted(_model, x, sigma, condition, *args, **kw):
            step = state['calls']
            state['calls'] += 1
            if callable(getattr(runtime, 'statistics_step', None)):
                runtime.statistics_step(step)
            result = original_denoise(x, sigma, condition, *args, **kw)
            if collect_packets:
                state['packets'].append({'sampler_evaluation': step, 'denoiser_call': step,
                    'x': cpu_tree(x), 'sigma': cpu_tree(sigma),
                    'condition': legacy._tree_copy_to_device(condition, torch.device('cpu'), torch),
                    'target': cpu_tree(legacy._denoised_tensor(result))})
            return result

        runtime.model.denoise = types.MethodType(counted, runtime.model)
        try:
            with torch.inference_mode():
                generated, original = runtime.model.generate_samples_from_batch(
                    batch, seed=int(context['seed']), n_sample=int(batch['video'].shape[0]),
                    num_steps=int(runtime.protocol['inference']['denoising_steps']),
                    is_negative_prompt=False,
                    use_variance_scale=bool(runtime.protocol['inference']['use_variance_scale']),
                    return_orig_clean_latent_frames=True)
        finally:
            runtime.model.denoise = original_denoise
        return extract_endpoint(runtime, generated, original, batch,
                                context.get('latent_indices'), state['calls'], state['packets'])
    finally:
        end_query(runtime)


def allocation_elapsed():
    value = os.environ.get('COSMOS_JOB_START_EPOCH')
    if not value:
        raise RuntimeError('Allocation start timestamp is required')
    return time.time() - float(value)


def has_time(seconds, estimate=60):
    return allocation_elapsed() + estimate + 120 < float(seconds)


def target_manifest(runtime):
    return [{'name': name, 'shape': [module.out_features, module.in_features],
             'weight_numel': module.weight.numel()}
            for name, module in runtime.target_linears.items()]


def restore_bf16(runtime):
    runtime.controller._restore_current()
    runtime.controller.arm = 'bf16'
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.current_arm = 'bf16'


def check_tensor_parity(candidate, reference, atol=1e-6, rtol=0.0):
    torch = torch_module()
    a, b = candidate.detach().float(), reference.detach().to(candidate.device).float()
    delta = a - b
    return {'passed': bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            'max_abs': float(delta.abs().max()), 'rmse': float(delta.square().mean().sqrt()),
            'atol': atol, 'rtol': rtol}

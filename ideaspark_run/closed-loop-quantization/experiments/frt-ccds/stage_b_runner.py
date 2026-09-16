"""Stage A/B FRT runner for a CCDS V100 allocation.

No top-level PyTorch/model import is used: main guards the allocation and GPU
before loading the existing DINO-WM smoke runtime.  The manifest freezes paths,
identities, episode splits, and the bounded fitting budget.  Stage B fits CAL
only, evaluates fresh residuals on DEV, and never opens TEST.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
import numpy as np
HERE = Path(__file__).resolve().parent
CORE_NAME = 'frt_core'
EPISODE_RE = re.compile('(?::|/)(\\d{3})$')
METHODS = ('clean', 'random', 'frt')

def _core():
    try:
        from frt_core import SourceHistoryAdapter, W4Spec, action_mask_like, atomic_json, atomic_npz, atomic_torch_save, estimate_wz, frt_loss, insert_slot_delta, mask_action_delta, materialize_hard, materialize_rtn, register_soft_quantizers, restore_modules, set_seed, shift_append, snapshot_modules, stable_hash, tensor_finite, transport, weighted_mse
        import frt_core
        return frt_core
    except ModuleNotFoundError:
        path = HERE / 'frt_core.py'
        spec = importlib.util.spec_from_file_location(CORE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[CORE_NAME] = module
        spec.loader.exec_module(module)
        return module

def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def _load_guard() -> Any:
    try:
        from allocation_guard import require_allocation
        return require_allocation
    except ModuleNotFoundError:
        path = HERE / 'allocation_guard.py'
        spec = importlib.util.spec_from_file_location('frt_allocation_guard', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load allocation guard {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.require_allocation

def _verify_gpu(allocation: Mapping[str, Any]) -> dict[str, Any]:
    """Check the actual visible allocation without importing torch."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('nvidia-smi GPU allocation verification failed') from exc
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 3:
            continue
        try:
            rows.append({'index': int(parts[0]), 'name': parts[1], 'memory_mib': int(parts[2])})
        except ValueError:
            continue
    if not rows:
        raise RuntimeError('nvidia-smi reported no visible GPU')
    job_id = str(allocation.get('job_id', ''))
    try:
        job_record = subprocess.run(['scontrol', 'show', 'job', '-o', job_id], check=True, text=True, capture_output=True, timeout=30).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('cannot verify SLURM AllocTRES for the current job') from exc
    fields = dict((item.split('=', 1) for item in job_record.split() if '=' in item))
    alloc_tres = str(fields.get('AllocTRES', ''))
    requested_gpu = 0
    for item in alloc_tres.split(','):
        if item.startswith(('gres/gpu=', 'gpu=', 'gres/gpu:')):
            try:
                requested_gpu = int(item.split('=', 1)[1].split('(', 1)[0])
            except ValueError:
                pass
    if requested_gpu < 1:
        raise RuntimeError(f'current SLURM job has no allocated GPU in AllocTRES={alloc_tres!r}')
    if len(rows) > requested_gpu:
        raise RuntimeError(f'nvidia-smi exposes {len(rows)} GPUs but job AllocTRES grants {requested_gpu}')
    if any(('v100' not in str(row['name']).casefold() for row in rows)):
        raise RuntimeError(f"FRT CCDS pilot requires V100, observed {[row['name'] for row in rows]}")
    if any((int(row['memory_mib']) < 30000 for row in rows)):
        raise RuntimeError(f'V100 allocation is smaller than 30 GiB: {rows}')
    return {'verified': True, 'gpus': rows, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}

def _resolve(path: Any, base: Path) -> Path:
    value = Path(str(path))
    return value.resolve() if value.is_absolute() else (base / value).resolve()

def _manifest(path: Path) -> dict[str, Any]:
    payload = _json(path)
    if not isinstance(payload, Mapping):
        raise ValueError('manifest must be a JSON object')
    schema = str(payload.get('schema', ''))
    if schema and (not schema.startswith('frt')):
        raise ValueError(f'unsupported FRT manifest schema: {schema!r}')
    return dict(payload)

def _stable_runtime(identity: Mapping[str, Any]) -> dict[str, Any]:
    keys = ('directory', 'checkpoint_file', 'recorded_epoch', 'checkpoint_size_bytes', 'source_commit', 'dinov2_source_commit', 'dtype', 'decoder', 'execution', 'torch', 'checkpoint_sha256', 'gpu')
    return {key: identity.get(key) for key in keys if key in identity}

def _load_smoke(manifest: Mapping[str, Any]) -> Any:
    configured = manifest.get('smoke_runner') or manifest.get('runtime', {}).get('smoke_runner')
    candidates = []
    if configured:
        candidates.append(Path(str(configured)))
    candidates.extend([HERE.parent.parent.parent / 'world-model-quantization' / 'experiments' / 'dino-wm-wall' / 'smoke_runner.py', HERE / 'smoke_runner.py', Path.cwd() / 'smoke_runner.py'])
    try:
        import smoke_runner
        return smoke_runner
    except ModuleNotFoundError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location('frt_smoke_runner', candidate.resolve())
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules['frt_smoke_runner'] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError('existing DINO-WM smoke_runner.py was not found')

def _load_runtime(manifest: Mapping[str, Any], root: Path, device: str | None) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    smoke = _load_smoke(manifest)
    runtime = smoke._runtime(root, device=device)
    identity = dict(smoke._checkpoint_identity(runtime))
    checkpoint = Path(runtime['checkpoint'])
    if checkpoint.is_file():
        identity['checkpoint_sha256'] = _core().sha256_file(checkpoint)
    return (smoke, runtime, identity)

def _check_identity(manifest: Mapping[str, Any], identity: Mapping[str, Any], gpu: Mapping[str, Any]) -> None:
    expected = manifest.get('runtime_identity') or manifest.get('identity') or {}
    if not isinstance(expected, Mapping):
        raise ValueError('manifest runtime_identity must be an object')
    actual = _stable_runtime(identity)
    for key, value in expected.items():
        if value is not None and (key not in actual or actual[key] != value):
            raise RuntimeError(f'runtime identity mismatch at {key!r}: {actual.get(key)!r} != {value!r}')
    expected_sha = manifest.get('checkpoint_sha256')
    if expected_sha and identity.get('checkpoint_sha256') != expected_sha:
        raise RuntimeError('checkpoint SHA256 does not match manifest')
    if not any(('v100' in str(row.get('name', '')).casefold() for row in gpu.get('gpus', []))):
        raise RuntimeError('actual GPU identity is not V100')

def _target_modules(smoke: Any, runtime: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[tuple[str, Any]], list[dict[str, Any]]]:
    model = runtime['model']
    groups = list(smoke._linear_groups(model))
    target = manifest.get('target_groups') or manifest.get('predictor_groups')
    target_ids = {str(value) for value in target} if isinstance(target, Sequence) and (not isinstance(target, (str, bytes))) else None
    modules: list[tuple[str, Any]] = []
    selected_groups = []
    for group in groups:
        if str(group.get('family')) != 'predictor':
            continue
        if target_ids is not None and str(group.get('group_id')) not in target_ids:
            continue
        selected_groups.append(dict(group))
        for linear in group.get('linear', []):
            path = str(linear['path'])
            module = smoke._module_for_path(model, 'predictor', int(group['index']), str(linear['relative']))
            modules.append((path, module))
    if not modules:
        raise RuntimeError('no predictor target Linear modules found')
    if target_ids is not None and {str(group.get('group_id')) for group in selected_groups} != target_ids:
        raise RuntimeError('manifest target_groups do not match runtime predictor groups')
    return (modules, selected_groups)

def _path_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _identity(manifest: Mapping[str, Any], manifest_path: Path, run_id: str, stage: str, identity: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], allocation: Mapping[str, Any], gpu: Mapping[str, Any]) -> dict[str, Any]:
    return {'schema': 'frt-ccds-progress-v1', 'run_id': run_id, 'stage': stage, 'manifest_sha256': _path_hash(manifest_path), 'runtime_identity': _stable_runtime(identity), 'target_groups': [{'group_id': group.get('group_id'), 'linear': group.get('linear'), 'numel': group.get('numel')} for group in groups], 'allocation': {key: allocation.get(key) for key in ('scheduler', 'job_id', 'hostname', 'nodelist')}, 'gpu': gpu}

def _progress_start(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    core = _core()
    if path.is_file():
        prior = _json(path)
        if prior.get('identity') != identity:
            raise RuntimeError('existing progress identity differs; refusing silent restart')
        return dict(prior)
    state = {'schema': 'frt-ccds-progress-v1', 'status': 'running', 'identity': dict(identity), 'completed': [], 'updated_at': time.time()}
    core.atomic_json(path, state)
    return state

def _progress(path: Path, state: MutableMapping[str, Any], **updates: Any) -> None:
    state.update(updates)
    state['updated_at'] = time.time()
    _core().atomic_json(path, state)

def _load_data(path: Path) -> Any:
    if path.is_dir():
        for name in ('records.pkl', 'bank.pkl', 'workload.pkl', 'records.json', 'bank.json'):
            candidate = path / name
            if candidate.is_file():
                return _load_data(candidate)
        raise FileNotFoundError(f'no record file under {path}')
    suffix = path.suffix.lower()
    if suffix == '.json':
        return _json(path)
    if suffix in ('.pkl', '.pickle'):
        with path.open('rb') as stream:
            return pickle.load(stream)
    if suffix == '.npz':
        arrays = np.load(path, allow_pickle=True)
        if 'records' in arrays.files:
            return arrays['records'].tolist()
        if 'metadata_json' in arrays.files:
            metadata = json.loads(str(arrays['metadata_json'].tolist()))
            if isinstance(metadata, Mapping):
                return {**metadata, '_arrays': {key: arrays[key] for key in arrays.files if key != 'metadata_json'}}
        return {key: arrays[key] for key in arrays.files}
    raise ValueError(f'unsupported record file: {path}')

def _record_like(node: Mapping[str, Any]) -> bool:
    keys = set(node)
    return bool(keys.intersection({'history', 'x_history', 'latent_history', 'full_history', 'obs_0'})) and bool(keys.intersection({'action', 'a', 'next_action', 'actions', 'action_sequence'}))

def _flatten_records(node: Any, split: str, inherited: Mapping[str, Any] | None=None) -> list[dict[str, Any]]:
    inherited = dict(inherited or {})
    if isinstance(node, Mapping):
        if _record_like(node):
            result = dict(inherited)
            result.update(dict(node))
            result.setdefault('split', split)
            return [result]
        for key in ('records', 'cases', 'items', 'episodes'):
            if key in node:
                metadata = dict(inherited)
                for name in ('episode_id', 'episode', 'record_index', 'local_index', 'dataset_index', 'subset_index', 'original_dataset_index', 'window_start', 'env_seed', 'cem_seed', 'target_fingerprint'):
                    if name in node:
                        metadata[name] = node[name]
                rows: list[dict[str, Any]] = []
                value = node[key]
                if isinstance(value, Mapping):
                    value = list(value.values())
                if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
                    for item in value:
                        rows.extend(_flatten_records(item, split, metadata))
                return rows
        if '_arrays' in node:
            return _flatten_records({key: value for key, value in node.items() if key != '_arrays'}, split, inherited)
    if isinstance(node, Sequence) and (not isinstance(node, (str, bytes))):
        rows: list[dict[str, Any]] = []
        for item in node:
            rows.extend(_flatten_records(item, split, inherited))
        return rows
    raise ValueError(f'cannot find FRT records in {type(node).__name__}')

def _split_source(source: Any, base: Path, split: str) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        return _flatten_records(_load_data(_resolve(source, base)), split)
    if isinstance(source, Mapping):
        if any((key in source for key in ('path', 'file', 'records_path', 'record_path'))):
            path = next((source[key] for key in ('path', 'file', 'records_path', 'record_path') if key in source))
            rows = _flatten_records(_load_data(_resolve(path, base)), split)
            extra = {key: value for key, value in source.items() if key not in {'path', 'file', 'records_path', 'record_path'}}
            for row in rows:
                for key, value in extra.items():
                    row.setdefault(key, value)
            return rows
        return _flatten_records(source, split)
    return _flatten_records(source, split)

def _split_sources(manifest: Mapping[str, Any], base: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    splits = manifest.get('splits')
    if isinstance(splits, Mapping):
        cal_source = splits.get('cal') or splits.get('CAL') or splits.get('calibration')
        dev_source = splits.get('dev') or splits.get('DEV') or splits.get('development')
    else:
        cal_source = manifest.get('cal') or manifest.get('calibration') or manifest.get('cal_path')
        dev_source = manifest.get('dev') or manifest.get('development') or manifest.get('dev_path')
    if cal_source is None or dev_source is None:
        raise ValueError('manifest must provide splits.cal and splits.dev')
    return (_split_source(cal_source, base, 'cal'), _split_source(dev_source, base, 'dev'))

def _episode_id(row: Mapping[str, Any], split: str, index: int) -> str:
    value = row.get('episode_id', row.get('episode'))
    return str(value) if value is not None else f'{split}:{index:03d}'

def _validate_split_rows(rows: list[dict[str, Any]], split: str, require_fresh: bool) -> list[dict[str, Any]]:
    for index, row in enumerate(rows):
        row.setdefault('episode_id', _episode_id(row, split, index))
        row.setdefault('split', split)
        if row.get('dataset_index') is not None:
            row['dataset_index'] = int(row['dataset_index'])
            if require_fresh and row['dataset_index'] <= 49:
                raise ValueError(f"{split} uses historical dataset index {row['dataset_index']}; need fresh >49")
        elif require_fresh:
            raise ValueError(f"{split} record {row['episode_id']} has no dataset_index")
    episodes = sorted({str(row['episode_id']) for row in rows})
    if len(episodes) != 6:
        raise ValueError(f'{split} must contain exactly six episodes, got {len(episodes)}')
    for episode in episodes:
        if not any((str(row['episode_id']) == episode for row in rows)):
            raise ValueError(f'empty {split} episode {episode}')
    return rows

def _validate_split_pair(cal: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]]) -> None:
    cal_ids = {str(row['episode_id']) for row in cal}
    dev_ids = {str(row['episode_id']) for row in dev}
    if cal_ids & dev_ids:
        raise ValueError(f'CAL/DEV episode overlap: {sorted(cal_ids & dev_ids)}')
    cal_indices = {int(row['dataset_index']) for row in cal if row.get('dataset_index') is not None}
    dev_indices = {int(row['dataset_index']) for row in dev if row.get('dataset_index') is not None}
    if cal_indices & dev_indices:
        raise ValueError(f'CAL/DEV dataset index overlap: {sorted(cal_indices & dev_indices)}')

def _declared_indices(manifest: Mapping[str, Any], split: str) -> set[int] | None:
    declared = manifest.get('dataset_indices')
    if isinstance(declared, Mapping) and declared.get(split) is not None:
        value = declared.get(split)
    else:
        splits = manifest.get('splits')
        value = splits.get(split) if isinstance(splits, Mapping) else None
        if isinstance(value, Mapping):
            value = value.get('dataset_indices', value.get('indices'))
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f'manifest dataset_indices.{split} must be a list')
    result = {int(item) for item in value}
    if len(result) != 6:
        raise ValueError(f'manifest dataset_indices.{split} must contain six unique indices')
    return result

def _array(row: Mapping[str, Any], names: Sequence[str], *, required: bool=True) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
        arrays = row.get('_arrays')
        if isinstance(arrays, Mapping) and name in arrays:
            return arrays[name]
    if required:
        raise ValueError(f'record is missing one of {tuple(names)}')
    return None

def _tensor(value: Any, torch: Any, device: Any, dtype: Any=None) -> Any:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device)
        return result if dtype is None else result.to(dtype=dtype)
    result = torch.as_tensor(np.asarray(value), device=device)
    return result if dtype is None else result.to(dtype=dtype)

def _obs_tensor(obs: Mapping[str, Any], torch: Any, device: Any) -> dict[str, Any]:
    result = {}
    for key, value in obs.items():
        result[key] = _tensor(value, torch, device, torch.float32)
    return result

def _record_tensors(row: Mapping[str, Any], adapter: Any, torch: Any, device: Any) -> dict[str, Any]:
    """Convert one raw collector record into a one-sample source history."""
    history_value = _array(row, ('history', 'full_history', 'latent_history', 'z_history', 'x_history'), required=False)
    actions_value = _array(row, ('actions', 'action_sequence'), required=False)
    source_obs = row.get('obs_0', row.get('initial_obs'))
    if history_value is None:
        if not isinstance(source_obs, Mapping) or actions_value is None:
            raise ValueError('record needs latent history or obs_0 plus action_sequence')
        source_actions = _tensor(actions_value, torch, device, torch.float32)
        if source_actions.ndim == 2:
            source_actions = source_actions.unsqueeze(0)
        if source_actions.ndim != 3:
            raise ValueError('action_sequence must be [B,T,A]')
        source_obs = _obs_tensor(source_obs, torch, device)
        if source_obs['visual'].ndim == 4:
            source_obs = {key: value.unsqueeze(0) for key, value in source_obs.items()}
        initial_len = int(source_obs['visual'].shape[1])
        history = adapter.encode(source_obs, source_actions[:, :initial_len, ...])
        current_value = source_actions[:, initial_len:initial_len + 1, ...]
        next_value = source_actions[:, initial_len + 1:initial_len + 2, ...]
        if next_value.shape[1] == 0:
            next_value = current_value
    else:
        history = _tensor(history_value, torch, device, torch.float32)
        if history.ndim == 3:
            history = history.unsqueeze(0)
        if history.ndim != 4:
            raise ValueError(f'history must be [B,T,P,D], got {tuple(history.shape)}')
        current_value = _array(row, ('action', 'a', 'current_action', 'action_current'), required=False)
        if current_value is None and actions_value is not None:
            sequence = _tensor(actions_value, torch, device, torch.float32)
            current_value = sequence[..., -1:, :] if sequence.ndim == 3 else sequence[-1:]
        if current_value is None:
            raise ValueError('record is missing current action')
        next_value = _array(row, ('next_action', 'a_next', 'action_next', 'future_action'), required=False)
        if next_value is None:
            raise ValueError('record is missing next_action; FRT transport needs a fixed next action')
    action = _tensor(current_value, torch, device, torch.float32)
    next_action = _tensor(next_value, torch, device, torch.float32)
    if action.ndim == 1:
        action = action.reshape(1, 1, -1)
    elif action.ndim == 2:
        action = action.unsqueeze(1)
    if next_action.ndim == 1:
        next_action = next_action.reshape(1, 1, -1)
    elif next_action.ndim == 2:
        next_action = next_action.unsqueeze(1)
    if action.ndim != 3 or next_action.ndim != 3 or action.shape[1] != 1 or (next_action.shape[1] != 1):
        raise ValueError('current/next actions must be [B,1,A]')
    if action.shape[0] != history.shape[0] or next_action.shape[0] != history.shape[0]:
        raise ValueError('history and action batch sizes disagree')
    if history.shape[1] > adapter.num_hist:
        raise ValueError(f'history length {history.shape[1]} exceeds model.num_hist={adapter.num_hist}')
    if history.shape[1] < max(1, min(2, adapter.num_hist)):
        raise ValueError('FRT records must contain more than one temporal context when the model supports it')
    return {'history': history, 'action': action, 'next_action': next_action}

def _expand_batched_rows(rows: Sequence[Mapping[str, Any]], adapter: Any, torch: Any, device: Any, split: str) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        converted = _record_tensors(row, adapter, torch, device)
        batch = int(converted['history'].shape[0])
        for item in range(batch):
            out = dict(row)
            out.update({key: value[item:item + 1] for key, value in converted.items()})
            out['episode_id'] = str(row.get('episode_id', f'{split}:{index:03d}'))
            expanded.append(out)
    return expanded

def _target_modules_identity(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{'group_id': group.get('group_id'), 'linear': group.get('linear'), 'numel': group.get('numel')} for group in groups]

def _snapshot_cpu(modules: Sequence[tuple[str, Any]], core: Any) -> dict[str, Any]:
    return core.snapshot_modules(modules, cpu=True)

def _fp_outputs(adapter: Any, history: Any, action: Any, next_action: Any, core: Any) -> dict[str, Any]:
    torch = __import__('torch')
    with torch.no_grad():
        fp_current = adapter.one_step(history, action)
        x_history = adapter.append(history, fp_current)
        fp_transport = core.transport(adapter, x_history, torch.zeros_like(fp_current), next_action)
    return {'fp_current': fp_current, 'x_history': x_history, 'fp_transport_zero': fp_transport}

def _build_bank(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], snapshot: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], adapter: Any, split: str, random_seed: int, wz: Any | None=None) -> dict[str, Any]:
    """Collect paired FP/Q0 residuals; all returned tensors are detached CPU arrays."""
    core = _core()
    torch = runtime['torch']
    model = runtime['model']
    if not rows:
        raise ValueError(f'{split} contains no records')
    core.restore_modules(modules, snapshot)
    fp_rows = []
    with torch.no_grad():
        for row in rows:
            history, action, next_action = (row['history'], row['action'], row['next_action'])
            fp_current = adapter.one_step(history, action)
            x_history = adapter.append(history, fp_current)
            fp_rows.append({'history': history, 'action': action, 'next_action': next_action, 'fp_current': fp_current, 'x_history': x_history})
    core.restore_modules(modules, snapshot)
    with torch.no_grad():
        q0_meta = core.materialize_rtn(modules, core.W4Spec())
        q0_rows = []
        for row in fp_rows:
            q0_current = adapter.one_step(row['history'], row['action'])
            delta = q0_current - row['fp_current']
            delta = core.mask_action_delta(delta, adapter.action_mask(delta))
            q0_transport = core.transport(adapter, row['x_history'], delta, row['next_action'])
            q0_rows.append({**row, 'q0_current': q0_current, 'delta': delta, 'q0_transport': q0_transport})
    core.restore_modules(modules, snapshot)
    with torch.no_grad():
        for row in q0_rows:
            q0_history = adapter.append(row['history'], row['q0_current'])
            expected = core.insert_slot_delta(row['x_history'], row['delta'])
            if not torch.allclose(q0_history, expected, rtol=1e-05, atol=1e-06):
                raise RuntimeError(f'{split} Q0 delta is not confined to the appended observation/proprio slot')
            row['fp_transport'] = core.transport(adapter, row['x_history'], row['delta'], row['next_action'])
    histories = torch.cat([row['history'] for row in q0_rows], dim=0).detach().cpu()
    x_histories = torch.cat([row['x_history'] for row in q0_rows], dim=0).detach().cpu()
    actions = torch.cat([row['action'] for row in q0_rows], dim=0).detach().cpu()
    next_actions = torch.cat([row['next_action'] for row in q0_rows], dim=0).detach().cpu()
    fp_current = torch.cat([row['fp_current'] for row in q0_rows], dim=0).detach().cpu()
    q0_current = torch.cat([row['q0_current'] for row in q0_rows], dim=0).detach().cpu()
    q0_transport = torch.cat([row['q0_transport'] for row in q0_rows], dim=0).detach().cpu()
    delta = torch.cat([row['delta'] for row in q0_rows], dim=0).detach().cpu()
    fp_transport = torch.cat([row['fp_transport'] for row in q0_rows], dim=0).detach().cpu()
    generator = torch.Generator(device='cpu').manual_seed(int(random_seed))
    random_delta = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
    random_delta = core.mask_action_delta(random_delta, adapter.action_mask(random_delta))
    if wz is None:
        wz = core.estimate_wz(fp_current, floor=0.001).detach().cpu()
    wz = wz.detach().cpu() if hasattr(wz, 'detach') else torch.as_tensor(wz, dtype=delta.dtype)
    if adapter.concat_dim == 1 and adapter.expanded_action_dim:
        wz[-adapter.expanded_action_dim:] = 0
    white_norm = random_delta.reshape(random_delta.shape[0], -1).norm(dim=1, keepdim=True).clamp_min(1e-12)
    target_norm = (delta * wz.reshape((1,) * (delta.ndim - 1) + (delta.shape[-1],))).reshape(delta.shape[0], -1).norm(dim=1, keepdim=True)
    white = random_delta / white_norm.reshape((-1,) + (1,) * (delta.ndim - 1))
    safe_wz = wz.clamp_min(1e-12).reshape((1,) * (delta.ndim - 1) + (delta.shape[-1],))
    random_delta = white * target_norm.reshape((-1,) + (1,) * (delta.ndim - 1)) / safe_wz
    with torch.no_grad():
        random_transport_parts = []
        for start in range(0, len(histories), 16):
            end = min(len(histories), start + 16)
            h = histories[start:end].to(device=next(model.parameters()).device)
            x = x_histories[start:end].to(device=h.device)
            a = next_actions[start:end].to(device=h.device)
            d = random_delta[start:end].to(device=h.device)
            random_transport_parts.append(core.transport(adapter, x, d, a).detach().cpu())
        random_transport = torch.cat(random_transport_parts, dim=0)
    provenance_keys = ('episode_id', 'record_index', 'local_index', 'dataset_index', 'subset_index', 'original_dataset_index', 'window_start', 'env_seed', 'cem_seed')
    metadata = []
    for row in rows:
        item = {key: row.get(key) for key in provenance_keys if key in row}
        if 'episode_id' in item:
            item['episode_id'] = str(item['episode_id'])
        metadata.append(item)
    return {'split': split, 'metadata': metadata, 'q0_meta': q0_meta, 'history': histories, 'x_history': x_histories, 'action': actions, 'next_action': next_actions, 'fp_current': fp_current, 'q0_current': q0_current, 'q0_transport': q0_transport, 'delta': delta, 'fp_transport': fp_transport, 'random_delta': random_delta, 'random_transport': random_transport, 'wz': wz, 'action_mask': adapter.action_mask(delta[:1])[0, 0].detach().cpu(), 'active_mask': (~adapter.action_mask(delta[:1])[0, 0]).detach().cpu()}

def _save_bank(path: Path, bank: Mapping[str, Any], core: Any) -> None:
    arrays = {}
    for key, value in bank.items():
        if isinstance(value, (np.ndarray,)):
            arrays[key] = value
        else:
            try:
                torch = __import__('torch')
                if isinstance(value, torch.Tensor):
                    arrays[key] = value.detach().cpu().numpy()
            except Exception:
                pass
    arrays['metadata_json'] = json.dumps({'split': bank.get('split'), 'metadata': bank.get('metadata'), 'q0_meta': bank.get('q0_meta')}, default=core.json_default)
    core.atomic_npz(path, arrays)

def _bank_device(bank: Mapping[str, Any], device: Any) -> dict[str, Any]:
    torch = __import__('torch')
    result = {}
    for key, value in bank.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device)
    return result

def _batch_indices(n: int, batch_size: int, seed: int, step: int) -> np.ndarray:
    if batch_size >= n:
        return np.arange(n, dtype=np.int64)
    generator = np.random.default_rng(int(seed) + int(step) * 1000003)
    return np.sort(generator.choice(n, size=batch_size, replace=False).astype(np.int64))

def _method_direction(bank: Mapping[str, Any], method: str) -> tuple[str, str]:
    if method == 'frt':
        return ('delta', 'fp_transport')
    if method == 'random_same_norm':
        return ('random_delta', 'random_transport')
    if method == 'clean':
        return ('delta', 'fp_transport')
    raise ValueError(f'unknown FRT method {method!r}')

def _rounding_config(manifest: Mapping[str, Any]) -> dict[str, float]:
    fit = manifest.get('fit', {})
    if not isinstance(fit, Mapping):
        fit = {}
    return {'temperature_start': float(fit.get('temperature_start', 2.0)), 'temperature_end': float(fit.get('temperature_end', 0.1)), 'regularization_start': float(fit.get('rounding_regularization_start', fit.get('rounding_regularization', 0.01))), 'regularization_end': float(fit.get('rounding_regularization_end', fit.get('rounding_regularization', 0.1)))}

def _fit_method(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], snapshot: Mapping[str, Any], bank: Mapping[str, Any], adapter: Any, method: str, seed: int, manifest: Mapping[str, Any], output: Path, progress_path: Path, progress_state: MutableMapping[str, Any]) -> dict[str, Any]:
    core = _core()
    torch = runtime['torch']
    model = runtime['model']
    fit_cfg = manifest.get('fit', {}) if isinstance(manifest.get('fit', {}), Mapping) else {}
    optimizer_cfg = fit_cfg.get('optimizer', {}) if isinstance(fit_cfg.get('optimizer', {}), Mapping) else {}
    loss_cfg = fit_cfg.get('loss', {}) if isinstance(fit_cfg.get('loss', {}), Mapping) else {}
    steps = int(fit_cfg.get('fit_updates', fit_cfg.get('steps', manifest.get('fit_steps', 0))))
    batch_size = int(fit_cfg.get('batch_size', manifest.get('batch_size', len(bank['history']))))
    lr = float(optimizer_cfg.get('lr', fit_cfg.get('lr', manifest.get('lr', 0.01))))
    lambda_value = loss_cfg.get('lambda_T', loss_cfg.get('lambda_transport', fit_cfg.get('lambda_transport', manifest.get('lambda_transport'))))
    if method != 'clean' and lambda_value is None:
        raise ValueError('manifest must explicitly freeze fit.loss.lambda_T/lambda_transport for transport methods')
    lambda_transport = float(lambda_value if lambda_value is not None else 0.0)
    if steps <= 0:
        raise ValueError('manifest fit.steps must be a positive bounded value, frozen from Stage A')
    if batch_size <= 0 or lr <= 0:
        raise ValueError('fit batch_size/lr must be positive')
    core.restore_modules(modules, snapshot)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handles = core.register_soft_quantizers(modules, core.W4Spec())
    params = []
    for handle in handles:
        params.extend([handle.parametrization.scale_raw, handle.parametrization.alpha])
    optimizer = torch.optim.Adam(params, lr=lr)
    directions, target_key = _method_direction(bank, method)
    device_bank = _bank_device(bank, next(model.parameters()).device)
    wz = device_bank['wz']
    fit_rows = []
    started = time.monotonic()
    rounding = _rounding_config(manifest)
    for step in range(steps):
        temperature = rounding['temperature_start'] + (rounding['temperature_end'] - rounding['temperature_start']) * ((step + 1) / steps)
        regularization_weight = rounding['regularization_start'] + (rounding['regularization_end'] - rounding['regularization_start']) * ((step + 1) / steps)
        for handle in handles:
            handle.parametrization.temperature = temperature
        indices = _batch_indices(len(device_bank['history']), min(batch_size, len(device_bank['history'])), seed, step)
        idx = torch.as_tensor(indices, device=next(model.parameters()).device, dtype=torch.long)
        with torch.enable_grad():
            history = device_bank['history'].index_select(0, idx)
            actions = device_bank['action'].index_select(0, idx)
            next_actions = device_bank['next_action'].index_select(0, idx)
            x_history = device_bank['x_history'].index_select(0, idx)
            direction = device_bank[directions].index_select(0, idx)
            clean_prediction = adapter.one_step(history, actions)
            transport_prediction = core.transport(adapter, x_history, direction, next_actions)
            losses = core.frt_loss(clean_prediction, device_bank['fp_current'].index_select(0, idx), transport_prediction, device_bank[target_key].index_select(0, idx), wz, lambda_transport if method != 'clean' else 0.0)
            regularizer = torch.stack([handle.parametrization.rounding_regularizer() for handle in handles]).mean()
            total = losses['total'] + regularization_weight * regularizer
            if not core.tensor_finite(total):
                raise FloatingPointError(f'non-finite {method} loss at step {step + 1}')
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        if step == 0 or step + 1 == steps or (step + 1) % max(1, min(25, steps // 5 or 1)) == 0:
            fit_rows.append({'step': step + 1, 'clean': float(losses['clean'].detach().item()), 'transport': float(losses['transport'].detach().item()), 'rounding_regularizer': float(regularizer.detach().item()), 'temperature': temperature, 'regularization_weight': regularization_weight, 'total': float(total.detach().item())})
            _progress(progress_path, progress_state, current_method=f'{method}:{seed}', current_step=step + 1)
    hard_ledger = core.materialize_hard(handles)
    elapsed = time.monotonic() - started
    checkpoint_path = output / 'checkpoints' / f'{method}_seed_{seed}.pt'
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    core.atomic_torch_save(checkpoint_path, {'schema': 'frt-hard-w4-checkpoint-v1', 'method': method, 'seed': seed, 'state_dict': state_dict, 'hard_ledger': hard_ledger})
    result = {'method': method, 'seed': int(seed), 'steps': steps, 'batch_size': batch_size, 'lr': lr, 'lambda_transport': lambda_transport, 'lambda_transport_source': 'fit.loss' if lambda_value is not None else 'clean_only_zero', 'rounding': {**rounding, 'final_temperature': temperature, 'final_regularization_weight': regularization_weight}, 'elapsed_seconds': elapsed, 'checkpoint': str(checkpoint_path.resolve()), 'hard_ledger': hard_ledger, 'fit_trace': fit_rows}
    return result

def _metric_row(value: Any) -> float:
    return float(value.detach().item())

def _evaluate_model(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], snapshot: Mapping[str, Any], bank: Mapping[str, Any], adapter: Any, wz: Any, method: str) -> dict[str, Any]:
    core = _core()
    torch = runtime['torch']
    model = runtime['model']
    device = next(model.parameters()).device
    data = _bank_device(bank, device)
    direction_key, target_key = _method_direction(bank, method)
    hard_snapshot = {path: module.weight.detach().clone() for path, module in modules}
    with torch.no_grad():
        clean = adapter.one_step(data['history'], data['action'])
        frozen_transport = core.transport(adapter, data['x_history'], data['delta'], data['next_action'])
        requested_transport = core.transport(adapter, data['x_history'], data[direction_key], data['next_action'])
        metrics = {'clean_mse': _metric_row(core.weighted_mse(clean, data['fp_current'], wz)), 'frozen_q0_transport_mse': _metric_row(core.weighted_mse(frozen_transport, data['fp_transport'], wz)), 'requested_transport_mse': _metric_row(core.weighted_mse(requested_transport, data[target_key], wz)), 'mean_q0_delta_norm': _metric_row(data['delta'].reshape(data['delta'].shape[0], -1).norm(dim=1).mean())}
        theta_current = clean
        theta_history = adapter.append(data['history'], theta_current)
        theta_next = adapter.one_step(theta_history, data['next_action'])
        core.restore_modules(modules, snapshot)
        fp_theta_next = adapter.one_step(theta_history, data['next_action'])
        theta_delta = core.mask_action_delta(theta_next - fp_theta_next, adapter.action_mask(theta_next))
        core.restore_modules(modules, hard_snapshot)
        theta_base_history = adapter.append(theta_history, fp_theta_next)
        theta_transport = core.transport(adapter, theta_base_history, theta_delta, data['next_action'])
        core.restore_modules(modules, snapshot)
        fp_theta_transport = core.transport(adapter, theta_base_history, theta_delta, data['next_action'])
        metrics['fresh_qtheta_history_clean_mse'] = _metric_row(core.weighted_mse(theta_next, fp_theta_next, wz))
        metrics['fresh_qtheta_transport_mse'] = _metric_row(core.weighted_mse(theta_transport, fp_theta_transport, wz))
        metrics['mean_qtheta_delta_norm'] = _metric_row(theta_delta.reshape(theta_delta.shape[0], -1).norm(dim=1).mean())
        metrics['finite'] = bool(torch.isfinite(clean).all().item() and torch.isfinite(theta_transport).all().item())
    core.restore_modules(modules, hard_snapshot)
    return metrics

def _stage_a_source_rows(manifest: Mapping[str, Any], base: Path) -> list[dict[str, Any]]:
    stage = manifest.get('stage_a', {})
    if not isinstance(stage, Mapping):
        stage = {}
    source = stage.get('records_path') or stage.get('records') or manifest.get('stage_a_records')
    if source is None:
        return []
    return _split_source(source, base, 'stage_a')

def _source_bridge_gate(rows: Sequence[Mapping[str, Any]], adapter: Any, torch: Any, device: Any) -> dict[str, Any]:
    for row in rows:
        obs = row.get('obs_0', row.get('initial_obs'))
        actions = _array(row, ('actions', 'action_sequence'), required=False)
        if not isinstance(obs, Mapping) or actions is None:
            continue
        obs_t = _obs_tensor(obs, torch, device)
        if obs_t['visual'].ndim == 4:
            obs_t = {key: value.unsqueeze(0) for key, value in obs_t.items()}
        actions_t = _tensor(actions, torch, device, torch.float32)
        if actions_t.ndim == 2:
            actions_t = actions_t.unsqueeze(0)
        if actions_t.ndim != 3 or actions_t.shape[1] <= obs_t['visual'].shape[1]:
            continue
        with torch.no_grad():
            expected, observed = adapter.source_rollout_one_step(obs_t, actions_t)
        close = bool(torch.allclose(expected, observed, rtol=1e-05, atol=1e-06))
        return {'available': True, 'passed': close, 'max_abs_error': float((expected - observed).abs().max().item())}
    return {'available': False, 'passed': False, 'reason': 'no obs_0+action_sequence source bridge record'}

def _physical_replay_gate(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage = manifest.get('stage_a', {})
    configured = stage.get('physical_replay') if isinstance(stage, Mapping) else None
    candidates: list[Any] = []
    if configured is not None:
        candidates.append(configured)
    candidates.extend((row.get('physical_replay') for row in rows if row.get('physical_replay') is not None))
    for value in candidates:
        if not isinstance(value, Mapping):
            continue
        state_a = value.get('states_a', value.get('states_first'))
        state_b = value.get('states_b', value.get('states_second'))
        xy_a = value.get('xy_a')
        xy_b = value.get('xy_b')
        if state_a is None or state_b is None:
            continue
        same_state = bool(np.array_equal(np.asarray(state_a), np.asarray(state_b)))
        same_xy = True if xy_a is None or xy_b is None else bool(np.array_equal(np.asarray(xy_a), np.asarray(xy_b)))
        xy = np.asarray(xy_a if xy_a is not None else state_a)
        finite_xy = bool(np.isfinite(xy).all() and xy.shape[-1] >= 2)
        return {'available': True, 'passed': same_state and same_xy and finite_xy, 'same_state_replay': same_state, 'same_xy_replay': same_xy, 'xy_shape': list(xy.shape), 'finite_xy': finite_xy}
    return {'available': False, 'passed': False, 'reason': 'no independent physical replay evidence'}

def _sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _timed_backward(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], snapshot: Mapping[str, Any], bank: Mapping[str, Any], adapter: Any, method: str, seed: int, manifest: Mapping[str, Any]) -> dict[str, Any]:
    core = _core()
    torch = runtime['torch']
    model = runtime['model']
    device = next(model.parameters()).device
    data = _bank_device(bank, device)
    core.restore_modules(modules, snapshot)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handles = core.register_soft_quantizers(modules, core.W4Spec())
    params = [parameter for handle in handles for parameter in (handle.parametrization.scale_raw, handle.parametrization.alpha)]
    fit_cfg = manifest.get('fit', {}) if isinstance(manifest.get('fit', {}), Mapping) else {}
    opt_cfg = fit_cfg.get('optimizer', {}) if isinstance(fit_cfg.get('optimizer', {}), Mapping) else {}
    optimizer = torch.optim.Adam(params, lr=float(opt_cfg.get('lr', fit_cfg.get('lr', 0.01))))
    direction_key, target_key = _method_direction(bank, method)
    with torch.enable_grad():
        for handle in handles:
            handle.parametrization.temperature = 2.0
        optimizer.zero_grad(set_to_none=True)
        warm_clean = adapter.one_step(data['history'], data['action'])
        warm_transport = core.transport(adapter, data['x_history'], data[direction_key], data['next_action'])
        warm_losses = core.frt_loss(warm_clean, data['fp_current'], warm_transport, data[target_key], data['wz'], 0.0 if method == 'clean' else 1.0)
        warm_reg = torch.stack([handle.parametrization.rounding_regularizer() for handle in handles]).mean()
        (warm_losses['total'] + 0.01 * warm_reg).backward()
        optimizer.zero_grad(set_to_none=True)
    _sync(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
    core.set_seed(seed)
    _sync(torch)
    start = time.perf_counter()
    with torch.enable_grad():
        clean = adapter.one_step(data['history'], data['action'])
        tr = core.transport(adapter, data['x_history'], data[direction_key], data['next_action'])
        losses = core.frt_loss(clean, data['fp_current'], tr, data[target_key], data['wz'], 0.0 if method == 'clean' else 1.0)
        reg = torch.stack([handle.parametrization.rounding_regularizer() for handle in handles]).mean()
        total = losses['total'] + 0.01 * reg
        total.backward()
        optimizer.step()
    _sync(torch)
    elapsed = time.perf_counter() - start
    peak_allocated = float(torch.cuda.max_memory_allocated(device=device) / 2 ** 30) if torch.cuda.is_available() else None
    peak_reserved = float(torch.cuda.max_memory_reserved(device=device) / 2 ** 30) if torch.cuda.is_available() else None
    core.discard_soft(handles)
    core.restore_modules(modules, snapshot)
    return {'method': method, 'seed': int(seed), 'elapsed_seconds': elapsed, 'seconds_per_fit_step': elapsed, 'peak_vram_allocated_gib': peak_allocated, 'peak_vram_reserved_gib': peak_reserved, 'soft_clean_loss': float(losses['clean'].detach().item()), 'soft_transport_loss': float(losses['transport'].detach().item()), 'soft_total_loss': float(total.detach().item()), 'rounding_regularizer': float(reg.detach().item()), 'temperature': 2.0, 'finite': core.tensor_finite(total)}

def _hard_reload_gate(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], snapshot: Mapping[str, Any], bank: Mapping[str, Any], adapter: Any, output: Path) -> dict[str, Any]:
    core = _core()
    torch = runtime['torch']
    model = runtime['model']
    device = next(model.parameters()).device
    core.restore_modules(modules, snapshot)
    ledger = core.materialize_rtn(modules, core.W4Spec())
    with torch.no_grad():
        hard_output = adapter.one_step(bank['history'].to(device), bank['action'].to(device)).detach()
    hard_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = output / 'hard_w4_reload.pt'
    core.atomic_torch_save(checkpoint, {'schema': 'frt-hard-w4-checkpoint-v1', 'method': 'Q0-RTN', 'state_dict': hard_state, 'hard_ledger': ledger})
    core.atomic_json(output / 'hard_w4_ledger.json', {'schema': 'frt-hard-w4-ledger-v1', 'rows': ledger})
    core.restore_modules(modules, snapshot)
    loaded = torch.load(checkpoint, map_location=device)
    model.load_state_dict(loaded['state_dict'], strict=True)
    with torch.no_grad():
        reloaded = adapter.one_step(bank['history'].to(device), bank['action'].to(device)).detach()
    same = bool(torch.allclose(hard_output, reloaded, rtol=0.0, atol=0.0))
    core.restore_modules(modules, snapshot)
    return {'passed': same, 'max_abs_error': float((hard_output - reloaded).abs().max().item()), 'checkpoint': str(checkpoint.resolve()), 'ledger_rows': len(ledger), 'ledger_sha256': core.stable_hash(ledger)}

def _stage_a(manifest: Mapping[str, Any], manifest_path: Path, args: argparse.Namespace, allocation: Mapping[str, Any], gpu: Mapping[str, Any], output: Path, progress_path: Path, progress_state: MutableMapping[str, Any]) -> dict[str, Any]:
    core = _core()
    root = _resolve(args.root or manifest.get('root') or manifest.get('runtime_root'), manifest_path.parent)
    loaded = getattr(args, '_loaded_runtime', None)
    if loaded is None:
        smoke, runtime, runtime_identity = _load_runtime(manifest, root, args.device)
    else:
        smoke, runtime, runtime_identity = loaded
    _check_identity(manifest, runtime_identity, gpu)
    modules, groups = _target_modules(smoke, runtime, manifest)
    torch = runtime['torch']
    model = runtime['model']
    model.eval()
    adapter = core.SourceHistoryAdapter(model)
    source_rows = _stage_a_source_rows(manifest, manifest_path.parent)
    if not source_rows:
        result = {'schema': 'frt-stage-a-v1', 'status': 'blocked', 'engineering_pass': False, 'reason': 'manifest has no stage_a records; collector must provide full history/action and replay evidence', 'allocation': dict(allocation), 'gpu': gpu, 'runtime_identity': runtime_identity}
        core.atomic_json(output / 'stage_a_summary.json', result)
        core.atomic_json(output / 'summary.json', result)
        _progress(progress_path, progress_state, status='blocked', result='stage_a_summary.json')
        return result
    source_rows = _expand_batched_rows(source_rows, adapter, torch, next(model.parameters()).device, 'stage_a')
    if not source_rows:
        raise ValueError('stage A record source produced no rows')
    snapshot = core.snapshot_modules(modules, cpu=True)
    full_bank = _build_bank(runtime, modules, snapshot, source_rows, adapter, 'stage_a', int(manifest.get('random_seed', 910001)))
    probe_count = max(1, min(2, len(source_rows)))
    bank = {key: value[:probe_count] if isinstance(value, torch.Tensor) and value.ndim and (value.shape[0] == len(source_rows)) else value for key, value in full_bank.items()}
    bank['wz'] = full_bank['wz'].to(device=next(model.parameters()).device)
    source_gate = _source_bridge_gate(source_rows, adapter, torch, next(model.parameters()).device)
    physical_gate = _physical_replay_gate(manifest, source_rows)
    device = next(model.parameters()).device
    data = _bank_device(bank, device)
    with torch.no_grad():
        zero_delta = torch.zeros_like(data['delta'])
        zero_transport = core.transport(adapter, data['x_history'], zero_delta, data['next_action'])
        old_history = data['x_history'].clone()
        inserted = core.insert_slot_delta(old_history, data['delta'])
        old_unchanged = bool(torch.equal(inserted[:, :-1], old_history[:, :-1]))
        action_mask = adapter.action_mask(data['delta'])
        action_zero = bool(torch.equal(data['delta'].masked_select(action_mask), torch.zeros_like(data['delta'].masked_select(action_mask))))
        delta_zero_gate = {'passed': float(zero_transport.abs().max().item()) <= 1e-07, 'max_abs_transport': float(zero_transport.abs().max().item())}
        fp_transport_a = core.transport(adapter, data['x_history'], data['delta'], data['next_action'])
        fp_transport_b = core.transport(adapter, data['x_history'], data['delta'], data['next_action'])
        fp_difference = fp_transport_a - fp_transport_b
        fp_null_gate = {'passed': bool(torch.allclose(fp_transport_a, fp_transport_b, rtol=0.0, atol=0.0)), 'max_abs_difference': float(fp_difference.abs().max().item())}
        invariant_gate = {'passed': old_unchanged and action_zero, 'old_history_unchanged': old_unchanged, 'action_delta_zero': action_zero}
    hard_gate = _hard_reload_gate(runtime, modules, snapshot, bank, adapter, output)
    learned_gate = core.one_step_hard_reload(model, modules, snapshot, adapter, bank, output)
    hard_gate['learned_fit'] = learned_gate
    _save_bank(output / 'bank_stage_a.npz', full_bank, core)
    timings = {}
    fit_cfg = manifest.get('fit', {}) if isinstance(manifest.get('fit', {}), Mapping) else {}
    timing_seeds = fit_cfg.get('seeds', manifest.get('seeds', [0]))
    if isinstance(timing_seeds, Sequence) and (not isinstance(timing_seeds, (str, bytes))):
        timing_seed = int(timing_seeds[0]) if timing_seeds else 0
    else:
        timing_seed = 0
    for method in ('clean', 'random_same_norm', 'frt'):
        timings[method] = _timed_backward(runtime, modules, snapshot, bank, adapter, method, timing_seed, manifest)
    max_step = max((float(row['seconds_per_fit_step']) for row in timings.values()))
    safe_budget = float(fit_cfg.get('safe_seconds', 1200.0) if isinstance(fit_cfg, Mapping) else 1200.0)
    safe_fit_steps = max(1, int(safe_budget / max(max_step, 1e-09)))
    q0_clean_mse = float(core.weighted_mse(full_bank['q0_current'], full_bank['fp_current'], full_bank['wz']).item())
    q0_transport_mse = float(core.weighted_mse(full_bank['q0_transport'], full_bank['fp_transport'], full_bank['wz']).item())
    lambda_raw = max(q0_clean_mse, 1e-12) / max(q0_transport_mse, 1e-12)
    lambda_t = min(4.0, max(0.25, lambda_raw))
    lambda_warning = []
    if q0_transport_mse <= 1e-12:
        lambda_warning.append('transport_component_floor')
    if lambda_t != lambda_raw:
        lambda_warning.append('lambda_clamped')
    summary = {'schema': 'frt-stage-a-v1', 'status': 'complete', 'engineering_pass': bool(source_gate['passed'] and physical_gate['passed'] and delta_zero_gate['passed'] and fp_null_gate['passed'] and invariant_gate['passed'] and hard_gate['passed'] and learned_gate['passed'] and all((row['finite'] for row in timings.values()))), 'allocation': dict(allocation), 'gpu': gpu, 'runtime_identity': runtime_identity, 'target_groups': _target_modules_identity(groups), 'gates': {'source_rollout_agreement': source_gate, 'physical_replay': physical_gate, 'delta_zero': delta_zero_gate, 'fp_transport_null': fp_null_gate, 'history_action_invariants': invariant_gate, 'hard_materialize_reload': hard_gate}, 'seconds_per_fit_step_by_method': {method: row['seconds_per_fit_step'] for method, row in timings.items()}, 'backward_timing': timings, 'peak_memory_by_method': {method: {'allocated_gib': row['peak_vram_allocated_gib'], 'reserved_gib': row['peak_vram_reserved_gib']} for method, row in timings.items()}, 'safe_1000_step_estimate_seconds_by_method': {method: float(row['seconds_per_fit_step'] * 1000 * 1.25) for method, row in timings.items()}, 'safe_fit_steps': safe_fit_steps, 'calibration_freeze': {'mean_L_clean_Q0': q0_clean_mse, 'mean_L_transport_Q0': q0_transport_mse, 'q0_clean_mse_wz': q0_clean_mse, 'q0_transport_mse_wz': q0_transport_mse, 'lambda_raw': lambda_raw, 'lambda_T': lambda_t, 'lambda_clip': [0.25, 4.0], 'lambda_clamped': bool(lambda_t != lambda_raw), 'scope_warning': lambda_warning}, 'q0_probe': {'records': len(source_rows), 'timing_records': probe_count, 'mean_delta_norm': float(data['delta'].reshape(data['delta'].shape[0], -1).norm(dim=1).mean().item())}, 'execution': 'fake_quant_emulation_only'}
    core.atomic_json(output / 'stage_a_summary.json', summary)
    core.atomic_json(output / 'summary.json', summary)
    _progress(progress_path, progress_state, status='complete', result='stage_a_summary.json', engineering_pass=summary['engineering_pass'])
    return summary

def _stage_b(manifest: Mapping[str, Any], manifest_path: Path, args: argparse.Namespace, allocation: Mapping[str, Any], gpu: Mapping[str, Any], output: Path, progress_path: Path, progress_state: MutableMapping[str, Any]) -> dict[str, Any]:
    from frt_stage_b import run_stage_b
    return run_stage_b(manifest, manifest_path, args, allocation, gpu, output, progress_path, progress_state, loaded=getattr(args, '_loaded_runtime', None))

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('a', 'b'), required=True)
    parser.add_argument('--root', type=Path, default=None, help='DINO-WM runtime root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--a-summary', type=Path, default=None)
    parser.add_argument('--device', default=None)
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    require_allocation = _load_guard()
    allocation = require_allocation()
    gpu = _verify_gpu(allocation)
    manifest_path = args.manifest.resolve()
    manifest = _manifest(manifest_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    core = _core()
    root = _resolve(args.root or manifest.get('root') or manifest.get('runtime_root'), manifest_path.parent)
    smoke, runtime, runtime_identity = _load_runtime(manifest, root, args.device)
    _check_identity(manifest, runtime_identity, gpu)
    _, groups = _target_modules(smoke, runtime, manifest)
    identity = _identity(manifest, manifest_path, args.run_id, args.stage, runtime_identity, groups, allocation, gpu)
    args._loaded_runtime = (smoke, runtime, runtime_identity)
    progress_path = output / 'progress.json'
    progress_state = _progress_start(progress_path, identity)
    summary_path = output / ('stage_a_summary.json' if args.stage == 'a' else 'stage_b_summary.json')
    if progress_state.get('status') == 'complete' and summary_path.is_file():
        print(summary_path.read_text(encoding='utf-8'), flush=True)
        return
    if args.stage == 'a':
        result = _stage_a(manifest, manifest_path, args, allocation, gpu, output, progress_path, progress_state)
    else:
        result = _stage_b(manifest, manifest_path, args, allocation, gpu, output, progress_path, progress_state)
    print(json.dumps({key: result.get(key) for key in ('schema', 'status', 'engineering_pass', 'safe_fit_steps', 'fit_updates', 'test_opened') if key in result}, indent=2), flush=True)
if __name__ == '__main__':
    main()

"""Independent raw-output verifier for the FRT CCDS-TC1 A/B pilot.

The verifier never imports the model and never opens an artifact before the
real SLURM allocation guard has passed. It audits arithmetic and metadata from
the raw NPZ/JSON outputs, then emits a compact summary. Raw arrays stay in the
compute-side artifact directory.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import re
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
METHODS = ('clean', 'random_same_norm', 'frt')
SEEDS = (1201, 1202, 1203)
BANKS = ('q0', 'random_same_norm', 'fresh:clean:1201', 'fresh:clean:1202', 'fresh:clean:1203', 'fresh:random_same_norm:1201', 'fresh:random_same_norm:1202', 'fresh:random_same_norm:1203', 'fresh:frt:1201', 'fresh:frt:1202', 'fresh:frt:1203')
EVALUATORS = ('fp32', 'q0_rtn', 'clean:1201', 'clean:1202', 'clean:1203', 'random_same_norm:1201', 'random_same_norm:1202', 'random_same_norm:1203', 'frt:1201', 'frt:1202', 'frt:1203')
Q0_BANK = 0
RANDOM_BANK = 1
FRESH_BANKS = tuple(range(2, 11))
FIT_EVALUATOR_INDEX = {f'{method}:{seed}': 2 + method_index * 3 + seed_index for method_index, method in enumerate(METHODS) for seed_index, seed in enumerate(SEEDS)}
EXPECTED_TARGETS = {'frt_cal': tuple(range(60, 66)), 'frt_dev': tuple(range(66, 72))}
EXPECTED_EPISODES = {'frt_cal': tuple((f'frt_cal:{index:03d}' for index in range(6))), 'frt_dev': tuple((f'frt_dev:{index:03d}' for index in range(6)))}
WZ_STD_FLOOR = 0.001
EPS = 1e-12
REL_TOL = 1e-06
VECTOR_RTOL = 2e-05
VECTOR_ATOL = 2e-07
MAX_ERRORS = 24

def _compact_errors(errors: Sequence[str]) -> list[str]:
    result = [str(error)[:240] for error in errors[:MAX_ERRORS]]
    if len(errors) > MAX_ERRORS:
        result.append(f'... {len(errors) - MAX_ERRORS} more errors')
    return result

def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=_json_default), encoding='utf-8')
    temporary.replace(path)
    if path.stat().st_size > 65536:
        raise RuntimeError('verification summary exceeds 64 KiB')

def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'cannot JSON encode {type(value)!r}')

def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _strings(values: Any) -> list[str]:
    array = np.asarray(values)
    return [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in array.reshape(-1).tolist()]

def _finite(name: str, value: Any, errors: list[str]) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        errors.append(f'{name} is not numeric')
        return array
    if not np.isfinite(array).all():
        errors.append(f'{name} contains non-finite values')
    return array

def _load_guard() -> Any:
    candidates = (Path(__file__).with_name('allocation_guard.py'), Path(__file__).parents[1] / 'cem-update-ptq-ccds' / 'allocation_guard.py')
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location('frt_allocation_guard', path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module.require_allocation
    raise FileNotFoundError('allocation_guard.py was not found beside FRT or old CCDS CEM')

def _runtime_errors(runtime: Any, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(runtime, Mapping):
        return ['runtime_identity is missing']
    expected = manifest.get('runtime_identity') or manifest.get('source_identity', {})
    if not isinstance(expected, Mapping):
        expected = {}
    if 'recorded_epoch' in expected and runtime.get('recorded_epoch') != expected['recorded_epoch']:
        errors.append('runtime recorded_epoch disagrees with manifest')
    if runtime.get('source_commit') != expected.get('source_commit'):
        errors.append('runtime source_commit disagrees with manifest')
    wanted_dino = expected.get('dinov2_source_commit', expected.get('dinov2_commit'))
    if runtime.get('dinov2_source_commit') != wanted_dino:
        errors.append('runtime dinov2_source_commit disagrees with manifest')
    if runtime.get('dtype') != expected.get('dtype', 'float32'):
        errors.append('runtime dtype disagrees with manifest')
    if runtime.get('decoder', 'missing') is not None:
        errors.append('runtime decoder is not None')
    if runtime.get('execution') not in ('emulation_only', 'weight_only_numerical_emulation'):
        errors.append('runtime execution is not emulation_only')
    if runtime.get('native_memory_claim') is not False:
        errors.append('runtime native_memory_claim must be false')
    gpu = str(runtime.get('gpu', ''))
    if 'v100' not in gpu.casefold():
        errors.append('runtime GPU is not a V100')
    return errors

def _manifest_audit(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get('schema') != 'frt-ccds-manifest-v1':
        errors.append('manifest schema mismatch')
    if manifest.get('protocol_id') != 'frt-ccds-v1':
        errors.append('protocol_id mismatch')
    scope = manifest.get('research_scope')
    if not isinstance(scope, Mapping) or scope.get('test_split') is not False or scope.get('stage_c') is not False:
        errors.append('manifest must declare no TEST and no Stage C')
    source = manifest.get('source_identity')
    if not isinstance(source, Mapping):
        errors.append('source_identity is missing')
    else:
        if source.get('recorded_epoch') != 65:
            errors.append('source epoch is not 65')
        if source.get('dtype') != 'float32' or source.get('decoder') is not None:
            errors.append('source dtype/decoder mismatch')
        if source.get('native_memory_claim') is not False:
            errors.append('manifest enables native memory claim')
        if source.get('dinov2_source_commit') != '7764ea0f912e53c92e82eb78a2a1631e92725fc8':
            errors.append('manifest DINOv2 commit mismatch')
    quantizer = manifest.get('quantizer')
    expected_blocks = [f'predictor.transformer.layers.{index}' for index in range(6)]
    if not isinstance(quantizer, Mapping) or quantizer.get('bits') != 4 or quantizer.get('q_min') != -7 or (quantizer.get('q_max') != 7) or (quantizer.get('scheme') != 'symmetric_per_output_channel_weight_only') or (list(quantizer.get('target_blocks', [])) != expected_blocks):
        errors.append('quantizer target/W4 contract mismatch')
    splits = manifest.get('splits')
    if not isinstance(splits, Mapping):
        errors.append('splits are missing')
    else:
        if splits.get('historical_reserved_minimum') != [0, 49]:
            errors.append('historical reserve is not 0..49')
        if splits.get('safety_reserved') != [50, 59]:
            errors.append('safety reserve is not 50..59')
        if splits.get('frt_reserved') != [60, 71]:
            errors.append('FRT reserve is not 60..71')
        for split_name, expected_indices in (('cal', EXPECTED_TARGETS['frt_cal']), ('dev', EXPECTED_TARGETS['frt_dev'])):
            node = splits.get(split_name)
            if not isinstance(node, Mapping) or tuple(node.get('dataset_indices', [])) != expected_indices:
                errors.append(f'{split_name} dataset index contract mismatch')
            if not isinstance(node, Mapping) or tuple(node.get('episode_ids', [])) != EXPECTED_EPISODES[node.get('split_id', '') if isinstance(node, Mapping) else 'frt_cal']:
                if isinstance(node, Mapping):
                    errors.append(f'{split_name} episode ID contract mismatch')
    fit = manifest.get('fit')
    if not isinstance(fit, Mapping):
        errors.append('fit section is missing')
    else:
        if tuple(fit.get('methods', [])) != METHODS:
            errors.append('method order mismatch')
        if tuple(fit.get('seeds', [])) != SEEDS:
            errors.append('fit seed order mismatch')
        updates = fit.get('fit_updates')
        if updates is not None and (not isinstance(updates, int) or updates <= 0):
            errors.append('fit_updates must be null or a positive integer')
        policy = fit.get('fit_update_policy', {})
        if not isinstance(policy, Mapping) or tuple(policy.get('candidate_values', [])) != (256, 512, 1000):
            errors.append('fit update candidate policy mismatch')
        batch = fit.get('batch_size_policy', {})
        if not isinstance(batch, Mapping) or tuple(batch.get('candidate_values', [])) != (1, 2, 4):
            errors.append('batch-size candidate policy mismatch')
    loss = manifest.get('loss')
    wz = loss.get('wz') if isinstance(loss, Mapping) else None
    if not isinstance(wz, Mapping) or float(wz.get('std_floor', -1)) != WZ_STD_FLOOR:
        errors.append('Wz std floor mismatch')
    lambda_rule = loss.get('lambda_rule') if isinstance(loss, Mapping) else None
    if not isinstance(lambda_rule, Mapping) or tuple(lambda_rule.get('clip', [])) != (0.25, 4.0):
        errors.append('lambda CAL clipping rule is missing or changed')
    random_control = manifest.get('random_control')
    if not isinstance(random_control, Mapping) or int(random_control.get('seed_base', -1)) != 940000:
        errors.append('random control seed base mismatch')
    gates = manifest.get('gates')
    if not isinstance(gates, Mapping):
        errors.append('gates are missing')
    else:
        clean = gates.get('clean_tolerance', {})
        transport = gates.get('transport_improvement', {})
        if not isinstance(clean, Mapping) or clean.get('per_seed_episode_requirement') != 4 or clean.get('seed_requirement') != 'at_least_2_of_3':
            errors.append('clean gate policy mismatch')
        if not isinstance(transport, Mapping) or tuple(transport.get('views', [])) != ('q0', 'fresh_union') or tuple(transport.get('comparators', [])) != ('clean', 'random_same_norm'):
            errors.append('transport gate policy mismatch')
    return {'pass': not errors, 'errors': _compact_errors(errors)}

def _record_audit(meta: Mapping[str, Any], arrays: Mapping[str, Any], split: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    expected_indices = EXPECTED_TARGETS[split]
    expected_episodes = EXPECTED_EPISODES[split]
    for key in ('record_keys', 'episode_ids', 'dataset_indices', 'env_seeds', 'cem_seeds'):
        if key not in arrays:
            errors.append(f'missing {key}')
    if errors:
        return (errors, {'record_count': 0})
    keys = _strings(arrays['record_keys'])
    episodes = _strings(arrays['episode_ids'])
    indices = np.asarray(arrays['dataset_indices']).reshape(-1)
    env_seeds = np.asarray(arrays['env_seeds']).reshape(-1)
    cem_seeds = np.asarray(arrays['cem_seeds']).reshape(-1)
    count = len(keys)
    if len(set(keys)) != count:
        errors.append('record_keys are duplicated')
    if not len(episodes) == len(indices) == len(env_seeds) == len(cem_seeds) == count:
        errors.append('record metadata lengths disagree')
        return (errors, {'record_count': count})
    if set(episodes) - set(expected_episodes):
        errors.append('unexpected episode ID in record arrays')
    if set(episodes) != set(expected_episodes):
        errors.append('record arrays do not cover exactly six frozen episodes')
    if set(indices.tolist()) - set(expected_indices):
        errors.append('historical or out-of-range dataset index in record arrays')
    episode_to_index: dict[str, int] = {}
    for episode, value in zip(episodes, indices.tolist()):
        current = int(value)
        prior = episode_to_index.setdefault(episode, current)
        if prior != current:
            errors.append(f'episode {episode} maps to multiple dataset indices')
    episode_counts: dict[str, int] = {}
    for episode in episodes:
        episode_counts[episode] = episode_counts.get(episode, 0) + 1
    if any((episode_counts.get(episode, 0) != 2 for episode in expected_episodes)):
        errors.append('each frozen episode must contribute exactly two nested records')
    for row, episode in enumerate(episodes):
        match = re.search(':([0-9]{3})$', episode)
        local = int(match.group(1)) if match else -1
        if local < 0 or local >= 6:
            errors.append(f'invalid local index in {episode}')
            continue
        if int(indices[row]) != expected_indices[local]:
            errors.append(f'{episode} dataset index mismatch')
        expected_env = 800000 if split == 'frt_cal' else 900000
        expected_cem = 810000 if split == 'frt_cal' else 910000
        if int(env_seeds[row]) != expected_env + local or int(cem_seeds[row]) != expected_cem + local:
            errors.append(f'{episode} seed metadata mismatch')
    records = meta.get('records')
    if records is None and isinstance(meta.get('record_keys'), Sequence) and (not isinstance(meta.get('record_keys'), (str, bytes))):
        records = meta.get('record_keys')
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        errors.append('JSON record metadata is missing')
    elif len(records) != count:
        errors.append('JSON and NPZ record counts disagree')
    return (errors, {'record_count': count, 'episode_count': len(set(episodes)), 'episodes': sorted(set(episodes)), 'dataset_indices': sorted(set((int(item) for item in indices.tolist())))})

def _weighted_mse(error: np.ndarray, wz: np.ndarray, active: np.ndarray) -> np.ndarray:
    weighted = np.where(active, error * wz, 0.0)
    denominator = max(int(np.asarray(active, dtype=bool).size), 1)
    return np.sum(weighted * weighted, axis=(-2, -1), dtype=np.float64) / denominator

def _vector_close(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=VECTOR_RTOL, atol=VECTOR_ATOL, equal_nan=False))

def _scalar_json(value: Any) -> Any:
    """Extract a scalar string from an NPZ metadata field without pickle."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError('metadata_json must be a scalar NPZ field')
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode('utf-8')
    return json.loads(str(item))

def _runner_bank_meta(arrays: Any) -> Mapping[str, Any]:
    if 'metadata_json' not in arrays:
        raise ValueError('runner bank is missing metadata_json')
    value = _scalar_json(arrays['metadata_json'])
    if not isinstance(value, Mapping):
        raise ValueError('runner bank metadata_json must be an object')
    return value

def _runner_record_audit(meta: Mapping[str, Any], arrays: Any, expected_split: str) -> tuple[list[str], dict[str, Any]]:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _runner_wz_view(wz: np.ndarray, slot_shape: tuple[int, int], errors: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _runner_stage_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _stage_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _fit_run_audit(meta: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    runs = meta.get('fit_runs')
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or len(runs) != 9:
        return (['fit_runs must contain exactly nine method-seed rows'], {})
    expected_updates = None
    fit = manifest.get('fit')
    if isinstance(fit, Mapping) and isinstance(fit.get('fit_updates'), int):
        expected_updates = int(fit['fit_updates'])
    configs: dict[str, tuple[Any, ...]] = {}
    seen: set[tuple[str, int]] = set()
    for row in runs:
        if not isinstance(row, Mapping):
            errors.append('fit_runs contains a non-object')
            continue
        method = str(row.get('method'))
        try:
            seed = int(row.get('seed'))
        except Exception:
            seed = -1
        key = (method, seed)
        seen.add(key)
        if method not in METHODS or seed not in SEEDS:
            errors.append(f'unexpected fit run {method}:{seed}')
        if row.get('status') != 'complete':
            errors.append(f'{method}:{seed} is not complete')
        try:
            completed = int(row.get('fit_updates_completed'))
            declared = int(row.get('fit_updates_expected'))
        except Exception:
            completed = declared = -1
            errors.append(f'{method}:{seed} lacks numeric fit update ledger')
        if declared <= 0 or completed != declared:
            errors.append(f'{method}:{seed} did not complete all fit updates')
        if expected_updates is not None and declared != expected_updates:
            errors.append(f'{method}:{seed} fit update count disagrees with manifest')
        for flag in ('binary_final', 'alpha_discarded', 'reload_equal', 'exact_w4'):
            if row.get(flag) is not True:
                errors.append(f'{method}:{seed} {flag} is not true')
        for key_name in ('checkpoint_sha256', 'hard_map_sha256', 'batch_schedule_hash', 'initialization_id', 'soft_schedule_id'):
            if not str(row.get(key_name, '')):
                errors.append(f'{method}:{seed} missing {key_name}')
        if row.get('checkpoint_sha256') and (not re.fullmatch('[0-9a-f]{64}', str(row.get('checkpoint_sha256')))):
            errors.append(f'{method}:{seed} checkpoint hash malformed')
        if row.get('hard_map_sha256') and (not re.fullmatch('[0-9a-f]{64}', str(row.get('hard_map_sha256')))):
            errors.append(f'{method}:{seed} hard map hash malformed')
        try:
            batch_size = int(row.get('batch_size'))
            exposure_count = int(row.get('exposure_count'))
        except Exception:
            batch_size = exposure_count = -1
            errors.append(f'{method}:{seed} lacks batch/exposure ledger')
        if batch_size not in (1, 2, 4):
            errors.append(f'{method}:{seed} batch size is outside A candidates')
        if exposure_count <= 0:
            errors.append(f'{method}:{seed} exposure count is not positive')
        configs.setdefault(str(seed), tuple((row.get(key) for key in ('batch_size', 'batch_schedule_hash', 'exposure_count', 'scale_learning_rate', 'rounding_learning_rate', 'initialization_id', 'soft_schedule_id', 'fit_updates_expected'))))
        if configs[str(seed)] != tuple((row.get(key) for key in ('batch_size', 'batch_schedule_hash', 'exposure_count', 'scale_learning_rate', 'rounding_learning_rate', 'initialization_id', 'soft_schedule_id', 'fit_updates_expected'))):
            errors.append(f'fit schedule/optimizer differs across methods for seed {seed}')
    expected_keys = {(method, seed) for method in METHODS for seed in SEEDS}
    if seen != expected_keys:
        errors.append('fit_runs do not cover exactly methods x seeds')
    fit_config = meta.get('fit_config')
    if not isinstance(fit_config, Mapping):
        errors.append('fit_config is missing')
    else:
        for key in ('fit_updates', 'batch_size', 'scale_learning_rate', 'rounding_learning_rate', 'schedule_hash'):
            if key not in fit_config:
                errors.append(f'fit_config.{key} is missing')
        if expected_updates is not None and fit_config.get('fit_updates') != expected_updates:
            errors.append('fit_config.fit_updates disagrees with manifest')
        for seed in SEEDS:
            row = configs.get(str(seed))
            if row is not None and fit_config.get('batch_size') != row[0]:
                errors.append(f'fit_config batch size disagrees for seed {seed}')
    return (errors, {'fit_run_count': len(runs), 'fit_config': dict(fit_config) if isinstance(fit_config, Mapping) else None})

def _runner_fit_audit(artifact: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _runner_bank_array_audit(arrays: Any, expected_split: str, errors: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _runner_stage_b_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str, stage_a: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _stage_b_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str, stage_a: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    resource = False
    if not ((artifact / 'stage_b.json').is_file() and (artifact / 'stage_b.npz').is_file()) and ((artifact / 'stage_b_summary.json').is_file() or (artifact / 'bank_dev.npz').is_file()):
        return _runner_stage_b_audit(artifact, manifest, manifest_hash, stage_a)
    meta_path = artifact / 'stage_b.json'
    arrays_path = artifact / 'stage_b.npz'
    if not meta_path.is_file() or not arrays_path.is_file():
        return {'pass': False, 'complete': False, 'engineering_pass': False, 'resource': True, 'errors': ['stage_b.json or stage_b.npz is missing']}
    meta = _json(meta_path)
    if meta.get('schema') != 'frt-stage-b-raw-v1':
        errors.append('Stage B schema mismatch')
    if meta.get('manifest_sha256') != manifest_hash:
        errors.append('Stage B manifest_sha256 does not match the active manifest revision')
    errors.extend(_runtime_errors(meta.get('runtime_identity'), manifest))
    if meta.get('completion_status') != 'complete' or meta.get('complete') is not True:
        resource = True
        errors.append('Stage B completion_status is incomplete')
    fit_errors, fit_summary = _fit_run_audit(meta, manifest)
    errors.extend(fit_errors)
    if any(('not complete' in item or 'did not complete' in item or 'missing fit' in item for item in fit_errors)):
        resource = True
    bank_ids = tuple((str(item) for item in meta.get('bank_ids', [])))
    evaluator_ids = tuple((str(item) for item in meta.get('evaluator_ids', [])))
    if bank_ids != BANKS:
        errors.append('bank_ids do not match the frozen common-bank order')
    if evaluator_ids != EVALUATORS:
        errors.append('evaluator_ids do not match the frozen all-evaluator order')
    if not meta.get('common_bank_all_evaluators') is True:
        errors.append('common_bank_all_evaluators is not true')
    if stage_a.get('pass') and meta.get('stage_a_wz_sha256') and (meta.get('stage_a_wz_sha256') != stage_a.get('wz_sha256')):
        errors.append('Stage B Wz identity disagrees with Stage A')
    record_summary: dict[str, Any] = {}
    metric_summary: dict[str, Any] = {}
    try:
        with np.load(arrays_path, allow_pickle=False) as arrays:
            required = ('record_keys', 'episode_ids', 'dataset_indices', 'active_mask', 'action_mask', 'wz', 'bank_ids', 'evaluator_ids', 'bank_deltas', 'clean_error', 'transport_error', 'clean_mse', 'transport_mse')
            missing = [key for key in required if key not in arrays]
            if missing:
                errors.extend((f'missing Stage B array {key}' for key in missing))
                return {'pass': False, 'complete': False, 'engineering_pass': False, 'mechanism_pass': False, 'resource': resource, 'errors': _compact_errors(errors), **fit_summary}
            record_errors, record_summary = _record_audit(meta, arrays, 'frt_dev')
            errors.extend(record_errors)
            record_keys = _strings(arrays['record_keys'])
            episodes = _strings(arrays['episode_ids'])
            n = len(record_keys)
            active = np.asarray(arrays['active_mask'], dtype=bool)
            action = np.asarray(arrays['action_mask'], dtype=bool)
            wz = _finite('wz', arrays['wz'], errors)
            array_bank_ids = tuple(_strings(arrays['bank_ids']))
            array_evaluator_ids = tuple(_strings(arrays['evaluator_ids']))
            if array_bank_ids != BANKS:
                errors.append('Stage B array bank_ids do not match the frozen common-bank order')
            if array_evaluator_ids != EVALUATORS:
                errors.append('Stage B array evaluator_ids do not match the frozen evaluator order')
            bank_deltas = _finite('bank_deltas', arrays['bank_deltas'], errors)
            clean_error = _finite('clean_error', arrays['clean_error'], errors)
            transport_error = _finite('transport_error', arrays['transport_error'], errors)
            clean_mse = _finite('clean_mse', arrays['clean_mse'], errors)
            transport_mse = _finite('transport_mse', arrays['transport_mse'], errors)
            if bank_deltas.ndim != 4 or bank_deltas.shape[0] != len(BANKS) or bank_deltas.shape[1] != n:
                errors.append('bank_deltas shape is not [11,N,P,D]')
            if clean_error.ndim != 4 or clean_error.shape[0] != len(EVALUATORS) or clean_error.shape[1] != n:
                errors.append('clean_error shape is not [11,N,P,D]')
            if transport_error.ndim != 5 or transport_error.shape[:3] != (len(EVALUATORS), len(BANKS), n):
                errors.append('transport_error shape is not [11,11,N,P,D]')
            if clean_mse.shape != (len(EVALUATORS), n):
                errors.append('clean_mse shape is not [11,N]')
            if transport_mse.shape != (len(EVALUATORS), len(BANKS), n):
                errors.append('transport_mse shape is not [11,11,N]')
            if len(record_keys) != len(set(record_keys)):
                errors.append('DEV record_keys are duplicated')
            if active.shape != action.shape or wz.shape != active.shape:
                errors.append('DEV masks/Wz shape mismatch')
            elif np.any(active & action) or np.any(~(active | action)):
                errors.append('DEV active/action masks are not disjoint and covering')
            if active.shape == wz.shape:
                if np.any(wz[action] != 0) or np.any(wz[active] <= 0):
                    errors.append('DEV Wz action/active values violate contract')
            stage_a_wz_hash = meta.get('stage_a_wz_sha256')
            actual_wz_hash = None
            if wz.ndim == 1:
                actual_wz_hash = hashlib.sha256(np.asarray(wz).reshape(-1).tobytes()).hexdigest()
            elif wz.ndim == 2 and wz.shape[0] > 0:
                if not np.array_equal(wz, np.broadcast_to(wz[0:1], wz.shape)):
                    errors.append('DEV Wz rows disagree; cannot establish Stage A vector identity')
                actual_wz_hash = hashlib.sha256(np.asarray(wz[0]).reshape(-1).tobytes()).hexdigest()
            else:
                errors.append('DEV Wz has no usable Stage A identity vector')
            if not isinstance(stage_a_wz_hash, str) or not re.fullmatch('[0-9a-f]{64}', stage_a_wz_hash):
                errors.append('Stage B stage_a_wz_sha256 is missing or malformed')
            elif actual_wz_hash is not None and actual_wz_hash != stage_a_wz_hash:
                errors.append('Stage B Wz array disagrees with stage_a_wz_sha256')
            if stage_a.get('pass') and stage_a.get('wz_sha256') and (actual_wz_hash is not None) and (actual_wz_hash != stage_a.get('wz_sha256')):
                errors.append('Stage B Wz array disagrees with independently audited Stage A Wz')
            if bank_deltas.ndim == 4 and active.shape == bank_deltas.shape[2:]:
                if np.any(bank_deltas * action.reshape((1, 1) + action.shape)):
                    errors.append('a DEV bank has nonzero action coordinates')
                q0_norm = np.sqrt(np.sum(np.where(active, bank_deltas[Q0_BANK] * wz, 0.0) ** 2, axis=(-2, -1)))
                random_norm = np.sqrt(np.sum(np.where(active, bank_deltas[RANDOM_BANK] * wz, 0.0) ** 2, axis=(-2, -1)))
                if np.any(q0_norm <= EPS) or not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append('DEV random bank does not match shared q0 norm or has zero q0 norm')
            if clean_error.ndim == 4 and clean_error.shape[0] == len(EVALUATORS) and (clean_error.shape[1] == n) and (wz.shape == clean_error.shape[2:]):
                recomputed_clean = _weighted_mse(clean_error, wz, active)
                if not np.allclose(clean_mse, recomputed_clean, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append('clean_mse does not match clean_error vectors')
            if transport_error.ndim == 5 and transport_error.shape[:3] == (len(EVALUATORS), len(BANKS), n) and (wz.shape == transport_error.shape[3:]):
                recomputed_transport = _weighted_mse(transport_error, wz, active)
                if not np.allclose(transport_mse, recomputed_transport, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append('transport_mse does not match transport_error vectors')
            if clean_mse.shape == (len(EVALUATORS), n) and transport_mse.shape == (len(EVALUATORS), len(BANKS), n):
                if not np.allclose(clean_mse[0], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                    errors.append('FP32 clean error is not zero')
                if not np.allclose(transport_mse[0], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                    errors.append('FP32 transport error is not zero')
                for evaluator in FIT_EVALUATOR_INDEX:
                    row = FIT_EVALUATOR_INDEX[evaluator]
                    if not np.isfinite(clean_mse[row]).all() or not np.isfinite(transport_mse[row]).all():
                        errors.append(f'{evaluator} has non-finite metrics')
                cosine_summary: dict[str, float] = {}
                q0 = bank_deltas[Q0_BANK]
                q0w = np.where(active, q0 * wz, 0.0)
                q0norm = np.sqrt(np.sum(q0w * q0w, axis=(-2, -1)))
                for bank_index, bank_id in zip(FRESH_BANKS, BANKS[2:]):
                    freshw = np.where(active, bank_deltas[bank_index] * wz, 0.0)
                    denom = q0norm * np.sqrt(np.sum(freshw * freshw, axis=(-2, -1)))
                    cos = np.sum(q0w * freshw, axis=(-2, -1)) / np.maximum(denom, EPS)
                    if not np.isfinite(cos).all():
                        errors.append(f'{bank_id} direction cosine is non-finite')
                    cosine_summary[bank_id] = float(np.mean(cos, dtype=np.float64))
                metric_summary['fresh_direction_cosine_mean'] = cosine_summary
                metric_summary['clean_mse_macro'] = {method: float(np.mean([np.mean(clean_mse[FIT_EVALUATOR_INDEX[f'{method}:{seed}']]) for seed in SEEDS])) for method in METHODS}
                gate_metrics = _mechanism_gates(clean_mse, transport_mse, episodes)
                metric_summary.update(gate_metrics)
    except Exception as exc:
        errors.append(f'Stage B array audit exception: {type(exc).__name__}: {exc}')
    engineering_pass = not errors and (not resource)
    mechanism_pass = bool(engineering_pass and metric_summary.get('all_gates_pass', False))
    if resource:
        status = 'resource_incomplete'
    elif not engineering_pass:
        status = 'engineering_fail'
    elif metric_summary.get('ambiguous'):
        status = 'ambiguous'
    elif mechanism_pass:
        status = 'conditional_signal'
    else:
        status = 'mechanism_no_go'
    return {'pass': engineering_pass and mechanism_pass, 'complete': bool(meta.get('complete') is True and (not resource)), 'engineering_pass': engineering_pass, 'mechanism_pass': mechanism_pass, 'resource': resource, 'status': status, 'errors': _compact_errors(errors), **record_summary, **fit_summary, 'metrics': metric_summary}

def _episode_seed_values(metric: np.ndarray, episodes: Sequence[str], evaluator: str, bank_indices: Sequence[int] | None) -> dict[int, np.ndarray]:
    row = FIT_EVALUATOR_INDEX[evaluator]
    result: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        method = evaluator.split(':', 1)[0]
        row = FIT_EVALUATOR_INDEX[f'{method}:{seed}']
        values = []
        for episode in EXPECTED_EPISODES['frt_dev']:
            mask = np.asarray([item == episode for item in episodes], dtype=bool)
            if bank_indices is None:
                values.append(float(np.mean(metric[row, mask], dtype=np.float64)))
            else:
                values.append(float(np.mean(metric[row, list(bank_indices)][:, mask], dtype=np.float64)))
        result[seed] = np.asarray(values, dtype=np.float64)
    return result

def _gate(values: dict[int, np.ndarray], comparator: dict[int, np.ndarray], ratio: float, name: str) -> dict[str, Any]:
    seed_rows: dict[str, Any] = {}
    macro_value = float(np.mean([np.mean(row, dtype=np.float64) for row in values.values()]))
    macro_comparator = float(np.mean([np.mean(row, dtype=np.float64) for row in comparator.values()]))
    macro_threshold = ratio * macro_comparator + EPS
    seed_passes = 0
    ambiguous = False
    for seed in SEEDS:
        left = values[seed]
        right = comparator[seed]
        threshold = ratio * right + EPS
        delta = left - threshold
        passed = delta <= 0.0
        near = np.abs(delta) <= REL_TOL * np.maximum(np.abs(threshold), EPS)
        if bool(np.any(near)):
            ambiguous = True
        seed_rows[str(seed)] = {'episode_pass_count': int(np.sum(passed)), 'required_episode_pass_count': 4, 'macro_value': float(np.mean(left)), 'macro_comparator': float(np.mean(right)), 'pass': bool(np.sum(passed) >= 4 and np.mean(left) <= np.mean(right) * ratio + EPS), 'ambiguous': bool(np.any(near))}
        if seed_rows[str(seed)]['pass']:
            seed_passes += 1
    macro_pass = macro_value <= macro_threshold
    if abs(macro_value - macro_threshold) <= REL_TOL * max(abs(macro_threshold), EPS):
        ambiguous = True
    return {'name': name, 'ratio': ratio, 'macro_value': macro_value, 'macro_comparator': macro_comparator, 'macro_pass': bool(macro_pass), 'seed_pass_count': seed_passes, 'required_seed_pass_count': 2, 'seed_pass': seed_passes >= 2, 'pass': bool(macro_pass and seed_passes >= 2 and (not ambiguous)), 'ambiguous': ambiguous, 'per_seed': seed_rows}

def _mechanism_gates(clean_mse: np.ndarray, transport_mse: np.ndarray, episodes: Sequence[str]) -> dict[str, Any]:
    clean: dict[str, dict[int, np.ndarray]] = {}
    transport_q0: dict[str, dict[int, np.ndarray]] = {}
    transport_fresh: dict[str, dict[int, np.ndarray]] = {}
    for method in METHODS:
        evaluator = f'{method}:1201'
        clean[method] = _episode_seed_values(clean_mse, episodes, evaluator, None)
        transport_q0[method] = _episode_seed_values(transport_mse, episodes, evaluator, (Q0_BANK,))
        transport_fresh[method] = _episode_seed_values(transport_mse, episodes, evaluator, FRESH_BANKS)
    gates = [_gate(clean['frt'], clean['clean'], 1.1, 'clean_tolerance_frt_vs_clean'), _gate(transport_q0['frt'], transport_q0['clean'], 0.95, 'transport_q0_frt_vs_clean'), _gate(transport_q0['frt'], transport_q0['random_same_norm'], 0.95, 'transport_q0_frt_vs_random'), _gate(transport_fresh['frt'], transport_fresh['clean'], 0.95, 'transport_fresh_union_frt_vs_clean'), _gate(transport_fresh['frt'], transport_fresh['random_same_norm'], 0.95, 'transport_fresh_union_frt_vs_random')]
    return {'gates': gates, 'all_gates_pass': bool(all((bool(item['pass']) for item in gates))), 'ambiguous': bool(any((bool(item['ambiguous']) for item in gates)))}

def _self_test() -> None:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('a', 'b', 'all'), default='all')
    parser.add_argument('--artifact-dir', '--run-dir', dest='artifact_dir', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.artifact_dir is None or args.manifest is None or args.output is None:
        parser.error('--artifact-dir, --manifest and --output are required')
    return args

def main() -> None:
    raise RuntimeError('B helper snapshot requires completed canonical stage_b artifacts')
if __name__ == '__main__':
    main()

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
    if set(indices.tolist()) - set(expected_indices):
        errors.append('historical or out-of-range dataset index in record arrays')
    episode_to_index: dict[str, int] = {}
    for episode, value in zip(episodes, indices.tolist()):
        current = int(value)
        prior = episode_to_index.setdefault(episode, current)
        if prior != current:
            errors.append(f'episode {episode} maps to multiple dataset indices')
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
    records = meta.get('records', [])
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        errors.append('JSON records list is missing')
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
    """Audit the actual frt_runner bank contract (including repeated rows)."""
    errors: list[str] = []
    try:
        bank_meta = _runner_bank_meta(arrays)
    except Exception as exc:
        return ([f'bank metadata parse failed: {type(exc).__name__}: {exc}'], {})
    split = str(bank_meta.get('split', ''))
    if split not in {expected_split, 'stage_a'}:
        errors.append(f'bank split is {split!r}, expected {expected_split!r}')
    rows = bank_meta.get('metadata')
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        errors.append('bank metadata list is missing')
        return (errors, {})
    episodes: list[str] = []
    indices: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append('bank metadata contains a non-object row')
            continue
        episode = str(row.get('episode_id', ''))
        if not episode:
            errors.append('bank metadata row has no episode_id')
            continue
        try:
            index = int(row['dataset_index'])
        except Exception:
            errors.append(f'{episode} has no numeric dataset_index')
            continue
        episodes.append(episode)
        indices.append(index)
    if not episodes:
        return (errors + ['bank metadata has no records'], {'record_count': 0})
    expected_indices = set(EXPECTED_TARGETS['frt_cal' if expected_split == 'stage_a' else 'frt_dev'])
    expected_episode_prefix = 'frt_cal:' if expected_split == 'stage_a' else 'frt_dev:'
    if set(indices) != expected_indices:
        errors.append(f'dataset index set {sorted(set(indices))} != {sorted(expected_indices)}')
    if any((not episode.startswith(expected_episode_prefix) for episode in episodes)):
        errors.append('bank contains an unexpected episode namespace')
    episode_to_index: dict[str, int] = {}
    for episode, index in zip(episodes, indices):
        prior = episode_to_index.setdefault(episode, index)
        if prior != index:
            errors.append(f'episode {episode} maps to multiple dataset indices')
    if set(episode_to_index) != set(EXPECTED_EPISODES['frt_cal' if expected_split == 'stage_a' else 'frt_dev']):
        errors.append('bank does not cover exactly six frozen episode identities')
    return (errors, {'record_count': len(episodes), 'episode_count': len(episode_to_index), 'record_episode_ids': episodes, 'episodes': sorted(episode_to_index), 'dataset_indices': sorted(set(indices)), 'metadata_split': split})

def _runner_wz_view(wz: np.ndarray, slot_shape: tuple[int, int], errors: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Normalize the runner's [D] Wz or explicit [P,D] Wz+mask contract."""
    if wz.shape == (slot_shape[1],):
        if not np.isfinite(wz).all() or np.any(wz < 0):
            errors.append('runner Wz[D] is not finite and non-negative')
        zero_columns = np.asarray(wz == 0, dtype=bool)
        if np.any(zero_columns):
            expanded = np.broadcast_to(wz.reshape((1, slot_shape[1])), slot_shape).copy()
            return (expanded, np.broadcast_to(zero_columns.reshape((1, slot_shape[1])), slot_shape).copy())
        return (wz.reshape((1, slot_shape[1])), None)
    if wz.shape == slot_shape:
        if not np.isfinite(wz).all() or np.any(wz < 0):
            errors.append('runner Wz[P,D] is not finite and non-negative')
        return (wz, np.asarray(wz == 0, dtype=bool))
    errors.append(f'runner Wz shape {wz.shape} disagrees with slot {slot_shape}')
    return (None, None)

def _runner_stage_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    """Independent audit of frt_runner.py's stage_a_summary + bank_stage_a."""
    errors: list[str] = []
    summary_path = artifact / 'stage_a_summary.json'
    bank_path = artifact / 'bank_stage_a.npz'
    if not summary_path.is_file() or not bank_path.is_file():
        return {'pass': False, 'complete': False, 'resource': True, 'errors': ['stage_a_summary.json or bank_stage_a.npz is missing'], 'record_count': 0}
    summary = _json(summary_path)
    if summary.get('schema') != 'frt-stage-a-v1':
        errors.append('runner Stage A schema mismatch')
    if summary.get('status') != 'complete':
        errors.append('runner Stage A status is incomplete')
    errors.extend(_runtime_errors(summary.get('runtime_identity'), manifest))
    gates = summary.get('gates')
    if not isinstance(gates, Mapping):
        errors.append('runner Stage A gates are missing')
    else:
        required_gates = ('source_rollout_agreement', 'physical_replay', 'delta_zero', 'fp_transport_null', 'history_action_invariants', 'hard_materialize_reload')
        for name in required_gates:
            gate = gates.get(name)
            if not isinstance(gate, Mapping) or gate.get('passed') is not True:
                errors.append(f'Stage A gate {name} did not pass')
    record_summary: dict[str, Any] = {}
    try:
        with np.load(bank_path, allow_pickle=False) as arrays:
            required = ('history', 'x_history', 'action', 'next_action', 'fp_current', 'q0_current', 'q0_transport', 'delta', 'fp_transport', 'random_delta', 'random_transport', 'wz', 'action_mask', 'active_mask', 'metadata_json')
            missing = [key for key in required if key not in arrays]
            if missing:
                errors.extend((f'missing runner Stage A bank array {key}' for key in missing))
                return {'pass': False, 'complete': False, 'resource': False, 'errors': _compact_errors(errors), 'record_count': 0}
            meta_errors, record_summary = _runner_record_audit(arrays['metadata_json'], arrays, 'stage_a')
            errors.extend(meta_errors)
            history = _finite('history', arrays['history'], errors)
            x_history = _finite('x_history', arrays['x_history'], errors)
            action = _finite('action', arrays['action'], errors)
            next_action = _finite('next_action', arrays['next_action'], errors)
            fp_current = _finite('fp_current', arrays['fp_current'], errors)
            q0_current = _finite('q0_current', arrays['q0_current'], errors)
            q0_transport = _finite('q0_transport', arrays['q0_transport'], errors)
            delta = _finite('delta', arrays['delta'], errors)
            fp_transport = _finite('fp_transport', arrays['fp_transport'], errors)
            random_delta = _finite('random_delta', arrays['random_delta'], errors)
            random_transport = _finite('random_transport', arrays['random_transport'], errors)
            wz_raw = _finite('wz', arrays['wz'], errors)
            if history.ndim != 4 or x_history.shape != history.shape:
                errors.append('history/x_history are not matching [N,T,P,D]')
            if fp_current.ndim != 4:
                errors.append('fp_current is not [N,1,P,D]')
            slot_shape = tuple(fp_current.shape[-2:]) if fp_current.ndim == 4 else ()
            if slot_shape and any((array.shape != fp_current.shape for array in (q0_current, q0_transport, delta, fp_transport, random_delta, random_transport))):
                errors.append('one or more current/delta/transport arrays disagree with fp_current')
            if history.ndim == 4 and fp_current.ndim == 4 and (history.shape[0] != fp_current.shape[0]):
                errors.append('history and fp_current record counts disagree')
            if action.ndim != 3 or next_action.shape != action.shape or action.shape[0] != fp_current.shape[0]:
                errors.append('action/next_action are not matching [N,1,A]')
            wz, inferred_action_mask = _runner_wz_view(wz_raw, slot_shape, errors) if slot_shape else (None, None)
            if wz is not None and delta.ndim == 4:
                if not _vector_close(q0_current - fp_current, delta):
                    errors.append('q0_current-fp_current does not equal delta')
                if history.ndim == 4 and x_history.shape == history.shape:
                    if history.shape[1] == 1:
                        if not _vector_close(x_history[:, -1], fp_current[:, -1]):
                            errors.append('x_history does not equal appended fp_current for num_hist=1')
                    elif not _vector_close(x_history[:, :-1], history[:, 1:]) or not _vector_close(x_history[:, -1], fp_current[:, -1]):
                        errors.append('x_history is not the shifted history with fp_current appended')
                wz_broadcast = wz.reshape((1, 1, 1, slot_shape[1])) if wz.shape == (1, slot_shape[1]) else wz.reshape((1, 1, slot_shape[0], slot_shape[1]))
                q0_norm = np.sqrt(np.sum((delta * wz_broadcast) ** 2, axis=(-2, -1)))
                random_norm = np.sqrt(np.sum((random_delta * wz_broadcast) ** 2, axis=(-2, -1)))
                if not np.isfinite(q0_norm).all() or not np.isfinite(random_norm).all():
                    errors.append('Q0/random weighted norms are non-finite')
                elif np.any(q0_norm <= EPS):
                    errors.append('one or more Q0 weighted residual norms are zero')
                elif not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append('random bank does not match Q0 weighted norm')
                action_mask = np.asarray(arrays['action_mask'], dtype=bool)
                active_mask = np.asarray(arrays['active_mask'], dtype=bool)
                if action_mask.shape != slot_shape or active_mask.shape != slot_shape:
                    errors.append('runner action_mask/active_mask shape disagrees with latent slot')
                elif np.any(action_mask & active_mask) or np.any(~(action_mask | active_mask)):
                    errors.append('runner action/active masks are not disjoint and covering')
                elif np.any(wz[action_mask] != 0) or np.any(wz[active_mask] <= 0):
                    errors.append('runner Wz violates action-zero/active-positive contract')
                elif inferred_action_mask is not None and (not np.array_equal(action_mask, inferred_action_mask)):
                    errors.append('runner action_mask disagrees with Wz zero coordinates')
                if np.any(delta[..., action_mask] != 0) or np.any(random_delta[..., action_mask] != 0):
                    errors.append('delta has nonzero action coordinates')
                clean_error = q0_current - fp_current
                transport_error = q0_transport - fp_transport
                clean_component = float(np.mean(_weighted_mse(clean_error, wz, active_mask), dtype=np.float64))
                transport_component = float(np.mean(_weighted_mse(transport_error, wz, active_mask), dtype=np.float64))
                lambda_raw = max(clean_component, EPS) / max(transport_component, EPS)
                lambda_expected = float(np.clip(lambda_raw, 0.25, 4.0))
                freeze = summary.get('calibration_freeze')
                if not isinstance(freeze, Mapping):
                    errors.append('Stage A calibration_freeze is missing')
                else:
                    for name, expected_value in (('mean_L_clean_Q0', clean_component), ('mean_L_transport_Q0', transport_component), ('lambda_raw', lambda_raw), ('lambda_T', lambda_expected)):
                        if name not in freeze or not math.isclose(float(freeze[name]), expected_value, rel_tol=VECTOR_RTOL, abs_tol=VECTOR_ATOL):
                            errors.append(f'calibration_freeze.{name} disagrees with raw Q0 vectors')
                    if bool(freeze.get('lambda_clamped')) != bool(lambda_expected != lambda_raw):
                        errors.append('calibration_freeze.lambda_clamped disagrees with raw ratio')
                record_summary.update({'mean_L_clean_Q0': clean_component, 'mean_L_transport_Q0': transport_component, 'lambda_raw': lambda_raw, 'lambda_T': lambda_expected})
            record_summary.update({'history_shape': list(history.shape), 'slot_shape': list(slot_shape), 'wz_shape': list(wz_raw.shape), 'wz_sha256': hashlib.sha256(np.asarray(wz_raw).tobytes()).hexdigest(), 'mean_q0_weighted_norm': float(np.mean(q0_norm)) if 'q0_norm' in locals() else None})
    except Exception as exc:
        errors.append(f'runner Stage A bank audit exception: {type(exc).__name__}: {exc}')
    complete = summary.get('status') == 'complete'
    return {'pass': bool(complete and (not errors)), 'complete': bool(complete and (not errors)), 'resource': False, 'errors': _compact_errors(errors), **record_summary}

def _stage_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    errors: list[str] = []
    if (artifact / 'stage_a_summary.json').is_file() or (artifact / 'bank_stage_a.npz').is_file():
        return _runner_stage_audit(artifact, manifest, manifest_hash)
    meta_path = artifact / 'stage_a.json'
    arrays_path = artifact / 'stage_a.npz'
    if not meta_path.is_file() or not arrays_path.is_file():
        return {'pass': False, 'complete': False, 'resource': True, 'errors': ['stage_a.json or stage_a.npz is missing'], 'record_count': 0}
    meta = _json(meta_path)
    if meta.get('schema') != 'frt-stage-a-raw-v1':
        errors.append('Stage A schema mismatch')
    if meta.get('manifest_sha256') and (not re.fullmatch('[0-9a-f]{64}', str(meta.get('manifest_sha256')))):
        errors.append('Stage A manifest_sha256 is malformed')
    if meta.get('manifest_sha256') not in (None, manifest_hash):
        errors.append('Stage A was produced under an earlier manifest revision')
    errors.extend(_runtime_errors(meta.get('runtime_identity'), manifest))
    fingerprints = meta.get('target_fingerprints', [])
    if isinstance(fingerprints, Mapping):
        fingerprints = list(fingerprints.values())
    if not isinstance(fingerprints, Sequence) or isinstance(fingerprints, (str, bytes)):
        errors.append('Stage A target_fingerprints are missing')
    elif len(fingerprints) != 6 or len(set(map(str, fingerprints))) != 6:
        errors.append('Stage A target fingerprints are not six unique values')
    contract = meta.get('history_contract')
    if not isinstance(contract, Mapping):
        errors.append('history_contract is missing')
    else:
        for key in ('full_history', 'shift_concat_real', 'same_action_rng_replay', 'x_delta_legal', 'action_replaced_before_delta'):
            if contract.get(key) is not True:
                errors.append(f'history_contract.{key} is not true')
    try:
        with np.load(arrays_path, allow_pickle=False) as arrays:
            required = ('record_keys', 'episode_ids', 'dataset_indices', 'env_seeds', 'cem_seeds', 'history_fp', 'history_q0_slot', 'next_fp', 'next_q0', 'delta_q0', 'delta_random', 'active_mask', 'action_mask', 'fp_output_variance', 'wz', 'q0_clean_error', 'q0_transport_error')
            missing = [key for key in required if key not in arrays]
            if missing:
                errors.extend((f'missing Stage A array {key}' for key in missing))
                return {'pass': False, 'complete': False, 'resource': False, 'errors': _compact_errors(errors), 'record_count': 0}
            record_errors, record_summary = _record_audit(meta, arrays, 'frt_cal')
            errors.extend(record_errors)
            history_fp = _finite('history_fp', arrays['history_fp'], errors)
            history_q0 = _finite('history_q0_slot', arrays['history_q0_slot'], errors)
            next_fp = _finite('next_fp', arrays['next_fp'], errors)
            next_q0 = _finite('next_q0', arrays['next_q0'], errors)
            delta_q0 = _finite('delta_q0', arrays['delta_q0'], errors)
            delta_random = _finite('delta_random', arrays['delta_random'], errors)
            variance = _finite('fp_output_variance', arrays['fp_output_variance'], errors)
            wz = _finite('wz', arrays['wz'], errors)
            active = np.asarray(arrays['active_mask'], dtype=bool)
            action = np.asarray(arrays['action_mask'], dtype=bool)
            shape = next_fp.shape[1:] if next_fp.ndim == 3 else ()
            if history_fp.ndim != 4 or history_q0.shape != history_fp.shape:
                errors.append('history arrays are not both [N,T,P,D]')
            if next_fp.ndim != 3 or next_q0.shape != next_fp.shape or delta_q0.shape != next_fp.shape or (delta_random.shape != next_fp.shape):
                errors.append('next/delta arrays are not all [N,P,D]')
            if len(shape) != 2 or active.shape != shape or action.shape != shape or (wz.shape != shape) or (variance.shape != shape):
                errors.append('mask/Wz/variance shape does not match [P,D]')
            if history_fp.ndim == 4 and next_fp.ndim == 3 and (history_fp.shape[0] != next_fp.shape[0]):
                errors.append('history and next record counts disagree')
            if active.shape == action.shape and (np.any(active & action) or np.any(~(active | action))):
                errors.append('active/action masks are not disjoint and covering')
            if active.shape == action.shape and np.any(wz[action] != 0):
                errors.append('Wz action coordinates are not zero')
            if active.shape == action.shape and np.any(wz[active] <= 0):
                errors.append('Wz active coordinates are not positive')
            if delta_q0.shape == next_q0.shape and action.shape == delta_q0.shape[1:]:
                if not np.array_equal(delta_q0[:, action], np.zeros_like(delta_q0[:, action])):
                    errors.append('delta_q0 action coordinates are not exact zero')
                if not np.array_equal(delta_random[:, action], np.zeros_like(delta_random[:, action])):
                    errors.append('delta_random action coordinates are not exact zero')
                if not _vector_close(next_q0 - next_fp, delta_q0):
                    errors.append('next_q0-next_fp does not equal delta_q0')
            if history_fp.ndim == 4 and history_q0.shape == history_fp.shape and (delta_q0.ndim == 3):
                history_diff = history_q0 - history_fp
                if history_fp.shape[1] < 1:
                    errors.append('history has no new slot')
                else:
                    if not np.allclose(history_diff[:, :-1], 0.0, rtol=0.0, atol=VECTOR_ATOL):
                        errors.append('old history slots changed under delta injection')
                    if not _vector_close(history_diff[:, -1], delta_q0):
                        errors.append('delta was not injected into only the newest slot')
            if next_fp.ndim == 3 and variance.shape == next_fp.shape[1:]:
                computed_var = np.var(next_fp.astype(np.float64), axis=0)
                expected_wz = np.zeros_like(computed_var, dtype=np.float64)
                if active.shape == computed_var.shape:
                    expected_wz[active] = 1.0 / np.maximum(np.sqrt(np.maximum(computed_var[active], 0.0)), WZ_STD_FLOOR)
                    if not _vector_close(wz, expected_wz):
                        errors.append('Wz does not match frozen CAL population variance and floor')
            if delta_q0.ndim == 3 and delta_random.shape == delta_q0.shape and (wz.shape == delta_q0.shape[1:]):
                q0_norm = np.sqrt(np.sum(np.where(active, delta_q0 * wz, 0.0) ** 2, axis=(-2, -1)))
                random_norm = np.sqrt(np.sum(np.where(active, delta_random * wz, 0.0) ** 2, axis=(-2, -1)))
                if not np.isfinite(q0_norm).all() or np.any(q0_norm <= EPS):
                    errors.append('one or more Q0 residual weighted norms are zero/invalid')
                elif not np.allclose(q0_norm, random_norm, rtol=VECTOR_RTOL, atol=VECTOR_ATOL):
                    errors.append('random bank does not match Q0 weighted norm per record')
            clean_error = _finite('q0_clean_error', arrays['q0_clean_error'], errors)
            transport_error = _finite('q0_transport_error', arrays['q0_transport_error'], errors)
            if clean_error.shape != next_fp.shape or transport_error.shape != next_fp.shape:
                errors.append('Q0 error vectors do not match [N,P,D]')
            if clean_error.shape == next_fp.shape and transport_error.shape == next_fp.shape and (wz.shape == next_fp.shape[1:]):
                clean_component = float(np.mean(_weighted_mse(clean_error, wz, active), dtype=np.float64))
                transport_component = float(np.mean(_weighted_mse(transport_error, wz, active), dtype=np.float64))
                lambda_raw = max(clean_component, EPS) / max(transport_component, EPS)
                lambda_expected = float(np.clip(lambda_raw, 0.25, 4.0))
                spec = meta.get('lambda_spec')
                if not isinstance(spec, Mapping):
                    errors.append('lambda_spec is missing')
                else:
                    for key, expected in (('mean_L_clean_Q0', clean_component), ('mean_L_transport_Q0', transport_component), ('lambda_raw', lambda_raw), ('lambda_T', lambda_expected)):
                        if key not in spec or not math.isclose(float(spec[key]), expected, rel_tol=VECTOR_RTOL, abs_tol=VECTOR_ATOL):
                            errors.append(f'lambda_spec.{key} disagrees with raw A vectors')
                    if spec.get('clamped') is not (lambda_raw < 0.25 or lambda_raw > 4.0):
                        errors.append('lambda_spec.clamped flag disagrees with lambda_raw')
                record_summary.update({'mean_L_clean_Q0': clean_component, 'mean_L_transport_Q0': transport_component, 'lambda_raw': lambda_raw, 'lambda_T': lambda_expected})
    except Exception as exc:
        errors.append(f'Stage A array audit exception: {type(exc).__name__}: {exc}')
    return {'pass': not errors, 'complete': bool(meta.get('complete') is True and (not errors)), 'resource': False, 'errors': _compact_errors(errors), **record_summary}

def _fit_run_audit(meta: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _runner_fit_audit(artifact: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _runner_bank_array_audit(arrays: Any, expected_split: str, errors: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _runner_stage_b_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str, stage_a: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _stage_b_audit(artifact: Path, manifest: Mapping[str, Any], manifest_hash: str, stage_a: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _episode_seed_values(metric: np.ndarray, episodes: Sequence[str], evaluator: str, bank_indices: Sequence[int] | None) -> dict[int, np.ndarray]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _gate(values: dict[int, np.ndarray], comparator: dict[int, np.ndarray], ratio: float, name: str) -> dict[str, Any]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _mechanism_gates(clean_mse: np.ndarray, transport_mse: np.ndarray, episodes: Sequence[str]) -> dict[str, Any]:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

def _self_test() -> None:
    raise RuntimeError('Stage A verification snapshot excludes B and self-test')

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
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    require_allocation = _load_guard()
    allocation = require_allocation()
    artifact = args.artifact_dir.resolve()
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    manifest_hash = _manifest_sha256(manifest_path)
    manifest_result = _manifest_audit(manifest)
    result: dict[str, Any] = {'schema': 'frt-verification-v1', 'status': 'engineering_fail', 'allocation': {key: allocation.get(key) for key in ('scheduler', 'job_id', 'hostname', 'nodelist', 'verified')}, 'manifest': {**manifest_result, 'sha256': manifest_hash}}
    stage_a: dict[str, Any] = {'pass': False, 'complete': False, 'errors': ['not run']}
    stage_b: dict[str, Any] = {'pass': False, 'complete': False, 'errors': ['not run']}
    if args.stage in ('a', 'all'):
        stage_a = _stage_audit(artifact, manifest, manifest_hash)
        stage_a['wz_sha256'] = None
        a_arrays = artifact / 'bank_stage_a.npz'
        if not a_arrays.is_file():
            a_arrays = artifact / 'stage_a.npz'
        if a_arrays.is_file():
            try:
                with np.load(a_arrays, allow_pickle=False) as arrays:
                    if 'wz' in arrays:
                        stage_a['wz_sha256'] = hashlib.sha256(np.asarray(arrays['wz']).tobytes()).hexdigest()
            except Exception:
                pass
    if args.stage == 'a':
        result['stage_a'] = stage_a
        result['status'] = 'complete' if manifest_result['pass'] and stage_a.get('pass') else 'engineering_fail'
    elif args.stage == 'b':
        stage_a_identity = {'pass': False, 'wz_sha256': None}
        stage_b = _stage_b_audit(artifact, manifest, manifest_hash, stage_a_identity)
        result['stage_b'] = stage_b
        result['status'] = stage_b.get('status', 'engineering_fail')
    else:
        stage_b = _stage_b_audit(artifact, manifest, manifest_hash, stage_a)
        result['stage_a'] = stage_a
        result['stage_b'] = stage_b
        if not manifest_result['pass'] or not stage_a.get('pass'):
            result['status'] = 'engineering_fail'
        else:
            result['status'] = stage_b.get('status', 'engineering_fail')
    result['engineering_pass'] = bool(manifest_result['pass'] and (stage_a.get('pass') if args.stage in ('a', 'all') else True) and (stage_b.get('engineering_pass') if args.stage in ('b', 'all') else True))
    result['mechanism_gate_pass'] = bool(stage_b.get('mechanism_pass', False)) if args.stage in ('b', 'all') else False
    _write_json(args.output.resolve(), result)
    print(json.dumps({'status': result['status'], 'engineering_pass': result['engineering_pass'], 'mechanism_gate_pass': result['mechanism_gate_pass']}, separators=(',', ':')), flush=True)
if __name__ == '__main__':
    main()

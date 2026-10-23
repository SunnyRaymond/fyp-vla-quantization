"""Reuse the two frozen final80 checkpoints and the proven episode runtime."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

ARMS = ('softbin', 'relational')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_source_receipt(protocol, arm, receipt):
    source, expected = protocol['arm_sources'][arm], protocol['source_expected']
    identity = {'protocol': source['training_protocol'], 'unit': source['deployment_unit'],
                'pbs_jobid': source['training_job'], 'complete': True, 'updates': 80,
                'update_order': protocol['calibration']['update_order'],
                'initial_parity_max_abs': 0, 'actual_infer_parity_max_abs': 0,
                'cached_query_parity': expected['cached_query_parity']}
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError(f'Frozen final80 training receipt differs: {arm}')
    frozen = ('codebooks_unchanged', 'indices_unchanged', 'row_gains_unchanged',
              'continuous_scale_metadata_separate', 'original_exponents_unchanged',
              'gradient_coupling_complete', 'input_absmax_chain')
    if not all(receipt.get(key) is True for key in frozen):
        raise ValueError(f'Source coupled path or frozen VQ contents differ: {arm}')
    history = receipt.get('history', [])
    if (len(history) != 80 or [item.get('update') for item in history] != list(range(1, 81))
            or [item.get('case_id') for item in history] != protocol['calibration']['update_order']
            or any(item.get('seed_index') != 0 or item.get('video_steps') != 10
                   or item.get('action_steps') != 10 for item in history)):
        raise ValueError(f'Source update sequence differs: {arm}')
    scaling, ratios = receipt.get('scaling', {}), receipt.get('clipping', {})
    if (scaling.get('modules') != 614 or scaling.get('group_size') != 1
            or scaling.get('channels') != expected['scale_channels']
            or scaling.get('raw_scale_metadata_bytes') != expected['scale_metadata_bytes']
            or scaling.get('scale_metadata_dtype') != 'float32_log2'
            or scaling.get('cached_compensated_weight_evaluation') is not True
            or ratios.get('count') != 614):
        raise ValueError(f'Source scale/clip coverage differs: {arm}')
    check = receipt.get('gradient_selfcheck' if arm == 'softbin' else 'channel_selfcheck', {})
    if (check.get('pbs_jobid') != source['training_job']
            or not all(check.get(key) is True for key in
                       ('passed', 'independent_reference_passed', 'old_surrogate_rejected'))):
        raise ValueError(f'Source coupled-gradient selfcheck failed: {arm}')
    return receipt


def validate_endpoint_identity(endpoint, protocol, arm, row):
    source, checks = protocol['arm_sources'][arm], protocol['startup_checks']
    expected = {'protocol': source['endpoint_protocol'], 'arm': source['endpoint_arm'],
                'split': 'calibration', 'updates': 80, 'plan_row': row,
                'case_id': checks['case_id'], 'seed_index': checks['seed_index'],
                'sampler_seed': row['sampler_seeds'][checks['seed_index']],
                'pbs_jobid': source['endpoint_job'],
                'counts': {'video_steps': 10, 'action_steps': 10}}
    if any(endpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Saved source endpoint is differently bound: {arm}')
    return endpoint


def validate_training_artifacts(protocol, arm):
    """Small JSON/stat checks; the caller must already hold a PBS allocation."""
    source = protocol['arm_sources'][arm]
    source_protocol = read_json(Path(__file__).with_name(f'source_{arm}_protocol.json'))
    if (source_protocol.get('protocol') != source['training_protocol']
            or source_protocol.get('remote_root') != source['training_root']
            or any(source_protocol.get(key) != protocol[key]
                   for key in ('calibration', 'training', 'clipping', 'scaling'))):
        raise ValueError(f'Pinned training protocol differs: {arm}')
    receipt = validate_source_receipt(protocol, arm, read_json(source['training_receipt']))
    if not Path(source['deployment']).is_file() or not Path(source['reference_tensor']).is_file():
        raise FileNotFoundError(f'Source checkpoint or full endpoint tensor missing: {arm}')
    plan_path = Path(protocol['sources']['calibration_plan'])
    plan = read_json(plan_path)
    if (plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c80') != protocol['calibration']['case_ids']):
        raise ValueError('Frozen original C80 input plan differs')
    rows = {int(row['case_id']): row for row in plan['inputs']}
    if (len(rows) != len(plan['inputs'])
            or any(rows[case].get('domain') != 'original' for case in protocol['calibration']['case_ids'])):
        raise ValueError('Calibration plan contains duplicate or non-original C80 inputs')
    row = rows[protocol['startup_checks']['case_id']]
    endpoint = validate_endpoint_identity(read_json(source['reference_record']), protocol, arm, row)
    return {'source_protocol': source_protocol, 'training_receipt': receipt,
            'reference_endpoint': endpoint, 'reference_row': row, 'plan_path': plan_path}


def expected_install_metadata(protocol, arm):
    source = protocol['arm_sources'][arm]
    return {'protocol': protocol['protocol'], 'arm': arm,
        'source_training_protocol': source['training_protocol'],
        'source_training_unit': source['deployment_unit'],
        'source_training_job': source['training_job'], 'source_deployment': source['deployment'],
        'source_endpoint_job': source['endpoint_job'],
        'baseline_deployment': protocol['sources']['baseline_deployment'],
        'module_count': 614, 'execution': 'decoded_bf16_vq', 'activation_bits': 4,
        'nominal_weight_bits': 4, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
        'native_int4_gemm_calls': 0, 'transform_implementation': 'batched_h128_v2',
        'smooth_alpha': .5, 'hadamard_block': 128, 'scale_block': 32,
        'channel_scale_group_size': 1, 'scale_metadata_dtype': 'float32_log2',
        'fixed_indices': True, 'fixed_scale_exponents': True,
        'train_weights': False, 'train_clips': True, 'train_channel_scales': True,
        'updates': 80, 'training_updates': 80, 'new_training_updates': 0,
        'calibration_ids': protocol['calibration']['case_ids'],
        'cached_compensated_weight_evaluation': True}


def _runtime_type():
    import runner as pilot

    spec = importlib.util.spec_from_file_location(
        'softbin_previous_adapter', Path(__file__).with_name('previous_vq_adapter.py'))
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    base = previous._runtime_type()

    class Runtime(base):
        def __init__(self, out, protocol, worker_id):
            import torch
            import batched_transform
            from channel_banks import ChannelBanks

            pilot.guard()
            self.worker_id = worker_id
            if type(worker_id) is not int or worker_id not in range(4):
                raise ValueError('Expected one of four frozen workers')
            arm = ARMS[worker_id // 2]
            preflight = validate_training_artifacts(protocol, arm)
            torch.backends.cuda.matmul.allow_tf32 = False
            pilot.Runtime.__init__(self, out)
            self.model.eval()
            self.arm('bf16')
            self.protocol, self.arm_name, self.current_arm = protocol, arm, arm
            self.out, self.preflight = Path(out), preflight
            self.run_queries = self.linear_calls = 0
            self.transform_implementation = batched_transform.IMPLEMENTATION
            if self.transform_implementation != 'batched_h128_v2':
                raise RuntimeError('Expected the frozen batched H128 implementation')
            source = protocol['arm_sources'][arm]
            deployment = torch.load(source['deployment'], map_location='cpu', weights_only=True)
            baseline = torch.load(protocol['sources']['baseline_deployment'], map_location='cpu', weights_only=True)
            if (deployment.get('protocol') != preflight['source_protocol']
                    or deployment.get('unit') != source['deployment_unit']
                    or deployment.get('updates') != 80
                    or deployment.get('baseline_checkpoint') != protocol['sources']['baseline_deployment']
                    or deployment.get('scale_metadata_dtype') != 'float32_log2'
                    or baseline.get('arm') != 'rho1.5' or baseline.get('updates') != 8
                    or baseline.get('target_linears') != 614):
                raise RuntimeError(f'Source final80 deployment or C8 initialization differs: {arm}')
            modules = set(self.q.modules)
            if (len(modules) != 614 or set(baseline.get('module_masters', {})) != modules
                    or set(deployment.get('ratios', {})) != modules
                    or set(deployment.get('log2_scales', {})) != modules):
                raise RuntimeError('Source deployment must cover all614 physical Linears')
            total_channels = 0
            for name, mod in self.q.modules.items():
                ratio, logs = deployment['ratios'][name], deployment['log2_scales'][name]
                if (not torch.is_tensor(ratio) or ratio.dtype != torch.float32 or ratio.numel() != 1
                        or not bool(torch.isfinite(ratio).all()) or not .5 <= float(ratio) <= 1):
                    raise RuntimeError(f'Invalid frozen clipping ratio: {name}')
                if (not torch.is_tensor(logs) or logs.dtype != torch.float32
                        or tuple(logs.shape) != (mod.weight.shape[1],)
                        or not bool(torch.isfinite(logs).all())
                        or not bool(((logs >= -1) & (logs <= 1)).all())):
                    raise RuntimeError(f'Invalid frozen per-input-channel log2D: {name}')
                total_channels += logs.numel()
            if total_channels != protocol['source_expected']['scale_channels']:
                raise RuntimeError('Frozen learned channel count differs')
            self.banks = ChannelBanks(self, protocol['calibration']['bank_directory'],
                                      baseline['module_masters'], protocol['clipping'], protocol['scaling'])
            self.banks.restore_clips(deployment['ratios'])
            self.banks.restore_scales(deployment['log2_scales'])
            for values in (self.banks.books.values(), self.banks.logs.values(),
                           self.banks.ratios.values(), self.banks.shadows.values()):
                for parameter in values:
                    parameter.requires_grad_(False)
            self.seen, self.forwards = self.banks.seen, self.banks.forwards
            for mod in self.q.modules.values():
                forward = mod.forward

                def counted(value, _forward=forward):
                    self.linear_calls += 1
                    return _forward(value)

                mod.forward = counted
            receipt = preflight['training_receipt']
            self.install_metadata = {**expected_install_metadata(protocol, arm),
                'ratios': receipt['clipping'], 'scaling': receipt['scaling'],
                'source_cached_query_parity': receipt['cached_query_parity']}
            self.stage('final80_loaded_before_source_replay')
            del baseline, deployment
            torch.cuda.empty_cache()

        def stage(self, phase, **extra):
            pilot.save(self.out / 'runtime_softbin.json', {
                'protocol': self.protocol['protocol'], 'worker_id': self.worker_id,
                'arm': self.current_arm, 'pbs_job_id': os.environ['PBS_JOBID'],
                'phase': phase, 'install_metadata': self.install_metadata, **extra})
            print(f'SOFTBIN_STAGE worker={self.worker_id} arm={self.current_arm} phase={phase}', flush=True)

        def prepare(self, results_root):
            import numpy as np
            import torch
            import sampler
            import calibration_source

            protocol, arm, preflight = self.protocol, self.current_arm, self.preflight
            source, checks = protocol['arm_sources'][arm], protocol['startup_checks']
            row, endpoint = preflight['reference_row'], preflight['reference_endpoint']
            saved = torch.load(source['reference_tensor'], map_location='cpu', weights_only=True)
            validate_endpoint_identity(saved, protocol, arm, row)
            for key, shape in (('action', (32, 7)), ('future', (1, 48, 3, 14, 28))):
                value = saved.get(key)
                if (not torch.is_tensor(value) or value.dtype != torch.float32
                        or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
                    raise RuntimeError(f'Invalid saved source {key}: {arm}')
            observation, metadata = calibration_source.load_observation(np, preflight['plan_path'].parent, row)
            self.description = metadata['description']
            datum, seed = self.datum(observation), endpoint['sampler_seed']

            def source_query():
                self.banks.seen.clear()
                with torch.no_grad():
                    action, counts, future = sampler.differentiable_action(
                        self, datum, seed, checkpointed=False, return_future=True)
                if (len(self.banks.seen) != 614 or counts != {'video_steps': 10, 'action_steps': 10}
                        or tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
                        or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())
                        or self.q._native_int4_gemm_calls != 0 or self.q._kv_active):
                    raise RuntimeError('Source query coverage/shape/finite/native gate failed')
                return action.detach().float().cpu(), future.detach().float().cpu()

            self.stage('replay_saved_full_source_tensors')
            uncached_action, uncached_future = source_query()
            source_action_error = float((uncached_action - saved['action']).abs().max())
            source_future_error = float((uncached_future - saved['future']).abs().max())
            self.banks.cache_compensated_weights()
            cached_action, cached_future = source_query()
            action_error = float((cached_action - uncached_action).abs().max())
            future_error = float((cached_future - uncached_future).abs().max())
            actual, _ = self.infer(datum, seed, measure=False)
            actual_error = float((actual.detach().float().cpu() - saved['action']).abs().max())
            errors = (source_action_error, source_future_error, action_error, future_error, actual_error)
            if (not all(math.isfinite(error) for error in errors)
                    or source_action_error > checks['saved_source_action_max_abs_tolerance']
                    or source_future_error > checks['saved_source_future_max_abs_tolerance']
                    or action_error != 0 or future_error != 0
                    or actual_error > checks['actual_infer_action_max_abs_tolerance']
                    or len(self.banks.seen) != 614 or not self.banks.cached):
                raise RuntimeError(f'Frozen source/cache/actual inference parity failed: {errors}')
            self.run_queries = 0
            self.stage('ready', source_action_max_abs=source_action_error,
                       source_future_max_abs=source_future_error)
            return {'protocol': protocol['protocol'], 'worker_id': self.worker_id, 'arm': arm,
                'pbs_job_id': os.environ['PBS_JOBID'], 'passed': True, 'observed_modules': 614,
                'case_id': row['case_id'], 'seed_index': checks['seed_index'], 'sampler_seed': seed,
                'source_action_max_abs': source_action_error, 'source_future_max_abs': source_future_error,
                'uncached_cached_action_max_abs': action_error, 'uncached_cached_future_max_abs': future_error,
                'actual_infer_action_max_abs': actual_error, 'startup_queries': 3,
                'source_training_job': source['training_job'], 'source_endpoint_job': source['endpoint_job'],
                'training_updates': 80, 'new_training_updates': 0,
                'cached_compensated_weights': True, 'install_metadata': self.install_metadata}

    return Runtime


class SoftbinRuntime:
    """Keep controller and metadata tests independent of Torch imports."""
    def __new__(cls, *args, **kwargs):
        return _runtime_type()(*args, **kwargs)

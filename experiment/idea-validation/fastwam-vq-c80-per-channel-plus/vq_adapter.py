"""Restore the completed per-channel checkpoint onto the proven Plus runtime."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

ARM = 'per_channel_clip'
METRICS = ('first10motor_mse', 'first10motor_rmse', 'full32motor_rmse',
           'late22motor_rmse', 'first10gripper_rmse', 'future_latent_nmse', 'mixed_loss')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_source_receipt(protocol, receipt):
    expected = protocol['source_expected']
    if (receipt.get('protocol') != protocol['sources']['training_protocol']
            or receipt.get('pbs_jobid') != protocol['sources']['training_job']
            or receipt.get('complete') is not True or receipt.get('updates') != 80
            or receipt.get('update_order') != protocol['calibration']['update_order']
            or receipt.get('initial_parity_max_abs') != 0
            or receipt.get('cached_query_parity') != expected['cached_query_parity']):
        raise ValueError('Source final80 checkpoint receipt or parity differs')
    frozen = ('codebooks_unchanged', 'indices_unchanged', 'row_gains_unchanged',
              'continuous_scale_metadata_separate', 'original_exponents_unchanged',
              'gradient_coupling_complete', 'input_absmax_chain')
    if not all(receipt.get(key) is True for key in frozen):
        raise ValueError('Source VQ contents or coupled per-channel path were not frozen')
    history = receipt.get('history', [])
    if (len(history) != 80
            or [item.get('update') for item in history] != list(range(1, 81))
            or [item.get('case_id') for item in history] != protocol['calibration']['update_order']
            or any(item.get('seed_index') != 0 or item.get('video_steps') != 10
                   or item.get('action_steps') != 10 for item in history)):
        raise ValueError('Source calibration is not the exact frozen80 update sequence')
    scaling, ratios = receipt.get('scaling', {}), receipt.get('clipping', {})
    if (scaling.get('modules') != 614 or scaling.get('group_size') != 1
            or scaling.get('channels') != expected['scale_channels']
            or scaling.get('raw_scale_metadata_bytes') != expected['raw_scale_metadata_bytes']
            or scaling.get('scale_metadata_dtype') != 'float32_log2'
            or scaling.get('cached_compensated_weight_evaluation') is not True
            or ratios.get('count') != 614):
        raise ValueError('Source per-channel scale or clipping coverage differs')
    check = receipt.get('gradient_selfcheck', {})
    if (check.get('pbs_jobid') != protocol['sources']['training_job']
            or not all(check.get(key) is True for key in
                       ('passed', 'independent_reference_passed', 'old_surrogate_rejected'))):
        raise ValueError('Source independent coupled-gradient selfcheck did not pass')
    return receipt


def compare_endpoint_metrics(actual, saved, protocol):
    config = protocol['closed_loop_reference']
    errors = {}
    for name in METRICS:
        a, b = actual.get(name), saved.get(name)
        if (isinstance(a, bool) or isinstance(b, bool)
                or not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                or not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < 0):
            raise ValueError(f'Nonfinite or missing source endpoint metric: {name}')
        errors[name] = abs(a - b)
        if not math.isclose(a, b, rel_tol=config['source_metric_relative_tolerance'],
                            abs_tol=config['source_metric_absolute_tolerance']):
            raise ValueError(f'Source endpoint metric differs for {name}: {a} versus {b}')
    return errors


def validate_training_artifacts(protocol, arm, training_dir):
    """JSON/stat checks only; all tensor loads occur after the PBS compute guard."""
    sources = protocol['sources']
    if arm != ARM or Path(training_dir) != Path(sources['training_results']):
        raise ValueError('Use the exact completed per-channel training directory')
    source_protocol = read_json(Path(__file__).with_name('source_protocol.json'))
    if (source_protocol.get('protocol') != sources['training_protocol']
            or source_protocol.get('remote_root') != sources['training_root']
            or source_protocol.get('arm') != ARM
            or any(source_protocol.get(key) != protocol[key]
                   for key in ('calibration', 'training', 'clipping', 'scaling'))):
        raise ValueError('Pinned source protocol differs from the requested final80 arm')
    receipt = validate_source_receipt(protocol, read_json(Path(training_dir) / 'training_receipt.json'))
    deployment = Path(sources['deployment'])
    if not deployment.is_file() or deployment.parent != Path(training_dir):
        raise ValueError('Completed source deployment is missing')
    plan_path = Path(sources['plan_file'])
    plan = read_json(plan_path)
    if (plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c80') != protocol['calibration']['case_ids']):
        raise ValueError('Frozen original C80 input plan differs')
    rows = {int(row['case_id']): row for row in plan['inputs']}
    reference = protocol['closed_loop_reference']
    row = rows[reference['case_id']]
    endpoint = read_json(sources['reference_endpoint'])
    if (endpoint.get('protocol') != sources['training_protocol'] or endpoint.get('arm') != ARM
            or endpoint.get('split') != 'calibration' or endpoint.get('updates') != 80
            or endpoint.get('pbs_jobid') != sources['training_job']
            or endpoint.get('case_id') != reference['case_id']
            or endpoint.get('seed_index') != reference['seed_index']
            or endpoint.get('sampler_seed') != row['sampler_seeds'][reference['seed_index']]
            or endpoint.get('plan_row') != row
            or endpoint.get('counts') != {'video_steps': 10, 'action_steps': 10}):
        raise ValueError('Saved final80 source endpoint is differently bound')
    compare_endpoint_metrics(endpoint, endpoint, protocol)
    return {'training_dir': Path(training_dir), 'deployment_path': deployment,
            'source_protocol': source_protocol, 'training_receipt': receipt,
            'reference_endpoint': endpoint, 'plan_path': plan_path, 'reference_row': row,
            'reference_action': {'sampler_seed': endpoint['sampler_seed'],
                                 'actual_infer_parity': receipt['cached_query_parity']}}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_type():
    import runner as pilot

    # Reuse the completed experiment's episode, query-meter and cleanup code.
    previous = _module('per_channel_previous_adapter', Path(__file__).with_name('previous_vq_adapter.py'))
    base = previous._runtime_type()

    class Runtime(base):
        def __init__(self, out, protocol, arm, training_dir, preflight):
            import torch
            import batched_transform
            from channel_banks import ChannelBanks
            from evaluate import expected_install_metadata, validate_install_metadata

            torch.backends.cuda.matmul.allow_tf32 = False
            pilot.Runtime.__init__(self, out)
            self.model.eval()
            self.arm('bf16')
            self.protocol, self.arm_name, self.current_arm = protocol, arm, arm
            self.preflight = preflight
            self.run_queries = self.linear_calls = 0
            self.transform_implementation = batched_transform.IMPLEMENTATION
            if self.transform_implementation != 'batched_h128_v2':
                raise RuntimeError('Expected the completed batched H128 implementation')
            deployment = torch.load(preflight['deployment_path'], map_location='cpu', weights_only=True)
            baseline = torch.load(protocol['sources']['baseline_deployment'], map_location='cpu', weights_only=True)
            if (deployment.get('protocol') != preflight['source_protocol']
                    or deployment.get('updates') != 80
                    or deployment.get('baseline_checkpoint') != protocol['sources']['baseline_deployment']
                    or deployment.get('scale_metadata_dtype') != 'float32_log2'
                    or baseline.get('arm') != 'rho1.5' or baseline.get('updates') != 8
                    or baseline.get('target_linears') != 614
                    or len(baseline.get('module_masters', {})) != 614):
                raise RuntimeError('Source final80 deployment or C8 initialization identity differs')
            modules = set(self.q.modules)
            if (len(modules) != 614 or set(baseline['module_masters']) != modules
                    or set(deployment.get('ratios', {})) != modules
                    or set(deployment.get('log2_scales', {})) != modules):
                raise RuntimeError('Source deployment must cover all614 Linears')
            total_channels = 0
            for name, mod in self.q.modules.items():
                ratio, logs = deployment['ratios'][name], deployment['log2_scales'][name]
                if (not torch.is_tensor(ratio) or ratio.dtype != torch.float32 or ratio.numel() != 1
                        or not bool(torch.isfinite(ratio).all()) or not .5 <= float(ratio) <= 1):
                    raise RuntimeError(f'Invalid source clipping ratio: {name}')
                if (not torch.is_tensor(logs) or logs.dtype != torch.float32
                        or tuple(logs.shape) != (mod.weight.shape[1],)
                        or not bool(torch.isfinite(logs).all())
                        or not bool(((logs >= -1) & (logs <= 1)).all())):
                    raise RuntimeError(f'Invalid per-input-channel log2D: {name}')
                total_channels += logs.numel()
            if total_channels != protocol['source_expected']['scale_channels']:
                raise RuntimeError('Total learned channel count differs from the source')
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
            source_receipt = preflight['training_receipt']
            self.install_metadata = {**expected_install_metadata(protocol),
                'ratios': source_receipt['clipping'], 'scaling': source_receipt['scaling'],
                'source_cached_query_parity': source_receipt['cached_query_parity']}
            validate_install_metadata(self.install_metadata, protocol, arm)
            pilot.save(Path(out) / 'runtime_per_channel.json', {
                'phase': 'source_final80_loaded_before_parity_and_weight_caching',
                'install_metadata': self.install_metadata, 'startup_queries_measured': False})
            del baseline, deployment
            torch.cuda.empty_cache()

        def validate_reference(self):
            import numpy as np
            import torch
            import sampler
            import calibration_source
            import latent_reference
            from metrics_reference import metrics

            preflight, protocol = self.preflight, self.protocol
            row, saved = preflight['reference_row'], preflight['reference_endpoint']
            observation, metadata = calibration_source.load_observation(np, preflight['plan_path'].parent, row)
            self.description = metadata['description']
            datum, seed = self.datum(observation), saved['sampler_seed']

            def source_query():
                self.banks.seen.clear()
                with torch.no_grad():
                    action, counts, future = sampler.differentiable_action(
                        self, datum, seed, checkpointed=False, return_future=True)
                if (len(self.banks.seen) != 614 or counts != {'video_steps': 10, 'action_steps': 10}
                        or tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
                        or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())):
                    raise RuntimeError('Source query coverage, shape or finite-value gate failed')
                return action.detach().cpu(), future.detach().cpu()

            uncached_action, uncached_future = source_query()
            teacher = torch.load(Path(protocol['sources']['teacher_directory']) / '2000_0.pt',
                                 map_location='cpu', weights_only=True)
            if teacher.get('plan_row') != row or teacher.get('sampler_seed') != seed:
                raise RuntimeError('Shared teacher does not match source endpoint')
            observed = metrics(torch, latent_reference, uncached_action, uncached_future,
                               teacher, protocol['training'])
            metric_errors = compare_endpoint_metrics(observed, saved, protocol)
            self.banks.cache_compensated_weights()
            if not self.banks.cached:
                raise RuntimeError('Per-channel compensation was not cached')
            cached_action, cached_future = source_query()
            action_error = float((cached_action.float() - uncached_action.float()).abs().max())
            future_error = float((cached_future.float() - uncached_future.float()).abs().max())
            actual, _ = self.infer(datum, seed, measure=False)
            actual_error = float((actual.detach().float().cpu() - cached_action.float()).abs().max())
            tolerance = protocol['closed_loop_reference']['max_abs_tolerance']
            if (action_error != 0 or future_error != 0 or not math.isfinite(actual_error)
                    or actual_error > tolerance or len(self.banks.seen) != 614):
                raise RuntimeError(f'Cached/source/actual inference parity failed: '
                                   f'{action_error}, {future_error}, {actual_error}')
            return {'protocol': protocol['protocol'], 'arm': self.current_arm,
                'case_id': row['case_id'], 'seed_index': 0, 'sampler_seed': seed,
                'pbs_job_id': os.environ['PBS_JOBID'], 'passed': True, 'observed_modules': 614,
                'max_abs_vs_reference': actual_error, 'tolerance': tolerance,
                'reference_kind': protocol['closed_loop_reference']['kind'],
                'reference_actual_infer_parity': preflight['reference_action']['actual_infer_parity'],
                'source_endpoint_metrics_passed': True, 'source_endpoint_metric_errors': metric_errors,
                'source_endpoint_metrics': observed,
                'uncached_cached_action_max_abs': action_error,
                'uncached_cached_future_max_abs': future_error,
                'actual_infer_action_max_abs': actual_error,
                'training_receipt_status': 'complete', 'startup_queries': 3,
                'source_training_job': protocol['sources']['training_job'],
                'install_metadata': self.install_metadata}

    return Runtime


class TrainedRuntime:
    """Keep metadata-only controller imports independent of Torch."""
    def __new__(cls, *args, **kwargs):
        return _runtime_type()(*args, **kwargs)

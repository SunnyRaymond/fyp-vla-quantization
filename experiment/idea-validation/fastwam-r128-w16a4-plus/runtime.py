"""Restore frozen H/C80-R128 W16A4 and reuse the proven episode loop."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

ARMS = ('H_W16A4', 'C80_R128_W16A4')
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'mixed_loss',
           'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def compare_source_metrics(actual, expected, tolerance):
    errors = {}
    for name in METRICS:
        left, right = actual.get(name), expected.get(name)
        if not finite(left) or not finite(right) or left < 0 or right < 0:
            raise ValueError(f'Missing/nonfinite source metric: {name}')
        errors[name] = abs(left - right)
    if max(errors.values()) > tolerance:
        raise ValueError(f'Pinned source metric replay differs: {errors}')
    return errors


def validate_source_documents(protocol, arm, source_protocol, summary, endpoint, arm_summary=None):
    source = protocol['arm_sources'][arm]
    if (source_protocol.get('protocol') != source['source_protocol']
            or source_protocol.get('remote_root') != source['source_root']
            or source_protocol.get('target_linears') != 614
            or source_protocol.get('calibration', {}).get('plan_file') != protocol['sources']['calibration_plan']
            or source_protocol.get('calibration', {}).get('bank_directory') != protocol['sources']['bank_directory']
            or source_protocol.get('quantization', {}).get('D') != 1
            or source_protocol.get('quantization', {}).get('clipping_ratio') != 1):
        raise ValueError(f'Frozen source protocol differs: {arm}')
    if (summary.get('protocol') != source['source_protocol']
            or summary.get('pbs_jobid') != source['source_job']
            or summary.get('complete') is not True or summary.get('status') != 'complete'
            or summary.get('target_linears') != 614):
        raise ValueError(f'Frozen source run is incomplete or differently bound: {arm}')
    checks = protocol['startup_checks']
    expected = {'protocol': source['source_protocol'], 'arm': source['endpoint_arm'],
        'split': 'calibration', 'case_id': checks['case_id'], 'seed_index': checks['seed_index'],
        'sampler_seed': checks['sampler_seed'], 'pbs_jobid': source['source_job'],
        'counts': {'video_steps': 10, 'action_steps': 10}}
    if any(endpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Frozen source endpoint binding differs: {arm}')
    compare_source_metrics(endpoint, endpoint, checks['source_metric_max_abs_tolerance'])
    if arm == ARMS[1]:
        report = summary.get('arms', {}).get(source['endpoint_arm'], {})
        required = {'updates': 8, 'training_exposures': 80, 'calibration_queries': 8,
                    'heldout_queries': 80, 'c80_queries': 80, 'all_parameters_frozen': True}
        if (any(report.get(key) != value for key, value in required.items())
                or not finite(report.get('formal_infer_max_abs'))
                or not 0 <= report['formal_infer_max_abs'] <= 1e-6
                or not isinstance(arm_summary, dict)
                or arm_summary.get('complete') is not True or arm_summary.get('history_updates') != 8
                or arm_summary.get('protocol') != source['source_protocol']
                or arm_summary.get('pbs_jobid') != source['source_job']
                or arm_summary.get('lane') != 'coverage_c80'
                or arm_summary.get('arm') != source['endpoint_arm']
                or arm_summary.get('report') != report
                or source_protocol.get('calibration', {}).get('c80_case_ids') != list(range(2000, 2080))):
            raise ValueError('C80 final-update8 checkpoint coverage or source summary differs')
    return endpoint


def expected_install_metadata(protocol, arm):
    source = protocol['arm_sources'][arm]
    return {'protocol': protocol['protocol'], 'arm': arm,
        'source_protocol': source['source_protocol'], 'source_job': source['source_job'],
        'source_rotation_deployment': source['rotation_deployment'],
        'source_bank': protocol['sources']['bank_directory'],
        'weight_bits': 16, 'activation_bits': 4, 'execution': 'bf16_weights_a4',
        'vq_enabled': False, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
        'native_int4_gemm_calls': 0, 'module_count': 614,
        'transform_implementation': 'signed_h128_cayley_cached_weights_v1',
        'smooth_alpha': .5, 'hadamard_block': 128, 'D': 1, 'clipping_ratio': 1,
        'rotation_shared_streams': ['action', 'video'], 'rotation_shape': [128, 128],
        'rotation_updates': source['updates'], 'new_training_updates': 0,
        'all_parameters_frozen': True, 'cached_transformed_weights': True}


def _preflight(protocol, arm):
    source = protocol['arm_sources'][arm]
    source_protocol = read_json(Path(__file__).with_name(f'source_{arm}_protocol.json'))
    summary, endpoint = read_json(source['source_summary']), read_json(source['endpoint'])
    arm_summary = read_json(source['arm_summary']) if 'arm_summary' in source else None
    validate_source_documents(protocol, arm, source_protocol, summary, endpoint, arm_summary)
    plan_path = Path(protocol['sources']['calibration_plan'])
    plan = read_json(plan_path)
    if (plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c80') != protocol['calibration']['case_ids']):
        raise ValueError('Frozen C80 plan differs')
    rows = {int(row['case_id']): row for row in plan['inputs']}
    row = rows[protocol['startup_checks']['case_id']]
    if (row.get('domain') != 'original' or row['sampler_seeds'][0] != endpoint['sampler_seed']
            or row['suite'] != endpoint['suite']):
        raise ValueError('Source endpoint and original observation plan disagree')
    return {'source_protocol': source_protocol, 'source_endpoint': endpoint,
            'plan_path': plan_path, 'reference_row': row}


def _runtime_type():
    import runner as pilot

    spec = importlib.util.spec_from_file_location(
        'r128_episode_base', Path(__file__).with_name('previous_vq_adapter.py'))
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    base = previous._runtime_type()

    class Runtime(base):
        def __init__(self, out, protocol, worker_id):
            import torch
            from rotation_bank import RotationBanks

            pilot.guard()
            if type(worker_id) is not int or worker_id not in range(4):
                raise ValueError('Expected one of four frozen workers')
            arm = ARMS[worker_id // 2]
            preflight = _preflight(protocol, arm)
            torch.backends.cuda.matmul.allow_tf32 = False
            pilot.Runtime.__init__(self, out)
            self.model.eval()
            self.arm('bf16')
            self.protocol, self.arm_name, self.current_arm = protocol, arm, arm
            self.out, self.worker_id, self.preflight = Path(out), worker_id, preflight
            self.run_queries = self.linear_calls = 0
            self.transform_implementation = protocol['quantization']['transform_implementation']
            if any(mod.weight.dtype != torch.bfloat16 for mod in self.q.modules.values()):
                raise RuntimeError('Expected the original BF16 target weights')
            self.banks = RotationBanks(self, protocol['sources']['bank_directory'])
            source = protocol['arm_sources'][arm]
            if source['rotation_deployment'] is not None:
                deployment = torch.load(source['rotation_deployment'], map_location='cpu', weights_only=True)
                if (deployment.get('protocol') != preflight['source_protocol']
                        or deployment.get('lane') != 'coverage_c80'
                        or deployment.get('arm') != source['endpoint_arm']
                        or deployment.get('updates') != 8 or deployment.get('target_linears') != 614
                        or set(deployment.get('rotations', {})) != {'action', 'video'}):
                    raise RuntimeError('C80 rotation deployment identity differs')
                if any(value.dtype != torch.float32 for value in deployment['rotations'].values()):
                    raise RuntimeError('Frozen raw rotations must remain FP32')
                self.banks.rotations.restore(deployment['rotations'])
                del deployment
            self.banks.rotations.freeze()
            if (set(self.banks.rotations.raw) != {'action', 'video'}
                    or len(self.banks.modules) != 614
                    or any(parameter.requires_grad for parameter in self.model.parameters())):
                raise RuntimeError('Frozen original W16 model or two-stream rotation coverage differs')
            with torch.no_grad():
                orthogonality = self.banks.rotations.orthogonality()
                movement = {stream: float((self.banks.rotations.matrix(stream)
                    - torch.eye(128, device='cuda')).norm()) for stream in ('action', 'video')}
            if (not all(finite(value) for value in (*orthogonality.values(), *movement.values()))
                    or max(orthogonality.values()) > protocol['startup_checks']['orthogonality_max_abs_tolerance']
                    or (arm == ARMS[0] and any(value != 0 for value in movement.values()))):
                raise RuntimeError('Frozen H/Cayley orthogonality or identity gate failed')
            self.seen, self.forwards = self.banks.seen, self.banks.original
            self.cached_weights = False
            for mod in self.q.modules.values():
                forward = mod.forward

                def counted(value, _forward=forward):
                    self.linear_calls += 1
                    return _forward(value)

                mod.forward = counted
            self.install_metadata = {**expected_install_metadata(protocol, arm),
                'orthogonality': orthogonality, 'rotation_movement': movement}
            self.stage('source_loaded_before_weight_caching')
            torch.cuda.empty_cache()

        def stage(self, phase, **extra):
            pilot.save(self.out / 'runtime_r128.json', {
                'protocol': self.protocol['protocol'], 'worker_id': self.worker_id,
                'arm': self.current_arm, 'pbs_job_id': os.environ['PBS_JOBID'], 'phase': phase,
                'install_metadata': self.install_metadata, **extra})
            print(f'R128_STAGE worker={self.worker_id} arm={self.current_arm} phase={phase}', flush=True)

        def cache_weights(self):
            import torch
            import torch.nn.functional as F

            if self.cached_weights:
                raise RuntimeError('Do not transform BF16 weights twice')
            with torch.no_grad():
                for name, mod in self.q.modules.items():
                    # Each module still holds its original W until its own source transform completes.
                    adjusted = self.banks.weight(name).to(torch.bfloat16)
                    if not bool(torch.isfinite(adjusted).all()):
                        raise RuntimeError(f'Nonfinite transformed original W16 weight: {name}')
                    mod.weight.data = adjusted

                    def cached_forward(value, _name=name, _mod=mod):
                        self.banks.seen.add(_name)
                        self.linear_calls += 1
                        return F.linear(self.banks.operand(value, _name), _mod.weight, _mod.bias)

                    mod.forward = cached_forward
            self.cached_weights = True
            torch.cuda.empty_cache()

        def prepare(self, results_root):
            import numpy as np
            import torch
            import sampler
            import calibration_source
            import latent_reference

            protocol, checks = self.protocol, self.protocol['startup_checks']
            arm, preflight = self.current_arm, self.preflight
            source = protocol['arm_sources'][arm]
            row, saved = preflight['reference_row'], preflight['source_endpoint']
            observation, metadata = calibration_source.load_observation(np, preflight['plan_path'].parent, row)
            self.description = metadata['description']
            datum, seed = self.datum(observation), saved['sampler_seed']
            receipt = read_json(protocol['sources']['teacher_receipt'])
            teacher_cache = torch.load(protocol['sources']['teacher_cache'], map_location='cpu', weights_only=True)
            if (receipt.get('passed') is not True or receipt.get('complete') is not True
                    or teacher_cache.get('normalization', {}).get('normalization_c') != protocol['training']['normalization_c']
                    or not any(item.get('case_id') == row['case_id'] and item.get('plan_row') == row
                               for item in teacher_cache.get('rows', []))):
                raise RuntimeError('Reused C8 BF16 teacher or original input binding differs')
            teacher = teacher_cache['teacher'][str(row['case_id'])]

            def source_query():
                self.banks.seen.clear()
                with torch.no_grad():
                    action, counts, future = sampler.differentiable_action(
                        self, datum, seed, checkpointed=False, return_future=True)
                if (counts != {'video_steps': 10, 'action_steps': 10}
                        or tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
                        or len(self.banks.seen) != 614 or self.q._native_int4_gemm_calls != 0
                        or self.q._kv_active or not bool(torch.isfinite(action).all())
                        or not bool(torch.isfinite(future).all())):
                    raise RuntimeError('Full-query source shape/coverage/finite gate failed')
                return action.detach(), future.detach()

            self.stage('replay_source_metrics')
            uncached_action, uncached_future = source_query()
            delta = uncached_action.float() - teacher['action'].to('cuda').float()
            a = delta[:10, :6].square().mean()
            z = latent_reference.future_nmse_loss(torch, uncached_future, teacher['future'].to('cuda'))
            metrics = {'first10motor_rmse': float(a.sqrt()), 'first10motor_mse': float(a),
                'future_latent_nmse': float(z), 'mixed_loss': float(a + protocol['training']['latent_weight'] * z),
                'full32motor_rmse': float(delta[:, :6].square().mean().sqrt()),
                'late22motor_rmse': float(delta[10:, :6].square().mean().sqrt()),
                'first10gripper_rmse': float(delta[:10, 6].square().mean().sqrt())}
            metric_errors = compare_source_metrics(metrics, saved, checks['source_metric_max_abs_tolerance'])
            uncached_action, uncached_future = uncached_action.float().cpu(), uncached_future.float().cpu()
            del teacher_cache, teacher, delta, a, z
            self.stage('cache_frozen_transformed_weights')
            self.cache_weights()
            cached_action, cached_future = source_query()
            action_error = float((cached_action.float().cpu() - uncached_action).abs().max())
            future_error = float((cached_future.float().cpu() - uncached_future).abs().max())
            actual, _ = self.infer(datum, seed, measure=False)
            actual_error = float((actual.float().cpu() - cached_action.float().cpu()).abs().max())
            if (action_error != 0 or future_error != 0 or not math.isfinite(actual_error)
                    or actual_error > checks['actual_infer_action_max_abs_tolerance']
                    or len(self.banks.seen) != 614 or not self.cached_weights):
                raise RuntimeError(f'Cached/uncached/formal query parity failed: {action_error}, {future_error}, {actual_error}')
            self.run_queries = 0
            self.stage('ready', source_metric_errors=metric_errors)
            return {'protocol': protocol['protocol'], 'worker_id': self.worker_id, 'arm': arm,
                'pbs_job_id': os.environ['PBS_JOBID'], 'passed': True, 'observed_modules': 614,
                'case_id': row['case_id'], 'seed_index': 0, 'sampler_seed': seed,
                'source_job': source['source_job'], 'source_endpoint': source['endpoint'],
                'source_metrics_passed': True, 'source_metric_errors': metric_errors,
                'source_metrics': metrics, 'uncached_cached_action_max_abs': action_error,
                'uncached_cached_future_max_abs': future_error,
                'actual_infer_action_max_abs': actual_error, 'startup_queries': 3,
                'new_training_updates': 0, 'cached_transformed_weights': True,
                'install_metadata': self.install_metadata}

    return Runtime


class R128Runtime:
    """Keep controller and metadata tests independent of Torch imports."""
    def __new__(cls, *args, **kwargs):
        return _runtime_type()(*args, **kwargs)

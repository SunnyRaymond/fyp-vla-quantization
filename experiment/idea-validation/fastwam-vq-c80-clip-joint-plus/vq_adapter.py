"""Install the frozen C80 clipping or joint VQ deployment in the episode runtime."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


ARMS = ('clip_only', 'joint')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _finite(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f'Invalid finite {label}')
    return float(value)


def _plan_rows(protocol):
    plan_path = Path(protocol['sources']['plan_file'])
    plan = read_json(plan_path)
    case_ids = protocol['calibration']['case_ids']
    if (plan.get('protocol') != 'fastwam-vq-calibration-suites-v1'
            or plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c80') != case_ids):
        raise RuntimeError('Frozen C80 calibration plan differs from the protocol')
    rows = {int(row['case_id']): row for row in plan.get('inputs', [])}
    if len(rows) != len(plan.get('inputs', [])) or not set(case_ids).issubset(rows):
        raise RuntimeError('Frozen calibration plan has duplicate or missing C80 rows')
    return plan_path, rows


def validate_reference_action(reference, protocol, arm, row):
    ref = protocol['closed_loop_reference']
    seed_index = ref['seed_index']
    sampler_seed = row['sampler_seeds'][seed_index]
    expected = {
        'schema_version': 1, 'protocol': protocol['protocol'], 'arm': arm,
        'case_id': ref['case_id'], 'seed_index': seed_index,
        'sampler_seed': sampler_seed, 'target_linears': 614,
    }
    if any(reference.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'C80 reference action binding differs for {arm}')
    if reference.get('plan_row') != row:
        raise RuntimeError(f'C80 reference action plan row differs for {arm}')
    parity = reference.get('actual_infer_parity', {})
    maximum = parity.get('max_abs')
    if (parity.get('passed') is not True or isinstance(maximum, bool)
            or not isinstance(maximum, (int, float)) or not math.isfinite(maximum)
            or maximum < 0 or maximum > ref['max_abs_tolerance']):
        raise RuntimeError(f'C80 saved reference action parity failed for {arm}')
    action = reference.get('action')
    if (not isinstance(action, list) or len(action) != 32
            or any(not isinstance(values, list) or len(values) != 7 for values in action)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for values in action for value in values)):
        raise RuntimeError(f'C80 reference action must be finite [32,7] for {arm}')
    return reference


def _expected_source(protocol):
    sources = protocol['sources']
    return {
        'source_c80_protocol': protocol['source_c80_protocol']['protocol'],
        'source_c80_root': sources['c80_root'],
        'source_c80_training_job': sources['clip_only_training_job'],
        'baseline_root': sources['baseline_root'],
        'baseline_arm': sources['baseline_arm'],
        'baseline_training_job': sources['baseline_training_job'],
        'baseline_validation_job': sources['baseline_validation_job'],
        'clip_root': sources['clip_root'],
        'runtime_artifacts': sources['runtime_artifacts'],
        'bank_directory': protocol['calibration']['bank_directory'],
        'teacher_directory': sources['teacher_directory'],
        'plan_file': sources['plan_file'],
    }


def validate_training_artifacts(protocol, arm, training_dir):
    """Check JSON metadata only; deployment tensors are inspected after runtime startup."""
    if arm not in ARMS:
        raise ValueError(f'Unknown C80 arm: {arm}')
    training_dir = Path(training_dir)
    receipt_path = training_dir / 'training_receipt.json'
    deployment_path = training_dir / 'deployment.pt'
    reference_action_path = training_dir / 'reference_action.json'
    receipt = read_json(receipt_path)
    if not deployment_path.is_file() or not reference_action_path.is_file():
        raise FileNotFoundError(f'C80 deployment/reference action is missing for {arm}')

    calibration = protocol['calibration']
    training = protocol['training']
    expected = {
        'schema_version': 1, 'protocol': protocol['protocol'], 'arm': arm,
        'status': 'complete', 'complete': True, 'updates': 80, 'epochs': 1,
        'target_linears': 614, 'calibration_ids': calibration['case_ids'],
        'update_order': calibration['update_order'], 'rho': 1.5,
        'weights_unchanged': arm == 'clip_only',
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'C80 training receipt is incomplete or differently bound for {arm}')
    normalization = _finite(receipt.get('normalization_c'), 'normalization_c')
    if not math.isclose(normalization, training['normalization_c'], rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f'C80 normalization differs from the frozen C8 baseline for {arm}')
    if any(receipt.get('source', {}).get(key) != value
           for key, value in _expected_source(protocol).items()):
        raise RuntimeError(f'C80 parent/source binding differs for {arm}')

    plan_path, rows = _plan_rows(protocol)
    reference = read_json(reference_action_path)
    row = rows[protocol['closed_loop_reference']['case_id']]
    validate_reference_action(reference, protocol, arm, row)
    return {'training_receipt': receipt, 'reference_action': reference,
            'reference_action_path': reference_action_path, 'deployment_path': deployment_path,
            'training_dir': training_dir, 'plan_path': plan_path, 'reference_row': row}


def _load_clip_banks(path):
    spec = importlib.util.spec_from_file_location('c80_clip_banks_runtime', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load C80 ClipBanks implementation: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ClipBanks


def _runtime_type():
    import runner as pilot

    class _TrainedRuntime(pilot.Runtime):
        def __init__(self, out, protocol, arm, training_dir, preflight):
            import torch
            import batched_transform as core

            torch.backends.cuda.matmul.allow_tf32 = False
            super().__init__(out)
            self.arm('bf16')
            self.protocol, self.arm_name = protocol, arm
            self.preflight = preflight
            self.transform_implementation = core.IMPLEMENTATION
            if self.transform_implementation != 'batched_h128_v2':
                raise RuntimeError(f'Unexpected transform implementation: {self.transform_implementation}')
            self.current_arm = arm
            self.run_queries = 0
            self.linear_calls = 0

            deployment_path = Path(training_dir) / 'deployment.pt'
            deployment = torch.load(deployment_path, map_location='cpu', weights_only=True)
            self._validate_deployment(torch, deployment, protocol, arm, preflight)
            clip_banks = _load_clip_banks(Path(protocol['sources']['clip_root']) / 'clip_banks.py')
            # Keep the clipping-ratio forward path active for both deployments;
            # the joint codebooks/log row gains are already in module_masters.
            self.banks = clip_banks(self, protocol['calibration']['bank_directory'],
                                    deployment['module_masters'], 'clip_only', protocol['clipping'])
            self.banks.restore_clips(deployment['ratios'])
            for values in (self.banks.books.values(), self.banks.logs.values(), self.banks.ratios.values()):
                for parameter in values:
                    parameter.requires_grad_(False)
            self.seen = self.banks.seen
            for name, module in self.q.modules.items():
                if module.weight.dtype != torch.bfloat16 or not bool(torch.isfinite(module.weight).all()):
                    raise RuntimeError(f'C80 decoded BF16 VQ weight is invalid: {name}')
                trained_forward = module.forward

                def counted_forward(value, _forward=trained_forward):
                    self.linear_calls += 1
                    return _forward(value)

                module.forward = counted_forward
            self.forwards = self.banks.forwards
            self.install_metadata = self._make_install_metadata(protocol, arm, preflight)
            pilot.save(Path(out) / 'runtime_c80.json', {
                'snapshot_phase': 'c80_trained_deployment_installed_before_action_replay',
                'install_metadata': self.install_metadata,
                'resident_weight_precision': 'decoded_bf16', 'kv_cache_dtype': 'bf16',
                'native_int4_gemm_calls': 0,
            })
            del deployment
            torch.cuda.empty_cache()

        @staticmethod
        def _validate_deployment(torch, deployment, protocol, arm, preflight):
            config = protocol['arm_configs'][arm]
            expected = {
                'schema_version': 1, 'protocol': protocol['protocol'], 'arm': arm,
                'updates': 80, 'epochs': 1, 'target_linears': 614,
                'normalization_c': protocol['training']['normalization_c'],
                'rho': config['rho'], 'case_ids': protocol['calibration']['case_ids'],
            }
            if any(deployment.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f'C80 deployment identity differs for {arm}')
            if deployment.get('source') != preflight['training_receipt'].get('source'):
                raise RuntimeError(f'C80 deployment source binding differs from its receipt for {arm}')
            masters, ratios = deployment.get('module_masters'), deployment.get('ratios')
            if not isinstance(masters, dict) or not isinstance(ratios, dict):
                raise RuntimeError('C80 deployment must contain module masters and clipping ratios')
            if len(masters) != 614 or set(ratios) != set(masters):
                raise RuntimeError('C80 deployment must cover exactly 614 master/ratio pairs')
            for name, value in masters.items():
                books, logs = value.get('books'), value.get('log_scales')
                if (not torch.is_tensor(books) or tuple(books.shape) != (2, 256, 4)
                        or books.dtype != torch.float32 or not bool(torch.isfinite(books).all())
                        or not torch.is_tensor(logs) or logs.ndim != 1
                        or logs.dtype != torch.float32 or not bool(torch.isfinite(logs).all())):
                    raise RuntimeError(f'Invalid C80 FP32 master tensors for {name}')
                ratio = ratios[name]
                if (not torch.is_tensor(ratio) or ratio.numel() != 1 or ratio.dtype != torch.float32
                        or not bool(torch.isfinite(ratio).all())
                        or not 0.5 <= float(ratio) <= 1.0):
                    raise RuntimeError(f'Invalid C80 FP32 clipping ratio for {name}')
            ref = preflight['reference_action']
            if ref.get('target_linears') != 614:
                raise RuntimeError('C80 reference action does not cover all target Linears')

        def _make_install_metadata(self, protocol, arm, preflight):
            config = protocol['arm_configs'][arm]
            receipt = preflight['training_receipt']
            return {
                'protocol': protocol['protocol'], 'arm': arm,
                'source_bank': protocol['calibration']['bank_directory'],
                'source_protocol': 'fastwam-vq-calibration-suites-v1', 'source_cohort': 'c80',
                'source_case_ids': list(protocol['calibration']['case_ids']),
                'calibration_ids': list(protocol['calibration']['case_ids']),
                'initialization_protocol': 'fastwam-vq-c8-action-latent-plus-v1',
                'initialization_cohort': 'c8', 'initialization_arm': 'rho1.5',
                'initialization_updates': 8,
                'initialization_training_job': protocol['sources']['baseline_training_job'],
                'module_count': 614, 'execution': 'decoded_bf16_vq',
                'activation_bits': 4, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
                'native_int4_gemm_calls': 0, 'rho': config['rho'], 'epochs': 1, 'updates': 80,
                'normalization_c': protocol['training']['normalization_c'],
                'train_weights': config['train_weights'], 'train_clips': config['train_clips'],
                'transform_implementation': 'batched_h128_v2', 'smooth_alpha': 0.5,
                'hadamard_block': 128, 'scale_block': 32, 'fixed_indices': True,
                'fixed_scale_exponents': True, 'deployed_books_dtype': 'bfloat16',
                'deployed_row_scales_dtype': 'bfloat16',
                'weights_unchanged': receipt['weights_unchanged'],
                'ratios': self.banks.clip_summary(),
            }

        def validate_reference(self):
            import numpy as np
            import torch
            import calibration_source as source

            preflight = self.preflight
            plan_path, row = preflight['plan_path'], preflight['reference_row']
            observation, metadata = source.load_observation(np, plan_path.parent, row)
            reference = preflight['reference_action']
            if reference.get('metadata') is not None and metadata != reference['metadata']:
                raise RuntimeError('C80 source observation metadata differs from trained reference action')
            if reference.get('description') is not None and metadata.get('description') != reference['description']:
                raise RuntimeError('C80 reference description differs from the frozen plan observation')
            seed = reference['sampler_seed']
            self.description = metadata['description']
            self.banks.seen.clear()
            action, _ = self.infer(self.datum(observation), seed, measure=False)
            expected = torch.tensor(reference['action'], dtype=torch.float32)
            delta = action.detach().float().cpu() - expected
            max_abs = float(delta.abs().max())
            passed = max_abs <= self.protocol['closed_loop_reference']['max_abs_tolerance']
            receipt = {
                'protocol': self.protocol['protocol'], 'arm': self.arm_name,
                'case_id': self.protocol['closed_loop_reference']['case_id'],
                'seed_index': self.protocol['closed_loop_reference']['seed_index'],
                'sampler_seed': seed, 'pbs_job_id': os.environ['PBS_JOBID'],
                'max_abs_vs_reference': max_abs,
                'tolerance': self.protocol['closed_loop_reference']['max_abs_tolerance'],
                'passed': passed, 'observed_modules': len(self.banks.seen),
                'reference_actual_infer_parity': reference['actual_infer_parity'],
                'training_receipt_status': preflight['training_receipt']['status'],
                'install_metadata': self.install_metadata,
            }
            if len(self.banks.seen) != 614 or not passed:
                raise RuntimeError(f'C80 trained action replay parity failed: max_abs={max_abs}')
            return receipt

        def infer(self, datum, seed, measure=True):
            self.banks.seen.clear()
            action, metrics = super().infer(datum, seed, measure=measure)
            if len(self.banks.seen) != 614:
                raise RuntimeError(f'Only {len(self.banks.seen)}/614 C80 Linears ran in this query')
            if self.q._native_int4_gemm_calls != 0 or self.q._kv_active:
                raise RuntimeError('C80 decoded VQ unexpectedly enabled native INT4 or KV4')
            if measure:
                metrics.update(transform_implementation=self.transform_implementation,
                               pbs_job_id=os.environ['PBS_JOBID'], first_query_in_run=self.run_queries == 0,
                               sampler_seed=seed)
                self.run_queries += 1
            return action, metrics

        def episode(self, row, index):
            started = time.monotonic()
            calls_before = self.linear_calls
            obs = self.task(row, index)
            cap = 700 if row['suite'] == 'libero_10' else 400
            steps, replans, success = 0, 0, False
            metrics, action_sample = [], []
            while steps < cap and not success:
                action, metric = self.infer(self.datum(obs), pilot.seed_for(index, replans))
                metric['replan'] = replans
                metrics.append(metric)
                if replans < 3:
                    action_sample.append(action.float().cpu().tolist())
                commands = self.command(action)
                for command in commands[:min(10, cap - steps)]:
                    obs, _, done, _ = self.env.step(command.tolist())
                    steps += 1
                    if done:
                        success = True
                        break
                replans += 1
                if replans % 5 == 0:
                    print(f'REPLAN variant={row["variant_id"]} arm={self.current_arm} index={replans} steps={steps}', flush=True)
            self.env.close()
            self.env = None
            evidence = {**self.install_metadata, 'observed_module_count': len(self.banks.seen),
                        'quantized_linear_calls': self.linear_calls - calls_before,
                        'native_int4_gemm_calls': self.q._native_int4_gemm_calls,
                        'kv_packed_prefills': self.q._kv_packed_prefills,
                        'kv_layer_reads': self.q._kv_layer_reads}
            if evidence['quantized_linear_calls'] <= 0:
                raise RuntimeError('Episode did not execute C80 decoded-BF16 VQ/A4 Linears')
            return {
                'protocol': self.protocol['protocol'], 'variant_id': str(row['variant_id']), 'index': index,
                'suite': row['suite'], 'dimension': row['dimension'], 'task_id': int(row['task_id']),
                'original_task': row.get('original_task'), 'subtype': row.get('subtype'),
                'arm': self.current_arm, 'complete': True, 'success': success,
                'termination': 'success' if success else 'control_step_timeout', 'steps': steps,
                'replans': replans, 'episode_seconds': time.monotonic() - started,
                'env_seed': pilot.seed_for(index, environment=True), 'sampler_seed': pilot.seed_for(index),
                'state_id': 0, 'initial_state_path': self.init_states_path,
                'description': self.description, 'metrics': metrics, 'first_action_chunks': action_sample,
                'transform_implementation': self.transform_implementation,
                'pbs_job_id': os.environ['PBS_JOBID'], 'vq_evidence': evidence,
            }

        def close(self):
            if self.env is not None:
                self.env.close()
                self.env = None
            for name, forward in self.forwards.items():
                self.q.modules[name].forward = forward

    return _TrainedRuntime


class TrainedRuntime:
    """Lazy runtime factory so metadata-only imports never load Torch or runner."""

    def __new__(cls, *args, **kwargs):
        return _runtime_type()(*args, **kwargs)

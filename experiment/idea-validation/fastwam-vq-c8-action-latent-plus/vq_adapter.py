"""Install a trained C8 fixed-index VQ deployment on the frozen episode runtime."""
from __future__ import annotations

import collections
import json
import math
import os
from pathlib import Path
import time

import runner as pilot


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_c8_metadata(protocol):
    calibration = protocol['calibration']
    suites = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
    return {
        'protocol': 'fastwam-vq-calibration-suites-v1',
        'cohort': 'c8',
        'source_job_id': calibration['source_job_id'],
        'case_ids': list(calibration['case_ids']),
        'suite_case_counts': {suite: 2 for suite in suites},
        'bank_directory': calibration['bank_directory'],
    }


def reference_binding(protocol):
    calibration = protocol['calibration']
    root = Path(calibration['root']).resolve()
    plan_file = Path(calibration['plan_file'])
    if not plan_file.is_absolute():
        plan_file = root / plan_file
    return {
        'protocol': protocol['protocol'],
        'remote_root': protocol['remote_root'],
        'calibration': {
            'root': str(root), 'plan_file': str(plan_file.resolve()),
            'case_ids': list(calibration['case_ids']),
            'source_job_id': calibration['source_job_id'],
            'bank_directory': str(Path(calibration['bank_directory']).resolve()),
            'cohort': 'c8', 'domain': 'original', 'seed_index': 0,
        },
    }


def _finite_positive(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'Invalid positive finite {label}')
    return float(value)


def _check_source(protocol, reference_dir):
    calibration = protocol['calibration']
    binding = reference_binding(protocol)
    source_summary_path = Path(calibration['root']) / 'results' / 'c8' / 'summary.json'
    source_summary = read_json(source_summary_path)
    expected_bank = Path(calibration['bank_directory']).resolve()
    if (source_summary.get('status') != 'complete'
            or source_summary.get('protocol') != 'fastwam-vq-calibration-suites-v1'
            or source_summary.get('cohort') != 'c8'
            or source_summary.get('pbs_jobid') != calibration['source_job_id']
            or source_summary.get('target_linears') != 614
            or source_summary.get('fitted_linears') != 614
            or source_summary.get('calibration_case_ids') != calibration['case_ids']
            or Path(source_summary.get('bank_directory', '')).resolve() != expected_bank
            or not expected_bank.is_dir()):
        raise RuntimeError('C8 source summary or fixed bank differs from the protocol')

    reference_dir = Path(reference_dir)
    cache_path = reference_dir / 'reference_cache.pt'
    receipt_path = reference_dir / 'reference_receipt.json'
    if not cache_path.is_file():
        raise FileNotFoundError(f'Prepared C8 reference cache is missing: {cache_path}')
    receipt = read_json(receipt_path)
    if (receipt.get('schema_version') != 1 or receipt.get('protocol') != protocol['protocol']
            or receipt.get('status') != 'complete' or receipt.get('complete') is not True
            or receipt.get('passed') is not True or receipt.get('binding') != binding
            or receipt.get('context_count') != 8 or receipt.get('target_linears') != 614
            or receipt.get('suite_case_counts') != source_c8_metadata(protocol)['suite_case_counts']
            or receipt.get('source_c8') != source_c8_metadata(protocol)):
        raise RuntimeError('Prepared C8 teacher/reference receipt is incomplete or differently bound')
    for branch in ('teacher', 'parent'):
        parity = receipt.get('parity', {}).get(branch, {})
        if set(parity) != {str(case) for case in calibration['case_ids']}:
            raise RuntimeError(f'C8 {branch} parity receipt does not cover the frozen eight cases')
        for value in parity.values():
            measured = value.get('pilot_infer_vs_sampler', {})
            max_abs, rmse = measured.get('max_abs'), measured.get('rmse')
            if (value.get('passed') is not True
                    or isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
                    or not math.isfinite(max_abs) or max_abs > protocol['training']['reference_max_abs_tolerance']
                    or isinstance(rmse, bool) or not isinstance(rmse, (int, float))
                    or not math.isfinite(rmse) or rmse > protocol['training']['reference_rmse_tolerance']):
                raise RuntimeError(f'C8 {branch} parity failed')
    normalization = receipt.get('normalization', {})
    normalization_c = _finite_positive(normalization.get('normalization_c'), 'normalization_c')
    if normalization.get('calibration_ids') != calibration['case_ids']:
        raise RuntimeError('C8 normalization is not bound to the frozen eight cases')
    return {'binding': binding, 'source_c8': source_c8_metadata(protocol),
            'source_summary': source_summary, 'reference_receipt': receipt,
            'normalization_c': normalization_c}


def _training_binding(protocol, arm, source, normalization_c):
    config = protocol['arm_configs'][arm]
    training = protocol['training']
    return {
        'reference_binding': reference_binding(protocol),
        'arm': arm,
        'rho': config['rho'],
        'epochs': config['epochs'],
        'learning_rate': training['learning_rate'],
        'gradient_norm_clip': training['gradient_norm_clip'],
        'optimizer_reset_after_update1': training['optimizer_reset_after_update1'],
        'source_c8': source,
        'normalization_c': normalization_c,
        'latent_weight': config['rho'] * normalization_c,
        'case_ids': list(protocol['calibration']['case_ids']),
    }


def validate_training_artifacts(protocol, arm, training_dir, reference_dir):
    if arm not in protocol['arm_configs']:
        raise ValueError(f'No frozen training config for {arm}')
    source = _check_source(protocol, reference_dir)
    training_dir = Path(training_dir)
    receipt = read_json(training_dir / 'training_receipt.json')
    deployment_path = training_dir / 'deployment.pt'
    reference_action_path = training_dir / 'reference_action.json'
    if not deployment_path.is_file() or not reference_action_path.is_file():
        raise FileNotFoundError(f'Trained deployment/reference action is missing for {arm}')
    config, c8, normalization_c = protocol['arm_configs'][arm], protocol['calibration'], source['normalization_c']
    expected = {
        'status': 'complete', 'complete': True, 'arm': arm, 'rho': config['rho'],
        'epochs': config['epochs'], 'updates': config['updates'], 'target_linears': 614,
        'case_ids': c8['case_ids'], 'source_c8': source['source_c8'],
        'optimizer_reset_after_update1': protocol['training']['optimizer_reset_after_update1'],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'C8 training receipt is incomplete or differently bound for {arm}')
    if receipt.get('protocol') != protocol['protocol']:
        raise RuntimeError(f'C8 training protocol differs for {arm}')
    if receipt.get('reference_binding') != source['binding']:
        raise RuntimeError(f'C8 training reference binding differs for {arm}')
    binding = receipt.get('binding')
    expected_binding = _training_binding(protocol, arm, source['source_c8'], normalization_c)
    if binding != expected_binding:
        raise RuntimeError(f'C8 training binding differs for {arm}')
    receipt_c = _finite_positive(receipt.get('normalization_c'), 'training normalization_c')
    latent_weight = _finite_positive(receipt.get('latent_weight'), 'training latent_weight')
    if (not math.isclose(receipt_c, normalization_c, rel_tol=1e-9, abs_tol=1e-12)
            or not math.isclose(latent_weight, config['rho'] * normalization_c,
                                rel_tol=1e-9, abs_tol=1e-12)):
        raise RuntimeError(f'C8 training normalization/lambda differs for {arm}')
    coverage = receipt.get('gradient_coverage', {})
    if (coverage.get('target_linears') != 614 or coverage.get('connected_linears') != 614
            or coverage.get('finite_linears') != 614 or not 0 < coverage.get('nonzero_gradient_linears', 0) <= 614):
        raise RuntimeError(f'C8 training gradient coverage is incomplete for {arm}')
    counts = receipt.get('counts', {})
    if (counts.get('reference_contexts') != 8 or counts.get('gradient_updates') != config['updates']
            or counts.get('reference_action_sampler_forwards') != 1
            or counts.get('reference_action_pilot_infer_calls') != 1
            or counts.get('teacher_forwards') != 0 or counts.get('training_performed') is not True
            or receipt.get('epochs_completed') != config['epochs']):
        raise RuntimeError(f'C8 training counts are incomplete for {arm}')
    receipt_parity = receipt.get('reference_action_parity', {})
    receipt_max_abs = receipt_parity.get('max_abs')
    if (isinstance(receipt_max_abs, bool) or not isinstance(receipt_max_abs, (int, float))
            or not math.isfinite(receipt_max_abs)
            or receipt_max_abs > protocol['source_parity_max_abs_tolerance']):
        raise RuntimeError(f'C8 training receipt reference-action parity failed for {arm}')

    reference = read_json(reference_action_path)
    row = _c8_rows(protocol)[protocol['reference_case_id']]
    seed = row['sampler_seeds'][protocol['reference_seed_index']]
    recipe = reference.get('recipe', {})
    if (reference.get('schema_version') != 1 or reference.get('protocol') != protocol['protocol']
            or reference.get('arm') != arm or reference.get('rho') != config['rho']
            or reference.get('case_id') != protocol['reference_case_id']
            or reference.get('seed_index') != protocol['reference_seed_index']
            or reference.get('sampler_seed') != seed
            or reference.get('source_c8') != source['source_c8']
            or reference.get('plan_row') != row
            or recipe.get('normalization_c') != normalization_c
            or recipe.get('latent_weight') != config['rho'] * normalization_c
            or recipe.get('epochs') != config['epochs'] or recipe.get('updates') != config['updates']
            or recipe.get('optimizer_reset_after_update1') is not True):
        raise RuntimeError(f'C8 trained reference action binding differs for {arm}')
    parity = reference.get('actual_infer_parity', {})
    tolerance = protocol['source_parity_max_abs_tolerance']
    max_abs = parity.get('max_abs')
    if (parity.get('passed', True) is False or isinstance(max_abs, bool)
            or not isinstance(max_abs, (int, float)) or not math.isfinite(max_abs)
            or max_abs > tolerance):
        raise RuntimeError(f'C8 trained reference action sampler/infer parity failed for {arm}')
    action = reference.get('action')
    if (not isinstance(action, list) or len(action) != 32
            or any(not isinstance(row_action, list) or len(row_action) != 7 for row_action in action)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for row_action in action for value in row_action)):
        raise RuntimeError(f'C8 trained reference action has invalid [32,7] shape for {arm}')
    return {'source': source, 'training_receipt': receipt, 'deployment_path': deployment_path,
            'reference_action': reference, 'reference_action_path': reference_action_path,
            'training_dir': training_dir}


def _c8_rows(protocol):
    binding = reference_binding(protocol)
    plan_path = Path(binding['calibration']['plan_file'])
    plan = read_json(plan_path)
    case_ids = protocol['calibration']['case_ids']
    if (plan.get('protocol') != 'fastwam-vq-calibration-suites-v1' or plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c8') != case_ids):
        raise RuntimeError('Frozen C8 plan is incomplete or differently bound')
    rows = {int(row['case_id']): row for row in plan.get('inputs', [])}
    if len(rows) != len(plan.get('inputs', [])) or not set(case_ids).issubset(rows):
        raise RuntimeError('Frozen C8 plan has duplicate or missing cases')
    selected = [rows[case] for case in case_ids]
    counts = collections.Counter(row.get('suite') for row in selected)
    if counts != collections.Counter({name: 2 for name in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')}):
        raise RuntimeError('Frozen C8 plan no longer has two contexts per suite')
    return rows


class TrainedRuntime(pilot.Runtime):
    def __init__(self, out, protocol, arm, training_dir, preflight):
        pilot.guard()
        import torch
        import batched_transform as core
        from train_banks import TrainableBanks

        torch.backends.cuda.matmul.allow_tf32 = False
        super().__init__(out)
        self.arm('bf16')
        self.protocol, self.arm_name = protocol, arm
        self.transform_implementation = core.IMPLEMENTATION
        if self.transform_implementation != 'batched_h128_v2':
            raise RuntimeError(f'Unexpected batched transform implementation: {self.transform_implementation}')
        self.current_arm = arm
        self.run_queries = 0
        self.linear_calls = 0
        self.banks = TrainableBanks(self, protocol['calibration']['bank_directory'])
        self.seen = self.banks.seen
        self.preflight = {**preflight, 'protocol': protocol, 'arm': arm,
                          'module_shapes': {
                              name: (tuple(self.banks.books[name].shape), tuple(self.banks.logs[name].shape))
                              for name in self.banks.modules}}
        deployment_path = Path(training_dir) / 'deployment.pt'
        deployment = torch.load(deployment_path, map_location='cpu', weights_only=True)
        self._validate_deployment(torch, deployment, self.preflight)
        state = {}
        for name in self.q.modules:
            state[name] = {
                'books': deployment['module_masters'][name]['books'].to('cuda'),
                'log_scales': deployment['module_masters'][name]['log_scales'].to('cuda'),
            }
        self.banks.restore(state)
        with torch.no_grad():
            for name, module in self.q.modules.items():
                if module.weight.dtype != torch.bfloat16 or not bool(torch.isfinite(module.weight).all()):
                    raise RuntimeError(f'Trained BF16 VQ decode is invalid: {name}')
                trained_forward = module.forward

                def counted_forward(value, _forward=trained_forward):
                    self.linear_calls += 1
                    return _forward(value)

                module.forward = counted_forward
        self.forwards = self.banks.forwards
        self.install_metadata = self._make_install_metadata(protocol, arm, preflight, deployment)
        pilot.save(out / 'runtime_c8.json', {
            'snapshot_phase': 'trained_c8_deployment_installed_before_action_replay',
            'install_metadata': self.install_metadata,
            'resident_weight_precision': 'decoded_bf16', 'kv_cache_dtype': 'bf16',
        })
        del deployment, state
        torch.cuda.empty_cache()

    @staticmethod
    def _validate_deployment(torch, deployment, preflight):
        protocol, arm = preflight['protocol'], preflight['arm']
        config = protocol['arm_configs'][arm]
        c8, source = protocol['calibration'], preflight['source']['source_c8']
        expected_fixed = {
            'books': 'FP32 master; BF16 deployed',
            'row_scales': 'FP32 log master; BF16 deployed',
            'indices': 'frozen uint8',
            'scale_exponents': 'frozen uint8; scale_block=32',
            'transform': {'name': 'smooth05_hadamard', 'alpha': 0.5, 'hadamard_block': 128},
            'activation': {'name': 'dynamic A4', 'qmax': 7, 'row_scale': 'absmax'},
            'biases': 'frozen',
            'non_target_model_parameters': 'frozen',
        }
        expected = {
            'schema_version': 1, 'protocol': protocol['protocol'], 'arm': arm,
            'rho': config['rho'], 'epochs': config['epochs'], 'updates': config['updates'],
            'target_linears': 614, 'case_ids': c8['case_ids'], 'source_c8': source,
            'reference_binding': preflight['source']['binding'], 'fixed': expected_fixed,
        }
        if any(deployment.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f'Trained C8 deployment identity/config differs for {arm}')
        normalization_c = _finite_positive(deployment.get('normalization_c'), 'deployment normalization_c')
        latent_weight = _finite_positive(deployment.get('latent_weight'), 'deployment latent_weight')
        if (not math.isclose(normalization_c, preflight['source']['normalization_c'], rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(latent_weight, config['rho'] * normalization_c,
                                    rel_tol=1e-9, abs_tol=1e-12)):
            raise RuntimeError(f'Trained C8 deployment lambda differs for {arm}')
        masters = deployment.get('module_masters')
        modules = preflight['module_shapes']
        if not isinstance(masters, dict) or set(masters) != set(modules) or len(masters) != 614:
            raise RuntimeError('Trained C8 deployment must cover the exact 614 target module names')
        for name, (book_shape, scale_shape) in modules.items():
            if book_shape != (2, 256, 4):
                raise RuntimeError(f'C8 bank book shape is invalid for trained master module {name}')
            value = masters[name]
            books, logs = value.get('books'), value.get('log_scales')
            if (not torch.is_tensor(books) or tuple(books.shape) != book_shape or books.dtype != torch.float32
                    or not bool(torch.isfinite(books).all())
                    or not torch.is_tensor(logs) or tuple(logs.shape) != scale_shape or logs.dtype != torch.float32
                    or not bool(torch.isfinite(logs).all())):
                raise RuntimeError(f'Invalid FP32 master tensors for trained C8 module {name}')

    @staticmethod
    def _make_install_metadata(protocol, arm, preflight, deployment):
        config = protocol['arm_configs'][arm]
        return {
            'arm': arm,
            'source_bank': protocol['calibration']['bank_directory'],
            'source_job_id': protocol['calibration']['source_job_id'],
            'source_protocol': 'fastwam-vq-calibration-suites-v1',
            'source_cohort': 'c8',
            'source_case_ids': list(protocol['calibration']['case_ids']),
            'module_count': 614,
            'execution': 'decoded_bf16_vq',
            'activation_bits': 4,
            'kv_cache_dtype': 'bf16',
            'native_low_bit_kernel': False,
            'rho': config['rho'],
            'epochs': config['epochs'],
            'updates': config['updates'],
            'normalization_c': float(deployment['normalization_c']),
            'latent_weight': float(deployment['latent_weight']),
            'transform_implementation': 'batched_h128_v2',
            'smooth_alpha': 0.5,
            'hadamard_block': 128,
            'scale_block': 32,
            'fixed_indices': True,
            'fixed_scale_exponents': True,
            'deployed_books_dtype': 'bfloat16',
            'deployed_row_scales_dtype': 'bfloat16',
            'reference_binding': preflight['source']['binding'],
            'training_deployment': str(preflight['deployment_path']),
        }

    def validate_reference(self):
        import numpy as np
        import torch
        import calibration_source as source

        row = _c8_rows(self.protocol)[self.protocol['reference_case_id']]
        plan_path = Path(reference_binding(self.protocol)['calibration']['plan_file'])
        observation, metadata = source.load_observation(np, plan_path.parent, row)
        reference = self.preflight['reference_action']
        if metadata != reference.get('metadata'):
            raise RuntimeError('C8 source observation metadata differs from trained reference action')
        seed_index = self.protocol['reference_seed_index']
        seed = row['sampler_seeds'][seed_index]
        if reference.get('sampler_seed') != seed:
            raise RuntimeError('C8 trained reference action uses a different sampler seed')
        self.description = metadata['description']
        self.banks.seen.clear()
        action, _ = self.infer(self.datum(observation), seed, measure=False)
        expected = torch.tensor(reference['action'], dtype=torch.float32)
        delta = action.detach().float().cpu() - expected
        max_abs = float(delta.abs().max())
        tolerance = self.protocol['source_parity_max_abs_tolerance']
        receipt = {
            'protocol': self.protocol['protocol'], 'arm': self.arm_name,
            'case_id': self.protocol['reference_case_id'], 'seed_index': seed_index,
            'sampler_seed': seed, 'pbs_job_id': os.environ['PBS_JOBID'],
            'max_abs_vs_reference': max_abs, 'tolerance': tolerance,
            'passed': max_abs <= tolerance, 'observed_modules': len(self.banks.seen),
            'reference_actual_infer_parity': reference['actual_infer_parity'],
            'source_c8': self.preflight['source']['source_c8'],
            'training_receipt_status': self.preflight['training_receipt']['status'],
            'install_metadata': self.install_metadata,
        }
        if len(self.banks.seen) != 614 or not receipt['passed']:
            raise RuntimeError(f'Trained C8 action replay parity failed: max_abs={max_abs}')
        return receipt

    def infer(self, datum, seed, measure=True):
        self.banks.seen.clear()
        action, metrics = super().infer(datum, seed, measure=measure)
        if len(self.banks.seen) != 614:
            raise RuntimeError(f'Only {len(self.banks.seen)}/614 trained C8 Linears ran in this query')
        if self.q._native_int4_gemm_calls != 0 or self.q._kv_active:
            raise RuntimeError('Trained VQ unexpectedly enabled native INT4 or KV4')
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
            raise RuntimeError('Episode did not execute trained C8 VQ/A4 Linears')
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

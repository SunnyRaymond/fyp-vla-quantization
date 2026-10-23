"""Stdlib checks for the frozen C80 workers and durable episode recovery."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import (  # noqa: E402
    ARMS, COHORT_PROTOCOL, PROTOCOL, allocation_elapsed_seconds,
    environment_seed, load_manifest, recover_records, sampler_seed,
    validate_install_metadata, validate_protocol, validate_record,
    worker_arm, worker_coverage, worker_indices,
)


def fixture_protocol():
    return json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))


def fixture_metadata(protocol, arm='clip_only'):
    config = protocol['arm_configs'][arm]
    return {
        'protocol': PROTOCOL, 'arm': arm,
        'source_bank': protocol['calibration']['bank_directory'],
        'source_protocol': 'fastwam-vq-calibration-suites-v1',
        'source_cohort': 'c80',
        'source_case_ids': protocol['calibration']['case_ids'],
        'calibration_ids': protocol['calibration']['case_ids'],
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
        'deployed_row_scales_dtype': 'bfloat16', 'weights_unchanged': arm == 'clip_only',
        'ratios': {'count': 614, 'min': 0.5, 'mean': 0.9, 'max': 1.0, 'below_one': 500},
    }


def fixture_manifest():
    variants = []
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        for dimension in ('d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'):
            for task_id in range(10):
                index = len(variants)
                variants.append({'index': index, 'variant_id': f'variant-{index}',
                                 'suite': suite, 'dimension': dimension,
                                 'task_id': task_id, 'state_id': 0})
    return {'protocol': COHORT_PROTOCOL, 'variants': variants}


def valid_record(rows, index, worker_id, query_index, run_id='pbs-a', first_in_run=True,
                 *, arm='clip_only', protocol=None, metadata=None):
    protocol = protocol or fixture_protocol()
    metadata = metadata or fixture_metadata(protocol, arm)
    row = rows[index]
    metric = {
        'query_index_in_arm': query_index, 'first_query_in_arm': query_index == 0,
        'first_query_in_run': first_in_run, 'transform_implementation': 'batched_h128_v2',
        'pbs_job_id': run_id, 'sampler_seed': sampler_seed(index, 0), 'replan': 0,
        'denoising_latency_ms': 10.0, 'full_query_wall_ms': 12.0,
        'peak_allocated_gpu_GB': 2.0, 'baseline_allocated_gpu_GB': 1.0,
        'peak_reserved_gpu_GB': 3.0,
    }
    return {
        'protocol': PROTOCOL, 'arm': arm, 'worker_id': worker_id, 'complete': True,
        'index': index, 'variant_id': row['variant_id'], 'suite': row['suite'],
        'dimension': row['dimension'], 'task_id': row['task_id'], 'state_id': 0,
        'env_seed': environment_seed(index), 'sampler_seed': sampler_seed(index),
        'success': False, 'episode_seconds': 1.0, 'replans': 1,
        'transform_implementation': 'batched_h128_v2', 'pbs_job_id': run_id,
        'vq_evidence': {**metadata, 'observed_module_count': 614,
                        'native_int4_gemm_calls': 0, 'kv_packed_prefills': 0,
                        'kv_layer_reads': 0, 'quantized_linear_calls': 614},
        'metrics': [metric],
    }


class EvaluationControlTests(unittest.TestCase):
    def setUp(self):
        self.protocol = fixture_protocol()
        self.manifest = fixture_manifest()
        self.rows = self.manifest['variants']
        self.metadata = fixture_metadata(self.protocol)

    def test_frozen_protocol_and_worker_pairing(self):
        self.assertIs(validate_protocol(self.protocol), self.protocol)
        changed = json.loads(json.dumps(self.protocol))
        changed['inference']['settling_steps'] += 1
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed = json.loads(json.dumps(self.protocol))
        changed['worker_assignment']['variants_per_worker'] = 140
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(self.manifest), encoding='utf-8')
            _, rows = load_manifest(path)
        self.assertEqual(len(rows), 280)
        assignments = {arm: [] for arm in ARMS}
        for worker in range(4):
            indices = worker_indices(worker, 'clip_only')
            self.assertEqual(len(indices), 70)
            self.assertEqual(indices, list(range(worker, 280, 4)))
            assignments['clip_only'].extend(indices)
        for arm in ARMS:
            if arm == 'joint':
                assignments[arm] = [i for worker in range(4) for i in worker_indices(worker, arm)]
            self.assertEqual(sorted(assignments[arm]), list(range(280)))
        self.assertEqual([worker_arm(i, 'joint') for i in range(4)], ['joint'] * 4)
        self.assertEqual(environment_seed(17), 800_017_000)
        self.assertEqual(sampler_seed(17), 900_017_000)
        self.assertEqual(sampler_seed(17, 4), 900_017_004)
        self.assertEqual(allocation_elapsed_seconds(1000, 86500), 85500)
        self.assertEqual(worker_coverage(3, 'clip_only', [3, 7])['completed_indices'], [3, 7])

    def test_episode_evidence_binds_arm_weights_clipping_and_seed(self):
        self.assertEqual(validate_install_metadata(self.metadata, self.protocol, 'clip_only'),
                         self.metadata)
        record = valid_record(self.rows, 0, 0, 0, arm='clip_only', protocol=self.protocol, metadata=self.metadata)
        self.assertEqual(validate_record(record, self.rows, 0, self.protocol, self.metadata,
                                         'clip_only'), 0)
        wrong_arm = dict(self.metadata, train_weights=True)
        with self.assertRaises(ValueError):
            validate_install_metadata(wrong_arm, self.protocol, 'clip_only')
        wrong_freeze_receipt = dict(self.metadata, weights_unchanged=False)
        with self.assertRaises(ValueError):
            validate_install_metadata(wrong_freeze_receipt, self.protocol, 'clip_only')
        joint_metadata = fixture_metadata(self.protocol, 'joint')
        joint_record = valid_record(self.rows, 1, 1, 0, arm='joint',
                                    protocol=self.protocol, metadata=joint_metadata)
        self.assertEqual(validate_record(joint_record, self.rows, 1, self.protocol,
                                         joint_metadata, 'joint'), 1)
        wrong_ratio = dict(self.metadata, ratios={'count': 614, 'min': 0.4,
                                                  'mean': 0.8, 'max': 1.0})
        with self.assertRaises(ValueError):
            validate_install_metadata(wrong_ratio, self.protocol, 'clip_only')
        wrong_seed = json.loads(json.dumps(record))
        wrong_seed['sampler_seed'] += 1
        with self.assertRaises(ValueError):
            validate_record(wrong_seed, self.rows, 0, self.protocol, self.metadata, 'clip_only')

    def test_recovery_keeps_only_durable_prefix_and_run_first_markers(self):
        indices = worker_indices(0, 'clip_only')
        first = valid_record(self.rows, indices[0], 0, 0, first_in_run=True,
                             arm='clip_only',
                             protocol=self.protocol, metadata=self.metadata)
        second = valid_record(self.rows, indices[1], 0, 1, first_in_run=False,
                              arm='clip_only',
                              protocol=self.protocol, metadata=self.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl'
            path.write_bytes((json.dumps(first) + '\n' + json.dumps(second) + '\n').encode('utf-8')
                             + b'{"index":')
            self.assertEqual(list(recover_records(path, self.rows, 0, self.protocol,
                                                  self.metadata, 'clip_only')), indices[:2])
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 2)
            gap = valid_record(self.rows, indices[3], 0, 2, first_in_run=False,
                               arm='clip_only',
                               protocol=self.protocol, metadata=self.metadata)
            path.write_text(json.dumps(first) + '\n' + json.dumps(gap) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                recover_records(path, self.rows, 0, self.protocol, self.metadata, 'clip_only')


if __name__ == '__main__':
    unittest.main()

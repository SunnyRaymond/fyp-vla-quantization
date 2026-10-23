"""Stdlib-only checks for frozen worker coverage, evidence, and resume control."""
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
    return {
        'protocol': PROTOCOL,
        'arms': list(ARMS),
        'arm_configs': {
            'rho0.1': {'rho': 0.1, 'epochs': 2, 'updates': 16},
            'rho1.5': {'rho': 1.5, 'epochs': 1, 'updates': 8},
        },
        'calibration': {
            'bank_directory': '/frozen/c8-bank', 'source_job_id': '25789685.pbs101',
            'case_ids': [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070],
        },
    }


def fixture_metadata(protocol, arm='rho0.1'):
    config = protocol['arm_configs'][arm]
    return {
        'arm': arm,
        'source_bank': protocol['calibration']['bank_directory'],
        'source_job_id': protocol['calibration']['source_job_id'],
        'source_protocol': 'fastwam-vq-calibration-suites-v1',
        'source_cohort': 'c8',
        'source_case_ids': protocol['calibration']['case_ids'],
        'module_count': 614,
        'execution': 'decoded_bf16_vq',
        'activation_bits': 4,
        'kv_cache_dtype': 'bf16',
        'native_low_bit_kernel': False,
        'rho': config['rho'],
        'epochs': config['epochs'],
        'updates': config['updates'],
        'normalization_c': 0.5,
        'latent_weight': config['rho'] * 0.5,
        'transform_implementation': 'batched_h128_v2',
        'smooth_alpha': 0.5,
        'hadamard_block': 128,
        'scale_block': 32,
        'fixed_indices': True,
        'fixed_scale_exponents': True,
        'deployed_books_dtype': 'bfloat16',
        'deployed_row_scales_dtype': 'bfloat16',
        'reference_binding': {'fixture': True},
        'training_deployment': '/training/rho0.1/deployment.pt',
    }


def fixture_manifest():
    variants = []
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        for dimension in ('d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'):
            for cell_variant in range(10):
                index = len(variants)
                variants.append({'index': index, 'variant_id': f'variant-{index}', 'suite': suite,
                                 'dimension': dimension, 'task_id': cell_variant, 'state_id': 0})
    return {'protocol': COHORT_PROTOCOL, 'variants': variants}


def valid_record(rows, index, worker_id, query_index, run_id='pbs-a', first_in_run=True,
                 *, protocol=None, metadata=None):
    protocol = protocol or fixture_protocol()
    arm = worker_arm(worker_id)
    metadata = metadata or fixture_metadata(protocol, arm)
    row = rows[index]
    metric = {
        'query_index_in_arm': query_index,
        'first_query_in_arm': query_index == 0,
        'first_query_in_run': first_in_run,
        'transform_implementation': 'batched_h128_v2',
        'pbs_job_id': run_id,
        'sampler_seed': sampler_seed(index, 0),
        'replan': 0,
        'denoising_latency_ms': 10.0,
        'full_query_wall_ms': 12.0,
        'peak_allocated_gpu_GB': 2.0,
        'baseline_allocated_gpu_GB': 1.0,
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

    def test_workers_partition_both_frozen_arms_and_seed_formulas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(self.manifest), encoding='utf-8')
            _, rows = load_manifest(path)
        self.assertEqual(len(rows), 280)
        assigned = {arm: [] for arm in ARMS}
        for worker in range(4):
            arm = worker_arm(worker)
            indices = worker_indices(worker)
            self.assertEqual(len(indices), 140)
            self.assertEqual(indices, list(range(worker % 2, 280, 2)))
            assigned[arm].extend(indices)
        for arm in ARMS:
            self.assertEqual(sorted(assigned[arm]), list(range(280)))
        self.assertEqual(environment_seed(17), 800_017_000)
        self.assertEqual(sampler_seed(17), 900_017_000)
        self.assertEqual(sampler_seed(17, 4), 900_017_004)
        self.assertEqual(allocation_elapsed_seconds(1000, 86500), 85500)
        partial = worker_coverage(3, [1, 3, 5])
        self.assertEqual(partial['indices'], list(range(1, 280, 2)))
        self.assertEqual(partial['completed_indices'], [1, 3, 5])

    def test_protocol_keeps_frozen_idm_and_c8_binding(self):
        protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
        self.assertIs(validate_protocol(protocol), protocol)
        changed = json.loads(json.dumps(protocol))
        changed['inference']['settling_steps'] = 29
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_trained_metadata_and_episode_evidence_are_bound(self):
        index, worker = 0, 0
        self.assertEqual(validate_install_metadata(self.metadata, self.protocol, 'rho0.1'), self.metadata)
        record = valid_record(self.rows, index, worker, 0, protocol=self.protocol, metadata=self.metadata)
        self.assertEqual(validate_record(record, self.rows, worker, self.protocol, self.metadata), index)

        wrong_lambda = dict(self.metadata, latent_weight=99.0)
        with self.assertRaises(ValueError):
            validate_install_metadata(wrong_lambda, self.protocol, 'rho0.1')
        wrong_rho = json.loads(json.dumps(record))
        wrong_rho['vq_evidence']['rho'] = 1.5
        with self.assertRaises(ValueError):
            validate_record(wrong_rho, self.rows, worker, self.protocol, self.metadata)
        wrong_seed = json.loads(json.dumps(record))
        wrong_seed['sampler_seed'] += 1
        with self.assertRaises(ValueError):
            validate_record(wrong_seed, self.rows, worker, self.protocol, self.metadata)

    def test_resume_recovers_only_prefix_and_truncates_torn_tail(self):
        indices = worker_indices(0)
        first = valid_record(self.rows, indices[0], 0, 0, first_in_run=True,
                             protocol=self.protocol, metadata=self.metadata)
        second = valid_record(self.rows, indices[1], 0, 1, first_in_run=False,
                              protocol=self.protocol, metadata=self.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl'
            path.write_bytes((json.dumps(first) + '\n' + json.dumps(second) + '\n').encode('utf-8')
                             + b'{"index":')
            recovered = recover_records(path, self.rows, 0, self.protocol, self.metadata)
            self.assertEqual(list(recovered), indices[:2])
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 2)

            gap = valid_record(self.rows, indices[3], 0, 2, first_in_run=False,
                               protocol=self.protocol, metadata=self.metadata)
            path.write_text(json.dumps(first) + '\n' + json.dumps(gap) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                recover_records(path, self.rows, 0, self.protocol, self.metadata)

    def test_first_formal_query_per_resumed_run_is_separate_from_arm_index(self):
        indices = worker_indices(0)
        records = [
            valid_record(self.rows, indices[0], 0, 0, run_id='pbs-a', first_in_run=True,
                         protocol=self.protocol, metadata=self.metadata),
            valid_record(self.rows, indices[1], 0, 1, run_id='pbs-b', first_in_run=True,
                         protocol=self.protocol, metadata=self.metadata),
        ]
        self.assertFalse(records[1]['metrics'][0]['first_query_in_arm'])
        self.assertTrue(records[1]['metrics'][0]['first_query_in_run'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl'
            path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
            recovered = recover_records(path, self.rows, 0, self.protocol, self.metadata)
        self.assertEqual(list(recovered), indices[:2])


if __name__ == '__main__':
    unittest.main()

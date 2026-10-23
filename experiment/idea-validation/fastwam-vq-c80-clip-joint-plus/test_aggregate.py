"""Stdlib regression tests for compact controller-readable aggregates."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate  # noqa: E402
from evaluate import ARMS, PROTOCOL  # noqa: E402


class CompactAggregateTests(unittest.TestCase):
    def test_training_validation_supports_one_arm_and_checks_frozen_normalization(self):
        protocol = {'training': {'normalization_c': 0.028303017887797623}}

        def artifacts(_protocol, arm, path):
            return {'training_receipt': {
                'normalization_c': protocol['training']['normalization_c']}}

        with mock.patch('vq_adapter.validate_training_artifacts', side_effect=artifacts):
            for arms in (('clip_only',), ('joint',), ARMS):
                with self.subTest(arms=arms):
                    self.assertEqual(set(aggregate._validate_source_and_training(
                        protocol, Path('/results'), arms)), set(arms))
        with mock.patch('vq_adapter.validate_training_artifacts', return_value={
                'training_receipt': {'normalization_c': 0.1}}):
            for arms in (('clip_only',), ARMS):
                with self.subTest(arms=arms):
                    with self.assertRaisesRegex(ValueError, 'frozen C8 normalization_c'):
                        aggregate._validate_source_and_training(protocol, Path('/results'), arms)

    def test_c8_trained_loader_resets_query_state_for_each_worker(self):
        old_protocol = json.loads((HERE.parent / 'fastwam-vq-c8-action-latent-plus'
                                   / 'protocol.json').read_text(encoding='utf-8'))
        old_name = old_protocol['protocol']
        old_arms = tuple(old_protocol['arms'])
        calibration = old_protocol['calibration']
        normalization_c = 0.5
        rows = [{'variant_id': index, 'suite': f'suite-{index // 70}',
                 'dimension': f'd{index % 7}', 'task_id': index}
                for index in range(280)]
        root_summary = {
            'protocol': old_name, 'status': 'complete', 'complete': True,
            'episode_slots': 560, 'manifest_variants': 280,
            'new_arms': list(old_arms), 'arm_episodes': {arm: 280 for arm in old_arms},
            'training_receipts': {
                arm: {'protocol': old_name, 'arm': arm, 'status': 'complete',
                      'complete': True, 'rho': config['rho'], 'epochs': config['epochs'],
                      'updates': config['updates'], 'target_linears': 614,
                      'case_ids': calibration['case_ids']}
                for arm, config in old_protocol['arm_configs'].items()},
        }
        worker_data = {}
        job_id = 'test-job.pbs101'
        for worker in range(4):
            arm = old_arms[worker // 2]
            config = old_protocol['arm_configs'][arm]
            indices = list(range(worker % 2, 280, 2))
            metadata = {
                'arm': arm, 'source_bank': calibration['bank_directory'],
                'source_job_id': calibration['source_job_id'],
                'source_protocol': 'fastwam-vq-calibration-suites-v1',
                'source_cohort': 'c8', 'source_case_ids': calibration['case_ids'],
                'module_count': 614, 'execution': 'decoded_bf16_vq',
                'activation_bits': 4, 'kv_cache_dtype': 'bf16',
                'native_low_bit_kernel': False, 'rho': config['rho'],
                'epochs': config['epochs'], 'updates': config['updates'],
                'transform_implementation': 'batched_h128_v2', 'smooth_alpha': 0.5,
                'hadamard_block': 128, 'scale_block': 32, 'fixed_indices': True,
                'fixed_scale_exponents': True, 'deployed_books_dtype': 'bfloat16',
                'deployed_row_scales_dtype': 'bfloat16',
                'normalization_c': normalization_c,
                'latent_weight': config['rho'] * normalization_c,
            }
            records = []
            for query_index, index in enumerate(indices):
                records.append({
                    'index': index, 'protocol': old_name, 'worker_id': worker,
                    'arm': arm, 'complete': True, 'success': index % 2 == 0,
                    'variant_id': str(rows[index]['variant_id']),
                    'suite': rows[index]['suite'], 'dimension': rows[index]['dimension'],
                    'task_id': rows[index]['task_id'], 'state_id': 0,
                    'env_seed': 800_000_000 + index * 1000,
                    'sampler_seed': 900_000_000 + index * 1000,
                    'transform_implementation': 'batched_h128_v2',
                    'pbs_job_id': job_id, 'vq_evidence': {
                        **metadata, 'observed_module_count': 614,
                        'native_int4_gemm_calls': 0, 'kv_packed_prefills': 0,
                        'kv_layer_reads': 0, 'quantized_linear_calls': 614,
                    },
                    'replans': 1, 'episode_seconds': 1.0,
                    'metrics': [{
                        'query_index_in_arm': query_index,
                        'first_query_in_arm': query_index == 0,
                        'first_query_in_run': query_index == 0,
                        'replan': 0,
                        'sampler_seed': 900_000_000 + index * 1000,
                        'transform_implementation': 'batched_h128_v2',
                        'pbs_job_id': job_id,
                        'denoising_latency_ms': 10.0, 'full_query_wall_ms': 12.0,
                        'peak_allocated_gpu_GB': 2.0,
                        'baseline_allocated_gpu_GB': 1.0,
                        'peak_reserved_gpu_GB': 3.0,
                    }],
                })
            worker_summary = {
                'protocol': old_name, 'worker_id': worker, 'arm': arm,
                'status': 'complete', 'complete': True, 'episodes': 140,
                'expected': 140, 'indices': indices, 'completed_indices': indices,
                'arm_episodes': {arm: 140}, 'install_metadata': metadata,
            }
            worker_data[worker] = {
                'summary': worker_summary, 'progress': worker_summary,
                'setup_receipt': {
                    'passed': True, 'protocol': old_name, 'arm': arm,
                    'observed_modules': 614, 'training_receipt_status': 'complete',
                    'install_metadata': metadata, 'max_abs_vs_reference': 0.0,
                },
                'records': records,
            }

        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory) / 'results'

            def read_json(path):
                path = Path(path)
                if path.name == 'protocol.json':
                    return old_protocol
                if path == results_root / 'summary.json':
                    return root_summary
                worker = int(path.parent.name.removeprefix('worker_'))
                return worker_data[worker][path.stem]

            def read_jsonl(path):
                worker = int(Path(path).parent.name.removeprefix('worker_'))
                return enumerate(worker_data[worker]['records'], 1)

            with mock.patch.object(aggregate, 'read_json', side_effect=read_json), \
                    mock.patch.object(aggregate, '_read_jsonl', side_effect=read_jsonl):
                found = aggregate._load_c8_trained(results_root, rows)
                self.assertEqual(set(found), {
                    (arm, index) for arm in old_arms for index in range(280)})
                self.assertEqual(len(found), 560)

                worker_data[2]['records'][0]['metrics'][0]['first_query_in_run'] = False
                with self.assertRaisesRegex(ValueError, 'query timing identity'):
                    aggregate._load_c8_trained(results_root, rows)

    def test_complete_summary_omits_legacy_c8_statistics_and_training_histories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c8_results = root / 'c8' / 'results'
            c8_results.mkdir(parents=True)
            c8_protocol = {
                'protocol': 'fastwam-vq-c8-action-latent-plus-v1',
                'arms': ['rho0.1', 'rho1.5'],
                'calibration': {
                    'cohort': 'c8', 'source_job_id': '25789685.pbs101',
                    'case_ids': list(range(8)), 'bank_directory': '/banks/c8',
                },
                'arm_configs': {
                    'rho0.1': {'rho': 0.1, 'epochs': 2, 'updates': 16},
                    'rho1.5': {'rho': 1.5, 'epochs': 1, 'updates': 8},
                },
            }
            (c8_results.parent / 'protocol.json').write_text(
                json.dumps(c8_protocol), encoding='utf-8')
            large_old_summary = {
                'protocol': c8_protocol['protocol'], 'status': 'complete', 'complete': True,
                'episode_slots': 560, 'manifest_variants': 280,
                'new_arms': ['rho0.1', 'rho1.5'],
                'arm_episodes': {'rho0.1': 280, 'rho1.5': 280},
                'normalization_c': 0.028303017887797623,
                'suites': {'legacy': 'large' * 5000},
                'dimensions': {'legacy': 'large' * 5000},
                'cells': {'legacy': 'large' * 30000},
                'timing': {'legacy': 'large' * 5000},
            }
            (c8_results / 'summary.json').write_text(
                json.dumps(large_old_summary), encoding='utf-8')

            protocol_path = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
            protocol_path['references']['c8_trained_results'] = str(c8_results)
            protocol_path['references']['c8_trained_job_id'] = '25822971.pbs101'
            candidates = ARMS
            reference_arms = [
                'bf16', 'quarot_adapted_w4a4', 'spinquant_adapted_w4a4',
                'smooth05_hadamard_scalar_w4a4', 'row_scale_vq_a4',
                'micro_scale_vq_a4', 'rho0.1', 'rho1.5',
            ]
            rows = [{'suite': f'suite-{i // 70}', 'dimension': f'd{i % 7}'}
                    for i in range(280)]

            def episode(index, success=False):
                metric = {
                    'first_query_in_run': True, 'full_query_wall_ms': 12.0,
                    'denoising_latency_ms': 10.0, 'peak_allocated_gpu_GB': 2.0,
                    'baseline_allocated_gpu_GB': 1.0, 'peak_reserved_gpu_GB': 3.0,
                }
                return {
                    'success': success, 'transform_implementation': 'batched_h128_v2',
                    'pbs_job_id': f'job-{index}', 'episode_seconds': 1.0, 'metrics': [metric],
                }

            records = {(arm, index): episode(index, index % 2 == 0)
                       for arm in candidates for index in range(280)}
            by_arm = {
                arm: {index: episode(index, index % 3 == 0) for index in range(280)}
                for arm in (*candidates, *reference_arms)
            }
            training = {}
            for arm in candidates:
                training_dir = root / 'training' / arm
                training_dir.mkdir(parents=True)
                training[arm] = {
                    'training_dir': training_dir,
                    'training_receipt': {
                        'schema_version': 1, 'protocol': PROTOCOL, 'arm': arm,
                        'status': 'complete', 'complete': True, 'updates': 80,
                        'epochs': 1, 'target_linears': 614, 'calibration_ids': list(range(2000, 2080)),
                        'update_order': list(range(2000, 2080)),
                        'normalization_c': 0.028303017887797623, 'rho': 1.5,
                        'weights_unchanged': arm == 'clip_only',
                        'history': ['large-history-' * 10000],
                        'source': {'plan_file': '/frozen/plan.json'},
                    },
                    'reference_action': {'actual_infer_parity': {'max_abs': 0.0, 'passed': True}},
                }

            summary = aggregate._make_summary(
                protocol_path, rows, records, by_arm, {arm: {} for arm in candidates}, {},
                training, [], candidates=candidates, status='complete',
                reference_slots=2240)
            compact_c8 = summary['source_receipts']['c8_training_summary']
            self.assertNotIn('suites', compact_c8)
            self.assertNotIn('dimensions', compact_c8)
            self.assertNotIn('cells', compact_c8)
            self.assertNotIn('timing', compact_c8)
            self.assertNotIn('history', summary['training_receipts']['clip_only'])
            self.assertEqual(summary['source_receipts']['c80_training_receipt_paths'], {
                arm: str(training[arm]['training_dir'] / 'training_receipt.json')
                for arm in candidates})

            size = aggregate.save_compact_summary(root / 'summary.json', summary)
            self.assertLessEqual(size, aggregate.MAX_SUMMARY_BYTES)
            serialized = (root / 'summary.json').read_text(encoding='utf-8')
            self.assertNotIn('large-history-', serialized)
            self.assertNotIn('largelegacy', serialized)
            self.assertIn('cells', summary)  # Keep this experiment's full current cell statistics.


if __name__ == '__main__':
    unittest.main()

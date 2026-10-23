from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate
import evaluate


def fixture_rows():
    rows = []
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        for dimension in ('d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'):
            for task_id in range(10):
                index = len(rows)
                rows.append({'index': index, 'variant_id': f'variant-{index}', 'suite': suite,
                             'dimension': dimension, 'task_id': task_id, 'state_id': 0})
    return rows


def metadata_fixture(arm):
    return {
        'protocol': evaluate.PROTOCOL, 'arm': arm,
        'weight_bits': 4, 'activation_bits': 4, 'group_size': 128,
        'execution': 'decoded_bf16_reference', 'kv_cache_dtype': 'bf16',
        'native_low_bit_kernel': False, 'module_count': 614,
        'calibration_contexts': 80, 'optimizer_updates': 0,
        'encoded_target_bpw': 4.40 if arm == evaluate.ARMS[0] else 4.45,
        'encoded_target_bytes': 3_400_000 if arm == evaluate.ARMS[0] else 3_450_000,
        'target_weight_parameters': 6_000_000,
        'low_rank_global_rank': 8 if arm == evaluate.ARMS[0] else 0,
        'representation': ('svdquant_low_rank_residual' if arm == evaluate.ARMS[0]
                           else 'viditq_channel_balance_rotation'),
        'transform_implementation': ('svdquant_smoothed_low_rank_v1' if arm == evaluate.ARMS[0]
                                     else 'viditq_structured_hadamard_v1'),
        'bank_manifest_pbs_jobid': 'bank-job-1',
        'extra_metadata_bytes': {'smoothing_vectors': 1024},
    }


def setup_fixtures():
    return {str(worker): {
        'protocol': evaluate.PROTOCOL, 'worker_id': worker,
        'arm': evaluate.worker_arm(worker), 'pbs_job_id': 'pbs-job-1', 'passed': True,
        'observed_modules': 614, 'bf16_teacher_max_abs': 0.0,
        'cross_worker_action_max_abs': 0.0,
        'component_selfcheck': {'passed': True, 'max_abs': 0.0},
        'bank_manifest': {'protocol': evaluate.PROTOCOL, 'pbs_job_id': 'bank-job-1',
                          'complete': True, 'calibration_contexts': 80, 'modules': 614},
        'install_metadata': metadata_fixture(evaluate.worker_arm(worker)),
    } for worker in range(4)}


def timed_record(index, arm):
    return {
        'success': index % 2 == 0,
        'transform_implementation': metadata_fixture(arm)['transform_implementation'],
        'pbs_job_id': 'pbs-job-1', 'episode_seconds': 1.0,
        'metrics': [{
            'first_query_in_run': index == 0, 'denoising_latency_ms': 10.0,
            'full_query_wall_ms': 12.0, 'peak_allocated_gpu_GB': 2.0,
            'baseline_allocated_gpu_GB': 1.0, 'peak_reserved_gpu_GB': 3.0,
        }],
    }


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = evaluate.validate_protocol(evaluate.read_json(HERE / 'protocol.json'))
        self.rows = fixture_rows()
        self.reference_arms = list(self.protocol['references']['arms'])
        self.references = {(arm, index): {'success': index % 3 == 0}
                           for arm in self.reference_arms for index in range(280)}

    def test_reference_slot_count_and_paired_win_loss_tie_counts(self):
        self.assertEqual(len(aggregate.validate_reference_slots(
            self.references, self.reference_arms)), 2240)
        missing = dict(self.references)
        missing.pop((self.reference_arms[0], 0))
        with self.assertRaisesRegex(ValueError, '2240'):
            aggregate.validate_reference_slots(missing, self.reference_arms)
        stats = aggregate._paired(
            {0: {'success': True}, 1: {'success': False}, 2: {'success': True}},
            {0: {'success': False}, 1: {'success': True}, 2: {'success': True}},
            [0, 1, 2, 3])
        self.assertEqual(stats, {
            'wins': 1, 'losses': 1, 'ties': 1, 'episodes': 3,
            'success_rate_difference': 0.0,
        })

    def test_worker_setup_metadata_and_shared_bank_contract(self):
        metadata = {arm: metadata_fixture(arm) for arm in evaluate.ARMS}
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            for worker in range(4):
                arm = evaluate.worker_arm(worker)
                worker_root = results / 'workers' / f'worker_{worker}'
                worker_root.mkdir(parents=True)
                receipt = {
                    'protocol': evaluate.PROTOCOL, 'worker_id': worker, 'arm': arm,
                    'status': 'checkpointed', 'complete': False, 'episodes': 0,
                    'completed': 0, 'expected': 140, 'soft_deadline_reached': True,
                    'indices': evaluate.worker_indices(worker), 'completed_indices': [],
                    'arm_episodes': {arm: 0}, 'install_metadata': metadata[arm],
                }
                setup = {
                    'protocol': evaluate.PROTOCOL, 'worker_id': worker, 'arm': arm,
                    'pbs_job_id': 'pbs-job-1', 'passed': True, 'observed_modules': 614,
                    'bf16_teacher_max_abs': 0.0, 'cross_worker_action_max_abs': 0.0,
                    'component_selfcheck': {'passed': True},
                    'bank_manifest': {
                        'protocol': evaluate.PROTOCOL, 'complete': True,
                        'pbs_job_id': 'bank-job-1', 'modules': 614,
                        'calibration_contexts': 80, 'relative_bpw_gap': 0.05 / 4.45,
                    },
                    'install_metadata': metadata[arm],
                }
                (worker_root / 'progress.json').write_text(json.dumps(receipt), encoding='utf-8')
                (worker_root / 'summary.json').write_text(json.dumps(receipt), encoding='utf-8')
                (worker_root / 'setup_receipt.json').write_text(json.dumps(setup), encoding='utf-8')
            loaded = aggregate.load_workers(results, self.protocol, {}, fixture_rows())
            records, by_arm, installed, setups, checkpointed, checks = loaded
            self.assertEqual(records, {})
            self.assertEqual(by_arm, {evaluate.ARMS[0]: {}, evaluate.ARMS[1]: {}})
            self.assertEqual(installed, metadata)
            self.assertEqual(setups.keys(), {'0', '1', '2', '3'})
            self.assertEqual(checkpointed, [0, 1, 2, 3])
            self.assertAlmostEqual(checks['encoded_target_bpw_relative_gap'], 0.05 / 4.45)

            worker_one = results / 'workers' / 'worker_1'
            broken = json.loads((worker_one / 'summary.json').read_text(encoding='utf-8'))
            broken['install_metadata'] = dict(metadata[evaluate.ARMS[0]], encoded_target_bytes=1)
            (worker_one / 'summary.json').write_text(json.dumps(broken), encoding='utf-8')
            (worker_one / 'progress.json').write_text(json.dumps(broken), encoding='utf-8')
            setup = json.loads((worker_one / 'setup_receipt.json').read_text(encoding='utf-8'))
            setup['install_metadata'] = broken['install_metadata']
            (worker_one / 'setup_receipt.json').write_text(json.dumps(setup), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different install metadata'):
                aggregate.load_workers(results, self.protocol, {}, fixture_rows())

    def test_partial_then_complete_writes_checkpoint_then_completion_marker(self):
        metadata = {arm: metadata_fixture(arm) for arm in evaluate.ARMS}
        setups = setup_fixtures()
        cross_worker = {
            'bank_manifest_pbs_jobidstr': 'bank-job-1',
            'encoded_target_bpw_relative_gap': 0.05 / 4.45,
            'transform_implementations': {arm: metadata[arm]['transform_implementation']
                                          for arm in evaluate.ARMS},
        }
        partial_records = {(evaluate.ARMS[0], 0): timed_record(0, evaluate.ARMS[0])}
        partial_by_arm = {evaluate.ARMS[0]: {0: partial_records[evaluate.ARMS[0], 0]},
                          evaluate.ARMS[1]: {}}
        complete_records = {
            (arm, index): timed_record(index, arm)
            for arm in evaluate.ARMS for index in range(280)}
        complete_by_arm = {
            arm: {index: complete_records[arm, index] for index in range(280)}
            for arm in evaluate.ARMS}
        partial = (partial_records, partial_by_arm, metadata, setups,
                   [0, 1, 2, 3], cross_worker)
        complete = (complete_records, complete_by_arm, metadata, setups, [], cross_worker)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'results'
            results.mkdir()
            protocol_path, manifest_path = root / 'protocol.json', root / 'manifest.json'
            protocol_path.write_text(json.dumps(self.protocol), encoding='utf-8')
            manifest_path.write_text(json.dumps({
                'protocol': self.protocol['cohort']['manifest_protocol'],
                'variants': self.rows,
            }), encoding='utf-8')
            args = SimpleNamespace(protocol=protocol_path, manifest=manifest_path,
                                   results=results)
            with mock.patch.object(aggregate, '_load_references',
                                   return_value=(self.references, self.reference_arms)), \
                    mock.patch.object(aggregate, 'load_workers', side_effect=[partial, complete]):
                first = aggregate.run(args)
                summary_path = results / 'summary.json'
                checkpoint = results / 'PIPELINE_CHECKPOINTED'
                self.assertEqual(first['status'], 'checkpointed')
                self.assertFalse(first['complete'])
                self.assertEqual(first['episode_slots'], 1)
                self.assertEqual(first['arm_counts'], {evaluate.ARMS[0]: 1,
                                                       evaluate.ARMS[1]: 0})
                self.assertEqual(first['reference_episode_slots'], 2240)
                self.assertTrue(checkpoint.is_file())
                self.assertFalse((results / 'PIPELINE_COMPLETE').exists())
                self.assertEqual(first['overall']['paired'][evaluate.ARMS[0]]
                                 [self.reference_arms[0]]['episodes'], 1)
                self.assertEqual(first['overall']['paired'][evaluate.ARMS[1]]
                                 [self.reference_arms[0]]['episodes'], 0)
                self.assertLess(summary_path.stat().st_size, aggregate.MAX_SUMMARY_BYTES)
                checkpoint_value = json.loads(checkpoint.read_text(encoding='utf-8'))
                self.assertEqual(checkpoint_value['episode_slots'], 1)
                self.assertFalse(checkpoint_value['complete'])

                final = aggregate.run(args)
                marker = json.loads((results / 'PIPELINE_COMPLETE').read_text(encoding='utf-8'))
                self.assertEqual(final['status'], 'complete')
                self.assertEqual(final['episode_slots'], 560)
                self.assertEqual(final['arm_episodes'], {arm: 280 for arm in evaluate.ARMS})
                self.assertEqual(final['arm_counts'], {arm: 280 for arm in evaluate.ARMS})
                self.assertEqual(final['reference_episode_slots'], 2240)
                self.assertEqual(marker['episode_slots'], 560)
                self.assertEqual(marker['reference_episode_slots'], 2240)
                self.assertTrue(marker['complete'])
                self.assertFalse(checkpoint.exists())
                paired = final['overall']['paired'][evaluate.ARMS[0]][evaluate.ARMS[1]]
                self.assertEqual(paired['episodes'], 280)
                self.assertEqual(paired['wins'] + paired['losses'] + paired['ties'], 280)


if __name__ == '__main__':
    unittest.main()

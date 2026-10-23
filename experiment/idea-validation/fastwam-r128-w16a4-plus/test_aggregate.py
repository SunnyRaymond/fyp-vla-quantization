"""Stdlib-only aggregate, checkpoint, reference-slot, and paired-stat checks."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import aggregate
import evaluate

HERE = Path(__file__).resolve().parent


def fixture_rows():
    rows = []
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        for dimension in ('d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'):
            for task_id in range(10):
                index = len(rows)
                rows.append({'index': index, 'variant_id': f'variant-{index}', 'suite': suite,
                             'dimension': dimension, 'task_id': task_id, 'state_id': 0})
    return rows


def metadata_fixture(protocol, arm):
    metadata = evaluate.expected_install_metadata(protocol, arm)
    metadata.update({
        'orthogonality': {'action': 2e-6, 'video': 3e-6},
        'rotation_movement': ({'action': 0.0, 'video': 0.0} if arm == 'H_W16A4'
                              else {'action': 0.12, 'video': 0.08}),
    })
    return metadata


def setup_fixture(protocol, worker, metadata):
    arm = evaluate.worker_arm(worker)
    source = protocol['arm_sources'][arm]
    return {
        'protocol': evaluate.PROTOCOL, 'worker_id': worker, 'arm': arm,
        'pbs_job_id': '25910000.pbs101', 'passed': True, 'observed_modules': 614,
        'case_id': 2000, 'seed_index': 0, 'sampler_seed': 910_004_000,
        'source_job': source['source_job'], 'source_endpoint': source['endpoint'],
        'source_metrics_passed': True,
        'source_metric_errors': {name: 0.0 for name in evaluate.SOURCE_METRICS},
        'uncached_cached_action_max_abs': 0.0, 'uncached_cached_future_max_abs': 0.0,
        'actual_infer_action_max_abs': 0.0, 'startup_queries': 3,
        'new_training_updates': 0, 'cached_transformed_weights': True,
        'install_metadata': metadata,
    }


def timed_record(rows, worker, index, metadata, query_index, pbs_job='25910000.pbs101'):
    arm, row = evaluate.worker_arm(worker), rows[index]
    transform = metadata['transform_implementation']
    evidence = dict(metadata)
    evidence.update({
        'observed_module_count': 614, 'kv_packed_prefills': 0, 'kv_layer_reads': 0,
        'quantized_linear_calls': 614,
    })
    metric = {
        'replan': 0, 'query_index_in_arm': query_index,
        'first_query_in_arm': query_index == 0, 'first_query_in_run': query_index == 0,
        'sampler_seed': evaluate.sampler_seed(index), 'pbs_job_id': pbs_job,
        'transform_implementation': transform,
        'denoising_latency_ms': 10.0, 'full_query_wall_ms': 12.0,
        'peak_allocated_gpu_GB': 2.0, 'baseline_allocated_gpu_GB': 1.0,
        'peak_reserved_gpu_GB': 3.0,
    }
    return {
        'protocol': evaluate.PROTOCOL, 'arm': arm, 'worker_id': worker,
        'complete': True, 'index': index, 'suite': row['suite'],
        'dimension': row['dimension'], 'task_id': row['task_id'],
        'variant_id': row['variant_id'], 'state_id': 0,
        'env_seed': evaluate.environment_seed(index), 'sampler_seed': evaluate.sampler_seed(index),
        'success': index % 2 == 0, 'pbs_job_id': pbs_job,
        'transform_implementation': transform, 'episode_seconds': 1.0,
        'replans': 1, 'metrics': [metric], 'vq_evidence': evidence,
    }


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = evaluate.validate_protocol(evaluate.read_json(HERE / 'protocol.json'))
        self.rows = fixture_rows()
        self.reference_arms = list(self.protocol['references']['arms'])
        self.references = {
            (arm, index): {'success': index % 3 == 0}
            for arm in self.reference_arms for index in range(280)}
        self.metadata = {arm: metadata_fixture(self.protocol, arm) for arm in evaluate.ARMS}
        self.setups = {str(worker): setup_fixture(
            self.protocol, worker, self.metadata[evaluate.worker_arm(worker)])
            for worker in range(4)}

    def test_reference_slots_and_paired_win_loss_ties(self):
        self.assertEqual(len(aggregate.validate_reference_slots(
            self.references, self.reference_arms)), 3640)
        missing = dict(self.references)
        missing.pop((self.reference_arms[0], 0))
        with self.assertRaisesRegex(ValueError, '3640'):
            aggregate.validate_reference_slots(missing, self.reference_arms)
        stats = aggregate._paired(
            {0: {'success': True}, 1: {'success': False}, 2: {'success': True}},
            {0: {'success': False}, 1: {'success': True}, 2: {'success': True}},
            [0, 1, 2, 3])
        self.assertEqual(stats, {
            'wins': 1, 'losses': 1, 'ties': 1, 'episodes': 3,
            'success_rate_difference': 0.0,
        })

    def test_worker_loader_accepts_checkpointed_empty_prefixes(self):
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
                    'arm_episodes': {arm: 0}, 'install_metadata': self.metadata[arm],
                }
                (worker_root / 'progress.json').write_text(json.dumps(receipt), encoding='utf-8')
                (worker_root / 'summary.json').write_text(json.dumps(receipt), encoding='utf-8')
                (worker_root / 'setup_receipt.json').write_text(
                    json.dumps(self.setups[str(worker)]), encoding='utf-8')
            records, by_arm, installed, setups, checkpointed = aggregate.load_workers(
                results, self.protocol, {}, self.rows)
            self.assertEqual(records, {})
            self.assertEqual(by_arm, {'H_W16A4': {}, 'C80_R128_W16A4': {}})
            self.assertEqual(installed, self.metadata)
            self.assertEqual(set(setups), {'0', '1', '2', '3'})
            self.assertEqual(checkpointed, [0, 1, 2, 3])

            worker_one = results / 'workers' / 'worker_1'
            broken = json.loads((worker_one / 'summary.json').read_text(encoding='utf-8'))
            broken['indices'] = list(range(140))
            (worker_one / 'summary.json').write_text(json.dumps(broken), encoding='utf-8')
            (worker_one / 'progress.json').write_text(json.dumps(broken), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'frozen 140-slot shard'):
                aggregate.load_workers(results, self.protocol, {}, self.rows)

    def test_partial_then_complete_writes_control_compatible_markers(self):
        partial_by_arm = {arm: {} for arm in evaluate.ARMS}
        first = timed_record(self.rows, 0, 0, self.metadata['H_W16A4'], 0)
        partial_records = {('H_W16A4', 0): first}
        partial_by_arm['H_W16A4'][0] = first

        complete_records, complete_by_arm = {}, {arm: {} for arm in evaluate.ARMS}
        for worker in range(4):
            arm = evaluate.worker_arm(worker)
            for query_index, index in enumerate(evaluate.worker_indices(worker)):
                record = timed_record(self.rows, worker, index, self.metadata[arm], query_index)
                complete_records[arm, index] = record
                complete_by_arm[arm][index] = record

        partial = (partial_records, partial_by_arm, self.metadata, self.setups, [0, 1, 2, 3])
        complete = (complete_records, complete_by_arm, self.metadata, self.setups, [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'results'
            results.mkdir()
            protocol_path, manifest_path = root / 'protocol.json', root / 'manifest.json'
            protocol_path.write_text(json.dumps(self.protocol), encoding='utf-8')
            manifest_path.write_text(json.dumps({
                'protocol': self.protocol['cohort']['manifest_protocol'], 'variants': self.rows,
            }), encoding='utf-8')
            args = SimpleNamespace(protocol=protocol_path, manifest=manifest_path, results=results)
            with mock.patch.object(aggregate, '_load_references',
                                   return_value=(self.references, self.reference_arms)), \
                    mock.patch.object(aggregate, 'load_workers', side_effect=[partial, complete]):
                first_summary = aggregate.run(args)
                checkpoint = results / 'PIPELINE_CHECKPOINTED'
                self.assertEqual(first_summary['status'], 'checkpointed')
                self.assertEqual(first_summary['episode_slots'], 1)
                self.assertEqual(first_summary['arm_counts'], {'H_W16A4': 1, 'C80_R128_W16A4': 0})
                self.assertEqual(first_summary['reference_episode_slots'], 3640)
                self.assertTrue(checkpoint.is_file())
                self.assertFalse((results / 'PIPELINE_COMPLETE').exists())
                self.assertEqual(
                    first_summary['overall']['paired']['H_W16A4']['C80_R128_W16A4']['episodes'], 0)

                final = aggregate.run(args)
                marker = json.loads((results / 'PIPELINE_COMPLETE').read_text(encoding='utf-8'))
                summary_path = results / 'summary.json'
                self.assertEqual(final['status'], 'complete')
                self.assertEqual(final['episode_slots'], 560)
                expected = {'H_W16A4': 280, 'C80_R128_W16A4': 280}
                self.assertEqual(final['arm_episodes'], expected)
                self.assertEqual(final['arm_counts'], expected)
                self.assertEqual(final['reference_episode_slots'], 3640)
                self.assertEqual(marker['episode_slots'], 560)
                self.assertEqual(marker['reference_episode_slots'], 3640)
                self.assertTrue(marker['complete'])
                self.assertFalse(checkpoint.exists())
                self.assertEqual(
                    final['overall']['paired']['H_W16A4']['C80_R128_W16A4']['episodes'], 280)
                self.assertEqual(final['overall']['paired']['H_W16A4']['bf16']['episodes'], 280)
                self.assertEqual(len(final['cells']), 28)
                self.assertEqual(len(final['suites']), 4)
                self.assertEqual(len(final['dimensions']), 7)
                self.assertIn('signed_h128_cayley_cached_weights_v1',
                              final['timing']['H_W16A4']['by_implementation'])
                self.assertIn('original BF16 weights', final['timing_boundary'])
                self.assertNotIn('VQ/A4', final['timing_boundary'])
                self.assertLess(summary_path.stat().st_size, aggregate.MAX_SUMMARY_BYTES)


if __name__ == '__main__':
    unittest.main()

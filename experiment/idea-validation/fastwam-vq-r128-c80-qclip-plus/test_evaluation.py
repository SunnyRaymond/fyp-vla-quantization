"""Stdlib-only checks for frozen R128 workers, metadata, records, and recovery."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import evaluate

HERE = Path(__file__).resolve().parent


def manifest_rows():
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
        'rotation_movement': {'action': 0.12, 'video': 0.08},
        'clipping': evaluate.CLIPPING_SUMMARY.copy(),
    })
    return metadata


def setup_fixture(protocol, worker, metadata=None):
    arm = evaluate.worker_arm(worker)
    source = protocol['arm_sources'][arm]
    return {
        'protocol': evaluate.PROTOCOL, 'worker_id': worker, 'arm': arm,
        'pbs_job_id': '25910000.pbs101', 'passed': True, 'observed_modules': 614,
        'case_id': 2000, 'seed_index': 0, 'sampler_seed': 910_004_000,
        'source_job': source['source_job'], 'source_endpoint': source['endpoint'],
        'source_metrics_passed': True,
        'source_metric_errors': {key: 0.0 for key in evaluate.SOURCE_METRICS},
        'q0_anchor_passed': True, 'q0_action_max_abs': 0.0,
        'q0_future_max_abs': 0.0, 'actual_infer_action_max_abs': 0.0,
        'startup_queries': 4, 'new_training_updates': 0,
        'install_metadata': metadata or metadata_fixture(protocol, arm),
    }


def episode(protocol, rows, worker, index, metadata, query_start=0):
    arm, row = evaluate.worker_arm(worker), rows[index]
    job = '25910000.pbs101'
    transform = metadata['transform_implementation']
    metrics = []
    for replan in range(2):
        query = query_start + replan
        metrics.append({
            'replan': replan, 'query_index_in_arm': query,
            'first_query_in_arm': query == 0, 'first_query_in_run': query == 0,
            'sampler_seed': evaluate.sampler_seed(index, replan),
            'pbs_job_id': job, 'transform_implementation': transform,
            'denoising_latency_ms': 10.0, 'full_query_wall_ms': 12.0,
            'peak_allocated_gpu_GB': 2.0, 'baseline_allocated_gpu_GB': 1.0,
            'peak_reserved_gpu_GB': 3.0,
        })
    evidence = dict(metadata)
    evidence.update({
        'observed_module_count': 614, 'kv_packed_prefills': 0, 'kv_layer_reads': 0,
        'quantized_linear_calls': 614,
    })
    return {
        'protocol': evaluate.PROTOCOL, 'arm': arm, 'worker_id': worker, 'complete': True,
        'index': index, 'suite': row['suite'], 'dimension': row['dimension'],
        'task_id': row['task_id'], 'variant_id': row['variant_id'], 'state_id': 0,
        'env_seed': evaluate.environment_seed(index), 'sampler_seed': evaluate.sampler_seed(index),
        'success': index % 2 == 0, 'pbs_job_id': job,
        'transform_implementation': transform, 'episode_seconds': 1.25,
        'replans': len(metrics), 'metrics': metrics, 'vq_evidence': evidence,
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = evaluate.read_json(HERE / 'protocol.json')
        self.rows = manifest_rows()

    def test_protocol_worker_shards_and_reference_denominator_are_frozen(self):
        self.assertEqual(evaluate.validate_protocol(self.protocol)['protocol'], evaluate.PROTOCOL)
        self.assertEqual(evaluate.ARMS, ('qclip',))
        shards = [evaluate.worker_indices(worker) for worker in range(4)]
        self.assertEqual([evaluate.worker_arm(worker) for worker in range(4)], ['qclip'] * 4)
        self.assertTrue(all(len(shard) == 70 for shard in shards))
        self.assertEqual([shard[0] for shard in shards], [0, 1, 2, 3])
        self.assertEqual(sorted(set().union(*map(set, shards))), list(range(280)))
        changed = copy.deepcopy(self.protocol)
        changed['references']['expected_reference_slots'] = 4759
        with self.assertRaises(ValueError):
            evaluate.validate_protocol(changed)
        changed = copy.deepcopy(self.protocol)
        changed['source_initialization']['calibration']['case_ids'].pop()
        with self.assertRaises(ValueError):
            evaluate.validate_protocol(changed)

    def test_install_metadata_binds_decoded_mxvq4_and_rotation_evidence(self):
        metadata = metadata_fixture(self.protocol, 'qclip')
        self.assertEqual(evaluate.validate_install_metadata(metadata, self.protocol, 'qclip'), metadata)
        self.assertEqual(metadata['weight_bits'], 4)
        self.assertIs(metadata['vq_enabled'], True)
        self.assertEqual(metadata['execution'], 'decoded_bf16_dense_vq')
        self.assertEqual(metadata['new_training_updates'], 0)
        bad = copy.deepcopy(metadata)
        bad['native_low_bit_kernel'] = True
        with self.assertRaises(ValueError):
            evaluate.validate_install_metadata(bad, self.protocol, 'qclip')
        bad = copy.deepcopy(metadata)
        bad['orthogonality']['video'] = 1e-3
        with self.assertRaises(ValueError):
            evaluate.validate_install_metadata(bad, self.protocol, 'qclip')
        bad = copy.deepcopy(metadata)
        bad['clipping']['count'] = 613
        with self.assertRaises(ValueError):
            evaluate.validate_install_metadata(bad, self.protocol, 'qclip')

    def test_setup_receipt_enforces_full_anchor_and_infer_parity(self):
        for worker in range(4):
            arm = evaluate.worker_arm(worker)
            metadata = metadata_fixture(self.protocol, arm)
            setup = setup_fixture(self.protocol, worker, metadata)
            self.assertEqual(evaluate.validate_setup_receipt(
                setup, self.protocol, worker, arm, metadata), metadata)
            for key, value in (
                ('q0_action_max_abs', 1e-8),
                ('q0_future_max_abs', 1e-8),
                ('actual_infer_action_max_abs', 2e-6),
                ('new_training_updates', 1),
                ('startup_queries', 5),
                ('source_endpoint', '/wrong/endpoint.json'),
            ):
                bad = dict(setup, **{key: value})
                with self.subTest(worker=worker, key=key), self.assertRaises(ValueError):
                    evaluate.validate_setup_receipt(bad, self.protocol, worker, arm, metadata)
            bad = copy.deepcopy(setup)
            bad['source_metric_errors']['first10motor_rmse'] = 2e-6
            with self.assertRaises(ValueError):
                evaluate.validate_setup_receipt(bad, self.protocol, worker, arm, metadata)

    def test_episode_records_decoded_vq_execution_evidence(self):
        metadata = metadata_fixture(self.protocol, 'qclip')
        record = episode(self.protocol, self.rows, 0, 0, metadata)
        self.assertEqual(evaluate.validate_record(
            record, self.rows, 0, self.protocol, metadata, 'qclip'), 0)
        wrong = copy.deepcopy(record)
        wrong['quantization_evidence'] = wrong.pop('vq_evidence')
        with self.assertRaises(ValueError):
            evaluate.validate_record(wrong, self.rows, 0, self.protocol, metadata, 'qclip')
        wrong = copy.deepcopy(record)
        wrong['metrics'][1]['sampler_seed'] += 1
        with self.assertRaises(ValueError):
            evaluate.validate_record(wrong, self.rows, 0, self.protocol, metadata, 'qclip')

    def test_recovery_keeps_only_a_valid_frozen_prefix(self):
        metadata = metadata_fixture(self.protocol, 'qclip')
        record = episode(self.protocol, self.rows, 0, 0, metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl'
            path.write_bytes((json.dumps(record) + '\n{"partial":').encode('utf-8'))
            recovered = evaluate.recover_records(path, self.rows, 0, self.protocol, metadata, 'qclip')
            self.assertEqual(list(recovered), [0])
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            skipped = episode(self.protocol, self.rows, 0, 8, metadata, query_start=2)
            path.write_text(json.dumps(record) + '\n' + json.dumps(skipped) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'prefix'):
                evaluate.recover_records(path, self.rows, 0, self.protocol, metadata, 'qclip')


if __name__ == '__main__':
    unittest.main()

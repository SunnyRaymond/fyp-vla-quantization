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
        'rotation_movement': ({'action': 0.0, 'video': 0.0} if arm == 'H_W16A4'
                              else {'action': 0.12, 'video': 0.08}),
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
        'source_metric_errors': {name: 0.0 for name in evaluate.SOURCE_METRICS},
        'uncached_cached_action_max_abs': 0.0, 'uncached_cached_future_max_abs': 0.0,
        'actual_infer_action_max_abs': 0.0, 'startup_queries': 3,
        'new_training_updates': 0, 'cached_transformed_weights': True,
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
        self.assertEqual(evaluate.ARMS, ('H_W16A4', 'C80_R128_W16A4'))
        shards = [evaluate.worker_indices(worker) for worker in range(4)]
        self.assertEqual([evaluate.worker_arm(worker) for worker in range(4)],
                         ['H_W16A4', 'H_W16A4', 'C80_R128_W16A4', 'C80_R128_W16A4'])
        self.assertTrue(all(len(shard) == 140 for shard in shards))
        self.assertEqual(shards[0], shards[2])
        self.assertEqual(shards[1], shards[3])
        self.assertEqual(sorted(set().union(*map(set, shards))), list(range(280)))
        changed = copy.deepcopy(self.protocol)
        changed['references']['expected_reference_slots'] = 2800
        with self.assertRaises(ValueError):
            evaluate.validate_protocol(changed)
        changed = copy.deepcopy(self.protocol)
        changed['calibration']['case_ids'].pop()
        with self.assertRaises(ValueError):
            evaluate.validate_protocol(changed)

    def test_install_metadata_binds_bf16_weights_and_rotation_evidence(self):
        for arm in evaluate.ARMS:
            metadata = metadata_fixture(self.protocol, arm)
            self.assertEqual(evaluate.validate_install_metadata(metadata, self.protocol, arm), metadata)
            self.assertEqual(metadata['weight_bits'], 16)
            self.assertIs(metadata['vq_enabled'], False)
            bad = copy.deepcopy(metadata)
            bad['vq_enabled'] = True
            with self.assertRaises(ValueError):
                evaluate.validate_install_metadata(bad, self.protocol, arm)
            bad = copy.deepcopy(metadata)
            bad['orthogonality']['video'] = 1e-3
            with self.assertRaises(ValueError):
                evaluate.validate_install_metadata(bad, self.protocol, arm)
        bad = metadata_fixture(self.protocol, 'C80_R128_W16A4')
        bad['rotation_movement']['action'] = 0.0
        with self.assertRaises(ValueError):
            evaluate.validate_install_metadata(bad, self.protocol, 'C80_R128_W16A4')

    def test_setup_receipt_enforces_source_metrics_cache_parity_and_binding(self):
        for worker in range(4):
            arm = evaluate.worker_arm(worker)
            metadata = metadata_fixture(self.protocol, arm)
            setup = setup_fixture(self.protocol, worker, metadata)
            self.assertEqual(evaluate.validate_setup_receipt(
                setup, self.protocol, worker, arm, metadata), metadata)
            for key, value in (
                ('uncached_cached_action_max_abs', 1e-8),
                ('actual_infer_action_max_abs', 2e-6),
                ('new_training_updates', 1),
                ('source_endpoint', '/wrong/endpoint.json'),
            ):
                bad = dict(setup, **{key: value})
                with self.subTest(worker=worker, key=key), self.assertRaises(ValueError):
                    evaluate.validate_setup_receipt(bad, self.protocol, worker, arm, metadata)
            bad = copy.deepcopy(setup)
            bad['source_metric_errors']['future_latent_nmse'] = 3e-6
            with self.assertRaises(ValueError):
                evaluate.validate_setup_receipt(bad, self.protocol, worker, arm, metadata)

    def test_episode_uses_legacy_evidence_key_without_claiming_vq(self):
        metadata = metadata_fixture(self.protocol, 'H_W16A4')
        record = episode(self.protocol, self.rows, 0, 0, metadata)
        self.assertEqual(evaluate.validate_record(
            record, self.rows, 0, self.protocol, metadata, 'H_W16A4'), 0)
        wrong = copy.deepcopy(record)
        wrong['quantization_evidence'] = wrong.pop('vq_evidence')
        with self.assertRaises(ValueError):
            evaluate.validate_record(wrong, self.rows, 0, self.protocol, metadata, 'H_W16A4')
        wrong = copy.deepcopy(record)
        wrong['metrics'][1]['sampler_seed'] += 1
        with self.assertRaises(ValueError):
            evaluate.validate_record(wrong, self.rows, 0, self.protocol, metadata, 'H_W16A4')

    def test_recovery_keeps_only_a_valid_frozen_prefix(self):
        metadata = metadata_fixture(self.protocol, 'H_W16A4')
        record = episode(self.protocol, self.rows, 0, 0, metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl'
            path.write_bytes((json.dumps(record) + '\n{"partial":').encode('utf-8'))
            recovered = evaluate.recover_records(path, self.rows, 0, self.protocol, metadata, 'H_W16A4')
            self.assertEqual(list(recovered), [0])
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            skipped = episode(self.protocol, self.rows, 0, 4, metadata, query_start=2)
            path.write_text(json.dumps(record) + '\n' + json.dumps(skipped) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'prefix'):
                evaluate.recover_records(path, self.rows, 0, self.protocol, metadata, 'H_W16A4')


if __name__ == '__main__':
    unittest.main()

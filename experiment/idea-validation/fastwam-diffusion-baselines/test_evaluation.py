from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate


def protocol_fixture():
    return evaluate.read_json(HERE / 'protocol.json')


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
    return {
        'protocol': evaluate.PROTOCOL, 'arm': arm,
        'weight_bits': 4, 'activation_bits': 4, 'group_size': 128,
        'execution': 'decoded_bf16_reference', 'kv_cache_dtype': 'bf16',
        'native_low_bit_kernel': False, 'module_count': 614,
        'calibration_contexts': 80, 'optimizer_updates': 0,
        'encoded_target_bpw': 4.4, 'encoded_target_bytes': 3_400_000,
        'target_weight_parameters': 6_000_000,
        'low_rank_global_rank': 8 if arm == evaluate.ARMS[0] else 0,
        'representation': ('svdquant_low_rank_residual' if arm == evaluate.ARMS[0]
                           else 'viditq_channel_balance_rotation'),
        'transform_implementation': ('svdquant_smoothed_low_rank_v1' if arm == evaluate.ARMS[0]
                                     else 'viditq_structured_hadamard_v1'),
        'bank_manifest_pbs_jobid': 'bank-job-1',
        'extra_metadata_bytes': {'smoothing_vectors': 1024},
    }


def episode(protocol, rows, worker, index, metadata, query_index, first_in_run):
    arm = evaluate.worker_arm(worker)
    row = rows[index]
    implementation = metadata['transform_implementation']
    job_id = 'job-1' if first_in_run else 'job-1'
    evidence = {
        **metadata, 'observed_module_count': 614, 'quantized_linear_calls': 614,
        'native_int4_gemm_calls': 0, 'kv_packed_prefills': 0, 'kv_layer_reads': 0,
        ('low_rank_calls' if arm == evaluate.ARMS[0] else 'rotation_calls'): 1,
    }
    return {
        'protocol': evaluate.PROTOCOL, 'arm': arm, 'worker_id': worker, 'complete': True,
        'index': index, 'variant_id': row['variant_id'], 'suite': row['suite'],
        'dimension': row['dimension'], 'task_id': row['task_id'], 'state_id': 0,
        'env_seed': evaluate.environment_seed(index),
        'sampler_seed': evaluate.sampler_seed(index), 'success': index % 2 == 0,
        'transform_implementation': implementation, 'pbs_job_id': job_id,
        'quantization_evidence': evidence, 'episode_seconds': 1.0, 'replans': 1,
        'metrics': [{
            'query_index_in_arm': query_index, 'first_query_in_arm': query_index == 0,
            'first_query_in_run': first_in_run, 'replan': 0,
            'sampler_seed': evaluate.sampler_seed(index),
            'transform_implementation': implementation, 'pbs_job_id': job_id,
            'denoising_latency_ms': 10.0, 'full_query_wall_ms': 12.0,
            'peak_allocated_gpu_GB': 2.0, 'baseline_allocated_gpu_GB': 1.0,
            'peak_reserved_gpu_GB': 3.0,
        }],
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = evaluate.validate_protocol(protocol_fixture())
        self.rows = manifest_rows()
        self.metadata = {arm: metadata_fixture(self.protocol, arm) for arm in evaluate.ARMS}

    def test_four_workers_cover_exact_560_new_episode_slots(self):
        shards = [evaluate.worker_indices(worker) for worker in range(4)]
        self.assertEqual([len(shard) for shard in shards], [140] * 4)
        self.assertEqual(shards, [list(range(worker % 2, 280, 2)) for worker in range(4)])
        self.assertEqual(set(shards[0]) | set(shards[1]), set(range(280)))
        self.assertEqual(set(shards[2]) | set(shards[3]), set(range(280)))
        self.assertEqual(sum(map(len, shards)), 560)
        self.assertEqual([evaluate.worker_arm(worker) for worker in range(4)],
                         [evaluate.ARMS[0], evaluate.ARMS[0], evaluate.ARMS[1], evaluate.ARMS[1]])

    def test_install_metadata_rejects_wrong_bits_and_overbudget_bpw(self):
        for arm in evaluate.ARMS:
            valid = self.metadata[arm]
            self.assertIs(evaluate.validate_install_metadata(valid, self.protocol, arm), valid)
            for field, value in (('weight_bits', 8), ('activation_bits', 8),
                                 ('encoded_target_bpw', 4.61)):
                with self.subTest(arm=arm, field=field):
                    with self.assertRaises(ValueError):
                        evaluate.validate_install_metadata(dict(valid, **{field: value}),
                                                           self.protocol, arm)

    def test_setup_receipt_binds_startup_gates_and_bank_manifest(self):
        worker, arm = 0, evaluate.ARMS[0]
        metadata = self.metadata[arm]
        setup = {
            'protocol': evaluate.PROTOCOL, 'worker_id': worker, 'arm': arm,
            'pbs_job_id': 'pbs-job-1', 'passed': True, 'observed_modules': 614,
            'bf16_teacher_max_abs': 0.0, 'cross_worker_action_max_abs': 0.0,
            'component_selfcheck': {'passed': True},
            'bank_manifest': {'protocol': evaluate.PROTOCOL, 'complete': True,
                              'pbs_job_id': 'bank-job-1', 'modules': 614,
                              'calibration_contexts': 80},
            'install_metadata': metadata,
        }
        self.assertIs(evaluate.validate_setup_receipt(
            setup, self.protocol, worker, arm), metadata)
        broken = dict(setup, bank_manifest=dict(setup['bank_manifest'],
                                               pbs_job_id='other-job'))
        with self.assertRaisesRegex(ValueError, 'job id'):
            evaluate.validate_setup_receipt(broken, self.protocol, worker, arm)

    def test_prefix_recovery_discards_crash_tail_and_rejects_nonprefix(self):
        arm, worker = evaluate.ARMS[0], 0
        metadata = self.metadata[arm]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl'
            first = episode(self.protocol, self.rows, worker, 0, metadata, 0, True)
            evaluate.append_episode(path, first)
            with path.open('ab') as stream:
                stream.write(b'{"partial":')
                stream.flush()
            recovered = evaluate.recover_records(path, self.rows, worker, self.protocol,
                                                 metadata, arm)
            self.assertEqual(list(recovered), [0])
            self.assertTrue(path.read_bytes().endswith(b'\n'))

            second = episode(self.protocol, self.rows, worker, 2, metadata, 1, False)
            evaluate.append_episode(path, second)
            recovered = evaluate.recover_records(path, self.rows, worker, self.protocol,
                                                 metadata, arm)
            self.assertEqual(list(recovered), [0, 2])

            nonprefix = episode(self.protocol, self.rows, worker, 4, metadata, 2, True)
            path.write_text(json.dumps(nonprefix) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'prefix'):
                evaluate.recover_records(path, self.rows, worker, self.protocol,
                                         metadata, arm)


if __name__ == '__main__':
    unittest.main()

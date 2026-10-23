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


def fixture_manifest():
    variants = []
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        for dimension in ('d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'):
            for task_id in range(10):
                index = len(variants)
                variants.append({
                    'index': index, 'variant_id': f'variant-{index}', 'suite': suite,
                    'dimension': dimension, 'task_id': task_id, 'state_id': 0,
                })
    return variants


def fixture_metadata(protocol):
    metadata = evaluate.expected_install_metadata(protocol)
    metadata.update({
        'native_int4_gemm_calls': 0,
        'source_cached_query_parity': protocol['source_expected']['cached_query_parity'],
        'ratios': {'count': 614, 'min': .7315, 'mean': .9572, 'max': 1.0, 'below_one': 526},
        'scaling': {
            'modules': 614, 'group_size': 1,
            'channels': protocol['source_expected']['scale_channels'],
            'raw_scale_metadata_bytes': protocol['source_expected']['raw_scale_metadata_bytes'],
            'scale_min': .5025, 'scale_max': 1.7373, 'mean_abs_log2': .0653,
            'cached_compensated_weight_evaluation': True,
        },
    })
    return metadata


def write_complete_workers(root, protocol, rows, metadata):
    arm = 'per_channel_clip'
    training = {'reference_action': {
        'sampler_seed': 910_004_000,
        'actual_infer_parity': protocol['source_expected']['cached_query_parity'],
    }}
    setup = {
        'passed': True, 'protocol': evaluate.PROTOCOL, 'arm': arm, 'case_id': 2000,
        'seed_index': 0, 'sampler_seed': training['reference_action']['sampler_seed'],
        'reference_actual_infer_parity': training['reference_action']['actual_infer_parity'],
        'observed_modules': 614, 'max_abs_vs_reference': 0.0,
        'training_receipt_status': 'complete', 'startup_queries': 3,
        'source_endpoint_metrics_passed': True,
        'uncached_cached_action_max_abs': 0.0,
        'uncached_cached_future_max_abs': 0.0,
        'actual_infer_action_max_abs': 0.0, 'install_metadata': metadata,
    }
    for worker in range(4):
        out = root / 'evaluations' / arm / f'worker_{worker}'
        out.mkdir(parents=True)
        indices = evaluate.worker_indices(worker, arm)
        summary = {
            'protocol': evaluate.PROTOCOL, 'worker_id': worker, 'arm': arm,
            'status': 'complete', 'complete': True, 'episodes': 70, 'expected': 70,
            'soft_deadline_reached': False, 'indices': indices,
            'completed_indices': indices, 'arm_episodes': {arm: 70},
            'install_metadata': metadata,
        }
        (out / 'progress.json').write_text(json.dumps(summary), encoding='utf-8')
        (out / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        (out / 'setup_receipt.json').write_text(json.dumps(setup), encoding='utf-8')
        records = []
        job = f'job-{worker}'
        for query_index, index in enumerate(indices):
            row = rows[index]
            records.append({
                'protocol': evaluate.PROTOCOL, 'worker_id': worker, 'arm': arm,
                'complete': True, 'index': index, 'variant_id': row['variant_id'],
                'suite': row['suite'], 'dimension': row['dimension'], 'task_id': row['task_id'],
                'state_id': 0, 'env_seed': evaluate.environment_seed(index),
                'sampler_seed': evaluate.sampler_seed(index), 'success': index % 2 == 0,
                'transform_implementation': 'batched_h128_v2', 'pbs_job_id': job,
                'vq_evidence': {**metadata, 'observed_module_count': 614,
                                'native_int4_gemm_calls': 0, 'kv_packed_prefills': 0,
                                'kv_layer_reads': 0, 'quantized_linear_calls': 614},
                'episode_seconds': 1.0, 'replans': 1,
                'metrics': [{
                    'query_index_in_arm': query_index,
                    'first_query_in_arm': query_index == 0,
                    'first_query_in_run': query_index == 0,
                    'replan': 0, 'sampler_seed': evaluate.sampler_seed(index),
                    'transform_implementation': 'batched_h128_v2', 'pbs_job_id': job,
                    'denoising_latency_ms': 10.0, 'full_query_wall_ms': 12.0,
                    'peak_allocated_gpu_GB': 2.0, 'baseline_allocated_gpu_GB': 1.0,
                    'peak_reserved_gpu_GB': 3.0,
                }],
            })
        (out / 'episodes.jsonl').write_text(
            ''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
    return training


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = evaluate.validate_protocol(evaluate.read_json(HERE / 'protocol.json'))
        self.rows = fixture_manifest()
        self.metadata = fixture_metadata(self.protocol)

    def test_complete_single_arm_workers_load_all_280_and_missing_worker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = write_complete_workers(root, self.protocol, self.rows, self.metadata)
            records, metadata, setup, checkpointed = aggregate.load_workers(
                root, self.protocol, self.rows, training, 'per_channel_clip')
            self.assertEqual(len(records), 280)
            self.assertEqual(set(records), {('per_channel_clip', i) for i in range(280)})
            self.assertEqual(metadata, {'per_channel_clip': self.metadata})
            self.assertEqual(set(setup), {'0', '1', '2', '3'})
            self.assertEqual(checkpointed, [])
            (root / 'evaluations' / 'per_channel_clip' / 'worker_3' / 'summary.json').unlink()
            with self.assertRaises(FileNotFoundError):
                aggregate.load_workers(root, self.protocol, self.rows, training, 'per_channel_clip')

    def test_reference_slots_and_paired_summary_counts(self):
        arms = self.protocol['references']['arms']
        references = {(arm, i): {'success': i % 3 == 0}
                      for arm in arms for i in range(280)}
        self.assertEqual(len(aggregate.validate_reference_slots(references, arms)), 2800)
        with self.assertRaises(ValueError):
            aggregate.validate_reference_slots({key: value for key, value in references.items()
                                                if key != (arms[0], 0)}, arms)
        new_arm = 'per_channel_clip'
        current = {(new_arm, i): {
            'success': i % 2 == 0, 'transform_implementation': 'batched_h128_v2',
            'pbs_job_id': 'job-0', 'episode_seconds': 1.0,
            'metrics': [{
                'first_query_in_run': i == 0, 'denoising_latency_ms': 10.0,
                'full_query_wall_ms': 12.0, 'peak_allocated_gpu_GB': 2.0,
                'baseline_allocated_gpu_GB': 1.0, 'peak_reserved_gpu_GB': 3.0,
            }],
        } for i in range(280)}
        all_records = {**references, **current}
        by_arm = {arm: {i: references[arm, i] for i in range(280)} for arm in arms}
        by_arm[new_arm] = {i: current[new_arm, i] for i in range(280)}
        training = {'training_receipt': {
            'protocol': 'fastwam-vq-per-channel-scaling-clipping-v1',
            'status': 'complete', 'complete': True, 'updates': 80, 'target_linears': 614,
            'history': ['must not be copied'],
        }, 'reference_action': {'actual_infer_parity': {'action_max_abs': 0.0}}}
        summary = aggregate._make_summary(
            self.protocol, self.rows, current, by_arm, {new_arm: self.metadata}, {},
            training, [], 'complete')
        self.assertEqual(summary['episode_slots'], 280)
        self.assertEqual(summary['reference_episode_slots'], 2800)
        self.assertEqual(len(summary['overall']['arms']), 11)
        self.assertEqual(summary['overall']['paired'][new_arm][arms[0]]['episodes'], 280)
        self.assertEqual(len(summary['cells']), 28)
        self.assertNotIn('history', summary['training_receipt'])
        self.assertEqual(len(all_records), 3080)

    def test_existing_c80_source_requires_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / 'results'
            results.mkdir()
            source_protocol = {
                'protocol': self.protocol['references']['c80_protocol'],
                'arms': ['clip_only', 'joint'],
            }
            (results.parent / 'protocol.json').write_text(json.dumps(source_protocol), encoding='utf-8')
            summary = {
                'protocol': source_protocol['protocol'], 'status': 'complete', 'complete': True,
                'manifest_variants': 280, 'episode_slots': 560, 'reference_episode_slots': 2240,
                'arm_episodes': {'clip_only': 280, 'joint': 280},
            }
            (results / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            (results / 'PIPELINE_COMPLETE').write_text(json.dumps({
                'protocol': source_protocol['protocol'], 'complete': False,
                'episode_slots': 560, 'arm_episodes': summary['arm_episodes'],
                'reference_episode_slots': 2240,
            }), encoding='utf-8')
            with self.assertRaises(ValueError):
                aggregate._load_c80_results(results, self.rows, self.protocol)

    def test_run_writes_checkpoint_marker_for_prefix_and_removes_it_on_completion(self):
        arm = 'per_channel_clip'
        arms = self.protocol['references']['arms']
        references = {(name, i): {'success': i % 3 == 0} for name in arms for i in range(280)}

        def record(index):
            return {
                'success': index % 2 == 0, 'transform_implementation': 'batched_h128_v2',
                'pbs_job_id': 'job-0', 'episode_seconds': 1.0,
                'metrics': [{'first_query_in_run': index == 0,
                    'denoising_latency_ms': 10.0, 'full_query_wall_ms': 12.0,
                    'peak_allocated_gpu_GB': 2.0, 'baseline_allocated_gpu_GB': 1.0,
                    'peak_reserved_gpu_GB': 3.0}],
            }

        prefix = {(arm, i): record(i) for i in range(2)}
        full = {(arm, i): record(i) for i in range(280)}
        training = {'training_receipt': {'complete': True, 'updates': 80},
                    'reference_action': {'actual_infer_parity': {'action_max_abs': 0.0}}}
        prefix_result = (prefix, {arm: self.metadata}, {}, [0, 1, 2, 3])
        complete_result = (full, {arm: self.metadata}, {}, [])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / 'protocol.json'
            manifest_path = root / 'manifest.json'
            results = root / 'results'
            results.mkdir()
            protocol_path.write_text(json.dumps(self.protocol), encoding='utf-8')
            manifest_path.write_text(json.dumps({
                'protocol': self.protocol['cohort']['manifest_protocol'], 'variants': self.rows,
            }), encoding='utf-8')
            episode_path = results / 'evaluations' / arm / 'worker_0' / 'episodes.jsonl'
            episode_path.parent.mkdir(parents=True)
            episode_path.write_text('{"saved":true}\n', encoding='utf-8')
            args = SimpleNamespace(protocol=protocol_path, manifest=manifest_path,
                                   results=results, arm=arm)
            with mock.patch.dict(sys.modules, {'runner': SimpleNamespace(guard=lambda: None)}), \
                    mock.patch.object(aggregate, '_validate_source_and_training', return_value=training), \
                    mock.patch.object(aggregate, 'load_workers', side_effect=[prefix_result, complete_result]), \
                    mock.patch.object(aggregate, '_load_references', return_value=(references, arms)):
                aggregate.run(args)
                output = results / 'evaluations' / arm
                prefix_summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
                prefix_marker = json.loads((output / 'PIPELINE_CHECKPOINTED').read_text(encoding='utf-8'))
                self.assertEqual(prefix_summary['status'], 'checkpointed')
                self.assertEqual(prefix_summary['episode_slots'], 2)
                self.assertFalse(prefix_summary['complete'])
                self.assertEqual(prefix_marker['episode_slots'], 2)
                self.assertFalse((output / 'PIPELINE_COMPLETE').exists())
                aggregate.run(args)
                self.assertTrue((output / 'PIPELINE_COMPLETE').is_file())
                self.assertFalse((output / 'PIPELINE_CHECKPOINTED').exists())
                self.assertEqual(json.loads((output / 'summary.json').read_text())['episode_slots'], 280)
                self.assertEqual(episode_path.read_text(encoding='utf-8'), '{"saved":true}\n')


if __name__ == '__main__':
    unittest.main()

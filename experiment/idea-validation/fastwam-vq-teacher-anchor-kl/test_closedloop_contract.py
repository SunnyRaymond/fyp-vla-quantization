"""Stdlib-only checks for the frozen paired closed-loop contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import closedloop
import closedloop_aggregate as aggregate

HERE = Path(__file__).resolve().parent


def fixture_manifest():
    rows = []
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        for dimension in ('d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'):
            for variant in range(10):
                index = len(rows)
                rows.append({'index': index, 'variant_id': f'variant-{index}', 'suite': suite,
                             'dimension': dimension, 'task_id': variant, 'state_id': 0})
    return rows


class ClosedLoopContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol = closedloop.read_json(HERE / 'protocol.json')
        self.source = closedloop.read_json(
            HERE.parent / 'fastwam-vq-softbin-relational-plus' / 'protocol.json')
        self.rows = fixture_manifest()

    def test_source_manifest_and_baseline_bind_to_frozen_local_protocol(self):
        self.assertEqual(closedloop.validate_protocol(self.protocol, self.source), self.protocol)
        changed = copy.deepcopy(self.protocol)
        changed['closedloop']['source_root'] += '-different'
        with self.assertRaises(ValueError):
            closedloop.validate_protocol(changed, self.source)
        changed = copy.deepcopy(self.protocol)
        changed['closedloop']['expected_reference_success'] = 202
        with self.assertRaises(ValueError):
            closedloop.validate_protocol(changed, self.source)

    def test_two_parity_shards_cover_exactly_280_variants(self):
        shards = [closedloop.worker_indices(worker) for worker in range(2)]
        self.assertEqual([len(shard) for shard in shards], [140, 140])
        self.assertEqual(shards[0], list(range(0, 280, 2)))
        self.assertEqual(shards[1], list(range(1, 280, 2)))
        self.assertEqual(sorted(shards[0] + shards[1]), list(range(280)))
        with self.assertRaises(ValueError):
            closedloop.worker_indices(2)

    def test_manifest_retains_the_frozen_four_by_seven_by_ten_cells(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'protocol': closedloop.COHORT_PROTOCOL,
                                        'variants': self.rows}), encoding='utf-8')
            doc, rows = closedloop.load_manifest(path)
        self.assertEqual(doc['protocol'], closedloop.COHORT_PROTOCOL)
        self.assertEqual(len(rows), 280)
        bad = copy.deepcopy(self.rows)
        bad[-1]['index'] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'protocol': closedloop.COHORT_PROTOCOL,
                                        'variants': bad}), encoding='utf-8')
            with self.assertRaises(ValueError):
                closedloop.load_manifest(path)

    def test_pair_counts_use_episodes_and_count_both_success_and_failure_as_ties(self):
        candidate = {i: {'success': i % 4 in (0, 1)} for i in range(280)}
        reference = {i: {'success': i % 4 in (0, 2)} for i in range(280)}
        result = aggregate._pair(candidate, reference, list(range(280)))
        self.assertEqual(result, {'wins': 70, 'losses': 70, 'ties': 140,
                                  'episodes': 280, 'success_rate_difference': 0.0})

    def test_deployment_must_be_exact_final80_anchor_unit(self):
        history = [{'update': update, 'case_id': case}
                   for update, case in enumerate(self.protocol['calibration']['update_order'], 1)]
        receipt = {'complete': True, 'updates': 80,
                   'update_order': self.protocol['calibration']['update_order'], 'history': history,
                   'clipping': {'count': 614},
                   'scaling': {'modules': 614, 'group_size': 1,
                               'channels': self.source['source_expected']['scale_channels'],
                               'raw_scale_metadata_bytes': self.source['source_expected']['scale_metadata_bytes'],
                               'scale_metadata_dtype': 'float32_log2',
                               'cached_compensated_weight_evaluation': True},
                   'cached_query_parity': self.source['source_expected']['cached_query_parity']}
        modules = {f'module-{index}': None for index in range(614)}
        deployment = {'protocol': self.protocol, 'unit': closedloop.ARM, 'updates': 80,
                      'baseline_checkpoint': closedloop.baseline_checkpoint(self.protocol),
                      'scale_metadata_dtype': 'float32_log2', 'ratios': modules,
                      'log2_scales': modules.copy()}
        closedloop.validate_training_artifacts(self.protocol, deployment, receipt, self.source)
        wrong = copy.deepcopy(deployment)
        wrong['unit'] = 'relational'
        with self.assertRaises(ValueError):
            closedloop.validate_training_artifacts(self.protocol, wrong, receipt, self.source)

    def test_startup_parity_receipt_requires_exact_replay(self):
        setup = {'protocol': closedloop.PROTOCOL, 'unit': 'closedloop_0', 'worker_id': 0,
                 'arm': closedloop.ARM, 'passed': True, 'observed_modules': 614,
                 'case_id': 2000, 'seed_index': 0, 'sampler_seed': 910_004_000,
                 'training_updates': 80, 'new_training_updates': 0,
                 'cached_compensated_weights': True, 'pbs_job_id': 'job-123',
                 'source_action_max_abs': 0.0, 'source_future_max_abs': 0.0,
                 'uncached_cached_action_max_abs': 0.0,
                 'uncached_cached_future_max_abs': 0.0,
                 'actual_infer_action_max_abs': 0.0,
                 'install_metadata': {'protocol': closedloop.PROTOCOL,
                                      'source_endpoint_job': 'endpoint-456'}}
        closedloop._validate_setup(setup, 0, self.protocol)
        changed = copy.deepcopy(setup)
        changed['source_action_max_abs'] = 1e-12
        with self.assertRaises(ValueError):
            closedloop._validate_setup(changed, 0, self.protocol)

    def test_summary_contains_overall_suite_dimension_cell_and_pair_views(self):
        candidates = {i: {'success': i % 4 in (0, 1)} for i in range(280)}
        references = {i: {'success': i % 4 in (0, 2)} for i in range(280)}
        setups = {str(worker): {'pbs_job_id': f'job-{worker}',
                  'source_action_max_abs': 0.0, 'source_future_max_abs': 0.0,
                  'uncached_cached_action_max_abs': 0.0, 'uncached_cached_future_max_abs': 0.0,
                  'actual_infer_action_max_abs': 0.0, 'install_metadata': {}}
                  for worker in range(2)}
        result = aggregate.summarize(self.protocol, self.rows, candidates, references, setups, [])
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['suites']), 4)
        self.assertEqual(len(result['dimensions']), 7)
        self.assertEqual(len(result['cells']), 28)
        self.assertEqual(result['overall']['paired'][closedloop.ARM]['per_channel_clip']['ties'], 140)


if __name__ == '__main__':
    unittest.main()

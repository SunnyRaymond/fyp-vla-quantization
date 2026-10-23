"""Stdlib-only contract tests for the C8 latent-anchor aggregate."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import aggregate

HERE = Path(__file__).resolve().parent
PROTOCOL = aggregate.read_json(HERE / 'protocol.json')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def make_records(split):
    baseline, candidate = {}, {}
    slots = aggregate.expected_slots(PROTOCOL, split)
    case_ids = list(dict.fromkeys(case for case, _ in slots))
    suite_by_case = {case: SUITES[index % 4] for index, case in enumerate(case_ids)}
    seed_values = {case: [900_000_000 + case * 1000 + seed
                          for seed in PROTOCOL['evaluation']['seed_indices']]
                   for case in case_ids}
    for case_id, seed_index in slots:
        suite = suite_by_case[case_id]
        row = {'case_id': case_id, 'suite': suite,
               'sampler_seeds': seed_values[case_id]}
        baseline_value = 2.0
        candidate_value = 3.0
        if split == 'heldout' and case_id == case_ids[0]:
            candidate_value = 1.0 if seed_index == 0 else 3.0
        base = {
            'protocol': PROTOCOL['baseline_cache_protocol']['protocol'],
            'arm': 'baseline', 'updates': 0, 'split': split, 'case_id': case_id,
            'suite': suite, 'plan_row': row, 'seed_index': seed_index,
            'sampler_seed': seed_values[case_id][seed_index],
            **{metric: baseline_value for metric in aggregate.COMMON_METRICS},
        }
        candidate_record = {
            'protocol': PROTOCOL['protocol'], 'arm': 'anchor_kl', 'updates': 8,
            'split': split, 'case_id': case_id, 'suite': suite, 'plan_row': copy.deepcopy(row),
            'seed_index': seed_index, 'sampler_seed': seed_values[case_id][seed_index],
            **{metric: candidate_value for metric in aggregate.COMMON_METRICS},
            'action_mse': 1.0, 'latent_nmse': 1.0, 'latent_anchor_kl': .1,
            'baseline_objective': .05, 'composite_objective': .06,
        }
        baseline[(case_id, seed_index)] = base
        candidate[(case_id, seed_index)] = candidate_record
    return baseline, candidate


class AggregateContractTests(unittest.TestCase):
    def test_protocol_binds_baseline_to_original_c8_rho15_parent(self):
        self.assertEqual(aggregate.validate_protocol(PROTOCOL), PROTOCOL)
        changed = copy.deepcopy(PROTOCOL)
        changed['baseline_cache_protocol']['source_clipping_protocol']['parent_updates'] = 80
        with self.assertRaises(ValueError):
            aggregate.validate_protocol(changed)

    def test_frozen_query_counts_are_eight_calibration_and_eighty_heldout(self):
        self.assertEqual(len(aggregate.expected_slots(PROTOCOL, 'calibration')), 8)
        heldout = aggregate.expected_slots(PROTOCOL, 'heldout')
        self.assertEqual(len(heldout), 80)
        self.assertEqual(len({case for case, _ in heldout}), 40)

    def test_heldout_pairs_average_seeds_before_case_wins_and_losses(self):
        baseline, candidate = make_records('heldout')
        result = aggregate._summarize_split(PROTOCOL, 'heldout', baseline, candidate)
        metric = result['overall']['first10motor_rmse']
        self.assertEqual(metric['paired']['cases'], 40)
        self.assertEqual(metric['paired']['ties'], 1)
        self.assertEqual(metric['paired']['losses'], 39)
        self.assertEqual(metric['paired']['wins'], 0)
        self.assertEqual(result['query_records_per_arm'], 80)
        self.assertEqual(len(result['suites']), 4)

    def test_missing_pair_fails_and_baseline_anchor_diagnostics_stay_unavailable(self):
        baseline, candidate = make_records('calibration')
        candidate.pop(next(iter(candidate)))
        with self.assertRaises(ValueError):
            aggregate._summarize_split(PROTOCOL, 'calibration', baseline, candidate)
        baseline, candidate = make_records('calibration')
        result = aggregate._summarize_split(PROTOCOL, 'calibration', baseline, candidate)
        self.assertFalse(result['baseline_anchor_kl_unavailable']['available'])
        self.assertFalse(result['baseline_composite_objective_unavailable']['available'])
        self.assertTrue(result['candidate_only_diagnostics']['latent_anchor_kl']['available'])

    def test_baseline_copy_must_preserve_record_and_add_only_provenance(self):
        source_record = {
            'protocol': PROTOCOL['baseline_cache_protocol']['protocol'], 'arm': 'baseline',
            'updates': 0, 'split': 'calibration', 'case_id': 2000,
            'suite': SUITES[0],
            'plan_row': {'case_id': 2000, 'suite': SUITES[0], 'sampler_seeds': [1]},
            'seed_index': 0, 'sampler_seed': 1,
            **{metric: .25 for metric in aggregate.COMMON_METRICS},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = copy.deepcopy(PROTOCOL)
            protocol['baseline_cache_root'] = str(root / 'cache')
            source = (Path(protocol['baseline_cache_root']) / 'results/calibration/baseline/queries/2000_0.json')
            target = root / 'results/baseline/calibration/queries/2000_0.json'
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            source.write_text(json.dumps(source_record), encoding='utf-8')
            copied = {**source_record, 'new_protocol': protocol['protocol'], 'source_path': str(source)}
            target.write_text(json.dumps(copied), encoding='utf-8')
            loaded = aggregate._load_baseline_copy(protocol, root / 'results', 'calibration', 2000, 0)
            self.assertEqual(loaded['new_protocol'], protocol['protocol'])
            copied['first10motor_rmse'] = .5
            target.write_text(json.dumps(copied), encoding='utf-8')
            with self.assertRaises(ValueError):
                aggregate._load_baseline_copy(protocol, root / 'results', 'calibration', 2000, 0)

    def test_prepare_gate_binds_exact_calibration_protocol_and_frozen_coefficients(self):
        prepare = {
            'complete': True, 'passed': True, 'protocol': PROTOCOL['protocol'],
            'heldout_used': False,
            'calibration_case_ids': PROTOCOL['calibration']['case_ids'],
            'coefficient_frozen': True, 'beta': .01, 'temperature': .5,
            'stats': {'latent': {}},
            'normalization_c': PROTOCOL['training']['normalization_c'],
        }
        aggregate.validate_prepare_summary(PROTOCOL, prepare)
        for key, bad_value in (
            ('protocol', 'other-protocol'),
            ('heldout_used', True),
            ('calibration_case_ids', PROTOCOL['calibration']['case_ids'][:-1]),
            ('coefficient_frozen', False),
        ):
            changed = copy.deepcopy(prepare)
            changed[key] = bad_value
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate.validate_prepare_summary(PROTOCOL, changed)

    def test_report_scope_explicitly_avoids_independent_task_inference(self):
        baseline_cal, candidate_cal = make_records('calibration')
        baseline_val, candidate_val = make_records('heldout')
        prepare = {
            'complete': True, 'passed': True, 'protocol': PROTOCOL['protocol'],
            'heldout_used': False,
            'calibration_case_ids': PROTOCOL['calibration']['case_ids'],
            'coefficient_frozen': True, 'beta': .01, 'temperature': .5,
            'stats': {'latent': {}},
            'normalization_c': PROTOCOL['training']['normalization_c'],
        }
        candidate_summary = {
            'new_teacher_queries': 0, 'evaluation_queries': 88, 'new_model_queries': 97,
        }
        summary = aggregate.summarize(
            PROTOCOL,
            {'calibration': baseline_cal, 'heldout': baseline_val},
            {'calibration': candidate_cal, 'heldout': candidate_val},
            prepare, candidate_summary, 'job-123')
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['baseline_records'], 88)
        self.assertEqual(summary['candidate_records'], 88)
        self.assertEqual(summary['candidate_evaluation_queries'], 88)
        self.assertEqual(summary['candidate_new_model_queries'], 97)
        self.assertIn('not independent tasks', summary['interpretation'])
        self.assertNotIn('p_value', json.dumps(summary).lower())
        report = aggregate.render_report(summary)
        self.assertIn('unavailable', report)
        with tempfile.TemporaryDirectory() as temp:
            aggregate.write_outputs(temp, summary)
            saved = aggregate.read_json(Path(temp) / 'aggregate' / 'summary.json')
            comparison = aggregate.read_json(Path(temp) / 'aggregate' / 'comparison.json')
            self.assertTrue(saved['complete'])
            self.assertEqual(saved['pbs_jobid'], 'job-123')
            self.assertEqual(comparison['baseline_records'], 88)
            self.assertTrue((Path(temp) / 'ANALYSIS.zh.md').is_file())

        candidate_summary.pop('new_model_queries')
        summary_without_counter = aggregate.summarize(
            PROTOCOL,
            {'calibration': baseline_cal, 'heldout': baseline_val},
            {'calibration': candidate_cal, 'heldout': candidate_val},
            prepare, candidate_summary, 'job-123')
        self.assertIsNone(summary_without_counter['candidate_new_model_queries'])


if __name__ == '__main__':
    unittest.main()

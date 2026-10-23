"""Small toy checks for the training pair plan and its frozen C80 binding."""
import unittest

from coverage_inventory import FAMILIES
from prepare_pairs import make_plan
from prepare_training_pairs import make_training_plan


def toy_inputs():
    rows = []
    for suite_index in range(4):
        for ordinal in range(20):
            case_id = 2000 + suite_index * 20 + ordinal
            rows.append({
                'case_id': case_id,
                'suite': f'suite_{suite_index}',
                'sampler_seeds': [910000000 + case_id],
                'observation_file': f'inputs/case_{case_id}.npz',
                'metadata_file': f'inputs/case_{case_id}.json',
            })
    p = {
        'protocol': 'fastwam-vq-control-kl-v1',
        'remote_root': '/toy-parent',
        'source_relational_protocol': {'calibration': {
            'case_ids': list(range(2000, 2080)), 'root': '/toy-source'}},
    }
    original = {
        'status': 'complete',
        'inputs': rows,
        'suite_order': [f'suite_{i}' for i in range(4)],
        'calibration_cohorts': {'c80': list(range(2000, 2080))},
    }
    config = {
        'protocol': 'control-kl-training-data-v1',
        'parent_protocol': p['protocol'],
        'case_ids': list(range(2000, 2080)),
        'corruption_seed_base': 860000000,
        'corruption_seed_stride': 100,
        'corruption_seed_offset': 29,
        'training_use': True,
        'dataset_subdir': 'data/training_pairs',
        'sampler_seed_index': 0,
        'new_demonstrations': 0,
        'new_states': 0,
        'heldout_inputs_used': False,
        'evaluation_trajectory_inputs_used': False,
    }
    return p, original, config


class TrainingPairPlanTest(unittest.TestCase):
    def test_plan_preserves_c80_balance_and_child_binding(self):
        p, original, config = toy_inputs()
        plan = make_training_plan(p, original, config)
        source_plan = make_plan(p, original)
        self.assertEqual(plan['protocol'], config['protocol'])
        self.assertEqual(plan['parent_protocol'], p['protocol'])
        self.assertEqual(plan['config'], config)
        self.assertEqual(sorted(case['case_id'] for case in plan['cases']), config['case_ids'])
        self.assertTrue(all(case['training_use'] is True for case in plan['cases']))
        for suite in original['suite_order']:
            cases = [case for case in plan['cases'] if case['suite'] == suite]
            self.assertEqual(len(cases), 20)
            for family in FAMILIES:
                selected = [case for case in cases if case['family'] == family]
                self.assertEqual(len(selected), 4)
                self.assertEqual(sorted(case['severity'] for case in selected), [2, 4, 7, 10])
        for trained, source in zip(plan['cases'], source_plan['cases']):
            self.assertEqual((trained['family'], trained['severity'], trained['shard']),
                             (source['family'], source['severity'], source['shard']))

    def test_seeds_are_independent_of_diagnostic_seeds(self):
        p, original, config = toy_inputs()
        plan = make_training_plan(p, original, config)
        for case in plan['cases']:
            expected = 860000000 + case['case_id'] * 100 + 29
            diagnostic = 760000000 + case['case_id'] * 100 + 17
            self.assertEqual(case['corruption_seed'], expected)
            self.assertNotEqual(case['corruption_seed'], diagnostic)

    def test_case_id_drift_is_rejected(self):
        p, original, config = toy_inputs()
        changed_config = {**config, 'case_ids': [*config['case_ids'][:-1], 2080]}
        with self.assertRaisesRegex(ValueError, 'case IDs'):
            make_training_plan(p, original, changed_config)
        changed_original = {**original, 'calibration_cohorts': {'c80': [*config['case_ids'][:-1], 2080]}}
        with self.assertRaisesRegex(ValueError, 'C80 IDs'):
            make_training_plan(p, changed_original, config)
        changed_source = {**p, 'source_relational_protocol': {'calibration': {
            'case_ids': [*config['case_ids'][:-1], 2080], 'root': '/toy-source'}}}
        with self.assertRaisesRegex(ValueError, 'case IDs'):
            make_training_plan(changed_source, original, config)


if __name__ == '__main__':
    unittest.main()

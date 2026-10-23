"""Cheap standard-library checks; tensor checks run on the allocated GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('clipping_experiment_contract', HERE / 'experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
P = json.loads((HERE / 'protocol.json').read_text())


class Contract(unittest.TestCase):
    def rows(self):
        return [{'case_id': case, 'seed_index': seed,
                 'suite': experiment.SUITES[(case - 3000) // 10],
                 **{m: float((case - 3000) // 10 + 1) for m in experiment.METRICS}}
                for case in P['evaluation']['case_ids'] for seed in (0, 1)]

    def test_equal_suite_aggregation_and_exact_denominator(self):
        rows = self.rows()
        result = experiment.aggregate(rows, P)
        self.assertEqual(result['queries'], 80)
        self.assertEqual(result['overall']['first10motor_rmse'], 2.5)
        with self.assertRaises(ValueError):
            experiment.aggregate(rows[:-1], P)
        with self.assertRaises(ValueError):
            experiment.aggregate(rows + rows[:1], P)

    def test_gate_rejects_joint_gain_explained_by_weight_training(self):
        def summary(value):
            return {'complete': True, 'queries': 80,
                    'overall': {'first10motor_rmse': value, 'future_latent_nmse': value}}
        summaries = {'baseline': summary(1.), 'weight_only': summary(.8),
                     'clip_only': summary(1.1), 'joint': summary(.9)}
        self.assertFalse(experiment.gate(summaries)['advance_to_channel_scaling'])
        summaries['joint'] = summary(.7)
        self.assertTrue(experiment.gate(summaries)['advance_to_channel_scaling'])
        summaries['joint']['overall']['future_latent_nmse'] = 1.01
        self.assertFalse(experiment.gate(summaries)['advance_to_channel_scaling'])

    def test_frozen_budget_and_inputs(self):
        self.assertEqual(P['training']['additional_updates'], len(P['calibration']['case_ids']))
        self.assertEqual(P['evaluation']['case_ids'], list(range(3000, 3040)))
        self.assertEqual(P['arms'], ['baseline', 'weight_only', 'clip_only', 'joint'])
        self.assertEqual(P['clipping']['initial_ratio'], 1.)
        self.assertLessEqual(P['allocation']['walltime'], '02:00:00')


if __name__ == '__main__':
    unittest.main()

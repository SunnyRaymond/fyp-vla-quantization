"""Small stdlib-only checks for the C80 endpoint bindings and aggregation."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
METRICS = ('first10motor_mse', 'first10motor_rmse', 'full32motor_rmse',
           'late22motor_rmse', 'first10gripper_rmse', 'future_latent_nmse', 'mixed_loss')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def fake_c80_module():
    module = types.ModuleType('c80_reference')
    module.METRICS = METRICS
    module.ref = types.SimpleNamespace(SUITES=SUITES)

    def queries(protocol, split):
        cases = protocol['calibration' if split == 'calibration' else 'evaluation']['case_ids']
        seeds = [0] if split == 'calibration' else [0, 1]
        return [(case, seed) for case in cases for seed in seeds]

    def aggregate_records(records, protocol, split):
        suites = {}
        cases_per_suite = 20 if split == 'calibration' else 10
        for suite in SUITES:
            rows = [record for record in records if record['suite'] == suite]
            cases = sorted({record['case_id'] for record in rows})
            if len(cases) != cases_per_suite:
                raise ValueError('suite case count')
            suites[suite] = {}
            for metric in METRICS:
                case_means = [sum(record[metric] for record in rows
                                  if record['case_id'] == case)
                              / sum(record['case_id'] == case for record in rows)
                              for case in cases]
                suites[suite][metric] = sum(case_means) / cases_per_suite
        overall = {metric: sum(suite[metric] for suite in suites.values()) / 4
                   for metric in METRICS}
        return {'queries': len(records), 'cases': 80 if split == 'calibration' else 40,
                'overall': overall, 'by_suite': suites}

    module.queries = queries
    module.aggregate_records = aggregate_records
    module.validate_binding = lambda *args: None
    module.guard = lambda: None
    return module


def load_worker():
    clipping = types.ModuleType('clipping_reference')
    clipping.METRICS = METRICS[:5]
    clipping.SUITES = SUITES
    c80 = fake_c80_module()
    with patch.dict(sys.modules, {'clipping_reference': clipping,
                                  'c80_reference': c80, 'torch': None}):
        spec = importlib.util.spec_from_file_location('joint_rmse_worker_under_test',
                                                      HERE / 'worker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, c80


def source_protocol(calibration_ids=None, evaluation_ids=None):
    calibration_ids = calibration_ids or list(range(2000, 2080))
    evaluation_ids = evaluation_ids or list(range(3000, 3040))
    return {'protocol': 'source-c80',
            'calibration': {'case_ids': calibration_ids},
            'evaluation': {'case_ids': evaluation_ids}}


def record(case, seed, suite, metric_values=None):
    return {'case_id': case, 'seed_index': seed, 'suite': suite,
            **{metric: (metric_values or {}).get(metric, 0.0) for metric in METRICS}}


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker, cls.c80 = load_worker()

    def test_exact_80_coverage_rejects_missing_and_duplicate(self):
        protocol = source_protocol()
        records = [record(case, 0, SUITES[index // 20])
                   for index, case in enumerate(protocol['calibration']['case_ids'])]
        self.worker.require_exact_coverage(records, protocol, 'calibration')
        with self.assertRaises(RuntimeError):
            self.worker.require_exact_coverage(records[:-1], protocol, 'calibration')
        with self.assertRaises(RuntimeError):
            self.worker.require_exact_coverage(records + [records[0]], protocol, 'calibration')

    def test_candidate_binding_catches_seed_and_checkpoint_mismatch(self):
        protocol = {'protocol': 'consumer', 'arm': 'joint_c80',
                    'joint_source': {'updates': 80, 'checkpoint': '/source/deployment.pt'}}
        row = {'case_id': 2000, 'suite': 'libero_spatial', 'sampler_seeds': [9123, 9124]}
        source_job = '25860001.pbs101'
        saved = self.worker.candidate_binding(protocol, row, 'heldout', 1, source_job)
        saved.update({'pbs_jobid': '25860002.pbs101', 'first10motor_rmse': 0.1})
        self.worker.validate_candidate_record(saved, protocol, row, 'heldout', 1,
                                              source_job, ('first10motor_rmse',))
        for key, value in (('sampler_seed', 9123), ('source_checkpoint', '/other.pt'),
                           ('source_pbs_jobid', '25860000.pbs101')):
            altered = dict(saved)
            altered[key] = value
            with self.assertRaises(RuntimeError):
                self.worker.validate_candidate_record(altered, protocol, row, 'heldout', 1,
                                                      source_job, ('first10motor_rmse',))

    def test_plan_loader_uses_source_c80_evaluation_cohort(self):
        row_cal = {'case_id': 2000, 'domain': 'original', 'suite': SUITES[0],
                   'sampler_seeds': [11, 12]}
        row_test = {'case_id': 3000, 'domain': 'original', 'suite': SUITES[0],
                    'sampler_seeds': [21, 22]}
        source_c80 = {'protocol': 'source-c80',
                      'calibration': {'case_ids': [2000]},
                      'evaluation': {'case_ids': [3000]}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'plan.json'
            path.write_text(json.dumps({'status': 'complete',
                                        'calibration_cohorts': {'c80': [2000]},
                                        'inputs': [row_cal, row_test]}), encoding='utf-8')
            consumer = {'calibration': {'case_ids': [2000], 'plan_file': str(path)},
                        'evaluation': {'case_ids': [3000]}}
            _, rows = self.worker._load_plan(consumer, source_c80)
        self.assertEqual(rows[2000], row_cal)
        self.assertEqual(rows[3000], row_test)

    def test_teacher_paths_use_original_calibration_and_heldout_roots(self):
        control = Path('/original/c80-control')
        clipping = Path('/original/activation-clipping')
        self.assertEqual(self.worker.teacher_path(control, clipping, 'calibration', 2000, 0),
                         control / 'results' / 'teacher' / 'calibration' / '2000_0.pt')
        self.assertEqual(self.worker.teacher_path(control, clipping, 'heldout', 3000, 1),
                         clipping / 'results' / 'teacher' / '3000_1.pt')

    def test_heldout_aggregation_averages_seeds_then_equal_suites(self):
        protocol = source_protocol()
        records = []
        for case_index, case in enumerate(protocol['evaluation']['case_ids']):
            suite_index = case_index // 10
            values = {'first10motor_rmse': float(1 + suite_index)}
            for seed in (0, 1):
                item = record(case, seed, SUITES[suite_index], values)
                item['first10motor_rmse'] += seed
                records.append(item)
        with patch.dict(sys.modules, {'c80_reference': self.c80}):
            stats = self.worker.aggregate_endpoints(records, protocol, 'heldout')
        self.assertEqual(stats['queries'], 80)
        self.assertEqual(stats['cases'], 40)
        self.assertTrue(math.isclose(stats['by_suite']['libero_spatial']['first10motor_rmse'], 1.5))
        self.assertTrue(math.isclose(stats['overall']['first10motor_rmse'], 3.0))

    def test_paired_case_counts_average_the_two_seeds(self):
        candidate, baseline = [], []
        for case in range(3000, 3040):
            suite = SUITES[(case - 3000) // 10]
            for seed in (0, 1):
                baseline.append({'case_id': case, 'seed_index': seed,
                                 'first10motor_rmse': 1.0})
                candidate.append({'case_id': case, 'seed_index': seed,
                                  'first10motor_rmse': 0.5 if case % 2 == 0 else 1.5})
        counts = self.worker.paired_case_counts(
            {'joint_c80': candidate, 'baseline': baseline},
            'joint_c80', 'baseline', 'first10motor_rmse')
        self.assertEqual(counts, {'cases': 40, 'improved': 20, 'worsened': 20, 'tied': 0})


if __name__ == '__main__':
    unittest.main()

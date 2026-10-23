"""Lightweight protocol and endpoint-binding checks; deliberately imports no torch."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('rollout_c80_worker_under_test', HERE / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


METRICS = worker.METRICS
SUITES = worker.SUITES


def fixture_protocol():
    return {
        'protocol': 'fastwam-vq-rollout-c80-v1', 'arm': 'rollout_joint',
        'evaluation': {'case_ids': list(range(3000, 3040)), 'seed_indices': [0, 1]},
    }


def fixture_rows():
    return {case: {'case_id': case, 'suite': SUITES[(case - 3000) // 10],
                   'sampler_seeds': [case * 2, case * 2 + 1]}
            for case in range(3000, 3040)}


def records(protocol):
    rows = fixture_rows()
    return [{
        'protocol': protocol['protocol'], 'arm': protocol['arm'], 'split': 'heldout',
        'updates': 80, 'case_id': case, 'suite': rows[case]['suite'],
        'plan_row': rows[case], 'seed_index': seed,
        'sampler_seed': rows[case]['sampler_seeds'][seed], 'source_checkpoint': '/deployment.pt',
        'source_jobid': '123.server', 'pbs_jobid': '456.server',
        **{metric: float(case - 3000 + seed) for metric in METRICS},
    } for case, seed in worker.query_pairs(protocol)]


class WorkerMetadataTests(unittest.TestCase):
    def test_exact_heldout_pairs_and_duplicate_rejection(self):
        protocol = fixture_protocol()
        rows = records(protocol)
        self.assertEqual(len(worker.query_pairs(protocol)), 80)
        worker.require_exact_coverage(rows, protocol)
        with self.assertRaises(RuntimeError):
            worker.require_exact_coverage(rows[:-1], protocol)
        with self.assertRaises(RuntimeError):
            worker.require_exact_coverage(rows + [rows[0]], protocol)

    def test_plan_requires_all_expected_rows_in_original_order(self):
        expected = {
            case: {'case_id': case, 'sampler_seeds': [case, case + 1]}
            for case in list(range(2000, 2080)) + list(range(3000, 3040))
        }
        plan = {'status': 'complete', 'calibration_cohorts': {'c80': list(range(2000, 2080))},
                'inputs': list(expected.values())}
        worker.validate_plan_document(plan, expected)
        with self.assertRaises(RuntimeError):
            worker.validate_plan_document({**plan, 'inputs': plan['inputs'][:-1]}, expected)
        with self.assertRaises(RuntimeError):
            worker.validate_plan_document({**plan, 'inputs': list(reversed(plan['inputs']))}, expected)

    def test_expected_rows_preserves_frozen_cohorts_without_demo_leakage(self):
        source_path = (HERE.parent / 'fastwam-vq-calibration-suites' / 'retrieved' /
                       'data' / 'plan.json')
        source = json.loads(source_path.read_text(encoding='utf-8'))
        self.assertEqual(source['status'], 'preparing')
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / 'plan.json'
            source['status'] = 'complete'
            plan_path.write_text(json.dumps(source), encoding='utf-8')
            protocol = worker.common.read(HERE / 'protocol.json')
            protocol['calibration']['original_plan'] = str(plan_path)
            rows = worker.common.expected_rows(protocol)

        original = {row['case_id']: row for row in source['inputs']}
        anchor_ids = protocol['rollout']['anchor_ids']
        visited_ids = protocol['rollout']['visited_ids']
        heldout_ids = protocol['evaluation']['case_ids']
        self.assertEqual(len(rows), 120)
        self.assertEqual(len(anchor_ids), 40)
        self.assertEqual(len(visited_ids), 40)
        self.assertEqual(len({(rows[case]['suite'], rows[case]['task_id'])
                              for case in visited_ids}), 40)
        for case in anchor_ids:
            self.assertEqual(rows[case], original[case])
        for case in visited_ids:
            self.assertEqual({key: value for key, value in rows[case].items() if key != 'rollout'},
                             original[case])
        for case in heldout_ids:
            self.assertEqual(rows[case], original[case])
        train_demos = {(rows[case]['suite'], rows[case]['task_id'], rows[case]['demo_id'])
                       for case in protocol['calibration']['case_ids']}
        heldout_demos = {(rows[case]['suite'], rows[case]['task_id'], rows[case]['demo_id'])
                         for case in heldout_ids}
        self.assertTrue(train_demos.isdisjoint(heldout_demos))

    def test_prepare_teacher_coverage_and_parity_gate(self):
        protocol = fixture_protocol()
        summary = {
            'protocol': protocol['protocol'], 'pbs_jobid': '123.server', 'complete': True,
            'contexts': 80, 'visited_contexts': 40, 'anchor_contexts': 40,
            'new_teacher_queries': 40, 'teacher_queries': 80, 'collector_queries': 40,
            'case_ids': list(range(2000, 2080)), 'teacher_seed_indices': [0],
            'source_parity_max_abs': 1e-7, 'render_parity_passed': True,
        }
        worker.validate_prepare_summary(summary, protocol)
        with self.assertRaises(RuntimeError):
            worker.validate_prepare_summary({**summary, 'new_teacher_queries': 39}, protocol)
        with self.assertRaises(RuntimeError):
            worker.validate_prepare_summary({**summary, 'source_parity_max_abs': 1e-4}, protocol)

    def test_candidate_record_requires_exact_plan_and_metric_binding(self):
        protocol = fixture_protocol()
        row = fixture_rows()[3000]
        record = records(protocol)[0]
        worker.validate_candidate_record(record, protocol, row, 0, '/deployment.pt', '123.server')
        with self.assertRaises(RuntimeError):
            worker.validate_candidate_record({**record, 'sampler_seed': -1}, protocol, row, 0,
                                             '/deployment.pt', '123.server')

    def test_seven_metric_aggregation_keeps_exact_suite_and_seed_means(self):
        class C80:
            @staticmethod
            def aggregate_records(rows, protocol, split):
                if split != 'heldout':
                    raise AssertionError('expected heldout split')
                out = {}
                for suite in SUITES:
                    suite_rows = [row for row in rows if row['suite'] == suite]
                    cases = sorted({row['case_id'] for row in suite_rows})
                    out[suite] = {metric: sum(
                        sum(row[metric] for row in suite_rows if row['case_id'] == case) / 2
                        for case in cases) / 10 for metric in METRICS}
                return {'queries': len(rows), 'cases': 40, 'by_suite': out,
                        'overall': {metric: sum(out[s][metric] for s in SUITES) / 4
                                    for metric in METRICS}}

        protocol = fixture_protocol()
        stats = worker.aggregate_records(records(protocol), C80, protocol)
        self.assertEqual(stats['queries'], 80)
        self.assertEqual(set(stats['overall']), set(METRICS))
        self.assertAlmostEqual(stats['overall']['first10motor_mse'], 20.0)

    def test_real_consumer_comparison_and_control_arm_roots(self):
        comparison_path = (HERE.parent / 'fastwam-vq-c80-joint-rmse' /
                           'results' / 'comparison.json')
        comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
        self.assertEqual(comparison['protocol'], worker.C80_CONSUMER_PROTOCOL)
        self.assertTrue(comparison['complete'])
        self.assertEqual({arm: comparison['arms']['heldout'][arm]['queries']
                          for arm in ('joint_c80', 'clip_only', 'baseline')},
                         {'joint_c80': 80, 'clip_only': 80, 'baseline': 80})
        for arm in ('joint_c80', 'clip_only', 'baseline'):
            summary = comparison['arms']['heldout'][arm]
            source_protocol = {'protocol': summary['protocol']}
            worker._validate_control_summary(summary, summary, arm, source_protocol)
        joint_summary = comparison['arms']['heldout']['joint_c80']
        changed = json.loads(json.dumps(joint_summary))
        changed['overall']['first10motor_rmse'] += 0.01
        with self.assertRaises(RuntimeError):
            worker._validate_control_summary(joint_summary, changed, 'joint_c80',
                                            {'protocol': 'source-protocol'})

        protocol = {
            'reference_root': '/reference-consumer',
            'source_c80_protocol': {'remote_root': '/source-controls'},
        }
        self.assertEqual(worker._control_root(protocol, 'joint_c80'), Path('/reference-consumer'))
        for arm in ('clip_only', 'baseline'):
            self.assertEqual(worker._control_root(protocol, arm), Path('/source-controls'))
        with self.assertRaises(RuntimeError):
            worker._control_root(protocol, 'unexpected')

    def test_module_load_does_not_import_torch(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()

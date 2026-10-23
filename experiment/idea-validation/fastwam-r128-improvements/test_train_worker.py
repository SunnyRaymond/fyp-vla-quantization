"""Stdlib-only checks for frozen bindings, schedules, and allocation ordering."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("r128_train_worker_tested", HERE / "train_worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))

    def test_frozen_protocol_and_supported_lanes(self):
        for lane in worker.LANES:
            worker.validate_protocol(self.protocol, lane)
        self.assertEqual(worker.LANES, ("optimization", "coverage_c80"))
        no_c8_arm = copy.deepcopy(self.protocol)
        no_c8_arm["lanes"].pop("coverage_c8", None)
        worker.validate_protocol(no_c8_arm, "coverage_c80")
        with self.assertRaises(ValueError):
            worker.validate_protocol(self.protocol, "coverage_c8")

    def test_rejects_vq_or_arm_drift(self):
        changed = copy.deepcopy(self.protocol)
        changed["training"]["vq_updates"] = 1
        with self.assertRaises(ValueError):
            worker.validate_protocol(changed, "optimization")
        changed = copy.deepcopy(self.protocol)
        changed["lanes"]["optimization"]["arms"][0]["order"][0] = 2019
        with self.assertRaises(ValueError):
            worker.validate_protocol(changed, "optimization")

    def test_optimization_update_orders(self):
        schedules = worker.training_schedules(self.protocol, "optimization", {})
        self.assertEqual(list(schedules), ["grouped_lr1e4", "interleaved_lr1e3", "interleaved_lr1e4"])
        self.assertEqual(schedules["grouped_lr1e4"], [[case] for case in worker.C8])
        interleaved = [[case] for case in [2000, 2021, 2051, 2066, 2019, 2025, 2057, 2070]]
        self.assertEqual(schedules["interleaved_lr1e3"], interleaved)
        self.assertEqual(schedules["interleaved_lr1e4"], interleaved)

    def test_identity_metric_replay_checks_all_metrics(self):
        expected = {key: float(index) for index, key in enumerate(worker.METRICS)}
        actual = dict(expected)
        actual["first10motor_rmse"] += 1e-6
        result = worker.compare_metrics(actual, expected)
        self.assertEqual(result["max_abs_delta"], 1e-6)
        self.assertEqual(set(result["deltas"]), set(worker.METRICS))
        actual["future_latent_nmse"] += 3e-6
        with self.assertRaises(ValueError):
            worker.compare_metrics(actual, expected)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.suites = ["s0", "s1", "s2", "s3"]

    def fake_rows(self, case_ids, per_suite):
        return {case: {"suite": self.suites[index // per_suite]}
                for index, case in enumerate(case_ids)}

    def test_c80_schedule_uses_each_case_once_and_balances_suites(self):
        cases = list(range(80))
        rows = self.fake_rows(cases, 20)
        schedule = worker.coverage_schedule(rows, self.suites, cases, seed=17)
        self.assertEqual(len(schedule), 8)
        self.assertTrue(all(len(update) == 10 for update in schedule))
        self.assertEqual(worker.Counter(case for update in schedule for case in update),
                         worker.Counter({case: 1 for case in cases}))
        self.assertEqual(worker.Counter(rows[case]["suite"] for update in schedule for case in update),
                         worker.Counter({suite: 20 for suite in self.suites}))


class RuntimeBoundaryTests(unittest.TestCase):
    def test_import_is_stdlib_only_and_guard_precedes_torch(self):
        source = (HERE / "train_worker.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        guard_line = min(node.lineno for node in calls
                         if isinstance(node.func, ast.Attribute) and node.func.attr == "_require_compute_allocation")
        torch_line = min(node.lineno for node in ast.walk(run)
                         if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
        teacher_binding_line = min(node.lineno for node in ast.walk(run)
                                   if isinstance(node, ast.Subscript)
                                   and isinstance(node.slice, ast.Constant)
                                   and node.slice.value == "c80_teacher_directory")
        student_query_line = min(node.lineno for node in calls
                                 if isinstance(node.func, ast.Attribute)
                                 and node.func.attr == "differentiable_action")
        query = next(node for node in ast.walk(run)
                     if isinstance(node, ast.FunctionDef) and node.name == "query")
        matrix_lines = [node.lineno for node in ast.walk(query)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "matrix"]
        sampler_lines = [node.lineno for node in ast.walk(query)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute)
                         and node.func.attr == "differentiable_action"]
        rotation_line = min(node.lineno for node in ast.walk(run)
                            if isinstance(node, ast.ImportFrom) and node.module == "rotation_bank")
        self.assertLess(guard_line, torch_line)
        self.assertLess(teacher_binding_line, rotation_line)
        self.assertGreater(student_query_line, rotation_line)
        self.assertTrue(matrix_lines and sampler_lines)
        self.assertLess(max(matrix_lines), min(sampler_lines))
        self.assertIn('random_seed = p["training"]["random_seed"]', source)
        self.assertIn('"H_W16A4" / "queries" / "2000_0.json"', source)
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
                             for node in tree.body))
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()

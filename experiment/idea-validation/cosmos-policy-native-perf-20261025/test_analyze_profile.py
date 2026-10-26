"""Stdlib-only tests for ChromeTrace aggregation using synthetic intervals."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import analyze_profile


class AnalyzeProfileTests(unittest.TestCase):
    def test_nested_cpu_intervals_subtract_direct_children(self):
        events = [
            {"ph": "X", "cat": "user_annotation", "name": "engine.scope", "ts": 0, "dur": 100, "pid": 1, "tid": 7},
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 10, "dur": 40, "pid": 1, "tid": 7},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 20, "dur": 10, "pid": 1, "tid": 7},
            {"ph": "X", "cat": "cpu_op", "name": "aten::copy_", "ts": 60, "dur": 20, "pid": 1, "tid": 7},
            {"ph": "X", "cat": "kernel", "name": "gemm_kernel", "ts": 21, "dur": 8, "pid": 2, "tid": 3},
        ]
        result = analyze_profile.analyze_events(events)
        by_key = {(row["category"], row["name"]): row for row in result["cpu_inclusive_top"]}
        self.assertEqual(by_key[("user_annotation", "engine.scope")]["inclusive_us"], 100)
        self.assertEqual(by_key[("user_annotation", "engine.scope")]["exclusive_us"], 40)
        self.assertEqual(by_key[("cpu_op", "aten::mm")]["exclusive_us"], 30)
        self.assertEqual(by_key[("cuda_runtime", "cudaLaunchKernel")]["exclusive_us"], 10)
        self.assertEqual(result["gpu_kernel_top"][0]["duration_us"], 8)

    def test_non_nested_overlap_is_flagged_and_excluded_from_strict_exclusive(self):
        events = [
            {"ph": "X", "cat": "cpu_op", "name": "aten::equal", "ts": 0, "dur": 10, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "cpu_op", "name": "aten::item", "ts": 5, "dur": 10, "pid": 1, "tid": 1},
        ]
        result = analyze_profile.analyze_events(events)
        focus_equal = result["focus_cpu_events"]["aten::equal"][0]
        focus_item = result["focus_cpu_events"]["aten::item"][0]
        self.assertEqual(focus_equal["exclusive_status"], "overlap_non_nested")
        self.assertEqual(focus_item["exclusive_status"], "overlap_non_nested")
        self.assertEqual(focus_equal["exclusive_valid_calls"], 0)
        self.assertEqual(result["exclusive_quality"]["non_nested_overlap_events"], 2)

    def test_duplicate_names_keep_their_categories_and_focus_coverage(self):
        events = [
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 12, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "user_annotation", "name": "aten::mm", "ts": 20, "dur": 9, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 30, "dur": 4, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaDeviceSynchronize", "ts": 35, "dur": 5, "pid": 1, "tid": 1},
        ]
        result = analyze_profile.analyze_events(events)
        mm_rows = [row for row in result["cpu_inclusive_top"] if row["name"] == "aten::mm"]
        self.assertEqual({row["category"] for row in mm_rows}, {"cpu_op", "user_annotation"})
        self.assertEqual(len(result["focus_cpu_events"]["aten::mm"]), 2)
        self.assertEqual(len(result["focus_cpu_events"]["cudaStreamSynchronize"]), 1)
        self.assertEqual(len(result["focus_cpu_events"]["cudaDeviceSynchronize"]), 1)

    def test_allocation_guard_rejects_login_host_gpu_and_nonmember(self):
        with tempfile.TemporaryDirectory() as temporary:
            nodefile = Path(temporary) / "nodes"
            nodefile.write_text("x1000c0s0b0n0\n", encoding="utf-8")
            analyze_profile.validate_allocation("42.pbs101", str(nodefile), "x1000c0s0b0n0")
            with self.assertRaisesRegex(RuntimeError, "compute allocation"):
                analyze_profile.validate_allocation("42.pbs101", str(nodefile), "asp2a-login")
            with self.assertRaisesRegex(RuntimeError, "CPU-only"):
                analyze_profile.validate_allocation("42.pbs101", str(nodefile), "x1000c0s0b0n0", "0")
            with self.assertRaisesRegex(RuntimeError, "outside"):
                analyze_profile.validate_allocation("42.pbs101", str(nodefile), "x1000c0s0b1n0")

    def test_summary_size_limit_and_trace_category_binding(self):
        self.assertTrue(analyze_profile.bounded_json_bytes({"status": "complete"}, limit=64))
        with self.assertRaisesRegex(ValueError, "exceeds 8 bytes"):
            analyze_profile.bounded_json_bytes({"long": "string-value"}, limit=8)
        row = analyze_profile.analyze_events([
            {"ph": "X", "cat": "user_annotation", "name": "engine.fp32_U_matmul", "ts": 0, "dur": 2, "pid": 1, "tid": 1}
        ])["engine_user_annotations"][0]
        self.assertEqual(row["category"], "user_annotation")
        self.assertEqual(row["name"], "engine.fp32_U_matmul")


if __name__ == "__main__":
    unittest.main()

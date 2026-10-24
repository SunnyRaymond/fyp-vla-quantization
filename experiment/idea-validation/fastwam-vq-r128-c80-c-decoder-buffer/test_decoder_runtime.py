"""Standard-library checks for the decoder tuner; never imports Torch or Triton."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).with_name("decoder_runtime.py")


def _load_pure_helper(name, extra_scope=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    isolated = ast.Module(body=[node], type_ignores=[])
    scope = {"math": math}
    scope.update(extra_scope or {})
    exec(compile(isolated, str(SOURCE), "exec"), scope)
    return scope[name]


class DecoderRuntimeStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(SOURCE))

    def test_source_parses_without_importing_runtime_dependencies(self):
        self.assertEqual(SOURCE.name, "decoder_runtime.py")
        names = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertIn("torch", names)
        self.assertIn("triton", names)
        # Parsing only above keeps this local test independent of CUDA packages.

    def test_vectorized_arange_bound_is_an_explicit_compile_time_constant(self):
        kernel = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_decode_vectorized')
        bound = next(node for node in kernel.body if isinstance(node, ast.AnnAssign)
                     and isinstance(node.target, ast.Name) and node.target.id == 'scales_per_tile')
        self.assertEqual(ast.unparse(bound.annotation), 'tl.constexpr')

    def test_candidates_are_bounded_and_aligned_to_both_encoding_blocks(self):
        assignment = next(node for node in self.tree.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "_CANDIDATES"
                                  for target in node.targets))
        candidates = ast.literal_eval(assignment.value)
        self.assertGreaterEqual(len(candidates), 4)
        self.assertLessEqual(len(candidates), 6)
        self.assertEqual(len({row["name"] for row in candidates}), len(candidates))
        self.assertTrue(all(row["block"] % 32 == 0 and row["block"] % 4 == 0
                            for row in candidates))

    def test_prepare_api_exposes_reuse_and_frequency_report_methods(self):
        functions = {node.name: node for node in self.tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        prepare = functions["prepare"]
        positional = [arg.arg for arg in prepare.args.args]
        keyword_only = [arg.arg for arg in prepare.args.kwonlyargs]
        self.assertEqual(positional, ["runtime", "bundle", "p", "out"])
        self.assertEqual(keyword_only, ["reuse_path"])
        update = next(node for node in ast.walk(self.tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "update_frequencies")
        self.assertIn("query_count", [arg.arg for arg in update.args.kwonlyargs])
        classes = {node.name: node for node in self.tree.body if isinstance(node, ast.ClassDef)}
        dispatcher_methods = {node.name for node in classes["DecoderDispatcher"].body
                              if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"__call__", "report", "update_frequencies"}.issubset(dispatcher_methods))

    def test_selection_uses_only_bitwise_valid_candidates_that_beat_reference(self):
        choose = _load_pure_helper("_choose_config")
        self.assertEqual(choose(1.0, [
            {"name": "slow", "median_ms": 1.2, "bitwise": True},
            {"name": "fast", "median_ms": 0.8, "bitwise": True},
            {"name": "wrong", "median_ms": 0.1, "bitwise": False},
        ]), "fast")
        self.assertEqual(choose(1.0, [
            {"name": "slower", "median_ms": 1.01, "bitwise": True},
        ]), "reference")
        self.assertEqual(choose(1.0, [
            {"name": "tie", "median_ms": 1.0, "bitwise": True},
        ]), "reference")
        self.assertEqual(choose(1.0, []), "reference")

    def test_receipt_marks_frequency_pending_until_measured(self):
        self.assertIn('"call_frequency_status": "pending"', self.source)
        self.assertIn('"frequency_weighted": None', self.source)
        self.assertIn('"method": "CUDA Graph replay; per decode"', self.source)
        self.assertIn('"reference_decode_ms_full_panel": reference_total', self.source)
        self.assertIn('"reference_decode_ms_per_query": reference_total / query_count', self.source)

    def test_reuse_accepts_prepared_vq_objects_and_requires_same_complete_run(self):
        known = {"reference", "block32_w4", "block64_w4", "block128_w4",
                 "block256_w4", "block128_w8"}
        validate = _load_pure_helper("_validate_reuse", {
            "Path": Path, "json": json, "deepcopy": deepcopy,
            "EXPECTED_MODULES": 614, "_PRIMARY_QUERIES": 80, "_KNOWN_CONFIGS": known,
        })
        modules = {f"module.{i}": SimpleNamespace(rows=4, width=32) for i in range(614)}
        checkpoint = {
            "source_job": "checkpoint.pbs101", "source_protocol": "source-v1",
            "source_export_job": "export.pbs101", "bundle_path": "/bundle.pt",
        }
        device = {"name": "GPU", "capability": [8, 0]}
        receipt = {
            "schema": "fastwam_decoder_tuning_v1", "protocol": "decoder-v1",
            "pbs_jobid": "current.pbs101", "mode": "c_decoder",
            "checkpoint_identity": checkpoint, "device": device,
            "parity": {"checked_modules": 614, "bitwise_passed": 614, "candidate_failures": []},
            "call_frequency_status": "complete",
            "frequency_panel": {
                "query_count": 80, "logical_decode_calls_full_panel": 614,
                "logical_decode_calls_per_query": 614 / 80,
            },
            "frequency_weighted": {
                "query_count": 80, "logical_decode_calls_full_panel": 614,
                "reference_decode_ms_full_panel": 8.0, "selected_decode_ms_full_panel": 8.0,
                "reference_decode_ms_per_query": 0.1, "selected_decode_ms_per_query": 0.1,
                "selected_over_reference": 1.0,
            },
            "benchmark": {}, "reference": {},
            "modules": [{"name": name, "shape": [4, 32], "calls": 1} for name in modules],
            "shapes": [{
                "shape": [4, 32], "selected": "reference",
                "reference": {"median_ms": 0.01}, "candidates": [],
            }],
        }
        shape_rows = [{"shape": [4, 32], "module_count": 614, "call_count": None}]
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tuning.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            result = validate(
                path, modules, shape_rows, device, protocol="decoder-v1",
                pbs_jobid="current.pbs101", checkpoint_identity=checkpoint, mode="c_buffer",
            )
            self.assertEqual(result["shapes"][0]["selected"], "reference")
            self.assertIsNone(shape_rows[0]["call_count"])

            incomplete = deepcopy(receipt)
            incomplete["call_frequency_status"] = "pending"
            path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "80-query"):
                validate(
                    path, modules, [{"shape": [4, 32]}], device,
                    protocol="decoder-v1", pbs_jobid="current.pbs101",
                    checkpoint_identity=checkpoint, mode="c_buffer",
                )


if __name__ == "__main__":
    unittest.main()

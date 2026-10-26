"""Local stdlib-only contract tests for the PBS profile worker."""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

import engine_profile


def protocol_stub() -> dict:
    return {"protocol": "cosmos-policy-vqwam-v1",
            "arms": ["bf16", "ours_vqwam"],
            "performance": {"case_ids": list(range(3000, 3040)),
                            "seed_indices": [0, 1], "queries": 80,
                            "blocks": 3, "repetitions": 3,
                            "latency_scope": "fixed formal scope"}}


class EngineProfileTests(unittest.TestCase):
    def test_diagnostic_keys_are_bounded_and_in_frozen_panel(self):
        self.assertEqual(engine_profile.diagnostic_queries(protocol_stub()),
                         [(3000, 0), (3009, 1), (3020, 0), (3039, 1)])
        protocol = protocol_stub()
        protocol["performance"]["case_ids"].remove(3039)
        with self.assertRaisesRegex(ValueError, "lacks diagnostic"):
            engine_profile.diagnostic_queries(protocol)

    def test_formal_panel_binding_retains_720_samples_per_arm(self):
        binding = engine_profile.formal_panel_binding(protocol_stub())
        self.assertEqual(binding["queries"], 80)
        self.assertEqual(binding["samples_per_arm"], 720)
        self.assertEqual(binding["case_ids"], list(range(3000, 3040)))

    def test_formal_panel_drift_is_rejected(self):
        protocol = protocol_stub()
        protocol["performance"]["repetitions"] = 2
        with self.assertRaisesRegex(ValueError, "formal panel"):
            engine_profile.formal_panel_binding(protocol)

    def test_output_is_confined_to_new_root_and_separate_from_old_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            old_root, root = base / "old", base / "new"
            out = root / "results" / "profile"
            old_root.mkdir()
            checked = engine_profile.validate_output_paths(old_root, root, out)
            self.assertEqual(checked, (old_root.resolve(), root.resolve(), out.resolve()))
            with self.assertRaisesRegex(ValueError, "inside the new --root"):
                engine_profile.validate_output_paths(old_root, root, base / "outside")
            with self.assertRaisesRegex(ValueError, "isolated"):
                engine_profile.validate_output_paths(old_root, old_root / "new", old_root / "new" / "out")

    def test_profile_json_has_a_hard_size_limit(self):
        self.assertTrue(engine_profile.bounded_json_bytes({"ok": True}, limit=32))
        with self.assertRaisesRegex(ValueError, "exceeds 8 bytes"):
            engine_profile.bounded_json_bytes({"payload": "0123456789"}, limit=8)

    def test_profiler_reports_self_and_inclusive_time_separately(self):
        class Event:
            key = "engine._Banks._operand"
            count = 280
            self_cpu_time_total = 10.0
            cpu_time_total = 25.0
            self_device_time_total = 4.0
            device_time_total = 7.0

        row = engine_profile.profiler_event_row(Event())
        self.assertEqual(row["self_cpu_us"], 10.0)
        self.assertEqual(row["inclusive_cpu_us"], 25.0)
        self.assertEqual(row["self_cuda_us"], 4.0)
        self.assertEqual(row["inclusive_cuda_us"], 7.0)

    def test_module_load_does_not_import_torch(self):
        source = Path(engine_profile.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("torch" in (node.module or "") for node in imports
                             if isinstance(node, ast.ImportFrom)))
        self.assertFalse(any(alias.name == "torch" for node in imports if isinstance(node, ast.Import)
                             for alias in node.names))

    def test_profile_flag_is_required_before_any_runtime_import(self):
        with self.assertRaisesRegex(ValueError, "--profile-only"):
            engine_profile.main(["--old-root", "old", "--root", "new", "--out", "new/out",
                                 "--arm", "bf16"])


if __name__ == "__main__":
    unittest.main()

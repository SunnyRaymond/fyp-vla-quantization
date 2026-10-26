import ast
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_atom


class AtomDiagnosticTests(unittest.TestCase):
    def test_import_is_torch_free_and_cli_guard_precedes_runtime_import(self):
        source = Path(diagnose_atom.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_torch_imports = [node for node in tree.body
                                   if isinstance(node, ast.Import)
                                   and any(alias.name == "torch" or alias.name.startswith("torch.")
                                           for alias in node.names)]
        top_level_torch_from_imports = [node for node in tree.body
                                        if isinstance(node, ast.ImportFrom)
                                        and (node.module == "torch" or
                                             (node.module or "").startswith("torch."))]
        self.assertEqual(top_level_torch_imports, [])
        self.assertEqual(top_level_torch_from_imports, [])
        self.assertIn("from prepare_sources import guard", source)
        self.assertLess(source.index("if not allocation_guarded:\n        guard()"),
                        source.index("import torch as torch_module_local"))

    def test_shape_and_tolerance_are_frozen(self):
        self.assertEqual((diagnose_atom.M, diagnose_atom.N, diagnose_atom.K), (129, 131, 256))
        self.assertEqual((diagnose_atom.PADDED_M, diagnose_atom.PADDED_N), (256, 256))
        self.assertEqual(diagnose_atom.GROUP_WIDTH, 128)
        self.assertEqual(diagnose_atom.GROUPS, 2)
        self.assertEqual((diagnose_atom.ATOL, diagnose_atom.RTOL), (0.125, 0.002))
        self.assertEqual(diagnose_atom._output_path(Path("x")), Path("x/atom_diagnostic.json"))

    def test_a_scale_candidates_match_independent_official_lane_reader(self):
        for padded_m in (128, 256, 1792):
            with self.subTest(padded_m=padded_m):
                official_paired = []
                for tile in range(0, padded_m, 128):
                    for lane in range(64):
                        row = tile + (lane // 8) * 16 + lane % 8
                        official_paired.extend((row, row + 8) * 4)
                candidates = diagnose_atom._activation_scale_source_rows
                atom_v3 = candidates("atom_v3", padded_m)
                legacy = candidates("legacy_row_major", padded_m)
                paired = candidates("paired_half2_repeat4", padded_m)
                self.assertEqual(paired, tuple(official_paired))
                self.assertEqual(len(atom_v3), 4 * padded_m)
                self.assertEqual(len(legacy), 4 * padded_m)
                self.assertEqual(paired[:8], (0, 8, 0, 8, 0, 8, 0, 8))
                self.assertEqual(atom_v3[:8], (0, 0, 0, 0, 8, 8, 8, 8))
                self.assertEqual(legacy[:8], (0, 0, 0, 0, 1, 1, 1, 1))
                self.assertEqual(Counter(paired), Counter({row: 4 for row in range(padded_m)}))
                # Keep the three historical candidates independent of the
                # production helper so labels cannot silently alias v4.
                self.assertNotEqual(atom_v3, paired)
                self.assertNotEqual(legacy, paired)

    def test_case_matrix_covers_code_scale_and_row_column_boundaries(self):
        source = Path(diagnose_atom.__file__).read_text(encoding="utf-8")
        for token in ("constant_codes_and_scales", "seed777_random_codes_unit_scales",
                      "one_hot_boundary_cases", "unique_A_scales_layout_candidates",
                      "isolated_weight_scale_group_{group}", "paired_half2_repeat4",
                      "production_packing_v4", "_viditq_activation_scales(scale_rows, torch)",
                      "original FP16 scale tensors", "training_updates",
                      "policy_forward_calls", "closed_loop_episodes"):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertIn("(0, 0, 0), (0, 1, 0), (0, 2, 0)", source)
        self.assertIn("(0, 1), (0, 2)", source)
        self.assertIn("extension_label: str | None = None", source)
        self.assertIn("extension_label='original' or 'general'", source)
        self.assertIn('"extension_label": extension_label', source)

    def test_adapter_receipt_is_pinned_and_uses_existing_build(self):
        source = Path(diagnose_atom.__file__).read_text(encoding="utf-8")
        self.assertEqual(diagnose_atom.PINNED_VIDITQ,
                         "43657d11b32192ec0f50f0dac8d3a16c3d841196")
        self.assertIn("native_diffusion_build.json", source)
        self.assertIn("viditq_atom_native.so", source)
        self.assertIn("native_diffusion import NATIVE_ADAPTER_VERSIONS", source)
        adapter = Path(diagnose_atom.__file__).with_name("native_diffusion.py").read_text(encoding="utf-8")
        self.assertIn("native_viditq_general_build.json", adapter)
        self.assertIn("viditq_atom_per_channel_native", adapter)
        self.assertIn("_viditq_general_build(root)", source)


if __name__ == "__main__":
    unittest.main()

"""stdlib-only shape and safety tests for the Nunchaku packing diagnosis."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("diagnose_nunchaku_pack_under_test",
                                              HERE / "diagnose_nunchaku_pack.py")
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnostic)


class NunchakuPackingDiagnosticTests(unittest.TestCase):
    def test_activation_scale_writes_cover_storage_once_for_256_and_1792_rows(self):
        for rows in (256, 1792):
            for groups in (1, 4):
                writes = diagnostic.activation_scale_writes(rows, groups)
                expected_size = rows * groups
                destinations = [destination for destination, _, _ in writes]
                self.assertEqual(len(writes), expected_size)
                self.assertEqual(sorted(destinations), list(range(expected_size)))
                for group in range(groups):
                    source_rows = [row for _, row, written_group in writes
                                   if written_group == group]
                    self.assertEqual(sorted(source_rows), list(range(rows)))

    def test_prior_32_packed_half2_stride_overran_scale_storage(self):
        rows, groups = 256, 4
        last_dst = ((0 * groups + (groups - 1)) * 8 + 7) * 32 + 15
        self.assertGreater(last_dst * 2 + 1, rows * groups - 1)
        self.assertEqual(rows * groups, 1024)

    def test_cuda_packer_uses_pinned_16_lane_half2_stride(self):
        source = (HERE / "nunchaku_repack.cu").read_text(encoding="utf-8")
        start = source.index("__global__ void pack_activation_scales_kernel")
        stop = source.index("__global__ void pack_weight_scales_kernel", start)
        kernel = source[start:stop]
        self.assertIn(") * 16 + lane;", kernel)
        self.assertNotIn(") * 32 + lane;", kernel)

    def test_gpu_script_import_is_stdlib_only_and_guard_precedes_torch(self):
        source = (HERE / "diagnose_nunchaku_pack.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_torch = [node for node in tree.body
                           if isinstance(node, (ast.Import, ast.ImportFrom))
                           and ((isinstance(node, ast.Import) and
                                 any(alias.name == "torch" for alias in node.names))
                                or (isinstance(node, ast.ImportFrom) and node.module == "torch"))]
        self.assertEqual(top_level_torch, [])
        run = source[source.index("def run(root: Path)"):source.index("\n\ndef main(")]
        self.assertLess(run.index("guard()"), run.index("import torch"))
        self.assertLess(run.index("import torch"), run.index("extension = _extension"))
        self.assertEqual(diagnostic.REPRO_SHAPE, (129, 131, 256))
        self.assertEqual(diagnostic.SEED, 777)

    def test_diagnostic_uses_checkpoint_adapter_packers_and_pinned_commit(self):
        source = (HERE / "diagnose_nunchaku_pack.py").read_text(encoding="utf-8")
        self.assertIn("_nunchaku_bank(bank, module, extension", source)
        self.assertIn("extension.pack_activation_scales", source)
        self.assertIn("extension.pack_weight_scales", source)
        self.assertIn("302e0e97024ebd68688fe890e5df83731edf7b54", source)
        self.assertIn("single_one_hot_m_n_k", source)
        self.assertIn("training_updates\": 0", source)


if __name__ == "__main__":
    unittest.main()

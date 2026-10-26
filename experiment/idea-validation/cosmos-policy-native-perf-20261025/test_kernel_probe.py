"""stdlib-only checks for the guarded GPU kernel probe."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("kernel_probe_under_test", HERE / "kernel_probe.py")
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


class KernelProbeTests(unittest.TestCase):
    def test_independent_probe_failure_preserves_passes_and_continues(self):
        receipt = {'status': 'running', 'diffusion_optional_probes': {}}
        calls = []
        def passing():
            calls.append('pass')
            return {'status': 'complete', 'max_abs_error': 0}
        def failing():
            calls.append('fail')
            raise AssertionError('wrong layout')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'probe.json'
            result = probe._collect_remaining_probes(receipt, [
                (receipt['diffusion_optional_probes'], 'svdquant_w4a4', passing),
                (receipt['diffusion_optional_probes'], 'viditq_w4a4', failing),
                (receipt, 'official_hadamard', passing)], output)
            self.assertEqual(json.loads(output.read_text()), result)
        self.assertEqual(calls, ['pass', 'fail', 'pass'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['diffusion_optional_probes']['svdquant_w4a4']['status'], 'complete')
        self.assertEqual(result['official_hadamard']['status'], 'complete')
        self.assertEqual(result['diffusion_optional_probes']['viditq_w4a4']['error'], 'wrong layout')

    def test_probe_shape_matrix_has_all_requested_cases(self):
        self.assertEqual(probe.QUAROT_SHAPES,
                         tuple((m, n) for m in (1, 32, 128, 1764)
                               for n in (128, 2048, 8192)))
        self.assertEqual(probe.QUAROT_K, 128)

    def test_output_path_accepts_file_or_directory(self):
        self.assertEqual(probe._output_path(Path("x/result.json")), Path("x/result.json"))
        self.assertEqual(probe._output_path(Path("x/out")), Path("x/out/kernel_probe.json"))

    def test_uuid_mapping_respects_visible_gpu_index_or_uuid(self):
        rows = "0, GPU-aaa\n1, GPU-bbb\n"
        self.assertEqual(probe._select_device_uuid(rows, "1", 0), "GPU-bbb")
        self.assertEqual(probe._select_device_uuid(rows, "GPU-aaa", 0), "GPU-aaa")
        self.assertEqual(probe._select_device_uuid(rows, None, 1), "GPU-bbb")

    def test_source_commit_can_bind_fused_only_build_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep = root / "results" / "preparation"
            prep.mkdir(parents=True)
            (prep / "quarot_fused_build.json").write_text(
                json.dumps({"source_commit": "fused-commit"}), encoding="utf-8")
            self.assertEqual(probe._source_commit(root), "fused-commit")

    def test_probe_import_does_not_import_torch_or_run_gpu_work(self):
        source = (HERE / "kernel_probe.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_torch_imports = [node for node in tree.body
                                   if isinstance(node, (ast.Import, ast.ImportFrom))
                                   and ((isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
                                        or (isinstance(node, ast.ImportFrom) and node.module == "torch"))]
        self.assertEqual(top_level_torch_imports, [])

    def test_probe_fails_closed_on_imported_extension_outside_build_dir(self):
        source = (HERE / "kernel_probe.py").read_text(encoding="utf-8")
        self.assertIn("Imported {module_name} from outside its build directory", source)
        self.assertIn("raise AssertionError", source)

    def test_fused_scaled_add_covers_fp16_bf16_and_multiple_groups(self):
        self.assertTrue(all(groups > 1 for _, _, groups, _ in probe.FUSED_SCALED_ADD_CASES))
        self.assertTrue(all(groups_to_test[0] == 0 and groups_to_test[-1] == groups - 1
                            for _, _, groups, groups_to_test in probe.FUSED_SCALED_ADD_CASES))
        source = (HERE / "kernel_probe.py").read_text(encoding="utf-8")
        self.assertIn('("float16", torch.float16), ("bfloat16", torch.bfloat16)', source)
        self.assertIn('"non_default_current_stream"', source)
        self.assertIn('torch.cuda.graph(graph, stream=graph_stream)', source)
        self.assertIn('"group": group', source)

    def test_scaled_add_reference_keeps_separate_fp32_operation_order(self):
        source = (HERE / "kernel_probe.py").read_text(encoding="utf-8")
        start = source.index("def _reference_scaled_add(")
        end = source.index("\n\ndef _assert_close_low_precision", start)
        reference = source[start:end]
        factor = reference.index("factor = a_scales.to(torch.float32)[:, None] *")
        product = reference.index("scaled_accum = accum.to(torch.float32) * factor")
        add = reference.index("return output + scaled_accum")
        self.assertLess(factor, product)
        self.assertLess(product, add)

    def test_diffusion_probes_call_their_checkpoint_adapter_packers(self):
        source = (HERE / "kernel_probe.py").read_text(encoding="utf-8")
        nunchaku = source[source.index("def _probe_nunchaku("):source.index("def _probe_viditq(")]
        viditq = source[source.index("def _probe_viditq("):source.index("def run_probe(")]
        self.assertIn("from native_diffusion import _nunchaku_bank, _nunchaku_output_shape", nunchaku)
        self.assertIn("_nunchaku_bank(bank, module, extension", nunchaku)
        self.assertIn("from native_diffusion import _viditq_bank", viditq)
        self.assertIn("_viditq_bank(bank", viditq)
        self.assertIn("4 * padded_m", viditq)
        run_probe = source[source.index("def run_probe("):]
        self.assertLess(run_probe.index("hardware = _hardware(torch, device)"),
                        run_probe.index("_probe_fused_quarot_scaled_add(torch, device, root)"))
        self.assertLess(run_probe.index("hardware = _hardware(torch, device)"),
                        run_probe.index("_probe_nunchaku(torch, device, root)"))


class RequiredProbeTests(unittest.TestCase):
    def test_missing_or_pending_native_probe_is_not_complete(self):
        receipt = {'fused_quarot_scaled_add': {'status': 'complete'},
                   'official_hadamard': {'status': 'complete'},
                   'diffusion_optional_probes': {'svdquant_w4a4': {'status': 'complete'},
                                                  'viditq_w4a4': {'status': 'complete'}}}
        probe.require_all_native(receipt)
        receipt['official_hadamard']['status'] = 'pending'
        with self.assertRaisesRegex(RuntimeError, 'official FHT'):
            probe.require_all_native(receipt)
        del receipt['diffusion_optional_probes']['viditq_w4a4']
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            probe.require_all_native(receipt)


if __name__ == "__main__":
    unittest.main()

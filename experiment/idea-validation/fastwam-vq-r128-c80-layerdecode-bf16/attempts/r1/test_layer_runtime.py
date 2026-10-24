"""CPU-only source contracts; deliberately does not import the runtime or CUDA stack."""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("layer_runtime.py")


class LayerRuntimeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(SOURCE))

    def test_expected_modes_and_entry_points(self):
        self.assertIn('"layer_decode_fp32"', self.source)
        self.assertIn('"layer_decode_bf16_u"', self.source)
        self.assertIn('"layer_decode_bf16_all"', self.source)
        functions = {node.name for node in self.tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({"make_runtime", "install", "decode_prepared"}.issubset(functions))

    def test_decoder_uses_exact_existing_kernel_and_shared_output(self):
        self.assertIn("from fused_vq import _decode_kernel", self.source)
        self.assertIn("self.scratch = torch.empty", self.source)
        self.assertIn("decode_prepared(self.prepared[name], self.scratch)", self.source)
        self.assertNotIn("decode_weight(", self.source)
        self.assertNotIn("module.weight =", self.source)

    def test_linear_preserves_torch_functional_path(self):
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "linear"
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "F"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 3)

    def test_all_layer_validation_entries_and_diagnostics_are_exposed(self):
        self.assertIn('self.validation = {name: {"status": "pending"} for name in self.modules}', self.source)
        self.assertIn("self.decoded_calls = Counter()", self.source)
        self.assertIn("self.scratch_bytes =", self.source)
        self.assertIn('"smooth_h_max_abs"', self.source)
        self.assertIn('"rotation_max_abs"', self.source)
        self.assertIn('"a4_max_abs"', self.source)

    def test_bf16_all_keeps_global_transform_storage_in_bf16(self):
        self.assertIn('output = torch.empty(values.shape, dtype=torch.bfloat16, device=x.device)', self.source)
        self.assertIn('self.reference_transforms[name] = dict(transform)', self.source)
        self.assertIn('transform["scale"].to(dtype=torch.bfloat16)', self.source)
        self.assertLess(self.source.index('self.reference_transforms[name] = dict(transform)'),
                        self.source.index('transform["scale"].to(dtype=torch.bfloat16)'))
        self.assertIn('u_rotation_output_dtype=', self.source)
        self.assertIn('u_rotation_uses_native_bf16_gemm=', self.source)
        self.assertIn('all_transform_math_native_bf16=False', self.source)
        self.assertNotIn('_scratch_lock', self.source)
        self.assertIn('concurrent streams need one buffer per stream', self.source)

    def test_test_module_stays_stdlib_only(self):
        imported = set()
        for node in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported.intersection({"torch", "triton"}))


if __name__ == "__main__":
    unittest.main()

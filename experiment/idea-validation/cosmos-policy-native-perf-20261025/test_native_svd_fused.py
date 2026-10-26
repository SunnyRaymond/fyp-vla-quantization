import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_svd_fused


class NativeSVDFusedContractTests(unittest.TestCase):
    def test_module_import_does_not_load_torch(self):
        # This test is run in a fresh stdlib test process; importing the adapter is CPU-only.
        self.assertNotIn("torch", sys.modules)

    def test_shape_contract_pads_true_m_n_and_sidecar_k(self):
        self.assertEqual(
            native_svd_fused._fusion_shape_contract(1764, 1912, 1792, 1920, 2304, 16),
            {"activation": (1792, 2304), "down": (2304, 16),
             "up": (1920, 16), "output": (1764, 1912)},
        )

    def test_shape_contract_rejects_invalid_padding_or_rank(self):
        invalid = (
            (1764, 1912, 1791, 1920, 2304, 16),
            (1764, 1912, 1792, 1919, 2304, 16),
            (1764, 1912, 1792, 1920, 2240, 16),
            (1764, 1912, 1792, 1920, 2304, 8),
            (0, 1912, 0, 1920, 2304, 16),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                native_svd_fused._fusion_shape_contract(*args)

    def test_packing_matches_exporter_axis_order(self):
        source = (HERE / "native_svd_fused.py").read_text(encoding="utf-8")
        for layout in ("permute(2, 0, 1, 3)", "permute(0, 2, 1, 3)",
                       "reshape(c_packs, r_packs, 2, 8, 1, 2, 4, 2)",
                       "permute(0, 1, 3, 6, 2, 5, 4, 7)"):
            with self.subTest(layout=layout):
                self.assertIn(layout, source)
        self.assertIn("deepcompressor/backend/nunchaku/utils.py:147-175", source)
        self.assertIn("69f3473f5e1c1504bae35cc50c7858ef900a9b17", source)

    def test_runtime_factor_cast_checks_finite_before_and_after(self):
        source = (HERE / "native_svd_fused.py").read_text(encoding="utf-8")
        cast_start = source.index("def _fp16_factor")
        cast_end = source.index("def _pack_lowrank_weight", cast_start)
        body = source[cast_start:cast_end]
        self.assertEqual(body.count("torch.isfinite"), 2)
        self.assertIn("dtype=torch.float16", body)
        self.assertIn("checkpoint tensors are unchanged", source)

    def test_extension_is_distinct_and_candidate_is_not_silent_fallback(self):
        source = (HERE / "native_svd_fused.py").read_text(encoding="utf-8")
        self.assertIn('"nunchaku_svdq_fused_native"', source)
        self.assertIn('"native_extensions" / "svdquant_w4a4_fused"', source)
        self.assertIn("if any(not callable(getattr(extension, name, None)) for name in required)", source)
        self.assertIn("candidate_only", source)

    def test_cuda_bridge_calls_pinned_official_down_and_up_apis_once(self):
        bridge = (HERE / "nunchaku_svd_fused.cu").read_text(encoding="utf-8")
        self.assertEqual(bridge.count("PYBIND11_MODULE("), 1)
        self.assertIn("Launcher::quantize_w4a4_act_fuse_lora", bridge)
        self.assertIn("Launcher::gemm_w4a4", bridge)
        self.assertIn("false, {1.0f}, false, false, 1.0f", bridge)
        self.assertIn("padded_m - out.size(0) < 256", bridge)
        self.assertIn("packed_up.size(0) == padded_n", bridge)
        self.assertIn("rank == 16", bridge)

    def test_metadata_preserves_main_activation_and_reports_extra_quant(self):
        source = (HERE / "native_svd_fused.py").read_text(encoding="utf-8")
        for field in ("existing_per_token_whole_K_symmetric_A4",
                      "sidecar_extra_groupwise_a4_quantization",
                      "sidecar_groupwise_a4_outputs_discarded",
                      "lora_factor_runtime_dtype",
                      "native_int4_kernel_used",
                      "gpu_execution_tested"):
            with self.subTest(field=field):
                self.assertIn(field, source)

    def test_forward_slices_padded_n_before_bias_and_reshape(self):
        source = (HERE / "native_svd_fused.py").read_text(encoding="utf-8")
        slice_position = source.index("result = output[:, :_out_features].to(torch.bfloat16)")
        bias_position = source.index("if _module.bias is not None:", slice_position)
        reshape_position = source.index("return result.reshape(*original_shape[:-1], _out_features)",
                                        slice_position)
        self.assertLess(slice_position, bias_position)
        self.assertLess(bias_position, reshape_position)
        # This is a real non-tile N: the native output is [1764,1920], the Linear is N=1912.
        self.assertEqual(native_svd_fused._fusion_shape_contract(
            1764, 1912, 1792, 1920, 2304)["output"], (1764, 1912))


if __name__ == "__main__":
    unittest.main()

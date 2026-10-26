import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class BuildSVDFusedTests(unittest.TestCase):
    def test_builder_is_importable_without_torch_or_build(self):
        import sys
        sys.path.insert(0, str(HERE))
        import build_svd_fused
        self.assertEqual(build_svd_fused.NUNCHAKU_COMMIT,
                         "302e0e97024ebd68688fe890e5df83731edf7b54")
        self.assertEqual(build_svd_fused.DEEPCOMPRESSOR_COMMIT,
                         "69f3473f5e1c1504bae35cc50c7858ef900a9b17")
        self.assertEqual(build_svd_fused.MODULE_NAME, "nunchaku_svdq_fused_native")
        self.assertEqual(build_svd_fused.BUILD_DIR.as_posix(),
                         "native_extensions/svdquant_w4a4_fused")
        self.assertNotIn("torch", sys.modules)

    def test_cpu_builder_requires_real_pbs_and_limits_jobs(self):
        source = (HERE / "build_svd_fused.py").read_text(encoding="utf-8")
        main_start = source.index("def main()")
        check_position = source.index("    guard()", main_start)
        env_position = source.index("    torch, cuda_home, nvcc, nvcc_version = _check_environment()", main_start)
        self.assertLess(check_position, env_position)
        self.assertIn("from prepare_sources import guard", source)
        self.assertIn('os.environ["MAX_JOBS"] = "2"', source)
        self.assertIn('os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"', source)

    def test_receipt_records_build_provenance_and_fusion_limits(self):
        source = (HERE / "build_svd_fused.py").read_text(encoding="utf-8")
        for field in ("pbs_job_id", "source_commit", "packing_exporter_source", "module_name", "native_kernel_fusion",
                      "sidecar_groupwise_A4_outputs_discarded", "gpu_execution_tested",
                      "native_svd_fused_build.json"):
            with self.subTest(field=field):
                self.assertIn(field, source)
        self.assertNotIn("pip install", source.lower())
        self.assertIn("gemm_w4a4_launch_fp16_int4.cu", source)
        self.assertIn("nunchaku_svd_fused.cu", source)


if __name__ == "__main__":
    unittest.main()

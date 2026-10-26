import contextlib
import io
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_svd_fused


class ProbeSVDFusedTests(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys, probe_svd_fused; assert 'torch' not in sys.modules; "
            "assert 'native_diffusion' not in sys.modules"], cwd=HERE,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_torch_is_loaded_after_guard_and_before_compiled_extension(self):
        source = (HERE / 'probe_svd_fused.py').read_text(encoding='utf-8')
        run = source[source.index('def _run(root: Path'):source.index('\n\ndef main(')]
        self.assertLess(run.index('guard()'), run.index('import torch'))
        self.assertLess(run.index('import torch'), run.index('extension = _load_extension'))

    def test_requested_non_tile_cases_are_fixed(self):
        first = probe_svd_fused._case_spec(129)
        second = probe_svd_fused._case_spec(1764)
        self.assertEqual((first["padded_m"], first["true_n"], first["padded_n"],
                          first["k"], first["weight_groups_g128"], first["rank"]),
                         (256, 131, 256, 256, 2, 16))
        self.assertEqual((second["padded_m"], second["true_n"], second["padded_n"]),
                         (1792, 131, 256))

    def test_bad_kernel_shapes_fail_before_execution(self):
        for args in ((0, 131, 256), (129, 0, 256), (129, 131, 192)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                probe_svd_fused._case_spec(*args)

    def test_missing_build_receipt_fails_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                probe_svd_fused._load_build_receipt(Path(temp))

    def test_wrong_build_pin_fails_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt_dir = root / "results" / "preparation"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "native_svd_fused_build.json").write_text(
                json.dumps({"status": "complete", "pbs_job_id": "1.pbs",
                            "arch": "sm_80", "arms": {}}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                probe_svd_fused._load_build_receipt(root)

    def test_missing_module_in_build_receipt_fails_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt_dir = root / "results" / "preparation"
            receipt_dir.mkdir(parents=True)
            build_dir = root / "native_extensions" / "svdquant_w4a4_fused"
            receipt = {
                "status": "complete", "pbs_job_id": "1.pbs", "arch": "sm_80",
                "arms": {probe_svd_fused.BUILD_ARM: {
                    "source_commit": probe_svd_fused.NUNCHAKU_COMMIT,
                    "module_name": probe_svd_fused.MODULE_NAME,
                    "module": str(build_dir / (probe_svd_fused.MODULE_NAME + ".so")),
                    "packing_exporter_source": {
                        "commit": probe_svd_fused.DEEPCOMPRESSOR_COMMIT,
                        "function": "NunchakuWeightPacker.pack_lowrank_weight"},
                    "native_kernel_fusion": {
                        "sidecar_groupwise_A4_outputs_discarded": True,
                        "checkpoint_codes_scales_factors_requantized": False},
                }},
            }
            path = receipt_dir / "native_svd_fused_build.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                probe_svd_fused._load_build_receipt(root)

    def test_probe_uses_adapter_packing_and_official_fused_entrypoints(self):
        source = (HERE / "probe_svd_fused.py").read_text(encoding="utf-8")
        for marker in ("native_diffusion._nunchaku_bank(",
                       "native_svd_fused._pack_lowrank_weight(",
                       "native_svd_fused._fusion_shape_contract(",
                       "native_diffusion._nunchaku_output_shape(",
                       "extension.quantize_down(",
                       "extension.matmul_lora_up(",
                       "torch.cuda.CUDAGraph()",
                       "torch.cuda.default_stream(device)",
                       "torch.cuda.Stream(device=device)"):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)

    def test_cuda_graph_capture_path_has_no_host_scalar_sync(self):
        source = (HERE / "probe_svd_fused.py").read_text(encoding="utf-8")
        start = source.index("    def native_forward(phase):")
        end = source.index("    def trim_and_check_padding", start)
        self.assertNotIn(".item()", source[start:end])
        self.assertLess(source.index("graph_stream.synchronize()"),
                        source.index('trim_and_check_padding(graph_native_output'))

    def test_probe_syncs_each_entrypoint_and_skips_sync_inside_graph_capture(self):
        source = (HERE / "probe_svd_fused.py").read_text(encoding="utf-8")
        self.assertIn("_SynchronizedProbeExtension", source)
        self.assertIn("current_stream().synchronize()", source)
        self.assertIn('"capture_recorded"', source)
        self.assertIn("diagnostic_synchronization", source)
        for entrypoint in probe_svd_fused.FORWARD_ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint):
                self.assertIn(f'extension.{entrypoint}(', source)

    def test_sync_error_names_kernel_and_preserves_partial_receipt(self):
        class Stream:
            def synchronize(self):
                raise RuntimeError("illegal memory access")

        class Extension:
            def matmul_lora_up(self):
                return "returned"

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            current_stream=lambda: Stream()))
        with tempfile.TemporaryDirectory() as temporary:
            partial_path = Path(temporary) / "probe.partial.json"
            progress = probe_svd_fused._progress_writer(
                partial_path, {"pbs_job_id": "test-job", "model_loaded": False})
            extension = probe_svd_fused._SynchronizedProbeExtension(
                Extension(), fake_torch, progress,
                lambda: {"true_m": 129}, lambda: "default_current_stream")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
                    self.assertRaisesRegex(RuntimeError, "matmul_lora_up"):
                extension.matmul_lora_up()
            receipt = json.loads(partial_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["last_kernel_entry"]["entrypoint"], "matmul_lora_up")
            self.assertEqual(receipt["last_kernel_entry"]["state"], "sync_error")
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["model_loaded"])

    def test_capture_records_entrypoint_without_stream_synchronization(self):
        calls = []

        class Stream:
            def synchronize(self):
                calls.append("sync")

        class Extension:
            def quantize_down(self):
                return "recorded"

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: True,
            current_stream=lambda: Stream()))
        progress_rows = []
        extension = probe_svd_fused._SynchronizedProbeExtension(
            Extension(), fake_torch,
            lambda case, phase, entrypoint, state: progress_rows.append(state),
            lambda: None, lambda: "cuda_graph_capture")
        self.assertEqual(extension.quantize_down(), "recorded")
        self.assertEqual(calls, [])
        self.assertEqual(progress_rows, ["entered", "returned", "capture_recorded"])

    def test_oracle_and_tolerances_are_explicit_and_never_relaxed(self):
        self.assertEqual(probe_svd_fused.NRMSE_LIMIT, 0.01)
        self.assertEqual(probe_svd_fused.FP16_MAX_ABS_ATOL, 0.125)
        source = (HERE / "probe_svd_fused.py").read_text(encoding="utf-8")
        self.assertIn("exact FP16 operands", source)
        self.assertIn("thresholds_automatically_relaxed", source)
        self.assertIn('"thresholds_automatically_relaxed": False', source)
        self.assertIn("allow_tf32 = False", source)

    def test_output_records_zero_policy_training_and_closed_loop(self):
        source = (HERE / "probe_svd_fused.py").read_text(encoding="utf-8")
        for key in ("policy_forward_calls", "training_steps", "closed_loop_episodes",
                    "build_pbs_job_id", "nunchaku_commit", "gpu_uuid",
                    "native_kernel_calls", "graph_replay"):
            with self.subTest(key=key):
                self.assertIn(key, source)

    def test_probe_entrypoint_requires_root_and_out(self):
        source = (HERE / "probe_svd_fused.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--root", type=Path, required=True', source)
        self.assertIn('parser.add_argument("--out", type=Path, required=True', source)


if __name__ == "__main__":
    unittest.main()

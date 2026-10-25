"""GPU contract checks; run only inside the matching PBS compute allocation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import baseline_quant as baseline


class BaselineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from common import guard
        guard()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Tensor contract checks require the allocated CUDA device")
        cls.torch = torch

    def test_local_h128_is_orthogonal(self):
        torch = self.torch
        x = torch.randn(3, 256, device="cuda")
        signs = torch.where(torch.arange(256, device="cuda") % 2 == 0, 1, -1).to(torch.int8)
        restored = baseline.rotate(baseline.rotate(x, signs), signs, inverse=True)
        self.assertLess(float((restored - x).abs().max()), 2e-5)

    def test_w4_and_w8_bank_shapes_and_decoding(self):
        torch = self.torch
        weight = torch.randn(17, 256, device="cuda")
        for bits, affine in ((4, False), (4, True), (8, True)):
            encoded = baseline.quantize_weight(weight, group=128, bits=bits, affine=affine)
            decoded = baseline.decode_weight(encoded, dtype=torch.bfloat16)
            self.assertEqual(tuple(decoded.shape), tuple(weight.shape))
            self.assertTrue(bool(torch.isfinite(decoded).all()))
            self.assertEqual(encoded["codes"].dtype, torch.uint8)

    def test_method_scope_is_explicitly_adapted(self):
        self.assertEqual(set(baseline.ARMS), {
            "quarot_w4a4", "svdquant_w4a4", "viditq_w4a4",
        })
        self.assertTrue(all("cosmos_adapted" in value for value in baseline.IMPLEMENTATIONS.values()))
        self.assertNotIn("official", " ".join(baseline.IMPLEMENTATIONS.values()))

    def test_cpu_global_and_timestep_stats_fit_cuda_svd_vidit_and_w8_protection(self):
        torch = self.torch
        width = 128
        row = {
            "absmax": torch.full((width,), 2.0),
            "sum_sq": torch.full((width,), 12.0),
            "count": torch.tensor(6.0),
        }
        stats = {
            **row,
            "by_timestep": {
                str(step): {
                    "absmax": torch.full((width,), 2.0 + 0.1 * step),
                    "sum_sq": torch.full((width,), 10.0 + step),
                    "count": torch.tensor(5.0),
                }
                for step in range(5)
            },
        }
        module = torch.nn.Linear(width, 16, bias=False, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            module.weight.copy_(torch.randn_like(module.weight))

        def assert_cuda_finite(value):
            if torch.is_tensor(value):
                self.assertEqual(value.device, module.weight.device)
                if value.is_floating_point():
                    self.assertTrue(bool(torch.isfinite(value).all()))
            elif isinstance(value, dict):
                for child in value.values():
                    assert_cuda_finite(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    assert_cuda_finite(child)

        global_steps, global_max = baseline._stats_rows(row, width, torch, module.weight.device)
        assert_cuda_finite((global_steps, global_max))
        self.assertEqual(global_max.dtype, torch.float32)

        svd = baseline._fit_svd(
            module, "toy.svd", 0, stats, 128, rank=3, seed=17,
            alphas=(0.25, 0.5), method={"svd_oversampling": 2, "svd_power_iterations": 1},
            deadline=None, torch=torch,
        )
        assert_cuda_finite({key: svd[key] for key in ("encoded", "smoothing", "up", "down")})
        self.assertEqual(svd["up"].dtype, torch.bfloat16)
        self.assertEqual(svd["down"].dtype, torch.bfloat16)

        vidit = baseline._fit_vidit(
            module, "toy.vidit", 0, stats, 128, seed=23,
            alphas=(0.25, 0.5), method={}, deadline=None, torch=torch,
        )
        assert_cuda_finite({key: vidit[key] for key in ("encoded", "smoothing", "signs")})
        self.assertEqual(vidit["smoothing"].dtype, torch.float32)

        # Exercise the same refit and W8 quantization used for selected layers.
        protected = baseline._fit_vidit(
            module, "toy.vidit", 0, stats, 128, seed=23,
            alphas=(float(vidit["calibration"]["selected_alpha"]),),
            method={}, deadline=None, torch=torch,
        )
        protected["encoded"] = baseline._vidit_quantized(
            module.weight.detach().float(), protected["smoothing"], protected["signs"],
            128, 8, torch,
        )
        protected["weight_bits"] = 8
        assert_cuda_finite({key: protected[key] for key in ("encoded", "smoothing", "signs")})
        self.assertEqual(protected["encoded"]["bits"], 8)

    def test_vidit_fresh_budget_plan_and_resume_upgrade_selected_whole_layer(self):
        torch = self.torch
        protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
        alphas = baseline._alpha_candidates(protocol, "viditq_w4a4")
        self.assertEqual(len(protocol["calibration"]["case_ids"]), 80)
        self.assertEqual(protocol["inference"]["denoising_steps"], 5)
        self.assertEqual(alphas, [0.0, 0.25, 0.5, 0.75, 1.0])

        torch.manual_seed(417)
        modules = {
            "a_small": torch.nn.Linear(128, 16, bias=False, device="cuda", dtype=torch.bfloat16),
            "z_large": torch.nn.Linear(128, 4096, bias=False, device="cuda", dtype=torch.bfloat16),
        }
        stats = {}
        for name in modules:
            base = {
                "absmax": torch.linspace(1.0, 2.0, 128),
                "sum_sq": torch.linspace(6.0, 10.0, 128),
                "count": torch.tensor(5.0),
            }
            stats[name] = {
                **base,
                "by_timestep": {
                    str(step): {
                        "absmax": base["absmax"] + 0.1 * step,
                        "sum_sq": base["sum_sq"] + step,
                        "count": torch.tensor(5.0),
                    }
                    for step in range(5)
                },
            }

        with tempfile.TemporaryDirectory(prefix="vidit-protection-contract-") as temp:
            root = Path(temp)
            stats_path = root / "results" / "teacher" / "statistics.pt"
            stats_path.parent.mkdir(parents=True)
            torch.save(stats, stats_path)
            out = root / "viditq"
            runtime = SimpleNamespace(target_linears=modules)
            original_deadline_check = baseline._deadline_check
            calls_before_protection = len(modules) * (1 + len(alphas))
            call_count = 0

            def stop_before_first_w8(deadline, reserve=30):
                nonlocal call_count
                call_count += 1
                if call_count == calls_before_protection + 1:
                    raise TimeoutError("contract test: checkpoint after plan, before first W8 upgrade")
                return original_deadline_check(deadline, reserve)

            with mock.patch.object(baseline, "EXPECTED_MODULES", len(modules)), \
                    mock.patch.object(baseline, "EXPECTED_WEIGHT_NUMEL",
                                      sum(module.weight.numel() for module in modules.values())), \
                    mock.patch.object(baseline, "_deadline_check", side_effect=stop_before_first_w8):
                checkpoint = baseline.calibrate_baseline(
                    runtime, protocol, "viditq_w4a4", root, out, soft_deadline_seconds=0,
                )
                self.assertEqual(checkpoint["phase"], "whole_layer_protection")
                manifest_path = out / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                plan = manifest["protection"]["selected_modules"]
                self.assertEqual(plan, ["a_small"])
                by_name = {entry["name"]: entry for entry in manifest["entries"]}
                self.assertEqual(by_name["a_small"]["weight_bits"], 4)
                self.assertEqual(by_name["z_large"]["weight_bits"], 4)

            with mock.patch.object(baseline, "EXPECTED_MODULES", len(modules)), \
                    mock.patch.object(baseline, "EXPECTED_WEIGHT_NUMEL",
                                      sum(module.weight.numel() for module in modules.values())):
                resumed = baseline.calibrate_baseline(
                    runtime, protocol, "viditq_w4a4", root, out, soft_deadline_seconds=0,
                )
                final_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                final_entries = {entry["name"]: entry for entry in final_manifest["entries"]}
                self.assertEqual(resumed["status"], "complete")
                self.assertEqual(final_manifest["protection"]["selected_modules"], ["a_small"])
                self.assertEqual(final_entries["a_small"]["weight_bits"], 8)
                self.assertEqual(final_entries["z_large"]["weight_bits"], 4)
                self.assertTrue(resumed["bpw_within_target_tolerance"])


if __name__ == "__main__":
    unittest.main()

"""CPU toy checks for scope dispatch, exception restoration, and reference counters."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from stage_quantization import StageQuantization, scope_for_name


class ToyLinear:
    def __init__(self, bias: torch.Tensor | None = None):
        self.in_features, self.out_features = 128, 2
        self.bias = bias
        codes = torch.zeros((2, 128), dtype=torch.int8)
        codes[0].fill_(1)
        codes[1].fill_(2)
        unsigned = codes.to(torch.int16) & 15
        self._quant_packed_weight = (
            unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
        ).to(torch.uint8)
        self._quant_weight_scales = torch.tensor([[0.5], [1.0]], dtype=torch.bfloat16)


class ToyQuantization:
    group_size = 128

    def __init__(self, fail: bool = False):
        self.modules = {
            "video_expert.block": ToyLinear(torch.tensor([0.25, -0.5], dtype=torch.bfloat16)),
            "action_expert.block": ToyLinear(),
            "proprio_encoder.block": ToyLinear(),
        }
        self._activation_bits = 8
        self._kv_active = False
        self._integer_gemm_calls = 0
        self._native_int4_gemm_calls = 0
        self._arm = None
        self.seen_bits: list[int] = []
        self.fail = fail
        self._linear = self.native_linear

    def enable(self, arm: str):
        self._arm = arm
        self._activation_bits = 8 if arm == "w4a8" else 4
        self._kv_active = arm == "w4a4kv4"
        return self

    def native_linear(self, module, x):
        self.seen_bits.append(self._activation_bits)
        self._integer_gemm_calls += 1
        if self.fail:
            raise RuntimeError("toy native failure")
        return torch.full(
            (*x.shape[:-1], module.out_features), float(self._activation_bits),
            dtype=torch.bfloat16,
        )


class StageQuantizationTest(unittest.TestCase):
    def test_scope_and_mixed_dispatch_restore(self):
        self.assertEqual(scope_for_name("video_expert.layers.0"), "video")
        self.assertEqual(scope_for_name("mot.mixtures.action.layers.0"), "action")
        self.assertEqual(scope_for_name("proprio_encoder.proj"), "proprio")
        q = ToyQuantization()
        stage_q = StageQuantization(q).configure((4, 8, 4))
        self.assertEqual(q._arm, "w4a4")
        self.assertFalse(q._kv_active)
        x = torch.ones((1, 128), dtype=torch.bfloat16)
        video, action = q.modules["video_expert.block"], q.modules["action_expert.block"]
        self.assertTrue(torch.equal(q._linear(video, x), torch.full((1, 2), 4, dtype=torch.bfloat16)))
        self.assertEqual(q._activation_bits, 4)
        self.assertTrue(torch.equal(q._linear(action, x), torch.full((1, 2), 8, dtype=torch.bfloat16)))
        self.assertEqual(q._activation_bits, 4)
        self.assertEqual(q.seen_bits, [4, 8])

    def test_pure_arm_matches_original_and_finally_restores_on_error(self):
        x = torch.ones((1, 128), dtype=torch.bfloat16)
        q = ToyQuantization()
        stage_q = StageQuantization(q)
        module = q.modules["video_expert.block"]
        for bits, arm in (((4, 4, 4), "w4a4"), ((8, 8, 8), "w4a8")):
            stage_q.configure(bits)
            expected = stage_q._native_linear(module, x)
            actual = q._linear(module, x)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(q._arm, arm)
            self.assertEqual(q._activation_bits, bits[0])

        failing_q = ToyQuantization(fail=True)
        StageQuantization(failing_q).configure((4, 8, 8))
        with self.assertRaisesRegex(RuntimeError, "toy native failure"):
            failing_q._linear(failing_q.modules["video_expert.block"], x)
        self.assertEqual(failing_q._activation_bits, 4)

    def test_reference_is_independent_and_trace_keeps_repeated_calls(self):
        q = ToyQuantization()
        stage_q = StageQuantization(q).configure((4, 4, 4))
        module = q.modules["video_expert.block"]
        x = torch.ones((1, 128), dtype=torch.bfloat16)
        before = (q._integer_gemm_calls, q._native_int4_gemm_calls)
        with stage_q.reference_mode():
            output = q._linear(module, x)
        expected = torch.tensor([[64.25, 255.5]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual((q._integer_gemm_calls, q._native_int4_gemm_calls), before)

        with tempfile.TemporaryDirectory() as temp_dir:
            with stage_q.trace_mode(Path(temp_dir), {"query": "toy"}) as trace:
                trace.set_stage("video", 0)
                q._linear(module, x)
                q._linear(module, x)
            report = stage_q.finish_trace()
            self.assertEqual(report["calls"], 2)
            self.assertEqual(report["coverage"][0]["calls"], 2)
            self.assertTrue(Path(report["trace_path"]).is_file())


if __name__ == "__main__":
    unittest.main()

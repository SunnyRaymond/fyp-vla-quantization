import unittest

from rotation import self_check


class RotationCoreTests(unittest.TestCase):
    def test_rotation_core_and_bank_state(self):
        result = self_check()
        self.assertTrue(result["passed"])
        self.assertTrue(result["fp32_rotation_equivalence"])
        self.assertTrue(result["static_sub_r128_proprio_tail"])
        self.assertTrue(result["ste_bf16_precision"]["passed"])
        self.assertEqual(
            result["ste_bf16_precision"]["fp32_reference_max_abs_error"], 0.0
        )
        self.assertGreater(
            result["ste_bf16_precision"]["legacy_bf16_operand_max_abs_error"], 0.0
        )
        self.assertTrue(result["learned_initial_equals_fixed"])
        self.assertTrue(result["w4a4_ste_rotation_gradient"])
        self.assertTrue(result["gradients_through_input_and_weight_rotations"])
        self.assertTrue(result["cayley_orthogonality_after_update"])
        self.assertTrue(result["native_bank_state_roundtrip"])
        self.assertTrue(result["native_packed_bank_switch"])
        self.assertTrue(result["native_bank_discard"])


if __name__ == "__main__":
    unittest.main()

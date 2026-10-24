"""Workspace lifetime and capacity checks without importing Torch locally."""
import unittest

from abc_runtime import WORKSPACE_ELEMENTS, workspace_layout


class WorkspaceLayout(unittest.TestCase):
    def test_borrowed_h_u_fit_and_a4_stays_independent(self):
        original = workspace_layout(113246208, False)
        shared = workspace_layout(113246208, True)
        self.assertEqual(original['independent_bytes'], 42147840)
        self.assertEqual(shared['independent_bytes'], WORKSPACE_ELEMENTS * 2)
        self.assertEqual(shared['borrowed_weight_scratch_bytes'], WORKSPACE_ELEMENTS * 8)
        self.assertEqual(original['independent_bytes'] - shared['independent_bytes'], 33718272)

    def test_small_weight_scratch_is_rejected(self):
        with self.assertRaises(ValueError):
            workspace_layout(WORKSPACE_ELEMENTS * 8 - 1, True)


if __name__ == '__main__':
    unittest.main()

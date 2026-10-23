"""Storage and startup interfaces only; numerical checks run on allocated GPUs."""
import json
from pathlib import Path
import unittest

import diffusion_quant as quant


class BudgetTests(unittest.TestCase):
    def test_budget_counts_padding_zero_points_and_bf16_factors(self):
        costs = quant.estimate_storage([(10, 129)], rank=3)
        svd, vidit = (costs[arm] for arm in quant.ARMS)
        self.assertEqual(svd['bytes_by_kind']['packed'], 1280)
        self.assertEqual(svd['bytes_by_kind']['scales'], 80)
        self.assertEqual(svd['bytes_by_kind']['low_rank_factors'], 2 * 3 * (10 + 129))
        self.assertEqual(vidit['bytes_by_kind']['zero_points'], 20)
        self.assertEqual(vidit['bytes_by_kind']['rotation_signs'], 129)
        self.assertEqual(svd['parameters'], 1290)

    def test_shape_only_rank_is_largest_affordable_nonzero_candidate(self):
        protocol = json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
        budget = quant.select_rank([(2048, 2048)] * 4, protocol)
        self.assertEqual(budget['global_rank'], 16)
        self.assertLessEqual(budget['relative_bpw_gap'], .05)
        self.assertTrue(all(cost['bpw'] <= 4.60 for cost in budget['costs'].values()))
        protocol['quantization']['maximum_target_bpw'] = 3.9
        with self.assertRaises(RuntimeError):
            quant.select_rank([(2048, 2048)] * 4, protocol)

    def test_full_hadamard_for_supported_widths_and_explicit_physical_tail(self):
        self.assertEqual(quant.rotation_layout(4096), 'full_power2')
        self.assertEqual(quant.rotation_layout(3072), 'full_h12_power2')
        self.assertEqual(quant.rotation_layout(7), 'dyadic_tail')
        pbs = Path(__file__).with_name('run.pbs').read_text(encoding='utf-8')
        self.assertNotIn('SCRIPT_DIR=$(cd', pbs)


if __name__ == '__main__':
    unittest.main()

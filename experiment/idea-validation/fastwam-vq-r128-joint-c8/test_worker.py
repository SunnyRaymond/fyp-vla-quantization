"""Frozen scientific scope checks without importing Torch or any model."""
import copy
import json
from pathlib import Path
import unittest
from worker import validate_protocol


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())

    def test_declared_protocol_is_accepted(self):
        validate_protocol(self.protocol)

    def test_changed_budget_lr_loss_or_cohort_are_rejected(self):
        mutations = [('training', 'joint_updates', 16), ('training', 'rotation_learning_rate', .001),
                     ('training', 'rho', .1), ('evaluation', 'case_ids', list(range(3000, 3004))),
                     ('training', 'update_order', list(reversed(self.protocol['calibration']['case_ids'])))]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.protocol); changed[section][key] = value
                with self.assertRaises(RuntimeError):
                    validate_protocol(changed)


if __name__ == '__main__':
    unittest.main()

"""Check the frozen refinement scope without importing numerical runtimes."""
import copy
import json
from pathlib import Path
import unittest
from worker import validate_protocol


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())

    def test_declared_scope_is_accepted(self):
        validate_protocol(self.protocol)

    def test_budget_rotation_lr_and_incumbent_loss_cannot_drift(self):
        for section, key, value in [('training', 'joint_updates', 16), ('training', 'control_updates', 16),
                ('training', 'rotation_learning_rate', .0001), ('training', 'rho', .1),
                ('evaluation', 'seed_indices', [0])]:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.protocol); changed[section][key] = value
                with self.assertRaises(RuntimeError):
                    validate_protocol(changed)


if __name__ == '__main__':
    unittest.main()

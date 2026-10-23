"""Pure-stdlib contract checks; GPU numerical checks stay in the allocation."""
import copy
import json
from pathlib import Path
import unittest

from path_worker import METRICS, validate_protocol, validate_record


class PathContracts(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())

    def test_fixed_orthogonal_path_endpoints(self):
        validate_protocol(self.protocol)
        changed = copy.deepcopy(self.protocol)
        changed['lanes']['path']['arms'][1]['t'] = .1
        with self.assertRaises(RuntimeError):
            validate_protocol(changed)

    def test_resume_cannot_reuse_wrong_arm_or_nonfinite_metrics(self):
        binding = dict(protocol=self.protocol['protocol'], arm='Rt025_W16A4', t=.25,
                       case_id=2000, seed_index=0)
        record = dict(binding, **{m: .001 for m in METRICS})
        self.assertEqual(validate_record(record, binding), record)
        record['t'] = .5
        with self.assertRaises(RuntimeError):
            validate_record(record, binding)
        record['t'] = .25
        record['mixed_loss'] = float('nan')
        with self.assertRaises(RuntimeError):
            validate_record(record, binding)


if __name__ == '__main__':
    unittest.main()

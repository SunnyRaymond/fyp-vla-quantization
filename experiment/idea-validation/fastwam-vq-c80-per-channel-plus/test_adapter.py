"""Small source-boundary checks; no Torch or model computation."""
import copy
import json
from pathlib import Path
import unittest

import vq_adapter

HERE = Path(__file__).resolve().parent


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
        cls.receipt = json.loads((HERE / 'source_training_receipt.json').read_text(encoding='utf-8'))
        cls.endpoint = json.loads((HERE / 'source_reference_endpoint.json').read_text(encoding='utf-8'))

    def test_completed_final80_source_receipt(self):
        self.assertIs(vq_adapter.validate_source_receipt(self.protocol, self.receipt), self.receipt)
        for key, value in (('updates', 79), ('indices_unchanged', False), ('pbs_jobid', 'other.pbs101')):
            with self.subTest(field=key):
                broken = copy.deepcopy(self.receipt)
                broken[key] = value
                with self.assertRaises(ValueError):
                    vq_adapter.validate_source_receipt(self.protocol, broken)
        broken = copy.deepcopy(self.receipt)
        broken['scaling']['group_size'] = 32
        with self.assertRaises(ValueError):
            vq_adapter.validate_source_receipt(self.protocol, broken)

    def test_reproduce_saved_source_metrics_without_claiming_saved_action_replay(self):
        errors = vq_adapter.compare_endpoint_metrics(self.endpoint, self.endpoint, self.protocol)
        self.assertEqual(set(errors), set(vq_adapter.METRICS))
        self.assertTrue(all(value == 0 for value in errors.values()))
        changed = dict(self.endpoint, first10motor_rmse=self.endpoint['first10motor_rmse'] + .001)
        with self.assertRaises(ValueError):
            vq_adapter.compare_endpoint_metrics(changed, self.endpoint, self.protocol)
        changed = dict(self.endpoint, future_latent_nmse=float('nan'))
        with self.assertRaises(ValueError):
            vq_adapter.compare_endpoint_metrics(changed, self.endpoint, self.protocol)


if __name__ == '__main__':
    unittest.main()

"""Small source-binding checks without Torch, model loading, or real episodes."""
import copy
import json
from pathlib import Path
import unittest

import runtime

HERE = Path(__file__).parent


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))

    def receipt(self, arm):
        return json.loads((HERE / f'source_{arm}_training_receipt.json').read_text(encoding='utf-8'))

    def test_both_completed_source_receipts(self):
        for arm in runtime.ARMS:
            runtime.validate_source_receipt(self.protocol, arm, self.receipt(arm))

    def test_swapped_checkpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            runtime.validate_source_receipt(self.protocol, 'softbin', self.receipt('relational'))

    def test_changed_update_order_is_rejected(self):
        receipt = self.receipt('softbin')
        receipt['history'][0]['case_id'] += 1
        with self.assertRaises(ValueError):
            runtime.validate_source_receipt(self.protocol, 'softbin', receipt)

    def test_changed_frozen_weights_are_rejected(self):
        receipt = self.receipt('relational')
        receipt['codebooks_unchanged'] = False
        with self.assertRaises(ValueError):
            runtime.validate_source_receipt(self.protocol, 'relational', receipt)

    def test_endpoint_requires_exact_replay_job_and_seed(self):
        for arm in runtime.ARMS:
            endpoint = json.loads((HERE / f'source_{arm}_endpoint.json').read_text(encoding='utf-8'))
            runtime.validate_endpoint_identity(endpoint, self.protocol, arm, endpoint['plan_row'])
            wrong = copy.deepcopy(endpoint)
            wrong['sampler_seed'] += 1
            with self.assertRaises(ValueError):
                runtime.validate_endpoint_identity(wrong, self.protocol, arm, endpoint['plan_row'])
            wrong = copy.deepcopy(endpoint)
            wrong['pbs_jobid'] = self.protocol['arm_sources'][arm]['training_job']
            if wrong['pbs_jobid'] != endpoint['pbs_jobid']:
                with self.assertRaises(ValueError):
                    runtime.validate_endpoint_identity(wrong, self.protocol, arm, endpoint['plan_row'])

    def test_metadata_records_checkpoint_reuse_without_training(self):
        for arm in runtime.ARMS:
            metadata = runtime.expected_install_metadata(self.protocol, arm)
            self.assertEqual(metadata['new_training_updates'], 0)
            self.assertEqual(metadata['training_updates'], 80)
            self.assertEqual(metadata['module_count'], 614)
            self.assertEqual(metadata['source_deployment'], self.protocol['arm_sources'][arm]['deployment'])


if __name__ == '__main__':
    unittest.main()

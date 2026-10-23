"""Stdlib checks for pinned source identity and metric replay boundaries."""
import copy
import json
from pathlib import Path
import unittest

import runtime

HERE = Path(__file__).parent


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))

    def documents(self, arm):
        names = ('source_protocol_file', 'source_summary', 'endpoint')
        result = [json.loads((HERE / f'source_{arm}_{name}.json').read_text(encoding='utf-8'))
                  for name in names]
        if arm == runtime.ARMS[1]:
            result.append(json.loads((HERE / f'source_{arm}_arm_summary.json').read_text(encoding='utf-8')))
        return result

    def test_both_completed_source_bindings(self):
        for arm in runtime.ARMS:
            runtime.validate_source_documents(self.protocol, arm, *self.documents(arm))

    def test_swapped_endpoint_is_rejected(self):
        docs = self.documents(runtime.ARMS[0])
        docs[2] = self.documents(runtime.ARMS[1])[2]
        with self.assertRaises(ValueError):
            runtime.validate_source_documents(self.protocol, runtime.ARMS[0], *docs)

    def test_changed_final_update_or_exposures_are_rejected(self):
        arm = runtime.ARMS[1]
        for key, wrong in (('updates', 7), ('training_exposures', 79)):
            docs = self.documents(arm)
            docs[1]['arms']['C80_once_lr1e4'][key] = wrong
            with self.assertRaises(ValueError):
                runtime.validate_source_documents(self.protocol, arm, *docs)

    def test_metric_replay_tolerance_and_nonfinite(self):
        expected = self.documents(runtime.ARMS[0])[2]
        values = {key: expected[key] for key in runtime.METRICS}
        self.assertEqual(max(runtime.compare_source_metrics(values, expected, 2e-6).values()), 0)
        for wrong in (values['first10motor_rmse'] + 1e-3, float('nan')):
            actual = copy.deepcopy(values)
            actual['first10motor_rmse'] = wrong
            with self.assertRaises(ValueError):
                runtime.compare_source_metrics(actual, expected, 2e-6)

    def test_metadata_keeps_w16_without_vq_or_new_training(self):
        for arm, updates in zip(runtime.ARMS, (0, 8)):
            metadata = runtime.expected_install_metadata(self.protocol, arm)
            self.assertEqual(metadata['weight_bits'], 16)
            self.assertEqual(metadata['activation_bits'], 4)
            self.assertFalse(metadata['vq_enabled'])
            self.assertEqual(metadata['rotation_updates'], updates)
            self.assertEqual(metadata['new_training_updates'], 0)


if __name__ == '__main__':
    unittest.main()

"""Small receipt checks; numerical validation runs inside the GPU allocation."""
import copy
import unittest

from validation import panel_keys, identity, valid_receipt


class ValidationContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {'protocol': 'p', 'performance': {
            'case_ids': list(range(3000, 3040)), 'seed_indices': [0, 1]}}
        self.expected = identity(self.protocol, {'source_job': '1.pbs101',
                                 'completed_updates': 8}, {'name': 'GPU'})
        self.receipt = {**self.expected, 'status': 'passed', 'checked_queries': 80,
                        'expected_queries': 80, 'failed_queries': [],
                        'cuda_graph': {'capture_errors': [], 'graph_replays': 2}}

    def test_full_panel_receipt_and_unique_keys(self):
        self.assertEqual(len(panel_keys(self.protocol)), 80)
        self.assertTrue(valid_receipt(self.receipt, self.expected))

    def test_wrong_artifact_incomplete_or_failed_graph_is_not_reused(self):
        for key, value in [('source_job', '2.pbs101'), ('checked_queries', 79),
                           ('failed_queries', [{'failed': True}]), ('status', 'checkpointed')]:
            changed = {**self.receipt, key: value}
            self.assertFalse(valid_receipt(changed, self.expected))
        changed = copy.deepcopy(self.receipt)
        changed['cuda_graph']['capture_errors'] = ['capture failed']
        self.assertFalse(valid_receipt(changed, self.expected))


if __name__ == '__main__':
    unittest.main()

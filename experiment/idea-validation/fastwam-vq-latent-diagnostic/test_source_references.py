"""The resumable evaluation table, not segment-local JSONL, owns cached actions."""
import unittest
from run_future import selected_references


class SelectedReferenceTests(unittest.TestCase):
    def test_selected_epoch_contains_all_three_splits(self):
        rows = [dict(case_id=case, seed_index=0, epoch=epoch, split=split, action=[[float(epoch)] * 7] * 32)
                for epoch in (1, 2) for case, split in ((0, 'calibration'), (1000, 'selection'), (1004, 'test'))]
        indexed = selected_references(rows, 2)
        self.assertEqual(set(indexed), {(0, 0), (1000, 0), (1004, 0)})
        self.assertEqual(indexed[0, 0]['action'][0][0], 2.0)

    def test_duplicate_selected_reference_rejected(self):
        row = dict(case_id=0, seed_index=0, epoch=2)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            selected_references([row, row], 2)


if __name__ == '__main__':
    unittest.main()

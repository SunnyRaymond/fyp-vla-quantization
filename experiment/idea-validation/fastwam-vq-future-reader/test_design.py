"""Small tests for pairing and denominators; no torch/model imports."""
import json
from pathlib import Path
import unittest
from common import aggregate, paired, contexts


class ScopeTests(unittest.TestCase):
    def test_full_scope(self):
        p = json.loads(Path(__file__).with_name('protocol.json').read_text())
        rows = contexts(p)
        self.assertEqual(len(rows), 60)
        self.assertEqual([sum(r['split'] == s for r in rows) for s in ('calibration', 'selection', 'test')], [8, 8, 44])
        for case in p['test_cases']:
            pair = [r for r in rows if r['case_id'] == case]
            self.assertEqual([r['seed_index'] for r in pair], [0, 1])
            self.assertEqual(len({r['shard'] for r in pair}), 1)

    def test_case_equal_weight(self):
        rows = [dict(case_id=1, seed_index=0, m=1), dict(case_id=1, seed_index=1, m=3), dict(case_id=2, seed_index=0, m=0)]
        self.assertEqual(aggregate(rows, 'm')['mean'], 1)
        with self.assertRaises(RuntimeError):
            aggregate(rows + rows[:1], 'm')

    def test_paired_sign(self):
        a = [dict(case_id=1, seed_index=0, m=1), dict(case_id=2, seed_index=0, m=4)]
        b = [dict(case_id=1, seed_index=0, m=2), dict(case_id=2, seed_index=0, m=3)]
        result = paired(a, b, 'm')
        self.assertEqual((result['improved'], result['worsened'], result['mean_delta']), (1, 1, 0))
        with self.assertRaises(RuntimeError):
            paired(a, b[:1], 'm')

    def test_gate_and_denominators(self):
        from analyze import summarize, METRICS
        p = json.loads(Path(__file__).with_name('protocol.json').read_text())
        rows = []
        for c in contexts(p):
            for donor, value in [('bf16', 0), ('parent', 2), ('action_only', 3), ('action_latent', 1)]:
                rows.append(dict(**c, reader='bf16', donor=donor, **{m: value for m in METRICS}))
        result = summarize(p, rows, 'teacher')
        self.assertTrue(result['cross_gate_passed'])
        self.assertEqual(result['tables']['test/bf16/action_latent']['first10motor_rmse']['context_count'], 44)
        self.assertEqual(result['comparisons']['test/bf16/action_only-minus-parent']['first10motor_rmse']['worsened'], 22)
        with self.assertRaises(RuntimeError):
            summarize(p, rows[:-1], 'teacher')


if __name__ == '__main__':
    unittest.main()

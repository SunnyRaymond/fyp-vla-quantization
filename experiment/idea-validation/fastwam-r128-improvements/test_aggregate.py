"""Meaningful paired-cohort and cross-lane report checks, with artificial data."""
import copy
import unittest

from aggregate import combine, comparisons, PROTOCOL
from cohort_metrics import aggregate_records, METRICS, SUITES


def fixture(value=1., cases=40):
    records = []
    for i in range(cases):
        for seed in (0, 1):
            metrics = {m: value for m in METRICS}
            records.append(dict(case_id=3000+i, seed_index=seed, suite=SUITES[i//(cases//4)], **metrics))
    return aggregate_records(records)


class Reports(unittest.TestCase):
    def test_factorial_contrasts_preserve_direction_and_unit(self):
        arms = {name: fixture(v) for name, v in [('H_W16A4', 2), ('R8_W16A4', 4),
            ('grouped_lr1e4', 3), ('interleaved_lr1e3', 2), ('interleaved_lr1e4', 1)]}
        x = comparisons('optimization', arms)
        self.assertEqual(x['grouped_lr1e4_vs_R8_W16A4']['candidate_relative_change_percent']['mixed_loss'], -25)
        self.assertEqual(x['interleaved_lr1e4_vs_grouped_lr1e4']['paired_case_win_loss']['mixed_loss']['cases'], 40)

    def test_missing_seed_is_not_a_complete_context(self):
        rows = [dict(case_id=3000+i, seed_index=0, suite=SUITES[i//10],
                     **{m: 1 for m in METRICS}) for i in range(40)]
        with self.assertRaises(ValueError):
            aggregate_records(rows)

    def test_combination_requires_all_lanes_and_identical_anchors(self):
        anchors = {'H_W16A4': fixture(2), 'R8_W16A4': fixture(4)}
        reports = []
        for lane in ['path', 'optimization', 'coverage_c80']:
            x = dict(protocol=PROTOCOL, complete=True, lane=lane, heldout=copy.deepcopy(anchors),
                     calibration=copy.deepcopy(anchors), comparisons={}, frozen_design={})
            if lane.startswith('coverage_'):
                name = 'C80_once_lr1e4'
                x['heldout'][name] = fixture(1)
                x['c80'] = {name: fixture(1)}
            reports.append(x)
        with self.assertRaises(ValueError):
            combine(reports[:-1])
        merged = combine(reports)
        self.assertEqual(merged['comparisons']['C80_once_lr1e4_vs_R8_W16A4']
                         ['candidate_relative_change_percent']['mixed_loss'], -75)
        reports[-1]['heldout']['H_W16A4']['overall']['mixed_loss'] = 3
        with self.assertRaises(ValueError):
            combine(reports)


if __name__ == '__main__':
    unittest.main()

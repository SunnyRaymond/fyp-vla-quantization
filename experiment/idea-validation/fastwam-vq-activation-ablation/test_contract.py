"""Small standard-library checks for source swaps and frozen endpoint coverage."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('clipping_reference',HERE.parent/'fastwam-vq-activation-clipping/experiment.py')
ref=importlib.util.module_from_spec(spec)
sys.modules['clipping_reference']=ref
spec.loader.exec_module(ref)
spec=importlib.util.spec_from_file_location('ablation_eval',HERE/'evaluate.py')
ev=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
P=json.loads((HERE/'protocol.json').read_text())


class Contract(unittest.TestCase):
    def test_source_swaps_and_total_scope(self):
        a=ev.binding(P,'joint_w_ratio1','heldout','joint_w_ratio1')
        b=ev.binding(P,'baseline_w_joint_ratios','heldout','baseline_w_joint_ratios')
        self.assertEqual((a['weights'],a['ratios']),('joint','one'))
        self.assertEqual((b['weights'],b['ratios']),('baseline','joint'))
        self.assertCountEqual([arm for unit in P['units'].values() for arm in unit['c8']],P['c8_evaluation']['arms'])
        self.assertEqual(2*len(P['evaluation']['case_ids'])*2+4*len(P['calibration']['case_ids']),192)
        self.assertEqual(P['training_updates'],0)

    def test_c8_duplicate_or_missing_cannot_pass(self):
        rows=[{'case_id':case,'seed_index':0,'suite':ref.SUITES[i//2],
               **{m:1. for m in P['c8_evaluation']['metrics']}}
              for i,case in enumerate(P['calibration']['case_ids'])]
        self.assertEqual(ev.aggregate_c8(rows,P)['queries'],8)
        with self.assertRaises(ValueError):ev.aggregate_c8(rows[:-1],P)
        with self.assertRaises(ValueError):ev.aggregate_c8(rows[:-1]+[rows[0]],P)
        bad=[dict(row) for row in rows]
        bad[0]['seed_index']=1
        with self.assertRaises(ValueError):ev.aggregate_c8(bad,P)


if __name__=='__main__':
    unittest.main()

import copy
import json
from pathlib import Path
import unittest

from aggregate import summarize
from evaluate import ARMS, TIMING_FIELDS


class PairedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))
        self.rows=[dict(suite=f's{i//70}',dimension=f'd{(i%70)//10}') for i in range(280)]
        self.workers={str(i):dict(complete=True) for i in range(4)}
        self.arms={arm:{i:dict(success=i<100,metrics=[dict(
            {field:1. for field in TIMING_FIELDS},first_query_in_run=False)])
            for i in range(280)} for arm in ARMS}

    def test_paired_denominator_and_direction(self):
        self.arms['ours'][99]['success']=False
        self.arms['ours'][100]['success']=True
        self.arms['ours'][101]['success']=True
        result=summarize(self.arms,self.rows,self.workers,self.protocol)
        self.assertTrue(result['complete'])
        self.assertEqual(result['episodes'],560)
        paired=result['overall']['ours_vs_reconstruction']
        self.assertEqual((paired['wins'],paired['losses'],paired['ties'],paired['episodes']),(2,1,277,280))
        self.assertAlmostEqual(paired['success_rate_difference'],1/280)
        self.assertEqual(result['by_suite']['s0']['expected_episodes_per_arm'],70)
        self.assertEqual(result['by_dimension']['d0']['expected_episodes_per_arm'],40)
        self.assertEqual(result['by_cell']['s0/d0']['expected_episodes_per_arm'],10)

    def test_partial_coverage_is_never_complete(self):
        del self.arms['ours'][279]
        result=summarize(self.arms,self.rows,self.workers,self.protocol)
        self.assertFalse(result['complete'])
        self.assertEqual(result['episodes'],559)
        self.assertEqual(result['overall']['ours_vs_reconstruction']['episodes'],279)
        self.assertEqual(result['overall']['expected_episodes_per_arm'],280)

    def test_complete_records_need_all_worker_receipts(self):
        self.workers['3']['complete']=False
        result=summarize(self.arms,self.rows,self.workers,self.protocol)
        self.assertFalse(result['complete'])

    def test_invalid_timing_cannot_be_summarized(self):
        self.arms['ours'][0]['metrics'][0]['full_query_wall_ms']=float('nan')
        with self.assertRaises(ValueError):
            summarize(self.arms,self.rows,self.workers,self.protocol)

    def test_real_aggregate_schema_matches_controller_gate(self):
        from test_control import control, complete_value
        value=complete_value()
        value['results']['summary']=summarize(self.arms,self.rows,self.workers,self.protocol)
        self.assertTrue(control.complete_gate(value))
        del self.arms['ours'][279]
        value['results']['summary']=summarize(self.arms,self.rows,self.workers,self.protocol)
        self.assertFalse(control.complete_gate(value))


if __name__=='__main__':
    unittest.main()

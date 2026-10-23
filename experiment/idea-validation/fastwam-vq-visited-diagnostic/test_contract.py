"""Small metadata tests; imports no model or numerical runtime."""
import copy
import json
from pathlib import Path
import unittest

import worker


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((Path(__file__).parent/'protocol.json').read_text())

    def test_frozen_full_scope(self):
        self.assertEqual(worker.validate_protocol(self.p)['expected_records'],600)
        for key,value in (('case_ids',self.p['case_ids'][:-1]),('training_updates',1),('seed_index',1)):
            p=copy.deepcopy(self.p);p[key]=value
            with self.assertRaises(ValueError):worker.validate_protocol(p)

    def test_missing_or_cross_input_record_rejected(self):
        records=[]
        for case in self.p['case_ids']:
            for arm in self.p['arms']:
                for variant in self.p['variants']:
                    records.append({'case_id':case,'arm':arm,'variant':variant,'sampler_seed':case,'input_perturbation':{'variant':variant}})
        worker.require_panel(records,self.p)
        with self.assertRaises(ValueError):worker.require_panel(records[:-1],self.p)
        records[0]['sampler_seed']=-1
        with self.assertRaises(ValueError):worker.require_panel(records,self.p)

    def test_geometry_detects_wrong_direction(self):
        def sample(value):return {'action':[[value]*7 for _ in range(32)]}
        base,positive,negative=sample(0),sample(1),sample(-1)
        exact=worker.geometry(base,positive,negative,base,positive,negative,1)
        reversed_=worker.geometry(base,negative,positive,base,positive,negative,1)
        flat=worker.geometry(base,base,base,base,positive,negative,1)
        self.assertEqual(exact['error_rms'],0)
        self.assertEqual(exact['cosine'],1)
        self.assertEqual(reversed_['cosine'],-1)
        self.assertEqual(reversed_['error_rms'],2)
        self.assertEqual(flat['error_rms'],1)
        self.assertIsNone(flat['cosine'])


if __name__=='__main__':unittest.main()

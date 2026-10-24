"""Small contract/paired aggregation checks; tensor/model checks run in PBS only."""
import ast
import copy
import json
from pathlib import Path
import unittest
from worker import validate, aggregate, METRICS


class ContractTests(unittest.TestCase):
    def test_rejects_changed_budget_loss_or_cohort(self):
        p=json.loads(Path(__file__).with_name('protocol.json').read_text())
        validate(p)
        for section,key,value in [('training','updates',8),('training','beta',0),
                                  ('clipping','initial_ratio',.9),('evaluation','seed_indices',[0])]:
            other=copy.deepcopy(p)
            other[section][key]=value
            with self.assertRaises(ValueError):validate(other)

    def test_paired_seed_then_context_then_suite_aggregation(self):
        cases=list(range(8));suites=['a','b','c','d']
        rows=[dict(case_id=c,seed_index=s,suite=suites[c//2],**{m:float(c+s) for m in METRICS})
              for c in cases for s in (0,1)]
        value=aggregate(rows,cases,[0,1],suites)
        self.assertEqual(value['queries'],16)
        self.assertEqual(value['overall']['first10motor_rmse'],4.)
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):aggregate(bad,cases,[0,1],suites)

    def test_sources_parse_without_importing_torch(self):
        for filename in ('worker.py','clip_bank.py'):
            ast.parse(Path(__file__).with_name(filename).read_text())


if __name__=='__main__':unittest.main()

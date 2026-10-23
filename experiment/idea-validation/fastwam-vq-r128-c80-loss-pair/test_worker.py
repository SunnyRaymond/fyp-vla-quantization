"""Recipe and completion gates; numerical/model checks run only in PBS."""
import copy
import json
from pathlib import Path
import unittest
from worker import validate_protocol, beta_from_means, PROTOCOL
from finalize import validate_arm


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads(Path(__file__).with_name('protocol.json').read_text())

    def test_matched_recipe_rejects_budget_or_initialization_drift(self):
        validate_protocol(self.p)
        for section,key,value in [('training','contexts_per_update',1),
                ('training','rotation_learning_rate',1e-5),
                ('training','post_update_diagnostic_forwards',True),
                ('source_initialization','job','different.pbs'),
                ('calibration','case_ids',list(range(2000,2008)))]:
            changed=copy.deepcopy(self.p)
            changed[section][key]=value
            with self.assertRaises(ValueError):validate_protocol(changed)

    def test_beta_matches_predeclared_initial_contribution(self):
        nmse,kl=.06,.02
        beta=beta_from_means(self.p,nmse,kl)
        self.assertAlmostEqual(beta*kl,.1*self.p['training']['latent_weight']*nmse)
        for bad in (0.,float('nan'),-1.):
            with self.assertRaises(ValueError):beta_from_means(self.p,nmse,bad)

    def test_terminal_receipt_requires_full_matched_budget(self):
        for arm,beta in [('az',0.),('reader_kl',.03)]:
            row=dict(protocol=PROTOCOL,arm=arm,pbs_jobid='123.pbs',complete=True,status='complete',
                updates=8,exposures=80,target_linears=614,frozen_rotation=True,final_anchor=True,
                new_teacher_queries=0,initial_parity_max_abs=0.,formal_parity_max_abs=0.,beta=beta)
            validate_arm(row,arm,'123.pbs')
            for key,value in [('exposures',79),('frozen_rotation',False),
                              ('formal_parity_max_abs',float('nan')),('pbs_jobid','stale.pbs')]:
                with self.assertRaises(ValueError):validate_arm({**row,key:value},arm,'123.pbs')


if __name__=='__main__':unittest.main()

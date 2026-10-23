import copy
import json
from pathlib import Path
import unittest

import control
import stage2

HERE=Path(__file__).parent
S=json.loads((HERE/'stage2_protocol.json').read_text(encoding='utf-8'))

class Stage2(unittest.TestCase):
    def test_frozen_scope(self):
        stage2.validate_config(control.P,S)
        bad=copy.deepcopy(S);bad['horizons']=[1,4,16]
        with self.assertRaises(RuntimeError):stage2.validate_config(control.P,bad)

    def test_eligible_candidate_not_merely_best_primary(self):
        baseline={m:1.0 for m in stage2.METRICS}
        low={**baseline,'first10motor_rmse':0.985,'future_latent_nmse':1.06}
        high={**baseline,'first10motor_rmse':0.988,'future_latent_nmse':1.02}
        decisions,chosen=stage2.select_candidates({'stage2_rho1':low,'stage2_rho225':high},baseline,S)
        self.assertFalse(decisions['stage2_rho1']['eligible'])
        self.assertTrue(decisions['stage2_rho225']['eligible']);self.assertEqual(chosen,'stage2_rho225')
        low['first10motor_rmse']=0.995;low['future_latent_nmse']=0.9
        high['first10motor_rmse']=1.01
        self.assertIsNone(stage2.select_candidates({'stage2_rho1':low,'stage2_rho225':high},baseline,S)[1])

    def test_constant_training_history_and_case_order_required(self):
        history=json.loads((HERE/'results/train_current/training_state.json').read_text(encoding='utf-8'))['history']
        state={'stage2_protocol':S['protocol'],'unit':'stage2_rho1','rho':1.0,'updates':16,
               'source_initial_checkpoint':S['source_initial_checkpoint'],'initial_state_exact':True,
               'history':[{**r,'rho':1.0} for r in history]}
        stage2.validate_training(state,control.P,S,'stage2_rho1')
        state['history'][8]['case_id']=2019
        with self.assertRaises(RuntimeError):stage2.validate_training(state,control.P,S,'stage2_rho1')

    def test_completion_requires_all_horizons_and_restore(self):
        row={'handle':{'unit':'stage2_rho1','job_id':'123.pbs101'},
             'scheduler':{'query_rc':0,'job_state':'F','Exit_status':'0'},'artifact_exit_code':0,'marker':True,
             'summary':{'protocol':control.P['protocol'],'pbs_jobid':'123.pbs101','complete':True,'updates':16,
                        'queries':24,'horizon_queries':{'1':8,'8':8,'16':8},'initial_state_exact':True,
                        'evaluation_restore_exact':True,'new_teacher_queries':0}}
        self.assertTrue(control.passed(row))
        broken=copy.deepcopy(row);del broken['summary']['horizon_queries']['8'];self.assertFalse(control.passed(broken))
        broken=copy.deepcopy(row);broken['summary']['evaluation_restore_exact']=False;self.assertFalse(control.passed(broken))

    def test_archived_dependency_requires_exact_terminal_receipt_and_fresh_artifacts(self):
        row={'handle':{'unit':'preflight','job_id':'123.pbs101'},'scheduler':{'query_rc':153,'job_state':None,'Exit_status':None},
             'artifact_exit_code':0,'marker':True,'summary':{'protocol':control.P['protocol'],'pbs_jobid':'123.pbs101','complete':True,'states':['initial','warm']}}
        archive={'complete':True,'checked_utc':'2026-10-19T19:18:08+00:00','units':{'preflight':{
            'job_id':'123.pbs101','passed':True,'artifact_exit_code':0,'marker':True,'matching_complete_summary':True,'pbs_state':'F','exit_status':'0'}}}
        self.assertTrue(control.dependency_passed(row,archive));self.assertFalse(control.passed(row))
        bad=copy.deepcopy(row);bad['scheduler']['query_rc']=1;self.assertFalse(control.dependency_passed(bad,archive))
        bad=copy.deepcopy(row);bad['marker']=False;self.assertFalse(control.dependency_passed(bad,archive))
        bad=copy.deepcopy(row);bad['handle']['job_id']='456.pbs101';self.assertFalse(control.dependency_passed(bad,archive))

if __name__=='__main__':unittest.main()

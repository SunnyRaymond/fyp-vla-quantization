import unittest
import control

class Completion(unittest.TestCase):
    def test_pbs_walltime(self):
        self.assertEqual(control.wall_seconds('01:59:00'),7140)
        self.assertEqual(control.wall_seconds('100:00:01'),360001)
        for value in ('01:60:00','00:00:60','unknown','-1:00:00'):
            with self.assertRaises(ValueError):control.wall_seconds(value)
    def row(self,unit,summary):
        return {'handle':{'unit':unit,'job_id':'123.pbs101'},
                'scheduler':{'query_rc':0,'job_state':'F','Exit_status':'0'},'artifact_exit_code':0,'marker':True,
                'summary':{'protocol':control.P['protocol'],'pbs_jobid':'123.pbs101','complete':True,**summary}}
    def test_complete_cohort_required(self):
        row=self.row('evaluate_low',{'queries':96,'split_queries':{'confirmation':16,'historical':80}})
        self.assertTrue(control.passed(row))
        row['summary']['queries']=95;self.assertFalse(control.passed(row))
    def test_preflight_repair_requires_failed_terminal(self):
        row=self.row('preflight',{'states':['initial','rho1.5_update8'],'gate_passed':False})
        self.assertTrue(control.retryable_preflight(row))
        row['summary']['gate_passed']=True;self.assertFalse(control.retryable_preflight(row))
        row['summary']['gate_passed']=False;row['scheduler']['job_state']='R';self.assertFalse(control.retryable_preflight(row))
        row['scheduler']['job_state']='F';row['scheduler']['Exit_status']='1';row['artifact_exit_code']=1
        row['progress']={'phase':'failed'};self.assertTrue(control.retryable_preflight(row))
        row['artifact_exit_code']=None;self.assertFalse(control.retryable_preflight(row))
    def test_exit_marker_and_job_binding(self):
        row=self.row('search',{'rounds':8,'updates':16});self.assertTrue(control.passed(row))
        row['scheduler']['query_rc']=1;self.assertFalse(control.passed(row));row['scheduler']['query_rc']=0
        row['summary']['pbs_jobid']='456.pbs101';self.assertFalse(control.passed(row));row['summary']['pbs_jobid']='123.pbs101'
        row['marker']=False;self.assertFalse(control.passed(row))

if __name__=='__main__':unittest.main()

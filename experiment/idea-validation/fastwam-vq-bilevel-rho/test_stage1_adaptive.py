import copy
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock,patch

import control
import stage1_adaptive as stage1

HERE=Path(__file__).parent

class Stage1(unittest.TestCase):
    def test_paired_cached_metrics_and_seed_aggregation(self):
        rows=json.loads((HERE/'analysis_failure_20261020/existing_endpoint_records.json').read_text(encoding='utf-8'))['records']
        records={split:{arm:[r for r in rows if r['split']==split and r['arm']==arm] for arm in ('current','selected')} for split in ('outer','confirmation')}
        for split in records:
            records[split]['adaptive']=[{**r,'arm':'adaptive'} for r in records[split]['current']]
        result=stage1.compare(records,control.P)
        self.assertAlmostEqual(result['summaries']['confirmation']['adaptive']['overall']['first10motor_rmse'],0.0431395914638415)
        self.assertEqual(result['adaptive_percent_vs_fixed']['confirmation']['current']['first10motor_rmse'],0)
        self.assertEqual(len(result['paired_cases']['confirmation']),8)
        self.assertEqual(len(result['paired_cases']['confirmation'][-1]['seed_records']),6)
        records['confirmation']['adaptive'].pop()
        with self.assertRaises(ValueError):stage1.compare(records,control.P)

    def source(self):
        selected=json.loads((HERE/'results/selected_rho.json').read_text(encoding='utf-8'))
        summary=json.loads((HERE/'results/search/summary.json').read_text(encoding='utf-8'))
        operations=json.loads((HERE/'analysis_failure_20261020/search_operations.json').read_text(encoding='utf-8'))
        training=[r for r in operations if r['kind']=='search_committed']
        state={'protocol':control.P['protocol'],'numerics':stage1.NUMERICS,'round':8,'updates':16,
               'commits_in_round':0,'round_update':None,'probes':{},'history':selected['history'],
               'training_history':training,'rho':selected['rho']}
        snapshot={'updates':16,'optimizer_reset_after_update1':True,'optimizer_state':{'state':{'nonempty':True}}}
        return {'search_state':state,'snapshot':snapshot},summary,selected

    def test_requires_complete_adaptive_history(self):
        saved,summary,selected=self.source()
        stage1.validate_search(saved,control.P,summary,selected)
        broken=copy.deepcopy(saved);broken['search_state']['training_history'][0]['rho']=0.1
        with self.assertRaises(RuntimeError):stage1.validate_search(broken,control.P,summary,selected)
        saved['snapshot']['updates']=14
        with self.assertRaises(RuntimeError):stage1.validate_search(saved,control.P,summary,selected)

    def test_completion_requires_scope_and_read_only_execution(self):
        row={'handle':{'unit':'stage1_adaptive','job_id':'123.pbs101'},
             'scheduler':{'query_rc':0,'job_state':'F','Exit_status':'0'},'artifact_exit_code':0,'marker':True,
             'summary':{'protocol':control.P['protocol'],'pbs_jobid':'123.pbs101','complete':True,
                        'queries':24,'split_queries':{'outer':8,'confirmation':16},'comparison_records':72,
                        'cached_baseline_queries':48,'new_training_updates':0,'new_teacher_queries':0,
                        'master_parameters_unchanged':True}}
        self.assertTrue(control.passed(row))
        for key,value in (('queries',23),('new_training_updates',1),('new_teacher_queries',1),('master_parameters_unchanged',False)):
            broken=copy.deepcopy(row);broken['summary'][key]=value;self.assertFalse(control.passed(broken))
        row['marker']=False;self.assertFalse(control.passed(row))

    def test_gpu_cost_includes_stage_number_and_rejects_empty_receipt(self):
        files=MagicMock();files.listdir.return_value=['stage1_adaptive_intent_0.json','search_intent_0.json']
        context=MagicMock();context.__enter__.return_value=files
        handle={'unit':'stage1_adaptive','job_id':'123.pbs101','gpus':1}
        query={'query_rc':0,'job_state':'F','Exit_status':'0','resources_used.walltime':'00:17:09'}
        with patch.object(control.remote,'sftp_client',return_value=context),patch.object(control,'read',return_value=handle),patch.object(control.source.source.previous,'scheduler',return_value=query):
            result=control.gpu_cost(None,'stage1_adaptive')
            self.assertTrue(result['complete']);self.assertEqual(len(result['gpu_jobs']),1)
            self.assertAlmostEqual(result['total_new_gpu_hours'],1029/3600)
            files.listdir.return_value=[]
            self.assertFalse(control.gpu_cost(None,'stage1_adaptive')['complete'])

if __name__=='__main__':unittest.main()

"""Source and numerical gates without loading Torch or a model locally."""
import copy
import json
from pathlib import Path
import unittest
from runtime import validate_source_documents,compare_source_metrics,expected_install_metadata,METRICS


class SourceTests(unittest.TestCase):
    def setUp(self):
        folder=Path(__file__).resolve().parent
        self.p=json.loads((folder/'protocol.json').read_text())
        self.source=json.loads((folder.parent/'fastwam-vq-r128-c80-qclip/protocol.json').read_text())
        binding=self.p['arm_sources']['qclip']
        self.summary=dict(protocol=binding['source_protocol'],pbs_jobid=binding['source_job'],status='complete',
            complete=True,updates=80,exposures=80,target_linears=614,weights_frozen=True,rotation_frozen=True,
            new_teacher_queries=0,closedloop_episodes=0,initial_parity_max_abs=0.,formal_parity_max_abs=0.)
        self.endpoint=dict(protocol=binding['source_protocol'],pbs_jobid=binding['source_job'],arm='qclip',
            split='calibration',case_id=2000,seed_index=0,sampler_seed=910004000,
            source_checkpoint=binding['source_checkpoint'],**{k:.05 for k in METRICS})

    def test_uses_complete_final80_clip_on_the_exact_final_kl_parent(self):
        validate_source_documents(self.p,self.source,self.summary,self.endpoint)
        for key,value in [('updates',79),('weights_frozen',False),('rotation_frozen',False),
                          ('pbs_jobid','other.pbs'),('formal_parity_max_abs',float('nan'))]:
            with self.assertRaises(ValueError):
                validate_source_documents(self.p,self.source,{**self.summary,key:value},self.endpoint)
        wrong=copy.deepcopy(self.source)
        wrong['source_arm']='az'
        with self.assertRaises(ValueError):validate_source_documents(self.p,wrong,self.summary,self.endpoint)
        with self.assertRaises(ValueError):
            validate_source_documents(self.p,self.source,self.summary,{**self.endpoint,'arm':'q0'})

    def test_fixed_input_replay_checks_all_metrics_and_rejects_nonfinite_values(self):
        values={k:.05 for k in METRICS}
        self.assertEqual(set(compare_source_metrics(values,values,1e-6)),set(METRICS))
        for key in METRICS:
            with self.assertRaises(ValueError):compare_source_metrics({**values,key:float('nan')},values,1e-6)
            with self.assertRaises(ValueError):compare_source_metrics({**values,key:.1},values,1e-6)

    def test_preserves_clipping_and_decoded_bf16_evidence_boundary(self):
        meta=expected_install_metadata(self.p,'qclip')
        self.assertEqual(meta['clipping_updates'],80)
        self.assertEqual(meta['vq_updates'],8)
        self.assertEqual(meta['execution'],'decoded_bf16_dense_vq')
        self.assertFalse(meta['native_low_bit_kernel'])
        self.assertEqual(meta['new_training_updates'],0)
        environment=Path(__file__).with_name('environment.sh').read_text()
        self.assertIn('CUBLAS_WORKSPACE_CONFIG=:4096:8',environment)


if __name__=='__main__':unittest.main()

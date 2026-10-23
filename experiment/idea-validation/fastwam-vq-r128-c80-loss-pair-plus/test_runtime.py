"""Checkpoint/anchor binding gates without importing Torch."""
import copy
import json
from pathlib import Path
import unittest
from runtime import validate_source_documents,expected_install_metadata


class BindingTests(unittest.TestCase):
    def setUp(self):
        folder=Path(__file__).resolve().parent
        self.p=json.loads((folder/'protocol.json').read_text())
        self.source=json.loads((folder.parent/'fastwam-vq-r128-c80-loss-pair/protocol.json').read_text())

    def test_requires_final_frozen_matched_checkpoint_and_anchor(self):
        for arm,beta in [('az',0.),('reader_kl',.02)]:
            s=self.p['arm_sources'][arm]
            common=dict(protocol=s['source_protocol'],arm=arm,pbs_jobid=s['source_job'],
                updates=8,exposures=80,target_linears=614,frozen_rotation=True)
            summary=dict(**common,complete=True,status='complete',final_anchor=True,
                new_teacher_queries=0,beta=beta,initial_parity_max_abs=0.,formal_parity_max_abs=0.)
            anchor=dict(**common,case_id=2000,seed_index=0,sampler_seed=910004000,
                        counts={'video_steps':10,'action_steps':10})
            validate_source_documents(self.p,arm,self.source,summary,anchor)
            for field,value in [('arm','wrong'),('updates',7),('frozen_rotation',False)]:
                with self.assertRaises(ValueError):
                    validate_source_documents(self.p,arm,self.source,summary,{**anchor,field:value})
            bad=copy.deepcopy(self.source)
            bad['source_initialization']['bank_directory']='different_banks'
            with self.assertRaises(ValueError):validate_source_documents(self.p,arm,bad,summary,anchor)

    def test_evidence_describes_vq_instead_of_uncompressed_weight_path(self):
        for arm in self.p['arms']:
            meta=expected_install_metadata(self.p,arm)
            self.assertTrue(meta['vq_enabled'])
            self.assertEqual(meta['weight_bits'],4)
            self.assertEqual(meta['vq_updates'],8)
            self.assertFalse(meta['native_low_bit_kernel'])
            self.assertNotIn('cached_transformed_weights',meta)

    def test_clean_container_receives_calibration_deterministic_workspace(self):
        folder=Path(__file__).resolve().parent
        environment=(folder/'environment.sh').read_text()
        self.assertIn('--env "CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=20261022"',environment)
        runtime=(folder/'runtime.py').read_text()
        self.assertLess(runtime.index("os.environ.get('CUBLAS_WORKSPACE_CONFIG')"),
                        runtime.index('pilot.Runtime.__init__(self,out)'))


if __name__=='__main__':unittest.main()

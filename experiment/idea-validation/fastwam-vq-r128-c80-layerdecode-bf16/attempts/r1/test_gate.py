"""The second stage must not run on a partial or failed first-stage result."""
import copy
import runpy
from pathlib import Path
import unittest

gate = runpy.run_path(str(Path(__file__).with_name('finalize.py')))['first_gate']


class GateTests(unittest.TestCase):
    def setUp(self):
        self.p = {'source':{'source_job':'frozen'}}
        template = dict(complete=True, queries=80, source_job='frozen', new_training_updates=0,
            deterministic_algorithms=True, fill_uninitialized_memory=True, allow_tf32=False,
            allow_bf16_reduced_precision_reduction=True, compiled=False, cuda_graph=False,
            gpu_uuid='same', anchor_action_max_abs=0, anchor_future_max_abs=0,
            coverage=dict(modules=614, timed_linear_calls=514880, encoded_target_modules=614,
                          dense_target_weights=0, decoded_calls=514880, scratch_bytes=113246208),
            metrics=dict(full_query_wall_ms=dict(p50=6000, mean=6000),
                         peak_allocated_gpu_GB=dict(maximum=25)),
            load_transient_memory=dict(peak_allocated_bytes=25_000_000_000))
        self.rows = {m:copy.deepcopy(template) for m in ('bf16','eager','input_fused','layer_decode_fp32')}
        candidate=self.rows['layer_decode_fp32']
        candidate['metrics']['full_query_wall_ms'].update(p50=4000,mean=4000)
        candidate['metrics']['peak_allocated_gpu_GB']['maximum']=17
        candidate['load_transient_memory']['peak_allocated_bytes']=17_000_000_000
        self.errors={m:dict(action_max_abs=0) for m in self.rows}
        self.validation=dict(complete=True,decoder_modules=614,decoder_all_bitwise=True,
                             linear_all_exact=True,linear_shape_count=12)

    def check(self):
        return gate(self.p,self.rows,self.errors,self.validation)

    def test_success_requires_both_resources_and_outputs(self):
        self.assertTrue(self.check()['passed'])
        self.errors['layer_decode_fp32']['action_max_abs']=0.001
        self.assertFalse(self.check()['passed'])

    def test_lower_memory_alone_cannot_unlock_stage_two(self):
        self.rows['layer_decode_fp32']['metrics']['full_query_wall_ms']['p50']=7000
        self.assertFalse(self.check()['passed'])

    def test_lower_latency_alone_cannot_unlock_stage_two(self):
        self.rows['layer_decode_fp32']['metrics']['peak_allocated_gpu_GB']['maximum']=26
        self.assertFalse(self.check()['passed'])

    def test_incomplete_cohort_or_decoder_cannot_unlock_stage_two(self):
        self.rows['eager']['queries']=79
        self.assertFalse(self.check()['passed'])
        self.rows['eager']['queries']=80
        self.validation['decoder_modules']=613
        self.assertFalse(self.check()['passed'])


if __name__ == '__main__':
    unittest.main()

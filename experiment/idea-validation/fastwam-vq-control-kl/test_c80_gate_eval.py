import os
import time
import unittest

from control_common import Deadline
from c80_gate_eval import (
    ACTION_SHAPE, NOISE_SHAPE, _reader_pass, _record_is_reusable,
)


class C80GateRecordTests(unittest.TestCase):
    def test_resume_requires_exact_proposal_and_full_pair_bindings(self):
        binding = {
            'gate_protocol': 'toy-gate-v1', 'unit': 'c80_eval_latent_0',
            'seed': 11, 'input_identity': {'clean': 'a', 'noisy': 'b'},
            'proposal': {'direction_unit': 'c80_direction_latent',
                         'direction_pbs_jobid': '123.server',
                         'candidate_path': '/tmp/candidate-0.1.pt', 'fraction': 0.1},
        }
        record = {
            'complete': True, 'binding': binding, 'created_pbs_jobid': 'toy.pbs',
            'case_loss': 0.25,
            'loss_components': {'clean': 0.2, 'noisy': 0.3},
            'native_noise_parity': {'clean': True, 'noisy': True},
            'reader_noise_parity': {'clean': True, 'noisy': True},
            'runtime_counts': {variant: {'video_steps': 10, 'action_steps': 10}
                               for variant in ('clean', 'noisy')},
            'reader_counts': {variant: {'video_prefill_calls': 1,
                                        'action_scheduler_steps': 10}
                              for variant in ('clean', 'noisy')},
        }
        self.assertTrue(_record_is_reusable(record, binding, 'latent_control_kl'))
        changed = {**binding, 'proposal': {**binding['proposal'],
                                           'direction_pbs_jobid': '124.server'}}
        self.assertFalse(_record_is_reusable(record, changed, 'latent_control_kl'))
        changed = {**binding, 'proposal': {**binding['proposal'],
                                           'candidate_path': '/tmp/repaired-candidate.pt'}}
        self.assertFalse(_record_is_reusable(record, changed, 'latent_control_kl'))
        changed = {**binding, 'seed': 12}
        self.assertFalse(_record_is_reusable(record, changed, 'latent_control_kl'))
        record['reader_noise_parity'] = {'clean': True, 'noisy': False}
        self.assertFalse(_record_is_reusable(record, binding, 'latent_control_kl'))

    def test_reader_pass_uses_live_control_deadline_and_count_contract(self):
        import torch

        key = 'FW_JOB_START_EPOCH'
        previous = os.environ.get(key)
        os.environ[key] = str(int(time.time()))
        try:
            deadline = Deadline()
            expected_noise = torch.zeros(NOISE_SHAPE)
            action = torch.zeros(ACTION_SHAPE)

            def reader(_runtime, _datum, _seed, _future, checkpointed=False):
                self.assertFalse(checkpointed)
                return action, {
                    'action_initial_noise': expected_noise,
                    'video_prefill_calls': 1,
                    'action_scheduler_steps': 10,
                }

            counters = {'reader_new_pass_attempted_count': 0,
                        'reader_new_pass_count': 0}
            actual_action, counts = _reader_pass(
                torch, reader, None, None, 11, torch.empty((0,)), expected_noise,
                deadline, counters)
            self.assertTrue(torch.equal(actual_action, action))
            self.assertEqual(counts, {'video_prefill_calls': 1,
                                      'action_scheduler_steps': 10})
            self.assertEqual(counters, {'reader_new_pass_attempted_count': 1,
                                        'reader_new_pass_count': 1})
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


if __name__ == '__main__':
    unittest.main()

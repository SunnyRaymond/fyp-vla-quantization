"""Small stdlib-only checks for the frozen cohort and source bindings."""
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))


class ProtocolContractTests(unittest.TestCase):
    def test_five_arms_and_endpoint_denominator(self):
        self.assertEqual(P['arms'], ['baseline', 'clip_only', 'joint_c80', 'per_channel_clip', 'joint_per_channel'])
        self.assertEqual(P['arm'], 'joint_per_channel')
        self.assertEqual(P['expected']['reused_endpoint_records'], 640)
        self.assertEqual(P['expected']['new_endpoint_records'], 160)
        self.assertEqual(P['expected']['total_endpoint_records'], 800)
        self.assertEqual(P['evaluation']['queries_per_arm'], 80)
        self.assertEqual(P['expected']['new_teacher_queries'], 0)

    def test_original_c80_paired_query_sets(self):
        self.assertEqual(len(P['calibration']['case_ids']), 80)
        self.assertEqual(len(set(P['calibration']['case_ids'])), 80)
        self.assertEqual(len(P['evaluation']['case_ids']), 40)
        self.assertEqual(P['evaluation']['seed_indices'], [0, 1])
        self.assertEqual(P['calibration']['seed_index'], 0)
        self.assertEqual(P['source_control_protocol']['calibration']['case_ids'], P['calibration']['case_ids'])
        self.assertEqual(P['source_control_protocol']['evaluation']['case_ids'], P['evaluation']['case_ids'])
        self.assertEqual(P['source_joint_protocol']['calibration']['case_ids'], P['calibration']['case_ids'])
        self.assertEqual(P['source_joint_protocol']['evaluation']['case_ids'], P['evaluation']['case_ids'])
        self.assertEqual(len(P['calibration']['case_ids']) * 1, 80)
        self.assertEqual(len(P['evaluation']['case_ids']) * len(P['evaluation']['seed_indices']), 80)

    def test_fresh_joint_parameter_groups(self):
        self.assertEqual(P['training']['additional_updates'], 80)
        self.assertEqual(P['training']['weight_learning_rate'], 0.0005)
        self.assertEqual(P['training']['clip_learning_rate'], 0.01)
        self.assertEqual(P['scaling']['learning_rate'], 0.02)
        self.assertEqual(P['scaling']['group_size'], 1)
        self.assertEqual(P['scaling']['initial_log2'], 0.0)
        self.assertEqual(P['clipping']['initial_ratio'], 1.0)
        self.assertIn('C8 rho1.5 baseline', P['scaling']['initialization'])
        self.assertIn('no B32 warm start', P['scaling']['initialization'])
        for group in ('614 ratios', 'per-input-channel log2D', 'VQ books', 'log row gains'):
            self.assertIn(group, P['parameters'])

    def test_source_control_roots_and_allocation_guards(self):
        self.assertEqual(P['control_root'], P['source_control_protocol']['remote_root'])
        self.assertEqual(P['per_channel_root'], P['source_per_channel_protocol']['remote_root'])
        self.assertEqual(P['joint_control_root'], P['source_joint_protocol']['remote_root'])
        self.assertEqual(P['source_joint_protocol']['protocol'], 'fastwam-vq-c80-joint-rmse-v1')
        self.assertEqual(P['source_joint_protocol']['joint_source']['updates'], 80)
        self.assertEqual(P['allocation']['walltime'], '01:59:00')
        self.assertEqual(P['allocation']['soft_deadline_seconds'], 6840)
        self.assertEqual(P['allocation']['gpu_total_budget_seconds'], 7200)
        self.assertIs(P['gpu_budget']['automatic_resubmission'], False)


if __name__ == '__main__':
    unittest.main()

"""Local standard-library contract tests; this module never imports Torch."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import calibrate
import backward_check


PROTOCOL_PATH = Path(__file__).with_name('protocol.json')


class CalibrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))

    def test_backward_probe_reuses_container_safe_allocation_guard(self):
        self.assertIs(backward_check.guard, calibrate.guard)
        with TemporaryDirectory() as tmp:
            nodefile = Path(tmp) / 'nodes'
            nodefile.write_text('compute01.cluster\n')
            with mock.patch.dict(os.environ, {'PBS_JOBID': '123.pbs101',
                                             'PBS_NODEFILE': str(nodefile)}), \
                 mock.patch.object(calibrate.socket, 'gethostname', return_value='compute01'):
                backward_check.guard()
                with mock.patch.object(calibrate.socket, 'gethostname', return_value='login01'):
                    with self.assertRaises(RuntimeError):
                        backward_check.guard()
                with mock.patch.object(calibrate.socket, 'gethostname', return_value='compute02'):
                    with self.assertRaises(RuntimeError):
                        backward_check.guard()
                with mock.patch.dict(os.environ, {'PBS_JOBID': ''}):
                    with self.assertRaises(RuntimeError):
                        backward_check.guard()

    def test_frozen_source_and_arm_contract(self):
        calibrate.validate_protocol(self.protocol)
        p = self.protocol
        self.assertEqual(p['arm_configs']['clip_only']['updates'], 80)
        self.assertFalse(p['arm_configs']['clip_only']['train_weights'])
        self.assertTrue(p['arm_configs']['clip_only']['train_clips'])
        self.assertTrue(p['arm_configs']['joint']['train_weights'])
        self.assertTrue(p['arm_configs']['joint']['train_clips'])
        self.assertEqual(len(set(p['calibration']['update_order'])), 80)

    def test_rejects_changed_update_order(self):
        p = copy.deepcopy(self.protocol)
        p['calibration']['update_order'][0], p['calibration']['update_order'][1] = (
            p['calibration']['update_order'][1], p['calibration']['update_order'][0])
        with self.assertRaises(ValueError):
            calibrate.validate_protocol(p)

    def test_rejects_changed_backward_implementation(self):
        p = copy.deepcopy(self.protocol)
        p['quantization']['weight_backward']['max_vectors_per_chunk'] = 32
        with self.assertRaises(ValueError):
            calibrate.validate_protocol(p)

    def test_deployment_and_receipt_need_exact_frozen_coverage(self):
        p = self.protocol
        arm = 'joint'
        masters = {f'linear_{i}': {'books': object(), 'log_scales': object()}
                   for i in range(614)}
        deployment = {
            'schema_version': 1, 'protocol': p['protocol'], 'arm': arm,
            'updates': 80, 'target_linears': 614, 'epochs': 1, 'rho': 1.5,
            'normalization_c': p['training']['normalization_c'],
            'case_ids': p['calibration']['case_ids'],
            'update_order': p['calibration']['update_order'],
            'module_masters': masters,
            'ratios': {name: object() for name in masters},
        }
        calibrate.validate_deployment(deployment, p, arm)
        history = [{'update': i + 1, 'case_id': case}
                   for i, case in enumerate(p['calibration']['update_order'])]
        receipt = {
            'schema_version': 1, 'protocol': p['protocol'], 'arm': arm,
            'status': 'complete', 'complete': True, 'updates': 80, 'epochs': 1,
            'target_linears': 614, 'calibration_ids': p['calibration']['case_ids'],
            'update_order': p['calibration']['update_order'],
            'normalization_c': p['training']['normalization_c'], 'rho': 1.5,
            'weights_unchanged': False, 'history_source': 'checkpoint.pt', 'history': history,
            'source': calibrate.source_bindings(p),
            'actual_infer_parity': {'max_abs': 0.0, 'passed': True},
            'reference_action_parity': {'max_abs': 0.0, 'passed': True},
            'backward_check': {'passed': True, 'implementation': 'bounded_vector_chunks_v2',
                               'max_vectors_per_chunk': 1048576},
        }
        calibrate.validate_receipt(receipt, p, arm)
        receipt['updates'] = 79
        with self.assertRaises(ValueError):
            calibrate.validate_receipt(receipt, p, arm)

    def test_backward_receipt_matches_loaded_parent_implementation(self):
        receipt = {'passed': True, 'implementation': 'bounded_vector_chunks_v2',
                   'max_vectors_per_chunk': 1048576}
        calibrate.validate_backward_receipt(receipt, 'bounded_vector_chunks_v2', 1048576)
        receipt['max_vectors_per_chunk'] = 2**20 - 1
        with self.assertRaises(RuntimeError):
            calibrate.validate_backward_receipt(receipt, 'bounded_vector_chunks_v2', 1048576)

    def test_resume_requires_matching_checkpointed_artifact_and_progress(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobid = '1234.pbs101'
            artifact = root / 'artifacts' / jobid
            training = root / 'results' / 'training' / 'joint'
            artifact.mkdir(parents=True)
            training.mkdir(parents=True)
            checkpoint = training / 'checkpoint.pt'
            checkpoint.write_bytes(b'checkpoint')
            saved = {'arm': 'joint', 'jobid': jobid, 'status': 'checkpointed',
                     'complete': False, 'updates': 3, 'checkpoint': str(checkpoint)}
            calibrate.save_json(artifact / 'calibration_summary.json', saved)
            calibrate.save_json(training / 'progress.json', saved)
            (artifact / 'exit_code.txt').write_text('75')
            (artifact / 'TRAINING_CHECKPOINTED').touch()
            self.assertEqual(
                calibrate.validate_resume_source(root, 'joint', jobid, checkpoint, saved)['updates'], 3)
            saved['updates'] = 80
            calibrate.save_json(artifact / 'calibration_summary.json', saved)
            calibrate.save_json(training / 'progress.json', saved)
            self.assertEqual(
                calibrate.validate_resume_source(root, 'joint', jobid, checkpoint, saved)['updates'], 80)
            saved['updates'] = 4
            with self.assertRaises(RuntimeError):
                calibrate.validate_resume_source(root, 'joint', jobid, checkpoint, saved)


if __name__ == '__main__':
    unittest.main()

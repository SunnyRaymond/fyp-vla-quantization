"""Stdlib metadata and parity checks for the C80 adapter."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vq_adapter  # noqa: E402


def fixture_protocol():
    return json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))


def plan_fixture(protocol):
    rows = []
    for case_id in protocol['calibration']['case_ids']:
        rows.append({'case_id': case_id, 'suite': 'libero_spatial',
                     'sampler_seeds': [900_000_000 + case_id, 900_001_000 + case_id],
                     'description': f'case {case_id}'})
    return {'protocol': 'fastwam-vq-calibration-suites-v1', 'status': 'complete',
            'calibration_cohorts': {'c80': protocol['calibration']['case_ids']},
            'inputs': rows}


def reference_fixture(protocol, arm, row):
    seed_index = protocol['closed_loop_reference']['seed_index']
    return {
        'schema_version': 1, 'protocol': protocol['protocol'], 'arm': arm,
        'case_id': row['case_id'], 'seed_index': seed_index,
        'sampler_seed': row['sampler_seeds'][seed_index], 'target_linears': 614,
        'plan_row': row, 'action': [[0.0] * 7 for _ in range(32)],
        'actual_infer_parity': {'max_abs': 0.0, 'passed': True},
    }


class AdapterMetadataTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_runner(self):
        completed = subprocess.run(
            [sys.executable, '-c',
             "import sys, vq_adapter; assert 'torch' not in sys.modules; assert 'runner' not in sys.modules"],
            cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_reference_action_identity_shape_and_parity(self):
        protocol = fixture_protocol()
        row = {'case_id': 2000, 'sampler_seeds': [12345, 67890]}
        reference = reference_fixture(protocol, 'clip_only', row)
        self.assertIs(vq_adapter.validate_reference_action(reference, protocol, 'clip_only', row), reference)
        changed = copy.deepcopy(reference)
        changed['action'][0][0] = float('nan')
        with self.assertRaises(RuntimeError):
            vq_adapter.validate_reference_action(changed, protocol, 'clip_only', row)
        changed = copy.deepcopy(reference)
        changed['actual_infer_parity']['max_abs'] = 2e-6
        with self.assertRaises(RuntimeError):
            vq_adapter.validate_reference_action(changed, protocol, 'clip_only', row)

    def test_training_preflight_checks_c80_receipt_without_loading_tensors(self):
        protocol = fixture_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / 'plan.json'
            plan = plan_fixture(protocol)
            plan_path.write_text(json.dumps(plan), encoding='utf-8')
            protocol['sources']['plan_file'] = str(plan_path)
            source = vq_adapter._expected_source(protocol)
            for arm in ('clip_only', 'joint'):
                training_dir = root / 'training' / arm
                training_dir.mkdir(parents=True)
                (training_dir / 'deployment.pt').write_bytes(b'payload is not deserialized in preflight')
                row = next(row for row in plan['inputs'] if row['case_id'] == 2000)
                receipt = {
                    'schema_version': 1, 'protocol': protocol['protocol'], 'arm': arm,
                    'status': 'complete', 'complete': True, 'updates': 80, 'epochs': 1,
                    'target_linears': 614, 'calibration_ids': protocol['calibration']['case_ids'],
                    'update_order': protocol['calibration']['update_order'],
                    'normalization_c': protocol['training']['normalization_c'], 'rho': 1.5,
                    'weights_unchanged': arm == 'clip_only', 'source': source,
                }
                (training_dir / 'training_receipt.json').write_text(
                    json.dumps(receipt), encoding='utf-8')
                (training_dir / 'reference_action.json').write_text(
                    json.dumps(reference_fixture(protocol, arm, row)), encoding='utf-8')
                result = vq_adapter.validate_training_artifacts(protocol, arm, training_dir)
                self.assertEqual(result['training_receipt']['updates'], 80)
                receipt['update_order'] = list(reversed(receipt['update_order']))
                (training_dir / 'training_receipt.json').write_text(
                    json.dumps(receipt), encoding='utf-8')
                with self.assertRaises(RuntimeError):
                    vq_adapter.validate_training_artifacts(protocol, arm, training_dir)

    def test_runtime_validate_reference_uses_saved_preflight_without_torch(self):
        protocol = fixture_protocol()
        row = {'case_id': 2000, 'suite': 'libero_spatial', 'sampler_seeds': [12345, 67890],
               'description': 'case 2000'}
        reference = reference_fixture(protocol, 'clip_only', row)
        preflight = {
            'plan_path': Path('frozen-plan.json'), 'reference_row': row,
            'reference_action': reference,
            'training_receipt': {'status': 'complete'},
        }

        class FakeTensor:
            def __init__(self, values):
                self.values = list(values)

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def __sub__(self, other):
                return FakeTensor([left - right for left, right in zip(self.values, other.values)])

            def abs(self):
                return FakeTensor([abs(value) for value in self.values])

            def max(self):
                return max(self.values, default=0.0)

        fake_torch = types.ModuleType('torch')
        fake_torch.float32 = object()
        fake_torch.tensor = lambda values, dtype=None: FakeTensor(
            [value for row_values in values for value in row_values])
        fake_numpy = types.ModuleType('numpy')
        fake_source = types.ModuleType('calibration_source')
        fake_source.load_observation = lambda np, directory, source_row: (
            {'observation': 'frozen'}, {'description': 'case 2000'})
        fake_runner = types.ModuleType('runner')
        fake_runner.Runtime = type('Runtime', (), {})

        def fake_infer(runtime, datum, seed, measure=False):
            self.assertEqual(datum, {'observation': 'frozen'})
            self.assertEqual(seed, reference['sampler_seed'])
            self.assertFalse(measure)
            runtime.banks.seen.update(range(614))
            return FakeTensor([0.0] * (32 * 7)), {}

        with patch.dict(sys.modules, {
                'runner': fake_runner, 'torch': fake_torch, 'numpy': fake_numpy,
                'calibration_source': fake_source}), \
                patch.dict(os.environ, {'PBS_JOBID': 'test-job'}):
            runtime_type = vq_adapter._runtime_type()
            runtime = object.__new__(runtime_type)
            runtime.protocol = protocol
            runtime.arm_name = 'clip_only'
            runtime.preflight = preflight
            runtime.banks = types.SimpleNamespace(seen=set())
            runtime.install_metadata = {'arm': 'clip_only'}
            runtime.datum = lambda observation: observation
            runtime.infer = lambda datum, seed, measure=False: fake_infer(
                runtime, datum, seed, measure)
            receipt = runtime.validate_reference()

        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['observed_modules'], 614)
        self.assertEqual(receipt['training_receipt_status'], 'complete')
        self.assertEqual(receipt['pbs_job_id'], 'test-job')


if __name__ == '__main__':
    unittest.main()

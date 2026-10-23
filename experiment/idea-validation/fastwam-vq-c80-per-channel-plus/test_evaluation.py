from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate


def protocol_fixture():
    return evaluate.read_json(HERE / 'protocol.json')


def metadata_fixture(protocol):
    metadata = evaluate.expected_install_metadata(protocol)
    metadata.update({
        'native_int4_gemm_calls': 0,
        'source_cached_query_parity': protocol['source_expected']['cached_query_parity'],
        'ratios': {'count': 614, 'min': .7315285801887512,
                   'mean': .9572885090831048, 'max': 1.0, 'below_one': 526},
        'scaling': {
            'modules': 614, 'group_size': 1,
            'channels': protocol['source_expected']['scale_channels'],
            'raw_scale_metadata_bytes': protocol['source_expected']['raw_scale_metadata_bytes'],
            'scale_min': .5025339126586914, 'scale_max': 1.7372326850891113,
            'mean_abs_log2': .0653400868177414,
            'cached_compensated_weight_evaluation': True,
        },
    })
    return metadata


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = protocol_fixture()
        self.metadata = metadata_fixture(self.protocol)

    def test_single_arm_partition_and_complete_coverage(self):
        self.assertEqual(evaluate.validate_protocol(self.protocol)['protocol'], evaluate.PROTOCOL)
        self.assertEqual(evaluate.ARMS, ('per_channel_clip',))
        shards = [evaluate.worker_indices(worker, 'per_channel_clip') for worker in range(4)]
        self.assertTrue(all(len(indices) == 70 for indices in shards))
        self.assertEqual(sorted(index for shard in shards for index in shard), list(range(280)))
        self.assertTrue(all(indices == list(range(worker, 280, 4))
                            for worker, indices in enumerate(shards)))
        self.assertEqual(evaluate.expected_install_metadata(self.protocol)['source_training_job'],
                         self.protocol['sources']['training_job'])

    def test_install_metadata_rejects_invalid_ratio_scale_and_source_parity(self):
        self.assertEqual(evaluate.validate_install_metadata(
            self.metadata, self.protocol, 'per_channel_clip'), self.metadata)
        for field, replacement in (
            ('ratios', {'count': 613, 'min': .7, 'mean': .9, 'max': 1.0, 'below_one': 500}),
            ('scaling', dict(self.metadata['scaling'], channels=1)),
            ('scaling', dict(self.metadata['scaling'], scale_max=2.1)),
            ('source_cached_query_parity', {'action_max_abs': 1.0, 'future_max_abs': 0.0}),
        ):
            with self.subTest(field=field, replacement=replacement):
                bad = dict(self.metadata, **{field: replacement})
                with self.assertRaises(ValueError):
                    evaluate.validate_install_metadata(bad, self.protocol, 'per_channel_clip')

    def test_setup_requires_endpoint_and_all_three_live_parities(self):
        reference_action = {
            'sampler_seed': 910_004_000,
            'actual_infer_parity': self.protocol['source_expected']['cached_query_parity'],
        }
        setup = {
            'passed': True, 'protocol': evaluate.PROTOCOL, 'arm': 'per_channel_clip',
            'case_id': 2000, 'seed_index': 0, 'sampler_seed': reference_action['sampler_seed'],
            'reference_actual_infer_parity': reference_action['actual_infer_parity'],
            'observed_modules': 614, 'max_abs_vs_reference': 0.0,
            'training_receipt_status': 'complete', 'startup_queries': 3,
            'source_endpoint_metrics_passed': True,
            'uncached_cached_action_max_abs': 0.0,
            'uncached_cached_future_max_abs': 0.0,
            'actual_infer_action_max_abs': 0.0,
            'install_metadata': self.metadata,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'setup.json'
            path.write_text(json.dumps(setup), encoding='utf-8')
            receipt, installed = evaluate._setup_receipt(
                path, 'per_channel_clip', self.protocol, reference_action)
            self.assertTrue(receipt['source_endpoint_metrics_passed'])
            self.assertEqual(installed, self.metadata)
            for key, value in (
                ('source_endpoint_metrics_passed', False),
                ('uncached_cached_action_max_abs', 2e-6),
                ('uncached_cached_future_max_abs', 2e-6),
                ('actual_infer_action_max_abs', 2e-6),
            ):
                broken = dict(setup, **{key: value})
                path.write_text(json.dumps(broken), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    evaluate._setup_receipt(path, 'per_channel_clip', self.protocol, reference_action)


if __name__ == '__main__':
    unittest.main()

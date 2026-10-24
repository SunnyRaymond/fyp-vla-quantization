"""Export accounting and refusal gates; no Torch/model loading on the local host."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('export_runtime', Path(__file__).with_name('export_runtime.py'))
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportContractTests(unittest.TestCase):
    def test_accounting_counts_shared_storage_once_and_keeps_padding(self):
        report = export.inventory_from_descriptors([
            dict(storage_key=('cuda:0', 1, 128), storage_bytes=128, logical_bytes=80, dtype='torch.bfloat16'),
            dict(storage_key=('cuda:0', 1, 128), storage_bytes=128, logical_bytes=40, dtype='torch.bfloat16'),
            dict(storage_key=('cuda:0', 2, 8), storage_bytes=8, logical_bytes=8, dtype='torch.uint8')])
        self.assertEqual(report['unique_storage_bytes'], 136)
        self.assertEqual(report['logical_tensor_bytes'], 128)
        self.assertEqual(report['unique_storage_count'], 2)

    def test_header_rejects_source_drift_and_dense_leaks(self):
        bundle = dict(format=export.FORMAT, source_job=export.SOURCE_JOB, arm='reader_kl',
            training_updates=0, quantized={str(i): {} for i in range(614)},
            rotations={'action': None, 'video': None}, excluded_model_state_keys=['x.weight'],
            model_state={'x.bias': None})
        export._validate_header(bundle)
        for key, value in [('source_job', 'wrong'), ('training_updates', 1), ('arm', 'az')]:
            invalid = dict(bundle, **{key: value})
            with self.assertRaises(ValueError):
                export._validate_header(invalid)
        invalid = copy.deepcopy(bundle)
        invalid['model_state']['x.weight'] = None
        with self.assertRaises(ValueError):
            export._validate_header(invalid)

    def test_guard_refuses_before_importing_torch(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'allocation'):
                export.load_bundle('unused.pt')
        self.assertNotIn('torch', sys.modules)

    def test_export_resolves_inference_nodes_without_hydra_output_directory(self):
        config = dict(model=dict(_target_='FastWAM', proprio_dim='${data.train.processor.proprio_output_dim}'),
            data=dict(train=dict(processor=dict(proprio_output_dim=7), video_size=[224, 448],
                                 concat_multi_camera='horizontal', num_frames=33)),
            EVALUATION=dict(output_dir='${hydra:runtime.output_dir}', action_infer_mode='idm',
                            num_inference_steps=10), seed=20261022)
        reads = []

        class FakeOmegaConf:
            @staticmethod
            def select(root, path, default=None):
                reads.append(path)
                node = root
                for key in path.split('.'):
                    if not isinstance(node, dict) or key not in node:
                        return default
                    node = node[key]
                return node

            @staticmethod
            def is_config(value):
                return isinstance(value, (dict, list))

            @staticmethod
            def to_container(value, **kwargs):
                if isinstance(value, dict):
                    return {key: FakeOmegaConf.to_container(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [FakeOmegaConf.to_container(item) for item in value]
                if value == '${data.train.processor.proprio_output_dim}':
                    return config['data']['train']['processor']['proprio_output_dim']
                if isinstance(value, str) and '${hydra:' in value:
                    raise RuntimeError('HydraConfig was not set')
                return value

            create = staticmethod(lambda value: value)
            to_yaml = staticmethod(lambda value, **kwargs: str(value))

        with tempfile.TemporaryDirectory() as folder:
            stats = Path(folder) / 'stats.json'
            stats.write_text('{"action":{},"state":{}}')
            config['EVALUATION']['dataset_stats_path'] = str(stats)
            runtime = types.SimpleNamespace(cfg=config)
            with patch.dict(sys.modules, {'omegaconf': types.SimpleNamespace(OmegaConf=FakeOmegaConf)}):
                assets = export._deployment_assets(runtime)
        deployed = assets['resolved_config']
        self.assertEqual(deployed['model']['proprio_dim'], 7)
        self.assertEqual(deployed['data']['train']['video_size'], [224, 448])
        self.assertEqual(deployed['EVALUATION']['num_inference_steps'], 10)
        self.assertEqual(assets['dataset_stats_json'], '{"action":{},"state":{}}')
        self.assertNotIn('output_dir', deployed['EVALUATION'])
        self.assertNotIn('EVALUATION.output_dir', reads)


if __name__ == '__main__':
    unittest.main()
